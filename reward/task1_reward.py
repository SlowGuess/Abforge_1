import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from openai import AsyncOpenAI

logger = logging.getLogger("task1_reward")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REWARD_DIR = Path(__file__).resolve().parent
PARENT_DIR = REWARD_DIR.parent
for path in (REWARD_DIR, PARENT_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from task1_candidate_utils import (  # noqa: E402
    approx_token_count,
    coerce_extra_info,
    compute_count_penalty_v2_from_env,
    compute_weighted_score,
    compute_wt_p_recall,
    extract_result_block,
    parse_candidates,
)


app = FastAPI()


def use_binary_scoring() -> bool:
    value = (os.environ.get("TASK1_BINARY_SCORING", "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Structured bullet check (mirrors eval parser parse_predicted_objectives).
# Substring match was easily gamed by placeholder lines like "[A list of target
# modules and research questions.]", so we now require actual bullet headers:
#   - Target Module: <name>
#     - Research Question: <q>
# Same regex shape as eval parser, so reward-pass and eval-parseable are aligned.
_TARGET_BULLET_RE = re.compile(
    r'\n\s*[-*]\s*(?:\*\*)?Target Module(?:\*\*)?\s*:', re.IGNORECASE)
_RQ_BULLET_RE = re.compile(
    r'\n\s*[-*]\s*(?:\*\*)?Research Question(?:\*\*)?\s*:', re.IGNORECASE)


def has_expected_format(result_text: str) -> bool:
    text = "\n" + (result_text or "")
    return bool(_TARGET_BULLET_RE.search(text)) and bool(_RQ_BULLET_RE.search(text))


def count_objective_pairs(result_text: str) -> int:
    """Count complete (Target Module, Research Question) bullet pairs.
    A pair = matched bullet headers; we conservatively use min(n_tm, n_rq).
    """
    text = "\n" + (result_text or "")
    n_tm = len(_TARGET_BULLET_RE.findall(text))
    n_rq = len(_RQ_BULLET_RE.findall(text))
    return min(n_tm, n_rq)


def extract_numbered_bullets(result_text: str) -> List[Dict[str, Any]]:
    """Parse result text into ordered (Target Module, Research Question) bullets.
    Returns list of dicts with 1-based idx, target_module, research_question.

    Used in v9 to: (1) number bullets in the judge prompt so the judge can pick a
    primary_match by index, and (2) server-side enforce the consumed-once rule.
    """
    if not result_text:
        return []
    text = "\n" + result_text
    tm_matches = list(_TARGET_BULLET_RE.finditer(text))
    rq_matches = list(_RQ_BULLET_RE.finditer(text))
    bullets: List[Dict[str, Any]] = []
    for i, tm in enumerate(tm_matches):
        tm_start = tm.end()
        next_tm_start = tm_matches[i + 1].start() if i + 1 < len(tm_matches) else len(text)
        rq_in_range = next((rq for rq in rq_matches if tm_start < rq.start() < next_tm_start), None)
        if rq_in_range is None:
            continue
        tm_text = text[tm_start:rq_in_range.start()].strip().lstrip(":").strip().rstrip("*").strip()
        rq_text = text[rq_in_range.end():next_tm_start].strip().lstrip(":").strip().rstrip("*").strip()
        # Trim verbose RQ to keep judge prompt manageable.
        if len(rq_text) > 400:
            rq_text = rq_text[:400] + "..."
        bullets.append({
            "idx": len(bullets) + 1,
            "target_module": tm_text[:300],
            "research_question": rq_text,
        })
    return bullets


def enforce_bipartite(
    eval_items: List[Dict[str, Any]],
    n_pred_bullets: int,
) -> Dict[str, Any]:
    """v12 score-based 1:1 enforce. Reverses two prior failures:

    v9 enforce_consumed_once was greedy by focus order — the first focus to
    claim a bullet kept it, even if a LATER focus would have been a higher-
    scoring match. This mis-zeroed legitimate matches.

    v10/v11 detect_reuse was telemetry-only — judge soft-violated the 1:1
    rule with impunity, inflating candidate_score (n_reuse_observed climbed
    from 0.17 → 0.56 per sample over 100 steps).

    v12 fix: when multiple focuses cite the same bullet, keep ONLY the
    highest-scoring focus's pairing; the other focuses' scores are forced to
    0.0. Tie-break: same score → earlier focus position wins. This is a
    score-based local approximation to Hungarian 1:1 assignment; not optimal
    but closes the math-level reuse hack discussed in v11 analysis.

    Mutates eval_items in place. Returns telemetry dict.
    """
    # Group focuses by their primary_match bullet
    by_bullet: Dict[int, List[Any]] = {}
    for pos, item in enumerate(eval_items):
        pm = item.get("primary_match")
        if pm is None:
            continue
        try:
            pm_int = int(pm)
        except (ValueError, TypeError):
            continue
        if pm_int <= 0 or pm_int > n_pred_bullets:
            continue
        try:
            score = float(item.get("score") or 0)
        except (ValueError, TypeError):
            score = 0.0
        by_bullet.setdefault(pm_int, []).append((pos, score))

    n_reuse_observed = 0
    n_reuse_zeroed = 0
    for bullet_idx, claims in by_bullet.items():
        if len(claims) <= 1:
            continue
        # Reuse: pick winner by (higher score, then earlier pos)
        claims.sort(key=lambda x: (-x[1], x[0]))
        winner_pos = claims[0][0]
        for pos, _ in claims[1:]:
            n_reuse_observed += 1
            n_reuse_zeroed += 1
            item = eval_items[pos]
            item["original_score"] = item.get("score")
            item["score"] = 0.0
            item["reuse_zeroed"] = True
            old_crit = (item.get("critique") or "")[:200]
            item["critique"] = (
                f"[V12-BIPARTITE: bullet {bullet_idx} consumed by focus at "
                f"position {winner_pos + 1} with higher score; this focus zeroed. "
                f"Original critique: {old_crit}]"
            )

    return {
        "n_reuse_observed": n_reuse_observed,
        "n_reuse_zeroed": n_reuse_zeroed,
        "n_bullets_used": len(by_bullet),
    }


def _two_tier_overlong_penalty(token_count: int, t1: int, r1: float, t2: int, r2: float,
                                t3: int = 0, r3: float = 0.0) -> float:
    """Additive multi-tier overlong penalty. Past each threshold the per-100tok
    rate adds to the previous tier(s), giving piecewise-linear growth.
    tier3 args are optional (default 0/0 = disabled) to keep existing callers."""
    overlong_t1 = max(0, token_count - t1)
    penalty = (overlong_t1 / 100.0) * r1
    if t2 > 0 and r2 > 0:
        overlong_t2 = max(0, token_count - t2)
        penalty += (overlong_t2 / 100.0) * r2
    if t3 > 0 and r3 > 0:
        overlong_t3 = max(0, token_count - t3)
        penalty += (overlong_t3 / 100.0) * r3
    return penalty


def compute_format_factor(raw_response: str, result_text: str) -> Dict[str, float]:
    """v6 format gate: multiplicative factor in [0, 1].

    v5's additive penalty (max 0.40) was too small relative to the [0, 1]
    candidate_score range — a non-compliant rollout with high content still
    out-scored a compliant rollout with mediocre content. Worse, the reward
    server's `extract_result_block` falls back to EOS when `</Result>` is
    missing, so the model still received content credit and learned that
    closing tag is optional (v5 inference: 198/200 missing `</Result>`).

    Multiplicative gate fixes this by zeroing content reward on hard
    violations: missing close tag, missing open tag, or zero parseable
    bullet pairs. Partial bullet compliance (n_pairs=1) gets factor=0.5.

    Base Qwen3-8B already produces compliant format (verified 200/200 on
    bench200), so factor=1.0 from the start of training — no gradient
    explosion risk like v4's binary 0→0.25 additive shock.
    """
    raw = raw_response or ""
    raw_lower = raw.lower()
    has_open = "<result>" in raw_lower
    has_close = "</result>" in raw_lower
    n_pairs = count_objective_pairs(result_text)

    if not has_open:
        return {"factor": 0.0, "reason": "no_result_open", "n_pairs": n_pairs}
    if not has_close:
        return {"factor": 0.0, "reason": "no_result_close", "n_pairs": n_pairs}
    if n_pairs == 0:
        return {"factor": 0.0, "reason": "no_bullet_pair", "n_pairs": n_pairs}
    if n_pairs == 1:
        return {"factor": 0.5, "reason": "single_pair", "n_pairs": n_pairs}
    return {"factor": 1.0, "reason": "ok", "n_pairs": n_pairs}


def compute_count_penalty(n_pred: int, n_gt: int) -> float:
    """GT-relative count penalty. Delegates to compute_count_penalty_v2_from_env.

    n_pred ≤ n_gt + TASK1_COUNT_FREE_EXTRA: no penalty.
    n_gt + TASK1_COUNT_FREE_EXTRA < n_pred ≤ TASK1_COUNT_HARD_THRESHOLD (default 7):
        TASK1_COUNT_SOFT_RATE per excess.
    n_pred > TASK1_COUNT_HARD_THRESHOLD: soft zone penalty + TASK1_COUNT_HARD_RATE per further excess.

    Env vars: TASK1_COUNT_SOFT_RATE (0.005), TASK1_COUNT_HARD_THRESHOLD (7),
              TASK1_COUNT_HARD_RATE (0.05), TASK1_COUNT_CAP (0.5),
              TASK1_COUNT_FREE_EXTRA (1).
    """
    return compute_count_penalty_v2_from_env(n_pred, n_gt)


def compute_rule_penalty(raw_response: str, result_text: str) -> Dict[str, float]:
    raw = raw_response or ""

    # Result-block length penalty (two tiers, additive per 100 tokens).
    res_t1 = int(os.environ.get("TASK1_LENGTH_THRESHOLD", "768"))
    res_r1 = float(os.environ.get("TASK1_LENGTH_PENALTY_RATE", "0.05"))
    res_t2 = int(os.environ.get("TASK1_LENGTH_THRESHOLD_TIER2", "0"))
    res_r2 = float(os.environ.get("TASK1_LENGTH_PENALTY_RATE_TIER2", "0"))
    token_count = approx_token_count(result_text)
    length_penalty = _two_tier_overlong_penalty(token_count, res_t1, res_r1, res_t2, res_r2)

    # Total-response length penalty (constrains <Think> bloat that result-only penalty cannot see).
    # Three-tier additive: soft nudge near SFT mean, medium past p95, hard brake on true bloat.
    total_t1 = int(os.environ.get("TASK1_TOTAL_LENGTH_THRESHOLD", "0"))
    total_r1 = float(os.environ.get("TASK1_TOTAL_LENGTH_PENALTY_RATE", "0"))
    total_t2 = int(os.environ.get("TASK1_TOTAL_LENGTH_THRESHOLD_TIER2", "0"))
    total_r2 = float(os.environ.get("TASK1_TOTAL_LENGTH_PENALTY_RATE_TIER2", "0"))
    total_t3 = int(os.environ.get("TASK1_TOTAL_LENGTH_THRESHOLD_TIER3", "0"))
    total_r3 = float(os.environ.get("TASK1_TOTAL_LENGTH_PENALTY_RATE_TIER3", "0"))
    total_token_count = approx_token_count(raw)
    length_penalty += _two_tier_overlong_penalty(
        total_token_count, total_t1, total_r1, total_t2, total_r2, total_t3, total_r3
    )

    return {
        "length_penalty": float(length_penalty),
        "token_count": float(token_count),
        "total_token_count": float(total_token_count),
    }


def _extract_full_rq_word_counts(result_text: str) -> List[int]:
    """Re-parse result_text and return word counts of each Research Question body
    WITHOUT the 400-char truncation that extract_numbered_bullets applies.
    Used by compute_rq_verbosity_penalty so verbose RQ stuffing is detected
    at its true length.
    """
    if not result_text:
        return []
    text = "\n" + result_text
    tm_matches = list(_TARGET_BULLET_RE.finditer(text))
    rq_matches = list(_RQ_BULLET_RE.finditer(text))
    counts = []
    for i, tm in enumerate(tm_matches):
        next_tm_start = tm_matches[i + 1].start() if i + 1 < len(tm_matches) else len(text)
        rq_in_range = next((rq for rq in rq_matches if tm.end() < rq.start() < next_tm_start), None)
        if rq_in_range is None:
            continue
        rq_body = text[rq_in_range.end():next_tm_start]
        counts.append(len(rq_body.split()))
    return counts


def compute_rq_verbosity_penalty(result_text: str) -> Dict[str, float]:
    """v11 per-bullet RQ verbosity penalty. Counters the "umbrella TM + verbose
    RQ stuffed with keywords" hack discovered in v10: bullets with broad TM names
    were writing 200-300 word RQs enumerating many keywords, causing the judge
    to award hierarchical-partial 0.5 on focus keyword overlap. Penalizing
    per-bullet RQ length encourages concise atomic ablation questions.

    Per bullet: if RQ exceeds threshold words, subtract a rate per 50 excess words.
    Sum across bullets, cap at TASK1_RQ_VERBOSITY_CAP.
    """
    threshold = int(os.environ.get("TASK1_RQ_VERBOSITY_THRESHOLD", "80"))
    rate = float(os.environ.get("TASK1_RQ_VERBOSITY_RATE", "0.05"))
    cap = float(os.environ.get("TASK1_RQ_VERBOSITY_CAP", "0.3"))

    rq_words_list = _extract_full_rq_word_counts(result_text)
    total_penalty = 0.0
    for rq_words in rq_words_list:
        excess = max(0, rq_words - threshold)
        total_penalty += (excess / 50.0) * rate
    capped = min(cap, total_penalty)
    mean_rq_words = (sum(rq_words_list) / len(rq_words_list)) if rq_words_list else 0.0
    max_rq_words = max(rq_words_list) if rq_words_list else 0
    return {
        "rq_verbosity_penalty": float(capped),
        "rq_verbosity_raw": float(total_penalty),
        "rq_words_mean": float(mean_rq_words),
        "rq_words_max": float(max_rq_words),
    }


# v16 doubling penalty — server-side detection of duplicate bullets in same paper
_DOUBLING_STOP = {
    'strategy','design','method','approach','framework','mechanism','protocol','system',
    'pipeline','algorithm','technique','procedure','evaluation','analysis','assessment',
    'validation','verification','robustness','generalization','adaptability','scalability',
    'efficiency','optimization','training','inference','composition','architecture',
    'configuration','setting','choice','selection','tuning','sensitivity','effectiveness',
    'performance','integrity','suitability','interaction','integration','combination',
    'components','component','module','modules','mechanisms','process','processes',
    'aspect','aspects','strategies','designs','techniques','approaches','methods',
    'function','functions','operation','operations',
    'term','terms','level','levels','rate','rates','size','sizes','range','ranges',
    'value','values','part','parts','number','numbers','version','versions','type','types',
    'and','with','for','the','in','of','to','on','from','by','vs','versus','via',
    'using','its','their','this','that','these','those','such',
    'each','some','any','all','one','two','three',
    'at','as','an','is','are','be','can','will','may','does',
}


def _doubling_core_words(tm: str) -> set:
    if not tm:
        return set()
    return set(
        w.lower() for w in re.split(r'[\W_]+', tm)
        if len(w) > 2 and w.lower() not in _DOUBLING_STOP
    )


def _doubling_normalize(tm: str) -> str:
    """lowercase, drop parentheticals, collapse whitespace — for substring containment."""
    s = (tm or "").lower()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[^a-z0-9\s]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _is_doubling_pair(tm_a: str, tm_b: str, threshold: float) -> bool:
    """Two-rule duplicate detection (v2 heuristic, validated at threshold=0.7
    on 22 hand-curated cases: precision 100%, recall 83% — no false positives
    on legitimate atomic decompositions like encoder/decoder, self/cross-attn,
    short-term/long-term memory, BiasNorm/LayerNorm).

    Rule 1: substring containment after normalization (both sides ≥8 chars).
    Rule 2: word-overlap ratio > threshold AND |overlap| >= 2 (filters single-
            word coincidence like shared 'image' / 'memory' / 'data').
    """
    na = _doubling_normalize(tm_a)
    nb = _doubling_normalize(tm_b)
    if len(na) >= 8 and len(nb) >= 8 and (na in nb or nb in na):
        return True
    ca = _doubling_core_words(tm_a)
    cb = _doubling_core_words(tm_b)
    if not ca or not cb:
        return False
    overlap = ca & cb
    if len(overlap) < 2:
        return False
    return (len(overlap) / min(len(ca), len(cb))) > threshold


def compute_doubling_penalty(result_text: str) -> Dict[str, float]:
    """v16 doubling penalty — penalize models that write multiple bullets
    covering the same paper component with different phrasing/sub-aspects.

    The bipartite-enforce already zeros reuse on the GT side, but doesn't
    discourage writing duplicates in the first place — model can write 5
    bullets including 1 duplicate pair, lose the duplicate on bipartite but
    keep the other 4. v16 adds a flat per-pair penalty to discourage this.

    Implementation walks bullets in order; each later bullet that duplicates
    an earlier (per _is_doubling_pair) accrues penalty. v18: rate per pair
    raised 0.05 → 0.06 to offset the softer v18 count_penalty (which had let
    doubling regress from v16's 2% to v17's 6.1% — multi-bullet became net
    positive again under v17 count_penalty=0.01 base). cap = 0.20.
    """
    threshold = float(os.environ.get("TASK1_DOUBLING_THRESHOLD", "0.7"))
    rate = float(os.environ.get("TASK1_DOUBLING_RATE", "0.06"))
    cap = float(os.environ.get("TASK1_DOUBLING_CAP", "0.20"))

    bullets = extract_numbered_bullets(result_text)
    seen = []
    n_doubles = 0
    examples = []
    for b in bullets:
        tm = b.get("target_module", "")
        is_dup = False
        for prev_tm in seen:
            if _is_doubling_pair(tm, prev_tm, threshold):
                is_dup = True
                if len(examples) < 3:
                    examples.append((prev_tm[:80], tm[:80]))
                break
        if is_dup:
            n_doubles += 1
        else:
            seen.append(tm)
    penalty = min(cap, rate * n_doubles)
    return {
        "doubling_penalty": float(penalty),
        "n_doubles": int(n_doubles),
        "doubling_examples": examples,
    }


def create_evaluation_prompt(
    paper_context: str,
    generated_result: str,
    candidate_items: List[Dict[str, Any]],
) -> str:
    # focus-only: description was paper-narrative + tables, which is noise for
    # judging whether the model's ablation objective overlaps with the canonical
    # focus phrase. weight is kept in candidate_items for downstream weighted
    # aggregation but not shown to the judge.
    candidate_block = "\n".join(
        [
            (
                f'<candidate num="{idx + 1}">\n'
                f'<focus>{item["focus"]}</focus>\n'
                f'</candidate>'
            )
            for idx, item in enumerate(candidate_items)
        ]
    )

    # v9: pre-parse the model's bullets and present them with explicit indices so
    # the judge can identify the primary_match by integer ID. This is the input
    # side of the server-side consumed-once enforcement.
    bullets = extract_numbered_bullets(generated_result)
    if bullets:
        bullets_block = "\n".join(
            [
                (
                    f'<bullet num="{b["idx"]}">\n'
                    f'<target_module>{b["target_module"]}</target_module>\n'
                    f'<research_question>{b["research_question"]}</research_question>\n'
                    f'</bullet>'
                )
                for b in bullets
            ]
        )
    else:
        bullets_block = "(no parseable bullets in the generated result)"

    binary = use_binary_scoring()
    if binary:
        score_format = "0|1"
    else:
        score_format = "0|0.5|1"

    return f"""You are a rigorous scientific reviewer evaluating ablation objective predictions against GT focuses.

<Paper_Context>
{paper_context}
</Paper_Context>

<Generated_Result>
{generated_result}
</Generated_Result>

<Generated_Bullets>
{bullets_block}
</Generated_Bullets>

<Reference_Focuses>
{candidate_block}
</Reference_Focuses>

EVALUATION PROCEDURE:
  Process the reference focuses 1..N IN ORDER. For each focus:
    1. Read the focus.
    2. Look at the UNCONSUMED candidate bullets (bullets not yet cited as primary_match by an earlier focus).
    3. For each candidate, judge whether its (Target Module + Research Question) addresses the same atomic ablation as the focus.
    4. Assign the score 0 / 0.5 / 1.0 per the rubric below. Record the bullet's integer ID in <primary_match> (or 0 if no bullet qualifies).
    5. A bullet picked as primary_match for the current focus is CONSUMED — do NOT cite it for any later focus.

ATOMIC ABLATION means: a single specific paper-introduced method, component, design choice, or experimental contrast (e.g., "Beam search width", "Prototype Attentive Module", "BiasNorm vs LayerNorm", "L_adv + L_unsup joint loss"). NOT a combo of abstract categories.

DUAL SIGNAL: Use BOTH the bullet's Target Module heading AND its Research Question text. TM tells you the bullet's main subject; RQ confirms (or disconfirms) it engages the focus's actual experimental concern. A bullet earns credit only when TM + RQ together point to the focus.

SCORING (v18 — tightened 0.5 + same-component 1.0):

  1.0 — ATOMIC MATCH ON SAME PAPER COMPONENT
        Award 1.0 in ANY of these cases (component identity matters more than verb identity):
          • DIRECT MATCH: TM clearly names the same atomic ablation as the focus AND RQ confirms the same experimental concern. Synonymous verbs are fine ("necessity" ≈ "contribution" ≈ "effect" ≈ "with-vs-without").
          • SINGLE-COMPONENT ATOMIZATION
            (Ex: focus="Effect of beam width on translation" ←→ TM="Beam search width" → 1.0)
          • COMPARISON ATOMIZATION: focus="A vs B" is ONE experiment; a bullet naming A, naming B, or naming the contrast all score 1.0.
            (Ex: focus="LoRA vs Fully Fine-tuning" ←→ TM="LoRA fine-tuning" with RQ about replacing full FT → 1.0)
          • JOINT COMPOSITE: focus names X AND Y as a paper-joint ablation, TM names BOTH.
            (Ex: focus="L_adv and L_unsup loss design" ←→ TM="L_adv + L_unsup joint loss" → 1.0)
          • SAME COMPONENT, DIFFERENT VALID ANGLE (v18 — NEW):
            TM names the same paper-specific component/mechanism as the focus, AND the RQ probes a different but legitimate experimental angle on that same component (sensitivity ↔ robustness ↔ necessity ↔ magnitude). The bullet must be testing the same paper artifact, just from another direction.
            (Ex: focus="Effect of Reference Corpus Size on NPM" ←→ TM="Reference Corpus Invariance" with RQ asking if model adapts when corpus is swapped → 1.0, because both ablate the reference corpus.)
            CONSTRAINT: This rule applies ONLY when the component is paper-specific (named module / mechanism / design choice). It does NOT promote generic-category bullets — "Encoder Architecture" or "Loss Function" cannot use this rule.

  0.5 — RESCUED MATCH (v18 — TIGHTENED, USE SPARINGLY)
        HARD PREREQUISITE: at least one of TM or RQ MUST mention a paper-specific term that the focus also references (a named module, mechanism, design choice, or technical detail visible in the focus or its description). If the bullet is generic enough to appear unchanged in any ML paper, score 0 — NOT 0.5. RQ keyword overlap with the focus alone is NOT sufficient — there must be paper-specific anchoring.

        Award 0.5 in the following cases (prereq must hold first):

        Case A — Vague TM rescued by precise RQ:
          TM is broader/vaguer than the focus, BUT the RQ contains a paper-specific term that locks the bullet to the focus's atomic concern.
          (Ex: focus="Effect of beam width" ←→ TM="Decoding parameter tuning" with RQ explicitly varying beam width on the paper's decoder → 0.5)

        Case B — Precise TM but generic RQ (v18 — NEW):
          TM exactly names the focus's paper-specific component, BUT the RQ is a generic template ("Is X critical for performance?" / "Does X improve results?") that adds no paper-specific detail beyond what the TM already says. The bullet shows it knows what to ablate but not how.
          (Ex: focus="MMD-based feature drift constraint" ←→ TM="MMD-based feature drift constraint" with RQ="Is the MMD constraint critical for performance?" → 0.5, because TM nails the component but RQ contributes no additional paper-specific anchor.)

        Case C — One side of TRUE joint/comparison:
          TM names ONE side of a TRUE joint ablation OR ONE side of a TRUE comparison, AND the RQ confirms the bullet IS engaging the focus's experimental point.
          (Ex: focus="Contribution of L_adv and L_unsup joint loss" ←→ TM="Adversarial loss" with RQ about removing GAN loss only → 0.5)

        Do NOT use 0.5 for "topically related" / "different aspect" / "higher abstraction" / "adjacent question" — those are 0.

  0.0 — NO MATCH. Apply 0 if ANY hold:

    (a) The bullet's Target Module is a GENERIC UMBRELLA — combines 2+ abstract method categories rather than naming paper-specific components.
        UMBRELLA examples (always 0):
          "Training Protocol and Optimization Strategy" / "Backbone Architecture and Network Design"
          "Loss Function and Regularization" / "Ablation and Sensitivity Analysis"
          "Reward and Exploration Signal Design" / "Mechanistic Analysis"
        NOT-UMBRELLA (paper-named, may qualify):
          "MeanNet and BiasNet" / "L_adv and L_unsup loss" / "Planning Algorithm (MCTS and P-UCB)"
        TEST: remove "and"/"with". Are remaining noun phrases SPECIFIC paper names? → not umbrella. Are they generic categories appearing in any paper? → umbrella.

    (b) The focus appears only inside a parenthetical enumeration "(e.g., A, B, ...)" of a bullet whose TM is about a different topic.

    (c) No unconsumed bullet specifically addresses the focus (different aspect / different component / vague catchall listing many components).

    (d) PURE PARAPHRASE (v18 — NEW): Both TM and RQ use only generic-category phrasings that could appear in many ML papers, even if topically aligned with the focus. There is no paper-specific anchor in either field — score 0, not 0.5.

SERVER-SIDE 1:1 ENFORCEMENT (safety net):
  Server enforces 1:1 by score. If you cite the same bullet for two focuses, server keeps the higher-scoring focus and zeros the other. Follow the procedure above strictly to avoid this — pick the next-best unconsumed bullet (or primary_match=0).

============================================================
SECTION B — PER-BULLET SPECIFICITY (v18.1 precision input)
============================================================
After scoring all focuses above, score each generated bullet ONCE for paper-specificity, INDEPENDENT of whether it matched any focus. Evaluate the (TM, RQ) pair as a unit:

  1.0 — SPECIFIC & VALID
        TM names a concrete paper-specific component/mechanism/design choice identifiable in <Paper_Context> (NOT a generic category like "Training Objective", "Encoder", "Attention Module"). RQ asks a meaningful non-trivial question about it.

  0.5 — RQ REDEEMS GENERIC TM
        TM uses a generic category-level name, BUT the RQ contains paper-specific technical terms that unambiguously identify which exact mechanism is being targeted (effectively supplying the specificity the TM lacks). The RQ must reference a concrete paper detail, not merely be detailed/wordy.

  0.0 — GENERIC OR INVALID
        Either: (i) both TM and RQ use generic language applicable to any ML paper; (ii) TM is a category name AND RQ fails to identify the specific mechanism; or (iii) the prediction is irrelevant to the paper's methodology.

When in doubt between 0.5 and 0.0, choose 0.0.

Output Format:
<Evaluation>
<candidate num="1">
<critique>Brief: which bullet is the match (cite TM + RQ evidence) and why (or why none).</critique>
<primary_match>integer bullet ID, or 0 if no match</primary_match>
<score>{score_format}</score>
</candidate>
...
</Evaluation>

<Bullet_Specificity>
<bullet num="1">
<spec_reason>One sentence: paper-specific anchor (or its absence) in TM and/or RQ.</spec_reason>
<spec_score>{score_format}</spec_score>
</bullet>
...
</Bullet_Specificity>

Return only the XML — both blocks.
"""


def parse_score(value: str) -> Optional[float]:
    cleaned = (value or "").strip().strip("[](){}").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        score = float(match.group(0))
    except Exception:
        return None
    if use_binary_scoring():
        allowed = (0.0, 1.0)
    else:
        allowed = (0.0, 0.5, 1.0)
    return min(allowed, key=lambda a: (abs(a - score), a))


def parse_eval_response(response: str, candidate_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    text = response or ""
    for idx, item in enumerate(candidate_items, start=1):
        pattern = (
            rf'<\s*candidate\b[^>]*(?:num|number|id)\s*=\s*["\']?\s*{idx}\s*["\']?[^>]*>'
            rf'(.*?)</\s*candidate\s*>'
        )
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        critique = ""
        score = None
        primary_match: Optional[int] = None
        if match:
            block = match.group(1)
            critique_match = re.search(
                r"<\s*critique\s*>(.*?)</\s*critique\s*>",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            score_match = re.search(
                r"<\s*score\s*>(.*?)</\s*score\s*>",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            pm_match = re.search(
                r"<\s*primary_match\s*>(.*?)</\s*primary_match\s*>",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            critique = critique_match.group(1).strip() if critique_match else ""
            score = parse_score(score_match.group(1).strip()) if score_match else None
            if pm_match:
                pm_int_match = re.search(r"-?\d+", pm_match.group(1))
                if pm_int_match:
                    try:
                        primary_match = int(pm_int_match.group(0))
                    except ValueError:
                        primary_match = None
        results.append(
            {
                "candidate_num": idx,
                "focus": item["focus"],
                "weight": item["weight"],
                "critique": critique,
                "score": score,
                "primary_match": primary_match,
            }
        )
    return results


def parse_bullet_specificity(response: str, n_bullets: int) -> Dict[int, float]:
    """Extract per-bullet specificity scores from <Bullet_Specificity> block.

    Returns a dict {bullet_id (1-indexed): score in {0.0, 0.5, 1.0}}.
    Missing bullets default to 1.0 to avoid wt(P) regression when the judge
    forgets the block — caller should treat None vs 1.0 as ambiguous.
    """
    text = response or ""
    out: Dict[int, float] = {}
    block_match = re.search(
        r"<\s*Bullet_Specificity\s*>(.*?)</\s*Bullet_Specificity\s*>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    block = block_match.group(1) if block_match else text  # fall back to whole text

    for idx in range(1, n_bullets + 1):
        pattern = (
            rf'<\s*bullet\b[^>]*(?:num|number|id)\s*=\s*["\']?\s*{idx}\s*["\']?[^>]*>'
            rf'(.*?)</\s*bullet\s*>'
        )
        m = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        sm = re.search(
            r"<\s*spec_score\s*>(.*?)</\s*spec_score\s*>",
            m.group(1),
            re.DOTALL | re.IGNORECASE,
        )
        if sm:
            sc = parse_score(sm.group(1).strip())
            if sc is not None:
                out[idx] = sc
    return out


_judge_client: Optional[AsyncOpenAI] = None
_judge_semaphore: Optional[asyncio.Semaphore] = None


def _get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is not None:
        return _judge_client

    base_url = os.environ.get("JUDGE_API_BASE")
    api_key = os.environ.get("JUDGE_API_KEY")

    if not base_url:
        raise RuntimeError("JUDGE_API_BASE is not set.")
    if not api_key:
        raise RuntimeError("JUDGE_API_KEY is not set.")

    _judge_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=int(os.environ.get("TASK1_JUDGE_REQUEST_TIMEOUT", "300")),
        max_retries=0,
    )
    return _judge_client


def _get_judge_semaphore() -> asyncio.Semaphore:
    global _judge_semaphore
    if _judge_semaphore is None:
        max_concurrent = int(os.environ.get("TASK1_JUDGE_MAX_CONCURRENT", "64"))
        _judge_semaphore = asyncio.Semaphore(max_concurrent)
    return _judge_semaphore


def _get_judge_extra_body() -> Dict[str, Any]:
    raw = os.environ.get("TASK1_JUDGE_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("TASK1_JUDGE_EXTRA_BODY must decode to a JSON object")
        return parsed
    except Exception as exc:
        logger.warning("ignoring malformed TASK1_JUDGE_EXTRA_BODY (%s): %s", exc, raw)
        return {}


async def _call_judge_chat(messages: List[Dict[str, Any]]) -> str:
    model = os.environ.get("JUDGE_MODEL")
    if not model:
        raise RuntimeError("JUDGE_MODEL is not set.")

    max_retries = int(os.environ.get("TASK1_JUDGE_MAX_RETRIES", "5"))
    max_output_tokens = int(os.environ.get("TASK1_JUDGE_MAX_OUTPUT_TOKENS", "1500"))
    extra_body = _get_judge_extra_body()
    client = _get_judge_client()
    semaphore = _get_judge_semaphore()

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=0,
                    extra_body=extra_body or None,
                )
            if not response.choices:
                raise RuntimeError(f"judge chat.completions returned no choices: {response}")
            content = response.choices[0].message.content or ""
            content = content.strip()
            if not content:
                raise RuntimeError(f"judge chat.completions returned empty content: {response}")
            return content
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"judge chat.completions request failed after {max_retries} attempts: {last_error}")


async def call_llm_judge(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a rigorous scientific reviewer. Return only the requested XML.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return await _call_judge_chat(messages)


async def build_reward_response(raw_request: Dict[str, Any]) -> Dict[str, Any]:
    extra_info = coerce_extra_info(raw_request.get("extra_info"))
    raw_response = raw_request.get("response_str", "")
    if isinstance(raw_response, list):
        raw_response = raw_response[0] if raw_response else ""

    result_text = extract_result_block(raw_response)
    candidate_items = parse_candidates(extra_info.get("candidates", ""))
    if not candidate_items:
        return {
            "score": 0.0,
            "candidate_score": 0.0,
            "format_factor": 0.0,
            "format_reason": "no_candidate_items",
            "n_pairs": 0,
            "length_penalty": 0.0,
            "token_count": 0.0,
            "error": "no_candidate_items",
        }

    prompt = create_evaluation_prompt(
        paper_context=extra_info.get("paper_context", ""),
        generated_result=result_text,
        candidate_items=candidate_items,
    )

    # Validation retry: re-call judge if too many items have invalid/missing scores.
    max_validation_retries = int(os.environ.get("TASK1_JUDGE_VALIDATION_RETRIES", "2"))
    miss_threshold = max(1, len(candidate_items) // 2)

    raw_judge_output = ""
    eval_items: List[Dict[str, Any]] = []
    missing = 0
    for attempt in range(max_validation_retries + 1):
        raw_judge_output = await call_llm_judge(prompt)
        eval_items = parse_eval_response(raw_judge_output, candidate_items)
        missing = sum(1 for it in eval_items if it.get("score") is None)
        if missing == 0:
            break
        if attempt < max_validation_retries and missing >= miss_threshold:
            logger.warning(
                "validation retry %d/%d: %d/%d candidates missing or invalid (threshold %d)",
                attempt + 1, max_validation_retries, missing, len(eval_items), miss_threshold,
            )
            continue
        break

    if missing:
        logger.warning(
            "final judge result missed %d/%d candidates after %d attempts",
            missing, len(eval_items), max_validation_retries + 1,
        )

    fmt = compute_format_factor(raw_response=raw_response, result_text=result_text)

    # v12: score-based 1:1 bipartite enforce. When multiple focuses cite the
    # same bullet as primary_match, keep only the highest-scoring focus; the
    # other re-citations are zeroed BEFORE weighted_score aggregation. This is
    # the structural fix for the "reuse + umbrella + multi-version" hack stack
    # discovered in v7-v11. Replaces v9 order-greedy enforce and v10/v11
    # telemetry-only no-op.
    reuse_stats = enforce_bipartite(eval_items, n_pred_bullets=int(fmt["n_pairs"]))

    # v18.1: parse per-bullet specificity and compute wt(P) recall as the main
    # candidate score. Plain recall is kept as a secondary telemetry field.
    bullet_spec = parse_bullet_specificity(raw_judge_output, n_bullets=int(fmt["n_pairs"]))
    candidate_score_plain = compute_weighted_score(eval_items)
    candidate_score = compute_wt_p_recall(eval_items, bullet_spec)
    if candidate_score is None:
        candidate_score = 0.0
    if candidate_score_plain is None:
        candidate_score_plain = 0.0
    penalties = compute_rule_penalty(raw_response=raw_response, result_text=result_text)
    rq_verb = compute_rq_verbosity_penalty(result_text=result_text)
    doubling = compute_doubling_penalty(result_text=result_text)
    gated_candidate = float(candidate_score) * fmt["factor"]
    count_penalty = compute_count_penalty(n_pred=fmt["n_pairs"], n_gt=len(candidate_items))
    total_score = max(
        0.0,
        gated_candidate
        - penalties["length_penalty"]
        - count_penalty
        - rq_verb["rq_verbosity_penalty"]
        - doubling["doubling_penalty"],
    )

    return {
        "score": total_score,
        "candidate_score": float(candidate_score),
        "candidate_score_plain": float(candidate_score_plain),
        "format_factor": fmt["factor"],
        "format_reason": fmt["reason"],
        "n_pairs": fmt["n_pairs"],
        "n_gt": len(candidate_items),
        "count_penalty": count_penalty,
        "length_penalty": penalties["length_penalty"],
        "rq_verbosity_penalty": rq_verb["rq_verbosity_penalty"],
        "rq_words_mean": rq_verb["rq_words_mean"],
        "rq_words_max": rq_verb["rq_words_max"],
        "doubling_penalty": doubling["doubling_penalty"],
        "n_doubles": doubling["n_doubles"],
        "token_count": penalties["token_count"],
        "total_token_count": penalties.get("total_token_count", 0.0),
        "missing_candidates": missing,
        "n_reuse_observed": reuse_stats["n_reuse_observed"],
        "n_reuse_zeroed": reuse_stats["n_reuse_zeroed"],
        "n_bullets_used": reuse_stats["n_bullets_used"],
        "n_bullet_spec_missing": int(fmt["n_pairs"]) - len(bullet_spec),
        "eval_items": eval_items,
        "bullet_spec": bullet_spec,
        "raw_judge_output": raw_judge_output,
    }


@app.post("/get_reward_task1")
async def get_reward_task1(request: Request):
    json_data = await request.json()
    result = await build_reward_response(json_data)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "binary_scoring": use_binary_scoring()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TASK1_REWARD_PORT", os.environ.get("TASK1_AZURE_REWARD_PORT", "6013")))
    uvicorn.run(app, host="0.0.0.0", port=port)
