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

PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from task1_candidate_utils import (  # noqa: E402
    approx_token_count,
    coerce_extra_info,
    compute_weighted_score,
    extract_result_block,
    parse_candidates,
)


app = FastAPI()


def use_binary_scoring() -> bool:
    value = (os.environ.get("TASK1_BINARY_SCORING", "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def rule_penalty_disabled() -> bool:
    """Disable additive structural penalties for ablation studies.

    The multiplicative format factor remains active because it gates
    parseability rather than applying a soft penalty.
    """
    return _truthy_env("TASK1_DISABLE_RULE_PENALTY")


# Task 1 scores a list of target modules/components that should be tested by
# ablation. The judge compares each predicted target against paper-specific
# ground-truth candidate focuses.
_TARGET_BULLET_RE = re.compile(
    r'\n\s*[-*]\s*(?:\*\*)?Target Module(?:\*\*)?\s*:', re.IGNORECASE)


def has_expected_format(result_text: str) -> bool:
    text = "\n" + (result_text or "")
    return bool(_TARGET_BULLET_RE.search(text))


def count_objective_pairs(result_text: str) -> int:
    """Count parseable Task 1 target bullets."""
    text = "\n" + (result_text or "")
    return len(_TARGET_BULLET_RE.findall(text))


def extract_numbered_bullets(result_text: str) -> List[Dict[str, Any]]:
    """Parse Task 1 target bullets into ordered entries."""
    if not result_text:
        return []
    text = "\n" + result_text
    tm_matches = list(_TARGET_BULLET_RE.finditer(text))
    bullets: List[Dict[str, Any]] = []
    for i, tm in enumerate(tm_matches):
        tm_start = tm.end()
        next_tm_start = tm_matches[i + 1].start() if i + 1 < len(tm_matches) else len(text)
        tm_text = text[tm_start:next_tm_start].strip().lstrip(":").strip().rstrip("*").strip()
        if not tm_text:
            continue
        bullets.append({
            "idx": len(bullets) + 1,
            "target_module": tm_text[:300],
            "research_question": "",
        })
    return bullets


def enforce_bipartite(
    eval_items: List[Dict[str, Any]],
    n_pred_bullets: int,
) -> Dict[str, Any]:
    """Enforce one-to-one assignment between GT focuses and predicted bullets.

    When multiple GT focuses cite the same predicted bullet, only the
    highest-scoring focus keeps the match; the rest are zeroed. This prevents a
    single broad bullet from receiving credit for many distinct targets.
    Mutates eval_items in place and returns telemetry.
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
    """Multiplicative parseability gate in [0, 1]."""
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
    """Progressive penalty for predicting too many targets.

    The first excess targets are cheap, while later excess targets cost more.
    This discourages broad over-generation without forcing a fixed number of
    predictions for every paper.
    """
    if rule_penalty_disabled():
        return 0.0
    tol = int(os.environ.get("TASK1_COUNT_TOL", "2"))
    base_rate = float(os.environ.get("TASK1_COUNT_BASE_RATE", "0.02"))
    step = float(os.environ.get("TASK1_COUNT_STEP", "0.01"))
    cap = float(os.environ.get("TASK1_COUNT_CAP", "0.5"))
    excess = max(0, int(n_pred) - int(n_gt) - tol)
    if excess <= 0:
        return 0.0
    pen = sum(base_rate + step * (i - 1) for i in range(1, excess + 1))
    return min(cap, pen)


def compute_rule_penalty(raw_response: str, result_text: str) -> Dict[str, float]:
    raw = raw_response or ""

    if rule_penalty_disabled():
        return {
            "length_penalty": 0.0,
            "token_count": float(approx_token_count(result_text)),
            "total_token_count": float(approx_token_count(raw)),
        }

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


def compute_rq_verbosity_penalty(result_text: str) -> Dict[str, float]:
    """Compatibility no-op for older reward aggregation fields."""
    return {
        "rq_verbosity_penalty": 0.0,
        "rq_verbosity_raw": 0.0,
        "rq_words_mean": 0.0,
        "rq_words_max": 0.0,
    }


# Duplicate-bullet detection for outputs that repeat the same component with
# slightly different wording.
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
    """Detect duplicate target bullets by containment or high word overlap."""
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
    """Penalize multiple bullets that cover the same paper component."""
    if rule_penalty_disabled():
        return {
            "doubling_penalty": 0.0,
            "n_doubles": 0,
            "doubling_examples": [],
        }

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

    # Target Module: pre-parse bullets and present with integer indices. RQ field
    # is dropped — the judge has only Target Module text + Paper Context to
    # match against GT Reference Focuses.
    bullets = extract_numbered_bullets(generated_result)
    if bullets:
        bullets_block = "\n".join(
            [
                (
                    f'<bullet num="{b["idx"]}">\n'
                    f'<target_module>{b["target_module"]}</target_module>\n'
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
    3. For each candidate, judge whether its Target Module addresses the same atomic ablation as the focus, using the Paper_Context to disambiguate when the TM heading is short.
    4. Assign the score 0 / 0.5 / 1.0 per the rubric below. Record the bullet's integer ID in <primary_match> (or 0 if no bullet qualifies).
    5. A bullet picked as primary_match for the current focus is CONSUMED — do NOT cite it for any later focus.

ATOMIC ABLATION means: a single specific paper-introduced method, component, design choice, or experimental contrast (e.g., "Beam search width", "Prototype Attentive Module", "BiasNorm vs LayerNorm", "L_adv + L_unsup joint loss"). NOT a combo of abstract categories.

TASK 1 MATCHING: Each bullet is a Target Module heading. Match by whether the heading names the same atomic ablation as the focus. Use the Paper_Context to resolve ambiguity when the phrasing is broad.

SCORING:

  1.0 — DEDICATED ATOMIC MATCH
        TM clearly names the same atomic ablation as the focus. Phrasing differences are fine (still 1.0):
          • synonymous verbs ("necessity" ≈ "contribution" ≈ "effect" ≈ "with-vs-without")
          • TM atomizes a single-component focus (Example: focus="Effect of beam width on translation" ←→ TM="Beam search width" → 1.0)
          • Comparison atomization: focus="A vs B" is ONE experiment; a bullet naming A, naming B, or naming the contrast all score 1.0.
            (Example: focus="LoRA vs Fully Fine-tuning" ←→ TM="LoRA fine-tuning" → 1.0)
          • Joint composite: focus names X AND Y as a paper-joint ablation, TM names BOTH.
            (Example: focus="L_adv and L_unsup loss design" ←→ TM="L_adv + L_unsup joint loss" → 1.0)

  0.5 — NEAR-1.0 WITH MINOR PHRASING FLAW (USE SPARINGLY)
        Reserve for cases where you have HIGH CONFIDENCE the bullet is about the SAME atomic ablation as the focus, but the TM heading has a small imperfection:
          • TM is slightly broader/vaguer in wording than the focus, BUT in the Paper_Context this TM uniquely maps to the focus's experimental concern.
            (Example: focus="Effect of beam width on translation" ←→ TM="Decoding parameter tuning" where the paper only varies beam width → 0.5)
          • TM names only ONE side of a TRUE joint ablation OR ONE side of a TRUE comparison.
            (Example: focus="Contribution of L_adv and L_unsup joint loss" ←→ TM="Adversarial loss" → 0.5)

        CRITICAL — 0.5 is closer to 1.0 than to 0. Give 0.5 only when you would give 1.0 if the TM phrasing were slightly cleaner. When in doubt:
          • If the TM does NOT uniquely point to the focus's concern (under Paper_Context) → score 0, NOT 0.5.
          • If the TM is a GENERIC UMBRELLA (see 0.0 (a) below) → score 0.
          • Do NOT use 0.5 for "topically related" / "different framing" / "higher abstraction" / "adjacent question" — those are 0.

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

    (c) No unconsumed bullet specifically addresses the focus.

SERVER-SIDE 1:1 ENFORCEMENT (safety net):
  Server enforces 1:1 by score. If you cite the same bullet for two focuses, server keeps the higher-scoring focus and zeros the other. Follow the procedure above strictly to avoid this — pick the next-best unconsumed bullet (or primary_match=0).

Output Format:
<Evaluation>
<candidate num="1">
<critique>Brief: which bullet is the match (cite TM heading and Paper_Context evidence) and why (or why none).</critique>
<primary_match>integer bullet ID, or 0 if no match</primary_match>
<score>{score_format}</score>
</candidate>
...
</Evaluation>

Return only the XML.
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

    # Enforce one-to-one matching before aggregation: when multiple focuses cite
    # the same bullet, only the highest-scoring focus keeps the match.
    reuse_stats = enforce_bipartite(eval_items, n_pred_bullets=int(fmt["n_pairs"]))

    candidate_score = compute_weighted_score(eval_items)
    if candidate_score is None:
        candidate_score = 0.0
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
        "eval_items": eval_items,
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

    port = int(os.environ.get("TASK1_REWARD_PORT", "6013"))
    uvicorn.run(app, host="0.0.0.0", port=port)
