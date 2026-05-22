#!/usr/bin/env python3
"""
OpenAI-compatible API evaluator for Task 1 (Ablation Objective Identification).

This eval mirrors the Task 1 reward judge prompt shipped in
reward/task1_reward.py. Differences vs. the RL reward judge are intentional:

  * eval input has no <Paper_Context>, so that block is dropped
  * judge output is JSON instead of per-candidate XML
  * server-side enforce_bipartite is reproduced post-hoc in parse_match_response
    (score-desc + gt-asc → 1:1 dedup)
  * count_penalty parameters are kept independent of the RL reward (see below)

Scoring rubric:
  * 1.0 — DEDICATED ATOMIC MATCH (includes comparison atomization and joint
          composite target headings that name both sides)
  * 0.5 — NEAR-1.0 WITH MINOR PHRASING FLAW.
  * 0.0 — NO MATCH (umbrella target / parenthetical-only mention / no unconsumed
          bullet addresses the focus / vague catchall / different aspect)

Usage:
    JUDGE_API_BASE=https://api.openai.com/v1 JUDGE_API_KEY=... JUDGE_MODEL=... \
      python evaluation/eval_task1.py --input <infer.jsonl>
"""

import argparse
import concurrent.futures
import json
import os
import re
import time
from threading import Lock
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


# ======================== Config ========================
JUDGE_TIMEOUT = 300
WORKERS = 3

JUDGE_MODEL: Optional[str] = None
JUDGE_CLIENT: Optional[OpenAI] = None
write_lock = Lock()


# ======================== Evaluation Prompt ========================
#
# Direct adaptation of reward/task1_reward.py create_evaluation_prompt with
# three eval-side adaptations:
#   1. No <Paper_Context> — eval inputs do not carry it.
#   2. JSON output instead of per-candidate XML.
#   3. Bipartite 1:1 enforce is done post-hoc by parse_match_response.

EVAL_PROMPT = """You are a rigorous scientific reviewer evaluating ablation objective predictions against GT focuses.

<Reference_Focuses>
{GT_OBJECTIVES}
</Reference_Focuses>

<Generated_Bullets>
{PRED_OBJECTIVES}
</Generated_Bullets>

EVALUATION PROCEDURE:
  Process the reference focuses 1..N IN ORDER. For each focus:
    1. Read the focus.
    2. Look at the UNCONSUMED candidate bullets (bullets not yet cited as primary_match by an earlier focus).
    3. For each candidate, judge whether its Target Module names the same atomic ablation as the focus.
    4. Assign the score 0 / 0.5 / 1.0 per the rubric below. Record the bullet's integer ID in primary_match (or 0 if no bullet qualifies).
    5. A bullet picked as primary_match for the current focus is CONSUMED — do NOT cite it for any later focus.

ATOMIC ABLATION means: a single specific paper-introduced method, component, design choice, or experimental contrast (e.g., "Beam search width", "Prototype Attentive Module", "BiasNorm vs LayerNorm", "L_adv + L_unsup joint loss"). NOT a combo of abstract categories.

TARGET-MODULE MATCHING: Each bullet is just a Target Module heading. Match by whether the target module names the same atomic ablation as the focus. There is no Research Question to disambiguate the bullet; judge purely from the target-module string.

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
          • TM is slightly broader/vaguer in wording than the focus, but unambiguously points to the same paper-level concern.
            (Example: focus="Effect of beam width on translation" ←→ TM="Decoding parameter tuning" → 0.5)
          • TM names only ONE side of a TRUE joint ablation OR ONE side of a TRUE comparison.
            (Example: focus="Contribution of L_adv and L_unsup joint loss" ←→ TM="Adversarial loss" → 0.5)

        CRITICAL — 0.5 is closer to 1.0 than to 0. Give 0.5 only when you would give 1.0 if the TM phrasing were slightly cleaner. When in doubt:
          • If the TM does NOT clearly point to the focus's concern → score 0, NOT 0.5.
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

    (c) No unconsumed bullet specifically addresses the focus (different aspect / different component / vague catchall listing many components).

1:1 ENFORCEMENT (post-hoc by evaluator):
  After your scoring, the evaluator script keeps each bullet's pairing only with the HIGHEST-scoring focus (tie-break: earlier focus wins). If you cite the same bullet for two focuses, the lower-scoring focus will be zeroed. Follow the procedure above strictly — pick the next-best unconsumed bullet, or primary_match=0.

**Output Format (strictly this JSON, no markdown fences, no commentary):**
{{
  "matches": [
    {{"gt_id": 1, "primary_match": 3, "score": 1.0,
      "reason": "<one sentence: cite TM + RQ evidence and why this score>"}}
  ],
  "unmatched_gt": [2]
}}"""


# ======================== IO ========================

def load_jsonl(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data


def append_to_jsonl(filename, data):
    with write_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()


def get_title(item):
    return (item.get("meta", {}).get("title", "") or "").strip()


def get_done_titles(*files):
    done = set()
    for f in files:
        for d in load_jsonl(f):
            t = (d.get("meta") or {}).get("title")
            if t:
                done.add(t.strip())
    return done


# ======================== Parsing ========================

# Target Module branch: only Target Module bullets are emitted/parsed; RQ is dropped.
# Mirrors reward/task1_reward.py for code-level consistency.
_TARGET_BULLET_RE = re.compile(
    r'\n\s*[-*]\s*(?:\*\*)?Target Module(?:\*\*)?\s*:', re.IGNORECASE)


def parse_gt_objectives(gt_candidates: str) -> List[str]:
    """Extract each <Investigation_Focus> from the gt_Candidates XML."""
    focuses = re.findall(
        r'<Investigation_Focus>(.*?)</Investigation_Focus>',
        gt_candidates,
        re.DOTALL,
    )
    return [f.strip() for f in focuses if f.strip()]


def parse_predicted_bullets(infer_response: str) -> List[Dict[str, str]]:
    """Target Module: parse infer response into ordered Target Module bullets.
    `research_question` is kept as an empty string for downstream schema
    compatibility but is never consumed by the judge prompt below.
    """
    if not infer_response:
        return []
    result_match = re.search(r'<Result>(.*?)(?:</Result>|$)', infer_response, re.DOTALL)
    result_text = result_match.group(1) if result_match else infer_response

    text = "\n" + result_text
    tm_matches = list(_TARGET_BULLET_RE.finditer(text))
    bullets: List[Dict[str, str]] = []
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


def format_pred_bullets(bullets: List[Dict[str, str]]) -> str:
    """Target Module: only emit <target_module>; no <research_question> element."""
    if not bullets:
        return "(no parseable bullets in the generated result)"
    parts = []
    for b in bullets:
        parts.append(
            f'<bullet num="{b["idx"]}">\n'
            f'<target_module>{b["target_module"]}</target_module>\n'
            f'</bullet>'
        )
    return "\n".join(parts)


def parse_match_response(response: str, n_gt: int, n_pred: int) -> Dict:
    """Parse judge JSON and apply bipartite 1:1 enforce (score-desc + gt-asc).

    Mirrors reward/task1_reward.py:enforce_bipartite — when two
    focuses cite the same bullet, keep the higher-scoring focus; tie-break by
    earlier focus position. Replaces the older `seen_gt`/`seen_pred` dedup that
    was order-dependent on the judge's output order.
    """
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return {"matches": [], "parse_error": "no JSON found", "raw": response[-500:]}

    try:
        result = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        return {"matches": [], "parse_error": str(e), "raw": response[-500:]}

    raw_matches = result.get("matches") or result.get("scores") or []
    # First pass: validate ranges, drop garbage, accept score==0 (kept for telemetry,
    # filtered later by score>0 when computing recall).
    candidates: List[Dict] = []
    for m in raw_matches:
        gt_id = m.get("gt_id")
        # Accept either primary_match or pred_id as the bullet pointer.
        pred_id = m.get("primary_match")
        if pred_id is None:
            pred_id = m.get("pred_id")
        score = m.get("score", 0)
        if not isinstance(gt_id, int) or gt_id < 1 or gt_id > n_gt:
            continue
        if not isinstance(score, (int, float)):
            continue
        # Normalize to {0, 0.5, 1.0}.
        score = float(score)
        score = min((0.0, 0.5, 1.0), key=lambda a: abs(a - score))
        # primary_match == 0 (or out of range) means "no bullet for this GT".
        if not isinstance(pred_id, int) or pred_id < 1 or pred_id > n_pred:
            pred_id = 0
            score = 0.0
        candidates.append({
            "gt_id": gt_id,
            "pred_id": pred_id,
            "score": score,
            "reason": m.get("reason", ""),
        })

    # Second pass: bipartite enforce — keep highest score per pred_id, then
    # one match per gt_id. Sort by (score desc, gt_id asc); both kept-checks
    # operate on the same sort order so the score-winning focus survives even
    # if the judge listed it later.
    candidates.sort(key=lambda x: (-x["score"], x["gt_id"]))
    seen_gt = set()
    seen_pred = set()
    valid: List[Dict] = []
    for c in candidates:
        if c["score"] <= 0 or c["pred_id"] == 0:
            continue
        if c["gt_id"] in seen_gt or c["pred_id"] in seen_pred:
            continue
        seen_gt.add(c["gt_id"])
        seen_pred.add(c["pred_id"])
        valid.append(c)

    # Re-sort matches by gt_id for stable downstream presentation.
    valid.sort(key=lambda x: x["gt_id"])
    result["matches"] = valid
    return result


def compute_match_rate(matches: List[Dict], n_gt: int) -> Optional[float]:
    if n_gt == 0:
        return None
    return sum(m.get("score", 1.0) for m in matches) / n_gt


# Count penalty is kept independent from the RL reward so evaluation remains a
# stable comparator across runs.
COUNT_PENALTY_TOLERANCE = 1
COUNT_PENALTY_RATE = 0.05
COUNT_PENALTY_CAP = 0.5


def compute_count_penalty(n_pred: int, n_gt: int) -> float:
    excess = max(0, n_pred - n_gt - COUNT_PENALTY_TOLERANCE)
    return min(COUNT_PENALTY_CAP, excess * COUNT_PENALTY_RATE)


# ======================== Judge API ========================

def call_judge(prompt_text: str, label: str = "", timeout: int = JUDGE_TIMEOUT) -> Optional[str]:
    tag = f"[{label}] " if label else ""
    if JUDGE_CLIENT is None or not JUDGE_MODEL:
        raise RuntimeError("Judge client/model is not configured")

    try:
        t0 = time.time()
        response = JUDGE_CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=2048,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        output = (response.choices[0].message.content or "").strip()
        if not output:
            print(f"  {tag}empty response ({elapsed:.1f}s)")
            return None
        print(f"  {tag}ok ({elapsed:.1f}s, {len(output)} chars)")
        return output

    except Exception as e:
        print(f"  {tag}error: {e}")
        return None


# ======================== Per-item evaluation ========================

def evaluate_item(item: dict) -> Tuple[str, dict]:
    title = get_title(item) or "?"
    tag = title[:30]
    meta = item.get("meta", {})

    try:
        gt_candidates = item.get("gt_Candidates", "")
        infer_response = item.get("infer_task1_response", "")

        if not gt_candidates or not infer_response:
            return ("FAIL", {"meta": meta,
                             "reason": "missing gt_Candidates or infer_task1_response"})

        gt_objectives = parse_gt_objectives(gt_candidates)
        if not gt_objectives:
            return ("FAIL", {"meta": meta,
                             "reason": "no GT objectives parsed from gt_Candidates"})

        pred_bullets = parse_predicted_bullets(infer_response)
        if not pred_bullets:
            return ("FAIL", {"meta": meta,
                             "reason": "no predicted bullets parsed from infer_task1_response"})

        gt_block = "\n".join(
            f'<focus num="{i+1}">{obj}</focus>'
            for i, obj in enumerate(gt_objectives)
        )
        pred_block = format_pred_bullets(pred_bullets)

        prompt = (EVAL_PROMPT
                  .replace("{GT_OBJECTIVES}", gt_block)
                  .replace("{PRED_OBJECTIVES}", pred_block))

        resp = call_judge(prompt, label=f"EVALt1|{tag}")
        if not resp:
            return ("FAIL", {"meta": meta, "reason": "judge call failed"})

        n_gt = len(gt_objectives)
        n_pred = len(pred_bullets)
        match_result = parse_match_response(resp, n_gt, n_pred)

        if "parse_error" in match_result:
            return ("FAIL", {
                "meta": meta,
                "reason": f"JSON parse error: {match_result['parse_error']}",
                "raw_response_tail": resp[-500:],
            })

        matches = match_result.get("matches", [])
        n_matched_full = sum(1 for m in matches if m.get("score", 0) >= 0.99)
        n_matched_partial = sum(1 for m in matches if 0.3 < m.get("score", 0) < 0.99)
        weighted_match_sum = sum(m.get("score", 0) for m in matches)
        recall = weighted_match_sum / n_gt if n_gt else 0.0
        count_penalty = compute_count_penalty(n_pred, n_gt)
        adjusted_score = max(0.0, recall - count_penalty)

        return ("SUCCESS", {
            "meta": meta,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_objectives": gt_objectives,
            "pred_bullets": pred_bullets,
            "matches": matches,
            "unmatched_gt": match_result.get("unmatched_gt", []),
            "n_matched_full": n_matched_full,
            "n_matched_partial": n_matched_partial,
            "n_matched": len(matches),
            "weighted_match_sum": weighted_match_sum,
            "match_rate": recall,
            "count_penalty": count_penalty,
            "adjusted_score": adjusted_score,
            "raw_eval_response": resp,
        })

    except Exception as e:
        return ("FAIL", {"meta": meta, "reason": f"exception: {e}"})


# ======================== Runner ========================

def run_eval(input_file: str, output_file: str, fail_file: str, limit: int = 0):
    items = load_jsonl(input_file)
    if limit > 0:
        items = items[:limit]
    done = get_done_titles(output_file, fail_file)
    pending = [it for it in items if get_title(it) not in done]

    print(f"Task 1 Eval: total={len(items)}, done={len(done)}, "
          f"pending={len(pending)}, workers={WORKERS}")
    print(f"Model: {JUDGE_MODEL}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file} / {fail_file}")

    if not pending:
        print("All done!")
        return

    success_count = fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {
            executor.submit(evaluate_item, item): item
            for item in pending
        }

        for future in concurrent.futures.as_completed(future_map):
            item = future_map[future]
            title = get_title(item) or "?"

            try:
                status, data = future.result()
            except Exception as e:
                status = "FAIL"
                data = {"meta": item.get("meta", {}),
                        "reason": f"future exception: {e}"}

            if status == "SUCCESS":
                append_to_jsonl(output_file, data)
                success_count += 1
                rate = data.get("match_rate")
                rate_str = f"{rate:.3f}" if rate is not None else "N/A"
                n_matched = data.get("n_matched", 0)
                n_gt = data.get("n_gt", 0)
                print(f"  [{success_count + fail_count}/{len(pending)}] "
                      f"{title[:50]} -> {rate_str} ({n_matched}/{n_gt})")
            else:
                append_to_jsonl(fail_file, data)
                fail_count += 1
                reason = data.get("reason", "unknown")
                print(f"  [{success_count + fail_count}/{len(pending)}] "
                      f"{title[:50]} -> FAIL: {reason}")

    print(f"\n{'='*60}")
    print(f"Task 1 Eval done: {success_count} success, {fail_count} fail")

    # Final stats include FAILs as zero; dropping parse failures inflates the
    # average for over-generating or malformed outputs.
    ok_results = load_jsonl(output_file)
    fail_results = load_jsonl(fail_file)
    total_n = len(ok_results) + len(fail_results)
    if total_n == 0:
        return

    def avg_with_fails(key):
        vals = [r.get(key) or 0.0 for r in ok_results]
        vals.extend([0.0] * len(fail_results))
        return sum(vals) / len(vals) if vals else 0.0

    avg_recall = avg_with_fails("match_rate")
    avg_pen = avg_with_fails("count_penalty")
    avg_adj = avg_with_fails("adjusted_score")
    pred_counts = [r.get("n_pred", 0) for r in ok_results]
    gt_counts = [r.get("n_gt", 0) for r in ok_results]

    print(f"Total scored: {total_n} ({len(ok_results)} ok + {len(fail_results)} fail counted as 0)")
    print()
    print(f"  match_rate       = {avg_recall:.4f}   (raw recall, bipartite-enforced)")
    print(f"  count_penalty    = {avg_pen:.4f}   (tol={COUNT_PENALTY_TOLERANCE}, rate={COUNT_PENALTY_RATE})")
    print(f"  adjusted_score   = {avg_adj:.4f}   ← main metric (recall - count_penalty)")
    if pred_counts:
        print()
        print(f"  n_pred  mean={sum(pred_counts)/len(pred_counts):.2f}  median={sorted(pred_counts)[len(pred_counts)//2]}")
        print(f"  n_gt    mean={sum(gt_counts)/len(gt_counts):.2f}  median={sorted(gt_counts)[len(gt_counts)//2]}")
        n_full = sum(r.get("n_matched_full", 0) for r in ok_results)
        n_part = sum(r.get("n_matched_partial", 0) for r in ok_results)
        n_total_gt = sum(gt_counts)
        if n_total_gt:
            print(f"  match mix       1.0={n_full}  0.5={n_part}  0={n_total_gt - n_full - n_part} "
                  f"(of {n_total_gt} GT slots)")


# ======================== Main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Task 1 inference results with an OpenAI-compatible judge API")
    parser.add_argument("--input",   required=True, help="Infer jsonl from run_inference_local.py --task 1")
    parser.add_argument("--output",  default=None,
                        help="Output jsonl (default: <input_stem>_eval_task1.jsonl)")
    parser.add_argument("--fail",    default=None,
                        help="Fail jsonl (default: <input_stem>_eval_task1_fail.jsonl)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", type=str, default=os.environ.get("JUDGE_MODEL"),
                        help="Judge model id. Defaults to JUDGE_MODEL.")
    parser.add_argument("--api-base", default=os.environ.get("JUDGE_API_BASE"),
                        help="OpenAI-compatible API base URL. Defaults to JUDGE_API_BASE.")
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY"),
                        help="API key. Defaults to JUDGE_API_KEY.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit",   type=int, default=0,
                        help="Only evaluate first N records (0 = all)")
    args = parser.parse_args()

    if not args.api_base:
        raise SystemExit("--api-base or JUDGE_API_BASE is required")
    if not args.api_key:
        raise SystemExit("--api-key or JUDGE_API_KEY is required")
    if not args.model:
        raise SystemExit("--model or JUDGE_MODEL is required")

    JUDGE_MODEL = args.model
    JUDGE_TIMEOUT = args.timeout
    JUDGE_CLIENT = OpenAI(api_key=args.api_key, base_url=args.api_base)
    WORKERS = args.workers

    if args.output is None:
        base = os.path.basename(args.input)
        stem = re.sub(r"_infer_task1\.jsonl$", "", base)
        if stem == base:
            stem = base[:-6] if base.endswith(".jsonl") else base
        out_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(out_dir, f"{stem}_eval_task1.jsonl")
    if args.fail is None:
        args.fail = args.output.replace(".jsonl", "_fail.jsonl")

    print(f"Config: model={JUDGE_MODEL}, api_base={args.api_base}, "
          f"workers={WORKERS}, timeout={JUDGE_TIMEOUT}s, limit={args.limit or 'all'}")
    run_eval(args.input, args.output, args.fail, limit=args.limit)
