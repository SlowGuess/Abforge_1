#!/usr/bin/env python3
"""
OpenAI-compatible API evaluator for Task 2 using the rubric v2 format.

This is the eval-time counterpart to reward/task2_rubric_reward.py.
Asymmetry by design (matches the paper):
  - Training judge sees <Paper_Context>, but NOT the GT plan, so the policy
    cannot learn to mimic a reference.
  - Eval judge sees <Reference_Design> (GT plan) for a stable comparison; it
    does NOT see <Paper_Context>.

Both share:
  - exactly 10 rubric items with fixed level + weight (rubric v2)
  - score set {0, 0.5, 1}, judge does NOT reassign weights
  - Design Score = Σ(w_pre · s) / Σ(w_pre), missing items excluded

Usage:
    JUDGE_API_BASE=https://api.openai.com/v1 JUDGE_API_KEY=... JUDGE_MODEL=... \
      python evaluation/eval_task2_rubric.py --input <infer.jsonl>
"""

import argparse
import concurrent.futures
import json
import os
import re
import time
from threading import Lock
from typing import Optional, Tuple, List, Dict

from openai import OpenAI

# ======================== Config ========================
JUDGE_TIMEOUT = 300
WORKERS = 3

JUDGE_MODEL: Optional[str] = None
JUDGE_CLIENT: Optional[OpenAI] = None
write_lock = Lock()


# ======================== Evaluation Prompt (v2) ========================
#
# Weights are FIXED in the rubric (see stage3_rubric_v2) — we do not ask the
# judge to reassign them. The judge only writes a short critique and a score
# in {0, 0.5, 1}.

EVAL_PROMPT_V2 = """You are a rigorous scientific reviewer acting as an automated evaluation judge for ablation experiment designs. Your task is to evaluate whether a generated ablation plan satisfies each rubric criterion.

<Ablation_Goal>
{GOAL}
</Ablation_Goal>

<Reference_Design>
{REFERENCE}
</Reference_Design>

<Generated_Design>
{GENERATED}
</Generated_Design>

<Evaluation_Rubric>
{RUBRIC}
</Evaluation_Rubric>

**Evaluation Instructions:**
The rubric has exactly 10 items. Each item already carries a fixed `level` and `weight` — do NOT reassign them. For each rubric item you must:

1. Produce a brief critique (1–3 sentences) explaining whether and how the generated design satisfies the requirement.
2. Assign a satisfaction score in {0, 0.5, 1}:
   - 1 : the generated design fully satisfies the rubric item
   - 0.5 : the generated design partially addresses it (notable gaps or ambiguities)
   - 0 : the generated design does not satisfy it

Be strict. Evaluate only against the specific condition stated in the rubric item; ignore whether other items would be satisfied.

**Output Format (strictly follow this XML format; produce exactly 10 <item> blocks):**
<Evaluation>
<item num="1">
<critique>[brief analysis]</critique>
<score>[0, 0.5, or 1]</score>
</item>
<item num="2">
<critique>[brief analysis]</critique>
<score>[0, 0.5, or 1]</score>
</item>
...
<item num="10">
<critique>[brief analysis]</critique>
<score>[0, 0.5, or 1]</score>
</item>
</Evaluation>"""


# ======================== Rubric parsing ========================

RUBRIC_BLOCK_RE = re.compile(r"<Rubric>(.*?)</Rubric>", re.DOTALL | re.IGNORECASE)
ITEM_V2_RE = re.compile(
    r'<item\s+num="(\d+)"\s+level="(critical|major|minor)"\s+weight="(\d+)"\s*>'
    r'(.*?)</item>',
    re.DOTALL | re.IGNORECASE,
)


def parse_rubric_v2(rubric_text: str) -> List[Dict]:
    """
    Parse the v2 rubric. Returns a list of dicts:
        [{num, level, weight, text}, ...]
    Accepts either the inner 10-item list or a full <Rubric>…</Rubric> block.
    Raises ValueError if the parsed count != 10.
    """
    if not rubric_text:
        raise ValueError("empty rubric")
    m = RUBRIC_BLOCK_RE.search(rubric_text)
    inner = m.group(1) if m else rubric_text
    items = ITEM_V2_RE.findall(inner)
    if len(items) != 10:
        raise ValueError(f"rubric has {len(items)} items, expected 10")
    parsed = []
    for num, level, weight, text in items:
        parsed.append({
            "num":    int(num),
            "level":  level.lower(),
            "weight": int(weight),
            "text":   text.strip(),
        })
    return parsed


# ======================== Judge response parsing ========================

def parse_eval_response_v2(response: str, num_items: int) -> List[Dict]:
    """Extract critique + score per item from the judge response."""
    results = []
    for i in range(1, num_items + 1):
        pattern = rf'<item\s+num="{i}">(.*?)</item>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            pattern = rf'<item\s+num={i}>(.*?)</item>'
            match = re.search(pattern, response, re.DOTALL)

        if match:
            block = match.group(1)
            critique = re.search(r'<critique>(.*?)</critique>', block, re.DOTALL)
            score = re.search(r'<score>(.*?)</score>', block, re.DOTALL)
            results.append({
                "item_num": i,
                "critique": critique.group(1).strip() if critique else "",
                "score":    _parse_score(score.group(1).strip()) if score else None,
            })
        else:
            results.append({
                "item_num": i,
                "critique": "PARSE_FAILED",
                "score":    None,
            })
    return results


def _parse_score(s: str) -> Optional[float]:
    try:
        v = float(s)
        if v < 0:   return 0.0
        if v > 1:   return 1.0
        return v
    except Exception:
        return None


def compute_design_score(rubric_items: List[Dict],
                         eval_items: List[Dict]) -> Tuple[Optional[float], Dict]:
    """
    Design Score = Σ(w_pre · s) / Σ(w_pre), using pre-assigned weights from
    the rubric (not the judge).

    Items whose score failed to parse are EXCLUDED from both numerator and
    denominator (so a single parse miss doesn't push the denominator up and
    penalize the model artificially). Returns (score, diagnostics).
    """
    assert len(rubric_items) == len(eval_items)

    total_w = 0
    total_ws = 0.0
    parsed = 0
    missing = 0
    per_level = {"critical": [], "major": [], "minor": []}

    for rub, ev in zip(rubric_items, eval_items):
        w = rub["weight"]
        lvl = rub["level"]
        s = ev["score"]
        if s is None:
            missing += 1
            continue
        parsed += 1
        total_w  += w
        total_ws += w * s
        per_level[lvl].append(s)

    if total_w == 0:
        return None, {"parsed": parsed, "missing": missing, "per_level": per_level}

    per_level_mean = {
        lvl: (sum(xs) / len(xs) if xs else None)
        for lvl, xs in per_level.items()
    }
    return total_ws / total_w, {
        "parsed": parsed,
        "missing": missing,
        "per_level_mean": per_level_mean,
    }


# ======================== IO helpers ========================

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
            max_tokens=4096,
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
        goal = item.get("Goal", "")
        gt_rubric = item.get("gt_Rubric", "")
        gt_plan = item.get("gt_refined_plan", "")
        raw_response = item.get("infer_task2_response", "")

        # Grade only the <Proposed_Plan>, not the <Think> part
        plan_match = re.search(r'<Proposed_Plan>(.*?)</Proposed_Plan>',
                               raw_response, re.DOTALL)
        generated = plan_match.group(1).strip() if plan_match else raw_response

        if not gt_rubric or not generated:
            return ("FAIL", {"meta": meta,
                             "reason": "missing gt_Rubric or infer_task2_response"})

        try:
            rubric_items = parse_rubric_v2(gt_rubric)
        except ValueError as e:
            return ("FAIL", {"meta": meta, "reason": f"rubric parse: {e}"})

        # Render rubric for the judge WITH attributes visible, so it can
        # see level/weight context for calibration.
        rubric_formatted = "\n".join(
            f'<item num="{r["num"]}" level="{r["level"]}" weight="{r["weight"]}">'
            f'{r["text"]}</item>'
            for r in rubric_items
        )

        prompt = (EVAL_PROMPT_V2
                  .replace("{GOAL}", goal)
                  .replace("{REFERENCE}", gt_plan)
                  .replace("{GENERATED}", generated)
                  .replace("{RUBRIC}", rubric_formatted))

        resp = call_judge(prompt, label=f"EVALv2|{tag}")
        if not resp:
            return ("FAIL", {"meta": meta, "reason": "judge call failed"})

        eval_results = parse_eval_response_v2(resp, len(rubric_items))

        # Attach pre-assigned level/weight to each eval entry for transparency
        for rub, ev in zip(rubric_items, eval_results):
            ev["level"]  = rub["level"]
            ev["weight"] = rub["weight"]

        if all(r["score"] is None for r in eval_results):
            return ("FAIL", {
                "meta": meta,
                "reason": "all scores parse failed",
                "raw_response_tail": resp[-500:] if resp else "",
            })

        design_score, diag = compute_design_score(rubric_items, eval_results)

        return ("SUCCESS", {
            "meta":              meta,
            "Goal":              goal,
            "num_rubric_items":  len(rubric_items),
            "eval_details":      eval_results,
            "design_score":      design_score,
            "score_diagnostics": diag,
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

    print(f"Task 2 Eval (rubric v2): total={len(items)}, done={len(done)}, "
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
                score = data.get("design_score")
                score_str = f"{score:.3f}" if score is not None else "N/A"
                print(f"  [{success_count + fail_count}/{len(pending)}] "
                      f"{title[:50]} -> {score_str}")
            else:
                append_to_jsonl(fail_file, data)
                fail_count += 1
                reason = data.get("reason", "unknown")
                print(f"  [{success_count + fail_count}/{len(pending)}] "
                      f"{title[:50]} -> FAIL: {reason}")

    print(f"\n{'='*60}")
    print(f"Task 2 Eval (v2) done: {success_count} success, {fail_count} fail")

    results = load_jsonl(output_file)
    scores = [r["design_score"] for r in results if r.get("design_score") is not None]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"Average Design Score: {avg:.4f} ({avg*100:.2f}%)")
        print(f"Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"Evaluated: {len(scores)} instances")

        # Per-level breakdown across all records.
        per_level_all = {"critical": [], "major": [], "minor": []}
        for r in results:
            pl = (r.get("score_diagnostics") or {}).get("per_level_mean") or {}
            for lvl, v in pl.items():
                if v is not None:
                    per_level_all[lvl].append(v)
        for lvl, xs in per_level_all.items():
            if xs:
                print(f"  {lvl:>8s}: mean={sum(xs)/len(xs):.4f}  n={len(xs)}")


# ======================== Main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Task 2 inference results with an OpenAI-compatible judge API")
    parser.add_argument("--input",   required=True, help="Infer jsonl with v2 gt_Rubric")
    parser.add_argument("--output",  default=None,
                        help="Output jsonl (default: <input_stem>_eval_task2_v2.jsonl)")
    parser.add_argument("--fail",    default=None,
                        help="Fail jsonl (default: <input_stem>_eval_task2_v2_fail.jsonl)")
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

    # Default output naming: replace "_infer_task2(_rubricv2).jsonl" with "_eval_task2_v2.jsonl"
    if args.output is None:
        base = os.path.basename(args.input)
        stem = re.sub(r"_infer_task2(_rubricv2)?\.jsonl$", "", base)
        if stem == base:
            stem = base[:-6] if base.endswith(".jsonl") else base
        out_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(out_dir, f"{stem}_eval_task2_v2.jsonl")
    if args.fail is None:
        args.fail = args.output.replace(".jsonl", "_fail.jsonl")

    print(f"Config: model={JUDGE_MODEL}, api_base={args.api_base}, "
          f"workers={WORKERS}, timeout={JUDGE_TIMEOUT}s, limit={args.limit or 'all'}")
    run_eval(args.input, args.output, args.fail, limit=args.limit)
