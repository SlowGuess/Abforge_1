"""
Preprocess ABForge SFT data from JSONL to verl parquet.

This script does not generate new supervision. It formats the released JSONL
records into the prompt/response columns consumed by the verl SFT trainer,
filters overlong examples, and creates a paper-grouped train/validation split.

Task 1 trains ablation-objective identification responses. Task 2 trains
ablation-plan synthesis responses.

Usage:
    python dataprocess/prepare_sft.py --task 1 \
        --sft_data_path data/train/sft_task1_45961.jsonl \
        --sft_remain_path data/train/SFT_50K.jsonl \
        --tokenizer_path Qwen/Qwen3-8B \
        --local_dir data/abforge_task1_sft \
        --val_size 200
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets


TASK1_USER_PROMPT = """You are an expert AI research scientist and a rigorous peer reviewer. Your task is to identify the key ablation research questions that should be investigated to rigorously validate a paper's central methodological claims.

<Research_Context>
(Paper context with ablation-related content removed)
{CONTENT}
</Research_Context>

**Task Instructions:**
1. Read the paper context carefully and infer the most important research questions that should be addressed by ablation or controlled analysis.
2. Identify which components, mechanisms, or assumptions are most scientifically vulnerable.
3. Consider what causal confounders or alternative explanations a skeptical reviewer would raise.
4. Focus on research questions that are necessary to verify whether the claimed gains truly come from the proposed method.
5. Stay at the level of research questions rather than detailed implementation.

**Important Constraints:**
- Do not propose full experimental plans, datasets, hyperparameters, or exact protocols.
- Prefer mechanistically meaningful and causally informative questions over superficial component toggles.
- Identify the 2-6 most critical ablation targets. Prioritize scientific necessity over completeness.

**Output Format:** Each bullet represents one atomic ablation target. Output the most scientifically necessary targets (typically 2-6).

<Think>
[A brief reasoning process explaining the most important scientific vulnerabilities and causal uncertainties.]
</Think>

<Result>
[A list of target modules and their corresponding high-level research questions.]

- Target Module: [Name of the component or design choice]
    - Research Question: [One precise sentence summarizing the exact hypothesis to test]
</Result>"""


TASK2_USER_PROMPT = """You are an expert AI research scientist specializing in scientific experimental design. Your task is to construct a rigorous and reproducible ablation plan for a given ablation goal, based on the paper's methodology context.

<Research_Context>
{CONTENT}
</Research_Context>

<Ablation_Goal>
{GOAL}
</Ablation_Goal>

**Task Instructions:**
1. Design a scientifically rigorous experimental plan that directly tests the given ablation goal.
2. Define a fair baseline.
3. Specify the most important ablation or control variants.
4. Isolate the intended causal factor as cleanly as possible.
5. Keep unrelated components fixed unless a change is explicitly required.
6. Include the critical protocols and evaluation metrics needed for a reproducible comparison.

**Important Constraints:**
- Focus on causal validity, fairness of comparison, and confounder isolation.
- Avoid introducing unnecessary complexity or speculative variants not grounded in the methodology context.
- If a stronger control is needed to rule out a plausible alternative explanation, include it.

**Output Format:**

<Think>
[A brief reasoning process explaining how the ablation goal maps to the key controls, baselines, and confounders.]
</Think>

<Proposed_Plan>
- Objective: [Brief statement of the design goal]
- Baseline Setup: [Clear definition of the control condition]
- Variants: [The main ablation or control conditions and what each one changes]
- Fixed Protocols & Metrics: [Key training constraints, datasets, evaluation settings, and primary metrics in a single paragraph]
</Proposed_Plan>"""


TASK_CFG = {
    1: {
        "prompt_template": TASK1_USER_PROMPT,
        "needs_goal": False,
        "think_field": "Global_CoT",
        "result_field": "Global_Result",
        "result_tag": "Result",
    },
    2: {
        "prompt_template": TASK2_USER_PROMPT,
        "needs_goal": True,
        "think_field": "detail_think",
        "result_field": "detail_plan",
        "result_tag": "Proposed_Plan",
    },
}


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_INVESTIGATION_FOCUS_RE = re.compile(r"<Investigation_Focus>", re.IGNORECASE)


def count_gt_focuses(record: Dict) -> int:
    return len(_INVESTIGATION_FOCUS_RE.findall(record.get("Candidates", "") or ""))


def paper_key(record: Dict) -> str:
    meta = record.get("meta") or {}
    return meta.get("pdf_url") or meta.get("title") or ""


def build_prompt(cfg: Dict, content: str, goal: str) -> str:
    text = cfg["prompt_template"].replace("{CONTENT}", content)
    if cfg["needs_goal"]:
        text = text.replace("{GOAL}", goal)
    return text


def build_response(cfg: Dict, think: str, result: str) -> str:
    tag = cfg["result_tag"]
    return f"<Think>\n{think}\n</Think>\n\n<{tag}>\n{result}\n</{tag}>"


def group_aware_split(
    rows: List[Dict],
    val_size: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Split rows into train/val grouped by paper_key so the same paper never
    appears in both splits. We greedily pick papers (in shuffled order) into
    the val pool until reaching val_size, then put the rest in train.
    """
    rng = random.Random(seed)
    by_key: Dict[str, List[Dict]] = {}
    for r in rows:
        by_key.setdefault(r["_paper_key"], []).append(r)

    keys = list(by_key.keys())
    rng.shuffle(keys)

    val_rows: List[Dict] = []
    train_rows: List[Dict] = []
    for k in keys:
        bucket = by_key[k]
        if len(val_rows) < val_size:
            val_rows.extend(bucket)
        else:
            train_rows.extend(bucket)
    # If we overshot val a lot, push the overflow back to train.
    if len(val_rows) > val_size:
        overflow_papers: Dict[str, List[Dict]] = {}
        for r in val_rows:
            overflow_papers.setdefault(r["_paper_key"], []).append(r)
        kept_val: List[Dict] = []
        for k, bucket in overflow_papers.items():
            if len(kept_val) + len(bucket) <= val_size:
                kept_val.extend(bucket)
            else:
                train_rows.extend(bucket)
        val_rows = kept_val
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--sft_data_path", required=True,
                        help="JSONL with detail_think/detail_plan or Global_CoT/Global_Result")
    parser.add_argument("--sft_remain_path", required=True,
                        help="JSONL with source Content/Goal/Rubric fields")
    parser.add_argument("--tokenizer_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=6144,
        help="Drop samples whose chat-templated prompt + response exceeds this",
    )
    parser.add_argument("--min_think_tokens", type=int, default=100)
    parser.add_argument("--min_result_tokens", type=int, default=50)
    parser.add_argument("--task1_min_gt", type=int, default=2,
                        help="For Task 1 only: keep records with at least this many GT focuses.")
    parser.add_argument("--task1_max_gt", type=int, default=6,
                        help="For Task 1 only: keep records with at most this many GT focuses.")
    args = parser.parse_args()

    cfg = TASK_CFG[args.task]
    sft_path = Path(os.path.expanduser(args.sft_data_path)).resolve()
    rem_path = Path(os.path.expanduser(args.sft_remain_path)).resolve()
    out_dir = Path(os.path.expanduser(args.local_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[task {args.task}] loading {rem_path.name} ...")
    remain_records = load_jsonl(rem_path)
    remain_map: Dict[str, Dict] = {}
    for r in remain_records:
        k = paper_key(r)
        if k:
            remain_map[k] = r
    print(f"[task {args.task}] remain unique papers: {len(remain_map)}")

    print(f"[task {args.task}] loading {sft_path.name} ...")
    sft_records = load_jsonl(sft_path)
    print(f"[task {args.task}] sft records: {len(sft_records)}")

    print(f"[task {args.task}] loading tokenizer ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    def render_prompt(user_text: str) -> str:
        msgs = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def tlen(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows: List[Dict] = []
    stats = {
        "total": 0,
        "miss_join": 0,
        "miss_field": 0,
        "short_think": 0,
        "short_result": 0,
        "too_long": 0,
        "kept": 0,
    }

    for idx, a in enumerate(sft_records):
        stats["total"] += 1
        k = paper_key(a)
        b = remain_map.get(k)
        if not b:
            stats["miss_join"] += 1
            continue
        if args.task == 1:
            n_gt = count_gt_focuses(b)
            if n_gt < args.task1_min_gt:
                stats["gt_too_few"] = stats.get("gt_too_few", 0) + 1
                continue
            if n_gt > args.task1_max_gt:
                stats["gt_too_many"] = stats.get("gt_too_many", 0) + 1
                continue
        think = a.get(cfg["think_field"], "") or ""
        result = a.get(cfg["result_field"], "") or ""
        if not think or not result:
            stats["miss_field"] += 1
            continue

        think_tok = tlen(think)
        result_tok = tlen(result)
        if think_tok < args.min_think_tokens:
            stats["short_think"] += 1
            continue
        if result_tok < args.min_result_tokens:
            stats["short_result"] += 1
            continue

        content = b.get("Content", "") or ""
        goal = b.get("Goal", "") or a.get("Goal", "") or ""
        user_msg = build_prompt(cfg, content=content, goal=goal)
        response = build_response(cfg, think=think, result=result)

        prompt_tok = len(
            tokenizer(render_prompt(user_msg), add_special_tokens=False)["input_ids"]
        )
        resp_tok = tlen(response) + 1  # account for the EOS appended at training time
        total_tok = prompt_tok + resp_tok
        if total_tok > args.max_total_tokens:
            stats["too_long"] += 1
            continue

        meta = a.get("meta", {}) or {}
        rows.append(
            {
                "data_source": f"abforge_task{args.task}_sft",
                "prompt": user_msg,
                "response": response,
                "_paper_key": k,
                "extra_info": {
                    "task": args.task,
                    "title": meta.get("title", ""),
                    "pdf_url": meta.get("pdf_url", ""),
                    "prompt_tokens": prompt_tok,
                    "response_tokens": resp_tok,
                    "total_tokens": total_tok,
                },
            }
        )
        stats["kept"] += 1
        if (idx + 1) % 1000 == 0:
            print(f"  processed {idx + 1}  kept={stats['kept']}")

    print(f"\n[task {args.task}] preprocess stats:")
    for k, v in stats.items():
        print(f"  {k:<14}: {v}")

    train_rows, val_rows = group_aware_split(rows, val_size=args.val_size, seed=args.seed)

    # strip bookkeeping field
    for r in train_rows + val_rows:
        r.pop("_paper_key", None)

    print(f"[task {args.task}] split: train={len(train_rows)} val={len(val_rows)}")

    datasets.Dataset.from_list(train_rows).to_parquet(str(out_dir / "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(str(out_dir / "val.parquet"))
    print(f"[task {args.task}] wrote {out_dir/'train.parquet'} and {out_dir/'val.parquet'}")


if __name__ == "__main__":
    main()
