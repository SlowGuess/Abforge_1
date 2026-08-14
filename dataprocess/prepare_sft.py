"""
Preprocess ABForge SFT data into verl parquet.

This script does not generate new supervision. It reads the unified table of
`SlowGuess/abforge-data`, keeps the rows belonging to the requested task's SFT
split, formats them into the prompt/response columns consumed by the verl SFT
trainer, filters overlong examples, and creates a paper-grouped train/validation
split.

Task 1 trains ablation-objective identification responses. Task 2 trains
ablation-plan synthesis responses.

Usage:
    python dataprocess/prepare_sft.py --task 1 \
        --tokenizer_path Qwen/Qwen3-8B \
        --local_dir data/abforge_task1_sft \
        --val_size 200

`--dataset` also accepts a local directory, e.g. one produced by
`huggingface-cli download SlowGuess/abforge-data --repo-type dataset --local-dir data`
(pass `data/unified`).
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
        "think_field": "global_cot",
        "result_field": "global_result",
        "result_tag": "Result",
        "split_flag": "in_sft_task1",
        "run_flag": "sft_run_task1",
    },
    2: {
        "prompt_template": TASK2_USER_PROMPT,
        "needs_goal": True,
        "think_field": "detail_think",
        "result_field": "detail_plan",
        "result_tag": "Proposed_Plan",
        "split_flag": "in_sft_task2",
        "run_flag": "sft_run_task2",
    },
}

# columns pulled from the unified table; everything else is dropped up front so
# the 1.9 GB table does not have to be materialized in full
KEEP_COLUMNS = ["pdf_url", "title", "content", "goal", "n_focuses",
                "global_cot", "global_result", "detail_think", "detail_plan",
                "in_sft_task1", "in_sft_task2", "sft_run_task1", "sft_run_task2"]


def load_unified(dataset: str, config: str, split: str, cfg: Dict, select: str):
    """Rows of the unified table for this task's SFT data.

    `select=run` takes the rows the released SFT run actually consumed, recorded in
    the table as `sft_run_task{1,2}`. That selection was made on an intermediate
    artifact which is not part of the release, so it cannot be recomputed from the
    published text — reading the flag is the only way to reproduce those splits.
    `select=view` instead takes the whole task view and re-derives the filters below.
    """
    import datasets as hfds

    if os.path.isdir(os.path.expanduser(dataset)):
        files = sorted(str(p) for p in Path(os.path.expanduser(dataset)).glob("*.parquet"))
        if not files:
            raise SystemExit(f"no parquet files under {dataset}")
        ds = hfds.load_dataset("parquet", data_files=files, split="train")
    else:
        ds = hfds.load_dataset(dataset, config, split=split)

    drop = [c for c in ds.column_names if c not in KEEP_COLUMNS]
    if drop:
        ds = ds.remove_columns(drop)
    flag = cfg["run_flag"] if select == "run" else cfg["split_flag"]
    if flag not in ds.column_names:
        raise SystemExit(f"column {flag} missing; the table predates --select {select}")
    ds = ds.filter(lambda r: r[flag] and (r["content"] or "").strip())
    return ds


def paper_key(record: Dict) -> str:
    return record.get("pdf_url") or record.get("title") or ""


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
    parser.add_argument("--dataset", default="SlowGuess/abforge-data",
                        help="HF dataset id, or a local directory of unified/*.parquet")
    parser.add_argument("--config", default="unified")
    parser.add_argument("--split", default="train")
    parser.add_argument("--select", choices=["run", "view"], default="run",
                        help="run: the rows the released SFT run used (reproduces the "
                             "reported training sizes). view: the whole task view, "
                             "re-filtered by the options below.")
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
    parser.add_argument("--task1_min_gt", type=int, default=1,
                        help="For Task 1 only: keep records with at least this many GT focuses. "
                             "Default 1 — the value used for the released runs.")
    parser.add_argument("--task1_max_gt", type=int, default=6,
                        help="For Task 1 only: keep records with at most this many GT focuses.")
    args = parser.parse_args()

    cfg = TASK_CFG[args.task]
    out_dir = Path(os.path.expanduser(args.local_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[task {args.task}] loading {args.dataset} ({args.config}), select={args.select} ...")
    sft_records = load_unified(args.dataset, args.config, args.split, cfg, args.select)
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
        "miss_field": 0,
        "short_think": 0,
        "short_result": 0,
        "too_long": 0,
        "kept": 0,
    }

    for idx, a in enumerate(sft_records):
        stats["total"] += 1
        k = paper_key(a)
        # in run mode the selection already encodes every filter the run applied,
        # so re-applying them here would only shrink it further
        if args.task == 1 and args.select == "view":
            n_gt = a.get("n_focuses") or 0
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

        content = a.get("content", "") or ""
        goal = a.get("goal", "") or ""
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

        rows.append(
            {
                "data_source": f"abforge_task{args.task}_sft",
                "prompt": user_msg,
                "response": response,
                "_paper_key": k,
                "extra_info": {
                    "task": args.task,
                    "title": a.get("title", ""),
                    "pdf_url": a.get("pdf_url", ""),
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
