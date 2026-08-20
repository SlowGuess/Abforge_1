"""
Preprocess ABForge Task 2 RL data into verl parquet.

This script does not generate new data. It reads the ABForge table of
`SlowGuess/abforge-data`, keeps the rows flagged `in_rl_task2`, and writes the
prompt/reward metadata columns expected by verl PPO/GRPO training together with
a train/validation split.

Usage:
    python dataprocess/prepare_task2_rl.py --local_dir data/abforge_task2_rl

`--dataset` also accepts a local directory of parquet shards, e.g. the
`data/train` produced by
`huggingface-cli download SlowGuess/abforge-data --repo-type dataset --local-dir data`,
or a JSONL file carrying the same fields.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import datasets


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


# rows flagged with this column form the Task 2 RL pool
SPLIT_FLAG = "in_rl_task2"

# columns pulled from the table; everything else is dropped up front so the
# full table does not have to be materialized
KEEP_COLUMNS = ["pdf_url", "title", "venue", "year", "content", "goal",
                "rubric", "refined_standard_plan", SPLIT_FLAG]

# the table and the older JSONL exports name the same things differently
ALIASES = {"Content": "content", "Goal": "goal", "Rubric": "rubric"}


def field(record: Dict, name: str) -> str:
    """Read a reference field from either input schema."""
    v = record.get(name)
    if v is None:
        v = record.get(ALIASES.get(name, name))
    return v or ""


def meta_of(record: Dict) -> Dict:
    m = record.get("meta")
    if isinstance(m, dict):
        return m
    return {k: record.get(k) or "" for k in ("title", "venue", "year", "pdf_url")}


def load_records(dataset: str, config, split: str) -> List[Dict]:
    """The Task 2 RL rows, from the released table or from a JSONL file."""
    src = os.path.expanduser(dataset)
    if os.path.isfile(src):
        with open(src, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if os.path.isdir(src):
        files = sorted(str(p) for p in Path(src).glob("**/*.parquet"))
        if not files:
            raise SystemExit(f"no .jsonl file and no parquet files at {dataset}")
        ds = datasets.load_dataset("parquet", data_files=files, split="train")
    elif config:
        ds = datasets.load_dataset(dataset, config, split=split)
    else:
        ds = datasets.load_dataset(dataset, split=split)

    drop = [c for c in ds.column_names if c not in KEEP_COLUMNS]
    if drop:
        ds = ds.remove_columns(drop)
    ds = ds.filter(lambda r: r[SPLIT_FLAG])
    return list(ds)


def build_prompt(content: str, goal: str, max_content_chars: int) -> str:
    if max_content_chars > 0:
        content = content[:max_content_chars]
    return TASK2_USER_PROMPT.replace("{CONTENT}", content).replace("{GOAL}", goal)


def convert_record(record: Dict, split: str, idx: int, max_content_chars: int) -> Dict:
    meta = meta_of(record)
    content = field(record, "Content")
    goal = field(record, "Goal")
    prompt = build_prompt(content=content, goal=goal, max_content_chars=max_content_chars)
    return {
        "data_source": "abforge_task2",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "ablation_design",
        "reward_model": {
            "style": "rubric",
            "ground_truth": field(record, "refined_standard_plan"),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "title": meta.get("title", ""),
            "paper_context": content,
            "goal": goal,
            "rubric": field(record, "Rubric"),
            "refined_standard_plan": field(record, "refined_standard_plan"),
            "meta": meta,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "--input", dest="dataset",
                        default="SlowGuess/abforge-data",
                        help="HF dataset id, a local directory of parquet shards, "
                             "or a JSONL file carrying the same fields")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_dir", default="data/abforge_task2_rl")
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_content_chars", type=int, default=60000)
    args = parser.parse_args()

    local_dir = Path(os.path.expanduser(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.dataset, args.config, args.split)
    print(f"Task 2 RL rows: {len(records)}")
    rng = random.Random(args.seed)
    rng.shuffle(records)

    val_size = min(max(args.val_size, 1), len(records) - 1)
    val_records = records[:val_size]
    train_records = records[val_size:]

    train_rows = [
        convert_record(record=record, split="train", idx=idx, max_content_chars=args.max_content_chars)
        for idx, record in enumerate(train_records)
    ]
    val_rows = [
        convert_record(record=record, split="val", idx=idx, max_content_chars=args.max_content_chars)
        for idx, record in enumerate(val_records)
    ]

    datasets.Dataset.from_list(train_rows).to_parquet(str(local_dir / "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(str(local_dir / "val.parquet"))

    print(f"Saved train={len(train_rows)} val={len(val_rows)} to {local_dir}")


if __name__ == "__main__":
    main()
