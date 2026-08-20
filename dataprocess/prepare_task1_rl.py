"""
Preprocess ABForge Task 1 RL data into verl parquet.

This script does not generate new data. It reads the ABForge table of
`SlowGuess/abforge-data`, keeps the rows flagged `in_rl_task1`, applies the
ground-truth focus-count filter, and writes the prompt/reward columns expected
by verl PPO/GRPO training together with a train/validation split.

Usage:
    python dataprocess/prepare_task1_rl.py --local_dir data/abforge_task1_rl

`--dataset` also accepts a local directory of parquet shards, e.g. the
`data/train` produced by
`huggingface-cli download SlowGuess/abforge-data --repo-type dataset --local-dir data`,
or a JSONL file carrying the same fields.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List

import datasets


_INVESTIGATION_FOCUS_RE = re.compile(r"<Investigation_Focus>", re.IGNORECASE)

# rows flagged with this column form the Task 1 RL pool
SPLIT_FLAG = "in_rl_task1"

# columns pulled from the table; everything else is dropped up front so the
# full table does not have to be materialized
KEEP_COLUMNS = ["pdf_url", "title", "venue", "year",
                "content", "candidates", "n_focuses", SPLIT_FLAG]

# the table and the older JSONL exports name the same things differently
ALIASES = {"Content": "content", "Candidates": "candidates"}


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


def count_gt_focuses(record: Dict) -> int:
    n = record.get("n_focuses")
    if isinstance(n, int):
        return n
    return len(_INVESTIGATION_FOCUS_RE.findall(field(record, "Candidates")))


INFER_TASK1 = """You are an expert AI research scientist and a rigorous peer reviewer. Your task is to identify the key ablation research questions that should be investigated to rigorously validate a paper's central methodological claims.

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


def load_records(dataset: str, config, split: str) -> List[Dict]:
    """The Task 1 RL rows, from the released table or from a JSONL file."""
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


def build_prompt(content: str, max_content_chars: int) -> str:
    if max_content_chars > 0:
        content = content[:max_content_chars]
    return INFER_TASK1.replace("{CONTENT}", content)


def convert_record(record: Dict, split: str, idx: int, max_content_chars: int) -> Dict:
    meta = meta_of(record)
    content = field(record, "Content")
    prompt = build_prompt(content=content, max_content_chars=max_content_chars)
    return {
        "data_source": "abforge_task1",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "ablation_objective_identification",
        "reward_model": {
            "style": "candidate_coverage",
            # The reference objectives live in `Candidates`. An earlier version read a
            # non-existent `Identification` key, which left this an empty string; the
            # Task 1 reward scores against `extra_info.candidates`, so training was
            # unaffected, but the field is now populated and consistent.
            "ground_truth": field(record, "Candidates"),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "title": meta.get("title", ""),
            "paper_context": content,
            "candidates": field(record, "Candidates"),
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
    parser.add_argument("--local_dir", default="data/abforge_task1_rl")
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_content_chars", type=int, default=50000)
    parser.add_argument("--min_gt", type=int, default=1,
                        help="Min GT Investigation_Focus count. Default 1 — the "
                             "value used for the released runs; single-objective "
                             "papers are kept even though the prompt asks for 2-6.")
    parser.add_argument("--max_gt", type=int, default=6,
                        help="Max GT Investigation_Focus count. Default 6 — "
                             "aligned with the upper end of the prompt constraint "
                             "and the eval/reward count_penalty hard_cap.")
    args = parser.parse_args()

    local_dir = Path(os.path.expanduser(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.dataset, args.config, args.split)
    n_total = len(records)
    records = [r for r in records
               if args.min_gt <= count_gt_focuses(r) <= args.max_gt]
    n_filt = len(records)
    print(f"GT filter [{args.min_gt}, {args.max_gt}]: {n_total} -> {n_filt} "
          f"({100*n_filt/max(n_total,1):.1f}% kept)")

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
