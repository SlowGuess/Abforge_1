"""
Preprocess ABForge Task 1 RL data from JSONL to verl parquet.

This script does not generate new data. It formats released JSONL records into
the prompt/reward metadata columns expected by verl PPO/GRPO training and
creates a train/validation split.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

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
- If a mechanism can be decomposed into multiple distinct causal factors, separate them explicitly.

**Output Format:**

<Think>
[A brief reasoning process explaining the most important scientific vulnerabilities and causal uncertainties.]
</Think>

<Result>
[A list of target modules — the components, mechanisms, or design choices that should be probed by ablation. Output ONE bullet per atomic target.]

- Target Module: [Name of the component or design choice]
- Target Module: [Name of the next component]
- Target Module: [...]
</Result>"""


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prompt(content: str, max_content_chars: int) -> str:
    if max_content_chars > 0:
        content = content[:max_content_chars]
    return TASK1_USER_PROMPT.replace("{CONTENT}", content)


def convert_record(record: Dict, split: str, idx: int, max_content_chars: int) -> Dict:
    meta = record.get("meta", {})
    content = record.get("Content", "")
    prompt = build_prompt(content=content, max_content_chars=max_content_chars)
    return {
        "data_source": "abforge_task1",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "ablation_objective_identification",
        "reward_model": {
            "style": "candidate_coverage",
            "ground_truth": record.get("Identification", ""),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "title": meta.get("title", ""),
            "paper_context": content,
            "candidates": record.get("Candidates", ""),
            "identification": record.get("Identification", ""),
            "meta": meta,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="ABForge RL JSONL or a local file exported from the Hugging Face dataset.")
    parser.add_argument("--local_dir", default="data/abforge_task1_rl")
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_content_chars", type=int, default=50000)
    args = parser.parse_args()

    input_path = Path(os.path.expanduser(args.input)).resolve()
    local_dir = Path(os.path.expanduser(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
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
