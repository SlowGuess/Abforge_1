"""
Preprocess ABForge Task 2 RL data from JSONL to verl parquet.

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


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prompt(content: str, goal: str, max_content_chars: int) -> str:
    if max_content_chars > 0:
        content = content[:max_content_chars]
    return TASK2_USER_PROMPT.replace("{CONTENT}", content).replace("{GOAL}", goal)


def convert_record(record: Dict, split: str, idx: int, max_content_chars: int) -> Dict:
    meta = record.get("meta", {})
    content = record.get("Content", "")
    goal = record.get("Goal", "")
    prompt = build_prompt(content=content, goal=goal, max_content_chars=max_content_chars)
    return {
        "data_source": "abforge_task2",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "ablation_design",
        "reward_model": {
            "style": "rubric",
            "ground_truth": record.get("refined_standard_plan", ""),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "title": meta.get("title", ""),
            "paper_context": content,
            "goal": goal,
            "rubric": record.get("Rubric", ""),
            "refined_standard_plan": record.get("refined_standard_plan", ""),
            "meta": meta,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="ABForge RL JSONL or a local file exported from the Hugging Face dataset.")
    parser.add_argument("--local_dir", default="data/abforge_task2_rl")
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_content_chars", type=int, default=60000)
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
