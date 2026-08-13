"""
Preprocess ABForge Task 1 RL data from jsonl to parquet.
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


def count_gt_focuses(record: Dict) -> int:
    cs = record.get("Candidates", "") or ""
    return len(_INVESTIGATION_FOCUS_RE.findall(cs))


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
    return INFER_TASK1.replace("{CONTENT}", content)


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
    parser.add_argument("--min_gt", type=int, default=1,
                        help="Min GT Investigation_Focus count. Default 1 — the "
                             "value used for the released runs; single-objective "
                             "papers are kept even though the prompt asks for 2-6.")
    parser.add_argument("--max_gt", type=int, default=6,
                        help="Max GT Investigation_Focus count. Default 6 — "
                             "aligned with the upper end of the prompt constraint "
                             "and the eval/reward count_penalty hard_cap.")
    args = parser.parse_args()

    input_path = Path(os.path.expanduser(args.input)).resolve()
    local_dir = Path(os.path.expanduser(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
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
