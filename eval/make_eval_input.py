#!/usr/bin/env python3
"""Turn published generations into scorer input.

`outputs/task{1,2}/generations/<slug>.jsonl` in the data release carries only
`pdf_url` / `title` / `model` / `response` — the reference fields are deliberately
not duplicated there. This joins them back by `meta.pdf_url` so the result can be
fed straight to `eval/eval_task1.py` or `eval/eval_task2_rubric.py`.

    Task 1: meta, gt_Candidates, infer_task1_response
    Task 2: meta, Goal, gt_refined_plan, gt_Rubric, infer_task2_response

Usage:
    python eval/make_eval_input.py --task 1 \
        --generations data/outputs/task1/generations/abforge.jsonl \
        --bench data/eval/ablationbench_200.jsonl \
        --output preds_task1.jsonl

`--bench` also accepts the unified table (a directory of `unified/*.parquet` or a
Hugging Face dataset id); use `--filter` to pick the split, e.g.
`--bench SlowGuess/abforge-data --filter in_bench_200`.

Running your own model instead? `run_inference_local.py` already writes this
schema directly and no join is needed.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

# the benchmark JSONL and the unified table name the same things differently
ALIASES = {
    "Content": "content", "Candidates": "candidates", "Goal": "goal",
    "Rubric": "rubric", "refined_standard_plan": "refined_standard_plan",
}


def field(item: Dict, name: str) -> str:
    v = item.get(name)
    if v is None:
        v = item.get(ALIASES.get(name, name))
    return v or ""


def meta_of(item: Dict) -> Dict:
    m = item.get("meta")
    if isinstance(m, dict):
        return m
    return {k: item.get(k, "") for k in ("title", "venue", "year", "pdf_url")}


def load_bench(source: str, filter_col: str) -> List[Dict]:
    src = os.path.expanduser(source)
    if os.path.isfile(src):
        with open(src, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    import datasets as hfds
    if os.path.isdir(src):
        files = sorted(str(p) for p in Path(src).glob("**/*.parquet"))
        if not files:
            raise SystemExit(f"no .jsonl file and no parquet files at {source}")
        ds = hfds.load_dataset("parquet", data_files=files, split="train")
    else:
        ds = hfds.load_dataset(src, "unified", split="train")
    if filter_col:
        ds = ds.filter(lambda r: r.get(filter_col))
    return list(ds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=[1, 2], required=True)
    ap.add_argument("--generations", required=True,
                    help="outputs/task{1,2}/generations/<slug>.jsonl")
    ap.add_argument("--bench", required=True,
                    help="benchmark JSONL, a directory of unified parquet, or a HF dataset id")
    ap.add_argument("--filter", default="",
                    help="for non-JSONL --bench: boolean column selecting the split")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bench = {}
    for r in load_bench(args.bench, args.filter):
        url = meta_of(r).get("pdf_url") or ""
        if url:
            bench[url] = r
    print(f"bench papers: {len(bench)}")

    gens = []
    with open(os.path.expanduser(args.generations), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gens.append(json.loads(line))
    print(f"generations:  {len(gens)}")

    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = missing = empty = 0
    with out_path.open("w", encoding="utf-8") as out:
        for g in gens:
            url = g.get("pdf_url") or ""
            b = bench.get(url)
            if b is None:
                missing += 1
                continue
            response = g.get("response") or ""
            if not response.strip():
                empty += 1
            if args.task == 1:
                row = {
                    "meta": meta_of(b),
                    "gt_Candidates": field(b, "Candidates"),
                    "infer_task1_response": response,
                }
            else:
                row = {
                    "meta": meta_of(b),
                    "Goal": field(b, "Goal"),
                    "gt_refined_plan": field(b, "refined_standard_plan"),
                    "gt_Rubric": field(b, "Rubric"),
                    "infer_task2_response": response,
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} rows -> {out_path}")
    if missing:
        print(f"  WARNING: {missing} generations had no matching pdf_url in the benchmark")
    if empty:
        print(f"  note: {empty} rows have an empty response (scored as a failure)")


if __name__ == "__main__":
    main()
