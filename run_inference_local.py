#!/usr/bin/env python3
"""Run a local ABForge checkpoint over AblationBench and write eval-ready JSONL.

The output of this script is exactly what `eval/eval_task1.py` and
`eval/eval_task2_rubric.py` expect as `--input`: each row carries the reference
fields next to the model's generation, so inference and scoring compose without
an intermediate join.

    Task 1: meta, gt_Candidates, infer_task1_response, local_model_path
    Task 2: meta, Goal, gt_refined_plan, gt_Rubric, infer_task2_response,
            local_model_path

Prompts and chat templating match training (`dataprocess/prepare_sft.py`):
`enable_thinking=False`, with the reasoning block produced by the model itself
inside `<Think>`.

Usage:
    python run_inference_local.py --task 1 \
        --input data/eval/ablationbench_200.jsonl \
        --output preds_task1.jsonl \
        --model-path SlowGuess/ABForge-Qwen3-8B \
        --dtype bf16 --device-map auto \
        --max-new-tokens 5120 --temperature 0.0 --stop-on '</Result>'

`--input` also accepts the ABForge table — a directory of `train/*.parquet` or
a Hugging Face dataset id — in which case `--filter` selects the split, e.g.
`--input SlowGuess/abforge-data --filter in_bench_200`.

Re-running resumes: rows already present in `--output` are skipped. `--overwrite`
starts the file from scratch (it truncates; it does not append a second copy).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List


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


INFER_TASK2 = """You are an expert AI research scientist specializing in scientific experimental design. Your task is to construct a rigorous and reproducible ablation plan for a given ablation goal, based on the paper's methodology context.

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


DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "auto": "auto"}

# the benchmark JSONL and the ABForge table name the same things differently
TABLE_ALIASES = {
    "Content": "content", "Candidates": "candidates", "Goal": "goal",
    "Rubric": "rubric", "refined_standard_plan": "refined_standard_plan",
}


def field(item: Dict, name: str) -> str:
    """Read a reference field from either input schema."""
    v = item.get(name)
    if v is None:
        v = item.get(TABLE_ALIASES.get(name, name))
    return v or ""


def meta_of(item: Dict) -> Dict:
    m = item.get("meta")
    if isinstance(m, dict):
        return m
    return {k: item.get(k, "") for k in ("title", "venue", "year", "pdf_url")}


def load_items(source: str, filter_col: str, limit: int) -> List[Dict]:
    src = os.path.expanduser(source)
    if os.path.isfile(src):
        rows = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        import datasets as hfds
        if os.path.isdir(src):
            files = sorted(str(p) for p in Path(src).glob("**/*.parquet"))
            if not files:
                raise SystemExit(f"no .jsonl file and no parquet files at {source}")
            ds = hfds.load_dataset("parquet", data_files=files, split="train")
        else:
            ds = hfds.load_dataset(src, split="train")
        if filter_col:
            ds = ds.filter(lambda r: r.get(filter_col))
        rows = list(ds)
    if limit > 0:
        rows = rows[:limit]
    return rows


def title_of(row: Dict) -> str:
    return (meta_of(row).get("title") or "").strip()


def done_titles(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = title_of(json.loads(line))
                if t:
                    out.add(t)
    return out


def build_prompt(item: Dict, task: int) -> str:
    content = field(item, "Content")
    if task == 1:
        return INFER_TASK1.replace("{CONTENT}", content)
    return INFER_TASK2.replace("{CONTENT}", content).replace("{GOAL}", field(item, "Goal"))


def build_row(item: Dict, task: int, response: str, model_path: str) -> Dict:
    if task == 1:
        return {
            "meta": meta_of(item),
            "gt_Candidates": field(item, "Candidates"),
            "infer_task1_response": response,
            "local_model_path": model_path,
        }
    return {
        "meta": meta_of(item),
        "Goal": field(item, "Goal"),
        "gt_refined_plan": field(item, "refined_standard_plan"),
        "gt_Rubric": field(item, "Rubric"),
        "infer_task2_response": response,
        "local_model_path": model_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=[1, 2], required=True)
    ap.add_argument("--input", required=True,
                    help="benchmark JSONL, a directory of parquet shards, or a HF dataset id")
    ap.add_argument("--filter", default="",
                    help="for non-JSONL inputs: boolean column selecting the split, "
                         "e.g. in_bench_200")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--stop-on", default=None,
                    help="stop as soon as this string appears in the generation, "
                         "e.g. '</Result>' or '</Proposed_Plan>'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="truncate --output and re-run every paper")
    args = ap.parse_args()

    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    items = load_items(args.input, args.filter, args.limit)
    already = done_titles(out_path)
    pending = [it for it in items if title_of(it) not in already]
    print(f"[task {args.task}] {len(items)} papers, {len(already)} already done, "
          f"{len(pending)} to run -> {out_path}", flush=True)
    if not pending:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    tok_path = args.tokenizer_path or args.model_path
    print(f"[task {args.task}] loading {args.model_path} ({args.dtype}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=args.trust_remote_code)
    dtype = getattr(torch, DTYPES[args.dtype]) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    class StopOnText(StoppingCriteria):
        """Stop once the marker shows up in the newly generated text."""

        def __init__(self, marker: str, prompt_len: int):
            self.marker, self.prompt_len = marker, prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            tail = tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
            return self.marker in tail

    for i, item in enumerate(pending, 1):
        user_msg = build_prompt(item, args.task)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if args.temperature and args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            gen_kwargs.update(do_sample=False)
        if args.stop_on:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopOnText(args.stop_on, prompt_len)])

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(build_row(item, args.task, response, args.model_path),
                               ensure_ascii=False) + "\n")
            f.flush()
        print(f"  [{i}/{len(pending)}] {title_of(item)[:55]} -> {len(response)} chars "
              f"({time.time() - t0:.1f}s)", flush=True)

    print(f"[task {args.task}] done -> {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
