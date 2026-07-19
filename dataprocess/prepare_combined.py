"""Merge the per-task ABForge parquets into the unified (mixed 1:1) training
sets used by the main SFT/RL pipeline.

ABForge trains a single model across both tasks; Task 1 and Task 2 examples
are mixed at a 1:1 ratio in both stages. Run the per-task prepare scripts
first (``prepare_sft.py``, ``prepare_task1_rl.py``, ``prepare_task2_rl.py``),
then merge their outputs with this script.

Two modes:

  --mode sft
    Concatenate the two SFT parquets (identical schema: prompt/response
    strings + extra_info) and shuffle. The SFT trainer keys on
    prompt/response only; the model learns the task from the prompt format.

  --mode rl
    Concatenate the two RL parquets. Their ``extra_info`` structs differ
    (Task 1 carries candidates/identification; Task 2 carries
    goal/rubric/refined_standard_plan), so a plain concat would hit an Arrow
    schema mismatch. Every row is coerced into one union ``extra_info``
    schema computed from the two inputs (missing fields -> ""). The reward
    server reads keys with .get(...), so empty extra fields are harmless.
    Each row keeps its own ``data_source`` ("abforge_task1" /
    "abforge_task2") so the unified reward server can route it. Rows are
    streamed via pyarrow to stay memory-flat.

Examples:
  python dataprocess/prepare_combined.py --mode sft \
      --task1_dir data/abforge_task1_sft --task2_dir data/abforge_task2_sft \
      --out_dir data/abforge_combined_sft
  python dataprocess/prepare_combined.py --mode rl \
      --task1_dir data/abforge_task1_rl --task2_dir data/abforge_task2_rl \
      --out_dir data/abforge_combined_rl
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------- SFT mode -----------------------------
def merge_sft(task1_dir: Path, task2_dir: Path, out_dir: Path, seed: int) -> None:
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        f1 = task1_dir / f"{split}.parquet"
        f2 = task2_dir / f"{split}.parquet"
        d1 = datasets.Dataset.from_parquet(str(f1))
        d2 = datasets.Dataset.from_parquet(str(f2))
        print(f"[sft:{split}] task1={len(d1)} task2={len(d2)}")
        # Both come from prepare_sft.py, so this cast is a no-op in practice;
        # it guards against incidental field-order drift.
        if d1.features != d2.features:
            print(f"[sft:{split}] features differ -> casting task2 to task1 schema")
            d2 = d2.cast(d1.features)
        merged = datasets.concatenate_datasets([d1, d2]).shuffle(seed=seed)
        merged.to_parquet(str(out_dir / f"{split}.parquet"))
        print(f"[sft:{split}] wrote {len(merged)} rows -> {out_dir / f'{split}.parquet'}")


# ----------------------------- RL mode ------------------------------
def _struct_field_names(struct_type: pa.DataType) -> List[str]:
    return [struct_type.field(i).name for i in range(struct_type.num_fields)]


def _union_extra_info_fields(paths: List[Path]) -> (List[str], List[str]):
    """Union of extra_info fields across the input parquets.

    Returns (string_fields, meta_keys); ``index`` and ``meta`` are handled
    separately (int64 and struct-of-strings respectively).
    """
    str_fields: List[str] = []
    meta_keys: List[str] = []
    for path in paths:
        schema = pq.ParquetFile(str(path)).schema_arrow
        ei = schema.field("extra_info").type
        for name in _struct_field_names(ei):
            if name == "index":
                continue
            if name == "meta":
                meta_type = ei.field("meta").type
                if pa.types.is_struct(meta_type):
                    for mk in _struct_field_names(meta_type):
                        if mk not in meta_keys:
                            meta_keys.append(mk)
                continue
            if name not in str_fields:
                str_fields.append(name)
    return str_fields, meta_keys


def _rl_schema(str_fields: List[str], meta_keys: List[str]) -> pa.Schema:
    prompt_type = pa.list_(
        pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())])
    )
    reward_model_type = pa.struct(
        [pa.field("ground_truth", pa.string()), pa.field("style", pa.string())]
    )
    extra_fields = [pa.field(f, pa.string()) for f in str_fields]
    extra_fields.append(pa.field("index", pa.int64()))
    if meta_keys:
        meta_type = pa.struct([pa.field(k, pa.string()) for k in meta_keys])
        extra_fields.append(pa.field("meta", meta_type))
    return pa.schema(
        [
            pa.field("data_source", pa.string()),
            pa.field("prompt", prompt_type),
            pa.field("ability", pa.string()),
            pa.field("reward_model", reward_model_type),
            pa.field("extra_info", pa.struct(extra_fields)),
        ]
    )


def _s(v: Any) -> str:
    return "" if v is None else str(v)


def _coerce_rl_row(row: Dict[str, Any], str_fields: List[str], meta_keys: List[str]) -> Dict[str, Any]:
    ei = row.get("extra_info") or {}
    if not isinstance(ei, dict):
        ei = {}
    meta = ei.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    new_ei: Dict[str, Any] = {f: _s(ei.get(f, "")) for f in str_fields}
    try:
        new_ei["index"] = int(ei.get("index", 0) or 0)
    except (TypeError, ValueError):
        new_ei["index"] = 0
    if meta_keys:
        new_ei["meta"] = {k: _s(meta.get(k, "")) for k in meta_keys}

    rm = row.get("reward_model") or {}
    if not isinstance(rm, dict):
        rm = {}
    new_rm = {"ground_truth": _s(rm.get("ground_truth", "")), "style": _s(rm.get("style", ""))}

    prompt = row.get("prompt") or []
    new_prompt = [
        {"content": _s(m.get("content", "")), "role": _s(m.get("role", ""))}
        for m in prompt
        if isinstance(m, dict)
    ]
    return {
        "data_source": _s(row.get("data_source", "")),
        "prompt": new_prompt,
        "ability": _s(row.get("ability", "")),
        "reward_model": new_rm,
        "extra_info": new_ei,
    }


def merge_rl(task1_dir: Path, task2_dir: Path, out_dir: Path, batch_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        f1 = task1_dir / f"{split}.parquet"
        f2 = task2_dir / f"{split}.parquet"
        out = out_dir / f"{split}.parquet"
        str_fields, meta_keys = _union_extra_info_fields([f1, f2])
        schema = _rl_schema(str_fields, meta_keys)
        writer = pq.ParquetWriter(str(out), schema, compression="snappy")
        written: Dict[str, int] = {}
        buf: List[Dict[str, Any]] = []

        def flush():
            if not buf:
                return
            writer.write_table(pa.Table.from_pylist(buf, schema=schema))
            buf.clear()

        try:
            for src in (f1, f2):
                pf = pq.ParquetFile(str(src))
                for batch in pf.iter_batches(batch_size=batch_size):
                    for row in batch.to_pylist():
                        r = _coerce_rl_row(row, str_fields, meta_keys)
                        written[r["data_source"]] = written.get(r["data_source"], 0) + 1
                        buf.append(r)
                        if len(buf) >= batch_size:
                            flush()
            flush()
        finally:
            writer.close()
        counts = " ".join(f"{k}={v}" for k, v in sorted(written.items()))
        print(f"[rl:{split}] wrote {sum(written.values())} rows ({counts}) -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sft", "rl"], required=True)
    p.add_argument("--task1_dir", required=True)
    p.add_argument("--task2_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=500)
    args = p.parse_args()

    t1 = Path(os.path.expanduser(args.task1_dir)).resolve()
    t2 = Path(os.path.expanduser(args.task2_dir)).resolve()
    out = Path(os.path.expanduser(args.out_dir)).resolve()
    for d in (t1, t2):
        for split in ("train", "val"):
            f = d / f"{split}.parquet"
            if not f.is_file():
                raise FileNotFoundError(f"missing input parquet: {f}")

    if args.mode == "sft":
        merge_sft(t1, t2, out, seed=args.seed)
    else:
        merge_rl(t1, t2, out, batch_size=args.batch_size)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
