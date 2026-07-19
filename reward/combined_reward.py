"""Unified ABForge reward service.

ABForge trains a single model on a 1:1 mixture of Task 1 and Task 2. During RL,
verl's reward manager POSTs one request per sample to ``reward_api`` and each
request carries the sample's ``data_source`` field ("abforge_task1" /
"abforge_task2", written by ``dataprocess/prepare_task{1,2}_rl.py``). This
server exposes a single ``/get_reward`` endpoint and routes every request to
the corresponding task-specific judge:

- ``abforge_task1*`` -> ``task1_reward.build_reward_response``
  (specificity-weighted objective matching + structural penalties)
- ``abforge_task2*`` -> ``task2_rubric_reward.build_reward_response``
  (weighted rubric score + format/length penalties)

Both judges call the same OpenAI-compatible endpoint configured via
``JUDGE_API_BASE`` / ``JUDGE_API_KEY`` / ``JUDGE_MODEL``.

The per-task endpoints ``/get_reward_task1`` and ``/get_reward_task2`` are also
served by this process, so single-task runs can point at the same server.
"""
import logging
import os
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from fastapi import FastAPI, Request

import task1_reward as t1
import task2_rubric_reward as t2

logger = logging.getLogger("combined_reward")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()


def _route(data_source: str) -> Tuple[Optional[str], Optional[Callable[[Dict[str, Any]], Coroutine]]]:
    ds = (data_source or "").strip().lower()
    if ds.startswith("abforge_task1"):
        return "task1", t1.build_reward_response
    if ds.startswith("abforge_task2"):
        return "task2", t2.build_reward_response
    return None, None


@app.post("/get_reward")
async def get_reward(request: Request):
    """Unified endpoint: route by the sample's data_source to the right judge."""
    json_data = await request.json()
    data_source = json_data.get("data_source", "")
    label, fn = _route(data_source)
    if fn is None:
        logger.warning("unknown data_source=%r -> returning score 0", data_source)
        return {"score": 0.0, "error": f"unknown_data_source:{data_source}"}
    try:
        result = await fn(json_data)
    except Exception as exc:  # never crash training on one bad sample
        logger.exception("reward error (%s) data_source=%s: %s", label, data_source, exc)
        return {"score": 0.0, "error": f"{label}_exception:{exc}"}
    if isinstance(result, dict):
        result.setdefault("routed_task", label)
    return result


@app.post("/get_reward_task1")
async def get_reward_task1(request: Request):
    return await t1.build_reward_response(await request.json())


@app.post("/get_reward_task2")
async def get_reward_task2(request: Request):
    return await t2.build_reward_response(await request.json())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "combined",
        "judge_model": os.environ.get("JUDGE_MODEL", ""),
        "judge_base": os.environ.get("JUDGE_API_BASE", ""),
        "task1_binary": t1.use_binary_scoring(),
        "task2_binary": t2.use_binary_scoring(),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("COMBINED_REWARD_PORT", "6010"))
    logger.info("combined reward server starting on :%d (judge=%s base=%s)",
                port, os.environ.get("JUDGE_MODEL", ""), os.environ.get("JUDGE_API_BASE", ""))
    uvicorn.run(app, host="0.0.0.0", port=port)
