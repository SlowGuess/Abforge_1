"""Task 2 reward service — rubric v2.

Rubric v2 behavior:
- rubric is parsed with task2_rubric_utils_v2.parse_rubric_items_v2, which
  reads fixed `level`/`weight` attributes instead of heuristic weighting
- judge prompt tells the LLM that weights are pre-assigned and must NOT be
  reassigned; no Reference_Design is shown to the judge (RL reward should
  not bias toward mimicking a reference plan)
- warns (not fails) if fewer than 10 items parsed; warns if some items
  failed to score — useful for diagnosing truncated judge outputs
- length_penalty and format_penalty logic reused; tuning is via env vars:
    TASK2_LENGTH_THRESHOLD          default 2400
    TASK2_LENGTH_PENALTY_RATE       default 0.03
    TASK2_JUDGE_MAX_OUTPUT_TOKENS   default 1500
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from openai import AsyncOpenAI

PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from task2_rubric_utils_v2 import (  # noqa: E402
    EXPECTED_ITEM_COUNT,
    approx_token_count,
    coerce_extra_info,
    compute_weighted_score,
    extract_proposed_plan,
    parse_rubric_items_v2,
)


logger = logging.getLogger("task2_rubric_reward")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


app = FastAPI()


def use_binary_scoring() -> bool:
    value = (os.environ.get("TASK2_BINARY_SCORING", "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def has_required_sections(plan_text: str) -> bool:
    lowered = (plan_text or "").lower()
    required_sections = (
        "objective:",
        "baseline setup:",
        "variants:",
        "fixed protocols & metrics:",
    )
    return all(section in lowered for section in required_sections)


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _two_tier_overlong_penalty(token_count: int, t1: int, r1: float, t2: int, r2: float) -> float:
    overlong_t1 = max(0, token_count - t1)
    penalty = (overlong_t1 / 100.0) * r1
    if t2 > 0 and r2 > 0:
        overlong_t2 = max(0, token_count - t2)
        penalty += (overlong_t2 / 100.0) * r2
    return penalty


def compute_rule_penalty(raw_response: str, plan_text: str) -> Dict[str, float]:
    raw = raw_response or ""
    raw_lower = raw.lower()
    has_open = "<proposed_plan>" in raw_lower
    has_close = "</proposed_plan>" in raw_lower

    # Token counts are computed unconditionally so the response logs
    # still expose verbosity even in the disabled-penalty ablation.
    plan_t1 = int(os.environ.get("TASK2_LENGTH_THRESHOLD", "2400"))
    plan_r1 = float(os.environ.get("TASK2_LENGTH_PENALTY_RATE", "0.03"))
    plan_t2 = int(os.environ.get("TASK2_LENGTH_THRESHOLD_TIER2", "0"))
    plan_r2 = float(os.environ.get("TASK2_LENGTH_PENALTY_RATE_TIER2", "0"))
    total_t1 = int(os.environ.get("TASK2_TOTAL_LENGTH_THRESHOLD", "0"))
    total_r1 = float(os.environ.get("TASK2_TOTAL_LENGTH_PENALTY_RATE", "0"))
    total_t2 = int(os.environ.get("TASK2_TOTAL_LENGTH_THRESHOLD_TIER2", "0"))
    total_r2 = float(os.environ.get("TASK2_TOTAL_LENGTH_PENALTY_RATE_TIER2", "0"))
    token_count = approx_token_count(plan_text)
    total_token_count = approx_token_count(raw)

    # Ablation switch (paper §6.4 "w/o rule-based penalty"): when enabled,
    # both format_penalty and length_penalty are forced to 0 so the reward
    # collapses to rubric_score alone. The token counts above are still
    # returned for logging/diagnostics. Default (unset) keeps prior behavior.
    if _truthy_env("TASK2_DISABLE_RULE_PENALTY"):
        return {
            "format_penalty": 0.0,
            "length_penalty": 0.0,
            "token_count": float(token_count),
            "total_token_count": float(total_token_count),
        }

    # Format penalty:
    #   plan tag truncated (open without close)  : -0.25 (unambiguous truncation)
    #   plan tag entirely missing                : -0.25 (model never opened the block)
    #   required sections missing inside plan    : -0.25 (additive)
    # Note: truncation magnitude was originally 0.5, but combined with lr=2e-5 it
    # produced a single-step advantage spike that collapsed entropy and locked
    # the policy into the 4096-cap mode. 0.25 keeps detection while staying gentle.
    format_penalty = 0.0
    if has_open and not has_close:
        format_penalty += 0.25
    elif not has_open:
        format_penalty += 0.25
    if not has_required_sections(plan_text):
        format_penalty += 0.25

    # Plan-only length penalty (two tiers, additive per 100 tokens).
    length_penalty = _two_tier_overlong_penalty(token_count, plan_t1, plan_r1, plan_t2, plan_r2)
    # Total-response length penalty (constrains <Think> bloat that plan-only penalty cannot see).
    total_length_penalty = _two_tier_overlong_penalty(total_token_count, total_t1, total_r1, total_t2, total_r2)
    length_penalty += total_length_penalty

    return {
        "format_penalty": float(format_penalty),
        "length_penalty": float(length_penalty),
        "token_count": float(token_count),
        "total_token_count": float(total_token_count),
    }


def render_rubric_for_judge(rubric_items: List[Dict[str, Any]]) -> str:
    """Keep the original attributes visible so the judge can calibrate on level."""
    return "\n".join(
        f'<item num="{item["item_num"]}" level="{item["level"]}" weight="{item["weight"]}">'
        f'{item["criterion"]}</item>'
        for item in rubric_items
    )


def create_evaluation_prompt(
    paper_context: str,
    goal: str,
    generated_design: str,
    rubric_items: List[Dict[str, Any]],
) -> str:
    binary = use_binary_scoring()
    if binary:
        score_set = "{0, 1}"
        score_lines = (
            "   - 1 : the generated design fully satisfies the rubric item\n"
            "   - 0 : the generated design does not satisfy it (or only partially)"
        )
        score_format = "0|1"
        strictness = (
            "Be strict. Treat ambiguous or partial evidence as 0, not 1. "
            "Evaluate only against the specific condition stated in the rubric item."
        )
    else:
        score_set = "{0, 0.5, 1}"
        score_lines = (
            "   - 1 : the generated design fully satisfies the rubric item\n"
            "   - 0.5 : the generated design partially addresses it (notable gaps or ambiguities)\n"
            "   - 0 : the generated design does not satisfy it"
        )
        score_format = "0|0.5|1"
        strictness = (
            "Be strict. Evaluate only against the specific condition stated in the "
            "rubric item; ignore whether other items would be satisfied. Generic "
            "scientific language should not receive full credit."
        )

    rubric_block = render_rubric_for_judge(rubric_items)
    expected = EXPECTED_ITEM_COUNT

    return f"""You are a rigorous scientific reviewer acting as an automated evaluation judge for ablation experiment designs. Your task is to evaluate whether a generated ablation plan satisfies each rubric criterion.

<Paper_Context>
{paper_context}
</Paper_Context>

<Ablation_Goal>
{goal}
</Ablation_Goal>

<Generated_Design>
{generated_design}
</Generated_Design>

<Evaluation_Rubric>
{rubric_block}
</Evaluation_Rubric>

**Evaluation Instructions:**
The rubric has exactly {expected} items. Each item already carries a fixed `level` and `weight` — do NOT reassign them. For each rubric item you must:

1. Produce a brief critique (1–3 sentences) explaining whether and how the generated design satisfies the requirement.
2. Assign a satisfaction score in {score_set}:
{score_lines}

{strictness}

**Output Format (strictly follow this XML format; produce exactly {expected} <item> blocks):**
<Evaluation>
<item num="1">
<critique>[brief analysis]</critique>
<score>[{score_format}]</score>
</item>
<item num="2">
<critique>[brief analysis]</critique>
<score>[{score_format}]</score>
</item>
...
<item num="{expected}">
<critique>[brief analysis]</critique>
<score>[{score_format}]</score>
</item>
</Evaluation>

Do not output any text outside <Evaluation>...</Evaluation>.
"""


def parse_score(value: str) -> Optional[float]:
    cleaned = (value or "").strip().strip("[](){}").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        score = float(match.group(0))
    except Exception:
        return None
    if use_binary_scoring():
        allowed = (0.0, 1.0)
    else:
        allowed = (0.0, 0.5, 1.0)
    return min(allowed, key=lambda a: (abs(a - score), a))


def parse_eval_response(response: str, rubric_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    text = response or ""
    for item in rubric_items:
        item_num = item["item_num"]
        pattern = (
            rf'<\s*item\b[^>]*(?:num|number|id)\s*=\s*["\']?\s*{item_num}\s*["\']?[^>]*>'
            rf'(.*?)</\s*item\s*>'
        )
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        critique = ""
        score: Optional[float] = None
        if match:
            block = match.group(1)
            critique_match = re.search(
                r"<\s*critique\s*>(.*?)</\s*critique\s*>",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            score_match = re.search(
                r"<\s*score\s*>(.*?)</\s*score\s*>",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            critique = critique_match.group(1).strip() if critique_match else ""
            score = parse_score(score_match.group(1).strip()) if score_match else None
        results.append(
            {
                "item_num": item_num,
                "level": item["level"],
                "weight": item["weight"],
                "criterion": item["criterion"],
                "critique": critique,
                "score": score,
            }
        )
    return results


_judge_client: Optional[AsyncOpenAI] = None
_judge_semaphore: Optional[asyncio.Semaphore] = None


def _get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is not None:
        return _judge_client

    base_url = os.environ.get("JUDGE_API_BASE")
    api_key = os.environ.get("JUDGE_API_KEY")

    if not base_url:
        raise RuntimeError("JUDGE_API_BASE is not set.")
    if not api_key:
        raise RuntimeError("JUDGE_API_KEY is not set.")

    _judge_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=int(os.environ.get("TASK2_JUDGE_REQUEST_TIMEOUT", "300")),
        max_retries=0,
    )
    return _judge_client


def _get_judge_semaphore() -> asyncio.Semaphore:
    global _judge_semaphore
    if _judge_semaphore is None:
        max_concurrent = int(os.environ.get("TASK2_JUDGE_MAX_CONCURRENT", "64"))
        _judge_semaphore = asyncio.Semaphore(max_concurrent)
    return _judge_semaphore


def _get_judge_extra_body() -> Dict[str, Any]:
    raw = os.environ.get("TASK2_JUDGE_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("TASK2_JUDGE_EXTRA_BODY must decode to a JSON object")
        return parsed
    except Exception as exc:
        logger.warning("ignoring malformed TASK2_JUDGE_EXTRA_BODY (%s): %s", exc, raw)
        return {}


async def _call_judge_chat(messages: List[Dict[str, Any]]) -> str:
    model = os.environ.get("JUDGE_MODEL")
    if not model:
        raise RuntimeError("JUDGE_MODEL is not set.")

    max_retries = int(os.environ.get("TASK2_JUDGE_MAX_RETRIES", "5"))
    max_output_tokens = int(os.environ.get("TASK2_JUDGE_MAX_OUTPUT_TOKENS", "1500"))
    extra_body = _get_judge_extra_body()
    client = _get_judge_client()
    semaphore = _get_judge_semaphore()

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=0,
                    extra_body=extra_body or None,
                )
            if not response.choices:
                raise RuntimeError(f"judge chat.completions returned no choices: {response}")
            content = response.choices[0].message.content or ""
            content = content.strip()
            if not content:
                raise RuntimeError(f"judge chat.completions returned empty content: {response}")
            return content
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"judge chat.completions request failed after {max_retries} attempts: {last_error}")


async def call_llm_judge(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a rigorous scientific reviewer. Return only the requested XML.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return await _call_judge_chat(messages)


async def build_reward_response(raw_request: Dict[str, Any]) -> Dict[str, Any]:
    extra_info = coerce_extra_info(raw_request.get("extra_info"))
    raw_response = raw_request.get("response_str", "")
    if isinstance(raw_response, list):
        raw_response = raw_response[0] if raw_response else ""

    plan_text = extract_proposed_plan(raw_response)
    rubric_items = parse_rubric_items_v2(extra_info.get("rubric", ""))

    if not rubric_items:
        logger.warning("no rubric items parsed — data_source=%s", raw_request.get("data_source"))
        return {
            "score": 0.0,
            "rubric_score": 0.0,
            "format_penalty": 0.0,
            "length_penalty": 0.0,
            "token_count": 0.0,
            "error": "no_rubric_items",
        }

    if len(rubric_items) != EXPECTED_ITEM_COUNT:
        logger.warning(
            "rubric parsed %d items, expected %d — data_source=%s",
            len(rubric_items),
            EXPECTED_ITEM_COUNT,
            raw_request.get("data_source"),
        )

    prompt = create_evaluation_prompt(
        paper_context=extra_info.get("paper_context", ""),
        goal=extra_info.get("goal", ""),
        generated_design=plan_text,
        rubric_items=rubric_items,
    )
    max_validation_retries = int(os.environ.get("TASK2_JUDGE_VALIDATION_RETRIES", "2"))
    miss_threshold = max(1, len(rubric_items) // 2)

    raw_judge_output = ""
    eval_items: List[Dict[str, Any]] = []
    missing = 0
    for attempt in range(max_validation_retries + 1):
        raw_judge_output = await call_llm_judge(prompt)
        eval_items = parse_eval_response(raw_judge_output, rubric_items)
        missing = sum(1 for it in eval_items if it.get("score") is None)
        if missing == 0:
            break
        if attempt < max_validation_retries and missing >= miss_threshold:
            logger.warning(
                "validation retry %d/%d: %d/%d items missing or invalid (threshold %d)",
                attempt + 1,
                max_validation_retries,
                missing,
                len(eval_items),
                miss_threshold,
            )
            continue
        break

    if missing:
        logger.warning(
            "final judge result missed %d/%d items after %d validation attempts",
            missing,
            len(eval_items),
            max_validation_retries + 1,
        )

    rubric_score = compute_weighted_score(eval_items)
    if rubric_score is None:
        rubric_score = 0.0

    penalties = compute_rule_penalty(raw_response=raw_response, plan_text=plan_text)
    total_score = max(
        0.0,
        float(rubric_score) - penalties["format_penalty"] - penalties["length_penalty"],
    )

    return {
        "score": total_score,
        "rubric_score": float(rubric_score),
        "format_penalty": penalties["format_penalty"],
        "length_penalty": penalties["length_penalty"],
        "token_count": penalties["token_count"],
        "missing_items": missing,
        "binary_scoring": use_binary_scoring(),
        "eval_items": eval_items,
        "raw_judge_output": raw_judge_output,
    }


@app.post("/get_reward_task2")
async def get_reward_task2(request: Request):
    json_data = await request.json()
    result = await build_reward_response(json_data)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "version": "v2", "binary_scoring": use_binary_scoring()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TASK2_REWARD_PORT", "6011"))
    uvicorn.run(app, host="0.0.0.0", port=port)
