"""Rubric v2 utilities for Task 2.

v2 rubric: each <item> carries fixed `level` and `weight` attributes.
Total of exactly 10 items per rubric. Σweight = 20 (3 critical x3 + 4 major x2 + 3 minor x1).

Kept separate from task2_rubric_utils.py so the original v1 parsing path
(heuristic weight) stays available for any legacy pipeline.
"""
import json
import math
import re
from typing import Any, Dict, List, Optional


RUBRIC_BLOCK_RE = re.compile(r"<Rubric>(.*?)</Rubric>", re.DOTALL | re.IGNORECASE)
ITEM_V2_RE = re.compile(
    r'<item\s+num="(\d+)"\s+level="(critical|major|minor)"\s+weight="(\d+)"\s*>'
    r'(.*?)</item>',
    re.DOTALL | re.IGNORECASE,
)
PLAN_PATTERN = re.compile(r'<Proposed_Plan>(.*?)</Proposed_Plan>', re.DOTALL | re.IGNORECASE)

EXPECTED_ITEM_COUNT = 10


def parse_rubric_items_v2(rubric_text: str) -> List[Dict[str, Any]]:
    if not rubric_text:
        return []
    match = RUBRIC_BLOCK_RE.search(rubric_text)
    inner = match.group(1) if match else rubric_text
    items: List[Dict[str, Any]] = []
    for raw_num, level, raw_weight, raw_text in ITEM_V2_RE.findall(inner):
        items.append(
            {
                "item_num": int(raw_num),
                "level": level.lower(),
                "weight": int(raw_weight),
                "criterion": normalize_whitespace(raw_text),
            }
        )
    return items


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_proposed_plan(response_text: str) -> str:
    response_text = response_text or ""
    match = PLAN_PATTERN.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def coerce_extra_info(extra_info: Any) -> Dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        extra_info = extra_info.strip()
        if not extra_info:
            return {}
        try:
            return json.loads(extra_info)
        except json.JSONDecodeError:
            return {"raw_extra_info": extra_info}
    return {}


def approx_token_count(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    whitespace_tokens = len(text.split())
    char_tokens = math.ceil(len(text) / 4)
    return max(whitespace_tokens, char_tokens)


def compute_weighted_score(eval_items: List[Dict[str, Any]]) -> Optional[float]:
    """Σ(w·s) / Σ(w), using the pre-assigned weights from the rubric.
    Items whose score failed to parse are excluded from both numerator and
    denominator, matching the offline evaluator's behavior.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for item in eval_items:
        score = item.get("score")
        if score is None:
            continue
        weight = float(item.get("weight", 1))
        total_weight += weight
        weighted_sum += weight * float(score)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight
