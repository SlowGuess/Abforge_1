import json
import math
import re
from typing import Any, Dict, List, Optional


CANDIDATE_PATTERN = re.compile(r"<Candidate>(.*?)</Candidate>", re.DOTALL)
FIELD_PATTERN_TEMPLATE = r"<{field}>(.*?)</{field}>"
RESULT_PATTERN = re.compile(r"<Result>(.*?)</Result>", re.DOTALL | re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def parse_candidates(candidates_text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for block in CANDIDATE_PATTERN.findall(candidates_text or ""):
        focus = extract_field(block, "Investigation_Focus")
        description = extract_field(block, "Description_Text")
        candidate_id = extract_field(block, "ID")
        if focus:
            candidates.append(
                {
                    "candidate_id": int(candidate_id) if candidate_id and candidate_id.isdigit() else None,
                    "focus": normalize_whitespace(focus),
                    "description": normalize_whitespace(description),
                    "weight": heuristic_weight(focus, description),
                }
            )
    return candidates


def extract_field(block: str, field: str) -> str:
    match = re.search(FIELD_PATTERN_TEMPLATE.format(field=field), block or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def heuristic_weight(focus: str, description: str) -> int:
    merged = f"{focus} {description}".lower()
    critical_markers = (
        "central claim",
        "core",
        "necessity",
        "causal",
        "main mechanism",
        "critical component",
    )
    if any(marker in merged for marker in critical_markers):
        return 3
    return 2


def extract_result_block(response_text: str) -> str:
    response_text = response_text or ""
    match = RESULT_PATTERN.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def approx_token_count(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    whitespace_tokens = len(text.split())
    char_tokens = math.ceil(len(text) / 4)
    return max(whitespace_tokens, char_tokens)


def compute_weighted_score(eval_items: List[Dict[str, Any]]) -> Optional[float]:
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

