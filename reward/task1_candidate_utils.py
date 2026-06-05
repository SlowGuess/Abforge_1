import json
import math
import os
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


COUNT_PENALTY_HARD_CAP_DEFAULT = 6
COUNT_PENALTY_BASE_RATE_DEFAULT = 0.02
COUNT_PENALTY_STEP_DEFAULT = 0.01
COUNT_PENALTY_CAP_DEFAULT = 0.5


def compute_count_penalty(
    n_pred: int,
    hard_cap: int = COUNT_PENALTY_HARD_CAP_DEFAULT,
    base_rate: float = COUNT_PENALTY_BASE_RATE_DEFAULT,
    step: float = COUNT_PENALTY_STEP_DEFAULT,
    cap: float = COUNT_PENALTY_CAP_DEFAULT,
) -> float:
    """Hard-cap count penalty shared by RL reward and eval.

    Returns 0 when n_pred <= hard_cap. Past the cap, the per-excess rate
    grows progressively (rate_i = base_rate + step*(i-1)), so marginal
    cost rises with excess. Total deduction is clamped to `cap`.

    With defaults (hard_cap=6, base=0.02, step=0.01, cap=0.5), excess →
    penalty: 1→0.02, 2→0.05, 3→0.09, 4→0.14, 5→0.20, 6→0.27, 7→0.35,
    ≥10→0.50.

    The hard cap matches the data-side filter that removes papers with
    n_gt > 6, so any model emitting more than 6 bullets is strictly
    over-generating relative to the worst-case GT size.
    """
    excess = max(0, int(n_pred) - int(hard_cap))
    if excess <= 0:
        return 0.0
    penalty = sum(base_rate + step * (i - 1) for i in range(1, excess + 1))
    return min(cap, penalty)


def compute_count_penalty_from_env(n_pred: int) -> float:
    """Same as compute_count_penalty but reads hyperparams from env vars.

    Env vars (with defaults):
      TASK1_COUNT_HARD_CAP   = 6
      TASK1_COUNT_BASE_RATE  = 0.02
      TASK1_COUNT_STEP       = 0.01
      TASK1_COUNT_CAP        = 0.5
    """
    return compute_count_penalty(
        n_pred,
        hard_cap=int(os.environ.get("TASK1_COUNT_HARD_CAP",
                                    str(COUNT_PENALTY_HARD_CAP_DEFAULT))),
        base_rate=float(os.environ.get("TASK1_COUNT_BASE_RATE",
                                       str(COUNT_PENALTY_BASE_RATE_DEFAULT))),
        step=float(os.environ.get("TASK1_COUNT_STEP",
                                  str(COUNT_PENALTY_STEP_DEFAULT))),
        cap=float(os.environ.get("TASK1_COUNT_CAP",
                                 str(COUNT_PENALTY_CAP_DEFAULT))),
    )


def compute_count_penalty_v2(
    n_pred: int,
    n_gt: int,
    soft_rate: float = 0.005,
    hard_threshold: int = 7,
    hard_rate: float = 0.05,
    cap: float = 0.5,
    free_extra: int = 1,
) -> float:
    """GT-relative count penalty.

    n_pred ≤ n_gt + free_extra: no penalty (free zone).
    n_gt + free_extra < n_pred ≤ hard_threshold: soft_rate per excess bullet above the free zone.
    n_pred > hard_threshold: soft zone penalty + hard_rate per bullet above threshold.

    With defaults (soft=0.005, hard_threshold=7, hard=0.05, free_extra=1):
      n_gt=3: n_pred 4→0.000, 5→0.005, 6→0.010, 7→0.015, 8→0.065
      n_gt=5: n_pred 6→0.000, 7→0.005, 8→0.055
      n_gt=6: n_pred 7→0.000, 8→0.050

    This leaves one exploratory bullet above GT at zero cost and makes count a
    weak preference rather than a dominant reward signal.
    """
    free_limit = int(n_gt) + max(0, int(free_extra))
    soft_excess = max(0, min(int(n_pred), int(hard_threshold)) - free_limit)
    hard_excess = max(0, int(n_pred) - int(hard_threshold))
    return min(cap, soft_rate * soft_excess + hard_rate * hard_excess)


def compute_count_penalty_v2_from_env(n_pred: int, n_gt: int) -> float:
    """Same as compute_count_penalty_v2 but reads params from env vars.

    Env vars (with defaults):
      TASK1_COUNT_SOFT_RATE      = 0.005
      TASK1_COUNT_HARD_THRESHOLD = 7
      TASK1_COUNT_HARD_RATE      = 0.05
      TASK1_COUNT_CAP            = 0.5
      TASK1_COUNT_FREE_EXTRA     = 1
    """
    return compute_count_penalty_v2(
        n_pred,
        n_gt,
        soft_rate=float(os.environ.get("TASK1_COUNT_SOFT_RATE", "0.005")),
        hard_threshold=int(os.environ.get("TASK1_COUNT_HARD_THRESHOLD", "7")),
        hard_rate=float(os.environ.get("TASK1_COUNT_HARD_RATE", "0.05")),
        cap=float(os.environ.get("TASK1_COUNT_CAP", "0.5")),
        free_extra=int(os.environ.get("TASK1_COUNT_FREE_EXTRA", "1")),
    )


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


def compute_wt_p_recall(
    eval_items: List[Dict[str, Any]],
    bullet_spec: Dict[int, float],
    default_spec: float = 1.0,
) -> Optional[float]:
    """v18.1 wt(P) recall = Σ(weight_i × match_score_i × bullet_P_i) / Σ(weight_i).

    For each focus, multiply its match score by the specificity (precision)
    score of the bullet it matched. Unmatched focuses (primary_match=0 or
    None) contribute 0. Missing bullet_spec entries fall back to
    `default_spec` (use 1.0 if you want a graceful degradation when the
    judge forgets the spec block; use 0.0 if you want strict).

    Returns None when total_weight is 0.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for item in eval_items:
        score = item.get("score")
        if score is None:
            continue
        weight = float(item.get("weight", 1))
        total_weight += weight
        pm = item.get("primary_match") or 0
        if pm and pm > 0:
            spec = bullet_spec.get(int(pm), default_spec)
        else:
            spec = 0.0  # no match → no recall contribution
        weighted_sum += weight * float(score) * float(spec)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight
