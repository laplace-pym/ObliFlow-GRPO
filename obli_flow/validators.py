import re
from typing import Any


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "in", "on", "with", "for", "of", "is", "are",
    "your", "you", "must", "need", "find", "buy", "put", "pick", "place", "then", "from",
}


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).lower()


def keyword_set(text: Any, min_len: int = 3) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-]+", normalize_text(text))
    return {token for token in tokens if len(token) >= min_len and token not in STOPWORDS}


def token_overlap(a: Any, b: Any) -> float:
    a_set = keyword_set(a)
    b_set = keyword_set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, min(len(a_set), len(b_set)))


def exact_contains(text: Any, target: Any) -> float:
    text_l = normalize_text(text)
    target_l = normalize_text(target).strip()
    if not target_l:
        return 0.0
    return 1.0 if target_l in text_l else 0.0


def info_won(info: dict[str, Any]) -> bool:
    return bool(info.get("won", False))


def task_score(info: dict[str, Any]) -> float:
    try:
        return float(info.get("task_score", 0.0))
    except Exception:
        return 0.0


def is_noop_feedback(text: Any) -> bool:
    text_l = normalize_text(text)
    noop_markers = (
        "nothing happens",
        "you can't",
        "you cannot",
        "not carrying anything",
        "invalid",
    )
    return any(marker in text_l for marker in noop_markers)


def validate_admissible_action(action: str, admissible_actions: Any) -> tuple[bool, float, str]:
    if admissible_actions is None:
        return True, 0.5, "missing admissible action list"
    if isinstance(admissible_actions, dict):
        candidates = []
        if admissible_actions.get("has_search_bar"):
            candidates.append("search[")
        candidates.extend(admissible_actions.get("clickables", []))
    else:
        candidates = list(admissible_actions) if isinstance(admissible_actions, (list, tuple, set)) else [str(admissible_actions)]
    action_l = normalize_text(action)
    for candidate in candidates:
        candidate_l = normalize_text(candidate)
        if candidate_l and (action_l == candidate_l or candidate_l in action_l or action_l in candidate_l):
            return True, 1.0, "action matches available action"
    return False, 0.0, "action not in available actions"


def validate_goal_overlap(action_or_obs: Any, task_text: Any) -> tuple[bool, float, str]:
    score = token_overlap(action_or_obs, task_text)
    return score > 0.0, score, "goal keyword overlap" if score > 0 else "no goal keyword overlap"
