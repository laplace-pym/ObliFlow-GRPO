import re

from .schema import ParsedAction


THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


def parse_response(raw_response: str, env_name: str = "") -> ParsedAction:
    raw_response = "" if raw_response is None else str(raw_response)
    think_match = THINK_RE.search(raw_response)
    action_match = ACTION_RE.search(raw_response)
    think = think_match.group(1).strip() if think_match else ""
    action = action_match.group(1).strip() if action_match else ""
    valid_format = bool(think_match and action_match and action)
    action_type, action_args = classify_action(action, env_name)
    if not valid_format:
        action_type = "invalid"
    return ParsedAction(
        think=think,
        action=action,
        valid_format=valid_format,
        action_type=action_type,
        action_args=action_args,
    )


def classify_action(action: str, env_name: str = "") -> tuple[str, dict]:
    action = "" if action is None else str(action).strip()
    action_l = action.lower()
    env_l = env_name.lower()

    if "webshop" in env_l:
        search_match = re.match(r"search\[(.*)\]\s*$", action_l)
        if search_match:
            return "search", {"query": search_match.group(1).strip()}
        click_match = re.match(r"click\[(.*)\]\s*$", action_l)
        if click_match:
            target = click_match.group(1).strip()
            if "buy" in target:
                return "buy", {"target": target}
            return "click", {"target": target}
        return ("invalid", {}) if not action_l else ("web_action", {"text": action_l})

    if "alfworld" in env_l or "alfred" in env_l:
        if action_l in {"look", "inventory", "help", "pass"}:
            return action_l, {}
        if action_l.startswith(("go to ", "goto ")):
            return "goto", {"target": action_l.replace("go to ", "", 1).replace("goto ", "", 1).strip()}
        verb_aliases = {
            "pick": ("pick", "take", "grab"),
            "put": ("put", "place"),
            "open": ("open",),
            "close": ("close",),
            "toggle": ("toggle", "turn on", "turn off"),
            "heat": ("heat",),
            "cool": ("cool",),
            "clean": ("clean", "wash"),
            "slice": ("slice",),
            "examine": ("examine", "inspect"),
        }
        for action_type, prefixes in verb_aliases.items():
            if any(action_l.startswith(prefix) for prefix in prefixes):
                return action_type, {"text": action_l}
        return ("invalid", {}) if not action_l else ("env_action", {"text": action_l})

    if action_l.startswith("search["):
        return "search", {"query": action[7:-1].strip("[] ")}
    if action_l.startswith("click["):
        return "click", {"target": action[6:-1].strip("[] ")}
    return ("invalid", {}) if not action_l else ("action", {"text": action_l})
