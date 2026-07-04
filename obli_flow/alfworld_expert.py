import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ALFWORLD_ROOT = "/home/tiger/.cache/alfworld/json_2.1.1"


def normalize_task_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def task_key(task_text: Any) -> str:
    normalized = normalize_task_text(task_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_uid(path: Any) -> str:
    value = str(path or "")
    marker = "json_2.1.1/"
    if marker in value:
        value = value.split(marker, 1)[1]
    value = value.replace("\\", "/").strip("/")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def traj_data_path_from_gamefile(gamefile: Any) -> Path | None:
    if not gamefile:
        return None
    path = Path(str(gamefile))
    if path.name == "traj_data.json":
        return path
    if path.name == "game.tw-pddl":
        return path.with_name("traj_data.json")
    if path.is_dir():
        return path / "traj_data.json"
    return None


def load_alfworld_expert(traj_data_path: str | Path) -> dict[str, Any]:
    path = Path(traj_data_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task = _first_task_desc(data)
    high_descs = _first_high_descs(data)
    high_pddl = _compact_high_pddl(data.get("plan", {}).get("high_pddl") or [])
    task_type = data.get("task_type") or _task_type_from_path(path)
    gamefile = path.with_name("game.tw-pddl")

    return {
        "source_path": str(path),
        "source_uid": source_uid(path),
        "gamefile": str(gamefile),
        "split": _split_from_path(path),
        "task_type": task_type,
        "task": task,
        "task_key": task_key(task),
        "high_descs": high_descs,
        "high_pddl": high_pddl,
        "pddl_params": data.get("pddl_params") or {},
    }


def expert_from_info(info: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(info, dict):
        return None
    if isinstance(info.get("expert"), dict):
        return info["expert"]

    for key in ("traj_data_path", "source_path", "alfworld_traj_data_path"):
        if info.get(key):
            path = Path(str(info[key]))
            if path.exists():
                return load_alfworld_expert(path)

    for key in ("extra.gamefile", "gamefile", "game_file", "game_file_path"):
        path = traj_data_path_from_gamefile(info.get(key))
        if path is not None and path.exists():
            return load_alfworld_expert(path)
    return None


def compact_expert_for_prompt(expert: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(expert, dict):
        return None
    return {
        "task_type": expert.get("task_type"),
        "task": expert.get("task"),
        "high_level_human_steps": expert.get("high_descs") or [],
        "high_level_pddl_actions": expert.get("high_pddl") or [],
        "pddl_params": expert.get("pddl_params") or {},
    }


def expert_signature(expert: dict[str, Any] | None) -> str | None:
    if not expert:
        return None
    compact = compact_expert_for_prompt(expert)
    raw = json.dumps(compact, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def iter_alfworld_traj_data(root: str | Path, splits: Iterable[str] | None = None) -> list[Path]:
    root_path = Path(root)
    if splits:
        paths: list[Path] = []
        for split in splits:
            split_path = root_path / split
            if split_path.exists():
                paths.extend(split_path.rglob("traj_data.json"))
        return sorted(paths)
    return sorted(root_path.rglob("traj_data.json"))


def _first_task_desc(data: dict[str, Any]) -> str:
    anns = data.get("turk_annotations", {}).get("anns", [])
    if isinstance(anns, list):
        for ann in anns:
            task = str((ann or {}).get("task_desc", "")).strip()
            if task:
                return task
    return str(data.get("task_desc") or data.get("goal") or "").strip()


def _first_high_descs(data: dict[str, Any]) -> list[str]:
    anns = data.get("turk_annotations", {}).get("anns", [])
    if isinstance(anns, list):
        for ann in anns:
            high_descs = (ann or {}).get("high_descs")
            if isinstance(high_descs, list) and high_descs:
                return [str(item).strip() for item in high_descs if str(item).strip()]
    template_descs = data.get("template", {}).get("high_descs")
    if isinstance(template_descs, list):
        return [str(item).strip() for item in template_descs if str(item).strip()]
    return []


def _compact_high_pddl(high_pddl: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for idx, item in enumerate(high_pddl):
        if not isinstance(item, dict):
            continue
        discrete = item.get("discrete_action") if isinstance(item.get("discrete_action"), dict) else {}
        planner = item.get("planner_action") if isinstance(item.get("planner_action"), dict) else {}
        action = discrete.get("action") or planner.get("action") or ""
        args = discrete.get("args") or _planner_args(planner)
        compact.append(
            {
                "index": int(item.get("high_idx", idx)),
                "action": str(action),
                "args": [str(arg) for arg in args],
            }
        )
    return compact


def _planner_args(planner: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key in ("objectId", "receptacleObjectId", "coordinateObjectId", "coordinateReceptacleObjectId", "location"):
        value = planner.get(key)
        if not value:
            continue
        if isinstance(value, list) and value:
            args.append(str(value[0]))
        else:
            args.append(str(value))
    return args


def _task_type_from_path(path: Path) -> str:
    try:
        return path.parent.parent.name.split("-", 1)[0]
    except Exception:
        return ""


def _split_from_path(path: Path) -> str:
    parts = path.parts
    if "json_2.1.1" in parts:
        idx = parts.index("json_2.1.1")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""
