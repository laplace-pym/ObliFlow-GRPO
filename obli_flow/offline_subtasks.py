import json
from pathlib import Path
from typing import Any

from obli_flow.alfworld_expert import normalize_task_text, source_uid, task_key, traj_data_path_from_gamefile
from obli_flow.llm_client import cfg_get, fallback_to_rules, warn_once
from obli_flow.llm_obligations import obligations_from_subtask_data
from obli_flow.schema import ObligationNode


_STORES: dict[str, "OfflineSubtaskStore"] = {}


class OfflineSubtaskStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.entries: list[dict[str, Any]] = []
        self.by_source: dict[str, dict[str, Any]] = {}
        self.by_task: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def lookup(self, task_text: str, info: dict[str, Any] | None, *, match_task_text: bool = True) -> dict[str, Any] | None:
        for key in _candidate_source_keys(info):
            entry = self.by_source.get(key)
            if entry is not None:
                return entry

        if not match_task_text:
            return None
        candidates = self.by_task.get(task_key(task_text)) or self.by_task.get(normalize_task_text(task_text)) or []
        if not candidates:
            return None
        if len(candidates) > 1:
            warn_once(
                "offline_subtasks_ambiguous_task",
                "Warning: multiple offline subtask entries match the same task text; using the first match. Prefer matching by ALFWorld extra.gamefile.",
            )
        return candidates[0]

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not isinstance(entry, dict) or not entry.get("subtasks"):
                    continue
                self.entries.append(entry)
                self._index_entry(entry)

    def _index_entry(self, entry: dict[str, Any]) -> None:
        source_keys = {
            entry.get("source_uid"),
            entry.get("source_path"),
            entry.get("gamefile"),
        }
        for value in (entry.get("source_path"), entry.get("gamefile")):
            source_keys.update(_path_keys(value))
        for key in source_keys:
            if key:
                self.by_source[str(key)] = entry

        text = entry.get("task") or ""
        for key in {entry.get("task_key"), task_key(text), normalize_task_text(text)}:
            if key:
                self.by_task.setdefault(str(key), []).append(entry)


def try_build_offline_obligations(
    env_name: str,
    traj_uid: str,
    task_text: str,
    info: dict[str, Any] | None = None,
    config: Any = None,
) -> list[ObligationNode] | None:
    path = str(cfg_get(config, "offline_subtask_path", "") or "")
    if not path or not bool(cfg_get(config, "use_offline_subtasks", True)):
        return None

    try:
        store = _get_store(path)
        entry = store.lookup(
            task_text=task_text,
            info=info,
            match_task_text=bool(cfg_get(config, "offline_subtask_match_task_text", True)),
        )
        if entry is None:
            return None
        return obligations_from_subtask_data(
            {"subtasks": entry.get("subtasks") or []},
            env_name=env_name,
            traj_uid=traj_uid,
            task_text=task_text,
            source="offline_expert",
            extra_metadata={
                "offline_subtask_path": path,
                "offline_source_uid": entry.get("source_uid"),
                "offline_source_path": entry.get("source_path"),
            },
        )
    except Exception as exc:
        if not fallback_to_rules(config):
            raise
        warn_once("offline_subtasks_failed", f"Warning: failed to load offline subtasks once; falling back to LLM/rule obligations. Error: {exc}")
        return None


def _get_store(path: str) -> OfflineSubtaskStore:
    if path not in _STORES:
        _STORES[path] = OfflineSubtaskStore(path)
    return _STORES[path]


def _candidate_source_keys(info: dict[str, Any] | None) -> set[str]:
    keys: set[str] = set()
    if not isinstance(info, dict):
        return keys

    for name in ("source_uid", "alfworld_source_uid"):
        if info.get(name):
            keys.add(str(info[name]))

    for name in ("traj_data_path", "source_path", "alfworld_traj_data_path"):
        keys.update(_path_keys(info.get(name)))

    for name in ("extra.gamefile", "gamefile", "game_file", "game_file_path"):
        gamefile = info.get(name)
        keys.update(_path_keys(gamefile))
        traj_data = traj_data_path_from_gamefile(gamefile)
        if traj_data is not None:
            keys.update(_path_keys(traj_data))
    return keys


def _path_keys(value: Any) -> set[str]:
    if not value:
        return set()
    text = str(value)
    keys = {text, text.replace("\\", "/")}
    try:
        path = Path(text)
        keys.add(str(path))
        keys.add(path.as_posix())
        if path.name == "game.tw-pddl":
            keys.add(path.with_name("traj_data.json").as_posix())
            keys.add(source_uid(path.with_name("traj_data.json")))
        elif path.name == "traj_data.json":
            keys.add(path.with_name("game.tw-pddl").as_posix())
            keys.add(source_uid(path))
        else:
            keys.add(source_uid(path))
    except Exception:
        pass
    return {key for key in keys if key}
