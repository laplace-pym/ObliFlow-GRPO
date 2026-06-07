from collections import defaultdict
from typing import Any

import numpy as np

from .action_parser import parse_response
from .schema import StepRecord


def _as_list(values: Any, size: int, default: Any = None) -> list[Any]:
    if values is None:
        return [default for _ in range(size)]
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    return [values for _ in range(size)]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value[0]
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    try:
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value[0]
        return bool(value)
    except Exception:
        return default


def build_step_records(batch, env_name: str) -> list[StepRecord]:
    non_tensor = batch.non_tensor_batch
    size = len(non_tensor["traj_uid"])

    uids = _as_list(non_tensor.get("uid"), size, "")
    traj_uids = _as_list(non_tensor.get("traj_uid"), size, "")
    step_ids = _as_list(non_tensor.get("step_id"), size, None)
    raw_responses = _as_list(non_tensor.get("text_actions"), size, "")
    task_texts = _as_list(non_tensor.get("task_text"), size, "")
    anchor_obs = _as_list(non_tensor.get("anchor_obs"), size, "")
    next_obs = _as_list(non_tensor.get("next_obs"), size, "")
    rewards = _as_list(non_tensor.get("rewards"), size, 0.0)
    dones = _as_list(non_tensor.get("dones"), size, False)
    infos = _as_list(non_tensor.get("env_infos"), size, {})
    active_masks = _as_list(non_tensor.get("active_masks"), size, True)
    is_action_valid = _as_list(non_tensor.get("is_action_valid"), size, True)
    episode_rewards = _as_list(non_tensor.get("episode_rewards"), size, 0.0)

    records: list[StepRecord] = []
    for i in range(size):
        info = infos[i] if isinstance(infos[i], dict) else {}
        raw_response = raw_responses[i]
        parsed = parse_response(raw_response, env_name=env_name)
        record = StepRecord(
            uid=str(uids[i]),
            traj_uid=str(traj_uids[i]),
            step_id=int(step_ids[i]) if step_ids[i] is not None else i,
            env_name=env_name,
            task_text=str(task_texts[i] or ""),
            anchor_obs=anchor_obs[i],
            raw_response=str(raw_response or ""),
            parsed_action=parsed,
            next_obs=next_obs[i],
            reward=_to_float(rewards[i]),
            done=_to_bool(dones[i]),
            info=info,
            active_mask=_to_bool(active_masks[i], True),
            is_action_valid=_to_bool(is_action_valid[i], True),
            episode_reward=_to_float(episode_rewards[i]),
        )
        records.append(record)
    return records


def group_records_by_traj(records: list[StepRecord]) -> dict[str, list[StepRecord]]:
    grouped: dict[str, list[StepRecord]] = defaultdict(list)
    for record in records:
        if record.active_mask:
            grouped[record.traj_uid].append(record)
    for traj_uid in grouped:
        grouped[traj_uid].sort(key=lambda item: item.step_id)
    return dict(grouped)
