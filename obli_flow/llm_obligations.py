import json
import re
from typing import Any

from obli_flow.alfworld_expert import compact_expert_for_prompt, expert_from_info, expert_signature
from obli_flow.llm_client import LLMUnavailable, chat_json, cfg_get, fallback_to_rules, llm_enabled, warn_once
from obli_flow.schema import ObligationNode


def try_build_llm_obligations(
    env_name: str,
    traj_uid: str,
    task_text: str,
    info: dict[str, Any] | None = None,
    config: Any = None,
) -> list[ObligationNode] | None:
    if not llm_enabled(config, "use_llm_decomposition", False):
        return None
    try:
        expert = _expert_for_decomposition(env_name=env_name, info=info, config=config)
        max_subtasks = int(cfg_get(config, "llm_max_subtasks", 6))
        data = chat_json(
            _decomposition_messages(env_name=env_name, task_text=task_text, max_subtasks=max_subtasks, expert=expert),
            config=config,
            purpose="decompose",
            cache_payload={
                "env_name": env_name,
                "task_text": task_text,
                "max_subtasks": max_subtasks,
                "expert_signature": expert_signature(expert),
            },
        )
        return obligations_from_subtask_data(
            data,
            env_name=env_name,
            traj_uid=traj_uid,
            task_text=task_text,
            source="llm_expert" if expert else "llm",
        )
    except Exception as exc:
        if not fallback_to_rules(config):
            raise
        warn_once("llm_decomposition_failed", f"Warning: ObliFlow LLM decomposition failed once; falling back to rule obligations. Error: {exc}")
    return None


def obligations_from_subtask_data(
    data: dict[str, Any],
    *,
    env_name: str,
    traj_uid: str,
    task_text: str,
    source: str,
    extra_metadata: dict[str, Any] | None = None,
) -> list[ObligationNode] | None:
    obligations = _parse_obligations(data, env_name=env_name, traj_uid=traj_uid, task_text=task_text)
    if not obligations:
        return None
    for obligation in obligations:
        obligation.metadata["subtask_source"] = source
        if extra_metadata:
            obligation.metadata.update(extra_metadata)
    obligations.append(_final_success_obligation(traj_uid))
    return obligations


def _decomposition_messages(
    env_name: str,
    task_text: str,
    max_subtasks: int,
    expert: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system = (
        "You decompose embodied or shopping agent tasks into semantic subtasks for RL process rewards. "
        "Return JSON only. Subtasks must be concrete, ordered, atomic, and checkable from the model's actual action plus environment observation history. "
        "If an expert trajectory is provided, treat it as a reference solution, not the only valid solution. "
        "Do not require the agent to follow the expert path, exact action order, exact navigation route, or optional exploration actions. "
        "Extract only invariant semantic milestones that any successful solution should satisfy. "
        "Do not include a final overall success subtask; the environment success signal is checked separately."
    )
    user = {
        "env_name": env_name,
        "task": task_text,
        "requirements": [
            f"Return 2 to {max_subtasks} subtasks.",
            "Use short stable ids such as identify_goal, find_object, clean_object, place_object.",
            "Each subtask needs description, success_criteria, and weight.",
            "For ALFWorld, include required object interaction/state-change/place subtasks.",
            "For WebShop, include query/search, inspect/select, option matching, and buy decision subtasks when applicable.",
        ],
        "schema": {
            "subtasks": [
                {
                    "id": "short_snake_case_id",
                    "description": "what the agent must accomplish",
                    "success_criteria": "observable evidence that means this subtask is complete",
                    "weight": 1.0,
                }
            ]
        },
    }
    if expert:
        user["expert_reference"] = compact_expert_for_prompt(expert)
        user["requirements"].extend(
            [
                "Use expert_reference only to infer required object, state-change, and receptacle milestones.",
                "Do not encode optional expert-specific route choices, look-around actions, or container-opening steps unless they are necessary semantic milestones.",
                "Prefer milestones such as find target object, pick target object, apply required clean/heat/cool state, find target receptacle, and place target object.",
            ]
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _expert_for_decomposition(env_name: str, info: dict[str, Any] | None, config: Any = None) -> dict[str, Any] | None:
    if "alfworld" not in env_name.lower():
        return None
    if not bool(cfg_get(config, "llm_use_expert_trajectory", True)):
        return None
    try:
        return expert_from_info(info)
    except Exception as exc:
        if not fallback_to_rules(config):
            raise
        warn_once("alfworld_expert_load_failed", f"Warning: failed to load ALFWorld expert trajectory once; using task-only LLM decomposition. Error: {exc}")
        return None


def _parse_obligations(data: dict[str, Any], env_name: str, traj_uid: str, task_text: str) -> list[ObligationNode]:
    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list):
        return []

    obligations: list[ObligationNode] = []
    seen: set[str] = set()
    for idx, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            continue
        description = str(subtask.get("description", "")).strip()
        success_criteria = str(subtask.get("success_criteria", "")).strip()
        if not description or not success_criteria:
            continue
        source_id = _slug(str(subtask.get("id") or description)) or f"subtask_{idx + 1}"
        if source_id in seen:
            source_id = f"{source_id}_{idx + 1}"
        seen.add(source_id)
        weight = _float_in_range(subtask.get("weight", 1.0), default=1.0, low=0.1, high=3.0)
        obligations.append(
            ObligationNode(
                id=f"{traj_uid}:o_llm_{idx + 1}_{source_id}",
                traj_uid=traj_uid,
                type="llm_subtask",
                target={
                    "task": task_text,
                    "env_name": env_name,
                    "subtask_id": source_id,
                    "description": description,
                    "success_criteria": success_criteria,
                },
                weight=weight,
                verifier="llm_subtask_completion",
                metadata={"llm_generated": True, "source_id": source_id, "order": idx + 1},
            )
        )
    return obligations


def _final_success_obligation(traj_uid: str) -> ObligationNode:
    return ObligationNode(
        id=f"{traj_uid}:o_final_success",
        traj_uid=traj_uid,
        type="final_success",
        target=True,
        weight=3.0,
        verifier="env_success",
        metadata={"llm_generated": False},
    )


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value[:48]


def _float_in_range(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))
