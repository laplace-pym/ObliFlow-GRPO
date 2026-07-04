import json
from dataclasses import dataclass
from typing import Any

from obli_flow.llm_client import LLMUnavailable, chat_json, cfg_get, fallback_to_rules, llm_enabled, warn_once
from obli_flow.schema import ObligationNode, StepRecord
from obli_flow.validators import token_overlap


@dataclass
class SubtaskCheck:
    completed: bool
    step_id: int | None
    score: float
    reason: str
    evidence: str = ""


def verify_llm_subtasks_for_trajectory(records: list[StepRecord], obligations: list[ObligationNode], config: Any = None) -> dict[str, SubtaskCheck]:
    if not records or not obligations:
        return {}
    if not llm_enabled(config, "use_llm_verifier", False):
        return _heuristic_checks(records, obligations)

    try:
        data = chat_json(
            _verification_messages(records=records, obligations=obligations),
            config=config,
            purpose="verify",
            cache_payload={
                "task_text": records[0].task_text,
                "env_name": records[0].env_name,
                "subtasks": [_subtask_payload(obligation) for obligation in obligations],
                "trajectory": [_record_payload(record) for record in records],
            },
        )
        checks = _parse_checks(data, obligations)
        if checks:
            return checks
    except Exception as exc:
        if not fallback_to_rules(config):
            raise
        warn_once("llm_verifier_failed", f"Warning: ObliFlow LLM verifier failed once; falling back to heuristic subtask checks. Error: {exc}")
    return _heuristic_checks(records, obligations)


def _verification_messages(records: list[StepRecord], obligations: list[ObligationNode]) -> list[dict[str, str]]:
    system = (
        "You judge whether decomposed agent subtasks were completed in a trajectory. "
        "Return JSON only. Mark a subtask completed only when there is observable evidence in the action or the following environment observation. "
        "If a step merely attempts or plans the subtask without evidence, completed must be false."
    )
    user = {
        "env_name": records[0].env_name,
        "task": records[0].task_text,
        "subtasks": [_subtask_payload(obligation) for obligation in obligations],
        "trajectory": [_record_payload(record) for record in records],
        "schema": {
            "subtasks": [
                {
                    "id": "same id from input",
                    "completed": True,
                    "step_id": 0,
                    "score": 1.0,
                    "reason": "brief reason",
                    "evidence": "short quote or paraphrase from action/observation",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _subtask_payload(obligation: ObligationNode) -> dict[str, Any]:
    target = obligation.target if isinstance(obligation.target, dict) else {"description": obligation.target}
    return {
        "id": obligation.metadata.get("source_id", obligation.id),
        "description": target.get("description", ""),
        "success_criteria": target.get("success_criteria", ""),
    }


def _record_payload(record: StepRecord) -> dict[str, Any]:
    return {
        "step_id": record.step_id,
        "observation_before": _clip(record.anchor_obs),
        "action": _clip(record.parsed_action.action),
        "reasoning": _clip(record.parsed_action.think, limit=500),
        "observation_after": _clip(record.next_obs),
        "reward": record.reward,
        "done": record.done,
        "won": bool(record.info.get("won", False)),
        "valid_action": bool(record.is_action_valid and record.parsed_action.valid_format),
    }


def _parse_checks(data: dict[str, Any], obligations: list[ObligationNode]) -> dict[str, SubtaskCheck]:
    raw_items = data.get("subtasks", [])
    if not isinstance(raw_items, list):
        return {}

    by_source = {obligation.metadata.get("source_id", obligation.id): obligation for obligation in obligations}
    checks: dict[str, SubtaskCheck] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id", ""))
        obligation = by_source.get(source_id)
        if obligation is None:
            continue
        completed = bool(item.get("completed", False))
        step_id = item.get("step_id", None)
        try:
            step_id = int(step_id) if step_id is not None else None
        except Exception:
            step_id = None
        if not completed:
            step_id = None
        checks[obligation.id] = SubtaskCheck(
            completed=completed,
            step_id=step_id,
            score=_score(item.get("score", 1.0 if completed else 0.0), completed=completed),
            reason=str(item.get("reason", "llm verifier")),
            evidence=str(item.get("evidence", "")),
        )
    return checks


def _heuristic_checks(records: list[StepRecord], obligations: list[ObligationNode]) -> dict[str, SubtaskCheck]:
    checks: dict[str, SubtaskCheck] = {}
    for obligation in obligations:
        best_step = None
        best_score = 0.0
        target = obligation.target
        for record in records:
            score = max(
                token_overlap(record.parsed_action.action, target),
                token_overlap(record.next_obs, target),
                token_overlap(record.anchor_obs, target),
            )
            if score > best_score:
                best_score = score
                best_step = record.step_id
        completed = best_score > 0.0
        checks[obligation.id] = SubtaskCheck(
            completed=completed,
            step_id=best_step if completed else None,
            score=max(0.1, best_score) if completed else 0.0,
            reason="heuristic keyword overlap fallback" if completed else "no heuristic evidence",
        )
    return checks


def _score(value: Any, *, completed: bool) -> float:
    try:
        score = float(value)
    except Exception:
        score = 1.0 if completed else 0.0
    score = max(0.0, min(1.0, score))
    return score if completed else 0.0


def _clip(value: Any, limit: int = 1200) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
