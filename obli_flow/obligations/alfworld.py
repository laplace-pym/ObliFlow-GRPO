from obli_flow.schema import ObligationNode
from obli_flow.validators import keyword_set


def build_alfworld_obligations(traj_uid: str, task_text: str, info: dict | None = None) -> list[ObligationNode]:
    task_l = (task_text or "").lower()
    keywords = sorted(keyword_set(task_text))
    obligations = [
        ObligationNode(
            id=f"{traj_uid}:o_task_parse",
            traj_uid=traj_uid,
            type="task_parse",
            target={"task": task_text, "keywords": keywords},
            weight=0.5,
            verifier="goal_overlap",
        )
    ]

    if any(word in task_l for word in ("clean", "wash")):
        obligations.append(_state_obligation(traj_uid, "clean"))
    if "heat" in task_l:
        obligations.append(_state_obligation(traj_uid, "heat"))
    if "cool" in task_l:
        obligations.append(_state_obligation(traj_uid, "cool"))
    if "slice" in task_l:
        obligations.append(_state_obligation(traj_uid, "slice"))

    if any(word in task_l for word in ("pick", "put", "place", "in ", "on ")):
        obligations.append(
            ObligationNode(
                id=f"{traj_uid}:o_object_handoff",
                traj_uid=traj_uid,
                type="object_handoff",
                target={"task": task_text, "keywords": keywords},
                weight=1.0,
                verifier="action_effect",
            )
        )

    obligations.append(
        ObligationNode(
            id=f"{traj_uid}:o_final_success",
            traj_uid=traj_uid,
            type="final_success",
            target=True,
            weight=3.0,
            verifier="env_success",
        )
    )
    return obligations


def _state_obligation(traj_uid: str, state: str) -> ObligationNode:
    return ObligationNode(
        id=f"{traj_uid}:o_state_{state}",
        traj_uid=traj_uid,
        type="state_transform",
        target=state,
        weight=1.0,
        verifier="state_transform",
    )
