from obli_flow.schema import ArtifactNode, StepRecord
from obli_flow.validators import keyword_set, normalize_text, is_noop_feedback

from .base import BaseExtractor


class AlfWorldExtractor(BaseExtractor):
    def extract(self, record: StepRecord, action_id: str) -> list[ArtifactNode]:
        base = f"{record.traj_uid}:s{record.step_id}"
        artifacts = [
            ArtifactNode(
                id=f"{base}:obs",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="observation_text",
                value=record.anchor_obs,
                producer_action=None,
                source="environment",
            ),
            ArtifactNode(
                id=f"{base}:visible_objects",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="visible_objects",
                value=sorted(keyword_set(record.anchor_obs)),
                producer_action=None,
                source="environment",
            ),
            ArtifactNode(
                id=f"{base}:action",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="env_action",
                value=record.parsed_action.action,
                producer_action=action_id,
                source="policy",
                confidence=1.0 if record.is_action_valid and record.parsed_action.valid_format else 0.0,
            ),
            ArtifactNode(
                id=f"{base}:next_obs",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="action_effect",
                value=record.next_obs,
                producer_action=action_id,
                source="environment_transition",
                confidence=0.2 if is_noop_feedback(record.next_obs) else 1.0,
            ),
        ]

        admissible = record.info.get("admissible_commands")
        if admissible is not None:
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:admissible",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="admissible_actions",
                    value=admissible,
                    producer_action=None,
                    source="environment",
                )
            )

        action_l = normalize_text(record.parsed_action.action)
        next_l = normalize_text(record.next_obs)
        for state in ("clean", "heat", "cool", "slice"):
            if state in action_l or state in next_l:
                artifacts.append(
                    ArtifactNode(
                        id=f"{base}:state_{state}",
                        traj_uid=record.traj_uid,
                        step_id=record.step_id,
                        type="object_state",
                        value=state,
                        producer_action=action_id,
                        source="environment_transition",
                        confidence=1.0 if state in next_l or not is_noop_feedback(record.next_obs) else 0.3,
                    )
                )

        if record.info.get("won", False) or normalize_text(record.next_obs) == "success":
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:success",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="success_state",
                    value=True,
                    producer_action=action_id,
                    source="environment_verifier",
                )
            )
        return artifacts
