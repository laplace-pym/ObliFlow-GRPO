from obli_flow.schema import ArtifactNode, StepRecord
from obli_flow.validators import keyword_set, task_score

from .base import BaseExtractor


class WebShopExtractor(BaseExtractor):
    def extract(self, record: StepRecord, action_id: str) -> list[ArtifactNode]:
        base = f"{record.traj_uid}:s{record.step_id}"
        artifacts = [
            ArtifactNode(
                id=f"{base}:page",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="page_observation",
                value=record.anchor_obs,
                producer_action=None,
                source="environment",
            ),
            ArtifactNode(
                id=f"{base}:page_terms",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="page_terms",
                value=sorted(keyword_set(record.anchor_obs)),
                producer_action=None,
                source="environment",
            ),
            ArtifactNode(
                id=f"{base}:action",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="web_action",
                value=record.parsed_action.action,
                producer_action=action_id,
                source="policy",
                confidence=1.0 if record.is_action_valid and record.parsed_action.valid_format else 0.0,
            ),
            ArtifactNode(
                id=f"{base}:next_page",
                traj_uid=record.traj_uid,
                step_id=record.step_id,
                type="page_transition",
                value=record.next_obs,
                producer_action=action_id,
                source="environment_transition",
            ),
        ]
        available_actions = record.info.get("available_actions", {})
        if available_actions:
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:clickables",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="available_actions",
                    value=available_actions,
                    producer_action=None,
                    source="environment",
                )
            )

        if record.parsed_action.action_type == "search":
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:query",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="search_query",
                    value=record.parsed_action.action_args.get("query", ""),
                    producer_action=action_id,
                    source="policy",
                )
            )
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:search_results",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="search_results",
                    value=record.next_obs,
                    producer_action=action_id,
                    source="environment_transition",
                )
            )
        elif record.parsed_action.action_type == "buy":
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:buy",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="buy_action",
                    value=True,
                    producer_action=action_id,
                    source="policy",
                )
            )
        elif record.parsed_action.action_type == "click":
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:click",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="click_target",
                    value=record.parsed_action.action_args.get("target", ""),
                    producer_action=action_id,
                    source="policy",
                )
            )
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:product_detail",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="product_detail",
                    value=record.next_obs,
                    producer_action=action_id,
                    source="environment_transition",
                )
            )

        score = task_score(record.info)
        if score > 0.0:
            artifacts.append(
                ArtifactNode(
                    id=f"{base}:task_score",
                    traj_uid=record.traj_uid,
                    step_id=record.step_id,
                    type="task_score",
                    value=score,
                    producer_action=action_id,
                    source="environment_verifier",
                    confidence=max(0.0, min(score, 1.0)),
                )
            )
        if record.info.get("won", False):
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
