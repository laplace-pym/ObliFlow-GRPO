from .extractors import get_extractor
from .obligations import build_obligations
from .schema import ActionNode, ArtifactNode, FlowEdge, FlowGraph, ObligationNode, StepRecord
from .validators import (
    exact_contains,
    info_won,
    is_noop_feedback,
    task_score,
    token_overlap,
    validate_admissible_action,
    validate_goal_overlap,
)


def build_flow_graph(records: list[StepRecord], env_name: str) -> FlowGraph:
    if not records:
        raise ValueError("Cannot build an ObliFlow graph from an empty trajectory.")
    first = records[0]
    last = records[-1]
    task_text = first.task_text
    obligations = build_obligations(env_name=env_name, traj_uid=first.traj_uid, task_text=task_text, info=last.info)
    graph = FlowGraph(traj_uid=first.traj_uid, env_name=env_name, obligations=obligations)
    extractor = get_extractor(env_name)

    task_artifact = ArtifactNode(
        id=f"{first.traj_uid}:task",
        traj_uid=first.traj_uid,
        step_id=-1,
        type="task_intent" if "webshop" not in env_name.lower() else "shopping_goal",
        value=task_text,
        producer_action=None,
        source="task",
    )
    graph.add_artifact(task_artifact)

    for record in records:
        action_id = f"{record.traj_uid}:a{record.step_id}"
        action = ActionNode(
            id=action_id,
            traj_uid=record.traj_uid,
            step_id=record.step_id,
            raw_response=record.raw_response,
            think=record.parsed_action.think,
            action=record.parsed_action.action,
            action_type=record.parsed_action.action_type,
            valid=bool(record.is_action_valid and record.parsed_action.valid_format),
            env_name=env_name,
        )
        graph.add_action(action)

        step_artifacts = extractor.extract(record, action_id=action_id)
        for artifact in step_artifacts:
            graph.add_artifact(artifact)
            if artifact.producer_action == action_id:
                graph.add_edge(
                    FlowEdge(
                        id=f"{action_id}->{artifact.id}",
                        traj_uid=record.traj_uid,
                        src=action_id,
                        dst=artifact.id,
                        edge_type="production",
                        step_id=record.step_id,
                        valid=artifact.confidence > 0.0,
                        score=artifact.confidence,
                        cost=_edge_cost(record, artifact),
                        reason="action produced artifact",
                    )
                )

        _add_consumption_edges(graph, record, action_id, task_artifact, step_artifacts)
        _add_discharge_edges(graph, record, step_artifacts)

    return graph


def _edge_cost(record: StepRecord, artifact: ArtifactNode) -> float:
    if not record.is_action_valid:
        return 2.0
    if artifact.type in {"success_state", "task_score"}:
        return 0.0
    if artifact.type in {"search_query", "search_results"}:
        return 1.0
    return 0.5


def _add_consumption_edges(
    graph: FlowGraph,
    record: StepRecord,
    action_id: str,
    task_artifact: ArtifactNode,
    step_artifacts: list[ArtifactNode],
) -> None:
    goal_valid, goal_score, goal_reason = validate_goal_overlap(record.parsed_action.action, record.task_text)
    graph.add_edge(
        FlowEdge(
            id=f"{task_artifact.id}->{action_id}",
            traj_uid=record.traj_uid,
            src=task_artifact.id,
            dst=action_id,
            edge_type="consumption",
            step_id=record.step_id,
            valid=goal_valid or record.step_id == 0,
            score=max(goal_score, 0.1 if record.step_id == 0 else 0.0),
            cost=0.0,
            reason=goal_reason,
        )
    )

    obs_artifacts = [artifact for artifact in step_artifacts if artifact.type in {"observation_text", "page_observation"}]
    for obs_artifact in obs_artifacts:
        valid = bool(record.is_action_valid and record.parsed_action.valid_format)
        score = 1.0 if valid else 0.0
        graph.add_edge(
            FlowEdge(
                id=f"{obs_artifact.id}->{action_id}",
                traj_uid=record.traj_uid,
                src=obs_artifact.id,
                dst=action_id,
                edge_type="consumption",
                step_id=record.step_id,
                valid=valid,
                score=score,
                cost=0.0,
                reason="current observation consumed by action" if valid else "invalid action cannot consume observation",
            )
        )

    available = record.info.get("admissible_commands") or record.info.get("available_actions")
    if available is not None:
        valid, score, reason = validate_admissible_action(record.parsed_action.action, available)
        for artifact in step_artifacts:
            if artifact.type in {"admissible_actions", "available_actions"}:
                graph.add_edge(
                    FlowEdge(
                        id=f"{artifact.id}->{action_id}",
                        traj_uid=record.traj_uid,
                        src=artifact.id,
                        dst=action_id,
                        edge_type="consumption",
                        step_id=record.step_id,
                        valid=valid,
                        score=score,
                        cost=0.0,
                        reason=reason,
                    )
                )


def _add_discharge_edges(graph: FlowGraph, record: StepRecord, step_artifacts: list[ArtifactNode]) -> None:
    for obligation in graph.obligations:
        for artifact in step_artifacts:
            score, valid, reason = _verify_discharge(record, artifact, obligation)
            if score <= 0.0 and not valid:
                continue
            graph.add_edge(
                FlowEdge(
                    id=f"{artifact.id}->{obligation.id}",
                    traj_uid=record.traj_uid,
                    src=artifact.id,
                    dst=obligation.id,
                    edge_type="discharge",
                    step_id=record.step_id,
                    valid=valid,
                    score=score,
                    cost=0.0,
                    reason=reason,
                )
            )


def _verify_discharge(record: StepRecord, artifact: ArtifactNode, obligation: ObligationNode) -> tuple[float, bool, str]:
    if obligation.type == "final_success":
        if artifact.type == "success_state" and (artifact.value is True or info_won(record.info)):
            return 1.0, True, "environment success verifier"
        return 0.0, False, "not a success artifact"

    if obligation.type == "buy_correct_product":
        if artifact.type == "task_score":
            score = max(0.0, min(float(artifact.value), 1.0))
            return score, score > 0.0, "webshop task_score partial verifier"
        if artifact.type == "success_state" and info_won(record.info):
            return 1.0, True, "webshop won verifier"
        return 0.0, False, "not a buy verifier artifact"

    if obligation.type == "state_transform":
        target = str(obligation.target)
        if artifact.type == "object_state" and exact_contains(artifact.value, target):
            return artifact.confidence, artifact.confidence > 0.0, "state transform artifact"
        if artifact.type == "action_effect" and exact_contains(artifact.value, target):
            return 0.5 * artifact.confidence, artifact.confidence > 0.0, "state appears in action effect"
        return 0.0, False, "state not observed"

    if obligation.type in {"task_parse", "goal_parse", "query_matches_goal", "inspect_candidate_product", "object_handoff"}:
        score = token_overlap(artifact.value, obligation.target)
        if artifact.type in {"search_query", "product_detail", "page_transition", "action_effect", "env_action", "web_action"} and score > 0.0:
            return max(0.1, score) * artifact.confidence, True, "artifact overlaps obligation target"
        return 0.0, False, "artifact does not overlap obligation target"

    if artifact.type == "action_effect" and not is_noop_feedback(artifact.value):
        return 0.2, True, "generic action effect"
    return 0.0, False, "no verifier match"
