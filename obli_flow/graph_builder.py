from .extractors import get_extractor
from .llm_verifier import verify_llm_subtasks_for_trajectory
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


def build_flow_graph(records: list[StepRecord], env_name: str, config=None) -> FlowGraph:
    if not records:
        raise ValueError("Cannot build an ObliFlow graph from an empty trajectory.")
    first = records[0]
    last = records[-1]
    task_text = first.task_text
    obligations = build_obligations(env_name=env_name, traj_uid=first.traj_uid, task_text=task_text, info=last.info, config=config)
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

    prior_handoff_artifacts: list[ArtifactNode] = []
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

        consumption_edges = _add_consumption_edges(graph, record, action_id, task_artifact, step_artifacts, prior_handoff_artifacts)
        _add_transformation_edges(graph, record, action_id, step_artifacts, consumption_edges)
        _add_discharge_edges(graph, record, step_artifacts, include_llm=False)
        prior_handoff_artifacts.extend(_handoff_artifacts(step_artifacts, action_id))

    _add_llm_subtask_discharge_edges(graph, records, config=config)
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
    prior_handoff_artifacts: list[ArtifactNode],
) -> list[FlowEdge]:
    added_edges: list[FlowEdge] = []
    goal_valid, goal_score, goal_reason = validate_goal_overlap(record.parsed_action.action, record.task_text)
    task_edge = FlowEdge(
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
    graph.add_edge(task_edge)
    added_edges.append(task_edge)

    obs_artifacts = [artifact for artifact in step_artifacts if artifact.type in {"observation_text", "page_observation"}]
    for obs_artifact in obs_artifacts:
        valid = bool(record.is_action_valid and record.parsed_action.valid_format)
        score = 1.0 if valid else 0.0
        obs_edge = FlowEdge(
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
        graph.add_edge(obs_edge)
        added_edges.append(obs_edge)

    available = record.info.get("admissible_commands") or record.info.get("available_actions")
    if available is not None:
        valid, score, reason = validate_admissible_action(record.parsed_action.action, available)
        for artifact in step_artifacts:
            if artifact.type in {"admissible_actions", "available_actions"}:
                available_edge = FlowEdge(
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
                graph.add_edge(available_edge)
                added_edges.append(available_edge)
    for artifact in prior_handoff_artifacts:
        valid, score, reason = _validate_prior_handoff(record, artifact)
        handoff_edge = FlowEdge(
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
        graph.add_edge(handoff_edge)
        added_edges.append(handoff_edge)
    return added_edges


def _handoff_artifacts(step_artifacts: list[ArtifactNode], action_id: str) -> list[ArtifactNode]:
    handoff_types = {
        "action_effect",
        "page_transition",
        "search_results",
        "product_detail",
        "object_state",
        "task_score",
        "success_state",
    }
    return [
        artifact
        for artifact in step_artifacts
        if artifact.producer_action == action_id and artifact.type in handoff_types
    ]


def _validate_prior_handoff(record: StepRecord, artifact: ArtifactNode) -> tuple[bool, float, str]:
    if not record.is_action_valid or not record.parsed_action.valid_format:
        return False, 0.0, "invalid action cannot consume prior artifact"
    overlap = max(
        token_overlap(artifact.value, record.anchor_obs),
        token_overlap(artifact.value, record.parsed_action.action),
        token_overlap(artifact.value, record.parsed_action.think),
    )
    adjacent_bonus = 0.5 if artifact.step_id == record.step_id - 1 else 0.0
    score = max(overlap, adjacent_bonus)
    return score > 0.0, score, "prior artifact consumed by later action" if score > 0.0 else "prior artifact not consumed"


def _add_transformation_edges(
    graph: FlowGraph,
    record: StepRecord,
    action_id: str,
    step_artifacts: list[ArtifactNode],
    consumption_edges: list[FlowEdge],
) -> None:
    produced_artifacts = [
        artifact
        for artifact in step_artifacts
        if artifact.producer_action == action_id and artifact.type not in {"env_action", "web_action"}
    ]
    for consume_edge in consumption_edges:
        for artifact in produced_artifacts:
            valid = bool(consume_edge.valid and artifact.confidence > 0.0 and record.is_action_valid)
            graph.add_edge(
                FlowEdge(
                    id=f"{consume_edge.src}->{action_id}->{artifact.id}",
                    traj_uid=record.traj_uid,
                    src=consume_edge.src,
                    dst=artifact.id,
                    edge_type="transformation",
                    step_id=record.step_id,
                    valid=valid,
                    score=min(consume_edge.score, artifact.confidence) if valid else 0.0,
                    cost=0.0,
                    reason="artifact transformed through action" if valid else "invalid artifact transformation",
                    metadata={"via_action": action_id},
                )
            )


def _add_discharge_edges(graph: FlowGraph, record: StepRecord, step_artifacts: list[ArtifactNode], include_llm: bool = True) -> None:
    for obligation in graph.obligations:
        if obligation.type == "llm_subtask" and not include_llm:
            continue
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


def _add_llm_subtask_discharge_edges(graph: FlowGraph, records: list[StepRecord], config=None) -> None:
    obligations = [obligation for obligation in graph.obligations if obligation.type == "llm_subtask"]
    if not obligations:
        return

    checks = verify_llm_subtasks_for_trajectory(records, obligations, config=config)
    if not checks:
        return

    final_step = records[-1].step_id if records else 0
    for obligation in obligations:
        check = checks.get(obligation.id)
        if check is None:
            continue
        step_id = check.step_id if check.completed and check.step_id is not None else final_step
        step_ids = {record.step_id for record in records}
        if step_id not in step_ids:
            step_id = final_step
        source_artifact = _source_artifact_for_llm_check(graph, step_id)
        if source_artifact is None:
            continue
        graph.add_edge(
            FlowEdge(
                id=f"{source_artifact.id}->{obligation.id}:llm_check",
                traj_uid=graph.traj_uid,
                src=source_artifact.id,
                dst=obligation.id,
                edge_type="discharge",
                step_id=step_id,
                valid=bool(check.completed),
                score=check.score,
                cost=0.0,
                reason=check.reason,
                metadata={"llm_checked": True, "evidence": check.evidence},
            )
        )


def _source_artifact_for_llm_check(graph: FlowGraph, step_id: int) -> ArtifactNode | None:
    preferred_types = (
        "success_state",
        "task_score",
        "object_state",
        "action_effect",
        "page_transition",
        "product_detail",
        "search_results",
        "env_action",
        "web_action",
    )
    artifacts = graph.artifacts_by_step(step_id)
    for artifact_type in preferred_types:
        for artifact in artifacts:
            if artifact.type == artifact_type:
                return artifact
    return artifacts[0] if artifacts else None


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
