from dataclasses import dataclass, field
from typing import Any

from .schema import FlowGraph, StepCredit
from .validators import is_noop_feedback


@dataclass
class FlowRewardResult:
    credits: list[StepCredit]
    metrics: dict[str, float] = field(default_factory=dict)
    graph_debug: dict[str, Any] = field(default_factory=dict)


def compute_step_credits(
    graph: FlowGraph,
    uid: str,
    *,
    lambda_cost: float = 0.02,
    eta_waste: float = 0.2,
    rho_break: float = 0.5,
    beta_cut: float = 0.3,
    use_min_cut: bool = True,
    use_waste_penalty: bool = True,
    use_terminal_reward: bool = True,
) -> FlowRewardResult:
    del beta_cut, use_terminal_reward
    step_ids = sorted({action.step_id for action in graph.actions})
    phi_by_step: dict[int, float] = {}
    for step_id in step_ids:
        phi_by_step[step_id] = compute_phi(graph, step_id, lambda_cost=lambda_cost)

    artifact_outflow = _artifact_outflow(graph)
    total_artifacts = [artifact for artifact in graph.artifacts if artifact.step_id >= 0]
    used_artifacts = [artifact for artifact in total_artifacts if artifact_outflow.get(artifact.id, 0) > 0]
    failed_obligations = _failed_obligations(graph)
    cut_by_step = _heuristic_cut_blame(graph, failed_obligations) if use_min_cut else {}

    credits: list[StepCredit] = []
    prev_phi = 0.0
    total_waste = 0.0
    total_break = 0.0
    for step_id in step_ids:
        action = graph.action_by_step(step_id)
        phi_curr = phi_by_step[step_id]
        delta_phi = phi_curr - prev_phi
        produced = [artifact for artifact in graph.artifacts if artifact.step_id == step_id and artifact.producer_action == (action.id if action else None)]
        waste_penalty = 0.0
        if use_waste_penalty and produced:
            wasted = [artifact for artifact in produced if artifact_outflow.get(artifact.id, 0) == 0 and artifact.type not in {"success_state", "task_score"}]
            waste_penalty = len(wasted) / max(1, len(produced))

        break_penalty = _break_penalty_for_step(graph, step_id)
        cut_blame = cut_by_step.get(step_id, 0.0)
        flow_reward = delta_phi - eta_waste * waste_penalty - rho_break * break_penalty + cut_blame
        total_waste += waste_penalty
        total_break += break_penalty
        credits.append(
            StepCredit(
                traj_uid=graph.traj_uid,
                uid=uid,
                step_id=step_id,
                phi_prev=prev_phi,
                phi_curr=phi_curr,
                delta_phi=delta_phi,
                waste_penalty=waste_penalty,
                break_penalty=break_penalty,
                cut_blame=cut_blame,
                flow_reward=flow_reward,
                obligation_coverage=_obligation_coverage(graph, step_id),
                artifact_utilization=len(used_artifacts) / max(1, len(total_artifacts)),
            )
        )
        prev_phi = phi_curr

    metrics = {
        "obliflow/phi_final": prev_phi,
        "obliflow/artifact_count": float(len(total_artifacts)),
        "obliflow/edge_count": float(len(graph.edges)),
        "obliflow/obligation_count": float(len(graph.obligations)),
        "obliflow/artifact_utilization_rate": len(used_artifacts) / max(1, len(total_artifacts)),
        "obliflow/obligation_coverage": _obligation_coverage(graph, max(step_ids) if step_ids else -1),
        "obliflow/waste_tool_ratio": _waste_tool_ratio(graph, artifact_outflow),
        "obliflow/waste_penalty_mean": total_waste / max(1, len(step_ids)),
        "obliflow/break_penalty_mean": total_break / max(1, len(step_ids)),
        "obliflow/broken_handoff_rate": _broken_handoff_rate(graph),
        "obliflow/discharge_edge_valid_rate": _discharge_edge_valid_rate(graph),
    }
    return FlowRewardResult(credits=credits, metrics=metrics, graph_debug=_graph_summary(graph))


def compute_phi(graph: FlowGraph, step_id: int, *, lambda_cost: float = 0.02) -> float:
    total = 0.0
    for obligation in graph.obligations:
        best = 0.0
        for edge in graph.edges:
            if edge.edge_type != "discharge" or edge.dst != obligation.id or edge.step_id > step_id:
                continue
            if edge.valid:
                best = max(best, edge.score)
        total += obligation.weight * best
    valid_cost = sum(edge.cost for edge in graph.edges if edge.step_id <= step_id and edge.valid)
    return total - lambda_cost * valid_cost


def _artifact_outflow(graph: FlowGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in graph.edges:
        if edge.edge_type in {"consumption", "transformation", "discharge"} and edge.valid:
            counts[edge.src] = counts.get(edge.src, 0) + 1
    return counts


def _break_penalty_for_step(graph: FlowGraph, step_id: int) -> float:
    action = graph.action_by_step(step_id)
    penalty = 0.0
    if action is not None and not action.valid:
        penalty += 1.0
    step_edges = graph.edges_by_step(step_id)
    invalid_consumption = [edge for edge in step_edges if edge.edge_type == "consumption" and not edge.valid]
    if invalid_consumption:
        penalty += min(1.0, len(invalid_consumption) / max(1, len(step_edges)))
    produced_effects = [
        artifact
        for artifact in graph.artifacts
        if artifact.step_id == step_id and artifact.type in {"action_effect", "page_transition"} and artifact.producer_action == (action.id if action else None)
    ]
    if any(is_noop_feedback(artifact.value) for artifact in produced_effects):
        penalty += 0.5
    return min(2.0, penalty)


def _failed_obligations(graph: FlowGraph) -> list[str]:
    failed = []
    for obligation in graph.obligations:
        valid_edges = [edge for edge in graph.edges if edge.edge_type == "discharge" and edge.dst == obligation.id and edge.valid and edge.score > 0.0]
        if not valid_edges:
            failed.append(obligation.id)
    return failed


def _heuristic_cut_blame(graph: FlowGraph, failed_obligations: list[str]) -> dict[int, float]:
    if not failed_obligations:
        return {}
    scored_steps: list[tuple[float, int]] = []
    for action in graph.actions:
        break_score = _break_penalty_for_step(graph, action.step_id)
        if break_score > 0.0:
            scored_steps.append((break_score, action.step_id))
    if not scored_steps and graph.actions:
        scored_steps.append((0.5, graph.actions[-1].step_id))
    if not scored_steps:
        return {}
    scored_steps.sort(reverse=True)
    blame = -min(1.0, 0.25 * len(failed_obligations) + 0.25 * scored_steps[0][0])
    return {scored_steps[0][1]: blame}


def _obligation_coverage(graph: FlowGraph, step_id: int) -> float:
    if not graph.obligations:
        return 0.0
    covered = 0
    for obligation in graph.obligations:
        if any(edge.edge_type == "discharge" and edge.dst == obligation.id and edge.step_id <= step_id and edge.valid and edge.score > 0.0 for edge in graph.edges):
            covered += 1
    return covered / len(graph.obligations)


def _broken_handoff_rate(graph: FlowGraph) -> float:
    consumption = [edge for edge in graph.edges if edge.edge_type == "consumption"]
    broken = [edge for edge in consumption if not edge.valid]
    return len(broken) / max(1, len(consumption))


def _waste_tool_ratio(graph: FlowGraph, artifact_outflow: dict[str, int]) -> float:
    action_count = 0
    waste_count = 0
    terminal_types = {"success_state", "task_score"}
    for action in graph.actions:
        produced = [
            artifact
            for artifact in graph.artifacts
            if artifact.producer_action == action.id and artifact.type not in terminal_types
        ]
        if not produced:
            continue
        action_count += 1
        if all(artifact_outflow.get(artifact.id, 0) == 0 for artifact in produced):
            waste_count += 1
    return waste_count / max(1, action_count)


def _discharge_edge_valid_rate(graph: FlowGraph) -> float:
    discharge_edges = [edge for edge in graph.edges if edge.edge_type == "discharge"]
    valid_edges = [edge for edge in discharge_edges if edge.valid]
    return len(valid_edges) / max(1, len(discharge_edges))


def _graph_summary(graph: FlowGraph) -> dict[str, Any]:
    return {
        "traj_uid": graph.traj_uid,
        "env_name": graph.env_name,
        "actions": len(graph.actions),
        "artifacts": len(graph.artifacts),
        "obligations": len(graph.obligations),
        "edges": len(graph.edges),
    }
