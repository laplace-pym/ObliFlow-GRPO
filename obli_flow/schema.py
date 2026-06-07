from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParsedAction:
    think: str
    action: str
    valid_format: bool
    action_type: str
    action_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    uid: str
    traj_uid: str
    step_id: int
    env_name: str
    task_text: str
    anchor_obs: Any
    raw_response: str
    parsed_action: ParsedAction
    next_obs: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    active_mask: bool = True
    is_action_valid: bool = True
    episode_reward: float = 0.0


@dataclass
class ActionNode:
    id: str
    traj_uid: str
    step_id: int
    raw_response: str
    think: str
    action: str
    action_type: str
    valid: bool
    env_name: str


@dataclass
class ArtifactNode:
    id: str
    traj_uid: str
    step_id: int
    type: str
    value: Any
    producer_action: Optional[str]
    source: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObligationNode:
    id: str
    traj_uid: str
    type: str
    target: Any
    weight: float
    verifier: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowEdge:
    id: str
    traj_uid: str
    src: str
    dst: str
    edge_type: str
    step_id: int
    valid: bool
    score: float
    cost: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepCredit:
    traj_uid: str
    uid: str
    step_id: int
    phi_prev: float
    phi_curr: float
    delta_phi: float
    waste_penalty: float
    break_penalty: float
    cut_blame: float
    flow_reward: float
    obligation_coverage: float
    artifact_utilization: float


@dataclass
class FlowGraph:
    traj_uid: str
    env_name: str
    obligations: list[ObligationNode] = field(default_factory=list)
    actions: list[ActionNode] = field(default_factory=list)
    artifacts: list[ArtifactNode] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)

    def add_action(self, action: ActionNode) -> None:
        self.actions.append(action)

    def add_artifact(self, artifact: ArtifactNode) -> None:
        self.artifacts.append(artifact)

    def add_edge(self, edge: FlowEdge) -> None:
        self.edges.append(edge)

    def artifacts_by_step(self, step_id: int) -> list[ArtifactNode]:
        return [artifact for artifact in self.artifacts if artifact.step_id == step_id]

    def edges_by_step(self, step_id: int) -> list[FlowEdge]:
        return [edge for edge in self.edges if edge.step_id == step_id]

    def action_by_step(self, step_id: int) -> Optional[ActionNode]:
        for action in self.actions:
            if action.step_id == step_id:
                return action
        return None
