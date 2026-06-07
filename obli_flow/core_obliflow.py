from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .graph_builder import build_flow_graph
from .metrics import aggregate_metric_dicts
from .reward import compute_step_credits
from .trajectory import build_step_records, group_records_by_traj


@dataclass
class BatchedFlowRewards:
    flow_rewards: torch.Tensor
    cut_blames: torch.Tensor
    extra_info: dict[str, np.ndarray]
    metrics: dict[str, float]


def _cfg_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return config.get(name, default) if hasattr(config, "get") else getattr(config, name, default)


def compute_flow_step_rewards(batch, config=None, env_name: str = "") -> BatchedFlowRewards:
    records = build_step_records(batch, env_name=env_name)
    grouped = group_records_by_traj(records)
    device = batch.batch["input_ids"].device
    size = len(records)
    by_key = {}
    metric_dicts: list[dict[str, float]] = []

    for traj_uid, traj_records in grouped.items():
        graph = build_flow_graph(traj_records, env_name=env_name)
        result = compute_step_credits(
            graph,
            uid=traj_records[0].uid,
            lambda_cost=float(_cfg_get(config, "lambda_cost", 0.02)),
            eta_waste=float(_cfg_get(config, "eta_waste", 0.2)),
            rho_break=float(_cfg_get(config, "rho_break", 0.5)),
            beta_cut=float(_cfg_get(config, "beta", 0.3)),
            use_min_cut=bool(_cfg_get(config, "use_min_cut", True)),
            use_waste_penalty=bool(_cfg_get(config, "use_waste_penalty", True)),
            use_terminal_reward=bool(_cfg_get(config, "use_terminal_reward", True)),
        )
        metric_dicts.append(result.metrics)
        for credit in result.credits:
            by_key[(credit.traj_uid, credit.step_id)] = credit

    flow_values = np.zeros(size, dtype=np.float32)
    cut_values = np.zeros(size, dtype=np.float32)
    delta_phi = np.zeros(size, dtype=np.float32)
    waste = np.zeros(size, dtype=np.float32)
    breaks = np.zeros(size, dtype=np.float32)
    coverage = np.zeros(size, dtype=np.float32)
    utilization = np.zeros(size, dtype=np.float32)
    for i, record in enumerate(records):
        credit = by_key.get((record.traj_uid, record.step_id))
        if credit is None:
            continue
        flow_values[i] = credit.flow_reward
        cut_values[i] = credit.cut_blame
        delta_phi[i] = credit.delta_phi
        waste[i] = credit.waste_penalty
        breaks[i] = credit.break_penalty
        coverage[i] = credit.obligation_coverage
        utilization[i] = credit.artifact_utilization

    metrics = aggregate_metric_dicts(metric_dicts)
    metrics.update(
        {
            "obliflow/flow_reward_mean": float(flow_values.mean()) if size else 0.0,
            "obliflow/flow_reward_std": float(flow_values.std()) if size else 0.0,
            "obliflow/cut_blame_mean": float(cut_values.mean()) if size else 0.0,
            "obliflow/delta_phi_mean": float(delta_phi.mean()) if size else 0.0,
        }
    )
    extra_info = {
        "obliflow_flow_reward": flow_values.astype(object),
        "obliflow_cut_blame": cut_values.astype(object),
        "obliflow_delta_phi": delta_phi.astype(object),
        "obliflow_waste_penalty": waste.astype(object),
        "obliflow_break_penalty": breaks.astype(object),
        "obliflow_obligation_coverage": coverage.astype(object),
        "obliflow_artifact_utilization": utilization.astype(object),
    }
    return BatchedFlowRewards(
        flow_rewards=torch.tensor(flow_values, dtype=torch.float32, device=device),
        cut_blames=torch.tensor(cut_values, dtype=torch.float32, device=device),
        extra_info=extra_info,
        metrics=metrics,
    )


def compute_obliflow_outcome_advantage(
    token_level_rewards: torch.Tensor,
    flow_rewards: torch.Tensor,
    cut_blames: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.3,
    mode: str = "mean_std_norm",
    use_terminal_reward: bool = True,
    epsilon: float = 1e-6,
):
    response_length = response_mask.shape[-1]
    terminal_scores = token_level_rewards.sum(dim=-1)
    if use_terminal_reward:
        traj_advantages = _trajectory_advantages(terminal_scores, index, traj_index, mode=mode, epsilon=epsilon)
    else:
        traj_advantages = torch.zeros_like(terminal_scores)
    raw_scores = traj_advantages + alpha * flow_rewards + beta * cut_blames

    normalized = raw_scores.clone()
    with torch.no_grad():
        for uid in np.unique(index):
            loc = np.where(index == uid)[0]
            loc_tensor = torch.tensor(loc, device=raw_scores.device, dtype=torch.long)
            group = raw_scores.index_select(0, loc_tensor)
            mean = group.mean()
            if mode == "mean_norm" or len(loc) == 1:
                normalized.index_copy_(0, loc_tensor, group - mean)
            elif mode == "mean_std_norm":
                normalized.index_copy_(0, loc_tensor, (group - mean) / (group.std(unbiased=False) + epsilon))
            else:
                raise ValueError(f"Unknown ObliFlow mode: {mode}")

    advantages = normalized.unsqueeze(-1).tile([1, response_length]) * response_mask
    return advantages, advantages


def _trajectory_advantages(
    terminal_scores: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    *,
    mode: str,
    epsilon: float,
) -> torch.Tensor:
    traj_advantages = torch.zeros_like(terminal_scores)
    with torch.no_grad():
        for uid in np.unique(index):
            traj_locs: dict[Any, int] = {}
            for row, (row_uid, row_traj) in enumerate(zip(index, traj_index)):
                if row_uid == uid and row_traj not in traj_locs:
                    traj_locs[row_traj] = row
            if not traj_locs:
                continue
            rows = list(traj_locs.values())
            row_tensor = torch.tensor(rows, device=terminal_scores.device, dtype=torch.long)
            group = terminal_scores.index_select(0, row_tensor)
            mean = group.mean()
            if mode == "mean_norm" or len(rows) == 1:
                norm_group = group - mean
            elif mode == "mean_std_norm":
                norm_group = (group - mean) / (group.std(unbiased=False) + epsilon)
            else:
                raise ValueError(f"Unknown ObliFlow mode: {mode}")
            traj_to_adv = {traj: norm_group[i] for i, traj in enumerate(traj_locs)}
            for row, (row_uid, row_traj) in enumerate(zip(index, traj_index)):
                if row_uid == uid:
                    traj_advantages[row] = traj_to_adv[row_traj]
    return traj_advantages
