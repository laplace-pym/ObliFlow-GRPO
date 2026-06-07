import numpy as np
import torch

from obli_flow.core_obliflow import compute_obliflow_outcome_advantage


def test_obliflow_advantage_group_normalizes_and_uses_flow():
    token_rewards = torch.zeros(4, 3)
    token_rewards[:, -1] = torch.tensor([10.0, 0.0, 0.0, 10.0])
    flow_rewards = torch.tensor([2.0, 1.0, -1.0, 0.5])
    cut_blames = torch.tensor([0.0, -0.5, -2.0, 0.0])
    mask = torch.ones(4, 3)
    index = np.array(["u", "u", "u", "u"], dtype=object)
    traj_index = np.array(["t1", "t2", "t3", "t4"], dtype=object)

    advantages, returns = compute_obliflow_outcome_advantage(
        token_level_rewards=token_rewards,
        flow_rewards=flow_rewards,
        cut_blames=cut_blames,
        response_mask=mask,
        index=index,
        traj_index=traj_index,
        alpha=1.0,
        beta=0.3,
        mode="mean_norm",
    )

    assert advantages.shape == mask.shape
    assert returns.shape == mask.shape
    assert torch.isclose(advantages[:, 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert advantages[0, 0] > advantages[2, 0]


def test_obliflow_can_disable_terminal_reward():
    token_rewards = torch.zeros(2, 3)
    token_rewards[:, -1] = torch.tensor([10.0, 0.0])
    flow_rewards = torch.tensor([0.0, 1.0])
    cut_blames = torch.zeros(2)
    mask = torch.ones(2, 3)
    index = np.array(["u", "u"], dtype=object)
    traj_index = np.array(["t1", "t2"], dtype=object)

    advantages, _ = compute_obliflow_outcome_advantage(
        token_level_rewards=token_rewards,
        flow_rewards=flow_rewards,
        cut_blames=cut_blames,
        response_mask=mask,
        index=index,
        traj_index=traj_index,
        alpha=1.0,
        beta=0.3,
        mode="mean_norm",
        use_terminal_reward=False,
    )

    assert advantages[1, 0] > advantages[0, 0]
