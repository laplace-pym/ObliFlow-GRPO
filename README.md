<h1 align="center">ObliFlow-GRPO</h1>

<p align="center">
  <strong>
    Training-Free Obligation-Flow Credit Assignment<br />
    for Long-Horizon Multi-Tool Agents
  </strong>
</p>

<p align="center">
  <a href="./arxiv_author.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-B31B1B?style=for-the-badge" alt="Paper PDF" />
  </a>
</p>

<p align="center">
  <a href="./arxiv_author.pdf">
    <img src="./image.png" alt="First-page preview of the ObliFlow-GRPO paper" width="900" />
  </a>
</p>

<p align="center"><sub>Click the preview to open the full paper.</sub></p>

ObliFlow-GRPO is a credit-assignment recipe for long-horizon agent
reinforcement learning. It augments outcome-level GRPO with verified
obligation flow rewards, so each tool/action step can receive dense feedback
for producing, using, transforming, and discharging task-relevant artifacts.

This repository is built on the `verl-agent` training stack and keeps the
existing agent environments and rollout infrastructure. The main additions are
the `obli_flow` package, the `recipe/ObliFlow` trainer, and practical launch
scripts for WebShop and ALFWorld.

## Why ObliFlow-GRPO

Long-horizon agent tasks often expose only a sparse terminal reward. This makes
GRPO-style training depend heavily on group-level outcome differences, even
when the useful signal is hidden in intermediate tool calls. ObliFlow-GRPO adds
a trajectory-level flow graph that answers more local questions:

- Did this action create a useful artifact?
- Was the artifact consumed by later actions?
- Did the trajectory satisfy a required subtask or obligation?
- Did a step break a previously useful handoff?
- Which missing obligation should receive blame when the final result fails?

The result is a dense shaping signal that remains tied to verifiable evidence
inside the trajectory instead of relying only on free-form reasoning traces.

## Core Idea

For each sampled trajectory, ObliFlow builds a graph with four kinds of
information:

- **Action nodes**: parsed tool calls, text responses, search operations, and
  environment actions.
- **Artifact nodes**: entities, observations, URLs, item attributes, rooms,
  inventory facts, and other task-specific evidence extracted from steps.
- **Obligation nodes**: required subtasks from offline decomposition, optional
  LLM decomposition, or deterministic rule fallbacks.
- **Flow edges**: production, consumption, transformation, and obligation
  discharge edges.

The graph is then scored with a potential-based flow reward:

```text
flow_reward_t =
  delta_phi_t
  - eta_waste * waste_penalty_t
  - rho_break * break_penalty_t
  + cut_blame_t
```

During advantage computation, ObliFlow-GRPO combines terminal outcome reward
with verified flow credit:

```text
score_t = terminal_advantage
          + alpha * flow_reward_t
          + beta  * cut_blame_t
```

The combined score is normalized within each prompt group and broadcast to the
response tokens used by the PPO/GRPO update.

## What Is New In This Codebase

- **Obligation-centered credit assignment**: tasks are decomposed into explicit
  obligations, then intermediate actions are rewarded when they verifiably move
  the trajectory toward those obligations.
- **Action-artifact flow graph**: ObliFlow tracks how evidence is produced and
  reused across a trajectory, instead of treating each step as an isolated text
  transition.
- **Verifier-backed dense rewards**: deterministic validators, offline
  subtasks, optional OpenAI-compatible LLM decomposition, and optional LLM
  verification can all contribute evidence.
- **Waste and broken-handoff penalties**: actions that create unused artifacts
  or invalidate useful handoffs are penalized.
- **GRPO integration**: `algorithm.adv_estimator=obliflow` plugs directly into
  the trainer and preserves the rest of the rollout and optimization pipeline.
- **Operational metrics**: local JSONL metrics and TensorBoard/W&B metrics
  expose flow reward quality, obligation coverage, artifact utilization, waste,
  broken handoffs, and verifier behavior.

## Repository Layout

```text
.
├── obli_flow/                         # ObliFlow graph, reward, verifier, schema
│   ├── core_obliflow.py               # trajectory grouping, reward tensors, advantage
│   ├── graph_builder.py               # action-artifact-obligation graph builder
│   ├── reward.py                      # flow potential, penalties, cut blame
│   ├── obligations/                   # offline, LLM, and rule obligation sources
│   ├── extractors/                    # task-specific artifact extraction
│   └── llm_client.py                  # OpenAI-compatible client and cache
├── recipe/ObliFlow/
│   ├── main_obliflow.py               # training entrypoint
│   ├── obliflow_ray_trainer.py        # Ray trainer with ObliFlow advantage path
│   ├── rollout_loop.py                # rollout collection with step records
│   └── config/obliflow_trainer.yaml   # main ObliFlow config
├── examples/obliflow_trainer/
│   ├── run_alfworld.sh                # standard ALFWorld launch example
│   └── run_webshop.sh                 # standard WebShop launch example
├── run_qwen2.5_1.5b_webshop_changed.sh # cloud-friendly WebShop 1.5B wrapper
└── run_qwen2.5_7b_webshop_changed.sh   # cloud-friendly WebShop 7B wrapper
```

The original baseline recipes such as GraphGPO and GiGPO are still present for
comparison.

## Installation

Follow the base environment setup used by `verl-agent`, then install this
repository in editable mode:

```bash
git clone https://github.com/laplace-pym/ObliFlow-GRPO.git
cd ObliFlow-GRPO
pip install -e .
```

For WebShop and ALFWorld runs, prepare the corresponding datasets and
environment assets before launching training.

## Quick Start

### 1. Configure common paths

```bash
export CHECKPOINTS_DIR=/path/to/checkpoints
export HF_HOME=/path/to/huggingface_cache
export WANDB_API_KEY=your_wandb_key

# Optional: offline subtask decomposition used by ObliFlow.
export OBLIFLOW_OFFLINE_SUBTASK_PATH=/path/to/subtasks.jsonl

# Optional: enable LLM decomposition or verification through an OpenAI-compatible API.
export GLM_API_KEY=your_api_key
export GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export GLM_MODEL=glm-4-flash
```

### 2. Run ALFWorld

```bash
bash examples/obliflow_trainer/run_alfworld.sh
```

### 3. Run WebShop

```bash
bash examples/obliflow_trainer/run_webshop.sh
```

### 4. Run the cloud WebShop wrappers

The root-level scripts are more operational wrappers around WebShop launch
commands. They set Java/Ray environment variables, link WebShop assets, expose
model and data overrides, and write local metrics under the selected output
directory.

```bash
# Qwen2.5-1.5B WebShop
bash run_qwen2.5_1.5b_webshop_changed.sh

# Qwen2.5-7B WebShop
bash run_qwen2.5_7b_webshop_changed.sh
```

Common overrides:

```bash
export REPO_DIR=/path/to/ObliFlow-GRPO
export PYTHON_BIN=/path/to/python
export MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/val.parquet
export WEBSHOP_SRC_DIR=/path/to/webshop_assets
export OBLIFLOW_OFFLINE_SUBTASK_PATH=/path/to/webshop_subtasks.jsonl
export OUT_DIR=/path/to/output

bash run_qwen2.5_1.5b_webshop_changed.sh
```

For the 7B wrapper, the most useful overrides are:

```bash
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export GPU_IDS=0,1,2,3
export ROLLOUT_TP=4
export BASE_WRAPPER=/path/to/run_qwen2.5_1.5b_webshop_changed.sh

bash run_qwen2.5_7b_webshop_changed.sh
```

## Main Configuration Knobs

The central config is
`recipe/ObliFlow/config/obliflow_trainer.yaml`.

| Key | Meaning |
| --- | --- |
| `algorithm.adv_estimator=obliflow` | Enables the ObliFlow advantage path. |
| `algorithm.obliflow.alpha` | Weight for dense flow reward. |
| `algorithm.obliflow.beta` | Weight for cut-blame credit. |
| `algorithm.obliflow.lambda_cost` | Edge/action cost used by flow scoring. |
| `algorithm.obliflow.eta_waste` | Penalty for unused or low-value artifacts. |
| `algorithm.obliflow.rho_break` | Penalty for broken handoffs. |
| `algorithm.obliflow.use_terminal_reward` | Keeps terminal outcome reward in the combined advantage. |
| `algorithm.obliflow.use_offline_subtasks` | Loads precomputed obligation decompositions. |
| `algorithm.obliflow.use_llm_decomposition` | Uses an LLM to decompose tasks when needed. |
| `algorithm.obliflow.use_llm_verifier` | Uses an LLM verifier for obligation discharge evidence. |
| `algorithm.obliflow.enable_posthoc_extractor` | Extracts missing artifacts after rollout. |
| `algorithm.obliflow.save_local_metrics` | Writes local JSONL metrics for debugging. |

The default recipe keeps deterministic and fallback paths enabled, so training
can still run when LLM decomposition or verification is unavailable.

## Metrics

ObliFlow logs metrics with the `obliflow/` prefix, including:

- `obliflow/flow_reward_mean`
- `obliflow/flow_reward_std`
- `obliflow/cut_blame_mean`
- `obliflow/phi_final`
- `obliflow/artifact_count`
- `obliflow/edge_count`
- `obliflow/obligation_count`
- `obliflow/artifact_utilization_rate`
- `obliflow/obligation_coverage`
- `obliflow/waste_tool_ratio`
- `obliflow/waste_penalty_mean`
- `obliflow/break_penalty_mean`
- `obliflow/broken_handoff_rate`
- `obliflow/discharge_edge_valid_rate`

When `save_local_metrics=true`, the trainer also writes:

```text
<output_dir>/local_metrics/metrics.jsonl
<output_dir>/local_metrics/important_metrics.jsonl
```

These files are useful for inspecting per-step flow reward, cut blame, verifier
outcomes, and rollout-level debugging signals without depending only on W&B.

## Implementation Notes

The training path is:

1. Rollout workers collect step-level action and observation records.
2. `compute_flow_step_rewards` groups records by trajectory.
3. `graph_builder.py` constructs the action-artifact-obligation graph.
4. `reward.py` computes flow potential, waste penalties, break penalties, and
   cut blame.
5. `compute_obliflow_outcome_advantage` combines terminal reward with flow
   credit and normalizes within the prompt group.
6. The trainer applies the standard PPO/GRPO optimization update.

This keeps ObliFlow localized to the reward and advantage path while reusing
the existing rollout, environment, model, and optimizer infrastructure.

## Acknowledgements

This project builds on `verl-agent`, `verl`, WebShop, ALFWorld, GraphGPO, and
GiGPO. ObliFlow-GRPO adds obligation-flow credit assignment and the corresponding
trainer integration for long-horizon agent RL.
