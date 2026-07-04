# ObliFlow-GRPO

ObliFlow-GRPO is the project-specific reinforcement learning recipe for long-horizon LLM agent training in this repository. It builds structured rollout graphs from agent-environment trajectories and uses graph-derived step signals together with episode-level outcomes to provide denser credit assignment than trajectory-only GRPO.

## Key Idea

During each training iteration, ObliFlow-GRPO:

1. Collects multi-turn trajectories via the agent-environment loop (same as GiGPO).
2. Builds a `ComplexGraph` per task group from the observed state transitions.
3. Runs reverse Dijkstra from the goal state (`next_obs = 'success'`) to compute the shortest-path distance of every observed state to the goal.
4. Assigns step returns as `10 × gamma^d` where `d` is the distance of `next_obs` to the goal — states closer to the goal receive higher returns.
5. Normalizes step-level advantages within same-state groups and optionally combines with episode-level advantages (same normalization as GiGPO).


## Configuration

ObliFlow-GRPO has its own trainer entry point (`recipe.obliflow_grpo.main_obliflow_grpo`) and Hydra config (`recipe/obliflow_grpo/config/obliflow_grpo_trainer.yaml`). Set `algorithm.adv_estimator=obliflow_grpo` to activate it.

### Core algorithm parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.adv_estimator` | `obliflow_grpo` | Must be set to `obliflow_grpo`. |
| `algorithm.gamma` | `0.10` | Decay factor for graph-distance returns. Step return = `10 × gamma^d`, so smaller gamma creates steeper reward differences between states at different distances. Typically much smaller than GiGPO's `0.95`. |
| `algorithm.obliflow_grpo.step_advantage_w` | `1.0` | Weight on the graph-derived step-level advantage. |
| `algorithm.obliflow_grpo.episode_advantage_w` | `1.0` | Weight on the episode-level outcome advantage (same normalization as GiGPO). Set to `0.0` to rely solely on the graph step signal. |
| `algorithm.obliflow_grpo.mode` | `mean_std_norm` | Normalization mode for both step and episode advantages. `mean_std_norm`: subtract mean and divide by std. `mean_norm`: subtract mean only. |
| `algorithm.obliflow_grpo.normalize_distance` | `False` | If `False`, step return = `10 × gamma^(d_new)` where `d_new` is the distance of `next_obs` to the goal. If `True`, step return = `10 × gamma^(d_new − d_old + 1)`, which rewards progress (closing distance) rather than proximity. |
| `algorithm.obliflow_grpo.enable_similarity` | `False` | If `True`, merge observations with similarity ≥ `similarity_thresh` into a single graph node, reducing graph fragmentation from minor text differences. |
| `algorithm.obliflow_grpo.similarity_thresh` | `0.95` | String similarity threshold used when `enable_similarity=True`. |

## Quick Start

Set the following environment variables before running:

```bash
export HF_HOME=<path to huggingface cache>
export WANDB_API_KEY=<your wandb key>
export WANDB_DIR=<path to wandb log dir>
export CHECKPOINTS_DIR=<path to save checkpoints>
```

### ALFWorld — Qwen2.5-1.5B

```bash
bash recipe/obliflow_grpo/run_qwen2.5_1.5b_alfworld_train.sh
```
[![W&B](https://img.shields.io/badge/W%26B-logs-yellow?logo=weightsandbiases)](https://api.wandb.ai/links/xincheng9215-nanyang-technological-university-singapore/jwmw0qbr)

### ALFWorld — Qwen2.5-7B

```bash
bash recipe/obliflow_grpo/run_qwen2.5_7b_alfworld_train.sh
```

### WebShop — Qwen2.5-1.5B

```bash
bash recipe/obliflow_grpo/run_qwen2.5_1.5b_webshop_train.sh
```

### WebShop — Qwen2.5-7B

```bash
bash recipe/obliflow_grpo/run_qwen2.5_7b_webshop_train.sh
```

### Sokoban — Qwen2.5-VL-3B

```bash
bash recipe/obliflow_grpo/run_qwen2.5_vl_3b_sokoban_train.sh
```

## Visualization

Two visualization scripts are provided in `recipe/obliflow_grpo/`.

### `compare_advantages.py` — advantage comparison across methods

Loads a saved rollout batch (`.pth`), builds the state-transition graph, and generates PDF plots comparing GRPO / GiGPO / ObliFlow-GRPO advantages side-by-side.

```bash
# Interactive — pick a task from a menu
python -m recipe.obliflow_grpo.compare_advantages <path_to_batch.pth>

# Non-interactive — specify task by index or UID
python -m recipe.obliflow_grpo.compare_advantages <path_to_batch.pth> 0
python -m recipe.obliflow_grpo.compare_advantages <path_to_batch.pth> <uid>
```

Output is written to `vis_<uid8>/`. Per-task contents:

| File | Description |
|------|-------------|
| `00_trajectories.pdf` | Graph with edges colored by trajectory (each trajectory gets a unique color). |
| `01_grpo_full.pdf` | Graph with edges colored by GRPO advantage (all trajectories). |
| `02_gigpo_full.pdf` | Graph with edges colored by GiGPO advantage (all trajectories). |
| `03_obliflow_grpo_full.pdf` | Graph with edges colored by ObliFlow-GRPO advantage (all trajectories). |
| `traj00–N_*_{grpo,gigpo,obliflow_grpo}.pdf` | Per-trajectory highlight: only one trajectory's edges are colored (by advantage rank), all others are greyed out. |

Edge color encodes advantage rank per source node: red = worst action from that state, green = best.

### `visualize_task.py` — single task graph viewer

Quick interactive viewer that prints all tasks in a batch and saves a PNG graph for the selected one.

```bash
python -m recipe.obliflow_grpo.visualize_task <path_to_batch.pth>
```

Generated `vis_*/` directories are local visualization outputs and are not part of the source tree.
