# ObliFlow-GRPO

<p align="center">
  <b>Structured credit assignment for long-horizon LLM agent training</b>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-3776AB">
  <img alt="RL" src="https://img.shields.io/badge/RL-GRPO-0A7">
  <img alt="Agent Tasks" src="https://img.shields.io/badge/tasks-ALFWorld%20%7C%20WebShop%20%7C%20Sokoban-purple">
</p>

ObliFlow-GRPO is a reinforcement-learning recipe for training long-horizon LLM agents. It is built on the veRL / verl-agent training stack and adds a project-specific credit-assignment path under [`recipe/obliflow_grpo`](./recipe/obliflow_grpo).

The current implementation constructs structured rollout graphs from agent-environment trajectories, derives dense step-level credit from graph distances, and combines it with episode outcomes through a GRPO-compatible trainer.

## Why This Repository Exists

Standard trajectory-level rewards are sparse for multi-turn agents: a final success or failure score often hides which intermediate tool call, observation, or action actually moved the task forward. ObliFlow-GRPO keeps the scalable rollout and distributed training machinery from veRL while adding a dedicated recipe for structured credit assignment across long interaction traces.

## Highlights

- **Dedicated ObliFlow-GRPO recipe**: trainer, Hydra config, rollout loop, environment manager, and visualization tools live together under `recipe/obliflow_grpo`.
- **Dense step credit**: rollout transitions are organized into task-level graphs, then converted into step returns before advantage normalization.
- **Episode + step control**: `step_advantage_w` and `episode_advantage_w` let you tune the balance between graph-derived feedback and terminal task outcomes.
- **Ready-to-run scripts**: ALFWorld, WebShop, and Sokoban launch scripts are provided for Qwen2.5 / Qwen2.5-VL models.
- **Visual diagnostics**: compare GRPO, GiGPO, and ObliFlow-GRPO advantages on the same saved rollout batch.

## Repository Layout

```text
.
├── recipe/obliflow_grpo/          # ObliFlow-GRPO training recipe
│   ├── config/                    # Hydra config
│   ├── core_graph.py              # rollout graph construction and step-credit logic
│   ├── main_obliflow_grpo.py      # training entry point
│   ├── obliflow_grpo_ray_trainer.py
│   ├── rollout_loop.py
│   ├── *_train.sh                 # runnable task/model scripts
│   └── compare_advantages.py      # visualization and diagnostics
├── agent_system/                  # inherited agent environments, memory, rollout utilities
├── examples/                      # inherited baseline examples and preprocessors
├── tests/                         # sanity, protocol, trainer, worker, and utility tests
└── verl/                          # veRL trainer, workers, models, and distributed runtime
```

## Installation

Create an environment first. Python 3.10+ is recommended for the current dependency stack.

```bash
git clone https://github.com/laplace-pym/ObliFlow-GRPO.git
cd ObliFlow-GRPO

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For development utilities:

```bash
pip install -r requirements.txt
```

Some rollout backends require extra runtime packages and GPU-specific versions. Match the backend to your environment before launching large runs:

```bash
# vLLM backend
pip install "vllm>=0.8.5,<=0.11.0"

# SGLang backend
pip install "sglang[srt,openai]==0.5.5"
```

## Quick Start

Set the common runtime variables:

```bash
export HF_HOME=/path/to/huggingface/cache
export WANDB_API_KEY=<your-wandb-key>
export WANDB_DIR=/path/to/wandb
export CHECKPOINTS_DIR=/path/to/checkpoints
export CUDA_VISIBLE_DEVICES=0,1
```

Run ALFWorld with Qwen2.5-1.5B:

```bash
bash recipe/obliflow_grpo/run_qwen2.5_1.5b_alfworld_train.sh
```

Other provided scripts:

```bash
bash recipe/obliflow_grpo/run_qwen2.5_7b_alfworld_train.sh
bash recipe/obliflow_grpo/run_qwen2.5_1.5b_webshop_train.sh
bash recipe/obliflow_grpo/run_qwen2.5_7b_webshop_train.sh
bash recipe/obliflow_grpo/run_qwen2.5_vl_3b_sokoban_train.sh
```

The scripts use `vllm` by default. You can pass the engine as the first argument:

```bash
bash recipe/obliflow_grpo/run_qwen2.5_1.5b_alfworld_train.sh sglang
```

## Core Configuration

The main config is [`recipe/obliflow_grpo/config/obliflow_grpo_trainer.yaml`](./recipe/obliflow_grpo/config/obliflow_grpo_trainer.yaml).

```yaml
algorithm:
  adv_estimator: obliflow_grpo
  gamma: 0.10
  obliflow_grpo:
    step_advantage_w: 1.0
    episode_advantage_w: 1.0
    mode: mean_std_norm
    enable_similarity: false
    similarity_thresh: 0.95
    normalize_distance: false
```

Important knobs:

| Parameter | Meaning |
| --- | --- |
| `algorithm.adv_estimator` | Must be `obliflow_grpo` to activate the recipe. |
| `algorithm.gamma` | Decay factor for graph-distance step returns. |
| `algorithm.obliflow_grpo.step_advantage_w` | Weight for graph-derived step advantage. |
| `algorithm.obliflow_grpo.episode_advantage_w` | Weight for episode-level outcome advantage. |
| `algorithm.obliflow_grpo.mode` | Advantage normalization mode: `mean_std_norm` or `mean_norm`. |
| `algorithm.obliflow_grpo.normalize_distance` | If true, use distance deltas instead of absolute distance to the goal. |
| `algorithm.obliflow_grpo.enable_similarity` | Merge similar observations into shared graph nodes. |

Hydra overrides can be passed directly through the training scripts:

```bash
bash recipe/obliflow_grpo/run_qwen2.5_1.5b_alfworld_train.sh \
  algorithm.gamma=0.2 \
  algorithm.obliflow_grpo.episode_advantage_w=0.5
```

## Visualization

To inspect how advantages differ on a saved rollout batch:

```bash
python -m recipe.obliflow_grpo.compare_advantages <path_to_batch.pth>
```

Non-interactive selection by task index or UID:

```bash
python -m recipe.obliflow_grpo.compare_advantages <path_to_batch.pth> 0
python -m recipe.obliflow_grpo.compare_advantages <path_to_batch.pth> <uid>
```

The script writes `vis_<uid8>/` directories with:

- `00_trajectories.pdf`: trajectory-colored graph
- `01_grpo_full.pdf`: GRPO advantage graph
- `02_gigpo_full.pdf`: GiGPO advantage graph
- `03_obliflow_grpo_full.pdf`: ObliFlow-GRPO advantage graph
- `traj*_*.pdf`: per-trajectory highlighted graphs

Generated `vis_*/` directories are ignored by Git.

## Development Checks

Fast sanity checks:

```bash
python3 -m pytest tests/sanity/test_project_branding.py tests/sanity/test_import.py -q
python3 -m compileall -q recipe/obliflow_grpo tests/sanity/test_project_branding.py
```

The branding sanity test prevents the old recipe name from reappearing in source paths or text files.

## Notes

- The installable Python package is still named `verl` because the project extends the veRL runtime and imports its trainer/workers directly.
- Large training runs require GPU-compatible versions of PyTorch, vLLM or SGLang, Ray, and model-specific dependencies.
- The recipe scripts assume prepared task datasets under the same conventions used by the inherited `examples.data_preprocess.prepare` entry point.

## Acknowledgements

This repository builds on [veRL](https://github.com/volcengine/verl) and the verl-agent training stack. The included environments and utilities also inherit work from ALFWorld, WebShop, Sokoban, Search-R1, Gym Cards, AppWorld, and related open-source projects. Please keep the original license and notice files when redistributing modified versions.

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`Notice.txt`](./Notice.txt).
