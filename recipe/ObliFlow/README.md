# ObliFlow Recipe

ObliFlow-GRPO is a GRPO recipe for long-horizon agent tasks. It adds post-hoc obligation-flow credit assignment on top of rollout trajectories:

1. Build per-trajectory obligations from offline subtasks, LLM decomposition, or rule-based fallbacks.
2. Extract action/observation artifacts from each rollout step.
3. Verify whether intermediate evidence discharges each obligation.
4. Convert verified obligation progress into dense step-level flow rewards.
5. Combine terminal outcome advantages with ObliFlow step rewards during GRPO advantage computation.

## Files

- `main_obliflow.py`: Hydra entry point.
- `obliflow_ray_trainer.py`: trainer integration and advantage computation hook.
- `rollout_loop.py`: multi-turn rollout collection.
- `config/obliflow_trainer.yaml`: default ObliFlow hyperparameters.

## Related Core Modules

- `obli_flow/core_obliflow.py`: batch-level reward and advantage API.
- `obli_flow/graph_builder.py`: artifact-obligation flow graph construction.
- `obli_flow/reward.py`: dense flow reward and blame assignment.
- `obli_flow/llm_obligations.py`: LLM task decomposition.
- `obli_flow/llm_verifier.py`: LLM or heuristic subtask completion checks.
- `obli_flow/offline_subtasks.py`: offline JSONL subtask loading.

## Run

```bash
python3 -m recipe.ObliFlow.main_obliflow algorithm.adv_estimator=obliflow
```

The example scripts under `examples/obliflow_trainer/` provide ALFWorld and WebShop launch commands with the expected environment variables.
