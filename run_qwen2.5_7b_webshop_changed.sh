#!/usr/bin/env bash
set -euo pipefail

BASE_WRAPPER="${BASE_WRAPPER:-/opt/tiger/run_qwen2.5_1.5b_webshop_changed.sh}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

export MODEL_PATH="${MODEL_PATH:-/opt/tiger/obliflow-grpo/models/Qwen2.5-7B-Instruct}"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export OUT_DIR="${OUT_DIR:-/mnt/bn/qutuo-my2/jinhongbo/pym/qwen2.5_7b_webshop/change_obliflow_offline_webshop_${TS}}"
export LOG_FILE="${LOG_FILE:-${OUT_DIR}/run.log}"

[[ -f "$BASE_WRAPPER" ]] || { echo "Missing BASE_WRAPPER: $BASE_WRAPPER" >&2; exit 1; }

exec bash "$BASE_WRAPPER" \
  trainer.experiment_name=qwen2.5_7b_webshop_changed \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  "$@"
