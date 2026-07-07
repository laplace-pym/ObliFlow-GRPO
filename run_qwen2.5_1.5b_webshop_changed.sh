#!/usr/bin/env bash
set -euo pipefail

REF_SCRIPT="${REF_SCRIPT:-/mnt/bn/qutuo-my2/jinhongbo/pym/run_qwen2.5_7b_alfworld_changed.sh}"
REPO_DIR="${REPO_DIR:-/opt/tiger/obliflow-grpo—change}"
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/.venvs/verl-agent-webshop/bin/python}"
SITE_DIR="${SITE_DIR:-/mnt/bn/qutuo-my2/jinhongbo/pym/no_torchvision_site}"
MODEL_PATH="${MODEL_PATH:-/opt/tiger/obliflow-grpo/models/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-/home/tiger/data/verl-agent/text/train.parquet}"
VAL_FILE="${VAL_FILE:-/home/tiger/data/verl-agent/text/test.parquet}"
SUBTASK_PATH="${OBLIFLOW_OFFLINE_SUBTASK_PATH:-/opt/tiger/webshop_subtasks.jsonl}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/mnt/bn/qutuo-my2/jinhongbo/pym/qwen2.5_1.5b_webshop/change_obliflow_offline_webshop_${TS}}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run.log}"

JAVA_HOME="${WEBSHOP_JAVA_HOME:-/home/tiger/tools/jdk-11.0.29+7}"
if [[ ! -d "$JAVA_HOME" ]]; then
  JAVA_HOME="$(find /home/tiger/tools -maxdepth 1 -type d -name 'jdk-11*' | head -1)"
fi

[[ -n "$JAVA_HOME" && -d "$JAVA_HOME" ]] || { echo "Missing JDK 11 under /home/tiger/tools" >&2; exit 1; }
[[ -f "$REF_SCRIPT" ]] || { echo "Missing REF_SCRIPT: $REF_SCRIPT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing executable PYTHON_BIN: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$REPO_DIR" ]] || { echo "Missing REPO_DIR: $REPO_DIR" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing MODEL_PATH: $MODEL_PATH" >&2; exit 1; }
[[ -f "$SUBTASK_PATH" ]] || { echo "Missing SUBTASK_PATH: $SUBTASK_PATH" >&2; exit 1; }
[[ -x "$JAVA_HOME/bin/java" ]] || { echo "Missing java executable: $JAVA_HOME/bin/java" >&2; exit 1; }

WEBSHOP_SRC_DIR="${WEBSHOP_SRC_DIR:-/opt/tiger/obliflow-grpo/agent_system/environments/env_package/webshop/webshop}"
WEBSHOP_DST_DIR="${WEBSHOP_DST_DIR:-${REPO_DIR}/agent_system/environments/env_package/webshop/webshop}"

link_required_dir() {
  local src="$1"
  local dst="$2"

  [[ -d "$src" ]] || { echo "Missing WebShop source asset directory: $src" >&2; exit 1; }
  if [[ -d "$dst" ]]; then
    return
  fi
  if [[ -e "$dst" || -L "$dst" ]]; then
    echo "WebShop asset path exists but is not a usable directory: $dst" >&2
    exit 1
  fi
  ln -s "$src" "$dst"
}

link_required_dir "$WEBSHOP_SRC_DIR/data" "$WEBSHOP_DST_DIR/data"
for webshop_asset_dir in indexes indexes_100 indexes_1k indexes_100k resources resources_100 resources_1k resources_100k; do
  link_required_dir "$WEBSHOP_SRC_DIR/search_engine/$webshop_asset_dir" "$WEBSHOP_DST_DIR/search_engine/$webshop_asset_dir"
done

[[ -f "$WEBSHOP_DST_DIR/data/items_shuffle_1000.json" ]] || { echo "Missing WebShop product file after asset setup" >&2; exit 1; }
[[ -d "$WEBSHOP_DST_DIR/search_engine/indexes_1k" ]] || { echo "Missing WebShop 1k search index after asset setup" >&2; exit 1; }

JAVA_VERSION_OUTPUT="$("$JAVA_HOME/bin/java" -version 2>&1)"
case "$JAVA_VERSION_OUTPUT" in
  *'version "11.'*|*'version "12.'*|*'version "13.'*|*'version "14.'*|*'version "15.'*|*'version "16.'*|*'version "17.'*|*'version "18.'*|*'version "19.'*|*'version "20.'*|*'version "21.'*|*'version "22.'*|*'version "23.'*|*'version "24.'*|*'version "25.'*)
    ;;
  *)
    echo "WebShop/pyserini requires Java 11+, but WEBSHOP_JAVA_HOME points to:" >&2
    printf '%s\n' "$JAVA_VERSION_OUTPUT" >&2
    exit 1
    ;;
esac

export JAVA_HOME
VENV_BIN="$(dirname "$PYTHON_BIN")"
export PATH="$JAVA_HOME/bin:$VENV_BIN:$PATH"

ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
CLEANED_LD_LIBRARY_PATH="$(
  printf '%s' "$ORIGINAL_LD_LIBRARY_PATH" |
    tr ':' '\n' |
    awk '
      $0 == "" { next }
      $0 ~ /^\/usr\/local\/cuda[^/]*\/lib64$/ { next }
      $0 ~ /^\/usr\/local\/cuda[^/]*\/lib64\/stubs$/ { next }
      $0 ~ /\/jdk8|jdk8u|jdk1\.8/ { next }
      { paths[++n] = $0 }
      END {
        for (i = 1; i <= n; i++) {
          printf "%s%s", (i == 1 ? "" : ":"), paths[i]
        }
      }
    '
)"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
[[ -f "$JVM_PATH" ]] || { echo "Missing JVM shared library: $JVM_PATH" >&2; exit 1; }
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$JAVA_HOME/lib/server:$JAVA_HOME/lib:$JAVA_HOME/lib/jli${CLEANED_LD_LIBRARY_PATH:+:$CLEANED_LD_LIBRARY_PATH}"
unset CUDA_HOME CUDA_PATH
RAY_WORKER_PATH="$JAVA_HOME/bin:$VENV_BIN:/usr/local/bin:/usr/bin:/bin"

export REPO_DIR PYTHON_BIN SITE_DIR MODEL_PATH TRAIN_FILE VAL_FILE OUT_DIR LOG_FILE
export OBLIFLOW_OFFLINE_SUBTASK_PATH="$SUBTASK_PATH"
export GPU_IDS ROLLOUT_NAME ROLLOUT_TP
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ "${NCCL_SOCKET_IFNAME:-}" == =* ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME#=}"
fi
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  for iface_path in /sys/class/net/*; do
    iface="$(basename "$iface_path")"
    case "$iface" in
      lo|docker*|veth*|br-*|bonding_masters) continue ;;
    esac
    export NCCL_SOCKET_IFNAME="$iface"
    break
  done
fi

exec bash "$REF_SCRIPT" \
  data.max_prompt_length=4096 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  env.env_name=Webshop \
  env.max_steps=15 \
  env.history_length=2 \
  env.resources_per_worker.num_cpus=0.01 \
  env.webshop.use_small=True \
  env.webshop.human_goals=False \
  trainer.project_name=verl_agent_webshop \
  trainer.experiment_name=qwen2.5_1.5b_webshop_changed \
  trainer.save_local_metrics=True \
  "trainer.metrics_local_dir=${OUT_DIR}/local_metrics" \
  ray_init.num_cpus=64 \
  "++ray_init.runtime_env.env_vars.JAVA_HOME='${JAVA_HOME}'" \
  "++ray_init.runtime_env.env_vars.JVM_PATH='${JVM_PATH}'" \
  "++ray_init.runtime_env.env_vars.PATH='${RAY_WORKER_PATH}'" \
  "++ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'" \
  "$@"
