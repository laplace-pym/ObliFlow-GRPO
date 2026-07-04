set -x

if [[ "${VLLM_ATTENTION_BACKEND:-}" == "XFORMERS" || -z "${VLLM_ATTENTION_BACKEND:-}" ]]; then
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
fi
export OBLIFLOW_OFFLINE_SUBTASK_PATH=${OBLIFLOW_OFFLINE_SUBTASK_PATH:-/opt/tiger/alfworld_subtasks.jsonl}
export MY_HOST_IP=${MY_HOST_IP:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
if [[ "${NCCL_SOCKET_IFNAME:-}" == "lo" ]]; then
    unset NCCL_SOCKET_IFNAME
fi

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=64
group_size=8
experiment_name=${EXP:-obliflow_change_qwen2.5_1.5b_alfworld_8gpu}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/mnt/bn/qutuo-my2/jinhongbo/pym/checkpoints}

python3 -m recipe.ObliFlow.main_obliflow \
    algorithm.adv_estimator=obliflow \
    algorithm.obliflow.alpha=1.0 \
    algorithm.obliflow.beta=0.3 \
    algorithm.obliflow.lambda_cost=0.02 \
    algorithm.obliflow.eta_waste=0.20 \
    algorithm.obliflow.rho_break=0.50 \
    algorithm.obliflow.mode=mean_std_norm \
    algorithm.obliflow.use_min_cut=True \
    algorithm.obliflow.use_waste_penalty=True \
    algorithm.obliflow.use_offline_subtasks=True \
    algorithm.obliflow.offline_subtask_path="${OBLIFLOW_OFFLINE_SUBTASK_PATH}" \
    algorithm.obliflow.use_llm_decomposition=False \
    algorithm.obliflow.use_llm_verifier=True \
    algorithm.obliflow.llm_fallback_to_rules=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=/home/tiger/data/verl-agent/text/train.parquet \
    data.val_files=/home/tiger/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/${experiment_name}" \
    trainer.save_local_metrics=True \
    trainer.metrics_local_dir="${CHECKPOINTS_DIR}/${experiment_name}/local_metrics" \
    trainer.metrics_jsonl_name=metrics.jsonl \
    trainer.important_metrics_jsonl_name=important_metrics.jsonl \
    trainer.val_before_train=True
