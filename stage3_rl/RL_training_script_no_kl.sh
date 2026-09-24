#!/bin/bash

# ============================================================================
# Stage 3 — ranking-aware GRPO, no KL regularization.
#
# Both KL switches are disabled, so verl does not register a RefPolicy worker
# or compute reference-policy log probabilities.
#
# Each prompt samples G=16 reasoning trajectories; conditioned on each one, a
# trie built from the category's SID catalog constrains every decoding step and
# beam search with B=10 returns an ordered list of catalog-valid items. The
# rank of the ground-truth SID in that beam defines the NDCG@10 reward.
#
# Usage:
#   CATEGORY=Video_Games bash stage3_rl/RL_training_script_no_kl.sh
#   CATEGORY=Office_Products bash stage3_rl/RL_training_script_no_kl.sh
# ============================================================================

# Tested stack: CUDA 12.8 · torch 2.8.0 · vllm 0.10.2 · vendored verl 0.6
# export NCCL_P2P_DISABLE=1       # Disable NVLink
# export NCCL_IB_DISABLE=1        # Disable InfiniBand
# export NCCL_NET_GDR_LEVEL=0     # Disable GDR
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Start a private, node-local Ray cluster. Without this, ray.init() attaches to
# any pre-existing multi-node Ray cluster on the host and can schedule workers
# onto GPUs that another job already occupies, which fails with a CUDA OOM
# inside vLLM's wake_up().
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

# Reuse the W&B login persisted by the Stage-1/2 wandb.login() call.
# Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. vLLM's
# sleep/wake path uses CuMemAllocator, which is incompatible with it.

# ================================
# Adjust the GPU and node counts for the target machine.
# ================================
n_gpus_per_node=8
nnodes=1
category="${CATEGORY:-Video_Games}"
case "$category" in
    Video_Games)               reward_tag=Games ;;
    Office_Products)           reward_tag=Office ;;
    Industrial_and_Scientific) reward_tag=Industrial ;;
    *) echo "unknown CATEGORY: ${category}" >&2; exit 1 ;;
esac
experiment_name="${EXPERIMENT_NAME:-${category}_stage3_rl_no_kl_Qwen3-1.7B}"
stage2_checkpoint="${STAGE2_CHECKPOINT:-./checkpoint/${category}/stage2/final_checkpoint}"
data_dir="./data/${category}/rl"
reward_file="./verl/utils/reward_score/direct_recommendation_StepRule_${reward_tag}.py"
checkpoint_dir="./checkpoint/${category}/stage3"
log_file="./logs/${experiment_name}.log"
# ================================

mkdir -p ./logs

{
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${data_dir}/train.parquet" \
    data.val_files="${data_dir}/validation.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${stage2_checkpoint}" \
    actor_rollout_ref.actor.optim.lr=7e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    +actor_rollout_ref.rollout.sid_constrained_beam_size=10 \
    +actor_rollout_ref.rollout.sid_validation_beam_size=10 \
    +actor_rollout_ref.rollout.sid_category="${category}" \
    +actor_rollout_ref.rollout.sid_length=3 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.wandb_exclude_prefixes='["val-core/","val-aux/","timing_s/","timing_per_token_ms/","response_length_non_aborted/","global_seqlen/","perf/","critic/","actor/"]' \
    custom_reward_function.path="${reward_file}" \
    custom_reward_function.name="rule_base_reward" \
    trainer.project_name="${WANDB_PROJECT:-EvoRec_Stage3_RankingAware_RL}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 "$@"
} > "${log_file}" 2>&1