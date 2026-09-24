#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

# Let the CUDA allocator reuse fragmented "reserved but unallocated" memory via
# expandable segments (same fix as Stage-1), avoiding mid-epoch OOM on long
# reasoning batches without touching micro_batch_size / LR / global batch.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CATEGORY="${CATEGORY:-Video_Games}"
BASE_MODEL="${BASE_MODEL:-./checkpoint/${CATEGORY}/stage1/final_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoint/${CATEGORY}/stage2_scorer}"
RUN_NAME="${RUN_NAME:-${CATEGORY}_stage2_reasoning_activation_Qwen3-1.7B}"
LOG_FILE="./logs/${RUN_NAME}.txt"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29519}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
EVAL_CUDA_LIST="${EVAL_CUDA_LIST:-0,1,2,3,4,5,6,7}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/recsys_eval"
EVAL_NUM_SAMPLES=-1
WANDB_PROJECT="${WANDB_PROJECT:-EvoRec_Stage2}"
WANDB_RUN_ID="${RUN_NAME}-$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_TO="${REPORT_TO:-wandb}"   # set REPORT_TO=none to train without W&B

mkdir -p ./logs

run_recsys_eval() {
    local checkpoint="$1"
    echo "Running post-training recommendation evaluation for ${checkpoint}"
    # The evaluator attaches to the training run with resume="must", so it can
    # only upload when the trainer actually created that run.
    local wandb_args=()
    if [[ "${REPORT_TO}" == "wandb" ]]; then
        wandb_args=(--upload-to-wandb
                    --wandb-project "${WANDB_PROJECT}"
                    --wandb-run-name "${RUN_NAME}"
                    --wandb-run-id "${WANDB_RUN_ID}")
    fi
    python "$REPO_ROOT/evaluation/evaluate_stage2_checkpoint.py" \
        --checkpoint "${checkpoint}" \
        --category "${CATEGORY}" \
        --output-dir "${EVAL_OUTPUT_DIR}" \
        --cuda-list "${EVAL_CUDA_LIST}" \
        --num-samples "${EVAL_NUM_SAMPLES}" \
        --num-beams 10 \
        ${wandb_args[@]+"${wandb_args[@]}"}
}

{
echo "category=${CATEGORY} | base_model=${BASE_MODEL} (Stage-1 checkpoint; all data read from ./data/${CATEGORY}/ by --category)"
    echo "wandb_project=${WANDB_PROJECT} | wandb_run_id=${WANDB_RUN_ID}"

# Explicit DeepSpeed launch (no HF Trainer): the training loop lives in sft_reasoning_activation.py::main.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} \
    "$SCRIPT_DIR/sft_reasoning_activation.py" \
    --base_model "${BASE_MODEL}" \
    --micro_batch_size 8 \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate 1e-5 \
    --cutoff_len 1024 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to "${REPORT_TO}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_run_id "${WANDB_RUN_ID}" \
    --category "${CATEGORY}" \
    --seed 42 \
    --zero_stage 2 \
    --dtype bf16 \
    --deepspeed

# epoch_N is the actual model state at the end of training; final_checkpoint is
# the loss-best convenience copy and may point to an earlier epoch.
POSTTRAIN_MODEL="${OUTPUT_DIR}/epoch_${NUM_EPOCHS}"
if [[ ! -d "${POSTTRAIN_MODEL}" ]]; then
    echo "Error: post-training checkpoint not found at ${POSTTRAIN_MODEL}"
    exit 1
fi
run_recsys_eval "${POSTTRAIN_MODEL}"
} > "${LOG_FILE}" 2>&1
