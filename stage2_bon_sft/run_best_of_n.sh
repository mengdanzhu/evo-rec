#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 2 — Best-of-N Rejection Sampling SFT on a single node (8 GPU).
#
# Four chained steps on one host:
#
#   1. SCORE   Freeze the single-CoT Stage-2 model (--scorer) and measure, for
#              every candidate trace,
#                  delta_k = log p(y*|h,z_k) - log p(y*|h)
#              i.e. think-mode-with-this-CoT minus non-think-no-CoT on the very
#              same prompt (Eq. 2). Runs stage2_bon_sft/score_cot_delta.py.
#   2. SELECT  Best-of-N: for each row keep the candidate with the LARGEST
#              delta, provided it clears MIN_DELTA; a row whose best candidate
#              still fails is rejected (Eq. 3).
#   3. TRAIN   Start from the Stage-1 SID-alignment checkpoint and run the
#              Stage-2 CoT trainer (stage2_bon_sft/train_stage2.py) on the one
#              selected trace per row -- completion-only SFT (Eq. 4).
#   4. EVAL    Thinking-mode trie-constrained beam search on the test split.
#              Runs automatically once training finishes (SKIP_EVAL=1 opts out).
#
# The global batch is 72 (8 GPUs x micro 9) and training runs for one epoch at
# lr 1e-5, matching the paper's Stage-2 recipe.
#
# Usage:
#   export WANDB_API_KEY=...
#   CATEGORY=Video_Games bash stage2_bon_sft/run_best_of_n.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CATEGORY="${CATEGORY:-Video_Games}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-9}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
CUTOFF_LEN="${CUTOFF_LEN:-1024}"
MASTER_PORT="${MASTER_PORT:-29633}"
SCORE_PORT="${SCORE_PORT:-29634}"
ZERO_STAGE="${ZERO_STAGE:-2}"
MIN_DELTA="${MIN_DELTA:-0.0}"
SCORE_BATCH_TOKENS="${SCORE_BATCH_TOKENS:-24576}"
SCORE_MAX_BATCH="${SCORE_MAX_BATCH:-48}"
SCORE_LIMIT="${SCORE_LIMIT:-0}"
DROP_INCOMPLETE="${DROP_INCOMPLETE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
FORCE_RESELECT="${FORCE_RESELECT:-0}"

# Stage-1 init and the frozen scorer share one directory so only one 6.6 GB
# copy of each lands on disk.
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoint}"
# Produced by Stage 1 (sft_Qwen3_enrich.sh).
BASE_MODEL="${BASE_MODEL:-${CKPT_DIR}/${CATEGORY}/stage1/final_checkpoint}"
# The scorer is the model that actually learned to consume these traces, so its
# likelihood gap is the meaningful accept/reject signal.
SCORER_MODEL="${SCORER_MODEL:-${CKPT_DIR}/${CATEGORY}/stage2_scorer/final_checkpoint}"

RUN_NAME="${RUN_NAME:-${CATEGORY}_stage2_bestofn_cot_v6_K5_Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoint/${CATEGORY}/stage2}"
LOG_DIR="${LOG_DIR:-stage2_bon_sft/logs}"
DELTA_DIR="${DELTA_DIR:-stage2_bon_sft/deltas}"
DATA_DIR="${DATA_DIR:-stage2_bon_sft/data}"
DELTA_FILE="${DELTA_FILE:-${DELTA_DIR}/${CATEGORY}_cot_v6_K5_delta.parquet}"
SELECTED_JSONL="${SELECTED_JSONL:-${DATA_DIR}/${CATEGORY}_cot_v6_K5_bestofn.jsonl}"
# Deltas depend only on (scorer, cot_dirs), never on the selection rule, so
# an already-computed parquet for this category can be reused as-is.
SHARED_DELTA_FILE="${SHARED_DELTA_FILE:-stage2_bon_sft/deltas/${CATEGORY}_cot_v6_K5_delta.parquet}"

# The N candidates. Order is irrelevant to best-of-N except as a tie-break.
if [[ -n "${COT_DIRS:-}" ]]; then
    read -r -a COT_DIR_LIST <<< "$COT_DIRS"
else
    COT_DIR_LIST=(
        "./data/${CATEGORY}/cot/sample_1"
        "./data/${CATEGORY}/cot/sample_2"
        "./data/${CATEGORY}/cot/sample_3"
        "./data/${CATEGORY}/cot/sample_4"
        "./data/${CATEGORY}/cot/sample_5"
    )
fi

for path in "$BASE_MODEL" "$SCORER_MODEL"; do
    if [[ ! -f "${path}/config.json" ]]; then
        echo "checkpoint is missing: ${path}" >&2
        echo "Stage 1:      bash stage1_sid_alignment/sft_Qwen3_enrich.sh ${CATEGORY}" >&2
        echo "Stage-2 scorer: CATEGORY=${CATEGORY} bash stage2_bon_sft/sft_reasoning_activation.sh" >&2
        exit 1
    fi
done
if [[ -z "${WANDB_API_KEY:-}" && "$SKIP_TRAIN" != "1" ]]; then
    echo "WANDB_API_KEY must be set for online logging." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$DELTA_DIR" "$DATA_DIR"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=online
export WANDB_DIR="$REPO_ROOT/stage2_bon_sft/wandb"
mkdir -p "$WANDB_DIR"
unset NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_NET_GDR_LEVEL

echo "gpus=${GPUS_PER_NODE}"
echo "base_model (train from) = ${BASE_MODEL}"
echo "scorer_model (delta)    = ${SCORER_MODEL}"
echo "cot_dirs (N=${#COT_DIR_LIST[@]}): ${COT_DIR_LIST[*]}"
echo "delta_file=${DELTA_FILE} min_delta=${MIN_DELTA}"
echo "selected=${SELECTED_JSONL}"
echo "output=${OUTPUT_DIR} global_batch=$((MICRO_BATCH_SIZE * GPUS_PER_NODE))"

# Stray workers keep holding GPU memory and would block the next stage. Match on
# process *name* plus cmdline so unrelated processes are never hit.
reap() {
    local script="$1"
    local pids=""
    local p
    for p in $(ps -eo pid=,comm= | awk '$2 ~ /^(python|pt_main_thread)$/ {print $1}'); do
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "$script"; then
            pids="$pids $p"
        fi
    done
    [[ -z "$pids" ]] && return 0
    kill -TERM $pids 2>/dev/null || true
    sleep 10
    kill -KILL $pids 2>/dev/null || true
    return 0
}

# -------------------------------- scoring -----------------------------------
if [[ "$FORCE_RESCORE" != "1" && ! -f "$DELTA_FILE" && -f "$SHARED_DELTA_FILE" ]]; then
    echo "[score] linking deltas from the first-accept run: ${SHARED_DELTA_FILE}"
    ln -sf "$REPO_ROOT/$SHARED_DELTA_FILE" "$DELTA_FILE"
    [[ -f "${SHARED_DELTA_FILE}.summary.json" ]] &&
        ln -sf "$REPO_ROOT/${SHARED_DELTA_FILE}.summary.json" "${DELTA_FILE}.summary.json"
fi

if [[ "$FORCE_RESCORE" != "1" && -e "$DELTA_FILE" ]]; then
    echo "[score] reusing existing ${DELTA_FILE} (FORCE_RESCORE=1 to redo)"
else
    SCORE_ARGS=(
        --scorer_model "$SCORER_MODEL"
        --category "$CATEGORY"
        --cot_dirs "${COT_DIR_LIST[@]}"
        --output "$DELTA_FILE"
        --cutoff_len "$CUTOFF_LEN"
        --batch_tokens "$SCORE_BATCH_TOKENS"
        --max_batch_size "$SCORE_MAX_BATCH"
        --dtype bf16
        --seed 42
    )
    [[ "$DROP_INCOMPLETE" == "1" ]] && SCORE_ARGS+=(--drop_incomplete_cot)
    [[ "$SCORE_LIMIT" != "0" ]] && SCORE_ARGS+=(--limit "$SCORE_LIMIT")

    echo "[score] delta = log p(y*|h,z) - log p(y*|h)"
    score_rc=0
    torchrun \
        --standalone \
        --nproc_per_node "$GPUS_PER_NODE" \
        --master_port "$SCORE_PORT" \
        stage2_bon_sft/score_cot_delta.py \
        "${SCORE_ARGS[@]}" > "${LOG_DIR}/score.${RUN_NAME}.log" 2>&1 || score_rc=$?
    reap "score_cot_delta.py"
    if [[ "$score_rc" -ne 0 ]]; then
        echo "SCORING_FAILED rc=${score_rc} log=${LOG_DIR}/score.${RUN_NAME}.log" >&2
        exit "$score_rc"
    fi
fi

if [[ -f "${DELTA_FILE}.summary.json" ]]; then
    echo "[score] summary:"
    sed 's/^/    /' "${DELTA_FILE}.summary.json"
fi

# ------------------------------- selection ----------------------------------
if [[ "$FORCE_RESELECT" != "1" && -f "$SELECTED_JSONL" ]]; then
    echo "[select] reusing existing ${SELECTED_JSONL} (FORCE_RESELECT=1 to redo)"
else
    SELECT_ARGS=(
        --delta_file "$DELTA_FILE"
        --cot_dirs "${COT_DIR_LIST[@]}"
        --output "$SELECTED_JSONL"
        --min_delta "$MIN_DELTA"
    )
    [[ "$DROP_INCOMPLETE" == "1" ]] && SELECT_ARGS+=(--drop_incomplete_cot)
    # score_cot_delta.py only scored the first N rows, so the selector has to
    # slice identically or the completeness check trips on unscored candidates.
    [[ "$SCORE_LIMIT" != "0" ]] && SELECT_ARGS+=(--limit "$SCORE_LIMIT")

    echo "[select] argmax delta per row, kept only when > ${MIN_DELTA}"
    select_rc=0
    python stage2_bon_sft/select_cot_best_of_n.py "${SELECT_ARGS[@]}" \
        > "${LOG_DIR}/select.${RUN_NAME}.log" 2>&1 || select_rc=$?
    if [[ "$select_rc" -ne 0 ]]; then
        echo "SELECTION_FAILED rc=${select_rc} log=${LOG_DIR}/select.${RUN_NAME}.log" >&2
        exit "$select_rc"
    fi
fi

if [[ -f "${SELECTED_JSONL}.stats.json" ]]; then
    echo "[select] stats:"
    sed 's/^/    /' "${SELECTED_JSONL}.stats.json"
fi

if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "SELECTION_ONLY_DONE selected=${SELECTED_JSONL}"
    exit 0
fi

# ------------------------------- training -----------------------------------
# train_stage2.py reads the traces from COT_REASONING_JSONL, so best-of-N needs
# no new trainer: it is the stock Stage-2 CoT SFT on a filtered, one-trace-per-
# row dataset.
export COT_REASONING_JSONL="$REPO_ROOT/$SELECTED_JSONL"
export COT_REASONING_CATEGORY="$CATEGORY"

# deepspeed defaults to /job/hostfile (multi-host here), which would fan the job
# across the cluster. A per-run single-host hostfile plus --no_ssh pins every
# rank to this node and skips ssh entirely.
RUN_HOSTFILE="stage2_bon_sft/.hostfile.${RUN_NAME}"
printf 'localhost slots=%s\n' "$GPUS_PER_NODE" > "$RUN_HOSTFILE"

echo "[train] starting ${RUN_NAME}"
train_rc=0
deepspeed \
    --hostfile "$RUN_HOSTFILE" \
    --no_ssh \
    --node_rank 0 \
    --num_nodes 1 \
    --num_gpus "$GPUS_PER_NODE" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    stage2_bon_sft/train_stage2.py \
    --base_model "$BASE_MODEL" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --cutoff_len "$CUTOFF_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT:-EvoRec_Stage2_BestOfN}" \
    --wandb_run_name "$RUN_NAME" \
    --category "$CATEGORY" \
    --seed 42 \
    --zero_stage "$ZERO_STAGE" \
    --dtype bf16 \
    --deepspeed > "${LOG_DIR}/${RUN_NAME}.log" 2>&1 || train_rc=$?

reap "train_stage2.py"

if [[ "$train_rc" -ne 0 ]]; then
    echo "TRAINING_FAILED rc=${train_rc} log=${LOG_DIR}/${RUN_NAME}.log" >&2
    exit "$train_rc"
fi

if [[ ! -f "${OUTPUT_DIR}/final_checkpoint/config.json" ]]; then
    echo "FATAL: no final checkpoint at ${OUTPUT_DIR}/final_checkpoint" >&2
    exit 1
fi

echo "TRAINING_DONE output=${OUTPUT_DIR}/final_checkpoint"

# ------------------------------ evaluation ----------------------------------
if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "SKIP_EVAL=1, stopping before evaluation"
    exit 0
fi

echo "[eval] thinking-mode constrained beam search on the test split"
eval_rc=0
CATEGORY="$CATEGORY" \
    MODEL="$REPO_ROOT/${OUTPUT_DIR}/final_checkpoint" \
    OUT_TAG="$RUN_NAME" \
    bash "$REPO_ROOT/evaluation/run_eval.sh" \
    > "${LOG_DIR}/eval.${RUN_NAME}.log" 2>&1 || eval_rc=$?

if [[ "$eval_rc" -ne 0 ]]; then
    echo "EVAL_FAILED rc=${eval_rc} log=${LOG_DIR}/eval.${RUN_NAME}.log" >&2
    exit "$eval_rc"
fi

echo "[eval] metrics:"
grep -E "R@|N@|NDCG|HR" "${LOG_DIR}/eval.${RUN_NAME}.log" | sed 's/^/    /' || true
echo "EVAL_DONE results=results/${RUN_NAME}/final_result_thinking_${CATEGORY}.json"
