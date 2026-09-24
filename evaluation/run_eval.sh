#!/bin/bash

# Parametrized thinking-mode evaluation driver for Stage-2 / Stage-3 checkpoints.
# Pipeline: split.py -> evaluate_Qwen3_think.py [beam search, constrained SID
# decode] -> merge.py -> calc.py. The domain and model path are parametrized so
# the same script runs unchanged for every category.
#
# Usage:
#   CATEGORY=Video_Games MODEL=/abs/path/to/actor_merged \
#     bash evaluation/run_eval.sh
#
# Env vars:
#   CATEGORY   Video_Games | Office_Products | Industrial_and_Scientific
#   MODEL      path to a merged HF checkpoint dir (actor_merged)
#   CUDA_LIST  space-separated GPU ids (default "0 1 2 3 4 5 6 7")
#   OUT_TAG    label for results subdir (default basename of MODEL's grandparent)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CATEGORY="${CATEGORY:?set CATEGORY}"
MODEL="${MODEL:?set MODEL (path to merged HF checkpoint)}"
CUDA_LIST="${CUDA_LIST:-0 1 2 3 4 5 6 7}"
CUDA_LIST_CSV="$(echo "${CUDA_LIST}" | tr ' ' ',')"

# Local data directories, resolved by data_io.py against DATA_ROOT (./data).
TEST_FILE="./data/${CATEGORY}/seqrec/test/"
INFO_FILE="./data/${CATEGORY}/catalog/"
ITEM_FILE="./data/${CATEGORY}/catalog/"
INDEX_FILE="./data/${CATEGORY}/catalog/"

exp_name="${MODEL}"
dir1=$(basename "$(dirname "$exp_name")")
dir2=$(basename "$exp_name")
dir0=$(basename "$(dirname "$(dirname "$exp_name")")")
exp_name_clean="${OUT_TAG:-${dir0}__${dir1}__${dir2}}"

echo "Processing category: ${CATEGORY} with model: ${exp_name_clean} (THINKING MODE)"

temp_dir="./temp/${CATEGORY}-${exp_name_clean}"
mkdir -p "${temp_dir}"

echo "Splitting test data..."
python "$SCRIPT_DIR/split.py" --input_path "${TEST_FILE}" --output_path "${temp_dir}" --cuda_list "${CUDA_LIST_CSV}"

echo "Starting parallel evaluation (THINKING MODE)..."
for i in ${CUDA_LIST}; do
    if [[ -f "${temp_dir}/${i}.csv" ]]; then
        echo "Starting evaluation on GPU ${i} for category ${CATEGORY}"
        mkdir -p "${temp_dir}/.cache/gpu${i}"
        CUDA_VISIBLE_DEVICES=${i} \
        TRITON_CACHE_DIR="${temp_dir}/.cache/gpu${i}/triton" \
        TORCHINDUCTOR_CACHE_DIR="${temp_dir}/.cache/gpu${i}/inductor" \
        VLLM_CACHE_ROOT="${temp_dir}/.cache/gpu${i}/vllm" \
        python -u "$SCRIPT_DIR/evaluate_Qwen3_think.py" \
            --base_model "${exp_name}" \
            --info_file "${INFO_FILE}" \
            --category "${CATEGORY}" \
            --test_data_path "${temp_dir}/${i}.csv" \
            --item_file "${ITEM_FILE}" \
            --index_file "${INDEX_FILE}" \
            --result_json_data "${temp_dir}/${i}.json" \
            --batch_size 4 \
            --num_beams 10 \
            --max_new_tokens 1024 \
            --length_penalty 0.0 &
    else
        echo "Warning: Split file ${temp_dir}/${i}.csv not found, skipping GPU ${i}"
    fi
done
echo "Waiting for all evaluation processes to complete..."
wait

result_files=$(find "${temp_dir}" -maxdepth 1 -name '*.json' | wc -l)
if [[ ${result_files} -eq 0 ]]; then
    echo "Error: No result files generated for category ${CATEGORY}"
    exit 1
fi

output_dir="./results/${exp_name_clean}"
mkdir -p "${output_dir}"

actual_cuda_list=""
for gpu in ${CUDA_LIST}; do
    if [[ -f "${temp_dir}/${gpu}.json" ]]; then
        actual_cuda_list="${actual_cuda_list}${gpu},"
    fi
done
actual_cuda_list="${actual_cuda_list%,}"

echo "Merging results from GPUs: ${actual_cuda_list}"
python "$SCRIPT_DIR/merge.py" \
    --input_path "${temp_dir}" \
    --output_path "${output_dir}/final_result_thinking_${CATEGORY}.json" \
    --cuda_list "${actual_cuda_list}"

echo "Calculating metrics..."
python "$SCRIPT_DIR/calc.py" \
    --path "${output_dir}/final_result_thinking_${CATEGORY}.json" \
    --item_path "${INFO_FILE}"

echo "Result JSON: ${output_dir}/final_result_thinking_${CATEGORY}.json"
