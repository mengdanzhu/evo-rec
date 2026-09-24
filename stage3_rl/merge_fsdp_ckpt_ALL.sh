#!/bin/bash

# Configure via environment:
#   CATEGORY      -> domain to merge (default Video_Games)
#   CKPT_ROOT     -> experiment root containing global_step_xx folders
#   EVAL_INTERVAL -> merge checkpoints every N steps
#
# Uses verl v0.6.0: `verl` is a top-level package dir at the repo root, so
# `from verl.model_merger ...` (with PYTHONPATH=$REPO_ROOT) resolves to it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CATEGORY="${CATEGORY:-Video_Games}"
CKPT_ROOT="${CKPT_ROOT:-./checkpoint/${CATEGORY}/stage3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

if [ -z "$CKPT_ROOT" ]; then
    echo "❌ ERROR: Please provide the experiment root directory containing global_step_xx folders."
    exit 1
fi

if ! [[ "$EVAL_INTERVAL" =~ ^[0-9]+$ ]] || [ "$EVAL_INTERVAL" -le 0 ]; then
    echo "❌ ERROR: eval_interval must be a positive integer."
    exit 1
fi

CKPT_ROOT="${CKPT_ROOT%/}"

if [ ! -d "$CKPT_ROOT" ]; then
    echo "❌ ERROR: Cannot find directory $CKPT_ROOT"
    exit 1
fi

MERGE_PY="$SCRIPT_DIR/merge_fsdp_checkpoint.py"
if [ ! -f "$MERGE_PY" ]; then
    echo "❌ ERROR: Cannot find merge_fsdp_checkpoint.py."
    echo "Expected at: $MERGE_PY"
    exit 1
fi

echo "🔍 Searching checkpoints under: $CKPT_ROOT"
echo "⏱️  Eval interval: every $EVAL_INTERVAL steps"
echo "🔧 Using merge script: $MERGE_PY"
echo ""

matches=()
while IFS= read -r actor_dir; do
    step_dir="$(basename "$(dirname "$actor_dir")")"
    if [[ "$step_dir" =~ ^global_step_([0-9]+)$ ]]; then
        step="${BASH_REMATCH[1]}"
        if (( step % EVAL_INTERVAL == 0 )); then
            matches+=("$step:$actor_dir")
        fi
    fi
done < <(find "$CKPT_ROOT" -maxdepth 2 -type d -name "actor" -path "*/global_step_*/*")

if [ ${#matches[@]} -eq 0 ]; then
    echo "⚠️  No actor checkpoints found matching interval $EVAL_INTERVAL under $CKPT_ROOT"
    exit 1
fi

IFS=$'\n' read -r -d '' -a sorted_matches < <(printf "%s\n" "${matches[@]}" | sort -t: -k1,1n && printf '\0')

for entry in "${sorted_matches[@]}"; do
    step="${entry%%:*}"
    actor_dir="${entry#*:}"
    output_dir="${actor_dir}_merged"

    echo "🚀 Merging step ${step}: $actor_dir -> $output_dir"
    python3 "$MERGE_PY" --checkpoint "$actor_dir" --output-dir "$output_dir"
    echo ""
done

echo "✅ All merges completed."
