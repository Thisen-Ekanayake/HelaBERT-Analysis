#!/usr/bin/env bash
# Launch the co-attention [c; c̃; T̃] (clsconcat) finetune scripts, each in its
# own detached tmux session (8 sessions total: 4 tasks x 2 model sizes).
# These write checkpoints to the retagged HelaBERT_*_coattention_*_clsconcat
# dirs; run scripts/run_inference_coattn_clsconcat.sh afterwards to evaluate.
#
# NOTE: each script is a separate session and starts training immediately, so
# all 8 contend for the GPU at once. On a single small GPU, run a subset by
# commenting out lines in SCRIPTS below (or launch, then attach and stagger).
set -euo pipefail

# Script lives in scripts/, so the repo root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    VENV_ACTIVATE="$REPO_ROOT/venv/bin/activate"
else
    echo "Error: no venv/.venv found under $REPO_ROOT" >&2
    exit 1
fi
LOG_DIR="$REPO_ROOT/logs/finetune/coattn_clsconcat"
mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed." >&2
    exit 1
fi

SCRIPTS=(
    "finetune/helabert_large/news_category_co_attn.py"
    "finetune/helabert_large/news_source_co_attn.py"
    "finetune/helabert_large/sentiment_co_attn.py"
    "finetune/helabert_large/writing_style_co_attn.py"
    "finetune/helabert_small/news_category_co_attn.py"
    "finetune/helabert_small/news_source_co_attn.py"
    "finetune/helabert_small/sentiment_co_attn.py"
    "finetune/helabert_small/writing_style_co_attn.py"
)

for rel_path in "${SCRIPTS[@]}"; do
    script_path="$REPO_ROOT/$rel_path"
    if [ ! -f "$script_path" ]; then
        echo "Warning: $script_path not found, skipping."
        continue
    fi

    size="$(basename "$(dirname "$rel_path")" | sed 's/helabert_//')"
    script_name="$(basename "$rel_path" .py)"
    session_name="finetune_${size}_${script_name}"
    log_file="$LOG_DIR/${size}_${script_name}.log"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Session '$session_name' already exists, skipping."
        continue
    fi

    cmd="cd '$REPO_ROOT' && source '$VENV_ACTIVATE' && python '$script_path' 2>&1 | tee '$log_file'; exec bash"
    tmux new-session -d -s "$session_name" "$cmd"
    echo "Started session '$session_name' -> $rel_path (log: $log_file)"
done

echo
echo "Co-attention (clsconcat) finetune sessions launched. List them with: tmux ls"
echo "Attach to one with: tmux attach -t <session_name>"
echo "When all have finished, evaluate with: scripts/run_inference_coattn_clsconcat.sh"
