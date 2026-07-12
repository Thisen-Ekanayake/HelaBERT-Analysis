#!/usr/bin/env bash
# Launch every inference script in inference/helabert_large, each in its own
# detached tmux session (8 sessions total). Run from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    VENV_ACTIVATE="$REPO_ROOT/venv/bin/activate"
else
    echo "Error: no venv/.venv found under $REPO_ROOT" >&2
    exit 1
fi
SCRIPT_DIR="$REPO_ROOT/inference/helabert_large"
LOG_DIR="$REPO_ROOT/logs/inference/large"
mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed." >&2
    exit 1
fi

for script_path in "$SCRIPT_DIR"/*.py; do
    [ -e "$script_path" ] || continue
    script_name="$(basename "$script_path" .py)"
    session_name="infer_large_${script_name}"
    log_file="$LOG_DIR/${script_name}.log"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Session '$session_name' already exists, skipping."
        continue
    fi

    cmd="cd '$REPO_ROOT' && source '$VENV_ACTIVATE' && python '$script_path' 2>&1 | tee '$log_file'; exec bash"
    tmux new-session -d -s "$session_name" "$cmd"
    echo "Started session '$session_name' -> $script_path (log: $log_file)"
done

echo
echo "All large-model inference sessions launched. List them with: tmux ls"
echo "Attach to one with: tmux attach -t <session_name>"
