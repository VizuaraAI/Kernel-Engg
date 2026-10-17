#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-status}; shift || true
CONFIG="$ROOT/.gpu.json"
command -v jq >/dev/null || { echo 'Install jq: brew install jq' >&2; exit 1; }
save_setting() {
    local key=$1 value=$2 tmp
    tmp=$(mktemp "$ROOT/.gpu.json.XXXXXX")
    if ! jq --arg key "$key" --arg value "$value" '.[$key]=$value' \
        "${CONFIG}" > "$tmp"; then rm -f "$tmp"; return 1; fi
    mv "$tmp" "$CONFIG"
}
BACKEND=${GPU_PROVIDER:-}
if [[ "$ACTION" == setup || "$ACTION" == select ]]; then BACKEND=${1:-modal}; fi
if [[ -z "$BACKEND" ]]; then
    BACKEND=$(if [[ -f "$CONFIG" ]]; then jq -r '.provider // "modal"' "$CONFIG"; else echo modal; fi)
fi
case "$BACKEND" in modal|jarvis) ;; *) echo "Unknown provider: $BACKEND" >&2; exit 1;; esac
if [[ "$ACTION" == select ]]; then
    [[ -f "$CONFIG" ]] || printf '{}\n' > "$CONFIG"
    save_setting provider "$BACKEND"
    echo "Default provider: $BACKEND"
    exit 0
fi
if [[ "$ACTION" == setup ]]; then
    if [[ "$BACKEND" == modal ]]; then
        command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
        [[ -x "$ROOT/.venv/bin/python" ]] || uv venv "$ROOT/.venv"
        uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/tools/requirements-modal.txt"
        "$ROOT/.venv/bin/modal" setup
    else
        bash "$ROOT/tools/gpu.sh" setup
        read -r -p 'Exact Jarvis instance name: ' INSTANCE_NAME
        [[ -n "$INSTANCE_NAME" ]] || { echo 'Instance name required' >&2; exit 1; }
        [[ -f "$CONFIG" ]] || printf '{}\n' > "$CONFIG"
        save_setting name "$INSTANCE_NAME"
    fi
    [[ -f "$CONFIG" ]] || printf '{}\n' > "$CONFIG"
    save_setting provider "$BACKEND"
    exit 0
fi
if [[ "$ACTION" == profile ]]; then export GPU_PROFILE=1; ACTION=run; fi
if [[ "$BACKEND" == jarvis ]]; then exec bash "$ROOT/tools/gpu.sh" "$ACTION" "$@"; fi
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run just setup modal first.' >&2; exit 1; }
if [[ "$ACTION" == status ]]; then
    echo "Provider: Modal; GPU: ${MODAL_GPU:-L4}"
    exec "$ROOT/.venv/bin/modal" profile current
fi
[[ "$ACTION" == run ]] || { echo "Unsupported action: $ACTION" >&2; exit 1; }
exec "$ROOT/.venv/bin/python" "$ROOT/tools/modal_runner.py" "$@"
