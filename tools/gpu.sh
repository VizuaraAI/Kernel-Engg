#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fail() { echo "Error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || fail "Missing $1. See tools/GPU.md for installation."; }
need jl
need jq
NAME=${JARVIS_INSTANCE_NAME:-}
USER_NAME=${JARVIS_SSH_USER:-ubuntu}
IDENTITY=${JARVIS_SSH_KEY:-$HOME/.ssh/id_ed25519}
if [[ -f "$ROOT/.gpu.json" ]]; then
    NAME=${JARVIS_INSTANCE_NAME:-$(jq -r '.name // empty' "$ROOT/.gpu.json")}
    USER_NAME=${JARVIS_SSH_USER:-$(jq -r '.user // "ubuntu"' "$ROOT/.gpu.json")}
    IDENTITY=${JARVIS_SSH_KEY:-$(jq -r '.identity // "~/.ssh/id_ed25519"' "$ROOT/.gpu.json")}
fi
IDENTITY=${IDENTITY/#\~/$HOME}
[[ "$USER_NAME" =~ ^[a-z_][a-z0-9_-]*$ ]] || fail 'Invalid SSH username'
ACTION=${1:-status}
shift || true
if [[ "$ACTION" == setup ]]; then
    jl setup
    echo 'Setup complete. Run: just status'
    exit 0
fi
[[ -n "$NAME" ]] || fail 'Configure your instance with just setup jarvis or set JARVIS_INSTANCE_NAME.'
case "$ACTION" in status|resume|pause|run) ;; *) fail "Unknown command: $ACTION";; esac
if [[ "$ACTION" == run ]]; then
    [[ $# -ge 1 ]] || fail 'Usage: just run path/to/kernel.cu [arguments...]'
    SOURCE=$1
    shift
    [[ ${1:-} != -- ]] || shift
    [[ -f "$SOURCE" ]] || fail "Source does not exist: $SOURCE"
    SOURCE_DIR=$(cd "$(dirname "$SOURCE")" && pwd)
    FILE=$(basename "$SOURCE")
    # Safe filenames also work with older rsync's remote-shell transport.
    [[ "$FILE" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*\.(cu|py)$ ]] || fail 'Use a .cu or .py filename containing letters, numbers, dots, underscores or hyphens'
fi
load_instance() {
INSTANCES=$(jl list --json) || fail 'Jarvis lookup failed. Run just setup to authenticate.'
INSTANCE=$(jq -ce --arg name "$NAME" '
    if type != "array" then error("Expected an instance list") else . end |
    map(select(.name == $name)) |
    if length != 1 then error("Instance name must match exactly one instance") else .[0] end
' <<< "$INSTANCES") || fail "Cannot uniquely find instance: $NAME"
ID=$(jq -er '.machine_id' <<< "$INSTANCE")
STATE=$(jq -er '.status' <<< "$INSTANCE")
[[ "$ID" =~ ^[0-9]+$ ]] || fail 'Invalid instance ID'
}
load_instance
case "$ACTION" in
    status) jq '{name, machine_id, status, public_ip}' <<< "$INSTANCE"; exit 0;;
    resume) jl resume "$ID"; load_instance;;
    pause) jl pause "$ID"; exit 0;;
esac
need ssh
need rsync
if [[ "$ACTION" == run ]]; then
    case "$STATE" in
        Paused|Running|Resuming|Creating) ;;
        *) fail "Cannot run on instance in state $STATE";;
    esac
    # Re-resolve by name on exit: resume may assign a different machine ID.
    # Preserve a program failure, but report cleanup failure after a successful run.
    pause_after_run() {
        local run_status=$?
        trap - EXIT INT TERM
        echo "Pausing $NAME..."
        local pause_status=0
        (
            load_instance
            case "$STATE" in
                Paused|Pausing) echo "Instance is already $STATE.";;
                *) jl pause "$ID" --yes;;
            esac
        ) || pause_status=$?
        if [[ "$pause_status" -ne 0 ]]; then
            echo "WARNING: Automatic pause failed. Check just status and pause the instance to stop compute billing." >&2
            if [[ "$run_status" -eq 0 ]]; then run_status=$pause_status; fi
        fi
        exit "$run_status"
    }
    trap pause_after_run EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [[ "$STATE" == Paused ]]; then
        echo "Resuming $NAME..."
        jl resume "$ID" --yes
    fi
fi
# Jarvis may rotate server keys on resume. User-approved exception, scoped to
# these subprocesses only; do not read/write the user's known_hosts files.
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1
    -o ServerAliveInterval=5 -o ServerAliveCountMax=2
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o GlobalKnownHostsFile=/dev/null -o VerifyHostKeyDNS=no
    -o UpdateHostKeys=no -i "$IDENTITY")
WAIT_TIMEOUT=${JARVIS_READY_TIMEOUT:-300}
[[ "$WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail 'JARVIS_READY_TIMEOUT must be positive seconds'
DEADLINE=$((SECONDS + WAIT_TIMEOUT))
PROBE=$(cat <<'REMOTE_PROBE'
command -v timeout >/dev/null || exit 76
[ -x /usr/local/cuda/bin/nvcc ] || exit 76
probe=$(mktemp -d) || exit 76
trap 'rm -rf "$probe"' EXIT
cat > "$probe/check.cu" <<'CUDA'
#include <cuda_runtime.h>
__global__ void check() {}
int main() {
    check<<<1,1>>>();
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 75;
}
CUDA
timeout 30 /usr/local/cuda/bin/nvcc "$probe/check.cu" -o "$probe/check" || exit 75
timeout 10 "$probe/check" || exit 75
REMOTE_PROBE
)
echo "Waiting for $NAME: Jarvis status, SSH, and CUDA kernel execution..."
while :; do
    load_instance
    case "$STATE" in
        Running|Resuming|Creating) ;;
        *) fail "Instance is $STATE. Resume paused instances with just resume.";;
    esac
    if [[ "$STATE" == Running ]]; then
        IP=$(jq -r '.public_ip // empty' <<< "$INSTANCE")
        if [[ -n "$IP" ]]; then
            [[ "$IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'Expected a VM IPv4 address'
            TARGET="$USER_NAME@$IP"
            ERROR_FILE=$(mktemp)
            if ssh "${SSH_OPTIONS[@]}" "$TARGET" "$PROBE" 2>"$ERROR_FILE"; then
                cat "$ERROR_FILE" >&2
                rm -f "$ERROR_FILE"
                echo "Ready: SSH and CUDA kernel execution passed ($IP)."
                break
            else
                PROBE_STATUS=$?
            fi
            ERROR_TEXT=$(cat "$ERROR_FILE")
            rm -f "$ERROR_FILE"
            if [[ "$PROBE_STATUS" == 76 ]]; then
                fail 'Remote readiness check requires timeout and /usr/local/cuda/bin/nvcc.'
            fi
            if [[ "$PROBE_STATUS" != 75 ]] && ! [[ "$PROBE_STATUS" == 255 && "$ERROR_TEXT" =~ (Connection\ refused|Connection\ timed\ out|Operation\ timed\ out|No\ route\ to\ host|Connection\ reset|Connection\ closed) ]]; then
                fail "Readiness check failed (exit $PROBE_STATUS): $ERROR_TEXT"
            fi
            echo "Not ready yet: SSH connection or CUDA execution unavailable."
        fi
    else
        echo "Jarvis status: $STATE"
    fi
    (( SECONDS < DEADLINE )) || fail "Readiness timed out after ${WAIT_TIMEOUT}s. Last status: $STATE; last probe: ${ERROR_TEXT:-not attempted}"
    sleep 3
done
[[ "$ACTION" != resume ]] || exit 0
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
REMOTE="/home/$USER_NAME/cuda-workspace/runs/$RUN_ID"
RESULTS="$ROOT/results/$RUN_ID"
mkdir -p "$RESULTS"
printf 'Instance: %s\nID: %s\nIP: %s\nSource: %s/%s\nRemote: %s\n' "$NAME" "$ID" "$IP" "$SOURCE_DIR" "$FILE" "$REMOTE" | tee "$RESULTS/metadata.txt"
printf 'Profiled: %s\n' "${GPU_PROFILE:-0}" >> "$RESULTS/metadata.txt"
ssh "${SSH_OPTIONS[@]}" "$TARGET" "mkdir -p '$REMOTE/src' '$REMOTE/results'"
if [[ ${GPU_PROFILE:-0} == 1 ]]; then
    cat "$ROOT/tools/profile.sh" | ssh "${SSH_OPTIONS[@]}" "$TARGET" "cat > '$REMOTE/profile.sh'"
fi
printf -v SSH_TRANSPORT '%q ' ssh "${SSH_OPTIONS[@]}"
rsync -az -e "$SSH_TRANSPORT" --exclude .git/ --exclude .venv/ --exclude __pycache__/ \
    --exclude results/ --exclude /bin/ --exclude /lib/ \
    "$SOURCE_DIR/" "$TARGET:$REMOTE/src/"
# Build remote command with quoted arguments, never interpolate program arguments as shell code.
COMMAND="cd '$REMOTE/src' && export GPU_RESULTS_DIR='$REMOTE/results' && "
PROFILE_PREFIX=''
if [[ ${GPU_PROFILE:-0} == 1 ]]; then PROFILE_PREFIX="bash '$REMOTE/profile.sh' "; fi
if [[ "$FILE" == *.cu ]]; then
    COMMAND+="/usr/local/cuda/bin/nvcc -O3 -lineinfo './$FILE' -o '$REMOTE/program' && "
    printf -v EXECUTION '%q ' "$REMOTE/program" "$@"
else
    PYTHON=${JARVIS_REMOTE_PYTHON:-/home/$USER_NAME/cuda-workspace/.venv/bin/python}
    COMMAND+='export CUDA_HOME=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib64 && '
    printf -v EXECUTION '%q ' "$PYTHON" "./$FILE" "$@"
fi
set +e
ssh "${SSH_OPTIONS[@]}" "$TARGET" "$COMMAND$PROFILE_PREFIX$EXECUTION" 2>&1 | tee "$RESULTS/run.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RESULTS/exit-code.txt"
# Retrieve artifacts even after compile/runtime failure without masking the program's exit code.
FETCH_STATUS=0
rsync -az -e "$SSH_TRANSPORT" "$TARGET:$REMOTE/results/" "$RESULTS/artifacts/" || FETCH_STATUS=$?
echo "Results: $RESULTS"
if [[ "$STATUS" -ne 0 ]]; then exit "$STATUS"; fi
exit "$FETCH_STATUS"
