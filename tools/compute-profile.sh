#!/usr/bin/env bash
# Remote Nsight Compute capture; selected launches only, no device policy changes.
set -euo pipefail
: "${GPU_RESULTS_DIR:?GPU_RESULTS_DIR must be set}"
OUT="$GPU_RESULTS_DIR/compute-profile"
mkdir -p "$OUT"
if ! command -v ncu >/dev/null; then
    for executable in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/NsightCompute-*/ncu; do
        if [[ -x "$executable" ]]; then export PATH="$(dirname "$executable"):$PATH"; break; fi
    done
fi
command -v ncu >/dev/null || { echo 'Nsight Compute CLI (ncu) is missing on this GPU host. Install the matching NVIDIA package first.' >&2; exit 69; }
NCU=$(command -v ncu)
USE_SUDO=${GPU_NCU_SUDO:-0}
[[ "$USE_SUDO" == 0 || "$USE_SUDO" == 1 ]] || { echo 'GPU_NCU_SUDO must be 0 or 1' >&2; exit 2; }
COLLECTOR=("$NCU")
if [[ "$USE_SUDO" == 1 ]]; then
    # Jarvis support recommends root-only collection, without changing driver policy.
    [[ ! -x /usr/local/cuda/bin/ncu ]] || NCU=/usr/local/cuda/bin/ncu
    command -v sudo >/dev/null || { echo 'Jarvis compute profiling requires sudo.' >&2; exit 69; }
    if ! sudo -n -- "$NCU" --version >/dev/null; then
        echo 'Noninteractive sudo for Nsight Compute is unavailable. Ask your VM administrator to permit it; the runner cannot prompt for a password.' >&2
        exit 1
    fi
    COLLECTOR=(sudo -n -- env "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}"
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
        "GPU_RESULTS_DIR=$GPU_RESULTS_DIR" "$NCU")
    OWNER="$(id -u):$(id -g)"
    restore_report_owner() {
        local status=$?
        trap - EXIT
        if [[ -f "$OUT/metrics.ncu-rep" ]]; then
            if ! sudo -n -- chown -- "$OWNER" "$OUT/metrics.ncu-rep"; then
                echo 'Could not restore report ownership for download.' >&2
                [[ "$status" -ne 0 ]] || status=1
            fi
        fi
        exit "$status"
    }
    trap restore_report_owner EXIT
    echo 'Jarvis compute collection uses sudo; the profiled program also runs as root.'
fi
KERNEL=${NCU_KERNEL:-regex:^(?!(spin|empty|cupy_fill|census)$).*}
COUNT=${NCU_LAUNCH_COUNT:-1}
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'NCU_LAUNCH_COUNT must be positive' >&2; exit 2; }
"$NCU" --version | tee "$OUT/version.txt"
printf 'Kernel filter: %s\nLaunch limit: %s\n' "$KERNEL" "$COUNT" | tee "$OUT/settings.txt"
echo 'COMPUTE PROFILE: kernel replay changes timing; use just run for benchmarks.'
set +e
"${COLLECTOR[@]}" --kernel-name "$KERNEL" --launch-count "$COUNT" --clock-control none \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
    --check-exit-code yes --force-overwrite --export "$OUT/metrics" --page details \
    "$@" 2>&1 | tee "$OUT/summary.txt"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$OUT/capture-exit-code.txt"
if grep -q 'ERR_NVGPUCTRPERM' "$OUT/summary.txt"; then
    echo 'GPU performance-counter access is restricted for this execution identity. Jarvis requires sudo collection; consult provider guidance if access still fails. No driver policy was changed.' >&2
    [[ "$STATUS" -ne 0 ]] || STATUS=1
fi
if [[ "$STATUS" -ne 0 ]]; then exit "$STATUS"; fi
[[ -s "$OUT/metrics.ncu-rep" ]] || { echo 'No compute report produced. Check the kernel filter and provider permissions.' >&2; exit 1; }
if [[ "$USE_SUDO" == 1 ]]; then
    sudo -n -- chown -- "$OWNER" "$OUT/metrics.ncu-rep"
fi
if ! "$NCU" --import "$OUT/metrics.ncu-rep" --page raw --csv > "$OUT/metrics.csv" 2> "$OUT/import-errors.txt"; then
    cat "$OUT/import-errors.txt" >&2
    exit 1
fi
if ! grep -Eq '(sm__|smsp__|dram__|gpu__)' "$OUT/metrics.csv"; then
    echo 'No hardware metrics confirmed in the report. Check permissions and the kernel filter.' >&2
    exit 1
fi
echo 'Compute metrics describe selected kernel launches, not whole-program GPU utilization.'
