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
KERNEL=${NCU_KERNEL:-regex:^(?!(spin|empty|cupy_fill|census)$).*}
COUNT=${NCU_LAUNCH_COUNT:-1}
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'NCU_LAUNCH_COUNT must be positive' >&2; exit 2; }
ncu --version | tee "$OUT/version.txt"
printf 'Kernel filter: %s\nLaunch limit: %s\n' "$KERNEL" "$COUNT" | tee "$OUT/settings.txt"
echo 'COMPUTE PROFILE: kernel replay changes timing; use just run for benchmarks.'
set +e
ncu --kernel-name "$KERNEL" --launch-count "$COUNT" --clock-control none \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
    --check-exit-code yes --force-overwrite --export "$OUT/metrics" --page details \
    "$@" 2>&1 | tee "$OUT/summary.txt"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$OUT/capture-exit-code.txt"
if grep -q 'ERR_NVGPUCTRPERM' "$OUT/summary.txt"; then
    echo 'GPU performance-counter access is blocked by the host/provider. Ask the provider to enable access or use a permitted VM. No security policy was changed.' >&2
    [[ "$STATUS" -ne 0 ]] || STATUS=1
fi
if [[ "$STATUS" -ne 0 ]]; then exit "$STATUS"; fi
[[ -s "$OUT/metrics.ncu-rep" ]] || { echo 'No compute report produced. Check the kernel filter and provider permissions.' >&2; exit 1; }
if ! ncu --import "$OUT/metrics.ncu-rep" --page raw --csv > "$OUT/metrics.csv" 2> "$OUT/import-errors.txt"; then
    cat "$OUT/import-errors.txt" >&2
    exit 1
fi
if ! grep -Eq '(sm__|smsp__|dram__|gpu__)' "$OUT/metrics.csv"; then
    echo 'No hardware metrics confirmed in the report. Check permissions and the kernel filter.' >&2
    exit 1
fi
echo 'Compute metrics describe selected kernel launches, not whole-program GPU utilization.'
