#!/usr/bin/env bash
# Runs remotely on Linux. Arguments are the program followed by its arguments.
set -euo pipefail
: "${GPU_RESULTS_DIR:?GPU_RESULTS_DIR must be set}"
mkdir -p "$GPU_RESULTS_DIR/profile"
if ! command -v nsys >/dev/null; then
    for executable in /opt/nvidia/nsight-systems/*/bin/nsys /opt/nvidia/nsight-systems/*/target-linux-x64/nsys; do
        if [[ -x "$executable" ]]; then export PATH="$(dirname "$executable"):$PATH"; break; fi
    done
fi
command -v nsys >/dev/null || {
    echo 'Nsight Systems CLI (nsys) is missing on this GPU host. Install it from NVIDIA, then retry just profile.' >&2
    exit 69
}
REPORT="$GPU_RESULTS_DIR/profile/trace"
nsys --version | tee "$GPU_RESULTS_DIR/profile/version.txt"
echo 'PROFILE RUN: instrumentation affects timings; use just run for ordinary benchmarks.'
# Disable CPU sampling/context switches so CUDA tracing does not require perf privileges.
set +e
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="$REPORT" "$@"
PROGRAM_STATUS=$?
set -e
printf '%s\n' "$PROGRAM_STATUS" > "$GPU_RESULTS_DIR/profile/capture-exit-code.txt"
if [[ ! -f "$REPORT.nsys-rep" ]]; then
    echo 'No Nsight report was produced. Check the log for profiler/permission errors.' >&2
    if [[ "$PROGRAM_STATUS" -ne 0 ]]; then exit "$PROGRAM_STATUS"; fi
    exit 1
fi
set +e
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format column \
    --force-export=true "$REPORT.nsys-rep" 2>&1 | tee "$GPU_RESULTS_DIR/profile/summary.txt"
SUMMARY_STATUS=${PIPESTATUS[0]}
# Check for actual kernel rows rather than claiming a successful CUDA capture from an empty report.
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true "$REPORT.nsys-rep" \
    > "$GPU_RESULTS_DIR/profile/kernels.csv" 2> "$GPU_RESULTS_DIR/profile/stats-errors.txt"
CSV_STATUS=$?
set -e
PROFILE_STATUS=0
if [[ "$SUMMARY_STATUS" -ne 0 || "$CSV_STATUS" -ne 0 ]] || ! grep -Eq '^"?[0-9]+([.,])' "$GPU_RESULTS_DIR/profile/kernels.csv"; then
    echo 'CUDA kernel trace could not be confirmed. Inspect profile/summary.txt and stats-errors.txt.' >&2
    PROFILE_STATUS=1
fi
printf '%s\n' "$PROFILE_STATUS" > "$GPU_RESULTS_DIR/profile/analysis-exit-code.txt"
echo 'Summary percentages are shares of captured kernel/memory-operation time, not overall GPU utilization.'
if [[ "$PROGRAM_STATUS" -ne 0 ]]; then exit "$PROGRAM_STATUS"; fi
exit "$PROFILE_STATUS"
