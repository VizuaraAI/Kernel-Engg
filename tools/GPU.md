# Run GPU kernels with just: Modal or JarvisLabs

Modal is the default provider, with one L4 GPU. Kernel files do not need Modal
annotations: the provider adapter runs standalone CUDA/C++ (`.cu`) and Python
(`.py`) programs. Python programs that already orchestrate Modal apps themselves
(such as `lecture3/dog.py`) should still use their original `modal run` workflow.

## One-time setup on your Mac

Install just, jq and uv, then authenticate your chosen provider:

```bash
brew install just jq uv
just setup modal
```

Modal setup creates the project's `.venv`, installs `tools/requirements-modal.txt`
there, and opens Modal's login flow. No global pip installation or shell activation
is needed. Container dependencies are separately installed in the Modal image.
Only the provider being used needs its SDK/CLI installed.

For Jarvis, install rsync and the official CLI into an isolated tool environment:

```bash
brew install rsync
uv tool install jarvislabs
just setup jarvis
ssh-add ~/.ssh/id_ed25519
```

Jarvis setup prompts for your API token through `jl setup`, then your exact unique
instance name. Do not share tokens or commit them. Local provider and instance
settings are stored in ignored `.gpu.json`; existing Jarvis settings are preserved.

## Run and switch providers

From this checkout:

```bash
just provider modal                 # save default provider locally
just run lecture4/vecadd.cu
just run lecture4/maxpool2d.py
just run lecture4/vecadd.cu -- 4096   # forward program arguments

just provider=jarvis run lecture4/vecadd.cu  # override for one run
just provider=modal gpu=L4 run lecture4/vecadd.cu
just provider jarvis                # save Jarvis as default
just status
```

`GPU_PROVIDER` and `MODAL_GPU` environment variables also provide overrides.
The GPU override applies only to Modal; Jarvis uses its instance's existing GPU.
`just` finds the justfile from subdirectories; source paths are relative to the
repository root. Absolute source paths work too. Never run concurrent jobs on the
same Jarvis instance: each run pauses it afterward.

## Shared behavior

- Upload the source directory, including sibling headers; files outside it are
  not uploaded. Git metadata, virtual environments and results are excluded.
- Compile `.cu` with `nvcc -O3`; execute only after compilation succeeds.
- Execute `.py` with the provider's remote Python environment.
- Stream output and save logs, provider metadata and `exit-code.txt` under
  `results/<run-id>/`. `just` reports failure with its own nonzero status.
- Write exported files to the remote `$GPU_RESULTS_DIR` to download them into
  `results/<run-id>/artifacts/`. Files elsewhere are not collected.
- Attempt compute cleanup on success, failure or Ctrl+C. Force-kill, power loss,
  or lost network/API access can prevent cleanup; check the provider dashboard.

Provider filesystem differences: Jarvis runs live on its persistent VM;
Modal runs in a new temporary container and only collected results survive.

## Modal

Uses a CUDA 13.0 development image, Python 3.12, CuPy 14.2.0, NumPy 2.3.5 and
Numba CUDA 0.30.4. Images are cached; the first run builds the environment.
The client SDK is pinned separately in `requirements-modal.txt` and installed
only in `.venv`. A temporary GPU sandbox is created per execution and explicitly
terminated after artifacts are collected. It has a 300-second maximum lifetime,
a 300-second idle timeout and a 240-second execution timeout. No service is deployed.
The run log records the actual GPU name in addition to the requested GPU in
metadata. Different GPUs can produce different performance measurements.

Usage is billed to the Modal workspace used for login and may consume credits.
Timeouts are not a monetary spending cap; configure workspace budgets separately.
This runner does not automatically fall back to another provider or GPU.

## JarvisLabs

`just run` automatically resumes a paused instance, resolves its current ID/IP by
unique name, waits for SSH and a tiny CUDA kernel to work, syncs sources, runs,
collects output and requests pause. Even an already-running instance is paused
at the end. The current ID is resolved again before pausing. Pause API acceptance
is asynchronous; the dashboard may briefly display Pausing. Cleanup failure is
reported and makes a successful execution return nonzero.

`just resume` and `just pause` are explicit Jarvis controls regardless of the
selected default provider. For status while Modal is default, use
`just provider=jarvis status`.

Configuration overrides: `JARVIS_INSTANCE_NAME`, `JARVIS_SSH_USER` (default ubuntu),
`JARVIS_SSH_KEY` (default ~/.ssh/id_ed25519), `JARVIS_REMOTE_PYTHON` (default
/home/ubuntu/cuda-workspace/.venv/bin/python), and `JARVIS_READY_TIMEOUT` (default
300 seconds). Your instance name has no personal hardcoded default.

Requires a directly accessible IPv4 VM on SSH port 22, remote Bash, GNU timeout,
rsync, and `/usr/local/cuda/bin/nvcc`. Python examples need dependencies installed
in the remote environment; a new VM must be provisioned separately.

As approved for this workflow, host-key verification is disabled only for the
runner's SSH/rsync subprocesses. Normal SSH configuration and known_hosts files
are unchanged. This handles rotating Jarvis server keys, but cannot detect server
impersonation. Client authentication still uses your private key.

References:
- https://modal.com/docs/guide/sandboxes
- https://modal.com/docs/guide/cuda
- https://docs.jarvislabs.ai/cli
- https://just.systems/man/en/

## Profile from the terminal

Use Nsight Systems to capture kernel launches and memory transfers:

```bash
just timeline-profile lecture4/vecadd.cu
just provider=modal gpu=H100 timeline-profile lecture3/dog.py
just provider=jarvis timeline-profile lecture4/vecadd.cu
```

`dog.py` requires the standalone adaptation from the `dog-just-workflow` branch.
The command prints kernel and memory-transfer timing tables, downloads the
`.nsys-rep` report, and releases compute using the same lifecycle as `just run`.
Profiling changes execution timings; use `just run` for ordinary benchmarks.

Reports are saved in `results/<run-id>/artifacts/profile/`:

- `summary.txt`: terminal timing tables.
- `kernels.csv`: kernel names, counts and timing statistics.
- `trace.nsys-rep`: full report for later inspection.
- `version.txt`: profiler version used.

Modal builds a separate profiling image with NVIDIA's CUDA-13-matched Nsight
Systems package. Normal runs do not install Nsight. Jarvis requires `nsys` on the
VM; if missing, install the matching Nsight Systems package from NVIDIA before
retrying. On Ubuntu VMs with NVIDIA's CUDA apt repository configured, for example:

```bash
# Run on the Jarvis VM; match the package to its CUDA version.
sudo apt-get update
sudo apt-get install cuda-nsight-systems-13-0
nsys --version
```

CPU sampling and context-switch tracing are disabled. CUDA tracing can still be
restricted by the provider/driver; empty captures or profiler failures return a
nonzero status instead of claiming successful profiling. Jarvis timeline capture was verified on an RTX PRO 6000 VM with CUDA 13.2. Cleanup still runs after profiling failure.

The `Time (%)` column is each kernel's share of captured kernel time, not overall
GPU utilization. These reports do not calculate SM occupancy or sample GPU-wide
utilization. No local Nsight installation or GUI is needed for terminal summaries.


## Compute profiling from the terminal

`just timeline-profile` captures execution timelines with Nsight Systems.
`just compute-profile` collects per-kernel hardware metrics with Nsight Compute.
The old `just profile` recipe has been renamed; use the explicit commands.

```bash
just compute-profile lecture4/vecadd.cu
just kernel=rgb_to_gray launches=1 compute-profile lecture3/dog.py
just provider=jarvis kernel=rgb_to_gray compute-profile lecture3/dog.py
```

By default, profile one matching launch and exclude `spin`, `empty`, `cupy_fill`
and `census`. `kernel` accepts an exact function name or `regex:...`; `launches`
is a positive count of matching launches. Corresponding environment variables
are `NCU_KERNEL` and `NCU_LAUNCH_COUNT`. All arguments after the source path are
passed to your program. Warm-up launches are eligible, so choose deliberately
when interpreting measurements. The rest of the program still executes.

Collects SpeedOfLight, Occupancy and MemoryWorkloadAnalysis sections, including
compute/memory throughput and occupancy metrics. Results are downloaded to
`results/<run-id>/artifacts/compute-profile/`: `summary.txt`, `metrics.csv`,
`metrics.ncu-rep`, `version.txt` and `settings.txt`. Compute metrics describe the
selected launch, not program-wide GPU utilization. Kernel replay adds substantial
overhead; do not compare instrumented benchmark timings with ordinary runs.

Modal installs the CUDA-13-matched Nsight Compute package in a separate image.
Jarvis requires `ncu` on the VM (for a CUDA 13.0 apt installation, the package is
`cuda-nsight-compute-13-0`). Jarvis compute collection uses `sudo -n` with the full
`/usr/local/cuda/bin/ncu` path, as recommended by Jarvis support. The profiler and
its target program run as root; the helper explicitly forwards CUDA library
paths and `GPU_RESULTS_DIR`. Python still uses the configured remote venv's
interpreter. Report ownership is restored to the SSH user for download and
unprivileged report import. Normal runs, timelines and Modal do not use sudo.
Passwordless sudo is required; if unavailable, the command fails without waiting
for a password. No driver configuration, sudoers rules or system-wide counter
permissions are changed. If the host still returns `ERR_NVGPUCTRPERM`, the command fails with an explanation and releases
compute; it does not change driver permissions or silently fall back to timelines.
Empty captures and unmatched filters also return nonzero. Existing five-minute
sandbox limits and cleanup behavior apply to both profiling modes.


### Provider validation

- Modal L4: timeline and compute captures verified with vector addition.
- Jarvis RTX PRO 6000, CUDA 13.2: timeline capture verified with vector addition
  after installing `cuda-nsight-systems-13-2` on the VM.
- Jarvis RTX PRO 6000, CUDA 13.2: compute capture, hardware metrics, report download
  and automatic pause verified with vector addition using sudo collection.
  Earlier unprivileged captures returned `ERR_NVGPUCTRPERM`; sudo resolved it,
  consistent with Jarvis support's guidance. The sudo path has not yet been
  live-tested on the H100 VM or with Python workloads.


## Run all stages with analyze

```bash
just analyze lecture4/vecadd.cu
just provider=modal gpu=H100 kernel=rgb_to_gray launches=1 analyze lecture3/dog.py
```

Runs normal execution, timeline profiling, and compute profiling in that order.
Provider, GPU, source path, and program arguments apply to all three executions;
`kernel` and `launches` select the compute capture. The first stage clears any
inherited profiling flag. Use only that stage for ordinary benchmark timings.
Each stage saves its own result directory and performs normal provider cleanup.
The command stops at the first failure, preserving earlier results. On Jarvis,
this means three resume/pause cycles; compute requires noninteractive sudo for counter access.
No stage is retried or silently skipped. Use the individual commands when you
only need one profile to avoid extra execution time and cost.
