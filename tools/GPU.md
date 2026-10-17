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
