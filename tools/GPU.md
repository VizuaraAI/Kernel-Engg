# Run kernels on JarvisLabs with just

## One-time setup on your Mac

```bash
brew install just jq rsync
uv tool install jarvislabs
just setup
ssh-add ~/.ssh/id_ed25519
```

Install `uv` first if needed: https://docs.astral.sh/uv/getting-started/installation/.
Enter your Jarvis API token only in `jl setup`'s interactive prompt, not in Git or
chat. Tokens are available at https://jarvislabs.ai/settings/api-keys.
The official `jl` CLI manages its own Python runtime through uv; our runner is Bash
and does not require activating a local Python environment.

The default instance is `running-first-kernel-code`, user `ubuntu`, private key
`~/.ssh/id_ed25519`. Keep the instance name unique and unchanged across resumes.
Existing ignored `.gpu.json` settings are still supported. Environment variables
`JARVIS_INSTANCE_NAME`, `JARVIS_SSH_USER`, and `JARVIS_SSH_KEY` override them.

## Everyday use

```bash
just status
just resume                         # optional: manually start compute billing
just run lecture4/vecadd.cu
just run lecture4/maxpool2d.py
just run lecture4/vecadd.cu -- 4096   # forward arguments to the program
just pause                          # optional: manually pause
```

Run from this checkout. `just` also finds the justfile from subdirectories; file
arguments are relative to the repository root. Absolute source paths work too,
including sources in your main checkout:

```bash
just run /Users/utkarshparekh/Kernel-Engg/lecture4/vecadd.cu
```

Every run:

1. Fetches the current instance ID and IP by exact name with `jl list --json`.
2. Resumes a paused instance automatically, then waits through Creating/Resuming and temporary SSH connection failures, then
   compiles and executes a tiny CUDA kernel before uploading. `just resume`
   performs the same check after Jarvis reports Running. Failed instances and
   authentication failures stop execution immediately. `just run` starts compute
   billing when it resumes an instance.
3. Syncs the source file's directory to a fresh remote run directory with rsync.
4. Compiles `.cu` with `/usr/local/cuda/bin/nvcc -O3` and executes only on success.
   `.py` uses `/home/ubuntu/cuda-workspace/.venv/bin/python`, the environment set up
   earlier. Override with `JARVIS_REMOTE_PYTHON` for another remote interpreter.
5. Streams combined stdout/stderr and saves `results/<run-id>/run.log`, metadata,
   and `exit-code.txt`. Artifacts written to the remote `$GPU_RESULTS_DIR` are
   downloaded to `results/<run-id>/artifacts/`, even after program failure.

The Bash helper preserves the remote program's failure code (also saved in
`exit-code.txt`); `just` reports recipe failures with its own nonzero exit code.
If the program
succeeds but artifact transfer fails, the command returns the transfer failure.
Upload/connection failures stop before execution. Remote run directories are
retained for inspection; local results are ignored by Git.

After results are collected, `just run` automatically requests pause, even after
compilation, readiness, transfer or execution failure. Ctrl+C/SIGTERM also trigger
cleanup. The current machine ID is looked up again before pausing. An instance
that was already running is also paused after the run; avoid concurrent runs or
other work on the same instance. Pause failures are reported prominently and make
a successful run return nonzero. A failing program retains its original failure.
Force-killing the runner, closing the host, or losing API access can prevent
cleanup; check the dashboard if the command is interrupted unexpectedly. Jarvis
pause acceptance is asynchronous; the dashboard may briefly show Pausing.

## Scope

Readiness checks retry every three seconds and stop as soon as CUDA execution
succeeds. The default timeout is 300 seconds, configurable with
`JARVIS_READY_TIMEOUT=600 just run lecture4/vecadd.cu`. An in-flight probe can
extend that limit by its bounded SSH/compiler timeout. Remote GNU `timeout` is
required alongside nvcc. Readiness checking does not execute your user program;
the user program is launched once after the check succeeds.

Supports directly accessible Jarvis VMs with IPv4, SSH port 22, Bash and rsync
installed remotely. The remote CUDA toolkit and Python dependencies must already
be installed; new VMs need provisioning separately. Local source filenames must
use letters, numbers, dots, underscores or hyphens. Sibling headers are synced;
files outside the source directory are not. Virtual environments, Git metadata,
results, and common venv bin/lib directories are excluded from uploads.
Remote programs run with their source directory as the working directory. Write
exported results to `$GPU_RESULTS_DIR`; files elsewhere remain on the VM.

## SSH host keys

As explicitly approved for this workflow, the runner disables server host-key
verification for its SSH and rsync subprocesses. It ignores both user and global
known-host files, disables DNS host-key verification, and does not save server
keys. Rotated Jarvis keys therefore do not block resume or execution.

This exception does not modify normal SSH settings or existing known-host files.
It prevents this runner from detecting a server impersonating the Jarvis VM.
Client authentication still uses your private key. Unlock passphrase-protected
client keys in your Mac terminal with `ssh-add` when necessary.

References: https://docs.jarvislabs.ai/cli and https://just.systems/man/en/.
