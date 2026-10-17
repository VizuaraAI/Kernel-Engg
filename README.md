# Kernel Engineering

Exercises for the [Vizuara Kernel Engineering Workshop](https://vizuara.ai).
Edit code on your Mac and run it on a remote GPU with Modal or JarvisLabs.
No local GPU or CUDA installation is required.

## Install the utilities

Install [Homebrew](https://brew.sh/) if needed, then:

```bash
git clone https://github.com/utkarshparekh/Kernel-Engg.git
cd Kernel-Engg
brew install just jq uv
```

Already cloned? Run the remaining commands from your repository root.
Choose one provider below and use your own account.

## Option 1: Modal

1. Create a [Modal account](https://modal.com/) and enable GPU access.
2. Run setup and complete the browser login:

   ```bash
   just setup modal
   ```

   Setup installs the SDK in the project's `.venv` and selects Modal as default.
   No environment activation is needed.
3. Run a kernel:

   ```bash
   just run lecture4/vecadd.cu
   just run lecture4/maxpool2d.py
   ```

Each run uploads code to an L4 sandbox, compiles or executes it, downloads results,
and terminates the sandbox. The first run builds the cached CUDA environment.
Usage is charged to your Modal workspace, including any available credits.

## Option 2: JarvisLabs

1. Create an Ubuntu GPU **VM** in [JarvisLabs](https://jarvislabs.ai/) with a
   unique name. It needs `/usr/local/cuda/bin/nvcc`, `rsync`, and GNU `timeout`.
2. Register your Mac's SSH public key with JarvisLabs. If needed, create a key:

   ```bash
   ssh-keygen -t ed25519
   ```

   Register `~/.ssh/id_ed25519.pub`; keep the private key on your Mac.
3. Install the CLI and configure the runner:

   ```bash
   brew install rsync
   uv tool install jarvislabs
   just setup jarvis
   ssh-add ~/.ssh/id_ed25519
   ```

   Setup asks for your API token and exact instance name, then selects Jarvis as
   default. The runner uses SSH user `ubuntu` and port 22.
4. Run CUDA code from your Mac:

   ```bash
   just run lecture4/vecadd.cu
   ```

The runner resumes the VM if needed, discovers its current IP, waits for CUDA,
uploads and runs code, downloads results, and pauses the VM afterward—even after
failure. It also pauses an already-running VM; do not share it with concurrent jobs.

### Jarvis: Python examples only

Resume the VM and connect using the dashboard's SSH command. With Python 3.12 and
CUDA 13.x installed, run **on the VM**:

```bash
python3 -m venv /home/ubuntu/cuda-workspace/.venv
/home/ubuntu/cuda-workspace/.venv/bin/python -m pip install \
  numba-cuda==0.30.4 cupy-cuda13x==14.2.0 numpy==2.3.5
```

If Python reports missing venv support, install `python3.12-venv` first.
Then return to your Mac and run:

```bash
just run lecture4/maxpool2d.py
```

## Switch providers

After setting up both providers, save a default:

```bash
just provider modal
# Or:
just provider jarvis
```

Override the provider or Modal GPU for one run:

```bash
just provider=jarvis run lecture4/vecadd.cu
just provider=modal gpu=A100 run lecture4/vecadd.cu
```

## Results and settings

- `just status` shows the selected provider.
- Logs and exit codes are in `results/<run-id>/`.
- Write output files to `$GPU_RESULTS_DIR` for automatic download.
- Settings are saved locally in `.gpu.json`; settings, results, and `.venv` are
  ignored by Git. Never commit API tokens or private keys.
- If a run is force-killed or loses connectivity, check the provider dashboard
  to confirm compute stopped.

See [tools/GPU.md](tools/GPU.md) for custom SSH settings, timeouts, and limitations.
The Jarvis runner disables server host-key verification for its connections;
this accommodates rotating keys but cannot detect server impersonation.

## Lecture-specific instructions

| Folder | Exercises |
|---|---|
| [lecture1](lecture1) | H100 bandwidth measurements and Qwen3-8B serving |
| [lecture3](lecture3) | CUDA image kernels and warp divergence |

Programs that already manage Modal themselves, such as `lecture3/dog.py`, use
their lecture's documented `modal run` command instead of `just run`.

## Profile GPU execution

```bash
just profile lecture4/vecadd.cu
```

Prints kernel and memory-transfer timing summaries in the terminal and saves
Nsight Systems reports in `results/<run-id>/artifacts/profile/`. Compute is
released afterward. See [profiling setup and limitations](tools/GPU.md#profile-from-the-terminal).
