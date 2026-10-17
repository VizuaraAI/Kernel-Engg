# Choose a provider with `just provider modal`, or override one invocation.
set positional-arguments
provider := env_var_or_default("GPU_PROVIDER", "")
kernel := env_var_or_default("NCU_KERNEL", "regex:^(?!(spin|empty|cupy_fill|census)$).*")
launches := env_var_or_default("NCU_LAUNCH_COUNT", "1")
gpu := env_var_or_default("MODAL_GPU", "")

default:
    @just --list

# Authenticate and configure a provider (default: Modal).
setup backend="modal":
    bash tools/provider.sh setup "$@"

# Save the default provider locally.
provider backend:
    bash tools/provider.sh select "$@"

# Show provider status.
status:
    GPU_PROVIDER="{{provider}}" bash tools/provider.sh status

# Explicit Jarvis resume.
resume:
    bash tools/gpu.sh resume

# Explicit Jarvis pause.
pause:
    bash tools/gpu.sh pause

# Execute with the chosen provider, collect results, and release compute.
run file *args:
    GPU_PROVIDER="{{provider}}" MODAL_GPU="{{gpu}}" bash tools/provider.sh run "$@"

# Capture CUDA activity, print timing summaries, and save an Nsight Systems report.
timeline-profile file *args:
    GPU_PROVIDER="{{provider}}" MODAL_GPU="{{gpu}}" bash tools/provider.sh timeline-profile "$@"

# Collect hardware metrics for selected kernel launches using Nsight Compute.
compute-profile file *args:
    GPU_PROVIDER="{{provider}}" MODAL_GPU="{{gpu}}" NCU_KERNEL="{{kernel}}" NCU_LAUNCH_COUNT="{{launches}}" bash tools/provider.sh compute-profile "$@"
