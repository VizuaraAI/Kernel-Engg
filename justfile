# Run `just` to list available commands.
set positional-arguments

default:
    @just --list

# Authenticate with JarvisLabs (one time).
setup:
    bash tools/gpu.sh setup

# Show the configured instance and its current IP.
status:
    bash tools/gpu.sh status

# Resume the configured instance explicitly.
resume:
    bash tools/gpu.sh resume

# Pause the configured instance explicitly.
pause:
    bash tools/gpu.sh pause

# Resume if needed, wait, sync, run, save results, then pause.
run file *args:
    bash tools/gpu.sh run "$@"
