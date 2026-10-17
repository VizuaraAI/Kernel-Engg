"""Modal implementation for `just run`; all client dependencies live in .venv."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import tarfile
import tempfile
import uuid

import modal
from modal.stream_type import StreamType

ROOT = Path(__file__).resolve().parents[1]


def execute(sb, command, log, timeout=240):
    process = sb.exec(*command, stderr=StreamType.STDOUT, timeout=timeout)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file() or source.suffix not in ('.cu', '.py'):
        parser.error('Provide an existing .cu or .py file')
    arguments = args.arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    profiling = os.environ.get('GPU_PROFILE') == '1'
    gpu = os.environ.get('MODAL_GPU') or 'L4'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    results = ROOT / 'results' / run_id
    results.mkdir(parents=True)
    metadata = dict(provider='modal', requested_gpu=gpu, source=str(source), profiled=profiling)
    sb = None
    status = 1
    try:
        image = (modal.Image.from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
                 .entrypoint([])
                 .pip_install('cupy-cuda13x==14.2.0', 'numpy==2.3.5', 'numba-cuda==0.30.4'))
        if profiling:
            image = (image.apt_install('cuda-nsight-systems-13-0')
                     .add_local_file(ROOT / 'tools/profile.sh', '/opt/gpu-profile.sh', copy=True))
        image = image.add_local_dir(source.parent, '/workspace/src', ignore=[
                     '**/.git/**', '**/.venv/**', '**/__pycache__/**', '**/results/**',
                     '/bin/**', '/lib/**', 'pyvenv.cfg', '.gpu.json', '.env', '.env.*'])
        with modal.enable_output():
            app = modal.App.lookup('kernel-engg-runner', create_if_missing=True)
            sb = modal.Sandbox.create('sleep', 'infinity', app=app, image=image, gpu=gpu, timeout=300,
                                      idle_timeout=300, workdir='/workspace/src')
        metadata['sandbox_id'] = sb.object_id
        (results / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Modal sandbox: {sb.object_id}; GPU: {gpu}', flush=True)
        with (results / 'run.log').open('w') as log:
            setup = ['bash', '-lc', 'mkdir -p /workspace/results && nvidia-smi --query-gpu=name --format=csv,noheader']
            status = execute(sb, setup, log)
            if status == 0:
                profile_prefix = 'bash /opt/gpu-profile.sh ' if profiling else ''
                if source.suffix == '.cu':
                    cmd = ('nvcc -O3 ' + ('-lineinfo ' if profiling else '') + shlex.quote('./' + source.name) + ' -o /workspace/program && '
                           + profile_prefix + shlex.join(['/workspace/program', *arguments]))
                else:
                    cmd = profile_prefix + shlex.join(['python', './' + source.name, *arguments])
                status = execute(sb, ['bash', '-lc',
                    'export GPU_RESULTS_DIR=/workspace/results CUDA_HOME=/usr/local/cuda; '
                    'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}; ' + cmd], log)
            # Collect results before termination, including after compilation/runtime failure.
            pack = sb.exec('tar', '-czf', '/tmp/gpu-results.tgz', '-C', '/workspace/results', '.', timeout=20)
            if pack.wait() != 0:
                raise RuntimeError('Could not archive Modal results')
            with tempfile.TemporaryDirectory() as staging:
                archive = Path(staging) / 'results.tgz'
                sb.filesystem.copy_to_local('/tmp/gpu-results.tgz', archive)
                with tarfile.open(archive, mode='r:gz') as tar:
                    for member in tar.getmembers():
                        if not (member.isfile() or member.isdir()):
                            raise RuntimeError('Artifact archive contains a link or special file')
                    tar.extractall(results / 'artifacts', filter='data')
    except KeyboardInterrupt:
        status = 130
        raise
    except Exception:
        if status == 0:
            status = 1
        raise
    finally:
        (results / 'exit-code.txt').write_text(str(status) + '\n')
        if sb is not None:
            print('Terminating Modal sandbox...', flush=True)
            try:
                sb.terminate(wait=True)
                metadata['terminated'] = True
                metadata['final_sandbox_status'] = sb.poll()
            except Exception as error:
                metadata['termination_error'] = str(error)
                print(f'WARNING: termination failed; check sandbox {sb.object_id}: {error}', file=sys.stderr)
                if status == 0:
                    status = 1
            (results / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (results / 'exit-code.txt').write_text(str(status) + '\n')
        print(f'Results: {results}', flush=True)
    return status


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
