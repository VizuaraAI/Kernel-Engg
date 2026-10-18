"""Tests use a fake ncu; no GPU or credits required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComputeProfileTests(unittest.TestCase):
    def capture(self, mode, sudo=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / 'ncu'
            executable.write_text('''#!/bin/bash
if [[ "$1" == --version ]]; then echo 'test ncu'; exit; fi
if [[ "$1" == --import ]]; then
 if [[ "$MODE" != empty ]]; then echo 'sm__throughput.avg.pct_of_peak_sustained_elapsed,12.0'; fi
 exit
fi
printf '%s\\n' "$@" > "$CALLS"
if [[ "$MODE" == permission ]]; then echo ERR_NVGPUCTRPERM; exit 1; fi
if [[ "$MODE" == failure ]]; then exit 7; fi
while [[ $# -gt 0 ]]; do
 if [[ "$1" == --export ]]; then shift; printf report > "$1.ncu-rep"; fi
 shift
done
if [[ "$MODE" == missing ]]; then rm -f "$GPU_RESULTS_DIR/compute-profile/metrics.ncu-rep"; fi
''')
            executable.chmod(0o755)
            sudo_executable = root / 'sudo'
            sudo_executable.write_text('''#!/bin/bash
if [[ "$MODE" == sudo_denied ]]; then echo 'sudo: a password is required' >&2; exit 1; fi
[[ "$1" == -n ]] && shift
[[ "$1" == -- ]] && shift
if [[ "$1" == chown ]]; then echo restored > "$OWNER_LOG"; exit 0; fi
exec "$@"
''')
            sudo_executable.chmod(0o755)
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}', MODE=mode,
                       CALLS=str(root / 'calls'), GPU_RESULTS_DIR=str(root / 'results'),
                       NCU_KERNEL='rgb_to_gray', NCU_LAUNCH_COUNT='1',
                       GPU_NCU_SUDO='1' if sudo else '0', OWNER_LOG=str(root/'owner'))
            result = subprocess.run(['bash', str(ROOT / 'tools/compute-profile.sh'), '/bin/true'],
                                    env=env, capture_output=True, text=True)
            if mode == 'sudo_denied':
                self.assertFalse((root/'calls').exists())
                return result
            if sudo and mode == 'success':
                self.assertTrue((root/'owner').exists())
            calls = (root / 'calls').read_text()
            self.assertIn('--kernel-name\nrgb_to_gray', calls)
            self.assertIn('--launch-count\n1', calls)
            return result

    def test_success(self): self.assertEqual(self.capture('success').returncode, 0)
    def test_failure(self): self.assertEqual(self.capture('failure').returncode, 7)
    def test_permission(self):
        result = self.capture('permission')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('restricted for this execution identity', result.stderr)
    def test_empty(self): self.assertNotEqual(self.capture('empty').returncode, 0)
    def test_no_report(self): self.assertNotEqual(self.capture('missing').returncode, 0)

    def test_sudo_success(self): self.assertEqual(self.capture('success', sudo=True).returncode, 0)
    def test_sudo_denied(self):
        result = self.capture('sudo_denied', sudo=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Noninteractive sudo', result.stderr)
    def test_sudo_program_failure(self): self.assertEqual(self.capture('failure', sudo=True).returncode, 7)
