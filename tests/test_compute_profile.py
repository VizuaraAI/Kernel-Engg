"""Tests use a fake ncu; no GPU or credits required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComputeProfileTests(unittest.TestCase):
    def capture(self, mode):
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
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}', MODE=mode,
                       CALLS=str(root / 'calls'), GPU_RESULTS_DIR=str(root / 'results'),
                       NCU_KERNEL='rgb_to_gray', NCU_LAUNCH_COUNT='1')
            result = subprocess.run(['bash', str(ROOT / 'tools/compute-profile.sh'), '/bin/true'],
                                    env=env, capture_output=True, text=True)
            calls = (root / 'calls').read_text()
            self.assertIn('--kernel-name\nrgb_to_gray', calls)
            self.assertIn('--launch-count\n1', calls)
            return result

    def test_success(self): self.assertEqual(self.capture('success').returncode, 0)
    def test_failure(self): self.assertEqual(self.capture('failure').returncode, 7)
    def test_permission(self):
        result = self.capture('permission')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('blocked by the host/provider', result.stderr)
    def test_empty(self): self.assertNotEqual(self.capture('empty').returncode, 0)
    def test_no_report(self): self.assertNotEqual(self.capture('missing').returncode, 0)
