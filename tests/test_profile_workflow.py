"""Profiler wrapper checks; no GPU or provider connection required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def capture(self, mode, target_status=0):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / 'nsys'
            executable.write_text('''#!/bin/bash
case "$1" in
--version) echo 'Nsight test';;
profile)
  for arg in "$@"; do
    case "$arg" in --output=*) report="${arg#--output=}";; esac
  done
  if [[ "$MODE" != missing ]]; then touch "$report.nsys-rep"; fi
  exit "${TARGET_STATUS:-0}";;
stats)
  [[ "$*" == *--force-export=true* ]] || exit 2
  if [[ "$MODE" == permission ]]; then echo 'CUPTI permission denied' >&2; exit 1; fi
  if [[ "$MODE" == empty ]]; then echo 'No CUDA events'; exit 0; fi
  if [[ "$*" == *csv* ]]; then printf 'Time (%%),Total Time,Instances,Name\\n100.0,1000,1,test_kernel\\n'; else echo 'test_kernel 1 launch'; fi;;
esac
''')
            executable.chmod(0o755)
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}',
                       GPU_RESULTS_DIR=str(root / 'results'), MODE=mode,
                       TARGET_STATUS=str(target_status))
            result = subprocess.run(['bash', str(ROOT / 'tools/profile.sh'), '/bin/true'],
                                    env=env, capture_output=True, text=True)
            return result

    def test_summary(self):
        result = self.capture('success')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('not overall GPU utilization', result.stdout)

    def test_program_failure(self):
        self.assertEqual(self.capture('success', 7).returncode, 7)

    def test_no_report(self):
        self.assertNotEqual(self.capture('missing').returncode, 0)

    def test_permission_failure(self):
        self.assertNotEqual(self.capture('permission').returncode, 0)

    def test_empty_capture_not_success(self):
        self.assertNotEqual(self.capture('empty').returncode, 0)


if __name__ == '__main__':
    unittest.main()
