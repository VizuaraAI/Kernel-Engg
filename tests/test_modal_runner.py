"""Lifecycle tests without allocating a GPU. Run with .venv/bin/python -m unittest discover -s tests."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location('modal_runner', Path(__file__).resolve().parents[1] / 'tools/modal_runner.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LifecycleTests(unittest.TestCase):
    def invoke(self, program_status=0, artifact_error=None, cleanup_error=None, profile_mode=''):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'kernel.cu'
            source.write_text('// test')
            sandbox = MagicMock()
            sandbox.object_id = 'sb-test'
            sandbox.poll.return_value = 137
            sandbox.terminate.side_effect = cleanup_error
            sandbox.exec.return_value.wait.return_value = 0
            def download(remote, local):
                if artifact_error:
                    raise artifact_error
                import tarfile
                with tarfile.open(local, 'w:gz'):
                    pass
            sandbox.filesystem.copy_to_local.side_effect = download
            with patch.object(runner, 'ROOT', root), patch.object(runner.sys, 'argv', ['runner', str(source)]), \
                 patch.object(runner.modal, 'Image'), patch.object(runner.modal, 'App'), \
                 patch.object(runner.modal, 'enable_output'), \
                 patch.object(runner.modal.Sandbox, 'create', return_value=sandbox) as create, \
                 patch.dict(os.environ, {'GPU_PROFILE': profile_mode}), \
                 patch.object(runner, 'execute', side_effect=[0, program_status]):
                if artifact_error:
                    with self.assertRaises(RuntimeError):
                        runner.main()
                else:
                    status = runner.main()
                    self.assertEqual(status, program_status or (1 if cleanup_error else 0))
            metadata = json.loads(next((root / 'results').glob('*/metadata.json')).read_text())
            mode = {'timeline': 'timeline', '1': 'timeline', 'compute': 'compute'}.get(profile_mode, 'run')
            self.assertTrue(metadata['sandbox_name'].startswith(f'kernel-{mode}-'))
            self.assertEqual(create.call_args.kwargs['name'], metadata['sandbox_name'])
            self.assertEqual(create.call_args.kwargs['tags'], metadata['tags'])
            self.assertEqual(metadata['tags']['program'], 'kernel')
            self.assertEqual(metadata['tags']['mode'], mode)
            self.assertEqual(metadata['sandbox_id'], 'sb-test')
            sandbox.terminate.assert_called_once_with(wait=True)
            saved = next((root / 'results').glob('*/exit-code.txt')).read_text().strip()
            self.assertEqual(saved, str(program_status or (1 if artifact_error or cleanup_error else 0)))

    def test_profile_names(self):
        for mode in ('timeline', 'compute', '1'):
            with self.subTest(mode=mode): self.invoke(profile_mode=mode)

    def test_name_limits_and_source(self):
        run_id = '20261018T222000Z-a1b2c3d4'
        self.assertEqual(runner.sandbox_name(Path('dog.py'), 'run', run_id), f'dog-run-{run_id}')
        self.assertEqual(runner.sandbox_name(Path('vecadd.cu'), 'run', run_id), f'vecadd-run-{run_id}')
        name = runner.sandbox_name(Path(('a' * 100) + ' dog 🐕.py'), 'compute', run_id)
        self.assertLess(len(name), 64)
        self.assertRegex(name, r'^[A-Za-z0-9_.-]+$')
        self.assertTrue(name.endswith(f'-compute-{run_id}'))

    def test_success(self): self.invoke()
    def test_program_failure(self): self.invoke(program_status=7)
    def test_artifact_failure(self): self.invoke(artifact_error=RuntimeError('download failed'))
    def test_cleanup_failure(self): self.invoke(cleanup_error=RuntimeError('termination failed'))
    def test_original_failure_preserved(self): self.invoke(program_status=7, cleanup_error=RuntimeError('termination failed'))


if __name__ == '__main__':
    unittest.main()
