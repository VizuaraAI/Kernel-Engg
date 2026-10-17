"""Lifecycle tests without allocating a GPU. Run with .venv/bin/python -m unittest discover -s tests."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location('modal_runner', Path(__file__).resolve().parents[1] / 'tools/modal_runner.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LifecycleTests(unittest.TestCase):
    def invoke(self, program_status=0, artifact_error=None, cleanup_error=None):
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
                 patch.object(runner.modal.Sandbox, 'create', return_value=sandbox), \
                 patch.object(runner, 'execute', side_effect=[0, program_status]):
                if artifact_error:
                    with self.assertRaises(RuntimeError):
                        runner.main()
                else:
                    status = runner.main()
                    self.assertEqual(status, program_status or (1 if cleanup_error else 0))
            sandbox.terminate.assert_called_once_with(wait=True)
            saved = next((root / 'results').glob('*/exit-code.txt')).read_text().strip()
            self.assertEqual(saved, str(program_status or (1 if artifact_error or cleanup_error else 0)))

    def test_success(self): self.invoke()
    def test_program_failure(self): self.invoke(program_status=7)
    def test_artifact_failure(self): self.invoke(artifact_error=RuntimeError('download failed'))
    def test_cleanup_failure(self): self.invoke(cleanup_error=RuntimeError('termination failed'))
    def test_original_failure_preserved(self): self.invoke(program_status=7, cleanup_error=RuntimeError('termination failed'))


if __name__ == '__main__':
    unittest.main()
