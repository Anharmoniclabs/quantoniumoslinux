"""Exercise local recovery and storage hooks without Drive access or a GPU."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import phistate_a100_storage_recovery as recovery


class StorageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.project = self.work / 'project'
        self.output = self.project / recovery.journal.OUTPUT
        self.relative = recovery.journal.OUTPUT + '/dense_20261101/latest.pt'
        self.checkpoint = self.project / self.relative
        self.checkpoint.parent.mkdir(parents=True)

    def payload(self, step):
        return {
            'experiment_request': {
                **recovery.RECOVERY['old_arm_request'],
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            },
            'phase_step': step, 'history': [{} for _ in range(step)],
            'phase_trained_tokens': step * recovery.journal.BATCH * recovery.journal.CONTEXT,
            'state_dict': {'weight': torch.tensor([1.0, 2.0])},
        }

    def state(self):
        return {
            'format': recovery.FORMAT,
            'experiment': recovery.RECOVERY['old_experiment_request_key'],
            'token': 'fixture', 'folder_id': 'fixture_folder', 'folders': {}, 'arms': {},
            'files': {self.relative: {'id': 'saved_step32', 'sha256': 'a' * 64}},
            'steps': {'dense_20261101': 32}, 'status': 'recovering',
        }

    def mocks_for_run(self, stack, *, result=None, failure=None):
        torch.save(self.payload(64), self.checkpoint)
        (self.work / 'transport.json').write_text(json.dumps(self.state()))
        stack.enter_context(patch.dict(recovery.RECOVERY, {'immutable_sources': []}))
        stack.enter_context(patch.object(recovery.training, 'require_a100'))
        stack.enter_context(patch.object(recovery, 'check_compute'))
        stack.enter_context(patch.object(recovery, 'ensure_local', return_value=self.work))
        store = Mock()
        store.quota_guard.return_value = {'quota_known': True, 'free_bytes': 20 * recovery.GIB}
        stack.enter_context(patch.object(recovery, 'DriveStore', return_value=store))
        backups = SimpleNamespace(checkpoint=Mock(), progress=Mock(), weights=Mock(), commit_index=Mock())
        stack.enter_context(patch.object(recovery, 'Backups', return_value=backups))
        journal = stack.enter_context(patch.object(
            recovery.journal, 'run', return_value=result or {'status': 'complete'}, side_effect=failure))
        return store, backups, journal

    def test_hooks_restore_on_error_and_preserve_exception_identity(self):
        original = recovery.training.atomic_checkpoint, recovery.training.atomic_json
        backups = SimpleNamespace(checkpoint=Mock(), progress=Mock())
        primary = RuntimeError('Drive backup failed')
        with self.assertRaises(RuntimeError) as caught:
            with recovery.storage_hooks(backups):
                self.assertIs(recovery.training.atomic_checkpoint, backups.checkpoint)
                self.assertIs(recovery.training.atomic_json, backups.progress)
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual((recovery.training.atomic_checkpoint, recovery.training.atomic_json), original)

    def test_later_local_optimizer_state_is_not_overwritten_by_old_cloud_index(self):
        with ExitStack() as stack:
            store, backups, journal = self.mocks_for_run(stack)
            before = self.checkpoint.read_bytes()
            result = recovery.run(self.work, service=object())
            self.assertEqual(result['status'], 'complete')
            store.download.assert_not_called()
            self.assertEqual(self.checkpoint.read_bytes(), before)
            self.assertEqual(torch.load(self.checkpoint, weights_only=True)['phase_step'], 64)
            journal.assert_called_once_with(self.project, recover=True)
            self.assertGreaterEqual(backups.weights.call_count, 1)
            self.assertTrue(all(call.args == (self.checkpoint, 64)
                                for call in backups.weights.call_args_list))

    def test_wrapper_preserves_primary_error_and_local_checkpoint(self):
        primary = RuntimeError('remote upload interrupted')
        original = recovery.training.atomic_checkpoint, recovery.training.atomic_json
        with ExitStack() as stack:
            self.mocks_for_run(stack, failure=primary)
            before = self.checkpoint.read_bytes()
            with self.assertRaises(RuntimeError) as caught:
                recovery.run(self.work, service=object())
            self.assertIs(caught.exception, primary)
            self.assertEqual(self.checkpoint.read_bytes(), before)
            record = json.loads((self.work / 'recovery_error.json').read_text())
            self.assertTrue(record['keep_runtime_connected'])
        self.assertEqual((recovery.training.atomic_checkpoint, recovery.training.atomic_json), original)

    def test_secondary_local_reporting_failure_does_not_hide_primary(self):
        primary = RuntimeError('original transfer error')
        normal_write = recovery.atomic_json

        def failing_error_write(path, info):
            if Path(path).name == 'recovery_error.json':
                raise OSError('local disk error')
            return normal_write(path, info)

        with ExitStack() as stack:
            self.mocks_for_run(stack, failure=primary)
            stack.enter_context(patch.object(recovery, 'atomic_json', side_effect=failing_error_write))
            with self.assertRaises(RuntimeError) as caught:
                recovery.run(self.work, service=object())
            self.assertIs(caught.exception, primary)
            self.assertIn('Local error reporting also failed', '\n'.join(primary.__notes__))

    def test_progress_uploads_only_at_backup_boundary_and_on_stop(self):
        backups = recovery.Backups(Mock(), self.work, self.state())
        path = self.checkpoint.with_name('progress.json')
        with patch.object(backups, 'weights') as weights, patch.object(backups, 'commit_index') as commit:
            backups.progress(path, {'completed_steps': 32, 'status': 'training'})
            weights.assert_not_called(); commit.assert_not_called()
            backups.progress(path, {'completed_steps': 128, 'status': 'training'})
            weights.assert_called_once_with(self.checkpoint, 128)
            commit.assert_called_once_with('training')
            weights.reset_mock(); commit.reset_mock()
            backups.progress(path, {'completed_steps': 160, 'status': 'stopped'})
            weights.assert_called_once_with(self.checkpoint, 160)
            commit.assert_called_once_with('stopped')
        self.assertEqual(json.loads(path.read_text())['completed_steps'], 160)

    def test_metadata_receipts_keep_canonical_local_relative_paths(self):
        root_meta = self.output / 'report.json'
        arm_meta = self.checkpoint.with_name('progress.json')
        root_meta.write_text('{"status":"running"}')
        arm_meta.write_text('{"completed_steps":64}')
        state = self.state()
        store = Mock()
        store.ensure_folder.return_value = 'arm_folder'

        def uploaded(path, parent, name, token, role):
            return {'id': 'receipt_' + path.stem,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'name': name}

        store.upload.side_effect = uploaded
        backups = recovery.Backups(store, self.work, state)
        backups.metadata()
        for path in (root_meta, arm_meta):
            relative = str(path.relative_to(self.project))
            self.assertIn(relative, state['files'])
            self.assertEqual(state['files'][relative]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        calls = store.upload.call_count
        backups.metadata()
        self.assertEqual(store.upload.call_count, calls)

    def test_invalid_checkpoint_exposures_are_rejected(self):
        state = self.payload(32)
        state['phase_trained_tokens'] += 1
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'update/token count mismatch'):
            recovery.validate_local_checkpoint(self.project)

    def test_research_source_hashes_still_match_saved_experiment(self):
        request = recovery.RECOVERY['old_experiment_request']
        with patch.object(recovery.torch, '__version__', request['torch']), \
                patch.object(recovery.torch.version, 'cuda', request['cuda']):
            recovery.check_compute()

    def test_paths_cannot_escape_or_alias_unknown_arms(self):
        for path in ('../outside.json', '/tmp/x.json', recovery.journal.OUTPUT + '/unknown/latest.pt',
                     recovery.journal.OUTPUT + '/dense_20261101/../request.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                recovery.safe_relative(path)


if __name__ == '__main__':
    unittest.main()
