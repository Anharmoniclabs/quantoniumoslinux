"""Failure paths for ownership cleanup and explicit same-runtime recovery."""

import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import a100_run_guard as guard


class GuardFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'experiment'
        self.root = self.out.parent / '.experiment.a100-owner'

    def previous(self):
        self.root.mkdir()
        generation = self.root / '0000000001'
        generation.mkdir()
        host, boot = guard._host()
        pid = guard._proc_pid()
        _, start = guard._process_identity(pid)
        record = dict(token='prior', host=host, boot_id=boot, pid=pid, process_start=start)
        (generation / 'owner.json').write_text(json.dumps(record))
        return generation, record

    def test_primary_checkpoint_error_is_preserved_when_guard_disappears(self):
        primary = RuntimeError('basic_ios::clear: iostream error')
        with self.assertRaises(RuntimeError) as caught:
            with guard.guarded_run(self.out) as owner:
                owner.error = OSError(errno.EIO, 'Drive unavailable')
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIn('cleanup also failed', '\n'.join(primary.__notes__))
        self.assertIsNone(guard._CURRENT.get())
        self.assertFalse((self.root / '0000000001/released.json').exists())

    def test_normal_completion_still_rejects_cleanup_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'heartbeat ownership'):
            with guard.guarded_run(self.out) as owner:
                owner.error = OSError(errno.EIO, 'Drive unavailable')
        self.assertIsNone(guard._CURRENT.get())

    def test_keyboard_interrupt_is_preserved(self):
        primary = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt) as caught:
            with guard.guarded_run(self.out) as owner:
                owner.error = OSError(errno.EIO, 'Drive unavailable')
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIsNone(guard._CURRENT.get())

    def test_release_write_error_does_not_replace_training_error(self):
        primary = ValueError('training failed')
        real_write = guard._write_once

        def fail_release(path, record):
            if path.name == 'released.json':
                raise OSError(errno.ENOSPC, 'No space left on device')
            return real_write(path, record)

        with patch.object(guard, '_write_once', side_effect=fail_release):
            with self.assertRaises(ValueError) as caught:
                with guard.guarded_run(self.out):
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertIn('No space left', '\n'.join(primary.__notes__))

    def test_missing_guard_keeps_underlying_filesystem_error(self):
        with self.assertRaisesRegex(RuntimeError, 'missing or unreadable') as caught:
            guard._latest(self.root)
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)

    def test_guard_symlink_is_rejected(self):
        other = self.out.parent / 'actual'
        other.mkdir()
        self.root.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'not a real directory'):
            guard._latest(self.root)

    def test_fresh_same_host_dead_owner_requires_explicit_recovery(self):
        old, _ = self.previous()
        with patch.object(guard, '_process_identity', side_effect=ProcessLookupError):
            with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
                guard._allow_next(old, recover=False)
            guard._allow_next(old, recover=True)

    def test_fresh_remote_owner_cannot_be_immediately_recovered(self):
        old, record = self.previous()
        record['host'] = 'different-Colab-runtime'
        (old / 'owner.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
            guard._allow_next(old, recover=True)

    def test_same_host_unknown_process_state_is_not_proof_of_exit(self):
        old, _ = self.previous()
        with patch.object(guard, '_process_identity', side_effect=PermissionError):
            with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
                guard._allow_next(old, recover=True)

    def test_same_live_process_remains_protected(self):
        old, _ = self.previous()
        with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
            guard._allow_next(old, recover=True)


if __name__ == '__main__':
    unittest.main()
