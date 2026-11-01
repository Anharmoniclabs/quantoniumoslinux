from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import a100_run_guard as guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)/'experiment'
        self.root = self.out.parent/f'.{self.out.name}.a100-owner'

    def previous(self, **overrides):
        self.root.mkdir()
        generation = self.root/'0000000001'
        generation.mkdir()
        host, boot = guard._host()
        pid = guard._proc_pid()
        _, start = guard._process_identity(pid)
        obj = dict(token='p', host=host, boot_id=boot, pid=pid, process_start=start)
        (generation/'owner.json').write_text(json.dumps({**obj, **overrides}))
        return generation

    def age(self, generation):
        t = time.time() - guard.STALE_SECONDS - 10
        for obj in generation.iterdir():
            os.utime(obj, (t, t))
        os.utime(generation, (t, t))

    def test_acquire_release_rerun_and_stop_preserved(self):
        self.out.mkdir()
        (self.out/'STOP').touch()
        with guard.guarded_run(self.out) as g:
            g.check()
            self.assertEqual(g.generation.name, '0000000001')
            with self.assertRaisesRegex(RuntimeError, 'already owns'):
                with guard.guarded_run(self.out, recover=True):
                    pass
        with guard.guarded_run(self.out) as g:
            self.assertEqual(g.generation.name, '0000000002')
        self.assertEqual({f.name for f in self.out.iterdir()}, {'STOP'})

    def test_remote_fresh_blocks_even_explicit_recovery(self):
        self.previous(host='other-Colab')
        for recover in (False, True):
            with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
                with guard.guarded_run(self.out, recover=recover):
                    pass

    def test_same_live_pid_blocks_even_stale(self):
        self.age(self.previous())
        with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
            with guard.guarded_run(self.out, recover=True):
                pass

    def test_stale_remote_requires_explicit_operator_recovery(self):
        old = self.previous(host='other-Colab')
        self.age(old)
        with self.assertRaisesRegex(RuntimeError, 'Stale A100 owner'):
            with guard.guarded_run(self.out):
                pass
        with guard.guarded_run(self.out, recover=True) as g:
            self.assertEqual(g.generation.name, '0000000002')
            self.assertEqual(guard._read_owner(old)['token'], 'p')

    def test_two_remote_recoveries_of_the_same_generation_have_one_winner(self):
        previous = self.previous(host='old-runtime')
        self.age(previous)
        barrier, loser_done = threading.Barrier(2), threading.Event()
        real_allow = guard._allow_next

        def observe_stale(generation, recover):
            real_allow(generation, recover)
            barrier.wait(timeout=3)  # Both contenders have observed the same stale generation.

        def contender():
            try:
                with guard.guarded_run(self.out, recover=True) as owner:
                    self.assertTrue(loser_done.wait(timeout=3))
                    owner.check()
                    return 'winner'
            except RuntimeError as exc:
                if 'Another runtime claimed' not in str(exc):
                    raise
                loser_done.set()
                return 'loser'

        with patch.object(guard, '_local_run_lock', lambda _: nullcontext()), \
                patch.object(guard, '_allow_next', observe_stale), ThreadPoolExecutor(2) as pool:
            result = [pool.submit(contender) for _ in range(2)]
            self.assertEqual(sorted(job.result() for job in result), ['loser', 'winner'])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['0000000001', '0000000002'])

    def test_crash_leaves_an_owner_and_recovery_preserves_generation(self):
        cmd = ('from pathlib import Path; import os,sys; from a100_run_guard import guarded_run; '
               'g=guarded_run(Path(sys.argv[1])); g.__enter__(); os._exit(71)')
        p = subprocess.run([sys.executable, '-c', cmd, str(self.out)], cwd=Path(__file__).resolve().parent)
        self.assertEqual(p.returncode, 71)
        old = self.root/'0000000001'
        self.age(old)
        with guard.guarded_run(self.out, recover=True) as g:
            self.assertNotEqual(old, g.generation)
            g.check()

    def test_abandoned_init_is_not_immediately_reclaimed(self):
        self.root.mkdir()
        old = self.root/'0000000001'
        old.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'Active or initializing'):
            with guard.guarded_run(self.out, recover=True):
                pass
        self.age(old)
        with guard.guarded_run(self.out, recover=True) as g:
            g.check()

    def test_replaced_generation_is_detected_and_cannot_release_successor(self):
        with self.assertRaisesRegex(RuntimeError, 'no longer holds'):
            with guard.guarded_run(self.out) as g:
                (self.root/'0000000002').mkdir()  # Emulate a newly fenced remote generation.
                with self.assertRaisesRegex(RuntimeError, 'no longer holds'):
                    g.check()
        self.assertFalse((self.root/'0000000002'/'released.json').exists())

    def test_heartbeat_updates_the_immutable_owner_record(self):
        with patch.object(guard, 'HEARTBEAT_SECONDS', .01):
            with guard.guarded_run(self.out) as g:
                path = g.generation/'owner.json'
                initial = path.stat().st_mtime_ns
                contents = path.read_bytes()
                deadline = time.monotonic() + 1
                while path.stat().st_mtime_ns <= initial and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertGreater(path.stat().st_mtime_ns, initial)
                self.assertEqual(path.read_bytes(), contents)
                g.check()

    def test_cleanup_exact_allowlist_and_only_unpublished_destinations(self):
        self.out.mkdir()
        for name in ('request.partial', 'STOP', 'arbitrary.partial'):
            (self.out/name).write_bytes(b'state')
        for arm in ('broad', 'paired', 'other'):
            d = self.out/arm
            d.mkdir()
            for name in ('latest.pt.partial', 'progress.partial', 'best.pt.partial', 'STOP'):
                (d/name).write_bytes(b'state')
        (self.out/'paired'/'latest.pt').write_bytes(b'valid')
        (self.out/'paired'/'progress.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'matching active'):
            guard.clean_orphans(self.out)
        with guard.guarded_run(self.out):
            self.assertEqual(set(guard.clean_orphans(self.out)), {
                self.out/'request.partial', self.out/'broad'/'latest.pt.partial',
                self.out/'broad'/'progress.partial',})
            self.assertEqual(guard.clean_orphans(self.out), [])
        self.assertEqual((self.out/'paired'/'latest.pt').read_bytes(), b'valid')
        for relative in ('STOP', 'arbitrary.partial', 'broad/best.pt.partial', 'broad/STOP',
                         'paired/STOP', 'paired/latest.pt.partial', 'other/latest.pt.partial'):
            self.assertTrue((self.out/relative).exists())

    def test_cleanup_does_not_follow_symlinks(self):
        self.out.mkdir()
        victim = self.out.parent/'external'
        victim.write_text('untouched')
        (self.out/'request.partial').symlink_to(victim)
        with guard.guarded_run(self.out), self.assertRaisesRegex(RuntimeError, 'orphan type'):
            guard.clean_orphans(self.out)
        self.assertEqual(victim.read_text(), 'untouched')
