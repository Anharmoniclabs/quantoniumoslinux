"""Journal ownership, interruption and reporting checks. Uses temporary files."""
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import coherence_recovery as coordinator
import continuation_journal as journal
from chat_recovery_train import run_lock


class JournalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / journal.EXPERIMENT
        self.ready = self.root / 'ready'
        self.lab = Path(journal.__file__).parent

    def put(self, file, content):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(content))

    def test_stored_request_drives_resume(self):
        self.put(self.out / 'request.json', {'steps_per_arm': 17, 'eval_every': 4, 'save_every': 2})
        (self.out / 'STOP').touch()
        def run(root, **args):
            with coordinator.run_lock(self.out / 'execution'):
                self.assertFalse((self.out / 'STOP').exists())
                self.assertEqual((args['steps'], args['eval_every'], args['save_every']), (17, 4, 2))
                self.assertFalse(args['clear_stop'])
                return {'status': 'failed_quality'}
        with patch.object(coordinator, 'run', run):
            self.assertEqual(journal._worker(self.root, self.ready, 500), 'failed_quality')

    def test_snapshot_reads_actual_progress_even_if_coordinator_report_is_old(self):
        self.put(self.out / 'request.json', {'steps_per_arm': 500})
        self.put(self.out / 'report.json', {'status': 'comparing'})
        quality = {'macro_score': .5, 'qualified': False, 'buckets': {'math': {'passed': 2, 'total': 8}},
                   'rows': [{'bucket': 'math', 'prompt': 'Add 1 and 1', 'text': '2', 'passed': True}]}
        self.put(self.out / 'original/progress.json', {'status': 'completed_failed_quality', 'phase_step': 500})
        self.put(self.out / 'protected/progress.json', {'status': 'stopped', 'phase_step': 81,
            'coverage': {'python': {'draws': 120, 'unique_indices': [3, 4], 'unique_examples': 2,
                                    'input_tokens': 6000, 'answer_tokens': 3000}},
            'last_evaluation': {'quality': quality}})
        (self.out / 'STOP').touch()
        state = journal.snapshot(self.root)
        self.assertEqual(state['status'], 'stopped')
        arm = state['arms']['protected']
        self.assertEqual(arm['phase_step'], 81)
        self.assertNotIn('unique_indices', arm['coverage']['python'])
        self.assertEqual(arm['coverage']['python']['draws'], 120)
        self.assertEqual(arm['quality']['examples']['math']['text'], '2')
        self.assertTrue((self.out / 'STOP').exists())

    def test_competing_execution_blocks_launch_and_never_touches_its_stop(self):
        self.put(self.out / 'request.json', {'steps_per_arm': 500, 'eval_every': 250, 'save_every': 50})
        (self.out / 'STOP').touch()
        with run_lock(self.out / 'execution'), patch.object(journal.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                journal.run_or_resume(self.root, self.lab)
            spawn.assert_not_called()
            self.assertTrue((self.out / 'STOP').exists())

    def test_owned_interrupt_writes_durable_stop(self):
        def run(root, **args):
            with coordinator.run_lock(self.out / 'execution'):
                self.out.mkdir()
                coordinator.atomic_json(self.out / 'request.json', dict(steps_per_arm=500))
                os.kill(os.getpid(), signal.SIGUSR1)
                self.assertTrue((self.out / 'STOP').is_file())
                return {'status': 'stopped'}
        with patch.object(coordinator, 'run', run):
            self.assertEqual(journal._worker(self.root, self.ready, 500), 'stopped')

    def test_interrupt_before_first_request_does_not_create_stray_output(self):
        def run(root, **args):
            with coordinator.run_lock(self.out / 'execution'):
                os.kill(os.getpid(), signal.SIGUSR1)
                self.fail('Early pause was ignored')
        with patch.object(coordinator, 'run', run):
            self.assertEqual(journal._worker(self.root, self.ready, 500), 'stopped')
        self.assertFalse(self.out.exists())

    def test_terminal_verification_uses_real_coordinator_and_rejects_changed_candidate(self):
        import torch
        from chat_recovery_data import sha
        source = self.root / 'broad_continuation_v1/latest.pt'
        corpus = self.root / 'corpus_repaired_v1'
        sft = self.root / 'chat_recovery_v1/data'
        for path in [source, self.out / 'source.pt', corpus / 'manifest.json',
                     corpus / 'tokenizer.model', sft / 'manifest.json',
                     self.out / 'protected/candidate.pt', self.out / 'qualification/final.json',
                     self.out / 'qualified_native/best.pt']:
            self.put(path, {'fixture': True})
        request = dict(version=coordinator.VERSION, source=str(source), source_sha256=sha(source),
                       steps_per_arm=5, eval_every=1, save_every=1,
                       code_sha256=coordinator.code_identity(), torch_version=str(torch.__version__),
                       sft_manifest_sha256=sha(sft / 'manifest.json'),
                       corpus_manifest_sha256=sha(corpus / 'manifest.json'),
                       tokenizer_sha256=sha(corpus / 'tokenizer.model'))
        self.put(self.out / 'request.json', request)
        candidate = self.out / 'protected/candidate.pt'
        final = {'identity': {'candidate_sha256': sha(candidate),
                              'suite_sha256': sha(self.out / 'qualification/final.json')},
                 'result': {'qualified': True}}
        self.put(self.out / 'final_result.json', final)
        self.put(self.out / 'report.json', {'status': 'qualified', 'request': request,
                                           'selected': 'protected', 'final': final})
        with patch.object(journal.subprocess, 'Popen') as spawn:
            self.assertEqual(journal.run_or_resume(self.root, self.lab)['status'], 'qualified')
            candidate.write_bytes(b'changed after evaluation')
            with self.assertRaisesRegex(RuntimeError, 'Final candidate or suite changed'):
                journal.run_or_resume(self.root, self.lab)
            source.write_bytes(b'changed after training')
            with self.assertRaisesRegex(RuntimeError, 'immutable'):
                journal.run_or_resume(self.root, self.lab)
            spawn.assert_not_called()

    def test_terminal_result_without_matching_request_is_rejected(self):
        self.put(self.out / 'report.json', {'status': 'qualified'})
        with self.assertRaisesRegex(RuntimeError, 'immutable request/source'):
            journal.run_or_resume(self.root, self.lab)

    def test_before_start_summary_shows_pending(self):
        with patch('sys.stdout', new_callable=io.StringIO) as display:
            journal.print_summary(self.root)
        self.assertIn('original pending', display.getvalue())
        self.assertNotIn('0 / None', display.getvalue())

    def test_tee_log_is_outside_new_experiment(self):
        class Finished:
            stdout = io.StringIO('ARM: original\nSAVED complete 2 / 2\n')
            def wait(self): return 0
            def poll(self): return 0
        def spawn(cmd, **options):
            self.assertFalse(self.out.exists())
            self.assertTrue(options['start_new_session'])
            Path(cmd[-2]).write_text('failed_quality')
            return Finished()
        with patch.object(journal.subprocess, 'Popen', spawn):
            self.assertEqual(journal.run_or_resume(self.root, self.lab)['status'], 'failed_quality')
        self.assertIn('SAVED complete', (self.root / 'coherence_continuation_journal.log').read_text())


if __name__ == '__main__':
    unittest.main()
