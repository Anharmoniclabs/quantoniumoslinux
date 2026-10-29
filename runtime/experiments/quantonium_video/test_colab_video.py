"""Snapshot interruption/recovery and optional CUDA execution contracts."""
import json
from pathlib import Path
import tempfile
import unittest
import torch

from colab_video_runtime import save_snapshot, restore_latest, verify_snapshot
from train_v07 import PoseInterpolator
from train_open_v08 import external_eval


def fixture(run,step):
    run.mkdir(exist_ok=True)
    state = dict(step=step,config={'version':'test'},model={'x':torch.ones(1)})
    for name in ('latest.pt','best.pt'):
        torch.save(state,run/name)
    (run/'metrics_latest.json').write_text(json.dumps({'phase_step':step}))
    (run/'run_manifest.json').write_text(json.dumps({'config':state['config']}))


class SnapshotTests(unittest.TestCase):
    def test_roundtrip_and_incomplete_write_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            saved=save_snapshot(run,root/'drive')
            (root/'drive/snapshots/incomplete').mkdir()
            restored=restore_latest(root/'drive',root/'restore',{'version':'test'})
            self.assertEqual(restored,saved)
            self.assertEqual(torch.load(root/'restore/latest.pt',weights_only=True)['step'],100)

    def test_corrupt_newer_snapshot_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            old=save_snapshot(run,root/'drive')
            fixture(run,125);new=save_snapshot(run,root/'drive')
            (new/'latest.pt').write_bytes(b'interrupted transfer')
            self.assertEqual(restore_latest(root/'drive',root/'restore',{'version':'test'}),old)

    def test_mismatched_lineage_cannot_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            save_snapshot(run,root/'drive')
            with self.assertRaisesRegex(ValueError,'silent reset'):
                restore_latest(root/'drive',root/'restore',{'version':'other'})

    def test_newer_local_checkpoint_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            save_snapshot(run,root/'drive');fixture(run,125)
            with self.assertRaisesRegex(ValueError,'newer'):
                restore_latest(root/'drive',run,{'version':'test'})
            self.assertEqual(torch.load(run/'latest.pt',weights_only=True)['step'],125)

    def test_nonfinite_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';fixture(run,100)
            s=torch.load(run/'latest.pt',weights_only=True);s['model']['x'].fill_(float('nan'))
            torch.save(s,run/'latest.pt')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                save_snapshot(run,root/'drive')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware not present')
    def test_cuda_forward_backward_and_evaluation(self):
        model=PoseInterpolator(128,4).cuda()
        a=torch.rand(1,4,128,128,device='cuda');b=torch.rand_like(a)
        g=torch.zeros(1,13,128,128,device='cuda')
        pred,*_=model(a,b,.5,0.,g);pred.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        class Data:
            items={'validation':[None]}
            def tensors(self,values):
                return a.cpu(),((a+b)/2).cpu(),b.cpu(),g.cpu()
        self.assertTrue(torch.isfinite(torch.tensor(external_eval(model,Data())['model_l1'])))


if __name__ == '__main__':
    unittest.main()
