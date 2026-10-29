"""Contracts for pretrained flow, real-video splits and checkpoint isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch

from open_flow import guides_from_flows, sha256, RAFT_SHA256
from prepare_open_video import reject_triplet
from train_open_v08 import OpenData, train
from train_v07 import PoseInterpolator


class OpenVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flow_direction_for_translation(self):
        ab = torch.zeros(1,2,128,128)
        ab[:,0] = 8
        g = guides_from_flows(ab,-ab,.25)
        self.assertTrue(torch.equal(g[:,9],torch.full_like(g[:,9],-2)))
        self.assertTrue(torch.equal(g[:,11],torch.full_like(g[:,11],6)))
        self.assertEqual(g[:,:9].abs().sum(),0)

    def test_exact_endpoints_with_external_guides(self):
        m = PoseInterpolator(128,4)
        a,b = torch.rand(2,4,128,128).unbind()
        g = guides_from_flows(torch.ones(1,2,128,128),-torch.ones(1,2,128,128),.5)
        for t,target in ((0,a),(1,b)):
            self.assertTrue(torch.equal(m(a[None],b[None],t,0.,g)[0],target[None]))

    def test_cut_and_blank_rejected(self):
        frames = torch.randint(0,256,(3,3,128,128),dtype=torch.uint8)
        self.assertTrue(reject_triplet(frames))
        self.assertTrue(reject_triplet(torch.zeros_like(frames)))
        self.assertFalse(reject_triplet(frames[:1].expand(3,-1,-1,-1)))

    def fixture(self,root):
        records = []
        for name,split in [('bunny','train'),('sintel','validation')]:
            file = name+'.pt'
            torch.save(dict(frames=torch.randint(0,256,(3,3,128,128),dtype=torch.uint8),
                ab=torch.zeros(2,128,128),ba=torch.zeros(2,128,128)),root/file)
            records.append(dict(file=file,sha256=sha256(root/file),source=name,split=split,t=.5))
        manifest = dict(schema=1,size=128,teacher_sha256=RAFT_SHA256,records=records,
            sources={'bunny':{'split':'train'},'sintel':{'split':'validation'}})
        (root/'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_loader_rgb_and_split_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            data = OpenData(root,128)
            a,y,b,g = data.tensors(data.items['train'])
            self.assertEqual(tuple(a.shape),(1,4,128,128))
            self.assertTrue(torch.equal(y[:,3],torch.ones_like(y[:,3])))
            self.assertEqual(tuple(g.shape),(1,13,128,128))

    def test_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with (root/'bunny.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                OpenData(root,128)

    def test_target_cannot_change_endpoint_guidance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            data = OpenData(root,128)
            a,y,b,g = data.tensors(data.items['train'])
            data.items['train'][0]['frames'][1].zero_()
            aa,yy,bb,gg = data.tensors(data.items['train'])
            self.assertFalse(torch.equal(y,yy))
            self.assertTrue(all(torch.equal(x,z) for x,z in ((a,aa),(b,bb),(g,gg))))

    def test_original_run_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(threads=2,init=str(Path(folder)/'latest.pt'),out=folder)
            with self.assertRaisesRegex(ValueError,'separate output'):
                train(args)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            m = self.fixture(root)
            m['records'][1]['source'] = 'bunny'
            (root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'split'):
                OpenData(root,128)

    def test_bad_time_rejected(self):
        with self.assertRaises(ValueError):
            guides_from_flows(torch.zeros(1,2,128,128),torch.zeros(1,2,128,128),1.5)


if __name__ == '__main__':
    unittest.main()
