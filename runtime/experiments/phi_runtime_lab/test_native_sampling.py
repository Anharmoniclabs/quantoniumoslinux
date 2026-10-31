"""Native sampling checks; forward logits and the legacy greedy API stay intact."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from moe_model import PhiStateMoELM, config_for
from native_cpp_trial import (NATIVE, compile_native, decode_arguments, export_model,
                              native_run, parity)


class NativeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261031)
        cls.folder = tempfile.TemporaryDirectory(prefix='native_sampling_test_')
        cls.root = Path(cls.folder.name)
        cls.binary = cls.root / 'phistate_moe_cli'
        cls.weights = cls.root / 'model.phimoe'
        compile_native(cls.binary)
        cls.model = PhiStateMoELM(config_for('mini', 400)).eval()
        export_model(cls.model, cls.weights)
        cls.ids = [7, 13, 9, 21, 4, 17, 11, 6]
        cls.previous_threads = os.environ.get('RFTMW_THREADS')
        os.environ['RFTMW_THREADS'] = '2'

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        if cls.previous_threads is None:
            os.environ.pop('RFTMW_THREADS', None)
        else:
            os.environ['RFTMW_THREADS'] = cls.previous_threads

    def run_native(self, config=None, count=24, eos=-1):
        return native_run(self.binary, self.weights, self.ids, 400, self.root,
                          count, eos, config)

    def test_legacy_greedy_and_top_k_one_are_identical(self):
        logits, legacy = self.run_native()
        explicit_logits, explicit = self.run_native({})
        _, top_one = self.run_native(dict(temperature=.8, top_k=1, seed=98))
        self.assertTrue(np.array_equal(logits, explicit_logits))
        self.assertEqual(legacy['generated_ids'], explicit['generated_ids'])
        self.assertEqual(legacy['generated_ids'], top_one['generated_ids'])
        self.assertTrue(parity(self.model, self.ids, logits)['passed'])

    def test_seeded_sampling_repeats_and_respects_token_bounds(self):
        config = dict(temperature=.8, top_p=.95, top_k=50, repetition_penalty=1.1, seed=654)
        _, a = self.run_native(config)
        _, b = self.run_native(config)
        self.assertEqual(a['generated_ids'], b['generated_ids'])
        self.assertEqual(len(a['generated_ids']), 24)
        self.assertTrue(all(0 <= value < 400 for value in a['generated_ids']))
        self.assertEqual(a['decoding_api'], 1)
        # Validate native logits along a sampled trajectory, not just the prompt.
        trajectory = self.ids + a['generated_ids']
        logits, _ = native_run(self.binary, self.weights, trajectory, 400, self.root)
        self.assertTrue(parity(self.model, trajectory, logits)['passed'])

    def test_eos_stops_without_appending_and_zero_count_is_allowed(self):
        _, first = self.run_native(count=1)
        eos = first['generated_ids'][0]
        _, ended = self.run_native(count=24, eos=eos)
        self.assertEqual(ended['generated_ids'], [])
        self.assertTrue(ended['eos'])
        _, empty = self.run_native(count=0)
        self.assertEqual(empty['generated_ids'], [])
        self.assertFalse(empty['eos'])

    def test_invalid_config_rejected_by_python_and_native(self):
        for config in [dict(temperature=float('nan')), dict(temperature=-1),
                       dict(top_p=0), dict(top_p=1.1), dict(top_k=-1),
                       dict(top_k=1.5), dict(repetition_penalty=.9),
                       dict(seed=-1), dict(seed=2**64), dict(unknown=1)]:
            with self.subTest(config=config), self.assertRaises((ValueError, TypeError)):
                decode_arguments(config)
        tokens = self.root / 'invalid_tokens.u32'
        np.asarray(self.ids, dtype='<u4').tofile(tokens)
        base = [str(self.binary), str(self.weights), str(tokens), str(self.root / 'invalid.f32'), '2', '-1']
        for extra in [['nan', '1', '0', '1', '0'], ['1', '0', '0', '1', '0'],
                      ['1', '1', '-1', '1', '0'], ['1', '1', '0', '1', '-1'],
                      ['1', '1', '0x', '1', '0'], ['1', '1', '0', '.5', '0']]:
            result = subprocess.run(base + extra, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, extra)
        bad_count = base.copy()
        bad_count[-2] = '4097'
        self.assertNotEqual(subprocess.run(bad_count, capture_output=True).returncode, 0)

    def test_sampler_penalty_sign_eos_exemption_and_nucleus_threshold(self):
        # Include the implementation directly to exercise controlled logits;
        # no model parameter configuration is needed to force edge conditions.
        source = self.root / 'sampler_unit.cpp'
        include = json.dumps(str(NATIVE / 'phistate_moe_cli.cpp'))
        source.write_text('#define main phistate_cli_entry\n#include ' + include + '''
#undef main
#include <cassert>
int main() {
    DecodeConfig cfg; cfg.repetition_penalty=2;
    std::mt19937_64 rng(7);
    std::vector<bool> seen{true,false};
    assert(choose_token(Vec{2,1.5},seen,-1,cfg,rng)==1);
    assert(choose_token(Vec{-1,-1.5},seen,-1,cfg,rng)==1);
    assert(choose_token(Vec{2,1.5},seen,0,cfg,rng)==0);
    assert(choose_token(Vec{2,1.5},std::vector<bool>{false,false},-1,cfg,rng)==0);
    cfg.temperature=1; cfg.repetition_penalty=1; cfg.top_p=.7;
    Vec logits{float(std::log(.6)),float(std::log(.3)),float(std::log(.1))};
    int counts[3]={0,0,0};
    for(int i=0;i<2000;++i) ++counts[choose_token(logits,std::vector<bool>(3,false),-1,cfg,rng)];
    assert(counts[0]>counts[1] && counts[1]>0 && counts[2]==0);
    assert(double(counts[0])/2000>.62 && double(counts[0])/2000<.71);
    cfg.top_p=.5;
    for(int i=0;i<20;++i) assert(choose_token(logits,std::vector<bool>(3,false),-1,cfg,rng)==0);
    bool rejected=false;
    try { choose_token(Vec{std::numeric_limits<float>::quiet_NaN()},std::vector<bool>(1,false),-1,cfg,rng); }
    catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
''')
        binary = self.root / 'sampler_unit'
        subprocess.run(['g++', '-O2', '-std=c++17', '-pthread', str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
