"""Synthetic CPU checks, without a training step or user data."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sentencepiece as spm
import torch

import chat_recovery_data as data
from moe_model import PhiStateMoELM, config_for
import phistate_a100_diagnostics as audit
import prompt_conditioning_eval as conditioning


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        cls.sft=cls.root/'sft'; cls.sft.mkdir()
        stream=io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter(['A short example question. Some words and Python code.']*20),
            model_writer=stream, vocab_size=350, model_type='bpe', hard_vocab_limit=False, byte_fallback=True,
            normalization_rule_name='identity', add_dummy_prefix=False, remove_extra_whitespaces=False,
            user_defined_symbols=['<|general|>','<|code|>'], minloglevel=2)
        tok=cls.sft/'tokenizer.model'; tok.write_bytes(stream.getvalue())
        tokenizer=spm.SentencePieceProcessor(model_file=str(tok))
        rows=[]
        for source in data.SOURCES:
            for i in range(2):
                msg=[dict(role='user',content=f'{source} question {i}?'),dict(role='assistant',content=f'A short answer {i}.')]
                x,y=data.encode_example(msg,tokenizer,256)
                rows.append(dict(split='val',source=source,id=data.digest(data.canonical(msg)),messages=msg,input_ids=x,labels=y))
        val=cls.sft/'val.jsonl'; val.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=dict(template=data.TEMPLATE,context=256,base_data_fingerprint='fixture-corpus',
                      files={'val.jsonl':data.sha(val)},tokenizer_sha256=data.sha(tok))
        manifest['data_fingerprint']=data.digest(data.canonical(manifest))
        (cls.sft/'manifest.json').write_text(json.dumps(manifest))
        cfg=config_for('mini',tokenizer.vocab_size()); model=PhiStateMoELM(cfg)
        state=dict(format='PhiStateMoE-v1',config=vars(cfg),state_dict=model.state_dict(),step=100,
                   data_fingerprint='fixture-corpus',tokenizer_sha256=data.sha(tok),sft_data_fingerprint=manifest['data_fingerprint'],
                   chat_template=data.TEMPLATE,sft_steps=1)
        cls.parent=cls.root/'parent.pt'; torch.save(state,cls.parent)
        # Known synthetic displacement for verifying the later read-only measurement.
        with torch.no_grad(): model.embedding.weight.add_(.125)
        state.update(step=101,sft_steps=2,source_sha256=data.sha(cls.parent))
        cls.source=cls.root/'source.pt'; torch.save(state,cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_probe_gradients_displacement_and_identity(self):
        hash_before=data.sha(self.source); rng_before=torch.get_rng_state().clone()
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimizer step')), \
             patch.object(torch.optim.Adafactor,'step',side_effect=AssertionError('optimizer step')), \
             torch.no_grad():
            r=audit.run(self.source,self.sft,self.root/'local_reports',limit=2,comparison_checkpoint=self.parent)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng_before))
        self.assertEqual(data.sha(self.source),hash_before)
        self.assertEqual(r['optimizer_steps_taken'],0)
        self.assertEqual(r['prompt_conditioning']['source'],'measured_on_loaded_weights')
        g=r['validation_gradient_probe']; self.assertGreater(g['global_gradient_l2'],0)
        self.assertEqual(len(g['rows']),4)
        for layer in g['routing_by_layer']:
            self.assertAlmostEqual(sum(layer['assignment_share']),1.)
        groups=g['by_parameter_class']
        self.assertAlmostEqual(groups['tied_embedding_output']['parent_to_current_net_delta_rms'],.125,places=5)
        self.assertEqual(groups['routed_experts']['parent_to_current_net_delta_rms'],0.)
        self.assertIn('recurrent_projection',groups)
        self.assertGreater(groups['router']['gradient_rms'],0)
        first=next(iter(r['recurrent_half_life_tokens'].values()))
        self.assertAlmostEqual(first['min'],.69314718/.2501,places=3)
        self.assertAlmostEqual(first['max'],.69314718/.0201,places=3)
        with patch.object(audit,'ckpt_load',side_effect=AssertionError('cached result should bypass model load')):
            self.assertEqual(audit.run(self.source,self.sft,self.root/'local_reports',limit=2,comparison_checkpoint=self.parent),r)

    def test_verified_existing_conditioning_and_wrong_parent(self):
        directory=self.root/'prior_conditioning'
        expected=conditioning.run(self.source,self.sft,directory,limit=2)
        with patch.object(audit,'_measure_conditioning',side_effect=AssertionError('prior report should be reused')):
            result=audit.run(self.source,self.sft,self.root/'cached_reports',limit=2,conditioning_dir=directory)
        self.assertEqual(result['prompt_conditioning']['source'],'verified_existing_report')
        self.assertEqual(result['prompt_conditioning']['report_sha256'],expected['report_sha256'])
        self.assertEqual(result['prompt_conditioning']['macro'],expected['macro'])
        with self.assertRaisesRegex(ValueError,'direct parent'):
            audit.run(self.source,self.sft,self.root/'wrong_parent',limit=2,comparison_checkpoint=self.source)
        prior=Path(expected['report_path']); damaged=json.loads(prior.read_text());damaged['macro']={}
        prior.write_text(json.dumps(damaged)); self.assertIsNone(audit._read_conditioning(directory,{}))


if __name__ == '__main__': unittest.main()
