"""Offline checks for frozen evaluation, safe code grading and decoding."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import coherence_quality as quality


class Tokenizer:
    def encode(self, text): return [0]
    def eos_id(self): return 3
    def decode(self, ids): return ''.join('abc'[i] for i in ids)


class Model(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.scores = scores

    def step(self, token, state):
        index = 0 if state is None else state
        return torch.tensor([[self.scores[min(index, len(self.scores)-1)]]]), index+1


class QualityTests(unittest.TestCase):
    def item(self, bucket, number=0, split='dev'):
        return [i for i in quality._bank(split) if i['bucket'] == bucket][number]

    def test_correct_paraphrases_and_bad_answers(self):
        ram = self.item('knowledge')
        self.assertTrue(quality.grade_answer(ram, 'RAM is temporary memory for data used by active programs.')['passed'])
        self.assertTrue(quality.grade_answer(ram, 'It temporarily holds information for running programs.')['passed'])
        self.assertFalse(quality.grade_answer(ram, "Hello, I'm here to help.")['passed'])
        self.assertFalse(quality.grade_answer(ram, 'RAM is not temporary storage for data.')['passed'])
        self.assertFalse(quality.grade_answer(ram, 'The Sun is a star.')['passed'])
        self.assertFalse(quality.grade_answer(ram, ' ')['passed'])
        self.assertTrue(quality.repetition_failure('The number of the number of the number of the number of the number.'))
        self.assertTrue(quality.repetition_failure(' '*80))
        self.assertFalse(quality.repetition_failure('RAM temporarily holds data for running programs.'))

    def test_nonword_repetition_and_code_indentation(self):
        for text in ('et'*30, ')'*40, '哈哈哈'*12, '\ufffd'*32, '同同据'*12):
            self.assertTrue(quality.repetition_failure(text), repr(text))
        for text in (
            'def maximum(a, b):\n    if a > b:\n        return a\n    return b',
            'def add(a, b):\n        result = a + b\n        return result',
            'A short emphasis: ha ha ha.',
            'A short prefix ' + 'x'*12 + ' followed by a normal explanatory sentence.',
        ):
            self.assertFalse(quality.repetition_failure(text), repr(text))

    def test_exact_and_arithmetic(self):
        item = self.item('instruction')
        self.assertTrue(quality.grade_answer(item, 'mango')['passed'])
        self.assertFalse(quality.grade_answer(item, 'The answer is mango.')['passed'])
        for split in ('dev', 'final'):
            for item in quality._bank(split):
                if item['rubric']['kind'] == 'exact':
                    self.assertTrue(quality.grade_answer(item, item['rubric']['expected'])['passed'])
                    self.assertFalse(quality.grade_answer(item, 'wrong')['passed'])

    def test_every_rubric_accepts_a_correct_reference(self):
        definitions = [
            'RAM temporarily holds data for running programs.',
            'Storage keeps files even when the power is off.',
            'The CPU executes program instructions.',
            'Python is a programming language.',
            'A variable is a name that refers to a value.',
            'An operating system manages hardware resources.',
            'Plants use sunlight to make sugar during photosynthesis.',
            'Gravity is a force that pulls objects toward Earth.',
        ]
        expressions = ['a+b', 'a-b', 'a*b', 'x*x', 'x % 2 == 0', 'max(a,b)', 'abs(x)', '2*x']
        for split in ('dev', 'final'):
            for item in quality._bank(split):
                rubric = item['rubric']
                number = int(item['id'].rsplit('-', 1)[1])
                if rubric['kind'] == 'keywords':
                    reference = definitions[number]
                elif rubric['kind'] == 'exact':
                    reference = rubric['expected']
                else:
                    reference = 'def ' + rubric['function'] + '(' + ','.join(rubric['args']) + '):\n    return ' + expressions[number]
                self.assertTrue(quality.grade_answer(item, reference)['passed'], (item['id'], reference))

    def test_code_behavior_and_safety(self):
        add = self.item('code')['rubric']
        self.assertTrue(quality.code_passes('```python\ndef add(a, b):\n    return a + b\n```', add))
        self.assertTrue(quality.code_passes('def add(a: int, b: int) -> int:\n    result = a+b\n    return result', add))
        for bad in [
            'def add(a,b):\n    return 0',
            'def add(a,b):\n    return a-b',
            'def subtract(a,b):\n    return a+b',
            'def add(a,b):\n    return True',
            'import os\ndef add(a,b):\n    return a+b',
            'def add(a,b):\n    __import__("os").system("touch /tmp/never")\n    return a+b',
            'def add(a,b):\n    if False:\n        import os\n    return a+b',
            '@danger\ndef add(a,b):\n    return a+b',
            'def add(a: danger(),b=1):\n    return a+b',
            'def add(a,b):\n    while True: pass\n    return a+b',
            'def add(a,b):\n    return (lambda: a+b)()',
            'def add(a,b):\n    return [a,b][0]',
            'def add(a,b):\n    return a +',
            'def add(a,b):\n    return\n    return a+b',
        ]:
            self.assertFalse(quality.code_passes(bad, add), bad)
        maximum = self.item('code', 5)['rubric']
        self.assertTrue(quality.code_passes('def maximum(a,b):\n    if a > b:\n        return a\n    return b', maximum))
        self.assertTrue(quality.code_passes('def maximum(a,b):\n    return max(a,b)', maximum))
        even = self.item('code', 4)['rubric']
        self.assertTrue(quality.code_passes('def is_even(x):\n    return x % 2 == 0', even))
        self.assertFalse(quality.code_passes('def is_even(x):\n    return 1', even))

    def test_frozen_disjoint_suites_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Force fallback wording for one otherwise frozen dev prompt.
            occupied = quality._bank('dev')[0]['prompt']
            row = dict(messages=[dict(role='user', content=occupied), dict(role='assistant', content='example')])
            (root/'train.jsonl').write_text(json.dumps(row)+'\n')
            (root/'val.jsonl').write_text('')
            paths = quality.build_suites(root, root/'suites')
            dev, final = (quality.load_suite(paths[k]) for k in ('dev', 'final'))
            dp = {quality.normalized(i['prompt']) for i in dev['items']}
            fp = {quality.normalized(i['prompt']) for i in final['items']}
            self.assertEqual(len(dp), 32)
            self.assertFalse(dp & fp)
            self.assertNotIn(quality.normalized(occupied), dp | fp)
            before = paths['dev'].read_bytes()
            quality.build_suites(root, root/'suites')
            self.assertEqual(before, paths['dev'].read_bytes())
            tampered = copy.deepcopy(dev)
            tampered['items'][0]['prompt'] = 'changed'
            paths['dev'].write_text(json.dumps(tampered))
            with self.assertRaises(ValueError): quality.load_suite(paths['dev'])
            with self.assertRaises(ValueError): quality.build_suites(root, root/'suites')

    def test_decoding_determinism_rng_preservation_and_penalty(self):
        tokenizer = Tokenizer()
        model = Model([[3., 2., 1., -5.], [1., 2., 3., -5.], [0., 0., 0., 10.]])
        model.train()
        before = torch.get_rng_state().clone()
        a = quality.generate_answer(model, tokenizer, 'question', decode_config={'temperature': .8}, seed=17)
        b = quality.generate_answer(model, tokenizer, 'question', decode_config={'temperature': .8}, seed=17)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)
        # Repeated negative logits must become less likely, not more likely.
        model = Model([[10., -5., -6., -20.], [-1., -1.2, -6., -20.], [-9., -9., -9., 10.]])
        self.assertEqual(quality.generate_answer(model, tokenizer, 'q', decode_config={'repetition_penalty': 2.})['text'], 'ab')
        self.assertEqual(quality.generate_answer(model, tokenizer, 'q')['text'], 'aa')
        self.assertEqual(quality.generate_answer(model, tokenizer, 'q', decode_config={'temperature': 1., 'top_k': 1})['text'], 'aa')
        with self.assertRaises(ValueError):
            quality.generate_answer(model, tokenizer, 'q', decode_config={'unsupported': True})
        with self.assertRaises(ValueError):
            quality.generate_answer(model, tokenizer, 'q', decode_config={'top_k': 1.5})

    def test_qualification_and_empty_output(self):
        suite = dict(format=quality.FORMAT, split='dev', criteria=quality.CRITERIA,
                     items=quality._bank('dev'))
        suite['fingerprint'] = quality.fingerprint(suite)
        with patch.object(quality, 'generate_answer', return_value=dict(text='', tokens=0, eos=True)):
            result = quality.evaluate_suite(None, None, suite)
        self.assertFalse(result['qualified'])
        self.assertEqual(result['macro_score'], 0.)
        self.assertEqual(result['nonempty_rate'], 0.)
        self.assertEqual(len(result['rows']), 32)
        with patch.object(quality, 'generate_answer', return_value=dict(text='ok', tokens=1, eos=True)), \
             patch.object(quality, 'grade_answer', return_value=dict(passed=True, nonempty=True, repetition_failure=False)):
            result = quality.evaluate_suite(None, None, suite)
        self.assertTrue(result['qualified'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
