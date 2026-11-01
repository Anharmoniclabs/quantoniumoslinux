"""Conversational probe contract and controlled CPU greedy evaluation."""

import unittest
from collections import Counter
from unittest.mock import patch

import torch

import a100_dialogue_checks as dialogue
from chat_recovery_data import canonical, digest, prompt_text


class HistoryTokenizer:
    """A complete history gets its own marker. The final user turn alone does not."""

    def __init__(self, pairs):
        self.lookup = {}
        self.answers = {}
        for pair in pairs:
            for item in pair['items']:
                index = len(self.lookup)
                self.lookup[prompt_text(item['messages'])] = index
                self.answers[100 + index] = item['reference']

    def encode(self, text):
        return [10 + self.lookup[text], 4]

    def decode(self, ids):
        return ' '.join(self.answers[i] for i in ids)

    def eos_id(self):
        return 2


class HistoryFixture(torch.nn.Module):
    """The neutral final token forces the fixture to consult its prior-state marker."""

    def __init__(self, ignore_change=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.ignore_change = ignore_change

    def step(self, tokens, cache):
        t = tokens[:, 0]
        previous = torch.zeros_like(t) if cache is None else cache
        marker = torch.where((t >= 10) & (t < 34), t - 10, previous)
        decision = marker - (marker % 2 if self.ignore_change else 0)
        predicted = torch.where(t == 4, 100 + decision, torch.full_like(t, 2))
        logits = torch.full((len(t), 1, 128), -10.0)
        logits[:, :, 2] = 0.0
        logits[torch.arange(len(t)), 0, predicted] = 10.0
        return logits, marker


class DialogueChecksTests(unittest.TestCase):
    def test_design_split_and_counterfactual_histories(self):
        prior_ids = set()
        prior_gold = set()
        prior_questions = set()
        for split in ('dev', 'final'):
            pairs = dialogue.bank(split)
            self.assertEqual(pairs, dialogue.bank(split))
            self.assertEqual(Counter(p['family'] for p in pairs), {f: 3 for f in dialogue.FAMILIES})
            answers = set()
            questions = set()
            histories = set()
            lengths = set()
            for pair in pairs:
                self.assertNotIn(pair['pair_id'], prior_ids)
                prior_ids.add(pair['pair_id'])
                left, right = pair['items']
                histories_a = left['messages']
                histories_b = right['messages']
                self.assertEqual(histories_a[-1], histories_b[-1])
                self.assertNotIn(histories_a[-1]['content'], questions)
                questions.add(histories_a[-1]['content'])
                changes = [i for i, (x, y) in enumerate(zip(histories_a, histories_b)) if x != y]
                self.assertEqual(len(changes), 1)
                self.assertLess(changes[0], len(histories_a) - 1)
                self.assertEqual(histories_a[changes[0]]['role'], 'user')
                self.assertEqual(len(histories_a), len(histories_b))
                for item in (left, right):
                    turns = item['messages']
                    lengths.add(sum(t['role'] == 'user' for t in turns))
                    formatted = prompt_text(turns)
                    self.assertTrue(formatted.endswith('Assistant:\n'))
                    self.assertNotIn(formatted, histories)
                    histories.add(formatted)
                    answers.add(item['reference'].casefold())
                self.assertFalse(set(left['accepted']) & set(right['accepted']))
            self.assertEqual(lengths, {2, 3})
            self.assertTrue(answers.isdisjoint(prior_gold))
            self.assertTrue(questions.isdisjoint(prior_questions))
            prior_gold.update(answers)
            prior_questions.update(questions)
        with self.assertRaises(ValueError):
            dialogue.bank('test')

    def test_declared_aliases_no_keyword_or_negation_shortcuts(self):
        for split in ('dev', 'final'):
            for pair in dialogue.bank(split):
                a, b = pair['items']
                for own, wrong in [(a, b), (b, a)]:
                    for accepted in own['accepted']:
                        self.assertTrue(dialogue.grade(own, '  ' + accepted.upper() + '.  '))
                        self.assertFalse(dialogue.grade(wrong, accepted))
                    self.assertFalse(dialogue.grade(own, 'not ' + own['reference']))
                    self.assertFalse(dialogue.grade(own, own['reference'] + ' or ' + wrong['reference']))
        numbered = dialogue.bank('dev')[5]['items'][0]
        self.assertTrue(dialogue.grade(numbered, 'Locker 4.'))
        self.assertFalse(dialogue.grade(numbered, 'It might be 4.'))

    def test_gold_without_eos_does_not_pass(self):
        pairs = dialogue.bank('dev')
        tokenizer = HistoryTokenizer(pairs)
        answers = [
            {'text': item['reference'], 'tokens': 24, 'eos': False}
            for pair in pairs
            for item in pair['items']
        ]
        with patch('coherence_quality.generate_answer', side_effect=answers):
            score = dialogue.evaluate(HistoryFixture(), tokenizer, 'dev')
        self.assertEqual((score['items_passed'], score['pairs_passed'], score['eos_items']), (0, 0, 0))
        self.assertTrue(all(i['text_passed'] for p in score['rows'] for i in p['items']))

    def test_cpu_greedy_history_and_eos_pair_aggregation(self):
        for split in ('dev', 'final'):
            pairs = dialogue.bank(split)
            tokenizer = HistoryTokenizer(pairs)
            model = HistoryFixture()
            score = dialogue.evaluate(model, tokenizer, split)
            self.assertTrue(model.training)
            self.assertEqual((score['pairs_passed'], score['pairs_total']), (12, 12))
            self.assertEqual((score['items_passed'], score['eos_items']), (24, 24))
            self.assertEqual(score['suite_fingerprint'], digest(canonical(pairs)))
            self.assertEqual(score['decoding']['backend'], 'serial_greedy')
            for row in score['rows']:
                self.assertEqual([item['tokens'] for item in row['items']], [1, 1])
                self.assertTrue(all(item['text'] == item['reference'] for item in row['items']))
            blind = dialogue.evaluate(HistoryFixture(ignore_change=True), tokenizer, split)
            self.assertEqual((blind['items_passed'], blind['pairs_passed']), (12, 0))
            self.assertEqual(blind['eos_items'], 24)


if __name__ == '__main__':
    unittest.main()
