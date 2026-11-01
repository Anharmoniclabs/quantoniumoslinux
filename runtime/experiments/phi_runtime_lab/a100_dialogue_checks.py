"""Matched, held-out conversational recall with fixed assistant acknowledgments.

Each pair has the same last question and a different earlier user statement.
Cases excuse only the explicitly enumerated short answer forms, ignoring case,
repeated whitespace, and a terminal period. Passing also requires generated EOS. This is a bounded test of using
chat history. Free-form conversational quality needs separate review.
"""

from collections import Counter

from chat_recovery_data import canonical, digest, prompt_text

VERSION = 'phistate-contextual-dialogue-v1'
FAMILIES = ('stated_fact', 'correction', 'prior_reference', 'instruction')
LIMIT = 24

# Each row gives a family, user turns, the two possible values in the changed
# turn, the corresponding accepted short answers, and optional answer wrappers.
# Earlier assistant text is fixed within each pair and never supplies the answer.
DEV = (
    (
        'stated_fact',
        ('Call me {v} when I sign in.', 'Which nickname did I give you?'),
        ('Leto', 'Rumi'),
        ('Leto', 'Rumi'),
        ('You asked me to call you {}',),
    ),
    (
        'stated_fact',
        ('For this rehearsal, I left the toolbox in the {v}.', 'Where did I put the toolbox?'),
        ('pantry', 'attic'),
        ('pantry', 'attic'),
        ('the {}', 'in the {}'),
    ),
    (
        'stated_fact',
        ('Bren carries the blue bag. The red bag goes to {v}.', 'Who gets the red bag?'),
        ('Suki', 'Noor'),
        ('Suki', 'Noor'),
        ('The red bag goes to {}',),
    ),
    (
        'correction',
        (
            'I wrote Tuesday on the meeting card.',
            'Correction: move that meeting to {v}.',
            'What day is the meeting now?',
        ),
        ('Thursday', 'Sunday'),
        ('Thursday', 'Sunday'),
        ('on {}', 'It is on {}'),
    ),
    (
        'correction',
        (
            'I ordered a tin of cobalt paint.',
            'Change the paint to {v}, keeping the same tin size.',
            'Which color did I settle on?',
        ),
        ('ochre', 'umber'),
        ('ochre', 'umber'),
        ('The color is {}',),
    ),
    (
        'correction',
        (
            'I put a cello in locker seven.',
            'Update its locker number to {v}.',
            'Which locker number is current?',
        ),
        ('four', 'nine'),
        (('four', '4'), ('nine', '9')),
        ('locker {}',),
    ),
    (
        'prior_reference',
        (
            'Track the trip led by {v}.',
            'Ada finishes at the pier. Milo finishes at the mill.',
            'Where does the trip I selected finish?',
        ),
        ('Ada', 'Milo'),
        ('pier', 'mill'),
        ('the {}', 'at the {}'),
    ),
    (
        'prior_reference',
        (
            'Keep an eye on the {v} ticket.',
            'The copper ticket opens the vault. The silver ticket opens the loft.',
            'Which room does my ticket open?',
        ),
        ('copper', 'silver'),
        ('vault', 'loft'),
        ('the {}',),
    ),
    (
        'prior_reference',
        (
            'I chose the {v}.',
            'The violin bears a star, and the drum bears a circle.',
            'Which mark is on my instrument?',
        ),
        ('violin', 'drum'),
        ('star', 'circle'),
        ('a {}', 'the {}'),
    ),
    (
        'instruction',
        (
            'For my next list, repeat the {v} entry.',
            'My list is plum, pear, lime.',
            'Using my rule, which entry do you send?',
        ),
        ('first', 'last'),
        ('plum', 'lime'),
        (),
    ),
    (
        'instruction',
        (
            'When you hear the cue, give me the {v}.',
            'Our swatch is purple linen.',
            'Here is the cue. What should you say?',
        ),
        ('color', 'material'),
        ('purple', 'linen'),
        (),
    ),
    (
        'instruction',
        (
            'For the next pair of labels, output the {v} label.',
            'The labels read kite and anchor.',
            'Apply my earlier rule to those labels.',
        ),
        ('shorter', 'longer'),
        ('kite', 'anchor'),
        (),
    ),
)
FINAL = (
    (
        'stated_fact',
        ('Put {v} on the badge I will wear.', 'What name should appear on my badge?'),
        ('Oona', 'Pavel'),
        ('Oona', 'Pavel'),
        ('The name is {}',),
    ),
    (
        'stated_fact',
        ('I left our map inside the {v}. My gloves are at the door.', 'Where can I find our map?'),
        ('shed', 'cabin'),
        ('shed', 'cabin'),
        ('the {}', 'in the {}'),
    ),
    (
        'stated_fact',
        ('The scorekeeper wears gray. Our referee is {v}.', 'Who is refereeing our game?'),
        ('Ines', 'Amir'),
        ('Ines', 'Amir'),
        ('The referee is {}',),
    ),
    (
        'correction',
        ('I planned to meet at dawn.', 'Meet at {v} instead, please.', 'What time have we agreed to meet?'),
        ('noon', 'dusk'),
        ('noon', 'dusk'),
        ('at {}',),
    ),
    (
        'correction',
        ('My draft calls the boat Minnow.', 'I have renamed the boat {v}.', 'What is the boat called now?'),
        ('Wren', 'Fern'),
        ('Wren', 'Fern'),
        ('The boat is called {}',),
    ),
    (
        'correction',
        (
            'My invitation assigns us table six.',
            'We have been moved to table {v}.',
            'Which table should we use?',
        ),
        ('two', 'eight'),
        (('two', '2'), ('eight', '8')),
        ('table {}',),
    ),
    (
        'prior_reference',
        (
            'We are taking the {v} trail.',
            'The left trail reaches the quarry. The right trail reaches the marsh.',
            'Where does our chosen trail lead?',
        ),
        ('left', 'right'),
        ('quarry', 'marsh'),
        ('the {}', 'to the {}'),
    ),
    (
        'prior_reference',
        (
            'Save the note from {v}.',
            'Tess signed with a leaf. Gabe signed with a shell.',
            'Which symbol is on the note I picked?',
        ),
        ('Tess', 'Gabe'),
        ('leaf', 'shell'),
        ('a {}', 'the {}'),
    ),
    (
        'prior_reference',
        (
            'Remember the {v} parcel.',
            'The wool parcel is for Esme. The felt parcel is for Dara.',
            'Who should receive my parcel?',
        ),
        ('felt', 'wool'),
        ('Dara', 'Esme'),
        ('It is for {}',),
    ),
    (
        'instruction',
        (
            'When I give a sequence, give me its {v} entry.',
            'The sequence is amber, pearl, slate.',
            'Which entry does my earlier instruction call for?',
        ),
        ('middle', 'final'),
        ('pearl', 'slate'),
        (),
    ),
    (
        'instruction',
        (
            'When I give you a postcard, tell me its {v}.',
            'The postcard is from Mara and is addressed to Dover.',
            'Which word from the postcard should you report?',
        ),
        ('sender', 'destination'),
        ('Mara', 'Dover'),
        (),
    ),
    (
        'instruction',
        (
            'For two words, return the one that comes {v} in alphabetical order.',
            'The words are robin and finch.',
            'Return the word using the rule I gave you.',
        ),
        ('earlier', 'later'),
        ('finch', 'robin'),
        (),
    ),
)
ACK = {'dev': ('Understood.', 'I have noted that.'), 'final': ('All right.', 'Got it.')}


def _normalize(answer):
    normalized = ' '.join(answer.split()).casefold()
    return normalized[:-1].rstrip() if normalized.endswith('.') else normalized


def grade(item, text):
    """Only an enumerated affirmative form passes. Additional claims do not."""
    return _normalize(text) in {_normalize(x) for x in item['accepted']}


def bank(split):
    if split not in ACK:
        raise ValueError('Dialogue split must be dev or final')
    pairs = []
    for index, (family, turns, cues, answers, wrappers) in enumerate(DEV if split == 'dev' else FINAL):
        if sum(t.count('{v}') for t in turns) != 1 or '{v}' in turns[-1]:
            raise ValueError('Each pair needs exactly one earlier changed user turn')
        pair_id = f'{VERSION}/{split}/{family}/{index:02d}'
        pair = {'pair_id': pair_id, 'family': family, 'items': []}
        for side, (cue, variants) in enumerate(zip(cues, answers)):
            messages = []
            for turn, content in enumerate(turns):
                if turn:
                    messages.append({'role': 'assistant', 'content': ACK[split][turn - 1]})
                messages.append({'role': 'user', 'content': content.format(v=cue)})
            prompt_text(messages)  # Verify the same alternating chat format as SFT.
            variants = (variants,) if isinstance(variants, str) else variants
            accepted = sorted({form.format(answer) for form in ('{}', *wrappers) for answer in variants})
            pair['items'].append(
                {
                    'id': f'{pair_id}/{side}',
                    'side': side,
                    'messages': messages,
                    'reference': variants[0],
                    'accepted': accepted,
                    'max_tokens': LIMIT,
                }
            )
        left, right = pair['items']
        if {_normalize(x) for x in left['accepted']} & {_normalize(x) for x in right['accepted']}:
            raise ValueError('Counterfactual dialogue answers overlap')
        pairs.append(pair)
    return pairs


def evaluate(model, tokenizer, split):
    """Reset context for every item and use the verified greedy decoder."""
    from a100_greedy_batch import generate_batch

    probes = bank(split)
    items = [item for pair in probes for item in pair['items']]
    if next(model.parameters()).device.type == 'cuda':
        outputs = generate_batch(model, tokenizer, items, batch_size=16, verify=2, serial_on_mismatch=True)
        backend = getattr(outputs, 'decode_backend', 'batched_greedy')
    else:
        from coherence_quality import generate_answer

        outputs = [
            generate_answer(model, tokenizer, item['messages'], max_tokens=item['max_tokens'])
            for item in items
        ]
        backend = 'serial_greedy'
    if len(outputs) != len(items):
        raise RuntimeError('Incomplete dialogue generation')
    answers = iter(outputs)
    rows = []
    passed_pairs = Counter()
    passed_items = Counter()
    counts = Counter()
    eos_items = 0
    for pair in probes:
        family = pair['family']
        records = []
        for item in pair['items']:
            answer = next(answers)
            text_passed = grade(item, answer['text'])
            passed = text_passed and answer['eos'] is True
            eos_items += int(answer['eos'])
            passed_items[family] += int(passed)
            records.append(dict(**item, **answer, text_passed=text_passed, passed=passed))
        win = all(row['passed'] for row in records)
        passed_pairs[family] += int(win)
        counts[family] += 1
        rows.append({'pair_id': pair['pair_id'], 'family': family, 'passed': win, 'items': records})
    n = len(rows)
    wins = sum(passed_pairs.values())
    single = sum(passed_items.values())
    buckets = {
        f: {
            'pairs_passed': passed_pairs[f],
            'pairs_total': counts[f],
            'items_passed': passed_items[f],
            'items_total': 2 * counts[f],
        }
        for f in FAMILIES
    }
    return {
        'format': VERSION,
        'split': split,
        'suite_fingerprint': digest(canonical(probes)),
        'pairs_passed': wins,
        'pairs_total': n,
        'pair_rate': wins / n,
        'items_passed': single,
        'items_total': 2 * n,
        'item_rate': single / (2 * n),
        'eos_items': eos_items,
        'buckets': buckets,
        'rows': rows,
        'decoding': {
            'temperature': 0.0,
            'repetition_penalty': 1.0,
            'max_tokens': LIMIT,
            'backend': backend,
        },
    }
