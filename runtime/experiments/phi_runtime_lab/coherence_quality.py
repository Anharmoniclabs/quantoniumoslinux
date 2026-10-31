"""Frozen, bounded functional smoke checks, not proof of general intelligence.

Suites are evaluation-only. The development set may select a checkpoint; run
the final set once after selection and do not use its results for tuning.
Generated Python is inspected and interpreted by a small numeric AST evaluator;
it is never passed to eval, exec, compile, a shell, or an external interpreter.
"""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from chat_recovery_data import prompt_text, read_jsonl, sha

FORMAT = 'PhiState-functional-smoke-v1'
BUCKETS = ('knowledge', 'instruction', 'arithmetic', 'code')
CRITERIA = dict(min_macro_score=.8, min_bucket_score=.5,
                max_repetition_failure_rate=.05, min_nonempty_rate=.95)
SCOPE = ('32 frozen functional smoke questions; lexical knowledge rubrics and '
         'small numeric code cases are limited proxies, not broad coherence proof.')


def normalized(text):
    return ' '.join(text.lower().split())


def fingerprint(value):
    clean = {k: v for k, v in value.items() if k != 'fingerprint'}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def _bank(split):
    final = split == 'final'
    # Each group is OR; every group must match. These deliberately narrow
    # rubrics are disclosed in the report and may reject valid unusual wording.
    knowledge = [
        ('What does computer RAM do?', 'Describe the purpose of RAM in a running computer.',
         [['temporary', 'temporarily', 'short-term', 'volatile'], ['data', 'information', 'program']]),
        ('What is computer storage used for?', 'Explain why files are kept on an SSD or hard drive.',
         [['file', 'data', 'information'], ['persist', 'permanent', 'long-term', 'power', 'retain', 'keep', 'store']]),
        ('What does a computer CPU do?', 'Explain the role of the processor inside a computer.',
         [['execute', 'run', 'process', 'perform'], ['instruction', 'calculation', 'computation', 'program']]),
        ('What is Python in computing?', 'Describe Python as a tool for writing software.',
         [['programming language', 'coding language', 'computer language']]),
        ('What is a variable in a computer program?', 'Explain what a programming variable represents.',
         [['name', 'named', 'identifier', 'label'], ['value', 'data', 'information']]),
        ('What is an operating system used for?', 'Describe the job of an operating system on a laptop.',
         [['manage', 'control', 'coordinate', 'allocate'], ['hardware', 'resource', 'memory', 'process']]),
        ('What do plants use sunlight for during photosynthesis?', 'Explain how photosynthesis helps a plant obtain food.',
         [['light', 'sun', 'solar'], ['food', 'sugar', 'glucose', 'energy']]),
        ('What does gravity do to objects?', 'Explain why a dropped object falls toward Earth.',
         [['attract', 'pull', 'force'], ['mass', 'object', 'earth', 'ground']]),
    ]
    items = []
    for i, (dev, held, groups) in enumerate(knowledge):
        items.append(dict(id=f'{split}-knowledge-{i}', bucket='knowledge',
                          prompt=held if final else dev, max_tokens=80,
                          rubric=dict(kind='keywords', groups=groups)))
    instructions = ([
        ('Reply with exactly the word cobalt.', 'cobalt'),
        ('Write apple in uppercase. Output only the result.', 'APPLE'),
        ('Write CLOUD in lowercase. Output only the result.', 'cloud'),
        ('Reverse the letters of dog. Output only the result.', 'god'),
        ('How many letters are in banana? Output only the integer.', '6'),
        ('Output exactly this JSON array: [2, 4, 6]', '[2, 4, 6]'),
        ('Return exactly the two words: silver moon', 'silver moon'),
        ('Repeat the following characters exactly: Q7-z', 'Q7-z'),
    ] if final else [
        ('Reply with exactly the word mango.', 'mango'),
        ('Write pear in uppercase. Output only the result.', 'PEAR'),
        ('Write RIVER in lowercase. Output only the result.', 'river'),
        ('Reverse the letters of cat. Output only the result.', 'tac'),
        ('How many letters are in orange? Output only the integer.', '6'),
        ('Output exactly this JSON array: [1, 3, 5]', '[1, 3, 5]'),
        ('Return exactly the two words: green leaf', 'green leaf'),
        ('Repeat the following characters exactly: K4-x', 'K4-x'),
    ])
    for i, (prompt, expected) in enumerate(instructions):
        items.append(dict(id=f'{split}-instruction-{i}', bucket='instruction',
                          prompt=prompt, max_tokens=24,
                          rubric=dict(kind='exact', expected=expected)))
    arithmetic = ([(17, '+', 26), (31, '+', 14), (29, '-', 12), (8, '-', 19),
                   (7, '*', 6), (9, '*', 4), (18, '//', 3), (23, '%', 5)] if final
                  else [(12, '+', 15), (24, '+', 8), (21, '-', 13), (5, '-', 12),
                        (6, '*', 8), (7, '*', 3), (20, '//', 4), (19, '%', 4)])
    operations = {'+': lambda a, b: a+b, '-': lambda a, b: a-b,
                  '*': lambda a, b: a*b, '//': lambda a, b: a//b, '%': lambda a, b: a % b}
    for i, (a, op, b) in enumerate(arithmetic):
        expression = (f'the integer quotient of {a} divided by {b}' if op == '//' else
                      f'the remainder when {a} is divided by {b}' if op == '%' else f'{a} {op} {b}')
        wording = f'Calculate {expression}. Respond with only the integer.' if final else \
                  f'Evaluate {expression}; return only the integer.'
        items.append(dict(id=f'{split}-arithmetic-{i}', bucket='arithmetic', prompt=wording,
                          max_tokens=24, rubric=dict(kind='exact', expected=str(operations[op](a, b)))))
    functions = [
        ('add', ['a', 'b'], 'return the sum of a and b', lambda a, b: a+b),
        ('subtract', ['a', 'b'], 'return a minus b', lambda a, b: a-b),
        ('multiply', ['a', 'b'], 'return the product of a and b', lambda a, b: a*b),
        ('square', ['x'], 'return x multiplied by itself', lambda x: x*x),
        ('is_even', ['x'], 'return a Boolean indicating whether integer x is even', lambda x: x % 2 == 0),
        ('maximum', ['a', 'b'], 'return the larger of a and b', lambda a, b: max(a, b)),
        ('absolute', ['x'], 'return the absolute value of x', lambda x: abs(x)),
        ('double', ['x'], 'return twice x', lambda x: 2*x),
    ]
    for i, (name, args, behavior, operation) in enumerate(functions):
        pairs = [(7, 2), (-4, 9), (0, -6), (5, 5)] if final else [(2, 3), (-7, 4), (0, 8), (-2, -2)]
        singles = [9, -4, 0, 12] if final else [3, -8, 0, 14]
        inputs = [list(p) for p in pairs] if len(args) == 2 else [[v] for v in singles]
        signature = name + '(' + ', '.join(args) + ')'
        wording = (f'Provide only Python code defining {signature}. It must {behavior}.' if final else
                   f'Write a Python function {signature} that will {behavior}. Return only code.')
        items.append(dict(id=f'{split}-code-{i}', bucket='code', prompt=wording, max_tokens=128,
                          rubric=dict(kind='code', function=name, args=args,
                                      cases=[dict(args=v, expected=operation(*v)) for v in inputs])))
    return items


def build_suites(prepared_dir, output_dir):
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    forbidden = set()
    provenance = {}
    for split in ('train', 'val'):
        path = prepared_dir / (split + '.jsonl')
        provenance[path.name] = sha(path)
        for row in read_jsonl(path):
            forbidden.add(normalized(row['messages'][-2]['content']))
    suites = {}
    for split in ('dev', 'final'):
        items = _bank(split)
        for item in items:
            original = item['prompt']
            alternatives = [original, 'Please answer this request: ' + original,
                            'Complete this task: ' + original]
            chosen = next((p for p in alternatives if normalized(p) not in forbidden), None)
            if chosen is None:
                raise ValueError('Frozen prompt alternatives overlap prepared data: ' + item['id'])
            item['prompt'] = chosen
            forbidden.add(normalized(chosen))
        suite = dict(format=FORMAT, split=split, scope=SCOPE, criteria=CRITERIA,
                     prepared_files=provenance, items=items)
        suite['fingerprint'] = fingerprint(suite)
        suites[split] = suite
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / (split + '.json') for split in suites}
    # Verify every existing artifact before writing either missing artifact.
    for split, path in paths.items():
        if path.exists() and load_suite(path) != suites[split]:
            raise ValueError('Existing frozen suite differs; use a new suite directory: ' + str(path))
    for split, path in paths.items():
        if not path.exists():
            with path.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(suites[split], indent=2, ensure_ascii=False) + '\n')
    return paths


def load_suite(path):
    suite = json.loads(Path(path).read_text())
    if suite.get('format') != FORMAT or suite.get('fingerprint') != fingerprint(suite):
        raise ValueError('Suite format/fingerprint mismatch')
    if suite.get('split') not in ('dev', 'final') or suite.get('criteria') != CRITERIA:
        raise ValueError('Unexpected suite split or qualification criteria')
    items = suite.get('items', [])
    if Counter(i['bucket'] for i in items) != Counter({b: 8 for b in BUCKETS}):
        raise ValueError('Expected eight questions in each of four buckets')
    if len({i['id'] for i in items}) != 32 or len({normalized(i['prompt']) for i in items}) != 32:
        raise ValueError('Duplicate suite identifiers/prompts')
    for item in items:
        if not isinstance(item['prompt'], str) or not item['prompt'].strip():
            raise ValueError('Empty suite prompt')
        if not 1 <= item['max_tokens'] <= 256:
            raise ValueError('Unbounded generation budget')
        if item['rubric']['kind'] not in ('keywords', 'exact', 'code'):
            raise ValueError('Unknown rubric')
    return suite


def repetition_failure(text):
    if len(text) >= 40 and sum(c.isspace() for c in text) / len(text) > .9:
        return True
    # Word trigrams miss punctuation, CJK, replacement-character and subword
    # loops. Require a substantial consecutive run occupying most of the text;
    # whitespace is excluded so ordinary Python indentation cannot match.
    if len(text) >= 24:
        for match in re.finditer(r'([^\s]{1,16})\1{3,}', text):
            run_length = match.end() - match.start()
            if run_length >= 24 and run_length > len(text) / 2:
                return True
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < 12:
        return False
    triples = Counter(tuple(words[i:i+3]) for i in range(len(words)-2))
    largest = max(triples.values(), default=0)
    return largest >= 3 and largest*3 >= .6*len(words)


class _Return(Exception):
    def __init__(self, value):
        self.value = value


def _numeric(value):
    if type(value) not in (int, float, bool) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError('Non-numeric or oversized value')
    return value


def _expression(node, env, depth=0):
    if depth > 24:
        raise ValueError('Expression too deep')
    recurse = lambda child: _expression(child, env, depth+1)
    if isinstance(node, ast.Constant):
        return _numeric(node.value)
    if isinstance(node, ast.Name):
        return _numeric(env[node.id])
    if isinstance(node, ast.UnaryOp):
        value = recurse(node.operand)
        if isinstance(node.op, ast.USub): return _numeric(-value)
        if isinstance(node.op, ast.UAdd): return _numeric(+value)
        if isinstance(node.op, ast.Not): return not value
    if isinstance(node, ast.BinOp):
        a, b = recurse(node.left), recurse(node.right)
        operations = {ast.Add: lambda: a+b, ast.Sub: lambda: a-b, ast.Mult: lambda: a*b,
                      ast.Div: lambda: a/b, ast.FloorDiv: lambda: a//b, ast.Mod: lambda: a % b}
        if type(node.op) in operations: return _numeric(operations[type(node.op)]())
        if isinstance(node.op, ast.Pow) and type(b) is int and 0 <= b <= 8:
            return _numeric(a**b)
    if isinstance(node, ast.Compare):
        left = recurse(node.left)
        for op, right_node in zip(node.ops, node.comparators):
            right = recurse(right_node)
            comparisons = {ast.Eq: left == right, ast.NotEq: left != right, ast.Lt: left < right,
                           ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right}
            if type(op) not in comparisons: raise ValueError('Unsupported comparison')
            if not comparisons[type(op)]: return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        value = recurse(node.values[0])
        for child in node.values[1:]:
            if isinstance(node.op, ast.And) and not value: return value
            if isinstance(node.op, ast.Or) and value: return value
            value = recurse(child)
        return value
    if isinstance(node, ast.IfExp):
        return recurse(node.body if recurse(node.test) else node.orelse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        functions = {'abs': abs, 'min': min, 'max': max}
        if node.func.id in functions and 1 <= len(node.args) <= 4:
            return _numeric(functions[node.func.id](*[recurse(a) for a in node.args]))
    raise ValueError('Unsupported expression: ' + type(node).__name__)


def _statements(statements, env, depth=0):
    if depth > 16:
        raise ValueError('Statements too deep')
    for statement in statements:
        if isinstance(statement, ast.Return):
            raise _Return(None if statement.value is None else _expression(statement.value, env))
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            env[statement.targets[0].id] = _expression(statement.value, env)
        elif isinstance(statement, ast.If):
            _statements(statement.body if _expression(statement.test, env) else statement.orelse, env, depth+1)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue  # A docstring has no execution effect.
        elif not isinstance(statement, ast.Return):
            raise ValueError('Unsupported statement: ' + type(statement).__name__)


def code_passes(text, rubric):
    try:
        if len(text) > 12000:
            return False
        blocks = re.findall(r'```(?:python)?[ \t]*\n(.*?)```', text, flags=re.S)
        if '```' in text:
            if len(blocks) != 1 or text.count('```') != 2: return False
            source = blocks[0]
        else:
            source = text.strip()
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 256 or len(tree.body) != 1:
            return False
        function = tree.body[0]
        if not isinstance(function, ast.FunctionDef) or function.name != rubric['function']:
            return False
        args = function.args
        if (function.decorator_list or args.defaults or args.kw_defaults or args.kwonlyargs or args.posonlyargs
                or args.vararg or args.kwarg or [a.arg for a in args.args] != rubric['args']):
            return False
        annotations = [a.annotation for a in args.args] + [function.returns]
        if any(a is not None and not (isinstance(a, ast.Name) and a.id in ('int', 'float', 'bool')) for a in annotations):
            return False
        # Reject unsupported constructs even in branches the cases never visit.
        forbidden = (ast.Import, ast.ImportFrom, ast.Attribute, ast.Lambda, ast.ClassDef,
                     ast.For, ast.While, ast.With, ast.Try, ast.Global, ast.Nonlocal,
                     ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.AsyncFunctionDef)
        for node in ast.walk(function):
            if isinstance(node, forbidden): return False
            if isinstance(node, ast.FunctionDef) and node is not function: return False
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in ('abs', 'min', 'max')):
                return False
        for case in rubric['cases']:
            try:
                _statements(function.body, dict(zip(rubric['args'], case['args'])))
                return False
            except _Return as result:
                expected = case['expected']
                if type(expected) is bool:
                    if type(result.value) is not bool or result.value != expected: return False
                elif type(result.value) not in (int, float) or result.value != expected:
                    return False
        return True
    except (SyntaxError, ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError, RecursionError):
        return False


def grade_answer(item, text):
    rubric = item['rubric']
    nonempty = bool(text.strip())
    repeated = repetition_failure(text)
    if rubric['kind'] == 'exact':
        correct = text.strip() == rubric['expected']
    elif rubric['kind'] == 'keywords':
        lower = text.lower()
        def positive_match(term):
            for match in re.finditer(r'\b' + re.escape(term) + r'\w*\b', lower):
                previous = re.findall(r"[a-z]+(?:'[a-z]+)?", lower[:match.start()])[-3:]
                if not set(previous) & {'not', 'never', "isn't", "doesn't", "cannot", "can't"}:
                    return True
            return False
        correct = all(any(positive_match(term) for term in group)
                      for group in rubric['groups'])
    else:
        correct = code_passes(text, rubric)
    return dict(passed=bool(correct and nonempty and not repeated), nonempty=nonempty,
                repetition_failure=repeated)


def generate_answer(model, tokenizer, prompt, max_tokens=128, decode_config=None, seed=20261031):
    import torch
    config = dict(temperature=0., top_p=.9, top_k=0, repetition_penalty=1.)
    unknown = set(decode_config or {}) - set(config)
    if unknown:
        raise ValueError('Unknown decoding settings: ' + ', '.join(sorted(unknown)))
    config.update(decode_config or {})
    temperature, top_p, penalty = (float(config[k]) for k in ('temperature', 'top_p', 'repetition_penalty'))
    top_k = config['top_k']
    if not (math.isfinite(temperature) and temperature >= 0 and 0 < top_p <= 1 and 1 <= penalty <= 2):
        raise ValueError('Invalid decoding parameters')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer; zero disables the filter')
    if not 1 <= max_tokens <= 256:
        raise ValueError('Generation limit must be 1..256 tokens')
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            ids = tokenizer.encode(prompt_text([dict(role='user', content=prompt)]))
            if not ids: raise ValueError('Empty encoded prompt')
            if len(ids) + max_tokens > 512:
                raise ValueError('Prompt and generation exceed the 512-token evaluation limit')
            cache, produced, ended = None, [], False
            for token in ids:
                logits, cache = model.step(torch.tensor([[token]], device=device), cache)
            for _ in range(max_tokens):
                scores = logits[0, -1].float().clone()
                if not torch.isfinite(scores).all(): raise RuntimeError('Nonfinite generation logits')
                # Penalize generated tokens only. EOS remains available unchanged.
                if penalty != 1 and produced:
                    seen = torch.tensor(sorted(set(produced) - {tokenizer.eos_id()}), device=device, dtype=torch.long)
                    scores[seen] = torch.where(scores[seen] < 0, scores[seen]*penalty, scores[seen]/penalty)
                if temperature == 0:
                    token = int(scores.argmax())
                else:
                    if top_k:
                        scores, choices = torch.topk(scores, min(top_k, scores.numel()))
                    else:
                        choices = torch.arange(scores.numel(), device=device)
                    probabilities, order = torch.softmax(scores/temperature, dim=-1).sort(descending=True)
                    choices = choices[order]
                    probabilities[probabilities.cumsum(0)-probabilities > top_p] = 0
                    pick = torch.multinomial(probabilities/probabilities.sum(), 1, generator=generator)
                    token = int(choices[pick].item())
                if token == tokenizer.eos_id():
                    ended = True
                    break
                produced.append(token)
                logits, cache = model.step(torch.tensor([[token]], device=device), cache)
            return dict(text=tokenizer.decode(produced), tokens=len(produced), eos=ended)
    finally:
        model.train(was_training)


def evaluate_suite(model, tokenizer, suite):
    if suite.get('fingerprint') != fingerprint(suite):
        raise ValueError('Suite modified after loading')
    buckets = {name: dict(passed=0, total=0) for name in BUCKETS}
    rows = []
    for item in suite['items']:
        generated = generate_answer(model, tokenizer, item['prompt'], max_tokens=item['max_tokens'])
        graded = grade_answer(item, generated['text'])
        rows.append(dict(id=item['id'], bucket=item['bucket'], prompt=item['prompt'], **generated, **graded))
        buckets[item['bucket']]['total'] += 1
        buckets[item['bucket']]['passed'] += int(graded['passed'])
    for counts in buckets.values():
        counts['rate'] = counts['passed']/counts['total']
    macro = sum(b['rate'] for b in buckets.values())/len(buckets)
    nonempty = sum(r['nonempty'] for r in rows)/len(rows)
    repeated = sum(r['repetition_failure'] for r in rows)/len(rows)
    qualified = (macro >= CRITERIA['min_macro_score'] and
                 min(b['rate'] for b in buckets.values()) >= CRITERIA['min_bucket_score'] and
                 repeated <= CRITERIA['max_repetition_failure_rate'] and nonempty >= CRITERIA['min_nonempty_rate'])
    return dict(macro_score=macro, qualified=qualified, buckets=buckets, rows=rows,
                nonempty_rate=nonempty, repetition_failure_rate=repeated, suite_fingerprint=suite['fingerprint'],
                split=suite['split'], scope=SCOPE, decoding=dict(temperature=0., repetition_penalty=1.))
