"""Small reviewed curriculum with distinct training, development and final prompts.

Reference matches are conservative task scores, not open-ended semantic grading.
Generated code is interpreted using a tiny arithmetic AST subset, never exec().
"""
import ast
import hashlib
import json
import operator
import re

# First three prompts train; fourth selects checkpoints; fifth is final-only.
FACTS = [
('ram', 'RAM temporarily holds data used by running programs.', [
 'What does RAM do?', 'Explain RAM in a computer.', 'What is computer memory called RAM used for?',
 'Describe the purpose of RAM.', 'Why does a running program need RAM?']),
('storage', 'Storage keeps files when the power is off.', [
 'What does storage do?', 'Explain computer storage.', 'What is a storage drive used for?',
 'Describe the purpose of persistent storage.', 'What keeps saved files after a computer shuts down?']),
('python', 'Python is a programming language.', [
 'What is Python?', 'Explain the Python language.', 'Tell me what Python is in computing.',
 'How would you describe Python?', 'Is Python a programming language or a storage device?']),
('linux', 'Linux is an operating system kernel.', [
 'What is Linux?', 'Explain Linux briefly.', 'Tell me what Linux is in computing.',
 'How would you describe Linux?', 'Is Linux an operating system kernel or a programming language?']),
('quantum', 'Quantum physics studies matter and energy at very small scales.', [
 'What is quantum physics?', 'Explain quantum physics simply.', 'What does quantum physics study?',
 'Give a short description of quantum physics.', 'Which scales of matter and energy does quantum physics study?']),
('astronomy', 'Astronomy studies stars, planets, and other objects in space.', [
 'What is astronomy?', 'Explain astronomy simply.', 'What does astronomy study?',
 'Give a short description of astronomy.', 'What kinds of objects do astronomers study?']),
('cpu', 'A CPU executes instructions in computer programs.', [
 'What does a CPU do?', 'Explain the purpose of a CPU.', 'What is a computer processor used for?',
 'Describe the job of the central processing unit.', 'Which job does a CPU perform when a program runs?']),
('gpu', 'A GPU performs many calculations in parallel.', [
 'What does a GPU do?', 'Explain the purpose of a GPU.', 'How does a GPU handle calculations?',
 'Describe parallel processing on a GPU.', 'Does a GPU perform many calculations in parallel?']),
('variable', 'A variable is a name that refers to a value.', [
 'What is a variable in programming?', 'Explain a programming variable.', 'What does a variable represent in code?',
 'Describe what a variable means in a program.', 'What connects a name to a value in programming?']),
('loop', 'A loop repeats a set of instructions.', [
 'What is a loop in programming?', 'Explain a programming loop.', 'What does a loop do in code?',
 'Describe the purpose of a loop.', 'How does a loop affect the instructions inside it?']),
('function', 'A function is a reusable block of code that performs a task.', [
 'What is a function in programming?', 'Explain a programming function.', 'What is a function used for in code?',
 'Describe the purpose of a function.', 'What is a reusable block of code for performing a task called?']),
('file', 'A file is a named collection of stored data.', [
 'What is a computer file?', 'Explain a file in computing.', 'What does the word file mean on a computer?',
 'Describe a computer file simply.', 'What is a named collection of stored data called?']),
]
CODES = [
 ('add', 'a, b', 'a + b', 'the sum of a and b'),
 ('subtract', 'a, b', 'a - b', 'a minus b'),
 ('multiply', 'a, b', 'a * b', 'the product of a and b'),
 ('square', 'x', 'x * x', 'x multiplied by itself'),
 ('negate', 'x', '-x', 'the negative of x'),
 ('is_even', 'x', 'x % 2 == 0', 'True if x is even and False otherwise'),
]
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul}
MATH_TRAIN = ['What is {a} {op} {b}? Reply with only the integer.',
              'Calculate {a} {op} {b}. Return only the integer.',
              'Give only the integer result of {a} {op} {b}.']
MATH_DEV = 'Evaluate {a} {op} {b}; answer with just the integer.'
MATH_FINAL = 'Return a single integer: {a} {op} {b} = ?'


def build():
    splits = {k: [] for k in ('train', 'dev', 'final')}
    def add(split, group, kind, prompt, answer, bucket=None):
        splits[split].append(dict(group=group, kind=kind, prompt=prompt, answer=answer,
                                  bucket=bucket or kind))
    for group, answer, prompts in FACTS:
        for i, prompt in enumerate(prompts):
            add('train' if i < 3 else ('dev' if i == 3 else 'final'), group, 'knowledge', prompt, answer)
    for name, args, expr, purpose in CODES:
        answer = f'```python\ndef {name}({args}):\n    return {expr}\n```'
        prompts = [f'Write a Python function {name}({args}) that returns {purpose}.',
                   f'Implement {name}({args}) in Python. It must return {purpose}.',
                   f'Give Python code for {name}({args}), returning {purpose}.',
                   f'Create a Python function named {name} with parameters {args}; return {purpose}.',
                   f'I need Python code: {name}({args}) should compute and return {purpose}.']
        for i, prompt in enumerate(prompts):
            add('train' if i < 3 else ('dev' if i == 3 else 'final'), name, 'code', prompt, answer)
    for op, func in OPS.items():
        for a in range(1, 6):
            for b in range(1, 6):
                for pattern in MATH_TRAIN:
                    add('train', op, 'arithmetic', pattern.format(a=a, b=b, op=op), str(func(a,b)))
        for split, pattern, pairs, new_pairs in [
            ('dev', MATH_DEV, [(1,2),(2,4),(4,3),(5,5)], [(6,7),(7,8),(8,9),(9,6)]),
            ('final', MATH_FINAL, [(1,3),(2,5),(3,4),(5,2)], [(10,11),(11,12),(12,10),(13,11)])]:
            for a,b in pairs:
                add(split, op, 'arithmetic', pattern.format(a=a,b=b,op=op), str(func(a,b)), 'math_rewording')
            for a,b in new_pairs:
                add(split, op, 'arithmetic', pattern.format(a=a,b=b,op=op), str(func(a,b)), 'math_new_inputs')
    sets = [{r['prompt'] for r in splits[s]} for s in ('train','dev','final')]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i+1,3))
    return splits


def fingerprint(splits):
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def code_passes(text, name):
    """Evaluate only a bounded one-return arithmetic AST, without executing code."""
    try:
        if len(text) > 2000:
            return False
        text = text.strip()
        if text.startswith('```python\n') and text.endswith('```'):
            text = text[len('```python\n'):-3].strip()
        tree = ast.parse(text)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False
        f = tree.body[0]
        spec = next(s for s in CODES if s[0] == name)
        names = spec[1].split(', ')
        if (f.name != name or f.decorator_list or f.returns or len(f.body) != 1 or
            not isinstance(f.body[0], ast.Return) or f.args.defaults or f.args.kw_defaults or
            f.args.vararg or f.args.kwarg or f.args.kwonlyargs or f.args.posonlyargs or
            [a.arg for a in f.args.args] != names or any(a.annotation for a in f.args.args)):
            return False
        expr = f.body[0].value
        nodes = list(ast.walk(expr))
        allowed = (ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
                   ast.Mod, ast.UnaryOp, ast.USub, ast.Compare, ast.Eq)
        if len(nodes) > 40 or any(not isinstance(n, allowed) for n in nodes):
            return False
        if any(isinstance(n, ast.Constant) and (type(n.value) not in (int,bool) or abs(n.value)>100) for n in nodes):
            return False
        def value(n, env):
            if isinstance(n, ast.Name): return env[n.id]
            if isinstance(n, ast.Constant): return n.value
            if isinstance(n, ast.UnaryOp): return -value(n.operand, env)
            if isinstance(n, ast.BinOp):
                fn = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul, ast.Mod:operator.mod}[type(n.op)]
                result = fn(value(n.left,env), value(n.right,env))
                if abs(result) > 10**9: raise ValueError('Bound exceeded')
                return result
            if isinstance(n, ast.Compare) and len(n.ops)==1:
                return value(n.left,env) == value(n.comparators[0],env)
            raise ValueError('Unsupported expression')
        target = {'add':lambda a,b:a+b, 'subtract':lambda a,b:a-b, 'multiply':lambda a,b:a*b,
                  'square':lambda x:x*x, 'negate':lambda x:-x, 'is_even':lambda x:x%2==0}[name]
        cases = [(-7,3), (0,0), (4,-5), (11,12), (2,8)] if len(names)==2 else [(-7,), (0,), (4,), (11,), (12,)]
        return all(type(v := value(expr, dict(zip(names,args)))) is type(target(*args)) and v == target(*args) for args in cases)
    except (ValueError, SyntaxError, StopIteration, KeyError, TypeError, ZeroDivisionError, RecursionError):
        return False


def score(item, answer, eos):
    if not eos:
        return False
    if item['kind'] == 'knowledge':
        return normalized(answer) == normalized(item['answer'])
    if item['kind'] == 'arithmetic':
        return bool(re.fullmatch(r'-?\d+', answer.strip())) and int(answer.strip()) == int(item['answer'])
    return code_passes(answer, item['group'])
