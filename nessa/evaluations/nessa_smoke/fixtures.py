"""Small deterministic smoke suite, not evidence of broad coding capability."""
TASKS = {
 'merge_ranges': {
  'prompt': 'Fix solution.merge_ranges(ranges). Return sorted merged closed integer intervals as tuples. Merge overlapping or endpoint-touching ranges (not merely adjacent integers), support empty input, do not mutate input. All pairs have start <= end.',
  'source': 'def merge_ranges(ranges):\n    return sorted(ranges)\n',
  'tests': '''from solution import merge_ranges as f
assert f([]) == []
a = [(7,9),(1,4),(3,8),(12,12)]
assert f(a) == [(1,9),(12,12)]
assert a == [(7,9),(1,4),(3,8),(12,12)]
assert f([(1,2),(2,3),(4,5)]) == [(1,3),(4,5)]
assert f([(-5,-1),(-4,-3),(-1,0)]) == [(-5,0)]
assert f([(2,2),(2,2)]) == [(2,2)]
''',
 'reference': '''def merge_ranges(ranges):
    out = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out
'''},
 'stable_unique': {
 'prompt': 'Fix solution.stable_unique(values). Return a new list retaining the first occurrence of each equal value in original order. Support lists and dictionaries as values, use Python equality semantics, and leave input unchanged.',
 'source': 'def stable_unique(values):\n    return list(set(values))\n',
 'tests': '''from solution import stable_unique as f
assert f([]) == []
assert f([3,1,3,2,1]) == [3,1,2]
a = [[1], {'x': 2}, [1], {'x': 2}, []]
assert f(a) == [[1], {'x': 2}, []]
assert a == [[1], {'x': 2}, [1], {'x': 2}, []]
assert f([1, True, 0, False]) == [1,0]
assert f(['a','A','a']) == ['a','A']
''',
 'reference': '''def stable_unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
'''},
 'parse_pairs': {
 'prompt': 'Fix solution.parse_pairs(text). Parse nonempty semicolon-separated key=value segments into a dict. Trim keys and values, ignore whitespace-only segments, split on the first equals sign only, keep the last duplicate key. Raise ValueError for missing equals or an empty trimmed key. Empty values are valid.',
 'source': "def parse_pairs(text):\n    return dict(piece.split('=') for piece in text.split(';'))\n",
 'tests': '''from solution import parse_pairs as f
assert f('') == {}
assert f(' ; ; ') == {}
assert f(' a = 1 ; b=two=three; a=last; c= ') == {'a':'last','b':'two=three','c':''}
assert f('x=;x=0') == {'x':'0'}
for text in ['bad', '=value', 'a=1; =2']:
    try: f(text)
    except ValueError: pass
    else: raise AssertionError(text)
''',
 'reference': '''def parse_pairs(text):
    result = {}
    for piece in text.split(';'):
        if not piece.strip():
            continue
        key, sep, value = piece.partition('=')
        if not sep or not key.strip():
            raise ValueError('invalid pair')
        result[key.strip()] = value.strip()
    return result
'''},
 'retry_call': {
 'prompt': 'Fix solution.retry_call(fn, attempts). attempts is an integer: reject attempts < 1 with ValueError before invoking fn. Invoke fn until it returns successfully, at most attempts times. Retry only ValueError, return the successful result unchanged, and re-raise the final ValueError when exhausted. Other exception types propagate immediately.',
 'source': 'def retry_call(fn, attempts):\n    for _ in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            pass\n',
 'tests': '''from solution import retry_call as f
calls = []
def flaky():
    calls.append(1)
    if len(calls) < 3: raise ValueError('again')
    return None
assert f(flaky, 3) is None and len(calls) == 3
for n in [0, -1]:
    try: f(lambda: calls.append(1), n)
    except ValueError: pass
    else: raise AssertionError('invalid attempts accepted')
assert len(calls) == 3
err = ValueError('last')
def fail(): raise err
try: f(fail, 2)
except ValueError as actual: assert actual is err
else: raise AssertionError('swallowed last error')
calls.clear()
def stop():
    calls.append(1)
    raise TypeError('stop')
try: f(stop, 5)
except TypeError: pass
else: raise AssertionError('swallowed TypeError')
assert len(calls) == 1
assert f(lambda: 42, 1) == 42
''',
 'reference': '''def retry_call(fn, attempts):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    for index in range(attempts):
        try:
            return fn()
        except ValueError:
            if index == attempts - 1:
                raise
'''}
}
