#!/usr/bin/env python3
"""Inference quality test: NumPy-only (no C++ kernels)."""
import os
os.environ['RFTMWPK_PATH'] = 'deepseek-r1-7b.rftmwpk'
os.environ['RFTMW_NO_NATIVE'] = '1'  # Force NumPy-only

from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine

e = Qwen25Engine('deepseek-r1-7b.rftmwpk')
e.load()

def build_prompt(messages):
    parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        parts.append(f'<|im_start|>{role}\n{content}<|im_end|>')
    parts.append('<|im_start|>assistant\n')
    return '\n'.join(parts)

messages = [{'role': 'user', 'content': 'hey cael'}]
prompt = build_prompt(messages)
print(f'Prompt: {repr(prompt)}')
print('--- NumPy-only, 50 tokens ---')
full = ''
for i, tok in enumerate(e.generate_stream(prompt, max_tokens=50, temperature=0.7)):
    full += tok
    print(f'  [{i}] {repr(tok)}')
print(f'\nFULL: {repr(full)}')
