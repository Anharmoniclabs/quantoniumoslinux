#!/usr/bin/env python3
"""Quick inference quality test."""
import os, sys
os.environ['RFTMWPK_PATH'] = 'deepseek-r1-7b.rftmwpk'
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
print('--- streaming 50 tokens ---')
full = ''
for i, tok in enumerate(e.generate_stream(prompt, max_tokens=50, temperature=0.7)):
    full += tok
    print(f'  [{i}] {repr(tok)}')
print(f'\nFULL OUTPUT: {repr(full)}')
