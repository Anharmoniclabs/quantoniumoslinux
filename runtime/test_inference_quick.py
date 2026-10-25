"""Quick inference test with C++ kernels enabled."""
import sys, os, time
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
# Do NOT set RFTMW_NO_NATIVE — we want C++ kernels
os.environ.pop('RFTMW_NO_NATIVE', None)

from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine

print('Loading engine...')
t0 = time.time()
e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()
print(f'Loaded in {time.time()-t0:.1f}s')

prompt = '<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
print(f'Prompt: {repr(prompt)}')
print('Generating (temp=0, greedy)...')

t0 = time.time()
tokens_out = []
for tok in e.generate_stream(prompt, max_tokens=30, temperature=0.0):
    tokens_out.append(tok)
    elapsed = time.time() - t0
    print(f'  [{elapsed:.1f}s] {repr(tok)}')
    sys.stdout.flush()

result = ''.join(tokens_out)
print(f'\nFinal result: [{result}]')
print(f'Total time: {time.time()-t0:.1f}s for {len(tokens_out)} tokens')
