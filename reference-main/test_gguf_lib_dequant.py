"""Use the gguf library to independently dequantize a weight and compare
against our dequant + matmul. This tests if our Q4K/Q6K dequant is correct."""
import sys, os, numpy as np
from gguf import GGUFReader

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

print('Reading GGUF with gguf library...')
reader = GGUFReader(GGUF)

# Find blk.0.attn_q.weight
for tensor in reader.tensors:
    if tensor.name == 'blk.0.attn_q.weight':
        q_tensor = tensor
        break

print(f'Tensor: {q_tensor.name}')
print(f'  Shape (ne): {q_tensor.shape}')
print(f'  Type: {q_tensor.tensor_type}')
print(f'  n_elements: {q_tensor.n_elements}')

# The gguf library can dequantize! 
# tensor.data gives raw bytes, but we need the dequantized values
# Let's check what the tensor.data looks like
print(f'  Data type: {type(q_tensor.data)}')
print(f'  Data shape: {q_tensor.data.shape}')
print(f'  Data dtype: {q_tensor.data.dtype}')

# The gguf library stores raw bytes. Let's try to get dequantized values.
# The GGUFReader exposes tensor data as numpy arrays.
# For quantized types, data is the raw quantized bytes.

# Let's manually dequant a single row from the gguf reader's data and compare
from quantonium_os_src.engine.rftmw_numpy_engine import _dequant_q4k_blocks

# Get row 0 from gguf reader
ne0 = q_tensor.shape[0]  # 2048 (in_features per row)  
ne1 = q_tensor.shape[1] if len(q_tensor.shape) > 1 else q_tensor.n_elements // ne0
bpr = (ne0 // 256) * 144  # Q4K bytes per row

# The data is stored as a flat array of uint8
raw_data = np.array(q_tensor.data, dtype=np.uint8)
print(f'\nRaw data from gguf reader: shape={raw_data.shape}')
print(f'Expected total bytes: {ne1 * bpr} = {ne1} rows * {bpr} bpr')

# Reshape to rows
if raw_data.size == ne1 * bpr:
    rows = raw_data.reshape(ne1, bpr)
elif raw_data.size == ne0 * ne1:
    # Might be dequantized already?
    print(f'  Data might be dequantized: {raw_data.size} = {ne0} * {ne1}')
    rows = None
else:
    # Hmm, different size
    n_blocks_total = raw_data.size // 144
    print(f'  Total Q4K blocks: {n_blocks_total}')
    blocks_per_row = ne0 // 256
    n_rows = n_blocks_total // blocks_per_row
    print(f'  Computed rows: {n_rows} (blocks_per_row={blocks_per_row})')
    rows = raw_data.reshape(n_rows, bpr)

if rows is not None:
    # Dequant row 0 using our function
    row0_blocks = rows[0].reshape(-1, 144)
    our_dequant = _dequant_q4k_blocks(row0_blocks).flatten()[:ne0]
    print(f'\nOur dequant row 0: first 8={our_dequant[:8]}')
    print(f'  min={our_dequant.min():.6f} max={our_dequant.max():.6f}')

# Now let's also try using the gguf library's own dequant if available
try:
    from gguf.quants import dequantize
    ref_all = dequantize(q_tensor.data, q_tensor.tensor_type)
    print(f'\nGGUF library dequant: shape={ref_all.shape}')
    
    # Compare row 0
    # ref_all shape might be (ne1, ne0) or flattened
    if ref_all.ndim == 1:
        ref_row0 = ref_all[:ne0]
    else:
        ref_row0 = ref_all[0, :ne0] if ref_all.shape[0] == ne1 else ref_all[:ne0, 0]
    
    print(f'Ref dequant row 0: first 8={ref_row0[:8]}')
    diff = np.abs(our_dequant - ref_row0)
    print(f'\nDequant comparison (our vs gguf library):')
    print(f'  max_diff: {diff.max():.2e}')
    print(f'  mean_diff: {diff.mean():.2e}')
    
    if diff.max() > 0.001:
        print('  *** DEQUANT MISMATCH! ***')
        worst_idx = np.argmax(diff)
        print(f'  Worst at index {worst_idx}: ours={our_dequant[worst_idx]:.6f} ref={ref_row0[worst_idx]:.6f}')
    else:
        print('  Dequant matches perfectly!')
        
    # Now compare a matmul using the reference-dequantized weight
    # Use a random activation
    x = np.random.randn(ne0).astype(np.float32)
    x = x / np.linalg.norm(x)  # normalize
    
    # Reference: full float32 matmul with gguf-lib dequanted weight
    ref_full = ref_all.reshape(ne1, ne0) if ref_all.ndim == 1 else ref_all
    if ref_full.shape == (ne0, ne1):
        ref_full = ref_full.T  # make it (ne1, ne0)
    ref_result = ref_full.astype(np.float32) @ x
    
    # Our engine matmul
    sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
    from quantonium_os_src.engine.rftmw_numpy_engine import QuantWeight
    qw = QuantWeight(raw_data.copy(), (ne1, bpr))
    our_result = qw.matmul(x)
    
    matmul_diff = np.abs(ref_result - our_result)
    print(f'\nMatmul comparison:')
    print(f'  ref first 5: {ref_result[:5]}')
    print(f'  our first 5: {our_result[:5]}')
    print(f'  max_diff: {matmul_diff.max():.2e}')
    print(f'  correlation: {np.corrcoef(ref_result, our_result)[0,1]:.8f}')

except ImportError:
    print('\ngguf.quants.dequantize not available')
except Exception as ex:
    print(f'\nError: {type(ex).__name__}: {ex}')
    import traceback; traceback.print_exc()
