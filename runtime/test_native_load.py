import sys, os
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
try:
    import rftmw_native
    print(f'Loaded: {rftmw_native.__file__}')
    print(f'HAS_AVX2: {rftmw_native.HAS_AVX2}')
    print(f'HAS_ASM: {rftmw_native.HAS_ASM_KERNELS}')
    has_infer = hasattr(rftmw_native, 'infer')
    print(f'Has infer: {has_infer}')
    if has_infer:
        avail = getattr(rftmw_native.infer, 'available', False)
        print(f'infer.available: {avail}')
        funcs = [x for x in dir(rftmw_native.infer) if not x.startswith('_')]
        print(f'infer funcs: {funcs}')
except Exception as ex:
    print(f'FAILED: {type(ex).__name__}: {ex}')
