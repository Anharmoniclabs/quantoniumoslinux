#!/usr/bin/env python3
"""CLI: compress any numpy array to .rftb using the unified RFT pipeline."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantonium_os_src.engine.rft_compress import compress, decompress

def main():
    parser = argparse.ArgumentParser(description="RFT Hybrid Compress CLI")
    parser.add_argument("input",  help="Input .npy file")
    parser.add_argument("output", help="Output .rftb file")
    parser.add_argument("--decompress", action="store_true")
    args = parser.parse_args()
    if args.decompress:
        data = decompress(Path(args.input).read_bytes())
        np.save(args.output, data)
        print(f"Decompressed -> {args.output}")
    else:
        arr = np.load(args.input)
        blob = compress(arr)
        Path(args.output).write_bytes(blob)
        ratio = arr.nbytes / len(blob)
        print(f"Compressed {arr.nbytes/1024:.1f} KB -> {len(blob)/1024:.1f} KB ({ratio:.2f}x)")

if __name__ == "__main__":
    main()
