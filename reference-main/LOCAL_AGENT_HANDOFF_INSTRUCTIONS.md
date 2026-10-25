# Local Agent Handoff: Reconcile Number Mismatch

Copy and paste the block below into your local agent exactly as written.

---

````text
Please reconcile benchmark number differences against Codespace baseline for QuantoniumOS.

## Codespace baseline numbers (April 2026)

Model: qwen2.5-coder_3b.rftmwpk (Qwen2.5-Coder 3B, 36 layers, hidden=2048, Q4K/Q6K mixed quant)
Prompt: "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
Output: "Hello! I'm an" (5 tokens)
Preload time: 25.3 s (253 tensors)

| Mode               | down_proj latency | Overall tok/s | Notes                         |
|--------------------|-------------------|---------------|-------------------------------|
| Q6K fused (C++)    | 42 ms             | 0.2 tok/s     | Dequant overhead per matmul   |
| FP32 + C++ AVX2    | 4.2 ms            | 0.6 tok/s     | NumPy BLAS after preload      |
| FP32 + ASM SGEMV   | 4.2 ms            | 0.6 tok/s     | Native ASM SGEMV              |
| Ollama (llama.cpp) | —                 | 3.5 tok/s     | Optimized runtime + mmap      |

Key detail: the "42 ms → 4.2 ms" improvement is the same down_proj weight
measured first via fused Q6K dequant+matmul, then via preloaded FP32 BLAS.
The 10x difference is expected — it is NOT a bug.

Ollama is 6x faster than our FP32+ASM path because:
1. mmap — no full model load, pages fault in on demand
2. Quantized inference — Q4_K stays quantized in cache, dequant in registers
3. Better cache utilization — tiled matmul with optimal blocking
For real-time inference, use Ollama. The native engine is for research/debugging.

## What each path actually does

1. Q6K fused (C++):
   - Env: RFTMW_REQUIRE_NATIVE=1, no preload_dense() call
      - Code: quantonium_os_src/engine/rftmw_numpy_engine.py ~line 570
         - Each matmul call dequants Q6K/Q4K bytes on the fly + AVX2 dot product
            - This is the chat app default (src/apps/chatbox_network.py ~line 174)

            2. FP32 preloaded:
               - Env: any (native or not)
                  - Code: engine.preload_dense() at ~line 1198
                     - Converts all 253 quantized tensors to FP32 upfront (25 s, ~12 GB RAM)
                        - Then matmul hits NumPy BLAS or native sgemv_f32
                           - Preload time is one-time cost; per-token latency drops to ~4 ms

                           3. Ollama/llama.cpp:
                              - Completely separate runtime, not our engine code
                                 - Uses GGUF directly with its own optimized kernels
                                    - Included as a comparison ceiling, not as a reconciliation target

                                    ## Why your local numbers can differ

                                    1. Native C++ kernels not loaded (check: RFTMW_NO_NATIVE env var set?)
                                    2. Running fused path on one machine, preloaded path on the other
                                    3. Different CPU (AVX2 vs AVX-512 vs no AVX), core count, thermal throttling
                                    4. Different BLAS backend or thread count (OMP_NUM_THREADS, etc.)
                                    5. Different prompt length or token count in measurement
                                    6. Measuring micro-benchmark latency vs end-to-end decode throughput
                                    7. Comparing against Ollama numbers when running our engine (apples vs oranges)

                                    ## Required reconciliation protocol

                                    Step 1 — Print env vars before each run:
                                      echo RFTMW_NO_NATIVE=$RFTMW_NO_NATIVE
                                        echo RFTMW_REQUIRE_NATIVE=$RFTMW_REQUIRE_NATIVE
                                          echo OMP_NUM_THREADS=$OMP_NUM_THREADS
                                            echo OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS
                                              echo MKL_NUM_THREADS=$MKL_NUM_THREADS

                                              Step 2 — Confirm native infer loaded:
                                                python -c "from quantonium_os_src.engine.rftmw_numpy_engine import _get_native_infer; ni = _get_native_infer(); print('native available:', ni is not None)"

                                                Step 3 — Run and report these three metrics separately:
                                                  - fused_q6k_down_proj_latency_ms    (no preload, native path)
                                                    - fp32_preloaded_down_proj_latency_ms (after preload_dense)
                                                      - e2e_decode_tok_per_s              (5 tokens, decode phase only)

                                                      Step 4 — Use identical settings:
                                                        - model: qwen2.5-coder_3b.rftmwpk
                                                          - prompt: "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                                                            - max_tokens: 5
                                                              - temperature: 0.3
                                                                - top_k: 40
                                                                  - 1 warmup run, then 3 measured runs, report median

                                                                  Step 5 — Do NOT merge metrics into a single "speed" number.

                                                                  ## Deliverable format

                                                                  Return a table:

                                                                  | mode | preload_time_s | down_proj_latency_ms | decode_tok_per_s | env_summary | notes |
                                                                  |------|----------------|----------------------|------------------|-------------|-------|
                                                                  | ...  | ...            | ...                  | ...              | ...         | ...   |

                                                                  Then include a one-paragraph root-cause summary explaining any mismatch
                                                                  with the Codespace baseline table above.
                                                                  ````

                                                                  ---

                                                                  ## Quick run checklist

                                                                  - [ ] Close heavy background tasks, let machine idle before measuring
                                                                  - [ ] Pin thread counts explicitly (e.g. OMP_NUM_THREADS=4)
                                                                  - [ ] Run 3 measured trials and report median
                                                                  - [ ] Keep prompt and model path identical between environments
                                                                  - [ ] If numbers still differ wildly, check `lscpu` / CPU-Z output and include it
                                                                  # Local Agent Handoff: Reconcile Number Mismatch

                                                                  Copy and paste the block below into your local agent exactly as written.

                                                                  ```text
                                                                  Please reconcile benchmark number differences against Codespace baseline for QuantoniumOS.

                                                                  What we already established:
                                                                  1. There are multiple valid performance numbers depending on execution path:
                                                                  - Q6K fused path measures dequant + matmul directly on quantized weights.
                                                                  - FP32 preload path measures BLAS matmul after converting weights to dense tensors.
                                                                  - End-to-end tok/s includes decode loop behavior and sampling overhead.

                                                                  2. In this repo, these paths are defined here:
                                                                  - quantonium_os_src/engine/rftmw_numpy_engine.py (fused native q4k/q6k matmul path)
                                                                  - quantonium_os_src/engine/rftmw_numpy_engine.py (preload_dense path and preload timing print)
                                                                  - src/apps/chatbox_network.py (chat app intentionally skips preload in native mode)
                                                                  - src/apps/chatbox_network.py (tok/s computed over decode phase)

                                                                  3. Native path is forced by env var in chat loader:
                                                                  - RFTMW_REQUIRE_NATIVE=1

                                                                  Why local numbers can differ:
                                                                  1. Native kernels are not loaded, or are disabled by env vars.
                                                                  2. One machine runs FP32 preloaded mode while the other runs fused quantized mode.
                                                                  3. Different CPU features, thread settings, thermal state, or power policy.
                                                                  4. Different prompt length, warmup behavior, random seed, or output token count.
                                                                  5. Comparing microbenchmark latency to end-to-end throughput.

                                                                  Required reconciliation protocol:
                                                                  1. Print and log these env vars before each run:
                                                                  - RFTMW_NO_NATIVE
                                                                  - RFTMW_REQUIRE_NATIVE
                                                                  - OMP_NUM_THREADS
                                                                  - OPENBLAS_NUM_THREADS
                                                                  - MKL_NUM_THREADS

                                                                  2. Confirm native infer availability before timing.

                                                                  3. Run and report metrics separately with exact labels:
                                                                  - fused_q6k_down_proj_latency_ms
                                                                  - fp32_preloaded_down_proj_latency_ms
                                                                  - e2e_decode_tok_per_s

                                                                  4. Use identical benchmark settings:
                                                                  - same model file
                                                                  - same prompt text
                                                                  - same max tokens
                                                                  - fixed seed
                                                                  - one warmup run before measured runs

                                                                  5. Do not merge metrics into a single “speed” number.

                                                                  Expected interpretation:
                                                                  - Fused Q6K may have higher per-op latency because dequant is included.
                                                                  - FP32 preloaded path usually lowers matmul latency after a large startup preload cost.
                                                                  - Ollama/llama.cpp may still show higher tok/s due to mmap/runtime pipeline optimizations.

                                                                  Deliverable format:
                                                                  Return a compact table with columns:
                                                                  - mode
                                                                  - preload_time_s
                                                                  - down_proj_latency_ms
                                                                  - decode_tok_per_s
                                                                  - env_summary
                                                                  - notes

                                                                  Then include a one-paragraph root-cause summary for any mismatch with Codespace.
                                                                  ```

                                                                  ## Optional: quick run checklist

                                                                  - Use one machine idle state (close heavy background tasks)
                                                                  - Pin thread counts explicitly
                                                                  - Run 3 measured trials and report median
                                                                  - Keep prompt and model path identical between environments
                                                                  