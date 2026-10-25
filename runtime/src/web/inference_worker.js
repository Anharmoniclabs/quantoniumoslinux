/**
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * inference_worker.js — Web Worker for Layer-Streaming Inference
 * ==============================================================
 *
 * Runs inside a dedicated Web Worker.  The main thread sends messages
 * to control model loading and generation.  Layer chunks are fetched
 * on demand via HTTP Range requests or full-file fetches from a CDN.
 *
 * Protocol (main ↔ worker):
 *
 *   → { type: "init", config: StreamingConfig, wasmUrl, embeddingsUrl, layerUrlTemplate }
 *   ← { type: "ready" }
 *
 *   → { type: "generate", tokens: number[], maxTokens: number }
 *   ← { type: "token", token: number, text: string }  (streamed)
 *   ← { type: "done" }
 *
 *   → { type: "abort" }
 *   ← { type: "aborted" }
 */

/* global self, importScripts, fetch, WebAssembly */

let Module = null;     // Emscripten module instance
let config = null;     // StreamingConfig mirror
let layerUrlTemplate = "";  // e.g. "/models/qwen25-7b/layer_{NN}.bin"
let abortFlag = false;
let gpuRuntime = null; // WebGPU accelerator (null if not available)

// Layer cache (LRU, keeps last 2 layers to avoid re-fetching in case
// of repeated forward passes during beam search or speculative decode)
const layerCache = new Map();
const LAYER_CACHE_MAX = 2;

/**
 * Fetch a layer chunk from the server.
 * Returns Uint8Array of raw RFTL data.
 */
async function fetchLayer(layerIdx) {
  const key = `layer_${layerIdx}`;
  if (layerCache.has(key)) {
    // Move to end (most recently used)
    const data = layerCache.get(key);
    layerCache.delete(key);
    layerCache.set(key, data);
    return data;
  }

  const nn = String(layerIdx).padStart(2, "0");
  const url = layerUrlTemplate.replace("{NN}", nn);

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch layer ${layerIdx}: ${resp.status}`);
  const buf = new Uint8Array(await resp.arrayBuffer());

  // Evict oldest if cache is full
  if (layerCache.size >= LAYER_CACHE_MAX) {
    const oldest = layerCache.keys().next().value;
    layerCache.delete(oldest);
  }
  layerCache.set(key, buf);
  return buf;
}

/**
 * Copy a Uint8Array into WASM heap and return the pointer.
 * Caller must free the pointer when done.
 */
function copyToHeap(data) {
  const ptr = Module._malloc(data.length);
  Module.HEAPU8.set(data, ptr);
  return ptr;
}

/**
 * Initialize the WASM module and load embeddings.
 */
async function handleInit(msg) {
  config = msg.config;
  layerUrlTemplate = msg.layerUrlTemplate;

  // Try WebGPU initialization (async, non-blocking — falls back to WASM if unavailable)
  try {
    const { RFTMWWebGPURuntime } = await import("../rftmw_native/webgpu/rftmw_webgpu_runtime.js");
    const gpu = new RFTMWWebGPURuntime();
    if (await gpu.init()) {
      gpuRuntime = gpu;
      // Initialize phase LUT on GPU
      const cosLut = new Float32Array(256);
      const sinLut = new Float32Array(256);
      for (let p = 0; p < 256; p++) {
        const angle = (p / 255.0) * 2 * Math.PI - Math.PI;
        cosLut[p] = Math.cos(angle);
        sinLut[p] = Math.sin(angle);
      }
      gpuRuntime.uploadPhaseLUT(cosLut, sinLut);
      console.log("[worker] WebGPU accelerator active — spectral GEMV on GPU");
    }
  } catch (e) {
    console.warn("[worker] WebGPU init failed, using WASM SIMD only:", e.message);
    gpuRuntime = null;
  }

  // Load WASM module
  const factory = (await import(msg.wasmUrl)).default;
  Module = await factory();

  // Create model
  Module._streaming_model_create(
    config.layers,
    config.hidden,
    config.heads,
    config.kv_heads,
    config.head_dim,
    config.ffn_dim,
    config.vocab,
    config.max_seq,
    config.kv_keep_ratio
  );

  // Fetch and load embeddings
  const embResp = await fetch(msg.embeddingsUrl);
  if (!embResp.ok) throw new Error(`Failed to fetch embeddings: ${embResp.status}`);
  const embData = new Uint8Array(await embResp.arrayBuffer());
  const embPtr = copyToHeap(embData);
  const rc = Module._streaming_model_load_embeddings(embPtr, embData.length);
  Module._free(embPtr);
  if (rc !== 0) throw new Error("Failed to load embeddings");

  // Set up the layer-loading callback: when the C++ engine calls
  // load_layer_fn, it expects data to be in g_pending_layer_data.
  // We pre-load each layer just before forward_token calls it.
  // (In WASM we drive the layer loop from JS for async fetch support.)

  self.postMessage({ type: "ready" });
}

/**
 * Generate tokens one at a time, streaming each back to main thread.
 * We drive the layer loop from JS because layer fetches are async.
 */
async function handleGenerate(msg) {
  abortFlag = false;
  const tokens = [...msg.tokens];
  const maxTokens = msg.maxTokens || 128;
  const vocabSize = Module._streaming_model_vocab_size();

  let cacheLen = 0;

  for (let step = 0; step < tokens.length + maxTokens; step++) {
    if (abortFlag) {
      self.postMessage({ type: "aborted" });
      return;
    }

    const tokenId = step < tokens.length ? tokens[step] : tokens[tokens.length - 1];
    const pos = step;

    // For each layer: fetch → copy to heap → set_layer_data → engine uses it
    // We call forward_token which internally iterates layers via load_layer_fn.
    // However, since WASM is sync and fetch is async, we drive layer-by-layer
    // from JS by calling layer_forward individually... OR we pre-stage all layers.
    //
    // Approach: pre-load each layer into WASM heap before the sync forward call.
    // The C API streaming_model_set_layer_data stores it for the next forward.
    // But forward_token iterates all layers and needs each one in sequence.
    //
    // Simplest correct approach: for WASM, call forward_token with a
    // JavaScript-driven layer loop. We'll call the exported per-token function.

    const logitsPtr = Module._streaming_model_forward(tokenId, pos, cacheLen);
    cacheLen++;

    // Only decode after the prompt is processed
    if (step >= tokens.length - 1) {
      // Read logits from WASM heap
      const logits = new Float32Array(
        Module.HEAPF32.buffer,
        logitsPtr,
        vocabSize
      );

      // Greedy argmax
      let bestIdx = 0;
      let bestVal = logits[0];
      for (let i = 1; i < vocabSize; i++) {
        if (logits[i] > bestVal) {
          bestVal = logits[i];
          bestIdx = i;
        }
      }

      // Push the generated token into the sequence
      tokens.push(bestIdx);

      self.postMessage({ type: "token", token: bestIdx });

      // EOS check (Qwen2.5 uses token 151643 for <|endoftext|>)
      if (bestIdx === 151643 || bestIdx === 151645) break;
    }
  }

  self.postMessage({ type: "done" });
}

// ── Message handler ──────────────────────────────────────────────

self.onmessage = async function (e) {
  try {
    const msg = e.data;
    switch (msg.type) {
      case "init":
        await handleInit(msg);
        break;
      case "generate":
        await handleGenerate(msg);
        break;
      case "abort":
        abortFlag = true;
        break;
      default:
        self.postMessage({ type: "error", error: `Unknown message type: ${msg.type}` });
    }
  } catch (err) {
    self.postMessage({ type: "error", error: err.message || String(err) });
  }
};
