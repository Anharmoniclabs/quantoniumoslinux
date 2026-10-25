/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_streaming_engine.hpp — Layer-Streaming Inference Engine
 * =============================================================
 *
 * Runs a full transformer forward pass with only 1-2 layers of weights
 * resident in memory at any time.  Designed to compile to both:
 *   - Native (Linux/macOS/Windows) via mmap or file I/O
 *   - WASM (browser) via fetch + heap alloc
 *
 * Memory budget for a 7B model:
 *   Embeddings + LM head:  ~150 MB (always resident)
 *   1-2 active layers:     ~250 MB (loaded, used, freed)
 *   KV-cache (compressed): ~80 MB  (RFT-compressed, see rftmw_kv_compress.hpp)
 *   Activations:           ~100 MB
 *   Total:                 ~630 MB
 *
 * The engine reads "RFTL" chunk files produced by split_gguf_layers.
 * Weights stay in Q4K/Q6K format — no re-encoding, no quality loss.
 * All inference kernels come from rftmw_inference.hpp (proven, verified).
 */

#pragma once

#include "rftmw_inference.hpp"
#include "rftmw_kv_compress.hpp"
#include "rftmw_spectral_inference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rftmw {
namespace streaming {

// ── GGUF type codes ──────────────────────────────────────────────
constexpr uint32_t GGUF_F32  = 0;
constexpr uint32_t GGUF_F16  = 1;
constexpr uint32_t GGUF_Q4_K = 12;
constexpr uint32_t GGUF_Q6_K = 14;

// ── RFTL chunk format reader ─────────────────────────────────────

struct ChunkTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t data_size = 0;
};

struct ChunkReader {
    std::vector<uint8_t> owned_data;  // holds the chunk bytes
    std::vector<ChunkTensor> tensors;

    bool load(const uint8_t* raw, size_t len) {
        // Copy into owned buffer so tensor pointers remain valid
        owned_data.assign(raw, raw + len);
        return parse(owned_data.data(), len);
    }

    bool load_from_file(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        size_t sz = static_cast<size_t>(std::ftell(f));
        std::fseek(f, 0, SEEK_SET);
        owned_data.resize(sz);
        size_t read = std::fread(owned_data.data(), 1, sz, f);
        std::fclose(f);
        if (read != sz) return false;
        return parse(owned_data.data(), sz);
    }

private:
    bool parse(const uint8_t* raw, size_t len) {
        if (len < 16) return false;
        const uint8_t* p = raw;
        if (std::memcmp(p, "RFTL", 4) != 0) return false;
        p += 4;

        uint32_t version, num_tensors, reserved;
        std::memcpy(&version, p, 4);     p += 4;
        std::memcpy(&num_tensors, p, 4); p += 4;
        std::memcpy(&reserved, p, 4);    p += 4;

        if (version != 1) return false;

        tensors.resize(num_tensors);
        for (uint32_t i = 0; i < num_tensors; ++i) {
            auto& t = tensors[i];

            uint32_t name_len;
            std::memcpy(&name_len, p, 4); p += 4;
            t.name.assign(reinterpret_cast<const char*>(p), name_len);
            p += name_len;

            std::memcpy(&t.type, p, 4); p += 4;

            uint32_t n_dims;
            std::memcpy(&n_dims, p, 4); p += 4;
            t.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                uint64_t dim_val;
                std::memcpy(&dim_val, p, 8); p += 8;
                t.shape[d] = dim_val;
            }

            std::memcpy(&t.data_size, p, 8); p += 8;
            t.data = p;
            p += t.data_size;

            // Skip 32-byte alignment padding
            size_t pos = static_cast<size_t>(p - raw);
            size_t aligned = (pos + 31u) & ~31u;
            p = raw + aligned;
        }
        return true;
    }

public:

    const ChunkTensor* find(const std::string& name) const {
        for (const auto& t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    }

    void release() {
        owned_data.clear();
        owned_data.shrink_to_fit();
        tensors.clear();
    }
};

// ── fp16 → f32 ───────────────────────────────────────────────────

inline float fp16_to_float(uint16_t h) {
    return inference::fp16_to_f32(h);
}

inline std::vector<float> tensor_to_f32(const ChunkTensor& t) {
    uint64_t n = 1;
    for (auto d : t.shape) n *= d;
    std::vector<float> out(static_cast<size_t>(n));
    if (t.type == GGUF_F32)
        std::memcpy(out.data(), t.data, static_cast<size_t>(n) * sizeof(float));
    else if (t.type == GGUF_F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(t.data);
        for (uint64_t i = 0; i < n; ++i)
            out[static_cast<size_t>(i)] = fp16_to_float(p[i]);
    } else
        throw std::runtime_error("tensor not dense f32/f16: " + t.name);
    return out;
}

// ── Weight2D wrapper ─────────────────────────────────────────────

struct StreamWeight2D {
    const uint8_t* raw_data = nullptr;
    int in = 0, out = 0, bytes_per_row = 0;
    uint32_t type = 0;
    std::vector<float> dense;  // only for F32/F16

    void bind(const ChunkTensor& t) {
        if (t.shape.size() < 2) throw std::runtime_error("2D tensor expected: " + t.name);
        in  = static_cast<int>(t.shape[0]);
        out = static_cast<int>(t.shape[1]);
        type = t.type;
        raw_data = t.data;

        if (t.type == GGUF_Q4_K) bytes_per_row = (in / 256) * 144;
        else if (t.type == GGUF_Q6_K) bytes_per_row = (in / 256) * 210;
        else dense = tensor_to_f32(t);
    }

    void matmul(const float* x, float* y) const {
        if (!dense.empty()) {
            inference::dense_sgemv_f32(dense.data(), x, y, out, in);
        } else if (type == GGUF_Q4_K) {
            inference::q4k_matmul_vec(raw_data, x, y, out, bytes_per_row);
        } else {
            inference::q6k_matmul_vec(raw_data, x, y, out, bytes_per_row);
        }
    }

    // Dequantize a single row to float32 (for embedding lookup)
    void dequant_row(int row, float* out_buf) const {
        if (!dense.empty()) {
            std::memcpy(out_buf, dense.data() + static_cast<size_t>(row) * in,
                        static_cast<size_t>(in) * sizeof(float));
            return;
        }
        const uint8_t* row_data = raw_data + static_cast<size_t>(row) * bytes_per_row;
        if (type == GGUF_Q4_K) {
            int nblocks = bytes_per_row / 144;
            for (int b = 0; b < nblocks; ++b)
                inference::dequant_q4k_block(row_data + static_cast<size_t>(b) * 144,
                                             out_buf + static_cast<size_t>(b) * 256);
        } else {
            int nblocks = bytes_per_row / 210;
            for (int b = 0; b < nblocks; ++b)
                inference::dequant_q6k_block(row_data + static_cast<size_t>(b) * 210,
                                             out_buf + static_cast<size_t>(b) * 256);
        }
    }
};

// ── Model config ─────────────────────────────────────────────────

struct StreamingConfig {
    int layers       = 36;
    int hidden       = 2048;
    int heads        = 16;
    int kv_heads     = 2;
    int head_dim     = 128;
    int ffn_dim      = 11008;
    float rms_eps    = 1e-6f;
    double rope_theta = 1000000.0;
    int vocab        = 151936;
    int max_seq      = 2048;

    // KV compression
    float kv_keep_ratio = 0.60f;  // 60% spectral retention
    bool kv_compress_enabled = true;

    // Spectral weight inference (RFT-domain matmul)
    bool spectral_enabled = false;
    float spectral_keep_ratio = 0.60f;  // coefficient retention for spectral weights
    int  spectral_block_size = 128;     // FFT block size for spectral decomposition
};

// ── Streaming layer (loaded and freed per-layer) ─────────────────

struct StreamingLayer {
    ChunkReader chunk;

    std::vector<float> attn_norm;
    std::vector<float> ffn_norm;

    StreamWeight2D q_proj, k_proj, v_proj, o_proj;
    std::vector<float> q_bias, k_bias, v_bias;

    StreamWeight2D ffn_gate, ffn_up, ffn_down;

    bool loaded = false;

    // Spectral weight representations (populated when spectral mode is on)
    spectral::SpectralLayerWeights spectral_weights;
    bool spectral_mode = false;

    bool load_from_chunk(const uint8_t* data, size_t size, int layer_idx) {
        if (!chunk.load(data, size)) return false;
        return bind_tensors(layer_idx);
    }

    bool load_from_file(const std::string& path, int layer_idx) {
        if (!chunk.load_from_file(path)) return false;
        return bind_tensors(layer_idx);
    }

    // Convert Q4K/Q6K weights to spectral form for RFT-domain matmul
    void convert_to_spectral(const spectral::SpectralConfig& scfg, int layer_idx) {
        if (!loaded) return;
        auto convert = [&](StreamWeight2D& w, spectral::SpectralWeight& sw, const char* name) {
            if (w.raw_data && w.type != GGUF_F32 && w.type != GGUF_F16) {
                const char* qt = (w.type == GGUF_Q4_K) ? "q4k" : "q6k";
                spectral::convert_layer_weight(
                    sw, w.raw_data, w.out, w.in, w.bytes_per_row,
                    qt, scfg, layer_idx, name);
            }
        };
        std::string p = "blk." + std::to_string(layer_idx) + ".";
        convert(q_proj,   spectral_weights.attn_q,    (p + "attn_q").c_str());
        convert(k_proj,   spectral_weights.attn_k,    (p + "attn_k").c_str());
        convert(v_proj,   spectral_weights.attn_v,    (p + "attn_v").c_str());
        convert(o_proj,   spectral_weights.attn_o,    (p + "attn_o").c_str());
        convert(ffn_gate, spectral_weights.ffn_gate,  (p + "ffn_gate").c_str());
        convert(ffn_up,   spectral_weights.ffn_up,    (p + "ffn_up").c_str());
        convert(ffn_down, spectral_weights.ffn_down,  (p + "ffn_down").c_str());
        spectral_mode = true;
    }

    void unload() {
        chunk.release();
        attn_norm.clear(); attn_norm.shrink_to_fit();
        ffn_norm.clear();  ffn_norm.shrink_to_fit();
        q_bias.clear(); q_bias.shrink_to_fit();
        k_bias.clear(); k_bias.shrink_to_fit();
        v_bias.clear(); v_bias.shrink_to_fit();
        q_proj = {}; k_proj = {}; v_proj = {}; o_proj = {};
        ffn_gate = {}; ffn_up = {}; ffn_down = {};
        loaded = false;
    }

private:
    bool bind_tensors(int layer_idx) {
        std::string p = "blk." + std::to_string(layer_idx) + ".";

        auto find_or_throw = [&](const std::string& name) -> const ChunkTensor& {
            const ChunkTensor* t = chunk.find(name);
            if (!t) throw std::runtime_error("missing tensor: " + name);
            return *t;
        };

        attn_norm = tensor_to_f32(find_or_throw(p + "attn_norm.weight"));
        ffn_norm  = tensor_to_f32(find_or_throw(p + "ffn_norm.weight"));

        q_proj.bind(find_or_throw(p + "attn_q.weight"));
        k_proj.bind(find_or_throw(p + "attn_k.weight"));
        v_proj.bind(find_or_throw(p + "attn_v.weight"));
        o_proj.bind(find_or_throw(p + "attn_output.weight"));

        q_bias = tensor_to_f32(find_or_throw(p + "attn_q.bias"));
        k_bias = tensor_to_f32(find_or_throw(p + "attn_k.bias"));
        v_bias = tensor_to_f32(find_or_throw(p + "attn_v.bias"));

        ffn_gate.bind(find_or_throw(p + "ffn_gate.weight"));
        ffn_up.bind(find_or_throw(p + "ffn_up.weight"));
        ffn_down.bind(find_or_throw(p + "ffn_down.weight"));

        loaded = true;
        return true;
    }
};

// ── KV cache per layer ───────────────────────────────────────────

struct LayerKVCache {
    // Uncompressed KV for the current position (written during attention)
    // Compressed KV for past positions (managed by kv_compress)
    std::vector<float> k_cache;  // [max_seq, kv_heads, head_dim]
    std::vector<float> v_cache;  // [max_seq, kv_heads, head_dim]

    void init(int max_seq, int kv_heads, int head_dim) {
        size_t sz = static_cast<size_t>(max_seq) * kv_heads * head_dim;
        k_cache.assign(sz, 0.0f);
        v_cache.assign(sz, 0.0f);
    }
};

// ── Streaming Model ──────────────────────────────────────────────

class StreamingModel {
public:
    StreamingConfig cfg;

    // Callback for loading layer chunks (set by the host — file I/O or fetch)
    using LoadLayerFn = std::function<bool(int layer_idx, std::vector<uint8_t>& out_data)>;
    LoadLayerFn load_layer_fn;

    // Embedding/output weights (always resident)
    ChunkReader embed_chunk;
    StreamWeight2D embed_weight;
    StreamWeight2D lm_head_weight;
    std::vector<float> output_norm;

    // KV caches (always resident, one per layer)
    std::vector<LayerKVCache> kv_caches;

    // KV compressor
    kv_compress::KVCompressor kv_compressor;

    // Working buffers (reused across layers)
    std::vector<float> hidden_state;
    std::vector<float> buf_q, buf_k, buf_v;
    std::vector<float> buf_attn_out, buf_normed;
    std::vector<float> buf_gate, buf_up, buf_ffn;
    std::vector<float> buf_logits;

    StreamingModel() = default;

    // Load embeddings (always in memory)
    bool load_embeddings(const std::string& path) {
        if (!embed_chunk.load_from_file(path)) return false;

        const ChunkTensor* emb = embed_chunk.find("token_embd.weight");
        if (!emb) return false;
        embed_weight.bind(*emb);

        const ChunkTensor* lm = embed_chunk.find("output.weight");
        if (!lm) return false;
        lm_head_weight.bind(*lm);

        const ChunkTensor* onorm = embed_chunk.find("output_norm.weight");
        if (!onorm) return false;
        output_norm = tensor_to_f32(*onorm);

        return true;
    }

    bool load_embeddings(const uint8_t* data, size_t size) {
        if (!embed_chunk.load(data, size)) return false;
        // Same binding as above
        const ChunkTensor* emb = embed_chunk.find("token_embd.weight");
        if (!emb) return false;
        embed_weight.bind(*emb);

        const ChunkTensor* lm = embed_chunk.find("output.weight");
        if (!lm) return false;
        lm_head_weight.bind(*lm);

        const ChunkTensor* onorm = embed_chunk.find("output_norm.weight");
        if (!onorm) return false;
        output_norm = tensor_to_f32(*onorm);
        return true;
    }

    // Initialize KV caches and working buffers
    void init() {
        kv_caches.resize(static_cast<size_t>(cfg.layers));
        for (auto& kv : kv_caches)
            kv.init(cfg.max_seq, cfg.kv_heads, cfg.head_dim);

        // Init KV compressor
        if (cfg.kv_compress_enabled) {
            kv_compress::KVCompressConfig kv_cfg;
            kv_cfg.head_dim = cfg.head_dim;
            kv_cfg.num_kv_heads = cfg.kv_heads;
            kv_cfg.base_keep_ratio = cfg.kv_keep_ratio;
            kv_cfg.total_layers = cfg.layers;
            kv_compressor.init(kv_cfg);
        }

        // Pre-allocate working buffers
        hidden_state.resize(static_cast<size_t>(cfg.hidden));
        buf_q.resize(static_cast<size_t>(cfg.heads * cfg.head_dim));
        buf_k.resize(static_cast<size_t>(cfg.kv_heads * cfg.head_dim));
        buf_v.resize(static_cast<size_t>(cfg.kv_heads * cfg.head_dim));
        buf_attn_out.resize(static_cast<size_t>(cfg.heads * cfg.head_dim));
        buf_normed.resize(static_cast<size_t>(cfg.hidden));
        buf_gate.resize(static_cast<size_t>(cfg.ffn_dim));
        buf_up.resize(static_cast<size_t>(cfg.ffn_dim));
        buf_ffn.resize(static_cast<size_t>(cfg.hidden));
        buf_logits.resize(static_cast<size_t>(cfg.vocab));
    }

    // Run one forward pass for a single token
    // Returns logits (vocab-sized)
    const float* forward_token(int token_id, int pos, int cache_len) {
        // Embedding lookup
        embed_weight.dequant_row(token_id, hidden_state.data());

        // Layer-by-layer forward (load → compute → unload)
        for (int i = 0; i < cfg.layers; ++i) {
            forward_layer(i, pos, cache_len);
        }

        // Final norm + LM head
        inference::rms_norm(hidden_state.data(), output_norm.data(),
                            buf_normed.data(), cfg.hidden, cfg.rms_eps);
        lm_head_weight.matmul(buf_normed.data(), buf_logits.data());

        return buf_logits.data();
    }

    // Greedy argmax over logits
    int sample_greedy(const float* logits, int vocab_size) const {
        return static_cast<int>(
            std::max_element(logits, logits + vocab_size) - logits);
    }

private:
    // Forward a single layer — loads weights, computes, frees weights
    void forward_layer(int layer_idx, int pos, int cache_len) {
        StreamingLayer layer;

        // Load layer weights (data kept alive for the full function scope)
        std::vector<uint8_t> layer_data;
        if (load_layer_fn) {
            if (!load_layer_fn(layer_idx, layer_data))
                throw std::runtime_error("failed to load layer " + std::to_string(layer_idx));
            if (!layer.load_from_chunk(layer_data.data(), layer_data.size(), layer_idx))
                throw std::runtime_error("failed to parse layer " + std::to_string(layer_idx));
        } else {
            throw std::runtime_error("no load_layer_fn set");
        }

        // Optionally convert to spectral form (RFT-domain matmul)
        if (cfg.spectral_enabled) {
            spectral::SpectralConfig scfg;
            scfg.block_size = cfg.spectral_block_size;
            scfg.keep_ratio = cfg.spectral_keep_ratio;
            scfg.enabled = true;
            scfg.nonuniform = true;
            scfg.total_layers = cfg.layers;
            layer.convert_to_spectral(scfg, layer_idx);
        }

        auto& kv = kv_caches[static_cast<size_t>(layer_idx)];
        const int nh  = cfg.heads;
        const int nkv = cfg.kv_heads;
        const int hd  = cfg.head_dim;
        const int kv_stride = nkv * hd;

        // Helper: use spectral matmul when available, else direct Q4K/Q6K
        auto do_matmul = [&](StreamWeight2D& w, spectral::SpectralWeight& sw,
                             const float* x, float* y) {
            if (layer.spectral_mode && sw.is_initialized()) {
                sw.matmul_vec(x, y);
            } else {
                w.matmul(x, y);
            }
        };

        // ── Attention ──
        inference::rms_norm(hidden_state.data(), layer.attn_norm.data(),
                            buf_normed.data(), cfg.hidden, cfg.rms_eps);

        do_matmul(layer.q_proj, layer.spectral_weights.attn_q,
                  buf_normed.data(), buf_q.data());
        do_matmul(layer.k_proj, layer.spectral_weights.attn_k,
                  buf_normed.data(), buf_k.data());
        do_matmul(layer.v_proj, layer.spectral_weights.attn_v,
                  buf_normed.data(), buf_v.data());

        // Add bias
        for (int j = 0; j < nh * hd; ++j)  buf_q[j] += layer.q_bias[j];
        for (int j = 0; j < nkv * hd; ++j) buf_k[j] += layer.k_bias[j];
        for (int j = 0; j < nkv * hd; ++j) buf_v[j] += layer.v_bias[j];

        // RoPE
        apply_rope_halved(buf_q.data(), nh,  hd, pos, cfg.rope_theta);
        apply_rope_halved(buf_k.data(), nkv, hd, pos, cfg.rope_theta);

        // Write to KV cache
        std::memcpy(kv.k_cache.data() + static_cast<size_t>(cache_len) * kv_stride,
                     buf_k.data(), static_cast<size_t>(kv_stride) * sizeof(float));
        std::memcpy(kv.v_cache.data() + static_cast<size_t>(cache_len) * kv_stride,
                     buf_v.data(), static_cast<size_t>(kv_stride) * sizeof(float));

        // GQA attention
        int seq_len = cache_len + 1;
        std::fill(buf_attn_out.begin(), buf_attn_out.end(), 0.0f);
        inference::gqa_attention(
            buf_q.data(),
            kv.k_cache.data(),
            kv.v_cache.data(),
            buf_attn_out.data(),
            nh, nkv, hd, seq_len);

        // O projection
        std::vector<float> o_out(static_cast<size_t>(cfg.hidden));
        do_matmul(layer.o_proj, layer.spectral_weights.attn_o,
                  buf_attn_out.data(), o_out.data());

        // Residual
        for (int j = 0; j < cfg.hidden; ++j)
            hidden_state[j] += o_out[j];

        // ── SwiGLU FFN ──
        inference::rms_norm(hidden_state.data(), layer.ffn_norm.data(),
                            buf_normed.data(), cfg.hidden, cfg.rms_eps);

        do_matmul(layer.ffn_gate, layer.spectral_weights.ffn_gate,
                  buf_normed.data(), buf_gate.data());
        do_matmul(layer.ffn_up, layer.spectral_weights.ffn_up,
                  buf_normed.data(), buf_up.data());

        // SiLU(gate) * up
        inference::silu_mul(buf_gate.data(), buf_up.data(),
                            buf_gate.data(), cfg.ffn_dim);

        do_matmul(layer.ffn_down, layer.spectral_weights.ffn_down,
                  buf_gate.data(), buf_ffn.data());

        // Residual
        for (int j = 0; j < cfg.hidden; ++j)
            hidden_state[j] += buf_ffn[j];

        // Compress KV cache for this layer (past positions only)
        if (cfg.kv_compress_enabled && cache_len > 0) {
            kv_compressor.compress_layer(
                layer_idx, cache_len,
                kv.k_cache.data(), kv.v_cache.data());
        }

        // Layer weights are freed when `layer` goes out of scope
    }

    // Half-split RoPE (matching qwen25_native_cli)
    static void apply_rope_halved(float* x, int n_heads, int hd,
                                  int pos, double theta) {
        int half = hd / 2;
        for (int h = 0; h < n_heads; ++h) {
            float* row = x + static_cast<size_t>(h) * hd;
            for (int i = 0; i < half; ++i) {
                double freq = 1.0 / std::pow(theta, static_cast<double>(2 * i) / hd);
                float c = static_cast<float>(std::cos(pos * freq));
                float s = static_cast<float>(std::sin(pos * freq));
                float x0 = row[i];
                float x1 = row[half + i];
                row[i]        = x0 * c - x1 * s;
                row[half + i] = x1 * c + x0 * s;
            }
        }
    }
};

// ── C API for WASM interop ───────────────────────────────────────
// These functions are exported with C linkage so JavaScript can call them
// through the WASM module's exported function table.

} // namespace streaming
} // namespace rftmw

#ifdef RFTMW_WASM_BUILD

extern "C" {

static rftmw::streaming::StreamingModel* g_model = nullptr;

int streaming_model_create(int layers, int hidden, int heads, int kv_heads,
                           int head_dim, int ffn_dim, int vocab, int max_seq,
                           float kv_keep_ratio) {
    if (g_model) { delete g_model; g_model = nullptr; }
    g_model = new rftmw::streaming::StreamingModel();
    g_model->cfg.layers = layers;
    g_model->cfg.hidden = hidden;
    g_model->cfg.heads = heads;
    g_model->cfg.kv_heads = kv_heads;
    g_model->cfg.head_dim = head_dim;
    g_model->cfg.ffn_dim = ffn_dim;
    g_model->cfg.vocab = vocab;
    g_model->cfg.max_seq = max_seq;
    g_model->cfg.kv_keep_ratio = kv_keep_ratio;
    g_model->cfg.kv_compress_enabled = (kv_keep_ratio < 1.0f);
    return 0;
}

int streaming_model_load_embeddings(const uint8_t* data, int size) {
    if (!g_model) return -1;
    if (!g_model->load_embeddings(data, static_cast<size_t>(size))) return -1;
    g_model->init();
    return 0;
}

// Forward a single token. Returns pointer to logits (vocab-sized float array).
// The pointer is valid until the next call to streaming_model_forward.
const float* streaming_model_forward(int token_id, int pos, int cache_len) {
    if (!g_model) return nullptr;
    return g_model->forward_token(token_id, pos, cache_len);
}

int streaming_model_vocab_size() {
    return g_model ? g_model->cfg.vocab : 0;
}

void streaming_model_destroy() {
    delete g_model;
    g_model = nullptr;
}

// Layer loading callback for WASM — called from JS via a registered function
static std::vector<uint8_t> g_pending_layer_data;

void streaming_model_set_layer_data(const uint8_t* data, int size) {
    g_pending_layer_data.assign(data, data + size);
}

} // extern "C"

#endif // RFTMW_WASM_BUILD
