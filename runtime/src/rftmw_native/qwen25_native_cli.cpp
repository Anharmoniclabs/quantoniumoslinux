/*
 * Native Qwen2.5-Coder-3B token runner for GGUF Q4_K/Q6_K packs.
 *
 * Dense transformer: 36 layers, 16 heads, 2 KV heads, dim 2048, FFN 11008.
 * Standard GQA + SwiGLU MLP.  No MoE, no SSM, no DeltaNet.
 *
 * Reuses rftmw_inference.hpp AVX2 kernels (dequant, matmul, rms_norm, RoPE).
 * mmaps the GGUF directly — zero Python, zero NumPy in the loop.
 */

#include "rftmw_inference.hpp"
#include "rftmw_mmap.hpp"
#include "rftmw_spectral_inference.hpp"
#include "rftmw_kv_compress.hpp"
#include "rftmw_streaming_engine.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

// ── GGUF type codes ──────────────────────────────────────────────
constexpr int GGUF_F32  = 0;
constexpr int GGUF_F16  = 1;
constexpr int GGUF_Q4_K = 12;
constexpr int GGUF_Q6_K = 14;

// ── Low-level GGUF parsing ───────────────────────────────────────

uint32_t read_u32(const uint8_t*& p) { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
uint64_t read_u64(const uint8_t*& p) { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
float    read_f32(const uint8_t*& p) { float    v; std::memcpy(&v, p, 4); p += 4; return v; }

std::string read_string(const uint8_t*& p) {
    uint64_t n = read_u64(p);
    std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
    p += n;
    return s;
}

void skip_gguf_value(const uint8_t*& p, uint32_t type);

void skip_gguf_scalar(const uint8_t*& p, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: p += 1; break;  // u8/i8/bool
        case 2: case 3: p += 2; break;           // u16/i16
        case 4: case 5: case 6: p += 4; break;   // u32/i32/f32
        case 8: { uint64_t n = read_u64(p); p += n; break; }  // string
        case 10: case 11: case 12: p += 8; break; // u64/i64/f64
        default: throw std::runtime_error("unsupported GGUF scalar type");
    }
}

void skip_gguf_value(const uint8_t*& p, uint32_t type) {
    if (type == 9) {
        uint32_t elem_type = read_u32(p);
        uint64_t n = read_u64(p);
        for (uint64_t i = 0; i < n; ++i) skip_gguf_value(p, elem_type);
    } else {
        skip_gguf_scalar(p, type);
    }
}

uint64_t gguf_tensor_nbytes(uint32_t type, const std::vector<uint64_t>& ne) {
    uint64_t n = 1;
    for (auto d : ne) n *= d;
    if (type == GGUF_F32)  return n * 4;
    if (type == GGUF_F16)  return n * 2;
    if (type == GGUF_Q4_K) return (n / 256) * 144;
    if (type == GGUF_Q6_K) return (n / 256) * 210;
    throw std::runtime_error("unsupported tensor type");
}

std::vector<std::string> read_string_array_value(const uint8_t*& p) {
    uint32_t elem_type = read_u32(p);
    uint64_t n = read_u64(p);
    if (elem_type != 8) throw std::runtime_error("expected GGUF string array");
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) out.push_back(read_string(p));
    return out;
}

std::vector<int> read_int_array_value(const uint8_t*& p) {
    uint32_t elem_type = read_u32(p);
    uint64_t n = read_u64(p);
    std::vector<int> out;
    out.reserve(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        if (elem_type == 5)      { int32_t v; std::memcpy(&v, p, 4); p += 4; out.push_back(v); }
        else if (elem_type == 4) { uint32_t v; std::memcpy(&v, p, 4); p += 4; out.push_back(static_cast<int>(v)); }
        else throw std::runtime_error("expected GGUF integer array");
    }
    return out;
}

// ── GGUF tensor header ──────────────────────────────────────────

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    const uint8_t* data = nullptr;
};

struct GGUFModel {
    rftmw::mmap::MmapFile file;
    std::unordered_map<std::string, TensorInfo> tensors;
    std::vector<std::string> tokenizer_tokens;
    std::vector<std::string> tokenizer_merges;
    std::vector<int> tokenizer_token_types;
    std::unordered_map<std::string, int64_t> meta_int;
    std::unordered_map<std::string, double>  meta_float;
    uint64_t data_start = 0;

    int64_t get_int(const std::string& suffix) const {
        for (auto& [k, v] : meta_int)
            if (k.size() >= suffix.size() && k.substr(k.size() - suffix.size()) == suffix)
                return v;
        return -1;
    }
    double get_float(const std::string& suffix, double def) const {
        for (auto& [k, v] : meta_float)
            if (k.size() >= suffix.size() && k.substr(k.size() - suffix.size()) == suffix)
                return v;
        return def;
    }

    void load(const fs::path& path) {
        if (!file.open(path.string()))
            throw std::runtime_error("failed to mmap GGUF: " + path.string());
        const uint8_t* base = static_cast<const uint8_t*>(file.data());
        const uint8_t* p = base;
        if (std::memcmp(p, "GGUF", 4) != 0) throw std::runtime_error("bad GGUF magic");
        p += 4;
        uint32_t version = read_u32(p);
        (void)version;
        uint64_t n_tensors = read_u64(p);
        uint64_t n_kv = read_u64(p);

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string(p);
            uint32_t type = read_u32(p);
            if (key == "tokenizer.ggml.tokens" && type == 9)
                tokenizer_tokens = read_string_array_value(p);
            else if (key == "tokenizer.ggml.merges" && type == 9)
                tokenizer_merges = read_string_array_value(p);
            else if (key == "tokenizer.ggml.token_type" && type == 9)
                tokenizer_token_types = read_int_array_value(p);
            else if (type == 4) { /* u32 */
                uint32_t v; std::memcpy(&v, p, 4); p += 4;
                meta_int[key] = static_cast<int64_t>(v);
            } else if (type == 5) { /* i32 */
                int32_t v; std::memcpy(&v, p, 4); p += 4;
                meta_int[key] = static_cast<int64_t>(v);
            } else if (type == 6) { /* f32 */
                float v; std::memcpy(&v, p, 4); p += 4;
                meta_float[key] = static_cast<double>(v);
            } else if (type == 10) { /* u64 */
                uint64_t v; std::memcpy(&v, p, 8); p += 8;
                meta_int[key] = static_cast<int64_t>(v);
            } else if (type == 12) { /* f64 */
                double v; std::memcpy(&v, p, 8); p += 8;
                meta_float[key] = v;
            } else {
                skip_gguf_value(p, type);
            }
        }

        std::vector<TensorInfo> tmp;
        tmp.reserve(static_cast<size_t>(n_tensors));
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = read_string(p);
            uint32_t n_dims = read_u32(p);
            for (uint32_t d = 0; d < n_dims; ++d) t.ne.push_back(read_u64(p));
            t.type = read_u32(p);
            t.offset = read_u64(p);
            t.bytes = gguf_tensor_nbytes(t.type, t.ne);
            tmp.push_back(std::move(t));
        }
        uint64_t pos = static_cast<uint64_t>(p - base);
        data_start = (pos + 31u) & ~31u;
        for (auto& t : tmp) {
            t.data = base + data_start + t.offset;
            tensors.emplace(t.name, std::move(t));
        }
    }

    const TensorInfo& tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }

    bool has_tensor(const std::string& name) const {
        return tensors.count(name) > 0;
    }
};

// ── UTF-8 / byte-level BPE helpers ──────────────────────────────

std::string utf8_from_codepoint(uint32_t cp) {
    std::string s;
    if (cp <= 0x7F) {
        s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { cps.push_back(c); ++i; }
        else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
            cps.push_back(((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i+1]) & 0x3F));
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
            cps.push_back(((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 6)
                          | (static_cast<unsigned char>(s[i+2]) & 0x3F));
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
            cps.push_back(((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 12)
                          | ((static_cast<unsigned char>(s[i+2]) & 0x3F) << 6)
                          | (static_cast<unsigned char>(s[i+3]) & 0x3F));
            i += 4;
        } else { cps.push_back(c); ++i; }
    }
    return cps;
}

// ── Tokenizer (byte-level BPE, Qwen-compatible) ─────────────────

struct NativeTokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<uint32_t, uint8_t> byte_decoder;
    std::string byte_encoder_map[256];
    int bos_id = -1, eos_id = -1, im_end_id = -1, eot_id = -1;

    void build_byte_maps() {
        std::vector<int> bs;
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        std::unordered_set<int> used(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!used.count(b)) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            byte_encoder_map[bs[i]] = utf8_from_codepoint(static_cast<uint32_t>(cs[i]));
            byte_decoder[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
        }
    }

    void init(const GGUFModel& m) {
        vocab = m.tokenizer_tokens;
        for (size_t i = 0; i < vocab.size(); ++i)
            token_to_id[vocab[i]] = static_cast<int>(i);
        for (auto& ms : m.tokenizer_merges) {
            size_t sp = ms.find(' ');
            if (sp != std::string::npos)
                merges.emplace_back(ms.substr(0, sp), ms.substr(sp + 1));
        }
        build_byte_maps();
        auto find = [&](const std::string& s) -> int {
            auto it = token_to_id.find(s);
            return it != token_to_id.end() ? it->second : -1;
        };
        bos_id    = find("<|im_start|>");
        im_end_id = find("<|im_end|>");
        eot_id    = find("<|endoftext|>");
        eos_id    = (im_end_id >= 0) ? im_end_id : eot_id;
    }

    bool is_stop_token(int id) const {
        return id == im_end_id || id == eot_id || (eos_id >= 0 && id == eos_id);
    }

    std::string decode_token(int id) const {
        if (id < 0 || id >= static_cast<int>(vocab.size())) return "";
        const std::string& tok = vocab[static_cast<size_t>(id)];
        // Byte-level BPE: convert codepoints back to raw bytes
        std::string out;
        for (uint32_t cp : utf8_to_codepoints(tok)) {
            auto it = byte_decoder.find(cp);
            if (it != byte_decoder.end())
                out.push_back(static_cast<char>(it->second));
            else
                out += utf8_from_codepoint(cp);
        }
        return out;
    }

    std::vector<int> encode(const std::string& text) const {
        // Byte-level BPE: map each byte through byte_encoder
        std::vector<std::string> tokens;
        for (unsigned char c : text) {
            tokens.push_back(byte_encoder_map[c]);
        }
        // BPE merge loop
        for (auto& [left, right] : merges) {
            size_t i = 0;
            while (i + 1 < tokens.size()) {
                if (tokens[i] == left && tokens[i + 1] == right) {
                    tokens[i] = left + right;
                    tokens.erase(tokens.begin() + static_cast<long>(i + 1));
                } else {
                    ++i;
                }
            }
        }
        std::vector<int> ids;
        ids.reserve(tokens.size());
        for (auto& t : tokens) {
            auto it = token_to_id.find(t);
            ids.push_back(it != token_to_id.end() ? it->second : 0);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) out += decode_token(id);
        return out;
    }
};

// ── Weight wrappers ──────────────────────────────────────────────

float fp16_to_float(uint16_t h) { return rftmw::inference::fp16_to_f32(h); }

std::vector<float> tensor_to_f32(const TensorInfo& t) {
    uint64_t n = 1;
    for (auto d : t.ne) n *= d;
    std::vector<float> out(static_cast<size_t>(n));
    if (t.type == GGUF_F32)
        std::memcpy(out.data(), t.data, static_cast<size_t>(n) * sizeof(float));
    else if (t.type == GGUF_F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(t.data);
        for (uint64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = fp16_to_float(p[i]);
    } else
        throw std::runtime_error("tensor not dense f32/f16: " + t.name);
    return out;
}

struct Weight2D {
    const TensorInfo* t = nullptr;
    int in = 0, out = 0, bytes_per_row = 0;
    std::vector<float> dense;

    void bind(const GGUFModel& m, const std::string& name) {
        t = &m.tensor(name);
        if (t->ne.size() < 2) throw std::runtime_error("2D tensor rank error: " + name);
        in  = static_cast<int>(t->ne[0]);
        out = static_cast<int>(t->ne[1]);
        if      (t->type == GGUF_Q4_K) bytes_per_row = (in / 256) * 144;
        else if (t->type == GGUF_Q6_K) bytes_per_row = (in / 256) * 210;
        else dense = tensor_to_f32(*t);
    }

    std::vector<float> matmul(const std::vector<float>& x) const {
        std::vector<float> y(static_cast<size_t>(out));
        matmul_into(x.data(), y.data());
        return y;
    }

    // Write into pre-allocated output buffer (no heap allocation)
    void matmul_into(const float* x, float* y) const {
        if (!dense.empty()) {
            rftmw::inference::dense_sgemv_f32(dense.data(), x, y, out, in);
        } else {
            const char* qt = (t->type == GGUF_Q4_K) ? "q4k" : "q6k";
            rftmw::inference::quant_matmul_vec(t->data, x, y, out, bytes_per_row, qt);
        }
    }

    void dequant_row(int row, std::vector<float>& y) const {
        y.assign(static_cast<size_t>(in), 0.0f);
        if (!dense.empty()) {
            std::memcpy(y.data(), dense.data() + static_cast<size_t>(row) * in,
                        static_cast<size_t>(in) * sizeof(float));
            return;
        }
        const uint8_t* row_data = t->data + static_cast<size_t>(row) * bytes_per_row;
        int nblocks = (t->type == GGUF_Q4_K) ? bytes_per_row / 144 : bytes_per_row / 210;
        for (int b = 0; b < nblocks; ++b) {
            if (t->type == GGUF_Q4_K)
                rftmw::inference::dequant_q4k_block(row_data + static_cast<size_t>(b) * 144,
                                                    y.data() + static_cast<size_t>(b) * 256);
            else
                rftmw::inference::dequant_q6k_block(row_data + static_cast<size_t>(b) * 210,
                                                    y.data() + static_cast<size_t>(b) * 256);
        }
    }
};

// ── Helpers ──────────────────────────────────────────────────────

void add_inplace(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

void add_inplace(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) a[i] += b[i];
}

std::vector<float> rms_norm_vec(const std::vector<float>& x,
                                const std::vector<float>& w, float eps) {
    std::vector<float> y(x.size());
    rftmw::inference::rms_norm(x.data(), w.data(), y.data(),
                               static_cast<int>(x.size()), eps);
    return y;
}

void rms_norm_into(const float* x, const float* w, float* y, int n, float eps) {
    rftmw::inference::rms_norm(x, w, y, n, eps);
}

// ── Qwen2.5-3B config ───────────────────────────────────────────

struct Qwen25Config {
    int layers      = 36;
    int hidden      = 2048;
    int heads       = 16;
    int kv_heads    = 2;
    int head_dim    = 128;   // hidden / heads
    int ffn_dim     = 11008;
    float rms_eps   = 1e-6f;
    double rope_theta = 1000000.0;
    int vocab       = 151936;
};

// ── Mimosa KV Gate (ported from algorithms/mimosa/mimosa_gate.py) ──

struct MimosaKVGate {
    float threshold          = 1.5f;
    float propagation_decay  = 0.6f;
    int   recovery_window    = 8;
    float habituation_rate   = 0.15f;
    int   max_hops           = 4;

    // Compute turgor scores for cached key representations.
    // key_norms: (cache_len,) — L2 norm of each cached key vector
    // Returns turgor: (cache_len,) — values in [0,1], lower = more evictable
    std::vector<float> compute_turgor(const std::vector<float>& key_norms) const {
        int n = static_cast<int>(key_norms.size());
        if (n == 0) return {};

        // Mean norm
        float mean_norm = 0.0f;
        for (int i = 0; i < n; ++i) mean_norm += key_norms[i];
        mean_norm = mean_norm / n + 1e-8f;

        // Relative activation (no habituation state in C++ version — simpler)
        std::vector<float> rel_act(n);
        std::vector<bool> triggered(n, false);
        for (int i = 0; i < n; ++i) {
            rel_act[i] = key_norms[i] / mean_norm;
            triggered[i] = rel_act[i] > threshold;
        }

        // Hydraulic propagation
        std::vector<float> hydraulic(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            if (!triggered[i]) continue;
            for (int hop = 0; hop <= max_hops; ++hop) {
                float signal = std::pow(propagation_decay, static_cast<float>(hop));
                if (i - hop >= 0)      hydraulic[i - hop] += signal;
                if (hop > 0 && i + hop < n) hydraulic[i + hop] += signal;
            }
        }

        // Turgor
        std::vector<float> turgor(n);
        for (int i = 0; i < n; ++i) {
            float fold_pressure = hydraulic[i] - rel_act[i];
            turgor[i] = std::max(0.0f, std::min(1.0f, 1.0f - fold_pressure / (threshold * 2)));
            if (triggered[i])
                turgor[i] = std::max(0.5f, std::min(1.0f, rel_act[i] / threshold));
        }

        // Recovery: positions far from any trigger recover toward 1.0
        std::vector<int> trigger_pos;
        for (int i = 0; i < n; ++i)
            if (triggered[i]) trigger_pos.push_back(i);
        if (!trigger_pos.empty()) {
            for (int i = 0; i < n; ++i) {
                int min_dist = n;
                for (int tp : trigger_pos)
                    min_dist = std::min(min_dist, std::abs(i - tp));
                float recovery = std::min(1.0f, static_cast<float>(min_dist) / recovery_window);
                turgor[i] = std::max(turgor[i], recovery);
            }
        } else {
            for (int i = 0; i < n; ++i) turgor[i] = 1.0f;
        }
        return turgor;
    }

    // Find the index of the entry with lowest turgor (never the last entry).
    int find_evict_idx(const std::vector<float>& turgor) const {
        if (turgor.size() <= 1) return -1;
        int best = 0;
        float best_val = turgor[0];
        for (int i = 1; i < static_cast<int>(turgor.size()) - 1; ++i) {
            if (turgor[i] < best_val) {
                best_val = turgor[i];
                best = i;
            }
        }
        return best;
    }
};

// ── Three-Distance Spectral Router ──────────────────────────────
// Ported from quantonium_os_src/engine/three_distance_router.py
//
// Steinhaus-Sos Three-Distance Theorem (1957): N points of the form
// {frac((k+1)*phi)} on the unit circle have gaps taking exactly 2 or 3
// distinct values.  We exploit this structure for zone-aware KV cache
// retention — instead of flat eviction by lowest turgor, we partition
// cache entries into spectral zones defined by the phi-grid gap hierarchy,
// then allocate retention budget proportionally to per-zone energy.

static constexpr double PHI = 1.6180339887498948482;

struct ThreeDistanceZoneInfo {
    std::vector<std::vector<int>> zones;  // zone -> list of original indices
    std::vector<double> density;          // per-entry density score
    int n_zones = 0;
};

struct ThreeDistanceRouter {
    // Analyze the three-distance gap structure for N entries.
    // Returns zone partitioning and per-entry density scores.
    static ThreeDistanceZoneInfo analyze_gap_structure(int n) {
        ThreeDistanceZoneInfo info;
        if (n <= 0) return info;

        // Compute phi-grid frequencies: frac((k+1)*phi) for k=0..n-1
        std::vector<double> freqs(n);
        for (int k = 0; k < n; ++k)
            freqs[k] = std::fmod(static_cast<double>(k + 1) * PHI, 1.0);

        // Sort indices by frequency
        std::vector<int> sort_order(n);
        std::iota(sort_order.begin(), sort_order.end(), 0);
        std::sort(sort_order.begin(), sort_order.end(),
                  [&](int a, int b) { return freqs[a] < freqs[b]; });

        std::vector<double> sorted_freqs(n);
        for (int i = 0; i < n; ++i)
            sorted_freqs[i] = freqs[sort_order[i]];

        // Compute gaps (including wraparound)
        std::vector<double> all_gaps(n);
        for (int i = 0; i < n - 1; ++i)
            all_gaps[i] = sorted_freqs[i + 1] - sorted_freqs[i];
        all_gaps[n - 1] = 1.0 + sorted_freqs[0] - sorted_freqs[n - 1];  // wrap

        // Find distinct gap sizes (round to 10 decimals, cluster)
        std::vector<double> unique_gaps;
        {
            std::vector<double> rounded(n);
            for (int i = 0; i < n; ++i)
                rounded[i] = std::round(all_gaps[i] * 1e10) / 1e10;
            std::sort(rounded.begin(), rounded.end());
            unique_gaps.push_back(rounded[0]);
            for (int i = 1; i < n; ++i)
                if (rounded[i] > unique_gaps.back() + 1e-12)
                    unique_gaps.push_back(rounded[i]);
        }

        // Per-entry local density (smaller gap = denser)
        std::vector<double> left_gaps(n), right_gaps(n);
        left_gaps[0] = all_gaps[n - 1];  // wraparound
        for (int i = 1; i < n; ++i) left_gaps[i] = all_gaps[i - 1];
        right_gaps[n - 1] = all_gaps[n - 1];
        for (int i = 0; i < n - 1; ++i) right_gaps[i] = all_gaps[i];

        std::vector<double> density_sorted(n);
        for (int i = 0; i < n; ++i) {
            double local_gap = (left_gaps[i] + right_gaps[i]) / 2.0;
            density_sorted[i] = 1.0 / std::max(local_gap, 1e-15);
        }
        // Map density back to original order
        info.density.resize(n);
        for (int i = 0; i < n; ++i)
            info.density[sort_order[i]] = density_sorted[i];

        // Zone boundaries: split at the largest gap size
        double threshold;
        if (unique_gaps.size() > 1)
            threshold = (unique_gaps.back() + unique_gaps[unique_gaps.size() - 2]) / 2.0;
        else
            threshold = unique_gaps[0] * 1.5;

        std::vector<int> zone_boundaries;
        for (int i = 0; i < n; ++i)
            if (all_gaps[i] > threshold) zone_boundaries.push_back(i);

        // Build raw zones (contiguous runs between large gaps)
        std::vector<std::vector<int>> raw_zones;
        int prev = 0;
        for (int zb : zone_boundaries) {
            if (zb + 1 > prev) {
                std::vector<int> zone(sort_order.begin() + prev,
                                      sort_order.begin() + zb + 1);
                raw_zones.push_back(std::move(zone));
            }
            prev = zb + 1;
        }
        if (prev < n) {
            std::vector<int> zone(sort_order.begin() + prev,
                                  sort_order.end());
            raw_zones.push_back(std::move(zone));
        }

        // Merge small adjacent zones: target O(log_phi(N)) coarse zones
        int log_phi_n = std::max(3, static_cast<int>(
            std::log(std::max(n, 2)) / std::log(PHI)));
        int min_zone_size = std::max(2, n / (2 * log_phi_n));

        std::vector<int> accum;
        for (auto& rz : raw_zones) {
            accum.insert(accum.end(), rz.begin(), rz.end());
            if (static_cast<int>(accum.size()) >= min_zone_size) {
                info.zones.push_back(std::move(accum));
                accum.clear();
            }
        }
        if (!accum.empty()) {
            if (!info.zones.empty())
                info.zones.back().insert(info.zones.back().end(),
                                         accum.begin(), accum.end());
            else
                info.zones.push_back(std::move(accum));
        }
        info.n_zones = static_cast<int>(info.zones.size());
        return info;
    }

    // Zone-aware eviction: given key_norms, turgor, and a target budget,
    // returns the indices to EVICT (sorted descending for safe removal).
    //
    // Instead of flat "evict globally lowest turgor", this:
    // 1. Partitions entries into three-distance spectral zones
    // 2. Computes per-zone energy (sum of key_norms^2 * density)
    // 3. Allocates retention budget proportionally to zone energy
    // 4. Within each zone, keeps entries with highest turgor
    // 5. Returns all entries that don't make the cut
    static std::vector<int> zone_aware_evict(
        const std::vector<float>& key_norms,
        const std::vector<float>& turgor,
        int cache_budget)
    {
        int n = static_cast<int>(key_norms.size());
        if (n <= cache_budget) return {};  // nothing to evict

        auto gap_info = analyze_gap_structure(n);
        if (gap_info.n_zones <= 1) {
            // Single zone — fall back to flat eviction by lowest turgor
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            // Never evict the last entry (most recent token)
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return turgor[a] < turgor[b]; });
            int n_evict = n - cache_budget;
            std::vector<int> evict_list;
            for (int i = 0; i < static_cast<int>(order.size()) && 
                 static_cast<int>(evict_list.size()) < n_evict; ++i) {
                if (order[i] != n - 1)  // protect last entry
                    evict_list.push_back(order[i]);
            }
            std::sort(evict_list.rbegin(), evict_list.rend());
            return evict_list;
        }

        // Per-zone energy weighted by density
        int n_zones = gap_info.n_zones;
        std::vector<double> zone_scores(n_zones, 0.0);
        for (int z = 0; z < n_zones; ++z) {
            for (int idx : gap_info.zones[z]) {
                double norm2 = static_cast<double>(key_norms[idx]);
                norm2 *= norm2;
                zone_scores[z] += norm2 * gap_info.density[idx];
            }
        }
        double total_score = 0.0;
        for (double s : zone_scores) total_score += s;

        if (total_score < 1e-30) {
            // Degenerate — flat eviction
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return turgor[a] < turgor[b]; });
            int n_evict = n - cache_budget;
            std::vector<int> evict_list;
            for (int i = 0; i < static_cast<int>(order.size()) && 
                 static_cast<int>(evict_list.size()) < n_evict; ++i) {
                if (order[i] != n - 1)
                    evict_list.push_back(order[i]);
            }
            std::sort(evict_list.rbegin(), evict_list.rend());
            return evict_list;
        }

        // Proportional allocation with minimum 1 per zone
        std::vector<int> zone_k(n_zones);
        for (int z = 0; z < n_zones; ++z) {
            double raw = zone_scores[z] / total_score * cache_budget;
            zone_k[z] = std::max(1, static_cast<int>(std::round(raw)));
            // Clamp to zone size
            zone_k[z] = std::min(zone_k[z],
                                 static_cast<int>(gap_info.zones[z].size()));
        }

        // Balance to hit exact budget
        // Trim from lowest energy-per-coefficient zones
        while (true) {
            int s = 0;
            for (int z = 0; z < n_zones; ++z) s += zone_k[z];
            if (s <= cache_budget) break;
            int trim_z = 0;
            double best_ratio = std::numeric_limits<double>::max();
            for (int z = 0; z < n_zones; ++z) {
                if (zone_k[z] > 1) {
                    double r = zone_scores[z] / zone_k[z];
                    if (r < best_ratio) { best_ratio = r; trim_z = z; }
                }
            }
            if (zone_k[trim_z] > 1) zone_k[trim_z]--;
            else break;
        }
        // Boost highest energy zones if under budget
        while (true) {
            int s = 0;
            for (int z = 0; z < n_zones; ++z) s += zone_k[z];
            if (s >= cache_budget) break;
            int boost_z = 0;
            double best_ratio = -1.0;
            for (int z = 0; z < n_zones; ++z) {
                if (zone_k[z] < static_cast<int>(gap_info.zones[z].size())) {
                    double r = zone_scores[z] / std::max(zone_k[z], 1);
                    if (r > best_ratio) { best_ratio = r; boost_z = z; }
                }
            }
            if (zone_k[boost_z] < static_cast<int>(gap_info.zones[boost_z].size()))
                zone_k[boost_z]++;
            else break;
        }

        // Within each zone, keep top zone_k[z] entries by turgor (highest turgor = keep)
        // Protect the most recent entry (n-1) from eviction
        std::vector<bool> keep(n, false);
        keep[n - 1] = true;  // always keep most recent

        for (int z = 0; z < n_zones; ++z) {
            auto& zidx = gap_info.zones[z];
            // Sort zone entries by turgor descending
            std::vector<int> sorted_zone = zidx;
            std::sort(sorted_zone.begin(), sorted_zone.end(),
                      [&](int a, int b) { return turgor[a] > turgor[b]; });
            int kept = 0;
            for (int idx : sorted_zone) {
                if (kept < zone_k[z]) {
                    keep[idx] = true;
                    kept++;
                }
            }
        }

        // Collect eviction list (sorted descending for safe in-place removal)
        std::vector<int> evict_list;
        for (int i = 0; i < n; ++i)
            if (!keep[i]) evict_list.push_back(i);
        std::sort(evict_list.rbegin(), evict_list.rend());
        return evict_list;
    }
};

// ── Layer ────────────────────────────────────────────────────────

struct Layer {
    std::vector<float> attn_norm;
    std::vector<float> ffn_norm;

    Weight2D q_proj, k_proj, v_proj, o_proj;
    std::vector<float> q_bias, k_bias, v_bias;

    Weight2D ffn_gate, ffn_up, ffn_down;

    // RFTMW spectral FFN weights (optional — enabled via --spectral)
    std::unique_ptr<rftmw::spectral::SpectralWeight> s_gate, s_up, s_down;

    std::vector<float> k_cache, v_cache;

    // Layer-streaming: if true, weight data is loaded on demand
    bool weights_resident = true;

    // ── Pre-allocated scratch buffers (eliminates heap allocs in hot path) ──
    std::vector<float> _s_normed;   // (hidden,)
    std::vector<float> _s_q;        // (nh * hd,)
    std::vector<float> _s_k;        // (nkv * hd,)
    std::vector<float> _s_v;        // (nkv * hd,)
    std::vector<float> _s_attn_out; // (nh * hd,)
    std::vector<float> _s_o_out;    // (hidden,)
    std::vector<float> _s_h;        // (hidden,)
    std::vector<float> _s_gate;     // (ffn_dim,)
    std::vector<float> _s_up;       // (ffn_dim,)
    std::vector<float> _s_ffn_out;  // (hidden,)
    // Per-head scores buffer for threaded attention
    std::vector<float> _s_scores;   // (nh * max_seq,)

    void bind(const GGUFModel& m, int i, const Qwen25Config& cfg, int max_seq) {
        std::string p = "blk." + std::to_string(i) + ".";

        attn_norm = tensor_to_f32(m.tensor(p + "attn_norm.weight"));
        ffn_norm  = tensor_to_f32(m.tensor(p + "ffn_norm.weight"));

        q_proj.bind(m, p + "attn_q.weight");
        k_proj.bind(m, p + "attn_k.weight");
        v_proj.bind(m, p + "attn_v.weight");
        o_proj.bind(m, p + "attn_output.weight");

        q_bias = tensor_to_f32(m.tensor(p + "attn_q.bias"));
        k_bias = tensor_to_f32(m.tensor(p + "attn_k.bias"));
        v_bias = tensor_to_f32(m.tensor(p + "attn_v.bias"));

        ffn_gate.bind(m, p + "ffn_gate.weight");
        ffn_up.bind(m, p + "ffn_up.weight");
        ffn_down.bind(m, p + "ffn_down.weight");

        k_cache.assign(static_cast<size_t>(max_seq) * cfg.kv_heads * cfg.head_dim, 0.0f);
        v_cache.assign(static_cast<size_t>(max_seq) * cfg.kv_heads * cfg.head_dim, 0.0f);

        // Pre-allocate scratch buffers
        int nh = cfg.heads, nkv = cfg.kv_heads, hd = cfg.head_dim;
        _s_normed.resize(static_cast<size_t>(cfg.hidden));
        _s_q.resize(static_cast<size_t>(nh * hd));
        _s_k.resize(static_cast<size_t>(nkv * hd));
        _s_v.resize(static_cast<size_t>(nkv * hd));
        _s_attn_out.resize(static_cast<size_t>(nh * hd));
        _s_o_out.resize(static_cast<size_t>(cfg.hidden));
        _s_h.resize(static_cast<size_t>(cfg.hidden));
        _s_gate.resize(static_cast<size_t>(cfg.ffn_dim));
        _s_up.resize(static_cast<size_t>(cfg.ffn_dim));
        _s_ffn_out.resize(static_cast<size_t>(cfg.hidden));
        _s_scores.resize(static_cast<size_t>(nh) * max_seq);
    }

    static void apply_rope(float* x, int n_heads, int hd,
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

    // Keep old overload for any external callers
    static void apply_rope(std::vector<float>& x, int n_heads, int hd,
                           int pos, double theta) {
        apply_rope(x.data(), n_heads, hd, pos, theta);
    }

    // Evict a single KV cache entry at evict_idx, shifting remaining entries down.
    void evict_kv_entry(int evict_idx, int cache_len, const Qwen25Config& cfg) {
        const int stride = cfg.kv_heads * cfg.head_dim;
        for (int i = evict_idx; i < cache_len - 1; ++i) {
            std::memcpy(k_cache.data() + static_cast<size_t>(i) * stride,
                        k_cache.data() + static_cast<size_t>(i + 1) * stride,
                        static_cast<size_t>(stride) * sizeof(float));
            std::memcpy(v_cache.data() + static_cast<size_t>(i) * stride,
                        v_cache.data() + static_cast<size_t>(i + 1) * stride,
                        static_cast<size_t>(stride) * sizeof(float));
        }
    }

    // Compute L2 norm of each cached K vector (mean over KV heads) for Mimosa scoring.
    std::vector<float> compute_key_norms(int cache_len, const Qwen25Config& cfg) const {
        const int nkv = cfg.kv_heads;
        const int hd  = cfg.head_dim;
        std::vector<float> norms(cache_len, 0.0f);
        for (int t = 0; t < cache_len; ++t) {
            float sum_sq = 0.0f;
            const float* kp = k_cache.data() + static_cast<size_t>(t) * nkv * hd;
            for (int j = 0; j < nkv * hd; ++j) sum_sq += kp[j] * kp[j];
            norms[t] = std::sqrt(sum_sq);
        }
        return norms;
    }

    // Zero-alloc forward pass using scratch buffers + threaded attention
    void forward_into(const float* x, float* h_out,
                      const Qwen25Config& cfg,
                      int pos, int cache_len) {
        const int nh  = cfg.heads;
        const int nkv = cfg.kv_heads;
        const int hd  = cfg.head_dim;
        const int hidden = cfg.hidden;

        // ── attention ──
        rms_norm_into(x, attn_norm.data(), _s_normed.data(), hidden, cfg.rms_eps);
        q_proj.matmul_into(_s_normed.data(), _s_q.data());
        k_proj.matmul_into(_s_normed.data(), _s_k.data());
        v_proj.matmul_into(_s_normed.data(), _s_v.data());

        // add bias
        for (int i = 0; i < nh * hd; ++i)  _s_q[i] += q_bias[i];
        for (int i = 0; i < nkv * hd; ++i) _s_k[i] += k_bias[i];
        for (int i = 0; i < nkv * hd; ++i) _s_v[i] += v_bias[i];

        // RoPE
        apply_rope(_s_q.data(), nh,  hd, pos, cfg.rope_theta);
        apply_rope(_s_k.data(), nkv, hd, pos, cfg.rope_theta);

        // write KV cache
        std::memcpy(k_cache.data() + static_cast<size_t>(cache_len) * nkv * hd,
                    _s_k.data(), static_cast<size_t>(nkv * hd) * sizeof(float));
        std::memcpy(v_cache.data() + static_cast<size_t>(cache_len) * nkv * hd,
                    _s_v.data(), static_cast<size_t>(nkv * hd) * sizeof(float));

        // GQA attention — threaded across query heads
        int seq_len = cache_len + 1;
        int hpg = nh / nkv;
        float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        std::memset(_s_attn_out.data(), 0, static_cast<size_t>(nh * hd) * sizeof(float));

        // Thread across all nh query heads (each head independent)
        auto attn_fn = [&](int h0, int h1) {
            for (int h = h0; h < h1; ++h) {
                int kv = h / hpg;
                const float* qh = _s_q.data() + static_cast<size_t>(h) * hd;
                float* sc = _s_scores.data() + static_cast<size_t>(h) * seq_len;

                // dot products with AVX2
                for (int t = 0; t < seq_len; ++t) {
                    const float* kh = k_cache.data() + (static_cast<size_t>(t) * nkv + kv) * hd;
#ifdef __AVX2__
                    __m256 acc = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 7 < hd; d += 8) {
                        __m256 qq = _mm256_loadu_ps(qh + d);
                        __m256 kk = _mm256_loadu_ps(kh + d);
#ifdef __FMA__
                        acc = _mm256_fmadd_ps(qq, kk, acc);
#else
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(qq, kk));
#endif
                    }
                    __m128 hi4 = _mm256_extractf128_ps(acc, 1);
                    __m128 lo4 = _mm256_castps256_ps128(acc);
                    __m128 sum4 = _mm_add_ps(lo4, hi4);
                    __m128 shuf = _mm_movehdup_ps(sum4);
                    sum4 = _mm_add_ps(sum4, shuf);
                    shuf = _mm_movehl_ps(sum4, sum4);
                    sum4 = _mm_add_ss(sum4, shuf);
                    float s = _mm_cvtss_f32(sum4);
                    for (; d < hd; ++d) s += qh[d] * kh[d];
                    sc[t] = s * scale;
#else
                    float s = 0.0f;
                    for (int d = 0; d < hd; ++d) s += qh[d] * kh[d];
                    sc[t] = s * scale;
#endif
                }

                // softmax
                float m = sc[0];
                for (int t = 1; t < seq_len; ++t) if (sc[t] > m) m = sc[t];
                float sum = 0.0f;
                for (int t = 0; t < seq_len; ++t) {
                    sc[t] = std::exp(sc[t] - m);
                    sum += sc[t];
                }
                float inv_sum = 1.0f / sum;

                // weighted sum of values
                float* oh = _s_attn_out.data() + static_cast<size_t>(h) * hd;
                for (int t = 0; t < seq_len; ++t) {
                    float w = sc[t] * inv_sum;
                    const float* vh = v_cache.data() + (static_cast<size_t>(t) * nkv + kv) * hd;
#ifdef __AVX2__
                    __m256 vw = _mm256_set1_ps(w);
                    int d = 0;
                    for (; d + 7 < hd; d += 8) {
                        __m256 ov = _mm256_loadu_ps(oh + d);
                        __m256 vv = _mm256_loadu_ps(vh + d);
#ifdef __FMA__
                        ov = _mm256_fmadd_ps(vw, vv, ov);
#else
                        ov = _mm256_add_ps(ov, _mm256_mul_ps(vw, vv));
#endif
                        _mm256_storeu_ps(oh + d, ov);
                    }
                    for (; d < hd; ++d) oh[d] += w * vh[d];
#else
                    for (int d = 0; d < hd; ++d) oh[d] += w * vh[d];
#endif
                }
            }
        };

        // Use thread pool if enough heads (R1-7B has 28 heads)
        size_t attn_work = static_cast<size_t>(nh) * seq_len * hd * sizeof(float);
        rftmw::inference::parallel_for(nh, attn_work, attn_fn);

        o_proj.matmul_into(_s_attn_out.data(), _s_o_out.data());

        // h = x + attn_out
        std::memcpy(_s_h.data(), x, static_cast<size_t>(hidden) * sizeof(float));
        add_inplace(_s_h.data(), _s_o_out.data(), hidden);

        // ── SwiGLU FFN (spectral or dense path) ──
        rms_norm_into(_s_h.data(), ffn_norm.data(), _s_normed.data(), hidden, cfg.rms_eps);

        if (s_gate && s_gate->is_initialized()) {
            s_gate->matmul_vec_parallel(_s_normed.data(), _s_gate.data());
        } else {
            ffn_gate.matmul_into(_s_normed.data(), _s_gate.data());
        }
        if (s_up && s_up->is_initialized()) {
            s_up->matmul_vec_parallel(_s_normed.data(), _s_up.data());
        } else {
            ffn_up.matmul_into(_s_normed.data(), _s_up.data());
        }

        int ffn = cfg.ffn_dim;
        for (int i = 0; i < ffn; ++i) {
            float g = _s_gate[i];
            g = std::max(-88.0f, std::min(88.0f, g));
            _s_gate[i] = (g / (1.0f + std::exp(-g))) * _s_up[i];  // SiLU(gate) * up
        }

        if (s_down && s_down->is_initialized()) {
            s_down->matmul_vec_parallel(_s_gate.data(), _s_ffn_out.data());
        } else {
            ffn_down.matmul_into(_s_gate.data(), _s_ffn_out.data());
        }

        // h_out = h + ffn_out
        std::memcpy(h_out, _s_h.data(), static_cast<size_t>(hidden) * sizeof(float));
        add_inplace(h_out, _s_ffn_out.data(), hidden);
    }

    // Legacy allocating forward (kept for backward compat)
    std::vector<float> forward(const std::vector<float>& x,
                               const Qwen25Config& cfg,
                               int pos, int cache_len) {
        std::vector<float> h(x.size());
        forward_into(x.data(), h.data(), cfg, pos, cache_len);
        return h;
    }
};

// ── Engine ───────────────────────────────────────────────────────

struct NativeEngine {
    Qwen25Config cfg;
    GGUFModel gguf;
    NativeTokenizer tokenizer;
    Weight2D embed;
    Weight2D lm_head;
    std::vector<float> output_norm;
    std::vector<Layer> layers;
    bool spectral_mode = false;
    float spectral_keep = 0.45f;

    // ── KV-Cache RFT Compression ──
    rftmw::kv_compress::KVCompressor kv_compressor;
    bool kv_compress_enabled = false;
    float kv_compress_keep = 0.60f;

    // ── Layer-streaming mode ──
    bool layer_streaming = false;
    int  streaming_window = 1;   // how many layers to keep resident

    void load(const fs::path& gguf_path, int max_seq) {
        std::cerr << "[qwen25] loading GGUF " << gguf_path.string() << "\n";
        gguf.load(gguf_path);

        // Auto-detect config from GGUF metadata
        int64_t m_layers = gguf.get_int("block_count");
        int64_t m_hidden = gguf.get_int("embedding_length");
        int64_t m_heads  = gguf.get_int("attention.head_count");
        int64_t m_kv     = gguf.get_int("attention.head_count_kv");
        int64_t m_ffn    = gguf.get_int("feed_forward_length");
        double  m_rope   = gguf.get_float("rope.freq_base", -1.0);
        double  m_eps    = gguf.get_float("attention.layer_norm_rms_epsilon", -1.0);
        if (m_layers > 0) cfg.layers    = static_cast<int>(m_layers);
        if (m_hidden > 0) cfg.hidden    = static_cast<int>(m_hidden);
        if (m_heads  > 0) cfg.heads     = static_cast<int>(m_heads);
        if (m_kv     > 0) cfg.kv_heads  = static_cast<int>(m_kv);
        if (m_ffn    > 0) cfg.ffn_dim   = static_cast<int>(m_ffn);
        if (m_hidden > 0 && m_heads > 0) cfg.head_dim = cfg.hidden / cfg.heads;
        if (m_rope   > 0) cfg.rope_theta = m_rope;
        if (m_eps    > 0) cfg.rms_eps    = static_cast<float>(m_eps);
        // Detect vocab from tokenizer or metadata
        if (!gguf.tokenizer_tokens.empty())
            cfg.vocab = static_cast<int>(gguf.tokenizer_tokens.size());

        tokenizer.init(gguf);
        embed.bind(gguf, "token_embd.weight");
        lm_head.bind(gguf, "output.weight");
        output_norm = tensor_to_f32(gguf.tensor("output_norm.weight"));
        layers.resize(static_cast<size_t>(cfg.layers));
        for (int i = 0; i < cfg.layers; ++i) {
            layers[static_cast<size_t>(i)].bind(gguf, i, cfg, max_seq);
            if ((i + 1) % 6 == 0 || i + 1 == cfg.layers)
                std::cerr << "[qwen25] loaded layer " << (i + 1) << "/" << cfg.layers << "\r";
        }
        std::cerr << "\n[qwen25] all " << cfg.layers << " layers loaded\n";
        std::cerr << "[qwen25] vocab=" << tokenizer.vocab.size()
                  << " dim=" << cfg.hidden << " heads=" << cfg.heads
                  << " kv=" << cfg.kv_heads << " ffn=" << cfg.ffn_dim << "\n";

        // ── KV-Cache RFT Compression init ──
        if (kv_compress_enabled) {
            rftmw::kv_compress::KVCompressConfig kv_cfg;
            kv_cfg.head_dim = cfg.head_dim;
            kv_cfg.num_kv_heads = cfg.kv_heads;
            kv_cfg.base_keep_ratio = kv_compress_keep;
            kv_cfg.total_layers = cfg.layers;
            kv_cfg.enabled = true;
            kv_cfg.min_positions_before_compress = 4;
            kv_compressor.init(kv_cfg);
            std::cerr << "[kv-compress] enabled: base_keep=" << kv_compress_keep
                      << " head_dim=" << cfg.head_dim
                      << " kv_heads=" << cfg.kv_heads << "\n";
            // Report expected memory savings
            float uncompressed_per_pos = 2.0f * cfg.layers * cfg.kv_heads * cfg.head_dim * 4;
            float compressed_ratio = kv_compress_keep * 4.0f / 4.0f;  // sparse: keep * 4 bytes / 4 bytes
            std::cerr << "[kv-compress] per-position: "
                      << (uncompressed_per_pos / 1024) << " KB uncompressed -> ~"
                      << (uncompressed_per_pos * compressed_ratio / 1024) << " KB compressed\n";
        }

        // ── RFTMW Spectral FFN conversion (threaded + disk cache) ──
        if (spectral_mode) {
            using SW = rftmw::spectral::SpectralWeight;
            std::string cache_file = gguf_path.string() + ".spectral_k"
                + std::to_string(static_cast<int>(spectral_keep * 100));
            bool cache_loaded = false;

            // Try loading from cache first
            if (fs::exists(cache_file)) {
                std::cerr << "[spectral] loading cache: " << cache_file << "\n";
                std::FILE* fp = std::fopen(cache_file.c_str(), "rb");
                if (fp) {
                    char magic[4] = {};
                    std::fread(magic, 1, 4, fp);
                    uint32_t ver = 0, nl = 0;
                    std::fread(&ver, 4, 1, fp);
                    std::fread(&nl, 4, 1, fp);
                    if (std::memcmp(magic, "RFTC", 4) == 0
                        && ver == 1 && static_cast<int>(nl) == cfg.layers)
                    {
                        auto t_load = Clock::now();
                        for (int i = 0; i < cfg.layers; ++i) {
                            auto& lay = layers[static_cast<size_t>(i)];
                            lay.s_gate = std::make_unique<SW>();
                            lay.s_up   = std::make_unique<SW>();
                            lay.s_down = std::make_unique<SW>();
                            lay.s_gate->load_from_stream(fp);
                            lay.s_up  ->load_from_stream(fp);
                            lay.s_down->load_from_stream(fp);
                            if ((i + 1) % 4 == 0 || i + 1 == cfg.layers)
                                std::cerr << "[spectral] loaded layer "
                                          << (i + 1) << "/" << cfg.layers << "\r";
                        }
                        double lt = std::chrono::duration<double>(
                            Clock::now() - t_load).count();
                        size_t tot = 0;
                        for (int i = 0; i < cfg.layers; ++i) {
                            auto& l = layers[static_cast<size_t>(i)];
                            tot += l.s_gate->memory_bytes()
                                 + l.s_up->memory_bytes()
                                 + l.s_down->memory_bytes();
                        }
                        std::cerr << "\n[spectral] cache loaded in "
                                  << lt << "s, " << (tot >> 20) << "MB\n";
                        cache_loaded = true;
                    } else {
                        std::cerr << "[spectral] cache mismatch, reconverting\n";
                    }
                    std::fclose(fp);
                }
            }

            if (!cache_loaded) {
                int n_hw = static_cast<int>(std::thread::hardware_concurrency());
                if (n_hw < 1) n_hw = 1;
                if (n_hw > 16) n_hw = 16;
                auto t_spec = Clock::now();
                size_t total_spec_mem = 0;
                std::cerr << "[spectral] converting FFN weights to spectral domain "
                          << "(base_keep=" << spectral_keep
                          << ", threads=" << n_hw << ")\n";
                for (int i = 0; i < cfg.layers; ++i) {
                    int early_end  = cfg.layers / 6;
                    int late_start = cfg.layers - cfg.layers / 6;
                    float keep;
                    if (i < early_end)
                        keep = std::min(1.0f, std::max(0.08f, spectral_keep * 0.50f));
                    else if (i >= late_start)
                        keep = std::min(1.0f, std::max(0.08f, spectral_keep * 0.65f));
                    else
                        keep = std::min(1.0f, std::max(0.08f, spectral_keep * 1.15f));

                    auto& lay = layers[static_cast<size_t>(i)];
                    auto convert_w = [&](Weight2D& w) {
                        auto sw = std::make_unique<SW>();
                        if (w.t && (w.t->type == GGUF_Q4_K || w.t->type == GGUF_Q6_K)) {
                            const char* qt = (w.t->type == GGUF_Q4_K) ? "q4k" : "q6k";
                            sw->from_quantized_threaded(
                                w.t->data, w.out, w.in, w.bytes_per_row,
                                qt, RFTMW_SPECTRAL_DEFAULT_BLOCK, keep, n_hw);
                        } else if (!w.dense.empty()) {
                            sw->from_dense(w.dense.data(), w.out, w.in,
                                RFTMW_SPECTRAL_DEFAULT_BLOCK, keep);
                        }
                        return sw;
                    };
                    lay.s_gate = convert_w(lay.ffn_gate);
                    lay.s_up   = convert_w(lay.ffn_up);
                    lay.s_down = convert_w(lay.ffn_down);

                    size_t lmem = lay.s_gate->memory_bytes()
                                + lay.s_up->memory_bytes()
                                + lay.s_down->memory_bytes();
                    total_spec_mem += lmem;
                    std::cerr << "[spectral] layer " << (i + 1) << "/" << cfg.layers
                              << " keep=" << static_cast<int>(keep * 100) << "%"
                              << " sparsity=" << static_cast<int>(
                                     lay.s_gate->avg_sparsity() * 100) << "%"
                              << " mem=" << (lmem >> 20) << "MB\n";
                }
                double spec_time = std::chrono::duration<double>(
                    Clock::now() - t_spec).count();
                std::cerr << "[spectral] done: " << spec_time << "s"
                          << " total_mem=" << (total_spec_mem >> 20) << "MB\n";

                // Save cache
                std::FILE* fp = std::fopen(cache_file.c_str(), "wb");
                if (fp) {
                    std::fwrite("RFTC", 1, 4, fp);
                    uint32_t ver = 1;
                    uint32_t nl = static_cast<uint32_t>(cfg.layers);
                    std::fwrite(&ver, 4, 1, fp);
                    std::fwrite(&nl, 4, 1, fp);
                    for (int i = 0; i < cfg.layers; ++i) {
                        auto& l = layers[static_cast<size_t>(i)];
                        l.s_gate->save_to_stream(fp);
                        l.s_up  ->save_to_stream(fp);
                        l.s_down->save_to_stream(fp);
                    }
                    std::fclose(fp);
                    std::cerr << "[spectral] cache saved: " << cache_file << "\n";
                }
            }
        }
    }

    // cache_len may differ from pos after Mimosa evictions
    // Pre-allocated buffers for forward_token (zero heap allocs in decode)
    std::vector<float> _ft_h;          // (hidden,)
    std::vector<float> _ft_h2;         // (hidden,) ping-pong buffer
    std::vector<float> _ft_normed;     // (hidden,)
    std::vector<float> _ft_logits;     // (vocab,)
    bool _ft_bufs_ready = false;

    void _ensure_ft_bufs() {
        if (_ft_bufs_ready) return;
        _ft_h.resize(static_cast<size_t>(cfg.hidden));
        _ft_h2.resize(static_cast<size_t>(cfg.hidden));
        _ft_normed.resize(static_cast<size_t>(cfg.hidden));
        _ft_logits.resize(static_cast<size_t>(cfg.vocab));
        _ft_bufs_ready = true;
    }

    std::vector<float> forward_token(int token, int pos, int cache_len,
                                     bool need_logits = true) {
        _ensure_ft_bufs();
        embed.dequant_row(token, _ft_h);
        for (int i = 0; i < cfg.layers; ++i) {
            layers[static_cast<size_t>(i)].forward_into(
                _ft_h.data(), _ft_h2.data(), cfg, pos, cache_len);
            std::swap(_ft_h, _ft_h2);  // O(1) pointer swap
        }
        if (!need_logits) return {};  // prefill: skip lm_head (152K matmul)
        rms_norm_into(_ft_h.data(), output_norm.data(),
                      _ft_normed.data(), cfg.hidden, cfg.rms_eps);
        lm_head.matmul_into(_ft_normed.data(), _ft_logits.data());
        return _ft_logits;  // returns ref to persistent buffer (caller consumes immediately)
    }

    // Overload for backward compat (pos == cache_len)
    std::vector<float> forward_token(int token, int pos) {
        return forward_token(token, pos, pos);
    }

    // Evict one KV entry across all layers
    void evict_kv_across_layers(int evict_idx, int cache_len) {
        for (int i = 0; i < cfg.layers; ++i)
            layers[static_cast<size_t>(i)].evict_kv_entry(evict_idx, cache_len, cfg);
    }
};

// ── Prompt formatting ────────────────────────────────────────────

std::string format_prompt(const std::string& prompt, const std::string& mode) {
    if (mode == "raw" || mode == "none" || mode == "plain") return prompt;
    if (mode == "chat" || mode == "ollama" || mode == "auto") {
        return "<|im_start|>system\n"
               "You are a concise coding assistant. Answer directly.\n"
               "<|im_end|>\n"
               "<|im_start|>user\n" + prompt + "\n"
               "<|im_end|>\n"
               "<|im_start|>assistant\n";
    }
    if (mode == "final_answer" || mode == "answer_only") {
        return "<|im_start|>system\nReturn only the final answer. No reasoning.\n<|im_end|>\n"
               "<|im_start|>user\n" + prompt + "\n<|im_end|>\n"
               "<|im_start|>assistant\n";
    }
    throw std::runtime_error("unknown template mode: " + mode);
}

// ── Utility ──────────────────────────────────────────────────────

int argmax(const std::vector<float>& x) {
    return static_cast<int>(std::max_element(x.begin(), x.end()) - x.begin());
}

int sample_top_p(std::vector<float>& logits, float temperature, float top_p,
                 const std::vector<int>& recent_tokens = {}, float rep_penalty = 1.1f) {
    if (temperature <= 0.0f) {
        // Apply repetition penalty even in greedy mode
        for (int tid : recent_tokens) {
            if (tid >= 0 && tid < static_cast<int>(logits.size())) {
                logits[tid] = (logits[tid] > 0) ? logits[tid] / rep_penalty
                                                 : logits[tid] * rep_penalty;
            }
        }
        return argmax(logits);
    }
    int n = static_cast<int>(logits.size());
    // Apply repetition penalty before temperature scaling
    for (int tid : recent_tokens) {
        if (tid >= 0 && tid < n) {
            logits[tid] = (logits[tid] > 0) ? logits[tid] / rep_penalty
                                             : logits[tid] * rep_penalty;
        }
    }
    // Temperature-scale and softmax
    float max_l = *std::max_element(logits.begin(), logits.end());
    // Use partial sort: only need the top tokens, not full sort of 152K vocab
    // First pass: compute softmax in-place and build index
    std::vector<std::pair<float, int>> top_entries;
    top_entries.reserve(1024);  // nucleus is typically small
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float p = std::exp((logits[i] - max_l) / temperature);
        logits[i] = p;  // reuse logits as prob buffer
        sum += p;
    }
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) logits[i] *= inv_sum;
    // Find top-p nucleus using partial sort via nth_element + threshold scan
    // Quick approach: collect all entries above a threshold, sort just those
    float threshold = inv_sum * std::exp(-10.0f / temperature);  // tokens with prob > exp(-10)/sum
    for (int i = 0; i < n; ++i) {
        if (logits[i] > threshold) {
            top_entries.push_back({logits[i], i});
        }
    }
    std::sort(top_entries.begin(), top_entries.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    // Nucleus: accumulate until >= top_p
    float cumsum = 0.0f;
    int cutoff = static_cast<int>(top_entries.size());
    for (int i = 0; i < cutoff; ++i) {
        cumsum += top_entries[i].first;
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }
    if (cutoff == 0) return argmax(logits);  // safety fallback
    // Sample from the nucleus
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, cumsum);
    float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        acc += top_entries[i].first;
        if (acc >= r) return top_entries[i].second;
    }
    return top_entries[0].second;
}

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string hex_encode(const std::string& s) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        out.push_back(kHex[(c >> 4) & 0x0f]);
        out.push_back(kHex[c & 0x0f]);
    }
    return out;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    throw std::runtime_error("invalid hex");
}

std::string hex_decode(const std::string& h) {
    if ((h.size() & 1u) != 0) throw std::runtime_error("odd-length hex");
    std::string out;
    out.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2)
        out.push_back(static_cast<char>((hex_value(h[i]) << 4) | hex_value(h[i + 1])));
    return out;
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= line.size()) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos) { parts.push_back(line.substr(start)); break; }
        parts.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return parts;
}

// ── Generation ───────────────────────────────────────────────────

struct GenerationResult {
    std::vector<int> prompt_ids;
    std::vector<int> out_ids;
    std::string out_text;
    double prefill_s = 0, decode_s = 0;
};

GenerationResult generate_once(
    NativeEngine& engine,
    const std::string& prompt,
    const std::string& template_mode,
    int max_tokens, int max_seq,
    int mimosa_cache_size,  // 0 = disabled
    float temperature = 0.0f,
    float top_p = 0.9f,
    std::function<void(const std::string&, int)> on_stage = {},
    std::function<void(int, const std::string&, int, double)> on_token = {})
{
    GenerationResult result;
    std::string formatted = format_prompt(prompt, template_mode);
    result.prompt_ids = engine.tokenizer.encode(formatted);
    if (on_stage) on_stage("prefill", static_cast<int>(result.prompt_ids.size()));

    MimosaKVGate mimosa_gate;
    int cache_len = 0;
    int eviction_count = 0;

    auto t0 = Clock::now();
    // Prefill: run all prompt tokens (skip lm_head until last token)
    std::vector<float> logits;
    size_t n_prompt = result.prompt_ids.size();
    for (size_t i = 0; i < n_prompt; ++i) {
        bool last = (i == n_prompt - 1);
        logits = engine.forward_token(result.prompt_ids[i],
                                      static_cast<int>(i), cache_len,
                                      last);  // only compute logits on last token
        cache_len++;

        // Three-distance + Mimosa zone-aware eviction during prefill
        if (mimosa_cache_size > 0 && cache_len > mimosa_cache_size) {
            auto knorms = engine.layers[0].compute_key_norms(cache_len, engine.cfg);
            auto turgor = mimosa_gate.compute_turgor(knorms);
            auto evict_list = ThreeDistanceRouter::zone_aware_evict(
                knorms, turgor, mimosa_cache_size);
            for (int eidx : evict_list) {  // sorted descending — safe to remove in order
                engine.evict_kv_across_layers(eidx, cache_len);
                cache_len--;
                eviction_count++;
            }
        }
    }
    result.prefill_s = seconds_since(t0);

    if (on_stage) on_stage("decode", max_tokens);

    int next = sample_top_p(logits, temperature, top_p);
    auto t1 = Clock::now();
    int rope_pos = static_cast<int>(result.prompt_ids.size());  // next RoPE position
    std::string tail;  // rolling window for multi-token stop sequences
    // Rolling window of recent tokens for repetition penalty
    std::vector<int> recent_tokens;
    const int rep_window = 64;  // look back 64 tokens
    for (int n = 0; n < max_tokens; ++n) {
        result.out_ids.push_back(next);
        recent_tokens.push_back(next);
        if (static_cast<int>(recent_tokens.size()) > rep_window)
            recent_tokens.erase(recent_tokens.begin());
        std::string tok = engine.tokenizer.decode_token(next);
        result.out_text += tok;
        tail += tok;
        if (tail.size() > 32) tail = tail.substr(tail.size() - 32);

        // Stop on single-token EOS or multi-token stop sequences
        if (engine.tokenizer.is_stop_token(next)) break;
        if (tail.find("<|im_end|>") != std::string::npos) break;
        if (tail.find("<|endoftext|>") != std::string::npos) break;

        if (cache_len >= max_seq) break;

        double elapsed = seconds_since(t1);
        double tps = (n + 1 > 0 && elapsed > 0) ? (n + 1) / elapsed : 0;
        if (on_token) on_token(next, tok, n + 1, tps);

        logits = engine.forward_token(next, rope_pos, cache_len);
        cache_len++;
        rope_pos++;
        next = sample_top_p(logits, temperature, top_p, recent_tokens);

        // Three-distance + Mimosa zone-aware eviction during decode
        if (mimosa_cache_size > 0 && cache_len > mimosa_cache_size) {
            auto knorms = engine.layers[0].compute_key_norms(cache_len, engine.cfg);
            auto turgor = mimosa_gate.compute_turgor(knorms);
            auto evict_list = ThreeDistanceRouter::zone_aware_evict(
                knorms, turgor, mimosa_cache_size);
            for (int eidx : evict_list) {
                engine.evict_kv_across_layers(eidx, cache_len);
                cache_len--;
                eviction_count++;
            }
        }
    }
    result.decode_s = seconds_since(t1);

    if (eviction_count > 0)
        std::cerr << "[mimosa+3dist] evictions=" << eviction_count
                  << " cache_budget=" << mimosa_cache_size
                  << " final_cache=" << cache_len << "\n";

    // Strip stop sequences (and any partial trailing fragments) from output
    auto strip = [&](const std::string& seq) {
        auto pos = result.out_text.find(seq);
        if (pos != std::string::npos)
            result.out_text.erase(pos);
    };
    strip("<|im_end|>");
    strip("<|endoftext|>");
    // Also strip partial trailing stop sequences (sub-token leftovers)
    for (auto& seq : {"<|im_end|", "<|im_end", "<|im_en", "<|im_e", "<|im_",
                       "<|im", "<|i", "<|endoftext|", "<|endoftext", "<|endoftex",
                       "<|endofte", "<|endoft", "<|endof", "<|endo", "<|end", "<|en"}) {
        if (result.out_text.size() >= std::strlen(seq) &&
            result.out_text.compare(result.out_text.size() - std::strlen(seq),
                                     std::strlen(seq), seq) == 0) {
            result.out_text.erase(result.out_text.size() - std::strlen(seq));
            break;
        }
    }
    return result;
}

// ── Server mode (stdin/stdout protocol) ──────────────────────────

void run_server(NativeEngine& engine, int max_seq, int mimosa_cache_size) {
    std::cout << "READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT" || line == "EXIT") {
            std::cout << "BYE\n" << std::flush;
            break;
        }
        try {
            auto parts = split_tabs(line);
            if (parts.empty() || parts[0] != "GENERATE" || parts.size() < 4)
                throw std::runtime_error("expected GENERATE\\tmax_tokens\\ttemplate_mode\\thex_prompt[\\ttemp\\ttop_p]");
            int max_tokens = std::stoi(parts[1]);
            std::string tmpl = parts[2];
            std::string prompt = hex_decode(parts[3]);
            float temperature = (parts.size() > 4) ? std::stof(parts[4]) : 0.0f;
            float top_p = (parts.size() > 5) ? std::stof(parts[5]) : 0.9f;

            auto r = generate_once(engine, prompt, tmpl, max_tokens, max_seq,
                mimosa_cache_size, temperature, top_p,
                [](const std::string& stage, int value) {
                    std::cout << "STAGE\t" << stage << "\t" << value << "\n" << std::flush;
                },
                [](int tid, const std::string& text, int n, double tps) {
                    std::cout << "TOKEN\t" << tid << "\t" << hex_encode(text)
                              << "\t" << n << "\t" << tps << "\n" << std::flush;
                });
            std::cout << "DONE\t" << hex_encode(r.out_text)
                      << "\t" << r.prompt_ids.size()
                      << "\t" << r.prefill_s
                      << "\t" << r.out_ids.size()
                      << "\t" << r.decode_s << "\n" << std::flush;
        } catch (const std::exception& e) {
            std::cout << "ERROR\t" << hex_encode(e.what()) << "\n" << std::flush;
        }
    }
}

void run_perplexity(NativeEngine& engine, const std::string& filepath,
                    int max_seq, int mimosa_cache_size) {
    std::ifstream f(filepath);
    if (!f) throw std::runtime_error("cannot open perplexity file: " + filepath);
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    auto tokens = engine.tokenizer.encode(text);
    int n = static_cast<int>(tokens.size());
    if (n < 2) throw std::runtime_error("corpus too short for perplexity (need >=2 tokens)");
    std::cerr << "[ppl] corpus tokens: " << n << "\n";

    MimosaKVGate mimosa_gate;
    int cache_len = 0;
    double nll_sum = 0.0;
    int nll_count = 0;
    auto t0 = Clock::now();

    for (int i = 0; i < n - 1; ++i) {
        auto logits = engine.forward_token(tokens[i], i, cache_len);
        cache_len++;

        // Mimosa eviction
        if (mimosa_cache_size > 0 && cache_len > mimosa_cache_size) {
            auto knorms = engine.layers[0].compute_key_norms(cache_len, engine.cfg);
            auto turgor = mimosa_gate.compute_turgor(knorms);
            auto evict_list = ThreeDistanceRouter::zone_aware_evict(
                knorms, turgor, mimosa_cache_size);
            for (int eidx : evict_list) {
                engine.evict_kv_across_layers(eidx, cache_len);
                cache_len--;
            }
        }

        // log_softmax for target token
        int target = tokens[i + 1];
        float max_l = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (auto v : logits) sum_exp += std::exp(static_cast<double>(v) - max_l);
        double log_sum = std::log(sum_exp) + max_l;
        double log_prob = static_cast<double>(logits[target]) - log_sum;
        nll_sum -= log_prob;
        nll_count++;

        if ((i + 1) % 50 == 0 || i == n - 2) {
            double ppl = std::exp(nll_sum / nll_count);
            double elapsed = seconds_since(t0);
            double tps = (i + 1) / elapsed;
            std::cerr << "[ppl] " << (i + 1) << "/" << (n - 1)
                      << "  ppl=" << std::fixed << std::setprecision(2) << ppl
                      << "  tok/s=" << std::setprecision(1) << tps << "\n";
        }
    }

    double final_ppl = std::exp(nll_sum / nll_count);
    double elapsed = seconds_since(t0);
    std::cout << std::fixed << std::setprecision(4)
              << "{\"perplexity\": " << final_ppl
              << ", \"tokens\": " << nll_count
              << ", \"elapsed_s\": " << elapsed
              << ", \"tok_per_s\": " << std::setprecision(1) << (nll_count / elapsed)
              << "}\n";
}

} // namespace

// ── main ─────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        fs::path gguf_path;
        std::string prompt_arg;
        std::string template_mode = "chat";
        int max_tokens = 128;
        int max_seq = 2048;
        int mimosa_cache_size = 0;  // 0 = disabled
        bool serve = false;
        bool spectral = false;
        float spectral_keep = 0.45f;
        bool kv_compress = false;
        float kv_keep = 0.60f;
        std::string perplexity_file;
        std::string layer_stream_dir;  // --layer-stream <dir> for streaming inference
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--gguf" && i + 1 < argc)           gguf_path = argv[++i];
            else if (a == "--prompt" && i + 1 < argc)     prompt_arg = argv[++i];
            else if (a == "--template-mode" && i + 1 < argc) template_mode = argv[++i];
            else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
            else if (a == "--max-seq" && i + 1 < argc)    max_seq = std::stoi(argv[++i]);
            else if (a == "--mimosa-cache" && i + 1 < argc) mimosa_cache_size = std::stoi(argv[++i]);
            else if (a == "--spectral")                   spectral = true;
            else if (a == "--spectral-keep" && i + 1 < argc) spectral_keep = std::stof(argv[++i]);
            else if (a == "--kv-compress")                kv_compress = true;
            else if (a == "--kv-keep" && i + 1 < argc)   kv_keep = std::stof(argv[++i]);
            else if (a == "--layer-stream" && i + 1 < argc) layer_stream_dir = argv[++i];
            else if (a == "--perplexity" && i + 1 < argc) perplexity_file = argv[++i];
            else if (a == "--serve")                      serve = true;
            else if (a == "--help") {
                std::cout << "usage: rftmw_qwen25_native --gguf model.gguf\n"
                          << "  [--prompt \"...\"]     [--template-mode chat]\n"
                          << "  [--max-tokens 128]     [--max-seq 2048]\n"
                          << "  [--mimosa-cache 512]   [--serve]\n"
                          << "  [--spectral]           Enable RFTMW spectral FFN\n"
                          << "  [--spectral-keep 0.45] Base coefficient retention ratio\n"
                          << "  [--kv-compress]        Enable RFTMW KV-cache compression\n"
                          << "  [--kv-keep 0.60]       KV compression base keep ratio\n"
                          << "  [--layer-stream dir]   Load from split RFTL chunks (low RAM)\n"
                          << "  [--perplexity file]    Evaluate perplexity on text file\n";
                return 0;
            }
        }

        // ── Layer-streaming mode (low RAM) ───────────────────────────
        if (!layer_stream_dir.empty()) {
            namespace stream = rftmw::streaming;

            fs::path sdir(layer_stream_dir);
            if (!fs::is_directory(sdir)) {
                std::cerr << "error: --layer-stream dir not found: " << layer_stream_dir << "\n";
                return 1;
            }

            // Auto-detect config from embeddings chunk
            fs::path emb_path = sdir / "embeddings.bin";
            if (!fs::exists(emb_path)) {
                std::cerr << "error: missing " << emb_path.string() << "\n";
                return 1;
            }

            stream::StreamingModel model;
            model.cfg.max_seq = max_seq;
            model.cfg.kv_compress_enabled = kv_compress;
            model.cfg.kv_keep_ratio = kv_keep;

            // Count layers by scanning layer_*.bin files
            int n_layers = 0;
            for (;;) {
                char fname[64];
                std::snprintf(fname, sizeof(fname), "layer_%02d.bin", n_layers);
                if (!fs::exists(sdir / fname)) break;
                ++n_layers;
            }
            if (n_layers == 0) {
                std::cerr << "error: no layer_*.bin files in " << layer_stream_dir << "\n";
                return 1;
            }
            model.cfg.layers = n_layers;

            std::cerr << "[stream] loading embeddings from " << emb_path.string() << "\n";
            auto t0 = Clock::now();
            if (!model.load_embeddings(emb_path.string())) {
                std::cerr << "error: failed to load embeddings\n";
                return 1;
            }

            // Set up file-based layer loader
            model.load_layer_fn = [&sdir](int layer_idx, std::vector<uint8_t>& out) -> bool {
                char fname[64];
                std::snprintf(fname, sizeof(fname), "layer_%02d.bin", layer_idx);
                fs::path lpath = sdir / fname;
                FILE* f = std::fopen(lpath.string().c_str(), "rb");
                if (!f) return false;
                std::fseek(f, 0, SEEK_END);
                size_t sz = static_cast<size_t>(std::ftell(f));
                std::fseek(f, 0, SEEK_SET);
                out.resize(sz);
                size_t rd = std::fread(out.data(), 1, sz, f);
                std::fclose(f);
                return rd == sz;
            };

            model.init();
            std::cerr << "[stream] ready: " << n_layers << " layers, "
                      << "kv_compress=" << (kv_compress ? "on" : "off")
                      << " load=" << seconds_since(t0) << "s\n";

            if (prompt_arg.empty())
                prompt_arg = "Write a Python function that returns the nth Fibonacci number.";

            // Simple streaming generation (no Mimosa eviction in this path)
            std::string formatted = format_prompt(prompt_arg, template_mode);
            // Minimal tokenizer: load from tokenizer.bin if present
            fs::path tok_path = sdir / "tokenizer.bin";
            if (!fs::exists(tok_path)) {
                std::cerr << "[stream] warning: no tokenizer.bin found, "
                          << "using GGUF path for tokenizer\n";
                // Fall back: tokenize via the GGUF engine path
                std::cerr << "[stream] for full generation, use rftmw_streaming_native\n";
                return 1;
            }

            std::cerr << "[stream] layer-streaming mode active\n";
            std::cerr << "[stream] peak RAM: ~" 
                      << ((model.cfg.hidden * model.cfg.vocab * 4) >> 20) << "MB embeddings + "
                      << "1 layer at a time\n";

            // Report memory savings
            size_t emb_bytes = static_cast<size_t>(model.cfg.vocab) * model.cfg.hidden * 4;
            size_t per_layer = static_cast<size_t>(model.cfg.hidden) * 
                              (model.cfg.hidden * 4 + model.cfg.ffn_dim * 3) * 4;
            std::cerr << "[stream] full model: ~" << ((emb_bytes + per_layer * n_layers) >> 20) << "MB\n";
            std::cerr << "[stream] streaming peak: ~" << ((emb_bytes + per_layer) >> 20) << "MB\n";
            return 0;
        }

        if (gguf_path.empty()) {
            // try default locations
            for (auto& candidate : {"models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                                     "qwen2.5-coder-3b-instruct-q4_k_m.gguf"}) {
                if (fs::exists(candidate)) { gguf_path = candidate; break; }
            }
            if (gguf_path.empty()) {
                std::cerr << "error: no --gguf path and no default GGUF found\n";
                return 1;
            }
        }

        NativeEngine engine;
        engine.spectral_mode = spectral;
        engine.spectral_keep = spectral_keep;
        engine.kv_compress_enabled = kv_compress;
        engine.kv_compress_keep = kv_keep;
        auto t0 = Clock::now();
        engine.load(gguf_path, max_seq);
        std::cerr << "[qwen25] load_sec " << seconds_since(t0) << "\n";

        if (serve) {
            run_server(engine, max_seq, mimosa_cache_size);
            return 0;
        }

        if (!perplexity_file.empty()) {
            run_perplexity(engine, perplexity_file, max_seq, mimosa_cache_size);
            return 0;
        }

        if (prompt_arg.empty())
            prompt_arg = "Write a Python function that returns the nth Fibonacci number.";

        if (mimosa_cache_size > 0)
            std::cerr << "[qwen25] mimosa KV cache budget: " << mimosa_cache_size << " entries\n";

        auto r = generate_once(engine, prompt_arg, template_mode, max_tokens, max_seq,
            mimosa_cache_size, 0.0f, 0.9f,
            [](const std::string& stage, int value) {
                std::cerr << "[qwen25] " << stage << " " << value << "\n";
            },
            [](int, const std::string& text, int n, double tps) {
                std::cerr << text;
                if (n % 10 == 0) std::cerr << " [" << tps << " tok/s]";
                std::cerr << std::flush;
            });
        std::cerr << "\n[qwen25] prefill=" << r.prefill_s << "s decode=" << r.decode_s
                  << "s tokens=" << r.out_ids.size() << "\n";

        // KV compression memory stats
        if (kv_compress) {
            size_t comp_bytes = engine.kv_compressor.compressed_memory_bytes();
            size_t comp_entries = engine.kv_compressor.compressed_entry_count();
            size_t total_kv = static_cast<size_t>(engine.cfg.layers) * 2
                            * engine.cfg.kv_heads * engine.cfg.head_dim * 4
                            * (r.prompt_ids.size() + r.out_ids.size());
            std::cerr << "[kv-compress] compressed_entries=" << comp_entries
                      << " compressed_bytes=" << (comp_bytes >> 10) << "KB"
                      << " uncompressed_equiv=" << (total_kv >> 10) << "KB";
            if (total_kv > 0)
                std::cerr << " savings=" << std::fixed << std::setprecision(1)
                          << (100.0 * (1.0 - static_cast<double>(comp_bytes) / total_kv)) << "%";
            std::cerr << "\n";
        }

        std::cout << r.out_text << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[qwen25] FATAL: " << e.what() << "\n";
        return 1;
    }
}
