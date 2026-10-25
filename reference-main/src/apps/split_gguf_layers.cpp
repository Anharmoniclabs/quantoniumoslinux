/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * split_gguf_layers.cpp — Split a GGUF model into per-layer chunks
 * =================================================================
 *
 * Offline tool that reads a GGUF file and emits self-contained binary
 * chunks suitable for layer-streaming inference:
 *
 *   <output_dir>/embeddings.bin   — token_embd, output_norm, output weights
 *   <output_dir>/layer_00.bin     — all tensors for blk.0.*
 *   <output_dir>/layer_01.bin     — all tensors for blk.1.*
 *   ...
 *   <output_dir>/tokenizer.bin    — tokenizer vocab/merges (for WASM)
 *
 * Each chunk has a minimal header:
 *   [4] magic "RFTL" (RFT Layer)
 *   [4] version (1)
 *   [4] num_tensors
 *   [4] reserved
 *   For each tensor:
 *     [4] name_len
 *     [name_len] name (UTF-8)
 *     [4] type (GGUF type code)
 *     [4] n_dims
 *     [n_dims * 8] shape (uint64 each)
 *     [8] data_size
 *     [data_size] raw tensor bytes (aligned to 32 bytes)
 *
 * No re-encoding — raw GGUF tensor bytes are copied verbatim.
 * This preserves bit-exact Q4K/Q6K quality.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// ── GGUF parsing (minimal, reused from qwen25_native_cli.cpp) ───

static uint32_t read_u32(const uint8_t*& p) { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
static uint64_t read_u64(const uint8_t*& p) { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }

static std::string read_string(const uint8_t*& p) {
    uint64_t n = read_u64(p);
    std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
    p += n;
    return s;
}

static void skip_gguf_scalar(const uint8_t*& p, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: p += 1; break;
        case 2: case 3: p += 2; break;
        case 4: case 5: case 6: p += 4; break;
        case 8: { uint64_t n = read_u64(p); p += n; break; }
        case 10: case 11: case 12: p += 8; break;
        default: throw std::runtime_error("unsupported GGUF scalar type");
    }
}

static void skip_gguf_value(const uint8_t*& p, uint32_t type) {
    if (type == 9) {
        uint32_t elem_type = read_u32(p);
        uint64_t n = read_u64(p);
        for (uint64_t i = 0; i < n; ++i) skip_gguf_value(p, elem_type);
    } else {
        skip_gguf_scalar(p, type);
    }
}

static std::vector<std::string> read_string_array(const uint8_t*& p) {
    uint32_t elem_type = read_u32(p);
    uint64_t n = read_u64(p);
    if (elem_type != 8) throw std::runtime_error("expected string array");
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) out.push_back(read_string(p));
    return out;
}

constexpr int GGUF_F32  = 0;
constexpr int GGUF_F16  = 1;
constexpr int GGUF_Q4_K = 12;
constexpr int GGUF_Q6_K = 14;

static uint64_t gguf_tensor_nbytes(uint32_t type, const std::vector<uint64_t>& ne) {
    uint64_t n = 1;
    for (auto d : ne) n *= d;
    if (type == GGUF_F32)  return n * 4;
    if (type == GGUF_F16)  return n * 2;
    if (type == GGUF_Q4_K) return (n / 256) * 144;
    if (type == GGUF_Q6_K) return (n / 256) * 210;
    throw std::runtime_error("unsupported tensor type " + std::to_string(type));
}

struct TensorEntry {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    const uint8_t* data = nullptr;
};

// ── Chunk writer ─────────────────────────────────────────────────

static void write_u32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); }
static void write_u64(std::ofstream& f, uint64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); }

static void write_chunk(const fs::path& path, const std::vector<TensorEntry>& tensors) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());

    // Header
    f.write("RFTL", 4);
    write_u32(f, 1);  // version
    write_u32(f, static_cast<uint32_t>(tensors.size()));
    write_u32(f, 0);  // reserved

    for (const auto& t : tensors) {
        // Name
        write_u32(f, static_cast<uint32_t>(t.name.size()));
        f.write(t.name.data(), static_cast<std::streamsize>(t.name.size()));

        // Type and shape
        write_u32(f, t.type);
        write_u32(f, static_cast<uint32_t>(t.ne.size()));
        for (auto d : t.ne) write_u64(f, d);

        // Data (align to 32 bytes for efficient loading)
        write_u64(f, t.bytes);
        f.write(reinterpret_cast<const char*>(t.data), static_cast<std::streamsize>(t.bytes));

        // Pad to 32-byte alignment
        uint64_t pos = static_cast<uint64_t>(f.tellp());
        uint64_t aligned = (pos + 31u) & ~31u;
        if (aligned > pos) {
            std::vector<char> pad(static_cast<size_t>(aligned - pos), 0);
            f.write(pad.data(), static_cast<std::streamsize>(pad.size()));
        }
    }

    std::cerr << "  wrote " << path.filename().string()
              << " (" << tensors.size() << " tensors, "
              << (f.tellp() / (1024 * 1024)) << " MB)\n";
}

// ── Layer classification ─────────────────────────────────────────

static int extract_layer_index(const std::string& name) {
    // Match "blk.N." pattern
    std::smatch m;
    static const std::regex blk_re("^blk\\.(\\d+)\\.");
    if (std::regex_search(name, m, blk_re)) {
        return std::stoi(m[1].str());
    }
    return -1;  // not a layer tensor
}

// ── Main ─────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: split_gguf_layers <model.gguf> <output_dir>\n"
                  << "\nSplits a GGUF model into per-layer chunks for\n"
                  << "layer-streaming inference (browser/WASM/mobile).\n"
                  << "\nOutput:\n"
                  << "  <output_dir>/embeddings.bin  — embeddings + output weights\n"
                  << "  <output_dir>/layer_NN.bin    — per-layer weight chunks\n";
        return 1;
    }

    fs::path gguf_path = argv[1];
    fs::path out_dir   = argv[2];

    // Memory-map the GGUF
    std::cerr << "[split] reading " << gguf_path.string() << "\n";
    std::ifstream gguf_file(gguf_path, std::ios::binary | std::ios::ate);
    if (!gguf_file) { std::cerr << "ERROR: cannot open " << gguf_path << "\n"; return 1; }
    size_t file_size = static_cast<size_t>(gguf_file.tellg());
    gguf_file.seekg(0);
    std::vector<uint8_t> file_data(file_size);
    gguf_file.read(reinterpret_cast<char*>(file_data.data()), static_cast<std::streamsize>(file_size));
    gguf_file.close();

    const uint8_t* base = file_data.data();
    const uint8_t* p = base;

    // Parse GGUF header
    if (std::memcmp(p, "GGUF", 4) != 0) { std::cerr << "ERROR: bad GGUF magic\n"; return 1; }
    p += 4;
    uint32_t version = read_u32(p);
    (void)version;
    uint64_t n_tensors = read_u64(p);
    uint64_t n_kv = read_u64(p);

    std::cerr << "[split] " << n_tensors << " tensors, " << n_kv << " metadata keys\n";

    // Parse metadata (skip, but capture tokenizer data)
    std::vector<std::string> tokenizer_tokens;
    std::vector<std::string> tokenizer_merges;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = read_string(p);
        uint32_t type = read_u32(p);
        if (key == "tokenizer.ggml.tokens" && type == 9)
            tokenizer_tokens = read_string_array(p);
        else if (key == "tokenizer.ggml.merges" && type == 9)
            tokenizer_merges = read_string_array(p);
        else
            skip_gguf_value(p, type);
    }

    // Parse tensor headers
    std::vector<TensorEntry> all_tensors;
    all_tensors.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorEntry t;
        t.name = read_string(p);
        uint32_t n_dims = read_u32(p);
        for (uint32_t d = 0; d < n_dims; ++d) t.ne.push_back(read_u64(p));
        t.type = read_u32(p);
        t.offset = read_u64(p);
        t.bytes = gguf_tensor_nbytes(t.type, t.ne);
        all_tensors.push_back(std::move(t));
    }

    // Compute data start (after all headers, aligned to 32)
    uint64_t header_end = static_cast<uint64_t>(p - base);
    uint64_t data_start = (header_end + 31u) & ~31u;

    // Set data pointers
    for (auto& t : all_tensors) {
        t.data = base + data_start + t.offset;
    }

    // Classify tensors
    std::unordered_map<int, std::vector<TensorEntry>> layer_tensors;
    std::vector<TensorEntry> embedding_tensors;
    int max_layer = -1;

    for (const auto& t : all_tensors) {
        int layer_idx = extract_layer_index(t.name);
        if (layer_idx >= 0) {
            layer_tensors[layer_idx].push_back(t);
            max_layer = std::max(max_layer, layer_idx);
        } else {
            // Embedding, output norm, output head, etc.
            embedding_tensors.push_back(t);
        }
    }

    std::cerr << "[split] " << embedding_tensors.size() << " embedding/output tensors, "
              << (max_layer + 1) << " layers\n";

    // Create output directory
    fs::create_directories(out_dir);

    // Write embedding chunk
    write_chunk(out_dir / "embeddings.bin", embedding_tensors);

    // Write per-layer chunks
    for (int i = 0; i <= max_layer; ++i) {
        auto it = layer_tensors.find(i);
        if (it == layer_tensors.end()) {
            std::cerr << "WARNING: no tensors for layer " << i << "\n";
            continue;
        }
        char fname[32];
        std::snprintf(fname, sizeof(fname), "layer_%02d.bin", i);
        write_chunk(out_dir / fname, it->second);
    }

    // Summary
    uint64_t total_bytes = 0;
    for (const auto& t : all_tensors) total_bytes += t.bytes;
    std::cerr << "[split] done: " << (max_layer + 1) << " layer chunks + embeddings\n"
              << "[split] total tensor data: " << (total_bytes / (1024 * 1024)) << " MB\n";

    return 0;
}
