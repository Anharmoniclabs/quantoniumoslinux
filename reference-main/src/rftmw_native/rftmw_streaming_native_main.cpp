/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_streaming_native_main.cpp — Native CLI for streaming inference
 * =====================================================================
 *
 * Usage:
 *   ./rftmw_streaming_native <split_dir> "<prompt>"
 *
 * Where <split_dir> contains:
 *   embeddings.bin, layer_00.bin, layer_01.bin, ..., layer_NN.bin
 * (produced by rftmw_split_gguf)
 */

#include "rftmw_streaming_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <split_dir> \"<prompt>\"\n", argv[0]);
        return 1;
    }

    std::string split_dir = argv[1];
    std::string prompt = argv[2];

    // Ensure trailing slash
    if (!split_dir.empty() && split_dir.back() != '/') split_dir += '/';

    std::fprintf(stderr, "[streaming] Loading model from %s\n", split_dir.c_str());

    rftmw::streaming::StreamingModel model;

    // Default config for Qwen2.5-7B
    model.cfg.layers    = 36;
    model.cfg.hidden    = 2048;
    model.cfg.heads     = 16;
    model.cfg.kv_heads  = 2;
    model.cfg.head_dim  = 128;
    model.cfg.ffn_dim   = 11008;
    model.cfg.vocab     = 151936;
    model.cfg.max_seq   = 2048;
    model.cfg.kv_keep_ratio = 0.60f;
    model.cfg.kv_compress_enabled = true;

    // Load embeddings
    std::string emb_path = split_dir + "embeddings.bin";
    if (!model.load_embeddings(emb_path)) {
        std::fprintf(stderr, "[streaming] ERROR: failed to load %s\n", emb_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "[streaming] Embeddings loaded\n");

    // Set up layer loading callback
    model.load_layer_fn = [&split_dir](int layer_idx, std::vector<uint8_t>& out) -> bool {
        char fname[256];
        std::snprintf(fname, sizeof(fname), "%slayer_%02d.bin", split_dir.c_str(), layer_idx);
        FILE* f = std::fopen(fname, "rb");
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
    std::fprintf(stderr, "[streaming] Model initialized, generating...\n");

    // Simple byte-level tokenization for demo (real use needs proper tokenizer)
    std::vector<int> tokens;
    for (unsigned char c : prompt) tokens.push_back(static_cast<int>(c));

    // Forward the prompt
    int cache_len = 0;
    const float* logits = nullptr;
    for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
        logits = model.forward_token(tokens[i], i, cache_len);
        cache_len++;
    }

    // Generate up to 128 tokens
    int max_gen = 128;
    for (int step = 0; step < max_gen; ++step) {
        int next = model.sample_greedy(logits, model.cfg.vocab);
        if (next == 151643 || next == 151645) break;  // EOS

        std::printf("%d ", next);
        std::fflush(stdout);

        int pos = static_cast<int>(tokens.size()) + step;
        logits = model.forward_token(next, pos, cache_len);
        cache_len++;
    }
    std::printf("\n");

    std::fprintf(stderr, "[streaming] Done. Peak RAM: layer-streaming mode.\n");
    return 0;
}
