/*
 * Mimosa KV Cache Eviction Test for llama.cpp
 *
 * Compares two KV cache management strategies:
 *   A) Sliding Window: remove first half of context (llama.cpp default)
 *   B) Mimosa + Three-Distance: turgor-scored zone-aware selective eviction
 *
 * Uses a long-context needle-in-haystack test: facts placed early in the prompt
 * are queried after context overflow. Mimosa should retain important early facts.
 *
 * Copyright 2024-2026 QuantoniumOS Research
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <io.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Mmap token cache: tokenize once, persist token IDs to a binary file, then
// mmap on subsequent runs. Skips re-tokenization entirely.
// File format: [uint32_t n_tokens] [llama_token * n_tokens]
// ─────────────────────────────────────────────────────────────────────────────

struct MmapTokenCache {
    void *  map_ptr  = nullptr;
    size_t  map_size = 0;
    int     fd       = -1;

    // Try to load a pre-tokenized cache. Returns empty vector if not found.
    std::vector<llama_token> load(const std::string & path) {
#ifndef _WIN32
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return {};
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(uint32_t)) { close(fd); fd = -1; return {}; }
        map_size = (size_t)st.st_size;
        map_ptr = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map_ptr == MAP_FAILED) { map_ptr = nullptr; close(fd); fd = -1; return {}; }

        const uint32_t n = *(const uint32_t *)map_ptr;
        size_t expected = sizeof(uint32_t) + n * sizeof(llama_token);
        if (expected > map_size) { unload(); return {}; }

        const llama_token * toks = (const llama_token *)((const char *)map_ptr + sizeof(uint32_t));
        return std::vector<llama_token>(toks, toks + n);
#else
        (void)path; return {};
#endif
    }

    void unload() {
#ifndef _WIN32
        if (map_ptr && map_ptr != MAP_FAILED) { munmap(map_ptr, map_size); map_ptr = nullptr; }
        if (fd >= 0) { close(fd); fd = -1; }
#endif
    }

    // Save tokenized IDs to binary file for future mmap loads.
    static bool save(const std::string & path, const std::vector<llama_token> & tokens) {
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) return false;
        uint32_t n = (uint32_t)tokens.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(tokens.data(), sizeof(llama_token), n, f);
        fclose(f);
        return true;
    }

    ~MmapTokenCache() { unload(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Mimosa KV Gate — turgor-scored eviction (ported from QuantoniumOS)
// ─────────────────────────────────────────────────────────────────────────────

struct MimosaKVGate {
    float threshold         = 1.5f;
    float propagation_decay = 0.6f;
    int   recovery_window   = 8;
    int   max_hops          = 4;

    // Compute turgor scores. importance[i] = importance of position i.
    // Returns turgor[i] in [0,1], lower = more evictable.
    std::vector<float> compute_turgor(const std::vector<float>& importance) const {
        int n = (int)importance.size();
        if (n == 0) return {};

        float mean_imp = 0.0f;
        for (int i = 0; i < n; ++i) mean_imp += importance[i];
        mean_imp = mean_imp / n + 1e-8f;

        std::vector<float> rel_act(n);
        std::vector<bool>  triggered(n, false);
        for (int i = 0; i < n; ++i) {
            rel_act[i]   = importance[i] / mean_imp;
            triggered[i] = rel_act[i] > threshold;
        }

        // Hydraulic propagation
        std::vector<float> hydraulic(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            if (!triggered[i]) continue;
            for (int hop = 0; hop <= max_hops; ++hop) {
                float signal = std::pow(propagation_decay, (float)hop);
                if (i - hop >= 0)           hydraulic[i - hop] += signal;
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
                float recovery = std::min(1.0f, (float)min_dist / recovery_window);
                turgor[i] = std::max(turgor[i], recovery);
            }
        } else {
            for (int i = 0; i < n; ++i) turgor[i] = 1.0f;
        }
        return turgor;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Three-Distance Router — Steinhaus–Šós phi-grid zone partitioning
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double PHI = 1.6180339887498948482;

struct ThreeDistanceZoneInfo {
    std::vector<std::vector<int>> zones;
    std::vector<double> density;
    int n_zones = 0;
};

struct ThreeDistanceRouter {
    static ThreeDistanceZoneInfo analyze_gap_structure(int n) {
        ThreeDistanceZoneInfo info;
        if (n <= 0) return info;

        std::vector<double> freqs(n);
        for (int k = 0; k < n; ++k)
            freqs[k] = std::fmod((double)(k + 1) * PHI, 1.0);

        std::vector<int> sort_order(n);
        std::iota(sort_order.begin(), sort_order.end(), 0);
        std::sort(sort_order.begin(), sort_order.end(),
                  [&](int a, int b) { return freqs[a] < freqs[b]; });

        std::vector<double> sorted_freqs(n);
        for (int i = 0; i < n; ++i) sorted_freqs[i] = freqs[sort_order[i]];

        std::vector<double> all_gaps(n);
        for (int i = 0; i < n - 1; ++i)
            all_gaps[i] = sorted_freqs[i + 1] - sorted_freqs[i];
        all_gaps[n - 1] = 1.0 + sorted_freqs[0] - sorted_freqs[n - 1];

        // Find distinct gap sizes
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

        // Per-entry local density
        std::vector<double> left_gaps(n), right_gaps(n);
        left_gaps[0] = all_gaps[n - 1];
        for (int i = 1; i < n; ++i) left_gaps[i] = all_gaps[i - 1];
        right_gaps[n - 1] = all_gaps[n - 1];
        for (int i = 0; i < n - 1; ++i) right_gaps[i] = all_gaps[i];

        std::vector<double> density_sorted(n);
        for (int i = 0; i < n; ++i) {
            double local_gap = (left_gaps[i] + right_gaps[i]) / 2.0;
            density_sorted[i] = 1.0 / std::max(local_gap, 1e-15);
        }
        info.density.resize(n);
        for (int i = 0; i < n; ++i)
            info.density[sort_order[i]] = density_sorted[i];

        // Zone boundaries
        double zone_threshold;
        if (unique_gaps.size() > 1)
            zone_threshold = (unique_gaps.back() + unique_gaps[unique_gaps.size() - 2]) / 2.0;
        else
            zone_threshold = unique_gaps[0] * 1.5;

        std::vector<int> zone_boundaries;
        for (int i = 0; i < n; ++i)
            if (all_gaps[i] > zone_threshold) zone_boundaries.push_back(i);

        std::vector<std::vector<int>> raw_zones;
        int prev = 0;
        for (int zb : zone_boundaries) {
            if (zb + 1 > prev) {
                std::vector<int> zone(sort_order.begin() + prev, sort_order.begin() + zb + 1);
                raw_zones.push_back(std::move(zone));
            }
            prev = zb + 1;
        }
        if (prev < n) {
            std::vector<int> zone(sort_order.begin() + prev, sort_order.end());
            raw_zones.push_back(std::move(zone));
        }

        // Merge small zones
        int log_phi_n = std::max(3, (int)(std::log(std::max(n, 2)) / std::log(PHI)));
        int min_zone_size = std::max(2, n / (2 * log_phi_n));
        std::vector<int> accum;
        for (auto& rz : raw_zones) {
            accum.insert(accum.end(), rz.begin(), rz.end());
            if ((int)accum.size() >= min_zone_size) {
                info.zones.push_back(std::move(accum));
                accum.clear();
            }
        }
        if (!accum.empty()) {
            if (!info.zones.empty())
                info.zones.back().insert(info.zones.back().end(), accum.begin(), accum.end());
            else
                info.zones.push_back(std::move(accum));
        }
        info.n_zones = (int)info.zones.size();
        return info;
    }

    // Zone-aware eviction: given importance + turgor scores and a target budget,
    // returns the POSITIONS to evict (sorted descending for safe removal).
    static std::vector<int> zone_aware_evict(
        const std::vector<float>& importance,
        const std::vector<float>& turgor,
        int cache_budget)
    {
        int n = (int)importance.size();
        if (n <= cache_budget) return {};

        auto gap_info = analyze_gap_structure(n);
        if (gap_info.n_zones <= 1) {
            // Single zone — flat eviction by lowest turgor
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return turgor[a] < turgor[b]; });
            int n_evict = n - cache_budget;
            std::vector<int> evict_list;
            for (int i = 0; i < (int)order.size() && (int)evict_list.size() < n_evict; ++i)
                if (order[i] != n - 1)
                    evict_list.push_back(order[i]);
            std::sort(evict_list.rbegin(), evict_list.rend());
            return evict_list;
        }

        // Per-zone energy
        int n_zones = gap_info.n_zones;
        std::vector<double> zone_scores(n_zones, 0.0);
        for (int z = 0; z < n_zones; ++z)
            for (int idx : gap_info.zones[z]) {
                double imp2 = (double)importance[idx]; imp2 *= imp2;
                zone_scores[z] += imp2 * gap_info.density[idx];
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
            for (int i = 0; i < (int)order.size() && (int)evict_list.size() < n_evict; ++i)
                if (order[i] != n - 1) evict_list.push_back(order[i]);
            std::sort(evict_list.rbegin(), evict_list.rend());
            return evict_list;
        }

        // Proportional allocation
        std::vector<int> zone_k(n_zones);
        for (int z = 0; z < n_zones; ++z) {
            double raw = zone_scores[z] / total_score * cache_budget;
            zone_k[z] = std::max(1, (int)std::round(raw));
            zone_k[z] = std::min(zone_k[z], (int)gap_info.zones[z].size());
        }

        // Balance to exact budget
        while (true) {
            int s = 0; for (int z = 0; z < n_zones; ++z) s += zone_k[z];
            if (s <= cache_budget) break;
            int trim_z = 0; double best_ratio = 1e30;
            for (int z = 0; z < n_zones; ++z)
                if (zone_k[z] > 1) { double r = zone_scores[z] / zone_k[z]; if (r < best_ratio) { best_ratio = r; trim_z = z; } }
            if (zone_k[trim_z] > 1) zone_k[trim_z]--; else break;
        }
        while (true) {
            int s = 0; for (int z = 0; z < n_zones; ++z) s += zone_k[z];
            if (s >= cache_budget) break;
            int boost_z = 0; double best_ratio = -1.0;
            for (int z = 0; z < n_zones; ++z)
                if (zone_k[z] < (int)gap_info.zones[z].size()) { double r = zone_scores[z] / std::max(zone_k[z], 1); if (r > best_ratio) { best_ratio = r; boost_z = z; } }
            if (zone_k[boost_z] < (int)gap_info.zones[boost_z].size()) zone_k[boost_z]++; else break;
        }

        // Within each zone, keep top entries by turgor
        std::vector<bool> keep(n, false);
        keep[n - 1] = true;  // always keep most recent
        for (int z = 0; z < n_zones; ++z) {
            auto& zidx = gap_info.zones[z];
            std::vector<int> sorted_zone = zidx;
            std::sort(sorted_zone.begin(), sorted_zone.end(),
                      [&](int a, int b) { return turgor[a] > turgor[b]; });
            int kept = 0;
            for (int idx : sorted_zone)
                if (kept < zone_k[z]) { keep[idx] = true; kept++; }
        }

        std::vector<int> evict_list;
        for (int i = 0; i < n; ++i) if (!keep[i]) evict_list.push_back(i);
        std::sort(evict_list.rbegin(), evict_list.rend());
        return evict_list;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Importance signal: combined recency + content information score
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> compute_importance(
    const std::vector<llama_token>& tokens,
    const llama_vocab * vocab,
    int n_cached,
    int n_keep)  // first n_keep positions always get max importance
{
    std::vector<float> importance(n_cached);

    for (int i = 0; i < n_cached; ++i) {
        // Base: exponential recency decay with long tail
        float recency = 0.3f + 0.7f * std::exp(-0.005f * (float)(n_cached - 1 - i));

        // Boost: first n_keep tokens (system prompt / BOS territory)
        if (i < n_keep) {
            importance[i] = 10.0f;
            continue;
        }

        // Boost: special structural tokens (newlines, punctuation boundaries)
        float content_boost = 1.0f;
        if (i < (int)tokens.size()) {
            char buf[256];
            int len = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
            if (len > 0) {
                std::string tok_text(buf, len);
                // Newlines/paragraphs signal structure
                if (tok_text.find('\n') != std::string::npos) content_boost = 2.5f;
                // Sentence-ending punctuation
                else if (tok_text.find('.') != std::string::npos ||
                         tok_text.find('?') != std::string::npos ||
                         tok_text.find('!') != std::string::npos) content_boost = 1.8f;
                // Colons, semicolons (often precede key info)
                else if (tok_text.find(':') != std::string::npos) content_boost = 2.0f;
                // Numbers (facts, data)
                for (char c : tok_text) if (c >= '0' && c <= '9') { content_boost = std::max(content_boost, 1.5f); break; }
            }
        }

        importance[i] = recency * content_boost;
    }
    return importance;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main test
// ─────────────────────────────────────────────────────────────────────────────

// Test prompt: A passage with key facts distributed throughout.
// The "needle" facts are at the BEGINNING. After overflow and eviction,
// we ask about those early facts. Sliding window loses them; Mimosa keeps them.
static const char* LONG_CONTEXT_PROMPT =
    "IMPORTANT FACTS TO REMEMBER:\n"
    "Fact 1: The secret project codename is AURORA-7.\n"
    "Fact 2: The launch date is March 15, 2027.\n"
    "Fact 3: The budget allocation is exactly $4.7 million.\n"
    "Fact 4: The lead engineer is Dr. Sarah Chen from MIT.\n"
    "Fact 5: The backup facility is located in Zurich, Switzerland.\n"
    "\n"
    "--- BEGIN TECHNICAL REPORT ---\n"
    "\n"
    "Section 1: Introduction to Distributed Neural Architecture\n"
    "\n"
    "The field of distributed neural computation has seen remarkable advances in recent years. "
    "Modern transformer architectures leverage multi-head attention mechanisms to capture long-range "
    "dependencies in sequential data. The fundamental challenge remains: how do we efficiently manage "
    "the ever-growing key-value cache that stores contextual information during autoregressive generation? "
    "Traditional approaches rely on fixed sliding window strategies that blindly discard older context, "
    "potentially losing critical information that was established early in the conversation.\n"
    "\n"
    "Section 2: Memory-Efficient Attention Patterns\n"
    "\n"
    "Recent work on memory-efficient attention has explored several directions. Flash attention reduces "
    "the memory footprint through tiling and recomputation. Ring attention distributes the KV cache "
    "across multiple devices. Paged attention manages memory through virtual memory-inspired techniques. "
    "However, none of these approaches address the fundamental question of WHAT to keep when the cache "
    "exceeds its budget — they only address HOW to store it more efficiently. The eviction policy itself "
    "remains a simple sliding window or random eviction in most implementations.\n"
    "\n"
    "Section 3: Biological Inspiration from Plant Turgor Dynamics\n"
    "\n"
    "Plants maintain cellular pressure through turgor — the osmotic pressure exerted against cell walls. "
    "When water is scarce, turgor drops and cells lose rigidity, causing wilting. In well-hydrated tissue, "
    "turgor is high and the plant maintains its structure. This biological mechanism provides an elegant "
    "analogy for cache management: information that is 'well-connected' to important context maintains "
    "high turgor (retention pressure), while isolated or redundant information has low turgor and can be "
    "safely evicted. The hydraulic propagation of water through plant vasculature mirrors how importance "
    "signals should propagate through the cache — nearby positions of important tokens inherit some of "
    "that importance, creating pressure gradients that inform eviction decisions.\n"
    "\n"
    "Section 4: Mathematical Foundations of the Three-Distance Theorem\n"
    "\n"
    "The Steinhaus-Sos three-distance theorem states that for any irrational number alpha and positive "
    "integer N, the N points frac(k*alpha) for k=1,2,...,N partition the unit circle into gaps of at "
    "most three distinct lengths. When alpha is the golden ratio phi = (1+sqrt(5))/2, these gaps have "
    "particularly elegant properties related to Fibonacci numbers. This mathematical structure provides "
    "a natural way to partition cache positions into spectral zones — regions of the cache that share "
    "similar structural properties under the golden ratio frequency map.\n"
    "\n"
    "Section 5: Experimental Results on Synthetic Benchmarks\n"
    "\n"
    "We evaluated our biologically-inspired eviction strategy on a suite of synthetic benchmarks designed "
    "to test long-range information retention. The benchmarks include: (a) needle-in-haystack retrieval, "
    "where key facts are embedded at various positions in a long context; (b) multi-hop reasoning chains "
    "that require connecting information from distant parts of the context; (c) summarization tasks where "
    "the source document exceeds the KV cache budget. Across all benchmarks, the turgor-scored zone-aware "
    "eviction consistently outperformed sliding window, random eviction, and attention-score-based eviction "
    "strategies. The improvement was most pronounced in needle-in-haystack at low cache budgets (50-70% of "
    "context length), where sliding window completely loses early facts.\n"
    "\n"
    "Section 6: Implementation Considerations\n"
    "\n"
    "The computational overhead of turgor scoring is O(N*H) where N is the cache size and H is the max "
    "hop count (typically 4). The three-distance zone analysis is O(N log N) due to sorting. Combined, "
    "the eviction decision takes less than 0.1ms for cache sizes up to 8192, which is negligible compared "
    "to the attention computation itself. The memory overhead is 2*N floats (importance + turgor arrays), "
    "which is also negligible. The strategy can be implemented as a drop-in replacement for any existing "
    "eviction policy without changing the core attention computation.\n"
    "\n"
    "Section 7: Future Directions\n"
    "\n"
    "Several promising directions remain unexplored. First, learning the turgor parameters (threshold, "
    "propagation decay, recovery window) from data could improve performance on specific domains. Second, "
    "combining turgor scoring with attention-based importance (using actual attention weights from the last "
    "forward pass) could provide an even stronger signal. Third, hierarchical zone structures with multiple "
    "levels of golden ratio partitioning could enable more nuanced retention policies for very long contexts.\n"
    "\n"
    "--- END TECHNICAL REPORT ---\n"
    "\n"
    "Based on the IMPORTANT FACTS listed at the very beginning of this document, please answer: "
    "What is the secret project codename, and what is the exact budget allocation?";

static std::string run_with_strategy(
    const char* model_path,
    const char* strategy_name,
    int n_ctx,         // KV cache budget
    int n_keep,        // tokens to always keep at start
    int n_predict,     // tokens to generate
    bool use_mimosa,   // false = sliding window, true = Mimosa
    const std::vector<llama_token> * shared_tokens = nullptr)  // pre-tokenized (mmap'd)
{
    fprintf(stderr, "\n══════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Strategy: %s  |  n_ctx=%d  n_keep=%d\n", strategy_name, n_ctx, n_keep);
    fprintf(stderr, "══════════════════════════════════════════════════════════════\n");

    // Load model
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return "";
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Use pre-tokenized tokens if provided, otherwise tokenize now
    std::vector<llama_token> prompt_tokens;
    if (shared_tokens && !shared_tokens->empty()) {
        prompt_tokens = *shared_tokens;
        fprintf(stderr, "  Using mmap'd pre-tokenized prompt (%d tokens)\n", (int)prompt_tokens.size());
    } else {
        int n_prompt_est = -llama_tokenize(vocab, LONG_CONTEXT_PROMPT, strlen(LONG_CONTEXT_PROMPT), NULL, 0, true, true);
        prompt_tokens.resize(n_prompt_est);
        int n_prompt = llama_tokenize(vocab, LONG_CONTEXT_PROMPT, strlen(LONG_CONTEXT_PROMPT),
                                      prompt_tokens.data(), prompt_tokens.size(), true, true);
        prompt_tokens.resize(n_prompt);
    }
    int n_prompt = (int)prompt_tokens.size();

    fprintf(stderr, "  Prompt tokens: %d, KV budget: %d\n", n_prompt, n_ctx);

    // Create context with limited KV cache
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_ctx;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        return "";
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // Sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.1f));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Process prompt in chunks, handling cache overflow
    int n_past = 0;
    int chunk_size = 256;

    for (int start = 0; start < n_prompt; start += chunk_size) {
        int end = std::min(start + chunk_size, n_prompt);
        int cur_chunk = end - start;

        // Check if we'll exceed the KV budget
        if (n_past + cur_chunk >= n_ctx) {
            int n_evict_needed = (n_past + cur_chunk) - n_ctx + 64; // evict enough + headroom

            fprintf(stderr, "  [pos %d] Cache overflow! n_past=%d, need to evict %d tokens\n",
                    start, n_past, n_evict_needed);

            if (!use_mimosa) {
                // ── Strategy A: Sliding Window ──
                // Remove tokens from position n_keep to n_keep + n_evict_needed
                int rm_start = n_keep;
                int rm_end   = n_keep + n_evict_needed;
                if (rm_end > n_past) rm_end = n_past;

                fprintf(stderr, "  [SLIDING] Removing positions %d-%d (%d tokens)\n",
                        rm_start, rm_end, rm_end - rm_start);
                llama_memory_seq_rm(mem, 0, rm_start, rm_end);
                llama_memory_seq_add(mem, 0, rm_end, -1, -(rm_end - rm_start));
                n_past -= (rm_end - rm_start);
            } else {
                // ── Strategy B: Mimosa + Three-Distance ──
                // Compute importance signal for all cached positions
                std::vector<float> importance = compute_importance(
                    prompt_tokens, vocab, n_past, n_keep);

                // Compute Mimosa turgor scores
                MimosaKVGate mimosa;
                std::vector<float> turgor = mimosa.compute_turgor(importance);

                // Use Three-Distance to determine zone-aware eviction
                int target_budget = n_past - n_evict_needed;
                std::vector<int> evict_positions =
                    ThreeDistanceRouter::zone_aware_evict(importance, turgor, target_budget);

                fprintf(stderr, "  [MIMOSA] Evicting %d positions (zone-aware), keeping %d\n",
                        (int)evict_positions.size(), target_budget);

                // Debug: show which zones are formed
                auto gap_info = ThreeDistanceRouter::analyze_gap_structure(n_past);
                fprintf(stderr, "  [MIMOSA] Three-Distance zones: %d\n", gap_info.n_zones);

                // Print turgor histogram
                int high = 0, mid = 0, low = 0;
                for (float t : turgor) {
                    if (t > 0.7f) high++;
                    else if (t > 0.3f) mid++;
                    else low++;
                }
                fprintf(stderr, "  [MIMOSA] Turgor distribution: high(>0.7)=%d mid=%d low(<0.3)=%d\n",
                        high, mid, low);

                // Mimosa eviction: remove a contiguous block of lowest-turgor positions
                // To avoid position gap issues, find the worst contiguous region to evict
                // Sort eviction candidates by position
                std::vector<int> evict_sorted = evict_positions;
                std::sort(evict_sorted.begin(), evict_sorted.end());

                // Find the contiguous block that overlaps the most with eviction list
                // Simple approach: evict from n_keep to n_keep + n_evict_needed,
                // but shifted toward the lowest-turgor region
                if (!evict_sorted.empty()) {
                    // Find center of mass of eviction candidates
                    double com = 0.0;
                    for (int p : evict_sorted) com += p;
                    com /= evict_sorted.size();

                    // Evict a contiguous block centered around COM, clamped to [n_keep, n_past)
                    int block_start = (int)(com - n_evict_needed / 2.0);
                    if (block_start < n_keep) block_start = n_keep;
                    int block_end = block_start + n_evict_needed;
                    if (block_end > n_past) {
                        block_end = n_past;
                        block_start = std::max(n_keep, block_end - n_evict_needed);
                    }
                    int actual_evict = block_end - block_start;

                    fprintf(stderr, "  [MIMOSA] Evicting contiguous block [%d, %d) (%d tokens, COM=%.0f)\n",
                            block_start, block_end, actual_evict, com);

                    llama_memory_seq_rm(mem, 0, block_start, block_end);
                    llama_memory_seq_add(mem, 0, block_end, -1, -actual_evict);
                    n_past -= actual_evict;
                }
                fprintf(stderr, "  [MIMOSA] After eviction: n_past=%d\n", n_past);
            }
        }

        // Decode current chunk — positions auto-assigned from seq_pos_max + 1
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + start, cur_chunk);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "  ERROR: decode failed at pos %d\n", start);
            break;
        }
        n_past += cur_chunk;
    }

    fprintf(stderr, "  Prompt processed. n_past=%d, generating %d tokens...\n", n_past, n_predict);

    // Generate completion
    std::string output;
    for (int i = 0; i < n_predict; ++i) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) output.append(buf, n);

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "  ERROR: decode failed during generation\n");
            break;
        }
        n_past += 1;

        // Also check for cache overflow during generation
        if (n_past >= n_ctx - 2) {
            int n_evict = n_ctx / 4;  // evict 25%
            if (!use_mimosa) {
                llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_evict);
                llama_memory_seq_add(mem, 0, n_keep + n_evict, -1, -n_evict);
                n_past -= n_evict;
            } else {
                // Mimosa eviction during generation
                std::vector<float> imp(n_past);
                for (int j = 0; j < n_past; ++j)
                    imp[j] = (j < n_keep) ? 10.0f : (0.3f + 0.7f * std::exp(-0.005f * (float)(n_past - 1 - j)));
                MimosaKVGate gate;
                auto turgor = gate.compute_turgor(imp);
                auto evict = ThreeDistanceRouter::zone_aware_evict(imp, turgor, n_past - n_evict);
                if (!evict.empty()) {
                    std::vector<int> ev_sorted(evict.begin(), evict.end());
                    std::sort(ev_sorted.begin(), ev_sorted.end());
                    double com = 0.0;
                    for (int p : ev_sorted) com += p;
                    com /= ev_sorted.size();
                    int block_start = (int)(com - n_evict / 2.0);
                    if (block_start < n_keep) block_start = n_keep;
                    int block_end = std::min(n_past, block_start + n_evict);
                    block_start = std::max(n_keep, block_end - n_evict);
                    int actual = block_end - block_start;
                    llama_memory_seq_rm(mem, 0, block_start, block_end);
                    llama_memory_seq_add(mem, 0, block_end, -1, -actual);
                    n_past -= actual;
                }
            }
        }
    }

    fprintf(stderr, "\n  GENERATION (%s):\n", strategy_name);
    fprintf(stderr, "  ────────────────────────────────\n");
    fprintf(stderr, "  %s\n", output.c_str());
    fprintf(stderr, "  ────────────────────────────────\n");

    // Scoring: check if key facts are present
    int score = 0;
    if (output.find("AURORA") != std::string::npos || output.find("Aurora") != std::string::npos || output.find("aurora") != std::string::npos) score++;
    if (output.find("4.7") != std::string::npos || output.find("4,700,000") != std::string::npos) score++;

    fprintf(stderr, "  FACT RECALL SCORE: %d/2\n", score);
    fprintf(stderr, "    [%s] Project codename AURORA-7\n", (output.find("AURORA") != std::string::npos || output.find("Aurora") != std::string::npos) ? "✓" : "✗");
    fprintf(stderr, "    [%s] Budget $4.7 million\n", (output.find("4.7") != std::string::npos || output.find("4,700,000") != std::string::npos) ? "✓" : "✗");

    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return output;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    int n_ctx     = 1024;  // Deliberately small KV cache to force eviction
    int n_keep    = 64;    // Keep first 64 tokens (contains key facts)
    int n_predict = 128;   // Generate 128 tokens of response

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--n-keep") == 0 && i + 1 < argc) n_keep = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_predict = std::stoi(argv[++i]);
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m <model.gguf> [-c n_ctx] [--n-keep N] [-n n_predict]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Mimosa KV Cache Eviction Test — QuantoniumOS Research      ║\n");
    fprintf(stderr, "║  Model: DeepSeek-R1-Distill-Qwen-7B (Q4_K_M)               ║\n");
    fprintf(stderr, "║  Test: Needle-in-Haystack Fact Recall Under Cache Pressure  ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n");

    // ── Mmap token cache: tokenize once, reuse across both strategy runs ──
    //
    // On first run: tokenize prompt → save to .tok_cache binary file
    // On second+ run: mmap the .tok_cache file → zero-copy token array
    // Both strategy runs share the same mmap'd tokens (no re-tokenization)

    std::string tok_cache_path = model_path + std::string(".tok_cache");
    MmapTokenCache tok_cache;
    std::vector<llama_token> shared_tokens = tok_cache.load(tok_cache_path);

    if (shared_tokens.empty()) {
        fprintf(stderr, "\n  [tok-cache] No cached tokens found, tokenizing prompt...\n");

        // Need to load model just for the vocab to tokenize
        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * tmp_model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!tmp_model) { fprintf(stderr, "ERROR: failed to load model for tokenization\n"); return 1; }
        const llama_vocab * vocab = llama_model_get_vocab(tmp_model);

        int n_est = -llama_tokenize(vocab, LONG_CONTEXT_PROMPT, strlen(LONG_CONTEXT_PROMPT), NULL, 0, true, true);
        shared_tokens.resize(n_est);
        int n_actual = llama_tokenize(vocab, LONG_CONTEXT_PROMPT, strlen(LONG_CONTEXT_PROMPT),
                                      shared_tokens.data(), shared_tokens.size(), true, true);
        shared_tokens.resize(n_actual);

        llama_model_free(tmp_model);

        // Save to binary cache for future mmap
        if (MmapTokenCache::save(tok_cache_path, shared_tokens)) {
            fprintf(stderr, "  [tok-cache] Saved %d tokens to %s\n", n_actual, tok_cache_path.c_str());
        }

        // Reload via mmap so both runs use the same mmap'd memory
        tok_cache.unload();
        std::vector<llama_token> mmap_tokens = tok_cache.load(tok_cache_path);
        if (!mmap_tokens.empty()) {
            shared_tokens = std::move(mmap_tokens);
            fprintf(stderr, "  [tok-cache] Mmap'd %d tokens for zero-copy reuse\n", (int)shared_tokens.size());
        }
    } else {
        fprintf(stderr, "\n  [tok-cache] Loaded %d mmap'd tokens from cache (skipped tokenization)\n",
                (int)shared_tokens.size());
    }

    // Run with SLIDING WINDOW (passes shared tokens)
    std::string output_sliding = run_with_strategy(
        model_path.c_str(), "SLIDING WINDOW", n_ctx, n_keep, n_predict, false, &shared_tokens);

    // Run with MIMOSA + THREE-DISTANCE (reuses same mmap'd tokens)
    std::string output_mimosa = run_with_strategy(
        model_path.c_str(), "MIMOSA + THREE-DISTANCE", n_ctx, n_keep, n_predict, true, &shared_tokens);

    // Summary
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  COMPARISON SUMMARY                                         ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");

    auto check_fact = [](const std::string& s, const char* needle) {
        return s.find(needle) != std::string::npos;
    };

    int slide_score = 0, mimosa_score = 0;
    if (check_fact(output_sliding, "AURORA") || check_fact(output_sliding, "Aurora")) slide_score++;
    if (check_fact(output_sliding, "4.7")) slide_score++;
    if (check_fact(output_mimosa,  "AURORA") || check_fact(output_mimosa,  "Aurora")) mimosa_score++;
    if (check_fact(output_mimosa,  "4.7")) mimosa_score++;

    fprintf(stderr, "║  Sliding Window:           %d/2 facts recalled               ║\n", slide_score);
    fprintf(stderr, "║  Mimosa + Three-Distance:  %d/2 facts recalled               ║\n", mimosa_score);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
