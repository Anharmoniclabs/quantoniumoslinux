/*
 * test_spectral_roundtrip.cpp - Spectral encode/decode correctness diagnostic
 *
 * Tests the RFTMW spectral pipeline in isolation to find the bug causing
 * catastrophic perplexity (40K+ at 40% keep, vs 15.2 baseline).
 *
 * Test 1: FFT -> IFFT roundtrip (should be ~1e-7)
 * Test 2: Forward transform -> manual inverse (should be ~1e-7)
 * Test 3: Full quantized GEMV at keep=1.0 vs dense matmul
 * Test 4: GEMV keep sweep to map error vs sparsification
 * Test 5: Single-coefficient reconstruction sanity check
 * Test 6: Non-uniform sparsity schedule check
 * Test 7: Real GGUF weight matrix test (if --gguf provided)
 */

#include "rftmw_spectral_kernels.h"
#include "rftmw_inference.hpp"
#include "rftmw_spectral_inference.hpp"
#include "rftmw_mmap.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

/* ── Minimal GGUF parser (just enough to load one tensor) ──── */

namespace {

constexpr int GGUF_F32  = 0;
constexpr int GGUF_F16  = 1;
constexpr int GGUF_Q4_K = 12;
constexpr int GGUF_Q6_K = 14;

uint32_t read_u32(const uint8_t*& p) { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
uint64_t read_u64(const uint8_t*& p) { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }

std::string read_string(const uint8_t*& p) {
    uint64_t n = read_u64(p);
    std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
    p += n;
    return s;
}

void skip_gguf_value(const uint8_t*& p, uint32_t type);
void skip_gguf_scalar(const uint8_t*& p, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: p += 1; break;
        case 2: case 3: p += 2; break;
        case 4: case 5: case 6: p += 4; break;
        case 8: { uint64_t n = read_u64(p); p += n; break; }
        case 10: case 11: case 12: p += 8; break;
        default: throw std::runtime_error("unsupported GGUF type");
    }
}
void skip_gguf_value(const uint8_t*& p, uint32_t type) {
    if (type == 9) {
        uint32_t et = read_u32(p); uint64_t n = read_u64(p);
        for (uint64_t i = 0; i < n; ++i) skip_gguf_value(p, et);
    } else { skip_gguf_scalar(p, type); }
}

uint64_t gguf_tensor_nbytes(uint32_t type, const std::vector<uint64_t>& ne) {
    uint64_t n = 1; for (auto d : ne) n *= d;
    if (type == GGUF_F32)  return n * 4;
    if (type == GGUF_F16)  return n * 2;
    if (type == GGUF_Q4_K) return (n / 256) * 144;
    if (type == GGUF_Q6_K) return (n / 256) * 210;
    throw std::runtime_error("unsupported tensor type");
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t total_bytes = 0;
    const uint8_t* data = nullptr;
};

struct MiniGGUF {
    rftmw::mmap::MmapFile file;
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, int64_t> meta_int;

    void load(const char* path) {
        if (!file.open(path))
            throw std::runtime_error(std::string("failed to mmap: ") + path);
        const uint8_t* base = static_cast<const uint8_t*>(file.data());
        const uint8_t* p = base;
        if (std::memcmp(p, "GGUF", 4) != 0) throw std::runtime_error("bad magic");
        p += 4;
        read_u32(p); // version
        uint64_t n_tensors = read_u64(p);
        uint64_t n_kv = read_u64(p);
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string(p);
            uint32_t type = read_u32(p);
            if (type == 4) { uint32_t v; std::memcpy(&v, p, 4); p += 4; meta_int[key] = v; }
            else if (type == 5) { int32_t v; std::memcpy(&v, p, 4); p += 4; meta_int[key] = v; }
            else if (type == 10) { uint64_t v; std::memcpy(&v, p, 8); p += 8; meta_int[key] = (int64_t)v; }
            else { skip_gguf_value(p, type); }
        }
        std::vector<TensorInfo> tmp;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = read_string(p);
            uint32_t nd = read_u32(p);
            for (uint32_t d = 0; d < nd; d++) t.ne.push_back(read_u64(p));
            t.type = read_u32(p);
            t.offset = read_u64(p);
            t.total_bytes = gguf_tensor_nbytes(t.type, t.ne);
            tmp.push_back(std::move(t));
        }
        uint64_t pos = static_cast<uint64_t>(p - base);
        uint64_t data_start = (pos + 31u) & ~31u;
        for (auto& t : tmp) {
            t.data = base + data_start + t.offset;
            tensors.emplace(t.name, std::move(t));
        }
    }

    int64_t get_int(const std::string& suffix) const {
        for (auto& [k, v] : meta_int)
            if (k.size() >= suffix.size() && k.substr(k.size() - suffix.size()) == suffix)
                return v;
        return -1;
    }
};

} // anonymous namespace

/* ── Helpers ─────────────────────────────────────────────────── */

static double vec_norm(const float* v, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (double)v[i] * (double)v[i];
    return std::sqrt(sum);
}

static double relative_error(const float* ref, const float* test, int n) {
    double err_sq = 0, norm_sq = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)ref[i] - (double)test[i];
        err_sq += d * d;
        norm_sq += (double)ref[i] * (double)ref[i];
    }
    if (norm_sq < 1e-30) return 0.0;
    return std::sqrt(err_sq / norm_sq);
}

static double max_abs_error(const float* ref, const float* test, int n) {
    double mx = 0;
    for (int i = 0; i < n; i++) {
        double d = std::fabs((double)ref[i] - (double)test[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* Dense matmul: y[rows] = W[rows x cols] @ x[cols] */
static void dense_matmul(const float* W, const float* x, float* y,
                         int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        double sum = 0;
        for (int c = 0; c < cols; c++)
            sum += (double)W[(size_t)r * cols + c] * (double)x[c];
        y[r] = (float)sum;
    }
}

/* ── Test 1: FFT roundtrip ───────────────────────────────────── */

static bool test_fft_roundtrip() {
    printf("Test 1: FFT -> IFFT roundtrip\n");
    const int N = 256;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float re[N], im[N], re_orig[N];
    for (int i = 0; i < N; i++) {
        re[i] = dist(rng);
        im[i] = 0.0f;
        re_orig[i] = re[i];
    }

    rftmw_fft_f32(re, im, N);
    rftmw_ifft_f32(re, im, N);

    double err = relative_error(re_orig, re, N);
    double max_err = max_abs_error(re_orig, re, N);
    printf("  relative error: %.2e\n", err);
    printf("  max abs error:  %.2e\n", max_err);

    bool pass = err < 1e-5;
    printf("  %s\n\n", pass ? "PASS" : "*** FAIL ***");
    return pass;
}

/* ── Test 2: Forward transform roundtrip ──────────────────────
 *
 * Forward: S = diag(E) * FFT(x) / sqrt(N)
 * Inverse: x_hat = sqrt(N) * IFFT(diag(conj(E)) * S)
 *
 * Since |E[k]| = 1 (unit circle), conj(E)*E = 1, so this
 * should recover x exactly (up to float precision).
 * ────────────────────────────────────────────────────────────── */

static bool test_forward_roundtrip() {
    printf("Test 2: Forward transform -> manual inverse\n");
    const int N = 256;
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    rftmw_spectral_basis_t basis;
    rftmw_spectral_basis_init(&basis, N);

    float x[N], spec_re[N], spec_im[N];
    for (int i = 0; i < N; i++) x[i] = dist(rng);

    // Forward
    rftmw_spectral_forward_block(&basis, x, spec_re, spec_im);

    // Manual inverse: multiply by conj(E), then IFFT, then * sqrt(N)
    float inv_re[N], inv_im[N];
    for (int k = 0; k < N; k++) {
        float e_re = basis.cos_table[k];
        float e_im = basis.sin_table[k];
        // conj(E[k]) = (e_re, -e_im)
        // S * conj(E) = (s_re*e_re + s_im*e_im) + i*(-s_re*e_im + s_im*e_re)
        inv_re[k] = spec_re[k] * e_re + spec_im[k] * e_im;
        inv_im[k] = -spec_re[k] * e_im + spec_im[k] * e_re;
    }

    rftmw_ifft_f32(inv_re, inv_im, N);

    float sqrt_n = std::sqrt((float)N);
    float x_hat[N];
    for (int i = 0; i < N; i++) x_hat[i] = inv_re[i] * sqrt_n;

    double err = relative_error(x, x_hat, N);
    double max_err = max_abs_error(x, x_hat, N);
    printf("  relative error: %.2e\n", err);
    printf("  max abs error:  %.2e\n", max_err);

    bool pass = err < 1e-5;
    printf("  %s\n\n", pass ? "PASS" : "*** FAIL ***");

    rftmw_spectral_basis_free(&basis);
    return pass;
}

/* ── Test 3: Full quantized GEMV at keep=1.0 ─────────────────
 *
 * The critical test. At keep=1.0, every spectral coefficient is
 * retained. The only error source should be:
 *   - 4-bit magnitude quantization (~6% per coefficient)
 *   - 8-bit phase quantization (~0.012 radians)
 *
 * We build a small dense matrix, compress it, and compare GEMV
 * output with dense matmul. Expected relative error: ~5-15%.
 * ────────────────────────────────────────────────────────────── */

static bool test_gemv_keep_1() {
    printf("Test 3: Full quantized GEMV at keep=1.0 (64x256)\n");
    const int rows = 64, cols = 256;
    std::mt19937 rng(999);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<float> W(rows * cols);
    for (auto& v : W) v = dist(rng);
    std::vector<float> x(cols);
    for (auto& v : x) v = dist(rng);

    // Dense reference
    std::vector<float> y_dense(rows);
    dense_matmul(W.data(), x.data(), y_dense.data(), rows, cols);

    // Spectral at keep=1.0
    rftmw_spectral_weight_t sw;
    rftmw_spectral_weight_init(&sw, W.data(), rows, cols, 256, 1.0f);
    std::vector<float> y_spec(rows);
    rftmw_spectral_gemv(&sw, x.data(), y_spec.data());

    double err = relative_error(y_dense.data(), y_spec.data(), rows);
    double max_err = max_abs_error(y_dense.data(), y_spec.data(), rows);
    double y_norm = vec_norm(y_dense.data(), rows);

    printf("  relative error: %.4e\n", err);
    printf("  max abs error:  %.4e  (||y||=%.4e)\n", max_err, y_norm);

    // Print first 5 values for inspection
    printf("  first 5 values:\n");
    for (int i = 0; i < 5 && i < rows; i++) {
        printf("    [%d] dense=%.6f  spectral=%.6f  diff=%.2e\n",
               i, y_dense[i], y_spec[i],
               std::fabs(y_dense[i] - y_spec[i]));
    }

    bool pass = err < 0.30;  // generous threshold — we want to see the number
    printf("  %s (expect ~0.05-0.15 from 4-bit quant; >0.30 = bug)\n\n",
           pass ? "PASS" : "*** FAIL ***");

    rftmw_spectral_weight_free(&sw);
    return pass;
}

/* ── Test 4: GEMV keep-ratio sweep ───────────────────────────── */

static void test_gemv_sweep() {
    printf("Test 4: GEMV keep sweep (64x256 synthetic)\n");
    const int rows = 64, cols = 256;
    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<float> W(rows * cols);
    for (auto& v : W) v = dist(rng);
    std::vector<float> x(cols);
    for (auto& v : x) v = dist(rng);

    std::vector<float> y_dense(rows);
    dense_matmul(W.data(), x.data(), y_dense.data(), rows, cols);

    float keeps[] = {1.0f, 0.80f, 0.60f, 0.40f, 0.30f, 0.20f, 0.10f, 0.05f};
    for (float k : keeps) {
        rftmw_spectral_weight_t sw;
        rftmw_spectral_weight_init(&sw, W.data(), rows, cols, 256, k);
        std::vector<float> y_spec(rows);
        rftmw_spectral_gemv(&sw, x.data(), y_spec.data());
        double err = relative_error(y_dense.data(), y_spec.data(), rows);
        printf("  keep=%.2f  relative_error=%.4e\n", k, err);
        rftmw_spectral_weight_free(&sw);
    }
    printf("\n");
}

/* ── Test 5: Single-coefficient reconstruction ────────────────
 *
 * Create a weight row that is a single spectral coefficient
 * (a pure cosine at a known frequency). Compress at keep=1.0.
 * The spectral output should have a single dominant peak.
 * Tests for index misplacement bugs.
 * ────────────────────────────────────────────────────────────── */

static bool test_single_coefficient() {
    printf("Test 5: Single-coefficient reconstruction check\n");
    const int N = 256;

    rftmw_spectral_basis_t basis;
    rftmw_spectral_basis_init(&basis, N);
    rftmw_phase_lut_t phase_lut;
    rftmw_phase_lut_init(&phase_lut);

    // Create a pure cosine signal at bin 17
    float signal[N], spec_re[N], spec_im[N];
    for (int i = 0; i < N; i++)
        signal[i] = std::cos(2.0f * 3.14159265f * 17.0f * i / N);

    // Forward transform to see where the energy lands
    rftmw_spectral_forward_block(&basis, signal, spec_re, spec_im);

    // Find the bin with max magnitude
    int max_bin = 0;
    float max_mag = 0;
    for (int k = 0; k < N; k++) {
        float mag = std::sqrt(spec_re[k] * spec_re[k] + spec_im[k] * spec_im[k]);
        if (mag > max_mag) { max_mag = mag; max_bin = k; }
    }

    // Compress this signal as a single row, keep=1.0
    rftmw_qsparse_row_t row;
    rftmw_spectral_compress_row(&basis, &phase_lut, signal, N, 1.0f, &row);

    // Report
    printf("  pure cosine at DFT bin 17:\n");
    printf("    max spectral energy at hybrid-RFT bin %d (mag=%.4f)\n",
           max_bin, max_mag);
    printf("    compressed block 0: n_nonzero=%d\n", row.blocks[0].n_nonzero);

    // Now test GEMV with identity-like input (e_0 = [1,0,0,...])
    // y = W @ e_0 should equal signal[0] = cos(0) = 1.0
    // But more usefully, test with signal itself: dot(signal, signal) = N/2
    rftmw_spectral_weight_t sw;
    rftmw_spectral_weight_init(&sw, signal, 1, N, N, 1.0f);
    float x[N], y[1];
    std::memcpy(x, signal, N * sizeof(float));
    rftmw_spectral_gemv(&sw, x, y);

    double expected = 0;
    for (int i = 0; i < N; i++) expected += (double)signal[i] * signal[i];
    double err = std::fabs(y[0] - (float)expected) / std::fabs(expected);

    printf("    dot(signal, signal) dense=%.4f  spectral=%.4f  rel_err=%.4e\n",
           expected, (double)y[0], err);

    bool pass = err < 0.15;
    printf("  %s\n\n", pass ? "PASS" : "*** FAIL ***");

    rftmw_spectral_weight_free(&sw);
    rftmw_qsparse_row_free(&row);
    rftmw_spectral_basis_free(&basis);
    return pass;
}

/* ── Test 6: Non-uniform sparsity schedule check ─────────────
 *
 * Verify that keep ratio schedule matches expectations.
 * ────────────────────────────────────────────────────────────── */

static void test_sparsity_schedule() {
    printf("Test 6: Non-uniform sparsity schedule (28 layers, base=0.35)\n");
    int total = 28;
    float base = 0.35f;
    for (int i = 0; i < total; i++) {
        float k = rftmw_layer_keep_ratio(i, total, base);
        const char* zone = (i < total/6) ? "early" :
                           (i >= total - total/6) ? "late" : "core";
        printf("  layer %2d  keep=%.3f  (%s)\n", i, k, zone);
    }
    printf("\n");
}

/* ── Test 7: GGUF real weight test ────────────────────────────
 *
 * Loads one FFN weight matrix from the actual model, dequantizes
 * it, compresses at keep=1.0, and tests GEMV.
 * ────────────────────────────────────────────────────────────── */

static bool test_gguf_weight(const char* gguf_path) {
    printf("Test 7: Real GGUF weight roundtrip (layer 14 gate_proj)\n");

    MiniGGUF gguf;
    gguf.load(gguf_path);

    int n_layers = (int)gguf.get_int("block_count");
    int hidden   = (int)gguf.get_int("embedding_length");
    int ffn_dim  = (int)gguf.get_int("feed_forward_length");
    printf("  model: %d layers, dim=%d, ffn=%d\n", n_layers, hidden, ffn_dim);

    int layer_idx = std::min(14, n_layers - 1);
    std::string tname = "blk." + std::to_string(layer_idx) + ".ffn_gate.weight";
    auto it = gguf.tensors.find(tname);
    if (it == gguf.tensors.end()) {
        printf("  tensor %s not found, SKIP\n\n", tname.c_str());
        return true;
    }
    auto& ti = it->second;

    printf("  tensor: %s\n", tname.c_str());
    printf("  shape: [%d, %d]  type=%s  bytes_per_row=%d\n",
           ffn_dim, hidden,
           ti.type == 12 ? "q4_k" : (ti.type == 14 ? "q6_k" : "unknown"),
           (int)(ti.total_bytes / ffn_dim));

    const char* qtype = (ti.type == 12) ? "q4k" : "q6k";
    int bytes_per_row = (int)(ti.total_bytes / ffn_dim);

    // Dequant just 64 rows for speed
    int test_rows = 64;
    std::vector<float> W_dense((size_t)test_rows * hidden);
    for (int r = 0; r < test_rows; r++) {
        const uint8_t* row_data = ti.data + (size_t)r * bytes_per_row;
        float* out = W_dense.data() + (size_t)r * hidden;
        int n_blocks_q = hidden / 256;
        if (ti.type == 12) {
            for (int b = 0; b < n_blocks_q; b++)
                rftmw::inference::dequant_q4k_block(row_data + b * 144, out + b * 256);
        } else {
            for (int b = 0; b < n_blocks_q; b++)
                rftmw::inference::dequant_q6k_block(row_data + b * 210, out + b * 256);
        }
    }

    // Weight statistics
    {
        float wmin = W_dense[0], wmax = W_dense[0];
        double wsum = 0, wsum2 = 0;
        for (size_t i = 0; i < W_dense.size(); i++) {
            float v = W_dense[i];
            if (v < wmin) wmin = v;
            if (v > wmax) wmax = v;
            wsum += v;
            wsum2 += (double)v * v;
        }
        int n = (int)W_dense.size();
        printf("  weight stats: min=%.4f max=%.4f mean=%.6f std=%.6f\n",
               wmin, wmax, wsum/n, std::sqrt(wsum2/n - (wsum/n)*(wsum/n)));
    }

    // Generate test input
    std::mt19937 rng(314);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> x(hidden);
    for (auto& v : x) v = dist(rng);

    // Dense matmul
    std::vector<float> y_dense(test_rows);
    dense_matmul(W_dense.data(), x.data(), y_dense.data(), test_rows, hidden);

    // Test at multiple keep ratios
    float keeps[] = {1.0f, 0.50f, 0.35f, 0.20f, 0.10f};
    printf("  GEMV comparison (64 rows x %d cols):\n", hidden);
    bool overall_pass = true;

    for (float k : keeps) {
        rftmw_spectral_weight_t sw;
        rftmw_spectral_weight_init(&sw, W_dense.data(), test_rows, hidden, 256, k);
        std::vector<float> y_spec(test_rows);
        rftmw_spectral_gemv(&sw, x.data(), y_spec.data());

        double err = relative_error(y_dense.data(), y_spec.data(), test_rows);
        double max_e = max_abs_error(y_dense.data(), y_spec.data(), test_rows);
        double y_norm = vec_norm(y_dense.data(), test_rows);

        const char* verdict;
        if (k >= 0.9f && err > 0.30) { verdict = "*** FAIL ***"; overall_pass = false; }
        else if (k >= 0.3f && err > 1.0) { verdict = "*** FAIL ***"; overall_pass = false; }
        else verdict = "ok";

        printf("    keep=%.2f  rel_err=%.4e  max_abs=%.4e  ||y||=%.4e  %s\n",
               k, err, max_e, y_norm, verdict);
        rftmw_spectral_weight_free(&sw);
    }

    // Detailed dump for keep=1.0
    {
        printf("  First 10 values at keep=1.0:\n");
        rftmw_spectral_weight_t sw;
        rftmw_spectral_weight_init(&sw, W_dense.data(), test_rows, hidden, 256, 1.0f);
        std::vector<float> y_spec(test_rows);
        rftmw_spectral_gemv(&sw, x.data(), y_spec.data());
        for (int i = 0; i < 10 && i < test_rows; i++) {
            printf("    [%2d] dense=%+.6f  spectral=%+.6f  diff=%+.2e\n",
                   i, y_dense[i], y_spec[i],
                   (double)y_spec[i] - (double)y_dense[i]);
        }
        rftmw_spectral_weight_free(&sw);
    }

    printf("  %s\n\n", overall_pass ? "PASS" : "*** FAIL - BUG CONFIRMED ***");
    return overall_pass;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    printf("============================================\n");
    printf("  RFTMW Spectral Roundtrip Diagnostic\n");
    printf("============================================\n\n");

    int pass = 0, fail = 0;

    // Core transform tests (no GGUF needed)
    test_fft_roundtrip()       ? pass++ : fail++;
    test_forward_roundtrip()   ? pass++ : fail++;
    test_gemv_keep_1()         ? pass++ : fail++;
    test_gemv_sweep();
    test_single_coefficient()  ? pass++ : fail++;
    test_sparsity_schedule();

    // Optional: real GGUF weight test
    const char* gguf_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--gguf") == 0 && i + 1 < argc)
            gguf_path = argv[++i];
    }

    if (gguf_path) {
        test_gguf_weight(gguf_path) ? pass++ : fail++;
    } else {
        printf("(Skipping GGUF test — pass --gguf <path> to enable)\n\n");
    }

    printf("============================================\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("============================================\n");
    return fail > 0 ? 1 : 0;
}
