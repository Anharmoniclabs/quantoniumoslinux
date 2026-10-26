// Benchmark + equivalence harness for the RFTMW compute kernels.
//
// Built twice by run_bench.sh: once against linux/bench/baseline (the original
// sources) and once against runtime/src/rftmw_native (current). Each run prints
// timings and writes every result vector to OUT so the two builds can be diffed.
//
//   bench_rftmw OUT [quick]
#include "rftmw_core.hpp"
#include "rftmw_inference.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

extern "C" void rftmw_dot_f32_asm(const float* w, const float* x, float* out, int cols);
extern "C" float rftmw_spectral_sparse_dot_asm(const uint16_t* idx, const float* cre, const float* cim,
                                               const float* xre, const float* xim, int n);

namespace {
FILE* g_out = nullptr;
bool g_quick = false;

template <class F>
double best_seconds(F&& f, double budget = 0.4, int min_reps = 3) {
    using clk = std::chrono::steady_clock;
    double best = 1e30, spent = 0;
    for (int rep = 0; rep < min_reps || spent < budget; ++rep) {
        auto t0 = clk::now();
        f();
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        best = std::min(best, dt);
        spent += dt;
        if (rep > 1000) break;
    }
    return best;
}

void dump(const char* tag, const double* v, size_t n) {
    uint64_t len = n;
    char name[32] = {};
    std::snprintf(name, sizeof name, "%s", tag);
    std::fwrite(name, 1, sizeof name, g_out);
    std::fwrite(&len, sizeof len, 1, g_out);
    std::fwrite(v, sizeof(double), n, g_out);
}

std::vector<double> randvec(size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> d;
    std::vector<double> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

std::vector<float> randvecf(size_t n, unsigned seed) {
    auto d = randvec(n, seed);
    return std::vector<float>(d.begin(), d.end());
}

void bench_core() {
    rftmw::RFTMWEngine eng;
    std::vector<size_t> sizes = g_quick ? std::vector<size_t>{64, 1024} : std::vector<size_t>{32, 64, 128, 256, 1024, 4096};
    for (size_t n : sizes) {
        auto x = randvec(n, 1000 + n);
        rftmw::ComplexVec X;
        rftmw::RealVec y;
        // First call separately: shows the cold (basis build) cost.
        auto t0 = std::chrono::steady_clock::now();
        X = eng.forward(x);
        double cold = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double tf = best_seconds([&] { X = eng.forward(x); }, n > 1024 ? 1.0 : 0.4, 2);
        double ti = best_seconds([&] { y = eng.inverse(X); }, n > 1024 ? 1.0 : 0.4, 2);
        double err = 0;
        if (n <= RFTMW_CANONICAL_MAX_N)
            for (size_t i = 0; i < n; ++i) err = std::max(err, std::abs(y[i] - x[i]));
        std::printf("rft     N=%-5zu cold %10.3f ms  forward %10.3f ms  inverse %10.3f ms  roundtrip_err %.2e\n",
                    n, cold * 1e3, tf * 1e3, ti * 1e3, err);
        std::string tag = "rft_fwd_" + std::to_string(n);
        dump(tag.c_str(), reinterpret_cast<const double*>(X.data()), 2 * X.size());
        tag = "rft_inv_" + std::to_string(n);
        dump(tag.c_str(), y.data(), y.size());
    }
}

void bench_sgemv() {
    struct Shape { int rows, cols; };
    std::vector<Shape> shapes = g_quick ? std::vector<Shape>{{896, 896}, {2048, 2048}}
                                        : std::vector<Shape>{{896, 896}, {2048, 2048}, {11008, 2048}, {2048, 11008}};
    for (auto s : shapes) {
        auto w = randvecf(size_t(s.rows) * s.cols, 7);
        auto x = randvecf(s.cols, 8);
        std::vector<float> y(s.rows);
        double t = best_seconds([&] { rftmw::inference::dense_sgemv_f32(w.data(), x.data(), y.data(), s.rows, s.cols); });
        double bytes = double(s.rows) * s.cols * 4;
        std::printf("sgemv   %5dx%-5d  %9.3f ms  %6.1f GB/s  %6.1f GFLOP/s\n",
                    s.rows, s.cols, t * 1e3, bytes / t / 1e9, 2.0 * s.rows * s.cols / t / 1e9);
        std::vector<double> yd(y.begin(), y.end());
        std::string tag = "sgemv_" + std::to_string(s.rows) + "x" + std::to_string(s.cols);
        dump(tag.c_str(), yd.data(), yd.size());
    }
}

void bench_asm() {
    // L1/L2-resident rows: measures the kernel itself, not DRAM.
    for (int cols : {256, 2048, 2051}) {
        auto w = randvecf(cols, 11), x = randvecf(cols, 12);
        float out = 0;
        const int reps = 20000;
        double t = best_seconds([&] {
            for (int r = 0; r < reps; ++r) rftmw_dot_f32_asm(w.data(), x.data(), &out, cols);
        });
        std::printf("asm dot cols=%-5d  %8.1f ns/call  %6.2f flop/cycle@3GHz\n",
                    cols, t / reps * 1e9, 2.0 * cols / (t / reps) / 3e9);
        double od = out;
        std::string tag = "asmdot_" + std::to_string(cols);
        dump(tag.c_str(), &od, 1);
    }
    for (int nnz : {64, 256, 1027}) {
        const int spec = 4096;
        auto xre = randvecf(spec, 21), xim = randvecf(spec, 22);
        auto cre = randvecf(nnz, 23), cim = randvecf(nnz, 24);
        std::vector<uint16_t> idx(nnz);
        std::mt19937 rng(25);
        for (auto& i : idx) i = uint16_t(rng() % spec);
        float r = 0;
        const int reps = 20000;
        double t = best_seconds([&] {
            for (int k = 0; k < reps; ++k)
                r = rftmw_spectral_sparse_dot_asm(idx.data(), cre.data(), cim.data(), xre.data(), xim.data(), nnz);
        });
        std::printf("asm spectral nnz=%-5d %8.1f ns/call\n", nnz, t / reps * 1e9);
        double od = r;
        std::string tag = "asmspec_" + std::to_string(nnz);
        dump(tag.c_str(), &od, 1);
    }
}
}  // namespace

// Declared by rftmw_inference.hpp; the bench never calls it.
void rftmw::inference::dense_sgemv_f32_bridge(const float*, const float*, float*, int, int) {}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s OUT [quick]\n", argv[0]); return 2; }
    g_out = std::fopen(argv[1], "wb");
    if (!g_out) { std::perror(argv[1]); return 2; }
    g_quick = argc > 2 && std::string(argv[2]) == "quick";
    std::printf("threads: %d\n", rftmw::inference::get_num_threads());
    bench_asm();
    bench_sgemv();
    bench_core();
    std::fclose(g_out);
}
