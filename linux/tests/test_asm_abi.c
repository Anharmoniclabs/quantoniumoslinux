/* Cross-ABI check for the RFTMW NASM kernels: runs on win64 and System V.
 * Each kernel is compared to a scalar reference; callee-saved registers the
 * caller relies on are exercised by keeping live values across the calls. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rftmw_dot_f32_asm(const float* w, const float* x, float* out, int cols);
float rftmw_spectral_sparse_dot_asm(const uint16_t* idx, const float* cre, const float* cim,
                                    const float* xre, const float* xim, int n);

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

int main(void) {
    int failures = 0;
    srand(1234);
    enum { N = 1037, S = 4096 };
    static float w[N], x[N], xre[S], xim[S], cre[N], cim[N];
    static uint16_t idx[N];
    for (int i = 0; i < N; ++i) { w[i] = frand(); x[i] = frand(); cre[i] = frand(); cim[i] = frand(); idx[i] = (uint16_t)(rand() % S); }
    for (int i = 0; i < S; ++i) { xre[i] = frand(); xim[i] = frand(); }

    int lens[] = {0, 1, 7, 8, 9, 64, 100, N};
    for (unsigned t = 0; t < sizeof lens / sizeof lens[0]; ++t) {
        int n = lens[t];
        volatile double keep = 3.25 * (t + 1); /* lives across the call */
        double ref = 0; for (int i = 0; i < n; ++i) ref += (double)w[i] * x[i];
        float got = -1.0f; rftmw_dot_f32_asm(w, x, &got, n);
        double sref = 0; for (int i = 0; i < n; ++i) sref += (double)cre[i] * xre[idx[i]] + (double)cim[i] * xim[idx[i]];
        float sgot = rftmw_spectral_sparse_dot_asm(idx, cre, cim, xre, xim, n);
        int ok = fabs(got - ref) < 1e-3 && fabs(sgot - sref) < 1e-3 && keep == 3.25 * (t + 1);
        printf("n=%4d  dot %+.6f/%+.6f  spectral %+.6f/%+.6f  %s\n", n, got, ref, sgot, sref, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    printf(failures ? "FAILED (%d)\n" : "all ABI checks passed\n", failures);
    return failures != 0;
}
