#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "p48.h"

/* Test vectors */
static uint8_t v1_data[8] = {10, 20, 30, 40, 50, 60, 0, 0};
static uint8_t v2_data[8] = {0, 10, 20, 30, 40, 50, 60, 0};
static uint8_t v3_data[8] = {63, 63, 63, 63, 63, 63, 63, 63}; /* max */

/* Nearest-neighbor search over a batch */
static void bench_search(const char* label, uint64_t* vecs, size_t n, uint64_t query) {
    clock_t start = clock();
    int best_idx = -1;
    int best_dist = INT32_MAX;
    for (size_t i = 0; i < n; i++) {
        int d = p48_dist_sq(vecs[i], query);
        if (d < best_dist) {
            best_dist = d;
            best_idx = (int)i;
        }
    }
    clock_t end = clock();
    double ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  %s: best=%d dist=%d time=%.3fms (%.0f vec/s)\n",
           label, best_idx, best_dist, ms, n / (ms / 1000.0));
}

int main(void) {
    printf("Pythagorean48 Tests\n");
    printf("===================\n\n");

    /* Pack/unpack round-trip */
    uint64_t p1 = p48_pack(v1_data);
    uint8_t v1_check[8];
    p48_unpack(p1, v1_check);
    int ok = 1;
    for (int i = 0; i < 8; i++) {
        if (v1_check[i] != v1_data[i]) { ok = 0; break; }
    }
    printf("[%s] Pack/unpack round-trip\n", ok ? "PASS" : "FAIL");

    /* Dot product */
    uint64_t p2 = p48_pack(v2_data);
    int dot = p48_dot(p1, p2);
    int dot_expected = 10*0 + 20*10 + 30*20 + 40*30 + 50*40 + 60*50 + 0*60 + 0*0;
    printf("[%s] Dot product: %d == %d\n",
           dot == dot_expected ? "PASS" : "FAIL", dot, dot_expected);

    /* Norm */
    int n1 = p48_norm_sq(p1);
    int n1_expected = 10*10 + 20*20 + 30*30 + 40*40 + 50*50 + 60*60;
    printf("[%s] Norm squared: %d == %d\n",
           n1 == n1_expected ? "PASS" : "FAIL", n1, n1_expected);

    /* Max norm */
    uint64_t p_max = p48_pack(v3_data);
    int n_max = p48_norm_sq(p_max);
    int n_max_expected = 8 * 63 * 63;
    printf("[%s] Max norm: %d == %d\n",
           n_max == n_max_expected ? "PASS" : "FAIL", n_max, n_max_expected);

    /* Distance */
    int d = p48_dist_sq(p1, p1);
    printf("[%s] Self-distance = 0\n", d == 0 ? "PASS" : "FAIL");
    int d2 = p48_dist_sq(p1, p_max);
    printf("[%s] Distance to max = %d\n", d2 > 0 ? "PASS" : "FAIL", d2);

    /* Float conversion */
    float fvec[8] = {0.5f, 0.25f, 0.75f, 0.0f, 1.0f, 0.1f, 0.9f, 0.33f};
    uint64_t pf = p48_quantize(fvec, 8);
    printf("[%s] Float quantization\n", pf != 0 ? "PASS" : "FAIL");

    /* Exact identity (Pythagorean property) */
    uint64_t same = p48_pack(v1_data);
    int dist_itself = p48_dist_sq(p1, same);
    printf("[%s] Pythagorean identity (unlimited drift = 0)\n",
           dist_itself == 0 ? "PASS" : "FAIL");

#if defined(__aarch64__)
    /* NEON SIMD check */
    uint64_t p_neon_a = p1;
    uint64_t p_neon_b = p2;
    int dot_neon = p48_dot_neon(p_neon_a, p_neon_b);
    int dist_neon = p48_dist_sq_neon(p_neon_a, p_neon_b);

    printf("[%s] NEON dot product %d == scalar %d\n",
           dot_neon == dot ? "PASS" : "FAIL", dot_neon, dot);
    printf("[%s] NEON distance %d == scalar %d\n",
           dist_neon == d2 ? "PASS" : "FAIL", dist_neon, d2);
#endif

    printf("\n--- Benchmarks ---\n\n");

    /* Generate 100,000 random P48 vectors */
    size_t N = 100000;
    uint64_t* batch = (uint64_t*)malloc(N * sizeof(uint64_t));
    srand(42);
    for (size_t i = 0; i < N; i++) {
        uint8_t c[8];
        for (int j = 0; j < 8; j++) {
            c[j] = rand() & 0x3F;
        }
        batch[i] = p48_pack(c);
    }

    uint64_t q = p48_pack(v1_data);

    /* Scalar nearest-neighbor */
    bench_search("Scalar NN 100k", batch, N, q);

#if defined(__aarch64__)
    /* NEON nearest-neighbor */
    {
        uint64_t v = q;
        clock_t start = clock();
        int best_idx = -1;
        int best_dist = INT32_MAX;
        for (size_t i = 0; i < N; i++) {
            int d = p48_dist_sq_neon(batch[i], v);
            if (d < best_dist) {
                best_dist = d;
                best_idx = (int)i;
            }
        }
        clock_t end = clock();
        double ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        printf("  NEON NN 100k: best=%d dist=%d time=%.3fms (%.0f vec/s)\n",
               best_idx, best_dist, ms, N / (ms / 1000.0));
    }
#endif

    free(batch);

    printf("\nAll tests complete.\n");
    return 0;
}
