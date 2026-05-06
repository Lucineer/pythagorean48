/* Benchmark: P48 NEON batch operations
 * Compares scalar vs NEON for warp-room's 13-vector P48 classification */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

/* Include p48.h for the scalar baseline */
#define P48_MAX_COMPS 104
#include "p48.h"
#include "p48_neon.h"

/* Generate random P48 vectors (6-bit components) */
static void rand_p48_vec(uint64_t *vec, int n_p48) {
    for (int i = 0; i < n_p48; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) {
            int c = rand() & 0x3F;
            v |= ((uint64_t)c) << (6 * j);
        }
        vec[i] = v;
    }
}

/* Scalar nearest neighbor for a batch of P48 vectors */
static int scalar_nn(const uint64_t *query, int query_len,
                     const uint64_t *vectors, int num_vectors) {
    int best_idx = -1;
    int best_dist = INT32_MAX;
    for (int vi = 0; vi < num_vectors; vi++) {
        int dist = 0;
        for (int pv = 0; pv < query_len; pv++) {
            uint64_t pa = query[pv];
            uint64_t pb = vectors[vi * query_len + pv];
            for (int i = 0; i < 8; i++) {
                int ca = (pa >> (6 * i)) & 0x3F;
                int cb = (pb >> (6 * i)) & 0x3F;
                int d = ca - cb;
                dist += d * d;
            }
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = vi;
        }
    }
    return best_idx;
}

/* Quick correctness test */
static int test_correctness(void) {
    printf("=== Correctness Test ===\n");

    uint64_t query[13] = {0};
    uint64_t vectors[100][13];

    /* Create vectors with known nearest neighbor */
    /* Vector 42 is a copy of the query with small perturbation */
    srand(42);
    rand_p48_vec(query, 13);

    for (int i = 0; i < 100; i++) {
        if (i == 42) {
            /* Copy query then flip one component by 1 */
            memcpy(vectors[i], query, 13 * sizeof(uint64_t));
            uint64_t v = vectors[i][0];
            int c = (v >> 6) & 0x3F;
            c = (c + 1) & 0x3F;
            v &= ~((uint64_t)0x3F << 6);
            v |= ((uint64_t)c) << 6;
            vectors[i][0] = v;
        } else {
            rand_p48_vec(vectors[i], 13);
        }
    }

    int scalar_best = scalar_nn(query, 13, (const uint64_t*)vectors, 100);

    /* NEON batch */
    int neon_results[100];
    p48_neon_batch_dot(query, 13, (const uint64_t*)vectors, 100,
                       neon_results, 0);
    int neon_best = 0;
    int neon_best_dist = neon_results[0];
    for (int i = 1; i < 100; i++) {
        if (neon_results[i] < neon_best_dist) {
            neon_best_dist = neon_results[i];
            neon_best = i;
        }
    }

    printf("  Scalar best index: %d\n", scalar_best);
    printf("  NEON best index:   %d\n", neon_best);

    if (scalar_best != neon_best) {
        printf("  MISMATCH! Scalar=%d NEON=%d\n", scalar_best, neon_best);
        for (int i = 0; i < 10; i++) {
            printf("  vec[%d]: scalar_sq=%d\n", i, 0);
        }
        return 0;
    }
    printf("  ✅ Correct\n");
    return 1;
}

/* Speed benchmark */
static void bench_speed(void) {
    printf("\n=== Speed Benchmark ===\n");

    srand(123);

    /* Warp-room scale: 13 P48 vectors per room, 4 rooms */
    int p48_len = 13;
    int n_rooms = 4;
    uint64_t rooms[4][13];
    uint64_t query[13];

    for (int r = 0; r < n_rooms; r++)
        rand_p48_vec(rooms[r], p48_len);
    rand_p48_vec(query, p48_len);

    /* Warmup */
    for (int i = 0; i < 100; i++)
        scalar_nn(query, p48_len, (const uint64_t*)rooms, n_rooms);

    /* Benchmark: single query vs 4 rooms, 100000 iterations */
    int iterations = 100000;
    clock_t start, end;

    /* Scalar */
    start = clock();
    for (int i = 0; i < iterations; i++) {
        scalar_nn(query, p48_len, (const uint64_t*)rooms, n_rooms);
    }
    end = clock();
    double scalar_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    /* NEON batch */
    int rbuf[4];
    start = clock();
    for (int i = 0; i < iterations; i++) {
        p48_neon_batch_dot(query, p48_len, (const uint64_t*)rooms,
                           n_rooms, rbuf, 0);
        /* Find min */
        int best = rbuf[0];
        for (int j = 1; j < n_rooms; j++)
            if (rbuf[j] < best) best = rbuf[j];
    }
    end = clock();
    double neon_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    printf("  Warp-room: 1 query vs 4 rooms, 13 P48 vecs each\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Scalar: %.2f ms (%.0f ops/s)\n", scalar_ms,
           iterations / (scalar_ms / 1000.0));
    printf("  NEON:   %.2f ms (%.0f ops/s)", neon_ms,
           iterations / (neon_ms / 1000.0));
    if (neon_ms > 0)
        printf(" — %.1fx speedup\n", scalar_ms / neon_ms);
    else
        printf("\n");

    /* Large batch benchmark: 100k vectors vs 1 query */
    int n_vecs = 100000;
    uint64_t *big_batch = (uint64_t*)malloc(n_vecs * p48_len * sizeof(uint64_t));
    if (!big_batch) return;

    for (int i = 0; i < n_vecs; i++)
        rand_p48_vec(&big_batch[i * p48_len], p48_len);

    /* Scalar large batch */
    start = clock();
    scalar_nn(query, p48_len, big_batch, n_vecs);
    end = clock();
    double s_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    /* NEON large batch */
    int *nres = (int*)malloc(n_vecs * sizeof(int));
    if (!nres) { free(big_batch); return; }

    start = clock();
    p48_neon_batch_dot(query, p48_len, big_batch, n_vecs, nres, 0);
    /* Find min */
    int nb = nres[0];
    for (int i = 1; i < n_vecs; i++)
        if (nres[i] < nb) nb = nres[i];
    end = clock();
    double n_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    printf("  Large batch: 1 query vs %d vectors\n", n_vecs);
    printf("  Scalar: %.2f ms (%.0f vec/s)\n", s_ms, n_vecs / (s_ms / 1000.0));
    printf("  NEON:   %.2f ms (%.0f vec/s)", n_ms, n_vecs / (n_ms / 1000.0));
    if (n_ms > 0)
        printf(" — %.1fx speedup\n", s_ms / n_ms);
    else
        printf("\n");

    free(big_batch);
    free(nres);
}

int main(void) {
    test_correctness();
    bench_speed();
    printf("\nDone.\n");
    return 0;
}
