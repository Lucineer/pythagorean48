#ifndef P48_NEON_H
#define P48_NEON_H

#include <stdint.h>
#include <arm_neon.h>

/*
 * NEON-accelerated P48 operations.
 * Since P48 stores 6-bit fields at bit boundaries (not byte boundaries),
 * we must unpack to 8 × uint8 before NEON operations.
 *
 * For maximum throughput: pre-unpack all vectors to byte arrays,
 * then process in bulk.
 */

/* Unpack 8 × 6-bit from uint64 into 8 × uint8 (NEON-friendly) */
static inline uint8x8_t p48_neon_unpack(uint64_t p) {
    /* Bit extraction: for each 6-bit field at bit position 6*i */
    const uint8_t shifts[8] = {0, 6, 12, 18, 24, 30, 36, 42};
    /* Load shift values */
    uint8x8_t vshifts = vld1_u8(shifts);
    /* Broadcast packed value */
    uint8x8_t vpacked = vdup_n_u8(0);
    /* Can't do variable shifts in NEON on uint8 easily.
       Fall back to scalar unpack then NEON operations */
    uint8_t comps[8];
    for (int i = 0; i < 8; i++) {
        comps[i] = (p >> shifts[i]) & 0x3F;
    }
    return vld1_u8(comps);
}

#endif /* P48_NEON_H */
