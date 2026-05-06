C fix for the NEON path.
The issue: p48_pack stores components at 6-bit boundaries, not byte boundaries.
NEON's vcreate_u8 treats the data as 8 × 8-bit bytes, not 8 × 6-bit fields.
This means we need a different approach for NEON:

1. Unpack 8 × 6-bit from uint64 into 8 × 8-bit uint8 (the "correct" NEON format)
2. Then use NEON on the unpacked data

Let me add an unpack-to-byte-array NEON path.
