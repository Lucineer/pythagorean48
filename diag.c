#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>
#include "src/p48.h"

int main(void) {
    uint8_t data[8] = {0x0A, 0x14, 0x1E, 0x28, 0x32, 0x3C, 0x00, 0x00};
    uint64_t p = p48_pack(data);
    
    printf("Packed: 0x%016lx\n", p);
    printf("Bytes:  ");
    uint8_t* bp = (uint8_t*)&p;
    for (int i = 0; i < 8; i++) printf("%02x ", bp[i]);
    printf("\n");
    
    uint8x8_t v = vcreate_u8(p);
    uint8_t vals[8];
    vst1_u8(vals, v);
    printf("NEON:   ");
    for (int i = 0; i < 8; i++) printf("%02x ", vals[i]);
    printf("\n");
    
    /* Scalar dot */
    uint8_t data2[8] = {0x00, 0x0A, 0x14, 0x1E, 0x28, 0x32, 0x3C, 0x00};
    uint64_t p2 = p48_pack(data2);
    printf("Scalar dot: %d\n", p48_dot(p, p2));
    
    /* NEON dot */
    printf("NEON dot:   %d\n", p48_dot_neon(p, p2));
    
    /* Let's calculate manually what NEON should get */
    uint8x8_t va = vcreate_u8(p);
    uint8x8_t vb = vcreate_u8(p2);
    uint16x8_t wa = vmovl_u8(va);
    uint16x8_t wb = vmovl_u8(vb);
    
    printf("\nComponents:\n");
    for (int i = 0; i < 8; i++) {
        uint16_t ca = wa[i];
        uint16_t cb = wb[i];
        printf("  [%d] %3u * %3u = %5u\n", i, ca, cb, ca*cb);
    }
    
    return 0;
}
