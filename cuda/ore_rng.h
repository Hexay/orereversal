// Xoroshiro128++ with Java's nextInt/nextLong/nextDouble semantics, ported from cubiomes rng.h.
// Function names follow cubiomes so the port can be checked line by line.
#ifndef ORE_RNG_H
#define ORE_RNG_H
#include <stdint.h>
#include "ore_config.h"

typedef struct {
    uint64_t lo, hi;
} Xoroshiro;

ORE_HD static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

ORE_HD static inline void xSetSeed(Xoroshiro* xr, uint64_t value) {
    const uint64_t XL = 0x9e3779b97f4a7c15ULL, XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL, B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

ORE_HD static inline uint64_t xNextLong(Xoroshiro* xr) {
    uint64_t l = xr->lo, h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

// Java nextLong(): two 32-bit halves.
ORE_HD static inline uint64_t xNextLongJ(Xoroshiro* xr) {
    int32_t a = (int32_t)(xNextLong(xr) >> 32);
    int32_t b = (int32_t)(xNextLong(xr) >> 32);
    return ((uint64_t)a << 32) + b;
}

// Java nextInt(n), including its rejection loop for non-power-of-two n.
ORE_HD static inline int xNextIntJ(Xoroshiro* xr, uint32_t n) {
    const int m = n - 1;
    if ((m & n) == 0) {
        uint64_t x = n * (xNextLong(xr) >> 33);
        return (int)((int64_t)x >> 31);
    }
    int bits, val;
    do {
        bits = (int)(xNextLong(xr) >> 33);
        val = bits % n;
    } while ((int32_t)((uint32_t)bits - val + m) < 0);
    return val;
}

ORE_HD static inline float xNextFloat(Xoroshiro* xr) {
    return (xNextLong(xr) >> (64 - 24)) * 5.9604645E-8F;
}

ORE_HD static inline double xNextDoubleJ(Xoroshiro* xr) {
    uint64_t a = xNextLong(xr), b = xNextLong(xr);
    return ((a >> (64 - 26) << 27) + (b >> (64 - 27))) * 1.1102230246251565E-16;
}

ORE_HD static inline int xNextIntBetween(Xoroshiro* xr, int min, int max) {
    return xNextIntJ(xr, (uint32_t)(max - min + 1)) + min;
}

// Per-chunk population seed (cubiomes getPopulationSeed, 1.18+ variant).
ORE_HD static inline uint64_t getPopulationSeed(uint64_t worldSeed, int blockX, int blockZ) {
    Xoroshiro xr;
    xSetSeed(&xr, worldSeed);
    uint64_t a = xNextLongJ(&xr) | 1ULL;
    uint64_t b = xNextLongJ(&xr) | 1ULL;
    return ((uint64_t)blockX * a + (uint64_t)blockZ * b) ^ worldSeed;
}

#endif
