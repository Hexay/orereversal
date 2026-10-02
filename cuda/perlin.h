// 1.18+ Perlin and double-Perlin noise, ported from cubiomes noise.c. Only the single-octave form is
// ported (each side of a DoublePerlinNoise has one octave), which is all the ore-vein noises use.
// Plain C, host and device; initialization is host-side.
#ifndef PERLIN_H
#define PERLIN_H
#include <math.h>
#include <stdint.h>
#include "ore_config.h"
#include "ore_rng.h"

typedef struct {
    uint8_t d[257];
    uint8_t h2;
    double a, b, c;
    double amplitude, lacunarity;
    double d2, t2;
} PerlinNoise;

typedef struct {
    PerlinNoise octA, octB;
    double amplitude;
} DoublePerlinNoise;

static void xPerlinInit(PerlinNoise* p, Xoroshiro* xr) {
    p->a = xNextDouble(xr) * 256.0;
    p->b = xNextDouble(xr) * 256.0;
    p->c = xNextDouble(xr) * 256.0;
    p->amplitude = 1.0;
    p->lacunarity = 1.0;
    for (int i = 0; i < 256; i++)
        p->d[i] = (uint8_t)i;
    for (int i = 0; i < 256; i++) {
        int j = xNextInt(xr, 256 - i) + i;
        uint8_t n = p->d[i];
        p->d[i] = p->d[j];
        p->d[j] = n;
    }
    p->d[256] = p->d[0];
    double i2 = floor(p->b);
    double d2 = p->b - i2;
    p->h2 = (uint8_t)(int)i2;
    p->d2 = d2;
    p->t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
}

// cubiomes xOctaveInit for amplitudes {1.0}: one octave seeded from md5("octave_<omin>").
static void xSingleOctaveInit(PerlinNoise* p, Xoroshiro* xr, int omin) {
    static const uint64_t MD5_OCTAVE[][2] = {
        {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, // octave_-8
        {0xf11268128982754fULL, 0x257a1d670430b0aaULL}, // octave_-7
        {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}, // octave_-6
        {0x6d7b49e7e429850aULL, 0x2e3063c622a24777ULL}, // octave_-5
    };
    uint64_t lo = xNextLong(xr), hi = xNextLong(xr);
    Xoroshiro pxr = {lo ^ MD5_OCTAVE[omin + 8][0], hi ^ MD5_OCTAVE[omin + 8][1]};
    xPerlinInit(p, &pxr);
    p->amplitude = 1.0;
    p->lacunarity = ldexp(1.0, omin);
}

// omin in -8..-5.
static void xDoublePerlinInit(DoublePerlinNoise* n, Xoroshiro* xr, int omin) {
    xSingleOctaveInit(&n->octA, xr, omin);
    xSingleOctaveInit(&n->octB, xr, omin);
    n->amplitude = 5.0 / 6.0; // (5/3) * len / (len + 1) for len 1
}

#define DEFINE_INDEXED_LERP(name, T)                                                                        \
    ORE_HD static inline T name(uint8_t idx, T a, T b, T c) {                                               \
        switch (idx & 0xf) {                                                                                \
        case 0: return a + b;                                                                               \
        case 1: return -a + b;                                                                              \
        case 2: return a - b;                                                                               \
        case 3: return -a - b;                                                                              \
        case 4: return a + c;                                                                               \
        case 5: return -a + c;                                                                              \
        case 6: return a - c;                                                                               \
        case 7: return -a - c;                                                                              \
        case 8: return b + c;                                                                               \
        case 9: return -b + c;                                                                              \
        case 10: return b - c;                                                                              \
        case 11: return -b - c;                                                                             \
        case 12: return a + b;                                                                              \
        case 13: return -b + c;                                                                             \
        case 14: return -a + b;                                                                             \
        default: return -b - c;                                                                             \
        }                                                                                                   \
    }
DEFINE_INDEXED_LERP(indexedLerp, double)
DEFINE_INDEXED_LERP(indexedLerpF32, float)

ORE_HD static inline double perlinFade(double d) {
    return d * d * d * (d * (d * 6.0 - 15.0) + 10.0);
}

ORE_HD static inline double perlinLerp(double t, double a, double b) {
    return a + t * (b - a);
}

// cubiomes samplePerlin with yamp = 0.
ORE_HD static inline double samplePerlin(const PerlinNoise* p, double d1, double d2, double d3) {
    uint8_t h2;
    double t2;
    if (d2 == 0.0) {
        d2 = p->d2;
        h2 = p->h2;
        t2 = p->t2;
    } else {
        d2 += p->b;
        double i2 = floor(d2);
        d2 -= i2;
        h2 = (uint8_t)(int)i2;
        t2 = perlinFade(d2);
    }
    d1 += p->a;
    d3 += p->c;
    double i1 = floor(d1), i3 = floor(d3);
    d1 -= i1;
    d3 -= i3;
    uint8_t h1 = (uint8_t)(int)i1, h3 = (uint8_t)(int)i3;
    double t1 = perlinFade(d1), t3 = perlinFade(d3);

    const uint8_t* idx = p->d;
    uint8_t a1 = (uint8_t)(idx[h1] + h2), b1 = (uint8_t)(idx[h1 + 1] + h2);
    uint8_t a2 = (uint8_t)(idx[a1] + h3), b2 = (uint8_t)(idx[b1] + h3);
    uint8_t a3 = (uint8_t)(idx[a1 + 1] + h3), b3 = (uint8_t)(idx[b1 + 1] + h3);

    double l1 = indexedLerp(idx[a2], d1, d2, d3);
    double l2 = indexedLerp(idx[b2], d1 - 1, d2, d3);
    double l3 = indexedLerp(idx[a3], d1, d2 - 1, d3);
    double l4 = indexedLerp(idx[b3], d1 - 1, d2 - 1, d3);
    double l5 = indexedLerp(idx[a2 + 1], d1, d2, d3 - 1);
    double l6 = indexedLerp(idx[b2 + 1], d1 - 1, d2, d3 - 1);
    double l7 = indexedLerp(idx[a3 + 1], d1, d2 - 1, d3 - 1);
    double l8 = indexedLerp(idx[b3 + 1], d1 - 1, d2 - 1, d3 - 1);

    l1 = perlinLerp(t1, l1, l2);
    l3 = perlinLerp(t1, l3, l4);
    l5 = perlinLerp(t1, l5, l6);
    l7 = perlinLerp(t1, l7, l8);
    l1 = perlinLerp(t2, l1, l3);
    l5 = perlinLerp(t2, l5, l7);
    return perlinLerp(t3, l1, l5);
}

// FP32 approximation of samplePerlin for screening: lattice cell and fractions come from FP64 like the
// exact version, everything after in FP32 (error ~1e-6, versus FP64 at 1/64 the rate on consumer GPUs).
ORE_HD static inline float samplePerlinF32(const PerlinNoise* p, double d1, double d2, double d3) {
    uint8_t h2;
    float f2;
    if (d2 == 0.0) {
        f2 = (float)p->d2;
        h2 = p->h2;
    } else {
        d2 += p->b;
        double i2 = floor(d2);
        f2 = (float)(d2 - i2);
        h2 = (uint8_t)(int)i2;
    }
    d1 += p->a;
    d3 += p->c;
    double i1 = floor(d1), i3 = floor(d3);
    float f1 = (float)(d1 - i1), f3 = (float)(d3 - i3);
    uint8_t h1 = (uint8_t)(int)i1, h3 = (uint8_t)(int)i3;
    float t1 = f1 * f1 * f1 * (f1 * (f1 * 6.0f - 15.0f) + 10.0f);
    float t2 = f2 * f2 * f2 * (f2 * (f2 * 6.0f - 15.0f) + 10.0f);
    float t3 = f3 * f3 * f3 * (f3 * (f3 * 6.0f - 15.0f) + 10.0f);

    const uint8_t* idx = p->d;
    uint8_t a1 = (uint8_t)(idx[h1] + h2), b1 = (uint8_t)(idx[h1 + 1] + h2);
    uint8_t a2 = (uint8_t)(idx[a1] + h3), b2 = (uint8_t)(idx[b1] + h3);
    uint8_t a3 = (uint8_t)(idx[a1 + 1] + h3), b3 = (uint8_t)(idx[b1 + 1] + h3);
    float l1 = indexedLerpF32(idx[a2], f1, f2, f3);
    float l2 = indexedLerpF32(idx[b2], f1 - 1, f2, f3);
    float l3 = indexedLerpF32(idx[a3], f1, f2 - 1, f3);
    float l4 = indexedLerpF32(idx[b3], f1 - 1, f2 - 1, f3);
    float l5 = indexedLerpF32(idx[a2 + 1], f1, f2, f3 - 1);
    float l6 = indexedLerpF32(idx[b2 + 1], f1 - 1, f2, f3 - 1);
    float l7 = indexedLerpF32(idx[a3 + 1], f1, f2 - 1, f3 - 1);
    float l8 = indexedLerpF32(idx[b3 + 1], f1 - 1, f2 - 1, f3 - 1);
    l1 += t1 * (l2 - l1);
    l3 += t1 * (l4 - l3);
    l5 += t1 * (l6 - l5);
    l7 += t1 * (l8 - l7);
    l1 += t2 * (l3 - l1);
    l5 += t2 * (l7 - l5);
    return l1 + t3 * (l5 - l1);
}

ORE_HD static inline float sampleDoublePerlinF32(const DoublePerlinNoise* n, double x, double y, double z) {
    const double f = 337.0 / 331.0;
    double la = n->octA.lacunarity, lb = n->octB.lacunarity;
    float v = (float)n->octA.amplitude * samplePerlinF32(&n->octA, x * la, y * la, z * la);
    v += (float)n->octB.amplitude * samplePerlinF32(&n->octB, x * f * lb, y * f * lb, z * f * lb);
    return v * (float)n->amplitude;
}

ORE_HD static inline double sampleOctave1(const PerlinNoise* p, double x, double y, double z) {
    double lf = p->lacunarity;
    return p->amplitude * samplePerlin(p, x * lf, y * lf, z * lf);
}

ORE_HD static inline double sampleDoublePerlin(const DoublePerlinNoise* n, double x, double y, double z) {
    const double f = 337.0 / 331.0;
    double v = sampleOctave1(&n->octA, x, y, z);
    v += sampleOctave1(&n->octB, x * f, y * f, z * f);
    return v * n->amplitude;
}

#endif
