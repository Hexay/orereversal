// Iron-vein tuff for the GPU pass: the deepslate-band part of harness/ore_veins.h (vanilla OreVeinifier),
// whose doc has the model. Copper veins (y 0..50) never reach the band, and iron ore isn't a GPU family,
// so only iron-vein filler (tuff) is produced. Plain C, host and device; noise init is host-side.
//
// Exact FP64 evaluation is slow on consumer GPUs, so it runs behind three FP32 screens, each of which only
// rejects what the exact test certainly rejects (margins >= 1e-4, FP32 error ~1e-6): corners too weak for a
// vein, cells whose corners are all weak, and blocks failing the strength or ridge test.
#ifndef IRON_VEINS_H
#define IRON_VEINS_H
#include "perlin.h"

#define IRON_VEIN_MIN_Y  (-60)
#define IRON_VEIN_MAX_Y  (-8)
#define VEIN_CORNERS_Y   9 // y = -64, -56, ..., 0: every corner an iron-vein block interpolates between
#define VEIN_CORNERS     (VEIN_CORNERS_Y * 5 * 5)
#define VEIN_CELLS       ((VEIN_CORNERS_Y - 1) * 4 * 4) // indexed like corners: y-major, then x, then z
#define IRON_VEIN_ROWS   (16 * (IRON_VEIN_MAX_Y - IRON_VEIN_MIN_Y + 1)) // (x, y) rows of one chunk
#define VEIN_STRENGTH_CUTOFF 0.39 // below this |toggle| no block can pass the 0.4 strength test
#define VEIN_SCREEN_MARGIN   1e-4f

typedef struct {
    DoublePerlinNoise veininess, veinA, veinB, gap;
    Xoroshiro posRandom;
} OreVeinNoise;

// Corner samples for one chunk, indexed [corner y][corner x][corner z], exact and as FP32 copies.
typedef struct {
    double toggle[VEIN_CORNERS_Y][5][5], a[VEIN_CORNERS_Y][5][5], b[VEIN_CORNERS_Y][5][5];
    float toggleF32[VEIN_CORNERS_Y][5][5], aF32[VEIN_CORNERS_Y][5][5], bF32[VEIN_CORNERS_Y][5][5];
} VeinCorners;

// cubiomes initOreVeinNoise.
static void initOreVeinNoise(OreVeinNoise* n, uint64_t seed) {
    Xoroshiro ws;
    xSetSeed(&ws, seed);
    uint64_t lo = xNextLong(&ws), hi = xNextLong(&ws);
    Xoroshiro ore = {lo ^ 0x9b88124de600116dULL, hi ^ 0x2ae68055aa4a7761ULL}; // md5("minecraft:ore")
    uint64_t a = xNextLong(&ore), b = xNextLong(&ore);
    n->posRandom.lo = a;
    n->posRandom.hi = b;
    Xoroshiro veininess = {lo ^ 0x6b86c7820a307171ULL, hi ^ 0xd87fb0fefd9c1624ULL};
    Xoroshiro veinA = {lo ^ 0x4cd8d69b9a841649ULL, hi ^ 0xcdd63f17bfe8f5edULL};
    Xoroshiro veinB = {lo ^ 0x6b26220b31f7c6c9ULL, hi ^ 0xae077edebf6aaec1ULL};
    Xoroshiro gap = {lo ^ 0x9c4cc6b2fb0be4bbULL, hi ^ 0xbd5964705573bb5eULL};
    xDoublePerlinInit(&n->veininess, &veininess, -8);
    xDoublePerlinInit(&n->veinA, &veinA, -7);
    xDoublePerlinInit(&n->veinB, &veinB, -7);
    xDoublePerlinInit(&n->gap, &gap, -5);
}

ORE_HD static inline double clampedMap(double v, double inMin, double inMax, double outMin, double outMax) {
    double t = (v - inMin) / (inMax - inMin);
    if (t <= 0)
        return outMin;
    if (t >= 1)
        return outMax;
    return perlinLerp(t, outMin, outMax);
}

// Block coordinates of corner `index` (y-major, then x, then z); returns 0 if vanilla zeroes the corner
// for lying below the vein range.
ORE_HD static inline int veinCornerAt(int chunkX, int chunkZ, int index, int* x, int* y, int* z) {
    *x = chunkX * 16 + index / 5 % 5 * 4;
    *y = -64 + index / 25 * 8;
    *z = chunkZ * 16 + index % 5 * 4;
    return *y >= IRON_VEIN_MIN_Y;
}

// Screen 1: 1 if corner `index` may be strong enough for an iron vein.
ORE_HD static inline int mayBeIronVeinCorner(const OreVeinNoise* n, int chunkX, int chunkZ, int index) {
    int x, y, z;
    return veinCornerAt(chunkX, chunkZ, index, &x, &y, &z) &&
           sampleDoublePerlinF32(&n->veininess, x * 1.5, y * 1.5, z * 1.5) < -VEIN_STRENGTH_CUTOFF;
}

// The 4x8x4 cell holding chunk-local block (lx, y, lz).
ORE_HD static inline int veinCellOf(int lx, int y, int lz) {
    return ((y + 64) >> 3) * 16 + (lx >> 2) * 4 + (lz >> 2);
}

// Screen 2: 1 if cell `cell` may hold iron-vein blocks. Its blocks interpolate between its 8 corners, so it
// needs a strong one. strongCorner[] holds screen 1 for every corner.
ORE_HD static inline int veinCellMayHaveIron(const unsigned char* strongCorner, int cell) {
    int cy = cell >> 4, i = cell >> 2 & 3, k = cell & 3;
    for (int d = 0; d < 8; d++)
        if (strongCorner[(cy + (d >> 2)) * 25 + (i + (d >> 1 & 1)) * 5 + k + (d & 1)])
            return 1;
    return 0;
}

// 1 if corner `index` bounds a live cell, i.e. its exact noise is needed. cellLive[] is indexed by cell.
ORE_HD static inline int veinCornerNeeded(const unsigned char* cellLive, int index) {
    int cy = index / 25, i = index / 5 % 5, k = index % 5;
    for (int d = 0; d < 8; d++) {
        int ccy = cy - (d >> 2), ci = i - (d >> 1 & 1), ck = k - (d & 1);
        if (ccy >= 0 && ccy < VEIN_CORNERS_Y - 1 && ci >= 0 && ci < 4 && ck >= 0 && ck < 4 &&
            cellLive[ccy * 16 + ci * 4 + ck])
            return 1;
    }
    return 0;
}

ORE_HD static inline void fillVeinCorner(const OreVeinNoise* n, int chunkX, int chunkZ, int index,
                                         VeinCorners* c) {
    int x, y, z, live = veinCornerAt(chunkX, chunkZ, index, &x, &y, &z);
    double t = live ? sampleDoublePerlin(&n->veininess, x * 1.5, y * 1.5, z * 1.5) : 0.0;
    double a = live ? sampleDoublePerlin(&n->veinA, x * 4.0, y * 4.0, z * 4.0) : 0.0;
    double b = live ? sampleDoublePerlin(&n->veinB, x * 4.0, y * 4.0, z * 4.0) : 0.0;
    (&c->toggle[0][0][0])[index] = t;
    (&c->a[0][0][0])[index] = a;
    (&c->b[0][0][0])[index] = b;
    (&c->toggleF32[0][0][0])[index] = (float)t;
    (&c->aF32[0][0][0])[index] = (float)a;
    (&c->bF32[0][0][0])[index] = (float)b;
}

// Trilinear interpolation in vanilla's order (y, then x, then z) at chunk-local (lx, ly = y + 64, lz).
#define DEFINE_INTERPOLATE_CORNERS(name, T)                                                                 \
    ORE_HD static inline T name(const T v[VEIN_CORNERS_Y][5][5], int lx, int ly, int lz) {                  \
        int i = lx >> 2, k = lz >> 2, cy = ly >> 3;                                                         \
        T dx = (T)(lx & 3) / (T)4, dy = (T)(ly & 7) / (T)8, dz = (T)(lz & 3) / (T)4;                        \
        T n00 = v[cy][i][k] + dy * (v[cy + 1][i][k] - v[cy][i][k]);                                         \
        T n10 = v[cy][i + 1][k] + dy * (v[cy + 1][i + 1][k] - v[cy][i + 1][k]);                             \
        T n01 = v[cy][i][k + 1] + dy * (v[cy + 1][i][k + 1] - v[cy][i][k + 1]);                             \
        T n11 = v[cy][i + 1][k + 1] + dy * (v[cy + 1][i + 1][k + 1] - v[cy][i + 1][k + 1]);                 \
        T n0 = n00 + dx * (n10 - n00), n1 = n01 + dx * (n11 - n01);                                         \
        return n0 + dz * (n1 - n0);                                                                         \
    }
DEFINE_INTERPOLATE_CORNERS(interpolateCorners, double)
DEFINE_INTERPOLATE_CORNERS(interpolateCornersF32, float)

// Screen 3: 0 only if ironVeinTuffAt certainly rejects the block on strength or ridge.
ORE_HD static inline int mayBeIronVeinTuff(const VeinCorners* c, int lx, int y, int lz) {
    int ly = y + 64, top = IRON_VEIN_MAX_Y - y, bottom = y - IRON_VEIN_MIN_Y;
    int edge = top < bottom ? top : bottom;
    float toggle = interpolateCornersF32(c->toggleF32, lx, ly, lz);
    float edgeFactor = edge >= 20 ? 0.0f : -0.2f + 0.01f * (float)edge;
    if (toggle > VEIN_SCREEN_MARGIN || -toggle + edgeFactor < 0.4f - VEIN_SCREEN_MARGIN)
        return 0;
    float ridge = fmaxf(fabsf(interpolateCornersF32(c->aF32, lx, ly, lz)),
                        fabsf(interpolateCornersF32(c->bF32, lx, ly, lz)));
    return ridge < 0.08f + VEIN_SCREEN_MARGIN;
}

// 1 if block (x, y, z) of chunk (chunkX, chunkZ) is iron-vein tuff. Same decision order as oreVeinAt.
ORE_HD static inline int ironVeinTuffAt(const OreVeinNoise* n, const VeinCorners* c, int chunkX, int chunkZ,
                                        int x, int y, int z) {
    if (y < IRON_VEIN_MIN_Y || y > IRON_VEIN_MAX_Y)
        return 0;
    int lx = x - chunkX * 16, ly = y + 64, lz = z - chunkZ * 16;
    double toggle = interpolateCorners(c->toggle, lx, ly, lz);
    if (toggle > 0.0)
        return 0; // a copper vein, which starts at y 0
    int top = IRON_VEIN_MAX_Y - y, bottom = y - IRON_VEIN_MIN_Y;
    double strength = fabs(toggle);
    if (strength + clampedMap(top < bottom ? top : bottom, 0.0, 20.0, -0.2, 0.0) < 0.4F)
        return 0;
    Xoroshiro rng = xAtPos(&n->posRandom, x, y, z);
    if (xNextFloat(&rng) > 0.7F)
        return 0;
    double ridge = -0.08F + fmax(fabs(interpolateCorners(c->a, lx, ly, lz)),
                                 fabs(interpolateCorners(c->b, lx, ly, lz)));
    if (ridge >= 0.0)
        return 0;
    if ((double)xNextFloat(&rng) < clampedMap(strength, 0.4F, 0.6F, 0.1F, 0.3F) &&
        sampleDoublePerlin(&n->gap, x, y, z) > -0.3F)
        return 0; // iron ore or raw iron
    return 1;
}

#endif
