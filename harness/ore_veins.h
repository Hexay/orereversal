// 1.18+ ore veins (vanilla OreVeinifier): large iron veins (iron ore, raw iron, tuff filler, y -60..-8)
// and copper veins (copper ore, raw copper, granite filler, y 0..50). They replace stone at noise-fill
// time, before ore features, so they add blocks no ore feature predicts, and they use no chunk RNG.
//
// cubiomes' getOreVeinBlockAt samples the vein noises per block. Vanilla interpolates veininess, vein_a
// and vein_b over 4x8x4 cells (only ore_gap is per block), so this evaluates them at cell corners once per
// chunk and lerps like vanilla's NoiseInterpolator (y, then x, then z). Against a real 1.18.2 world this
// explained 3547 of 3550 deep tuff blocks no tuff feature predicts, with no false predictions on plain
// deepslate (docs/research-log.md P8).
#ifndef ORE_VEINS_H
#define ORE_VEINS_H
#include <math.h>
#include "finders.h"

#define VEIN_MIN_Y    (-60) // corners outside [VEIN_MIN_Y, VEIN_END_Y) contribute 0, as in vanilla
#define VEIN_END_Y    51
#define CELL_XZ       4
#define CELL_Y        8
#define CHUNK_CELLS_Y ((320 + 64) / CELL_Y)

enum { VEIN_NONE = 0, VEIN_ORE, VEIN_RAW, VEIN_FILLER };

typedef struct {
    double v[CHUNK_CELLS_Y + 1][5][5]; // [cell y corner][x corner][z corner]
} CornerGrid;

typedef struct {
    CornerGrid toggle, a, b;
} ChunkVeinNoise;

static double cornerNoise(const DoublePerlinNoise* n, double scale, int x, int y, int z) {
    if (y < VEIN_MIN_Y || y >= VEIN_END_Y)
        return 0.0;
    return sampleDoublePerlin(n, x * scale, y * scale, z * scale);
}

static void fillCorners(CornerGrid* g, const DoublePerlinNoise* n, double scale, int cx, int cz) {
    for (int cy = 0; cy <= CHUNK_CELLS_Y; cy++)
        for (int i = 0; i < 5; i++)
            for (int k = 0; k < 5; k++)
                g->v[cy][i][k] =
                    cornerNoise(n, scale, cx * 16 + i * CELL_XZ, -64 + cy * CELL_Y, cz * 16 + k * CELL_XZ);
}

static void initChunkVeinNoise(ChunkVeinNoise* c, OreVeinParameters* p, int cx, int cz) {
    fillCorners(&c->toggle, &p->oreVeininess, 1.5, cx, cz);
    fillCorners(&c->a, &p->oreVeinA, 4.0, cx, cz);
    fillCorners(&c->b, &p->oreVeinB, 4.0, cx, cz);
}

// Trilinear interpolation at block (x, y, z) of the chunk the grid was built for.
static double interpolate(const CornerGrid* g, int cx, int cz, int x, int y, int z) {
    int lx = x - cx * 16, lz = z - cz * 16, ly = y + 64;
    int i = lx / CELL_XZ, k = lz / CELL_XZ, cy = ly / CELL_Y;
    double dx = (lx % CELL_XZ) / (double)CELL_XZ, dy = (ly % CELL_Y) / (double)CELL_Y,
           dz = (lz % CELL_XZ) / (double)CELL_XZ;
    double n00 = g->v[cy][i][k] + dy * (g->v[cy + 1][i][k] - g->v[cy][i][k]);
    double n10 = g->v[cy][i + 1][k] + dy * (g->v[cy + 1][i + 1][k] - g->v[cy][i + 1][k]);
    double n01 = g->v[cy][i][k + 1] + dy * (g->v[cy + 1][i][k + 1] - g->v[cy][i][k + 1]);
    double n11 = g->v[cy][i + 1][k + 1] + dy * (g->v[cy + 1][i + 1][k + 1] - g->v[cy][i + 1][k + 1]);
    double n0 = n00 + dx * (n10 - n00), n1 = n01 + dx * (n11 - n01);
    return n0 + dz * (n1 - n0);
}

// VEIN_* for block (x, y, z); *isCopper tells which vein type. Same decision order as vanilla OreVeinifier.
static int oreVeinAt(OreVeinParameters* p, const ChunkVeinNoise* c, int cx, int cz, int x, int y, int z,
                     int* isCopper) {
    if (y < VEIN_MIN_Y || y > 50)
        return VEIN_NONE;
    double toggle = interpolate(&c->toggle, cx, cz, x, y, z);
    *isCopper = toggle > 0.0;
    int minY = *isCopper ? 0 : -60, maxY = *isCopper ? 50 : -8;
    int top = maxY - y, bottom = y - minY;
    if (top < 0 || bottom < 0)
        return VEIN_NONE;
    double strength = fabs(toggle);
    if (strength + clampedMap(top < bottom ? top : bottom, 0.0, 20.0, -0.2, 0.0) < 0.4F)
        return VEIN_NONE;
    Xoroshiro rng = xAtPos(&p->posRandom, x, y, z);
    if (xNextFloat(&rng) > 0.7F)
        return VEIN_NONE;
    double ridge =
        -0.08F + fmax(fabs(interpolate(&c->a, cx, cz, x, y, z)), fabs(interpolate(&c->b, cx, cz, x, y, z)));
    if (ridge >= 0.0)
        return VEIN_NONE;
    if ((double)xNextFloat(&rng) < clampedMap(strength, 0.4F, 0.6F, 0.1F, 0.3F) &&
        sampleDoublePerlin(&p->oreGap, x, y, z) > -0.3F)
        return xNextFloat(&rng) < 0.02F ? VEIN_RAW : VEIN_ORE;
    return VEIN_FILLER;
}

#endif
