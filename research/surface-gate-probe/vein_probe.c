// Research probe: feature candidates (all Y) + per-vein surface-gate info + ore-vein predictions
// (cubiomes direct sampling and vanilla-style cell interpolation).
// Usage: vein_probe <seed> <cxMin> <cxMax> <czMin> <czMax>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "generator.h"
#include "finders.h"
#include "biomenoise.h"
#include "noise.h"
#define PI 3.14159265358979323846

typedef struct { int oreType; const char* fam; } FO;
static const FO ORES[] = {
    {TuffOre, "tuff"}, {GravelOre, "gravel"}, {LowerGraniteOre, "granite"}, {UpperGraniteOre, "granite"},
    {CopperOre, "copper"}, {LargeCopperOre, "copper"}, {MiddleIronOre, "iron"}, {SmallIronOre, "iron"},
    {UpperIronOre, "iron"}, {RedstoneOre, "redstone"}, {LowerRedstoneOre, "redstone"},
    {LapisOre, "lapis"}, {BuriedLapisOre, "lapis"},
};

static void genConfig(const Generator* g, const SurfaceNoise* sn, OreConfig config, int cx, int cz, const char* fam, int ungated) {
    uint64_t ps = getPopulationSeed(g->mc, g->seed, cx << 4, cz << 4);
    CREATE_RANDOM_SOURCE(rnd, 0);
    rnd.setSeed(rnd.state, ps + config.index + 10000 * config.step);
    int oreType = config.oreType;
    int repeat = (oreType == UpperGraniteOre) ? (rnd.nextFloat(rnd.state) < 1.0F / config.repeatCount)
                                              : config.repeatCount;
    for (int i = 0; i < repeat; i++) {
        Pos3 pos = generateBaseOrePosition(g->mc, config, cx, cz, rnd);
        int biome = getBiomeAt(g, 1, pos.x, pos.y, pos.z);
        if (!isViableOreBiome(g->mc, oreType, biome))
            continue;
        float angle = rnd.nextFloat(rnd.state) * (float)PI;
        float size = (float)config.size / 8.0F;
        int amort = ceil(((float)config.size / 16.0F * 2.0F + 1.0F) / 2.0F);
        double xp = (double)pos.x + sin(angle) * (double)size, xn = (double)pos.x - sin(angle) * (double)size;
        double zp = (double)pos.z + cos(angle) * (double)size, zn = (double)pos.z - cos(angle) * (double)size;
        double yp = pos.y + rnd.nextInt(rnd.state, 3) - 2;
        double yn = pos.y + rnd.nextInt(rnd.state, 3) - 2;
        int sx = pos.x - ceil(size) - amort, sy = pos.y - 2 - amort, sz = pos.z - ceil(size) - amort;
        int oreSize = 2 * (ceil(size) + amort), radius = 2 * (2 + amort);
        int maxH = -9999, pass = 0;
        for (int x = sx; x <= sx + oreSize; ++x)
            for (int z = sz; z <= sz + oreSize; ++z) {
                float y;
                mapApproxHeight(&y, 0, g, sn, x >> 2, z >> 2, 1, 1);
                int h = (int)floor(y);
                if (h > maxH) maxH = h;
            }
        pass = ungated || sy <= maxH;
        // G,oreType,cx,cz,vein,startX,startY,startZ,oreSize,maxApproxH,pass
        if (!ungated)
            printf("G,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", oreType, cx, cz, i, sx, sy, sz, oreSize, maxH, pass);
        if (pass) {
            Pos3List l;
            createPos3List(&l, 4096);
            generateVeinPart(g->mc, config, rnd, xp, xn, zp, zn, yp, yn, sx, sy, sz, oreSize, radius, &l);
            for (int k = 0; k < l.size; k++)
                printf("%s,%s,%d,%d,%d\n", ungated ? "U" : "C", fam,l.pos3s[k].x, l.pos3s[k].y, l.pos3s[k].z);
            freePos3List(&l);
        }
    }
}

static double cornerNoise(const DoublePerlinNoise* n, double s, int x, int y, int z) {
    if (y < -60 || y >= 51) return 0.0;
    return sampleDoublePerlin(n, x * s, y * s, z * s);
}

// vanilla NoiseInterpolator order: y, then x, then z
static double interp(const DoublePerlinNoise* n, double s, int x, int y, int z) {
    int x0 = (int)floor(x / 4.0) * 4, z0 = (int)floor(z / 4.0) * 4;
    int y0 = -64 + (int)floor((y + 64) / 8.0) * 8;
    double dx = (x - x0) / 4.0, dy = (y - y0) / 8.0, dz = (z - z0) / 4.0;
    double n000 = cornerNoise(n, s, x0, y0, z0), n001 = cornerNoise(n, s, x0, y0, z0 + 4);
    double n100 = cornerNoise(n, s, x0 + 4, y0, z0), n101 = cornerNoise(n, s, x0 + 4, y0, z0 + 4);
    double n010 = cornerNoise(n, s, x0, y0 + 8, z0), n011 = cornerNoise(n, s, x0, y0 + 8, z0 + 4);
    double n110 = cornerNoise(n, s, x0 + 4, y0 + 8, z0), n111 = cornerNoise(n, s, x0 + 4, y0 + 8, z0 + 4);
    double n00 = n000 + dy * (n010 - n000), n10 = n100 + dy * (n110 - n100);
    double n01 = n001 + dy * (n011 - n001), n11 = n101 + dy * (n111 - n101);
    double n0 = n00 + dx * (n10 - n00), n1 = n01 + dx * (n11 - n01);
    return n0 + dz * (n1 - n0);
}


// 0 none, 1 ore, 2 raw, 3 filler; *copper set to vein type
static int veinInterp(int x, int y, int z, OreVeinParameters* p, int* copper) {
    double toggle = interp(&p->oreVeininess, 1.5, x, y, z);
    int isCopper = toggle > 0.0;
    *copper = isCopper;
    int minY = isCopper ? 0 : -60, maxY = isCopper ? 50 : -8;
    double a = fabs(toggle);
    int top = maxY - y, bot = y - minY;
    if (bot < 0 || top < 0) return 0;
    int off = top < bot ? top : bot;
    if (a + clampedMap(off, 0.0, 20.0, -0.2, 0.0) < 0.4F) return 0;
    Xoroshiro xr = xAtPos(&p->posRandom, x, y, z);
    if (xNextFloat(&xr) > 0.7F) return 0;
    double ridged = -0.08F + fmax(fabs(interp(&p->oreVeinA, 4.0, x, y, z)), fabs(interp(&p->oreVeinB, 4.0, x, y, z)));
    if (ridged >= 0.0) return 0;
    double thr = clampedMap(a, 0.4F, 0.6F, 0.1F, 0.3F);
    if ((double)xNextFloat(&xr) < thr && sampleDoublePerlin(&p->oreGap, x, y, z) > -0.3F)
        return xNextFloat(&xr) < 0.02F ? 2 : 1;
    return 3;
}

static const char* veinName(int k, int copper) {
    static const char* n[2][4] = {{"", "iron", "rawiron", "tuff"}, {"", "copper", "rawcopper", "granite"}};
    return n[copper][k];
}

int main(int argc, char** argv) {
    uint64_t seed = strtoull(argv[1], 0, 10);
    int cx0 = atoi(argv[2]), cx1 = atoi(argv[3]), cz0 = atoi(argv[4]), cz1 = atoi(argv[5]);
    Generator g;
    setupGenerator(&g, MC_1_18, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, seed);
    OreVeinParameters ov;
    initOreVeinNoise(&ov, seed, MC_1_18);
    for (int cx = cx0; cx <= cx1; cx++)
        for (int cz = cz0; cz <= cz1; cz++) {
            for (size_t i = 0; i < sizeof(ORES) / sizeof(ORES[0]); i++) {
                OreConfig c;
                if (getOreConfig(ORES[i].oreType, MC_1_18, 0, &c)) {
                    genConfig(&g, &sn, c, cx, cz, ORES[i].fam, 0);
                    if (ORES[i].oreType != UpperIronOre && ORES[i].oreType != GravelOre)
                        genConfig(&g, &sn, c, cx, cz, ORES[i].fam, 1);
                }
            }
            for (int x = cx * 16; x < cx * 16 + 16; x++)
                for (int z = cz * 16; z < cz * 16 + 16; z++)
                    for (int y = -60; y <= 50; y++) {
                        int b = getOreVeinBlockAt(x, y, z, &ov);
                        if (b >= 0) {
                            const char* nm = b == IRON_ORE ? "iron" : b == RAW_IRON_BLOCK ? "rawiron"
                                           : b == TUFF ? "tuff" : b == COPPER_ORE ? "copper"
                                           : b == RAW_COPPER_BLOCK ? "rawcopper" : b == GRANITE ? "granite" : "?";
                            printf("VD,%s,%d,%d,%d\n", nm, x, y, z);
                        }
                        int cop, k = veinInterp(x, y, z, &ov, &cop);
                        if (k)
                            printf("VI,%s,%d,%d,%d\n", veinName(k, cop), x, y, z);
                    }
            // per-4x4-cell approx height for the chunk
            for (int qx = cx * 4; qx < cx * 4 + 4; qx++)
                for (int qz = cz * 4; qz < cz * 4 + 4; qz++) {
                    float y;
                    mapApproxHeight(&y, 0, &g, &sn, qx, qz, 1, 1);
                    printf("H,%d,%d,%d\n", qx, qz, (int)floor(y));
                }
        }
    return 0;
}
