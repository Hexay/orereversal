// region_dump — candidate ore blocks for a chunk box, via cubiomes, as CSV (family,x,y,z).
// Real ore is a subset of these (cubiomes skips the air/replaceable checks). The GPU matcher's refine
// pass and python/candidates.py call it; tests/regress.sh diffs cuda/oretest against it.
//
// Usage: region_dump <seed> <version> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax] [family ...] [+veins]
//                    [+branch]
//   Y band defaults to -64..-1. Families: tuff redstone lapis gravel granite copper iron diamond (buried
//   diamond only); default: all.
//   +veins also emits ore-vein blocks (ore_veins.h) under their family: tuff/iron from iron veins,
//   granite/copper from copper veins. Off by default so the output stays comparable with cuda/oretest.
//   +branch emits both outcomes of borderline surface-gate decisions for gravel, copper and lapis
//   (ore_branch.h), tagging such blocks family~<group>~<variant>.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "generator.h"
#include "finders.h"
#include "biomenoise.h"
#include "ore_veins.h"
#include "ore_branch.h"

typedef struct {
    int oreType; // cubiomes enum Ores
    const char* family;
} FamilyOre;

// Ore types whose candidates match real worldgen in the deepslate band (see cuda/README.md). Ore veins
// (iron and copper veins) are separate noise, emitted with +veins.
static const FamilyOre ORES[] = {
    {TuffOre, "tuff"},
    {RedstoneOre, "redstone"},
    {LowerRedstoneOre, "redstone"},
    {LapisOre, "lapis"},
    {BuriedLapisOre, "lapis"},
    {GravelOre, "gravel"},
    {LowerGraniteOre, "granite"},
    {UpperGraniteOre, "granite"},
    {CopperOre, "copper"},
    {LargeCopperOre, "copper"},
    {MiddleIronOre, "iron"},
    {SmallIronOre, "iron"},
    {UpperIronOre, "iron"},
    {BuriedDiamondOre, "diamond"},
};
#define ORE_COUNT ((int)(sizeof(ORES) / sizeof(ORES[0])))

// Accepts 1.N and 1.N.p; releases after 1.21 (incl. year-based 26.x) share 1.21's ore configs.
static int parseVersion(const char* s) {
    static const int byMinor[] = {MC_1_16, MC_1_17, MC_1_18, MC_1_19, MC_1_20, MC_1_21};
    int minor = strncmp(s, "1.", 2) == 0 ? atoi(s + 2) : 99;
    if (minor < 16)
        return MC_UNDEF;
    return minor > 21 ? MC_1_21 : byMinor[minor - 16];
}

// Prints the chunk's ore-vein blocks in [yMin, yMax]; raw ore blocks have no family and are skipped.
static long emitVeinBlocks(OreVeinParameters* p, int cx, int cz, int yMin, int yMax) {
    if (yMax < VEIN_MIN_Y || yMin >= VEIN_END_Y)
        return 0;
    static ChunkVeinNoise noise; // ~30 KB; static keeps it off the stack
    initChunkVeinNoise(&noise, p, cx, cz);
    long count = 0;
    for (int y = yMin < VEIN_MIN_Y ? VEIN_MIN_Y : yMin; y <= yMax && y < VEIN_END_Y; y++)
        for (int x = cx * 16; x < cx * 16 + 16; x++)
            for (int z = cz * 16; z < cz * 16 + 16; z++) {
                int isCopper, kind = oreVeinAt(p, &noise, cx, cz, x, y, z, &isCopper);
                if (kind == VEIN_ORE || kind == VEIN_FILLER) {
                    const char* family =
                        kind == VEIN_ORE ? (isCopper ? "copper" : "iron") : (isCopper ? "granite" : "tuff");
                    printf("%s,%d,%d,%d\n", family, x, y, z);
                    count++;
                }
            }
    return count;
}

static long emitBlocks(const char* family, const char* tag, const Pos3List* blocks, int yMin, int yMax) {
    long count = 0;
    for (int k = 0; k < blocks->size; k++) {
        Pos3 p = blocks->pos3s[k];
        if (p.y < yMin || p.y > yMax)
            continue;
        printf("%s%s,%d,%d,%d\n", family, tag, p.x, p.y, p.z);
        count++;
    }
    return count;
}

// One gate-sensitive config in one chunk. A single variant prints as usual; several print as
// "family~<group>~<variant>" so refine can pick one per group (ore_branch.h).
static long emitBranched(HeightCache* heights, int mc, uint64_t seed, OreConfig config, BranchWindow window,
                         const char* family, int cx, int cz, int yMin, int yMax) {
    BranchVariants v;
    generateBranchedOres(heights, mc, seed, config, window, cx, cz, &v);
    long count = 0;
    for (int k = 0; k < v.count; k++) {
        char tag[64] = "";
        if (v.count > 1) {
            snprintf(tag, sizeof(tag), "~%d_%d_%d~%d", cx, cz, config.index, k);
            printf("%s%s,0,-1000,0\n", family, tag); // declares the variant even if none of it is in the band
        }
        count += emitBlocks(family, tag, &v.variants[k], yMin, yMax);
        freePos3List(&v.variants[k]);
    }
    return count;
}

// Returns 0 for configs whose surface gate is never branched.
static int branchWindowFor(int oreType, BranchWindow* window) {
    if (oreType == GravelOre || oreType == CopperOre)
        *window = GRAVEL_COPPER_WINDOW;
    else if (oreType == LapisOre || oreType == BuriedLapisOre)
        *window = LAPIS_WINDOW;
    else
        return 0;
    return 1;
}

static int looksNumeric(const char* s) {
    return s[0] == '-' || (s[0] >= '0' && s[0] <= '9');
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <seed> <version> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax] [family ...] "
                "[+veins] [+branch]\n",
                argv[0]);
        return 2;
    }
    uint64_t seed = (uint64_t)strtoll(argv[1], NULL, 10);
    int mc = parseVersion(argv[2]);
    if (mc == MC_UNDEF) {
        fprintf(stderr, "unknown version '%s'\n", argv[2]);
        return 2;
    }
    int cxMin = atoi(argv[3]), cxMax = atoi(argv[4]), czMin = atoi(argv[5]), czMax = atoi(argv[6]);
    int yMin = -64, yMax = -1, firstFamilyArg = 7;
    if (argc >= 9 && looksNumeric(argv[7]) && looksNumeric(argv[8])) {
        yMin = atoi(argv[7]);
        yMax = atoi(argv[8]);
        firstFamilyArg = 9;
    }
    int veins = 0, branch = 0, familyArgs = 0;
    for (int a = firstFamilyArg; a < argc; a++) {
        if (!strcmp(argv[a], "+veins"))
            veins = 1;
        else if (!strcmp(argv[a], "+branch"))
            branch = 1;
        else
            familyArgs++;
    }
    int enabled[ORE_COUNT];
    for (int i = 0; i < ORE_COUNT; i++)
        enabled[i] = familyArgs == 0; // no family arguments: everything
    for (int a = firstFamilyArg; a < argc; a++)
        for (int i = 0; i < ORE_COUNT; i++)
            if (!strcmp(argv[a], ORES[i].family))
                enabled[i] = 1;
    OreVeinParameters veinParams;
    if (veins && !initOreVeinNoise(&veinParams, seed, mc))
        veins = 0; // before 1.18: no ore veins

    Generator g;
    setupGenerator(&g, mc, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, seed);

    fprintf(stderr, "seed=%" PRId64 " mc=%d chunks x[%d..%d] z[%d..%d] y[%d..%d]\n", (int64_t)seed, mc, cxMin,
            cxMax, czMin, czMax, yMin, yMax);
    printf("family,x,y,z\n");
    long total = 0;
    HeightCache heights;
    for (int cx = cxMin; cx <= cxMax; cx++) {
        for (int cz = czMin; cz <= czMax; cz++) {
            initHeightCache(&heights, &g, &sn, cx, cz);
            for (int i = 0; i < ORE_COUNT; i++) {
                OreConfig config;
                if (!enabled[i] || !getOreConfig(ORES[i].oreType, mc, 0, &config))
                    continue;
                // A config whose veins can't reach the Y band is skipped outright. Every config has its own
                // RNG stream, so this leaves the others' output unchanged (and drops upper iron's 90 veins).
                if (config.h1 - config.size > yMax || config.h2 + config.size < yMin)
                    continue;
                BranchWindow window;
                if (branch && branchWindowFor(ORES[i].oreType, &window)) {
                    total += emitBranched(&heights, mc, seed, config, window, ORES[i].family, cx, cz, yMin,
                                          yMax);
                    continue;
                }
                Pos3List blocks = generateOres(&g, &sn, config, cx, cz);
                total += emitBlocks(ORES[i].family, "", &blocks, yMin, yMax);
                freePos3List(&blocks);
            }
            if (veins)
                total += emitVeinBlocks(&veinParams, cx, cz, yMin, yMax);
        }
    }
    fprintf(stderr, "emitted %ld candidate blocks\n", total);
    return 0;
}
