// ore_dump — P1 validation harness.
// Dumps cubiomes' predicted ore *candidate* block positions for a given
// seed / MC version / chunk. Output is CSV (ore,index,x,y,z) sorted for stable
// diffing against ground truth. These are RNG candidate positions only: cubiomes
// does NOT apply the replace-block / air-exposure filters (see NOTES.md), so real
// in-game ores are a SUBSET of this output.
//
// Usage:
//   ore_dump <seed> <version> <chunkX> <chunkZ> [oreName ...]
//   ore_dump --list
// version: e.g. 1.21 1.20 1.19 1.18 1.21.3 (latest patch of that release)
// If no oreName given, dumps every overworld ore that exists for the version.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "generator.h"
#include "finders.h"
#include "biomenoise.h"

// Names indexed by enum Ores (finders.h:421). Order MUST match the enum exactly.
static const char *ORE_NAMES[ORE_NUM] = {
    "AndesiteOre", "BlackstoneOre", "BuriedDiamondOre", "BuriedLapisOre", "ClayOre",
    "CoalOre", "CopperOre", "DeepslateOre", "DeltasGoldOre", "DeltasQuartzOre",
    "DiamondOre", "DioriteOre", "DirtOre", "EmeraldOre", "ExtraGoldOre",
    "GoldOre", "GraniteOre", "GravelOre", "IronOre", "LapisOre",
    "LargeCopperOre", "LargeDebrisOre", "LargeDiamondOre", "LowerAndesiteOre", "LowerCoalOre",
    "LowerDioriteOre", "LowerGoldOre", "LowerGraniteOre", "LowerRedstoneOre", "MagmaOre",
    "MediumDiamondOre", "MiddleIronOre", "NetherGoldOre", "NetherGravelOre", "NetherQuartzOre",
    "RedstoneOre", "SmallDebrisOre", "SmallIronOre", "SoulSandOre", "TuffOre",
    "UpperAndesiteOre", "UpperCoalOre", "UpperDioriteOre", "UpperGraniteOre", "UpperIronOre",
};

static int parseVersion(const char *s) {
    struct { const char *name; int mc; } map[] = {
        {"1.21.11", MC_1_21_11}, {"1.21.9", MC_1_21_9}, {"1.21.6", MC_1_21_6},
        {"1.21.5", MC_1_21_5}, {"1.21.4", MC_1_21_4}, {"1.21.3", MC_1_21_3},
        {"1.21.1", MC_1_21_1}, {"1.21", MC_1_21}, {"1.20", MC_1_20},
        {"1.19", MC_1_19}, {"1.19.2", MC_1_19_2}, {"1.18", MC_1_18}, {"1.17", MC_1_17},
        {"1.16", MC_1_16},
    };
    for (size_t i = 0; i < sizeof(map)/sizeof(map[0]); i++)
        if (!strcmp(s, map[i].name)) return map[i].mc;
    return MC_UNDEF;
}

static int oreByName(const char *s) {
    for (int i = 0; i < ORE_NUM; i++)
        if (!strcmp(s, ORE_NAMES[i])) return i;
    return -1;
}

static int cmpPos3(const void *a, const void *b) {
    const Pos3 *p = a, *q = b;
    if (p->y != q->y) return p->y - q->y;
    if (p->x != q->x) return p->x - q->x;
    return p->z - q->z;
}

static void dumpOre(const Generator *g, const SurfaceNoise *sn, int oreType,
                    int cx, int cz, FILE *out, long *grandTotal) {
    OreConfig oc;
    if (!getOreConfig(oreType, g->mc, 0, &oc)) return;     // not in this version
    if (oc.dim != DIM_OVERWORLD) return;                    // overworld-only for now

    Pos3List ores = generateOres(g, sn, oc, cx, cz);
    if (ores.size > 0)
        qsort(ores.pos3s, ores.size, sizeof(Pos3), cmpPos3);

    fprintf(stderr, "  %-18s index=%-3d step=%-2d size=%-3d repeat=%-3d -> %d blocks\n",
            ORE_NAMES[oreType], oc.index, oc.step, oc.size, oc.repeatCount, ores.size);
    for (int i = 0; i < ores.size; i++) {
        Pos3 p = ores.pos3s[i];
        fprintf(out, "%s,%d,%d,%d,%d\n", ORE_NAMES[oreType], oc.index, p.x, p.y, p.z);
    }
    *grandTotal += ores.size;
    freePos3List(&ores);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--list")) {
        for (int i = 0; i < ORE_NUM; i++) printf("%2d  %s\n", i, ORE_NAMES[i]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--config")) {
        int mc = parseVersion(argv[2]);
        if (mc == MC_UNDEF) { fprintf(stderr, "unknown version '%s'\n", argv[2]); return 2; }
        printf("ore,index,step,size,repeat,h1,h2,h3,discard,dim\n");
        for (int ot = 0; ot < ORE_NUM; ot++) {
            OreConfig oc;
            if (!getOreConfig(ot, mc, 0, &oc)) continue;
            printf("%s,%d,%d,%d,%d,%d,%d,%d,%.2f,%d\n", ORE_NAMES[ot], oc.index, oc.step,
                   oc.size, oc.repeatCount, oc.h1, oc.h2, oc.h3, oc.discardChanceOnAirExposure, oc.dim);
        }
        return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s <seed> <version> <chunkX> <chunkZ> [oreName ...]\n", argv[0]);
        fprintf(stderr, "       %s --list\n", argv[0]);
        return 2;
    }

    uint64_t seed = (uint64_t)strtoll(argv[1], NULL, 10);
    int mc = parseVersion(argv[2]);
    if (mc == MC_UNDEF) { fprintf(stderr, "unknown version '%s'\n", argv[2]); return 2; }
    int cx = atoi(argv[3]);
    int cz = atoi(argv[4]);

    Generator g;
    setupGenerator(&g, mc, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, seed);

    fprintf(stderr, "seed=%" PRId64 " version=%s(mc=%d) chunk=(%d,%d) blocks=(%d..%d, %d..%d)\n",
            (int64_t)seed, argv[2], mc, cx, cz, cx<<4, (cx<<4)+15, cz<<4, (cz<<4)+15);

    printf("ore,index,x,y,z\n");
    long total = 0;

    if (argc > 5) {
        for (int i = 5; i < argc; i++) {
            int ot = oreByName(argv[i]);
            if (ot < 0) { fprintf(stderr, "unknown ore '%s' (use --list)\n", argv[i]); continue; }
            dumpOre(&g, &sn, ot, cx, cz, stdout, &total);
        }
    } else {
        for (int ot = 0; ot < ORE_NUM; ot++)
            dumpOre(&g, &sn, ot, cx, cz, stdout, &total);
    }

    fprintf(stderr, "total candidate blocks: %ld\n", total);
    return 0;
}
