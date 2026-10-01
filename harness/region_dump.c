// region_dump — bulk Tier-1 deepslate ore candidate generator (P2).
// Emits CSV (family,x,y,z) for every candidate ore block in a chunk box, within a Y band.
// One process for a whole region (vs ore_dump's single chunk) so the matcher gets a fast feed.
// These are RNG candidate positions (real ores are a subset); only the discard-free Tier-1 families
// that we validated as bit-exact are emitted. See docs/research-log.md.
//
// Usage: region_dump <seed> <version> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax] [family ...]
//   default Y band: -64..-1 (deepslate). Optional family filter (tuff redstone lapis gravel granite copper).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "generator.h"
#include "finders.h"
#include "biomenoise.h"

typedef struct {
    int ore;
    const char* family;
} OreFam;
// Tier-1: discard-free (0 or 1) feature ores that bit-match real worldgen with no terrain sim.
static const OreFam TIER1[] = {
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
    // iron: discard-free but the separate ore-vein noise system isn't simulated, so candidates
    // miss a few real iron blocks (~2/16 deep). 1.18 deep iron = small/middle; upper is y80+.
    {MiddleIronOre, "iron"},
    {SmallIronOre, "iron"},
    {UpperIronOre, "iron"},
};
static const int TIER1_N = sizeof(TIER1) / sizeof(TIER1[0]);

static int parseVersion(const char* s) {
    struct {
        const char* n;
        int mc;
    } m[] = {
        {"1.21", MC_1_21}, {"1.20", MC_1_20}, {"1.19", MC_1_19},
        {"1.18", MC_1_18}, {"1.17", MC_1_17}, {"1.16", MC_1_16},
    };
    for (size_t i = 0; i < sizeof(m) / sizeof(m[0]); i++)
        if (!strcmp(s, m[i].n))
            return m[i].mc;
    return MC_UNDEF;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <seed> <version> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax] [family ...]\n",
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
    int yMin = -64, yMax = -1, ai = 7;
    if (argc >= 9 && argv[7][0] != '\0' && (argv[7][0] == '-' || argv[7][0] >= '0' && argv[7][0] <= '9') &&
        (argv[8][0] == '-' || argv[8][0] >= '0' && argv[8][0] <= '9')) {
        yMin = atoi(argv[7]);
        yMax = atoi(argv[8]);
        ai = 9;
    }
    // optional family filter
    int useFam[16];
    for (int i = 0; i < TIER1_N; i++)
        useFam[i] = (ai >= argc); // if no filter, use all
    for (int a = ai; a < argc; a++)
        for (int i = 0; i < TIER1_N; i++)
            if (!strcmp(argv[a], TIER1[i].family))
                useFam[i] = 1;

    Generator g;
    setupGenerator(&g, mc, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, seed);

    fprintf(stderr, "seed=%" PRId64 " mc=%d chunks x[%d..%d] z[%d..%d] y[%d..%d]\n", (int64_t)seed, mc, cxMin,
            cxMax, czMin, czMax, yMin, yMax);
    printf("family,x,y,z\n");
    long total = 0;
    OreConfig oc;
    for (int cx = cxMin; cx <= cxMax; cx++) {
        for (int cz = czMin; cz <= czMax; cz++) {
            for (int i = 0; i < TIER1_N; i++) {
                if (!useFam[i])
                    continue;
                if (!getOreConfig(TIER1[i].ore, mc, 0, &oc))
                    continue;
                // Skip configs whose entire vein band can't reach the query window. Each config has its
                // own decorator RNG (config.index), so skipping one never shifts another -> bit-identical
                // output. Kills UpperIronOre (y80..384, repeatCount=90) for deepslate queries: pure waste.
                if (oc.h1 - oc.size > yMax || oc.h2 + oc.size < yMin)
                    continue;
                Pos3List ores = generateOres(&g, &sn, oc, cx, cz);
                for (int k = 0; k < ores.size; k++) {
                    Pos3 p = ores.pos3s[k];
                    if (p.y < yMin || p.y > yMax)
                        continue;
                    printf("%s,%d,%d,%d\n", TIER1[i].family, p.x, p.y, p.z);
                    total++;
                }
                freePos3List(&ores);
            }
        }
    }
    fprintf(stderr, "emitted %ld candidate blocks\n", total);
    return 0;
}
