// oretest — host driver for oregen.h (and, with +veins, iron_veins.h) that prints harness/region_dump's
// CSV format, so the two can be diffed for bit-exactness (tests/regress.sh "golden_diff").
// Usage: oretest <seed> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax [version]] [+veins]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oregen.h"
#include "iron_veins.h"

static long emitIronVeinTuff(const OreVeinNoise* noise, int cx, int cz, int yMin, int yMax) {
    // Same staging as kIronVeins, so the diff against cubiomes also checks that screening drops no block.
    static VeinCorners corners;
    unsigned char strongCorner[VEIN_CORNERS], cellLive[VEIN_CELLS];
    for (int i = 0; i < VEIN_CORNERS; i++)
        strongCorner[i] = (unsigned char)mayBeIronVeinCorner(noise, cx, cz, i);
    for (int i = 0; i < VEIN_CELLS; i++)
        cellLive[i] = (unsigned char)veinCellMayHaveIron(strongCorner, i);
    for (int i = 0; i < VEIN_CORNERS; i++)
        if (veinCornerNeeded(cellLive, i))
            fillVeinCorner(noise, cx, cz, i, &corners);
    long count = 0;
    for (int y = yMin; y <= yMax; y++)
        for (int x = cx * 16; x < cx * 16 + 16; x++)
            for (int z = cz * 16; z < cz * 16 + 16; z++)
                if (y >= IRON_VEIN_MIN_Y && y <= IRON_VEIN_MAX_Y &&
                    cellLive[((y + 64) >> 3) * 16 + ((x - cx * 16) >> 2) * 4 + ((z - cz * 16) >> 2)] &&
                    mayBeIronVeinTuff(&corners, x - cx * 16, y, z - cz * 16) &&
                    ironVeinTuffAt(noise, &corners, cx, cz, x, y, z)) {
                    printf("tuff,%d,%d,%d\n", x, y, z);
                    count++;
                }
    return count;
}

int main(int argc, char** argv) {
    int veins = argc > 1 && !strcmp(argv[argc - 1], "+veins");
    if (veins)
        argc--;
    if (argc < 6) {
        fprintf(stderr, "usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax [version]] [+veins]\n",
                argv[0]);
        return 2;
    }
    uint64_t seed = (uint64_t)strtoll(argv[1], NULL, 10);
    int cxMin = atoi(argv[2]), cxMax = atoi(argv[3]), czMin = atoi(argv[4]), czMax = atoi(argv[5]);
    int yMin = -64, yMax = -1;
    if (argc >= 8) {
        yMin = atoi(argv[6]);
        yMax = atoi(argv[7]);
    }
    int era = oreEraFromVersion(argc >= 9 ? argv[8] : "1.18");
    if (era < 0) {
        fprintf(stderr, "unsupported version (1.18 or later)\n");
        return 2;
    }
    static OreVeinNoise noise;
    initOreVeinNoise(&noise, seed);

    static OrePos positions[200000];
    printf("family,x,y,z\n");
    long total = 0;
    for (int cx = cxMin; cx <= cxMax; cx++) {
        for (int cz = czMin; cz <= czMax; cz++) {
            for (int c = 0; c < ORE_CONFIG_COUNT; c++) {
                int count = 0;
                CandidateSink sink;
                memset(&sink, 0, sizeof(sink));
                sink.list = positions;
                sink.listCount = &count;
                sink.listCapacity = (int)(sizeof(positions) / sizeof(positions[0]));
                generateOreConfig(seed, &ORE_CONFIGS[c], era, cx, cz, &sink);
                const char* family = FAMILY_NAMES[ORE_CONFIGS[c].family];
                for (int k = 0; k < count; k++) {
                    if (positions[k].y < yMin || positions[k].y > yMax)
                        continue;
                    printf("%s,%d,%d,%d\n", family, positions[k].x, positions[k].y, positions[k].z);
                    total++;
                }
            }
            if (veins)
                total += emitIronVeinTuff(&noise, cx, cz, yMin, yMax);
        }
    }
    fprintf(stderr, "emitted %ld candidate blocks\n", total);
    return 0;
}
