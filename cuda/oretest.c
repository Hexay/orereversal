// oretest — host driver for oregen.h that prints harness/region_dump's CSV format, so the two can be
// diffed for bit-exactness (tests/regress.sh "golden_diff"). 1.18 only.
// Usage: oretest <seed> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax [version]]
#include <stdio.h>
#include <stdlib.h>
#include "oregen.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax [version]]\n", argv[0]);
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
        }
    }
    fprintf(stderr, "emitted %ld candidate blocks\n", total);
    return 0;
}
