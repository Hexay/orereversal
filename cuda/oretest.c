// oretest — CPU driver for the ported ore-gen (oregen.h). Mirrors region_dump.exe output
// (CSV family,x,y,z over a chunk box + Y band) so we can diff for bit-exactness. 1.18 only.
// Usage: oretest <seed> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax]
#include <stdio.h>
#include <stdlib.h>
#include "oregen.h"

static const char* FAM_NAME[F_COUNT] = {"tuff", "redstone", "lapis", "gravel", "granite", "copper"};

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax> [yMin yMax]\n", argv[0]);
        return 2;
    }
    uint64_t seed = (uint64_t)strtoll(argv[1], NULL, 10);
    int cxMin = atoi(argv[2]), cxMax = atoi(argv[3]), czMin = atoi(argv[4]), czMax = atoi(argv[5]);
    int yMin = -64, yMax = -1;
    if (argc >= 8) {
        yMin = atoi(argv[6]);
        yMax = atoi(argv[7]);
    }

    static OrePos buf[200000];
    printf("family,x,y,z\n");
    long total = 0;
    for (int cx = cxMin; cx <= cxMax; cx++)
        for (int cz = czMin; cz <= czMax; cz++) {
            for (int ci = 0; ci < ORE_NCFG; ci++) {
                int n = 0;
                OreEmit em;
                memset(&em, 0, sizeof(em));
                em.out = buf;
                em.n = &n;
                em.cap = (int)(sizeof(buf) / sizeof(buf[0]));
                generateOreType(seed, &ORE_CFGS_118[ci], cx, cz, &em);
                const char* fam = FAM_NAME[ORE_CFGS_118[ci].family];
                for (int k = 0; k < n; k++) {
                    if (buf[k].y < yMin || buf[k].y > yMax)
                        continue;
                    printf("%s,%d,%d,%d\n", fam, buf[k].x, buf[k].y, buf[k].z);
                    total++;
                }
            }
        }
    fprintf(stderr, "emitted %ld candidate blocks\n", total);
    return 0;
}
