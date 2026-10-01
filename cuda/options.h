// Command-line options of the matcher.
#ifndef OPTIONS_H
#define OPTIONS_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define MAX_TILE_SIZE 320 // larger tiles risk a Windows TDR reset (a kernel running > 2 s)

struct Options {
    uint64_t seed;
    int chunkMinX, chunkMaxX, chunkMinZ, chunkMaxZ;
    const char* observationPath;
    int tolerance = 0;        // --error
    int absenceTolerance = 0; // --abs-error, see docs/research-log.md P7
    float absenceWeight = 1.0f;
    float minPresenceFraction = 0.5f;
    int tileSize = 256; // chunks per tile side
    int topK = 4096;
    int refineCount = 64;
    bool refine = true;
    bool legacyGenerator = false;
    bool generateAllFamilies = false; // --no-gate: skip family gating (debugging)
};

static const char* USAGE =
    "usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv> [options]\n"
    "  --error E       ore-cell position tolerance in blocks (default 0)\n"
    "  --abs-error A   bare-cell position tolerance in blocks (default 0)\n"
    "  --absw W        weight of each absence hit (default 1.0)\n"
    "  --minfrac F     presence a hypothesis needs to survive pass 1, as a fraction (default 0.5)\n"
    "  --tile T        tile size in chunks, at most 320 (default 256)\n"
    "  --topk K        pass-1 survivors kept across tiles (default 4096)\n"
    "  --refine N      hypotheses re-scored with all families in pass 2 (default 64)\n"
    "  --no-refine     GPU pass only\n"
    "  --legacy-gen    one-thread-per-chunk reference generator\n"
    "  --no-gate       generate every GPU family even when the observation doesn't need it\n";

// Returns false (after printing why) on bad input.
static bool parseOptions(int argc, char** argv, Options& o) {
    if (argc < 7) {
        fprintf(stderr, USAGE, argv[0]);
        return false;
    }
    o.seed = (uint64_t)strtoll(argv[1], NULL, 10);
    o.chunkMinX = atoi(argv[2]);
    o.chunkMaxX = atoi(argv[3]);
    o.chunkMinZ = atoi(argv[4]);
    o.chunkMaxZ = atoi(argv[5]);
    o.observationPath = argv[6];
    for (int i = 7; i < argc; i++) {
        const char* a = argv[i];
        bool hasValue = i + 1 < argc;
        if (!strcmp(a, "--legacy-gen"))
            o.legacyGenerator = true;
        else if (!strcmp(a, "--no-refine"))
            o.refine = false;
        else if (!strcmp(a, "--no-gate"))
            o.generateAllFamilies = true;
        else if (!strcmp(a, "--error") && hasValue)
            o.tolerance = atoi(argv[++i]);
        else if (!strcmp(a, "--abs-error") && hasValue)
            o.absenceTolerance = atoi(argv[++i]);
        else if (!strcmp(a, "--absw") && hasValue)
            o.absenceWeight = (float)atof(argv[++i]);
        else if (!strcmp(a, "--minfrac") && hasValue)
            o.minPresenceFraction = (float)atof(argv[++i]);
        else if (!strcmp(a, "--tile") && hasValue)
            o.tileSize = atoi(argv[++i]);
        else if (!strcmp(a, "--topk") && hasValue)
            o.topK = atoi(argv[++i]);
        else if (!strcmp(a, "--refine") && hasValue)
            o.refineCount = atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown or incomplete option: %s\n", a);
            fprintf(stderr, USAGE, argv[0]);
            return false;
        }
    }
    if (o.tileSize > MAX_TILE_SIZE) {
        fprintf(stderr, "warning: tile>%d risks a Windows TDR kill (kScore launch >2s); clamping to 256\n",
                MAX_TILE_SIZE);
        o.tileSize = 256;
    }
    return true;
}

#endif
