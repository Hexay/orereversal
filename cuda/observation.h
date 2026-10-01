// Loading an observation CSV (docs/observation-format.md) and deriving what the search needs from it.
#ifndef OBSERVATION_H
#define OBSERVATION_H
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "common.h"
#include "region_dump.h"

struct Observation {
    std::vector<ObsCell> ore; // GPU families first: pass 1 only scores ore[0 .. gpuOreCount)
    std::vector<ObsCell> bare;
    int familyCounts[F_COUNT] = {0};
    int gpuOreCount = 0;
    int anchorFamily = -1; // rarest GPU family observed; its candidates seed the hypotheses
    ObsCell anchorCell;
    int maxExtent = 1; // largest |x| or |z| of any cell
};

// Returns false (after printing why) if the file can't be read or has no GPU-family ore.
static bool loadObservation(const char* path, Observation& obs) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        char name[32];
        int x, y, z;
        if (sscanf(line, "%31[^,],%d,%d,%d", name, &x, &y, &z) != 4)
            continue; // also skips the header
        if (!strcmp(name, "bare")) {
            obs.bare.push_back({x, y, z, -1});
            continue;
        }
        int family = familyFromName(name);
        if (family >= 0)
            obs.ore.push_back({x, y, z, family});
    }
    fclose(f);

    std::stable_partition(obs.ore.begin(), obs.ore.end(),
                          [](const ObsCell& c) { return c.family < GPU_FAMILY_COUNT; });
    for (const ObsCell& c : obs.ore) {
        obs.familyCounts[c.family]++;
        if (c.family < GPU_FAMILY_COUNT)
            obs.gpuOreCount++;
    }
    int fewest = 1 << 30;
    for (int family = 0; family < GPU_FAMILY_COUNT; family++)
        if (obs.familyCounts[family] > 0 && obs.familyCounts[family] < fewest) {
            fewest = obs.familyCounts[family];
            obs.anchorFamily = family;
        }
    if (obs.anchorFamily < 0) {
        fprintf(stderr, "no GPU-generated ore family in observation\n");
        return false;
    }
    for (const ObsCell& c : obs.ore)
        if (c.family == obs.anchorFamily) {
            obs.anchorCell = c;
            break;
        }
    for (const ObsCell& c : obs.ore)
        obs.maxExtent = std::max(obs.maxExtent, std::max(abs(c.x), abs(c.z)));
    for (const ObsCell& c : obs.bare)
        obs.maxExtent = std::max(obs.maxExtent, std::max(abs(c.x), abs(c.z)));
    return true;
}

// Chunks of margin around a tile so every vein reaching a hypothesis inside it is generated.
static inline int marginChunks(const Observation& obs) {
    return obs.maxExtent / 16 + 2;
}

// The GPU configs to generate. Presence only needs the observed families, but absence checks every GPU
// family, so any bare cell requires all of them. Configs seed their RNG independently, so skipping one
// leaves the others bit-identical.
static std::vector<int> selectGpuConfigs(const Observation& obs, bool generateAll) {
    std::vector<int> ids;
    for (int c : gpuConfigIds())
        if (generateAll || !obs.bare.empty() || obs.familyCounts[ORE_CONFIGS_118_HOST[c].family] > 0)
            ids.push_back(c);
    return ids;
}

#endif
