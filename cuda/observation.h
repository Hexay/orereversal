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
    int anchorFamily = -1; // its candidates seed the hypotheses
    ObsCell anchorCell;    // the observed cell of anchorFamily those candidates are aligned to
    int maxExtent = 1;     // largest |x| or |z| of any cell
    int anchorReach = 0;   // largest x/z (Chebyshev) distance from anchorCell to any cell, in any orientation
    int footprint = 0;     // horizontal size: max(x span, z span) over all cells
};

// Sets the anchor family (`family`, or the rarest observed GPU family if -1) and its first observed cell.
// Returns false if the family wasn't observed.
static bool chooseAnchor(Observation& obs, int family) {
    if (family < 0) {
        int fewest = 1 << 30;
        for (int f = 0; f < GPU_FAMILY_COUNT; f++)
            if (obs.familyCounts[f] > 0 && obs.familyCounts[f] < fewest) {
                fewest = obs.familyCounts[f];
                family = f;
            }
    }
    if (family < 0 || obs.familyCounts[family] == 0) {
        fprintf(stderr, "no observed cell of the anchor family\n");
        return false;
    }
    obs.anchorFamily = family;
    obs.anchorCell = *std::find_if(obs.ore.begin(), obs.ore.end(),
                                   [&](const ObsCell& c) { return c.family == family; });
    obs.anchorReach = 0;
    for (const std::vector<ObsCell>* list : {&obs.ore, &obs.bare})
        for (const ObsCell& c : *list)
            obs.anchorReach = std::max(obs.anchorReach,
                                       std::max(abs(c.x - obs.anchorCell.x), abs(c.z - obs.anchorCell.z)));
    return true;
}

// The family to re-anchor on when a lapis-anchored search isn't confident, or -1. In low terrain the GPU
// generator misses ~20% of real lapis (no surface gate, docs/research-log.md P9), and if the anchor cell is
// one of those the true origin is never hypothesized. Redstone and granite are unaffected; tuff would flood
// the anchor buffer.
static int retryAnchorFamily(const Observation& obs) {
    if (obs.anchorFamily != F_LAPIS)
        return -1;
    int best = -1;
    for (int family : {F_REDSTONE, F_GRANITE})
        if (obs.familyCounts[family] > 0 && (best < 0 || obs.familyCounts[family] < obs.familyCounts[best]))
            best = family;
    return best;
}

// Returns false (after printing why) if the file can't be read or has no usable anchor.
static bool loadObservation(const char* path, int anchorFamily, Observation& obs) {
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
    if (!obs.gpuOreCount) {
        fprintf(stderr, "no GPU-generated ore family in observation\n");
        return false;
    }
    if (!chooseAnchor(obs, anchorFamily))
        return false;
    int minX = obs.ore[0].x, maxX = minX, minZ = obs.ore[0].z, maxZ = minZ;
    for (const std::vector<ObsCell>* cells : {&obs.ore, &obs.bare})
        for (const ObsCell& c : *cells) {
            obs.maxExtent = std::max(obs.maxExtent, std::max(abs(c.x), abs(c.z)));
            minX = std::min(minX, c.x);
            maxX = std::max(maxX, c.x);
            minZ = std::min(minZ, c.z);
            maxZ = std::max(maxZ, c.z);
        }
    obs.footprint = std::max(maxX - minX, maxZ - minZ);
    return true;
}

// Chunks of margin a refine window needs around the origin.
static inline int marginChunks(const Observation& obs) {
    return obs.maxExtent / 16 + 2;
}

// Chunks of margin a GPU tile needs: hypotheses are anchored inside the tile and probe up to anchorReach
// blocks away, plus one chunk because veins spill up to 13 blocks out of their own chunk.
static inline int tileMarginChunks(const Observation& obs) {
    return (obs.anchorReach + 15) / 16 + 1;
}

// The GPU configs to generate. Presence only needs the observed families, but absence checks every GPU
// family, so any bare cell requires all of them. Configs seed their RNG independently, so skipping one
// leaves the others bit-identical.
static std::vector<int> selectGpuConfigs(const Observation& obs, bool generateAll) {
    std::vector<int> ids;
    for (int c : gpuConfigIds())
        if (generateAll || !obs.bare.empty() || obs.familyCounts[ORE_CONFIGS_HOST[c].family] > 0)
            ids.push_back(c);
    return ids;
}

#endif
