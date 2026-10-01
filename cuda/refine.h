// Pass 2 on the CPU: re-score the best pass-1 hypotheses with all seven families. Gravel, copper and
// iron come from region_dump, since the GPU doesn't generate them.
#ifndef REFINE_H
#define REFINE_H
#include <algorithm>
#include <unordered_set>
#include <vector>
#include "common.h"
#include "observation.h"
#include "options.h"
#include "region_dump.h"

typedef std::unordered_set<uint64_t> BlockSet;

// Exact key for any block in the world (|x|, |z| < 2^25 covers +-30M; y - BAND_MIN_Y < 2^7).
static inline uint64_t blockKey(int x, int y, int z) {
    return ((uint64_t)(uint32_t)(x + (1 << 25)) << 33) | ((uint64_t)(uint32_t)(y - BAND_MIN_Y) << 26) |
           (uint32_t)(z + (1 << 25));
}

static bool containsNear(const BlockSet& s, int x, int y, int z, int tolerance) {
    for (int dx = -tolerance; dx <= tolerance; dx++)
        for (int dy = -tolerance; dy <= tolerance; dy++)
            for (int dz = -tolerance; dz <= tolerance; dz++)
                if (s.count(blockKey(x + dx, y + dy, z + dz)))
                    return true;
    return false;
}

// Candidate blocks of every family over the chunk box, deepslate band only.
static void generateAllFamilies(uint64_t seed, int chunkMinX, int chunkMaxX, int chunkMinZ, int chunkMaxZ,
                                BlockSet* candidates) {
    std::vector<OrePos> positions(200000);
    for (int cx = chunkMinX; cx <= chunkMaxX; cx++)
        for (int cz = chunkMinZ; cz <= chunkMaxZ; cz++)
            for (int c : gpuConfigIds()) {
                const OreConfig* config = &ORE_CONFIGS_118_HOST[c];
                int count = 0;
                CandidateSink sink = {};
                sink.list = positions.data();
                sink.listCount = &count;
                sink.listCapacity = (int)positions.size();
                generateOreConfig(seed, config, cx, cz, &sink);
                for (int k = 0; k < count; k++)
                    if (inBand(positions[k].y))
                        candidates[config->family].insert(
                            blockKey(positions[k].x, positions[k].y, positions[k].z));
            }
    runRegionDump(seed, chunkMinX, chunkMaxX, chunkMinZ, chunkMaxZ, "gravel copper iron",
                  [&](int family, int x, int y, int z) { candidates[family].insert(blockKey(x, y, z)); });
}

static Result rescore(const Result& r, const Observation& obs, const BlockSet* candidates,
                      const Options& opt) {
    Result out = r;
    out.present = 0;
    for (const ObsCell& c : obs.ore) {
        int dx, dz;
        orientXZ(c.x, c.z, r.rotation, r.mirror, &dx, &dz);
        if (containsNear(candidates[c.family], r.originX + dx, r.originY + c.y, r.originZ + dz,
                         opt.tolerance))
            out.present++;
    }
    out.absenceHits = 0;
    for (const ObsCell& c : obs.bare) {
        int dx, dz;
        orientXZ(c.x, c.z, r.rotation, r.mirror, &dx, &dz);
        for (int family = 0; family < F_COUNT; family++)
            if (containsNear(candidates[family], r.originX + dx, r.originY + c.y, r.originZ + dz,
                             opt.absenceTolerance)) {
                out.absenceHits++;
                break;
            }
    }
    out.score = out.present - opt.absenceWeight * out.absenceHits;
    return out;
}

// Re-scores the leading hypotheses of `top` (sorted best first) and returns them sorted best first.
static std::vector<Result> refine(const std::vector<Result>& top, const Observation& obs,
                                  const Options& opt) {
    // The refine-only families can add at most this much to any hypothesis, so hypotheses further than
    // this below the best can't overtake it. Always refine at least 8 so a margin can be shown.
    int maxGain = obs.familyCounts[F_GRAVEL] + obs.familyCounts[F_COPPER] + obs.familyCounts[F_IRON];
    int count = std::min((int)top.size(), opt.refineCount);
    float best = top.empty() ? 0 : top[0].score;
    for (int t = 8; t < count; t++)
        if (top[t].score < best - (float)maxGain) {
            count = t;
            break;
        }

    int margin = marginChunks(obs);
    std::vector<Result> refined(count);
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < count; t++) {
        const Result& r = top[t];
        BlockSet candidates[F_COUNT];
        generateAllFamilies(opt.seed, ((r.originX - obs.maxExtent) >> 4) - margin,
                            ((r.originX + obs.maxExtent) >> 4) + margin,
                            ((r.originZ - obs.maxExtent) >> 4) - margin,
                            ((r.originZ + obs.maxExtent) >> 4) + margin, candidates);
        refined[t] = rescore(r, obs, candidates, opt);
    }
    std::sort(refined.begin(), refined.end(), rankedBefore);
    return refined;
}

#endif
