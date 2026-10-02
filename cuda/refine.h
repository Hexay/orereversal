// Pass 2 on the CPU: re-score the best pass-1 hypotheses with every family. Gravel, copper, iron, buried
// diamond and ore veins come from region_dump, since the GPU doesn't generate them.
#ifndef REFINE_H
#define REFINE_H
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common.h"
#include "observation.h"
#include "options.h"
#include "region_dump.h"
#include "top_k.h"

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

// One surface-gate-sensitive config in one chunk: alternative outcomes, exactly one of them real
// (harness/ore_branch.h). Variant 0 is cubiomes' own guess.
struct VariantGroup {
    int family;
    std::vector<std::vector<OrePos>> variants;
};

// Candidate blocks of every family over the chunk box, deepslate band only. Gate-sensitive gravel and
// copper come as groups of alternatives instead.
static void generateAllFamilies(const Options& opt, int chunkMinX, int chunkMaxX, int chunkMinZ,
                                int chunkMaxZ, BlockSet* candidates, std::vector<VariantGroup>& groups) {
    std::vector<OrePos> positions(200000);
    for (int cx = chunkMinX; cx <= chunkMaxX; cx++)
        for (int cz = chunkMinZ; cz <= chunkMaxZ; cz++)
            for (int c : gpuConfigIds()) {
                const OreConfig* config = &ORE_CONFIGS_HOST[c];
                int count = 0;
                CandidateSink sink = {};
                sink.list = positions.data();
                sink.listCount = &count;
                sink.listCapacity = (int)positions.size();
                generateOreConfig(opt.seed, config, opt.era, cx, cz, &sink);
                for (int k = 0; k < count; k++)
                    if (inBand(positions[k].y))
                        candidates[config->family].insert(
                            blockKey(positions[k].x, positions[k].y, positions[k].z));
            }
    std::unordered_map<std::string, int> groupIndex;
    runRegionDump(opt.seed, opt.version, chunkMinX, chunkMaxX, chunkMinZ, chunkMaxZ,
                  "gravel copper iron diamond +veins +branch",
                  [&](int family, const char* group, int variant, int x, int y, int z) {
                      if (!group[0]) {
                          candidates[family].insert(blockKey(x, y, z));
                          return;
                      }
                      auto it = groupIndex.emplace(group, (int)groups.size()).first;
                      if (it->second == (int)groups.size())
                          groups.push_back({family, {}});
                      VariantGroup& g = groups[it->second];
                      if ((int)g.variants.size() <= variant)
                          g.variants.resize(variant + 1);
                      if (inBand(y))
                          g.variants[variant].push_back({x, y, z});
                  });
}

static bool anyOreAt(const BlockSet* candidates, int x, int y, int z) {
    for (int family = 0; family < F_COUNT; family++)
        if (candidates[family].count(blockKey(x, y, z)))
            return true;
    return false;
}

// Absence with tolerance is an erosion: a bare cell only counts against a hypothesis if ore is predicted
// at every position within +-tolerance, so a misread bare cell next to ore doesn't (docs/research-log.md P8).
static bool oreThroughout(const BlockSet* candidates, int x, int y, int z, int tolerance) {
    for (int dx = -tolerance; dx <= tolerance; dx++)
        for (int dy = -tolerance; dy <= tolerance; dy++)
            for (int dz = -tolerance; dz <= tolerance; dz++)
                if (!anyOreAt(candidates, x + dx, y + dy, z + dz))
                    return false;
    return true;
}

static Result scoreHypothesis(const Result& r, const Observation& obs, const BlockSet* candidates,
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
        if (oreThroughout(candidates, r.originX + dx, r.originY + c.y, r.originZ + dz, opt.absenceTolerance))
            out.absenceHits++;
    }
    out.score = out.present - opt.absenceWeight * out.absenceHits;
    return out;
}

static bool touchesFootprint(const std::vector<OrePos>& blocks, const Result& r, int reach) {
    for (const OrePos& p : blocks)
        if (abs(p.x - r.originX) <= reach && abs(p.z - r.originZ) <= reach)
            return true;
    return false;
}

// For each variant group, adds to `candidates` the variant that scores best for r (cubiomes' guess on
// ties). Groups that don't reach the observation's footprint just get cubiomes' guess. Every hypothesis
// gets the same freedom, so this doesn't favour the true location by construction.
static void chooseVariants(const Result& r, const Observation& obs, BlockSet* candidates,
                           const std::vector<VariantGroup>& groups, const Options& opt) {
    int reach = obs.maxExtent + std::max(opt.tolerance, opt.absenceTolerance) + 1;
    for (const VariantGroup& g : groups) {
        int best = 0;
        bool relevant = std::any_of(g.variants.begin(), g.variants.end(), [&](const std::vector<OrePos>& v) {
            return touchesFootprint(v, r, reach);
        });
        if (relevant) {
            float bestScore = -1e30f;
            for (int v = 0; v < (int)g.variants.size(); v++) {
                std::vector<uint64_t> added;
                for (const OrePos& p : g.variants[v])
                    if (candidates[g.family].insert(blockKey(p.x, p.y, p.z)).second)
                        added.push_back(blockKey(p.x, p.y, p.z));
                float score = scoreHypothesis(r, obs, candidates, opt).score;
                for (uint64_t key : added)
                    candidates[g.family].erase(key);
                if (score > bestScore) {
                    bestScore = score;
                    best = v;
                }
            }
        }
        for (const OrePos& p : g.variants[best])
            candidates[g.family].insert(blockKey(p.x, p.y, p.z));
    }
}

// With --error e the anchor cell is itself off by up to e, so the hypothesis' origin is too. Re-scoring
// every origin within +-e (same orientation) and keeping the best recovers the exact origin.
static Result rescore(const Result& r, const Observation& obs, const BlockSet* candidates,
                      const Options& opt) {
    Result best = scoreHypothesis(r, obs, candidates, opt);
    int e = opt.tolerance;
    for (int dx = -e; dx <= e; dx++)
        for (int dy = -e; dy <= e; dy++)
            for (int dz = -e; dz <= e; dz++) {
                Result shifted = r;
                shifted.originX += dx;
                shifted.originY += dy;
                shifted.originZ += dz;
                Result scored = scoreHypothesis(shifted, obs, candidates, opt);
                if (rankedBefore(scored, best))
                    best = scored;
            }
    return best;
}

// Re-scores the leading hypotheses of `top` (sorted best first) and returns them sorted best first. Also
// re-scores the best hypothesis more than `separation` blocks from the leader, so the report always has a
// genuine competitor to measure the margin against.
static std::vector<Result> refine(const std::vector<Result>& top, const Observation& obs, const Options& opt,
                                  int separation) {
    // The refine-only families can add at most this much to any hypothesis, so hypotheses further than
    // this below the best can't overtake it. Always refine at least 8.
    int maxGain = 0;
    for (int family = GPU_FAMILY_COUNT; family < F_COUNT; family++)
        maxGain += obs.familyCounts[family];
    int count = std::min((int)top.size(), opt.refineCount);
    float best = top.empty() ? 0 : top[0].score;
    for (int t = 8; t < count; t++)
        if (top[t].score < best - (float)maxGain) {
            count = t;
            break;
        }
    std::vector<Result> chosen(top.begin(), top.begin() + count);
    for (const Result& r : top)
        if (!top.empty() && chebyshev(r, top[0]) > separation) {
            if (std::none_of(chosen.begin(), chosen.end(),
                             [&](const Result& c) { return chebyshev(c, r) == 0; }))
                chosen.push_back(r);
            break;
        }

    int margin = marginChunks(obs);
    std::vector<Result> refined(chosen.size());
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < (int)chosen.size(); t++) {
        const Result& r = chosen[t];
        BlockSet candidates[F_COUNT];
        std::vector<VariantGroup> groups;
        generateAllFamilies(opt, ((r.originX - obs.maxExtent) >> 4) - margin,
                            ((r.originX + obs.maxExtent) >> 4) + margin,
                            ((r.originZ - obs.maxExtent) >> 4) - margin,
                            ((r.originZ + obs.maxExtent) >> 4) + margin, candidates, groups);
        chooseVariants(r, obs, candidates, groups, opt);
        refined[t] = rescore(r, obs, candidates, opt);
    }
    std::sort(refined.begin(), refined.end(), rankedBefore);
    // Re-centring can move two hypotheses onto (nearly) the same origin; keep the better one.
    std::vector<Result> distinct;
    for (const Result& r : refined)
        if (std::none_of(distinct.begin(), distinct.end(),
                         [&](const Result& d) { return chebyshev(d, r) <= 2 * opt.tolerance + 1; }))
            distinct.push_back(r);
    return distinct;
}

#endif
