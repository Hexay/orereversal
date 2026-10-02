// Console output of the matcher.
#ifndef REPORT_H
#define REPORT_H
#include <algorithm>
#include <cstdio>
#include <vector>
#include "common.h"
#include "observation.h"
#include "options.h"
#include "top_k.h"

static void printObservationSummary(const Options& opt, const Observation& obs) {
    long long chunks = (long long)(opt.chunkMaxX - opt.chunkMinX + 1) * (opt.chunkMaxZ - opt.chunkMinZ + 1);
    printf("obs: %zu ore + %zu bare | mc=%s | search %lld chunks | anchor=%s(%d) tile=%d margin=%dch "
           "minfrac=%.2f e=%d ae=%d w=%.1f\n",
           obs.ore.size(), obs.bare.size(), opt.version, chunks, FAMILY_NAMES[obs.anchorFamily],
           obs.familyCounts[obs.anchorFamily], opt.tileSize, tileMarginChunks(obs), opt.minPresenceFraction,
           opt.tolerance, opt.absenceTolerance, opt.absenceWeight);
}

static void printGenerationGating(const std::vector<int>& configIds, bool allFamilies) {
    printf("gen-gating: %d/%d configs [", (int)configIds.size(), (int)gpuConfigIds().size());
    bool listed[GPU_FAMILY_COUNT] = {false};
    const char* separator = "";
    for (int c : configIds) {
        int family = ORE_CONFIGS_HOST[c].family;
        if (!listed[family]) {
            printf("%s%s", separator, FAMILY_NAMES[family]);
            separator = ",";
            listed[family] = true;
        }
    }
    printf("]%s\n", allFamilies ? " (all GPU families: bare-absence)" : " (rare-gated)");
}

// Up to 10 ranked hypotheses, then the margin between the best one and the best one more than
// `separation` blocks from it: a copy of the winner shifted by a few blocks is the same place, not a
// competing location. CONFIDENT means that margin is at least max(3, 0.3 * oreTotal).
static void printRanking(const char* title, const std::vector<Result>& results, int oreTotal, int separation,
                         const char* confidentLabel) {
    printf("\n%s\n", title);
    printf("%4s %22s %12s %7s %10s %8s %9s\n", "rank", "world_origin", "chunk", "orient", "present", "absH",
           "final");
    for (size_t i = 0; i < results.size() && i < 10; i++) {
        const Result& r = results[i];
        char origin[40], chunk[32];
        snprintf(origin, sizeof(origin), "(%d, %d, %d)", r.originX, r.originY, r.originZ);
        snprintf(chunk, sizeof(chunk), "(%d, %d)", r.originX >> 4, r.originZ >> 4);
        printf("%4zu %22s %12s    r%dm%d %d/%d %8d %9.1f\n", i + 1, origin, chunk, r.rotation, r.mirror,
               r.present, oreTotal, r.absenceHits, r.score);
    }
    if (results.empty())
        return;
    const Result* competitor = nullptr;
    for (const Result& r : results)
        if (chebyshev(r, results[0]) > separation) {
            competitor = &r; // results are best first
            break;
        }
    if (!competitor) {
        printf("\ntop_final=%.1f margin=n/a (no other surviving hypothesis >%d blocks away)\n",
               results[0].score, separation);
        return;
    }
    // Pass 1 drops hypotheses below --minfrac presence, so the competitor is the best *survivor*.
    float margin = results[0].score - competitor->score;
    printf("\ntop_final=%.1f margin=%.1f (vs best surviving hypothesis >%d blocks away) => %s\n",
           results[0].score, margin, separation,
           (margin >= std::max(3.0, 0.3 * oreTotal)) ? confidentLabel : "shortlist");
}

#endif
