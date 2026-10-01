// Console output of the matcher.
#ifndef REPORT_H
#define REPORT_H
#include <algorithm>
#include <cstdio>
#include <vector>
#include "common.h"
#include "observation.h"
#include "options.h"

static void printObservationSummary(const Options& opt, const Observation& obs) {
    long long chunks = (long long)(opt.chunkMaxX - opt.chunkMinX + 1) * (opt.chunkMaxZ - opt.chunkMinZ + 1);
    printf("obs: %zu ore + %zu bare | search %lld chunks | anchor=%s(%d) tile=%d margin=%dch minfrac=%.2f "
           "e=%d ae=%d w=%.1f\n",
           obs.ore.size(), obs.bare.size(), chunks, FAMILY_NAMES[obs.anchorFamily],
           obs.familyCounts[obs.anchorFamily], opt.tileSize, marginChunks(obs), opt.minPresenceFraction,
           opt.tolerance, opt.absenceTolerance, opt.absenceWeight);
}

static void printGenerationGating(const std::vector<int>& configIds, bool allFamilies) {
    printf("gen-gating: %d/%d configs [", (int)configIds.size(), (int)gpuConfigIds().size());
    bool listed[GPU_FAMILY_COUNT] = {false};
    const char* separator = "";
    for (int c : configIds) {
        int family = ORE_CONFIGS_118_HOST[c].family;
        if (!listed[family]) {
            printf("%s%s", separator, FAMILY_NAMES[family]);
            separator = ",";
            listed[family] = true;
        }
    }
    printf("]%s\n", allFamilies ? " (all GPU families: bare-absence)" : " (rare-gated)");
}

// Up to 10 ranked hypotheses, then the margin between the best two. CONFIDENT means that margin is at
// least max(3, 0.3 * oreTotal).
static void printRanking(const char* title, const std::vector<Result>& results, int oreTotal,
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
    if (results.size() >= 2) {
        float margin = results[0].score - results[1].score;
        printf("\ntop_final=%.1f margin=%.1f => %s\n", results[0].score, margin,
               (margin >= std::max(3.0, 0.3 * oreTotal)) ? confidentLabel : "shortlist");
    }
}

#endif
