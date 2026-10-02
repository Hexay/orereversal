// matcher — find where in a known-seed world an observed ore pattern is.
//   Pass 1 (GPU, gpu_search.cuh): tile the chunk region; per tile generate the four GPU families,
//     hypothesize an origin at every candidate of the observation's rarest family in 8 orientations,
//     and score each by presence and soft absence. Keep the global top-K.
//   Pass 2 (CPU, refine.h): re-score the leaders with all eight families.
//   A lapis-anchored result that isn't confident is retried on another anchor (retryAnchorFamily).
// Usage: see options.h, or run without arguments. Design notes: cuda/README.md.
#include <cstdio>
#include <vector>
#include "common.h"
#include "gpu_search.cuh"
#include "observation.h"
#include "options.h"
#include "refine.h"
#include "region_dump.h"
#include "report.h"

// Runs pass 1 anchored on obs.anchorFamily and merges its survivors into `top`. Returns false on failure.
static bool searchPass1(const Options& opt, const Observation& obs, std::vector<Result>& top) {
    bool allFamilies = opt.noFamilyGating || !obs.bare.empty();
    std::vector<int> configIds = selectGpuConfigs(obs, opt.noFamilyGating);
    printGenerationGating(configIds, allFamilies);

    GpuSearch search(opt, obs, configIds);
    search.printGeneratorInfo();
    std::vector<Result> found;
    SearchStats stats;
    if (!search.run(found, stats))
        return false;
    printf("scanned %d tiles | %lld anchor candidates | %lld survivors (minfrac pre-filter) | top-K=%zu\n",
           stats.tiles, stats.anchors, stats.survivors, found.size());
    printf("[timing] generate=%.0f ms (setup=%.0f fill=%.0f) | score=%.0f ms | generate/score=%.2f\n",
           stats.msGenerate, stats.msSetup, stats.msGenerate - stats.msSetup, stats.msScore,
           stats.msScore > 0 ? stats.msGenerate / stats.msScore : 0);
    if (stats.anchorsDropped || stats.survivorsDropped)
        fprintf(stderr,
                "warning: buffers full, dropped %lld anchors and %lld survivors; the true location may be "
                "missing (use a smaller --tile or a higher --minfrac)\n",
                stats.anchorsDropped, stats.survivorsDropped);
    mergeTopK(top, found, 2 * opt.tolerance + 1, opt.topK);
    return true;
}

// Pass 2, or pass 1's ranking as-is under --no-refine. Returns false (after printing why) on failure.
static bool rankResults(const Options& opt, const Observation& obs, const std::vector<Result>& top,
                        int separation, std::vector<Result>& results) {
    if (!opt.refine) {
        results = top;
        return true;
    }
    if (refine(top, obs, opt, separation, results))
        return true;
    fprintf(stderr, "refine: region_dump failed; run it by hand (%s) to see why\n", g_regionDumpPath);
    return false;
}

// region_dump rejects what the matcher can't refine (e.g. an unknown version), so fail before pass 1.
static bool checkRegionDump(const Options& opt) {
    if (!regionDumpExists()) {
        fprintf(stderr, "region_dump not found at %s — run harness/build.sh, or pass --no-refine\n",
                g_regionDumpPath);
        return false;
    }
    if (!runRegionDump(opt.seed, opt.version, 0, 0, 0, 0, "tuff", [](int, const char*, int, int, int, int) {})) {
        fprintf(stderr, "region_dump %s rejected version '%s'\n", g_regionDumpPath, opt.version);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parseOptions(argc, argv, opt))
        return 2;
    resolveRegionDumpPath(argv[0]);
    if (opt.refine && !checkRegionDump(opt))
        return 1;
    Observation obs;
    if (!loadObservation(opt.observationPath, opt.anchorFamily, obs))
        return 1;
    printObservationSummary(opt, obs);

    std::vector<Result> top;
    if (!searchPass1(opt, obs, top))
        return 1;
    int separation = std::max(2 * opt.tolerance + 1, obs.footprint); // see printRanking
    int oreTotal = opt.refine ? (int)obs.ore.size() : obs.gpuOreCount;
    std::vector<Result> results;
    if (!rankResults(opt, obs, top, separation, results))
        return 1;

    int retryFamily = opt.anchorFamily < 0 ? retryAnchorFamily(obs) : -1;
    if (retryFamily >= 0 && !isConfident(results, oreTotal, separation)) {
        printf("\nnot confident with a lapis anchor: retrying pass 1 anchored on %s (docs/research-log.md P9)\n",
               FAMILY_NAMES[retryFamily]);
        chooseAnchor(obs, retryFamily);
        if (!searchPass1(opt, obs, top) || !rankResults(opt, obs, top, separation, results))
            return 1;
    }

    if (!opt.refine) {
        printRanking("(GPU families only: tuff/redstone/lapis/granite)", results, oreTotal, separation,
                     "CONFIDENT");
        return 0;
    }
    char title[96];
    snprintf(title, sizeof(title), "(refined top %d with all 8 families incl. gravel/copper/iron/diamond)",
             (int)std::min((size_t)opt.refineCount, top.size()));
    printRanking(title, results, oreTotal, separation, "CONFIDENT (unique)");
    return 0;
}
