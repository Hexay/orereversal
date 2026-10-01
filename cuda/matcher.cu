// matcher — find where in a known-seed world an observed ore pattern is.
//   Pass 1 (GPU, gpu_search.cuh): tile the chunk region; per tile generate the four GPU families,
//     hypothesize an origin at every candidate of the observation's rarest family in 8 orientations,
//     and score each by presence and soft absence. Keep the global top-K.
//   Pass 2 (CPU, refine.h): re-score the leaders with all seven families.
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

int main(int argc, char** argv) {
    Options opt;
    if (!parseOptions(argc, argv, opt))
        return 2;
    resolveRegionDumpPath(argv[0]);
    if (opt.refine && !regionDumpExists()) {
        fprintf(stderr, "region_dump not found at %s — run harness/build.sh, or pass --no-refine\n",
                g_regionDumpPath);
        return 1;
    }
    Observation obs;
    if (!loadObservation(opt.observationPath, obs))
        return 1;
    printObservationSummary(opt, obs);

    bool allFamilies = opt.generateAllFamilies || !obs.bare.empty();
    std::vector<int> configIds = selectGpuConfigs(obs, opt.generateAllFamilies);
    printGenerationGating(configIds, allFamilies);

    GpuSearch search(opt, obs, configIds);
    search.printGeneratorInfo();
    std::vector<Result> top;
    SearchStats stats;
    if (!search.run(top, stats))
        return 1;
    printf("scanned %d tiles | %lld anchor candidates | %lld survivors (minfrac pre-filter) | top-K=%zu\n",
           stats.tiles, stats.anchors, stats.survivors, top.size());
    printf("[timing] generate=%.0f ms (setup=%.0f fill=%.0f) | score=%.0f ms | generate/score=%.2f\n",
           stats.msGenerate, stats.msSetup, stats.msGenerate - stats.msSetup, stats.msScore,
           stats.msScore > 0 ? stats.msGenerate / stats.msScore : 0);

    if (!opt.refine) {
        printRanking("(GPU families only: tuff/redstone/lapis/granite)", top, obs.gpuOreCount, "CONFIDENT");
        return 0;
    }
    std::vector<Result> refined = refine(top, obs, opt);
    char title[96];
    snprintf(title, sizeof(title), "(refined top %d with all 7 families incl. gravel/copper/iron)",
             (int)std::min((size_t)opt.refineCount, top.size()));
    printRanking(title, refined, (int)obs.ore.size(), "CONFIDENT (unique)");
    return 0;
}
