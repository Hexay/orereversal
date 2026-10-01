// Pass 1: search the chunk region tile by tile on the GPU and keep the global top-K hypotheses.
#ifndef GPU_SEARCH_CUH
#define GPU_SEARCH_CUH
#include <algorithm>
#include <vector>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "common.h"
#include "generate.cuh"
#include "observation.h"
#include "options.h"
#include "score.cuh"
#include "top_k.h"

#define ANCHOR_CAPACITY     32000000
#define SURVIVOR_CAPACITY   4000000
#define MAX_NODES_PER_CHUNK 512 // worst case surviving vein nodes; sizes the setup/fill scratch
#define MAX_VEINS_PER_CHUNK 24

struct SearchStats {
    int tiles = 0;
    long long anchors = 0, survivors = 0;
    float msGenerate = 0, msSetup = 0, msScore = 0;
};

template <class T> static T* deviceAlloc(size_t count) {
    T* p;
    CUDA_CHECK(cudaMalloc(&p, std::max(count, (size_t)1) * sizeof(T)));
    return p;
}

template <class T> static T* deviceCopy(const std::vector<T>& v) {
    T* p = deviceAlloc<T>(v.size());
    if (!v.empty())
        CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

template <class T> static T readDevice(const T* p) {
    T v;
    CUDA_CHECK(cudaMemcpy(&v, p, sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

static float elapsedMs(cudaEvent_t from, cudaEvent_t to) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, from, to));
    return ms;
}

// Owns every device buffer, sized once for the largest tile and reused across tiles.
class GpuSearch {
  public:
    GpuSearch(const Options& opt, const Observation& obs, const std::vector<int>& configIds)
        : opt_(opt), obs_(obs), configCount_((int)configIds.size()), margin_(marginChunks(obs)) {
        int maxSide = opt.tileSize + 2 * margin_;
        int64_t maxBlocksSide = (int64_t)maxSide * 16;
        maxWordsPerFamily_ = (maxBlocksSide * BAND_HEIGHT * maxBlocksSide + 31) / 32;
        occupancy_ = deviceAlloc<uint32_t>(GPU_FAMILY_COUNT * maxWordsPerFamily_);
        anchors_ = deviceAlloc<int3>(ANCHOR_CAPACITY);
        mortonKeys_ = deviceAlloc<uint32_t>(ANCHOR_CAPACITY);
        anchorCount_ = deviceAlloc<int>(1);
        survivors_ = deviceAlloc<Result>(SURVIVOR_CAPACITY);
        survivorCount_ = deviceAlloc<int>(1);
        nextVein_ = deviceAlloc<int>(1);
        configIds_ = deviceCopy(configIds);
        ore_ = deviceCopy(obs.ore);
        bare_ = deviceCopy(obs.bare);

        cudaDeviceProp props;
        int smCount = cudaGetDeviceProperties(&props, 0) == cudaSuccess ? props.multiProcessorCount : 1;
        fillBlocks_ = smCount * 16; // persistent; surplus warps find the queue empty and exit
        if (!opt.legacyGenerator) {
            int64_t maxChunks = (int64_t)maxSide * maxSide;
            scratch_.nodeCapacity = (int)std::min((int64_t)500000000, maxChunks * MAX_NODES_PER_CHUNK);
            scratch_.veinCapacity = (int)std::min((int64_t)60000000, maxChunks * MAX_VEINS_PER_CHUNK);
            scratch_.nodes = deviceAlloc<VeinNode>(scratch_.nodeCapacity);
            scratch_.veins = deviceAlloc<VeinRecord>(scratch_.veinCapacity);
            scratch_.nodeCount = deviceAlloc<int>(1);
            scratch_.veinCount = deviceAlloc<int>(1);
        }
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&setupDone_));
        CUDA_CHECK(cudaEventCreate(&generated_));
        CUDA_CHECK(cudaEventCreate(&scored_));
    }

    void printGeneratorInfo() const {
        if (opt_.legacyGenerator)
            printf("generator: legacy (1 thread/chunk, bit-exact reference)\n");
        else
            printf("generator: two-kernel (kSetupVeins+kFillVeins), node capacity=%dM vein capacity=%dM, "
                   "%d fill blocks x %d\n",
                   scratch_.nodeCapacity / 1000000, scratch_.veinCapacity / 1000000, fillBlocks_,
                   FILL_BLOCK_SIZE);
    }

    // Returns false if the vein scratch overflowed (after printing why).
    bool run(std::vector<Result>& top, SearchStats& stats) {
        for (int tileX = opt_.chunkMinX; tileX <= opt_.chunkMaxX; tileX += opt_.tileSize)
            for (int tileZ = opt_.chunkMinZ; tileZ <= opt_.chunkMaxZ; tileZ += opt_.tileSize) {
                stats.tiles++;
                if (!searchTile(tileX, tileZ, top, stats))
                    return false;
            }
        return true;
    }

  private:
    bool searchTile(int tileX, int tileZ, std::vector<Result>& top, SearchStats& stats) {
        int tileEndX = std::min(tileX + opt_.tileSize - 1, opt_.chunkMaxX);
        int tileEndZ = std::min(tileZ + opt_.tileSize - 1, opt_.chunkMaxZ);
        // Generate a margin around the tile, but only take anchors from the tile itself, so neighbouring
        // tiles never produce the same hypothesis.
        ChunkGrid chunks = {tileX - margin_, tileZ - margin_, tileEndX - tileX + 1 + 2 * margin_,
                            tileEndZ - tileZ + 1 + 2 * margin_};
        OccupancyGrid grid;
        grid.words = occupancy_;
        grid.originX = chunks.originX * 16;
        grid.originZ = chunks.originZ * 16;
        grid.sizeX = chunks.countX * 16;
        grid.sizeZ = chunks.countZ * 16;
        grid.wordsPerFamily = ((int64_t)grid.sizeX * BAND_HEIGHT * grid.sizeZ + 31) / 32;
        AnchorSink anchors = {anchors_,           anchorCount_, ANCHOR_CAPACITY,   tileX * 16,
                              tileEndX * 16 + 15, tileZ * 16,   tileEndZ * 16 + 15};

        CUDA_CHECK(cudaMemset(occupancy_, 0, GPU_FAMILY_COUNT * grid.wordsPerFamily * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(anchorCount_, 0, sizeof(int)));
        CUDA_CHECK(cudaEventRecord(start_));
        if (opt_.legacyGenerator) {
            int chunkCount = chunks.countX * chunks.countZ;
            kGenerateLegacy<<<(chunkCount + 63) / 64, 64>>>(opt_.seed, chunks, grid, anchors,
                                                            obs_.anchorFamily, configIds_, configCount_);
        } else if (!generateTwoKernel(chunks, grid, anchors, stats)) {
            return false;
        }
        CUDA_CHECK(cudaEventRecord(generated_));
        CUDA_CHECK(cudaEventSynchronize(generated_));
        stats.msGenerate += elapsedMs(start_, generated_);

        int anchorCount = std::min(readDevice(anchorCount_), ANCHOR_CAPACITY);
        stats.anchors += anchorCount;
        if (!anchorCount)
            return true;
        std::vector<Result> survivors = score(grid, anchorCount, stats);
        stats.survivors += survivors.size();
        if (!survivors.empty())
            mergeTopK(top, survivors, 2 * opt_.tolerance + 1, opt_.topK);
        return true;
    }

    bool generateTwoKernel(const ChunkGrid& chunks, const OccupancyGrid& grid, const AnchorSink& anchors,
                           SearchStats& stats) {
        CUDA_CHECK(cudaMemset(scratch_.nodeCount, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(scratch_.veinCount, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(nextVein_, 0, sizeof(int)));
        int64_t items = (int64_t)chunks.countX * chunks.countZ * configCount_;
        kSetupVeins<<<(int)((items + 255) / 256), 256>>>(opt_.seed, chunks, obs_.anchorFamily, configIds_,
                                                         configCount_, scratch_);
        CUDA_CHECK(cudaEventRecord(setupDone_));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        stats.msSetup += elapsedMs(start_, setupDone_);
        int veinCount = readDevice(scratch_.veinCount), nodeCount = readDevice(scratch_.nodeCount);
        if (veinCount > scratch_.veinCapacity || nodeCount > scratch_.nodeCapacity) {
            fprintf(stderr, "scratch overflow (veins %d/%d nodes %d/%d) — raise caps\n", veinCount,
                    scratch_.veinCapacity, nodeCount, scratch_.nodeCapacity);
            return false;
        }
        kFillVeins<<<fillBlocks_, FILL_BLOCK_SIZE>>>(grid, anchors, scratch_.nodes, scratch_.veins, veinCount,
                                                     nextVein_);
        return true;
    }

    std::vector<Result> score(const OccupancyGrid& grid, int anchorCount, SearchStats& stats) {
        ScoringInput in;
        in.ore = ore_;
        in.oreCount = obs_.gpuOreCount;
        in.bare = bare_;
        in.bareCount = (int)obs_.bare.size();
        in.anchorX = obs_.anchorCell.x;
        in.anchorY = obs_.anchorCell.y;
        in.anchorZ = obs_.anchorCell.z;
        in.tolerance = opt_.tolerance;
        in.absenceTolerance = opt_.absenceTolerance;
        in.absenceWeight = opt_.absenceWeight;
        in.minPresence = (int)(opt_.minPresenceFraction * obs_.gpuOreCount);
        ResultSink out = {survivors_, survivorCount_, SURVIVOR_CAPACITY};

        CUDA_CHECK(cudaMemset(survivorCount_, 0, sizeof(int)));
        CUDA_CHECK(cudaEventRecord(generated_));
        kMortonKeys<<<(anchorCount + 255) / 256, 256>>>(anchors_, anchorCount, grid.originX, grid.originZ,
                                                        mortonKeys_);
        thrust::sort_by_key(thrust::device, mortonKeys_, mortonKeys_ + anchorCount, anchors_);
        int64_t hypotheses = (int64_t)anchorCount * 8;
        kScoreHypotheses<<<(int)((hypotheses + 127) / 128), 128>>>(anchors_, anchorCount, grid, in, out);
        CUDA_CHECK(cudaEventRecord(scored_));
        CUDA_CHECK(cudaEventSynchronize(scored_));
        stats.msScore += elapsedMs(generated_, scored_);

        int count = std::min(readDevice(survivorCount_), SURVIVOR_CAPACITY);
        std::vector<Result> survivors(count);
        if (count)
            CUDA_CHECK(
                cudaMemcpy(survivors.data(), survivors_, count * sizeof(Result), cudaMemcpyDeviceToHost));
        return survivors;
    }

    const Options& opt_;
    const Observation& obs_;
    int configCount_, margin_, fillBlocks_;
    int64_t maxWordsPerFamily_;
    uint32_t *occupancy_, *mortonKeys_;
    int3* anchors_;
    int *anchorCount_, *survivorCount_, *nextVein_, *configIds_;
    Result* survivors_;
    ObsCell *ore_, *bare_;
    VeinScratch scratch_ = {};
    cudaEvent_t start_, setupDone_, generated_, scored_;
};

#endif
