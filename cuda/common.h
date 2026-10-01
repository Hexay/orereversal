// Definitions shared by the matcher's host and device code.
#ifndef COMMON_H
#define COMMON_H
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "oregen.h"

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        cudaError_t err_ = (call);                                                                           \
        if (err_ != cudaSuccess) {                                                                           \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));                \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

// One observed block, relative to the observation's own origin. family is F_* (ore) or -1 (bare).
struct ObsCell {
    int x, y, z, family;
};

// One scored hypothesis: the observation's (0,0,0) placed at origin, turned by rotation/mirror.
struct Result {
    int originX, originY, originZ;
    int rotation, mirror;
    int present, absenceHits;
    float score;
};

// Best first: higher score, ties broken by position and orientation so rankings don't depend on the
// order the GPU happened to append results in.
static inline bool rankedBefore(const Result& a, const Result& b) {
    if (a.score != b.score)
        return a.score > b.score;
    if (a.originX != b.originX)
        return a.originX < b.originX;
    if (a.originY != b.originY)
        return a.originY < b.originY;
    if (a.originZ != b.originZ)
        return a.originZ < b.originZ;
    if (a.rotation != b.rotation)
        return a.rotation < b.rotation;
    return a.mirror < b.mirror;
}

// Rotate (x, z) by rotation * 90 degrees after optionally mirroring x. 4 rotations x 2 mirrors = the 8
// horizontal orientations every hypothesis is tested in.
ORE_HD static inline void orientXZ(int x, int z, int rotation, int mirror, int* outX, int* outZ) {
    x *= mirror;
    for (int i = 0; i < rotation; i++) {
        int nx = -z, nz = x;
        x = nx;
        z = nz;
    }
    *outX = x;
    *outZ = z;
}

// Indices into ORE_CONFIGS_118 of the configs whose family the GPU generates, in table order.
static inline std::vector<int> gpuConfigIds() {
    std::vector<int> ids;
    for (int c = 0; c < ORE_CONFIG_COUNT; c++)
        if (ORE_CONFIGS_118_HOST[c].family < GPU_FAMILY_COUNT)
            ids.push_back(c);
    return ids;
}

#endif
