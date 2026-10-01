// GPU scoring of hypotheses: sort anchors for memory locality, then score every (anchor, orientation).
#ifndef SCORE_CUH
#define SCORE_CUH
#include "common.h"

struct ScoringInput {
    const ObsCell* ore; // GPU families only
    int oreCount;
    const ObsCell* bare;
    int bareCount;
    int anchorX, anchorY, anchorZ; // the observation cell each anchor candidate is aligned to
    int tolerance;                 // ore cells match within +-tolerance
    int absenceTolerance;          // bare cells count as hits within +-absenceTolerance
    float absenceWeight;
    int minPresence; // hypotheses below this are dropped
};

struct ResultSink {
    Result* items;
    int* count;
    int capacity;
};

__device__ __forceinline__ bool occupied(const OccupancyGrid& grid, int family, int x, int y, int z) {
    if (!inGrid(&grid, x, y, z))
        return false;
    int64_t bit = occBitIndex(&grid, x, y, z);
    return (grid.words[family * grid.wordsPerFamily + (bit >> 5)] >> (bit & 31)) & 1u;
}

__device__ __forceinline__ bool occupiedNear(const OccupancyGrid& grid, int family, int x, int y, int z,
                                             int tolerance) {
    if (tolerance == 0)
        return occupied(grid, family, x, y, z);
    for (int dx = -tolerance; dx <= tolerance; dx++)
        for (int dy = -tolerance; dy <= tolerance; dy++)
            for (int dz = -tolerance; dz <= tolerance; dz++)
                if (occupied(grid, family, x + dx, y + dy, z + dz))
                    return true;
    return false;
}

// Spread the low 16 bits of n to the even bit positions.
__device__ __forceinline__ uint32_t spreadBits(uint32_t n) {
    n &= 0x0000ffffu;
    n = (n | (n << 8)) & 0x00FF00FFu;
    n = (n | (n << 4)) & 0x0F0F0F0Fu;
    n = (n | (n << 2)) & 0x33333333u;
    n = (n | (n << 1)) & 0x55555555u;
    return n;
}

// Morton (z-order) key of each anchor's grid-relative (x, z). Sorting by it makes neighbouring scoring
// threads probe overlapping parts of the grid.
__global__ void kMortonKeys(const int3* anchors, int count, int originX, int originZ, uint32_t* keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    keys[i] = spreadBits((uint32_t)(anchors[i].x - originX)) |
              (spreadBits((uint32_t)(anchors[i].z - originZ)) << 1);
}

// One thread per (anchor, orientation). Presence first, abandoning the hypothesis as soon as it can no
// longer reach minPresence; absence only for the survivors.
__global__ void kScoreHypotheses(const int3* anchors, int anchorCount, OccupancyGrid grid, ScoringInput in,
                                 ResultSink out) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (int64_t)anchorCount * 8)
        return;
    int orientation = (int)(tid % 8);
    int rotation = orientation & 3, mirror = (orientation < 4) ? 1 : -1;
    int3 anchor = anchors[tid / 8];
    int ax, az;
    orientXZ(in.anchorX, in.anchorZ, rotation, mirror, &ax, &az);
    int originX = anchor.x - ax, originY = anchor.y - in.anchorY, originZ = anchor.z - az;

    int present = 0;
    for (int i = 0; i < in.oreCount; i++) {
        int dx, dz;
        orientXZ(in.ore[i].x, in.ore[i].z, rotation, mirror, &dx, &dz);
        if (occupiedNear(grid, in.ore[i].family, originX + dx, originY + in.ore[i].y, originZ + dz,
                         in.tolerance))
            present++;
        if (present + (in.oreCount - 1 - i) < in.minPresence)
            return;
    }

    int absenceHits = 0;
    for (int i = 0; i < in.bareCount; i++) {
        int dx, dz;
        orientXZ(in.bare[i].x, in.bare[i].z, rotation, mirror, &dx, &dz);
        int x = originX + dx, y = originY + in.bare[i].y, z = originZ + dz;
        for (int family = 0; family < GPU_FAMILY_COUNT; family++)
            if (occupiedNear(grid, family, x, y, z, in.absenceTolerance)) {
                absenceHits++;
                break;
            }
    }

    int slot = atomicAdd(out.count, 1);
    if (slot < out.capacity) {
        Result r;
        r.originX = originX;
        r.originY = originY;
        r.originZ = originZ;
        r.rotation = rotation;
        r.mirror = mirror;
        r.present = present;
        r.absenceHits = absenceHits;
        r.score = present - in.absenceWeight * absenceHits;
        out.items[slot] = r;
    }
}

#endif
