// GPU generation of the four GPU families into the occupancy grid, plus anchor collection.
//
// Default path, two kernels (design and measurements: docs/gpu-optimization.md):
//   kSetupVeins  one thread per (chunk, config): runs the RNG and writes each vein's nodes to scratch.
//   kFillVeins   one warp per vein: lanes own whole (x, y) columns of the vein box and OR each column's
//                z-run into the grid in one go.
// kGenerateLegacy is the original one-thread-per-chunk generator, kept as the bit-exact reference.
#ifndef GENERATE_CUH
#define GENERATE_CUH
#include "common.h"

#define FULL_WARP       0xffffffffu
#define WARPS_PER_BLOCK 4
#define FILL_BLOCK_SIZE (WARPS_PER_BLOCK * 32)
#define CULL_WINDOW     8 // containment cull only compares nodes this far apart along the vein

struct ChunkGrid {
    int originX, originZ; // chunk coords
    int countX, countZ;
};

struct VeinRecord {
    int firstNode, nodeCount;
    int boxMinX, boxMinY, boxMinZ, boxSizeXZ, boxSizeY;
    int family, isAnchorFamily;
};

// Fixed-capacity scratch that kSetupVeins fills and kFillVeins drains.
struct VeinScratch {
    VeinNode* nodes;
    int nodeCapacity;
    int* nodeCount;
    VeinRecord* veins;
    int veinCapacity;
    int* veinCount;
};

__global__ void kGenerateLegacy(uint64_t seed, ChunkGrid chunks, OccupancyGrid grid, AnchorSink anchors,
                                int anchorFamily, const int* configIds, int configCount) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunks.countX * chunks.countZ)
        return;
    int chunkX = chunks.originX + tid % chunks.countX, chunkZ = chunks.originZ + tid / chunks.countX;
    CandidateSink sink = {};
    sink.grid = grid;
    sink.anchors = anchors;
    for (int c = 0; c < configCount; c++) {
        const OreConfig* config = &ORE_CONFIGS_118[configIds[c]];
        sink.family = config->family;
        sink.isAnchorFamily = config->family == anchorFamily;
        generateOreConfig(seed, config, chunkX, chunkZ, &sink);
    }
}

// Config-major thread order, so a warp shares one config and its loop bounds.
__global__ void kSetupVeins(uint64_t seed, ChunkGrid chunks, int anchorFamily, const int* configIds,
                            int configCount, VeinScratch scratch) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t chunkCount = (int64_t)chunks.countX * chunks.countZ;
    if (tid >= chunkCount * configCount)
        return;
    const OreConfig* config = &ORE_CONFIGS_118[configIds[tid / chunkCount]];
    int64_t chunkIndex = tid % chunkCount;
    int chunkX = chunks.originX + (int)(chunkIndex % chunks.countX);
    int chunkZ = chunks.originZ + (int)(chunkIndex / chunks.countX);
    int size = config->size;

    Xoroshiro rng = oreConfigRng(seed, config, chunkX, chunkZ);
    int attempts = veinAttempts(config, &rng);
    for (int a = 0; a < attempts; ++a) {
        VeinShape v = nextVeinShape(config, &rng, chunkX, chunkZ);
        int firstNode = atomicAdd(scratch.nodeCount, size);
        int slot = atomicAdd(scratch.veinCount, 1);
        bool overflow = slot >= scratch.veinCapacity || firstNode + size > scratch.nodeCapacity;
        // Keep drawing nodes on overflow: later veins in this chunk must see the same RNG stream.
        for (int i = 0; i < size; ++i) {
            VeinNode n = nextVeinNode(&v, i, size, &rng);
            if (!overflow)
                scratch.nodes[firstNode + i] = n;
        }
        if (overflow)
            continue;
        VeinRecord r;
        r.firstNode = firstNode;
        r.nodeCount = size;
        r.boxMinX = v.boxMinX;
        r.boxMinY = v.boxMinY;
        r.boxMinZ = v.boxMinZ;
        r.boxSizeXZ = v.boxSizeXZ;
        r.boxSizeY = v.boxSizeY;
        r.family = config->family;
        r.isAnchorFamily = config->family == anchorFamily;
        scratch.veins[slot] = r;
    }
}

// A node's centre relative to the vein box, and its squared radius, in FP32.
struct NodeF32 {
    float x, y, z, radiusSq;
};

// Approximate version of cullContainedNodes: windowed, and in FP32. Keeping a contained node is harmless
// (its blocks are a subset of its container's and the grid is a union), so a node is only culled when it
// is contained by more than CULL_EPS, which dwarfs FP32 and FP64 rounding: no block changes.
__device__ __forceinline__ void cullContainedNeighbors(const NodeF32* nodes, const float* radius, int count,
                                                       unsigned char* alive, int lane) {
    const float CULL_EPS = 1e-2f;
    for (int j = lane; j < count; j += 32) {
        NodeF32 nj = nodes[j];
        float rj = radius[j];
        int lo = max(j - CULL_WINDOW, 0), hi = min(j + CULL_WINDOW, count - 1);
        for (int i = lo; i <= hi; i++) {
            float dr = radius[i] - rj;
            if (i == j || dr <= 0.0f)
                continue;
            float dx = nodes[i].x - nj.x, dy = nodes[i].y - nj.y, dz = nodes[i].z - nj.z;
            if (dr * dr > dx * dx + dy * dy + dz * dz + CULL_EPS) {
                alive[j] = 0;
                break;
            }
        }
    }
}

// Bitmask over box-relative z of the blocks in column (boxX, boxY) covered by any live node, limited to
// [zFirst, zLast]. Classifies in FP32 with margin EPS and settles only the thin boundary shell in FP64,
// which gives exactly the FP64 answer (see docs/gpu-optimization.md, step 8).
__device__ __forceinline__ uint64_t columnZMask(int boxX, int boxY, int worldX, int worldY,
                                                const VeinRecord& v, const VeinNode* nodes,
                                                const NodeF32* nodesF32, const unsigned char* alive,
                                                int zFirst, int zLast) {
    const float EPS = 1e-2f;
    uint64_t mask = 0;
    for (int i = 0; i < v.nodeCount; ++i) {
        if (!alive[i])
            continue;
        NodeF32 n = nodesF32[i];
        float dx = (float)boxX + 0.5f - n.x, dy = (float)boxY + 0.5f - n.y;
        float slack = n.radiusSq - dx * dx - dy * dy; // < 0: the column misses this sphere
        if (slack <= -EPS)
            continue;
        float halfChord = (slack > 0.0f) ? sqrtf(slack) : 0.0f;
        int zLo = (int)floorf(n.z - 0.5f - halfChord) - 1, zHi = (int)floorf(n.z - 0.5f + halfChord) + 1;
        if (zLo < zFirst)
            zLo = zFirst;
        if (zHi > zLast)
            zHi = zLast;
        for (int z = zLo; z <= zHi; z++) {
            float dz = (float)z + 0.5f - n.z, distSq = dx * dx + dy * dy + dz * dz;
            if (distSq >= n.radiusSq + EPS)
                continue;
            if (distSq > n.radiusSq - EPS) {
                const VeinNode& d = nodes[i];
                double ddx = (double)worldX + 0.5 - d.x, ddy = (double)worldY + 0.5 - d.y,
                       ddz = (double)(v.boxMinZ + z) + 0.5 - d.z;
                if (ddx * ddx + ddy * ddy + ddz * ddz >= d.radius * d.radius)
                    continue;
            }
            mask |= (1ULL << z);
        }
    }
    return mask;
}

// OR a run of up to 64 z-consecutive blocks (bit 0 = block z) into one family's bitmap.
__device__ __forceinline__ void orZRun(const OccupancyGrid& grid, int family, int x, int y, int z,
                                       uint64_t run) {
    int64_t bit = occBitIndex(&grid, x, y, z);
    uint32_t* words = grid.words + family * grid.wordsPerFamily + (bit >> 5);
    int shift = (int)(bit & 31);
    uint64_t low = run << shift;
    if ((uint32_t)low)
        atomicOr(&words[0], (uint32_t)low);
    if ((uint32_t)(low >> 32))
        atomicOr(&words[1], (uint32_t)(low >> 32));
    if (shift && (uint32_t)(run >> (64 - shift)))
        atomicOr(&words[2], (uint32_t)(run >> (64 - shift)));
}

__device__ __forceinline__ void collectZRunAnchors(const AnchorSink& anchors, int x, int y, int z,
                                                   uint64_t run) {
    if (x < anchors.minX || x > anchors.maxX)
        return;
    while (run) {
        int b = __ffsll((long long)run) - 1;
        run &= run - 1;
        int bz = z + b;
        if (bz >= anchors.minZ && bz <= anchors.maxZ) {
            int slot = atomicAdd(anchors.count, 1);
            if (slot < anchors.capacity)
                anchors.items[slot] = make_int3(x, y, bz);
        }
    }
}

// Persistent warps pull veins from a shared counter until all are filled.
__global__ void __launch_bounds__(FILL_BLOCK_SIZE)
    kFillVeins(OccupancyGrid grid, AnchorSink anchors, const VeinNode* allNodes, const VeinRecord* veins,
               int veinCount, int* nextVein) {
    __shared__ unsigned char sharedAlive[WARPS_PER_BLOCK][MAX_VEIN_NODES];
    __shared__ NodeF32 sharedNodesF32[WARPS_PER_BLOCK][MAX_VEIN_NODES];
    __shared__ float sharedRadiusF32[WARPS_PER_BLOCK][MAX_VEIN_NODES];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned char* alive = sharedAlive[warp];
    NodeF32* nodesF32 = sharedNodesF32[warp];
    float* radiusF32 = sharedRadiusF32[warp];
    for (;;) {
        int veinIndex;
        if (lane == 0)
            veinIndex = atomicAdd(nextVein, 1);
        veinIndex = __shfl_sync(FULL_WARP, veinIndex, 0);
        if (veinIndex >= veinCount)
            break;
        VeinRecord v = veins[veinIndex];
        const VeinNode* nodes = allNodes + v.firstNode;

        // Box-relative offsets keep the FP32 values small, so they are exact enough for the EPS margins.
        for (int i = lane; i < v.nodeCount; i += 32) {
            double r = nodes[i].radius;
            nodesF32[i].x = (float)(nodes[i].x - v.boxMinX);
            nodesF32[i].y = (float)(nodes[i].y - v.boxMinY);
            nodesF32[i].z = (float)(nodes[i].z - v.boxMinZ);
            nodesF32[i].radiusSq = (float)(r * r);
            radiusF32[i] = (float)r;
            alive[i] = 1;
        }
        __syncwarp();
        cullContainedNeighbors(nodesF32, radiusF32, v.nodeCount, alive, lane);
        __syncwarp();

        for (int column = lane; column < v.boxSizeXZ * v.boxSizeY; column += 32) {
            int boxX = column % v.boxSizeXZ, boxY = column / v.boxSizeXZ;
            int x = v.boxMinX + boxX, y = v.boxMinY + boxY;
            if (!inBand(y) || x < grid.originX || x >= grid.originX + grid.sizeX)
                continue;
            // Box-relative z range that falls inside the grid.
            int zFirst = (grid.originZ > v.boxMinZ) ? (grid.originZ - v.boxMinZ) : 0;
            int zLast = v.boxSizeXZ - 1;
            if (v.boxMinZ + zLast > grid.originZ + grid.sizeZ - 1)
                zLast = grid.originZ + grid.sizeZ - 1 - v.boxMinZ;
            if (zFirst > zLast)
                continue;
            uint64_t mask = columnZMask(boxX, boxY, x, y, v, nodes, nodesF32, alive, zFirst, zLast);
            if (!mask)
                continue;
            uint64_t run = mask >> zFirst;
            orZRun(grid, v.family, x, y, v.boxMinZ + zFirst, run);
            if (v.isAnchorFamily)
                collectZRunAnchors(anchors, x, y, v.boxMinZ + zFirst, run);
        }
        __syncwarp();
    }
}

#endif
