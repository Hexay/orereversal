// kIronVeins: ORs iron-vein tuff (iron_veins.h) into the tuff bitmap, one block per chunk.
#ifndef IRON_VEIN_KERNEL_CUH
#define IRON_VEIN_KERNEL_CUH
#include "generate.cuh"
#include "iron_veins.h"

#define IRON_VEIN_BLOCK_SIZE 256

__global__ void __launch_bounds__(IRON_VEIN_BLOCK_SIZE)
    kIronVeins(const OreVeinNoise* __restrict__ noise, ChunkGrid chunks, OccupancyGrid grid,
               AnchorSink anchors, int tuffIsAnchor) {
    __shared__ VeinCorners corners;
    int chunkX = chunks.originX + blockIdx.x % chunks.countX;
    int chunkZ = chunks.originZ + blockIdx.x / chunks.countX;
    __shared__ unsigned char strongCorner[VEIN_CORNERS], cellLive[VEIN_CELLS];
    int strong = 0;
    for (int i = threadIdx.x; i < VEIN_CORNERS; i += blockDim.x)
        strong |= strongCorner[i] = mayBeIronVeinCorner(noise, chunkX, chunkZ, i);
    if (!__syncthreads_or(strong))
        return;
    for (int i = threadIdx.x; i < VEIN_CELLS; i += blockDim.x)
        cellLive[i] = veinCellMayHaveIron(strongCorner, i);
    __syncthreads();
    for (int i = threadIdx.x; i < VEIN_CORNERS; i += blockDim.x)
        if (veinCornerNeeded(cellLive, i))
            fillVeinCorner(noise, chunkX, chunkZ, i, &corners);
    __syncthreads();

    for (int row = threadIdx.x; row < IRON_VEIN_ROWS; row += blockDim.x) {
        int x = chunkX * 16 + (row & 15), y = IRON_VEIN_MIN_Y + (row >> 4), z = chunkZ * 16;
        const unsigned char* live = cellLive + veinCellOf(row & 15, y, 0);
        uint64_t run = 0;
        for (int lz = 0; lz < 16; lz++)
            if (live[lz >> 2] && mayBeIronVeinTuff(&corners, row & 15, y, lz) &&
                ironVeinTuffAt(noise, &corners, chunkX, chunkZ, x, y, z + lz))
                run |= 1ULL << lz;
        if (!run)
            continue;
        orZRun(grid, F_TUFF, x, y, z, run);
        if (tuffIsAnchor)
            collectZRunAnchors(anchors, x, y, z, run);
    }
}

#endif
