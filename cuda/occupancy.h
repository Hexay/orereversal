// Occupancy grid: one bit per block of the deepslate band over a tile, per GPU family. Bit order is
// z-fastest, then y, then x, so a z-run of blocks is a contiguous run of bits. Plain C, host + device.
#ifndef OCCUPANCY_H
#define OCCUPANCY_H
#include <stdint.h>
#include "ore_config.h"

#define BAND_MIN_Y  (-64) // inclusive
#define BAND_END_Y  0     // exclusive
#define BAND_HEIGHT 64

#ifndef __CUDACC__
typedef struct {
    int x, y, z;
} int3;
#endif

typedef struct {
    uint32_t* words; // GPU_FAMILY_COUNT consecutive bitmaps
    int64_t wordsPerFamily;
    int originX, originZ; // block coords of the grid's (0, 0) column
    int sizeX, sizeZ;     // extent in blocks
} OccupancyGrid;

// Collects candidate blocks of the anchor family whose column lies in [minX..maxX] x [minZ..maxZ].
typedef struct {
    int3* items;
    int* count;
    int capacity;
    int minX, maxX, minZ, maxZ;
} AnchorSink;

ORE_HD static inline int inBand(int y) {
    return y >= BAND_MIN_Y && y < BAND_END_Y;
}

ORE_HD static inline int inGrid(const OccupancyGrid* g, int x, int y, int z) {
    return inBand(y) && x >= g->originX && x < g->originX + g->sizeX && z >= g->originZ &&
           z < g->originZ + g->sizeZ;
}

// Bit index within one family's bitmap. Caller checks inGrid.
ORE_HD static inline int64_t occBitIndex(const OccupancyGrid* g, int x, int y, int z) {
    return ((int64_t)(x - g->originX) * BAND_HEIGHT + (y - BAND_MIN_Y)) * g->sizeZ + (z - g->originZ);
}

ORE_HD static inline int inAnchorArea(const AnchorSink* s, int x, int z) {
    return x >= s->minX && x <= s->maxX && z >= s->minZ && z <= s->maxZ;
}

#endif
