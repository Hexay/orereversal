// Bit-exact port of cubiomes' 1.18 ore generation (finders.c generateOres / generateVeinPart), plain C
// for both host and device. Only the families in ore_config.h are ported; the biome check is omitted
// because none of them is biome-gated and it consumes no RNG. Verified against harness/region_dump
// by the golden diff in tests/regress.sh.
#ifndef OREGEN_H
#define OREGEN_H
#include <math.h>
#include <string.h>
#include "ore_config.h"
#include "ore_rng.h"
#include "occupancy.h"

#define ORE_PI          3.14159265358979323846
#define MAX_VEIN_NODES  64   // largest OreConfig.size
#define VEIN_SEEN_BYTES 1600 // dedup bitset over the largest vein box (26 * 14 * 26 cells)

typedef struct {
    int32_t x, y, z;
} OrePos;

// A vein is a line segment from `from` to `to`, sampled into `size` spheres. Its blocks are confined to
// a box starting at boxMin (cubiomes names the box extents oreSize and radius).
typedef struct {
    double fromX, fromY, fromZ, toX, toY, toZ;
    int boxMinX, boxMinY, boxMinZ;
    int boxSizeXZ, boxSizeY;
} VeinShape;

typedef struct {
    double x, y, z, radius; // radius <= 0 marks a node culled as contained in another
} VeinNode;

// Where generated blocks go: appended to a list (host), or set in an occupancy grid with anchor
// collection (device, legacy generator).
typedef struct {
    OrePos* list;
    int* listCount;
    int listCapacity;
    OccupancyGrid grid;
    int family;
    AnchorSink anchors;
    int isAnchorFamily;
} CandidateSink;

ORE_HD static inline void emitCandidate(CandidateSink* s, int x, int y, int z) {
    if (s->list) {
        if (*s->listCount < s->listCapacity) {
            OrePos* p = &s->list[*s->listCount];
            p->x = x;
            p->y = y;
            p->z = z;
            (*s->listCount)++;
        }
        return;
    }
#ifdef __CUDA_ARCH__ // grid mode uses device atomics; host callers always use list mode
    if (!inGrid(&s->grid, x, y, z))
        return;
    int64_t bit = occBitIndex(&s->grid, x, y, z);
    atomicOr(&s->grid.words[s->family * s->grid.wordsPerFamily + (bit >> 5)], 1u << (bit & 31));
    if (s->isAnchorFamily && inAnchorArea(&s->anchors, x, z)) {
        int slot = atomicAdd(s->anchors.count, 1);
        if (slot < s->anchors.capacity)
            s->anchors.items[slot] = make_int3(x, y, z);
    }
#endif
}

ORE_HD static inline double lerp(double t, double a, double b) {
    return a + t * (b - a);
}

// The RNG stream for one ore config in one chunk.
ORE_HD static inline Xoroshiro oreConfigRng(uint64_t worldSeed, const OreConfig* c, int era, int chunkX,
                                            int chunkZ) {
    uint64_t populationSeed = getPopulationSeed(worldSeed, chunkX << 4, chunkZ << 4);
    Xoroshiro rng;
    xSetSeed(&rng, populationSeed + (uint64_t)c->index[era] + 10000ULL * (uint64_t)c->step);
    return rng;
}

ORE_HD static inline int veinAttempts(const OreConfig* c, Xoroshiro* rng) {
    if (c->rare)
        return (xNextFloat(rng) < 1.0f / (float)c->attempts) ? 1 : 0;
    return c->attempts;
}

ORE_HD static inline int veinHeight(const OreConfig* c, Xoroshiro* rng) {
    if (c->minY > c->maxY)
        return c->minY;
    int range = c->maxY - c->minY;
    if (c->heightKind == HEIGHT_UNIFORM || range <= 0)
        return xNextIntBetween(rng, c->minY, c->maxY);
    int half = range / 2;
    int a = xNextIntBetween(rng, 0, range - half);
    int b = xNextIntBetween(rng, 0, half);
    return c->minY + a + b;
}

ORE_HD static inline VeinShape nextVeinShape(const OreConfig* c, Xoroshiro* rng, int chunkX, int chunkZ) {
    int bx = (chunkX << 4) + xNextIntJ(rng, 16);
    int bz = (chunkZ << 4) + xNextIntJ(rng, 16);
    int by = veinHeight(c, rng);
    float angle = xNextFloat(rng) * (float)ORE_PI;
    float halfLength = (float)c->size / 8.0F;
    int pad = (int)ceil(((float)c->size / 16.0F * 2.0F + 1.0F) / 2.0F);
    VeinShape v;
    v.fromX = (double)bx + sin(angle) * (double)halfLength;
    v.toX = (double)bx - sin(angle) * (double)halfLength;
    v.fromZ = (double)bz + cos(angle) * (double)halfLength;
    v.toZ = (double)bz - cos(angle) * (double)halfLength;
    v.fromY = by + xNextIntJ(rng, 3) - 2;
    v.toY = by + xNextIntJ(rng, 3) - 2;
    v.boxMinX = bx - (int)ceil(halfLength) - pad;
    v.boxMinY = by - 2 - pad;
    v.boxMinZ = bz - (int)ceil(halfLength) - pad;
    v.boxSizeXZ = 2 * ((int)ceil(halfLength) + pad);
    v.boxSizeY = 2 * (2 + pad);
    return v;
}

ORE_HD static inline VeinNode nextVeinNode(const VeinShape* v, int i, int size, Xoroshiro* rng) {
    float t = (float)i / (float)size;
    VeinNode n;
    n.x = lerp(t, v->fromX, v->toX);
    n.y = lerp(t, v->fromY, v->toY);
    n.z = lerp(t, v->fromZ, v->toZ);
    double length = xNextDoubleJ(rng) * (double)size / 16.0;
    n.radius = ((sin((float)ORE_PI * t) + 1.0F) * length + 1.0) / 2.0;
    return n;
}

// Marks every node whose sphere lies inside another node's sphere (cubiomes' O(size^2) pass).
ORE_HD static inline void cullContainedNodes(VeinNode* nodes, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (nodes[i].radius <= 0.0)
            continue;
        for (int j = i + 1; j < size; ++j) {
            if (nodes[j].radius <= 0.0)
                continue;
            double dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y, dz = nodes[i].z - nodes[j].z;
            double dr = nodes[i].radius - nodes[j].radius;
            if (dr * dr <= dx * dx + dy * dy + dz * dz)
                continue;
            if (dr > 0.0)
                nodes[j].radius = -1.0;
            else
                nodes[i].radius = -1.0;
        }
    }
}

ORE_HD static inline int floorToInt(double v) {
    return (int)floor(v);
}

// cubiomes generateVeinPart: emits every in-box block inside a live node's sphere, each block once.
ORE_HD static inline void generateVeinPart(const OreConfig* c, Xoroshiro* rng, const VeinShape* v,
                                           CandidateSink* sink) {
    const int minBuildY = -64, maxBuildY = 320;
    VeinNode nodes[MAX_VEIN_NODES];
    for (int i = 0; i < c->size; ++i)
        nodes[i] = nextVeinNode(v, i, c->size, rng);
    cullContainedNodes(nodes, c->size);

    // cubiomes dedups by box-relative index without bounds-checking it; replicate exactly.
    char seen[VEIN_SEEN_BYTES];
    memset(seen, 0, sizeof(seen));
    for (int i = 0; i < c->size; ++i) {
        const VeinNode* n = &nodes[i];
        if (n->radius < 0.0)
            continue;
        int minX = floorToInt(n->x - n->radius), maxX = floorToInt(n->x + n->radius);
        int minY = floorToInt(n->y - n->radius), maxY = floorToInt(n->y + n->radius);
        int minZ = floorToInt(n->z - n->radius), maxZ = floorToInt(n->z + n->radius);
        if (minX < v->boxMinX)
            minX = v->boxMinX;
        if (minY < v->boxMinY)
            minY = v->boxMinY;
        if (minZ < v->boxMinZ)
            minZ = v->boxMinZ;
        if (maxX < minX)
            maxX = minX;
        if (maxY < minY)
            maxY = minY;
        if (maxZ < minZ)
            maxZ = minZ;
        for (int x = minX; x <= maxX; ++x) {
            double dx = ((double)x + 0.5 - n->x) / n->radius;
            if (dx * dx >= 1.0)
                continue;
            for (int y = minY; y <= maxY; ++y) {
                double dy = ((double)y + 0.5 - n->y) / n->radius;
                if (dx * dx + dy * dy >= 1.0)
                    continue;
                for (int z = minZ; z <= maxZ; ++z) {
                    double dz = ((double)z + 0.5 - n->z) / n->radius;
                    if (dx * dx + dy * dy + dz * dz >= 1.0)
                        continue;
                    if (y < minBuildY || y >= maxBuildY)
                        continue;
                    int cell = x - v->boxMinX + (y - v->boxMinY) * v->boxSizeXZ +
                               (z - v->boxMinZ) * v->boxSizeXZ * v->boxSizeY;
                    if (seen[cell >> 3] & (char)(1 << (cell & 7)))
                        continue;
                    seen[cell >> 3] |= (char)(1 << (cell & 7));
                    emitCandidate(sink, x, y, z);
                }
            }
        }
    }
}

// All candidate blocks of one ore config in one chunk. Skips cubiomes' surface-height gate: exact for tuff
// and redstone; in low terrain a gated high lapis vein shifts later lapis veins (gravel and copper are hit
// far harder, so they come from region_dump instead). See docs/research-log.md P9, P10.
ORE_HD static inline void generateOreConfig(uint64_t worldSeed, const OreConfig* c, int era, int chunkX,
                                            int chunkZ, CandidateSink* sink) {
    Xoroshiro rng = oreConfigRng(worldSeed, c, era, chunkX, chunkZ);
    int attempts = veinAttempts(c, &rng);
    for (int a = 0; a < attempts; ++a) {
        VeinShape v = nextVeinShape(c, &rng, chunkX, chunkZ);
        generateVeinPart(c, &rng, &v, sink);
    }
}

#endif
