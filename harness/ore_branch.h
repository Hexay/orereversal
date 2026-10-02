// Gate-branched ore generation for surface-gated configs (gravel, copper).
//
// cubiomes approximates Minecraft's per-vein surface gate with a climate-noise height, and it is wrong
// for a few percent of veins. A wrongly gated vein also shifts the RNG of every later vein of that config
// in the chunk. When a vein's start height is within BRANCH_MARGIN of the approximate surface, both
// outcomes are plausible, so this enumerates the decision tree (up to MAX_BRANCH_POINTS borderline veins)
// and returns each leaf as a variant; refine keeps whichever variant fits each hypothesis best.
#ifndef ORE_BRANCH_H
#define ORE_BRANCH_H
#include <math.h>
#include <string.h>
#include "finders.h"
#include "generator.h"

#define CUBIOMES_PI       3.14159265358979323846 // finders.c's PI, which isn't exported
#define BRANCH_MARGIN     12 // 24 gave competitors freedom too: real rooms 457->419 vs 457 at 12 (P8)
#define MAX_BRANCH_POINTS 3
#define MAX_VARIANTS      (1 << MAX_BRANCH_POINTS)
#define HEIGHT_CACHE      24 // quart cells per side cached around one chunk

typedef struct {
    const Generator* g;
    const SurfaceNoise* sn;
    int originQx, originQz; // quart coords of cache cell [0][0]
    int known[HEIGHT_CACHE][HEIGHT_CACHE];
    int height[HEIGHT_CACHE][HEIGHT_CACHE];
} HeightCache;

static void initHeightCache(HeightCache* h, const Generator* g, const SurfaceNoise* sn, int cx, int cz) {
    h->g = g;
    h->sn = sn;
    h->originQx = cx * 4 - 8;
    h->originQz = cz * 4 - 8;
    memset(h->known, 0, sizeof(h->known));
}

static int approxHeight(HeightCache* h, int qx, int qz) {
    int i = qx - h->originQx, k = qz - h->originQz;
    int cached = i >= 0 && i < HEIGHT_CACHE && k >= 0 && k < HEIGHT_CACHE;
    if (cached && h->known[i][k])
        return h->height[i][k];
    float y;
    mapApproxHeight(&y, 0, h->g, h->sn, qx, qz, 1, 1);
    int floored = (int)floor(y);
    if (cached) {
        h->known[i][k] = 1;
        h->height[i][k] = floored;
    }
    return floored;
}

// Highest approximate surface over the vein footprint, as cubiomes' gate loop sees it.
static int footprintHeight(HeightCache* h, int startX, int startZ, int oreSize) {
    int best = -1000000;
    for (int qx = startX >> 2; qx <= (startX + oreSize) >> 2; qx++)
        for (int qz = startZ >> 2; qz <= (startZ + oreSize) >> 2; qz++) {
            int y = approxHeight(h, qx, qz);
            if (y > best)
                best = y;
        }
    return best;
}

// Generates one leaf of the decision tree. Borderline veins take forced[j] for the j-th one met (j <
// nForced); later borderline veins (up to MAX_BRANCH_POINTS) follow cubiomes and their default is recorded in
// defaults[]. Returns the number of borderline veins met, capped at MAX_BRANCH_POINTS.
static int generateBranch(HeightCache* h, int mc, uint64_t seed, OreConfig c, int cx, int cz,
                          const int* forced, int nForced, int* defaults, Pos3List* out) {
    Xoroshiro xr;
    RandomSource rnd = createXoroshiro(&xr);
    rnd.setSeed(rnd.state, getPopulationSeed(mc, seed, cx << 4, cz << 4) + c.index + 10000 * c.step);
    int met = 0;
    for (int i = 0; i < c.repeatCount; i++) {
        Pos3 pos = generateBaseOrePosition(mc, c, cx, cz, rnd);
        // Same draws as cubiomes generateOrePositions (regular ore) up to the gate.
        float angle = rnd.nextFloat(rnd.state) * (float)CUBIOMES_PI;
        float size = (float)c.size / 8.0F;
        int amortizedSize = ceil(((float)c.size / 16.0F * 2.0F + 1.0F) / 2.0F);
        double offsetXPos = (double)pos.x + sin(angle) * (double)size;
        double offsetXNeg = (double)pos.x - sin(angle) * (double)size;
        double offsetZPos = (double)pos.z + cos(angle) * (double)size;
        double offsetZNeg = (double)pos.z - cos(angle) * (double)size;
        double offsetYPos = pos.y + rnd.nextInt(rnd.state, 3) - 2;
        double offsetYNeg = pos.y + rnd.nextInt(rnd.state, 3) - 2;
        int startX = pos.x - ceil(size) - amortizedSize;
        int startY = pos.y - 2 - amortizedSize;
        int startZ = pos.z - ceil(size) - amortizedSize;
        int oreSize = 2 * (ceil(size) + amortizedSize);
        int radius = 2 * (2 + amortizedSize);

        int surface = footprintHeight(h, startX, startZ, oreSize);
        int place = startY <= surface;
        if (abs(startY - surface) <= BRANCH_MARGIN && met < MAX_BRANCH_POINTS) {
            if (met < nForced)
                place = forced[met];
            else
                defaults[met] = place;
            met++;
        }
        if (place)
            generateVeinPart(mc, c, rnd, offsetXPos, offsetXNeg, offsetZPos, offsetZNeg, offsetYPos,
                             offsetYNeg, startX, startY, startZ, oreSize, radius, out);
    }
    return met;
}

typedef struct {
    Pos3List variants[MAX_VARIANTS];
    int count;
} BranchVariants;

static void branchFrom(HeightCache* h, int mc, uint64_t seed, OreConfig c, int cx, int cz, int* forced,
                       int nForced, BranchVariants* out) {
    int defaults[MAX_BRANCH_POINTS];
    Pos3List blocks;
    createPos3List(&blocks, 256);
    int met = generateBranch(h, mc, seed, c, cx, cz, forced, nForced, defaults, &blocks);
    if (met <= nForced) { // a leaf: no further borderline vein
        out->variants[out->count++] = blocks;
        return;
    }
    freePos3List(&blocks);
    for (int choice = 0; choice < 2; choice++) {
        forced[nForced] = choice ? !defaults[nForced] : defaults[nForced]; // default path first
        branchFrom(h, mc, seed, c, cx, cz, forced, nForced + 1, out);
    }
}

// All variants of one config in one chunk; variants[0] is cubiomes' own output.
static void generateBranchedOres(HeightCache* h, int mc, uint64_t seed, OreConfig c, int cx, int cz,
                                 BranchVariants* out) {
    int forced[MAX_BRANCH_POINTS];
    out->count = 0;
    branchFrom(h, mc, seed, c, cx, cz, forced, 0, out);
}

#endif
