// Keeping the best K hypotheses across tiles, at most one per neighbourhood.
#ifndef TOP_K_H
#define TOP_K_H
#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include "common.h"

static inline int floorDiv(int a, int b) {
    return a >= 0 ? a / b : -(((-a) + b - 1) / b);
}

static inline long long bucketKey(int bx, int by, int bz) {
    return (long long)bx * 0x9E3779B1LL + (long long)by * 0xC2B2AE35LL + (long long)bz * 0x27D4EB2FLL;
}

static inline int chebyshev(const Result& a, const Result& b) {
    return std::max(std::max(abs(a.originX - b.originX), abs(a.originY - b.originY)),
                    abs(a.originZ - b.originZ));
}

// Adds `incoming` to `top`, then keeps the highest-scoring results, skipping any within `separation`
// (Chebyshev) of one already kept, up to k. Near-duplicates are found via a hash grid of
// separation-sized buckets: anything within `separation` lies in the 3x3x3 neighbouring buckets.
static void mergeTopK(std::vector<Result>& top, const std::vector<Result>& incoming, int separation, int k) {
    top.insert(top.end(), incoming.begin(), incoming.end());
    std::sort(top.begin(), top.end(), rankedBefore);
    std::vector<Result> kept;
    std::unordered_map<long long, std::vector<int>> buckets;
    buckets.reserve(k * 2);
    for (const Result& r : top) {
        int bx = floorDiv(r.originX, separation), by = floorDiv(r.originY, separation),
            bz = floorDiv(r.originZ, separation);
        bool isDuplicate = false;
        for (int dx = -1; dx <= 1 && !isDuplicate; dx++)
            for (int dy = -1; dy <= 1 && !isDuplicate; dy++)
                for (int dz = -1; dz <= 1 && !isDuplicate; dz++) {
                    auto it = buckets.find(bucketKey(bx + dx, by + dy, bz + dz));
                    if (it == buckets.end())
                        continue;
                    for (int index : it->second)
                        if (chebyshev(r, kept[index]) <= separation) {
                            isDuplicate = true;
                            break;
                        }
                }
        if (!isDuplicate) {
            buckets[bucketKey(bx, by, bz)].push_back((int)kept.size());
            kept.push_back(r);
        }
        if ((int)kept.size() >= k)
            break;
    }
    top.swap(kept);
}

#endif
