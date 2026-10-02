// veintest — runs kIronVeins on the GPU over a chunk box and prints the iron-vein tuff it produces in
// harness/region_dump's CSV format. tests/regress.sh diffs it against cubiomes, since nvcc may contract the
// device code differently from the host build that oretest checks.
// Usage: veintest <seed> <cxMin> <cxMax> <czMin> <czMax>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common.h"
#include "iron_vein_kernel.cuh"

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax>\n", argv[0]);
        return 2;
    }
    uint64_t seed = (uint64_t)strtoll(argv[1], NULL, 10);
    int cxMin = atoi(argv[2]), cxMax = atoi(argv[3]), czMin = atoi(argv[4]), czMax = atoi(argv[5]);
    ChunkGrid chunks = {cxMin, czMin, cxMax - cxMin + 1, czMax - czMin + 1};
    OccupancyGrid grid;
    grid.originX = cxMin * 16;
    grid.originZ = czMin * 16;
    grid.sizeX = chunks.countX * 16;
    grid.sizeZ = chunks.countZ * 16;
    grid.wordsPerFamily = ((int64_t)grid.sizeX * BAND_HEIGHT * grid.sizeZ + 31) / 32;
    size_t bytes = GPU_FAMILY_COUNT * grid.wordsPerFamily * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&grid.words, bytes));
    CUDA_CHECK(cudaMemset(grid.words, 0, bytes));

    OreVeinNoise noise;
    initOreVeinNoise(&noise, seed);
    OreVeinNoise* deviceNoise;
    CUDA_CHECK(cudaMalloc(&deviceNoise, sizeof(noise)));
    CUDA_CHECK(cudaMemcpy(deviceNoise, &noise, sizeof(noise), cudaMemcpyHostToDevice));
    AnchorSink noAnchors = {};
    kIronVeins<<<chunks.countX * chunks.countZ, IRON_VEIN_BLOCK_SIZE>>>(deviceNoise, chunks, grid, noAnchors, 0);
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint32_t> tuff(grid.wordsPerFamily);
    CUDA_CHECK(cudaMemcpy(tuff.data(), grid.words + F_TUFF * grid.wordsPerFamily,
                          tuff.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("family,x,y,z\n");
    long count = 0;
    for (int x = grid.originX; x < grid.originX + grid.sizeX; x++)
        for (int y = BAND_MIN_Y; y < BAND_END_Y; y++)
            for (int z = grid.originZ; z < grid.originZ + grid.sizeZ; z++) {
                int64_t bit = occBitIndex(&grid, x, y, z);
                if (tuff[bit >> 5] >> (bit & 31) & 1) {
                    printf("tuff,%d,%d,%d\n", x, y, z);
                    count++;
                }
            }
    fprintf(stderr, "emitted %ld iron-vein tuff blocks\n", count);
    return 0;
}
