// Ore families and the 1.18 ore feature configs, transcribed from cubiomes finders.c getOreConfig.
// Plain C, shared by host and device code.
#ifndef ORE_CONFIG_H
#define ORE_CONFIG_H
#include <stdint.h>

#ifdef __CUDACC__
#define ORE_HD __host__ __device__
#else
#define ORE_HD
#endif

// The four GPU-generated families come first so a family id doubles as its occupancy-grid slot.
enum { F_TUFF = 0, F_REDSTONE, F_LAPIS, F_GRANITE, F_GRAVEL, F_COPPER, F_IRON, F_COUNT };
#define GPU_FAMILY_COUNT 4

static const char* const FAMILY_NAMES[F_COUNT] = {"tuff",   "redstone", "lapis", "granite",
                                                  "gravel", "copper",   "iron"};

enum { HEIGHT_UNIFORM = 0, HEIGHT_TRIANGLE };

typedef struct {
    int32_t index, step; // decorator seed = population seed + index + 10000 * step
    int32_t size;        // nodes per vein
    int32_t attempts;    // veins per chunk (rare: one vein with probability 1/attempts)
    int32_t heightKind;  // HEIGHT_UNIFORM | HEIGHT_TRIANGLE
    int32_t minY, maxY;
    int32_t rare;
    int32_t family; // F_*
} OreConfig;

// One entry per cubiomes ore *type*; a family can have several (redstone + lower_redstone). Every type
// here has discardChanceOnAirExposure 0 (buried_lapis: 1), so no RNG is spent on air exposure.
#define ORE_CONFIG_COUNT 9
// clang-format off
#define ORE_CONFIGS_118_INIT {                                                              \
    /* index step size attempts height          minY maxY rare  family */                   \
    {  8,    6,   64,  2,       HEIGHT_UNIFORM,  -64,   0, 0,   F_TUFF     }, /* tuff */             \
    { 16,    6,    8,  4,       HEIGHT_UNIFORM,  -64,  15, 0,   F_REDSTONE }, /* redstone */         \
    { 17,    6,    8,  8,       HEIGHT_TRIANGLE, -96, -32, 0,   F_REDSTONE }, /* lower_redstone */   \
    { 21,    6,    7,  2,       HEIGHT_TRIANGLE, -32,  32, 0,   F_LAPIS    }, /* lapis */            \
    { 22,    6,    7,  4,       HEIGHT_UNIFORM,  -64,  64, 0,   F_LAPIS    }, /* buried_lapis */     \
    {  1,    6,   33, 14,       HEIGHT_UNIFORM,  -64, 319, 0,   F_GRAVEL   }, /* gravel */           \
    {  3,    6,   64,  2,       HEIGHT_UNIFORM,    0,  60, 0,   F_GRANITE  }, /* lower_granite */    \
    {  2,    6,   64,  6,       HEIGHT_UNIFORM,   64, 128, 1,   F_GRANITE  }, /* upper_granite */    \
    { 24,    6,   10, 16,       HEIGHT_TRIANGLE, -16, 112, 0,   F_COPPER   }, /* copper */           \
}
// clang-format on

// Device code can't read a host table and vice versa, so under nvcc there are two copies.
#ifdef __CUDACC__
static __device__ const OreConfig ORE_CONFIGS_118[ORE_CONFIG_COUNT] = ORE_CONFIGS_118_INIT;
static const OreConfig ORE_CONFIGS_118_HOST[ORE_CONFIG_COUNT] = ORE_CONFIGS_118_INIT;
#else
static const OreConfig ORE_CONFIGS_118[ORE_CONFIG_COUNT] = ORE_CONFIGS_118_INIT;
#define ORE_CONFIGS_118_HOST ORE_CONFIGS_118
#endif

#endif
