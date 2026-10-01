// oregen.h — portable (host + CUDA device) port of cubiomes' 1.18+ Tier-1 ore generation.
// Hand-ported from xpple/cubiomes finders.c/rng.h. Bit-exactness is verified by diffing the CPU
// driver (oretest.c) against region_dump.exe — see cuda/README.md. Only the discard-free Tier-1
// families we validated are ported; the biome system is omitted (all these ores are isOverworld =
// always viable, no RNG cost) so LargeCopperOre (dripstone/deep_dark gated) is excluded.
//
// SURFACE GATE: cubiomes' generateOrePositions gates each vein on mapApproxHeight (surface noise).
// For deep ores startY << surface so the gate always passes; ORE_SKIP_SURFACE replaces it with
// "always generate". The diff test reveals whether any high-Y family (gravel/copper/upper_granite)
// needs the real gate. Define ORE_WITH_SURFACE to require a host-supplied surface_floor(x,z) hook.
#ifndef OREGEN_H
#define OREGEN_H
#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef __CUDACC__
#define ORE_HD __host__ __device__
#else
#define ORE_HD
#endif

#define ORE_PI 3.14159265358979323846
#define ORE_MAXSIZE 64           // largest config.size (tuff/granite = 64)
#define ORE_MAXSLOTS 1600        // ceil(maxOreSize*maxRadius*maxOreSize / 8); tuff -> 26*14*26=9464 bits

// ---- Tier-1 family ids (stable, used by driver + matcher) ----
// GPU-ACTIVE (bit-exact w/o surface noise): tuff, redstone, lapis, granite.
// DEFERRED (need the mapApproxHeight surface gate; high Y-range desyncs RNG): gravel, copper.
// Verified bit-exact vs region_dump.exe over 64 chunks (cuda/README.md). See docs/research-log.md P3.
enum { F_TUFF=0, F_REDSTONE, F_LAPIS, F_GRAVEL, F_GRANITE, F_COPPER, F_IRON, F_COUNT };
#define ORE_GPU_ACTIVE(fam) ((fam)==F_TUFF||(fam)==F_REDSTONE||(fam)==F_LAPIS||(fam)==F_GRANITE)

// ---- height provider kinds ----
enum { HP_UNIFORM=0, HP_TRIANGLE };

typedef struct {
    int32_t index, step, size, repeatCount;
    int32_t hp;          // HP_UNIFORM | HP_TRIANGLE
    int32_t h1, h2;      // min, max offset
    int32_t rare;        // rareOrePlacement (upper_granite): repeatCount via nextFloat<1/repeatCount
    float   discard;     // discardChanceOnAirExposure (0 or 1 for Tier-1)
    int32_t family;      // F_*
} OreCfg;

// 1.18 (MC_1_18 <= mc < MC_1_20) Tier-1 configs. Values transcribed from finders.c getOreConfig.
// One OreCfg per cubiomes ore *type* (a family can have several types, e.g. redstone + lower_redstone).
#define ORE_NCFG 9
#ifdef __CUDACC__
#define ORE_TABLE static __device__ const
#else
#define ORE_TABLE static const
#endif
ORE_TABLE OreCfg ORE_CFGS_118[ORE_NCFG] = {
    // index, step, size, repeat, hp,          h1,  h2, rare, discard, family
    {  8, 6, 64,  2, HP_UNIFORM,  -64,   0, 0, 0.0f, F_TUFF     }, // tuff
    { 16, 6,  8,  4, HP_UNIFORM,  -64,  15, 0, 0.0f, F_REDSTONE }, // redstone
    { 17, 6,  8,  8, HP_TRIANGLE, -96, -32, 0, 0.0f, F_REDSTONE }, // lower_redstone
    { 21, 6,  7,  2, HP_TRIANGLE, -32,  32, 0, 0.0f, F_LAPIS    }, // lapis
    { 22, 6,  7,  4, HP_UNIFORM,  -64,  64, 0, 1.0f, F_LAPIS    }, // buried_lapis (discard=1)
    {  1, 6, 33, 14, HP_UNIFORM,  -64, 319, 0, 0.0f, F_GRAVEL   }, // gravel
    {  3, 6, 64,  2, HP_UNIFORM,    0,  60, 0, 0.0f, F_GRANITE  }, // lower_granite
    {  2, 6, 64,  6, HP_UNIFORM,   64, 128, 1, 0.0f, F_GRANITE  }, // upper_granite (rare)
    { 24, 6, 10, 16, HP_TRIANGLE, -16, 112, 0, 0.0f, F_COPPER   }, // copper
};

typedef struct { int32_t x, y, z; } OrePos;
#ifndef __CUDACC__
typedef struct { int x, y, z; } int3;   // CPU fallback (CUDA provides int3)
#endif

// Emit sink for generated candidate blocks. Array mode (out!=0) used by the CPU driver; occ mode
// (out==0, CUDA only) writes a per-family occupancy bitmask directly + collects anchor candidates —
// no intermediate array, so it scales to large tiles.
typedef struct {
    OrePos *out; int *n; int cap;                 // array mode
    uint32_t *occ; long wpf; int fa;              // occ mode: bitmask base, words/family, family idx
    int X0, Z0, DX, DZ;                           // occ region (block origin + extent), y band -64..-1
    int3 *anchorList; int *anchorCount; int anchorCap, isAnchor;
    int icx0, icx1, icz0, icz1;                   // interior block bounds for anchor enumeration
} OreEmit;

ORE_HD static inline void oreEmit(OreEmit *em, int x, int y, int z){
    if(em->out){ if(*em->n < em->cap){ em->out[*em->n].x=x; em->out[*em->n].y=y; em->out[*em->n].z=z; (*em->n)++; } return; }
#ifdef __CUDA_ARCH__   // occ mode is device-only (atomics); host callers always use array mode
    if(y<-64||y>=0) return;
    if(x<em->X0||x>=em->X0+em->DX||z<em->Z0||z>=em->Z0+em->DZ) return;
    long idx=((long)(x-em->X0)*64+(y+64))*em->DZ+(z-em->Z0);
    atomicOr(&em->occ[(long)em->fa*em->wpf + (idx>>5)], 1u<<(idx&31));
    if(em->isAnchor && x>=em->icx0 && x<=em->icx1 && z>=em->icz0 && z<=em->icz1){
        int s=atomicAdd(em->anchorCount,1);
        if(s<em->anchorCap) em->anchorList[s]=make_int3(x,y,z);
    }
#endif
}

// ===================== RNG (xoroshiro128++, Java-faithful) =====================
typedef struct { uint64_t lo, hi; } XR;

ORE_HD static inline uint64_t ore_rotl(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }

ORE_HD static inline void xSetSeed(XR *xr, uint64_t value){
    const uint64_t XL=0x9e3779b97f4a7c15ULL, XH=0x6a09e667f3bcc909ULL;
    const uint64_t A=0xbf58476d1ce4e5b9ULL, B=0x94d049bb133111ebULL;
    uint64_t l=value^XH, h=l+XL;
    l=(l^(l>>30))*A; h=(h^(h>>30))*A;
    l=(l^(l>>27))*B; h=(h^(h>>27))*B;
    l=l^(l>>31); h=h^(h>>31);
    xr->lo=l; xr->hi=h;
}
ORE_HD static inline uint64_t xNextLong(XR *xr){
    uint64_t l=xr->lo, h=xr->hi, n=ore_rotl(l+h,17)+l;
    h^=l; xr->lo=ore_rotl(l,49)^h^(h<<21); xr->hi=ore_rotl(h,28);
    return n;
}
ORE_HD static inline uint64_t xNextLongJ(XR *xr){
    int32_t a=(int32_t)(xNextLong(xr)>>32), b=(int32_t)(xNextLong(xr)>>32);
    return ((uint64_t)a<<32)+b;
}
ORE_HD static inline int xNextIntJ(XR *xr, uint32_t n){
    int bits,val; const int m=n-1;
    if((m&n)==0){ uint64_t x=n*(xNextLong(xr)>>33); return (int)((int64_t)x>>31); }
    do { bits=(int)(xNextLong(xr)>>33); val=bits%n; }
    while((int32_t)((uint32_t)bits-val+m)<0);
    return val;
}
ORE_HD static inline float  xNextFloat(XR *xr){ return (xNextLong(xr)>>(64-24))*5.9604645E-8F; }
ORE_HD static inline double xNextDoubleJ(XR *xr){
    uint64_t a=xNextLong(xr), b=xNextLong(xr);
    return ((a>>(64-26)<<27)+(b>>(64-27)))*1.1102230246251565E-16;
}
ORE_HD static inline int xNextIntBetween(XR *xr, int mn, int mx){ return xNextIntJ(xr,(uint32_t)(mx-mn+1))+mn; }

ORE_HD static inline uint64_t getPopulationSeed(uint64_t ws, int x, int z){
    XR xr; xSetSeed(&xr, ws);
    uint64_t a=xNextLongJ(&xr)|1ULL, b=xNextLongJ(&xr)|1ULL;
    return ((uint64_t)x*a + (uint64_t)z*b) ^ ws;
}

ORE_HD static inline int ore_height(const OreCfg *c, XR *rnd){
    if(c->hp==HP_UNIFORM){
        if(c->h1>c->h2) return c->h1;
        return xNextIntBetween(rnd, c->h1, c->h2);
    } else { // triangle
        if(c->h1>c->h2) return c->h1;
        int range=c->h2-c->h1;
        if(range<=0) return xNextIntBetween(rnd, c->h1, c->h2);
        int mid=range/2, mid2=range-mid;
        int a=xNextIntBetween(rnd,0,mid2), b=xNextIntBetween(rnd,0,mid);
        return c->h1+a+b;
    }
}

// ---- bit array (matches cubiomes BITSET/BITTEST) ----
#define ORE_BITSET(a,b)  ((a)[(b)>>3] |= (char)(1<<((b)&7)))
#define ORE_BITTEST(a,b) ((a)[(b)>>3] &  (char)(1<<((b)&7)))

ORE_HD static inline int ore_floor(double v){ return (int)floor(v); }
ORE_HD static inline double ore_lerp(double p,double a,double b){ return a + p*(b-a); }

// Faithful port of generateVeinPart for discard in {0,1} (Tier-1). Appends to out[*n] (capacity cap).
ORE_HD static inline void generateVeinPart(const OreCfg *c, XR *rnd,
        double oXP,double oXN,double oZP,double oZN,double oYP,double oYN,
        int startX,int startY,int startZ,int oreSize,int radius,
        OreEmit *em)
{
    const int minBuildHeight=-64, maxBuildHeight=320;
    char bitSet[ORE_MAXSLOTS]; memset(bitSet,0,sizeof(bitSet));
    int size=c->size;
    double store[4*ORE_MAXSIZE];

    for(int i=0;i<size;++i){
        float percent=(float)i/(float)size;
        double x=ore_lerp(percent,oXP,oXN), y=ore_lerp(percent,oYP,oYN), z=ore_lerp(percent,oZP,oZN);
        double length=xNextDoubleJ(rnd)*(double)size/16.0;
        double offset=((sin((float)ORE_PI*percent)+1.0F)*length+1.0)/2.0;
        store[i*4]=x; store[i*4+1]=y; store[i*4+2]=z; store[i*4+3]=offset;
    }
    for(int i=0;i<size-1;++i){
        if(store[i*4+3]<=0.0) continue;
        for(int j=i+1;j<size;++j){
            if(store[j*4+3]<=0.0) continue;
            double dX=store[i*4]-store[j*4], dY=store[i*4+1]-store[j*4+1], dZ=store[i*4+2]-store[j*4+2];
            double off=store[i*4+3]-store[j*4+3];
            if(off*off<=dX*dX+dY*dY+dZ*dZ) continue;
            if(off>0.0) store[j*4+3]=-1.0; else store[i*4+3]=-1.0;
        }
    }
    for(int i=0;i<size;++i){
        double offset=store[i*4+3];
        if(offset<0.0) continue;
        double x=store[i*4], y=store[i*4+1], z=store[i*4+2];
        int minX=ore_floor(x-offset); if(minX<startX)minX=startX;
        int minY=ore_floor(y-offset); if(minY<startY)minY=startY;
        int minZ=ore_floor(z-offset); if(minZ<startZ)minZ=startZ;
        int maxX=ore_floor(x+offset); if(maxX<minX)maxX=minX;
        int maxY=ore_floor(y+offset); if(maxY<minY)maxY=minY;
        int maxZ=ore_floor(z+offset); if(maxZ<minZ)maxZ=minZ;
        for(int X=minX;X<=maxX;++X){
            double xS=((double)X+0.5-x)/offset; if(xS*xS>=1.0) continue;
            for(int Y=minY;Y<=maxY;++Y){
                double yS=((double)Y+0.5-y)/offset; if(xS*xS+yS*yS>=1.0) continue;
                for(int Z=minZ;Z<=maxZ;++Z){
                    double zS=((double)Z+0.5-z)/offset; if(xS*xS+yS*yS+zS*zS>=1.0) continue;
                    if(Y<minBuildHeight||Y>=maxBuildHeight) continue;
                    int area=X-startX+(Y-startY)*oreSize+(Z-startZ)*oreSize*radius;
                    if(ORE_BITTEST(bitSet,area)) continue;
                    ORE_BITSET(bitSet,area);
                    // Tier-1: discard 0 -> append; discard 1 -> append (no nextFloat). Both append.
                    oreEmit(em,X,Y,Z);
                }
            }
        }
    }
}

// One ore *type* in one chunk. Appends candidate block positions to out. ORE_SKIP_SURFACE: no gate.
ORE_HD static inline void generateOreType(uint64_t worldSeed,const OreCfg *c,int chunkX,int chunkZ,
        OreEmit *em)
{
    uint64_t popSeed=getPopulationSeed(worldSeed, chunkX<<4, chunkZ<<4);
    XR rnd; xSetSeed(&rnd, popSeed + (uint64_t)c->index + 10000ULL*(uint64_t)c->step);

    int repeat;
    if(c->rare) repeat = (xNextFloat(&rnd) < 1.0f/(float)c->repeatCount) ? 1 : 0;
    else repeat = c->repeatCount;

    for(int it=0; it<repeat; ++it){
        int bx=(chunkX<<4)+xNextIntJ(&rnd,16);
        int bz=(chunkZ<<4)+xNextIntJ(&rnd,16);
        int by=ore_height(c,&rnd);
        // isViableOreBiome == true (overworld) for all ported families; no RNG consumed.
        float angle=xNextFloat(&rnd)*(float)ORE_PI;
        float fsize=(float)c->size/8.0F;
        int amort=(int)ceil(((float)c->size/16.0F*2.0F+1.0F)/2.0F);
        double oXP=(double)bx+sin(angle)*(double)fsize, oXN=(double)bx-sin(angle)*(double)fsize;
        double oZP=(double)bz+cos(angle)*(double)fsize, oZN=(double)bz-cos(angle)*(double)fsize;
        double oYP=by+xNextIntJ(&rnd,3)-2, oYN=by+xNextIntJ(&rnd,3)-2;
        int startX=bx-(int)ceil(fsize)-amort, startY=by-2-amort, startZ=bz-(int)ceil(fsize)-amort;
        int oreSize=2*((int)ceil(fsize)+amort), radius=2*(2+amort);
#ifdef ORE_WITH_SURFACE
        // host must provide: int ore_surface_floor(int x,int z) -> floor(mapApproxHeight)
        int gated=0;
        for(int X=startX; X<=startX+oreSize && !gated; ++X)
            for(int Z=startZ; Z<=startZ+oreSize; ++Z)
                if(startY <= ore_surface_floor(X,Z)){ gated=1; break; }
        if(!gated) continue;
#endif
        generateVeinPart(c,&rnd,oXP,oXN,oZP,oZN,oYP,oYN,startX,startY,startZ,oreSize,radius,em);
    }
}

#endif
