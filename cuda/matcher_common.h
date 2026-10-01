// matcher_common.h — shared defs for the GPU matcher translation unit: family id maps, occupancy
// probes, result/observation structs, orientation transform. Included by matcher.cu,
// matcher_kernels.cuh, and matcher_refine.cuh (single TU — nvcc compiles matcher.cu only).
//
// Glossary of the terse names used across this TU (mirrors cubiomes source where noted):
//   occ          per-family occupancy bitmask; bit set => candidate ore block at that cell
//   wpf          occ words-per-family: uint32 count of one family's bitmask for the current tile
//   fa           compacted family index 0..6 (tuff,redstone,lapis,granite,gravel,copper,iron)
//   X0,Z0,DX,DZ  tile block-origin and block-extent; the y band is fixed to -64..-1
//   anchor       the sparsest observed GPU family; each of its cells seeds a candidate world origin
//   oXP/oXN      vein line-segment endpoints, ± of the base point (cubiomes naming)
//   vn / nf      per-vein node array (FP64 source) / hoisted FP32 box-local params (kFill)
//   e            match tolerance radius (Chebyshev, in blocks)
#ifndef MATCHER_COMMON_H
#define MATCHER_COMMON_H
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "oregen.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#define NGPU 4         // GPU-generated families: tuff,redstone,lapis,granite (occ idx 0..3)
#define NACTIVE 7      // + gravel(4), copper(5), iron(6) for CPU refine
#define N_ACTIVE_CFG 7
static __device__ const int ACTIVE_CFG[N_ACTIVE_CFG]  = {0,1,2,3,4,6,7}; // device (kGenerate)
static const int           ACTIVE_CFG_H[N_ACTIVE_CFG] = {0,1,2,3,4,6,7}; // host (refineTop)

ORE_HD static inline int famActive(int F){
    switch(F){ case F_TUFF:return 0; case F_REDSTONE:return 1; case F_LAPIS:return 2;
               case F_GRANITE:return 3; case F_GRAVEL:return 4; case F_COPPER:return 5; case F_IRON:return 6; }
    return -1;
}
static const char* FAMNAME[F_COUNT]={"tuff","redstone","lapis","gravel","granite","copper","iron"};
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#define DEVNULL "NUL"
#define PATHSEP "\\"
#define EXE_SUFFIX ".exe"
#else
#define DEVNULL "/dev/null"
#define PATHSEP "/"
#define EXE_SUFFIX ""
#endif
#define RD_RELPATH ".." PATHSEP "harness" PATHSEP "region_dump" EXE_SUFFIX
static char g_rdexe[600]=RD_RELPATH;   // resolved from argv[0] in main()

struct ObsCell { int x,y,z,fam; };
struct Result  { int ox,oy,oz; int r,mir,pres,absH; float fin; };

ORE_HD static inline void orient_xz(int x,int z,int r,int mir,int*ox,int*oz){
    x*=mir; for(int i=0;i<r;i++){ int nx=-z,nz=x; x=nx; z=nz; } *ox=x; *oz=z;
}
__device__ static inline bool occHit(const uint32_t*occ,long wpf,int fa,int x,int y,int z,int X0,int Z0,int DX,int DZ){
    if(x<X0||x>=X0+DX||y<-64||y>=0||z<Z0||z>=Z0+DZ) return false;
    long idx=((long)(x-X0)*64+(y+64))*DZ+(z-Z0);
    return (occ[(long)fa*wpf+(idx>>5)]>>(idx&31))&1u;
}
__device__ static inline bool occHitTol(const uint32_t*occ,long wpf,int fa,int x,int y,int z,int X0,int Z0,int DX,int DZ,int e){
    if(e==0) return occHit(occ,wpf,fa,x,y,z,X0,Z0,DX,DZ);
    for(int dx=-e;dx<=e;dx++)for(int dy=-e;dy<=e;dy++)for(int dz=-e;dz<=e;dz++)
        if(occHit(occ,wpf,fa,x+dx,y+dy,z+dz,X0,Z0,DX,DZ)) return true;
    return false;
}
#endif
