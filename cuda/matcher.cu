// matcher.cu — world-scale GPU ore-pattern localizer (P3.3). Tiles an arbitrary chunk region so memory
// is bounded regardless of search size (300k x 300k and beyond). Two passes:
//   PASS 1 (GPU, tiled): per tile, generate the 4 bit-exact families (tuff/redstone/lapis/granite) into a
//     reused occupancy bitmask, enumerate the anchor family's candidates in the tile interior, and score
//     (presence + soft absence) with an aggressive presence pre-filter + early-termination. Survivors
//     (compacted) merge into a global top-K.
//   PASS 2 (CPU refine): re-score the top-K with all 7 families (gravel/copper/iron from region_dump.exe)
//     so their margin is recovered without per-tile injection. iron is not bit-exact (ore-vein noise).
// Validated vs solve.py + the single-region matcher on the real test world (cuda/README, NOTES P3).
//
// Usage: matcher <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv>
//        [--error E] [--absw W] [--minfrac F] [--tile T] [--topk K] [--refine N] [--no-refine]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
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
static char g_rdexe[600]="..\\harness\\region_dump.exe";   // resolved from argv[0] in main()

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

__global__ void kGenerate(uint64_t seed,int ocx0,int ocz0,int gcx,int gcz,
        int X0,int Z0,int DX,int DZ,uint32_t*occ,long wpf,int anchorFamily,
        int icx0,int icx1,int icz0,int icz1,int3*anchorList,int*anchorCount,int anchorCap,
        const int*selCfg,int nSel){
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=gcx*gcz) return;
    int cx=ocx0+tid%gcx, cz=ocz0+tid/gcx;
    OreEmit em; em.out=0; em.occ=occ; em.wpf=wpf; em.X0=X0; em.Z0=Z0; em.DX=DX; em.DZ=DZ;
    em.anchorList=anchorList; em.anchorCount=anchorCount; em.anchorCap=anchorCap;
    em.icx0=icx0; em.icx1=icx1; em.icz0=icz0; em.icz1=icz1;
    for(int c=0;c<nSel;c++){
        const OreCfg*cfg=&ORE_CFGS_118[selCfg[c]];
        em.fa=famActive(cfg->family); em.isAnchor=(cfg->family==anchorFamily);
        generateOreType(seed,cfg,cx,cz,&em);
    }
}

// ---- Two-kernel generator (the divergence fix). The 1-thread-per-chunk kGenerate ran the fill at
// ~5/32 active lanes (ncu); a warp-cooperative single kernel backfired because serializing the RNG +
// cull on lane 0 (which legacy already does at full utilization) cost more than the fill win. The
// split keeps each part where it's efficient:
//   kSetup — 1 thread per (chunk,config), config-major so a warp shares one config (uniform loop
//            bounds => low divergence). Runs the RNG chain + order-dependent cull, compacts each
//            vein's surviving node list to global scratch via atomic counters.
//   kFill  — persistent warps, atomic vein queue, 1 warp per vein: stripe the sphere-fill cells
//            across 32 lanes (full utilization). Per-warp shared bitSet replicates cubiomes'
//            bounded-index collision-suppression so occupancy stays bit-exact.
#define COOP_WPB 4
#define COOP_BLK (COOP_WPB*32)
#define FULLMASK 0xffffffffu
#define CULL_W 8           // kFill containment-cull neighbor window (occ-invariant; tunes cull vs fill cost)

struct VeinRec { int nodeOff,nNodes,startX,startY,startZ,oreSize,radius,fa,isAnchor; };

__global__ void kSetup(uint64_t seed,int ocx0,int ocz0,int gcx,int gcz,int anchorFamily,
        double* nodes,int nodeCap,int* nodeCnt,VeinRec* veins,int veinCap,int* veinCnt,
        const int* selCfg,int nSel){
    long tid=(long)blockIdx.x*blockDim.x+threadIdx.x;
    long nChunks=(long)gcx*gcz, total=nChunks*nSel;
    if(tid>=total) return;
    int cfgI=(int)(tid/nChunks); long chunkLin=tid%nChunks;     // config-major: warp shares one config
    int cx=ocx0+(int)(chunkLin%gcx), cz=ocz0+(int)(chunkLin/gcx);
    const OreCfg* c=&ORE_CFGS_118[selCfg[cfgI]];
    int fa=famActive(c->family), isAnc=(c->family==anchorFamily);
    XR rnd; uint64_t popSeed=getPopulationSeed(seed,cx<<4,cz<<4);
    xSetSeed(&rnd, popSeed + (uint64_t)c->index + 10000ULL*(uint64_t)c->step);
    int repeat = c->rare ? ((xNextFloat(&rnd)<1.0f/(float)c->repeatCount)?1:0) : c->repeatCount;
    int size=c->size;
    for(int it=0; it<repeat; ++it){
        int bx=(cx<<4)+xNextIntJ(&rnd,16);
        int bz=(cz<<4)+xNextIntJ(&rnd,16);
        int by=ore_height(c,&rnd);
        float angle=xNextFloat(&rnd)*(float)ORE_PI;
        float fsize=(float)c->size/8.0F;
        int amort=(int)ceil(((float)c->size/16.0F*2.0F+1.0F)/2.0F);
        double oXP=(double)bx+sin(angle)*(double)fsize, oXN=(double)bx-sin(angle)*(double)fsize;
        double oZP=(double)bz+cos(angle)*(double)fsize, oZN=(double)bz-cos(angle)*(double)fsize;
        double oYP=by+xNextIntJ(&rnd,3)-2, oYN=by+xNextIntJ(&rnd,3)-2;
        int startX=bx-(int)ceil(fsize)-amort, startY=by-2-amort, startZ=bz-(int)ceil(fsize)-amort;
        int oreSize=2*((int)ceil(fsize)+amort), radius=2*(2+amort);
        // The overlap cull (removing nodes whose sphere is contained in a neighbor's) is occ-invariant and
        // RNG-free, so it's deferred to kFill -> nS==size is known here. Reserve the node slots first and
        // stream each node STRAIGHT to global scratch; no local staging array (was a 2KB/thread stack frame
        // = the LG-throttle source in kSetup). RNG must still advance fully even on overflow for bit-exactness.
        int nOff=atomicAdd(nodeCnt,size), vIdx=atomicAdd(veinCnt,1);
        int ovf=(vIdx>=veinCap || nOff+size>nodeCap);     // caps sized for max tile; overflow is the rare path
        for(int i=0;i<size;++i){
            float percent=(float)i/(float)size;
            double x=ore_lerp(percent,oXP,oXN), y=ore_lerp(percent,oYP,oYN), z=ore_lerp(percent,oZP,oZN);
            double length=xNextDoubleJ(&rnd)*(double)size/16.0;
            double offset=((sin((float)ORE_PI*percent)+1.0F)*length+1.0)/2.0;
            if(!ovf){ long b=(long)(nOff+i)*4; nodes[b]=x; nodes[b+1]=y; nodes[b+2]=z; nodes[b+3]=offset; }
        }
        if(ovf) continue;
        VeinRec r; r.nodeOff=nOff; r.nNodes=size; r.startX=startX; r.startY=startY; r.startZ=startZ;
        r.oreSize=oreSize; r.radius=radius; r.fa=fa; r.isAnchor=isAnc; veins[vIdx]=r;
    }
}

__global__ void __launch_bounds__(COOP_BLK) kFill(int X0,int Z0,int DX,int DZ,uint32_t*occ,long wpf,
        int icx0,int icx1,int icz0,int icz1,int3*anchorList,int*anchorCount,int anchorCap,
        const double* nodes,const VeinRec* veins,int nVeins,int* veinWork){
    // Column-analytic union (Option C): instead of classifying every cell of every node's bounding box and
    // de-duping in a shared bitSet, stripe the vein-box (Xr,Yr) COLUMNS across the warp. Each lane owns whole
    // columns, unions the per-node z-intervals into a private 64-bit register mask, then batch-writes the
    // column's z-run to occ. This removes the per-cell shared atomicOr (the profiler's #1 stall) AND the
    // per-cell integer-division decode. occ stays a union, so output is byte-identical to the per-cell path.
    // Nodes are read straight from global (L1-resident, <=2KB/vein); no shared node cache.
    __shared__ unsigned char sAlive[COOP_WPB][ORE_MAXSIZE];
    __shared__ float sNF[COOP_WPB][ORE_MAXSIZE*4];   // per-node FP32 params (xrf,yrf,zrf,foff2), hoisted
    int lane=threadIdx.x&31, w=threadIdx.x>>5;
    unsigned char* alive=sAlive[w]; float* nf=sNF[w];
    const float EPS=1e-2f;
    for(;;){
        int v;
        if(lane==0) v=atomicAdd(veinWork,1);
        v=__shfl_sync(FULLMASK,v,0);
        if(v>=nVeins) break;
        VeinRec r=veins[v];
        int nN=r.nNodes;
        const double* vn=nodes+(long)r.nodeOff*4;     // bit-exact global node source (x,y,z,offset per node)
        for(int i=lane;i<nN;i+=32) alive[i]=1;
        __syncwarp();
        // Parallel containment cull (occ-invariant: a missed node is just unioned redundantly). Nodes lie on
        // a lerp line, so containment is almost always by a near neighbor -> +-CULL_W catches it at O(size*W).
        for(int j=lane;j<nN;j+=32){
            double xj=vn[j*4],yj=vn[j*4+1],zj=vn[j*4+2],oj=vn[j*4+3];
            int lo=j-CULL_W; if(lo<0)lo=0; int hi=j+CULL_W; if(hi>=nN)hi=nN-1;
            for(int i=lo;i<=hi;i++){ if(i==j) continue;
                double off=vn[i*4+3]-oj; if(off<=0.0) continue;
                double dX=vn[i*4]-xj, dY=vn[i*4+1]-yj, dZ=vn[i*4+2]-zj;
                if(off*off>dX*dX+dY*dY+dZ*dZ){ alive[j]=0; break; }   // j contained in nearby i
            }
        }
        __syncwarp();
        // Hoist each node's FP32 box-local params out of the column loop (else 3 FP64 reductions per node PER
        // column -> FP64 blowup on Ada). The rare boundary-shell FP64 still reads the original doubles below.
        for(int i=lane;i<nN;i+=32){ double o=vn[i*4+3];
            nf[i*4]=(float)(vn[i*4]-r.startX); nf[i*4+1]=(float)(vn[i*4+1]-r.startY);
            nf[i*4+2]=(float)(vn[i*4+2]-r.startZ); nf[i*4+3]=(float)(o*o); }
        __syncwarp();
        int xext=r.oreSize, yext=r.radius, zext=r.oreSize;   // vein-box extents (X,Y,Z)
        long fbase=(long)r.fa*wpf;
        for(int col=lane; col<xext*yext; col+=32){
            int Xr=col%xext, Yr=col/xext;
            int X=r.startX+Xr, Y=r.startY+Yr;
            if(Y<-64||Y>=0) continue;                         // occ y-band (matches oreEmit)
            if(X<X0||X>=X0+DX) continue;                      // tile x bound
            int zc0=(Z0>r.startZ)?(Z0-r.startZ):0;            // lowest/highest in-tile Zr
            int zc1=zext-1; if(r.startZ+zc1>Z0+DZ-1) zc1=Z0+DZ-1-r.startZ;
            if(zc0>zc1) continue;
            uint64_t mask=0;                                  // bit Zr set if any node covers (X,Y,startZ+Zr)
            for(int i=0;i<nN;++i){
                if(!alive[i]) continue;
                float xrf=nf[i*4], yrf=nf[i*4+1], zrf=nf[i*4+2], foff2=nf[i*4+3];
                float dxf=(float)Xr+0.5f-xrf, dyf=(float)Yr+0.5f-yrf;
                float pf=foff2-dxf*dxf-dyf*dyf;               // z-slack (offset^2 - radial^2); <0 => column misses
                if(pf<=-EPS) continue;
                float sf=(pf>0.0f)?sqrtf(pf):0.0f;
                // FP32 candidate z-window (+-1 margin); exact membership decided per cell so the window only
                // needs to be a superset of the true contiguous interval. Same predicate as the per-cell path.
                int zlo=(int)floorf(zrf-0.5f-sf)-1, zhi=(int)floorf(zrf-0.5f+sf)+1;
                if(zlo<zc0)zlo=zc0; if(zhi>zc1)zhi=zc1;
                for(int Zr=zlo; Zr<=zhi; Zr++){
                    float dzf=(float)Zr+0.5f-zrf, q=dxf*dxf+dyf*dyf+dzf*dzf;
                    if(q>=foff2+EPS) continue;                                  // FP32-confident: outside
                    if(q>foff2-EPS){                                           // boundary shell: settle in FP64
                        double x=vn[i*4],y=vn[i*4+1],z=vn[i*4+2],offset=vn[i*4+3];
                        double ddx=(double)X+0.5-x, ddy=(double)Y+0.5-y, ddz=(double)(r.startZ+Zr)+0.5-z;
                        if(ddx*ddx+ddy*ddy+ddz*ddz>=offset*offset) continue;
                    }
                    mask|=(1ULL<<Zr);
                }
            }
            if(!mask) continue;
            // Batched occ write: deposit the column's z-run (>=1 atomics) instead of one atomic per cell.
            long base0=((long)(X-X0)*64+(Y+64))*(long)DZ + ((long)r.startZ+zc0-Z0);   // occ bit of Zr=zc0, >=0
            uint64_t m2=mask>>zc0;                            // align lowest in-tile Zr to bit 0
            int pw=(int)(base0>>5), psh=(int)(base0&31);
            uint64_t dep=m2<<psh;
            if((uint32_t)dep) atomicOr(&occ[fbase+pw],(uint32_t)dep);
            uint32_t hi=(uint32_t)(dep>>32); if(hi) atomicOr(&occ[fbase+pw+1],hi);
            if(psh){ uint32_t ov=(uint32_t)(m2>>(64-psh)); if(ov) atomicOr(&occ[fbase+pw+2],ov); }
            if(r.isAnchor && X>=icx0 && X<=icx1){
                uint64_t am=m2;
                while(am){ int b=__ffsll((long long)am)-1; am&=am-1; int Z=r.startZ+zc0+b;
                    if(Z>=icz0 && Z<=icz1){ int a=atomicAdd(anchorCount,1); if(a<anchorCap) anchorList[a]=make_int3(X,Y,Z); } }
            }
        }
        __syncwarp();
    }
}

// Morton key over tile-local (x,z) so a spatial sort of anchors gives kScore L1/L2 locality: adjacent
// threads then probe overlapping occ[] footprints. x,z-X0/Z0 fit in 12 bits (tile <=4096 blocks).
__device__ static inline uint32_t mortonPart(uint32_t n){
    n&=0x0000ffffu; n=(n|(n<<8))&0x00FF00FFu; n=(n|(n<<4))&0x0F0F0F0Fu;
    n=(n|(n<<2))&0x33333333u; n=(n|(n<<1))&0x55555555u; return n;
}
__global__ void kAnchorKey(const int3*a,int n,int X0,int Z0,uint32_t*keys){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    keys[i]=mortonPart((uint32_t)(a[i].x-X0))|(mortonPart((uint32_t)(a[i].z-Z0))<<1);
}

__global__ void kScore(const int3*anchorList,int nAnchor,const ObsCell*ore,int nOre,
        const ObsCell*bare,int nBare,int oax,int oay,int oaz,const uint32_t*occ,long wpf,
        int X0,int Z0,int DX,int DZ,int e,float w,int minPres,Result*out,int*outCount,int outCap){
    long tid=(long)blockIdx.x*blockDim.x+threadIdx.x; if(tid>=(long)nAnchor*8) return;
    int ai=(int)(tid/8),orient=(int)(tid%8),r=orient&3,mir=(orient<4)?1:-1;
    int3 a=anchorList[ai];
    int odx,odz; orient_xz(oax,oaz,r,mir,&odx,&odz);
    int ox=a.x-odx,oy=a.y-oay,oz=a.z-odz,pres=0;
    for(int i=0;i<nOre;i++){
        int dx,dz; orient_xz(ore[i].x,ore[i].z,r,mir,&dx,&dz);
        if(occHitTol(occ,wpf,ore[i].fam,ox+dx,oy+ore[i].y,oz+dz,X0,Z0,DX,DZ,e)) pres++;
        if(pres+(nOre-1-i)<minPres) return;          // early-out: can't reach threshold
    }
    int absH=0;
    for(int i=0;i<nBare;i++){
        int dx,dz; orient_xz(bare[i].x,bare[i].z,r,mir,&dx,&dz);
        int wx=ox+dx,wy=oy+bare[i].y,wz=oz+dz; bool any=false;
        for(int fa=0;fa<NGPU;fa++) if(occHitTol(occ,wpf,fa,wx,wy,wz,X0,Z0,DX,DZ,e)){any=true;break;}
        if(any) absH++;
    }
    int s=atomicAdd(outCount,1);
    if(s<outCap){ Result rr; rr.ox=ox;rr.oy=oy;rr.oz=oz;rr.r=r;rr.mir=mir;rr.pres=pres;rr.absH=absH;
        rr.fin=pres-w*absH; out[s]=rr; }
}

static void loadObs(const char*path,std::vector<ObsCell>&ore,std::vector<ObsCell>&bare){
    FILE*f=fopen(path,"r"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    char line[256];
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='#'||line[0]=='\n') continue;
        char fam[32]; int x,y,z;
        if(sscanf(line,"%31[^,],%d,%d,%d",fam,&x,&y,&z)!=4) continue;
        if(!strcmp(fam,"family")) continue;
        if(!strcmp(fam,"bare")){ bare.push_back({x,y,z,-1}); continue; }
        int F=-1; for(int i=0;i<F_COUNT;i++) if(!strcmp(fam,FAMNAME[i])){F=i;break;}
        int fa=famActive(F); if(fa>=0) ore.push_back({x,y,z,fa});
    }
    fclose(f);
}

// ---- CPU refine: full 7-family score of one hypothesis over a small window (gravel/copper/iron injected) ----
static inline long key3(int x,int y,int z){ return (((long)(x+1000000))*512 + (y+64))* (long)2000000 + (z+1000000); }
struct Refined { Result r; int pres6; int absH6; float fin6; };

static void refineTop(uint64_t seed,std::vector<Result>&top,int nRefine,
        const std::vector<ObsCell>&ore,const std::vector<ObsCell>&bare,
        int maxExt,float w,int gravelMax,std::vector<Refined>&outv){
    int marginCh=maxExt/16+2;
    float topFin = top.empty()?0:top[0].fin;
    // gravel/copper can add at most gravelMax; results further below the pass-1 top can't overtake. Always
    // refine >=8 so a margin can be shown; beyond that, stop once no contender remains. top is fin-descending
    // and topFin is fixed, so the cutoff is monotonic -> precompute nDo, making the loop body dependency-free
    // (each hypothesis: independent host-gen + its own region_dump spawn) and safe to run in parallel.
    int nDo=std::min((int)top.size(),nRefine);
    for(int t=8;t<nDo;t++) if(top[t].fin < topFin-(float)gravelMax){ nDo=t; break; }
    outv.assign(nDo,Refined{});                    // index-write per t (no push_back race)
    #pragma omp parallel for schedule(dynamic)
    for(int t=0;t<nDo;t++){
        Result R=top[t];
        // window covering the oriented footprint
        int wx0=R.ox-maxExt,wx1=R.ox+maxExt,wz0=R.oz-maxExt,wz1=R.oz+maxExt;
        int cx0=(wx0>>4)-marginCh,cx1=(wx1>>4)+marginCh,cz0=(wz0>>4)-marginCh,cz1=(wz1>>4)+marginCh;
        std::unordered_set<long> occ[NACTIVE];   // 0-3 GPU, 4 gravel, 5 copper, 6 iron
        std::vector<OrePos> gb(200000);          // per-thread scratch (was a shared static; not reentrant)
        // 4 GPU families on host
        for(int cx=cx0;cx<=cx1;cx++)for(int cz=cz0;cz<=cz1;cz++)
            for(int c=0;c<N_ACTIVE_CFG;c++){
                const OreCfg*cfg=&ORE_CFGS_118[ACTIVE_CFG_H[c]];
                int n=0; OreEmit em; memset(&em,0,sizeof(em)); em.out=gb.data(); em.n=&n; em.cap=200000;
                generateOreType(seed,cfg,cx,cz,&em);
                int fa=famActive(cfg->family);
                for(int k=0;k<n;k++) if(gb[k].y>=-64&&gb[k].y<0) occ[fa].insert(key3(gb[k].x,gb[k].y,gb[k].z));
            }
        // gravel + copper + iron from region_dump.exe (one call, all three families)
        { char cmd[700]; snprintf(cmd,sizeof(cmd),"\"%s\" %llu 1.18 %d %d %d %d -64 -1 gravel copper iron 2>NUL",
                g_rdexe,(unsigned long long)seed,cx0,cx1,cz0,cz1);
            FILE*p=_popen(cmd,"r"); if(p){ char ln[128];
                while(fgets(ln,sizeof(ln),p)){ char f[32]; int x,y,z;
                    if(sscanf(ln,"%31[^,],%d,%d,%d",f,&x,&y,&z)!=4||y<-64||y>=0) continue;
                    int fa = !strcmp(f,"gravel")?4 : !strcmp(f,"copper")?5 : !strcmp(f,"iron")?6 : -1;
                    if(fa>=0) occ[fa].insert(key3(x,y,z)); }
                _pclose(p); } }
        // score this hypothesis with all 7 families
        int pres=0; for(auto&o:ore){ int dx,dz; orient_xz(o.x,o.z,R.r,R.mir,&dx,&dz);
            if(occ[o.fam].count(key3(R.ox+dx,R.oy+o.y,R.oz+dz))) pres++; }
        int absH=0; for(auto&b:bare){ int dx,dz; orient_xz(b.x,b.z,R.r,R.mir,&dx,&dz);
            long k=key3(R.ox+dx,R.oy+b.y,R.oz+dz); bool any=false;
            for(int fa=0;fa<NACTIVE;fa++) if(occ[fa].count(k)){any=true;break;} if(any) absH++; }
        outv[t]={R,pres,absH,pres-w*absH};
    }
    std::sort(outv.begin(),outv.end(),[](const Refined&a,const Refined&b){return a.fin6>b.fin6;});
}

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv> [--error E] [--absw W] [--minfrac F] [--tile T] [--topk K] [--refine N] [--no-refine] [--legacy-gen]\n",argv[0]); return 2; }
    uint64_t seed=(uint64_t)strtoll(argv[1],NULL,10);
    // resolve region_dump.exe relative to this exe's dir (../harness/), robust to CWD
    { const char*a=argv[0]; int cut=-1; for(int i=0;a[i];i++) if(a[i]=='/'||a[i]=='\\') cut=i;
      if(cut>=0) snprintf(g_rdexe,sizeof(g_rdexe),"%.*s\\..\\harness\\region_dump.exe",cut,a); }
    int cxMin=atoi(argv[2]),cxMax=atoi(argv[3]),czMin=atoi(argv[4]),czMax=atoi(argv[5]);
    const char* obspath=argv[6];
    int e=0,tile=256,topk=4096,nRefine=64; float w=1.0f,minFrac=0.5f; bool refine=true; bool useCoop=true; bool gateOff=false;
    for(int i=7;i<argc;i++){
        if(!strcmp(argv[i],"--legacy-gen")) useCoop=false; else if(!strcmp(argv[i],"--coop")) useCoop=true; else
        if(!strcmp(argv[i],"--error")&&i+1<argc) e=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--absw")&&i+1<argc) w=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--minfrac")&&i+1<argc) minFrac=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--tile")&&i+1<argc) tile=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--topk")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--refine")&&i+1<argc) nRefine=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--no-refine")) refine=false;
        else if(!strcmp(argv[i],"--no-gate")) gateOff=true;
    }
    if(tile>320){ fprintf(stderr,"warning: tile>320 risks a Windows TDR kill (kScore launch >2s); clamping to 256\n"); tile=256; }
    std::vector<ObsCell> ore,bare; loadObs(obspath,ore,bare);
    int cnt[NACTIVE]={0}; for(auto&o:ore) cnt[o.fam]++;
    // PASS1 occ holds only the NGPU generated families, so it can only score those. Partition the GPU
    // families to the front and bound kScore's presence loop to them; gravel/copper/iron are scored
    // only at refine (host occ, full NACTIVE). (Reading occ for fam>=NGPU here would be out of bounds.)
    std::stable_partition(ore.begin(),ore.end(),[](const ObsCell&o){return o.fam<NGPU;});
    int nOreGpu=0; for(auto&o:ore) if(o.fam<NGPU) nOreGpu++;
    int anchorFa=-1,best=1<<30;
    for(int fa=0;fa<NGPU;fa++) if(cnt[fa]>0&&cnt[fa]<best){best=cnt[fa];anchorFa=fa;}
    if(anchorFa<0){fprintf(stderr,"no GPU-generated ore family in observation\n");return 1;}
    int anchorFamily=-1; for(int F=0;F<F_COUNT;F++) if(famActive(F)==anchorFa){anchorFamily=F;break;}
    int oax=0,oay=0,oaz=0; for(auto&o:ore) if(o.fam==anchorFa){oax=o.x;oay=o.y;oaz=o.z;break;}
    int maxExt=1; for(auto&o:ore) maxExt=std::max(maxExt,std::max(abs(o.x),abs(o.z)));
    for(auto&b:bare) maxExt=std::max(maxExt,std::max(abs(b.x),abs(b.z)));
    int marginCh=maxExt/16+2, minPres=(int)(minFrac*nOreGpu);

    long totCh=(long)(cxMax-cxMin+1)*(czMax-czMin+1);
    printf("obs: %zu ore + %zu bare | search %ld chunks | anchor=%s(%d) tile=%d margin=%dch minfrac=%.2f e=%d w=%.1f\n",
        ore.size(),bare.size(),totCh,FAMNAME[anchorFamily],best,tile,marginCh,minFrac,e,w);

    // Family-gated generation: only generate the GPU families this observation scores. kScore presence
    // reads just the obs ore families; absence (bare cells) probes ALL NGPU families, so fall back to the
    // full set whenever any bare cell is present. Families seed RNG independently (oregen.h), so skipping
    // a family is bit-exact for the rest. Big win when the obs is rare-only (drops the dense tuff fill).
    std::vector<int> selCfg; bool needAll=gateOff||!bare.empty();
    for(int c=0;c<N_ACTIVE_CFG;c++){ int fa=famActive(ORE_CFGS_118[ACTIVE_CFG_H[c]].family);
        if(needAll||cnt[fa]>0) selCfg.push_back(ACTIVE_CFG_H[c]); }
    int nSel=(int)selCfg.size(); int* dSelCfg; CK(cudaMalloc(&dSelCfg,nSel*sizeof(int)));
    CK(cudaMemcpy(dSelCfg,selCfg.data(),nSel*sizeof(int),cudaMemcpyHostToDevice));
    { printf("gen-gating: %d/%d configs [", nSel, N_ACTIVE_CFG);
      bool seen[NGPU]={false}; for(int s:selCfg){ int fa=famActive(ORE_CFGS_118[s].family);
          if(!seen[fa]){ printf("%s%s", FAMNAME[ORE_CFGS_118[s].family], ""); seen[fa]=true; } }
      printf("]%s\n", needAll?" (all GPU families: bare-absence)":" (rare-gated)"); }

    // buffers sized for the largest tile (+margin)
    int sideMax=tile+2*marginCh; long DXm=(long)sideMax*16, volM=DXm*64*DXm, wpfMax=(volM+31)/32;
    uint32_t* dOcc; CK(cudaMalloc(&dOcc,(long)NGPU*wpfMax*4));
    int anchorCap=32000000; int3* dAnchor; CK(cudaMalloc(&dAnchor,(long)anchorCap*sizeof(int3)));
    uint32_t* dKeys; CK(cudaMalloc(&dKeys,(long)anchorCap*sizeof(uint32_t)));   // Morton keys for anchor sort
    int* dAcnt; CK(cudaMalloc(&dAcnt,4));
    int outCap=4000000; Result* dOut; CK(cudaMalloc(&dOut,(long)outCap*sizeof(Result)));
    int* dOcnt; CK(cudaMalloc(&dOcnt,4));
    int* dWork; CK(cudaMalloc(&dWork,4));
    int numSM=1; { cudaDeviceProp p; if(cudaGetDeviceProperties(&p,0)==cudaSuccess) numSM=p.multiProcessorCount; }
    int coopBlocks=numSM*16;   // persistent kFill warps; extras drain the vein queue and exit
    // two-kernel generator scratch (compacted vein node lists), sized for the largest tile
    double *dNodes=0; VeinRec *dVeins=0; int *dNodeCnt=0,*dVeinCnt=0; int nodeCap=0,veinCap=0;
    if(useCoop){
        long maxCh=(long)sideMax*sideMax;
        nodeCap=(int)std::min((long)500000000,maxCh*512);   // ~<=512 surviving nodes/chunk worst case
        veinCap=(int)std::min((long)60000000,maxCh*24);
        CK(cudaMalloc(&dNodes,(long)nodeCap*4*sizeof(double)));
        CK(cudaMalloc(&dVeins,(long)veinCap*sizeof(VeinRec)));
        CK(cudaMalloc(&dNodeCnt,4)); CK(cudaMalloc(&dVeinCnt,4));
        printf("generator: two-kernel (kSetup+kFill), nodeCap=%dM veinCap=%dM, %d kFill blocks x %d\n",
            nodeCap/1000000,veinCap/1000000,coopBlocks,COOP_BLK);
    } else printf("generator: legacy (1 thread/chunk, bit-exact reference)\n");
    ObsCell *dOre,*dBare; CK(cudaMalloc(&dOre,ore.size()*sizeof(ObsCell)));
    CK(cudaMalloc(&dBare,std::max((size_t)1,bare.size())*sizeof(ObsCell)));
    CK(cudaMemcpy(dOre,ore.data(),ore.size()*sizeof(ObsCell),cudaMemcpyHostToDevice));
    if(bare.size()) CK(cudaMemcpy(dBare,bare.data(),bare.size()*sizeof(ObsCell),cudaMemcpyHostToDevice));

    std::vector<Result> top; long long totAnchor=0,totSurv=0; int nTiles=0;   // 64-bit: counts hit billions at world scale (Windows long is 32-bit)
    cudaEvent_t evA,evB,evC,evS; CK(cudaEventCreate(&evA)); CK(cudaEventCreate(&evB)); CK(cudaEventCreate(&evC)); CK(cudaEventCreate(&evS));
    float msGen=0,msScore=0,msSetup=0;
    for(int tcx0=cxMin;tcx0<=cxMax;tcx0+=tile){
      for(int tcz0=czMin;tcz0<=czMax;tcz0+=tile){
        nTiles++;
        int tcx1=std::min(tcx0+tile-1,cxMax), tcz1=std::min(tcz0+tile-1,czMax);
        int ocx0=tcx0-marginCh,ocz0=tcz0-marginCh, ocx1=tcx1+marginCh,ocz1=tcz1+marginCh;
        int gcx=ocx1-ocx0+1,gcz=ocz1-ocz0+1;
        int X0=ocx0*16,Z0=ocz0*16,DX=gcx*16,DZ=gcz*16; long wpf=((long)DX*64*DZ+31)/32;
        CK(cudaMemset(dOcc,0,(long)NGPU*wpf*4)); CK(cudaMemset(dAcnt,0,4));
        CK(cudaEventRecord(evA));
        if(useCoop){
            CK(cudaMemset(dNodeCnt,0,4)); CK(cudaMemset(dVeinCnt,0,4)); CK(cudaMemset(dWork,0,4));
            long nItems=(long)gcx*gcz*nSel;
            kSetup<<<(int)((nItems+255)/256),256>>>(seed,ocx0,ocz0,gcx,gcz,anchorFamily,
                dNodes,nodeCap,dNodeCnt,dVeins,veinCap,dVeinCnt,dSelCfg,nSel);
            CK(cudaEventRecord(evS));
            int nVeins; CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            { float ms; CK(cudaEventElapsedTime(&ms,evA,evS)); msSetup+=ms; }
            CK(cudaMemcpy(&nVeins,dVeinCnt,4,cudaMemcpyDeviceToHost));
            int nNodesUsed; CK(cudaMemcpy(&nNodesUsed,dNodeCnt,4,cudaMemcpyDeviceToHost));
            if(nVeins>veinCap||nNodesUsed>nodeCap){ fprintf(stderr,"scratch overflow (veins %d/%d nodes %d/%d) — raise caps\n",nVeins,veinCap,nNodesUsed,nodeCap); return 1; }
            kFill<<<coopBlocks,COOP_BLK>>>(X0,Z0,DX,DZ,dOcc,wpf,
                tcx0*16,tcx1*16+15,tcz0*16,tcz1*16+15,dAnchor,dAcnt,anchorCap,dNodes,dVeins,nVeins,dWork);
        } else {
            kGenerate<<<(gcx*gcz+63)/64,64>>>(seed,ocx0,ocz0,gcx,gcz,X0,Z0,DX,DZ,dOcc,wpf,anchorFamily,
                tcx0*16,tcx1*16+15,tcz0*16,tcz1*16+15,dAnchor,dAcnt,anchorCap,dSelCfg,nSel);
        }
        CK(cudaEventRecord(evB)); CK(cudaEventSynchronize(evB));
        { float ms; CK(cudaEventElapsedTime(&ms,evA,evB)); msGen+=ms; }
        int nAnchor; CK(cudaMemcpy(&nAnchor,dAcnt,4,cudaMemcpyDeviceToHost));
        if(nAnchor>anchorCap) nAnchor=anchorCap; totAnchor+=nAnchor;
        if(!nAnchor) continue;
        CK(cudaMemset(dOcnt,0,4));
        long nHyp=(long)nAnchor*8;
        CK(cudaEventRecord(evB));
        kAnchorKey<<<(nAnchor+255)/256,256>>>(dAnchor,nAnchor,X0,Z0,dKeys);   // spatial sort for occ[] locality
        thrust::sort_by_key(thrust::device,dKeys,dKeys+nAnchor,dAnchor);
        kScore<<<(nHyp+127)/128,128>>>(dAnchor,nAnchor,dOre,nOreGpu,dBare,(int)bare.size(),
            oax,oay,oaz,dOcc,wpf,X0,Z0,DX,DZ,e,w,minPres,dOut,dOcnt,outCap);
        CK(cudaEventRecord(evC)); CK(cudaEventSynchronize(evC));
        { float ms; CK(cudaEventElapsedTime(&ms,evB,evC)); msScore+=ms; }
        int nSurv; CK(cudaMemcpy(&nSurv,dOcnt,4,cudaMemcpyDeviceToHost));
        if(nSurv>outCap) nSurv=outCap; totSurv+=nSurv;
        if(!nSurv) continue;
        std::vector<Result> surv(nSurv); CK(cudaMemcpy(surv.data(),dOut,(long)nSurv*sizeof(Result),cudaMemcpyDeviceToHost));
        for(auto&s:surv) top.push_back(s);
        // keep global top-K: highest-fin first, spatial NMS within Chebyshev sep. The near-kept lookup is
        // a bucket grid (cell=sep) instead of an O(K^2) scan — at world scale this merge ran per-tile over
        // topk(=4096) survivors. Buckets are sep-wide so any point within sep lies in the 3x3x3 neighborhood;
        // the bucket key is a hash (collisions are harmless — every candidate is distance-verified anyway).
        std::sort(top.begin(),top.end(),[](const Result&a,const Result&b){return a.fin>b.fin;});
        std::vector<Result> kept; int sep=2*e+1;
        auto fdiv=[](int a,int b){ return a>=0 ? a/b : -(((-a)+b-1)/b); };
        auto bkey=[&](int x,int y,int z)->long long{
            return (long long)fdiv(x,sep)*0x9E3779B1LL + (long long)fdiv(y,sep)*0xC2B2AE35LL + (long long)fdiv(z,sep)*0x27D4EB2FLL; };
        std::unordered_map<long long,std::vector<int>> grid; grid.reserve(topk*2);
        for(auto&rr:top){ bool ok=true;
            int bx=fdiv(rr.ox,sep),by=fdiv(rr.oy,sep),bz=fdiv(rr.oz,sep);
            for(int dx=-1;dx<=1&&ok;dx++)for(int dy=-1;dy<=1&&ok;dy++)for(int dz=-1;dz<=1&&ok;dz++){
                auto it=grid.find((long long)(bx+dx)*0x9E3779B1LL+(long long)(by+dy)*0xC2B2AE35LL+(long long)(bz+dz)*0x27D4EB2FLL);
                if(it==grid.end()) continue;
                for(int idx:it->second) if(std::max(std::max(abs(rr.ox-kept[idx].ox),abs(rr.oy-kept[idx].oy)),abs(rr.oz-kept[idx].oz))<=sep){ok=false;break;} }
            if(ok){ grid[bkey(rr.ox,rr.oy,rr.oz)].push_back((int)kept.size()); kept.push_back(rr); }
            if((int)kept.size()>=topk) break; }
        top.swap(kept);
      }
    }
    printf("scanned %d tiles | %lld anchor candidates | %lld survivors (minfrac pre-filter) | top-K=%zu\n",
        nTiles,totAnchor,totSurv,top.size());
    printf("[timing] kGenerate=%.0f ms (kSetup=%.0f kFill=%.0f) | kScore=%.0f ms | gen/score=%.2f\n",msGen,msSetup,msGen-msSetup,msScore,msScore>0?msGen/msScore:0);

    if(!refine){
        printf("\n%4s %22s %12s %7s %8s %8s %9s\n","rank","world_origin","chunk","orient","pres4","absH4","final4");
        for(size_t i=0;i<top.size()&&i<10;i++){ auto&t=top[i]; char o[24],ch[16];
            snprintf(o,sizeof(o),"(%d, %d, %d)",t.ox,t.oy,t.oz); snprintf(ch,sizeof(ch),"(%d, %d)",t.ox>>4,t.oz>>4);
            printf("%4zu %22s %12s    r%dm%d %d/%zu %8d %9.1f\n",i+1,o,ch,t.r,t.mir,t.pres,ore.size(),t.absH,t.fin); }
        if(top.size()>=2){ float m=top[0].fin-top[1].fin;
            printf("\ntop_final4=%.1f margin=%.1f => %s\n",top[0].fin,m,(m>=std::max(3.0,0.3*ore.size()))?"CONFIDENT":"shortlist"); }
        return 0;
    }
    int gravelMax=cnt[4]+cnt[5]+cnt[6];   // max margin gravel/copper/iron can add to any hypothesis
    std::vector<Refined> rf; refineTop(seed,top,nRefine,ore,bare,maxExt,w,gravelMax,rf);
    printf("\n(refined top %d with all 7 families incl. gravel/copper/iron)\n",(int)std::min((size_t)nRefine,top.size()));
    printf("%4s %22s %12s %7s %10s %8s %9s\n","rank","world_origin","chunk","orient","present6","absH6","final6");
    for(size_t i=0;i<rf.size()&&i<10;i++){ auto&t=rf[i]; char o[24],ch[16];
        snprintf(o,sizeof(o),"(%d, %d, %d)",t.r.ox,t.r.oy,t.r.oz); snprintf(ch,sizeof(ch),"(%d, %d)",t.r.ox>>4,t.r.oz>>4);
        printf("%4zu %22s %12s    r%dm%d %d/%zu %8d %9.1f\n",i+1,o,ch,t.r.r,t.r.mir,t.pres6,ore.size(),t.absH6,t.fin6); }
    if(rf.size()>=2){ float m=rf[0].fin6-rf[1].fin6;
        printf("\ntop_final6=%.1f margin=%.1f => %s\n",rf[0].fin6,m,(m>=std::max(3.0,0.3*ore.size()))?"CONFIDENT (unique)":"shortlist"); }
    return 0;
}
