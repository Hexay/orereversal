// matcher_kernels.cuh — the device kernels of the GPU pass: bit-exact ore generation into a
// per-family occupancy bitmask (legacy kGenerate, or the two-kernel kSetup+kFill), Morton keying of
// anchor candidates, and the presence/absence scorer kScore. See matcher_common.h for the glossary.
#ifndef MATCHER_KERNELS_CUH
#define MATCHER_KERNELS_CUH
#include "matcher_common.h"

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
#endif
