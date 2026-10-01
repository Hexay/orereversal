// matcher_refine.cuh — host side of the matcher: CSV observation loader + the PASS-2 CPU refine that
// re-scores the GPU top-K with all 7 families (gravel/copper/iron injected from region_dump.exe).
// See matcher_common.h for the name glossary.
#ifndef MATCHER_REFINE_CUH
#define MATCHER_REFINE_CUH
#include <cstring>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "matcher_common.h"

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
// exact pack for the whole world (|x|,|z| < 2^25 covers ±30M; y+64 < 2^7). Not `long`: 32-bit on MSVC.
static inline uint64_t key3(int x,int y,int z){
    return ((uint64_t)(uint32_t)(x+(1<<25))<<33) | ((uint64_t)(uint32_t)(y+64)<<26) | (uint32_t)(z+(1<<25)); }
struct Refined { Result r; int pres; int absH; float fin; };

// host mirror of occHitTol: any candidate within Chebyshev distance e
static bool hitTol(const std::unordered_set<uint64_t>&s,int x,int y,int z,int e){
    for(int dx=-e;dx<=e;dx++)for(int dy=-e;dy<=e;dy++)for(int dz=-e;dz<=e;dz++)
        if(s.count(key3(x+dx,y+dy,z+dz))) return true;
    return false;
}

static void refineTop(uint64_t seed,std::vector<Result>&top,int nRefine,
        const std::vector<ObsCell>&ore,const std::vector<ObsCell>&bare,
        int maxExt,int e,int ae,float w,int gravelMax,std::vector<Refined>&outv){
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
        std::unordered_set<uint64_t> occ[NACTIVE];   // 0-3 GPU, 4 gravel, 5 copper, 6 iron
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
        // gravel + copper + iron from region_dump (one call, all three families)
        { char cmd[700]; snprintf(cmd,sizeof(cmd),"\"%s\" %llu 1.18 %d %d %d %d -64 -1 gravel copper iron 2>" DEVNULL,
                g_rdexe,(unsigned long long)seed,cx0,cx1,cz0,cz1);
            FILE*p=popen(cmd,"r"); if(p){ char ln[128];
                while(fgets(ln,sizeof(ln),p)){ char f[32]; int x,y,z;
                    if(sscanf(ln,"%31[^,],%d,%d,%d",f,&x,&y,&z)!=4||y<-64||y>=0) continue;
                    int fa = !strcmp(f,"gravel")?4 : !strcmp(f,"copper")?5 : !strcmp(f,"iron")?6 : -1;
                    if(fa>=0) occ[fa].insert(key3(x,y,z)); }
                pclose(p); } }
        // score this hypothesis with all 7 families
        int pres=0; for(auto&o:ore){ int dx,dz; orient_xz(o.x,o.z,R.r,R.mir,&dx,&dz);
            if(hitTol(occ[o.fam],R.ox+dx,R.oy+o.y,R.oz+dz,e)) pres++; }
        int absH=0; for(auto&b:bare){ int dx,dz; orient_xz(b.x,b.z,R.r,R.mir,&dx,&dz);
            bool any=false;
            for(int fa=0;fa<NACTIVE;fa++) if(hitTol(occ[fa],R.ox+dx,R.oy+b.y,R.oz+dz,ae)){any=true;break;} if(any) absH++; }
        outv[t]={R,pres,absH,pres-w*absH};
    }
    std::sort(outv.begin(),outv.end(),[](const Refined&a,const Refined&b){return a.fin>b.fin;});
}
#endif
