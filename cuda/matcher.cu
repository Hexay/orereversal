// matcher.cu — world-scale GPU ore-pattern localizer (P3.3). Tiles an arbitrary chunk region so memory
// is bounded regardless of search size (300k x 300k and beyond). Two passes:
//   PASS 1 (GPU, tiled): per tile, generate the 4 bit-exact families (tuff/redstone/lapis/granite) into a
//     reused occupancy bitmask, enumerate the anchor family's candidates in the tile interior, and score
//     (presence + soft absence) with an aggressive presence pre-filter + early-termination. Survivors
//     (compacted) merge into a global top-K.
//   PASS 2 (CPU refine): re-score the top-K with all 7 families (gravel/copper/iron from region_dump.exe)
//     so their margin is recovered without per-tile injection. iron is not bit-exact (ore-vein noise).
// Validated vs solve.py + the single-region matcher on the real test world (cuda/README.md, docs/research-log.md P3).
//
// Split across the TU (nvcc compiles this file only; the rest are headers it includes):
//   matcher_common.h    — family maps, occ probes, Result/ObsCell, orient_xz, and the name glossary.
//   matcher_kernels.cuh — the device kernels (kGenerate / kSetup+kFill / kAnchorKey / kScore).
//   matcher_refine.cuh  — loadObs + the PASS-2 CPU refine.
//
// Usage: matcher <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv>
//        [--error E] [--abs-error A] [--absw W] [--minfrac F] [--tile T] [--topk K] [--refine N]
//        [--no-refine] [--legacy-gen]
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
#include "matcher_common.h"
#include "matcher_kernels.cuh"
#include "matcher_refine.cuh"

int main(int argc,char**argv){
    const char* usage="usage: %s <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv> [--error E] [--abs-error A] [--absw W] [--minfrac F] [--tile T] [--topk K] [--refine N] [--no-refine] [--legacy-gen]\n";
    if(argc<7){ fprintf(stderr,usage,argv[0]); return 2; }
    uint64_t seed=(uint64_t)strtoll(argv[1],NULL,10);
    // resolve region_dump relative to this binary's dir (../harness/), robust to CWD
    { const char*a=argv[0]; int cut=-1; for(int i=0;a[i];i++) if(a[i]=='/'||a[i]=='\\') cut=i;
      if(cut>=0) snprintf(g_rdexe,sizeof(g_rdexe),"%.*s" PATHSEP RD_RELPATH,cut,a); }
    int cxMin=atoi(argv[2]),cxMax=atoi(argv[3]),czMin=atoi(argv[4]),czMax=atoi(argv[5]);
    const char* obspath=argv[6];
    // ae: absence tolerance, separate from e — see docs/research-log.md "P7"
    int e=0,ae=0,tile=256,topk=4096,nRefine=64; float w=1.0f,minFrac=0.5f; bool refine=true; bool useCoop=true; bool gateOff=false;
    for(int i=7;i<argc;i++){
        if(!strcmp(argv[i],"--legacy-gen")) useCoop=false; else if(!strcmp(argv[i],"--coop")) useCoop=true; else
        if(!strcmp(argv[i],"--error")&&i+1<argc) e=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--abs-error")&&i+1<argc) ae=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--absw")&&i+1<argc) w=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--minfrac")&&i+1<argc) minFrac=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--tile")&&i+1<argc) tile=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--topk")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--refine")&&i+1<argc) nRefine=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--no-refine")) refine=false;
        else if(!strcmp(argv[i],"--no-gate")) gateOff=true;
        else { fprintf(stderr,"unknown or incomplete option: %s\n",argv[i]); fprintf(stderr,usage,argv[0]); return 2; }
    }
    if(tile>320){ fprintf(stderr,"warning: tile>320 risks a Windows TDR kill (kScore launch >2s); clamping to 256\n"); tile=256; }
    if(refine){ FILE*t=fopen(g_rdexe,"rb");   // refine would otherwise silently score without gravel/copper/iron
        if(!t){ fprintf(stderr,"region_dump not found at %s — run harness/build.sh, or pass --no-refine\n",g_rdexe); return 1; }
        fclose(t); }
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
    printf("obs: %zu ore + %zu bare | search %ld chunks | anchor=%s(%d) tile=%d margin=%dch minfrac=%.2f e=%d ae=%d w=%.1f\n",
        ore.size(),bare.size(),totCh,FAMNAME[anchorFamily],best,tile,marginCh,minFrac,e,ae,w);

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
      bool seen[NGPU]={false}; const char* sep="";
      for(int s:selCfg){ int fa=famActive(ORE_CFGS_118[s].family);
          if(!seen[fa]){ printf("%s%s", sep, FAMNAME[ORE_CFGS_118[s].family]); sep=","; seen[fa]=true; } }
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
            oax,oay,oaz,dOcc,wpf,X0,Z0,DX,DZ,e,ae,w,minPres,dOut,dOcnt,outCap);
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
        printf("\n(GPU families only: tuff/redstone/lapis/granite)\n");
        printf("%4s %22s %12s %7s %10s %8s %9s\n","rank","world_origin","chunk","orient","present","absH","final");
        for(size_t i=0;i<top.size()&&i<10;i++){ auto&t=top[i]; char o[24],ch[16];
            snprintf(o,sizeof(o),"(%d, %d, %d)",t.ox,t.oy,t.oz); snprintf(ch,sizeof(ch),"(%d, %d)",t.ox>>4,t.oz>>4);
            printf("%4zu %22s %12s    r%dm%d %d/%d %8d %9.1f\n",i+1,o,ch,t.r,t.mir,t.pres,nOreGpu,t.absH,t.fin); }
        if(top.size()>=2){ float m=top[0].fin-top[1].fin;
            printf("\ntop_final=%.1f margin=%.1f => %s\n",top[0].fin,m,(m>=std::max(3.0,0.3*nOreGpu))?"CONFIDENT":"shortlist"); }
        return 0;
    }
    int gravelMax=cnt[4]+cnt[5]+cnt[6];   // max margin gravel/copper/iron can add to any hypothesis
    std::vector<Refined> rf; refineTop(seed,top,nRefine,ore,bare,maxExt,e,ae,w,gravelMax,rf);
    printf("\n(refined top %d with all 7 families incl. gravel/copper/iron)\n",(int)std::min((size_t)nRefine,top.size()));
    printf("%4s %22s %12s %7s %10s %8s %9s\n","rank","world_origin","chunk","orient","present","absH","final");
    for(size_t i=0;i<rf.size()&&i<10;i++){ auto&t=rf[i]; char o[24],ch[16];
        snprintf(o,sizeof(o),"(%d, %d, %d)",t.r.ox,t.r.oy,t.r.oz); snprintf(ch,sizeof(ch),"(%d, %d)",t.r.ox>>4,t.r.oz>>4);
        printf("%4zu %22s %12s    r%dm%d %d/%zu %8d %9.1f\n",i+1,o,ch,t.r.r,t.r.mir,t.pres,ore.size(),t.absH,t.fin); }
    if(rf.size()>=2){ float m=rf[0].fin-rf[1].fin;
        printf("\ntop_final=%.1f margin=%.1f => %s\n",rf[0].fin,m,(m>=std::max(3.0,0.3*ore.size()))?"CONFIDENT (unique)":"shortlist"); }
    return 0;
}
