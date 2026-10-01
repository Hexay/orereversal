# solve.py — consume an ore observation (see docs/observation-format.md) and return ranked candidate world
# locations for a known seed. Two-stage, recall-safe:
#   stage 1 (presence): anchor-enumeration (every region candidate of the rarest observed family is a
#     hypothesis -> true location never dropped) + full-score ore cells vs dilated candidate sets. The true
#     location always has MAX presence (real ores subset candidates), so it survives to stage 2.
#   stage 2 (soft absence): on the top presence survivors, penalize ore predicted on 'bare' cells.
#     final = presence - w * absence_hits. Crushes false positives, esp. via the dense families.
# Bounded region here (CPU). World-scale = GPU phase (same logic).
import candidates as C
import argparse, csv, collections

def orient_xz(x,z,r,mir):
    x*=mir
    for _ in range(r): x,z=-z,x
    return x,z

def dilate(pts,rad):
    if rad==0: return set(pts)
    out=set(); R=range(-rad,rad+1)
    for (X,Y,Z) in pts:
        for dx in R:
            for dy in R:
                for dz in R: out.add((X+dx,Y+dy,Z+dz))
    return out

def load_obs(path):
    ore=[]; bare=[]
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip() in ("family","") or row[0].startswith("#"): continue
            fam=row[0].strip(); p=(int(row[1]),int(row[2]),int(row[3]))
            (bare if fam=="bare" else ore).append((p[0],p[1],p[2],fam))
    return ore, bare

def solve(cand, ore, bare, e, w=1.0, topk=300, topn=8):
    D=2*e
    dil={f:dilate(s,D) for f,s in cand.items() if any(o[3]==f for o in ore) or f in C.USABLE}
    combined=set().union(*[dil[f] for f in dil]) if dil else set()   # "any usable ore here" (for absence)
    fams=[f for f in {o[3] for o in ore} if cand.get(f)]
    if not fams: return [], None, 0
    anchor=min(fams, key=lambda f: len(cand[f]))
    ax,ay,az=next((x,y,z) for (x,y,z,f) in ore if f==anchor)
    def presence(r,mir,ox,oy,oz):
        s=0
        for (x,y,z,f) in ore:
            dx,dz=orient_xz(x,z,r,mir)
            if (ox+dx,oy+y,oz+dz) in dil.get(f,()): s+=1
        return s
    def absence(r,mir,ox,oy,oz):
        s=0
        for (x,y,z,f) in bare:
            dx,dz=orient_xz(x,z,r,mir)
            if (ox+dx,oy+y,oz+dz) in combined: s+=1     # predicted an ore where we saw bare
        return s
    # stage 1: presence over all hypotheses
    bylocs={}
    nhyp=0
    for r in range(4):
        for mir in (1,-1):
            adx,adz=orient_xz(ax,az,r,mir)
            for (cx,cy,cz) in cand[anchor]:
                nhyp+=1
                ox,oy,oz=cx-adx,cy-ay,cz-adz
                p=presence(r,mir,ox,oy,oz)
                k=(ox,oy,oz)
                if p>bylocs.get(k,(-1,))[0]: bylocs[k]=(p,(r,mir))
    survivors=sorted(bylocs.items(), key=lambda kv:-kv[1][0])[:topk]
    # stage 2: soft absence on survivors
    scored=[]
    for (loc,(p,(r,mir))) in survivors:
        ah=absence(r,mir,*loc) if bare else 0
        scored.append((loc,p,ah,p-w*ah,(r,mir)))
    scored.sort(key=lambda t:-t[3])
    out=[]
    for (loc,p,ah,fin,om) in scored:
        if all(max(abs(loc[0]-d[0][0]),abs(loc[1]-d[0][1]),abs(loc[2]-d[0][2]))>2*e+1 for d in out):
            out.append((loc,p,ah,fin,om))
        if len(out)>=topn: break
    return out, anchor, nhyp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("observation")
    ap.add_argument("--seed", default="123"); ap.add_argument("--version", default="1.18")
    ap.add_argument("--region", type=int, default=13)
    ap.add_argument("--error", type=int, default=0)
    ap.add_argument("--absence-weight", type=float, default=1.0)
    a=ap.parse_args()
    ore,bare=load_obs(a.observation)
    R=a.region
    cand=C.region_dump(a.seed,a.version,-R-1,R+1,-R-1,R+1)
    res,anchor,nhyp=solve(cand, ore, bare, a.error, a.absence_weight)
    print(f"obs: {len(ore)} ore + {len(bare)} bare | search {(2*R+1)**2} ch | anchor={anchor} hyps={nhyp} err=+-{a.error} w={a.absence_weight}")
    if not res: print("no candidates."); return
    print(f"\n{'rank':>4} {'world_origin':>20} {'chunk':>11} {'orient':>7} {'present':>8} {'absHits':>8} {'final':>9}")
    for i,(loc,p,ah,fin,(r,mir)) in enumerate(res):
        print(f"{i+1:>4} {str(loc):>20} {str((loc[0]>>4,loc[2]>>4)):>11} {f'r{r}m{mir}':>7} {p}/{len(ore)} {ah:>8} {fin:>9.1f}")
    top=res[0][3]; second=res[1][3] if len(res)>1 else 0
    print(f"\ntop_final={top:.1f}  margin_to_next={top-second:.1f}  "
          f"=> {'CONFIDENT (unique)' if top-second>=max(3,0.3*len(ore)) else 'shortlist'}")

if __name__=="__main__":
    main()
