# P2e — recall-safe matcher (correctness demo). Anchor-ENUMERATION (every region candidate of the rarest
# observed family is a hypothesis -> the true anchor is always enumerated, so the true hypothesis can never
# be dropped), full-scored against DILATED candidate sets for O(1) tolerance matching. Dilation radius 2e
# absorbs single-anchor + per-point reading noise in one membership test.
# Python handles exact/±1 on a bounded region; ±2 over world-size regions is compute-bound -> that's P3 (GPU).
import matcher as M, matcher_room as R3
import collections, sys

SPARSE = {"lapis","copper","granite","redstone"}

def dilate(pts, rad):
    if rad==0: return set(pts)
    out=set(); rng=range(-rad,rad+1)
    for (x,y,z) in pts:
        for dx in rng:
            for dy in rng:
                for dz in rng: out.add((x+dx,y+dy,z+dz))
    return out

def recover(cand, obs, truth, e):
    D=2*e
    dil={f:dilate(s,D) for f,s in cand.items()}
    # anchor = rarest observed family
    fams={f for *_,f in obs}
    anchor=min((f for f in SPARSE if f in fams), key=lambda f: len(cand.get(f,())), default=None)
    a0,b0,c0=next((a,b,c) for (a,b,c,f) in obs if f==anchor)
    def full_score(r,mir,ox,oy,oz):
        s=0
        for (a,b,c,f) in obs:
            da,db=R3.orient(a,b,r,mir)
            if (ox+da,oy+c,oz+db) in dil.get(f,()): s+=1
        return s
    best=(-1,None); nhyp=0
    for r in range(4):
        for mir in (1,-1):
            da0,db0=R3.orient(a0,b0,r,mir)
            for (cx,cy,cz) in cand.get(anchor,()):
                nhyp+=1
                sc=full_score(r,mir,cx-da0,cy-c0,cz-db0)
                if sc>best[0]: best=(sc,(r,mir,(cx-da0,cy-c0,cz-db0)))
    cheb=lambda T:max(abs(T[0]-truth[0]),abs(T[1]-truth[1]),abs(T[2]-truth[2]))
    bt=best[1]
    ok = bt and bt[0]==0 and bt[1]==1 and cheb(bt[2])<=D
    return best[0], len(obs), anchor, nhyp, ok, (cheb(bt[2]) if bt else -1)

def run(R, e, box):
    cand=M.region_dump(-R-1,R+1,-R-1,R+1)
    obs,truth=R3.make_room(cand,box,noise=e)
    sc,n,anchor,nhyp,ok,dist=recover(cand,obs,truth,e)
    print(f"R={R} ({(2*R+1)**2} ch) e={e}: anchor={anchor} hyps={nhyp} N={n} "
          f"best={sc}/{n} dist_to_truth={dist} TRUTH_RECOVERED={ok}", flush=True)

if __name__=="__main__":
    box=(-6,25,-52,-37,-6,25)
    run(13, 0, box)   # exact, larger region
    run(5,  1, box)   # +-1 reading error, smaller region (dilation memory)
