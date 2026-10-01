# P2b — carved-room (3D) matcher + WORLD-scale uniqueness analysis.
# A carved room exposes ores on its inner shell -> full 3D relative typed points (depth known, unlike a
# single wall). Orientation still unknown -> try 8 (4 horizontal rotations x mirror); Y ~ deepslate band.
#
# World-uniqueness reasoning: dense ores (tuff/gravel) are correlated blobs and CANNOT establish
# uniqueness; only sparse ores (~independent) can. The matcher anchors on the rarest family and tries
# every anchor candidate, so #placements ~= 8 * (#anchor candidates in world). A wrong placement is a
# false positive only if the OTHER sparse points also coincide. Hence:
#   E[world false positives] ~= 8 * occ[anchor]*world_cells * PROD over other sparse points of p_hit.
# We report that expectation; <1 => world-unique. We also empirically measure the margin in a test region.
import matcher as M
import collections, math, random

BAND = 64  # deepslate band height used for occupancy/world-cell counting (-64..-1)

def orient(a, b, r, mir):
    a *= mir
    for _ in range(r): a, b = -b, a   # rotate (a,b) by r*90 deg
    return a, b

def make_room(cand, box, noise=0, seed=1):
    x0,x1,y0,y1,z0,z1 = box
    rnd = random.Random(seed)
    def shell(x,y,z):
        inside  = x0<=x<=x1 and y0<=y<=y1 and z0<=z<=z1
        near    = x0-1<=x<=x1+1 and y0-1<=y<=y1+1 and z0-1<=z<=z1+1
        return near and not inside
    obs=[]
    for fam,pts in cand.items():
        for (x,y,z) in pts:
            if shell(x,y,z):
                a,b,c = x-x0, z-z0, y-y0
                if noise: a+=rnd.randint(-noise,noise); b+=rnd.randint(-noise,noise); c+=rnd.randint(-noise,noise)
                obs.append((a,b,c,fam))
    return obs, (x0,y0,z0)

def match3d(cand, obs, tolfn):
    fams={f for *_,f in obs}
    present=[f for f in M.ABUND if f in fams] or list(fams)
    anchor=min(present,key=lambda f: len(cand.get(f,())))
    a0,b0,c0=next((a,b,c) for (a,b,c,f) in obs if f==anchor)
    others=[(a,b,c,f) for (a,b,c,f) in obs if not(a==a0 and b==b0 and c==c0 and f==anchor)]
    by_origin={}
    for r in range(4):
        for mir in (1,-1):
            da0,db0=orient(a0,b0,r,mir)
            for (cx,cy,cz) in cand.get(anchor,()):
                ox,oy,oz = cx-da0, cy-c0, cz-db0
                m=1
                for (a,b,c,f) in others:
                    da,db=orient(a,b,r,mir)
                    if M.hit(cand,f,ox+da,oy+c,oz+db,tolfn(f)): m+=1
                key=(ox,oy,oz,r,mir)
                if m>by_origin.get(key,0): by_origin[key]=m
    return by_origin,anchor

def world_fp(obs, occ, tolfn, anchor):
    world_cells = M.WORLD_CHUNKS*256*BAND
    n_anchor_world = 8 * occ[anchor]*world_cells
    sparse=[f for (a,b,c,f) in obs if f not in M.DENSE]
    if anchor in sparse: sparse.remove(anchor)   # anchor used for alignment, not discrimination
    prod=1.0
    for f in sparse:
        p = min(1.0, len(M.NEI)*occ[f]) if tolfn(f) else occ[f]
        prod*=p
    return n_anchor_world*prod, len(sparse)+1

def run(label, cand, occ, box, noise=0, tol_rare=0, tol_dense=0):
    tolfn=lambda f:(tol_dense if f in M.DENSE else tol_rare)
    obs,truth=make_room(cand,box,noise)
    if len(obs)<2: print(f"[{label}] {len(obs)} obs - skip"); return
    fc=collections.Counter(f for *_,f in obs)
    by_origin,anchor=match3d(cand,obs,tolfn)
    cheb=lambda o:max(abs(o[0]-truth[0]),abs(o[1]-truth[1]),abs(o[2]-truth[2]))
    eval_tol=noise+1
    best=max(by_origin.values())
    near=min((cheb(o) for o,m in by_origin.items() if m==best),default=99)
    far=max((m for o,m in by_origin.items() if cheb(o)>eval_tol),default=0)
    fp,nsparse=world_fp(obs,occ,tolfn,anchor)
    print(f"\n[{label}] box={box} noise={noise} obs={len(obs)} {dict(fc)}")
    print(f"   anchor={anchor} best={best}/{len(obs)}@{near} far={far} MARGIN={best-far} "
          f"sparse_pts={nsparse} | est_world_FP={fp:.2e} -> {'WORLD-UNIQUE' if fp<1 else 'NOT unique'}")

if __name__=="__main__":
    R=12
    print(f"dumping region [-{R+1}..{R+1}]^2 ...")
    cand=M.region_dump(-R-1,R+1,-R-1,R+1)
    occ=M.occupancy(cand,R,-64,-1)
    print("occupancy:",{f:round(p,4) for f,p in occ.items()})
    # rooms centered in chunk (0,0)-ish, deepslate band. (x0,x1,y0,y1,z0,z1)
    run("room-8  noiseless", cand, occ, (4,11,-40,-33,4,11))
    run("room-12 noiseless", cand, occ, (2,13,-44,-37,2,13))
    run("room-20 noiseless", cand, occ, (0,19,-48,-41,0,19))
    run("room-20 noise1",    cand, occ, (0,19,-48,-41,0,19), noise=1, tol_rare=1, tol_dense=0)
    run("room-32 noise1",    cand, occ, (-6,25,-52,-37,-6,25), noise=1, tol_rare=1, tol_dense=0)
