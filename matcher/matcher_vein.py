# P2c — vein-level uniqueness. The independent discriminator is the sparse VEIN, not the block
# (a wrong placement that lands in a vein matches all its blocks at once). So we:
#   1. cluster sparse ore candidates into veins (connected components),
#   2. match on vein centroids (anchor-align one vein, verify the rest land near a same-family vein),
#   3. sweep region size to extrapolate the false-positive margin toward world scale.
# Dense tuff/gravel are excluded entirely — they don't establish uniqueness. See NOTES.md.
import matcher as M
import matcher_room as R3
import collections, math

SPARSE = ["lapis","redstone","copper","granite"]
GAP = 2          # blocks within this Chebyshev distance (same family) belong to one vein
VTOL = 4         # a predicted vein centroid matches if a same-family vein centroid is within this

def cluster(points, gap=GAP):
    """Union-find clustering of (x,y,z) within Chebyshev<=gap. Returns list of centroids (int)."""
    pts = list(points); idx = {p:i for i,p in enumerate(pts)}; parent = list(range(len(pts)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def uni(a,b): parent[find(a)]=find(b)
    S=set(pts)
    offs=[(dx,dy,dz) for dx in range(-gap,gap+1) for dy in range(-gap,gap+1) for dz in range(-gap,gap+1)]
    for (x,y,z) in pts:
        for dx,dy,dz in offs:
            q=(x+dx,y+dy,z+dz)
            if q in S and q!=(x,y,z): uni(idx[(x,y,z)],idx[q])
    groups=collections.defaultdict(list)
    for p in pts: groups[find(idx[p])].append(p)
    cents=[]
    for g in groups.values():
        n=len(g); cents.append((round(sum(p[0] for p in g)/n), round(sum(p[1] for p in g)/n),
                                round(sum(p[2] for p in g)/n)))
    return cents

class VeinIndex:
    """Spatial hash of vein centroids per family for near-queries."""
    def __init__(self, fam_cents, cell=VTOL):
        self.cell=cell; self.grid={f:collections.defaultdict(list) for f in fam_cents}
        self.fams=fam_cents
        for f,cs in fam_cents.items():
            for c in cs: self.grid[f][(c[0]//cell,c[1]//cell,c[2]//cell)].append(c)
    def near(self, f, p, tol=VTOL):
        if f not in self.grid: return False
        cx,cy,cz=p[0]//self.cell,p[1]//self.cell,p[2]//self.cell
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                for dz in(-1,0,1):
                    for c in self.grid[f].get((cx+dx,cy+dy,cz+dz),()):
                        if abs(c[0]-p[0])<=tol and abs(c[1]-p[1])<=tol and abs(c[2]-p[2])<=tol: return True
        return False

def region_veins(cxlo,cxhi,czlo,czhi):
    cand=M.region_dump(cxlo,cxhi,czlo,czhi)        # all families; we use sparse only
    fam_cents={f:cluster(cand[f]) for f in SPARSE if f in cand}
    return fam_cents

def match_veins(idx, obs_veins, noise=0):
    """obs_veins = list of (a,b,c,fam) centroids (relative 3D). Returns (best, far) given idx truth region."""
    tol=VTOL+noise
    fams={f for *_,f in obs_veins}
    present=[f for f in SPARSE if f in fams] or list(fams)
    anchor=min(present,key=lambda f: len(idx.fams.get(f,())))
    a0,b0,c0=next((a,b,c) for (a,b,c,f) in obs_veins if f==anchor)
    others=[(a,b,c,f) for (a,b,c,f) in obs_veins if not(a==a0 and b==b0 and c==c0 and f==anchor)]
    by_origin={}
    for r in range(4):
        for mir in (1,-1):
            da0,db0=R3.orient(a0,b0,r,mir)
            for (cx,cy,cz) in idx.fams.get(anchor,()):
                ox,oy,oz=cx-da0,cy-c0,cz-db0
                m=1
                for (a,b,c,f) in others:
                    da,db=R3.orient(a,b,r,mir)
                    if idx.near(f,(ox+da,oy+c,oz+db),tol): m+=1
                key=(ox,oy,oz,r,mir)
                if m>by_origin.get(key,0): by_origin[key]=m
    return by_origin,anchor

if __name__=="__main__":
    # Build a fixed observed pattern from a room at the true location, then test against growing regions.
    print("dumping truth-local candidates for the room ...")
    local=M.region_dump(-3,3,-3,3)
    # large room to expose several sparse veins
    box=(-6,25,-52,-37,-6,25)
    obs_blocks,truth=R3.make_room(local,box,noise=0)
    obs_sparse=[(a,b,c,f) for (a,b,c,f) in obs_blocks if f in SPARSE]
    # cluster observed sparse blocks into veins (relative coords)
    by_fam=collections.defaultdict(list)
    for (a,b,c,f) in obs_sparse: by_fam[f].append((a,b,c))
    obs_veins=[]
    for f,pts in by_fam.items():
        for cen in cluster(pts): obs_veins.append((cen[0],cen[1],cen[2],f))
    vc=collections.Counter(f for *_,f in obs_veins)
    print(f"room exposes sparse blocks={len(obs_sparse)} -> sparse VEINS={len(obs_veins)} {dict(vc)}")

    true_origin=(truth[0],truth[1],truth[2])
    print(f"\n{'R':>3} {'chunks':>8} {'truthVeins':>11} {'far':>4} {'margin':>7}")
    for Rr in (3,6,9,13,18):
        fam_cents=region_veins(-Rr-1,Rr+1,-Rr-1,Rr+1)
        idx=VeinIndex(fam_cents)
        by_origin,anchor=match_veins(idx,obs_veins,noise=0)
        cheb=lambda o:max(abs(o[0]-true_origin[0]),abs(o[1]-true_origin[1]),abs(o[2]-true_origin[2]))
        best=max(by_origin.values())
        tnear=max((m for o,m in by_origin.items() if cheb(o)<=VTOL),default=0)
        far=max((m for o,m in by_origin.items() if cheb(o)>VTOL+2),default=0)
        print(f"{Rr:>3} {(2*Rr+1)**2:>8} {tnear:>11} {far:>4} {tnear-far:>7}")
    print(f"\n(total observed sparse veins = {len(obs_veins)}; "
          f"far should stay well below truth as region grows for world-uniqueness)")
