# P2 — synthetic round-trip matcher (CPU reference).
# Pick a SECRET chunk + carved wall -> expose its Tier-1 ores onto a 2D plane (drop depth, forget
# absolute pos & orientation, optionally add reading noise) -> hand only that typed point-set to the
# matcher -> confirm it recovers the secret location, AND quantify the margin so we can reason about
# WORLD-scale uniqueness, not just the small test region. See docs/research-log.md.
import collections, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # python/ for candidates
import candidates

SEED, VERSION = "123", "1.18"
ABUND = ["lapis","copper","granite","redstone","gravel","tuff"]   # sparsest first
DENSE = {"tuff","gravel"}
WORLD_CHUNKS = 3_750_000**2   # ~1.4e13

def region_dump(cxlo,cxhi,czlo,czhi,ylo=-64,yhi=-1):
    return candidates.region_dump(SEED,VERSION,cxlo,cxhi,czlo,czhi,ylo,yhi)

def occupancy(cand, R, ylo, yhi):
    cells = ((2*R+3)*16)**2 * (yhi-ylo+1)          # region volume (with margin) in blocks
    return {f: len(s)/cells for f,s in cand.items()}

# ---- synthetic wall ----
def make_wall(cand, axis, depth, ylo, yhi, blo, bhi, thickness=1, noise=0, only=None, rng_seed=1):
    import random; rnd = random.Random(rng_seed)
    obs = []
    for fam, pts in cand.items():
        if only and fam not in only: continue
        for (x,y,z) in pts:
            a = x if axis=='x' else z
            b = z if axis=='x' else x
            if depth<=a<depth+thickness and ylo<=y<=yhi and blo<=b<=bhi:
                h,v = b-blo, y-ylo
                if noise: h+=rnd.randint(-noise,noise); v+=rnd.randint(-noise,noise)
                obs.append((h,v,fam))
    return obs

# ---- matcher ----
NEI = [(dx,dy,dz) for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1)]
def hit(cand, fam, x, y, z, tol):
    s = cand.get(fam)
    if not s: return False
    if tol==0: return (x,y,z) in s
    return any((x+dx,y+dy,z+dz) in s for dx,dy,dz in NEI)
ORIENTS = [('x',1),('x',-1),('z',1),('z',-1)]

def match(cand, obs, tolfn):
    """tolfn(fam)->tol. Returns list of (score, origin_world, axis, sgn) over all placements."""
    fams = {f for _,_,f in obs}
    present = [f for f in ABUND if f in fams] or list(fams)
    anchor = min(present, key=lambda f: len(cand.get(f,())))
    h0,v0 = next((h,v) for (h,v,f) in obs if f==anchor)
    others = [(h,v,f) for (h,v,f) in obs if not (h==h0 and v==v0 and f==anchor)]
    out = []
    for axis,sgn in ORIENTS:
        for (cx,cy,cz) in cand.get(anchor,()):
            offA = cx if axis=='x' else cz
            offB = (cz if axis=='x' else cx) - sgn*h0
            offY = cy - v0
            m = 1
            for (h,v,f) in others:
                if axis=='x': wx,wy,wz = offA, v+offY, sgn*h+offB
                else:         wx,wy,wz = sgn*h+offB, v+offY, offA
                if hit(cand,f,wx,wy,wz,tolfn(f)): m += 1
            origin = (offA,offY,offB) if axis=='x' else (offB,offY,offA)
            out.append((m, origin, axis, sgn))
    return out, anchor

def run(label, cand, occ, axis, depth, ylo, yhi, blo, bhi, thickness=1, noise=0,
        tol_rare=0, tol_dense=0, only=None):
    obs = make_wall(cand, axis, depth, ylo, yhi, blo, bhi, thickness, noise, only)
    fc = collections.Counter(f for _,_,f in obs)
    n = len(obs)
    if n < 2: print(f"\n[{label}] only {n} obs — skip"); return
    tolfn = lambda f: (tol_dense if f in DENSE else tol_rare)
    res, anchor = match(cand, obs, tolfn)
    true_origin = (depth, ylo, blo) if axis=='x' else (blo, ylo, depth)
    by_origin = {}
    for m,o,ax,sg in res: by_origin[o] = max(by_origin.get(o,0), m)
    cheb = lambda a,b: max(abs(a[0]-b[0]),abs(a[1]-b[1]),abs(a[2]-b[2]))
    eval_tol = noise + (thickness-1) + 1     # how far the recovered origin may drift from truth
    global_best = max(by_origin.values())
    best_origins = [o for o,m in by_origin.items() if m==global_best]
    near = min((cheb(o,true_origin) for o in best_origins), default=99)
    # the true competitor: best score among origins NOT near truth
    far_best = max((m for o,m in by_origin.items() if cheb(o,true_origin)>eval_tol), default=0)
    margin = global_best - far_best
    recovered = (near<=eval_tol) and (margin>0)
    # honest world-FP proxy: extrapolate the empirical far-competitor margin, NOT an independence model.
    # rough sparse-ore discriminator count (points whose family occupancy is low)
    sparse_pts = sum(1 for _,_,f in obs if f not in DENSE)
    print(f"\n[{label}] axis={axis} obs={n} {dict(fc)} anchor={anchor} tol(rare={tol_rare},dense={tol_dense}) eval_tol={eval_tol}")
    print(f"   best={global_best}/{n} @dist {near} from truth | far_competitor={far_best} | MARGIN={margin} "
          f"| sparse_pts={sparse_pts} | RECOVERED={recovered}")

if __name__ == "__main__":
    R = 12
    print(f"dumping region [-{R+1}..{R+1}]^2 ...", file=sys.stderr)
    cand = region_dump(-R-1,R+1,-R-1,R+1)
    occ = occupancy(cand,R,-64,-1)
    print("region candidates:", {f:len(s) for f,s in cand.items()}, file=sys.stderr)
    print("occupancy:", {f:round(p,4) for f,p in occ.items()}, file=sys.stderr)

    run("full-noiseless",  cand, occ, 'x', 8, -60, -1, 0, 15)
    run("full-noise1-blanket-tol1", cand, occ, 'x', 8, -60, -1, 0, 15, noise=1, tol_rare=1, tol_dense=1)
    run("full-noise1-smart(rare tol1, dense tol0)", cand, occ, 'x', 8, -60, -1, 0, 15, noise=1, tol_rare=1, tol_dense=0)
    run("half-noiseless",  cand, occ, 'x', 8, -60, -31, 0, 15)
    run("rareonly-noiseless", cand, occ, 'x', 8, -64, -1, 0, 15, only={"lapis","copper","granite","redstone"})
    run("rareonly-noise1", cand, occ, 'x', 8, -64, -1, 0, 15, noise=1, tol_rare=1, only={"lapis","copper","granite","redstone"})
