# P2d — precision budget. Noise-robust Hough-voting matcher + sweep of reading-error vs exposed-ore.
# Voting: every sparse observed point votes for an (orientation, translation) hypothesis (consensus -> no
# single-anchor bias). Coarse bins of size 2e+1 absorb reading noise. Top hypotheses are then SCORED with
# the FULL ore set (all families) at tolerance e. Output: truth vs best-wrong margin as f(reading_error,
# amount of exposed ore) -> tells us how precise transcription must be for world-scale uniqueness.
import matcher as M, matcher_room as R3
import collections

SPARSE = {"lapis","copper","granite","redstone"}

def vote_and_score(cand, obs, e, topk=400):
    b = 2*e+1
    acc = collections.Counter()
    sparse_obs = [(a,bb,c,f) for (a,bb,c,f) in obs if f in SPARSE]
    for r in range(4):
        for mir in (1,-1):
            for (a,bb,c,f) in sparse_obs:
                da,db = R3.orient(a,bb,r,mir)
                for (cx,cy,cz) in cand.get(f,()):
                    tx,ty,tz = cx-da, cy-c, cz-db
                    acc[(r,mir,tx//b,ty//b,tz//b)] += 1
    def score(r,mir,ox,oy,oz):
        s=0
        for (a,bb,c,f) in obs:
            da,db = R3.orient(a,bb,r,mir)
            if M.hit(cand,f,ox+da,oy+c,oz+db,e): s+=1
        return s
    out=[]
    for (r,mir,bx,by,bz),_ in acc.most_common(topk):
        best=0; bestT=None
        for dx in range(b):
            for dy in range(b):
                for dz in range(b):
                    T=(bx*b+dx, by*b+dy, bz*b+dz)
                    sc=score(r,mir,*T)
                    if sc>best: best=sc; bestT=(r,mir,T)
        out.append((best,bestT))
    return out

def analyze(label, cand, box, e):
    obs,truth = R3.make_room(cand, box, noise=e)
    if len(obs)<3: print(f"[{label}] e={e} {len(obs)} obs - skip"); return
    res = vote_and_score(cand, obs, e)
    cheb = lambda T: max(abs(T[0]-truth[0]),abs(T[1]-truth[1]),abs(T[2]-truth[2]))
    truth_score = max((s for s,bt in res if bt and bt[0]==0 and bt[1]==1 and cheb(bt[2])<=e), default=0)
    far_score   = max((s for s,bt in res if bt and cheb(bt[2])>e+4), default=0)
    sp = sum(1 for *_,f in obs if f in SPARSE)
    print(f"[{label}] e={e:>1} N={len(obs):>4} sparse={sp:>3} truth={truth_score:>4} "
          f"best_wrong={far_score:>4} margin={truth_score-far_score:>4} "
          f"{'UNIQUE-in-region' if truth_score>far_score else 'COLLISION'}")

def score_fixed(cand, obs, truth, e):
    res = vote_and_score(cand, obs, e)
    cheb = lambda T: max(abs(T[0]-truth[0]),abs(T[1]-truth[1]),abs(T[2]-truth[2]))
    ts = max((s for s,bt in res if bt and bt[0]==0 and bt[1]==1 and cheb(bt[2])<=e), default=0)
    fw = max((s for s,bt in res if bt and cheb(bt[2])>e+4), default=0)
    return ts, fw

if __name__=="__main__":
    box = (-6,25,-52,-37,-6,25)   # big room
    local = M.region_dump(-3,3,-3,3)
    print("WORLD-SCALE EXTRAPOLATION (big room). Does best_wrong climb toward truth as region grows?")
    for e in (0,1,2):
        obs,truth = R3.make_room(local, box, noise=e)
        N = len(obs)
        print(f"\n e={e}  N={N}  (truth should stay ~{N}, best_wrong should plateau well below)")
        print(f"   {'Rchunks':>8} {'truth':>6} {'best_wrong':>11} {'margin':>7}")
        for R in (6,13,20,28):
            cand = M.region_dump(-R-1,R+1,-R-1,R+1)
            ts,fw = score_fixed(cand, obs, truth, e)
            print(f"   {(2*R+1)**2:>8} {ts:>6} {fw:>11} {ts-fw:>7}")
