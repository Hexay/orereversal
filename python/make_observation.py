# Generate an example ore observation from a KNOWN location (tests solve.py; example of the format your
# extraction mod should emit). Models a carved room: every exposed inner-shell cell is labeled with its
# usable-ore family, or `bare` if it's plain stone/deepslate (enables soft-absence scoring).
import candidates as C
import argparse, collections, os

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", default="123")
    ap.add_argument("--version", default="1.18")
    ap.add_argument("--box", default="-6,25,-52,-37,-6,25", help="x0,x1,y0,y1,z0,z1 carved air box")
    ap.add_argument("--noise", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.ROOT,"examples","obs_big_room.csv"))
    a=ap.parse_args()
    x0,x1,y0,y1,z0,z1=(int(v) for v in a.box.split(","))
    local=C.region_dump(a.seed,a.version,(x0>>4)-2,(x1>>4)+2,(z0>>4)-2,(z1>>4)+2)
    fam_at=C.family_at(local)
    import random; rnd=random.Random(1)
    def inside(x,y,z): return x0<=x<=x1 and y0<=y<=y1 and z0<=z<=z1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fc=collections.Counter()
    with open(a.out,"w",newline="\n") as f:
        f.write("family,x,y,z\n")
        for X in range(x0-1,x1+2):
            for Y in range(y0-1,y1+2):
                for Z in range(z0-1,z1+2):
                    if inside(X,Y,Z): continue          # carved air interior - not an exposed face
                    fam=fam_at.get((X,Y,Z),"bare")      # shell cell: ore family or bare
                    rx,ry,rz=X-x0,Y-y0,Z-z0
                    if a.noise and fam!="bare":
                        rx+=rnd.randint(-a.noise,a.noise); ry+=rnd.randint(-a.noise,a.noise); rz+=rnd.randint(-a.noise,a.noise)
                    f.write(f"{fam},{rx},{ry},{rz}\n"); fc[fam]+=1
    ores=sum(v for k,v in fc.items() if k!="bare")
    sparse=sum(v for k,v in fc.items() if k in {"lapis","redstone","copper","granite"})
    print(f"wrote {a.out}: {ores} ore cells + {fc['bare']} bare cells {dict(fc)} (sparse_ore={sparse})")
    print(f"TRUE origin (world of relative 0,0,0) = ({x0},{y0},{z0})  secret_chunk=({x0>>4},{z0>>4})")

if __name__=="__main__":
    main()
