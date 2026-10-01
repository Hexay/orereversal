# Emit a FLAT exposed-wall observation: a single vertical face (constant x), W wide x H tall.
# Models digging a flat wall and reading the block at each exposed face cell. One block deep -> no
# 3D shell, far less fine structure than a carved room. For testing how much info a wall carries.
import candidates as C
import argparse, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="123")
    ap.add_argument("--x0", type=int, required=True)  # the exposed face plane (world x)
    ap.add_argument("--y0", type=int, required=True)
    ap.add_argument("--z0", type=int, required=True)
    ap.add_argument("--w", type=int, default=10)  # wall width  (along z)
    ap.add_argument("--h", type=int, default=10)  # wall height (along y)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cx, cz = a.x0 >> 4, a.z0 >> 4
    local = C.region_dump(a.seed, "1.18", cx - 2, cx + 2, cz - 2, ((a.z0 + a.w) >> 4) + 2)
    fam_at = C.family_at(local)
    fc = collections.Counter()
    with open(a.out, "w", newline="\n") as f:
        f.write("family,x,y,z\n")
        for Y in range(a.y0, a.y0 + a.h):
            for Z in range(a.z0, a.z0 + a.w):
                fam = fam_at.get((a.x0, Y, Z), "bare")  # solid stone synthetic -> ore or bare
                f.write(f"{fam},0,{Y - a.y0},{Z - a.z0}\n")
                fc[fam] += 1
    ores = sum(v for k, v in fc.items() if k != "bare")
    sparse = sum(v for k, v in fc.items() if k in {"lapis", "redstone", "copper", "granite"})
    print(f"wrote {a.out}: {ores} ore + {fc['bare']} bare {dict(fc)} (sparse={sparse})")
    print(f"TRUE origin = ({a.x0},{a.y0},{a.z0})  secret_chunk=({cx},{cz})")


if __name__ == "__main__":
    main()
