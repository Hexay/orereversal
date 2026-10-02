"""Compare real 1.18.2 world (gt/world) against cubiomes feature candidates, the surface gate, and ore veins."""
import anvil, os, collections, math, sys
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(S, "..", "..", "gt", "world", "region")
LO, HI = -10, 10  # interior chunks (all neighbours fully generated)

TERRAIN_KEYS = ("stone", "deepslate", "dirt", "grass_block", "sand", "gravel", "granite", "diorite", "andesite",
                "tuff", "calcite", "clay", "_ore", "terracotta", "snow_block", "powder_snow", "packed_ice",
                "blue_ice", "podzol", "mycelium", "moss_block", "bedrock", "magma", "obsidian", "dripstone_block",
                "raw_", "amethyst", "budding", "smooth_basalt", "mud")
FAM = {"tuff": "tuff", "granite": "granite", "gravel": "gravel", "copper_ore": "copper",
       "deepslate_copper_ore": "copper", "iron_ore": "iron", "deepslate_iron_ore": "iron",
       "raw_iron_block": "rawiron", "raw_copper_block": "rawcopper",
       "redstone_ore": "redstone", "deepslate_redstone_ore": "redstone",
       "lapis_ore": "lapis", "deepslate_lapis_ore": "lapis"}


NAMES, DEEP, BLK = {}, {}, {}


def block_at(x, y, z):
    inv = block_at.inv
    return inv[int(BLK[(x >> 4, z >> 4)][y + 64, z & 15, x & 15])]


def is_terrain(name):
    return any(k in name for k in TERRAIN_KEYS) and "slab" not in name and "stairs" not in name and "wall" not in name


def decode_chunk(reg, lx, lz):
    """Returns (dict fam->set of (x,y,z) local, topSolid[16,16] absolute y)."""
    d = reg.chunk_data(lx, lz)
    fams = collections.defaultdict(set)
    top = np.full((16, 16), -65, dtype=np.int32)
    deep = np.zeros((128, 16, 16), dtype=np.int16)
    DEEP[(lx, lz, id(reg))] = deep
    for sec in d["sections"]:
        sy = sec["Y"].value
        if "block_states" not in sec:
            continue
        bs = sec["block_states"]
        pal = [p["Name"].value.replace("minecraft:", "") for p in bs["palette"]]
        if "data" not in bs:
            idx = np.zeros(4096, dtype=np.int64)
        else:
            bits = max(4, math.ceil(math.log2(len(pal))))
            per = 64 // bits
            longs = np.array([v & 0xFFFFFFFFFFFFFFFF for v in bs["data"].value], dtype=np.uint64)
            shifts = (np.arange(per, dtype=np.uint64) * np.uint64(bits))
            vals = (longs[:, None] >> shifts[None, :]) & np.uint64((1 << bits) - 1)
            idx = vals.reshape(-1)[:4096].astype(np.int64)
        idx = idx.reshape(16, 16, 16)  # y,z,x
        if -4 <= sy <= 3:
            gid = np.array([NAMES.setdefault(p, len(NAMES)) for p in pal])
            deep[(sy + 4) * 16:(sy + 5) * 16] = gid[idx]
        terr = np.array([is_terrain(p) for p in pal])
        tmask = terr[idx]
        for ly in range(16):
            y = sy * 16 + ly
            m = tmask[ly]
            top[m] = np.maximum(top[m], y)  # top indexed [z][x]
        for pi, name in enumerate(pal):
            f = FAM.get(name)
            if f:
                ys, zs, xs = np.nonzero(idx == pi)
                for a, b, c in zip(ys, zs, xs):
                    fams[f].add((int(c), int(sy * 16 + a), int(b)))
    return fams, top


def load_real():
    real = collections.defaultdict(set)
    top = {}
    regs = {}
    for cx in range(LO - 1, HI + 2):
        for cz in range(LO - 1, HI + 2):
            rx, rz = cx >> 5, cz >> 5
            if (rx, rz) not in regs:
                regs[(rx, rz)] = anvil.Region.from_file(f"{W}/r.{rx}.{rz}.mca")
            fams, t = decode_chunk(regs[(rx, rz)], cx & 31, cz & 31)
            BLK[(cx, cz)] = DEEP.pop((cx & 31, cz & 31, id(regs[(rx, rz)])))
            for z in range(16):
                for x in range(16):
                    top[(cx * 16 + x, cz * 16 + z)] = int(t[z, x])
            if LO <= cx <= HI and LO <= cz <= HI:
                for f, s in fams.items():
                    for (x, y, z) in s:
                        real[f].add((cx * 16 + x, y, cz * 16 + z))
    return real, top


def load_probe():
    cand = collections.defaultdict(set)
    ung = collections.defaultdict(set)
    vd = collections.defaultdict(set)
    vi = collections.defaultdict(set)
    gates = []
    approx = {}
    with open(S + "/probe.csv") as fh:
        for ln in fh:
            p = ln.rstrip().split(",")
            k = p[0]
            if k == "C":
                cand[p[1]].add((int(p[2]), int(p[3]), int(p[4])))
            elif k == "U":
                ung[p[1]].add((int(p[2]), int(p[3]), int(p[4])))
            elif k == "VD":
                vd[p[1]].add((int(p[2]), int(p[3]), int(p[4])))
            elif k == "VI":
                vi[p[1]].add((int(p[2]), int(p[3]), int(p[4])))
            elif k == "G":
                gates.append(tuple(int(v) for v in p[1:]))
            elif k == "H":
                approx[(int(p[1]), int(p[2]))] = int(p[3])
    return cand, ung, vd, vi, gates, approx


def main():
    real, top = load_real()
    cand, ung, vd, vi, gates, approx = load_probe()
    inside = lambda x, z: LO * 16 <= x < (HI + 1) * 16 and LO * 16 <= z < (HI + 1) * 16

    print("== family recall: real blocks NOT in cubiomes feature candidates ==")
    for fam, rng in [("tuff", (-64, 0)), ("iron", (-64, 0)), ("redstone", (-64, 0)), ("lapis", (-64, 0)),
                     ("gravel", (-64, 0)), ("copper", (-64, 0)), ("granite", (-64, 0)),
                     ("granite", (0, 51)), ("copper", (0, 51)), ("gravel", (0, 320))]:
        R = {p for p in real[fam] if rng[0] <= p[1] < rng[1] and p[1] <= top[(p[0], p[2])] - 2}
        miss = R - cand[fam]
        vfam = {"tuff": "tuff", "iron": "iron", "granite": "granite", "copper": "copper"}.get(fam)
        exd = len(miss & vd[vfam]) if vfam else 0
        exi = len(miss & vi[vfam]) if vfam else 0
        print(f"  {fam:8} y[{rng[0]},{rng[1]}) buried real={len(R):6} miss={len(miss):5} "
              f"explained by vein: direct={exd:5} interp={exi:5}")
    for fam in ("rawiron", "rawcopper"):
        R = real[fam]
        print(f"  {fam:8} real={len(R)} in VD={len(R & vd[fam])} in VI={len(R & vi[fam])}")

    print("== vein prediction precision on buried solid cells (y<=top-2) ==")
    for fam in ("tuff", "iron", "granite", "copper"):
        for name, V in (("direct", vd), ("interp", vi)):
            P = {p for p in V[fam] if inside(p[0], p[2]) and p[1] <= top[(p[0], p[2])] - 2}
            hit = len(P & real[fam])
            print(f"  {fam:8} {name}: predicted={len(P):6} real-same-block={hit:6} ({100.0*hit/max(1,len(P)):.1f}%)")

    print("== surface gate: cubiomes approx vs real terrain top (OCEAN_FLOOR_WG ~ top+1) ==")
    errs = []
    for (qx, qz), h in approx.items():
        cols = [top.get((qx * 4 + i, qz * 4 + j)) for i in range(4) for j in range(4)]
        if None in cols:
            continue
        errs.append(h - (max(cols) + 1))
    e = np.array(errs)
    print(f"  4x4 cells={len(e)} approx-real: mean={e.mean():.1f} median={np.median(e):.0f} "
          f"p5={np.percentile(e,5):.0f} p95={np.percentile(e,95):.0f} |err|<=4: {100*np.mean(abs(e)<=4):.0f}%")
    names = {17: "gravel", 6: "copper", 27: "lower_granite", 43: "upper_granite", 44: "upper_iron", 39: "tuff",
             31: "middle_iron", 37: "small_iron", 35: "redstone", 28: "lower_redstone", 19: "lapis",
             3: "buried_lapis", 20: "large_copper"}
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    disagree = []
    marg = []
    for (ot, cx, cz, vein, sx, sy, sz, osz, maxH, ok) in gates:
        realH = None
        ok2 = True
        for x in range(sx, sx + osz + 1):
            for z in range(sz, sz + osz + 1):
                t = top.get((x, z))
                if t is None:
                    ok2 = False
                    break
                realH = t + 1 if realH is None else max(realH, t + 1)
            if not ok2:
                break
        if not ok2:
            continue
        rp = int(sy <= realH)
        s = per[ot]
        s[0] += 1
        s[1] += ok
        s[2] += rp
        s[3] += int(rp != ok)
        if rp != ok:
            disagree.append((cx, cz, ot))
        marg.append((ot, maxH - sy, rp != ok))
    for ot, (n, cp, rp, dis) in sorted(per.items()):
        print(f"  oreType {ot:2} {names.get(ot, ''):14} veins={n:6} cubiomesPass={cp:6} realPass={rp:6} "
              f"DISAGREE={dis:5} ({100.0*dis/max(1,n):.2f}%)")
    print("== GPU-equivalent (ungated) vs cubiomes-gated recall, deep band, buried ==")
    for fam in ("tuff", "redstone", "lapis", "granite", "copper", "iron"):
        R = {p for p in real[fam] if p[1] < 0 and p[1] <= top[(p[0], p[2])] - 2}
        print(f"  {fam:8} real={len(R):6} miss gated(cubiomes)={len(R - cand[fam]):5} "
              f"miss ungated(GPU-style)={len(R - ung[fam]):5}  ungated-not-gated cands={len(ung[fam]-cand[fam])}")
    print("== are deep misses localised to chunks whose gate decision disagrees with real terrain? ==")
    for fam, ots in (("gravel", {17}), ("copper", {6}), ("lapis", {3}), ("iron", {31, 37})):
        dis = {(cx, cz) for cx, cz, ot in disagree if ot in ots}
        near = {(a + i, b + j) for a, b in dis for i in (-1, 0, 1) for j in (-1, 0, 1)}
        R = {p for p in real[fam] if p[1] < 0 and p[1] <= top[(p[0], p[2])] - 2}
        miss = R - cand[fam]
        if fam == "iron":
            miss -= vi["iron"]
        inn = sum(1 for p in miss if (p[0] >> 4, p[2] >> 4) in near)
        print(f"  {fam:7} disagreeing chunks={len(dis):3} (3x3 nbhd covers {len(near)}/441) "
              f"deep misses={len(miss):5} of which inside nbhd={inn:5}")
    print("== ambiguity band: veins with |cubiomesMaxH - startY| <= T (candidates for branch-both) ==")
    for ots, nm in (({17}, "gravel"), ({6}, "copper"), ({3}, "buried_lapis"), ({31, 37}, "mid+small iron")):
        rows = [(m, d) for ot, m, d in marg if ot in ots]
        nd = sum(d for _, d in rows)
        out = []
        for T in (4, 8, 12, 16, 24):
            amb = [d for m, d in rows if abs(m) <= T]
            out.append(f"T={T}: amb={100.0*len(amb)/len(rows):.1f}% caught={sum(amb)}/{nd}")
        print(f"  {nm:14} " + " | ".join(out))
    block_at.inv = {v: k for k, v in NAMES.items()}
    print("== ghost candidates: deep candidate cells whose real block is plain deepslate/stone (absence-hit fuel) ==")
    for fam in ("tuff", "redstone", "lapis", "granite", "gravel", "copper", "iron"):
        for nm, C in (("gated", cand), ("ungated", ung)):
            if fam not in C:
                continue
            P = [p for p in C[fam] if inside(p[0], p[2]) and p[1] < 0]
            g = sum(1 for p in P if block_at(*p) in ("deepslate", "stone"))
            print(f"  {fam:8} {nm:8} deep cands={len(P):7} on bare deepslate/stone={g:6} ({100.0*g/max(1,len(P)):.2f}%)")
    print("== what real block sits on interp-vein-predicted cells that are not that block (deep, buried) ==")
    block_at.inv = {v: k for k, v in NAMES.items()}
    for fam in ("tuff", "iron"):
        c = collections.Counter()
        for p in vi[fam]:
            if inside(p[0], p[2]) and p[1] < 0 and p[1] <= top[(p[0], p[2])] - 2 and p not in real[fam]:
                c[block_at(*p)] += 1
        print(f"  {fam}: {c.most_common(8)}")
    for fam in ("tuff", "iron"):
        c = collections.Counter()
        for p in vd[fam]:
            if inside(p[0], p[2]) and p[1] < 0 and p[1] <= top[(p[0], p[2])] - 2 and p not in real[fam]:
                c[block_at(*p)] += 1
        print(f"  direct {fam}: {c.most_common(6)}")


main()
