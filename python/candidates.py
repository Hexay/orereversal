# Ore candidate positions for a seed, via harness/region_dump.exe (cubiomes). Real ores are a subset.
import collections, os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE = os.path.join(ROOT, "harness", "region_dump" + (".exe" if os.name == "nt" else ""))
# Families an observation may name (see docs/observation-format.md). Ordered: on overlapping candidates
# the later family wins in family_at(), so labels are deterministic.
USABLE = ("tuff", "gravel", "granite", "copper", "iron", "redstone", "lapis")


def region_dump(seed, version, cxlo, cxhi, czlo, czhi, ylo=-64, yhi=-1):
    r = subprocess.run(
        [EXE, str(seed), version, str(cxlo), str(cxhi), str(czlo), str(czhi), str(ylo), str(yhi)],
        capture_output=True,
        text=True,
        check=True,
    )
    cand = collections.defaultdict(set)
    for ln in r.stdout.splitlines()[1:]:
        f, x, y, z = ln.split(",")
        cand[f].add((int(x), int(y), int(z)))
    return cand


def family_at(cand):
    return {p: f for f in USABLE for p in cand.get(f, ())}
