"""Candidate ore positions for a seed, from harness/region_dump (cubiomes). Real ore is a subset of these."""

import collections
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGION_DUMP = os.path.join(ROOT, "harness", "region_dump" + (".exe" if os.name == "nt" else ""))

# Families an observation may name. Ordered: where candidates of two families overlap, the later one
# wins in family_at(), so labels are deterministic.
USABLE = ("tuff", "gravel", "granite", "copper", "iron", "redstone", "lapis")


def region_dump(seed, version, chunk_min_x, chunk_max_x, chunk_min_z, chunk_max_z, min_y=-64, max_y=-1):
    """Returns {family: {(x, y, z), ...}} for the chunk box and y range."""
    bounds = (chunk_min_x, chunk_max_x, chunk_min_z, chunk_max_z, min_y, max_y)
    result = subprocess.run(
        [REGION_DUMP, str(seed), version, *map(str, bounds)], capture_output=True, text=True, check=True
    )
    candidates = collections.defaultdict(set)
    for line in result.stdout.splitlines()[1:]:
        family, x, y, z = line.split(",")
        candidates[family].add((int(x), int(y), int(z)))
    return candidates


def family_at(candidates):
    """{(x, y, z): family} over the usable families."""
    return {pos: family for family in USABLE for pos in candidates.get(family, ())}
