"""Reading and writing observation CSVs (docs/observation-format.md)."""

import collections
import csv
import os
from typing import NamedTuple

SPARSE_FAMILIES = {"lapis", "redstone", "copper", "granite"}


class Cell(NamedTuple):
    x: int
    y: int
    z: int
    family: str  # an ore family, or "bare"


def load_observation(path):
    """Returns (ore cells, bare cells)."""
    ore, bare = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip() in ("family", "") or row[0].startswith("#"):
                continue
            cell = Cell(int(row[1]), int(row[2]), int(row[3]), row[0].strip())
            (bare if cell.family == "bare" else ore).append(cell)
    return ore, bare


def write_observation(path, cells):
    """Writes cells with Unix line endings (byte-identical across platforms); returns a summary line."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = collections.Counter()
    with open(path, "w", newline="\n") as f:
        f.write("family,x,y,z\n")
        for c in cells:
            f.write(f"{c.family},{c.x},{c.y},{c.z}\n")
            counts[c.family] += 1
    ore = sum(n for family, n in counts.items() if family != "bare")
    sparse = sum(n for family, n in counts.items() if family in SPARSE_FAMILIES)
    return f"wrote {path}: {ore} ore + {counts['bare']} bare {dict(counts)} (sparse={sparse})"
