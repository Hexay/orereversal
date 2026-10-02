"""Candidate ore positions for a seed, from harness/region_dump (cubiomes). Real ore is a subset of these."""

import argparse
import collections
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGION_DUMP = os.path.join(ROOT, "harness", "region_dump" + (".exe" if os.name == "nt" else ""))

# Families an observation may name. Ordered: where candidates of two families overlap, the later one
# wins in family_at(), so labels are deterministic.
USABLE = ("tuff", "gravel", "granite", "copper", "iron", "diamond", "redstone", "lapis")
# Families whose candidates contain every real block, so they are safe anchors (cuda/README.md).
ANCHOR_FAMILIES = ("tuff", "redstone", "lapis", "granite")


def minecraft_version(text):
    """argparse type: a Java version string, 1.18 or later (1.N, 1.N.p, or year-based like 26.1)."""
    minor = int(text.split(".")[1]) if text.startswith("1.") else 99
    if minor < 18:
        raise argparse.ArgumentTypeError(f"unsupported version '{text}': 1.18 or later is required")
    return text


def add_world_args(parser):
    parser.add_argument("--seed", default="123", help="world seed")
    parser.add_argument(
        "--version", type=minecraft_version, default="1.18", help="Minecraft version, 1.18 or later"
    )


def region_dump(
    seed, version, chunk_min_x, chunk_max_x, chunk_min_z, chunk_max_z, min_y=-64, max_y=-1, veins=True
):
    """Returns {family: {(x, y, z), ...}} for the chunk box and y range, including ore-vein blocks
    (iron-vein tuff/iron, copper-vein granite/copper) unless veins=False."""
    bounds = (chunk_min_x, chunk_max_x, chunk_min_z, chunk_max_z, min_y, max_y)
    extra = ["+veins"] if veins else []
    result = subprocess.run(
        [REGION_DUMP, str(seed), version, *map(str, bounds), *extra],
        capture_output=True,
        text=True,
        check=True,
    )
    candidates = collections.defaultdict(set)
    for line in result.stdout.splitlines()[1:]:
        family, x, y, z = line.split(",")
        candidates[family].add((int(x), int(y), int(z)))
    return candidates


def family_at(candidates):
    """{(x, y, z): family} over the usable families."""
    return {pos: family for family in USABLE for pos in candidates.get(family, ())}
