"""Reference CPU solver: rank where in a known-seed world an observation (docs/observation-format.md) is.

The same two-stage algorithm as the GPU matcher in cuda/, over a small region around the origin:
  1. Presence. Every candidate block of the observation's rarest family is a hypothesis, in all 8
     orientations, scored by how many observed ore cells the seed predicts there. Real ore is a subset
     of the candidates, so the true location always has the top presence score and is never pruned.
  2. Soft absence. The best hypotheses lose a point (times the weight) for every bare cell where the seed
     predicts ore.
Tolerances dilate the candidate sets by twice the given error, because the anchor cell is itself
uncertain by that much.
"""

import argparse
from typing import NamedTuple

import candidates as C
from observation import load_observation

ORIENTATIONS = [(rotation, mirror) for rotation in range(4) for mirror in (1, -1)]


class Hypothesis(NamedTuple):
    origin: tuple  # world position of the observation's (0, 0, 0)
    present: int
    absence_hits: int
    score: float
    rotation: int
    mirror: int


def orient_xz(x, z, rotation, mirror):
    x *= mirror
    for _ in range(rotation):
        x, z = -z, x
    return x, z


def dilate(points, radius):
    if radius == 0:
        return set(points)
    offsets = range(-radius, radius + 1)
    return {
        (x + dx, y + dy, z + dz) for x, y, z in points for dx in offsets for dy in offsets for dz in offsets
    }


def placed(cell, origin, rotation, mirror):
    dx, dz = orient_xz(cell.x, cell.z, rotation, mirror)
    return (origin[0] + dx, origin[1] + cell.y, origin[2] + dz)


def count_present(ore, origin, rotation, mirror, candidates_by_family):
    return sum(placed(c, origin, rotation, mirror) in candidates_by_family.get(c.family, ()) for c in ore)


def count_absence_hits(bare, origin, rotation, mirror, any_ore):
    return sum(placed(c, origin, rotation, mirror) in any_ore for c in bare)


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def solve(candidates, ore, bare, tolerance, absence_weight=1.0, absence_tolerance=0, keep=300, top_n=8):
    """Returns (best hypotheses, anchor family, hypothesis count)."""
    observed = {c.family for c in ore}
    scored_families = [f for f in candidates if f in observed or f in C.USABLE]
    present_sets = {f: dilate(candidates[f], 2 * tolerance) for f in scored_families}
    any_ore = set().union(*(dilate(candidates[f], 2 * absence_tolerance) for f in scored_families))

    anchor_families = [f for f in sorted(observed) if candidates.get(f)]  # sorted: deterministic ties
    if not anchor_families:
        return [], None, 0
    anchor_family = min(anchor_families, key=lambda f: len(candidates[f]))
    anchor = next(c for c in ore if c.family == anchor_family)

    # Stage 1: best presence (and its orientation) per origin, over every hypothesis.
    best_by_origin = {}
    hypothesis_count = 0
    for rotation, mirror in ORIENTATIONS:
        ax, az = orient_xz(anchor.x, anchor.z, rotation, mirror)
        for cx, cy, cz in candidates[anchor_family]:
            hypothesis_count += 1
            origin = (cx - ax, cy - anchor.y, cz - az)
            present = count_present(ore, origin, rotation, mirror, present_sets)
            if present > best_by_origin.get(origin, (-1,))[0]:
                best_by_origin[origin] = (present, rotation, mirror)
    survivors = sorted(best_by_origin.items(), key=lambda item: -item[1][0])[:keep]

    # Stage 2: soft absence on the survivors, then keep one hypothesis per neighbourhood.
    hypotheses = []
    for origin, (present, rotation, mirror) in survivors:
        hits = count_absence_hits(bare, origin, rotation, mirror, any_ore) if bare else 0
        hypotheses.append(
            Hypothesis(origin, present, hits, present - absence_weight * hits, rotation, mirror)
        )
    hypotheses.sort(key=lambda h: -h.score)
    best = []
    for h in hypotheses:
        if all(chebyshev(h.origin, kept.origin) > 2 * tolerance + 1 for kept in best):
            best.append(h)
        if len(best) >= top_n:
            break
    return best, anchor_family, hypothesis_count


def print_report(args, ore, bare, results, anchor_family, hypothesis_count):
    size = 2 * args.region + 1
    print(
        f"obs: {len(ore)} ore + {len(bare)} bare | search {size**2} ch | anchor={anchor_family} "
        f"hyps={hypothesis_count} err=+-{args.error} abs_err=+-{args.abs_error} w={args.absence_weight}"
    )
    if not results:
        print("no candidates.")
        return
    print(
        f"\n{'rank':>4} {'world_origin':>20} {'chunk':>11} {'orient':>7} {'present':>8} {'absHits':>8} {'final':>9}"
    )
    for rank, h in enumerate(results, 1):
        chunk = (h.origin[0] >> 4, h.origin[2] >> 4)
        orient = f"r{h.rotation}m{h.mirror}"
        print(
            f"{rank:>4} {str(h.origin):>20} {str(chunk):>11} {orient:>7} {h.present}/{len(ore)} "
            f"{h.absence_hits:>8} {h.score:>9.1f}"
        )
    top = results[0].score
    second = results[1].score if len(results) > 1 else 0
    verdict = "CONFIDENT (unique)" if top - second >= max(3, 0.3 * len(ore)) else "shortlist"
    print(f"\ntop_final={top:.1f}  margin_to_next={top - second:.1f}  => {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("observation")
    parser.add_argument("--seed", default="123")
    parser.add_argument("--version", default="1.18")
    parser.add_argument("--region", type=int, default=13, help="search chunks -R..R on both axes")
    parser.add_argument("--error", type=int, default=0, help="ore-cell position tolerance in blocks")
    parser.add_argument("--abs-error", type=int, default=0, help="bare-cell position tolerance in blocks")
    parser.add_argument("--absence-weight", type=float, default=1.0)
    args = parser.parse_args()

    ore, bare = load_observation(args.observation)
    r = args.region
    candidates = C.region_dump(args.seed, args.version, -r - 1, r + 1, -r - 1, r + 1)
    results, anchor_family, hypothesis_count = solve(
        candidates, ore, bare, args.error, args.absence_weight, args.abs_error
    )
    print_report(args, ore, bare, results, anchor_family, hypothesis_count)


if __name__ == "__main__":
    main()
