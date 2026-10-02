"""Reference CPU solver: rank where in a known-seed world an observation (docs/observation-format.md) is.

The GPU matcher's two-stage algorithm (cuda/), over a small region around the origin and without its
surface-gate variants or --minfrac pre-filter:
  1. Presence. Every candidate block of the anchor family (the rarest observed of tuff, redstone, lapis
     and granite, whose candidates contain every real block) is a hypothesis, in all 8 orientations,
     scored by how many observed ore cells the seed predicts there.
  2. Soft absence. The best hypotheses lose a point (times the weight) for every bare cell where the seed
     predicts ore.
A lapis-anchored result that isn't confident is retried anchored on redstone or granite, as in the matcher.
With --error e, ore cells match candidates within +-e, and the best hypotheses are re-centred: every
origin within +-e is re-scored, because the anchor cell (and so the origin) is itself off by up to e.
With --abs-error A, a bare cell only counts against a hypothesis if ore is predicted throughout +-A.
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


class Scoring(NamedTuple):
    tolerance: int
    absence_weight: float
    absence_tolerance: int


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


def erode(points, radius):
    """The points whose whole +-radius neighbourhood is in `points`."""
    if radius == 0:
        return set(points)
    offsets = range(-radius, radius + 1)
    return {
        (x, y, z)
        for x, y, z in points
        if all((x + dx, y + dy, z + dz) in points for dx in offsets for dy in offsets for dz in offsets)
    }


def placed(cell, origin, rotation, mirror):
    dx, dz = orient_xz(cell.x, cell.z, rotation, mirror)
    return (origin[0] + dx, origin[1] + cell.y, origin[2] + dz)


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def distinct(hypotheses, spacing, limit=None):
    """Best first, dropping any hypothesis within `spacing` of one already kept."""
    kept = []
    for h in sorted(hypotheses, key=lambda h: (-h.score, h.origin)):
        if all(chebyshev(h.origin, k.origin) > spacing for k in kept):
            kept.append(h)
            if limit and len(kept) >= limit:
                break
    return kept


def choose_anchor_family(ore):
    counts = {f: sum(c.family == f for c in ore) for f in C.ANCHOR_FAMILIES}
    observed = [f for f in C.ANCHOR_FAMILIES if counts[f]]
    return min(observed, key=lambda f: counts[f]) if observed else None


def retry_anchor_family(ore, anchor_family):
    """See retryAnchorFamily in cuda/observation.h."""
    if anchor_family != "lapis":
        return None
    counts = {f: sum(c.family == f for c in ore) for f in ("redstone", "granite")}
    observed = [f for f in counts if counts[f]]
    return min(observed, key=lambda f: counts[f]) if observed else None


class Solver:
    def __init__(self, candidates, ore, bare, scoring):
        self.ore, self.bare, self.scoring = ore, bare, scoring
        self.candidates = candidates
        self.present_sets = {f: dilate(candidates[f], scoring.tolerance) for f in C.USABLE}
        self.absent_set = erode(set().union(*(candidates[f] for f in C.USABLE)), scoring.absence_tolerance)

    def count_present(self, origin, rotation, mirror):
        return sum(
            placed(c, origin, rotation, mirror) in self.present_sets.get(c.family, ()) for c in self.ore
        )

    def hypothesis_at(self, origin, rotation, mirror, present=None):
        if present is None:
            present = self.count_present(origin, rotation, mirror)
        hits = sum(placed(c, origin, rotation, mirror) in self.absent_set for c in self.bare)
        return Hypothesis(
            origin, present, hits, present - self.scoring.absence_weight * hits, rotation, mirror
        )

    def recentred(self, h):
        shifts = range(-self.scoring.tolerance, self.scoring.tolerance + 1)
        nearby = [
            self.hypothesis_at((h.origin[0] + dx, h.origin[1] + dy, h.origin[2] + dz), h.rotation, h.mirror)
            for dx in shifts
            for dy in shifts
            for dz in shifts
        ]
        return min(nearby, key=lambda n: (-n.score, n.origin))

    def score_anchor(self, anchor_family, keep=300):
        """Stage 1 over every hypothesis of the anchor family, then stage 2 on the `keep` best.
        Returns (hypotheses, hypothesis count)."""
        anchor = next(c for c in self.ore if c.family == anchor_family)
        best_by_origin = {}
        count = 0
        for rotation, mirror in ORIENTATIONS:
            ax, az = orient_xz(anchor.x, anchor.z, rotation, mirror)
            for cx, cy, cz in self.candidates[anchor_family]:
                count += 1
                origin = (cx - ax, cy - anchor.y, cz - az)
                present = self.count_present(origin, rotation, mirror)
                if present > best_by_origin.get(origin, (-1,))[0]:
                    best_by_origin[origin] = (present, rotation, mirror)
        survivors = sorted(best_by_origin.items(), key=lambda item: -item[1][0])[:keep]
        return [self.hypothesis_at(origin, r, m, present) for origin, (present, r, m) in survivors], count

    def shortlist(self, hypotheses, top_n=8):
        """The best distinct hypotheses (re-centred under --error), plus the best one more than the
        verdict separation away, so the margin always has a genuine competitor."""
        spacing = 2 * self.scoring.tolerance + 1
        best = distinct(hypotheses, spacing, top_n)
        if self.scoring.tolerance:
            best = distinct({self.recentred(h) for h in best}, spacing)
        verdict = Verdict.of(best, self.ore, self.bare, self.scoring.tolerance)
        if best and verdict.competitor is None:
            far = distinct(
                (h for h in hypotheses if chebyshev(h.origin, best[0].origin) > verdict.separation), 0, 1
            )
            best += far
        return best


class Verdict(NamedTuple):
    separation: int  # results closer than this to the winner are shifted copies of it
    competitor: Hypothesis
    margin: float
    confident: bool

    @staticmethod
    def of(results, ore, bare, tolerance):
        xs = [c.x for c in ore + bare]
        zs = [c.z for c in ore + bare]
        separation = max(2 * tolerance + 1, max(xs) - min(xs), max(zs) - min(zs))
        if not results:
            return Verdict(separation, None, 0.0, False)
        competitor = next((h for h in results if chebyshev(h.origin, results[0].origin) > separation), None)
        margin = results[0].score - competitor.score if competitor else 0.0
        return Verdict(
            separation, competitor, margin, competitor is not None and margin >= max(3, 0.3 * len(ore))
        )


def solve(candidates, ore, bare, scoring):
    """Returns (best hypotheses, anchor families tried, hypothesis count)."""
    anchor_family = choose_anchor_family(ore)
    if anchor_family is None:
        return [], [], 0
    solver = Solver(candidates, ore, bare, scoring)
    hypotheses, count = solver.score_anchor(anchor_family)
    results = solver.shortlist(hypotheses)
    retry = retry_anchor_family(ore, anchor_family)
    if retry and not Verdict.of(results, ore, bare, scoring.tolerance).confident:
        more, more_count = solver.score_anchor(retry)
        return solver.shortlist(hypotheses + more), [anchor_family, retry], count + more_count
    return results, [anchor_family], count


def print_report(args, ore, bare, results, anchors, hypothesis_count):
    side = 2 * args.region + 3
    print(
        f"obs: {len(ore)} ore + {len(bare)} bare | search {side**2} ch | anchor={'+'.join(anchors) or '-'} "
        f"hyps={hypothesis_count} err=+-{args.error} abs_err=+-{args.abs_error} w={args.absw}"
    )
    if not results:
        print("no candidates.")
        return
    print(
        f"\n{'rank':>4} {'world_origin':>20} {'chunk':>11} {'orient':>7} {'present':>8} {'absH':>8} {'final':>9}"
    )
    for rank, h in enumerate(results, 1):
        chunk = (h.origin[0] >> 4, h.origin[2] >> 4)
        orient = f"r{h.rotation}m{h.mirror}"
        print(
            f"{rank:>4} {str(h.origin):>20} {str(chunk):>11} {orient:>7} {h.present}/{len(ore)} "
            f"{h.absence_hits:>8} {h.score:>9.1f}"
        )
    verdict = Verdict.of(results, ore, bare, args.error)
    top = results[0]
    if verdict.competitor is None:
        print(
            f"\ntop_final={top.score:.1f}  margin=n/a (nothing scored more than {verdict.separation} blocks away)"
        )
        return
    label = "CONFIDENT (unique)" if verdict.confident else "shortlist"
    print(
        f"\ntop_final={top.score:.1f}  margin={verdict.margin:.1f} (vs best >{verdict.separation} blocks away)"
        f"  => {label}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("observation")
    C.add_world_args(parser)
    parser.add_argument("--region", type=int, default=13, help="search chunks -R..R on both axes")
    parser.add_argument("--error", type=int, default=0, help="ore-cell position tolerance in blocks")
    parser.add_argument("--abs-error", type=int, default=0, help="bare-cell position tolerance in blocks")
    parser.add_argument(
        "--absw", "--absence-weight", type=float, default=1.0, help="weight of each absence hit"
    )
    args = parser.parse_args()

    ore, bare = load_observation(args.observation)
    r = args.region
    candidates = C.region_dump(args.seed, args.version, -r - 1, r + 1, -r - 1, r + 1)
    scoring = Scoring(args.error, args.absw, args.abs_error)
    results, anchors, hypothesis_count = solve(candidates, ore, bare, scoring)
    print_report(args, ore, bare, results, anchors, hypothesis_count)


if __name__ == "__main__":
    main()
