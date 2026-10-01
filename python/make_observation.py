"""Synthetic observation of a carved room at a known location, for testing the solvers.

Every block of the one-block shell around the carved box is labelled with the seed's candidate family
there, or bare. --noise jitters ore cells (not bare ones) by up to +-N blocks per axis.
"""

import argparse
import os
import random

import candidates as C
from observation import Cell, write_observation


def shell_cells(x0, x1, y0, y1, z0, z1):
    """World positions of the blocks bordering the box, x outermost, z innermost."""
    for x in range(x0 - 1, x1 + 2):
        for y in range(y0 - 1, y1 + 2):
            for z in range(z0 - 1, z1 + 2):
                if not (x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1):
                    yield x, y, z


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", default="123")
    parser.add_argument("--version", default="1.18")
    parser.add_argument("--box", default="-6,25,-52,-37,-6,25", help="carved air box: x0,x1,y0,y1,z0,z1")
    parser.add_argument("--noise", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(C.ROOT, "examples", "obs_big_room.csv"))
    args = parser.parse_args()

    x0, x1, y0, y1, z0, z1 = (int(v) for v in args.box.split(","))
    nearby = C.region_dump(
        args.seed, args.version, (x0 >> 4) - 2, (x1 >> 4) + 2, (z0 >> 4) - 2, (z1 >> 4) + 2
    )
    family_at = C.family_at(nearby)
    rng = random.Random(1)
    cells = []
    for x, y, z in shell_cells(x0, x1, y0, y1, z0, z1):
        family = family_at.get((x, y, z), "bare")
        rx, ry, rz = x - x0, y - y0, z - z0
        if args.noise and family != "bare":
            rx += rng.randint(-args.noise, args.noise)
            ry += rng.randint(-args.noise, args.noise)
            rz += rng.randint(-args.noise, args.noise)
        cells.append(Cell(rx, ry, rz, family))

    print(write_observation(args.out, cells))
    print(f"true origin of relative (0,0,0): ({x0},{y0},{z0}), chunk ({x0 >> 4},{z0 >> 4})")


if __name__ == "__main__":
    main()
