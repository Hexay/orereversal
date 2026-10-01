"""Synthetic observation of a single flat wall (constant world x), for testing how much a wall reveals.

A one-block-deep face carries far less structure than a carved room's shell.
"""

import argparse

import candidates as C
from observation import Cell, write_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", default="123")
    parser.add_argument("--x0", type=int, required=True, help="world x of the wall face")
    parser.add_argument("--y0", type=int, required=True, help="world y of the wall's bottom row")
    parser.add_argument("--z0", type=int, required=True, help="world z of the wall's first column")
    parser.add_argument("--w", type=int, default=10, help="width along z")
    parser.add_argument("--h", type=int, default=10, help="height along y")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cx, cz = args.x0 >> 4, args.z0 >> 4
    nearby = C.region_dump(args.seed, "1.18", cx - 2, cx + 2, cz - 2, ((args.z0 + args.w) >> 4) + 2)
    family_at = C.family_at(nearby)
    cells = [
        Cell(0, y - args.y0, z - args.z0, family_at.get((args.x0, y, z), "bare"))
        for y in range(args.y0, args.y0 + args.h)
        for z in range(args.z0, args.z0 + args.w)
    ]

    print(write_observation(args.out, cells))
    print(f"true origin of relative (0,0,0): ({args.x0},{args.y0},{args.z0}), chunk ({cx},{cz})")


if __name__ == "__main__":
    main()
