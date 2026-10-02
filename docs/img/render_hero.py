"""Renders docs/img/hero.png: the example room's exposed ore and the match ranking it produces.

Needs matplotlib. The ranking comes from a matcher run over 512 x 512 chunks:
    cuda/matcher 123 -256 255 -256 255 examples/obs_big_room.csv > ranking.txt
    py -3 docs/img/render_hero.py ranking.txt
"""

import pathlib
import re
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
COLORS = {
    "tuff": "#7a7d6a",
    "redstone": "#d6362b",
    "lapis": "#2a5bd7",
    "granite": "#c08466",
    "gravel": "#a39a8f",
    "copper": "#e07a36",
    "iron": "#d9b48c",
    "diamond": "#3fd0c9",
}


def load_room(path):
    rows = [line.split(",") for line in path.read_text().splitlines()[1:]]
    return [(f, int(x), int(y), int(z)) for f, x, y, z in rows]


def load_ranking(path):
    pattern = re.compile(r"^ +(\d+) +\((-?\d+), (-?\d+), (-?\d+)\).* (-?[\d.]+)$")
    return [
        (int(m[1]), (int(m[2]), int(m[3]), int(m[4])), float(m[5]))
        for m in map(pattern.match, path.read_text().splitlines())
        if m
    ]


def draw_room(ax, room):
    bare = [c for c in room if c[0] == "bare"]
    ax.scatter(
        [c[1] for c in bare],
        [c[3] for c in bare],
        [c[2] for c in bare],
        s=2,
        c="#c9c9c9",
        alpha=0.25,
        depthshade=False,
    )
    for family, color in COLORS.items():
        cells = [c for c in room if c[0] == family]
        if cells:
            ax.scatter(
                [c[1] for c in cells],
                [c[3] for c in cells],
                [c[2] for c in cells],
                s=9,
                c=color,
                label=f"{family} ({len(cells)})",
                depthshade=False,
            )
    ax.set_title("Input: ore exposed on the walls of a dug-out room", fontsize=11, pad=0)
    ax.set_axis_off()
    ax.set_box_aspect(None, zoom=1.12)
    ax.view_init(elev=22, azim=-58)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.06), fontsize=8, frameon=False, ncol=4, markerscale=1.5
    )


def draw_ranking(ax, ranking, ore_total):
    labels = [f"#{rank}  ({x}, {y}, {z})" for rank, (x, y, z), _ in ranking]
    scores = [score for *_, score in ranking]
    colors = ["#2f9e44"] + ["#adb5bd"] * (len(scores) - 1)
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.axvline(0, color="#495057", linewidth=0.8)
    ax.set_xlim(min(scores) * 1.15, ore_total * 1.12)
    ax.text(scores[0], len(scores) - 1, f"  {scores[0]:.0f}", va="center", fontsize=9)
    ax.set_xlabel("score = ore cells matched − ore predicted on plain stone", fontsize=9)
    ax.set_title("Output: best match is unique across 262,144 chunks", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    room = load_room(ROOT / "examples" / "obs_big_room.csv")
    ranking = load_ranking(pathlib.Path(sys.argv[1]))
    fig = plt.figure(figsize=(11, 4.2), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1))
    draw_room(fig.add_subplot(grid[0], projection="3d"), room)
    draw_ranking(fig.add_subplot(grid[1]), ranking, sum(c[0] != "bare" for c in room))
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "img" / "hero.png", facecolor="white")


if __name__ == "__main__":
    main()
