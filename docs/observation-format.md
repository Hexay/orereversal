# Ore observation format

Both solvers (`cuda/matcher.exe` and `python/solve.py`) consume a CSV of observed ore/stone-type blocks extracted from a target build.
Your extraction tool (world copy / mod) should emit this format.

## File: CSV, one cell per line
```
family,x,y,z
```
- **family** — one of:
  - a usable ore family: `tuff, redstone, lapis, gravel, granite, copper, iron, diamond`. Tuff, redstone,
    lapis and granite carry the search; gravel, copper and iron miss a few real blocks, so treat their
    margin as a bonus. `diamond` only counts when the seed placed a *buried* diamond there (the only
    diamond config that doesn't desync); other diamonds are simply not credited. (NOT gold/coal — they
    desync; NOT dirt/clay — contaminated. See ../cuda/README.md "Which ore families".)
  - `bare` — an exposed cell that is **plain `stone` or `deepslate` only**. These enable SOFT ABSENCE
    scoring: a candidate location that predicts an ore where you saw `bare` is penalized, sharply cutting
    false positives (esp. via the dense families).

  **Emit rule (two allowlists — everything else is omitted):**
  1. block is a usable family (`tuff/redstone/lapis/gravel/granite/copper/iron/diamond`, incl. deepslate
     variants) → emit that family.
  2. block is `stone` or `deepslate` → emit `bare`.
  3. **anything else — air, lava, water, gold/coal ore, andesite/diorite — OMIT** (don't
     write a row). Cells you never exposed are likewise omitted (= unobserved, no info).

  Why omit instead of marking `bare`: cubiomes assumes every cell is solid stone, so it predicts ores
  inside cells that are actually cave air/lava or a desync ore (gold/coal/diamond). Calling those `bare`
  penalizes the TRUE location for a prediction that's only "wrong" because cubiomes can't see the cave.
  Real-world test: a room cutting a lava cave self-penalized the truth by 30 under the all-else-bare rule;
  omitting fixed it (research-log.md P6).
- **x,y,z** — block coordinates. May be **relative to any origin** you choose; the solver recovers the
  absolute position. **Orientation may be unknown** — the solver tries all 8 horizontal rotations/mirrors.
  - `y` should be the in-game Y (the solver searches the deepslate band, default -64..-1). Keep y on the
    same scale as x,z (1 unit = 1 block).
  - Axes: x = world-east, z = world-south, y = up — but since orientation is brute-forced, any consistent
    right-handed-ish horizontal labeling works; the vertical axis (y) must be correct (gravity is not
    rotated).

## What to include
- **Every** exposed usable-family block you can identify, not just the rare ones. The dense families
  (tuff/gravel) provide the fine structure that makes the match world-unique — include them.
- More blocks + more family diversity = stronger, more unique localization. A sizable carved region with
  hundreds of blocks incl. ~20+ sparse (redstone/lapis/copper/granite) localizes uniquely (see research-log.md P2d).

## Example
```
family,x,y,z
lapis,0,-49,0
redstone,4,-50,2
redstone,4,-51,2
tuff,1,-48,3
gravel,6,-52,1
...
```

## Coordinate precision
- Exact coords (mod/world-copy extraction): use tolerance e=0 -> strongest margin.
- If positions are uncertain by a block or two (e.g. read from an image), pass `--error N` to the solver;
  it matches ore cells within +-N and then re-centres the best hypotheses, so the exact origin is still
  recovered (+-2 jitter: exact origin, CONFIDENT; research-log.md P8).
- `--error` applies to ore cells only; `bare` cells stay exact unless you also pass `--abs-error N`, which
  makes a bare cell count only if ore is predicted throughout +-N. Use it when bare positions are
  uncertain too (research-log.md P7, P8).
