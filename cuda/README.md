# cuda/: the GPU matcher

A CUDA port of cubiomes' 1.18+ ore generation, plus the two-pass localizer built on it. It is the GPU
version of [`python/solve.py`](../python/solve.py) and uses the same scoring. For build and usage, see
the [top-level README](../README.md#quick-start).

## Files

The matcher is a single translation unit: `matcher.cu` includes everything else.

| File | Contents |
|---|---|
| **Ore generation** (plain C, host and device) | |
| `ore_config.h` | Ore families and the 1.18 config table. The four GPU families come first, so a family id is also its occupancy-grid slot. |
| `ore_rng.h` | Xoroshiro128++ with Java semantics. Function names follow cubiomes' `rng.h`. |
| `occupancy.h` | The occupancy grid (one bit per deepslate-band block per GPU family) and the anchor sink. |
| `oregen.h` | Vein generation, a line-by-line port of cubiomes' `generateOres` / `generateVeinPart`. |
| `oretest.c` | Prints `oregen.h`'s output in `harness/region_dump` format, for the bit-exact diff. |
| **Matcher** | |
| `matcher.cu` | `main()`: parse options, load the observation, run pass 1, refine, report. |
| `options.h` | Command-line options and usage text. |
| `observation.h` | Loading the observation CSV and choosing the anchor family and GPU configs. |
| `generate.cuh` | Generation kernels: `kSetupVeins` + `kFillVeins` (default) and `kGenerateLegacy`. |
| `score.cuh` | `kMortonKeys` (anchor sort keys) and `kScoreHypotheses`. |
| `gpu_search.cuh` | `GpuSearch`: device buffers and the per-tile loop of pass 1. |
| `top_k.h` | Merging tile survivors into the global top-K, one per neighbourhood. |
| `refine.h` | Pass 2 on the CPU. |
| `region_dump.h` | Locating and running `harness/region_dump`. |
| `report.h` | Console output. |
| `common.h` | `ObsCell`, `Result`, the 8 orientations, `CUDA_CHECK`. |
| `build.sh` / `rebuild.bat` | Build `oretest` and `matcher` on Linux / Windows. The host driver is compiled without FMA contraction (`-ffp-contract=off` / `/fp:strict`) so it stays bit-exact. |

## Pipeline

1. **Pass 1 (GPU, per tile).** `kSetupVeins` runs one thread per (chunk, ore config) and writes each
   vein's nodes to scratch. `kFillVeins` then runs one warp per vein and ORs the vein's blocks into
   the occupancy grid for tuff, redstone, lapis and granite. Every candidate of the rarest observed
   family becomes an anchor. `kScoreHypotheses` tests each anchor in 8 orientations for presence and
   soft absence. Hypotheses that pass the `--minfrac` filter are merged into a global top-K.
2. **Pass 2 (CPU refine).** The top-K hypotheses are re-scored with all 8 families. Gravel, copper,
   iron, buried diamond and ore-vein blocks come from `region_dump`. Pass 2 stops at the point where
   the refine-only families together can no longer change the ranking. It runs in parallel with OpenMP.

`--legacy-gen` swaps in the original one-thread-per-chunk `kGenerateLegacy`. It is the bit-exact
reference that every optimization is validated against.

## Which ore families, and why

Real worldgen couples ore to terrain in three independent ways. Only the first one leaves a family
usable for matching.

1. **Replaceable-block intersection (all families).** cubiomes fills every sphere cell, while
   Minecraft only places ore in stone-replaceable cells. cubiomes is therefore a superset of the real
   ore, and no RNG is skipped, so deep and mostly solid regions match exactly.
2. **`discardChanceOnAirExposure > 0` (diamond, gold, lower coal).** Minecraft rolls `nextFloat()`
   for each replaceable cell and skips the roll on air. That desyncs the RNG stream, so replaying it
   needs the full 1.18 terrain. Against the real world, 4 of 16 diamonds and 6 of 14 gold were
   missed. These families are **excluded**.
3. **Gravity after generation (gravel).** Gravel is bit-exact at generation, but it falls on the
   first block update. Treat its margin as a bonus.

| Family | How it's generated | Why |
|---|---|---|
| tuff, redstone, lapis, granite | GPU, bit-exact | Discard-free, and need no surface gate |
| gravel, copper | CPU refine via `region_dump.exe` | Their high Y ranges hit cubiomes' `mapApproxHeight` surface gate, which isn't ported |
| iron | CPU refine via `region_dump.exe` | Discard-free ore features, plus iron ore veins (`harness/ore_veins.h`) |

**Ore veins.** 1.18+ also places large iron veins (iron ore, raw iron and tuff filler, y −60..−8) from
position-only noise, before ore features run. `region_dump +veins` models them with vanilla's cell
interpolation, and refine adds their tuff and iron to the candidates. The GPU pass doesn't generate them,
so in pass 1 vein tuff is a presence miss, absorbed by `--minfrac`. On a real room crossing a vein this took
the true location from 190/269 to 243/269 cells (margin 115 → 183, `examples/real_vein_room.csv`).
| diamond (buried only) | CPU refine via `region_dump.exe` | Discard 1.0 rolls no RNG, so it's exact. Diamonds from the other diamond configs aren't credited |
| gold, coal, other diamond | Excluded | Air-exposure discard (see 2 above) |

## Validation

**Bit-exact generation.** For 64 chunks in the deepslate band, tuff, redstone, lapis and granite show
zero position diffs. Only gravel and copper differ, because they need the surface gate.

```sh
./oretest.exe 123 0 7 0 7 -64 -1 | tail +2 | sort > port.csv
../harness/region_dump.exe 123 1.18 0 7 0 7 -64 -1 | tail +2 | sort > ref.csv
diff port.csv ref.csv   # only gravel/copper lines
```

**End to end against a real 1.18.2 world (seed 123).**

| Room | True origin | Ore cells | Margin |
|---|---|---|---|
| Chunk (1, 1) | (16, -55, 16) | 352 / 352 | 294 |
| Chunk (-5, -5) | (-72, -55, -72) | 589 / 589 | 522 |

Both rooms rank first with no absence hits. With only the four GPU families, the margin stays roughly
flat as the region grows 31×.

**Regression.** Use `examples/obs_big_room.csv` over chunks -32..31. The expected result is
633/633 at (-6, -52, -6), margin 452. With `--no-refine` the ranking must be byte-identical to
`--legacy-gen`.

## Performance

[`docs/gpu-optimization.md`](../docs/gpu-optimization.md) records the profiling history: every
optimization step, its measured effect, and the dead ends that shouldn't be retried.
