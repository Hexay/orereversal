# cuda/: the GPU matcher

A CUDA port of cubiomes' 1.18+ ore generation, plus the two-pass localizer built on it. It is the GPU
version of [`python/solve.py`](../python/solve.py) and uses the same scoring. For build and usage, see
the [top-level README](../README.md#quick-start).

## Files

| File | Contents |
|---|---|
| `oregen.h` | Portable host + `__device__` port of cubiomes ore generation (`rng.h`/`finders.c`). It avoids VLAs and GCC builtins so it compiles under both MSVC and nvcc. |
| `matcher.cu` | Entry point. Tiles the search region, runs pass 1 on the GPU and pass 2 on the CPU, and prints the ranking. |
| `matcher_kernels.cuh` | Device kernels: `kSetup` + `kFill` (default generator), `kGenerate` (legacy generator), `kAnchorKey`, `kScore`. |
| `matcher_refine.cuh` | Observation loader and the pass-2 CPU refine. |
| `matcher_common.h` | Family index maps, occupancy probes, shared structs, and a glossary of the short names used in the kernels. |
| `oretest.c` | CPU driver that prints in `harness/region_dump.exe` format, for the bit-exact diff test. |
| `build.sh` / `rebuild.bat` | Build `oretest` and `matcher` on Linux / Windows. The host driver is compiled without FMA contraction (`-ffp-contract=off` / `/fp:strict`) so it stays bit-exact. |

## Pipeline

1. **Pass 1 (GPU, per tile).** `kSetup` runs one thread per (chunk, ore config) and writes vein node
   lists. `kFill` then runs one warp per vein and fills that vein's spheres into a per-family
   occupancy bitmask for tuff, redstone, lapis and granite. Every cell of the rarest observed family
   becomes an anchor. `kScore` tests each anchor in 8 orientations for presence and soft absence.
   Hypotheses that pass the `--minfrac` filter are merged into a global top-K.
2. **Pass 2 (CPU refine).** The top-K hypotheses are re-scored with all 7 families. Gravel, copper
   and iron come from `region_dump.exe`. Pass 2 stops at the point where gravel, copper and iron
   together can no longer change the ranking. It runs in parallel with OpenMP.

`--legacy-gen` swaps in the original one-thread-per-chunk `kGenerate`. It is the bit-exact reference
that every optimization is validated against.

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
| iron | CPU refine via `region_dump.exe` | Discard-free, but the separate 1.18 ore-vein noise isn't simulated (about 2 of 16 real deep blocks are missed) |
| diamond, gold, coal | Excluded | Air-exposure discard (see 2 above) |

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
