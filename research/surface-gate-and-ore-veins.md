# Surface gate (gravel/copper) and 1.18 ore veins (iron/tuff): can we do better?

Seed 123, real 1.18.2 world `gt/world`, 441 fully generated interior chunks (-10..10 squared), all Y.
Probe: `research/surface-gate-probe/` (`vein_probe.c` links cubiomes; `analyze.py` decodes the .mca
files directly). Raw numbers: `research/surface-gate-probe/output.txt`. CPU only, no GPU used.

## Verdict (ranked)

| # | Build | Gain | Effort | Bit-exactness risk |
|---|---|---|---|---|
| 1 | **Interpolated ore-vein model (iron + tuff filler, y -60..-8)** in `region_dump` and refine first, then on the GPU | Explains **3547 of 3550** real deep tuff blocks the tuff family misses, and **761 of 989** iron misses. Zero predictions land on plain deepslate. Adds `raw_iron_block`, which is never wrong (15 of 15 real blocks predicted) | S: ~80 LOC in region_dump. GPU: ~150 LOC on a shared Perlin port | **None.** It's a pure function of position with no chunk RNG, so it can't desync anything. Don't use cubiomes' `getOreVeinBlockAt`: it skips interpolation and is wrong (see B) |
| 2 | **Drop gravel/copper/iron from refine's absence term**, or weight them below 1 | Stops ghost candidates from penalising the true location. Deep-band ghost rates measured against real: gravel 3.2%, copper 9.5%, iron 1.7% | XS: one condition in `refine.h` `rescore` | None (it's a scoring change) |
| 3 | **GPU port of cubiomes' approximate surface gate**, shared by gravel, copper and buried_lapis | Parity with region_dump (not with real). Puts gravel in pass 1 (presence only), removes refine's region_dump spawns, and makes GPU lapis match cubiomes (deep misses 88 → 50) | M: ~300 LOC plus a golden test. ~22 KB constant data | Bit-exact **to cubiomes** if the noise is compiled with `--fmad=false` (or `__dmul_rn`/`__dadd_rn`). **Not** bit-exact to the real world (see A) |
| 4 | **Evidence-branched gate at refine**: when \|approxH - startY\| <= T, generate both RNG continuations and keep the better-scoring one per (chunk, config) | At T=24 it catches 59/61 gravel, 282/289 copper and 19/19 buried-lapis gate disagreements (T=12 is enough for lapis) | M, CPU only | Not exact, but it turns a desync into a choice between two candidates. Every hypothesis gets the same freedom |
| — | Don't: an exact OCEAN_FLOOR_WG port (the full 1.18 noise router plus a column search) | Would close the last ~1-4% of gate errors | XL (1.5k+ LOC, noise router, aquifers, caves) | High. This is the "partial terrain sim" the project already rejected |

**New holes in "bit-exact" claims** (the earlier validation covered chunks 0..7 and two rooms; this wider area exposes them):
- **Tuff vs the real world**: ore-vein filler adds tuff at y -60..-8 that no tuff feature predicts. That's 0.7% of deep
  tuff, concentrated inside veins. The GPU port is bit-exact to cubiomes, but in vein regions the real world is not a
  subset of cubiomes. Fixed by #1.
- **Lapis**: `buried_lapis` covers y -64..64, so its veins *do* hit the surface gate in low or ocean terrain.
  cubiomes rejected 45 of 2080 veins here and the GPU rejects none. Against the real world: GPU-style lapis misses
  88/5837 deep blocks and has 154 ghosts; cubiomes misses 50 and has 136 ghosts. All misses sit in the 3x3 neighbourhoods
  of the 17 chunks where the gate disagrees.
- **Granite**: the GPU (no gate) is *closer* to the real world than cubiomes. The real world kept all 909
  lower_granite veins; cubiomes rejected 5, leaving 30 ghosts vs 0 for the GPU. Keep granite ungated even if #3 lands.
- **Redstone**: exact (0 misses, 0 ghosts). Tuff ghosts: 3 out of 565k.

**Is GPU gravel worth it?** It adds about the room's gravel count to the score (refine measured +71/+75), on top of a
pass-1 margin of 200-500 that is already recall-safe and *grows* with search area. The only real gain is robustness for
sparse or gravel-heavy exposures. Cost: roughly 2.4 band-reaching gravel veins × 33 nodes per chunk plus the gate, about
+20-30% kFill. In pass 1 it has to be presence-only (3.2% ghosts). That is low value, so do #3 for the perf and
architecture win (no region_dump in refine) and for lapis parity, not for margin.

## A. The surface gate

**What cubiomes computes.** `finders.c:2036-2045` (`generateOrePositions`) loops over every column of the
(oreSize+1)² vein footprint (gravel 17², copper 9², buried lapis 9²). For each column it calls
`mapApproxHeight(&y, 0, g, sn, x>>2, z>>2, 1, 1)` and places the vein if `startY <= floor(y)` for any column. The
answer is "any" (order doesn't matter), so it's just max over the footprint ≥ startY.

`mapApproxHeight` in 1.18+ (`generator.c:623-640`) ignores `SurfaceNoise` entirely. It is **biome-climate noise**:
`sampleBiomeNoise(&g->bn, np, qx, 0, qz, 0, 0)` and then `y = np[NP_DEPTH]/76.0` (`biomenoise.c:1144-1193`):
- shift ("offset"): 2 DoublePerlin samples, 3 octaves × 2 sets → 12 Perlin evaluations
- continentalness 9×2 = 18, erosion 4×2 = 8 (amp {1,1,0,1,1}), weirdness 3×2 = 6 → **44 Perlin octave evaluations**
- the depth spline `getSpline` (float32, recursive, depth ≤ 4, 42 spline nodes + 151 fixed). Then
  `d = 1 - 83/160 + off` (float), `np = (int64)(10000.0F*d)`, `/76.0`
- temperature, humidity and `climateToBiome` are also computed but unused. Skip them with `SAMPLE_NO_BIOME`, or don't port them.
- separately, `generateOres` calls `getBiomeAt` per vein for `isViableOreBiome`. Only LargeCopperOre (dripstone caves)
  is biome-gated. Exclude it on the GPU rather than port the biome tree. It has its own RNG stream, so excluding it is
  harmless.

**Port estimate.** Device code: Perlin/octave/DoublePerlin sampling (~70 LOC), iterative getSpline (~35), depth assembly
(~15), per-chunk quart-height cache plus gate (~40). Host: flatten the spline pointers to indices and upload the octave
structs (~60). Golden test vs region_dump (~40). About **250-300 LOC**. Constant data: 38 `PerlinNoise` × 320 B ≈ 12 KB,
plus about 10 KB of splines, so ~22 KB fits in `__constant__`. The `d[257]` permutation lookups are data-dependent, so
put them in shared memory or read them through `__ldg`, not constant memory (divergent constant reads serialise).

**Cost.** cubiomes recomputes each column, up to 289 calls × ~52 octaves for a *rejected* gravel vein, which is the
common case (only 36% of gravel veins pass). Cache by quart cell instead: a chunk's veins only touch about 9×9 quart
cells, each shared by neighbouring chunks. Precompute one quart-height grid per tile: 16 cells/chunk × 44 octaves ×
~50 FP64 ops ≈ 35k FP64 ops/chunk. At consumer Ada FP64 rate (~0.7 TFLOP/s) that's ~20M chunks/s, under 10% of the
current ~1.1M chunk/s generator. The gate itself then costs 25-36 byte lookups per vein, far less than a vein fill.
- *CPU grid plus upload*: uses cubiomes' own `mapApproxHeight`, so it's exact by construction. It's simple, but CPU-bound
  at world scale (~14 µs/chunk single-threaded, ~5 min on 16 threads for 300k²). Fine for bounded searches.
- *GPU noise port*: scales, and gives the GPU a shared Perlin layer that #1 reuses. Recommended once #1 is done.

**Bit-exactness on GPU.** All inputs are integer quart coordinates. The noise is IEEE FP64 and the spline is FP32,
with no transcendental functions. cubiomes is built with GCC `-O3` and no `-march`, so it has no FMA. The GPU must
match that by disabling contraction in these functions (`--fmad=false` on that compilation unit, or `__dmul_rn`/`__dadd_rn`).
The decision is `startY <= floor(y)`, an integer compare, so any FP drift could only matter right at an integer boundary.

**But cubiomes' gate is not Minecraft's gate.** I disassembled the 1.18.2 server jar (`gt/server.jar` →
`META-INF/versions/1.18.2/server-1.18.2.jar`, class `cyj` = OreFeature). Its loop calls
`level.getHeight(Heightmap.Types.c, x, z)` with `ctw$a.c` = `OCEAN_FLOOR_WG`, per *block* column: the actual terrain at
feature time. cubiomes approximates that with the climate depth parameter at 4×4 resolution. Measured against the real
terrain top (`analyze.py` heuristic: highest terrain block, excluding vegetation and fluids): approx − real is median
−1, p5 −10, p95 +2, and within ±4 blocks in 83% of quart cells. Gate decisions per vein, cubiomes vs real:

| config | veins | disagree | deep-band effect vs real (cubiomes-gated) |
|---|---|---|---|
| gravel | 6736 | 61 (0.91%) | 2618/102334 misses (2.6%), 3716 ghosts (3.2%) |
| copper | 8103 | 289 (3.6%) | 258/2129 misses (12%), 215 ghosts (9.5%) |
| buried_lapis | 2080 | 19 (0.9%) | 50/5837 misses, 136 ghosts |
| small_iron | 5185 | 81 (1.6%) | most of the 228 non-vein iron misses |
| lower_granite | 909 | 5 (0.55%) | cubiomes worse than ungated (30 ghosts vs 0) |

A disagreement desyncs every later vein of that config in that chunk. That's why a rare gate error shows up as deep-band
damage, and why gravel/copper/iron can't be "exact" through any port of cubiomes. Ungated copper is far worse
(1461 misses, 53% ghosts), so copper does need a gate.

## B. 1.18 ore veins (OreVeinifier)

**Does cubiomes implement it?** Partly. `finders.c:2148-2230` (`initOreVeinNoise`, `getOreVeinBlockAt`) has the right
noises, seeds, thresholds and VeinTypes. It samples the noise **directly at each block**. Vanilla doesn't: the
1.18.2 router (`cud`, NoiseRouterData) builds veinToggle, veinA and veinB as
`interpolated(rangeChoice(y, -60, 51, noise(..), 0))`. I verified the marker enum `ctq$l$a.a` = "interpolated" and the
helper `cud.a(ctp,ctp,III)`. Only `ore_gap` is uninterpolated. So the three noises are sampled at 4×8×4 cell corners
and trilinearly lerped (corners outside y∈[-60,51) give 0, which damps the bottom cell).

The decision logic matches `cuj.a` (OreVeinifier) byte for byte: toggle>0 → copper, else iron; edge taper
`clampedMap(min(top,bot),0,20,-0.2,0)`; positional rng `nextFloat()>0.7` → none; ridge `-0.08+max(|A|,|B|)>=0` → none;
`nextFloat()<clampedMap(|t|,0.4,0.6,0.1,0.3) && gap>-0.3` → `nextFloat()<0.02 ? raw : ore`; else filler.
The cubiomes evaluation order differs but is equivalent, because the positional RNG is fresh per block.

**VeinTypes (from `cuj$a` + `cdr` Blocks):** IRON = deepslate_iron_ore / raw_iron_block / **tuff**, y **-60..-8**.
COPPER = copper_ore / raw_copper_block / **granite**, y **0..50**.

**Mechanism.** The veinifier is the second `MaterialRuleList` entry in NoiseChunk (`cua`), after the aquifer/density
rule. It runs at noise fill, only for cells that are solid, and **replaces** the default stone *before* surface rules,
carvers and features. So in the real world it **adds** iron ore, raw blocks and tuff/granite filler that no feature
predicts. It consumes no chunk RNG, so it can't desync anything. Later ore features can still overwrite vein filler,
because tuff ∈ deepslate_ore_replaceables and granite ∈ stone_ore_replaceables. Vein ore blocks aren't replaceable, so
a feature cell landing on one stays iron. That keeps cubiomes a superset for features.

**Measured (deep band, buried cells, 441 chunks):**

| | real | missed by features | explained by vein: cubiomes direct | interpolated |
|---|---|---|---|---|
| tuff | 491222 | 3550 | 2824 | **3547** |
| iron | 9737 | 989 | 623 | **761** (other 228 = gate desync) |
| granite y0..50 | 368808 | 11839 | 8499 | 10734 |
| copper y0..50 | 29901 | 5885 | 2589 | 3344 (rest = copper gate desync) |
| raw_iron / raw_copper | 15 / 70 | — | 13 / 63 | **15 / 70** |

Where the interpolated prediction disagrees with the real block, the real block is air/water/lava (caves), clay,
gravel, geode calcite/basalt, or a later redstone feature, **never plain deepslate**. The direct (cubiomes) prediction
puts 337 tuff and 65 iron on plain deepslate. Interpolated iron precision is 95% overall, 100% on undisturbed deepslate.
The interpolation order used (y, then x, then z, as in vanilla NoiseInterpolator) is in `vein_probe.c` `interp()`.

**Implications.**
- Tuff: the "bit-exact" claim holds against cubiomes but has holes against the real world wherever an iron vein sits
  at y -60..-8. That's ~0.7% of deep tuff, but locally dense: a room cut through a vein loses all of that tuff as
  presence at the true location. It's presence-only, so it never causes absence hits.
- Granite: its vein holes are at y 0..50, outside today's -64..-1 band. They matter only if the band is ever raised.
- The research-log Tier-2 note ("`getOreVeinBlockAt`, already in cubiomes, NOT a patch") is wrong in practice. It
  needs the interpolation shown here.

**Port of #1.** For each chunk, 5×5 xz corners × 9 y levels (y -64..0) × 3 interpolated noises × 2 Perlin octaves
≈ 1350 Perlin evaluations, plus gap noise and `xAtPos` for the few cells that survive. Skip a cell whenever max
|toggle corner| + taper < 0.4. That bound is safe because a lerp can't exceed its corner maximum, and it rejects most
cells. On the CPU: add a `veins` family to `region_dump.c` (it already links cubiomes' `sampleDoublePerlin`/`xAtPos`)
that emits tuff/iron/raw_iron in the band, and map it into refine's `candidates[F_TUFF]`/`[F_IRON]`. On the GPU: write
vein tuff into the tuff occupancy in kGenerate, so pass 1 also stops losing it.

## Reproduce
```sh
gcc -O2 -Icubiomes research/surface-gate-probe/vein_probe.c cubiomes/build/libcubiomes_static.a -lm \
    -o research/surface-gate-probe/vein_probe.exe
research/surface-gate-probe/vein_probe.exe 123 -12 12 -12 12 > research/surface-gate-probe/probe.csv   # ~35 s
py -3 research/surface-gate-probe/analyze.py                                                          # ~10 s
```
Caveats: one seed, one 21×21-chunk area (spawn, varied terrain). The real height is a final-world heuristic, not the
feature-time OCEAN_FLOOR_WG, so read the gate-disagreement counts as estimates. The miss, ghost and vein numbers are exact.
