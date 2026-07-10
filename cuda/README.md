# cuda/ — GPU ore-pattern localizer (P3)

Ports cubiomes' 1.18+ Tier-1 ore generation + the recall-safe two-stage matcher (matcher/solve.py)
to CUDA for world-region search. Known seed in, observed ore pattern in, ranked world locations out.

## Files
- `oregen.h` — portable (host + `__device__`) port of cubiomes ore-gen (rng.h/finders.c). No VLAs / GCC
  builtins so it compiles under MSVC + nvcc. GPU-ACTIVE families (bit-exact w/o surface noise):
  **tuff, redstone, lapis, granite**. Gravel/copper/iron are INJECT-ONLY (region_dump at refine): they
  need the mapApproxHeight surface gate for GPU gen, and iron also has the un-ported ore-vein noise.
  Diamond/gold/coal are EXCLUDED — `discardChanceOnAirExposure > 0` couples them to terrain (see below).
- `oretest.c` — CPU driver mirroring `harness/region_dump.exe` output, for the bit-exact diff test.
- `matcher.cu` — the GPU matcher. Generation (default): **two-kernel** — `kSetup` (1 thread per
  (chunk,config), config-major; RNG node lists to scratch) + `kFill` (persistent warps, 1 warp per
  vein; parallel containment cull + division-free sphere fill across 32 lanes → per-family occupancy
  bitmask + anchor list). `--legacy-gen` selects the original 1-thread-per-chunk `kGenerate` (bit-exact
  reference). `kScore`: per (anchor candidate × 8 orientations) presence + soft absence. Mirrors solve.py.

## Build (from a VS BuildTools x64 dev shell, i.e. after vcvars64.bat; needs CUDA bin on PATH)
```
cl /O2 /fp:strict oretest.c /Fe:oretest.exe
nvcc -O2 -arch=sm_89 matcher.cu -o matcher.exe        # sm_89 = RTX 4070 Ti SUPER
```
`rebuild.bat` does both (calls vcvars64 + prepends the CUDA v12.9 bin) for a non-dev shell.

## Validation
1. **Bit-exact generation** (oretest vs region_dump.exe), 64 chunks, deepslate band:
   tuff / redstone / lapis / granite — ZERO position diffs. gravel / copper desync (high Y-range veins
   fail the surface gate in real worldgen; skipping it shifts RNG for later veins). -> deferred.
   ```
   ./oretest.exe 123 0 7 0 7 -64 -1 | tail +2 | sort > port.csv
   ./../harness/region_dump.exe 123 1.18 0 7 0 7 -64 -1 | tail +2 | sort > ref.csv
   diff port.csv ref.csv   # only gravel/copper lines
   ```
2. **End-to-end matcher vs real world** (gt seed 123). All 6 families (gravel/copper injected from
   region_dump.exe; tuff/redstone/lapis/granite GPU-generated):
   - loc1 (true chunk 1,1): rank1 (16,-55,16) 352/352 absHits0, margin 294 — CONFIDENT
   - loc2 (true chunk -5,-5): rank1 (-72,-55,-72) 589/589 absHits0, margin 522 — CONFIDENT
   - gravel adds ~+70 margin (=its cell count); copper ~0 for deep rooms (only generates y>-18).
   - 4-family GPU-only (no injection) also localizes: loc1 223, loc2 447; margin stays ~flat as region
     grows 31x (region13=223 .. region80/25921ch=201) -> world-unique. GPU-only ~0.3-1.1s; with CPU
     gravel/copper injection ~5s (single-threaded region_dump over the region).

## Which ore families, and why (the three "air dependencies")
Real worldgen couples ore to terrain three independent ways; only the first leaves a family matchable:
1. **Replaceable/air intersection** (ALL families): cubiomes fills every sphere cell; real MC only
   stone-replaceable cells. One-directional (cubiomes is a superset), no RNG shift -> deep, mostly-solid
   regions match exactly. This is why the discard-free families bit-match the real `gt/world` deep band.
2. **`discardChanceOnAirExposure > 0`** (diamond, gold, lower-coal): a per-replaceable-cell `nextFloat()`
   roll real MC skips on air/non-stone cells -> the RNG stream desyncs + air-exposed ore is dropped.
   Needs the full 1.18 density-function terrain to replay. Confirmed against the real world (deep band):
   diamond 4/16 miss, gold 6/14 miss. NOT feasible without a terrain port -> excluded. ("Deepslate
   variants" are the same vein below y0, not a separate feature, so they inherit the parent's discard.)
   Coal: upper-coal is discard-0 but y0-127 (never deep); lower-coal is discard-0.5 -> no deep signal.
3. **Post-gen gravity** (gravel): a falling block. Worldgen places it suspended; it drops on the first
   block update (player visit). `gt/world` is freshly generated/unexplored so it can't be measured there
   (deep gravel showed 0 miss) — in a real explored room, air-below (ceiling-facing) gravel will have
   fallen. Gravel is bit-exact at GENERATION (discard 0) but treat its margin as a bonus, not load-bearing.

Iron is **discard-free** (all 1.18 configs 0.0) but contaminated by the separate 1.18 ore-vein noise
system (un-ported) -> ~2/16 deep candidates missed against the real world. Injected like gravel/copper;
presence-only so it never penalizes the truth (a missed iron just doesn't score). Measured on the real
carved room (`gt/extract_real.py`, exposes 3 iron): +3 margin (263->266), 3/3 matched, no false hits,
**zero wall-time cost** (region_dump's per-call generator init dominates; +1 family is free). Copper is
clean (0 deep miss). Diamond/gold/coal stay out.

NOTE (bug fixed while adding iron): PASS1 `dOcc` holds only the NGPU=4 generated families, but `kScore`
read `occ[fam*wpf..]` for ANY obs family — so gravel/copper/iron probed out of bounds. Harmless to the
final answer (refine re-scores on the host) but it polluted PASS1 candidate selection. Fixed: partition
GPU families to the front of the obs, bound the presence loop + `minfrac` threshold to them; gravel/
copper/iron are scored only at refine.

## World-scale (P3.3)
Tiled so memory is bounded for ANY region size (validated to 30M chunks; ~7 min projected for 300k x 300k
with mixed-precision gen + Morton-sorted kScore, vs ~9 min mixed-only, ~18 min FP64 two-kernel, ~48 min legacy).
- PASS 1 (GPU, per tile): generate 4 bit-exact families -> reused occupancy bitmask; enumerate anchor
  candidates in the tile interior; score presence + soft absence with an aggressive presence pre-filter
  (`--minfrac`, default 0.5) and per-cell early-termination. Survivors merge into a global top-K.
- PASS 2 (CPU refine): re-score the top-K with all 7 families (gravel/copper/iron from region_dump.exe). Pruned
  to only contenders within `gravelMax` of the top (gravel can add at most its cell count) -> ~few windows.
- Perf (RTX 4070 Ti SUPER, mixed-precision gen + Morton-sorted kScore): refine fixed ~8s. 30M chunks:
  gen 27.3s + score 7.2s = 34.5s wall (was 92s at the start of this opt arc). Generation is ~80% of wall,
  kScore ~20% (Morton anchor sort cut it ~2.1x). Tile >320 risks a Windows TDR kill -> clamped.

## Generator optimization (two-kernel 2.8x, then mixed-precision for ~6x total over legacy)
Profiled with `ncu` (needs GPU perf counters enabled: NVIDIA CP -> Developer -> Manage GPU Performance
Counters -> allow all users). The 1-thread-per-chunk `kGenerate` is 92% of GPU time and ran the sphere
fill at ~5/32 active lanes (load imbalance: a warp holds 32 chunks of wildly different vein sizes).
Findings, in order:
1. Reducing per-thread local memory (drop bitSet) did NOTHING — not memory-bound. (reverted)
2. Warp-cooperative single kernel BACKFIRED — serializing RNG+cull on lane 0 cost more than the fill win.
3. Two-kernel split (kSetup serial-per-vein at full lane use + kFill cooperative): fixed divergence ->
   1.3x. Bottleneck moved to the O(size^2) overlap cull (FP64).
4. Cull is occ-INVARIANT (it only removes spheres contained in others; bitSet dedups the rest) -> made
   it an approximate windowed parallel cull in kFill. Modest.
5. **The real floor was FP64 division in the fill.** Testing `(d/offset)^2>=1` as `d^2>=offset^2`
   (precompute offset^2) kills 3 FP64 divisions/cell. kGenerate 174s -> 76.7s. Consumer Ada FP64 is
   1/64 rate and division is multi-cycle; this was the dominant cost. Occ stays equivalent (validated:
   identical full-pipeline ranking vs legacy, true loc 589/589). The div-form is "occ-approximate" at
   FP rounding boundaries; `--legacy-gen` is the guaranteed bit-exact path if ever needed.
6. **DFMA fusion in the fill (~9% more).** Reprofile (per-kernel): kFill is now ~85% of generation
   (kSetup 3%, kScore the rest). kFill is FP64-pipeline-bound (85% FP64 util) BUT mostly *stalled*
   (IPC 0.38, 90% no-eligible-warp, 68% of stalls = shared-mem "short scoreboard" from the per-cell
   bitSet atomicOr); occupancy capped at 50% by 14.85 KB smem/block; memory is a non-issue (2.5%).
   The per-term early-out branches (`q+=dy*dy; if(q>=off2)continue;`) blocked nvcc from contracting
   into double-FMA. Computing `q = dx*dx+dy*dy+dz*dz` in one expression (2 DFMA + 1 mul, no branch)
   warm 177 -> 161 ms/tile (~9%). Less than ncu's ~40%-of-FP64 ceiling because the kernel is
   stall-bound, not purely compute-bound. Ranking byte-identical to legacy (632/632 margin 449).
7. **Occupancy lift (~2.5% more, much leaner kernel).** Stopped caching the node list in shared
   (`sNode`, 2 KB/warp): the fill already hoists its node to registers, and the cull reads neighbors
   straight from global `nodes[]` in double (L1-resident, ~2KB/vein, bit-exact — same values, no
   precision change so no over-cull). Smem 14.85->6.66 KB/block and regs 62->48 -> occupancy 50%->~83%.
   warm 161 -> 157 ms/tile. Small because kFill is bound by the per-warp dependency chain (bitSet
   atomicOr -> read-old -> branch per cell), not warp supply -> more occupancy can't hide a serial
   chain. Ranking still byte-identical to legacy (632/632 margin 449).
8. **THE SECOND BIG ONE: mixed-precision distance test (kFill 157->67 ms, 2.3x; 30M gen 76.7->28.3s,
   2.7x).** Diagnostic (all-FP32 test, timing-only) showed kFill would run at 53 ms vs 157 — i.e. FP64 is
   ~2/3 of the kernel. Captured it bit-exactly: reduce coords to the vein-local origin (`x-startX`, small
   + exact by Sterbenz even at world scale) so the FP32 path has no catastrophic cancellation; classify
   every cell in FP32 with a margin EPS=1e-2; only the thin boundary shell (within EPS, ~0.1% of cells)
   recomputes in FP64 using the original world-scale double. EPS is ~75x the max FP32 error (~1.3e-4), so
   NO cell is ever misclassified -> occ[] is provably IDENTICAL to all-FP64 (not merely ranking-equivalent).
   Validated: top-3 byte-identical to legacy and stable across runs; the deep-rank tail wobble is
   pre-existing kScore atomic-race noise (legacy wobbles run-to-run identically), not from this change.
9. **kScore Morton anchor sort (score 15->7.2s at 30M, 2.1x).** kScore probes occ[] at the oriented obs
   footprint per (anchor x orient); anchors arrived in scrambled generation order, so adjacent threads hit
   unrelated occ regions (L2 hit 87%, 74% no-eligible warps, latency-bound). Morton-sorting anchors by
   tile-local (x,z) before kScore (thrust::sort_by_key, ~1ms/tile, counted in score time) makes adjacent
   threads probe overlapping footprints -> L2 hit 87->97.6%, no-eligible 74->45%, issue rate 2.1x. Ranking
   unchanged (survivor SET is order-independent; host re-sorts by fin). occ[] layout is z-fastest.

## Usage
```
matcher.exe <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv>
            [--error E] [--absw W] [--minfrac F] [--tile T] [--topk K] [--refine N] [--no-refine] [--legacy-gen]
# --legacy-gen : use the original 1-thread-per-chunk generator (bit-exact reference; ~2.8x slower)
# obs.csv = OBSERVATION_FORMAT.md (family,x,y,z incl. bare). All 7 families used (gravel/copper/iron at refine).
# --minfrac : presence pre-filter, fraction of N (default 0.5; lower to see runner-ups, raise to prune harder)
# --tile T  : tile size in chunks (default 256; keep <=320 to avoid TDR). --no-refine for pure 4-family GPU.
```

## TODO (optional, perf only — not correctness)
- DONE: kScore Morton anchor sort (was the #2 cost) -> L2 hit 87->97.6%, no-eligible 74->45%, 2.1x.
  Generation (kFill) is again ~80% of wall and at its floor.
- kFill bitSet suppression is an EMPIRICAL DEAD-END (don't re-try): stripping it (timing-only) was a WASH
  (158 vs 157ms) — it's load-bearing, it CUTS global occ[] emit atomics. Removing it just trades shared-
  atomic for global-atomic traffic.
- The per-node __syncwarp (line ~216) is ALSO load-bearing for perf, NOT just dedup ordering (don't re-try):
  removing it + continuous cross-node lane striping was byte-identical (occ is an idempotent union) but 37%
  SLOWER at world scale (kGenerate 26.5->36.3s). Consecutive lerp-line nodes overlap ~80%, and the barrier
  is what lets node i suppress node i+1's overlapping cells BEFORE they emit; without it those cells race the
  bitSet and double-emit to global occ -> more global atomics.
- BATCHED occ write tried (Option D, 2026-06-18, REVERTED — informative null): re-laid bitSet z-fastest +
  wrote occ ONCE per (x,y) column z-run (~9x fewer global atomics) + dropped the per-node barrier (safe here
  since occ isn't touched per-cell). Byte-identical but kGenerate 26.5->26.6s = ZERO change. The global atomic
  stream + barrier are NOT the bottleneck. (D's null does NOT clear the SHARED bitSet atomic, the profiler's
  #1 stall — that needs Option C, below. An earlier note here wrongly bundled C into D's null; corrected.)
- DONE (Option C, 2026-06-18, kFill ~2x: kGenerate 26.5->13.5s @29.2M, BYTE-IDENTICAL): replaced the whole
  per-cell box-iteration + shared bitSet dedup with a COLUMN-ANALYTIC union. Stripe the vein-box (Xr,Yr)
  columns across the warp; each lane owns whole columns, computes each node's z-interval directly (FP32 sqrt
  of the z-slack, +-1 window, exact membership via the same FP32/EPS+FP64-shell predicate), unions them into
  a private 64-bit register mask, and batch-writes the column z-run to occ. Kills the per-cell shared atomicOr
  (profiler's #1 stall, 68%), the per-cell integer-division index decode, AND the bitSet (smem 6656->4352 B,
  regs 52->48 -> more occupancy). GOTCHA that mattered: the node's FP32 box-local params (xrf,yrf,zrf,foff2)
  MUST be precomputed once per node into shared (sNF) — doing the (float)(x-startX) reductions inline made them
  per-(node,column) FP64 subs and the first cut was 33-37s (SLOWER); hoisting them gave the 13.5s. This
  REVERSED the earlier "kFill at its floor / suppression angle closed" conclusion — we were at the floor of
  the per-cell ALGORITHM, not a fundamental one. Validated: --no-refine byte-identical vs --legacy-gen;
  obs_big_room --refine 16 margin 450 unique.
- HARDWARE (full-rate-FP64 A100/H100) would help the residual FP64 but is off the table (consumer only).
- REFINE is the wall-clock bottleneck for the COMMON case (single room, modest region): GPU PASS1 ~0.23s
  but refine ~8s. region_dump.exe is ~6ms/chunk, called once per refined hypothesis over a ~maxExt window.
  DONE: band-skip in region_dump.c drops configs whose [h1-size,h2+size] can't reach the query band — killed
  UpperIronOre (y80..384, repeatCount=90, 100% discarded for deepslate queries): iron 12.3->0.3 ms/ch, full
  run 26->8.4s (3.1x), ranking byte-identical.
- DONE: parallelize refine (OpenMP). Each hypothesis is an independent host-gen + its own region_dump spawn;
  the gravelMax cutoff is monotonic over the fin-descending top-K so the refine count is precomputed (nDo),
  the loop body made dependency-free, and `#pragma omp parallel for schedule(dynamic)` runs the windows
  concurrently (per-thread `gb` scratch; index-write into a pre-sized `outv`, no push_back race). Measured
  5.13x on a 52-hypothesis batch (16 cores); rankings BYTE-IDENTICAL to serial (obs_big_room 632/632 margin
  450). Build adds `-Xcompiler /openmp`. Remaining irreducible refine cost: gravel vein-fill (size 33 x
  repeat 14 — RNG order requires generating the high veins too) and the per-spawn region_dump init.
- GPU surface-gate port (1.18 climate-depth noise) would let gravel/copper/iron be GPU-generated, removing
  the refine region_dump dependency (iron would also need the ore-vein noise). Not needed now: refine is
  bounded to the top contenders.
- DONE: host top-K merge (was ~14% of world-scale wall, 2nd after kFill). The per-tile spatial NMS dedup was
  O(topk^2) (topk=4096 x 484 tiles ~ 10^10 cmps). Replaced the O(K^2) scan with a bucket grid (cell=sep,
  3x3x3 neighbor lookup, hashed key, distance-verified -> byte-identical ranking). 30M-chunk host overhead
  5.1s -> 1.77s; wall 36.1 -> 33.6s. ~40s off a 300k^2 run. What's left of host overhead is the survivor
  memcpy + sort, now small vs kFill.
- kScore memory locality: DONE (P3.5) -- Morton anchor sort took occ L2 hit to 97.6%; near its floor now.
- CAPSULE TWO-PASS: DEAD-END (built + validated byte-identical + measured 2026-06-18, don't re-try). Capsule-fill
  is 1.7x faster than sphere-fill, but scoring against the denser capsule-occ is ~1.7x SLOWER (early-out fires
  later) and the analytic verify is catastrophic (196s @minfrac 0.5) -> SLOWER at every threshold (21s@0.7 vs
  19s exact). occ is the optimal representation: looser=cheaper-to-build but denser=costlier-to-query, and
  querying dominates (reads ~100x fills). NOTES.md P3.15.
- HIERARCHICAL / tile pruning: DEAD-END (measured 2026-06-18, don't re-try). Concentration probe @29.2M:
  only 3.9% of tiles are survivor-free and survivors are SPATIALLY UNIFORM (top10 tiles = 4.7%, max 62 vs
  mean 21). minfrac=0.5 partial-matches are a dense uniform background; the true room separates only at
  refine. No coarse region to prune -> a perfect free rejector saves <=3.9%, a sound one less.
- WORLD-SCALE (post-Option-C audit, 29.2M chunks, --no-refine): kFill 10.3s (50% wall), kScore 6.1s (30%),
  kSetup 2.7s (13%), host ~1.5s (7%) -> wall ~20.6s (was 33.6s). 300k-block run ~7min -> ~4.3min.
  AT THE LIMIT for this algorithm: kFill is at its analytic-union floor (Option C), kScore at its locality
  floor (97.6% L2), no spatial sparsity to exploit. Remaining options are <10% micro-tuning or HARDWARE
  (full-rate-FP64 A100/H100, off the table). The perf arc is DONE.
