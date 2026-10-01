# GPU optimization log

How the GPU matcher in [`cuda/`](../cuda) got from ~48 min to ~4 min for a 300k × 300k-block search,
including the measured dead ends. Kept as history; the current design is summarized in
[`cuda/README.md`](../cuda/README.md). Per-phase context lives in [`research-log.md`](research-log.md).

The log uses the names the code had at the time. Since the 2026-10 readability refactor: `kSetup` →
`kSetupVeins`, `kFill` → `kFillVeins`, `kGenerate` → `kGenerateLegacy`, `kAnchorKey` → `kMortonKeys`,
`kScore` → `kScoreHypotheses`, `occ` → the occupancy grid (`OccupancyGrid`), `bitSet` → `seen`.

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

## Further work and dead ends (perf only — not correctness)
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
  querying dominates (reads ~100x fills). research-log.md P3.15.
- HIERARCHICAL / tile pruning: DEAD-END (measured 2026-06-18, don't re-try). Concentration probe @29.2M:
  only 3.9% of tiles are survivor-free and survivors are SPATIALLY UNIFORM (top10 tiles = 4.7%, max 62 vs
  mean 21). minfrac=0.5 partial-matches are a dense uniform background; the true room separates only at
  refine. No coarse region to prune -> a perfect free rejector saves <=3.9%, a sound one less.
- WORLD-SCALE (post-Option-C audit, 29.2M chunks, --no-refine): kFill 10.3s (50% wall), kScore 6.1s (30%),
  kSetup 2.7s (13%), host ~1.5s (7%) -> wall ~20.6s (was 33.6s). 300k-block run ~7min -> ~4.3min.
  AT THE LIMIT for this algorithm: kFill is at its analytic-union floor (Option C), kScore at its locality
  floor (97.6% L2), no spatial sparsity to exploit. Remaining options are <10% micro-tuning or HARDWARE
  (full-rate-FP64 A100/H100, off the table). The perf arc is DONE.
