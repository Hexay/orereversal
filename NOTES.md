# Ore-based location finding in Minecraft — design notes

## Goal
Given a **known world seed** + a set of **observed naturally-spawned ore positions** (e.g. visible in
someone's base/mine), find **where in the world** (chunk X/Z) that ore pattern occurs. CUDA-accelerated
brute-force / signature search over chunk space. Analogous to bedrock cracking, but for location not seed.

## Environment (verified 2026-06-17)
- GPU: RTX 4070, 16 GB, driver supports CUDA 12.9. Good for this.
- **CUDA toolkit NOT installed** (`nvcc` missing) and **no MSVC `cl`**. To build CUDA we'll need:
  NVIDIA CUDA Toolkit + Visual Studio Build Tools (host compiler). Setup task before any kernel work.
- git available.

## How ore gen works (the parts that matter)
Source of truth: xpple/cubiomes fork (`finders.c`, `rng.h`) + decompiled Yarn.

1. **Population seed** (per chunk, from world seed):
   `popSeed = (blockX * a + blockZ * b) ^ worldSeed`
   where `blockX = chunkX<<4`, `blockZ = chunkZ<<4`; `a,b = two nextLongs from worldSeed, each |1 (odd)`.
   `a,b` drawn from **xoroshiro128++ (MC ≥ 1.18)** or **Java LCG (< 1.18)**. Formula identical across versions.
   cubiomes: `getPopulationSeed()` at `finders.c:41`.
2. **Per-feature (decorator) seed**: `setSeed(popSeed + index + 10000*step)`. Ores are **step 6**.
   cubiomes applies this in `generateOres()` (`finders.c:1917`).
3. **Count placement**: N vein-attempts/chunk (diamond 7, iron-upper 90, coal-upper 30, etc.).
4. **Per vein**: random X/Z in chunk (`nextInt(16)`), Y from uniform/trapezoid height provider, then a
   line-segment ellipsoid blob of spheres (angle, length, per-sphere radius via nextFloat/nextDouble).
   cubiomes: `generateBaseOrePosition` (1944), `generateOrePositions` (1962), `generateVeinPart` (2048).

## Critical subtlety: candidate positions vs actually-placed ore
cubiomes ore gen produces **RNG candidate block positions only**. It has explicit TODOs and does NOT:
- check the target block is replaceable (stone/deepslate) — i.e. skips the air/cave/water filter,
- apply per-block air-exposure discard (diamond/coal/gold 0.5, buried lapis 1.0),
- decide stone-vs-deepslate variant (depends on existing block, not Y).
So **real visible ores = a SUBSET of cubiomes candidate positions** (terrain only ever *removes* blocks).

### Key insight that simplifies everything (for KNOWN-seed localization)
Because terrain filtering only removes candidate blocks, the set of actually-visible ore blocks is a
**subset** of the RNG-predicted candidate blocks. So we can match with the test:
  **every observed ore block ⊆ predicted candidate blocks for chunk (cx,cz)**
WITHOUT simulating terrain/noise/carvers at all. Wrong chunks fail because they won't contain the observed
blocks as a subset (given enough observed blocks). This avoids porting the entire worldgen pipeline to GPU
— we only need the RNG + ore-vein geometry, which is exactly what cubiomes already gives us.
(Full terrain sim would only be needed to *predict* what's visible, or to reduce false positives — not to
*confirm* a candidate location.)

## Information theory
- World ±30M blocks → 3.75M chunks/axis → ~1.4e13 chunks → **~44 bits** to localize a unique chunk.
- Rough estimate **~10–20 bits per observed vein** → **~4–8 well-characterized veins, or one fully-mapped
  chunk's ore layout**, to localize uniquely. (Derived estimate, not a canonical figure.)
- Seed: 64-bit world seed; lower 48 bits = "structure seed" (Java Random masks to 48). Not needed for
  known-seed localization.

## Compute feasibility (known-seed brute force)
- Matching one rare ore type (e.g. diamond): per chunk ~7 veins × small expansion = cheap (~1e3–1e4 ops).
- Full world 1.4e13 chunks × ~3e3 ops ≈ 4e16 ops → order ~tens of minutes on a 4070 for full world,
  single ore type; bounded regions (near spawn / known radius) are seconds-to-minutes.
- Strategy: cheap per-chunk signature (vein anchor positions) first-pass filter, then verify survivors
  with full blob subset match. Classic needle-in-haystack reduction.

## Version fork (big decision)
- **1.18–1.21 (xoroshiro128++)**: modern, most relevant. RNG inversion (for seed cracking) is hard/unsolved.
  Known-seed localization is forward-computable and unaffected — works fine.
- **< 1.18 (Java LCG)**: RNG fully invertible; seed-cracking from ores is tractable (LattiCG/TreeCracker
  pattern). Air-exposure discard didn't exist before 1.17.

## Seed-cracking (the harder, optional direction)
Recovering an UNKNOWN seed from ores:
- Needs to invert per-feature RNG from observed ore positions → popSeed → then invert popSeed formula
  (linear/lattice solve mod 2^N) for chunk coords. Coordinate solve is cheap; the RNG inversion front-end
  is the bottleneck.
- 1.16 LCG: tractable, tooled. 1.18+ xoroshiro: no published clean inversion from bounded ore outputs.
- Prior art: SeedcrackerX historically used **emerald ore** as a signal (≤1.17), now deprecated. No known
  tool does ore-based *location* finding — genuine gap.

## Why ores are a worse fingerprint than bedrock (context)
Bedrock = independent per-position bit, terrain-free, fully observable. Ores = sequentially-coupled RNG,
terrain-contaminated, partially hidden (can't tell "no ore" from "hidden ore"). Ore *localization* with a
known seed sidesteps most of this via the subset-match trick above.

## Settled decisions (2026-06-17 interview)
1. **Scope**: known-seed localization only. No seed cracking.
2. **Version**: modern 1.18–1.21 (xoroshiro128++). Overworld.
3. **Input**: screenshot of a carved wall. User hand-transcribes ore positions as offsets from a chosen
   origin. So: relative (Δhorizontal, Δvertical) read fairly well; **depth into wall poorly constrained**;
   **orientation unknown**; **ore type per block is readable from texture** (strong label per observation).
4. **Y level**: not exact. Rough band — "almost always below deepslate", i.e. deepslate layer (~y -64..0).
   Search the vertical band; ores will be deepslate variants.
5. **Scene size**: several veins (≈4–10) across a few ore types. Borderline-sufficient for unique
   localization (~44 bits needed) → expect a small **ranked candidate list**, not always a unique hit.
6. **Orientation**: unknown → brute all ~8 horizontal orientations + mirrors in the matcher.
7. **Search region**: configurable box; default whole world, allow narrowing with a prior.

## Refined matching model (2D wall → 3D)
- Observed data = typed points {(h_i, y_i, type_i)} on a near-plane; depth free; orientation 1-of-8; ±1
  block read error; y absolute only to within the deepslate band.
- Match test = find region-window + translation + orientation s.t. observed typed points map onto chunk's
  predicted candidate blocks. Subset property survives projection (visible ⊆ predicted 3D candidates).
- **Walls cross chunk borders** → search unit is a sliding multi-chunk window, not a single chunk.
- Use **geometric hashing on relative vectors between distinctively-typed blocks** (rare-type pairs first,
  e.g. diamond↔diamond) for the first-pass filter; verify survivors with full typed subset/score match.
- Must be **error-tolerant** (soft score / allow k mismatches), NOT strict subset: a single mis-read or
  mis-typed block must not reject the true location. This is the main accuracy lever.

## Deepslate-band implications
- Ores observed are deepslate variants (y ≲ 0). Diamond has multiple passes (large trapezoid + buried
  uniform -64..-4 with discard=1.0); use correct per-version OreConfigs (cubiomes has them).
- We do NOT simulate the stone↔deepslate boundary or discard/air filters — subset-match tolerates that
  predicted ⊇ visible. The risk is only the reverse (mis-read rejecting a true chunk) → handled by soft scoring.

## #1 validation requirement (do this before trusting any match)
cubiomes' xoroshiro path may not be bit-perfect vs Mojang (see MC-239059). **Validate cubiomes ore output
against ground truth** (real MC world / authoritative reference) for a known seed+chunk BEFORE building the
matcher. If vein positions don't match exactly, everything downstream is worthless.

## Proposed phased plan (no code yet — for discussion)
- **P0 Setup**: install CUDA Toolkit + VS Build Tools; clone cubiomes; build it CPU-side.
- **P1 Validation harness (CPU)**: generate ores for a known seed/chunk via cubiomes; compare to ground
  truth (real MC export or Chunkbase/orefinder cross-check). Lock down exact xoroshiro + vein geometry.
- **P2 CPU reference matcher**: implement the 2D-wall → candidate subset/score matcher on CPU, small region.
  Prove it localizes a synthetic screenshot (generate ores, project to a fake wall, recover the location).
- **P3 CUDA candidate generation**: port RNG + ore-vein geometry to a kernel; one thread per (window, orient).
- **P4 CUDA matcher + ranking**: full configurable-region search returning ranked candidate locations.
- **P5 Input tooling**: turn a transcribed screenshot into the typed point set (later; manual format first).

## Progress log
- **2026-06-17 P0 done (CPU half).** Toolchain: MinGW-W64 gcc/g++ 13.2.0 (via C:\Strawberry), cmake 4.1,
  mingw32-make present. No MSVC, no CUDA toolkit. cubiomes cloned to `cubiomes/` (HEAD 62007b8), built
  `cubiomes/build/libcubiomes_static.a`. **cubiomes CMake hard-errors on MSVC** → CUDA (P3) must be a
  *port* of the ore-gen subset, with this CPU build as the reference oracle (not linkable into nvcc/MSVC).
- **2026-06-17 P1 harness done.** `harness/ore_dump.c` → `harness/ore_dump.exe`. Build via `harness/build.sh`.
  Usage: `ore_dump <seed> <version> <chunkX> <chunkZ> [oreName...]` | `--list`. Emits CSV `ore,index,x,y,z`
  (sorted) of candidate blocks; stderr gives per-ore index/step/size/repeat/count. Verified sane on
  seed 123 / 1.21 / chunk(0,0): diamond candidates in deepslate band, blobs spill past chunk edge as expected.
  Confirmed real signatures: `initSurfaceNoise(sn, dim, seed)`, `generateOres` needs a seeded Generator
  (it calls getBiomeAt + isViableOreBiome) AND SurfaceNoise. Pos3List = {Pos3* pos3s; int capacity,size}.
- **CUDA/VS install (P0 other half): DEFERRED, recommend not installing yet.** Disk is 20 GB free / 98%
  full; CUDA Toolkit + VS Build Tools (C++ workload) is ~8–15 GB and needs admin. Not needed until P3.
  Recommend freeing disk first, then install VS Build Tools + CUDA 12.x when we reach kernel work.
- **2026-06-17 P1 ground-truth comparison: DONE — landmark finding below.**
  Ground truth = real 1.18.2 vanilla server (`gt/`, Java 21, seed 123), `/forceload add 0 0 15 15`,
  parsed `world/region/r.0.0.mca` with `anvil-parser2` (`gt/compare.py`). Compared real ores in chunk
  (0,0) against cubiomes candidates over the 3x3 chunk neighborhood (blob spill), per ore family.
  Toolchain note: MC-239059 is faithfully replicated by cubiomes (`xNextIntJ`, rng.h), so the RNG itself
  is fine — confirmed because lapis/redstone match bit-exactly.

### ⚠️ KEY VALIDATION RESULT — cubiomes ore fidelity is split by discard chance
- **Exact match (real ⊆ cubiomes, NO terrain needed): redstone, lapis** (and regular iron/copper modulo
  the separate ore-vein system). These have `discardChanceOnAirExposure == 0` (or `==1`, no roll).
- **DESYNC (real ⊄ cubiomes): diamond (0.5), large-diamond (0.7), gold (0.5), lower-coal (0.5).**
  Root cause: cubiomes stubs the replace-block check to `if(1)` (finders.c:2124), so it rolls the discard
  `nextFloat` (line 2130) on EVERY blob cell. The real game (`OreFeature.canPlaceOre`) rolls it ONLY on
  cells whose existing block is stone/deepslate. Where a blob overlaps air/cave, cubiomes consumes an
  extra `nextFloat` and that ore type's stream desyncs for the rest of the chunk. **Terrain-dependent.**
- Per-ore-type streams are independent (`popSeed + index + 10000*step`), so a desync in diamond does NOT
  affect redstone/lapis — verified empirically (diamond 9/13, redstone 41/41, lapis 17/17).

### Consequence for the project (REVISES the core simplification)
The "match without simulating terrain" shortcut is SOUND only for discard-free ores. For diamond/gold/coal
it is UNSOUND with stock cubiomes — real ore blocks appear where cubiomes predicted none.
**Decision: fingerprint primarily on REDSTONE + LAPIS** (both abundant on deepslate walls, discard-free,
bit-exact, no vein-system contamination). Treat diamond/gold/coal as weak, error-tolerant corroboration
only. Options if diamonds become essential later:
  (B) port the density/aquifer/carver "is-this-cell-stone" test so the discard roll fires exactly as
      in-game — this is the partial-terrain sim we hoped to avoid (noise router + carvers, not full gen);
  (C) add cubiomes' ore-vein system (`getOreVeinBlockAt`) if we want iron/copper too.
This must be reflected in P2 (matcher) and P3/P4 (kernel): the candidate generator is trustworthy as-is
for redstone/lapis; anything else needs the stone-cell test first.

### Two-tier signal architecture (decided 2026-06-17)
- **Tier 1 — wide first-pass filter (cheap, sparse RNG points, bit-exact, no terrain):**
  `tuff`, `deepslate_redstone_ore`, `gravel`, `granite`, `deepslate_lapis_ore`, `deepslate_copper_ore`
  (feature ores, discard 0/1). This narrows the whole search box to a handful of candidate locations.
- **Tier 2 — verifier on survivors only (heavier):** the **iron ore-vein system**
  (`initOreVeinNoise` + `getOreVeinBlockAt`, already in cubiomes, NOT a patch). It is NOISE/per-position
  based → position-independent → cannot desync → `real ⊆ cubiomes` holds with no terrain. But it samples
  3D Perlin over a volume (~13k evals/chunk), far costlier than sparse features, so it runs only on the
  few Tier-1 survivors to confirm/rank — never in the brute-force pass. Iron veins sit at y −60..−8 (our
  band) and are big/distinctive → good discriminator. Side benefit: vein filler is TUFF (copper→GRANITE),
  so modeling veins also tightens the tuff/granite families in vein-bearing chunks.
- Iron-vein output still needs its own P1-style ground-truth validation before we trust the verifier.
- Diamond/gold/coal remain excluded unless we later add a partial terrain (stone-cell) sim.

## P2 — synthetic round-trip matcher (2026-06-17): CONCEPT PROVEN, with caveats
Files: `harness/region_dump.c` (bulk Tier-1 candidate gen over a chunk box) + `matcher/matcher.py`
(secret chunk -> expose ores on a 2D plane, drop depth/abs-pos/orientation, optional ±1 noise ->
anchor-alignment match over all 4 orientations -> recover location). Region 25x25 chunks, seed 123, 1.18.

Matcher = pick rarest observed family as anchor; for each orientation x each region candidate of that
family the rigid transform is fully determined (translation + discrete orientation, NO rotation/scale),
so just verify the rest land on candidates. Evaluate recovery by NEAREST recovered origin (noise shifts
the single-anchor alignment), report margin = best_score - best_score_far_from_truth.

Results (margin = separation between truth and best wrong location):
- full wall (16x60, ~111 pts) noiseless: best 111, competitor 77, **margin 34** ✅
- full wall + ±1 reading noise: best 108, competitor 83, **margin 25** ✅
- half wall (~85 pts): margin 6 ✅ (thin)
- rare-ores-only (~5 pts) noiseless margin 2; + noise margin 0 ❌

### What P2 established
1. The localization MECHANISM works: a single carved deepslate wall recovers the exact chunk, including
   orientation, with no terrain sim — even with ±1 reading noise — given a reasonably sized wall.
2. **Dense ores (tuff/gravel) are large contiguous blobs (size-64) → shift-tolerant → weak discriminators.**
   A wrong placement still matches ~70% of points by sitting inside the same blobs. The real margin is
   carried by SPARSE ores (lapis/copper/granite/redstone) + blob-edge structure. So an independence-based
   FP model is over-optimistic; the empirical far-competitor is the honest signal.
3. Margin scales with wall size / #sparse points. Rare-only or small walls don't localize under noise.
4. Single-anchor alignment propagates the anchor's own noise to all points — needs RANSAC/consensus
   (vote in origin space) or tol >= 2x reading noise. Worked here via nearest-origin eval; harden later.

### Open before world-scale claims (next decisions)
- WORLD uniqueness not yet proven: margin measured in a 625-chunk region. Need (a) bigger-region
  extrapolation of far-competitor growth, and/or (b) rarity-weighted scoring that down-weights shift-
  tolerant dense blobs so margin reflects true information content.
- Model a carved ROOM (3D volume, ores on all faces) vs a single thin wall — a room gives far more sparse
  discriminators and should localize much more strongly. Worth comparing.
- Harden matcher to RANSAC/Hough origin-voting for robust noise handling.

## P2b — carved-room 3D matcher + world-scale uniqueness (2026-06-17)
File: `matcher/matcher_room.py`. Room = carved air box; observed = ore candidates on its 1-block inner
shell -> full 3D relative typed points (depth known). Orientation unknown -> 8 (4 rotations x mirror).
World-FP estimate = 8 * occ[anchor]*world_cells * PROD(p_hit over OTHER sparse points); <1 => world-unique.
(Dense tuff/gravel excluded from the FP product — correlated blobs can't establish uniqueness.)

Results (seed 123, 1.18, deepslate band):
- room 12^3, happened to expose 9 redstone (10 pts): world_FP 5.8e-6 -> UNIQUE.
- room 20^3 in an ore-poor pocket: 202 pts but ALL tuff/gravel, ~0 sparse -> world_FP 3e16 -> NOT unique.
- room 32^3 + 1-block noise: 654 pts, 24 sparse (23 redstone +1 lapis): margin 247, world_FP 1e-12 -> UNIQUE.

### Conclusions (the project's viability verdict)
1. **World-scale localization of a carved room is feasible** and the criterion is simple: enough SPARSE
   ore observations (lapis/redstone/copper/granite). Tuff/gravel quantity is irrelevant to uniqueness.
2. Rough budget: ~6-8 sparse points (exact) or ~15+ (with +-1 reading noise) -> world-unique. A room in a
   rare-ore-poor pocket may be UN-localizable regardless of size; user must expose enough surface to catch
   several rare veins.
3. **CAVEAT (next refinement):** the world-FP model treats sparse points as independent, but sparse ores
   also cluster into VEINS (redstone ~size-8 blobs). The true independent unit is the sparse VEIN and its
   relative arrangement, not the block. So the block-based FP is OPTIMISTIC — e.g. the 9 redstone in room
   12^3 are likely 1-2 veins, so that case is weaker than 5.8e-6 suggests. Next: count sparse VEINS (and
   match on inter-vein geometry), and add bigger-region empirical extrapolation. The room-32 margin (247)
   is empirically robust regardless.

## P2c — vein-level uniqueness (2026-06-17): PIVOTAL FINDING — noise is the whole game
File: `matcher/matcher_vein.py`. Clustered sparse candidates into veins (connected components), matched on
vein centroids (anchor-align + verify others within tolerance VTOL=4), swept region size R=3..18.

Result for the 32^3 room: 24 sparse blocks collapsed to just **8 sparse veins (7 redstone + 1 lapis)**, and
**margin = 0 at EVERY region size** (a wrong location matches all 8 veins even in 49 chunks) -> NOT world-
unique at vein resolution. This flatly contradicts the block-level margin of 247.

### Reconciling the two — the real picture
- The block-level margin (247, noiseless) was REAL but came from matching the EXACT fine structure of every
  ore (tuff/gravel/redstone blocks) at tol 0. Exact data IS world-unique (random placement matches only
  ~5/111 blocks; reaching all is astronomically unlikely).
- The vein-centroid abstraction THREW AWAY that fine structure, and what remains — the arrangement of ~8
  veins, mostly one family — is NOT unique, because deepslate sparse ores are common at the VEIN level
  (redstone ~4 veins/chunk, lapis ~1-2/chunk). Their arrangement at expected density carries little info.
- **Therefore uniqueness lives in the EXACT block fine-structure, and survives only as long as we can match
  near-exactly. Reading NOISE is the dominant risk:** any +-1 misread forces a match tolerance, and tolerance
  on the dense ores (27-cell ball vs ~8% fill ~ 86% random-hit) collapses the margin. Sparse-only FP under
  noise also fails to scale (8 x occ[anchor] x world_cells x PROD(27*occ_sparse) >> 1).

### Verdict (honest)
- **Exact transcription (no +-1 error) -> world-unique with modest data** (a clear wall/room). Viable.
- **Noisy transcription -> NOT unique** without a large, family-DIVERSE excavation; deepslate sparse ores
  alone are too common at vein level.
- The project's success hinges on **TRANSCRIPTION PRECISION**, not data volume. The lever is a good input
  method that pins each ore to an exact relative grid coordinate (a P5 concern), plus a noise-robust
  fine-structure matcher (RANSAC/ICP over blocks that tolerates small jitter WITHOUT the blanket dense-ore
  tolerance). Vein-centroid matching is the wrong abstraction — it discards the discriminating detail.

## P2d — precision budget (2026-06-17): WORLD-SCALE VIABILITY PROVEN for a large room
File: `matcher/matcher_budget.py`. Noise-robust matcher = Hough voting (every sparse point votes on the
(orientation,translation) hypothesis -> no single-anchor bias; coarse bins of 2e+1 absorb noise), then
SCORE top hypotheses with the FULL ore set (all families, incl. dense) at per-block tolerance = reading
error e. Big room = ~650 exposed ore blocks, 24 sparse.

WORLD-SCALE EXTRAPOLATION — does best_wrong climb toward truth as the search region grows?
  e=0:  truth 654, best_wrong 190 -> 188 across 169..3249 chunks (FLAT). margin ~465.
  e=1:  truth 654, best_wrong 336 -> 301 across 169..3249 chunks (FLAT/falling). margin ~320-353.
  e=2:  truth 596 (direct score), best_wrong 356 @3249 chunks. margin ~240.
**best_wrong is FLAT as region grows 19x** -> the false-positive ceiling is set by dense-ore tolerance
coincidence and does NOT scale with search size, while truth stays pinned at top. => WORLD-UNIQUE at
+-0, +-1, AND +-2 reading error for a large room.

### This RESOLVES the P2c pessimism (and corrects it)
Uniqueness lives in the EXACT fine structure of the FULL ore set matched at TIGHT per-block tolerance —
NOT vein centroids (P2c discarded it -> false margin 0) and NOT sparse-only. The dense tuff/gravel DO
contribute when matched tightly: the best_wrong "floor" is ~29% (e0) rising to ~54% (e2); the remaining
margin is earned by the true alignment of all 654 blocks incl. the 24 sparse veins, which a wrong location
cannot fake even over 3249 chunks.

### FINAL precision/data budget
- VIABLE: a sizable carved room (~hundreds of exposed ore blocks incl. ~20+ sparse: lapis/redstone/copper/
  granite), reading error up to +-2 blocks. World-unique with stable margin.
- FAILS: tiny excavations (~7 blocks), and ore-poor pockets (a 20^3 room here exposed 0 sparse -> cannot
  localize). Localizability is location-dependent; user must expose enough surface to catch sparse veins.
- Matcher caveat: the vote-ranking can DROP the true hypothesis at high noise/large region (e2 artifact).
  P3+ matcher must retain/seed the true-orientation hypotheses robustly (bigger topk, better voting, or
  ICP refinement). Discriminating power is proven; the search must not lose the needle.

## P2e — recall-safe matcher (2026-06-17): DONE
File: `matcher/matcher_robust.py`. Fix for the dropped-hypothesis artifact = anchor ENUMERATION: every
region candidate of the rarest observed family is a hypothesis (true anchor always enumerated -> cannot be
dropped), full-scored against DILATED candidate sets (radius 2e -> O(1) tolerance membership, absorbs
single-anchor + per-point reading noise). Results: e=0 over 729 chunks -> 101544 hyps, best 654/654, TRUTH
RECOVERED; e=1 over 121 chunks -> best 654/654, TRUTH RECOVERED. Recall problem solved.

Performance note (the real lesson): robust matching at +-2 over WORLD-size regions is compute-bound in
Python — redstone is so dense that within +-2 nearly every cell is "near" a redstone, so per-point pruning
saturates and discrimination lives only in the JOINT 654-point match. That joint full-score over billions
of chunk hypotheses is exactly the workload for the GPU (P3). CPU reference is correct and recall-safe;
scale is the GPU's job.

## P5 — observation format + solver consumer (2026-06-17): DONE (end-to-end)
Input is EXACT 3D block data (user will extract via world-copy/mod later, not noisy screenshots) -> the
strong case: exact coords, real 3D, all usable families incl. tuff. My side = the consumer:
- `matcher/OBSERVATION_FORMAT.md` — CSV `family,x,y,z` (relative coords ok; orientation brute-forced;
  y=vertical; usable families only: tuff/redstone/lapis/gravel/granite/copper).
- `matcher/make_observation.py` — generates example/test observation from a known location (also the spec
  the extraction mod should emit). Example: `matcher/examples/obs_big_room.csv`.
- `matcher/solve.py` — recall-safe matcher (anchor-enumeration + dilated O(1) scoring, 8 orientations) ->
  ranked candidate world locations + confidence verdict. `--error N` for +-N tolerance, `--region R` box.
END-TO-END TEST: generated 654-block obs at secret origin (-6,-52,-6) chunk (-1,-1); solve.py blind-recovered
rank1 = (-6,-52,-6) chunk (-1,-1) score 654/654, next distinct 302, margin 265 -> CONFIDENT (unique). PASS.
(Minor polish TODO: tighten near-truth dedup; rank2 was a +-2 shift of truth, same chunk.)

## P0 toolchain — DONE (2026-06-17). CUDA build verified end-to-end.
GPU: **RTX 4070 Ti SUPER (sm_89, Ada), 16 GB**. Disk: 232 GB free (earlier "20 GB" was a df artifact).
Installed: **CUDA Toolkit 12.9** (nvcc V12.9.86, matches driver 576.x) + **VS2022 BuildTools** with C++
workload (**MSVC 14.44.35207** + **Windows SDK 10.0.26100**). Note: the VS *Community* instance's modify
kept failing (exit 87 / channel-manifest errors + pending file-renames from CUDA) — used standalone
BuildTools instead. nvcc/cl are NOT on the default PATH.

**Build recipe (P3 must use this — vcvars sets the MSVC host env nvcc needs):**
```
vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
cmd /c "<vcvars> && <nvcc> -arch=sm_89 file.cu -o out.exe"
```
Verified: `cuda_test/hello.cu` compiles+runs a kernel -> "kernel returned 1337 | no error | RTX 4070 Ti SUPER".
(Setup scripts kept at repo root: setup_toolchain.ps1, vs_cpp.ps1, buildtools.ps1; logs _*.log.)

## P5b — soft absence (option 2), 2026-06-17: DONE
Decision: observation includes the carved/exposed volume, not just ore cells. Format now has `bare` rows
(exposed cells confirmed to have NO usable ore). Matcher = TWO-STAGE, recall-safe:
- stage 1 presence: anchor-enumeration + ore-cell match vs dilated candidates (true loc always has MAX
  presence -> never dropped), keep top-K survivors.
- stage 2 soft absence: penalize survivors for ores predicted on `bare` cells. final = presence - w*absHits.
  Soft (penalty, not veto) because a few true-location candidates can be legitimately absent (cave/air).
Powerful because: bare cells >> ore cells, and it makes the DENSE families (tuff/gravel) strong
discriminators (a wrong loc predicts tuff all over your bare cells). Two-stage keeps it cheap & mirrors GPU.
END-TO-END: big room (632 ore + 3792 bare). True loc final 632/632 absHits 0; best false present 176 absHits 94
final 82. **Margin 265 (presence-only) -> 449 (with absence).** Confirmed in solve.py/make_observation.py.

Answers to the two pre-port questions:
- Cross-chunk spill / non-chunk-aligned: matching is free block-translation + 8 orientations (never chunk-
  aligned); spill handled by candidate gen over region + 1-chunk margin. GPU port MUST gen all chunks a
  hypothesis footprint touches + 1 margin (not a single chunk).
- Absence: now USED via soft absence (above). Presence already exploited the strong direction (true loc
  must predict all observed ores); absence adds the carved-but-bare direction.

## P6 — REAL-WORLD end-to-end validation (2026-06-17): PREMISE PROVEN ON REAL DATA
All prior matcher results (P2–P5b) were cubiomes-vs-cubiomes (self-consistent by construction). P6 ran the
FULL pipeline on the REAL 1.18.2 gt world (seed 123): read the shell of a carved box from anvil, classify
each real block, run solve.py vs cubiomes candidates. Tool: gt/extract_real.py (emits relative coords +
two labeling policies). Two independent deepslate-band rooms (chunks (1,1) and (-5,-5)):
- Loc 1 (352 ore: tuff/gravel/redstone/lapis): rank-1 = TRUE, present 352/352, absHits 0, margin ~380.
- Loc 2 (589 ore; cut a LAVA/AIR cave): rank-1 = TRUE, present 589/589, margin ~547+.
KEY: present = N/N at both -> discard-free families (tuff/gravel/redstone/lapis) are bit-exact END-TO-END
vs real worldgen, and true-loc absHits = 0 (cubiomes predicts no usable ore on any real stone/deepslate cell).
World-localization premise is PROVEN on real data, not just synthetic.

RESOLVES pre-port question #2 (labeling of non-usable exposed cells) WITH DATA:
- Policy A "everything-else = bare": loc2 true-loc absHits 30 (penalized!) because cubiomes predicts ores
  inside lava/air cells that A wrongly called bare. Still localized here, but a real self-penalty.
- Policy B "only stone/deepslate = bare; omit air/lava/water + gold/coal/diamond/iron + andesite/diorite":
  true-loc absHits 0, margin HIGHER (574 vs 547). ADOPTED. Simpler for the mod (two allowlists, no exception
  list). OBSERVATION_FORMAT.md updated with the emit rule.

## P3 — CUDA port (2026-06-17): GENERATOR BIT-EXACT + GPU MATCHER VALIDATED (see cuda/)
Hand-ported cubiomes 1.18+ ore-gen to portable host+device C (cuda/oregen.h): xoroshiro128++,
getPopulationSeed, the Tier-1 1.18 configs, generateOres/BaseOrePosition/OrePositions/VeinPart. No VLAs/
GCC builtins -> compiles under MSVC + nvcc (the cubiomes CMake forced MinGW; only VLAs blocked MSVC).
- BIT-EXACT vs region_dump.exe over 64 chunks for **tuff, redstone, lapis, granite** (zero position diffs).
- gravel + copper DESYNC: high Y-range (gravel -64..319, copper -16..112) veins fail the mapApproxHeight
  surface gate in real worldgen; cubiomes skips them (no RNG consumed), so generating them raw shifts the
  RNG for later veins and desyncs even deepslate-band output. The 1.18 gate needs the full climate-depth
  noise sampler -> DEFERRED. Decision (user): ship 4 bit-exact families now, add gravel/copper later via a
  CPU-precomputed surface grid the GPU gate looks up. (Biome system omitted entirely: these families are
  isOverworld=always-viable, no RNG; only LargeCopperOre is biome-gated -> excluded.)
- GPU matcher (cuda/matcher.cu): kGenerate (1 thread/chunk +1 margin -> per-family occupancy bitmask over
  region deepslate band + anchor candidate list) then kScore (1 thread per anchor-candidate x 8 orientations
  -> presence + soft absence, cheap presence pre-filter). Mirrors solve.py.
  END-TO-END on REAL world (seed 123, 4 families): loc1 rank1 (16,-55,16) 281/281 absHits0 margin 223;
  loc2 rank1 (-72,-55,-72) 525/525 absHits0 margin 447. Both CONFIDENT. Margin ~flat as region grows 31x
  (region13=223 -> region80(25921 ch)=201) = world-uniqueness reconfirmed on GPU. ~1.1s for 25921 chunks /
  3.13M hypotheses on the 4070 Ti SUPER. (Margins below solve.py's 380/574 because gravel dropped.)
GRAVEL/COPPER (added via injection, not a port): region_dump.exe already gates gravel/copper correctly,
so matcher.cu shells out to it (CPU, bit-exact) and uploads those candidates into occ[gravel],occ[copper];
no climate-noise port needed. RESULT (real world, all 6 families = matches solve.py at the true loc):
loc1 352/352 margin 294 (+71 vs 4-fam, ~= gravel cell count); loc2 589/589 margin 522 (+75). Copper ~0
effect for deep rooms (only generates y>-18). COST: wall 316ms -> 5.1s (single-threaded region_dump over
the region x2); fine for bounded searches. Margin gain ~= gravel count confirms gravel's value.
## P3.3 — WORLD-SCALE tiling (2026-06-17): DONE. Handles 300k x 300k / unbounded.
matcher.cu rewritten to tile the search region so GPU memory is bounded for ANY size. Generation now
writes occupancy DIRECTLY via an emit callback (oregen.h OreEmit) — no per-chunk scratch (that didn't
scale to big tiles). Two passes:
- PASS 1 (GPU, per tile): 4 bit-exact families -> reused occupancy bitmask (one tile in VRAM at a time);
  enumerate anchor candidates in the tile INTERIOR (margin chunks generated for vein spill but not
  enumerated -> no cross-tile double count); score presence+absence with aggressive presence pre-filter
  (--minfrac, default 0.5) + per-cell early-termination (drop once can't reach threshold). Survivors
  compacted via atomicAdd -> global top-K.
- PASS 2 (CPU refine): re-score top-K with all 6 families (gravel/copper from region_dump.exe), PRUNED to
  contenders within gravelMax of the top (gravel adds <= its cell count) -> only ~a few windows.
PERF (RTX 4070 Ti SUPER): GPU pass ~8.3s / 1M chunks (~120k ch/s); refine fixed ~8s. 1M-chunk run: rank-1 =
true loc (-72,-55,-72) 589/589 margin 590, CONFIDENT. Margin GROWS with area (absence crushes far false
positives). 300k^2 (~351.5M chunks) projects to ~48-50 min. tile>320 risks Windows TDR (single kScore
launch >2s) -> clamped to 256.
OPTIONAL LATER (perf only): GPU surface-gate port (removes refine's region_dump dep — not needed, refine
is bounded); kScore occupancy-read locality (L2-latency bound) via spatial anchor sort / smaller tiles.

## P3.4 — generator optimization (2026-06-18): DONE. 2.8x (300k^2 ~48 -> ~18 min).
ncu (GPU perf counters enabled) showed kGenerate = 92% of GPU time, running the sphere fill at ~5/32
active lanes (load imbalance: 1 thread/chunk = 32 different-sized chunks per warp). Iteration:
- bitSet local-mem removal: no effect (NOT memory-bound). reverted.
- warp-cooperative single kernel: BACKFIRED (237->334s) — serializing RNG+cull on lane 0 (which legacy
  does at full lane use) cost more than the fill win.
- TWO-KERNEL (now DEFAULT): kSetup (1 thread/(chunk,config), config-major -> low divergence; RNG node
  lists -> global scratch, ~1GB/tile, bounded) + kFill (persistent warps, 1 warp/vein, fill striped
  across 32 lanes, per-warp shared bitSet for cubiomes collision-suppression). Fixed divergence -> 1.3x.
- Cull is OCC-INVARIANT (kills only spheres CONTAINED in another, containment transitive; bitSet dedups
  the rest) -> moved to a parallel windowed cull in kFill (any window safe; missed nodes just re-filled).
- **THE WIN: FP64 division.** Fill test `(d/offset)^2>=1` rewritten as `d^2>=offset^2` (precompute
  offset^2) -> kills 3 FP64 divisions/cell. kGenerate 174->77s. Consumer Ada FP64 = 1/64 rate and div is
  multi-cycle; this was the floor. Validated: identical full-pipeline ranking vs legacy (rank-1 589/589
  margin 635). div-form is occ-approximate at FP rounding boundaries -> `--legacy-gen` = bit-exact ref.
PERF: 30M chunks 261->92s wall (gen 77s + score 13s). Note the +-1 presence wobble at deep false
positives is PRE-EXISTING in legacy (downstream ranking, not occ), not introduced by the fast path.
- REPROFILE (per-kernel): kFill = ~85% of gen, kSetup 3%, kScore the rest. kFill is FP64-bound (85% util)
  but actually STALL-bound: IPC 0.38, 90% no-eligible-warp, 68% of stalls = shared-mem short-scoreboard
  (per-cell bitSet atomicOr); occ capped 50% by 14.85KB smem/block; mem non-issue (2.5%). DFMA fusion
  (drop per-term early-out branches that blocked nvcc contraction; q=dx*dx+dy*dy+dz*dz one expr) -> warm
  177->161ms/tile (~9%), ranking byte-identical to legacy (632/632 margin 449). Short of ncu's 40%-FP64
  ceiling because stall-bound, not compute-bound.
- OCCUPANCY LEVER (tried): stopped caching nodes in shared (sNode 2KB/warp) — fill hoists to regs, cull
  reads neighbors from global nodes[] in double (L1, bit-exact, no over-cull). smem 14.85->6.66KB, regs
  62->48 -> occ 50%->~83%. Only 161->157ms (~2.5%): kFill is NOT occ-starved, it's bound by the per-warp
  dep chain (bitSet atomicOr->read-old->branch/cell) + FP64; more warps can't hide a serial chain. Session
  177->157ms (~11%) + leaner kernel, ranking identical to legacy throughout.
- MIXED-PRECISION (BIG, 2026-06-18): all-FP32 diagnostic showed kFill would be 53ms vs 157 -> FP64 is ~2/3
  of kFill. Captured bit-exactly: reduce coords to vein-local origin (x-startX small+exact even at world
  scale) -> FP32 classify with margin EPS=1e-2 -> only boundary shell (~0.1% cells) recomputes FP64 with the
  ORIGINAL double. EPS ~75x max FP32 error -> no cell misclassified -> occ PROVABLY identical to all-FP64.
  kFill 157->67ms (2.3x); 30M gen 76.7->28.3s (2.7x); wall 92->43s (2.1x); 300k^2 ~18->~9 min. Validated:
  top-3 identical+stable vs legacy, tail wobble is pre-existing kScore race (legacy wobbles too).
- kScore MORTON ANCHOR SORT (2026-06-18): kScore probes occ[] over the oriented obs footprint per
  (anchor x orient); anchors arrived scrambled -> adjacent threads hit unrelated occ (L2 87%, 74% no-elig).
  Morton-sort anchors by tile-local (x,z) (thrust::sort_by_key, ~1ms/tile, in score time) -> overlapping
  footprints -> L2 87->97.6%, no-elig 74->45%, score 15->7.2s at 30M (2.1x). Ranking unchanged. NET ARC:
  30M wall 92->34.5s (2.67x); gen 76.7->27.3, score ->7.2; 300k^2 ~48min legacy -> ~7 min. kFill ~80% wall
  at floor; biggest remaining lever is HARDWARE (A100/H100 full-rate FP64 1/2 vs consumer Ada 1/64).
- SUPPRESSION-REMOVAL DEAD-END (measured, don't re-try): stripped bitSet atomic entirely (timing-only) ->
  WASH (min 158 vs 157ms). The atomic is LOAD-BEARING: suppressing overlapping/aliased cells CUTS global
  atomicOr emits into occ[]; removing it trades shared-atomic for more global-atomic traffic. No removable
  overhead. kFill is the MEASURED floor for this algorithm; a real win needs a different gen approach.

## P3.5 — family audit + IRON (2026-06-18): DONE. Why diamond/gold/coal stay out; iron added (free).
"Air-dependent" = 3 independent mechanisms, only #1 leaves a family matchable:
1. replaceable/air intersection (ALL families): cubiomes fills every sphere cell, real MC only
   stone-replaceable -> cubiomes SUPERSET, no RNG shift -> deep solid regions match (why discard-free
   families bit-match real gt/world deep band).
2. discardChanceOnAirExposure>0 (diamond/gold/lower-coal): per-replaceable-cell nextFloat() roll real MC
   skips on air cells -> RNG desync + air ore dropped -> needs full 1.18 density-fn terrain. Measured vs
   real gt/world deep band: diamond 4/16 miss, gold 6/14 miss -> EXCLUDED. Coal: upper=discard0 but y0-127
   (never deep), lower=0.5 -> no deep signal. "Deepslate variants" = same vein <y0, inherit parent discard.
3. post-gen gravity (gravel, falling block): suspended at worldgen, drops on first block update ->
   UNMEASURABLE on gt/world (fresh/unexplored, 0 deep miss) but real explored rooms WILL have ceiling-facing
   gravel fallen. Gravel bit-exact at GEN (discard0); treat margin as bonus, not load-bearing.
IRON added: discard-free (all 1.18 cfgs 0.0) but un-ported ore-vein noise -> ~2/16 deep miss; injected like
gravel/copper (presence-only, never penalizes truth). Real carved room (extract_real, 3 iron exposed): +3
margin (263->266), 3/3 matched. Copper clean (0 deep miss, already injected). 1.18 iron = Middle/SmallIronOre
(deep) + UpperIronOre(y80+); IronOre 1.17-only. Files: oregen.h(F_IRON), region_dump.c(TIER1),
matcher.cu(NACTIVE 7, refine), extract_real.py(USABLE).

## P3.6 — refine perf: region_dump band-skip (2026-06-18): DONE. 26->8.4s wall (3.1x).
Rechecked metrics: GPU PASS1 is trivial (kGenerate 24ms, kScore 5ms, no-refine wall 232ms); REFINE is ~99%
of wall (8 region_dump calls, one per refined hypothesis). Per-family profiling exposed the real cost: iron
3567ms/289ch vs tuff 198 — because UpperIronOre has repeatCount=90 at y80..384, generating 90 veins/chunk
that the y[-64..-1] filter then discards 100%. (The earlier "iron ZERO wall cost" was bogus: it compared obs
WITH/WITHOUT iron rows while region_dump requested iron in BOTH legs, hiding the cost.) FIX: region_dump.c
skips any config whose [h1-size,h2+size] misses the query band — independent decorator RNG per config makes
it bit-identical. iron 3567->92ms, full family set 5298->1733ms, full matcher run 26->8.4s, ranking
byte-identical (real room rank-1 true 355/355 margin 367; obs_big_room 632/632 margin 450). Remaining refine
cost = gravel vein-fill (size33 x repeat14, irreducible: RNG order needs the high veins) + N SERIAL
region_dump spawns.

## P3.6b — PARALLELIZE refine (OpenMP) (2026-06-18): DONE, 5.13x. Corrects P3.6's "don't parallelize".
P3.6 declined this because refine is "~2% of a 300k^2 run". That was world-scale tunnel-vision: for the
ACTUAL day-to-day workflow (one carved room, modest search region) refine is ~99% of wall (~8s) while the
whole GPU pass is ~0.23s. The refine loop is embarrassingly parallel — each hypothesis = independent host-gen
of the 4 GPU families + its OWN region_dump _popen spawn + scoring; the only couplings were the push_back into
outv and the gravelMax early-break. The break is monotonic (top is fin-descending, topFin fixed) so the refine
count nDo is precomputed up front -> loop body is dependency-free. FIX: per-thread `gb` scratch (was a shared
`static OrePos[200000]` — not reentrant), index-write `outv[t]` into a pre-sized vector (no push_back race),
`#pragma omp parallel for schedule(dynamic)`. Build: nvcc `-Xcompiler /openmp`. Measured (16 logical cores,
seed 123, search -8..8^2): 2-hyp batch 2.2->1.2s; 52-hyp batch (--minfrac 0.3 --refine 64) 7.66->1.49s =
5.13x. Rankings BYTE-IDENTICAL serial(OMP_NUM_THREADS=1) vs parallel (obs_big_room 632/632 margin 450).
Irreducible floor now = per-spawn region_dump init + gravel vein-fill, hidden behind core count.

## P3.7 — world-scale profiling + host-merge spatial hash (2026-06-18): DONE.
Measured GPU pass on 29.2M chunks (12x smaller than 300k^2, linear): kGenerate 26.5s (73%), kScore 4.5s
(13%), HOST top-K merge ~5.1s (14%), refine fixed (~2% at world scale). Bottleneck = kFill (gen), at its
software floor (stall-bound on the bit-exact bitSet atomicOr + Ada-1/64 residual FP64; suppression-removal
was a wash; mixed-precision already cut 2/3 of FP64). Only kFill levers left = HARDWARE (A100/H100 full-rate
FP64) or a new suppression algorithm. The one tractable SOFTWARE lever was the host merge: per-tile spatial
NMS dedup was O(topk^2) (topk=4096 x 484 tiles ~10^10 cmps). FIXED: bucket-grid dedup (cell=sep, 3x3x3
neighbor lookup, hashed key, distance-verified) -> ranking byte-identical, host overhead 5.1->1.77s, wall
36.1->33.6s (30M), ~40s off a 300k^2 run. matcher.cu top-K merge block (~line 435).
BUG FIXED while adding iron (pre-existing): PASS1 dOcc holds only NGPU=4 families but kScore read occ for
ANY obs family -> gravel/copper/iron OOB (harmless to final answer via refine, but polluted PASS1 selection;
caused a bogus "+72" before fix). FIX: partition GPU families to front, kScore presence loop + minPres bound
to nOreGpu; non-GPU families scored only at refine. Regression: obs_big_room 632/632 margin 450, two-kernel
== --legacy-gen byte-identical.

## P3.8 — kFill node-loop flatten ATTEMPT (2026-06-18): REVERTED, negative result.
Tried the one structural kFill lever left: remove the per-node __syncwarp + continuous cross-node lane
striping ((base+t)%32==lane) to kill the per-node tail + barriers (gravel ~462/vein, tuff ~128/vein).
Hypothesis: bitSet dedup is non-load-bearing (occ is an idempotent union; dup anchors collapse at the host
merge). CORRECTNESS held -- two-kernel == --legacy-gen byte-identical. But it was 37% SLOWER (kGenerate
26.5->36.3s @29.2M). Root cause: consecutive lerp-line nodes overlap ~80%; the barrier lets node i suppress
node i+1's overlapping cells via the shared bitSet BEFORE they emit, so they skip the global occ atomicOr.
Without it they race and double-emit -> ~2x global atomic traffic on overlap cells. kFill is global-atomic-
bound (same root cause as the "stripping suppression is a wash" result), so ANY node-loop flatten that breaks
cross-node suppression fights the actual bottleneck. Reverted to barrier + per-lane striping. DON'T re-try
node-loop restructuring; the remaining kFill levers are genuinely HARDWARE (full-rate FP64) or a fundamentally
different suppression scheme that keeps cross-node dedup without a barrier.

## P3.9 — kFill batched occ write (Option D) ATTEMPT (2026-06-18): REVERTED, decisive null.
Goal: test whether the global occ atomicOr stream is the kFill bottleneck. Re-laid the per-vein bitSet to
z-fastest (occ-aligned), kept the fill/classify untouched, but replaced the per-cell oreEmit (1 global atomic
each) with a post-fill sweep that ORs each (x,y) column's contiguous z-run into occ in <=3 atomics (~9x fewer
global atomics). Also dropped the per-node __syncwarp -- SAFE here (unlike P3.8) because occ is never touched
per-cell, so losing cross-node bitSet suppression can't double-write occ; the bitSet just accumulates the
union via commutative atomicOr. Result: byte-identical vs --legacy-gen, but kGenerate 26.5->26.6s @29.2M =
ZERO change. CONCLUSION (the important part): the atomic streams (shared bitSet + global occ) are NOT the
bottleneck -- they're fully hidden behind the per-cell FP32 distance classify. kFill is ARITHMETIC-bound,
sitting at ~50ms/tile (26.5s/484 tiles, ~94% kFill) vs the ~53ms all-FP32 diagnostic floor from the reprofile
arc. This CLOSES the entire "different suppression scheme" line of attack (incl. the column-analytic Option C,
which would ADD a sqrt per node-column -> more arithmetic, the actual bottleneck). The profiler's "stripping
suppression = wash" wasn't because atomics are free; it's because at the natural emit rate they're already
hidden, and removing suppression just raised them past the hidden threshold. Reverted (no gain, +complexity,
52->63 regs). Only remaining kFill software lever = cut per-cell arithmetic or cells classified (~48% of
bounding-box cells are outside-sphere rejects) -- a different, smaller game; HW is off the table.

## P3.10 — kFill COLUMN-ANALYTIC union (Option C) (2026-06-18): DONE. ~2x (kGenerate 26.5->13.5s @29.2M).
The "are we at the limit" answer was NO. P3.9 (Option D null) only cleared the GLOBAL atomic + barrier; it
LEFT the shared bitSet atomic (profiler's #1 stall, 68%) in place. I'd wrongly bundled Option C into D's null
-- corrected. Option C: replace the entire per-cell box-iteration + shared-bitSet dedup with a column-analytic
union. Stripe the vein-box (Xr,Yr) columns across the warp; each lane owns whole columns, computes each alive
node's z-interval directly (pf = foff2 - dxf^2 - dyf^2 = z-slack; sf=sqrtf(pf); FP32 +-1 window; exact
membership per cell via the SAME FP32/EPS + FP64-shell predicate as before, so it's a superset-window + exact
filter -> provably same set), ORs the intervals into a private uint64 column mask, batch-writes the column
z-run to occ (z-fastest, <=3 atomics), enumerates anchors from the mask. Removes: per-cell shared atomicOr,
per-cell integer-division decode, the bitSet (smem 6656->4352B, regs 52->48 -> more occupancy). GOTCHA: the
node FP32 params (xrf,yrf,zrf,foff2) MUST be hoisted once per node into shared (sNF[]) -- computing
(float)(x-startX) inline made them per-(node,column) FP64 subs and the first cut was 33-37s (SLOWER than
baseline); hoisting -> 13.5s. Validated: --no-refine byte-identical vs --legacy-gen; obs_big_room --refine 16
margin 450 unique; kGenerate stable 13.0-13.9s over 3 runs. World wall 33.6->~21.7s @29.2M; 300k ~7->~4.3min.
LESSON: "at the floor" meant the floor of the per-cell ALGORITHM, not fundamental. The dead-ends (strip
suppression / flatten / batch atomics) all kept the per-cell structure; changing the structure was the win.

## P3.11 — post-Option-C bottleneck audit + hierarchical-pruning probe (2026-06-18): AT THE LIMIT.
Added a permanent kSetup/kFill split to the cudaEvent timing (was bundled as kGenerate). Real split @29.2M
(seed 123, --no-refine): kFill 10.3s (50% wall), kScore 6.1s (30%), kSetup 2.7s (13%), host ~1.5s (7%);
wall 20.6s. kFill is STILL the giant after Option C; my guess that kSetup might dominate was WRONG.
Levers audited & CLOSED:
- kScore (30%): ALREADY optimized -- Morton anchor sort put occ L2 hit at 97.6% (P3.5). Near its floor.
- FUSE generate->score: RETRACTED idea -- dOcc lives in VRAM the whole time, kScore reads it in place.
  There is NO host round-trip to remove; fusion buys nothing.
- HIERARCHICAL / tile pruning: DEAD (measured, don't re-try). One-shot concentration probe over 484 tiles:
  zeroSurv=19 (3.9%), totSurv=10302, top10 tiles hold 4.7%, maxTile=62 vs mean ~21. Survivors are 96%-dense
  and SPATIALLY UNIFORM -> no coarse region to reject. Cause: minfrac=0.5 admits partial matches, which are a
  dense uniform background across a real world; the true room only separates at REFINE (already cheap at
  world scale). A perfect free tile-rejector would save <=3.9%; a SOUND superset pass even less.
- kSetup (13%): FP64 RNG; halving it is ~6% wall against bit-exactness risk -- not worth it alone.
VERDICT: both dominant kernels (kFill, kScore) are at their software floors and there is no spatial sparsity
to exploit. Remaining options are all <10% micro-tuning or HARDWARE (full-rate FP64, off the table). The perf
arc is DONE: wall 33.6->20.6s @29.2M (300k ~7->~4.3min).

## P3.12 — ncu profile of kSetup + kScore (2026-06-18, for a 115-day BLIND GLOBAL search where %=days).
Whole-world (border +-30M blocks = 60M^2 = ~1.4e13 chunks) at measured 1.4M ch/s = ~115 days on 1x 4070 Ti S.
Generation (kSetup+kFill, 63%) is a HARD FLOOR: every chunk must be generated once, no biome/terrain gate, no
RNG periodicity -> ~72 days irreducible on 1 GPU. So kernel work is bounded; only PARALLELISM (multi-GPU/cloud)
or a location prior breaks the order of magnitude. ncu (1 tile, sm_89) overturned the pre-profile guesses:
- kSETUP (13%) is NOT FP64-bound (guess was WRONG, don't chase the cull's FP64): FP64 pipe 21%, Compute(SM) 17%,
  IPC 0.15, issue-slots 3.8%. It is MEMORY/LG-THROTTLE bound -- 61.6% of a 175-cyc stall = "LG instruction queue
  full" (local+global mem ops), L2 81%, DRAM 51%. Cause: 2080-byte per-thread STACK FRAME (local-mem node buffer
  built before the bulk global node-scratch write; 0 spills but the local array IS local-mem traffic) + 56 regs
  capping occupancy at 66.7%. Only lever = shrink the local buffer / cut regs to raise occupancy -> ~1.3-1.5x on
  kSetup = ~4-5 days. Bit-exact-safe (storage reorder, cull is occ-invariant) but real kernel surgery.
- kSCORE (30%) is L2-LATENCY bound, already near floor: Memory 70% vs Compute 47%, DRAM only 4% (so it's L2-
  RESIDENT, 97.6% hit from the Morton sort), 60% of stall = L1TEX/global load scoreboard, occupancy 91%. Smaller
  tiles do NOT help (occ already in L2; the cost is intrinsic L2 latency, not misses). Only lever = cut warp
  divergence (24.7/32 active threads, ~13% est) / probe count from the x8 orientations -> ~10-15% = ~2-3 days.
- kFILL (50%) confirmed at its analytic floor.
NET HONEST CEILING for kernel work on the blind global search: ~1 week off 115 days (~108). Does NOT make blind
global tractable. The real levers remain parallelism (16 GPU ~1wk, ~100 spot-cloud ~1 day for a one-time job)
or a location prior to shrink the region. Recorded so the kSetup-FP64 angle isn't re-tried.

IMPLEMENTED (2026-06-18):
- kSETUP local-buffer removal: DONE, ~18% on kSetup (2688->~2190 ms @29.2M, ~2.5 days at world scale), BYTE-
  IDENTICAL vs --legacy-gen. The `double store[4*ORE_MAXSIZE]` staging array (the 2080-byte stack frame / LG-
  throttle source) was dead weight once the cull moved to kFill: nS==size is known up front, so reserve node
  slots via atomicAdd FIRST and stream each node straight to global (RNG still advances fully on overflow for
  bit-exactness). Stack frame 2080->32 bytes. Regs stayed 56 (occ still 66.7%, register-capped) -- the win was
  killing local-mem traffic, not occupancy. Pushing regs<=51 for more occupancy risks spills (back to local
  mem) for marginal gain; not pursued.
- kSCORE 4-wide probe batching (MLP to hide L2 latency): TRIED, REVERTED (no-op, don't re-try). Byte-identical
  but ZERO change (6.4s either way). Misread the ncu "60% L1TEX stall": kScore is L2-THROUGHPUT bound (70%), and
  at 91% occupancy the latency is ALREADY hidden by warp-switching (TLP), so within-thread MLP is redundant.
  Fewer L2 accesses would help (same probe count = same throughput demand); more parallelism for the same
  accesses does nothing. kScore is genuinely at its floor. Reverted to keep the hot loop simple.
NET of #1: ~2.5 days off 115 (kSetup only). kScore + kFill confirmed at floor.

## P3.13 — Tier B (skip-kFill via coarse pre-filter): measured, DEAD (2026-06-18). The deepest probe yet.
Premise: kFill (50% wall) only MATERIALIZES occ for probing. If a CHEAP filter using just kSetup vein origins
(no fill) could prune the world to a few candidates, fill+verify only those -> skip world-scale kFill -> 3-5x.
Hinged on two unknowns, measured both:
1. DISCRIMINATIVE POWER of the presence predicate (minfrac sweep on existing pipeline; occ-presence == vein-
   presence, same predicate, so this is a free lower bound). RESULT: a CLIFF. minfrac 0.50->10302 survivors,
   0.70->5, 0.80->4, 0.90->1, 1.00->1 (over 29.2M chunks). The 10302 "uniform/dense" survivors from P3.11 exist
   ONLY at the 0.5 noise-floor threshold; at 0.7+ the world empties to the truth. Reading noise is absorbed by
   the tolerance `e` (real rooms hit full presence), so the threshold can stay high. => signal is ENORMOUS.
   (This also corrects P3.11: the "4% cap on pruning" was measured at exactly the worst threshold; it bounds
   the weak minfrac-0.5 survivor set, NOT a stronger sound filter. Over-generalized then.)
2. Can a CHEAP coarse filter capture that power? Built kFillBox (fill each vein's padded BOUNDING BOX = cheapest
   node-free sound superset of the sphere) into a 2nd occ buffer; kScore presence vs box-occ; counted survivors.
   RESULT (decisive NO): box survivors @ minfrac 0.70=968,845,675  0.80=504,177,580  0.90=139,673,027 -- vs
   sphere's 5/4/1. Even at 0.90 the box leaves 140M (vs 1). AND kFillBox (12.6s) is SLOWER than kFill (10.5s)
   (boxes > spheres). So the cheapest coarse filter prunes ~nothing AND isn't cheap.
CONCLUSION (deep, final): the discriminative signal lives almost ENTIRELY in the EXACT sphere boundary -- the
razor-thin gap between "in the padded box" (968M) and "in the sphere" (5). Any representation cheap enough to
skip kFill (boxes) loses ~all discrimination; any representation that keeps it (per-node distance tests) costs
the same as kFill. You CANNOT separate "cheaply locate" from "expensively generate" -- the locating signal IS
the expensive geometry. Quantitatively confirms [[uniqueness-needs-exact-transcription]] (coarse/centroid
matching fails) with a hard number. Tier B dead; the ~115-day blind-global floor stands. Scaffolding (kFillBox,
--measure-box, presOcc, dOccBox) reverted; build byte-identical vs --legacy-gen.

## P3.14 — CAPSULE predicate: discriminates cheaply, but the win hinges on fill cost (2026-06-18).
Pushed past the box: tested the CAPSULE predicate (cell within maxOff of the vein's lerp-line AXIS SEGMENT;
sound superset of the union-of-node-spheres since every node lies on the segment w/ radius<=maxOff; cheaper
than exact = 1 segment-distance/cell vs N node-distances). Survivors @ minfrac 0.70/0.80/0.90 = 1148/10/4 vs
sphere 5/4/1 vs box 968M/504M/140M. => CORRECTION to P3.13: discrimination is NOT all in the exact per-node
geometry; the cheap capsule (axis+radius) keeps it (440M -> ~10-1148, ~400000x prune). BUT (1) scoring isn't
reduced -- a per-cell filter over 440M anchors is still ~10^12 probes and occ already makes each a 1-read; the
de-amortization wall (reads>>fills) is untouched, and any analytic per-cell test is >=3x a read. (2) naive
kFillCapsule (per-cell FP64 divide) = 52s vs kFill 12s. The ONE credible bit-exact path it opens: capsule-occ
as a fast APPROXIMATE PASS1 -> exact-verify the ~1148 survivors (sound: truth always in the superset). Wins
IFF analytic capsule-fill < sphere-fill (12s). A-priori doubt: capsule is a SUPERSET -> ~2x more occ cells ->
more atomic writes; if kFill is write-bound, capsule-fill can't beat sphere-fill regardless of cheaper geometry.
Optimized capsule-fill (divide hoisted, FP32 bulk) measured 6.1-6.4s vs kFill 10.5s = 1.7x FASTER (a-priori
"superset->more writes->slower" doubt was WRONG; the geometry saving -- 1 axis-distance/cell vs N node-distances,
tuff has 64 nodes -- swamps the extra writes). Capsule PASS1 survivors @ minfrac 0.50/0.60/0.70 = 1.4M/61K/1148.

## P3.15 — CAPSULE TWO-PASS built, validated, MEASURED DECISIVELY SLOWER -> REVERTED (2026-06-18). FINAL.
Built the full bit-exact two-pass under --capsule: kFill(anchorOnly) for exact anchors + kFillCapsule (capsule-
occ) -> kScore on capsule-occ (PASS1 superset) -> CSR vein index (by origin chunk) -> kVerify (analytic exact
rescore, nodeCovers reproduces kFill's FP32/EPS+FP64-shell predicate bit-for-bit) -> compact. VALIDATED byte-
identical top-K vs the exact pipeline (601^2 region; survivor counts match exactly: 10302 @0.5). Then measured
@29.2M chunks:
  EXACT baseline:   kGen 12.7s + kScore 6.3s = 19.0s
  capsule mf=0.50:  kGen 10.5s + kScore+verify 195.8s = 206s   (10x SLOWER)
  capsule mf=0.60:  kGen 10.5s + 66.8s = 77s                   (4x slower)
  capsule mf=0.70:  kGen 10.5s + 10.7s = 21s                   (STILL slower than 19s, best case)
SLOWER AT EVERY THRESHOLD. Two independent causes, both the de-amortization wall from the OTHER side:
1. capsule-occ is a DENSER superset (more set bits) -> presence early-out fires LATER -> PASS1 scoring is ~1.7x
   slower (10.5s vs 6.3s). This penalty (~4.4s) ALONE exceeds the fill saving (~2.1s). The very looseness that
   makes capsule cheap to FILL makes it expensive to SCORE.
2. kVerify (per-cell analytic membership via neighborhood vein scan) is catastrophic at real survivor counts
   (1.4M PASS1 survivors @0.5 -> 196s) -- recomputing coverage per probe is exactly what occ memoizes away.
DECISIVE, FINAL CONCLUSION: occ IS the optimal representation. ANY looser-and-cheaper-to-BUILD structure is
denser-and-costlier-to-QUERY, and querying dominates (reads ~100x fills). Tier B and every consolation under it
(box, capsule, analytic verify) are now measured dead from every angle. The exact-occ pipeline is at the true
floor. Scaffolding (kFillCapsule, kVerify, vein index, --capsule, anchorOnly, presOcc) fully REVERTED; build
byte-identical vs --legacy-gen. Net durable win from this whole arc = kSetup local-buffer removal (~2.5 days).
The ONLY remaining levers are non-algorithmic: parallelism (multi-GPU/cloud) or a location prior.

## Overall status
Science + full CPU pipeline COMPLETE and validated on REAL worldgen (P6). GPU generator bit-exact + GPU
matcher validated end-to-end on real data (P3). Remaining:
**P3 (GPU port** for world-scale search SPEED — same format/logic; toolchain now installed) and the user's
extraction mod (emits OBSERVATION_FORMAT.md incl. `bare` cells).
