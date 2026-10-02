# Ore coverage audit (2026-10-01)

> **Status (2026-10-02)** — findings below are kept as written; see `docs/research-log.md` P8-P10.
> 1. Fixed (P8). `cuda/ore_config.h` carries `index[ERA_COUNT]` (1.18-1.19 / 1.20+); the matcher takes
>    `--version` (`cuda/options.h`) and passes it to region_dump (`cuda/region_dump.h` `runRegionDump`);
>    `python/gen_wall.py` passes `args.version`. The "hardcodes 1.18" references in item 1 and the
>    Evidence section are stale, except `python/research/matcher.py:10` (research script, still 1.18).
>    The "GPU OK for 1.18-1.19" entries in the version table are stale: lapis/buried_lapis/copper/
>    large_copper are now correct for 1.20+ too.
> 2. Scope still holds; vanilla 1.21.x index layout is still not verified against a real 1.21 world.
> 3. GPU surface-gate port decided against (P8): vs the real world the ungated port is barely worse for
>    lapis and better for lower granite. Refine branches lapis gate outcomes (P10, window startY-surface
>    in [0,12]); pass 1 retries a non-CONFIDENT lapis-anchored search on redstone/granite (P9). Pass-1
>    lapis stays ungated; `tests/regress.sh` golden_diff_low pins the divergence.
> 4. Adopted (P8): refine family `diamond` (buried diamond only).
> 5. Skipped (P8): lower diorite/andesite only reach y -6..-1 and would each cost a GPU occupancy slot.
> 6. Still open (band not extended).
> 7. Vein tuff/iron modelled: CPU in P8 (`harness/ore_veins.h`, region_dump `+veins`), GPU iron-vein tuff
>    in P9 (`cuda/iron_veins.h`, `kIronVeins`). Large copper in deep_dark vs vanilla still unverified.

Scope: does the GPU generator (`cuda/ore_config.h`, `cuda/oregen.h`) and the CPU refine
(`harness/region_dump.c` via `cuda/region_dump.h`) match cubiomes (xpple fork @62007b8) across
MC 1.18–1.21, and which unused overworld ore types could add signal. Line refs are `cubiomes/finders.c`
unless noted. Empirical diffs: `oretest.exe` (CPU build of the GPU port) vs `region_dump.exe`, no GPU used.

## Verdict (ranked by expected impact)

1. **BUG — lapis is wrong for every 1.20+ world (GPU), copper wrong for 1.20+ (CPU refine).**
   MC 1.20 inserted `ore_diamond_medium` (index 19, finders.c:1513/1720-1722) into the
   UNDERGROUND_ORES step, shifting every later decorator index by +1: lapis 21→22, buried_lapis 22→23,
   large_copper 23→24, copper 24→25 (finders.c:1490-1491, 1425-1426, 1493-1494, 1436-1437; switch at
   1612-1615, 1625-1628, 1687-1692, 1694-1696). `ore_config.h:41-42,46` hardcode the 1.18 indices and
   `cuda/region_dump.h:57` hardcodes `1.18`. Measured, seed 123, chunks 0..15², band -64..-1:
   1.20 and 1.21 lapis: 3831 of 3835 cubiomes blocks missing from the port, 3730 of 3734 port blocks
   spurious (i.e. completely uncorrelated). tuff/redstone/granite: 0 diffs in 1.19/1.20/1.21.
   Effect: on a 1.20+ world every observed lapis block is a presence miss at the true location and a
   random hit elsewhere, and every copper block likewise. Lapis is one of the rare "anchor" families.
   Fix: version-select the index (or carry `index` per version) for lapis/buried_lapis/copper/
   large_copper; pass the real version to region_dump; add a `--version` flag to the matcher.
2. **Correctness scope, precisely:**
   - **1.18.x and 1.19.x: correct as-is.** No used config branches between 1.18 and 1.19 (the only
     1.19 change is emerald's index, 1654-1655, unused).
   - **1.20.x through 1.21.11 (and 26.x per cubiomes): tuff, redstone, granite, gravel, iron correct;
     lapis + copper wrong** (item 1). cubiomes has *no* ore branch above MC_1_20 (biomes.h:34-45 enum;
     every `< MC_1_20` / `else` in getOreConfig), so 1.21.x == 1.20 in cubiomes. Whether vanilla
     1.21.x added any UNDERGROUND_ORES feature (which would shift indices again) is **not verified**
     here — only a real 1.20/1.21 ground-truth world (the gt/ procedure) can confirm.
   - Everything else is version-invariant for 1.18+: population seed (xoroshiro branch at
     finders.c:47), RNG kind (1915), vein geometry and build limits (2050-2051), base position (1944-1959).
3. **LATENT BUG — "no surface gate needed" is false for lower_granite and buried_lapis (and lapis
   in deep ocean).** `oregen.h:213-214` says the gate "never rejects a deepslate-band vein", but the
   gate (finders.c:2036-2044) rejects *high* veins in the same per-config RNG stream, and a rejected
   vein skips its `nextDouble × size` draws (2063), shifting every later vein in that chunk — exactly
   the mechanism that made gravel/copper desync (research-log P3). Max startY (= baseY-2-amortized)
   per GPU config: tuff -7, lower_redstone -35, redstone 12, lapis 29, **lower_granite 53,
   buried_lapis 61**. Any chunk whose approx surface (`mapApproxHeight`, generator.c:623-640) is below
   those over the vein footprint (oceans, rivers, low valleys) can desync. Evidence: on the *land*
   validation box (seed 123, chunks 0..15², y 0..80) the port already emits 4238 granite + 6 lapis
   blocks cubiomes gates out. **Measured in the band, MC 1.18 itself** (table "Gate probe"): over
   16k-chunk boxes, 3-6% of cubiomes' lapis blocks and 1.6-2.8% of its granite blocks are missing
   from the port and replaced by wrong ones; tuff and redstone are exact everywhere. The 64-chunk
   golden diff (`tests/regress.sh`, chunks 0..7, land) cannot catch this. Fix options: port the
   gate (needs the surface grid already planned for gravel/copper), or cheaper, move lower_granite
   and buried_lapis to the CPU refine path; ranks below item 1 because it hits a few % of blocks.
4. **GAIN — buried diamond is discard-free and unused.** `o_buried_diamond_118` has
   discardChanceOnAirExposure **1.0** (1422), which, like buried lapis, takes the `chance >= 1.0`
   branch at 2130 and rolls **no** RNG: bit-exact, not biome-gated, max startY 13 (gate-safe like
   redstone), size 8 × 4 veins/chunk across the whole band (triangle peak at y=-64). It produces a large
   share of all real diamonds. Catch: observed diamonds also come from regular (0.5) and large (0.7)
   diamond, which desync, so diamond observations can't use strict presence; they need one-sided
   scoring (credit hits, don't penalize unexplained diamonds; absence via `bare` still works). The
   research-log "diamond 9/13 matched" (P1) is consistent with buried veins being the matching part.
5. **SMALL GAIN — lower diorite / lower andesite are mechanically identical to lower granite**
   (1500, 1504 vs 1509: size 64, count 2, uniform 0..60, discard 0, any overworld biome). No rationale
   for excluding them exists in the repo: "andesite/diorite" only appears in the Policy-B *bare* rule
   (research-log:379, observation-format.md:23), i.e. "don't call them bare", not "they're
   contaminated". In the -64..-1 band they only reach y ≥ -6 (bases 0..5, ~0.2 veins/chunk each), so
   value is small, and they share granite's surface-gate caveat (item 3).
6. **Extending the band upward (y ≥ 0) does not add bit-exact families cheaply.** Above y=0 every
   relevant config (lower granite/diorite/andesite, dirt, gravel, copper, lapis, buried lapis, iron)
   is surface-gated, and cubiomes' gate is itself approximate (climate depth / 76, generator.c:637,
   vs vanilla's OCEAN_FLOOR_WG heightmap), so near-surface veins are not exact vs the real world even
   in cubiomes. Copper ore-veins also drop GRANITE filler at y 0..50 (2195), contaminating granite.
   Worth it only after porting a surface grid (the gravel/copper plan) and validating vs real terrain.
7. **Minor:** tuff is contaminated by iron ore-vein filler in y -60..-8 (finders.c:2196, TUFF filler);
   not a version issue, but real tuff inside iron veins is unpredicted (presence miss).
   `cubiomes` gates LargeCopperOre to `dripstone_caves || deep_dark` (1875); verify deep_dark vs vanilla
   before relying on large copper in 1.19+.

## Evidence

### Version table for the configs in `ore_config.h` (cubiomes getOreConfig)

Only `index` ever differs between versions for 1.18+; step (6), size, count, height, rarity and discard
are identical across 1.18/1.19/1.20/1.21 for every row. "1.21" = MC_1_21_11; cubiomes has no ore
branch above MC_1_20 (= 1.20.6), so 1.20.x, 1.21.x, 26.x all take the `_120` row.

| type (ore_config.h line) | def. line | idx 1.18 | 1.19 | 1.20 | 1.21 | size | count | height | discard | GPU OK for |
|---|---|---|---|---|---|---|---|---|---|---|
| tuff (38) | 1525 | 8 | 8 | 8 | 8 | 64 | 2 | U -64..0 | 0 | all |
| redstone (39) | 1520 | 16 | 16 | 16 | 16 | 8 | 4 | U -64..15 | 0 | all |
| lower_redstone (40) | 1511 | 17 | 17 | 17 | 17 | 8 | 8 | T -96..-32 | 0 | all |
| lapis (41) | 1490/1491 | 21 | 21 | **22** | **22** | 7 | 2 | T -32..32 | 0 | 1.18-1.19 |
| buried_lapis (42) | 1425/1426 | 22 | 22 | **23** | **23** | 7 | 4 | U -64..64 | 1.0 (no roll) | 1.18-1.19 |
| gravel (43, CPU) | 1481 | 1 | 1 | 1 | 1 | 33 | 14 | U -64..319 | 0 | all (gated) |
| lower_granite (44) | 1509 | 3 | 3 | 3 | 3 | 64 | 2 | U 0..60 | 0 | all, minus gate (item 3) |
| upper_granite (45) | 1536 | 2 | 2 | 2 | 2 | 64 | rare 1/6 | U 64..128 | 0 | never reaches band |
| copper (46, CPU) | 1436/1437 | 24 | 24 | **25** | **25** | 10 | 16 | T -16..112 | 0 | 1.18-1.19 |
| large_copper (CPU) | 1493/1494 | 23 | 23 | **24** | **24** | 20 | 16 | T -16..112 | 0 | 1.18-1.19 |
| middle/small/upper iron (CPU) | 1515/1522/1538 | 12/13/11 | same | same | same | 9/4/9 | 10/10/90 | T -24..56 / U -64..72 / T 80..384 | 0 | all |

Seeding/logic version branches for 1.18+: none besides the index. getPopulationSeed switches to
xoroshiro at `mc >= MC_1_18` (47-52) and never again; CREATE_RANDOM_SOURCE legacy only `<= 1.17` (1915);
build limits only `<= 1.17` (2050-2051); base-position order only `<= 1.14` (1949); emerald/lower-gold
scatter special case (1946, 1965). `region_dump.c:41-53` parses 1.18/1.19/1.20/1.21 correctly; the
matcher just never passes anything but `1.18` (`cuda/region_dump.h:57`; also `python/gen_wall.py:24`,
`python/research/matcher.py:10`; `solve.py`/`make_observation.py` default to 1.18 but expose `--version`).

Measured (oretest.exe vs region_dump.exe, seed 123, chunks 0..15², y -64..-1, per family
onlyCubiomes/onlyPort): 1.18 and 1.19 all 0/0; 1.20 and 1.21 lapis 3831/3730 of 3835/3734, others 0/0.

### Gate probe (MC 1.18, y -64..-1, onlyCubiomes / onlyPort blocks)

| seed, chunk box | tuff | redstone | lapis | granite |
|---|---|---|---|---|
| 123, -64..63 x -64..63 | exact | 0 / 0 of 649687 | 7738 / 7501 of 248021 (3.1%) | 10349 / 9284 of 639452 (1.6%) |
| 123, 300..363 x -400..-337 | exact | 0 / 0 of 162870 | 3804 / 3690 of 61945 (6.1%) | 4467 / 4597 of 156974 (2.8%) |
| 42, -64..63 x -64..63 | exact | 0 / 0 of 649750 | 10370 / 10326 of 247685 (4.2%) | 11757 / 6227 of 652608 (1.8%) |

Mechanism: `generateOrePositions` returns before `generateVeinPart` when no footprint column has
`floor(approxHeight) >= startY` (2036-2044), so that vein's `size` nextDouble draws (2063) are never
consumed and every later vein of that config in that chunk shifts. The port always draws them
(`oregen.h:215-222`). A rejected vein is never itself in the band; the damage is to its successors.

### All overworld ore types (1.18 values; reach = approx block-Y span of one vein around its base Y)

Vein Y span from 2020-2034 / 2095-2117: size 64 ≈ base-6..+4, 33 ≈ -4..+2, 17/12/10/9 ≈ -3..+1,
8/7/4/3 ≈ -3..0. "Gate-safe" = max startY below any plausible surface.

| type | line | idx 18/20 | size | count / rarity | height | discard | biome gate (1827-1899) | reaches -64..-1? | status |
|---|---|---|---|---|---|---|---|---|---|
| DirtOre | 1454 | 0 | 33 | 7 | U 0..160 | 0 | none | y ≥ -4 only (bases 0..3) | unused: surface-gated, overwritten by all later types |
| GravelOre | 1481 | 1 | 33 | 14 | U -64..319 | 0 | none | yes | CPU refine (gated) |
| Upper granite/diorite/andesite | 1536/1533/1528 | 2/4/6 | 64 | rare 1/6 (1922-1924) | U 64..128 | 0 | none | no | irrelevant to band |
| LowerGraniteOre | 1509 | 3 | 64 | 2 | U 0..60 | 0 | none | y ≥ -6 | GPU (gate bug) |
| LowerDioriteOre | 1504 | 5 | 64 | 2 | U 0..60 | 0 | none | y ≥ -6 | **unused, usable** (same caveat) |
| LowerAndesiteOre | 1500 | 7 | 64 | 2 | U 0..60 | 0 | none | y ≥ -6 | **unused, usable** (same caveat) |
| TuffOre | 1525 | 8 | 64 | 2 | U -64..0 | 0 | none | yes | GPU, exact |
| UpperCoalOre | 1530 | 9 | 17 | 30 | U 136..319 | 0 | none | no | n/a |
| LowerCoalOre | 1502 | 10 | 17 | 20 | T 0..192 | **0.5** | none | y ≥ -3, rare | excluded: discard roll |
| UpperIronOre | 1538 | 11 | 9 | 90 | T 80..384 | 0 | none | no | n/a (region_dump skips it, region_dump.c:105) |
| MiddleIronOre | 1515 | 12 | 9 | 10 | T -24..56 | 0 | none | yes | CPU refine; vein-noise iron missing |
| SmallIronOre | 1522 | 13 | 4 | 10 | U -64..72 | 0 | none | yes | CPU refine |
| GoldOre | 1472 | 14 | 9 | 4 | T -64..32 | **0.5** | none | yes | excluded: discard roll |
| LowerGoldOre | 1507 | 15 | 9 | UniformInt(0,1) | U -64..-48 | **0.5** | none | yes | excluded: discard roll |
| RedstoneOre / Lower | 1520/1511 | 16/17 | 8 | 4 / 8 | U -64..15 / T -96..-32 | 0 | none | yes | GPU, exact |
| DiamondOre | 1445 | 18 | 4 | 7 | T -144..16 | **0.5** | none | yes | excluded: discard roll |
| MediumDiamondOre (1.20+) | 1513 | -/19 | 8 | 2 | U -64..-4 | **0.5** | none | yes | excluded; its existence causes the index shift |
| LargeDiamondOre | 1497 | 19/20 | 12 | rare 1/9 | T -144..16 | **0.7** | none | yes | excluded: discard roll |
| **BuriedDiamondOre** | 1422/1423 | 20/21 | 8 | 4 | T -144..16 | **1.0, no roll** | none | yes, dense near -64 | **unused, bit-exact candidate** (item 4) |
| LapisOre / BuriedLapis | 1490/1425 | 21/22 → 22/23 | 7 | 2 / 4 | T -32..32 / U -64..64 | 0 / 1.0 | none | yes | GPU; index bug 1.20+, gate bug |
| LargeCopperOre | 1493 | 23 → 24 | 20 | 16 | T -16..112 | 0 | dripstone_caves, deep_dark (1875) | yes | CPU refine; biome-gated per vein (1936-1937) |
| CopperOre | 1436 | 24 → 25 | 10 | 16 | T -16..112 | 0 | none | yes | CPU refine (gated) |
| ClayOre | 1428/1429 | 26 → 27 | 33 | 46 | U 0..256 | 0 | lush_caves (1877) | y ≥ -4 only | unused: biome + surface gate |
| ExtraGoldOre | 1466 | 27 → 28 | 9 | 50 | U 32..256 | **0.5** | badlands (1871) | no | n/a |
| EmeraldOre | 1459-1461 | 31 / 32 (1.19) / 33 | 3 | 100 | T -16..480 | 0 | mountain biomes (1866-1869) | y ≥ -3, very rare | unused: biome gate + surface gate on 100 veins |
| DeepslateOre, plain Granite/Diorite/Andesite/Coal/Iron | 1439, 1476, 1449, 1420, 1433, 1485 | - | - | - | - | - | - | - | ≤ 1.17 only (returns 0 for 1.18+: 1607, 1632, 1643, 1675, 1686, 1624) |

### Why each excluded family is excluded (cause line)

- **diamond (regular/large/medium), gold, lower gold, lower coal:** discard 0.5/0.7 -> `rnd.nextFloat`
  rolled per cell at 2130; vanilla only rolls on replaceable cells, so cubiomes' stub `if (1)` at
  2124 over-rolls in caves and the stream desyncs. Terrain-dependent. Buried diamond is the exception.
- **dirt:** discard 0 and not biome-gated, but U 0..160 means most veins hit the surface gate
  (2040); it is index 0 so every later type overwrites it; band reach only y ≥ -4.
- **clay:** lush_caves biome gate (1877) checked per vein after its base-position draws (1936-1937),
  plus the surface gate; band reach only y ≥ -4.
- **emerald:** mountain-biome gate (1866-1869) on 100 veins/chunk, plus the surface gate; ~0 band blocks.
- **andesite/diorite (lower):** no technical exclusion. Policy B only says not to emit them as
  `bare` (research-log:379); the "contaminated" wording in observation-format.md:13-14 refers to
  dirt/clay and has no recorded reason (likely the surface/biome gates above). Identical to lower granite.
- **gravel, copper (GPU):** surface gate (2036-2044); kept on the CPU refine path.
- **large copper:** biome gate (1875). Note cubiomes includes deep_dark; vanilla adds large copper to
  dripstone caves only, so 1.19+ deep_dark chunks may be over-predicted (unverified).
- **iron:** feature ores are exact, but the ore-vein system (2195-2196: iron veins y -60..-8, raw iron
  blocks, TUFF filler) adds unpredicted iron and tuff.

### Upward extension (y ≥ 0)

Port vs cubiomes on land (seed 123, chunks 0..15², y 0..80, 1.18): tuff and redstone exact, lapis +6
port-only, granite +4238 port-only, i.e. the gate already fires on land. Families that would enter:
lower diorite/andesite/granite (big, 0..60), dirt, more lapis/copper/iron/gravel. All are surface-gated,
and the cubiomes gate approximates height as `np[NP_DEPTH] / 76` (generator.c:637) rather than vanilla's
OCEAN_FLOOR_WG heightmap, so cubiomes is itself not guaranteed exact near the surface. Granite in y 0..50
is also contaminated by copper ore-vein filler (2195). Not bit-exact without a validated surface model.
