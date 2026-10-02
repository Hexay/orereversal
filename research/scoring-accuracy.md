# Scoring accuracy under positional noise (2026-10-01)

> **Status (2026-10-02)** — adopted in P8 (`docs/research-log.md`): #1 re-centring ±e (`cuda/refine.h`,
> `python/solve.py`); #2 solve.py presence dilates by e, not 2e; #4 `--abs-error A` is now erosion
> (`cuda/refine.h` `oreThroughout`, solve.py `--abs-error`), default 0; #5 margin is measured against the
> best result more than max(2e+1, footprint) away (`cuda/matcher.cu`, `cuda/report.h`). #3 rarity
> weighting skipped. `research/scratch/*` cited below is local-only (gitignored), not in the repo.

CPU-only study of how the solvers score noisy observations. No GPU was run; GPU behaviour is emulated
(presence tolerance ±e Chebyshev, top-K dedup separation 2e+1, same anchor-cell choice).

## Verdict (ranked)

1. **Re-centre the winners after the search: ADOPT.** This is the biggest win, and it fixes the
   off-by-anchor-noise origin. For each of the top ~20 distinct hypotheses, re-score every origin
   within ±e of it (same orientation) at presence tolerance ±e and exact absence, and keep the best.
   Over 20 trials (e=1,2 × 5 jitter seeds × R=8/13) it recovered the **exact origin 20/20**. The
   current solvers got it 4/20. It also raises the margin:
   - e=2, R=13: solve.py 115 → **394**; GPU-style 244 → **394** (min over seeds 59/190 → 374).
   - e=1, R=13: solve.py 374, GPU-style 457 → **510**.
   It works because at the true origin every jittered cell is within ±e of its true block, so presence
   is full and the exact bare probes land on true bare cells (absHits 154 → 0). The anchor-derived
   origin is off by the anchor's own noise, and that offset costs ~150 absence hits.
   Cost: (2e+1)^3 origins per refined hypothesis (125 at e=2). That is a refine-stage job, CPU or GPU.
   Re-centring ±2e is no better (e=2: 402 vs 414) and slightly worse at e=1 (481 vs 506), because
   wrong hypotheses get more room to improve. Use ±e.
2. **Presence tolerance: the GPU's ±e is right, and solve.py's 2e is wrong.** Once re-centring
   exists, the 2e dilation buys nothing: final results match (e=2 R=8: 414 vs 400, both 5/5 exact).
   Without re-centring it halves the margin (e=2: 124 vs 247), because a 5^3 dilation lets dense
   tuff match almost anywhere. ±e is also recall-safe in practice: the truth's anchor-derived presence
   was 604/633 (95%) at e=2, far above the GPU's `minPresenceFraction` 0.5. So the fix for the "OPEN"
   item in research-log P7 is: **solve.py should dilate by e (not 2e), and both solvers should
   re-centre.** Keep the GPU kernel as is; add re-centring to `refine.h` (`rescore` over ±e
   neighbours).
3. **Family weighting by −log2(density): small, consistent gain. Optional.** Weights come from the
   dilated candidate density in the band (R=8, e=0: tuff 4.0, gravel 6.2, redstone 9.0, granite 8.9,
   lapis 10.4, iron 9.8, copper 12.0). They shrink with tolerance (e=2: tuff 2.2 … copper 7.1).
   Margins below are normalised by the mean per-cell weight, so they compare directly with flat:
   - e=0: 452 → 484 (+7%)
   - e=1 + re-centre: 510 → 543 (+6%, R=13)
   - e=2 + re-centre: 394 → 422 (+7%, R=13)
   It never changed the rank-1 result or the origin. Risk: it up-weights iron (9.8), which is not
   bit-exact (misses some real blocks), and any desynced family. A true-location miss would then cost
   ~2.5x a tuff miss. Gating weights to bit-exact families (or capping iron at 1) is advised if adopted.
   Not worth it before items 1-2. ~7% does not change any verdict.
4. **Absence under noise: keep EXACT as the default. Never dilate. Offer erosion for noisy bare
   cells.**
   - Bare cells exact (control): exact beats erode-1 by a lot (e=2: 402 vs 263; e=1: 481 vs 331).
   - Bare cells jittered like ore (±e): exact and erode-1 have about the same margin (e=2: 249 vs 254
     at R=8, 242 vs 251 at R=13; e=1: 373 vs 347). Erode-1 recovers the exact origin 5/5 vs 2/5 for
     exact at e=2, because a misread bare cell landing on a true ore edge no longer counts.
   - Dilate-1 (the `--abs-error 1` semantics when this was written; now erosion, see Status) is the worst option everywhere: e=2 margin 121
     (min 45), origin exact 0/5, truth absHits 678.
   - Absence off: 191 (e=2) / 282 (e=1). Absence still adds ~30% even with noisy bare cells.
   - Weight 0.5: no better than 1 (243 vs 249).
   Recommendation: redefine `--abs-error A` as *erosion* (a hit only if every cell within ±A predicts
   ore) instead of dilation. Dilation was already measured harmful (P7), and erosion is the
   noise-robust version. Default stays A=0.
5. **Report margin against a different location, not a self-shift: low-cost fix.** At e=0 the dedup
   separation 2e+1 = 1, so "rank 2" is the truth shifted by 2 blocks. The reported margin 452
   understates the margin over the best genuinely different location (632). With re-centring at e>0,
   rank 2 was always ≥29 blocks away. Suggest a separation of at least ~4 blocks, or the observation
   half-extent, when choosing rank 2 for the margin/verdict.

Not separately measured: **multi-anchor voting.** Re-centring searches the whole ±e neighbourhood,
so it covers anything that voting across anchor cells could recover, and it got 20/20. Voting would
only matter if the enumerated hypothesis landed more than e from the truth, which never happened.

## Method

- Scratch code: `research/scratch/exp_lib.py` (a vectorised numpy re-implementation of `solve.py`
  that reproduces it exactly: `solve.py --region 6 --error 2` on `make_observation --noise 2` gives
  633/633, absHits 203, margin 37, origin off (-1,0,-2) in both). Driver:
  `research/scratch/run_exp.py <exp> <R>`. Raw output is in `research/scratch/out_*.txt`.
- Observation: `obs_big_room` box (true origin (-6,-52,-6), identity orientation). 633 ore (508 tuff,
  100 gravel, 23 redstone, 1 lapis, 1 iron) + 3791 bare. Anchor = the single lapis cell, as in both
  solvers.
- Jitter: each ore cell independently uniform ±N per axis (same RNG as `make_observation.py`; seed
  1 reproduces it byte for byte). Seeds 1-5 per row. For "bare jittered", bare cells are jittered the
  same way (local generator, tracked file untouched).
- All 7 families are scored in a single pass, like solve.py, not the GPU pass-1/refine split.
- Metrics, aggregated over the 5 seeds:
  - **rank1**: rank 1 is within 8 blocks of the truth with the correct orientation.
  - **exact**: rank 1's origin equals the truth.
  - **|off|**: Chebyshev distance of rank 1 from the truth.
  - **margin_next**: rank1 − rank2 after dedup at separation 2e+1, the solvers' own definition.
  - **margin_far**: best near-truth score minus the best score > 8 blocks away. It equalled
    margin_next in every noisy row.
  - Margins are divided by the mean per-cell weight, which is 1 for flat scoring.
- Region: R=8 (17² = 289 chunks), with key rows repeated at R=13 (729 chunks). Margins are almost
  flat with region size (compare tables A and D), consistent with P2d.

## A. Presence tolerance and re-centring (ore jittered, bare exact, exact absence, R=8)

| config | e | rank1 | exact origin | mean/max \|off\| | truth pres | truth absH | margin mean (min) |
|---|---|---|---|---|---|---|---|
| solve.py: search @2e | 1 | 5/5 | 2/5 | 0.6/1 | 633 | 56 | 379 (339) |
| GPU: search @e | 1 | 5/5 | 2/5 | 0.6/1 | 624 | 58 | 455 (410) |
| @2e → re-centre ±e @e | 1 | 5/5 | 5/5 | 0/0 | 633 | 0 | 506 (489) |
| @2e → re-centre ±2e @e | 1 | 5/5 | 5/5 | 0/0 | 633 | 0 | 481 (447) |
| @e → re-centre ±e @e | 1 | 5/5 | 5/5 | 0/0 | 633 | 0 | **507 (498)** |
| @e → re-centre ±2e @e | 1 | 5/5 | 5/5 | 0/0 | 633 | 0 | 470 (447) |
| anchors dilated ±e, @e (exhaustive)* | 1 | 5/5 | 5/5 | 0/0 | 633 | 0 | 504 (489) |
| solve.py: search @2e | 2 | 5/5 | 0/5 | 1.6/2 | 633 | 154 | 124 (55) |
| GPU: search @e | 2 | 5/5 | 0/5 | 1.6/2 | 604 | 154 | 247 (186) |
| @2e → re-centre ±e @e | 2 | 5/5 | 5/5 | 0/0 | 633 | 0 | 414 (374) |
| @2e → re-centre ±2e @e | 2 | 5/5 | 5/5 | 0/0 | 633 | 0 | 402 (374) |
| @e → re-centre ±e @e | 2 | 5/5 | 5/5 | 0/0 | 633 | 0 | **400 (374)** |
| @e → re-centre ±2e @e | 2 | 5/5 | 5/5 | 0/0 | 633 | 0 | 397 (374) |
| anchors dilated ±e, @e (exhaustive)* | 2 | 5/5 | 5/5 | 0/0 | 633 | 0 | 493 (451) |

\*The exhaustive variant enumerates every origin within ±e of each anchor candidate, which costs
(2e+1)^3 times more hypotheses. Its higher e=2 margin is probably an artefact: near-truth origins
crowd the 300-survivor cap and push out far competitors. Don't read it as headroom.

Clean regression (noise 0, R=8): e=0 exact gives margin 452 (rank 2 = self-shift at distance 2;
margin vs a far location 632). Running the re-centre pipeline at e=1 on the clean observation gives
exact origin, margin 487. Re-centring does not hurt a clean input.

## B. Family weighting (R=8; margins in mean-weight units)

| noise | config | flat margin (min) | info margin (min) |
|---|---|---|---|
| 0 | e=0 exact | 452 (452) | 484 (484) |
| ±1 | GPU search @e, no re-centre | 455 (410) | 480 (434) |
| ±1 | @2e → re-centre ±2e @e | 481 (447) | 523 (484) |
| ±2 | GPU search @e, no re-centre | 247 (186) | 269 (199) |
| ±2 | @2e → re-centre ±2e @e | 402 (374) | 421 (398) |

Rank 1, origin and exact-origin counts were identical between flat and info in every row.

## C. Absence when bare cells are jittered too (search @2e → re-centre ±2e @e, R=8)

| noise (ore & bare) | absence | exact origin | truth absH | margin mean (min) |
|---|---|---|---|---|
| ±1 | off (w=0) | 5/5 | 124 | 282 (278) |
| ±1 | exact | 5/5 | 124 | **373 (341)** |
| ±1 | exact, w=0.5 | 5/5 | 124 | 345 (338) |
| ±1 | dilate 1 | 1/5 | 690 | 174 (161) |
| ±1 | erode 1 | 5/5 | 0 | 347 (340) |
| ±2 | off (w=0) | 5/5 | 187 | 191 (177) |
| ±2 | exact | 2/5 | 182 | 249 (217) |
| ±2 | exact, w=0.5 | 4/5 | 185 | 243 (222) |
| ±2 | dilate 1 | 0/5 | 678 | 121 (45) |
| ±2 | erode 1 | **5/5** | 15 | **254 (236)** |
| ±2 | erode 2 | 5/5 | 0 | 202 (192) |
| control ±1, bare exact | exact | 5/5 | 0 | **481 (447)** |
| control ±1, bare exact | erode 1 | 5/5 | 0 | 331 (306) |
| control ±2, bare exact | exact | 5/5 | 0 | **402 (374)** |
| control ±2, bare exact | erode 1 | 5/5 | 0 | 263 (251) |

Truth rank 1 held in every row (5/5). Erosion helps only when the bare cells really are noisy, and
costs about a third of the margin when they are exact. That is why it should be opt-in, not default.

## D. Confirmation at R=13 (729 chunks)

| noise | config | exact origin | margin mean (min) |
|---|---|---|---|
| ±1 | solve.py @2e | 2/5 | 374 (309) |
| ±1 | GPU @e | 2/5 | 457 (410) |
| ±1 | @e → re-centre ±e @e | 5/5 | 510 (498) |
| ±1 | same, info weights | 5/5 | 543 (532) |
| ±2 | solve.py @2e | 0/5 | 115 (59) |
| ±2 | GPU @e | 0/5 | 244 (190) |
| ±2 | @e → re-centre ±e @e | 5/5 | 394 (374) |
| ±2 | same, info weights | 5/5 | 422 (401) |
| ±2 ore & bare | exact absence, re-centre | 2/5 | 242 (216) |
| ±2 ore & bare | erode-1 absence, re-centre | 5/5 | 251 (236) |

The GPU-style R=8→13 numbers (247→244, e=2) track the main session's 64²-chunk figure (153 after
refine with gravel). That is consistent with the P2d flat-ceiling result, so these gains should
carry over to world-scale regions.

## Caveats

- One room geometry (obs_big_room) and synthetic, self-consistent cubiomes data. Real-world misses
  (iron, caves) aren't modelled. The weighting risk in item 3 comes from exactly that.
- The jitter is independent per cell, uniform ±N, which is harsher than a rigid misregistration. A
  rigid offset is fully absorbed by re-centring.
- Margins are in flat score points (presence − absHits), so the verdict thresholds
  (max(3, 0.3·N) ≈ 190) apply. With re-centring, e=2 moves from "shortlist" (GPU 153 at 64²
  chunks) to CONFIDENT territory (~394 here).
