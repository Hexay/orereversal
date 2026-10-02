# CUDA bit-exact reversing — paradigms & playbook

A reusable methodology for porting a reference generator (Minecraft / cubiomes, or any
deterministic algorithm) to CUDA for large-scale search, distilled from orereversal's
[`cuda/`](../cuda). Project-agnostic — copy this into a new repo and follow it.

Core principle: **bit-exactness is a proven property, never an assumption.** Every optimization
is validated byte-identical against a reference you always keep. Predictability > cleverness.

---

## 1. The portable bit-exact header

Put the ported generator in ONE header that compiles on **both** host (MSVC) and device (nvcc).

```c
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
```

Rules that keep it portable:
- **No VLAs, no GCC/Clang builtins, no STL** in the header — MSVC + nvcc must both accept it.
- **Config-as-data**: put generator parameters in a `static const` table, not in code paths.
  Gate the table's storage class so it's `__device__` under nvcc, plain host array otherwise:
  ```c
  #ifdef __CUDACC__
  #define TABLE static __device__ const
  #else
  #define TABLE static const
  #endif
  ```
- **Feature flags for the parts you can't port bit-exactly.** Anything that couples to an
  un-ported subsystem (terrain/surface noise, secondary noise systems) goes behind a
  `#define` (hypothetical example, not flags in this repo: `SKIP_SURFACE` vs `WITH_SURFACE` requiring a
  host-supplied hook). Document
  exactly which inputs each flag changes.
- Hand-transcribe constants from the reference source with the source line cited in a comment.
  Transcription errors are the #1 source of desync — make them auditable.

## 2. Know your families/cases — and why each is in or out

Before porting, classify every case by whether it can be made bit-exact **without** the
subsystems you're skipping. In the ore project the three "air dependencies" were the taxonomy:
1. one-directional supersets (cubiomes fills a superset → deep/solid regions match exactly) — **IN**;
2. RNG-desyncing couplings (`discardChanceOnAirExposure` rolls that shift the stream) — **OUT** without a terrain port;
3. post-gen mutations (gravity-fallen blocks) — **IN at generation, treat as bonus signal only**.

Write the classification down with the *measured* evidence (e.g. "diamond 4/16 miss vs real
world"). This is the single most valuable artifact — it tells future-you what's safe to trust
and stops you re-litigating closed questions.

## 3. The golden diff test (do this FIRST, before any GPU code)

1. Build a **CPU driver** (`*test.c`) that calls your portable header and prints the same
   output format as a trusted reference (the original library's binary).
2. Diff sorted output against the reference over a representative region:
   ```sh
   ./port.exe   <args> | tail +2 | sort > port.csv
   ./reference.exe <args> | tail +2 | sort > ref.csv
   diff port.csv ref.csv   # must be empty (or only the families you classified OUT)
   ```
3. **Zero positional diffs on the IN families is the gate.** You do not write a kernel until
   the host port passes. The GPU then only has to match the host port.

## 4. Two-pass GPU→CPU architecture

- **PASS 1 (GPU, the hot loop):** generate the bit-exact cases into a compact reused buffer
  (e.g. an occupancy bitmask), enumerate candidates, score with an **aggressive pre-filter +
  per-cell early-termination**, merge survivors into a global top-K.
- **PASS 2 (CPU refine):** re-score only the top-K with the expensive or un-portable cases
  (inject them from the reference binary). Prune to real contenders with a monotonic cutoff so
  the refine count is tiny and precomputable.

This keeps the un-portable physics out of the hot loop without losing its signal in the final
ranking.

## 5. Tiling for bounded memory at any scale

Split the search region into tiles so device memory is **independent of total search size**
(validated to 30M+ chunks). Clamp tile size to dodge the Windows **TDR** watchdog (≤320 in the
ore project — a kernel that runs >~2s gets killed). Survivors from each tile merge into a global
top-K via a **bucket grid** for spatial NMS dedup (cell = separation distance, 3×3×3 neighbor
lookup) — never an O(K²) scan.

## 6. The optimization discipline (the most transferable asset)

This is a loop, run with `ncu` (enable: NVIDIA CP → Developer → Manage GPU Performance Counters
→ allow all users):

1. **Always keep a legacy reference path** (here `--legacy-gen`) — the slow, obviously-correct kernel. Never
   delete it. Every optimization is validated **byte-identical** against it (and against the
   golden diff test).
2. **Profile before cutting.** Find the actual bottleneck; don't guess. Record the per-kernel
   split (it moves as you optimize).
3. **On consumer (Ada/Ampere) GPUs, FP64 is the usual floor** — 1/64 rate, division is
   multi-cycle. The two biggest wins in the ore project were both FP64 attacks:
   - eliminate division: test `d² ≥ r²` not `(d/r)² ≥ 1` (precompute `r²`); fuse into DFMA.
   - **mixed precision with a *provable* margin**: classify in FP32 with an EPS margin sized
     ≫ max FP32 error (so NO cell is ever misclassified → result provably identical, not merely
     "ranking-equivalent"); recompute only the thin boundary shell (~0.1% of cells) in FP64.
     Reduce coordinates to a local origin first so FP32 has no catastrophic cancellation
     (Sterbenz-exact subtraction even at world scale).
4. **Memory locality via Morton/Z-order sort** of work items before the scoring kernel, so
   adjacent threads probe overlapping memory (L2 hit 87→97.6% in the ore project).
5. **Validate every step.** "Top-3 byte-identical, deep-tail wobble is pre-existing atomic-race
   noise" — distinguish your change's effect from baseline nondeterminism.

## 7. The lab notebook (here: [research-log.md](research-log.md)) — record dead-ends

Keep a running notebook. For every optimization tried, record the result AND the ones that
**failed**, marked `DEAD-END (measured <date>, don't re-try)` with the reason:
- capsule two-pass: cheaper to build, costlier to query, querying dominates → net slower.
- hierarchical tile pruning: survivors spatially uniform, <3.9% prunable → not worth it.
- batched occ write: global-atomic stream wasn't the bottleneck → zero change.

Dead-ends are as valuable as wins — they stop you burning the same path twice. State each
rationale once, at the most relevant site.

## 8. Build setup (Windows, non-dev shell)

`rebuild.bat` — self-contained, calls vcvars then prepends the CUDA bin so it runs from any shell:
```bat
@echo off
cd /d "<project-dir>"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
echo === building host driver ===
cl /nologo /O2 /fp:strict <driver>.c /Fe:<driver>.exe >nul
echo host_exit=%ERRORLEVEL%
echo === building device matcher ===
nvcc -O2 -std=c++17 -arch=sm_89 -Xcompiler /openmp -Xptxas -v <matcher>.cu -o <matcher>.exe
echo device_exit=%ERRORLEVEL%
```
- `/fp:strict` on the host driver — float contraction must match the reference for bit-exactness.
- `-arch=sm_89` = RTX 4070 Ti SUPER (Ada). Change per GPU.
- `-Xptxas -v` to watch registers/smem (occupancy). `-Xcompiler /openmp` for parallel refine.
- CUDA toolkit pinned (v12.9 here).

## 9. Standard CUDA scaffolding conventions

- Error-check macro on every CUDA call:
  ```c
  #define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
      fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
  ```
- Two-kernel split when a single kernel suffers load imbalance: a setup kernel doing the
  serial/order-dependent work at full lane utilization (1 thread per work-item, **work-major
  layout so a warp shares uniform loop bounds** → low divergence), then a cooperative fill/score
  kernel (persistent warps, atomic work queue, 1 warp per item, stripe across 32 lanes).
- Some `__syncwarp`/suppression steps are **load-bearing for performance**, not just
  correctness — removing them can be byte-identical but slower (more global-atomic traffic).
  Measure before deleting; annotate "don't re-try".
- Resolve helper-binary paths from `argv[0]` so the exe is relocatable.

## 10. Process (from global prefs, applied here)

- Interview before implementing non-trivial work: one decision at a time, recommend an answer,
  resolve dependencies in order. Settle the key decisions (target version, output, which cases
  are in scope) **before** writing the port.
- Check whether the source already answers a question (read it, grep it) before asking.
- No god files (>300 lines split by responsibility). Comment only for genuinely non-obvious
  logic, gotchas, and TODOs — terse, stated once.
- A function's name reflects everything it does; no implicit side effects.

---

### Checklist for a new project
- [ ] Identify the reference generator + a binary that emits ground truth.
- [ ] Classify cases: which are bit-exact-portable, which couple to skipped subsystems, why.
- [ ] Portable header (`HD` macro, config table, feature flags), constants cited from source.
- [ ] CPU driver + golden diff test → **zero diffs before any kernel**.
- [ ] Legacy reference kernel (slow, correct) — keep forever.
- [ ] Tiled two-pass GPU→CPU pipeline.
- [ ] Optimize: profile → attack FP64 (div-free, provable mixed precision) → locality (Morton)
      → each step byte-identical vs legacy.
- [ ] Research log (here `docs/research-log.md`): wins AND dead-ends with dates and reasons.
