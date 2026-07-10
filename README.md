# oreReversal

Localize a player's position in a **known-seed** Minecraft world from nothing but the
**ore pattern** they can see on a carved wall. Given the seed and a typed point-set of exposed
ore blocks (family + relative x/y/z, orientation and absolute position unknown), the matcher
recovers the true world location and quantifies how unique that match is — well enough to work at
world scale (validated to 30M chunks; ~minutes projected for a 300k×300k region).

The core idea: 1.18+ deep-slate ore generation is deterministic from the world seed, and the
discard-free families (**tuff, redstone, lapis, granite**) generate *bit-exactly* without needing
the full terrain density function. That fine structure is a fingerprint. Match the full
fingerprint (not vein centroids) and a single carved room is world-unique.

## Repository layout

| Path | What it is |
|------|-----------|
| `cuda/` | **The GPU matcher (main deliverable).** `oregen.h` is a portable host+device bit-exact port of cubiomes' 1.18 ore generation; `matcher.cu` is the tiled two-pass world-region localizer. See `cuda/README.md`. |
| `matcher/` | Python CPU reference matchers + the synthetic round-trip validation harness. Observation format in `matcher/OBSERVATION_FORMAT.md`. |
| `harness/` | Small C tools (`region_dump`, `ore_dump`) that dump reference ore positions from cubiomes — the ground truth the GPU port is diffed against. |
| `other/` | `decorationreverse_v3.cu` — a related decoration-reversal experiment. |
| `mushroom/` | Incomplete WIP fragment (mushroom-island finder); does not build standalone yet. |
| `NOTES.md` | The full research log (design decisions, dead-ends, why each ore family is or isn't matchable). |
| `CUDA_PARADIGMS.md` | GPU optimization notes. |

## Building

**Toolchain** (what this was developed against — adjust for your machine):
- NVIDIA CUDA Toolkit 12.9, GPU arch `sm_89` (RTX 40-series). Change `-arch` for your card.
- MSVC (VS 2022 Build Tools, C++ workload) as the `nvcc` host compiler, plus OpenMP.
- MinGW `gcc` + CMake to build cubiomes (its CMake hard-errors on MSVC).

**1. Clone cubiomes** (third-party, not vendored — pinned to the fork with the 1.18 ore config):
```sh
git clone https://github.com/xpple/cubiomes.git
cd cubiomes && git checkout 62007b8c6260290a3951f8ea9ce4a41e60dd1b54 && cd ..
```

**2. Build the harness** (cubiomes static lib + dump tools):
```sh
bash harness/build.sh
```

**3. Build the GPU matcher** (from a VS BuildTools x64 dev shell, CUDA `bin` on PATH):
```sh
cd cuda
nvcc -O2 -arch=sm_89 -Xcompiler /openmp matcher.cu -o matcher.exe
```
`cuda/rebuild.bat` does this from a non-dev shell (calls `vcvars64.bat` + prepends the CUDA bin).

## Usage

```sh
matcher.exe <seed> <cxMin> <cxMax> <czMin> <czMax> <obs.csv> \
            [--error E] [--absw W] [--minfrac F] [--tile T] [--topk K] [--refine N] [--no-refine]
```
`obs.csv` is the observed ore pattern (`family,x,y,z`, one per line, plus `bare` cells for soft
absence) — see `matcher/OBSERVATION_FORMAT.md`. Output is a ranked list of world origins with a
presence/absence score and a uniqueness margin.

Details, validation results, and the perf arc are in `cuda/README.md` and `NOTES.md`.

## Attribution

Depends on [cubiomes](https://github.com/Cubitect/cubiomes) by Cubitect (MIT), via the
[xpple/cubiomes](https://github.com/xpple/cubiomes) fork (commit `62007b8`) for the 1.18 ore config.
The ore-generation port in `cuda/oregen.h` is hand-derived from that source.

## License

MIT — see [LICENSE](LICENSE). (cubiomes remains under its own MIT license.)
