#!/usr/bin/env bash
# Linux build of oretest + matcher (Windows: rebuild.bat). ARCH defaults to the local GPU.
set -e
cd "$(dirname "$0")"
ARCH="${ARCH:-native}"

# -ffp-contract=off: same role as MSVC /fp:strict — no FMA contraction, so the port stays bit-exact
gcc -O2 -ffp-contract=off oretest.c -lm -o oretest
echo "built cuda/oretest"
nvcc -O2 -std=c++17 -arch="$ARCH" -Xcompiler -fopenmp -lgomp matcher.cu -o matcher
echo "built cuda/matcher (arch=$ARCH)"
