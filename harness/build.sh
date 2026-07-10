#!/usr/bin/env bash
# Build cubiomes (static, MinGW) + the ore_dump harness. Run from repo root or here.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. cubiomes static lib (MinGW only — its CMake hard-errors on MSVC)
if [ ! -f cubiomes/build/libcubiomes_static.a ]; then
  cmake -G "MinGW Makefiles" -S cubiomes -B cubiomes/build -DCMAKE_BUILD_TYPE=Release
  cmake --build cubiomes/build -j 8
fi

# 2. harness
gcc -O2 -Icubiomes harness/ore_dump.c cubiomes/build/libcubiomes_static.a -lm -o harness/ore_dump.exe
echo "built harness/ore_dump.exe"
