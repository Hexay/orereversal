#!/usr/bin/env bash
# Build cubiomes (static) + the region_dump/ore_dump harness. Run from repo root or here.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) GEN="MinGW Makefiles"; EXE=.exe ;;   # cubiomes' CMake hard-errors on MSVC
  *)                    GEN="Unix Makefiles"; EXE= ;;
esac

# 1. cubiomes static lib
if [ ! -f cubiomes/CMakeLists.txt ]; then
  echo "cubiomes/ is missing: clone xpple/cubiomes at 62007b8 into it (see README Quick start)"
  exit 1
fi
if [ ! -f cubiomes/build/libcubiomes_static.a ]; then
  cmake -G "$GEN" -S cubiomes -B cubiomes/build -DCMAKE_BUILD_TYPE=Release
  cmake --build cubiomes/build -j "$(nproc 2>/dev/null || echo 8)"
fi

# 2. harness
for t in region_dump ore_dump; do
  gcc -O2 -Icubiomes harness/$t.c cubiomes/build/libcubiomes_static.a -lm -o harness/$t$EXE
  echo "built harness/$t$EXE"
done
