#!/usr/bin/env bash
# Byte-for-byte regression of the matcher, oretest and solver against tests/expected/.
#   bash tests/regress.sh            compare (exit 1 on any difference)
#   bash tests/regress.sh --update   regenerate the expected outputs
#   bash tests/regress.sh --python   also run the (slow, ~1 min) Python solver case
#   bash tests/regress.sh --build    rebuild harness + cuda first
set -u
shopt -s lastpipe # `cmd | check` must run check in this shell, or its FAIL=1 is lost
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EXP=tests/expected
TMP=tests/tmp
mkdir -p "$EXP" "$TMP"

UPDATE=0; WITH_PY=0; BUILD=0
for a in "$@"; do
  case "$a" in
    --update) UPDATE=1 ;;
    --python) WITH_PY=1 ;;
    --build) BUILD=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WINDOWS=1; EXE=.exe ;; *) WINDOWS=0; EXE= ;; esac

if [ $BUILD = 1 ]; then
  bash harness/build.sh > "$TMP/build.log" 2>&1 || { tail -20 "$TMP/build.log"; exit 1; }
  if [ $WINDOWS = 1 ]; then cmd //c "cuda\\rebuild.bat" >> "$TMP/build.log" 2>&1
  else bash cuda/build.sh >> "$TMP/build.log" 2>&1; fi
  if grep -q -E "exit=[1-9]|error" "$TMP/build.log"; then grep -E -B2 -A4 "error|exit=[1-9]" "$TMP/build.log" | head -40; exit 1; fi
  echo "build ok ($(grep -c -i warning "$TMP/build.log") warnings)"
fi
MATCHER=cuda/matcher$EXE
ORETEST=cuda/oretest$EXE
REGION_DUMP=harness/region_dump$EXE
VEINTEST=cuda/veintest$EXE
for bin in "$MATCHER" "$ORETEST" "$REGION_DUMP" "$VEINTEST"; do
  [ -x "$bin" ] || { echo "$bin not found: build first (bash tests/regress.sh --build)"; exit 1; }
done
if command -v py >/dev/null 2>&1; then PY="py -3"; else PY=python3; fi

FAIL=0
# Timing lines vary run to run; the generator line differs between --legacy-gen and the default.
normalize() { tr -d '\r' | grep -v -E '^\[timing\]|^generator:'; }

check() {   # check <name> <expected-name>  (reads actual output on stdin)
  local name=$1 expected=$EXP/$2.txt
  normalize > "$TMP/$name.txt"
  if [ $UPDATE = 1 ] && [ "$1" = "$2" ]; then
    cp "$TMP/$name.txt" "$expected"; echo "updated  $name"
  elif diff -q "$expected" "$TMP/$name.txt" >/dev/null 2>&1; then
    echo "ok       $name"
  else
    echo "FAIL     $name"; diff "$expected" "$TMP/$name.txt" | head -10; FAIL=1
  fi
}

ROOM=examples/obs_big_room.csv
REGION="123 -32 31 -32 31"

$PY python/make_observation.py --noise 1 --out "$TMP/noise1.csv" >/dev/null
$PY python/make_observation.py --version 1.20 --out "$TMP/room_120.csv" >/dev/null
grep -E '^(family|lapis|redstone|granite),' "$ROOM" > "$TMP/rare_only.csv"

$MATCHER $REGION $ROOM                                  2>&1 | check clean_refine clean_refine
$MATCHER $REGION $ROOM --no-refine                      2>&1 | check clean_gpu clean_gpu
$MATCHER $REGION $ROOM --no-refine --legacy-gen         2>&1 | check clean_legacy clean_gpu
$MATCHER $REGION "$TMP/noise1.csv" --error 1            2>&1 | check noise1 noise1
$MATCHER $REGION "$TMP/noise1.csv" --error 1 --abs-error 1 2>&1 | check noise1_abs noise1_abs
$MATCHER $REGION "$TMP/rare_only.csv"                   2>&1 | check rare_only rare_only
$MATCHER 123 -16 15 -16 15 examples/real_polA.csv       2>&1 | check real_polA real_polA
$MATCHER $REGION "$TMP/room_120.csv" --version 1.20     2>&1 | check room_120 room_120
# A patch version must refine exactly like its minor (region_dump once rejected it and refine went quiet).
$MATCHER $REGION "$TMP/room_120.csv" --version 1.20.4 2>&1 | sed 's/mc=1\.20\.4/mc=1.20/' \
  | check room_1204 room_120
$MATCHER 123 -16 15 -16 15 examples/real_vein_room.csv  2>&1 | check real_vein_room real_vein_room

golden_diff() {   # golden_diff <version> [chunk box]: per-family count of blocks where port != cubiomes
  local box=${2:-"0 7 0 7"}
  $ORETEST 123 $box -64 -1 "$1" 2>/dev/null | tr -d '\r' | tail -n +2 | LC_ALL=C sort > "$TMP/port.csv"
  $REGION_DUMP 123 "$1" $box -64 -1 tuff redstone lapis gravel granite copper iron 2>/dev/null \
    | tr -d '\r' | tail -n +2 | LC_ALL=C sort > "$TMP/ref.csv"
  diff "$TMP/port.csv" "$TMP/ref.csv" | grep -E '^[<>]' | cut -d, -f1 | LC_ALL=C sort | uniq -c
}
# Land: only gravel/copper (surface gate) and iron (not ported) may differ.
golden_diff 1.18 | check golden_diff golden_diff
golden_diff 1.20 | check golden_diff_120 golden_diff_120
# Low terrain: lapis also differs, since the port never applies the surface gate (docs/research-log.md P9).
golden_diff 1.18 "-16 -9 -16 -9" | check golden_diff_low golden_diff_low

vein_diff() {   # iron-vein tuff (iron_veins.h) vs harness/ore_veins.h: must be identical
  $ORETEST 123 -12 -5 -16 -9 -64 -1 +veins 2>/dev/null | tr -d '\r' | grep '^tuff,' | LC_ALL=C sort > "$TMP/port.csv"
  $REGION_DUMP 123 1.18 -12 -5 -16 -9 -64 -1 tuff +veins 2>/dev/null | tr -d '\r' | grep '^tuff,' \
    | LC_ALL=C sort > "$TMP/ref.csv"
  echo "tuff blocks: $(wc -l < "$TMP/ref.csv"), differing: $(diff "$TMP/port.csv" "$TMP/ref.csv" | grep -c '^[<>]')"
}
vein_diff | check vein_diff vein_diff

vein_diff_gpu() {   # the same, but from kIronVeins on the device (vein tuff only, compared as sets)
  $VEINTEST 123 -32 31 -32 31 2>/dev/null | tr -d '\r' | tail -n +2 | LC_ALL=C sort -u > "$TMP/port.csv"
  $REGION_DUMP 123 1.18 -32 31 -32 31 -64 -1 diamond +veins 2>/dev/null | tr -d '\r' | grep '^tuff,' \
    | LC_ALL=C sort -u > "$TMP/ref.csv"
  echo "tuff blocks: $(wc -l < "$TMP/ref.csv"), differing: $(diff "$TMP/port.csv" "$TMP/ref.csv" | grep -c '^[<>]')"
}
vein_diff_gpu | check vein_diff_gpu vein_diff_gpu

if [ $WITH_PY = 1 ]; then
  $PY python/solve.py $ROOM 2>&1 | check solve_py solve_py
fi

[ $FAIL = 0 ] && echo "all passed" || echo "REGRESSION"
exit $FAIL
