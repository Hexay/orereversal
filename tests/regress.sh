#!/usr/bin/env bash
# Byte-for-byte regression of the matcher, oretest and solver against tests/expected/.
#   bash tests/regress.sh            compare (exit 1 on any difference)
#   bash tests/regress.sh --update   regenerate the expected outputs
#   bash tests/regress.sh --python   also run the (slow, ~1 min) Python solver case
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EXP=tests/expected
TMP=tests/tmp
mkdir -p "$EXP" "$TMP"

UPDATE=0; WITH_PY=0
for a in "$@"; do
  case "$a" in
    --update) UPDATE=1 ;;
    --python) WITH_PY=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; *) EXE= ;; esac
MATCHER=cuda/matcher$EXE
ORETEST=cuda/oretest$EXE
REGION_DUMP=harness/region_dump$EXE
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
grep -E '^(family|lapis|redstone|granite),' "$ROOM" > "$TMP/rare_only.csv"

$MATCHER $REGION $ROOM                                  2>&1 | check clean_refine clean_refine
$MATCHER $REGION $ROOM --no-refine                      2>&1 | check clean_gpu clean_gpu
$MATCHER $REGION $ROOM --no-refine --legacy-gen         2>&1 | check clean_legacy clean_gpu
$MATCHER $REGION "$TMP/noise1.csv" --error 1            2>&1 | check noise1 noise1
$MATCHER $REGION "$TMP/noise1.csv" --error 1 --abs-error 1 2>&1 | check noise1_abs noise1_abs
$MATCHER $REGION "$TMP/rare_only.csv"                   2>&1 | check rare_only rare_only
$MATCHER 123 -16 15 -16 15 examples/real_polA.csv       2>&1 | check real_polA real_polA

{ $ORETEST 123 0 7 0 7 -64 -1 2>/dev/null | tr -d '\r' | tail -n +2 | LC_ALL=C sort > "$TMP/port.csv"
  $REGION_DUMP 123 1.18 0 7 0 7 -64 -1 2>/dev/null | tr -d '\r' | tail -n +2 | LC_ALL=C sort > "$TMP/ref.csv"
  diff "$TMP/port.csv" "$TMP/ref.csv" | grep -E '^[<>]' | cut -d, -f1 | LC_ALL=C sort | uniq -c; } | check golden_diff golden_diff

if [ $WITH_PY = 1 ]; then
  $PY python/solve.py $ROOM 2>&1 | check solve_py solve_py
fi

[ $FAIL = 0 ] && echo "all passed" || echo "REGRESSION"
exit $FAIL
