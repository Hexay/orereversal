#!/usr/bin/env bash
# GPU pass timing on a large region (--no-refine), repeated to expose run-to-run noise.
#   bash tests/bench.sh [half-side-in-chunks=1024] [runs=3]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; *) EXE= ;; esac
MATCHER=cuda/matcher$EXE
[ -x "$MATCHER" ] || { echo "$MATCHER not found: build it first (see README)"; exit 1; }
R=${1:-1024}
RUNS=${2:-3}
echo "region: $((2*R)) x $((2*R)) chunks, examples/obs_big_room.csv, --no-refine"
for i in $(seq "$RUNS"); do
  start=$(date +%s.%N)
  status=0
  out=$("$MATCHER" 123 $((-R)) $((R-1)) $((-R)) $((R-1)) examples/obs_big_room.csv --no-refine 2>&1) || status=$?
  end=$(date +%s.%N)
  if [ $status != 0 ]; then
    printf 'run %d  FAILED (exit %d):\n%s\n' "$i" "$status" "$(echo "$out" | tail -5)"
    exit 1
  fi
  printf 'run %d  wall %.2fs  %s\n' "$i" "$(awk "BEGIN{print $end - $start}")" "$(echo "$out" | grep -o 'generate=.*')"
done
