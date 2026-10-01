#!/usr/bin/env bash
# GPU pass timing on a large region (--no-refine), repeated to expose run-to-run noise.
#   bash tests/bench.sh [half-side-in-chunks=1024] [runs=3]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; *) EXE= ;; esac
R=${1:-1024}
RUNS=${2:-3}
echo "region: $((2*R)) x $((2*R)) chunks, examples/obs_big_room.csv, --no-refine"
for i in $(seq "$RUNS"); do
  start=$(date +%s.%N)
  out=$(cuda/matcher$EXE 123 $((-R)) $((R-1)) $((-R)) $((R-1)) examples/obs_big_room.csv --no-refine 2>&1)
  end=$(date +%s.%N)
  printf 'run %d  wall %.2fs  %s\n' "$i" "$(awk "BEGIN{print $end - $start}")" "$(echo "$out" | grep -o 'generate=.*')"
done
