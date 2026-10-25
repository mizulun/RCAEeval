#!/usr/bin/env bash
# Baseline benchmark runner for the paper: baro, circa, microrank, tracerca.
#
# Per-method AC@K summaries + wall time go to output/baseline-summaries/<method>_<dataset>.txt.
# Each run gets its own --result-dir, so nothing touches output/results and
# runs cannot pollute each other. All runs are serial so wall times are honest.
#
# Usage: bash run_baselines.sh
set -u
cd "$(dirname "$0")"

# Override with PY=/path/to/python; Git Bash on Windows uses the Scripts/ venv layout.
if [[ -z ${PY:-} ]]; then
  PY=./py310/bin/python
  [[ -x $PY ]] || PY=./py310/Scripts/python.exe
fi
OUT=output/baseline-summaries
mkdir -p "$OUT"

run_one() { # harness method dataset
  local harness="$1" method="$2" ds="$3"
  local rdir="output/baseline_${method}_${ds}"
  local log="$OUT/${method}_${ds}.txt"
  # already completed (wall_sec is written last) -> skip, so the script is resumable
  if [ -f "$log" ] && grep -q "wall_sec" "$log"; then
    echo "=== $method $ds already done, skip ==="
    return
  fi
  rm -rf "$rdir"
  echo "=== $method $ds start $(date '+%F %T') ==="
  local t0=$SECONDS
  "$PY" "$harness" --method "$method" --dataset "$ds" --result-dir "$rdir" 2>/dev/null \
    | grep -E "AC@1-|AC@3-|AC@5-|Avg@5-|Avg speed" > "$log"
  echo "wall_sec: $((SECONDS - t0))" >> "$log"
  cat "$log"
}

# baro (metric, author's method) — numbers must match the author's tables
for ds in re2-tt re2-ob re2-ss; do run_one main.py baro "$ds"; done

# circa (metric + causal graph): full run where feasible, 1-case probe elsewhere
run_one main.py circa re2-ob
for ds in re2-ss re2-tt; do
  if [ -s "$OUT/circa_probe_${ds}.txt" ] && grep -qE "OK|TIMEOUT" "$OUT/circa_probe_${ds}.txt"; then
    echo "=== circa probe $ds already done, skip ==="
    continue
  fi
  echo "=== circa probe $ds $(date '+%F %T') ==="
  "$PY" probe_circa.py "$ds" 600 > "$OUT/circa_probe_${ds}.txt" 2>&1
  cat "$OUT/circa_probe_${ds}.txt"
done

# trace-based single-source baselines
for ds in re2-tt re2-ob re2-ss; do run_one main_trace.py microrank "$ds"; done
for ds in re2-tt re2-ob re2-ss; do run_one main_trace.py tracerca "$ds"; done

echo "ALL BASELINES DONE $(date '+%F %T')"
