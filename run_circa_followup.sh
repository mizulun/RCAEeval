#!/usr/bin/env bash
# Follow-up: probes showed circa IS feasible on re2-ss/re2-tt, so run them in
# full. Waits for run_baselines.sh to finish first (keeps all runs serial so
# wall times stay honest). Resumable like run_baselines.sh.
set -u
cd "$(dirname "$0")"
PY=./py310/bin/python

while pgrep -f "bash run_baselines.sh" > /dev/null; do sleep 60; done

for ds in re2-ss re2-tt; do
  log="baseline_results/circa_${ds}.txt"
  if [ -f "$log" ] && grep -q "wall_sec" "$log"; then
    echo "=== circa $ds already done, skip ==="
    continue
  fi
  rdir="output/baseline_circa_${ds}"
  rm -rf "$rdir"
  echo "=== circa $ds start $(date '+%F %T') ==="
  t0=$SECONDS
  "$PY" main.py --method circa --dataset "$ds" --result-dir "$rdir" 2>/dev/null \
    | grep -E "AC@1-|AC@3-|AC@5-|Avg@5-|Avg speed" > "$log"
  echo "wall_sec: $((SECONDS - t0))" >> "$log"
  cat "$log"
done
echo "CIRCA FOLLOWUP DONE $(date '+%F %T')"
