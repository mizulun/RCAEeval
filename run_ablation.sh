#!/usr/bin/env bash
# Ablation study runner for the gabanzhaf method.
#
# Runs each ablation mode on each dataset and prints AC@1/AC@3/AC@5/Avg@5 per
# fault type. Clears output/results before every run (the evaluator globs the
# whole folder, so stale files would pollute results).
#
# Usage:
#   bash run_ablation.sh                         # all modes, all RE2 datasets
#   bash run_ablation.sh --datasets re2-tt       # one dataset
#   bash run_ablation.sh --modes full no-metric  # selected modes
#   DATASETS="re2-tt re2-ob" MODES="full no-trace no-metric" bash run_ablation.sh

set -u
cd "$(dirname "$0")"

PY=./py310/bin/python
DATASETS="${DATASETS:-re2-tt re2-ob re2-ss}"
# Modes: full | no-trace | no-metric | metric-only | metric+trace | no-coverage
MODES="${MODES:-full no-trace no-metric}"

# Where summaries are saved: one file per dataset+mode, plus a combined log
# stamped with the run time so old runs are not overwritten.
OUTDIR="${OUTDIR:-ablation_results}"
STAMP="$(date +%Y%m%d-%H%M%S)"
COMBINED="$OUTDIR/ablation_$STAMP.txt"
mkdir -p "$OUTDIR"

# parse simple flags
while [ $# -gt 0 ]; do
  case "$1" in
    --datasets) shift; DATASETS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do DATASETS="$DATASETS $1"; shift; done ;;
    --modes)    shift; MODES="";    while [ $# -gt 0 ] && [[ "$1" != --* ]]; do MODES="$MODES $1"; shift; done ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# mode -> environment variables that select it
mode_env() {
  case "$1" in
    full)         echo "" ;;
    no-trace)     echo "GABANZHAF_DISABLE_TRACE=1" ;;
    no-metric)    echo "GABANZHAF_RANK_MODE=trace_evidence" ;;
    metric-only)  echo "GABANZHAF_RANK_MODE=service_evidence" ;;
    metric+trace) echo "GABANZHAF_RANK_MODE=service_evidence_trace" ;;
    no-coverage)  echo "GABANZHAF_DISABLE_COVERAGE=1" ;;
    *) echo "BADMODE" ;;
  esac
}

for ds in $DATASETS; do
  echo "############################################################"
  echo "# DATASET: $ds"
  echo "############################################################"
  for mode in $MODES; do
    env_vars="$(mode_env "$mode")"
    if [ "$env_vars" = "BADMODE" ]; then
      echo ">> skipping unknown mode: $mode"; continue
    fi
    rm -rf output/results && mkdir -p output/results
    per_file="$OUTDIR/${ds}__${mode}.txt"
    header="----- dataset: $ds   mode: $mode   [${env_vars:-default}]   ($STAMP) -----"
    echo "$header" | tee "$per_file" | tee -a "$COMBINED"
    t0=$(date +%s)
    env $env_vars "$PY" main.py --method gabanzhaf --dataset "$ds" \
        --result-dir output/results 2>/dev/null \
      | grep -E "AC@1-|AC@3-|AC@5-|Avg@5-" | sed 's/^/   /' \
      | tee -a "$per_file" | tee -a "$COMBINED"
    t1=$(date +%s)
    echo "   [wall: $((t1 - t0))s]" | tee -a "$per_file" | tee -a "$COMBINED"
  done
done
echo "DONE"
echo "Saved: per-mode files in $OUTDIR/<dataset>__<mode>.txt ; combined log: $COMBINED"
