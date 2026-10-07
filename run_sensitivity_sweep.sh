#!/usr/bin/env bash
# Post-hoc hyperparameter sensitivity sweep for the merit method.
#
# NOT a hyperparameter selection procedure.  The defaults are already fixed in
# RCAEval/e2e/merit.py:_GabanzhafConfig; this script measures how much AC@1 /
# Avg@5 move in a neighbourhood around each default, so the thesis can report
# robustness instead of claiming the values were tuned.
#
# Each run gets its own --result-dir, so runs are isolated from each other and
# from output/results.  That is what makes the -P parallelism safe: the
# evaluator globs a whole result folder, and a shared folder would mix runs.
#
# Usage:
#   bash run_sensitivity_sweep.sh                        # all params, all RE2 datasets
#   bash run_sensitivity_sweep.sh --datasets re2-tt      # one dataset
#   bash run_sensitivity_sweep.sh --params victim_floor  # one parameter
#   JOBS=1 bash run_sensitivity_sweep.sh                 # serial
#   PARAMS="canary" bash run_sensitivity_sweep.sh --datasets re2-tt   # plumbing check
#
# One re2-tt run measured 285-293s wall (2026-08-11, 16-core darwin), so the
# full 6-parameter x 3-dataset sweep is ~87 runs; at JOBS=6 expect ~1h.

set -u
cd "$(dirname "$0")"

PY=./py310/bin/python
DATASETS="${DATASETS:-re2-tt re2-ob re2-ss}"
PARAMS="${PARAMS:-top_k_svcs exact_candidate_k coverage_conf coverage_loud_thr victim_floor trace_topology_conf}"
TEST_ARGS="${TEST_ARGS:-}"
JOBS="${JOBS:-6}"

OUTDIR="${OUTDIR:-output/sensitivity-results}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUNROOT="${RUNROOT:-output/sensitivity_runs/$STAMP}"

# Workers are re-invoked as "$0 <dataset> <param> <value>", so they must not
# fall into the flag parser.
if [ "${SWEEP_WORKER:-0}" != "1" ]; then
while [ $# -gt 0 ]; do
  case "$1" in
    --datasets) shift; DATASETS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do DATASETS="$DATASETS $1"; shift; done ;;
    --params)   shift; PARAMS="";   while [ $# -gt 0 ] && [[ "$1" != --* ]]; do PARAMS="$PARAMS $1"; shift; done ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done
fi

# param -> environment variable read by merit() (RCAEval/e2e/merit.py:1799-1879)
param_env_name() {
  case "$1" in
    top_k_svcs)          echo "GABANZHAF_TOP_K_SVCS" ;;
    exact_candidate_k)   echo "GABANZHAF_EXACT_CANDIDATE_K" ;;
    coverage_conf)       echo "GABANZHAF_COVERAGE_CONF" ;;
    coverage_loud_thr)   echo "GABANZHAF_COVERAGE_LOUD_THR" ;;
    victim_floor)        echo "GABANZHAF_VICTIM_FLOOR" ;;
    trace_topology_conf) echo "GABANZHAF_TRACE_TOPOLOGY_CONF" ;;
    canary)              echo "GABANZHAF_TOP_K_SVCS" ;;
    *) echo "BADPARAM" ;;
  esac
}

# Sweep points.  The current default is included in every list so the default
# row and its neighbourhood come from the same sweep.
param_values() {
  case "$1" in
    top_k_svcs)          echo "10 12 15 20 25" ;;
    exact_candidate_k)   echo "6 8 10 12 15" ;;
    coverage_conf)       echo "0.5 0.6 0.7 0.8 0.9" ;;
    coverage_loud_thr)   echo "0.3 0.4 0.5 0.6" ;;
    victim_floor)        echo "0.3 0.4 0.5 0.6 0.7" ;;
    trace_topology_conf) echo "0.7 0.8 0.85 0.9 0.95" ;;
    # Plumbing check only: 1 vs 15 must NOT produce identical numbers, otherwise
    # the environment variable never reached _GabanzhafConfig.
    canary)              echo "1 15" ;;
    *) echo "BADPARAM" ;;
  esac
}

# Worker: run one (dataset, param, value) and write its own result file.
# Invoked via xargs, so it has to be re-enterable from a fresh shell.
if [ "${SWEEP_WORKER:-0}" = "1" ]; then
  ds="$1"; p="$2"; v="$3"
  var="$(param_env_name "$p")"
  rdir="$RUNROOT/${ds}__${p}__${v}"
  out="$OUTDIR/${ds}__${p}__${v}.txt"
  rm -rf "$rdir"; mkdir -p "$rdir"
  {
    echo "----- dataset: $ds   $p=$v   [$var=$v]   ($STAMP) -----"
    t0=$(date +%s)
    env "$var=$v" "$PY" main.py --method merit --dataset "$ds" \
        --result-dir "$rdir" $TEST_ARGS 2>/dev/null \
      | grep -E "AC@1-|AC@3-|AC@5-|Avg@5-" | sed 's/^/   /'
    t1=$(date +%s)
    echo "   [wall: $((t1 - t0))s]"
  } > "$out"
  echo "done: $out"
  exit 0
fi

mkdir -p "$OUTDIR" "$RUNROOT"
JOBFILE="$OUTDIR/jobs_$STAMP.txt"
: > "$JOBFILE"
for ds in $DATASETS; do
  for p in $PARAMS; do
    if [ "$(param_env_name "$p")" = "BADPARAM" ]; then
      echo ">> skipping unknown param: $p"; continue
    fi
    for v in $(param_values "$p"); do
      echo "$ds $p $v" >> "$JOBFILE"
    done
  done
done

n=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "sweep: $n runs, JOBS=$JOBS, results in $OUTDIR (stamp $STAMP)"
export SWEEP_WORKER=1 OUTDIR RUNROOT STAMP PY TEST_ARGS
# xargs does not search the working directory, so pass an absolute path.
xargs -P "$JOBS" -n 3 bash "$PWD/$(basename "$0")" < "$JOBFILE"

COMBINED="$OUTDIR/sensitivity_$STAMP.txt"
: > "$COMBINED"
while read -r ds p v; do
  cat "$OUTDIR/${ds}__${p}__${v}.txt" >> "$COMBINED"
done < "$JOBFILE"
echo "combined log: $COMBINED"
