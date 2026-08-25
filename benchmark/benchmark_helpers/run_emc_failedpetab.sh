#!/usr/bin/env bash
# run_emc_failedpetab.sh: exa GPU + CPU passes for FAILED_MODELS (PEtab.jl cannot compile them,
# so their rows are exa-only with no reference objective). Escalation on, small models first.
# Intended to run after run_emc.sh and run_emc_escalate.sh.
#
#   bash benchmark_helpers/run_emc_failedpetab.sh   # from benchmark/
set -u
cd "$(dirname "$0")/.."
HELP=benchmark_helpers
LOG="$HELP/logs"
THREADS=$(julia --project=. -e 'include("options.jl"); print(BENCH_CPU_THREADS)')
GPU_ID=1   # matches run_emc.sh
MODELS=Raia_CancerResearch2011,Lang_PLOSComputBiol2024,Froehlich_CellSystems2018
mkdir -p "$LOG"

retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

echo "[failedpetab gpu] $MODELS"
retry 100 env BENCH_ESCALATE=1 BENCH_SUBSET="$MODELS" \
    julia --project=. -t 1 "$HELP/run_examodels.jl" "$GPU_ID" 1 0 > "$LOG/failedpetab_gpu.log" 2>&1

echo "[failedpetab cpu] $MODELS"
retry 100 env BENCH_ESCALATE=1 BENCH_BACKEND=cpu BENCH_SUBSET="$MODELS" \
    OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    julia --project=. -t 1 "$HELP/run_examodels.jl" 0 1 0 > "$LOG/failedpetab_cpu.log" 2>&1

julia --project=. "$HELP/results_table.jl"
julia --project=. "$HELP/results_plot.jl"
echo "[failedpetab] DONE $(date)"
