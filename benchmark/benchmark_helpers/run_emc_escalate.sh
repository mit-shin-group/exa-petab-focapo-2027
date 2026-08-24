#!/usr/bin/env bash
# run_emc_escalate.sh: rerun the exa-infeasible models with mesh escalation (BENCH_ESCALATE=1).
#
# For each backend, models whose <prefix>term_status is INFEASIBLE_PROBLEM_DETECTED are reset
# (that prefix only, the prior file kept as <model>_results.txt.pre_escalate) and rerun with the
# subdivide-doubling loop in run_examodels.jl (cap BENCH_SD_CAP). The report regenerates at the
# end. Intended to run after run_emc.sh completes.
#
#   bash benchmark_helpers/run_emc_escalate.sh   # from benchmark/
set -u
cd "$(dirname "$0")/.."
HELP=benchmark_helpers
LOG="$HELP/logs"
RD=benchmark_results/benchmark_results_$(grep -E '^const BENCH_TAG' options.jl | sed -E 's/.*"([^"]*)".*/\1/')
THREADS=$(julia --project=. -e 'include("options.jl"); print(BENCH_CPU_THREADS)')
GPU_ID=1   # matches run_emc.sh
mkdir -p "$LOG"

retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

affected() {  # $1 = prefix (exagpu|exacpu): reset that prefix for infeasible models, list them
    local pfx=$1 out=() f m
    for f in "$RD"/*_results.txt; do
        grep -q "^${pfx}_term_status=INFEASIBLE_PROBLEM_DETECTED" "$f" || continue
        m=$(basename "$f" _results.txt)
        cp -n "$f" "$f.pre_escalate"
        sed -i "/^${pfx}_/d" "$f"
        out+=("$m")
    done
    (IFS=,; echo "${out[*]-}")
}

julia --project=. "$HELP/snapshot_options.jl"   # refresh _config.toml (BENCH_SD_CAP)

GPUM=$(affected exagpu)
if [ -n "$GPUM" ]; then
    echo "[escalate gpu] $GPUM"
    retry 100 env BENCH_ESCALATE=1 BENCH_SUBSET="$GPUM" \
        julia --project=. -t 1 "$HELP/run_examodels.jl" "$GPU_ID" 1 0 > "$LOG/escalate_gpu.log" 2>&1
else
    echo "[escalate gpu] none"
fi

CPUM=$(affected exacpu)
if [ -n "$CPUM" ]; then
    echo "[escalate cpu] $CPUM"
    retry 100 env BENCH_ESCALATE=1 BENCH_BACKEND=cpu BENCH_SUBSET="$CPUM" \
        OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
        julia --project=. -t 1 "$HELP/run_examodels.jl" 0 1 0 > "$LOG/escalate_cpu.log" 2>&1
else
    echo "[escalate cpu] none"
fi

julia --project=. "$HELP/results_table.jl"
julia --project=. "$HELP/results_plot.jl"
echo "[escalate] DONE $(date)"
