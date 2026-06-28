#!/usr/bin/env bash
# run_petab.sh — runs run_petab.jl for every benchmark model except the warmup. Each model is a
# separate Julia process that benchmarks all PEtab optimizers in series. Result txts go to
# benchmark_results/ (resumable: models with petab_alldone=true are skipped); logs go to
# benchmark_helpers/debugging/logs/. Each worker uses BENCH_CPU_THREADS BLAS threads (Optim +
# Fides/numpy via OMP_NUM_THREADS), so keep max_parallel_workers × BENCH_CPU_THREADS ≤ cores.
#
# Usage (from repo root):
#   bash benchmark_helpers/run_petab.sh [max_parallel_workers]   # default 1
set -u
cd "$(dirname "$0")/.."
RD=benchmark_results/benchmark_results_$(grep -E '^const BENCH_TAG' options.jl | sed -E 's/.*"([^"]*)".*/\1/')
LD=benchmark_helpers/debugging/logs
mkdir -p "$RD" "$LD"
PAR=${1:-1}

# Benchmarked models minus the warmup, and the per-worker thread count, from options.jl.
mapfile -t MODELS < <(julia --project=. -e 'include("options.jl"); foreach(println, filter(!=(BENCH_WARMUP_MODEL), BENCHMARK_MODELS))')
THREADS=$(julia --project=. -e 'include("options.jl"); print(BENCH_CPU_THREADS)')

petab_terminal() {  # $1 = model; returns 0 (skip) once all optimizers are terminal
    local f="$RD/$1_results.txt"
    [ -e "$f" ] || return 1
    [ "$(grep -h '^petab_alldone=' "$f" | cut -d= -f2)" = "true" ]
}

echo "[run_petab] $(date)  PAR=$PAR"
for m in "${MODELS[@]}"; do
    if petab_terminal "$m"; then
        echo "[skip] $m"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
    echo "[run ] $m"
    env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
        julia --project=. -t 1 benchmark_helpers/run_petab.jl "$m" \
        > "$LD/${m}_petab.log" 2>&1 &
done
wait
echo "[run_petab] $(date) ALL DONE"
