#!/usr/bin/env bash
# run_petab.sh — runs run_petab.jl for every benchmark model (minus the Bruno
# JIT warmup, which run_bruno.jl covers) in parallel. Each model is a separate Julia
# process. Result txts go to benchmark_results/ (resumable — models with a terminal petab
# result are skipped); per-model run logs go to benchmark_helpers/debugging/logs/.
#
# The MODELS array below is the canonical BENCHMARK_MODELS list (options.jl) minus Bruno.
#
# Usage (from repo root):
#   bash benchmark_helpers/run_petab.sh [max_parallel_workers]
# Default: 12 concurrent workers.
set -u
cd "$(dirname "$0")/.."
RD=benchmark_results
LD=benchmark_helpers/debugging/logs
mkdir -p "$RD" "$LD"
PAR=${1:-12}

# Canonical benchmarked set (options.jl) minus the shared JIT warmup (Bruno, run via run_bruno.jl).
mapfile -t MODELS < <(julia --project=. -e 'include("options.jl"); foreach(println, filter(!=(BENCH_WARMUP_MODEL), BENCHMARK_MODELS))')

petab_terminal() {  # $1 = model; returns 0 (skip) if petab result is terminal
    local f="$RD/$1_results.txt"
    [ -e "$f" ] || return 1
    local cs ss
    cs=$(grep -h "^petab_compile_status=" "$f" | cut -d= -f2)
    ss=$(grep -h "^petab_solve_status="   "$f" | cut -d= -f2)
    case "$cs" in
        ok)    case "$ss" in ok|error|timeout) return 0;; *) return 1;; esac ;;
        error|missing_yaml) return 0 ;;
        *) return 1 ;;
    esac
}

echo "[run_petab] $(date)  PAR=$PAR"
for m in "${MODELS[@]}"; do
    if petab_terminal "$m"; then
        echo "[skip] $m"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
    echo "[run ] $m"
    julia --project=. -t 1 benchmark_helpers/run_petab.jl "$m" \
        > "$LD/${m}_petab.log" 2>&1 &
done
wait
echo "[run_petab] $(date) ALL DONE"
