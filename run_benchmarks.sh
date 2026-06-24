#!/usr/bin/env bash
# run_benchmarks.sh — ONE command to reproduce the whole benchmark.
#
# Runs, over the canonical model set (options.jl BENCHMARK_MODELS):
#   1. PEtab.jl + Optim.IPNewton            (CPU baseline, parallel workers)
#   2. ExaModelsPEtab + MadNLP, GPU         (CUDSS)            — skipped if no GPU
#   3. ExaModelsPEtab + MadNLP, CPU         (HSL, BENCH_CPU_SOLVER = ma27/ma57/ma97)
#   4. Bruno_JExpBot2016                     (shared JIT-warmup model, benchmarked separately)
# then regenerates the report + figure.
#
# Per-model results + a config.toml snapshot -> benchmark_results_<BENCH_RUN>/ (resumable: terminal
# results are skipped). The run number is set in options.jl (BENCH_RUN); run 0 is the reference.
# Final outputs for the selected run -> results_table.txt , results_plot.png  (repo root).
# ALL settings (tolerances, limits, K, the CPU HSL solver, ...) live in options.jl.
#
#   bash run_benchmarks.sh                  # full suite (GPU auto-detected)
#   BENCH_CPU_SOLVER=ma57 bash run_benchmarks.sh   # swap the CPU HSL solver
#   CPU_INST=8 bash run_benchmarks.sh       # CPU exa parallelism (default 4)
set -u
cd "$(dirname "$0")"
HELP=benchmark_helpers
LOG="$HELP/debugging/logs"

CPU_INST=${CPU_INST:-4}
RUN=$(julia --project=. -e 'include("options.jl"); print(BENCH_RUN)')   # experiment number (options.jl / BENCH_RUN env)
mkdir -p "benchmark_results_$RUN" "$LOG"
julia --project=. "$HELP/snapshot_options.jl"          # write config.toml snapshot into benchmark_results_$RUN
GPU=$(julia --project=. -e 'using CUDA; print(CUDA.functional())' 2>/dev/null || echo false)
echo "[run_benchmarks] $(date)  RUN=$RUN  GPU=$GPU  CPU_INST=$CPU_INST  CPU_SOLVER=${BENCH_CPU_SOLVER:-ma27}"

# watchdog-style retry: a model's hard-deadline self-kill exits the worker non-zero (esp. the GPU
# cuDSS hang); just resume — the worker skips already-terminal models. Mirrors run_examodels.sh.
retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

# ── 1. PEtab.jl baseline (CPU) ──────────────────────────────────────────────────
echo "[1/5] PEtab.jl baseline ..."
bash "$HELP/run_petab.sh"

# ── 2. ExaModels + MadNLP, GPU (CUDSS) ──────────────────────────────────────────
if [ "$GPU" = true ]; then
    echo "[2/5] ExaModels GPU ..."
    bash "$HELP/run_examodels.sh"
else
    echo "[2/5] ExaModels GPU SKIPPED (no functional GPU)"
fi

# ── 3. ExaModels + MadNLP, CPU (HSL) — parallel instances strided over the model set ─
echo "[3/5] ExaModels CPU (HSL) x$CPU_INST ..."
for idx in $(seq 0 $((CPU_INST - 1))); do
    ( retry 100 env BENCH_BACKEND=cpu julia --project=. -t 1 \
        "$HELP/run_examodels.jl" 0 "$CPU_INST" "$idx" ) > "$LOG/exacpu_inst$idx.log" 2>&1 &
done
wait

# ── 4. Bruno (shared warmup model) — GPU pass then CPU pass ──────────────────────
echo "[4/5] Bruno ..."
[ "$GPU" = true ] && retry 100 julia --project=. -t 1 "$HELP/run_bruno.jl" Bruno_JExpBot2016 \
    > "$LOG/bruno_gpu.log" 2>&1
retry 100 env BENCH_BACKEND=cpu julia --project=. -t 1 "$HELP/run_bruno.jl" Bruno_JExpBot2016 \
    > "$LOG/bruno_cpu.log" 2>&1

# ── 5. Report + figure ──────────────────────────────────────────────────────────
echo "[5/5] table + figure ..."
julia --project=. "$HELP/results.jl"
julia --project=. "$HELP/results_plot.jl"

echo "[run_benchmarks] $(date) DONE -> results_table.txt , results_plot.png"
