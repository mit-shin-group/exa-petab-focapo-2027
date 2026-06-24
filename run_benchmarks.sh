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
# Per-model results + a _config.toml snapshot -> benchmark_results/benchmark_results_<BENCH_TAG>/
# (resumable: terminal results are skipped). The run tag is set in options.jl (BENCH_TAG); focapo is the paper reference.
# Final outputs for the selected run -> results_table.txt , results_plot.png  (repo root).
# ALL settings (tolerances, limits, K, the CPU HSL solver, ...) live in options.jl.
#
#   bash run_benchmarks.sh                  # full suite (GPU auto-detected); solvers/tag set in options.jl
#   CPU_INST=4 bash run_benchmarks.sh       # opt into parallel exa-CPU (default 1 = serial)
#
# Timed solves run SERIALLY by default (PEtab PAR=1, exa-CPU CPU_INST=1, exa-GPU 1 instance) so the
# CPU timings are uncontended and the ExaModels-vs-PEtab comparison is fair. Raise the knobs for
# throughput at the cost of timing accuracy.
set -u
cd "$(dirname "$0")"
HELP=benchmark_helpers
LOG="$HELP/debugging/logs"

CPU_INST=${CPU_INST:-1}   # serial timed CPU solves by default (fair, uncontended)
TAG=$(grep -E '^const BENCH_TAG' options.jl | sed -E 's/.*"([^"]*)".*/\1/')   # run tag (from options.jl)
[ -n "$TAG" ] || { echo "[run_benchmarks] Set BENCH_TAG (a non-empty string) in options.jl first."; exit 1; }
mkdir -p "$LOG"
julia --project=. "$HELP/snapshot_options.jl"          # creates benchmark_results/benchmark_results_$TAG + its _config.toml
GPU=$(julia --project=. -e 'using CUDA; print(CUDA.functional())' 2>/dev/null || echo false)
echo "[run_benchmarks] $(date)  TAG=$TAG  GPU=$GPU  CPU_INST=$CPU_INST  (solvers in options.jl / _config.toml)"

# watchdog-style retry: a model's hard-deadline self-kill exits the worker non-zero (esp. the GPU
# cuDSS hang); just resume — the worker skips already-terminal models. Mirrors run_examodels.sh.
retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

# ── 1. PEtab.jl baseline (CPU) ──────────────────────────────────────────────────
echo "[1/5] PEtab.jl baseline ..."
bash "$HELP/run_petab.sh" 1   # PAR=1: serial, uncontended PEtab solve timing

# ── 2. ExaModels + MadNLP, GPU (CUDSS) ──────────────────────────────────────────
if [ "$GPU" = true ]; then
    echo "[2/5] ExaModels GPU ..."
    bash "$HELP/run_examodels.sh" 1   # NINST=1: single GPU instance, uncontended timing
else
    echo "[2/5] ExaModels GPU SKIPPED (no functional GPU)"
fi

# ── 3. ExaModels + MadNLP, CPU (HSL) — CPU_INST instances strided over the model set (default 1 = serial) ─
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
julia --project=. "$HELP/results_table.jl"
julia --project=. "$HELP/results_plot.jl"

echo "[run_benchmarks] $(date) DONE -> results_table.txt , results_plot.png"
