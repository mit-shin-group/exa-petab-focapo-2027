#!/usr/bin/env bash
# run_benchmarks.sh — ONE command to run the benchmark for the configuration set in options.jl.
#
# Configure the run by editing options.jl (which backend via BENCH_INCLUDE_*, the CPU HSL solver, the
# PEtab optimizer + Hessian method, ...) and give it a BENCH_TAG label. This script runs only the
# included backend over the canonical model set (BENCHMARK_MODELS) — keep ONE backend included so its
# solve timing is uncontended — plus the shared JIT-warmup model (Bruno, benchmarked separately) for
# that side, then regenerates the report + figure. The runs we file for the paper:
#   GPU         INCLUDE_EXA_GPU=true                                   ExaModels + MadNLP GPU (CUDSS)
#   CPUma27/57/97  INCLUDE_EXA_CPU=true, BENCH_CPU_SOLVER=Ma27/57/97   ExaModels + MadNLP CPU (HSL)
#   IPNewton    INCLUDE_PETAB=true, OPTIMIZER=Optim.IPNewton()         PEtab.jl (ForwardDiff Hessian)
#   GaussNewton INCLUDE_PETAB=true, OPTIMIZER=Fides.CustomHessian()    PEtab.jl (Gauss–Newton Hessian)
#   BFGS        INCLUDE_PETAB=true, OPTIMIZER=Fides.BFGS()             PEtab.jl (self-approx Hessian)
#
# Per-model results + a _config.toml snapshot -> benchmark_results/benchmark_results_<BENCH_TAG>/
# (resumable: terminal results are skipped). results_table.jl auto-assembles the paper table from the
# separate per-tag dirs (GPU + CPUma57 + IPNewton/GaussNewton/BFGS). ALL settings live in options.jl.
#
#   bash run_benchmarks.sh                    # run the configuration in options.jl
#   CPU_INST=4 bash run_benchmarks.sh         # parallel exa-CPU instances (default 1 = serial)
#
# Timed solves run SERIALLY by default (PEtab PAR=1, exa-CPU CPU_INST=1, exa-GPU 1 instance) so the
# timings are uncontended and the ExaModels-vs-PEtab comparison is fair.
set -u
cd "$(dirname "$0")"
HELP=benchmark_helpers
LOG="$HELP/debugging/logs"

CPU_INST=${CPU_INST:-1}   # serial timed CPU solves by default (fair, uncontended)
# Tag is read from options.jl (the single source of truth — Julia and these scripts agree).
TAG=$(grep -E '^const BENCH_TAG' options.jl | sed -E 's/.*"([^"]*)".*/\1/')
[ -n "$TAG" ] || { echo "[run_benchmarks] Set BENCH_TAG (a non-empty string) in options.jl first."; exit 1; }
mkdir -p "$LOG"

# Per-tag stage gates (single source of truth = options.jl BENCH_INCLUDE_*).
read -r INC_GPU INC_CPU INC_PETAB < <(julia --project=. -e \
  'include("options.jl"); print(BENCH_INCLUDE_EXA_GPU," ",BENCH_INCLUDE_EXA_CPU," ",BENCH_INCLUDE_PETAB)')
julia --project=. "$HELP/snapshot_options.jl"          # creates benchmark_results/benchmark_results_$TAG + its _config.toml
GPU=$(julia --project=. -e 'using CUDA; print(CUDA.functional())' 2>/dev/null || echo false)
echo "[run_benchmarks] $(date)  TAG=$TAG  GPU=$GPU  CPU_INST=$CPU_INST  include[gpu=$INC_GPU cpu=$INC_CPU petab=$INC_PETAB]"

# watchdog-style retry: a model's hard-deadline self-kill exits the worker non-zero (esp. the GPU
# cuDSS hang); just resume — the worker skips already-terminal models. Mirrors run_examodels.sh.
retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

# ── 1. PEtab.jl (CPU) ────────────────────────────────────────────────────────────
if [ "$INC_PETAB" = true ]; then
    echo "[1/5] PEtab.jl ($TAG) ..."
    bash "$HELP/run_petab.sh" 1   # PAR=1: serial, uncontended PEtab solve timing
else
    echo "[1/5] PEtab.jl SKIPPED (tag $TAG)"
fi

# ── 2. ExaModels + MadNLP, GPU (CUDSS) ──────────────────────────────────────────
if [ "$INC_GPU" = true ] && [ "$GPU" = true ]; then
    echo "[2/5] ExaModels GPU ..."
    bash "$HELP/run_examodels.sh" 1   # NINST=1: single GPU instance, uncontended timing
elif [ "$INC_GPU" = true ]; then
    echo "[2/5] ExaModels GPU SKIPPED (no functional GPU)"
else
    echo "[2/5] ExaModels GPU SKIPPED (tag $TAG)"
fi

# ── 3. ExaModels + MadNLP, CPU (HSL) — CPU_INST instances strided over the model set ─
if [ "$INC_CPU" = true ]; then
    echo "[3/5] ExaModels CPU (HSL) x$CPU_INST ..."
    for idx in $(seq 0 $((CPU_INST - 1))); do
        ( retry 100 env BENCH_BACKEND=cpu julia --project=. -t 1 \
            "$HELP/run_examodels.jl" 0 "$CPU_INST" "$idx" ) > "$LOG/exacpu_inst$idx.log" 2>&1 &
    done
    wait
else
    echo "[3/5] ExaModels CPU SKIPPED (tag $TAG)"
fi

# ── 4. Bruno (shared warmup model) — only the side this tag runs ──────────────────
echo "[4/5] Bruno ..."
if [ "$INC_GPU" = true ] && [ "$GPU" = true ]; then
    retry 100 julia --project=. -t 1 "$HELP/run_bruno.jl" Bruno_JExpBot2016 > "$LOG/bruno_gpu.log" 2>&1
fi
if [ "$INC_CPU" = true ]; then
    retry 100 env BENCH_BACKEND=cpu julia --project=. -t 1 "$HELP/run_bruno.jl" Bruno_JExpBot2016 > "$LOG/bruno_cpu.log" 2>&1
fi
if [ "$INC_PETAB" = true ]; then
    retry 100 julia --project=. -t 1 "$HELP/run_bruno.jl" Bruno_JExpBot2016 > "$LOG/bruno_petab.log" 2>&1
fi

# ── 5. Report + figure ──────────────────────────────────────────────────────────
echo "[5/5] table + figure ..."
julia --project=. "$HELP/results_table.jl"
julia --project=. "$HELP/results_plot.jl"

echo "[run_benchmarks] $(date) DONE -> results_table.txt , results_plot.png"
