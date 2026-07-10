#!/usr/bin/env bash
# run_benchmarks.sh — run the benchmark for the configuration set in options.jl.
#
# Configure the run in options.jl (which backends via BENCH_INCLUDE_*, CPU HSL solver, the PEtab
# optimizer list + hessians) and set a BENCH_TAG label. One run executes every enabled stage in series
# into benchmark_results/benchmark_results_<BENCH_TAG>/, distinguished by key prefix:
#   ExaGPU (exagpu_*) -> ExaCPU (exacpu_*) -> PEtab (petab_<optimizer>_*, each optimizer in series)
# plus the shared warmup model, then regenerates the report + figure. Resumable: terminal results are
# skipped. results_table.jl / results_plot.jl assemble from that one dir.
#
#   bash run_benchmarks.sh                    # run the configuration in options.jl
#   CPU_INST=4 bash run_benchmarks.sh         # parallel exa-CPU instances (default 1 = serial)
#
# Timed solves run serially by default (PEtab PAR=1, exa-CPU CPU_INST=1, exa-GPU 1 instance) for
# uncontended timing. BENCH_CPU_THREADS is the shared CPU BLAS thread budget for every CPU solver:
# exa-CPU Ma57 (madnlp blas_num_threads), PEtab Optim, and Fides/numpy (OMP_NUM_THREADS). Julia -t is
# unused — no path here is Julia-threaded.
set -u
cd "$(dirname "$0")"
HELP=benchmark_helpers
LOG="$HELP/logs"

CPU_INST=${CPU_INST:-1}   # serial timed CPU solves by default
# Tag read from options.jl.
TAG=$(grep -E '^const BENCH_TAG' options.jl | sed -E 's/.*"([^"]*)".*/\1/')
[ -n "$TAG" ] || { echo "[run_benchmarks] Set BENCH_TAG (a non-empty string) in options.jl first."; exit 1; }
mkdir -p "$LOG"

# Stage gates + CPU thread count from options.jl.
read -r INC_GPU INC_CPU INC_PETAB THREADS < <(julia --project=. -e \
  'include("options.jl"); print(BENCH_INCLUDE_EXAGPU," ",BENCH_INCLUDE_EXACPU," ",BENCH_INCLUDE_PETAB," ",BENCH_CPU_THREADS)')
julia --project=. "$HELP/snapshot_options.jl"          # creates benchmark_results_$TAG + its _config.toml
GPU=$(julia --project=. -e 'using CUDA; print(CUDA.functional())' 2>/dev/null || echo false)
echo "[run_benchmarks] $(date)  TAG=$TAG  GPU=$GPU  CPU_INST=$CPU_INST  THREADS=$THREADS  include[gpu=$INC_GPU cpu=$INC_CPU petab=$INC_PETAB]"

# Retry: resume a worker that exited non-zero; already-terminal models are skipped.
retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

# ── 1. PEtab.jl (CPU) ────────────────────────────────────────────────────────────
if [ "$INC_PETAB" = true ]; then
    echo "[1/5] PEtab.jl ($TAG) ..."
    bash "$HELP/run_petab.sh" 1   # PAR=1: serial, uncontended timing
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
        ( retry 100 env BENCH_BACKEND=cpu OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
            julia --project=. -t 1 "$HELP/run_examodels.jl" 0 "$CPU_INST" "$idx" ) > "$LOG/exacpu_inst$idx.log" 2>&1 &
    done
    wait
else
    echo "[3/5] ExaModels CPU SKIPPED (tag $TAG)"
fi

# ── 4. Warmup model (BENCH_WARMUP_MODEL) — only the side this tag runs ─────────────
echo "[4/5] warmup ..."
if [ "$INC_GPU" = true ] && [ "$GPU" = true ]; then
    retry 100 env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
        julia --project=. -t 1 "$HELP/run_warmup.jl" > "$LOG/warmup_gpu.log" 2>&1
fi
if [ "$INC_CPU" = true ]; then
    retry 100 env BENCH_BACKEND=cpu OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
        julia --project=. -t 1 "$HELP/run_warmup.jl" > "$LOG/warmup_cpu.log" 2>&1
fi
if [ "$INC_PETAB" = true ]; then
    retry 100 env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
        julia --project=. -t 1 "$HELP/run_warmup.jl" > "$LOG/warmup_petab.log" 2>&1
fi

# ── 5. Report + figure ──────────────────────────────────────────────────────────
echo "[5/5] table + figure ..."
julia --project=. "$HELP/results_table.jl"
julia --project=. "$HELP/results_plot.jl"

echo "[run_benchmarks] $(date) DONE -> results_table.txt , results_plot.png"
