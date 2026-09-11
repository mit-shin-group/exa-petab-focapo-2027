#!/usr/bin/env bash
# Run the benchmark: warmup model, then every model on gpu, cpu and petab, then the report.
#   bash run_benchmarks.sh
#   GPU_ID=1 bash run_benchmarks.sh
#   bash run_benchmarks.sh Model_A Model_B   TODO (REVIEW) only these models, no warmup or report
set -u
cd "$(dirname "$0")"
mkdir -p logs results

read -r BACKENDS THREADS BUILD_LIMIT SOLVE_LIMIT N_RERUNS WARMUP < <(julia --project=. -e '
    include("options.jl")
    print(join(RUN_BACKENDS, ","), " ", CPU_THREADS, " ", Int(BUILD_LIMIT), " ", Int(SOLVE_LIMIT), " ", N_RERUNS, " ", WARMUP_MODEL)')
MODELS=${*:-$(julia --project=. -e 'include("options.jl"); print(join(MODELS, " "))')}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
BACKSTOP=$((BUILD_LIMIT + SOLVE_LIMIT * (1 + N_RERUNS) + 900))

enabled() { [[ ",$BACKENDS," == *",$1,"* ]]; }

# Run one (backend, model) in its own Julia process, skipped when already done
run() {
    local backend=$1 model=$2 warmup=$3 result=results/$model.txt
    [ -f "$result" ] && grep -q "^${backend}_stage=done\|^${backend}_exit_code=" "$result" && return
    echo "[$(date '+%F %T')] $backend $model"
    timeout -s KILL "$BACKSTOP" julia --project=. -t 1 src/run_model.jl "$backend" "$model" "$warmup" \
        > "logs/${backend}_$model.log" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && echo "${backend}_exit_code=$rc" >> "$result"
}

report() { julia --project=. src/results_table.jl && julia --project=. src/results_plot.jl; }

if [ $# -eq 0 ]; then
    echo "[0/4] warmup model $WARMUP"
    julia --project=. src/run_warmup.jl
    report
fi

stage=1
for backend in gpu cpu petab; do
    if enabled "$backend"; then
        echo "[$stage/4] $backend"
        for model in $MODELS; do
            [ "$model" = "$WARMUP" ] || run "$backend" "$model" "$WARMUP"
        done
    else
        echo "[$stage/4] $backend skipped"
    fi
    stage=$((stage + 1))
done

if [ $# -eq 0 ]; then
    echo "[4/4] report"
    report
fi
