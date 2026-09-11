#!/usr/bin/env bash
# Submit the benchmark as a Slurm job array: warmup, then PARALLEL models at a time, then the report.
#   cd benchmark && bash submit_array.sh
set -eu
cd "$(dirname "$0")"
PARALLEL=8
MODELS=($(julia --project=. -e 'include("options.jl"); print(join(filter(!=(WARMUP_MODEL), MODELS), " "))'))
warmup=$(sbatch --parsable -J exa-warmup run_benchmarks_array.sbatch warmup)
models=$(sbatch --parsable -J exa-models --array=1-${#MODELS[@]}%$PARALLEL --dependency=afterok:$warmup run_benchmarks_array.sbatch "${MODELS[@]}")
sbatch -J exa-report --dependency=afterany:$models run_benchmarks_array.sbatch report
echo "warmup $warmup, models $models (${#MODELS[@]} tasks, $PARALLEL at a time)"
