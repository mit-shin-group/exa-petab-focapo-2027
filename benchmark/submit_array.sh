#!/usr/bin/env bash
# Submit the benchmark as a Slurm job array: warmup, then PARALLEL models at a time, then the report.
#   cd benchmark && bash submit_array.sh
#   PARTITION=ou_cheme_gpu,mit_preemptable bash submit_array.sh
set -eu
cd "$(dirname "$0")"
PARALLEL=8
PARTITION=${PARTITION:-ou_cheme_gpu}
SB="sbatch --parsable -p $PARTITION --requeue"
MODELS=($(julia --project=. -e 'include("options.jl"); print(join(filter(!=(WARMUP_MODEL), MODELS), " "))'))
warmup=$($SB -J exa-warmup -t 6:00:00 run_benchmarks_array.sbatch warmup)
models=$($SB -J exa-models --array=1-${#MODELS[@]}%$PARALLEL --dependency=afterok:$warmup run_benchmarks_array.sbatch "${MODELS[@]}")
$SB -J exa-report -t 1:00:00 --dependency=afterany:$models run_benchmarks_array.sbatch report
echo "warmup $warmup, models $models (${#MODELS[@]} tasks, $PARALLEL at a time)"
