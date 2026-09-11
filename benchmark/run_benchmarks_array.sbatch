#!/usr/bin/env bash
# One Slurm job of the array benchmark: `warmup`, `report`, or the model picked by SLURM_ARRAY_TASK_ID.
#   submitted by submit_array.sh
#SBATCH -p ou_cheme_gpu
#SBATCH -G h200:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out
cd "$SLURM_SUBMIT_DIR" && mkdir -p logs results
case $1 in
    warmup) julia --project=. src/run_warmup.jl ;;
    report) julia --project=. src/results_table.jl && julia --project=. src/results_plot.jl ;;
    *)      bash run_benchmarks.sh "${!SLURM_ARRAY_TASK_ID}" ;;
esac
