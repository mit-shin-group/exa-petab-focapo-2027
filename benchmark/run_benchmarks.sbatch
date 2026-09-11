#!/usr/bin/env bash
# Run the benchmark on one exclusive Slurm GPU node.
#   cd benchmark && sbatch run_benchmarks.sbatch
#SBATCH -J exa-petab
#SBATCH -p ou_cheme_gpu
#SBATCH -G 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -o slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
GPU_ID=0 bash run_benchmarks.sh
