#!/usr/bin/env bash
# Run the benchmark on one H200 of a Slurm GPU node.
#   cd benchmark && sbatch run_benchmarks.sbatch
#SBATCH -J exa-petab
#SBATCH -p ou_cheme_gpu
#SBATCH -G h200:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 7-00:00:00
#SBATCH -o slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
GPU_ID=0 bash run_benchmarks.sh
