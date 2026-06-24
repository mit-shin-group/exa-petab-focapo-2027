# exa-petab-focapo-2026
This repository contains the scripts for reproducing the benchmark results in "REVISITING SIMULTANEOUS METHODS FOR DYNAMIC OPTIMIZATION IN THE GPU ERA" by Joseph W. Choi and Sungho Shin.

## How to run the benchmark
First, install Julia (we recommend [juliaup](https://github.com/JuliaLang/juliaup)).

The GPU pass requires an NVIDIA GPU for CUDA + CUDSS. If a compatible GPU is not available, the GPU columns are skipped and the rest of the suite still runs.

The CPU (HSL) runs (`ma27`/`ma57`/`ma97`, set via `BENCH_CPU_SOLVER` in `options.jl`) require a licensed [libHSL](https://licences.stfc.ac.uk/product/libhsl): download `HSL_jll.jl` from STFC and run `julia --project -e 'import Pkg; Pkg.develop(path="path/to/HSL_jll.jl"); Pkg.instantiate()'` before benchmarking. The GPU (CUDSS) and PEtab passes do not need it.

Otherwise, instantiate the pinned dependencies and run:
```
$ julia --project -e 'import Pkg; Pkg.instantiate()'
$ bash run_benchmarks.sh
```
This runs the full suite and writes `results_table.txt` and `results_plot.png`. 

## Benchmark options
All ExaModels.jl, MadNLP.jl, PEtab.jl and its solver options can be configured in `options.jl`.

Here, the benchmark run can also be tagged by changing `BENCH_TAG = <tag>`. 
Each tagged run is stored in `benchmark_results/benchmark_results_<tag>` which contains the following:
- `<model>_results.txt` for every model in the `BENCHMARK_MODELS` set
- `_config.toml` a complete snapshot of the settings used
The benchmark is resumable, so re-running skips models that already have a terminal result.

This tag also selects which run `results_table.txt` and `results_plot.png` are reported from, so different configurations can be kept side by side — e.g. set `BENCH_TAG = "ma57"` and `BENCH_CPU_SOLVER = MadNLPHSL.Ma57Solver` in `options.jl`, then:
```
$ bash run_benchmarks.sh   # -> benchmark_results/benchmark_results_ma57/
```
For tagged runs that are complete, `run_benchmarks.sh` only regenerates the table and plot by pulling the figure options from `/benchmark_helpers`.

## Existing results
The reference run for the paper is `benchmark_results/benchmark_results_focapo/` (set `BENCH_TAG = "focapo"` in `options.jl`), generated with the options chosen for a fair comparison between ExaModels and PEtab.

## Issues
For support, please contact [@sshin23](https://github.com/sshin23).
