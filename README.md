# exa-petab-focapo-2027
This repository contains the scripts for reproducing the benchmark results in "REVISITING SIMULTANEOUS METHODS FOR DYNAMIC OPTIMIZATION IN THE GPU ERA" by Joseph W. Choi and Sungho Shin.

## How to run the benchmark
The following hardware/software/licenses are required to run the benchmark:
* an NVIDIA GPU
* [julia](https://julialang.org/downloads/): we recommend [juliaup](https://github.com/JuliaLang/juliaup)
* [libHSL](https://licences.stfc.ac.uk/product/libhsl): a library for sparse linear algebra. After downloading, install `HSL_jll` into the benchmark project with
```
$ make -C benchmark hsl HSL=/full/path/to/HSL_jll.jl
```

Run the benchmark with
```
$ make -C benchmark
```

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
The reference run for the paper is `benchmark_results/benchmark_results_focapo/` (set `BENCH_TAG = "focapo"` in `options.jl`). The `_config.toml` file contains all of the options used.

## Issues
For support, contact [@jsphchoi](https://github.com/jsphchoi).
