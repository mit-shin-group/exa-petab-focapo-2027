# exa-petab-focapo-2026
This repository stores the scripts for reproducing the benchmark results in "REVISITING SIMULTANEOUS METHODS FOR DYNAMIC OPTIMIZATION IN THE GPU ERA" by Joseph W. Choi and Sungho Shin.

## How to run the script?
First, install Julia (we recommend [juliaup](https://github.com/JuliaLang/juliaup)).

The benchmark assumes an NVIDIA GPU — the GPU pass uses CUDA + CUDSS. Without one, the GPU columns are skipped and the rest of the suite still runs.

The CPU solver uses the HSL linear solvers through `MadNLPHSL`. The default `ma27` works out of the box from the freely-redistributable HSL subset bundled by `HSL_jll`. To benchmark with `ma57`/`ma97` instead (set `BENCH_CPU_SOLVER` in `options.jl`), you additionally need a full [libHSL](https://licences.stfc.ac.uk/product/libhsl) build; point the project at it before instantiating:
```
$ julia --project -e 'import Pkg; Pkg.develop(path="path/to/HSL_jll"); Pkg.instantiate()'
```

Otherwise, just instantiate the pinned dependencies and run:
```
$ julia --project -e 'import Pkg; Pkg.instantiate()'
$ bash run_benchmarks.sh
```
This runs the full suite and writes `results_table.txt` and `results_plot.png`. All settings — tolerances, wall/compile limits, mesh size `K`, and the CPU solver — live in `options.jl`.

## Experiments and existing data
Each experiment is a numbered run. `BENCH_RUN` in `options.jl` (or the `BENCH_RUN` env var) selects the run: the benchmark reads/writes its per-model `*_results.txt` under `benchmark_results_<n>/`, alongside an auto-generated `config.toml` snapshotting every setting used. The same `BENCH_RUN` chooses which run `results_table.txt` / `results_plot.png` are generated from. This makes it easy to keep, e.g., separate runs for different CPU solvers:
```
$ BENCH_RUN=1 BENCH_CPU_SOLVER=ma27 bash run_benchmarks.sh   # -> benchmark_results_1/
$ BENCH_RUN=2 BENCH_CPU_SOLVER=ma57 bash run_benchmarks.sh   # -> benchmark_results_2/
```
We provide our results in `benchmark_results_0/` (the reference run). The benchmark is resumable, so re-running skips models that already have a terminal result.

## Issues
For support, please contact [@sshin23](https://github.com/sshin23).
