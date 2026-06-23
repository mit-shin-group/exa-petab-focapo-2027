# Benchmark-Models-PEtab

The 35 model directories in this folder are the complete set of parameter estimation
problems from [The PEtab Benchmark Collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab),
copied here as the benchmark inputs for this repository. Each model is specified in the
[PEtab](https://github.com/PEtab-dev/PEtab) format.

The benchmark scripts consume these models directly (see `Benchmarks/options.jl` for how
the full set is partitioned into the reported model lists).

### References

- The PEtab Benchmark Collection contributors. *The PEtab Benchmark Collection of
  parameter estimation problems.* Version v2026.01.24. 2026.
  https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab
- H. Hass et al., "Benchmark problems for dynamic modeling of intracellular
  processes," *Bioinformatics* 35(17):3073–3082, 2019.
- L. Schmiester et al., "PEtab—Interoperable specification of parameter estimation
  problems in systems biology," *PLOS Comput. Biol.* 17(1):e1008646, 2021.
