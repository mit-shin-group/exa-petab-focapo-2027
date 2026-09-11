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
make -C benchmark
```

## Issues
For support, contact [@jsphchoi](https://github.com/jsphchoi).
