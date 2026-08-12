# exa-petab-focapo-2027 — benchmark and paper repo

`benchmark/options.jl` is the source of truth for every setting and model list, and
`run_benchmarks.sh` documents its own staging. Read them. Do not restate them here and do not invent
settings.

Live per-model status lives in `benchmark/benchmark_results/benchmark_results_<TAG>/` and
`results_table.txt`. Never carry pass/fail in notes.

## The one constraint the code cannot express

**Never set both an absolute and a relative tolerance on a PEtab optimizer.** The options are
PEtab.jl's own recommended per-optimizer defaults, overridden only by the wall and iteration caps.
Adding Fides `grtol`/`xtol` or Optim `f_abstol`/`x_reltol` hands PEtab extra early-stop criteria and
unfairly favors it, which is the whole argument of the comparison. The code shows what *is* set; it
cannot show what must never be added. A previous run was corrupted exactly this way and discarded.
