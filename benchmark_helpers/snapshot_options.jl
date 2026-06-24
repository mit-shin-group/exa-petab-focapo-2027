# snapshot_options.jl — dump the complete, resolved benchmark configuration into the run's results
# dir as _config.toml. Auto-introspects every BENCH_* setting from options.jl, resolving the zero-arg
# solver/optimizer accessors to their objects. Loads the solver packages (this runs once per
# benchmark) so those accessors resolve. Run-level config lives here, not in each *_results.txt.
using CUDA, MadNLP, MadNLPGPU, MadNLPHSL, CUDSS, Optim
include(joinpath(@__DIR__, "..", "options.jl"))
mkpath(RESULTDIR)
tomlval(v) = v isa Bool ? string(v) : v isa Number ? string(v) : "\"" * string(v) * "\""
resolve(v) = v isa Function ? (try v() catch; v end) : v   # call the zero-arg solver/optimizer accessors
open(joinpath(RESULTDIR, "_config.toml"), "w") do io
    println(io, "# benchmark_results_$(BENCH_TAG) — options.jl configuration snapshot (auto-generated)")
    for s in sort(names(Main; all = true))
        startswith(string(s), "BENCH_") || continue
        println(io, string(s), " = ", tomlval(resolve(getfield(Main, s))))
    end
    println(io, "n_benchmark_models = ", length(BENCHMARK_MODELS))
end
println("wrote ", joinpath(RESULTDIR, "_config.toml"))
