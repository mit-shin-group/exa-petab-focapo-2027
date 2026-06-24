# snapshot_options.jl — dump the complete, resolved benchmark configuration into the run's results
# dir as config.toml. Auto-introspects every BENCH_* setting from options.jl (env overrides included),
# so the snapshot is complete by construction. This is where run-level config lives — it is no longer
# duplicated into each per-model *_results.txt.
include(joinpath(@__DIR__, "..", "options.jl"))
mkpath(RESULTDIR)
tomlval(v) = v isa Bool ? string(v) : v isa Number ? string(v) : "\"" * string(v) * "\""
open(joinpath(RESULTDIR, "config.toml"), "w") do io
    println(io, "# benchmark_results_$(BENCH_RUN) — options.jl configuration snapshot (auto-generated)")
    for s in sort(names(Main; all = true))
        startswith(string(s), "BENCH_") || continue
        println(io, string(s), " = ", tomlval(getfield(Main, s)))
    end
    println(io, "n_benchmark_models = ", length(BENCHMARK_MODELS))
end
println("wrote ", joinpath(RESULTDIR, "config.toml"))
