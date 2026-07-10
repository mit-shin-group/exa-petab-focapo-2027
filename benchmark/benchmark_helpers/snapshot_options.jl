using CUDA, MadNLP, MadNLPGPU, CUDSS, Optim, Fides

try
    using MadNLPHSL
catch
    @warn "MadNLPHSL not installed; CPU/HSL solver will be recorded unresolved in _config.toml"
end
include(joinpath(@__DIR__, "..", "options.jl"))
mkpath(RESULTDIR)
tomlval(v) = v isa Bool ? string(v) : v isa Number ? string(v) : "\"" * string(v) * "\""
resolve(v) = v isa Function ? (try v() catch; v end) : v
open(joinpath(RESULTDIR, "_config.toml"), "w") do io
    println(io, "# benchmark_results_$(BENCH_TAG) — options.jl configuration snapshot (auto-generated)")
    for s in sort(names(Main; all = true))
        startswith(string(s), "BENCH_") || continue
        println(io, string(s), " = ", tomlval(resolve(getfield(Main, s))))
    end
    println(io, "n_benchmark_models = ", length(BENCHMARK_MODELS))
end
println("wrote ", joinpath(RESULTDIR, "_config.toml"))
