# Shared helpers for the benchmark scripts

include(joinpath(@__DIR__, "..", "options.jl"))

get_yaml(model) = (d = joinpath(MODELDIR, model); joinpath(d, first(filter(f -> endswith(f, ".yaml"), readdir(d)))))
result_path(model) = joinpath(RESULTDIR, model * ".txt")

# Shifted geometric mean of times [s]
t_sgmdelta(times, δ = SGM_SHIFT) = exp(sum(t -> log(t + δ), times) / length(times)) - δ

# Read a result file into a Dict
function read_result(model)
    d = Dict{String, String}()
    isfile(result_path(model)) || return d
    for line in eachline(result_path(model))
        i = findfirst('=', line)
        i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    return d
end

# Merge updates into the result file, keys sorted
function write_result(model, updates)
    d = merge(read_result(model), Dict(string(k) => replace(string(v), '\n' => ' ') for (k, v) in updates))
    mkpath(RESULTDIR)
    open(result_path(model), "w") do io
        for k in sort(collect(keys(d)))
            println(io, k, "=", d[k])
        end
    end
end

# Run f, SIGKILL the process if it takes longer than seconds
function with_hard_deadline(f, seconds)
    watchdog = run(`bash -c "sleep $seconds; kill -9 $(getpid())"`; wait = false)
    try
        return f()
    finally
        kill(watchdog)
    end
end

# Estimated parameter ids in parameters-table order, the theta order of ExaModelsPEtab
function theta_ids(yaml)
    petab = ExaModelsPEtab._parse_yaml(yaml)
    return [Symbol(p.parameter_id) for p in petab.parameters if p.estimate]
end

# Objective at theta by a forward ODE solve, NaN on failure
function objective_at(yaml, theta)
    try
        return evaluate_objective(yaml, theta)
    catch e
        @warn "evaluate_objective failed" exception = (e, catch_backtrace())
        return NaN
    end
end

# Parse a CSV of floats from the result file
parse_times(s) = [parse(Float64, t) for t in split(s, ","; keepempty = false)]
