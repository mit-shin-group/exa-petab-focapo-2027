# Benchmark the warmup model on every backend in RUN_BACKENDS, warmed up by WARMUP_WARMUP_MODEL

include(joinpath(@__DIR__, "common.jl"))

for backend in RUN_BACKENDS
    d = read_result(WARMUP_MODEL)
    get(d, backend * "_stage", "") == "done" && continue
    haskey(d, backend * "_exit_code") && continue
    log = joinpath(@__DIR__, "..", "logs", "$(backend)_$(WARMUP_MODEL).log")
    cmd = `julia --project=$(joinpath(@__DIR__, "..")) -t 1 $(joinpath(@__DIR__, "run_model.jl")) $backend $WARMUP_MODEL $WARMUP_WARMUP_MODEL`
    println("[warmup] $backend $WARMUP_MODEL")
    proc = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
    proc.exitcode == 0 || write_result(WARMUP_MODEL, Dict(backend * "_exit_code" => proc.exitcode))
end
