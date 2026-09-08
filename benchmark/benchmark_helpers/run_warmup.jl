# run_warmup.jl — benchmarks the shared JIT-warmup model BENCH_WARMUP_MODEL (options.jl).
#
# That model is the JIT warmup for run_examodels.jl and run_petab.jl, so it is excluded from their
# timed runs and benchmarked here instead, JIT-warmed on a different model for a clean compile time.
# Backend follows BENCH_BACKEND ("gpu" -> exagpu_*, "cpu" -> exacpu_*); the PEtab side is
# backend-independent. Runs only the side the current tag covers (BENCH_INCLUDE_* in options.jl).
#   julia --project=. -t 1 benchmark_helpers/run_warmup.jl [gpu_id]
#   BENCH_BACKEND=cpu julia --project=. -t 1 benchmark_helpers/run_warmup.jl

using ExaModelsPEtab, CUDA, MadNLP, MadNLPGPU, CUDSS, ExaModels, LinearAlgebra
# HSL is a licensed, user-supplied dependency (see README); only the CPU pass needs it.
if lowercase(get(ENV, "BENCH_BACKEND", "gpu")) == "cpu"
    using MadNLPHSL
end

# ─── CONFIGURABLE SETTINGS (single source of truth = options.jl) ──────────────────
include(joinpath(@__DIR__, "..", "options.jl"))   # MODELDIR + RESULTDIR + model sets + BENCH_* config

const TARGET    = BENCH_WARMUP_MODEL   # the warmup model, benchmarked here
const JIT_MODEL = TARGET == "Crauste_CellSystems2017" ? "Bruno_JExpBot2016" : "Crauste_CellSystems2017"  # JIT-warm on a different model

const TOL           = BENCH_TOL
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N
const SGM_SHIFT     = BENCH_SGM_SHIFT
const ACCEPT_TOL    = BENCH_ACCEPT_TOL
const ACCEPT_ITER   = BENCH_ACCEPT_ITER

const BACKEND = lowercase(get(ENV, "BENCH_BACKEND", "gpu"))
const IS_GPU  = BACKEND != "cpu"
const PFX     = IS_GPU ? "exagpu_" : "exacpu_"
const KKT_OPTS = (kkt_system = BENCH_KKT_SYSTEM(),
                  equality_treatment = BENCH_EQUALITY_TREATMENT(),
                  fixed_variable_treatment = BENCH_FIXED_VAR_TREATMENT())
const LINEAR_SOLVER = IS_GPU ? BENCH_GPU_SOLVER() : BENCH_CPU_SOLVER()
solve_madnlp(model) = madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER,
    max_iter=MAX_ITER, max_wall_time=SOLVE_LIMIT, linear_solver=LINEAR_SOLVER,
    blas_num_threads = IS_GPU ? 1 : BENCH_CPU_THREADS, KKT_OPTS...)
gpu_reclaim() = IS_GPU && (GC.gc(); CUDA.reclaim())
# ────────────────────────────────────────────────────────────────────────────────

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end
result_path(m) = joinpath(RESULTDIR, "$(m)_results.txt")

function read_result(path)
    d = Dict{String,String}(); isfile(path) || return d
    for line in eachline(path)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    return d
end
function write_result(path, updates)
    existing = read_result(path)
    merged = merge(existing, Dict(string(k) => replace(string(v), '\n' => ' ', '\r' => ' ')
                                  for (k, v) in updates))
    open(path, "w") do io
        for k in sort(collect(keys(merged))); println(io, "$k=", merged[k]); end
        flush(io)
    end
end
function with_hard_deadline(f, seconds::Real)
    pid = getpid()
    w = run(`bash -c "sleep $(seconds); kill -9 $(pid)"`; wait=false)
    try; return f(); finally; try; kill(w); catch; end; end
end
# inf-norm (max) constraint violation max(lcon - c(x), c(x) - ucon, 0) at point x; 0 if unconstrained.
function max_constr_viol(model, x)
    model.meta.ncon == 0 && return 0.0
    c = similar(x, model.meta.ncon); ExaModels.cons!(model, x, c)
    c = Array(c); lc = Array(model.meta.lcon); uc = Array(model.meta.ucon)
    maximum(max.(lc .- c, c .- uc, 0.0))
end

# ─── ExaModels build (backend-parametrized) ──────────────
# ─── build one ExaModel ──
# The public API call, timed whole: there is no PEtab.jl setup phase left to separate out.
function build_model(yaml)
    mdl = examodel_petab(yaml; backend = IS_GPU ? CUDA.CUDABackend() : nothing)
    IS_GPU && CUDA.synchronize()
    return mdl, mdl.meta.nvar, mdl.meta.ncon
end

is_converged(solve_status) = uppercase(solve_status) in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")

# theta at the solution, the vector evaluate_objective takes
function theta_star(model, solution)
    haskey(model.refs, :theta) || return Float64[]
    theta = model.refs.theta
    return Array(solution)[theta.offset .+ (1:theta.length)]
end

# Draw warm reruns until N_SGM_RERUNS of them converge, allowing one extra attempt.
function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict(PFX*"sgm_status" => "running"))
    solve_times = Float64[]; iters = Int[]; solve_statuses = String[]; objectives = Float64[]
    max_attempts = N_SGM_RERUNS + 1
    while length(solve_times) < max_attempts && count(is_converged, solve_statuses) < N_SGM_RERUNS
        attempt = length(solve_times) + 1
        @info "[$m] SGM solve $attempt/$max_attempts ..."
        try
            IS_GPU && GC.gc()     # clear dead GPU allocs OUTSIDE timing; stops a GC/allocator stall landing in a timed solve (keeps pool+kernels warm; no reclaim)
            t0 = time(); res = solve_madnlp(model); push!(solve_times, time() - t0)
            push!(iters, res.iter); push!(solve_statuses, string(res.status)); push!(objectives, res.objective)
        catch e
            @error "[$m] SGM solve $attempt failed" exception=(e, catch_backtrace())
            write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e))); return
        end
    end
    converged = is_converged.(solve_statuses)
    converged_times = solve_times[converged]
    out = Dict{String,Any}(PFX*"sgm_status"       => "ok",
                           PFX*"solve_times"      => join(converged_times, ","),
                           PFX*"sgm_times_all"    => join(solve_times, ","),
                           PFX*"sgm_iters"        => join(iters, ","),
                           PFX*"sgm_solve_status" => join(solve_statuses, ","),
                           PFX*"sgm_objectives"   => join(objectives, ","),
                           PFX*"sgm_n"            => "$(count(converged))/$(length(solve_times))")
    if isempty(converged_times)
        out[PFX*"sgm_status"] = "nonconverged"
        out[PFX*"sgm_error"]  = "0 of $(length(solve_times)) reruns converged, no SGM"
        @info "[$m] SGM: 0/$(length(solve_times)) reruns converged, no SGM"
    else
        sorted_times = sort(converged_times)
        out[PFX*"sgm_solve_time"] = t_sgmdelta(converged_times, SGM_SHIFT)
        out[PFX*"sgm_median"]     = sorted_times[cld(length(sorted_times), 2)]
        out[PFX*"sgm_min"]        = sorted_times[1]
        out[PFX*"sgm_max"]        = sorted_times[end]
        @info "[$m] SGM done: solve=$(out[PFX*"sgm_solve_time"]) s over $(out[PFX*"sgm_n"]) converged reruns"
    end
    write_result(rp, out)
end

# ─── ExaModels benchmark for the target (compile + first solve + SGM) ───────────
function bench_exa(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    yaml === nothing && (write_result(rp, Dict(PFX*"compile_status" => "missing_yaml", PFX*"solve_status" => "skipped")); return)
    model = nothing

    write_result(rp, Dict(
        PFX*"compile_status" => "compiling", PFX*"compile_time" => "",
        PFX*"solve_status"   => "skipped",   PFX*"solve_time"   => "", PFX*"term_status"   => "",
        PFX*"objective"      => "",          PFX*"iter"         => "",
        PFX*"nvar"           => "",          PFX*"ncon"         => "", PFX*"error" => "",
        PFX*"constr_viol"    => "",          PFX*"theta_star"   => "",
    ))
    @info "[$m] EXA compiling on $BACKEND ..."
    try
        t0 = time()
        mdl, nvar, ncon = with_hard_deadline(COMPILE_LIMIT) do; build_model(yaml); end
        model = mdl
        write_result(rp, Dict(
            PFX*"compile_status" => "ok", PFX*"compile_time" => time() - t0,
            PFX*"nvar" => nvar, PFX*"ncon" => ncon,
        ))
    catch e
        write_result(rp, Dict(PFX*"compile_status" => "error", PFX*"error" => sprint(showerror, e)))
        @error "[$m] EXA compile failed" exception=(e, catch_backtrace()); return
    end

    @info "[$m] EXA solving with MadNLP/$BACKEND..."
    write_result(rp, Dict(PFX*"solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 3600.0) do; solve_madnlp(model); end
        write_result(rp, Dict(
            PFX*"solve_status" => "ok", PFX*"solve_time" => time() - t0,
            PFX*"term_status"  => string(res.status), PFX*"objective" => res.objective,
            PFX*"iter"         => res.iter,
            PFX*"constr_viol"  => max_constr_viol(model, res.solution),
            PFX*"theta_star"   => join(theta_star(model, res.solution), ","),
        ))
    catch e
        write_result(rp, Dict(PFX*"solve_status" => "error", PFX*"error" => sprint(showerror, e)))
        @error "[$m] EXA solve failed" exception=(e, catch_backtrace()); model = nothing; gpu_reclaim(); return
    end

    if uppercase(get(read_result(rp), PFX*"term_status", "")) in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")
        run_sgm_reruns(m, rp, model)
    else
        write_result(rp, Dict(PFX*"sgm_status" => "skipped"))
    end
    model = nothing; gpu_reclaim()
    @info "[$m] EXA done"
end

function warmup_exa()
    yaml = get_yaml(JIT_MODEL); yaml === nothing && return
    @info "EXA warmup: JIT build+solve on $JIT_MODEL ($BACKEND) ..."
    try
        mdl, _, _ = build_model(yaml); IS_GPU && CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
               max_wall_time=250.0, linear_solver=LINEAR_SOLVER, KKT_OPTS...)
        mdl = nothing; gpu_reclaim(); @info "EXA warmup done"
    catch e; @warn "EXA warmup failed" exception=(e, catch_backtrace()); end
end
function main()
    gpu_idx = findfirst(a -> occursin(r"^\d+$", a), ARGS)
    gpu_id  = gpu_idx === nothing ? 0 : parse(Int, ARGS[gpu_idx])
    IS_GPU && CUDA.device!(gpu_id)
    mkpath(RESULTDIR)
    @info "run_warmup: target=$TARGET jit=$JIT_MODEL backend=$BACKEND prefix=$PFX"

    # Run only the side the current tag covers (BENCH_INCLUDE_* in options.jl).
    run_exa = (IS_GPU && BENCH_INCLUDE_EXAGPU) || (!IS_GPU && BENCH_INCLUDE_EXACPU)

    if run_exa
        warmup_exa()
        bench_exa(TARGET)
    end
    @info "run_warmup complete (exa=$run_exa)"
end

main()
