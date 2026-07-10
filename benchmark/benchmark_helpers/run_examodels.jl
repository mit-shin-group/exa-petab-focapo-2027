# run_examodels.jl — ExaModelsPEtab + MadNLP benchmark (GPU or CPU)
#
# Builds petab_examodel and solves it with MadNLP using the LiftedKKT (condensed-space) regime;
# only the linear solver differs (GPU: CUDSS, CPU: MadNLP default). Results are written to
# benchmark_results/{Model}_results.txt under a backend-specific prefix:
#   BENCH_BACKEND=gpu (default) -> exagpu_*
#   BENCH_BACKEND=cpu           -> exacpu_*
#
# Each solve also stores <prefix>petab_obj := PEtab's objective at the ExaModels optimum.
#
# Compile timing splits into Phase 1 (PEtab setup + ODE presolve) and Phase 2 (ExaModels build);
# after a converged first solve, N_SGM_RERUNS warm reruns give the SGM solve time. Resumable.
#
# Usage (GPU, strided one instance per GPU):
#   julia --project=. -t 1 benchmark_helpers/run_examodels.jl <gpu_id> <num_instances> <instance_idx>
# CPU (gpu_id arg ignored; run several instances to parallelize across cores):
#   BENCH_BACKEND=cpu julia --project=. -t 1 benchmark_helpers/run_examodels.jl 0 <num_instances> <instance_idx>

using ExaModelsPEtab, PEtab, CUDA, MadNLP, MadNLPGPU, CUDSS, ExaModels
# HSL is a licensed, user-supplied dependency (see README); only the CPU pass needs it.
if lowercase(get(ENV, "BENCH_BACKEND", "gpu")) == "cpu"
    using MadNLPHSL
end

# ─── CONFIGURABLE SETTINGS (single source of truth = options.jl) ────────────────
include(joinpath(@__DIR__, "..", "options.jl"))   # MODELDIR + RESULTDIR + model sets + BENCH_* config

const K             = BENCH_K
const TOL           = BENCH_TOL
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N
const SGM_SHIFT     = BENCH_SGM_SHIFT
const ACCEPT_TOL    = BENCH_ACCEPT_TOL
const ACCEPT_ITER   = BENCH_ACCEPT_ITER
const WARMUP_MODEL  = BENCH_WARMUP_MODEL

# Backend: "gpu" (CUDA + CUDSS, default) or "cpu". PFX prefixes every result key.
const BACKEND = lowercase(get(ENV, "BENCH_BACKEND", "gpu"))
const IS_GPU  = BACKEND != "cpu"
const PFX     = IS_GPU ? "exagpu_" : "exacpu_"

# LiftedKKT (condensed-space) MadNLP regime; passed through from options.jl.
const KKT_OPTS = (kkt_system = BENCH_KKT_SYSTEM(),
                  equality_treatment = BENCH_EQUALITY_TREATMENT(),
                  fixed_variable_treatment = BENCH_FIXED_VAR_TREATMENT())
const LINEAR_SOLVER = IS_GPU ? BENCH_GPU_SOLVER() : BENCH_CPU_SOLVER()
# blas_num_threads gives Ma57's dense CPU factorization BENCH_CPU_THREADS BLAS threads (MadNLP pins
# BLAS to 1 otherwise); irrelevant on GPU (CUDSS does the factorization), so 1 there.
solve_madnlp(model) = madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER,
    max_iter=MAX_ITER, max_wall_time=SOLVE_LIMIT, linear_solver=LINEAR_SOLVER,
    blas_num_threads = IS_GPU ? 1 : BENCH_CPU_THREADS, KKT_OPTS...)
gpu_reclaim() = IS_GPU && (GC.gc(); CUDA.reclaim())
# ──────────────────────────────────────────────────────────────────────────────

# Benchmarked set minus the warmup model; override via BENCH_SUBSET.
const RUN_MODELS = haskey(ENV, "BENCH_SUBSET") ?
    String.(split(ENV["BENCH_SUBSET"], ',')) : filter(!=(BENCH_WARMUP_MODEL), BENCHMARK_MODELS)

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

# Read-modify-write so the other backend's keys in the same file are preserved.
function write_result(path, updates)
    existing = read_result(path)
    merged = merge(existing, Dict(string(k) => replace(string(v), '\n' => ' ', '\r' => ' ')
                                  for (k, v) in updates))
    open(path, "w") do io
        for k in sort(collect(keys(merged))); println(io, "$k=", merged[k]); end
        flush(io)
    end
end

# Run f under a hard wall-clock deadline. If exceeded, an external watchdog creates `flag` (so a
# genuine timeout is distinguishable on disk from an external kill like a reboot) then SIGKILLs us.
function with_hard_deadline(f, seconds::Real; flag::AbstractString="")
    pid = getpid()
    isempty(flag) || rm(flag; force=true)
    touchcmd = isempty(flag) ? "" : "touch '$(flag)'; "
    w = run(`bash -c "sleep $(seconds); $(touchcmd)kill -9 $(pid)"`; wait=false)
    try; return f(); finally; try; kill(w); catch; end; end
end

# true if this backend has finished model m
function exa_finished(m)
    d = read_result(result_path(m))
    cs  = get(d, PFX * "compile_status", "")
    ss  = get(d, PFX * "solve_status",   "")
    sgm = get(d, PFX * "sgm_status",     "")
    cs in ("timeout", "error", "missing_yaml", "skipped") && return true
    cs != "ok" && return false
    ss in ("timeout", "error") && return true
    ss != "ok" && return false
    return sgm in ("ok", "error", "skipped")
end

# PEtab's objective at the ExaModels optimum; NaN if the eval fails.
function petab_obj_at_exa(PEprob, res)
    Np = PEprob.nparameters_estimate
    xstar = Array(res.solution)[1:Np]
    try; return PEprob.nllh(xstar); catch; return NaN; end
end

# max constraint violation max(lcon - c(x), c(x) - ucon, 0) at x; 0 if unconstrained
function max_constr_viol(model, x)
    model.meta.ncon == 0 && return 0.0
    c = similar(x, model.meta.ncon); ExaModels.cons!(model, x, c)
    c = Array(c); lc = Array(model.meta.lcon); uc = Array(model.meta.ucon)
    maximum(max.(lc .- c, c .- uc, 0.0))
end

# ─── build one ExaModel; returns the PEtabODEProblem too ──
function build_model(yaml, t_origin)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    backend = IS_GPU ? CUDA.CUDABackend() : nothing
    c = ExaModels.ExaCore(; backend, concrete=Val(true))

    if ExaModelsPEtab._is_steady_state(PEmodel)
        c, PEinfo = ExaModelsPEtab._create_variables_ss(c, PEmodel, PEprob)
        t_phase1 = time() - t_origin
        c = ExaModelsPEtab._create_constraints_ss(c, PEmodel, PEprob, PEinfo)
        c, y0, sigma0 = ExaModelsPEtab._create_objective_ss(c, PEmodel, PEprob, PEinfo)
    else
        c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)
        t_phase1 = time() - t_origin
        c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo)
        c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo)
        c, y0, sigma0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo)
    end
    mdl = ExaModels.ExaModel(c)
    ExaModels.set_start!(mdl, c.y, y0)
    ExaModels.set_start!(mdl, c.sigma, sigma0)
    IS_GPU && CUDA.synchronize()
    return mdl, PEprob, mdl.meta.nvar, mdl.meta.ncon, t_phase1
end

# ─── SGM warm reruns (timing only) ──────────────────────────────────────────────
function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict(PFX*"sgm_status" => "running"))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            IS_GPU && GC.gc()     # clear dead GPU allocs OUTSIDE timing; stops a GC/allocator stall landing in a timed solve (keeps pool+kernels warm; no reclaim)
            t0 = time()
            solve_madnlp(model)   # warm rerun, timed bare
            push!(solve_times, time() - t0)
        catch e
            @error "[$m] SGM solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e)))
            return
        end
    end
    sgm_solve = t_sgmdelta(solve_times, SGM_SHIFT)
    write_result(rp, Dict(PFX*"sgm_status" => "ok",
                          PFX*"solve_times" => join(solve_times, ","), PFX*"sgm_solve_time" => sgm_solve))
    @info "[$m] SGM done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

# ─── main per-model benchmark ─────────────────────────────────────────────────
function bench_one(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    d = read_result(rp)
    first_run_done = get(d, PFX*"compile_status", "") == "ok" && get(d, PFX*"solve_status", "") == "ok"
    model = nothing

    if !first_run_done
        if yaml === nothing
            write_result(rp, Dict(PFX*"compile_status" => "missing_yaml", PFX*"solve_status" => "skipped"))
            return
        end

        # ── COMPILE ──────────────────────────────────────────────────────────
        write_result(rp, Dict(
            PFX*"compile_status" => "compiling", PFX*"compile_time" => "", PFX*"presolve_time" => "",
            PFX*"solve_status"   => "skipped",   PFX*"solve_time"   => "", PFX*"term_status"   => "",
            PFX*"objective"      => "",          PFX*"petab_obj"    => "", PFX*"iter" => "",
            PFX*"nvar"           => "",          PFX*"ncon"         => "", PFX*"error" => "",
            PFX*"constr_viol"    => "",
        ))
        @info "[$m] compiling on $BACKEND (K=$K, compile_limit=$(COMPILE_LIMIT)s)..."
        local PEprob
        try
            t0 = time()
            mdl, PEprob, nvar, ncon, t_phase1 = with_hard_deadline(COMPILE_LIMIT; flag=rp*".$(PFX)compiling") do
                build_model(yaml, t0)
            end
            model = mdl
            write_result(rp, Dict(
                PFX*"compile_status" => "ok",
                PFX*"compile_time"   => time() - t0,
                PFX*"presolve_time"  => t_phase1,
                PFX*"nvar" => nvar, PFX*"ncon" => ncon,
            ))
        catch e
            write_result(rp, Dict(PFX*"compile_status" => "error", PFX*"compile_time" => "", PFX*"error" => sprint(showerror, e)))
            @error "[$m] compile failed" exception=(e, catch_backtrace())
            return
        end

        # ── SOLVE (first run includes one-time JIT) ───────────────
        @info "[$m] solving with MadNLP/$BACKEND (max_wall_time=$(SOLVE_LIMIT)s)..."
        write_result(rp, Dict(PFX*"solve_status" => "solving"))
        try
            t0 = time()
            res = with_hard_deadline(SOLVE_LIMIT + 3600.0; flag=rp*".$(PFX)solving") do; solve_madnlp(model); end
            write_result(rp, Dict(
                PFX*"solve_status" => "ok",
                PFX*"solve_time"   => time() - t0,
                PFX*"term_status"  => string(res.status),
                PFX*"objective"    => res.objective,
                PFX*"petab_obj"    => petab_obj_at_exa(PEprob, res),
                PFX*"iter"         => res.iter,
                PFX*"constr_viol"  => max_constr_viol(model, res.solution),
            ))
        catch e
            write_result(rp, Dict(PFX*"solve_status" => "error", PFX*"error" => sprint(showerror, e)))
            @error "[$m] solve failed" exception=(e, catch_backtrace())
            model = nothing; gpu_reclaim()
            return
        end
    end

    # ── SGM SOLVE RERUNS (only after a converged first solve) ──────────────────
    d2 = read_result(rp)
    if get(d2, PFX*"compile_status", "") == "ok" && get(d2, PFX*"solve_status", "") == "ok" &&
       get(d2, PFX*"sgm_status", "") ∉ ("ok", "error", "skipped")
        if !(uppercase(get(d2, PFX*"term_status", "")) in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL"))
            write_result(rp, Dict(PFX*"sgm_status" => "skipped",
                PFX*"sgm_error" => "first solve term=$(get(d2, PFX*"term_status", ""))∉{SUCCEEDED,ACCEPTABLE}; SGM skipped"))
            @info "[$m] SGM skipped (solve term=$(get(d2, PFX*"term_status", "")))"
        else
            if model === nothing  # resuming: rebuild and prime once
                yaml === nothing && return
                @info "[$m] rebuilding for SGM (resume)..."
                try
                    t0 = time()
                    model, _, _, _, _ = with_hard_deadline(COMPILE_LIMIT) do; build_model(yaml, t0); end
                    solve_madnlp(model)
                catch e
                    @error "[$m] SGM rebuild failed" exception=(e, catch_backtrace())
                    write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e)))
                    model = nothing; gpu_reclaim(); return
                end
            end
            run_sgm_reruns(m, rp, model)
        end
    end

    model = nothing; gpu_reclaim()
    @info "[$m] done"
end

function warmup()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "warmup: JIT build+solve on $WARMUP_MODEL ($BACKEND) ..."
    try
        t0 = time()
        mdl, _, _, _, _ = build_model(yaml, t0)
        IS_GPU && CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
               max_wall_time=250.0, linear_solver=LINEAR_SOLVER, KKT_OPTS...)
        mdl = nothing; gpu_reclaim()
        @info "warmup done"
    catch e; @warn "warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    gpu_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    ninst  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    idx    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
    if IS_GPU
        CUDA.device!(gpu_id)
        @info "instance $idx/$ninst on GPU $gpu_id ($(CUDA.name(CUDA.device())))  prefix=$PFX"
    else
        @info "instance $idx/$ninst on CPU  prefix=$PFX"
    end
    mkpath(RESULTDIR)

    # Resolve in-progress sentinels left by a killed run (this backend only). A genuine watchdog
    # timeout left a .compiling/.solving flag → record terminal timeout. No flag ⇒ the kill was
    # external (e.g. reboot) → clear the sentinel so the model re-runs. SGM has no watchdog, so a
    # leftover 'running' is always an external kill → re-run.
    for m in RUN_MODELS
        rp = result_path(m); d = read_result(rp)
        cflag = rp*".$(PFX)compiling"; sflag = rp*".$(PFX)solving"
        if get(d, PFX*"compile_status", "") == "compiling"
            isfile(cflag) ?
                write_result(rp, Dict(PFX*"compile_status" => "timeout", PFX*"compile_time" => string(COMPILE_LIMIT))) :
                write_result(rp, Dict(PFX*"compile_status" => "", PFX*"solve_status" => ""))
            rm(cflag; force=true)
        elseif get(d, PFX*"compile_status", "") == "ok" && get(d, PFX*"solve_status", "") == "solving"
            isfile(sflag) ?
                write_result(rp, Dict(PFX*"solve_status" => "timeout")) :
                write_result(rp, Dict(PFX*"solve_status" => ""))
            rm(sflag; force=true)
        elseif get(d, PFX*"sgm_status", "") == "running"
            write_result(rp, Dict(PFX*"sgm_status" => "interrupted"))
        end
    end

    mine = [m for (i, m) in enumerate(RUN_MODELS) if (i - 1) % ninst == idx]
    todo = filter(!exa_finished, mine)
    @info "instance $idx: $(length(mine)) models assigned, $(length(todo)) remaining"
    isempty(todo) && (@info "nothing to do"; return)

    warmup()
    for (i, m) in enumerate(todo)
        println("\n" * "="^60); println("[$idx][$i/$(length(todo))] $m ($BACKEND)"); println("="^60)
        bench_one(m)
    end
    @info "instance $idx complete"
end

main()
