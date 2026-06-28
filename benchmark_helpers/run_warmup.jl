# run_warmup.jl — benchmarks the shared JIT-warmup model BENCH_WARMUP_MODEL (options.jl).
#
# That model is the JIT warmup for run_examodels.jl and run_petab.jl, so it is excluded from their
# timed runs and benchmarked here instead, JIT-warmed on a different model for a clean compile time.
# Backend follows BENCH_BACKEND ("gpu" -> exagpu_*, "cpu" -> exacpu_*); the PEtab side is
# backend-independent. Runs only the side the current tag covers (BENCH_INCLUDE_* in options.jl).
#   julia --project=. -t 1 benchmark_helpers/run_warmup.jl [gpu_id]
#   BENCH_BACKEND=cpu julia --project=. -t 1 benchmark_helpers/run_warmup.jl

using ExaModelsPEtab, PEtab, CUDA, MadNLP, MadNLPGPU, CUDSS, ExaModels, Optim, MadNLPHSL, Fides, LinearAlgebra

# ─── CONFIGURABLE SETTINGS (single source of truth = options.jl) ──────────────────
include(joinpath(@__DIR__, "..", "options.jl"))   # MODELDIR + RESULTDIR + model sets + BENCH_* config

const TARGET    = BENCH_WARMUP_MODEL   # the warmup model, benchmarked here
const JIT_MODEL = TARGET == "Crauste_CellSystems2017" ? "Bruno_JExpBot2016" : "Crauste_CellSystems2017"  # JIT-warm on a different model

const K             = BENCH_K
const TOL           = BENCH_TOL
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const PETAB_COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N
const SGM_SHIFT     = BENCH_SGM_SHIFT
const ACCEPT_TOL    = BENCH_ACCEPT_TOL
const ACCEPT_ITER   = BENCH_ACCEPT_ITER
const PETAB_SGM_N   = BENCH_SGM_N

# Optimizers run in series, paired by index with their hessian. Normalize a single entry to a vector.
const OPTIMIZERS    = BENCH_PETAB_OPTIMIZERS() isa AbstractVector ? BENCH_PETAB_OPTIMIZERS() : [BENCH_PETAB_OPTIMIZERS()]
const HESSIANS      = BENCH_PETAB_HESSIANS   isa AbstractVector ? BENCH_PETAB_HESSIANS   : [BENCH_PETAB_HESSIANS]
length(OPTIMIZERS) == length(HESSIANS) ||
    error("BENCH_PETAB_OPTIMIZERS and BENCH_PETAB_HESSIANS must be both single entries or vectors of equal length (got $(length(OPTIMIZERS)) vs $(length(HESSIANS)))")
# Per-optimizer label (result-key prefix), Fides check, and nothing-aware problem build.
petab_label(opt) = opt isa Optim.IPNewton      ? "IPNewton"    :
                   opt isa Fides.BFGS          ? "BFGS"        :
                   opt isa Fides.CustomHessian ? "GaussNewton" : string(nameof(typeof(opt)))
is_fides(opt)    = opt isa Fides.HessianUpdate
build_pe(yaml, hess) = hess === nothing ? PEtabODEProblem(PEtabModel(yaml)) :
                                          PEtabODEProblem(PEtabModel(yaml); hessian_method = hess)

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
function petab_obj_at_exa(PEprob, res)
    Np = PEprob.nparameters_estimate
    xstar = Array(res.solution)[1:Np]
    try; return PEprob.nllh(xstar); catch; return NaN; end
end

# inf-norm (max) constraint violation max(lcon - c(x), c(x) - ucon, 0) at point x; 0 if unconstrained.
function max_constr_viol(model, x)
    model.meta.ncon == 0 && return 0.0
    c = similar(x, model.meta.ncon); ExaModels.cons!(model, x, c)
    c = Array(c); lc = Array(model.meta.lcon); uc = Array(model.meta.ucon)
    maximum(max.(lc .- c, c .- uc, 0.0))
end

# ─── ExaModels build (backend-parametrized; returns PEtabODEProblem) ──────────────
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

function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict(PFX*"sgm_status" => "running"))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time(); solve_madnlp(model); push!(solve_times, time() - t0)
        catch e
            @error "[$m] SGM solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e))); return
        end
    end
    sgm_solve = t_sgmdelta(solve_times, SGM_SHIFT)
    write_result(rp, Dict(PFX*"sgm_status" => "ok",
                          PFX*"solve_times" => join(solve_times, ","), PFX*"sgm_solve_time" => sgm_solve))
    @info "[$m] SGM done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

# ─── ExaModels benchmark for the target (compile + first solve + SGM) ───────────
function bench_exa(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    yaml === nothing && (write_result(rp, Dict(PFX*"compile_status" => "missing_yaml", PFX*"solve_status" => "skipped")); return)
    model = nothing

    write_result(rp, Dict(
        PFX*"compile_status" => "compiling", PFX*"compile_time" => "", PFX*"presolve_time" => "",
        PFX*"solve_status"   => "skipped",   PFX*"solve_time"   => "", PFX*"term_status"   => "",
        PFX*"objective"      => "",          PFX*"petab_obj"    => "", PFX*"iter" => "",
        PFX*"nvar"           => "",          PFX*"ncon"         => "", PFX*"error" => "",
        PFX*"constr_viol"    => "",
    ))
    @info "[$m] EXA compiling on $BACKEND (K=$K)..."
    local PEprob
    try
        t0 = time()
        mdl, PEprob, nvar, ncon, t_phase1 = with_hard_deadline(COMPILE_LIMIT) do; build_model(yaml, t0); end
        model = mdl
        write_result(rp, Dict(
            PFX*"compile_status" => "ok", PFX*"compile_time" => time() - t0,
            PFX*"presolve_time"  => t_phase1, PFX*"nvar" => nvar, PFX*"ncon" => ncon,
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
            PFX*"petab_obj"    => petab_obj_at_exa(PEprob, res), PFX*"iter" => res.iter,
            PFX*"constr_viol"  => max_constr_viol(model, res.solution),
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

# ─── PEtab benchmark for the target ──────────────────────────────────────────────
# Per-optimizer options = PEtab.jl's recommended defaults + the uniform wall/iteration caps.
# PEtab.jl's own default optimizer options (Optim DEFAULT_OPT / bare FidesOptions), overriding only
# the wall-time + iteration caps. g_abstol = BENCH_TOL = 1e-6 matches PEtab's Optim g_tol default.
petab_options(isf) = isf ?
    Fides.FidesOptions(gatol=TOL, maxiter=MAX_ITER, maxtime=SOLVE_LIMIT) :
    Optim.Options(g_abstol=TOL, f_reltol=1e-8, x_abstol=0.0, allow_f_increases=true,
                  successive_f_tol=3, iterations=MAX_ITER, time_limit=SOLVE_LIMIT, show_trace=false)

function has_events(PEprob)
    cbs = PEprob.model_info.simulation_info.callbacks
    for (_, cb) in cbs
        cc = hasproperty(cb, :continuous_callbacks) ? getproperty(cb, :continuous_callbacks) : ()
        dc = hasproperty(cb, :discrete_callbacks)   ? getproperty(cb, :discrete_callbacks)   : ()
        (!isempty(cc) || !isempty(dc)) && return true
    end
    return false
end

# Benchmark ONE optimizer for the warmup model, under prefix petab_<label>_.
function bench_petab_one(m, rp, yaml, pfx, opt, hess)
    isf  = is_fides(opt)
    opts = petab_options(isf)

    write_result(rp, Dict(
        pfx*"compile_status" => "compiling", pfx*"compile_time" => "",
        pfx*"solve_status"   => "skipped",   pfx*"solve_time"   => "",
        pfx*"objective"      => "",           pfx*"iter"        => "",
        pfx*"optimum_found"  => "",           pfx*"has_events"  => "", pfx*"error" => "",
    ))
    @info "[$m] PEtab compiling [$pfx]..."
    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(PETAB_COMPILE_LIMIT) do
            p  = build_pe(yaml, hess)
            x0 = get_x(p); np = length(x0)
            p.nllh(x0); p.grad!(zeros(np), x0); try; p.hess!(zeros(np, np), x0); catch; end; p
        end
        write_result(rp, Dict(pfx*"compile_status" => "ok", pfx*"compile_time" => time() - t0,
                              pfx*"has_events" => has_events(PEprob)))
    catch e
        write_result(rp, Dict(pfx*"compile_status" => "error", pfx*"error" => sprint(showerror, e))); return
    end

    @info "[$m] PEtab solving [$pfx]..."
    write_result(rp, Dict(pfx*"solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 600.0) do; calibrate(PEprob, get_x(PEprob), opt; options=opts); end
        # Convergence: Optim per-criterion booleans vs Fides retcode.
        gconv = ""; gres = ""; fconv = ""; xconv = ""
        optimum_found = "false"; solve_err = !isfinite(res.fmin)
        if isf
            rc = res.converged
            optimum_found = string(rc in (:GTOL, :FTOL, :XTOL))
            gconv = string(rc === :GTOL); fconv = string(rc === :FTOL); xconv = string(rc === :XTOL)
            solve_err |= rc in (:DID_NOT_RUN, :NOT_FINITE, :Optimisation_failed, :Optmisation_failed)
        else
            optimum_found = string(res.converged === true)
            solve_err |= (res.converged === :Optimisation_failed)
            try; o = res.original
                gconv = string(Optim.g_converged(o)); gres = string(Optim.g_residual(o))
                fconv = string(Optim.f_converged(o)); xconv = string(Optim.x_converged(o)); catch; end
        end
        write_result(rp, Dict(
            pfx*"solve_status"  => solve_err ? "error" : "ok",
            pfx*"solve_time"    => time() - t0, pfx*"objective" => res.fmin,
            pfx*"iter"          => res.niterations, pfx*"optimum_found" => optimum_found,
            pfx*"retcode"       => string(res.converged),
            pfx*"gconverged"    => gconv, pfx*"gresidual" => gres,
            pfx*"fconverged"    => fconv, pfx*"xconverged" => xconv,
        ))
    catch e
        write_result(rp, Dict(pfx*"solve_status" => "error", pfx*"error" => sprint(showerror, e)))
    end

    if PETAB_SGM_N > 0 && get(read_result(rp), pfx*"optimum_found", "") == "true"
        write_result(rp, Dict(pfx*"sgm_status" => "running"))
        ptimes = Float64[]; ok = true
        for i in 1:PETAB_SGM_N
            @info "[$m] PEtab SGM solve [$pfx] $i/$PETAB_SGM_N ..."
            try
                t0 = time()
                calibrate(PEprob, get_x(PEprob), opt; options=opts)   # warm rerun, timed bare
                push!(ptimes, time() - t0)
            catch e
                @error "[$m] PEtab SGM solve $i failed" exception=(e, catch_backtrace())
                write_result(rp, Dict(pfx*"sgm_status" => "error", pfx*"sgm_error" => sprint(showerror, e))); ok = false; break
            end
        end
        if ok
            psgm = t_sgmdelta(ptimes, SGM_SHIFT)
            write_result(rp, Dict(pfx*"sgm_status" => "ok",
                                  pfx*"solve_times" => join(ptimes, ","), pfx*"sgm_solve_time" => psgm))
            @info "[$m] PEtab SGM done [$pfx]: solve=$psgm s (n=$PETAB_SGM_N)"
        end
    end
end

# Run every optimizer in series for the warmup model.
function bench_petab(m)
    LinearAlgebra.BLAS.set_num_threads(BENCH_CPU_THREADS)   # restore CPU BLAS threads (the exa solve pins them)
    rp   = result_path(m)
    yaml = get_yaml(m)
    if yaml === nothing
        for opt in OPTIMIZERS
            pfx = "petab_$(petab_label(opt))_"
            write_result(rp, Dict(pfx*"compile_status" => "missing_yaml", pfx*"solve_status" => "skipped"))
        end
        return
    end
    for (opt, hess) in zip(OPTIMIZERS, HESSIANS)
        bench_petab_one(m, rp, yaml, "petab_$(petab_label(opt))_", opt, hess)
    end
end

function warmup_exa()
    yaml = get_yaml(JIT_MODEL); yaml === nothing && return
    @info "EXA warmup: JIT build+solve on $JIT_MODEL ($BACKEND) ..."
    try
        t0 = time(); mdl, _, _, _, _ = build_model(yaml, t0); IS_GPU && CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
               max_wall_time=250.0, linear_solver=LINEAR_SOLVER, KKT_OPTS...)
        mdl = nothing; gpu_reclaim(); @info "EXA warmup done"
    catch e; @warn "EXA warmup failed" exception=(e, catch_backtrace()); end
end
function warmup_petab()
    yaml = get_yaml(JIT_MODEL); yaml === nothing && return
    @info "PEtab warmup: JIT calibrate each optimizer on $JIT_MODEL ..."
    for (opt, hess) in zip(OPTIMIZERS, HESSIANS)
        try
            wp = build_pe(yaml, hess)
            wx = get_x(wp); try; wp.hess!(zeros(length(wx), length(wx)), wx); catch; end
            wopts = is_fides(opt) ? Fides.FidesOptions(maxiter=3) : Optim.Options(iterations=3)
            calibrate(wp, wx, opt; options=wopts)
        catch e; @warn "PEtab warmup failed for $(petab_label(opt))" exception=(e, catch_backtrace()); end
    end
    @info "PEtab warmup done"
end

function main()
    gpu_idx = findfirst(a -> occursin(r"^\d+$", a), ARGS)
    gpu_id  = gpu_idx === nothing ? 0 : parse(Int, ARGS[gpu_idx])
    IS_GPU && CUDA.device!(gpu_id)
    mkpath(RESULTDIR)
    @info "run_warmup: target=$TARGET jit=$JIT_MODEL backend=$BACKEND prefix=$PFX"

    # Run only the side the current tag covers (BENCH_INCLUDE_* in options.jl); BENCH_PETAB_ONLY=1
    # forces the PEtab-only side.
    petab_only = get(ENV, "BENCH_PETAB_ONLY", "0") == "1"
    run_exa = !petab_only && ((IS_GPU && BENCH_INCLUDE_EXAGPU) || (!IS_GPU && BENCH_INCLUDE_EXACPU))
    run_pet = petab_only || BENCH_INCLUDE_PETAB

    if run_exa
        warmup_exa()
        bench_exa(TARGET)
    end
    if run_pet
        warmup_petab()
        bench_petab(TARGET)
    end
    @info "run_warmup complete (exa=$run_exa petab=$run_pet)"
end

main()
