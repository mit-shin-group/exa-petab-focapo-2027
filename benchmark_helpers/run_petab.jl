# run_petab.jl — PEtab.jl optimizer benchmark
#
# For each model, runs every optimizer in BENCH_PETAB_OPTIMIZERS() in series, paired by index with
# its BENCH_PETAB_HESSIANS entry (nothing => PEtab default). Results go to
# benchmark_results/{Model}_results.txt under per-optimizer keys petab_<label>_* (label = IPNewton /
# GaussNewton / BFGS), so the optimizers and the ExaModels backends coexist in one file.
#
# Launched once per model (many in parallel from a shell loop):
#   for m in <model_list>; do
#     julia --project=. -t 1 benchmark_helpers/run_petab.jl "$m" &
#   done
# Or run serially:
#   julia --project=. -t 1 benchmark_helpers/run_petab.jl Bachmann_MSB2011
#
# To run all 35 models in parallel (up to PAR concurrent workers):
#   PAR=12
#   for m in $(julia --project=. -e 'include("benchmark_helpers/run_petab.jl"); print_models()'); do
#     while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 2; done
#     julia --project=. -t 1 benchmark_helpers/run_petab.jl "$m" &
#   done; wait

using PEtab, Optim, Fides, LinearAlgebra

# ─── CONFIGURABLE SETTINGS ────────────────────────────────────────────────────
# All solver/benchmark settings live in options.jl; below we only alias the BENCH_* constants.
include(joinpath(@__DIR__, "..", "options.jl"))  # MODELDIR + RESULTDIR + model sets + BENCH_* config

LinearAlgebra.BLAS.set_num_threads(BENCH_CPU_THREADS)   # Optim's dense Hessian factorization (Fides BLAS via OMP env)

const TOL           = BENCH_TOL                  # gradient tol — Optim g_abstol / Fides gatol
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT          # Optim time_limit [s]
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT        # PEtab build wall cap [s]
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N                 # SGM rerun count
const SGM_SHIFT     = BENCH_SGM_SHIFT             # SGM shift δ
const WARMUP_MODEL  = BENCH_WARMUP_MODEL          # warmup model; benchmarked via run_warmup.jl
# Optimizers run in series, paired by index with their hessian. Normalize a single entry to a vector.
const OPTIMIZERS    = BENCH_PETAB_OPTIMIZERS() isa AbstractVector ? BENCH_PETAB_OPTIMIZERS() : [BENCH_PETAB_OPTIMIZERS()]
const HESSIANS      = BENCH_PETAB_HESSIANS   isa AbstractVector ? BENCH_PETAB_HESSIANS   : [BENCH_PETAB_HESSIANS]
length(OPTIMIZERS) == length(HESSIANS) ||
    error("BENCH_PETAB_OPTIMIZERS and BENCH_PETAB_HESSIANS must be both single entries or vectors of equal length (got $(length(OPTIMIZERS)) vs $(length(HESSIANS)))")
# When set, preserve an already-computed SGM and only re-solve to re-capture the convergence flags.
const KEEP_SGM      = get(ENV, "BENCH_PETAB_KEEP_SGM", "0") == "1"

# Per-optimizer label: result-key prefix + table TAG column (matches the historical optimizer tags).
petab_label(opt) = opt isa Optim.IPNewton      ? "IPNewton"    :
                   opt isa Fides.BFGS          ? "BFGS"        :
                   opt isa Fides.CustomHessian ? "GaussNewton" : string(nameof(typeof(opt)))
is_fides(opt)    = opt isa Fides.HessianUpdate
# Build a PEtabODEProblem; a nothing hessian omits hessian_method so PEtab picks its own default.
build_pe(yaml, hess) = hess === nothing ? PEtabODEProblem(PEtabModel(yaml)) :
                                          PEtabODEProblem(PEtabModel(yaml); hessian_method = hess)
# ──────────────────────────────────────────────────────────────────────────────

# All benchmark models minus the warmup model. Used by print_models() / the .sh driver;
# run_worker() benchmarks whatever model name is passed as ARGS[1].
const RUN_MODELS = filter(!=(WARMUP_MODEL), BENCHMARK_MODELS)  # 34

print_models() = foreach(m -> print(m, " "), RUN_MODELS)

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end

result_path(m) = joinpath(RESULTDIR, "$(m)_results.txt")

function read_result(path)
    d = Dict{String,String}()
    isfile(path) || return d
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

function has_events(PEprob)
    cbs = PEprob.model_info.simulation_info.callbacks
    for (_, cb) in cbs
        cc = hasproperty(cb, :continuous_callbacks) ? getproperty(cb, :continuous_callbacks) : ()
        dc = hasproperty(cb, :discrete_callbacks)   ? getproperty(cb, :discrete_callbacks)   : ()
        (!isempty(cc) || !isempty(dc)) && return true
    end
    return false
end

# PEtab.jl's own default optimizer options (Optim DEFAULT_OPT / bare FidesOptions), overriding only
# the wall-time + iteration caps. g_abstol = BENCH_TOL = 1e-6 matches PEtab's Optim g_tol default.
petab_options(isf) = isf ?
    Fides.FidesOptions(gatol = TOL, maxiter = MAX_ITER, maxtime = SOLVE_LIMIT) :
    Optim.Options(g_abstol = TOL, f_reltol = 1e-8, x_abstol = 0.0, allow_f_increases = true,
                  successive_f_tol = 3, iterations = MAX_ITER, time_limit = SOLVE_LIMIT, show_trace = false)

# ─── PEtab SGM solve reruns ─────────────────────────────────────────────────────
# Re-solve the same compiled problem from the same start N_SGM_RERUNS times and store the shifted
# geometric mean solve time under <pfx>sgm_*. Only run after a converged first solve.
function run_petab_sgm(rp, PEprob, pfx, opt, opts)
    write_result(rp, Dict(pfx*"sgm_status" => "running"))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$pfx SGM] solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            calibrate(PEprob, get_x(PEprob), opt; options = opts)   # warm rerun, timed bare (no watchdog)
            push!(solve_times, time() - t0)
        catch e
            @error "[$pfx SGM] solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict(pfx*"sgm_status" => "error",
                                  pfx*"sgm_error"  => sprint(showerror, e)))
            return
        end
    end
    sgm_solve = t_sgmdelta(solve_times, SGM_SHIFT)
    write_result(rp, Dict(pfx*"sgm_status"     => "ok",
                          pfx*"solve_times"    => join(solve_times, ","),
                          pfx*"sgm_solve_time" => sgm_solve))
    @info "[$pfx SGM] done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

# Benchmark ONE optimizer for model m: compile + first solve + SGM, all under prefix petab_<label>_.
function bench_optimizer(m, rp, yaml, pfx, opt, hess)
    isf  = is_fides(opt)
    opts = petab_options(isf)

    # resumability: skip if this optimizer is already terminal for this model
    d0 = read_result(rp)
    cs0 = get(d0, pfx*"compile_status", ""); ss0 = get(d0, pfx*"solve_status", "")
    if cs0 in ("error", "missing_yaml") ||
       (cs0 == "ok" && ss0 in ("ok", "error", "timeout") &&
        (N_SGM_RERUNS == 0 || get(d0, pfx*"sgm_status", "") in ("ok", "error", "skipped")))
        @info "[$m] $pfx already terminal — skip"
        return
    end

    # ── COMPILE ──────────────────────────────────────────────────────────────
    clear = Dict(
        pfx*"compile_status" => "compiling", pfx*"compile_time" => "",
        pfx*"solve_status"   => "skipped",   pfx*"solve_time"   => "",
        pfx*"objective"      => "",           pfx*"iter"        => "",
        pfx*"optimum_found"  => "",           pfx*"has_events"  => "",
        pfx*"error"          => "",           pfx*"retcode"      => "",
        pfx*"gconverged"     => "",           pfx*"gresidual"    => "",
        pfx*"fconverged"     => "",           pfx*"xconverged"   => "",
    )
    # clear stale SGM outputs so a fresh invocation re-runs SGM; KEEP_SGM preserves them.
    KEEP_SGM || merge!(clear, Dict(pfx*"sgm_status" => "", pfx*"sgm_solve_time" => "", pfx*"solve_times" => ""))
    write_result(rp, clear)
    @info "[$m] PEtab compiling [$pfx] (limit=$(COMPILE_LIMIT)s)..."

    # JIT warmup: pay this optimizer's JIT cost on the warmup model; best-effort hess! prime.
    try
        wy = get_yaml(WARMUP_MODEL)
        if wy !== nothing
            wp = build_pe(wy, hess)
            wx = get_x(wp); try; wp.hess!(zeros(length(wx), length(wx)), wx); catch; end
            wopts = isf ? Fides.FidesOptions(maxiter = 3) : Optim.Options(iterations = 3)
            calibrate(wp, wx, opt; options = wopts)
        end
    catch; end

    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(COMPILE_LIMIT) do
            p  = build_pe(yaml, hess)
            x0 = get_x(p); np = length(x0)
            # Prime nllh + grad; hess! best-effort (skipped when the problem is hessian-free).
            p.nllh(x0); p.grad!(zeros(np), x0); try; p.hess!(zeros(np, np), x0); catch; end
            p
        end
        write_result(rp, Dict(pfx*"compile_status" => "ok",
            pfx*"compile_time" => time() - t0, pfx*"has_events" => has_events(PEprob)))
    catch e
        write_result(rp, Dict(pfx*"compile_status" => "error", pfx*"error" => sprint(showerror, e)))
        return
    end

    # ── SOLVE ─────────────────────────────────────────────────────────────────
    @info "[$m] PEtab solving [$pfx] (wall_limit=$(SOLVE_LIMIT)s)..."
    write_result(rp, Dict(pfx*"solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 600.0) do
            calibrate(PEprob, get_x(PEprob), opt; options = opts)
        end
        # Record which convergence criterion fired (g/f/x):
        #  • Optim: per-criterion booleans + gradient residual from res.original.
        #  • Fides: retcode Symbol — :GTOL/:FTOL/:XTOL converged, else timeout/failure.
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
            try
                o = res.original
                gconv = string(Optim.g_converged(o)); gres = string(Optim.g_residual(o))
                fconv = string(Optim.f_converged(o)); xconv = string(Optim.x_converged(o))
            catch; end
        end
        write_result(rp, Dict(
            pfx*"solve_status"  => solve_err ? "error" : "ok",
            pfx*"solve_time"    => time() - t0,
            pfx*"objective"     => res.fmin,
            pfx*"iter"          => res.niterations,
            pfx*"optimum_found" => optimum_found,
            pfx*"retcode"       => string(res.converged),
            pfx*"gconverged"    => gconv, pfx*"gresidual"  => gres,
            pfx*"fconverged"    => fconv, pfx*"xconverged" => xconv,
        ))
    catch e
        write_result(rp, Dict(pfx*"solve_status" => "error", pfx*"error" => sprint(showerror, e)))
    end

    # ── SGM SOLVE RERUNS (timing-only; needs a converged first solve) ─────────────
    if N_SGM_RERUNS > 0
        d = read_result(rp)
        if get(d, pfx*"optimum_found", "") == "true" && get(d, pfx*"sgm_status", "") ∉ ("ok", "error")
            run_petab_sgm(rp, PEprob, pfx, opt, opts)
        elseif get(d, pfx*"optimum_found", "") != "true"
            write_result(rp, Dict(pfx*"sgm_status" => "skipped",
                pfx*"sgm_error" => "first solve optimum_found=$(get(d,pfx*"optimum_found",""))≠true; SGM skipped"))
        end
    end
end

# True once every optimizer's result for model m is terminal (used for the run_petab.sh resume skip).
function petab_all_terminal(rp)
    d = read_result(rp)
    all(OPTIMIZERS) do opt
        p = "petab_$(petab_label(opt))_"
        cs = get(d, p*"compile_status", ""); ss = get(d, p*"solve_status", "")
        cs in ("error", "missing_yaml") ||
            (cs == "ok" && ss in ("ok", "error", "timeout") &&
             (N_SGM_RERUNS == 0 || get(d, p*"sgm_status", "") in ("ok", "error", "skipped")))
    end
end

# Run every optimizer in series for model m, each under its own petab_<label>_ prefix.
function run_worker(m)
    mkpath(RESULTDIR)
    rp = result_path(m)
    yaml = get_yaml(m)
    if yaml === nothing
        for opt in OPTIMIZERS
            pfx = "petab_$(petab_label(opt))_"
            write_result(rp, Dict(pfx*"compile_status" => "missing_yaml", pfx*"solve_status" => "skipped"))
        end
    else
        for (opt, hess) in zip(OPTIMIZERS, HESSIANS)
            bench_optimizer(m, rp, yaml, "petab_$(petab_label(opt))_", opt, hess)
        end
    end
    petab_all_terminal(rp) && write_result(rp, Dict("petab_alldone" => "true"))
end

isempty(ARGS) && error("usage: run_petab.jl <model_name>")
run_worker(ARGS[1])
