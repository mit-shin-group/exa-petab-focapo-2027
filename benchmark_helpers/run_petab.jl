# run_petab.jl — PEtab.jl optimizer benchmark
#
# Compiles and solves each PEtab model with the tag-selected PEtab.jl optimizer
# (BENCH_PETAB_OPTIMIZER + BENCH_PETAB_HESSIAN in options.jl): Optim.IPNewton (ForwardDiff),
# Fides.CustomHessian (Gauss–Newton), or Fides.BFGS. Results written to
# benchmark_results/{Model}_results.txt using prefixed keys (petab_*) so ExaModels results in the
# same file are preserved.
#
# Intended to be launched once per model (many in parallel from a shell loop):
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

using PEtab, Optim, Fides

# ─── CONFIGURABLE SETTINGS ────────────────────────────────────────────────────
# ALL solver/benchmark settings live in options.jl (single source of truth, shared with
# run_examodels.jl). Change them THERE, not here — below we only alias the BENCH_* constants.
include(joinpath(@__DIR__, "..", "options.jl"))  # MODELDIR + RESULTDIR + model sets + BENCH_* config

const TOL           = BENCH_TOL                  # Optim g_tol (== MadNLP tol)
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT          # Optim time_limit [s] (== MadNLP max_wall_time)
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT        # PEtab build wall cap [s]
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N                 # shared SGM rerun count (== exa's)
const SGM_SHIFT     = BENCH_SGM_SHIFT             # shared shift δ (== exa's)
const WARMUP_MODEL  = BENCH_WARMUP_MODEL          # benchmark Bruno separately via run_bruno.jl (Crauste-warmed)
# Whether the configured optimizer is a Fides (Python) algorithm — derived from its type, so the
# user just sets BENCH_PETAB_OPTIMIZER directly. Selects FidesOptions vs Optim.Options + the retcode map.
const IS_FIDES      = BENCH_PETAB_OPTIMIZER() isa Fides.HessianUpdate
# When set, PRESERVE an already-computed SGM (petab_sgm_*) and only re-solve once to (re)capture the
# x/f/g convergence flags — a cheap flag-only re-pass after a full SGM run, avoiding redoing SGM.
const KEEP_SGM      = get(ENV, "BENCH_PETAB_KEEP_SGM", "0") == "1"
# ──────────────────────────────────────────────────────────────────────────────

# PEtab attempts EVERY benchmark model (optimum-found is only known after solving), minus the
# shared JIT warmup Bruno (benchmarked via run_bruno.jl). Only used by print_models() /
# the .sh driver; run_worker() benchmarks whatever model name is passed as ARGS[1].
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

# Per-tag optimizer convergence options. DEFAULT optimizer settings except the gradient-convergence
# tol (relaxed to BENCH_TOL) and the uniform benchmark wall/iteration caps. Optim ⇒ Optim.Options
# (g_tol is the gradient criterion); Fides ⇒ FidesOptions (gatol is the gradient criterion — its
# default already equals BENCH_TOL, set explicitly here). Same `options=` kwarg drives both calibrate
# methods, so the call site is backend-agnostic.
petab_options() = IS_FIDES ?
    Fides.FidesOptions(gatol = TOL, maxiter = MAX_ITER, maxtime = SOLVE_LIMIT) :
    Optim.Options(g_tol = TOL, iterations = MAX_ITER, time_limit = SOLVE_LIMIT, show_trace = false)

# ─── PEtab SGM solve reruns ─────────────────────────────────────────────────────
# Mirrors the exa SGM pass: re-solve the SAME compiled PEtabODEProblem from the SAME nominal
# start N_SGM_RERUNS times and store the shifted geometric mean (δ=10s) solve time under petab_sgm_* — so the
# PEtab column is an n=10 head-to-head against exa_sgm_solve_time. Only meaningful after a
# converged first solve (petab_optimum_found=true); skipped otherwise.
function run_petab_sgm(rp, PEprob)
    write_result(rp, Dict("petab_sgm_status" => "running"))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[petab SGM] solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            calibrate(PEprob, get_x(PEprob), BENCH_PETAB_OPTIMIZER(); options=petab_options())   # warm rerun — timed bare (no watchdog) for a clean SGM
            push!(solve_times, round(time() - t0; digits=2))
        catch e
            @error "[petab SGM] solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict("petab_sgm_status" => "error",
                                  "petab_sgm_error"  => sprint(showerror, e)))
            return
        end
    end
    sgm_solve = shifted_geomean(solve_times, SGM_SHIFT)
    write_result(rp, Dict("petab_sgm_status"     => "ok",
                          "petab_solve_times"    => join(solve_times, ","),
                          "petab_sgm_solve_time" => sgm_solve))
    @info "[petab SGM] done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

function run_worker(m)
    mkpath(RESULTDIR)
    rp = result_path(m)
    yaml = get_yaml(m)

    if yaml === nothing
        write_result(rp, Dict("petab_compile_status" => "missing_yaml",
                               "petab_solve_status"   => "skipped"))
        return
    end

    # ── COMPILE ──────────────────────────────────────────────────────────────
    clear = Dict(
        "petab_compile_status" => "compiling", "petab_compile_time" => "",
        "petab_solve_status"   => "skipped",   "petab_solve_time"   => "",
        "petab_objective"      => "",           "petab_iter"        => "",
        "petab_optimum_found"  => "",           "petab_has_events"  => "",
        "petab_error"          => "",
        # always clear the convergence flags — they are recomputed in the SOLVE block below
        "petab_gconverged"     => "",           "petab_gresidual"      => "",
        "petab_fconverged"     => "",           "petab_xconverged"     => "",
        "petab_retcode"        => "",
    )
    # clear stale SGM outputs so a fresh invocation actually re-runs SGM (the gate below skips SGM if
    # petab_sgm_status is already ok/error). KEEP_SGM preserves them for a flag-only re-pass.
    KEEP_SGM || merge!(clear, Dict(
        "petab_sgm_status"     => "",           "petab_sgm_solve_time" => "",
        "petab_solve_times"    => "",
    ))
    write_result(rp, clear)
    @info "[$m] PEtab compiling (limit=$(COMPILE_LIMIT)s)..."

    # JIT warmup: pay the generic optimizer JIT cost on a small model. Build with the SAME Hessian
    # method and prime hess! once (priming hess! is also the Fides cache-prime fix, see compile below).
    try
        wy = get_yaml(WARMUP_MODEL)
        if wy !== nothing
            wp = PEtabODEProblem(PEtabModel(wy); hessian_method = BENCH_PETAB_HESSIAN)
            wx = get_x(wp); wp.hess!(zeros(length(wx), length(wx)), wx)
            wopts = IS_FIDES ? Fides.FidesOptions(maxiter = 3) : Optim.Options(iterations = 3)
            calibrate(wp, wx, BENCH_PETAB_OPTIMIZER(); options = wopts)
        end
    catch; end

    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(COMPILE_LIMIT) do
            p  = PEtabODEProblem(PEtabModel(yaml); hessian_method = BENCH_PETAB_HESSIAN)
            x0 = get_x(p); np = length(x0)
            # Prime nllh + grad + hess. The hess! call is REQUIRED for the Fides BFGS tag: PEtab's
            # Fides extension omits the cache-priming nllh it runs for CustomHessian, so BFGS's
            # nllh_grad objective would hit an unpopulated odesols_derivatives cache (KeyError). The
            # Gauss–Newton hess! solve populates that cache; one compile-time call fixes every solve.
            p.nllh(x0); p.grad!(zeros(np), x0); p.hess!(zeros(np, np), x0)
            p
        end
        write_result(rp, Dict(
            "petab_compile_status" => "ok",
            "petab_compile_time"   => round(time() - t0; digits=2),
            "petab_has_events"     => has_events(PEprob),
        ))
    catch e
        write_result(rp, Dict(
            "petab_compile_status" => "error",
            "petab_error"          => sprint(showerror, e),
        ))
        return
    end

    # ── SOLVE ─────────────────────────────────────────────────────────────────
    @info "[$m] PEtab solving ($(BENCH_TAG), wall_limit=$(SOLVE_LIMIT)s)..."
    write_result(rp, Dict("petab_solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 600.0) do
            calibrate(PEprob, get_x(PEprob), BENCH_PETAB_OPTIMIZER(); options=petab_options())
        end
        # Record WHICH convergence criterion fired so the table can distinguish a genuine first-order-
        # stationary stop (gradient) from an objective-plateau/zero-step stop. The two optimizer
        # families report this differently:
        #  • Optim: `converged` is x||f||g (so an IPNewton "success" may stop via f/x with |g| ≫ g_tol);
        #    capture the per-criterion booleans (g/f/x) + the gradient residual from res.original.
        #  • Fides: a single retcode Symbol — :GTOL / :FTOL / :XTOL = converged (gradient / f / x);
        #    :MAXTIME / :MAXITER = ran out (timeout); others = failure. res.original is a Fides struct
        #    (no Optim accessors), so map the retcode straight onto the same g/f/x flags.
        gconv = ""; gres = ""; fconv = ""; xconv = ""
        optimum_found = "false"; solve_err = !isfinite(res.fmin)
        if IS_FIDES
            rc = res.converged                                # Fides retcode Symbol
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
            "petab_solve_status"  => solve_err ? "error" : "ok",
            "petab_solve_time"    => round(time() - t0; digits=2),
            "petab_objective"     => res.fmin,
            "petab_iter"          => res.niterations,
            "petab_optimum_found" => optimum_found,
            "petab_retcode"       => string(res.converged),
            "petab_gconverged"    => gconv,
            "petab_gresidual"     => gres,
            "petab_fconverged"    => fconv,
            "petab_xconverged"    => xconv,
        ))
    catch e
        write_result(rp, Dict(
            "petab_solve_status" => "error",
            "petab_error"        => sprint(showerror, e),
        ))
    end

    # ── SGM SOLVE RERUNS (timing-only; needs a converged first solve) ─────────────
    if N_SGM_RERUNS > 0
        d = read_result(rp)
        if get(d, "petab_optimum_found", "") == "true" &&
           get(d, "petab_sgm_status", "") ∉ ("ok", "error")
            run_petab_sgm(rp, PEprob)
        elseif get(d, "petab_optimum_found", "") != "true"
            write_result(rp, Dict("petab_sgm_status" => "skipped",
                "petab_sgm_error" => "first solve optimum_found=$(get(d,"petab_optimum_found",""))≠true; SGM skipped"))
        end
    end
end

isempty(ARGS) && error("usage: run_petab.jl <model_name>")
run_worker(ARGS[1])
