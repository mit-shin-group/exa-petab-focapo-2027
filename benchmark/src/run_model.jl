# Benchmark one model on one backend
#   julia --project=. src/run_model.jl <gpu|cpu|petab> <Model> <WarmupModel>

include(joinpath(@__DIR__, "common.jl"))

const BACKEND, MODEL, WARMUP = ARGS[1], ARGS[2], ARGS[3]
const PFX = BACKEND * "_"

using ExaModelsPEtab
if BACKEND == "petab"
    using PEtab, Optim, Fides
elseif BACKEND == "gpu"
    using MadNLP, MadNLPGPU, CUDA, CUDSS
else
    using MadNLP, MadNLPHSL
end

# ExaModels + MadNLP
if BACKEND != "petab"
    const ExaModels = ExaModelsPEtab.ExaModels

    build(yaml) = examodel_petab(yaml; backend = BACKEND == "gpu" ? CUDA.CUDABackend() : nothing)

    solve(model; max_wall_time = SOLVE_LIMIT) = madnlp(model;
        tol = TOL, acceptable_tol = ACCEPT_TOL, acceptable_iter = ACCEPT_ITER, max_iter = MAX_ITER,
        max_wall_time, linear_solver = BACKEND == "gpu" ? GPU_SOLVER() : CPU_SOLVER(),
        blas_num_threads = BACKEND == "gpu" ? 1 : CPU_THREADS, KKT_OPTS()...)

    warmup_solve(model) = solve(model; max_wall_time = 600.0)
    sync() = BACKEND == "gpu" && CUDA.synchronize()
    reclaim() = (GC.gc(); BACKEND == "gpu" && CUDA.reclaim())
    size_keys(model) = Dict(PFX * "nvar" => model.meta.nvar, PFX * "ncon" => model.meta.ncon)
    status(res) = string(res.status)
    converged(res) = res.status in (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
    theta(model, res) = Array(ExaModels.solution(res, model.theta))
    iterations(res) = res.iter
    objective(res) = res.objective
end

# PEtab.jl
if BACKEND == "petab"
    build(yaml) = PEtabODEProblem(PEtabModel(yaml; verbose = false); verbose = false)

    # PEtab.jl's recommended optimizer by its own model size
    function optimizer(prob)
        model_size = PEtab._get_model_size(prob.model_info.model.sys_mutated, prob.model_info)
        return model_size == :Small  ? Optim.IPNewton() :
               model_size == :Medium ? Fides.CustomHessian() : Fides.BFGS()
    end
    optimizer_label(prob) = optimizer(prob) isa Optim.IPNewton ? "IPN" :
                            optimizer(prob) isa Fides.CustomHessian ? "GN" : "BFGS"

    # PEtab.jl's default options with the iteration and wall caps of the benchmark
    options(opt; iterations = MAX_ITER, time_limit = SOLVE_LIMIT) = opt isa Fides.HessianUpdate ?
        FidesOptions(maxiter = iterations, maxtime = time_limit) :
        Optim.Options(; iterations, time_limit, show_trace = false, allow_f_increases = true,
            successive_f_tol = 3, f_reltol = 1.0e-8, g_tol = 1.0e-6, x_abstol = 0.0)

    solve(prob) = calibrate(prob, get_x(prob), optimizer(prob); options = options(optimizer(prob)))
    warmup_solve(prob) = calibrate(prob, get_x(prob), optimizer(prob);
        options = options(optimizer(prob); iterations = 20, time_limit = 600.0))
    sync() = nothing
    reclaim() = GC.gc()
    size_keys(prob) = Dict(PFX * "nvar" => prob.nparameters_estimate, PFX * "optimizer" => optimizer_label(prob))

    function status(res)
        c = res.converged
        c === :Optmisation_failed && return "error"
        c === true && return Optim.g_converged(res.original) ? "converged_g" :
                             Optim.f_converged(res.original) ? "converged_f" : "converged_x"
        c === :GTOL && return "converged_g"
        c === :FTOL && return "converged_f"
        c === :XTOL && return "converged_x"
        (c === :MAXTIME || res.runtime >= SOLVE_LIMIT) && return "timeout"
        return "not_converged"
    end
    converged(res) = startswith(status(res), "converged")
    theta(prob, res) = [res.xmin[findfirst(==(id), prob.xnames)] for id in theta_ids(get_yaml(MODEL))]
    iterations(res) = res.niterations
    objective(res) = res.fmin
end

# Warmup: untimed build and solve of WARMUP so the target does not pay the JIT
function warmup()
    @info "warmup on $WARMUP"
    model = build(get_yaml(WARMUP))
    sync()
    warmup_solve(model)
    model = nothing
    reclaim()
end

function main()
    yaml = get_yaml(MODEL)
    warmup()

    # Build
    write_result(MODEL, Dict(PFX * "stage" => "build"))
    @info "build $MODEL"
    local model
    try
        t = time()
        model = with_hard_deadline(BUILD_LIMIT) do
            m = build(yaml)
            sync()
            m
        end
        write_result(MODEL, merge(Dict{String, Any}(PFX * "build_time" => time() - t), size_keys(model)))
    catch e
        write_result(MODEL, Dict(PFX * "status" => "error", PFX * "error" => sprint(showerror, e), PFX * "stage" => "done"))
        @error "build failed" exception = (e, catch_backtrace())
        return
    end

    # Solve
    write_result(MODEL, Dict(PFX * "stage" => "solve"))
    @info "solve $MODEL"
    local res
    try
        t = time()
        res = with_hard_deadline(SOLVE_LIMIT + 300.0) do
            solve(model)
        end
        write_result(MODEL, Dict(
            PFX * "solve_time" => time() - t,
            PFX * "status" => status(res),
            PFX * "iter" => iterations(res),
            PFX * "objective" => objective(res),
        ))
    catch e
        write_result(MODEL, Dict(PFX * "status" => "error", PFX * "error" => sprint(showerror, e), PFX * "stage" => "done"))
        @error "solve failed" exception = (e, catch_backtrace())
        return
    end

    # Objective at the final iterate by a forward ODE solve
    th = theta(model, res)
    write_result(MODEL, Dict(PFX * "theta" => join(th, ","), PFX * "objective_eval" => objective_at(yaml, th)))

    # Reruns
    if converged(res)
        write_result(MODEL, Dict(PFX * "stage" => "rerun"))
        times = Float64[]
        rerun_status = "ok"
        for i in 1:N_RERUNS
            @info "rerun $i/$N_RERUNS $MODEL"
            reclaim()
            t = time()
            r = with_hard_deadline(SOLVE_LIMIT + 300.0) do
                solve(model)
            end
            push!(times, time() - t)
            write_result(MODEL, Dict(PFX * "rerun_times" => join(times, ",")))
            converged(r) || (rerun_status = status(r); break)
        end
        write_result(MODEL, Dict(PFX * "rerun_status" => rerun_status))
    end

    write_result(MODEL, Dict(PFX * "stage" => "done"))
    @info "done $MODEL"
end

main()
