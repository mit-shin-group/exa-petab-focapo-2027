# Assemble results/*.txt into results_table.txt

using Printf
include(joinpath(@__DIR__, "common.jl"))

const REPORT = joinpath(@__DIR__, "..", "results_table.txt")

g(d, k) = get(d, k, "")
num(s) = tryparse(Float64, s)
short_name(model) = split(model, '_')[1]

# Build time, the limit on a killed build
function cmpl(d, pfx)
    g(d, pfx * "stage") == "build" && return @sprintf("%.2f", BUILD_LIMIT)
    t = num(g(d, pfx * "build_time"))
    t === nothing && return isempty(g(d, pfx * "error")) ? "-" : "E"
    return @sprintf("%.2f", t)
end

# Shifted geometric mean over the reruns, the limit on a timed-out first solve
function sol(d, pfx)
    g(d, pfx * "rerun_status") == "ok" && return @sprintf("%.3f", t_sgmdelta(parse_times(g(d, pfx * "rerun_times"))))
    st = g(d, pfx * "status")
    (st == "timeout" || st == "WALLTIME_EXCEEDED" || g(d, pfx * "stage") == "solve") && return @sprintf("%.3f", SOLVE_LIMIT)
    return "-"
end

# Relative objective gap of backend pfx against PEtab.jl
function rog(d, pfx)
    a, b = num(g(d, pfx * "objective_eval")), num(g(d, "petab_objective_eval"))
    (a === nothing || b === nothing || !isfinite(a) || !isfinite(b) || b == 0) && return nothing
    return (a - b) / abs(b)
end
rog_str(d, pfx) = (r = rog(d, pfx); r === nothing ? "-" : replace(@sprintf("%+.1e", r), "e+0" => "e+", "e-0" => "e-"))

const MADNLP_CODES = [
    ("SOLVE_SUCCEEDED", "0"), ("SOLVED_TO_ACCEPTABLE_LEVEL", "0A"), ("WALLTIME_EXCEEDED", "T"),
    ("RESTORATION_FAILED", "R"), ("SEARCH_DIRECTION_BECOMES_TOO_SMALL", "D"),
    ("INFEASIBLE_PROBLEM_DETECTED", "I"), ("DIVERGING_ITERATES", "DV"), ("MAXIMUM_ITERATIONS_EXCEEDED", "M"),
    ("timeout", "T"), ("error", "E"),
]
const PETAB_CODES = [
    ("converged_g", "0"), ("converged_f", "0F"), ("converged_x", "0X"), ("timeout", "T"),
    ("not_converged", "1"), ("error", "E"),
]
code(st, codes) = (i = findfirst(c -> c[1] == st, codes); i === nothing ? (isempty(st) ? "-" : "?") : codes[i][2])

# Status of backend pfx: the first solve, or the failed rerun with a star
function stat(d, pfx, codes)
    stage = g(d, pfx * "stage")
    stage == "build" && return "T"
    stage == "solve" && return "T"
    stage == "rerun" && return "T*"
    rerun = g(d, pfx * "rerun_status")
    rerun ∉ ("", "ok") && return code(rerun, codes) * "*"
    c = code(g(d, pfx * "status"), codes)
    r = rog(d, pfx)
    startswith(c, "0") && r !== nothing && r >= SUBOPT_ROG && (c *= "S")
    return c
end

# Table
const W_NAME, W_CMPL, W_SOL, W_STAT, W_ROG, W_OPT = 14, 8, 8, 4, 8, 4
const W_EXA   = W_CMPL + 1 + W_SOL + 2 + W_STAT + 1 + W_ROG
const W_PETAB = W_CMPL + 1 + W_SOL + 2 + W_STAT + 1 + W_OPT
center(s, n) = (l = div(n - length(s), 2); " "^l * s * " "^(n - length(s) - l))

D = Dict(m => read_result(m) for m in ALL_MODELS)
buf = IOBuffer()
sub_hdr = @sprintf("%-*s | %*s %*s  %*s %*s | %*s %*s  %*s %*s | %*s %*s  %*s %*s",
    W_NAME, "Model",
    W_CMPL, "CMPL(s)", W_SOL, "SOL(s)", W_STAT, "STAT", W_ROG, "ROG(-)",
    W_CMPL, "CMPL(s)", W_SOL, "SOL(s)", W_STAT, "STAT", W_ROG, "ROG(-)",
    W_CMPL, "CMPL(s)", W_SOL, "SOL(s)", W_STAT, "STAT", W_OPT, "OPT")
major_hdr = @sprintf("%-*s | %s | %s | %s", W_NAME, "",
    center("ExaModels + MadNLP (GPU)", W_EXA), center("ExaModels + MadNLP (CPU)", W_EXA), center("PEtab.jl (CPU)", W_PETAB))
bar, sep = "="^length(sub_hdr), "-"^length(sub_hdr)

println(buf, bar); println(buf, major_hdr); println(buf, sub_hdr); println(buf, sep)
for m in ALL_MODELS
    d = D[m]
    @printf(buf, "%-*s | %*s %*s  %*s %*s | %*s %*s  %*s %*s | %*s %*s  %*s %*s\n",
        W_NAME, short_name(m),
        W_CMPL, cmpl(d, "gpu_"), W_SOL, sol(d, "gpu_"), W_STAT, stat(d, "gpu_", MADNLP_CODES), W_ROG, rog_str(d, "gpu_"),
        W_CMPL, cmpl(d, "cpu_"), W_SOL, sol(d, "cpu_"), W_STAT, stat(d, "cpu_", MADNLP_CODES), W_ROG, rog_str(d, "cpu_"),
        W_CMPL, cmpl(d, "petab_"), W_SOL, sol(d, "petab_"), W_STAT, stat(d, "petab_", PETAB_CODES), W_OPT, (isempty(g(d, "petab_optimizer")) ? "-" : g(d, "petab_optimizer")))
end
println(buf, sep)

# Summary
built(pfx) = count(m -> num(g(D[m], pfx * "build_time")) !== nothing, ALL_MODELS)
solved(pfx, codes) = count(m -> startswith(stat(D[m], pfx, codes), "0"), ALL_MODELS)
subopt(pfx, codes) = count(m -> occursin("S", stat(D[m], pfx, codes)), ALL_MODELS)
n = length(ALL_MODELS)
println(buf, "\nSUMMARY")
@printf(buf, "  ExaModels built (GPU)  : %2d / %d\n", built("gpu_"), n)
@printf(buf, "  ExaModels solved (GPU) : %2d / %d  (%d suboptimal)\n", solved("gpu_", MADNLP_CODES), n, subopt("gpu_", MADNLP_CODES))
@printf(buf, "  ExaModels built (CPU)  : %2d / %d\n", built("cpu_"), n)
@printf(buf, "  ExaModels solved (CPU) : %2d / %d  (%d suboptimal)\n", solved("cpu_", MADNLP_CODES), n, subopt("cpu_", MADNLP_CODES))
@printf(buf, "  PEtab.jl built         : %2d / %d\n", built("petab_"), n)
@printf(buf, "  PEtab.jl solved        : %2d / %d\n", solved("petab_", PETAB_CODES), n)

println(buf, "\nTABLE KEY")
println(buf, "  CMPL(s) := model build time, $(Int(BUILD_LIMIT)) = timed out, E = errored")
println(buf, "  SOL(s)  := shifted geometric mean (δ = $(SGM_SHIFT) s) of $N_RERUNS solve reruns after a converged first solve, $(Int(SOLVE_LIMIT)) = timed out")
println(buf, "  STAT    := solver status of the first solve, * = a rerun ended with this status, S = suboptimal (ROG >= $SUBOPT_ROG)")
println(buf, "  ROG(-)  := relative objective gap (obj(theta) - obj(theta_petab)) / |obj(theta_petab)|, both by evaluate_objective")
println(buf, "  OPT     := PEtab.jl optimizer (IPN = Optim.IPNewton, GN = Fides.CustomHessian, BFGS = Fides.BFGS)")

madnlp_desc = Dict("0" => "SOLVE_SUCCEEDED", "0A" => "SOLVED_TO_ACCEPTABLE_LEVEL", "T" => "WALLTIME_EXCEEDED (timeout)",
    "R" => "RESTORATION_FAILED", "D" => "SEARCH_DIRECTION_BECOMES_TOO_SMALL", "I" => "INFEASIBLE_PROBLEM_DETECTED",
    "DV" => "DIVERGING_ITERATES", "M" => "MAXIMUM_ITERATIONS_EXCEEDED", "E" => "error", "?" => "other", "-" => "not run")
petab_desc = Dict("0" => "converged by gradient", "0F" => "converged by objective", "0X" => "converged by step",
    "T" => "timeout", "1" => "not converged", "E" => "error", "?" => "other", "-" => "not run")
base(c) = rstrip(rstrip(c, '*'), 'S')
present(pfx, codes) = Set(base(stat(D[m], pfx, codes)) for m in ALL_MODELS)
println(buf, "\n  MadNLP STATUS KEY")
for c in ["0", "0A", "T", "R", "D", "I", "DV", "M", "E", "?", "-"]
    c in union(present("gpu_", MADNLP_CODES), present("cpu_", MADNLP_CODES)) && println(buf, "   ", lpad(c, 3), " : ", madnlp_desc[c])
end
println(buf, "\n  PEtab STATUS KEY")
for c in ["0", "0F", "0X", "T", "1", "E", "?", "-"]
    c in present("petab_", PETAB_CODES) && println(buf, "   ", lpad(c, 3), " : ", petab_desc[c])
end
println(buf, "\n", bar)

report = String(take!(buf))
print(report)
write(REPORT, report)
