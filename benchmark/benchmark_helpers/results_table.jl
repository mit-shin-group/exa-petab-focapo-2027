# results_table.jl — assembles the benchmark table from the single results dir and writes it to
# results_table.txt. GPU, CPU, and the PEtab optimizers all live in one dir (the configured BENCH_TAG,
# e.g. focapo), distinguished by key prefix: exagpu_* (GPU), exacpu_* (CPU), petab_<label>_*
# (label = IPNewton / GaussNewton / BFGS). The PEtab column shows the fastest converged optimizer per
# model; TAG names the winner.
#   julia --project=. benchmark_helpers/results_table.jl          # warm (SGM) solve times — DEFAULT
#   julia --project=. benchmark_helpers/results_table.jl --cold   # cold first-run solve times instead

using Printf

const REPORT_TXT = joinpath(@__DIR__, "..", "results_table.txt")
const USE_SGM    = !("--cold" in ARGS)

include(joinpath(@__DIR__, "..", "options.jl"))   # MODELDIR + RESULTDIR + model sets + BENCH_* + t_sgmdelta

const MODELS    = BENCHMARK_MODELS  # benchmarked set (already sorted)
const SGM_N     = BENCH_SGM_N
const SGM_SHIFT = BENCH_SGM_SHIFT

# ROG threshold: a converged solve worse than PEtab's objective by ≥ this is suboptimal (0S/0AS),
# excluded from the "ExaModels solved" count.
const SUBOPT_ROG = 0.02

# Short labels for the PEtab TAG subcolumn.
const TAG_ABBR = Dict("IPNewton" => "IPN", "GaussNewton" => "GN", "BFGS" => "BFGS")

# ─── helpers ──────────────────────────────────────────────────────────────────
function read_result(dir, m)
    d = Dict{String,String}(); p = joinpath(dir, "$(m)_results.txt")
    isfile(p) || return d
    for line in eachline(p)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    d
end
read_model(m) = read_result(RESULTDIR, m)
g(d, k)   = get(d, k, "")
fparse(s) = tryparse(Float64, s)
short_name(m) = split(m, '_')[1]
# format a numeric string for display only (values are stored raw); pass "-" through
disp(s)     = (v = tryparse(Float64, s); v === nothing ? s : string(round(v; digits = 2)))  # CMPL: 2 dp
disp_sol(s) = (v = tryparse(Float64, s); v === nothing ? s : @sprintf("%.3f", v))            # SOL: fixed 3 dp

function center_str(s, n)
    len = length(s); len >= n && return s[1:n]
    l = div(n - len, 2); " "^l * s * " "^(n - len - l)
end

# PEtab optimizer labels present for a model (those with a petab_<label>_solve_status key).
function petab_labels(d)
    labs = String[]
    for k in keys(d)
        mt = match(r"^petab_(.+)_solve_status$", k)
        mt !== nothing && push!(labs, mt.captures[1])
    end
    sort!(unique!(labs))
end

# ROG(-) = (exa's petab_obj − po) / |po|, po = the PEtab reference objective. Reads <pfx>petab_obj,
# falling back to <pfx>objective.
function gap_val(d, pfx, po)
    eo = fparse(g(d, pfx * "petab_obj")); eo === nothing && (eo = fparse(g(d, pfx * "objective")))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) && return nothing
    (eo - po) / abs(po)
end
gap_str(d, pfx, po) = (gp = gap_val(d, pfx, po); gp === nothing ? "-" :
                   replace(@sprintf("%+.1e", gp), "e+0" => "e+", "e-0" => "e-"))

# MadNLP status code for backend pfx (exagpu_/exacpu_); po is the PEtab reference objective for ROG.
function madnlp_code(d, pfx, po)
    cs = g(d, pfx * "compile_status")
    cs == "timeout" && return "T"   # build exceeded COMPILE_LIMIT
    cs == "error"   && return "E"   # build failed
    cs != "ok"      && return "-"   # missing_yaml / skipped / not run
    ss = g(d, pfx * "solve_status")
    ss == "skipped" && return "-"
    ss == "timeout" && return "T"
    ss == "error" && return "E"
    term = uppercase(g(d, pfx * "term_status"))
    isempty(term) && return "-"
    if occursin("SUCCEEDED", term) || occursin("ACCEPTABLE", term)
        base = occursin("ACCEPTABLE", term) ? "0A" : "0"
        gp   = gap_val(d, pfx, po)
        return (gp !== nothing && gp >= SUBOPT_ROG) ? base * "S" : base
    end
    occursin("WALLTIME",         term) && return "T"
    occursin("RESTORATION",      term) && return "R"
    occursin("SEARCH_DIRECTION", term) && return "D"
    occursin("INFEASIBLE",       term) && return "I"
    return "5"
end

# PEtab status code for optimizer prefix pfx.
function petab_code(d, pfx)
    cs = g(d, pfx * "compile_status")
    cs == "timeout" && return "T"   # build exceeded COMPILE_LIMIT
    cs == "error"   && return "E"   # build failed
    cs != "ok"      && return "-"   # missing_yaml / skipped / not run
    ss = g(d, pfx * "solve_status")
    ss == "timeout" && return "T"
    ss == "error" && return "E"
    if g(d, pfx * "optimum_found") == "true"
        # which convergence criterion fired; precedence: gradient → F objective → X step
        g(d, pfx * "gconverged") == "true" && return "0"
        g(d, pfx * "fconverged") == "true" && return "0F"
        g(d, pfx * "xconverged") == "true" && return "0X"
        return "0"
    end
    if g(d, pfx * "optimum_found") == "false"
        st = fparse(g(d, pfx * "solve_time"))
        (st !== nothing && st >= 0.99 * BENCH_SOLVE_LIMIT) && return "T"
        return "1"
    end
    return "-"
end

pct_exa_str(d, pfx) = begin
    ct = fparse(g(d, pfx * "compile_time")); pt = fparse(g(d, pfx * "presolve_time"))
    (ct === nothing || pt === nothing || ct <= 0.0) ? "-" : @sprintf("%4.0f%%", 100.0 * (ct - pt) / ct)
end
fmt_cmp(d, pfx) = (g(d, pfx * "compile_status") == "ok" && !isempty(g(d, pfx * "compile_time"))) ? g(d, pfx * "compile_time") : "-"
fmt_slv(d, pfx) = begin
    if USE_SGM && g(d, pfx * "sgm_status") == "ok"
        raw = g(d, pfx * "solve_times")
        ts  = isempty(raw) ? Float64[] : Float64[x for x in (tryparse(Float64, s) for s in split(raw, ",")) if x !== nothing]
        if !isempty(ts)
            string(t_sgmdelta(ts, SGM_SHIFT))
        elseif !isempty(g(d, pfx * "sgm_solve_time"))
            g(d, pfx * "sgm_solve_time")
        else
            "-"
        end
    elseif !USE_SGM && g(d, pfx * "solve_status") == "ok" && !isempty(g(d, pfx * "solve_time"))
        g(d, pfx * "solve_time")
    elseif occursin("WALLTIME", uppercase(g(d, pfx * "term_status"))) && !isempty(g(d, pfx * "solve_time"))
        g(d, pfx * "solve_time")
    elseif !isempty(g(d, pfx * "solve_time")) &&
           (st = fparse(g(d, pfx * "solve_time"))) !== nothing && st >= 0.99 * BENCH_SOLVE_LIMIT
        g(d, pfx * "solve_time")
    else
        "-"
    end
end

# numeric solve time used to rank optimizers (smaller is better); Inf if not usable.
slv_num(d, pfx) = (v = tryparse(Float64, fmt_slv(d, pfx)); v === nothing ? Inf : v)
petab_solved(d, pfx) = g(d, pfx * "optimum_found") == "true"

# Winning PEtab optimizer label for a model: prefer a converged optimizer, then the fastest. "" if none.
function pick_petab(d)
    labs = petab_labels(d)
    isempty(labs) && return ""
    cand = [(lab, petab_solved(d, "petab_$(lab)_"), slv_num(d, "petab_$(lab)_")) for lab in labs]
    conv = filter(x -> x[2], cand); pool = isempty(conv) ? cand : conv
    best = pool[1]
    for x in pool[2:end]; x[3] < best[3] && (best = x); end
    best[1]
end

# ─── column widths ────────────────────────────────────────────────────────────
const W_NAME = 14
const W_CMPL, W_PCT, W_SOL, W_STAT, W_GAP, W_TAG = 7, 6, 8, 4, 8, 4   # W_SOL=8 fits fixed 3-dp solve times
const W_EXA_INNER   = W_CMPL + 1 + W_PCT + 1 + W_SOL + 2 + W_STAT + 1 + W_GAP   # GPU & CPU groups
const W_PETAB_INNER = W_CMPL + 1 + W_SOL + 2 + W_STAT + 1 + W_TAG               # PEtab (no GAP; + TAG)

# ─── build report ───────────────────────────────────────────────────────────────
buf = IOBuffer()
# One dict per model (all backends + optimizers); the PEtab winner label + its reference objective.
D         = [read_model(m) for m in MODELS]
petab_win = [pick_petab(d) for d in D]
petab_pfx = [isempty(petab_win[i]) ? "petab_" : "petab_$(petab_win[i])_" for i in eachindex(MODELS)]
petab_po  = [isempty(petab_win[i]) ? nothing : fparse(g(D[i], petab_pfx[i] * "objective")) for i in eachindex(MODELS)]

major_hdr = @sprintf("%-*s | %s | %s | %s",
    W_NAME, "",
    center_str("ExaModels + MadNLP (GPU)", W_EXA_INNER),
    center_str("ExaModels + MadNLP (CPU)", W_EXA_INNER),
    center_str("PEtab.jl + Best Optimizer", W_PETAB_INNER))
sub_hdr = @sprintf("%-*s | %*s %*s %*s  %*s %*s | %*s %*s %*s  %*s %*s | %*s %*s  %*s %*s",
    W_NAME, "Model",
    W_CMPL,"CMPL(s)", W_PCT,"EXA(%)", W_SOL,"SOL(s)", W_STAT,"STAT", W_GAP,"ROG(-)",
    W_CMPL,"CMPL(s)", W_PCT,"EXA(%)", W_SOL,"SOL(s)", W_STAT,"STAT", W_GAP,"ROG(-)",
    W_CMPL,"CMPL(s)", W_SOL,"SOL(s)", W_STAT,"STAT", W_TAG,"TAG")
bar = "="^length(sub_hdr); sep = "-"^length(sub_hdr)

println(buf, bar); println(buf, major_hdr); println(buf, sub_hdr); println(buf, sep)
for (i, m) in enumerate(MODELS)
    d = D[i]; pp = petab_pfx[i]; po = petab_po[i]
    tag_lbl = isempty(petab_win[i]) ? "-" : get(TAG_ABBR, petab_win[i], petab_win[i])
    @printf(buf, "%-*s | %*s %*s %*s  %*s %*s | %*s %*s %*s  %*s %*s | %*s %*s  %*s %*s\n",
        W_NAME, short_name(m),
        W_CMPL,disp(fmt_cmp(d,"exagpu_")), W_PCT,pct_exa_str(d,"exagpu_"), W_SOL,disp_sol(fmt_slv(d,"exagpu_")), W_STAT,madnlp_code(d,"exagpu_",po), W_GAP,gap_str(d,"exagpu_",po),
        W_CMPL,disp(fmt_cmp(d,"exacpu_")), W_PCT,pct_exa_str(d,"exacpu_"), W_SOL,disp_sol(fmt_slv(d,"exacpu_")), W_STAT,madnlp_code(d,"exacpu_",po), W_GAP,gap_str(d,"exacpu_",po),
        W_CMPL,disp(fmt_cmp(d,pp)),        W_SOL,disp_sol(fmt_slv(d,pp)),                          W_STAT,petab_code(d,pp), W_TAG,tag_lbl)
end
println(buf, sep)

# ─── summary (GPU is the primary ExaModels backend) ─────────────────────────────
exa_opt(i)    = madnlp_code(D[i],"exagpu_",petab_po[i]) in ("0", "0A")
exa_subopt(i) = madnlp_code(D[i],"exagpu_",petab_po[i]) in ("0S", "0AS")

println(buf, "\nSUMMARY (ExaModelsPEtab target set: continuous + PEtab-solved)")
@printf(buf, "  Target models          : %2d       (of %d; %d 'Possible Discontinuities', %d PEtab.jl failed compile)\n",
        length(BENCHMARK_MODELS), length(ALL_MODELS), length(EXCLUDED_MODELS), length(FAILED_MODELS))
@printf(buf, "  ExaModels solved (GPU) : %2d / %2d  (status 0 + 0A, full/acceptable optimum)\n", count(exa_opt, eachindex(MODELS)), length(MODELS))
@printf(buf, "  Solved-but-suboptimal  : %2d       (0S / 0AS, converged but ROG ≥ %.2f vs PEtab; excluded above)\n", count(exa_subopt, eachindex(MODELS)), SUBOPT_ROG)

println(buf, "")
println(buf, "  Source dir: benchmark_results_$(BENCH_TAG)/  (GPU=exagpu_*, CPU=exacpu_*, PEtab=petab_<opt>_*, fastest optimizer shown)")
println(buf, "  CMPL(s) := Model compilation time")
println(buf, "  EXA(%)  := Fraction of model compile time spent on actual ExaModels build (PEtab setup + mesh generation)")
println(buf, "  SOL(s)  := Solver solve time, shifted geometric mean (by δ = $(SGM_SHIFT)s) over n=$SGM_N reruns")
println(buf, "  STAT    := Solver status")
println(buf, "  ROG(-)  := relative objective gap = (petab.nllh(exa_p*) - petab_obj) / |petab_obj|  (negative => ExaModels lower)")
println(buf, "  TAG     := Fastest PEtab optimizer for the model (IPN=Optim.IPNewton, GN=Fides.CustomHessian/GaussNewton, BFGS=Fides.BFGS)")

# ─── status key (only codes present in the table) ───────────────────────────────
madnlp_desc = Dict("0"=>"SOLVE_SUCCEEDED", "0A"=>"SOLVED_TO_ACCEPTABLE_LEVEL",
    "0S"=>"SOLVE_SUCCEEDED, suboptimal (ROG ≥ $(SUBOPT_ROG) vs PEtab)",
    "0AS"=>"SOLVED_TO_ACCEPTABLE_LEVEL, suboptimal (ROG ≥ $(SUBOPT_ROG) vs PEtab)",
    "T"=>"WALLTIME_EXCEEDED (timeout)", "R"=>"RESTORATION_FAILED", "D"=>"SEARCH_DIRECTION_BECOMES_TOO_SMALL",
    "I"=>"INFEASIBLE_PROBLEM_DETECTED", "5"=>"other", "E"=>"Error", "-"=>"compile_failed/not_run")
const MADNLP_ORDER = ["0","0A","0S","0AS","T","R","D","I","5","E","-"]
petab_desc = Dict(
    "0"  => "Converged by gradient, ‖g‖∞ ≤ tol",
    "0F" => "Converged by objective (F), |Δf| ≤ ftol·|f| (with ‖g‖ > tol)",
    "0X" => "Converged by step (X), ‖Δx‖∞ ≤ xtol (with ‖g‖ > tol)",
    "T"  => "Walltime exceeded (timeout)",
    "1"  => "Not converged",
    "E"  => "Error", "-" => "compile_failed/not_run")
const PETAB_ORDER = ["0","0F","0X","T","1","E","-"]

madnlp_present = Set{String}()
for i in eachindex(MODELS)
    push!(madnlp_present, madnlp_code(D[i],"exagpu_",petab_po[i]))
    push!(madnlp_present, madnlp_code(D[i],"exacpu_",petab_po[i]))
end
petab_present = Set(petab_code(D[i], petab_pfx[i]) for i in eachindex(MODELS))
keyline(code, desc) = "   " * lpad(code, 3) * " : " * desc

println(buf, "\n  MadNLP (Status)")
for code in MADNLP_ORDER; code in madnlp_present && println(buf, keyline(code, madnlp_desc[code])); end
println(buf, "\n  PEtab  (Status)")
for code in PETAB_ORDER;  code in petab_present  && println(buf, keyline(code, petab_desc[code]));  end
println(buf, "\n", bar)

# ─── output ───────────────────────────────────────────────────────────────────
report = String(take!(buf))
print(report)
open(REPORT_TXT, "w") do io; print(io, report); end
println("\nReport saved to: $REPORT_TXT")
