# results_table.jl — assembles the benchmark table from the per-tag result dirs and writes it to
# results_table.txt. Each tag is backend-isolated (see options.jl / run_benchmarks.sh), so the
# columns are pulled from DIFFERENT dirs under benchmark_results/:
#   ExaModels GPU  <- the GPU tag                                   (exagpu_*)
#   ExaModels CPU  <- the pinned CPU tag (CPU_PIN = CPUma57)         (exacpu_*)
#   PEtab          <- FASTEST of the PEtab optimizer tags per model (petab_*), winner shown in TAG
# For a single-tag run only that tag's dir is populated, so the other columns show "-"; once the GPU,
# CPUma57 and PEtab tag dirs all exist the assembled report fills every column.
#   julia --project=. benchmark_helpers/results_table.jl          # warm (SGM) solve times — DEFAULT
#   julia --project=. benchmark_helpers/results_table.jl --cold   # cold first-run solve times instead

using Printf

const REPORT_TXT = joinpath(@__DIR__, "..", "results_table.txt")
const USE_SGM    = !("--cold" in ARGS)

include(joinpath(@__DIR__, "..", "options.jl"))   # MODELDIR + model sets + BENCH_* + shifted_geomean

const MODELS    = BENCHMARK_MODELS  # the benchmarked set (already sorted)
const SGM_N     = BENCH_SGM_N
const SGM_SHIFT = BENCH_SGM_SHIFT

# A SOLVE_SUCCEEDED/ACCEPTABLE whose objective is worse than PEtab's by ≥ this % is converged-but-
# suboptimal (0S / 0AS) and is EXCLUDED from the "ExaModels solved" count.
const SUBOPT_ROG = 0.02

# ─── tag discovery + per-backend source selection ──────────────────────────────
const RESULTS_ROOT = joinpath(@__DIR__, "..", "benchmark_results")
# All tag dirs present (benchmark_results_<tag>/), minus the legacy _0 scratch run.
available_tags() = sort(String[replace(d, "benchmark_results_" => "")
    for d in readdir(RESULTS_ROOT)
    if isdir(joinpath(RESULTS_ROOT, d)) && startswith(d, "benchmark_results_") && d != "benchmark_results_0"])
tagdir(t) = joinpath(RESULTS_ROOT, "benchmark_results_$t")

# Backend → candidate tags (string predicates mirror options.jl BENCH_INCLUDE_*). exa-CPU is pinned
# to one solver tag for now (no TAG column); set CPU_PIN = nothing to enable fastest-across-CPU-tags.
const TAGS       = available_tags()
const GPU_TAGS   = filter(==("GPU"), TAGS)
const CPU_PIN    = "CPUma57"
const CPU_TAGS   = CPU_PIN === nothing ? filter(t -> startswith(t, "CPUma"), TAGS) :
                                         filter(==(CPU_PIN), TAGS)
const PETAB_TAGS = filter(t -> t in ("IPNewton", "GaussNewton", "BFGS"), TAGS)
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
g(d, k)   = get(d, k, "")
fparse(s) = tryparse(Float64, s)
short_name(m) = split(m, '_')[1]

function center_str(s, n)
    len = length(s); len >= n && return s[1:n]
    l = div(n - len, 2); " "^l * s * " "^(n - len - l)
end

# ROG(-) = (petab_obj(exa_p*) - petab_obj) / |petab_obj| — dimensionless relative objective gap; ExaModels' optimal parameters scored
# under PEtab's OWN objective, vs PEtab's optimum (fair, same-objective). The benchmark stores
# petab_obj(exa_p*) as `<pfx>petab_obj`; until a run populates it we fall back to the ExaModels
# objective `<pfx>objective` so the column is not empty. The PEtab objective comes from the SAME dict
# (each backend dict carries its own petab_objective; for exa-vs-PEtab gap we pass the PEtab dict's).
function gap_val(d, pfx, pd)
    eo = fparse(g(d, pfx * "petab_obj")); eo === nothing && (eo = fparse(g(d, pfx * "objective")))
    po = fparse(g(pd, "petab_objective"))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) && return nothing
    (eo - po) / abs(po)
end
gap_str(d, pfx, pd) = (gp = gap_val(d, pfx, pd); gp === nothing ? "-" :
                   replace(@sprintf("%+.1e", gp), "e+0" => "e+", "e-0" => "e-"))

# ExaModels MadNLP status code for backend `pfx` (exagpu_ / exacpu_). `pd` = the PEtab dict for GAP.
function madnlp_code(d, pfx, pd)
    g(d, pfx * "compile_status") != "ok" && return "-"
    ss = g(d, pfx * "solve_status")
    ss == "skipped" && return "-"
    ss == "timeout" && return "T"
    ss == "error" && return "E"
    term = uppercase(g(d, pfx * "term_status"))
    isempty(term) && return "-"
    if occursin("SUCCEEDED", term) || occursin("ACCEPTABLE", term)
        base = occursin("ACCEPTABLE", term) ? "0A" : "0"
        gp   = gap_val(d, pfx, pd)
        return (gp !== nothing && gp >= SUBOPT_ROG) ? base * "S" : base
    end
    occursin("WALLTIME",         term) && return "T"
    occursin("RESTORATION",      term) && return "R"
    occursin("SEARCH_DIRECTION", term) && return "D"
    return "5"
end

function petab_code(d)
    g(d, "petab_compile_status") != "ok" && return "-"
    ss = g(d, "petab_solve_status")
    ss in ("skipped", "timeout") && return "-"
    ss == "error" && return "E"
    if g(d, "petab_optimum_found") == "true"
        # WHICH convergence criterion fired (Optim per-criterion booleans / Fides retcode → same flags).
        # Precedence: gradient (genuine first-order stationary) → F objective plateau → X zero-step.
        g(d, "petab_gconverged") == "true" && return "0"
        g(d, "petab_fconverged") == "true" && return "0F"
        g(d, "petab_xconverged") == "true" && return "0X"
        return "0"
    end
    if g(d, "petab_optimum_found") == "false"
        st = fparse(g(d, "petab_solve_time"))
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
            string(shifted_geomean(ts, SGM_SHIFT))
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

# numeric solve time used to rank tags within a backend (smaller is better); Inf if not usable.
slv_num(d, pfx) = (v = tryparse(Float64, fmt_slv(d, pfx)); v === nothing ? Inf : v)
exa_solved(d, pfx) = madnlp_code(d, pfx, d) in ("0", "0A", "0S", "0AS")
petab_solved(d)    = g(d, "petab_optimum_found") == "true"

# Pick the best (tag, dict) for model m across candidate `tags`: prefer a converged solve, then the
# fastest. Returns ("", empty) if no candidate tag has a record for this model.
function pick(m, tags, pfx, solved)
    cands = [(t, read_result(tagdir(t), m)) for t in tags]
    cands = [c for c in cands if !isempty(c[2])]
    isempty(cands) && return ("", Dict{String,String}())
    conv = filter(c -> solved(c[2]), cands)
    pool = isempty(conv) ? cands : conv
    best = pool[1]
    for c in pool[2:end]
        slv_num(c[2], pfx) < slv_num(best[2], pfx) && (best = c)
    end
    best
end

# ─── column widths ────────────────────────────────────────────────────────────
const W_NAME = 14
const W_CMPL, W_PCT, W_SOL, W_STAT, W_GAP, W_TAG = 7, 6, 7, 4, 8, 4
const W_EXA_INNER   = W_CMPL + 1 + W_PCT + 1 + W_SOL + 2 + W_STAT + 1 + W_GAP   # GPU & CPU groups
const W_PETAB_INNER = W_CMPL + 1 + W_SOL + 2 + W_STAT + 1 + W_TAG               # PEtab (no GAP; + TAG)

# ─── build report ───────────────────────────────────────────────────────────────
buf = IOBuffer()
# Per-model selected dicts (per backend): GPU/CPU from their tag dirs, PEtab the fastest tag + winner.
gpu_sel   = [pick(m, GPU_TAGS,   "exagpu_", d -> exa_solved(d, "exagpu_")) for m in MODELS]
cpu_sel   = [pick(m, CPU_TAGS,   "exacpu_", d -> exa_solved(d, "exacpu_")) for m in MODELS]
petab_sel = [pick(m, PETAB_TAGS, "petab_",  petab_solved)                  for m in MODELS]
all_dg = [s[2] for s in gpu_sel]

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
    dg = gpu_sel[i][2]; dc = cpu_sel[i][2]; tp, dp = petab_sel[i]
    tag_lbl = isempty(tp) ? "-" : get(TAG_ABBR, tp, tp)
    @printf(buf, "%-*s | %*s %*s %*s  %*s %*s | %*s %*s %*s  %*s %*s | %*s %*s  %*s %*s\n",
        W_NAME, short_name(m),
        W_CMPL,fmt_cmp(dg,"exagpu_"), W_PCT,pct_exa_str(dg,"exagpu_"), W_SOL,fmt_slv(dg,"exagpu_"), W_STAT,madnlp_code(dg,"exagpu_",dp), W_GAP,gap_str(dg,"exagpu_",dp),
        W_CMPL,fmt_cmp(dc,"exacpu_"), W_PCT,pct_exa_str(dc,"exacpu_"), W_SOL,fmt_slv(dc,"exacpu_"), W_STAT,madnlp_code(dc,"exacpu_",dp), W_GAP,gap_str(dc,"exacpu_",dp),
        W_CMPL,fmt_cmp(dp,"petab_"),  W_SOL,fmt_slv(dp,"petab_"),      W_STAT,petab_code(dp), W_TAG,tag_lbl)
end
println(buf, sep)

# ─── summary (GPU is the primary ExaModels backend) ─────────────────────────────
# Classify each model from (GPU dict, PEtab dict) so ROG is scored against PEtab's own objective.
# The GPU dict has no `petab_objective` key — passing it as `pd` silently zeroes the gap and the
# suboptimal (0S/0AS) flag never fires, undercounting the suboptimal class.
all_dgp = [(gpu_sel[i][2], petab_sel[i][2]) for i in eachindex(MODELS)]
exa_opt(p)    = madnlp_code(p[1],"exagpu_",p[2]) in ("0", "0A")
exa_subopt(p) = madnlp_code(p[1],"exagpu_",p[2]) in ("0S", "0AS")

println(buf, "\nSUMMARY (ExaModelsPEtab target set: continuous + PEtab-solved)")
@printf(buf, "  Target models          : %2d       (of %d; %d 'Possible Discontinuities', %d PEtab.jl failed compile)\n",
        length(BENCHMARK_MODELS), length(ALL_MODELS), length(EXCLUDED_MODELS), length(FAILED_MODELS))
@printf(buf, "  ExaModels solved (GPU) : %2d / %2d  (status 0 + 0A, full/acceptable optimum)\n", count(exa_opt, all_dgp), length(MODELS))
@printf(buf, "  Solved-but-suboptimal  : %2d       (0S / 0AS, converged but ROG ≥ %.2f vs PEtab; excluded above)\n", count(exa_subopt, all_dgp), SUBOPT_ROG)

println(buf, "")
println(buf, "  Sources: GPU=[$(join(GPU_TAGS, ","))]  CPU=[$(join(CPU_TAGS, ","))]  PEtab=[$(join(PETAB_TAGS, ","))]  (tag dirs under benchmark_results/)")
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
    "5"=>"other", "E"=>"Error", "-"=>"compile_failed/not_run")
const MADNLP_ORDER = ["0","0A","0S","0AS","T","R","D","5","E","-"]
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
    push!(madnlp_present, madnlp_code(gpu_sel[i][2],"exagpu_",petab_sel[i][2]))
    push!(madnlp_present, madnlp_code(cpu_sel[i][2],"exacpu_",petab_sel[i][2]))
end
petab_present = Set(petab_code(s[2]) for s in petab_sel)
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
