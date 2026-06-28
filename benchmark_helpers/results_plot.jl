# results_plot.jl — single scatter figure of solver speedup vs PEtab.jl over the exa target set.
#   x = nvar (NLP size), log10            — exa GPU and CPU share a model's nvar; PEtab points are
#                                            placed at that same model's nvar (PEtab has no NLP).
#   y = SGM10-time speedup vs PEtab.jl    — petab_best_sgm / solver_sgm  (log10). PEtab ⇒ 1 (the 10^0 line).
# Three series (legend order): ExaModels MadNLP (GPU), ExaModels MadNLP (CPU), PEtab.jl + Best optimizer (CPU).
#
# Like results_table.jl, columns come from DIFFERENT per-tag dirs under benchmark_results/:
#   GPU   <- the GPU tag (exagpu_*);  CPU <- the pinned CPU tag (CPU_PIN = CPUma57) (exacpu_*);
#   PEtab <- FASTEST converged of the PEtab optimizer tags per model (petab_*).
#
# Usage: julia --project=. benchmark_helpers/results_plot.jl   # writes results_plot.png at repo root.

using Plots
using Plots.PlotMeasures   # mm units for plot margins
gr()

const HERE      = @__DIR__
include(joinpath(HERE, "..", "options.jl"))   # model sets, shifted_geomean, BENCH_*

# ─── tag discovery + per-backend source selection (mirrors results_table.jl) ─────
const RESULTS_ROOT = joinpath(HERE, "..", "benchmark_results")
available_tags() = sort(String[replace(d, "benchmark_results_" => "")
    for d in readdir(RESULTS_ROOT)
    if isdir(joinpath(RESULTS_ROOT, d)) && startswith(d, "benchmark_results_") && d != "benchmark_results_0"])
tagdir(t) = joinpath(RESULTS_ROOT, "benchmark_results_$t")

const TAGS       = available_tags()
const GPU_TAGS   = filter(==("GPU"), TAGS)
const CPU_PIN    = "CPUma57"   # set to nothing to take the fastest across all CPUma* tags
const CPU_TAGS   = CPU_PIN === nothing ? filter(t -> startswith(t, "CPUma"), TAGS) :
                                         filter(==(CPU_PIN), TAGS)
const PETAB_TAGS = filter(t -> t in ("IPNewton", "GaussNewton", "BFGS"), TAGS)

function read_result(dir, m)
    d = Dict{String,String}(); p = joinpath(dir, "$(m)_results.txt")
    isfile(p) || return d
    for line in eachline(p)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    d
end

# SGM10 [s] for a backend prefix: report-time shift from raw solve_times when present, else the
# stored aggregate (legacy files). Returns nothing if there is no warm timing.
function sgm10(d, pfx)
    get(d, pfx * "sgm_status", "") == "ok" || return nothing
    raw = get(d, pfx * "solve_times", "")
    if !isempty(raw)
        ts = Float64[x for x in (tryparse(Float64, s) for s in split(raw, ",")) if x !== nothing]
        !isempty(ts) && return shifted_geomean(ts, BENCH_SGM_SHIFT)
    end
    tryparse(Float64, get(d, pfx * "sgm_solve_time", ""))
end

function get_nvar(d)
    for k in ("exagpu_nvar", "exacpu_nvar")
        v = tryparse(Int, get(d, k, "")); v !== nothing && return v
    end
    return nothing
end

# ROG and MadNLP status code, mirroring results_table.jl. A SOLVE_SUCCEEDED/ACCEPTABLE whose objective
# is ≥ SUBOPT_ROG worse than PEtab's (dimensionless relative objective gap) is converged-but-suboptimal
# (0S / 0AS) — often a fast bail to a worse optimum — and is EXCLUDED so the plot shows only 0 / 0A.
const SUBOPT_ROG = 0.02
# `po` is PEtab's reference optimum (key `petab_objective`), which lives in the PEtab tag dict, NOT
# the ExaModels dict `d`. It MUST be passed in: scoring the ROG against a value read off `d` (which
# only has `<pfx>petab_obj`, the PEtab objective AT the exa solution) silently zeroes the gap, so the
# suboptimal `S` flag never fires and 0S/0AS points get plotted instead of X'd. Same bug class as the
# results_table.jl fix.
function gap_val(d, pfx, po)
    eo = tryparse(Float64, get(d, pfx * "petab_obj", ""))
    eo === nothing && (eo = tryparse(Float64, get(d, pfx * "objective", "")))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) && return nothing
    (eo - po) / abs(po)
end
function madnlp_code(d, pfx, po)
    get(d, pfx * "compile_status", "") == "ok" || return "-"
    ss = get(d, pfx * "solve_status", "")
    ss == "skipped" && return "-"
    ss == "timeout" && return "T"
    ss == "error"   && return "E"
    term = uppercase(get(d, pfx * "term_status", ""))
    isempty(term) && return "-"
    if occursin("SUCCEEDED", term) || occursin("ACCEPTABLE", term)
        base = occursin("ACCEPTABLE", term) ? "0A" : "0"
        gp   = gap_val(d, pfx, po)
        return (gp !== nothing && gp >= SUBOPT_ROG) ? base * "S" : base
    end
    return "5"
end
# PEtab reference optimum for model m (or nothing) — the ROG denominator scored against the exa obj.
petab_ref_obj(pd) = tryparse(Float64, get(pd, "petab_objective", ""))

# Best converged PEtab baseline for model m: fastest petab_sgm across the PEtab optimizer tags.
# Returns (dict, sgm) or (empty, nothing) if no PEtab tag converged on this model.
function pick_petab(m)
    best = nothing
    for t in PETAB_TAGS
        d = read_result(tagdir(t), m)
        get(d, "petab_optimum_found", "") == "true" || continue
        s = sgm10(d, "petab_"); (s === nothing || s <= 0) && continue
        (best === nothing || s < best[2]) && (best = (d, s))
    end
    best === nothing ? (Dict{String,String}(), nothing) : best
end
# Single GPU tag dict (or empty); fastest converged CPU tag (by exacpu SGM) or empty.
gpu_result(m) = isempty(GPU_TAGS) ? Dict{String,String}() : read_result(tagdir(GPU_TAGS[1]), m)
function pick_cpu(m, po)
    best = nothing
    for t in CPU_TAGS
        d = read_result(tagdir(t), m)
        madnlp_code(d, "exacpu_", po) in ("0", "0A") || continue
        s = sgm10(d, "exacpu_"); (s === nothing || s <= 0) && continue
        (best === nothing || s < best[2]) && (best = (d, s))
    end
    best === nothing ? (Dict{String,String}(), nothing) : best
end
# nvar from whichever exa tag built the model (GPU first, then CPU tags).
function model_nvar(m)
    for t in vcat(GPU_TAGS, CPU_TAGS)
        v = get_nvar(read_result(tagdir(t), m)); v !== nothing && return v
    end
    nothing
end

# Julia logo colors
const J_PURPLE = RGB(0.584, 0.345, 0.698)
const J_GREEN  = RGB(0.220, 0.596, 0.149)
const J_RED    = RGB(0.796, 0.235, 0.200)

gpu_x = Float64[]; gpu_y = Float64[]
cpu_x = Float64[]; cpu_y = Float64[]
pet_x  = Float64[]; pet_y  = Float64[]   # PEtab baseline where ExaModels (GPU) also solved (0/0A)
petx_x = Float64[]; petx_y = Float64[]   # PEtab solved but ExaModels (GPU) did NOT ⇒ X'd boxes
for m in BENCHMARK_MODELS
    pd, pt = pick_petab(m); (pt === nothing || pt <= 0) && continue  # PEtab baseline must have solved
    po = petab_ref_obj(pd)                                           # ROG denominator (PEtab optimum)
    nv = model_nvar(m);     nv === nothing && continue               # never built ⇒ no x position
    gd = gpu_result(m)
    if madnlp_code(gd, "exagpu_", po) in ("0", "0A")                 # GPU solved cleanly (0/0A only; 0S/0AS excluded)
        g = sgm10(gd, "exagpu_"); (g !== nothing && g > 0) && (push!(gpu_x, nv); push!(gpu_y, pt / g))
        push!(pet_x, nv); push!(pet_y, 1.0)
    else                                                             # PEtab solved but GPU failed/suboptimal
        push!(petx_x, nv); push!(petx_y, 1.0)
    end
    _, c = pick_cpu(m, po); (c !== nothing && c > 0) && (push!(cpu_x, nv); push!(cpu_y, pt / c))  # CPU independent
end

# ticks at every order of 10 over the data range
prange(v) = isempty(v) ? (0:0) : (floor(Int, log10(minimum(v))):ceil(Int, log10(maximum(v))))
xt = [10.0^k for k in prange(vcat(gpu_x, cpu_x, pet_x, petx_x))]
yt = [10.0^k for k in prange(vcat(gpu_y, cpu_y, pet_y, petx_y))]

# common x-range (small log-padding) so the axis + trend lines span the full plot, edge to edge
_xa = vcat(gpu_x, cpu_x, pet_x, petx_x)
_xlo, _xhi = isempty(_xa) ? (1.0, 10.0) : extrema(_xa)
_pf  = (_xhi / _xlo) ^ 0.02            # ~2% log-padding each side
XLIM = (_xlo / _pf, _xhi * _pf)

# ── series 1: ExaModels + MadNLP (GPU). Global cosmetics live on this scatter() call. ──
plt = scatter(gpu_x, gpu_y;
              label  = "ExaModels + MadNLP (GPU)",  # legend text
              marker = :circle,                     # marker shape
              ms     = 8,                            # marker size
              mc     = J_PURPLE,                     # marker fill color
              msc    = :black,                       # marker outline color
              msw    = 1.5,                          # outline width
              malpha = 1.0,                          # marker opacity (0–1)
              xscale = :log10,                       # x log scale
              yscale = :log10,                       # y log scale
              xticks = xt,                           # x ticks (powers of 10)
              yticks = yt,                           # y ticks (powers of 10)
              xlims  = XLIM,                          # x-limits (trend lines span this)
              xlabel = "Number of variables",        # x axis label
              ylabel = "Speedup",                    # y axis label
              guidefontsize  = 15,                   # axis-label font size
              tickfontsize   = 11,                   # tick-number font size
              legendfontsize = 11,                   # legend font size
              legend     = :bottomright,                # legend position
              size       = (820, 420),               # figure size (px)
              grid       = true,                     # gridlines on/off
              gridalpha  = 0.2,                      # gridline opacity
              framestyle = :box,                     # axes frame style
              left_margin   = 3mm,                   # room for y-label (raise if clipped)
              bottom_margin = 3mm,                   # room for x-label (raise if clipped)
              top_margin    = 1mm,                   # room above plot
              right_margin  = 0mm)                   # room at right
# ── series 2 & 3: only per-series overrides (shape, size, colors); cosmetics inherit from above ──
scatter!(plt, cpu_x, cpu_y;
         label = "ExaModels + MadNLP (CPU)", marker = :utriangle,  # green triangles
         ms = 7, mc = J_GREEN, msc = :black, msw = 1.0, malpha = 1.0)
scatter!(plt, pet_x, pet_y;
         label = "PEtab.jl + Best optimizer (CPU)", marker = :square,    # red squares (baseline)
         ms = 6, mc = J_RED, msc = :black, msw = 1.0, malpha = 1.0)
# PEtab solved but ExaModels did NOT reach an optimum: red box stamped with a black X.
# Drawn as a red square with a black ✕ overlaid; the ✕ carries the legend entry.
scatter!(plt, petx_x, petx_y;
         label = "", marker = :square,                             # red box (unlabeled; X overlay labels it)
         ms = 6, mc = J_RED, msc = :black, msw = 1.0, malpha = 1.0)
scatter!(plt, petx_x, petx_y;
         label = "ExaModels failed or suboptimal", marker = :xcross,  # black X inside the box
         ms = 5, mc = :black, msc = :black, msw = 2.0, malpha = 1.0)

# least-squares trend line per solver series, fit in log–log space (power-law); thin dashed, no legend
function regline!(plt, x, y, color, xspan)
    length(x) < 2 && return
    lx = log10.(x); ly = log10.(y); n = length(lx)
    mx = sum(lx) / n; my = sum(ly) / n; sxx = sum((lx .- mx) .^ 2)
    sxx == 0 && return
    b = sum((lx .- mx) .* (ly .- my)) / sxx; a = my - b * mx     # ly = a + b·lx  (fit on the data)
    xs = collect(xspan)                                          # but draw across the full plot width
    plot!(plt, xs, 10.0 .^ (a .+ b .* log10.(xs)); ls = :dash, lw = 1, lc = color, label = "")
end
regline!(plt, gpu_x, gpu_y, J_PURPLE, XLIM)   # GPU trend (purple)
regline!(plt, cpu_x, cpu_y, J_GREEN, XLIM)    # CPU trend (green)

hline!(plt, [1.0]; ls = :dash, lc = :gray, lw = 1, label = "")     # dashed PEtab baseline at y=1

out = joinpath(HERE, "..", "results_plot.png")
savefig(plt, out)
println("saved: $out")
println("points plotted — GPU: $(length(gpu_x)), CPU: $(length(cpu_x)), PEtab: $(length(pet_x)), PEtab-only (X'd): $(length(petx_x))")
plt   # return the plot so `include("results_plot.jl")` displays it in the REPL / IDE plot pane
