# results_plot.jl — single scatter figure of solver speedup vs PEtab.jl over the exa target set.
#   x = nvar (NLP size), log10 — PEtab points placed at the model's exa nvar.
#   y = SGM10-time speedup vs PEtab.jl — petab_best_sgm / solver_sgm (log10). PEtab ⇒ 1.
# Three series (legend order): ExaModels MadNLP (GPU), ExaModels MadNLP (CPU), PEtab.jl + Best optimizer (CPU).
#
# All series come from the one results dir (BENCH_TAG), by key prefix: GPU exagpu_*, CPU exacpu_*,
# PEtab petab_<label>_* (fastest converged optimizer per model).
#
# Usage: julia --project=. benchmark_helpers/results_plot.jl   # writes results_plot.png at repo root.

using Plots
using Plots.PlotMeasures   # mm units for plot margins
gr()

const HERE      = @__DIR__
include(joinpath(HERE, "..", "options.jl"))   # model sets, t_sgmdelta, BENCH_*

# ─── single results dir (the configured BENCH_TAG); backends/optimizers by key prefix ─────
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

# PEtab optimizer labels present (petab_<label>_solve_status keys).
function petab_labels(d)
    labs = String[]
    for k in keys(d)
        mt = match(r"^petab_(.+)_solve_status$", k)
        mt !== nothing && push!(labs, mt.captures[1])
    end
    sort!(unique!(labs))
end

# SGM10 [s] for a backend prefix: shift from raw solve_times when present, else the stored
# aggregate. Returns nothing if there is no warm timing.
function sgm10(d, pfx)
    get(d, pfx * "sgm_status", "") == "ok" || return nothing
    raw = get(d, pfx * "solve_times", "")
    if !isempty(raw)
        ts = Float64[x for x in (tryparse(Float64, s) for s in split(raw, ",")) if x !== nothing]
        !isempty(ts) && return t_sgmdelta(ts, BENCH_SGM_SHIFT)
    end
    tryparse(Float64, get(d, pfx * "sgm_solve_time", ""))
end

function get_nvar(d)
    for k in ("exagpu_nvar", "exacpu_nvar")
        v = tryparse(Int, get(d, k, "")); v !== nothing && return v
    end
    return nothing
end

# MadNLP status code with ROG. A SUCCEEDED/ACCEPTABLE whose objective is ≥ SUBOPT_ROG worse than
# PEtab's (relative objective gap) is converged-but-suboptimal (0S/0AS) and is excluded; plot shows only 0/0A.
const SUBOPT_ROG = 0.02
# `po` is PEtab's reference optimum (the winning optimizer's objective), passed in as the ROG denominator.
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
# Fastest converged PEtab optimizer for a model: returns (prefix, sgm10, reference_objective).
function pick_petab(d)
    best = nothing  # (pfx, sgm)
    for lab in petab_labels(d)
        pfx = "petab_$(lab)_"
        get(d, pfx * "optimum_found", "") == "true" || continue
        s = sgm10(d, pfx); (s === nothing || s <= 0) && continue
        (best === nothing || s < best[2]) && (best = (pfx, s))
    end
    best === nothing && return ("", nothing, nothing)
    (best[1], best[2], tryparse(Float64, get(d, best[1] * "objective", "")))
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
    d = read_model(m)
    _, pt, po = pick_petab(d); (pt === nothing || pt <= 0) && continue  # PEtab baseline must have solved
    nv = get_nvar(d); nv === nothing && continue                        # never built ⇒ no x position
    if madnlp_code(d, "exagpu_", po) in ("0", "0A")                     # GPU solved cleanly (0/0A only; 0S/0AS excluded)
        g = sgm10(d, "exagpu_"); (g !== nothing && g > 0) && (push!(gpu_x, nv); push!(gpu_y, pt / g))
        push!(pet_x, nv); push!(pet_y, 1.0)
    else                                                                # PEtab solved but GPU failed/suboptimal
        push!(petx_x, nv); push!(petx_y, 1.0)
    end
    c = sgm10(d, "exacpu_")                                             # CPU independent (only if converged)
    (c !== nothing && c > 0 && madnlp_code(d, "exacpu_", po) in ("0", "0A")) && (push!(cpu_x, nv); push!(cpu_y, pt / c))
end

# ticks at every order of 10 over the data range
prange(v) = isempty(v) ? (0:0) : (floor(Int, log10(minimum(v))):ceil(Int, log10(maximum(v))))
xt = [10.0^k for k in prange(vcat(gpu_x, cpu_x, pet_x, petx_x))]
yt = [10.0^k for k in prange(vcat(gpu_y, cpu_y, pet_y, petx_y))]

# common x-range with small log-padding so axis + trend lines span the full plot
_xa = vcat(gpu_x, cpu_x, pet_x, petx_x)
_xlo, _xhi = isempty(_xa) ? (1.0, 10.0) : extrema(_xa)
_pf  = (_xhi / _xlo) ^ 0.02            # ~2% log-padding each side
XLIM = (_xlo / _pf, _xhi * _pf)

# common y-range with the same ~2% log-padding each side
_ya = vcat(gpu_y, cpu_y, pet_y, petx_y)
_ylo, _yhi = isempty(_ya) ? (1.0, 10.0) : extrema(_ya)
_pfy = (_yhi / _ylo) ^ 0.075
YLIM = (_ylo / _pfy, _yhi * _pfy)

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
              xlims  = XLIM,                         # x-limits (trend lines span this)
              ylims  = YLIM,                         # y-limits (±2% log-padding, like x)
              xlabel = "Number of variables",        # x axis label
              ylabel = "Speedup",                    # y axis label
              guidefontsize  = 15,                   # axis-label font size
              tickfontsize   = 11,                   # tick-number font size
              legendfontsize = 10,                    # legend font size
              legend     = (0.225, 0.925),             # upper-left quarter, fully inside the axes
              size       = (820, 420),               # figure size (px)
              grid       = true,                     # gridlines on/off
              gridalpha  = 0.2,                      # gridline opacity
              framestyle = :box,                     # axes frame style
              left_margin   = 3mm,                   # room for y-label
              bottom_margin = 3mm,                   # room for x-label
              top_margin    = 1mm,                   # room above plot
              right_margin  = 0mm)                   # room at right
# ── series 2 & 3: only per-series overrides (shape, size, colors); cosmetics inherit from above ──
scatter!(plt, cpu_x, cpu_y;
         label = "ExaModels + MadNLP (CPU)", marker = :utriangle,  # green triangles
         ms = 7, mc = J_GREEN, msc = :black, msw = 1.0, malpha = 1.0)
scatter!(plt, pet_x, pet_y;
         label = "PEtab.jl + Best optimizer (CPU)", marker = :square,    # red squares (baseline)
         ms = 6, mc = J_RED, msc = :black, msw = 1.0, malpha = 1.0)
# PEtab solved but ExaModels did not reach an optimum: red square with a black ✕ overlaid;
# the ✕ carries the legend entry.
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
plt   # return the plot so include displays it in the REPL / IDE plot pane
