# Speedup of ExaModels + MadNLP over PEtab.jl against nvar, from results/*.txt

using Plots
using Plots.PlotMeasures
gr()
include(joinpath(@__DIR__, "common.jl"))

const OUT = joinpath(@__DIR__, "..", "results_plot.png")

g(d, k) = get(d, k, "")
num(s) = tryparse(Float64, s)
sgm(d, pfx) = g(d, pfx * "rerun_status") == "ok" ? t_sgmdelta(parse_times(g(d, pfx * "rerun_times"))) : nothing
function suboptimal(d, pfx)
    a, b = num(g(d, pfx * "objective_eval")), num(g(d, "petab_objective_eval"))
    return a !== nothing && b !== nothing && isfinite(a) && isfinite(b) && b != 0 && (a - b) / abs(b) >= SUBOPT_ROG
end

# Points: (nvar, speedup) per backend, split into clean and suboptimal
points = Dict((pfx, kind) => (Float64[], Float64[]) for pfx in ("gpu_", "cpu_"), kind in (:clean, :subopt))
petab = (Float64[], Float64[])
failed = (Float64[], Float64[])
for m in ALL_MODELS
    d = read_result(m)
    tp = sgm(d, "petab_")
    tp === nothing && continue
    nvar = num(g(d, "gpu_nvar")) !== nothing ? num(g(d, "gpu_nvar")) : num(g(d, "cpu_nvar"))
    nvar === nothing && continue
    solved = false
    for pfx in ("gpu_", "cpu_")
        t = sgm(d, pfx)
        t === nothing && continue
        solved = true
        x, y = points[(pfx, suboptimal(d, pfx) ? :subopt : :clean)]
        push!(x, nvar); push!(y, tp / t)
    end
    push!(solved ? petab[1] : failed[1], nvar)
    push!(solved ? petab[2] : failed[2], 1.0)
end

const J_PURPLE = RGB(0.584, 0.345, 0.698)
const J_GREEN  = RGB(0.220, 0.596, 0.149)
const J_RED    = RGB(0.796, 0.235, 0.200)

xs = vcat(petab[1], failed[1], (points[k][1] for k in keys(points))...)
ys = vcat(petab[2], failed[2], (points[k][2] for k in keys(points))...)
function lims(v, pad)
    lo, hi = isempty(v) ? (1.0, 10.0) : extrema(v)
    return (lo / (hi / lo)^pad, hi * (hi / lo)^pad)
end
ticks(v) = isempty(v) ? [1.0] : [10.0^k for k in floor(Int, log10(minimum(v))):ceil(Int, log10(maximum(v)))]
XLIM, YLIM = lims(xs, 0.02), lims(ys, 0.075)

plt = scatter(points[("gpu_", :clean)]...; label = "ExaModels + MadNLP (GPU)", marker = :circle, ms = 8,
    mc = J_PURPLE, msc = :black, msw = 1.5, xscale = :log10, yscale = :log10, xticks = ticks(xs), yticks = ticks(ys),
    xlims = XLIM, ylims = YLIM, xlabel = "Number of variables (nvar)", ylabel = "Speedup",
    guidefontsize = 15, tickfontsize = 11, legendfontsize = 9, legend = (0.20, 0.925), size = (820, 420),
    grid = true, gridalpha = 0.2, framestyle = :box, left_margin = 3mm, bottom_margin = 3mm, top_margin = 1mm, right_margin = 0mm)
scatter!(plt, points[("gpu_", :subopt)]...; label = "", marker = :circle, ms = 8, mc = J_PURPLE, msc = :black, msw = 1.5)
scatter!(plt, points[("cpu_", :clean)]...; label = "ExaModels + MadNLP (CPU)", marker = :utriangle, ms = 7, mc = J_GREEN, msc = :black, msw = 1.0)
scatter!(plt, points[("cpu_", :subopt)]...; label = "", marker = :utriangle, ms = 7, mc = J_GREEN, msc = :black, msw = 1.0)
scatter!(plt, petab...; label = "PEtab.jl (CPU)", marker = :square, ms = 6, mc = J_RED, msc = :black, msw = 1.0)
scatter!(plt, failed...; label = "", marker = :square, ms = 6, mc = J_RED, msc = :black, msw = 1.0)
isempty(failed[1]) || scatter!(plt, failed...; label = "ExaModels failed to solve", marker = :xcross, ms = 5, mc = :black, msc = :black, msw = 2.0)
subopt = (vcat(points[("gpu_", :subopt)][1], points[("cpu_", :subopt)][1]), vcat(points[("gpu_", :subopt)][2], points[("cpu_", :subopt)][2]))
isempty(subopt[1]) || scatter!(plt, subopt...; label = "ExaModels suboptimal solve", marker = :cross, ms = 5, mc = :black, msc = :black, msw = 2.0)

# Least-squares trend line in log-log space
function trend!(plt, x, y, color)
    length(x) < 2 && return
    lx, ly = log10.(x), log10.(y)
    mx, my = sum(lx) / length(lx), sum(ly) / length(ly)
    sxx = sum((lx .- mx) .^ 2)
    sxx == 0 && return
    b = sum((lx .- mx) .* (ly .- my)) / sxx
    a = my - b * mx
    xs = collect(XLIM)
    plot!(plt, xs, 10.0 .^ (a .+ b .* log10.(xs)); ls = :dash, lw = 1, lc = color, label = "")
end
trend!(plt, points[("gpu_", :clean)]..., J_PURPLE)
trend!(plt, points[("cpu_", :clean)]..., J_GREEN)
hline!(plt, [1.0]; ls = :dash, lc = :gray, lw = 1, label = "")

savefig(plt, OUT)
println("saved: $OUT")
