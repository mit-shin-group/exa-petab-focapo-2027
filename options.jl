# options.jl — single source of truth for benchmark settings and model sets.
# Included by Benchmarks/{run_examodels,run_bruno,run_petab,results,results_plot}.jl.
# Pure data (no package deps): MadNLP/Optim choices are stored as Symbols and resolved by the
# consuming script. Three config sections: (1) shared, (2) ExaModels/MadNLP, (3) PEtab.jl/Optim.

# ── 1. SHARED (both backends) ────────────────────────────────────────────────────
const BENCH_TOL           = 1e-6           # convergence tol (MadNLP tol == Optim g_tol)
const BENCH_SOLVE_LIMIT   = 7200.0         # solve wall cap [s] (MadNLP max_wall_time / Optim time_limit)
const BENCH_COMPILE_LIMIT = 3600.0         # build/compile wall cap [s] (exa build & PEtab compile)
const BENCH_MAX_ITER      = 100_000_000    # max solver iterations (wall time is the real cap)
const BENCH_SGM_N         = 5              # warm reruns for the shifted-geomean solve time (0 disables)
const BENCH_SGM_SHIFT     = 10             # shift δ [s] for the shifted geometric mean
const BENCH_WARMUP_MODEL  = "Bruno_JExpBot2016"  # shared JIT-warmup model (benchmarked alone via run_bruno.jl)

# Shifted geometric mean of solve times [s]: exp(mean(log(tᵢ+δ))) − δ  (δ=0 ⇒ ordinary GM).
shifted_geomean(times, δ = BENCH_SGM_SHIFT) = round(exp(sum(t -> log(t + δ), times) / length(times)) - δ; digits = 2)

# ── 2. ExaModels / MadNLP ────────────────────────────────────────────────────────
const BENCH_K           = 4                # collocation points per mesh interval
const BENCH_ACCEPT_TOL  = 1e-4             # MadNLP acceptable_tol (ε-optimal fallback)
const BENCH_ACCEPT_ITER = 15               # iters at acceptable_tol before accepting
# LiftedKKT (condensed-space) regime — resolved against MadNLP in the run scripts.
const BENCH_KKT_SYSTEM          = :SparseCondensedKKTSystem
const BENCH_EQUALITY_TREATMENT  = :RelaxEquality
const BENCH_FIXED_VAR_TREATMENT = :RelaxBound
# Linear solver: GPU = CUDSS (MadNLPGPU); CPU = HSL via MadNLPHSL (ma27 | ma57 | ma97).
const BENCH_GPU_SOLVER  = :CUDSSSolver
const BENCH_CPU_SOLVER  = Symbol(get(ENV, "BENCH_CPU_SOLVER", "ma27"))

# ── 3. PEtab.jl / Optim ──────────────────────────────────────────────────────────
const BENCH_OPTIMIZER               = :IPNewton  # Optim optimizer (g_tol/time_limit/iterations come from §1)
const BENCH_PETAB_F_RELTOL          = 0.0        # Optim.Options f_reltol
const BENCH_PETAB_SUCCESSIVE_FTOL   = 2          # IPNewton successive_f_tol
const BENCH_PETAB_X_ABSTOL          = 0.0        # Optim.Options x_abstol
const BENCH_PETAB_ALLOW_F_INCREASES = true       # IPNewton allow_f_increases

# ── MODEL SETS ───────────────────────────────────────────────────────────────────
const MODELDIR   = joinpath(@__DIR__, "Benchmark-Models-PEtab")
const ALL_MODELS = sort(filter(m -> isdir(joinpath(MODELDIR, m)), readdir(MODELDIR)))

# Models that contain the problem feature "Possible Discontinuities" are not yet supported by
# ExaModelsPEtab and thus excluded from the benchmark set.
const EXCLUDED_MODELS = [
    "Alkan_SciSignal2018", "Beer_MolBioSystems2014", "Brannmark_JBC2010", "Chen_MSB2009",
    "Fujita_SciSignal2010", "Giordano_Nature2020", "Isensee_JCB2018", "Liu_IFACPapersOnLine2025",
    "Oliveira_NatCommun2021", "Raimundez_PCB2020", "Smith_BMCSystBiol2013", "Weber_BMC2015",
]

# Models which PEtab.jl fails to compile are excluded from the benchmark set.
const FAILED_MODELS = ["Froehlich_CellSystems2018", "Lang_PLOSComputBiol2024", "Raia_CancerResearch2011"]

# The set of benchmarked models.
const BENCHMARK_MODELS = filter(m -> m ∉ EXCLUDED_MODELS && m ∉ FAILED_MODELS, ALL_MODELS)
