# Options for running the benchmark scripts

# ── SOLVERS / OPTIMIZER ──────────────────────────────────────────────────────────
# These are zero-arg accessors, so including this file pulls in NO packages — each resolves
# (against the package the calling run script already `using`d) only when called. So PEtab-only
# and reporting scripts never load the GPU/MadNLP stack. To use a different optimizer/solver, write
# its constructor in the accessor; the run script that calls it must load the relevant package.

# ── EXPERIMENT TRACKING ──────────────────────────────────────────────────────────
# Each run reads/writes per-model results + a _config.toml snapshot under
# benchmark_results/benchmark_results_<tag>/; the tag also selects which run results_table.jl /
# results_plot.jl report.
const BENCH_TAG = ""   # Provide a ::String benchmark results tag for the configured run.
const RESULTDIR = joinpath(@__DIR__, "benchmark_results", "benchmark_results_$(BENCH_TAG)")

# ── 1. SHARED (both backends) ────────────────────────────────────────────────────
const BENCH_TOL           = 1e-6           # convergence tol (MadNLP tol == Optim g_tol)
const BENCH_SOLVE_LIMIT   = 3600.0         # solve wall cap [s] (MadNLP max_wall_time / Optim time_limit)
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
# LiftedKKT (condensed-space) regime — passed straight through as madnlp kwargs.
BENCH_KKT_SYSTEM()          = MadNLP.SparseCondensedKKTSystem
BENCH_EQUALITY_TREATMENT()  = MadNLP.RelaxEquality
BENCH_FIXED_VAR_TREATMENT() = MadNLP.RelaxBound
# Linear solver (madnlp `linear_solver`):
BENCH_GPU_SOLVER() = MadNLPGPU.CUDSSSolver   # GPU
BENCH_CPU_SOLVER() = MadNLPHSL.Ma27Solver    # CPU (HSL: Ma27Solver | Ma57Solver | Ma97Solver)

# ── 3. PEtab.jl / Optim ──────────────────────────────────────────────────────────
BENCH_OPTIMIZER() = Optim.IPNewton()                   # optimizer passed to PEtab.calibrate
const BENCH_PETAB_F_RELTOL          = 0.0               # Optim.Options f_reltol
const BENCH_PETAB_SUCCESSIVE_FTOL   = 2                 # IPNewton successive_f_tol
const BENCH_PETAB_X_ABSTOL          = 0.0               # Optim.Options x_abstol
const BENCH_PETAB_ALLOW_F_INCREASES = true              # IPNewton allow_f_increases

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
