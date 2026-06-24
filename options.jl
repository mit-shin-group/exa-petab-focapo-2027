# Options for running the benchmark scripts

# ── SOLVERS / OPTIMIZER ──────────────────────────────────────────────────────────
# These are zero-arg accessors, so including this file pulls in NO packages — each resolves
# (against the package the calling run script already `using`d) only when called. So PEtab-only
# and reporting scripts never load the GPU/MadNLP stack. To use a different optimizer/solver, write
# its constructor in the accessor; the run script that calls it must load the relevant package.

# ── EXPERIMENT TRACKING ──────────────────────────────────────────────────────────
# Each run reads/writes per-model results + a _config.toml snapshot under
# benchmark_results/benchmark_results_<tag>/; the tag also selects which run results_table.jl /
# results_plot.jl report. BENCH_TAG is just a label for THIS run — configure the run by setting the
# flags/settings below, then give it a tag. Keep ONE backend included per run so its solve timing is
# uncontended; the assembled paper reference (tag "focapo") is built by results_table.jl from the
# separate runs. Typical configurations and the tag we file them under:
#   GPU         : INCLUDE_EXA_GPU=true                                          (CUDSS)
#   CPUma27     : INCLUDE_EXA_CPU=true,  BENCH_CPU_SOLVER = Ma27Solver          (HSL ma27)
#   CPUma57/97  : INCLUDE_EXA_CPU=true,  BENCH_CPU_SOLVER = Ma57Solver/Ma97Solver
#   IPNewton    : INCLUDE_PETAB=true,    OPTIMIZER = Optim.IPNewton(),      HESSIAN = :ForwardDiff
#   GaussNewton : INCLUDE_PETAB=true,    OPTIMIZER = Fides.CustomHessian(), HESSIAN = :GaussNewton
#   BFGS        : INCLUDE_PETAB=true,    OPTIMIZER = Fides.BFGS(),          HESSIAN = :GaussNewton
const BENCH_TAG = "GPU"   # ::String label for this run's results dir.

const RESULTDIR = joinpath(@__DIR__, "benchmark_results", "benchmark_results_$(BENCH_TAG)")

# Which backend(s) this run benchmarks (run_benchmarks.sh gates each stage on these). Keep one true.
const BENCH_INCLUDE_EXA_GPU = true
const BENCH_INCLUDE_EXA_CPU = false
const BENCH_INCLUDE_PETAB   = false

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

# ── 3. PEtab.jl / Optim+Fides ─────────────────────────────────────────────────────
# The optimizer object passed to PEtab.calibrate — set it directly (the run script reads its TYPE to
# choose FidesOptions vs Optim.Options, so no extra flag). One of PEtab.jl's recommended algorithms:
#   Optim.IPNewton()         (small models)   — pair with BENCH_PETAB_HESSIAN = :ForwardDiff
#   Fides.CustomHessian()    (medium models)  — pair with BENCH_PETAB_HESSIAN = :GaussNewton
#   Fides.BFGS()             (large models)   — pair with BENCH_PETAB_HESSIAN = :GaussNewton
# Default optimizer settings except the gradient-convergence tol (relaxed to BENCH_TOL) and the
# wall/iteration caps. Lazy accessor — the run script `using`s both Optim and Fides.
BENCH_PETAB_OPTIMIZER() = Optim.IPNewton()
# PEtabODEProblem Hessian method (:ForwardDiff full Hessian, or :GaussNewton approximation). BFGS
# self-approximates the Hessian during the solve, but PEtab's Fides extension omits the cache-priming
# `nllh` call it does for CustomHessian, so BFGS's `nllh_grad` objective hits an unpopulated
# `odesols_derivatives` cache (KeyError). Setting :GaussNewton and priming `hess!` once at compile
# time (run_petab.jl) populates that cache and fixes it — a one-time cost; the solve stays Hessian-free.
const BENCH_PETAB_HESSIAN = :ForwardDiff

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
