# CHANGE THE OPTIONS FOR RUNNING THE BENCHMARK SCRIPTS HERE

# ── TAG THE BENCHMARK RUN ────────────────────────────────────────────────────
# Label the 'run_benchmarks.sh' run with a <tag>::String
# Each tagged run creates a folder /benchmark_results/benchmark_results_<tag>
# which contains a snapshot of the options used in _config.toml as well as
# <model>_results.txt in for each model benchmark_results_<tag>/
const BENCH_TAG = "emc"

# ── WHICH BACKENDS TO INCLUDE IN THE RUN ────────────────────────────────────────────────────
# Choose which backend(s) to benchmark in this tagged run. 
# If more than one, it runs in series: ExaGPU -> ExaCPU -> PEtab
const BENCH_INCLUDE_EXAGPU = true
const BENCH_INCLUDE_EXACPU = true
const BENCH_INCLUDE_PETAB  = true

# ── 1. SHARED OPTIONS (both backends) ────────────────────────────────────────────────────
const BENCH_TOL           = 1e-6           # gradient-based convergence tol
const BENCH_SOLVE_LIMIT   = 3600.0         # optimizer timeout [s]
const BENCH_COMPILE_LIMIT = 3600.0         # build/compile timeout [s]
const BENCH_MAX_ITER      = 100_000_000    # max solver iterations
const BENCH_SGM_N         = 5              # number of reruns for t_SGMδ
const BENCH_SGM_SHIFT     = 0.01           # shift δ [s] for the shifted geometric mean
const BENCH_WARMUP_MODEL  = "Bruno_JExpBot2016"  # warmup model

# ── 2. ExaModelsPEtab ────────────────────────────────────────────────────────
# ExaModelsPEtab options
const BENCH_K = 4 # number of interpolation points points per mesh interval
# MadNLP options
const BENCH_ACCEPT_TOL  = 1e-4             # MadNLP acceptable_tol
const BENCH_ACCEPT_ITER = 15               # iters at acceptable_tol before accepting
BENCH_KKT_SYSTEM()          = MadNLP.SparseCondensedKKTSystem # LiftedKKT
BENCH_EQUALITY_TREATMENT()  = MadNLP.RelaxEquality # LiftedKKT
BENCH_FIXED_VAR_TREATMENT() = MadNLP.RelaxBound # LiftedKKT
# MadNLP linear solver for GPU/CPU
BENCH_GPU_SOLVER() = MadNLPGPU.CUDSSSolver # GPU
BENCH_CPU_SOLVER() = MadNLPHSL.Ma57Solver # CPU

# ── 3. PEtab.jl ─────────────────────────────────────────────────────
# PEtab.jl recommends the following optimizers:
#   Small models        : Optim.IPNewton() with BENCH_PETAB_HESSIAN = :ForwardDiffv(hessian_method = ... in PEtabODEProblem)
#   Medium-sized models : Fides.CustomHessian() with BENCH_PETAB_HESSIAN = :GaussNewton (hessian_method = ... in PEtabODEProblem)
#   Large models        : Fides.BFGS() with BENCH_PETAB_HESSIAN = nothing 
#                         (i.e., hessian_method not specified, calibrate(petab_prob, x0, Fides.BFGS()))
# Can either put a single entry or vector with an equal-length vector pair for PETAB_HESSIANS
BENCH_PETAB_OPTIMIZERS() = [Optim.IPNewton(), Fides.CustomHessian(), Fides.BFGS()]
const BENCH_PETAB_HESSIANS = [:ForwardDiff, :GaussNewton, nothing]

# Shared CPU BLAS thread budget for every CPU solver: exa-CPU Ma57 dense factorization (madnlp
# blas_num_threads), PEtab Optim's Julia BLAS, and Fides/numpy (via OMP_NUM_THREADS). 64 =
# scipy-openblas default = what prior PEtab/Fides results ran at. Set to 1 for single-thread timing.
const BENCH_CPU_THREADS = 64

# ── MODEL SETS ───────────────────────────────────────────────────────────────────
const RESULTDIR  = joinpath(@__DIR__, "benchmark_results", "benchmark_results_$(BENCH_TAG)")
const MODELDIR   = joinpath(@__DIR__, "Benchmark-Models-PEtab")
const ALL_MODELS = sort(filter(m -> isdir(joinpath(MODELDIR, m)), readdir(MODELDIR)))

# Models carrying an SBML <event> are excluded from the benchmark set: ExaModelsPEtab discretizes
# fixed-time piecewise(time) gates, not <event> triggers. Liu's trigger is state-dependent
# (U < 1e-8); Smith's three are fixed-time (time >= t_ins) but still spelled as <event>.
const EXCLUDED_MODELS = [
    "Liu_IFACPapersOnLine2025", "Smith_BMCSystBiol2013",
]

# Models which PEtab.jl fails to compile are excluded from the benchmark set.
const FAILED_MODELS = ["Froehlich_CellSystems2018", "Lang_PLOSComputBiol2024", "Raia_CancerResearch2011"]

# The set of benchmarked models.
const BENCHMARK_MODELS = filter(m -> m ∉ EXCLUDED_MODELS && m ∉ FAILED_MODELS, ALL_MODELS)

# ── METRIC DEFINITIONS ───────────────────────────────────────────────────────────────────
# Shifted geometric mean (SGM) of solve times [s]
t_sgmdelta(times, δ = BENCH_SGM_SHIFT) = exp(sum(t -> log(t + δ), times) / length(times)) - δ