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
const BENCH_INCLUDE_EXACPU = false
const BENCH_INCLUDE_PETAB  = false

# ── 1. SHARED OPTIONS (both backends) ────────────────────────────────────────────────────
const BENCH_TOL           = 1e-6           # gradient-based convergence tol
const BENCH_SOLVE_LIMIT   = 900.0          # optimizer timeout [s]
const BENCH_COMPILE_LIMIT = 900.0          # build/compile timeout [s]
const BENCH_MAX_ITER      = 100_000_000    # max solver iterations
const BENCH_SGM_N         = 5              # number of reruns for t_SGMδ
const BENCH_SGM_SHIFT     = 0.01           # shift δ [s] for the shifted geometric mean
const BENCH_WARMUP_MODEL  = "Bruno_JExpBot2016"  # warmup model

# ── 2. ExaModelsPEtab ────────────────────────────────────────────────────────
# ExaModelsPEtab picks the mesh and K itself from the model size, so there is nothing to set here.
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
# PEtab.jl, Optim and Fides are out of the project while BENCH_INCLUDE_PETAB = false, so the
# optimizer settings that named them are removed with them. Restore both together.

# Shared CPU BLAS thread budget for every CPU solver: exa-CPU Ma57 dense factorization (madnlp
# blas_num_threads), PEtab Optim's Julia BLAS, and Fides/numpy (via OMP_NUM_THREADS). 64 =
# scipy-openblas default = what prior PEtab/Fides results ran at. Set to 1 for single-thread timing.
const BENCH_CPU_THREADS = 64

# ── MODEL SETS ───────────────────────────────────────────────────────────────────
const RESULTDIR  = joinpath(@__DIR__, "benchmark_results", "benchmark_results_$(BENCH_TAG)")
const MODELDIR   = joinpath(@__DIR__, "Benchmark-Models-PEtab")
const ALL_MODELS = sort(filter(m -> isdir(joinpath(MODELDIR, m)), readdir(MODELDIR)))

# Models ExaModelsPEtab cannot transcribe are excluded from the ExaModels target set
# (PEtab.jl still attempts them):
# - SBML <event>: ExaModelsPEtab discretizes fixed-time piecewise(time) gates, not <event>
#   triggers. Liu's trigger is state-dependent (U < 1e-8); Smith's three are fixed-time
#   (time >= t_ins) but still spelled as <event>.
# - Estimated event time: the switch time is a decision variable, so the gate cannot be
#   resolved at build time. Oliveira estimates t_1/t_2 directly; Beer's condition table maps
#   tau to estimated tau_* parameters.
# - Does not build on the emc rewrite: Fiedler's six closed-form initial steady states do not
#   come back from build_function, and Froehlich does not get through create_objective.
#   Raia and Raimundez build since the form-grouped create_objective.
const EXCLUDED_MODELS = [
    "Liu_IFACPapersOnLine2025", "Smith_BMCSystBiol2013",   # SBML <event>
    "Oliveira_NatCommun2021", "Beer_MolBioSystems2014",    # estimated event time
    "Fiedler_BMCSystBiol2016", "Froehlich_CellSystems2018", # does not build on the emc rewrite
]

# The ExaModels target set. The table still lists every model in the collection.
const BENCHMARK_MODELS = filter(m -> m ∉ EXCLUDED_MODELS, ALL_MODELS)
const PETAB_MODELS     = ALL_MODELS

# ── METRIC DEFINITIONS ───────────────────────────────────────────────────────────────────
# Shifted geometric mean (SGM) of solve times [s]
t_sgmdelta(times, δ = BENCH_SGM_SHIFT) = exp(sum(t -> log(t + δ), times) / length(times)) - δ