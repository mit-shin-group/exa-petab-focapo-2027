# Benchmark options

# Solver
const TOL         = 1e-6
const ACCEPT_TOL  = 1e-4
const ACCEPT_ITER = 15
const MAX_ITER    = 100_000_000
const CPU_THREADS = 8

# Limits [s]
const BUILD_LIMIT = 3600.0
const SOLVE_LIMIT = 3600.0

# Reruns for the shifted geometric mean
const N_RERUNS  = 5
const SGM_SHIFT = 0.01

# Relative objective gap above which a converged solve is suboptimal
const SUBOPT_ROG = 0.02

# Warmup: WARMUP_MODEL warms every other model and is itself warmed by WARMUP_WARMUP_MODEL
const WARMUP_MODEL        = "Bertozzi_PNAS2020"
const WARMUP_WARMUP_MODEL = "Blasi_CellSystems2016"

# MadNLP: LiftedKKT and linear solvers
KKT_OPTS() = (
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)
GPU_SOLVER() = MadNLPGPU.CUDSSSolver
CPU_SOLVER() = MadNLPHSL.Ma57Solver

# Models
const MODELDIR   = joinpath(@__DIR__, "Benchmark-Models-PEtab")
const RESULTDIR  = joinpath(@__DIR__, "results")
const ALL_MODELS = sort(filter(m -> isdir(joinpath(MODELDIR, m)), readdir(MODELDIR)))

# Flags: columns to run and models to skip
const RUN_BACKENDS = ["gpu", "cpu", "petab"]
const SKIP_MODELS  = String[]
const MODELS       = filter(m -> m ∉ SKIP_MODELS, ALL_MODELS)
