#!/usr/bin/env bash
# run_emc.sh: the emc-tagged benchmark run (ExaModelsPEtab v0.2, EMC backend).
#
# Differences from run_benchmarks.sh:
# - PEtab.jl runs a per-model optimizer subset instead of every optimizer. Each original target
#   model runs only its fastest optimizer from the focapo results (results_table.txt TAG column).
#   Each newly added model runs the two Fides optimizers. Indices follow
#   BENCH_PETAB_OPTIMIZERS() = [IPNewton, GaussNewton, BFGS] = 1, 2, 3.
# - The exa stages run an explicit model order for faster result turnover: previously clean
#   models, then the newly added models, then the models that previously consumed the walltime.
# Resumable the same way run_benchmarks.sh is: terminal results are skipped on rerun.
#
#   bash benchmark_helpers/run_emc.sh          # from benchmark/
set -u
cd "$(dirname "$0")/.."
HELP=benchmark_helpers
LOG="$HELP/logs"
mkdir -p "$LOG"

THREADS=$(julia --project=. -e 'include("options.jl"); print(BENCH_CPU_THREADS)')
GPU_ID=1   # matches the focapo run (run_examodels.sh single-instance default)

retry() { local n=$1; shift; for a in $(seq 1 "$n"); do "$@" && return 0; echo "  (retry $a)"; done; }

# ── 1. snapshot ─────────────────────────────────────────────────────────────────
julia --project=. "$HELP/snapshot_options.jl"

# ── 2. PEtab.jl, per-model optimizer subset (serial, uncontended) ───────────────
# Original targets with their focapo-fastest optimizer, then every model new to the PEtab
# benchmark (both Fides), small models first. PEtab.jl attempts all 35, including the models
# the ExaModels target set excludes. Bruno (warmup model) is benchmarked by run_warmup.jl in
# stage 5.
PETAB_JOBS=(
    "Perelson_Science1996 3"
    "Bertozzi_PNAS2020 3"
    "Blasi_CellSystems2016 3"
    "Zhao_QuantBiol2020 3"
    "Okuonghae_ChaosSolitonsFractals2020 3"
    "Zheng_PNAS2012 3"
    "Sneyd_PNAS2002 3"
    "Borghans_BiophysChem1997 3"
    "Schwen_PONE2014 3"
    "Armistead_CellDeathDis2024 2"
    "Rahman_MBS2016 2"
    "Crauste_CellSystems2017 2"
    "Elowitz_Nature2000 2"
    "Laske_PLOSComputBiol2019 2"
    "Bachmann_MSB2011 2"
    "SalazarCavazos_MBoC2020 2"
    "Lucarelli_CellSystems2018 3"
    "Boehm_JProteomeRes2014 1"
    "Fiedler_BMCSystBiol2016 1"
    "Fujita_SciSignal2010 2 3"
    "Brannmark_JBC2010 2 3"
    "Weber_BMC2015 2 3"
    "Giordano_Nature2020 2 3"
    "Alkan_SciSignal2018 2 3"
    "Raimundez_PCB2020 2 3"
    "Isensee_JCB2018 2 3"
    "Chen_MSB2009 2 3"
    "Oliveira_NatCommun2021 2 3"
    "Beer_MolBioSystems2014 2 3"
    "Smith_BMCSystBiol2013 2 3"
    "Liu_IFACPapersOnLine2025 2 3"
    "Raia_CancerResearch2011 2 3"
    "Lang_PLOSComputBiol2024 2 3"
    "Froehlich_CellSystems2018 2 3"
)
echo "[1/6] PEtab.jl (per-model optimizer subset) ..."
for job in "${PETAB_JOBS[@]}"; do
    set -- $job; m=$1; shift
    for i in "$@"; do
        echo "[petab] $m opt=$i $(date)"
        retry 3 env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
            julia --project=. -t 1 "$HELP/run_petab.jl" "$m" "$i" >> "$LOG/${m}_petab.log" 2>&1
    done
done

# ── exa model order: previously clean, then new, then previous walltime consumers ──
EXA_ORDER=Crauste_CellSystems2017,Perelson_Science1996,Bertozzi_PNAS2020,Armistead_CellDeathDis2024,Boehm_JProteomeRes2014,Okuonghae_ChaosSolitonsFractals2020,Blasi_CellSystems2016,Rahman_MBS2016,Schwen_PONE2014,Sneyd_PNAS2002,Borghans_BiophysChem1997,Elowitz_Nature2000,Zhao_QuantBiol2020,Zheng_PNAS2012,Laske_PLOSComputBiol2019,SalazarCavazos_MBoC2020,Fujita_SciSignal2010,Brannmark_JBC2010,Weber_BMC2015,Giordano_Nature2020,Alkan_SciSignal2018,Raimundez_PCB2020,Isensee_JCB2018,Chen_MSB2009,Bachmann_MSB2011,Lucarelli_CellSystems2018,Fiedler_BMCSystBiol2016

# ── 3. ExaModels + MadNLP, GPU (single instance, uncontended) ───────────────────
echo "[2/6] ExaModels GPU ..."
retry 100 env BENCH_SUBSET="$EXA_ORDER" \
    julia --project=. -t 1 "$HELP/run_examodels.jl" "$GPU_ID" 1 0 > "$LOG/exa_emc_gpu.log" 2>&1

# ── 4. ExaModels + MadNLP, CPU (HSL, single instance) ───────────────────────────
echo "[3/6] ExaModels CPU ..."
retry 100 env BENCH_BACKEND=cpu BENCH_SUBSET="$EXA_ORDER" \
    OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    julia --project=. -t 1 "$HELP/run_examodels.jl" 0 1 0 > "$LOG/exa_emc_cpu.log" 2>&1

# ── 5. warmup model (Bruno), all sides ──────────────────────────────────────────
echo "[4/6] warmup ..."
retry 100 env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    julia --project=. -t 1 "$HELP/run_warmup.jl" "$GPU_ID" > "$LOG/warmup_emc_gpu.log" 2>&1
retry 100 env BENCH_BACKEND=cpu OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    julia --project=. -t 1 "$HELP/run_warmup.jl" > "$LOG/warmup_emc_cpu.log" 2>&1
retry 100 env BENCH_PETAB_ONLY=1 OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    julia --project=. -t 1 "$HELP/run_warmup.jl" > "$LOG/warmup_emc_petab.log" 2>&1

# ── 6. report ───────────────────────────────────────────────────────────────────
echo "[5/6] table + figure ..."
julia --project=. "$HELP/results_table.jl"
julia --project=. "$HELP/results_plot.jl"
echo "[6/6] DONE $(date)"
