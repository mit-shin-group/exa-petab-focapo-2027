# Makefile — one-command reproduction of the benchmark in
# "Revisiting Simultaneous Methods for Dynamic Optimization in the GPU Era"
# by Joseph W. Choi and Sungho Shin.
#
#   make               instantiate the pinned Julia env, run the full benchmark
#                      (resumable), then regenerate results_table.txt + results_plot.png
#   make instantiate   only instantiate the pinned Manifest.toml (no solves)
#   make report        regenerate the table + figure from existing results (no solves)
#   make clean         remove run logs
#   make veryclean     also remove the generated CondaPkg (Fides) env
#
# Configure the backends, solver options, and run tag (BENCH_TAG) in options.jl.
# HSL is a required, user-supplied dependency (see ../README.md): the CPU linear
# solver is MadNLPHSL.Ma57Solver. Set CPU_INST=N for N parallel exa-CPU instances.

JULIA    ?= julia
CPU_INST ?= 1
HELP      = benchmark_helpers
TAG      := $(shell grep -E '^const BENCH_TAG' options.jl | sed -E 's/.*"([^"]*)".*/\1/')

.PHONY: all
all: benchmark

# Full run: instantiate the pinned env, then execute every enabled stage in series.
# run_benchmarks.sh is resumable (terminal models are skipped) and regenerates the
# table + figure at the end, so it is the single entry point for reproduction.
.PHONY: benchmark
benchmark: instantiate
	CPU_INST=$(CPU_INST) bash run_benchmarks.sh

.PHONY: instantiate
instantiate:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'

# Install YOUR licensed HSL_jll, overriding the dummy stub that instantiate pulls in.
# Required for the exa-CPU stage (see ../README.md).
#   make hsl HSL=/full/path/to/HSL_jll.jl
.PHONY: hsl
hsl:
	@[ -n "$(HSL)" ] || { echo "Usage: make hsl HSL=/full/path/to/HSL_jll.jl"; exit 1; }
	$(JULIA) --project=. -e 'using Pkg; Pkg.develop(path="$(HSL)")'

# Rebuild only the report artifacts from whatever results already exist for BENCH_TAG.
.PHONY: report
report:
	$(JULIA) --project=. $(HELP)/results_table.jl
	$(JULIA) --project=. $(HELP)/results_plot.jl
	@echo "Wrote results_table.txt , results_plot.png  (tag=$(TAG))"

.PHONY: clean
clean:
	@echo "Removing run logs..."
	rm -rf $(HELP)/logs

.PHONY: veryclean
veryclean: clean
	@echo "Removing generated CondaPkg (Fides) env..."
	rm -rf .CondaPkg
