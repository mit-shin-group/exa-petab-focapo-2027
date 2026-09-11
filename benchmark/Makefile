# Benchmark for "Revisiting Simultaneous Methods for Dynamic Optimization in the GPU Era"
#
#   make                 instantiate the environment and run the whole benchmark
#   make hsl HSL=<path>  install a licensed HSL_jll.jl into the environment
#   make report          regenerate results_table.txt and results_plot.png from results/
#   make clean           remove logs/
#   make clean-results   remove results/, results_table.txt and results_plot.png

JULIA ?= julia

.PHONY: all benchmark instantiate hsl report clean clean-results

all: benchmark

benchmark: instantiate
	bash run_benchmarks.sh

instantiate:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'

hsl:
	@[ -n "$(HSL)" ] || { echo "Usage: make hsl HSL=/full/path/to/HSL_jll.jl"; exit 1; }
	$(JULIA) --project=. -e 'using Pkg; Pkg.develop(path="$(HSL)")'

report:
	$(JULIA) --project=. src/results_table.jl
	$(JULIA) --project=. src/results_plot.jl

clean:
	rm -rf logs

clean-results:
	rm -rf results results_table.txt results_plot.png
