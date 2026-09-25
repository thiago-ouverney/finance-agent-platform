PYTHON ?= python3
ANALYTICS_VENV ?= analytics/.venv
ANALYTICS_PYTHON := $(ANALYTICS_VENV)/bin/python
RESULTS_DIR ?= services/inference-runtime/results
BENCH_TARGET ?= bench-all
BENCH_VARS ?=
REMOTE_HOST ?=
REMOTE_PORT ?= 22
REMOTE_DIR ?= /workspace/finance-agent-platform/services/inference-runtime
REMOTE_IDENTITY ?=

.PHONY: setup install test build verify test-whatsapp test-inference test-analytics check-quantization \
        setup-notebook notebook benchmark-dataset benchmark-tags predict-benchmark \
        benchmark-local benchmark-remote

setup: install setup-notebook

install:
	npm --prefix apps/whatsapp-adapter install

test: test-whatsapp test-inference test-analytics check-quantization

build:
	npm --prefix apps/whatsapp-adapter run build

verify: test build

test-whatsapp:
	npm --prefix apps/whatsapp-adapter test

test-inference:
	cd services/inference-runtime && python -m unittest discover -s tests

check-quantization:
	$(PYTHON) -m py_compile pipelines/qwen-quantization/quantize.py pipelines/qwen-quantization/benchmark.py pipelines/qwen-quantization/perplexity.py

test-analytics:
	$(PYTHON) -m unittest discover -s analytics/tests

setup-notebook:
	$(PYTHON) -m venv $(ANALYTICS_VENV)
	$(ANALYTICS_PYTHON) -m pip install --upgrade pip
	$(ANALYTICS_PYTHON) -m pip install -r analytics/requirements.txt
	$(ANALYTICS_PYTHON) -m ipykernel install --user --name finance-agent-analytics --display-name "Python 3 (finance-agent analytics)"

benchmark-dataset:
	$(ANALYTICS_PYTHON) -m analytics.src.build_dataset --results "$(RESULTS_DIR)" --output analytics/data/benchmark_dataset.csv

benchmark-tags:
	$(ANALYTICS_PYTHON) -m analytics.src.generate_tags --input analytics/data/benchmark_dataset.csv --output analytics/data/benchmark_tagged.csv

predict-benchmark:
	$(ANALYTICS_PYTHON) -m analytics.src.predict_benchmark --input analytics/data/benchmark_tagged.csv --output analytics/models/ttft.joblib

notebook:
	$(ANALYTICS_PYTHON) -m jupyter lab analytics/notebooks/benchmark-analysis.ipynb

benchmark-local:
	$(PYTHON) scripts/run_benchmark.py --mode local --target "$(BENCH_TARGET)" $(foreach item,$(BENCH_VARS),--set "$(item)")

benchmark-remote:
	@test -n "$(REMOTE_HOST)" || { echo 'Informe REMOTE_HOST com um alias SSH da UFF ou RunPod.'; exit 1; }
	$(PYTHON) scripts/run_benchmark.py --mode ssh --host "$(REMOTE_HOST)" --port "$(REMOTE_PORT)" --remote-dir "$(REMOTE_DIR)" --target "$(BENCH_TARGET)" $(if $(REMOTE_IDENTITY),--identity "$(REMOTE_IDENTITY)") $(foreach item,$(BENCH_VARS),--set "$(item)")
