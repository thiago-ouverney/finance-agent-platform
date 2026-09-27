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
EVAL_SSH_HOST ?=
EVAL_LOCAL_PORT ?= 18000
EVAL_REMOTE_PORT ?= 11434
EVAL_OLLAMA_HOST ?= http://127.0.0.1:$(EVAL_LOCAL_PORT)
EVAL_DATASETS_DIR ?= /tmp/mopep-bmc-tags
EVAL_RESULTS_DIR ?= analytics/results/mopep-tags
EVAL_CONSOLIDATED_DIR ?= analytics/data
EVAL_MODEL ?=
EVAL_DATASET ?=
EVAL_LIMIT ?=
EVAL_SEED ?= 42

.DEFAULT_GOAL := help

.PHONY: help setup install test build verify test-whatsapp test-inference test-analytics check-quantization access-add \
        setup-notebook notebook benchmark-dataset benchmark-tags predict-benchmark \
        benchmark-local benchmark-remote eval-tunnel eval-check eval-models eval-datasets eval-run \
        eval-results eval-consolidate

help:
	@printf '%s\n' \
		'Finance Agent Platform — comandos disponíveis' \
		'' \
		'Preparação:' \
		'  make setup                         Instala o adapter e prepara o ambiente analítico' \
		'  make install                       Instala as dependências do adapter WhatsApp' \
		'  make setup-notebook                Cria o ambiente Python/Jupyter de analytics' \
		'  make access-add PHONE=5521...      Autoriza um telefone no adapter' \
		'' \
		'Validação:' \
		'  make test                          Executa todos os testes e checks' \
		'  make build                         Compila o adapter WhatsApp' \
		'  make verify                        Executa testes e build' \
		'  make test-whatsapp                 Executa os testes do adapter' \
		'  make test-inference                Executa os testes do runtime de inferência' \
		'  make test-analytics                Executa os testes de analytics' \
		'  make check-quantization            Valida os scripts de quantização' \
		'' \
		'Analytics e benchmarks:' \
		'  make notebook                      Abre o notebook de analytics' \
		'  make benchmark-dataset             Consolida resultados dos benchmarks' \
		'  make benchmark-tags                Gera tags para o dataset' \
		'  make predict-benchmark             Treina o preditor de benchmark' \
		'  make benchmark-local               Executa um benchmark local' \
		'  make benchmark-remote              Executa um benchmark remoto via SSH' \
		'' \
		'Avaliação de tags com Ollama:' \
		'  make eval-tunnel                   Abre o túnel SSH para um Ollama remoto' \
		'  make eval-check                    Verifica a conexão com o Ollama' \
		'  make eval-models                   Lista os modelos disponíveis' \
		'  make eval-datasets                 Lista os datasets disponíveis' \
		'  make eval-run                      Avalia um modelo em um dataset' \
		'  make eval-results                  Resume as avaliações realizadas' \
		'  make eval-consolidate              Consolida runs e predições em CSV'

setup: install setup-notebook

install:
	npm --prefix apps/whatsapp-adapter install

access-add:
	@test -n "$(PHONE)" || { echo 'Informe PHONE com país e DDD, somente dígitos.'; exit 1; }
	cd apps/whatsapp-adapter && ./node_modules/.bin/tsx src/scripts/access-add.ts --phone "$(PHONE)"

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
	$(ANALYTICS_PYTHON) -m unittest discover -s analytics/tests

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

eval-tunnel:
	@test -n "$(EVAL_SSH_HOST)" || { echo 'Informe EVAL_SSH_HOST com um alias SSH da UFF ou RunPod.'; exit 1; }
	ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
		-L 127.0.0.1:$(EVAL_LOCAL_PORT):127.0.0.1:$(EVAL_REMOTE_PORT) "$(EVAL_SSH_HOST)"

eval-check:
	$(ANALYTICS_PYTHON) -m analytics.src.evaluate_mopep_tags \
		--ollama-host "$(EVAL_OLLAMA_HOST)" check

eval-models:
	$(ANALYTICS_PYTHON) -m analytics.src.evaluate_mopep_tags \
		--ollama-host "$(EVAL_OLLAMA_HOST)" models

eval-datasets:
	$(ANALYTICS_PYTHON) -m analytics.src.evaluate_mopep_tags \
		--datasets-dir "$(EVAL_DATASETS_DIR)" datasets

eval-run:
	@test -n "$(EVAL_MODEL)" || { echo 'Informe EVAL_MODEL com o nome exibido por make eval-models.'; exit 1; }
	@test -n "$(EVAL_DATASET)" || { echo 'Informe EVAL_DATASET com um ID ou caminho de CSV.'; exit 1; }
	$(ANALYTICS_PYTHON) -m analytics.src.evaluate_mopep_tags \
		--ollama-host "$(EVAL_OLLAMA_HOST)" \
		--datasets-dir "$(EVAL_DATASETS_DIR)" \
		--output-root "$(EVAL_RESULTS_DIR)" \
		run --model "$(EVAL_MODEL)" --dataset "$(EVAL_DATASET)" \
		--seed "$(EVAL_SEED)" $(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)")

eval-results:
	$(ANALYTICS_PYTHON) -m analytics.src.consolidate_mopep_evals \
		--results-root "$(EVAL_RESULTS_DIR)" results

eval-consolidate:
	$(ANALYTICS_PYTHON) -m analytics.src.consolidate_mopep_evals \
		--results-root "$(EVAL_RESULTS_DIR)" \
		--output-dir "$(EVAL_CONSOLIDATED_DIR)" consolidate
