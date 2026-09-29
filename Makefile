PYTHON ?= python3
ANALYTICS_VENV ?= analytics/.venv
ANALYTICS_PYTHON := $(ANALYTICS_VENV)/bin/python
INFERENCE_PYTHON ?= $(abspath services/inference-runtime/.venv/bin/python)
RESULTS_DIR ?= services/inference-runtime/results
OBSERVABILITY_RESULTS_DIR ?= services/inference-runtime/results-from-pod
OBSERVABILITY_DATASET_DIR ?= analytics/data/runtime-observability
MOPEP_PERF_SPLIT ?= calibration
MOPEP_PERF_DATASET ?= $(LOCAL_GOLDEN_DIR)/$(MOPEP_PERF_SPLIT).csv
MOPEP_PERF_WARMUP_DATASET ?= $(LOCAL_GOLDEN_DIR)/train.csv
MOPEP_PERF_WORKLOAD_DIR ?= /tmp/mopep-runtime-workload/$(MOPEP_PERF_SPLIT)
MOPEP_PERF_TOKENIZER ?= Qwen/Qwen2.5-7B-Instruct
MOPEP_PERF_TOKENIZER_REVISION ?=
MOPEP_PERF_THRESHOLD_MANIFEST ?=
MOPEP_PERF_SAMPLE_PER_BUCKET ?=
MOPEP_PERF_SAMPLE_SEED ?= 42
MOPEP_PERF_SAMPLE_DATASET ?=
MOPEP_PERF_REMOTE_DIR ?= /workspace/data/mopep-runtime-workload/$(MOPEP_PERF_SPLIT)
MOPEP_PERF_RESULTS_DIR ?= services/inference-runtime/results-from-pod
MOPEP_PERF_DATASET_DIR ?= analytics/data/mopep-runtime-comparison
MOPEP_PERF_GOLDEN ?= $(LOCAL_GOLDEN_DIR)/$(MOPEP_PERF_SPLIT).csv
MOPEP_PERF_CANONICAL_RESPONSES ?=
BENCH_TARGET ?= bench-all
BENCH_VARS ?=
REMOTE_HOST ?=
REMOTE_PORT ?= 22
REMOTE_DIR ?= /workspace/finance-agent-platform/services/inference-runtime
REMOTE_IDENTITY ?=
QUANT_REMOTE_PORT ?=
QUANT_SSH_OPTIONS = $(if $(strip $(QUANT_REMOTE_PORT)),-p "$(QUANT_REMOTE_PORT)") $(if $(REMOTE_IDENTITY),-i "$(REMOTE_IDENTITY)")
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
SILVER_BASE_URL ?= http://127.0.0.1:18000/v1
SILVER_MODEL ?= qwen-silver
SILVER_MODEL_REVISION ?=
SILVER_DATASET ?=
SILVER_OUTPUT_DIR ?= analytics/results/mopep-silver/qwen
SILVER_LIMIT ?=
SILVER_SEED ?= 42
SILVER_MAX_OUTPUT_TOKENS ?= 512
SILVER_TIMEOUT_SECONDS ?= 180
SILVER_MAX_ATTEMPTS ?= 3
SILVER_RETRY_DELAY_SECONDS ?= 2
QUANTIZATION_DIR ?= pipelines/qwen-quantization
LOCAL_GOLDEN_DIR ?= analytics/results/mopep-golden-shareable
REMOTE_GOLDEN_DIR ?= /workspace/data/mopep-golden
REMOTE_GOLDEN_PARENT = $(dir $(REMOTE_GOLDEN_DIR))

.DEFAULT_GOAL := help

.PHONY: help setup install test build verify test-whatsapp test-inference test-analytics check-quantization access-add \
        setup-notebook notebook observability-dataset observability-notebook openai-silver-notebook silver-qwen benchmark-dataset benchmark-tags predict-benchmark \
        benchmark-local benchmark-remote eval-tunnel eval-check eval-models eval-datasets eval-run \
        eval-results eval-consolidate push-quantization-data push-imatrix-train quantization-doctor quantization-setup \
        quantization-data-check smoke-quantize-model run-quantize-model verify-quantized-model \
        upload-quantized-model

.PHONY: mopep-performance-workload push-mopep-performance-workload mopep-performance-dataset mopep-performance-notebook

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
		'Quantização no RunPod:' \
		'  make push-quantization-data         Envia os CSVs privados ao Pod por SSH' \
		'  make push-imatrix-train             Envia somente train.csv para o build GGUF/iMatrix' \
		'  make quantization-doctor            Diagnostica GPU, disco, RAM e HF_TOKEN' \
		'  make quantization-setup             Instala o ambiente persistente no Pod' \
		'  make quantization-data-check        Valida o golden e a amostra escolhida' \
		'  make smoke-quantize-model           Testa Qwen 0.5B, GPTQ 4-bit e 8 exemplos' \
		'  make run-quantize-model             Quantiza o modelo configurado' \
		'  make verify-quantized-model         Recarrega e verifica o artefato' \
		'  make upload-quantized-model         Publica a pasta validada no HF' \
		'' \
		'Analytics e benchmarks:' \
		'  make notebook                      Abre o notebook de analytics' \
		'  make observability-dataset         Consolida os pacotes Prometheus baixados' \
		'  make observability-notebook        Analisa o dataset temporal localmente' \
		'  make mopep-performance-workload     Gera workload privado short/medium/heavy' \
		'  make push-mopep-performance-workload Envia workload privado ao Pod' \
		'  make mopep-performance-dataset      Une qualidade MOPEP e observabilidade' \
		'  make mopep-performance-notebook     Analisa qualidade e performance MOPEP' \
		'  make openai-silver-notebook        Abre o notebook da silver GPT/Batch' \
		'  make silver-qwen                   Gera a silver Qwen via API OpenAI-compatible' \
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
	@test -x "$(INFERENCE_PYTHON)" || { echo 'Ambiente do benchmark ausente; execute make -C services/inference-runtime install-benchmark prometheus-install.' >&2; exit 1; }
	cd services/inference-runtime && "$(INFERENCE_PYTHON)" -m unittest discover -s tests

check-quantization:
	$(MAKE) -C $(QUANTIZATION_DIR) check SYSTEM_PYTHON="$(PYTHON)"

push-quantization-data:
	@test -n "$(REMOTE_HOST)" || { echo 'Informe REMOTE_HOST=runpod-qwen.' >&2; exit 1; }
	@for name in train.csv calibration.csv test.csv; do \
		test -f "$(LOCAL_GOLDEN_DIR)/$$name" || { echo "Arquivo ausente: $(LOCAL_GOLDEN_DIR)/$$name" >&2; exit 1; }; \
	done
	@transfer_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$transfer_dir"' 0 1 2 3 15; \
	cp "$(LOCAL_GOLDEN_DIR)/train.csv" "$(LOCAL_GOLDEN_DIR)/calibration.csv" "$(LOCAL_GOLDEN_DIR)/test.csv" "$$transfer_dir/"; \
	(cd "$$transfer_dir" && sha256sum train.csv calibration.csv test.csv > SHA256SUMS); \
	tar -C "$$transfer_dir" -czf - train.csv calibration.csv test.csv SHA256SUMS | \
		ssh $(QUANT_SSH_OPTIONS) "$(REMOTE_HOST)" \
		"set -eu; umask 077; mkdir -p '$(REMOTE_GOLDEN_PARENT)'; incoming=\$$(mktemp -d '$(REMOTE_GOLDEN_PARENT)mopep-golden-transfer.XXXXXX'); trap 'rm -rf \"\$$incoming\"' 0 1 2 3 15; tar -xzf - -C \"\$$incoming\"; cd \"\$$incoming\"; sha256sum --check --strict SHA256SUMS; chmod 600 train.csv calibration.csv test.csv; mkdir -p '$(REMOTE_GOLDEN_DIR)'; mv train.csv calibration.csv test.csv '$(REMOTE_GOLDEN_DIR)/'; sha256sum '$(REMOTE_GOLDEN_DIR)'/*.csv; cd /; rm -f \"\$$incoming/SHA256SUMS\"; rmdir \"\$$incoming\"; trap - 0 1 2 3 15"

push-imatrix-train:
	@test -n "$(REMOTE_HOST)" || { echo 'Informe REMOTE_HOST=runpod-qwen.' >&2; exit 1; }
	@test -f "$(LOCAL_GOLDEN_DIR)/train.csv" || { echo 'Arquivo ausente: $(LOCAL_GOLDEN_DIR)/train.csv' >&2; exit 1; }
	@transfer_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$transfer_dir"' 0 1 2 3 15; \
	cp "$(LOCAL_GOLDEN_DIR)/train.csv" "$$transfer_dir/"; \
	(cd "$$transfer_dir" && sha256sum train.csv > SHA256SUMS); \
	tar -C "$$transfer_dir" -czf - train.csv SHA256SUMS | \
		ssh $(QUANT_SSH_OPTIONS) "$(REMOTE_HOST)" \
		"set -eu; umask 077; mkdir -p '$(REMOTE_GOLDEN_PARENT)'; incoming=\$$(mktemp -d '$(REMOTE_GOLDEN_PARENT)mopep-imatrix-transfer.XXXXXX'); trap 'rm -rf \"\$$incoming\"' 0 1 2 3 15; tar -xzf - -C \"\$$incoming\"; cd \"\$$incoming\"; sha256sum --check --strict SHA256SUMS; chmod 600 train.csv; mkdir -p '$(REMOTE_GOLDEN_DIR)'; mv train.csv '$(REMOTE_GOLDEN_DIR)/train.csv'; sha256sum '$(REMOTE_GOLDEN_DIR)/train.csv'; cd /; rm -f \"\$$incoming/SHA256SUMS\"; rmdir \"\$$incoming\"; trap - 0 1 2 3 15"

quantization-doctor:
	$(MAKE) -C $(QUANTIZATION_DIR) doctor

quantization-setup:
	$(MAKE) -C $(QUANTIZATION_DIR) setup

quantization-data-check:
	$(MAKE) -C $(QUANTIZATION_DIR) data-check

smoke-quantize-model:
	$(MAKE) -C $(QUANTIZATION_DIR) smoke-quantize-model

run-quantize-model:
	$(MAKE) -C $(QUANTIZATION_DIR) run-quantize-model

verify-quantized-model:
	$(MAKE) -C $(QUANTIZATION_DIR) verify-quantized-model

upload-quantized-model:
	$(MAKE) -C $(QUANTIZATION_DIR) upload-quantized-model

test-analytics:
	$(ANALYTICS_PYTHON) -m unittest discover -s analytics/tests

setup-notebook:
	$(PYTHON) -m venv $(ANALYTICS_VENV)
	$(ANALYTICS_PYTHON) -m pip install --upgrade pip
	$(ANALYTICS_PYTHON) -m pip install -r analytics/requirements.txt
	$(ANALYTICS_PYTHON) -m ipykernel install --user --name finance-agent-analytics --display-name "Python 3 (finance-agent analytics)"

benchmark-dataset:
	$(ANALYTICS_PYTHON) -m analytics.src.build_dataset --results "$(RESULTS_DIR)" --output analytics/data/benchmark_dataset.csv

observability-dataset:
	$(ANALYTICS_PYTHON) -m analytics.src.build_observability_dataset \
		--results-dir "$(OBSERVABILITY_RESULTS_DIR)" \
		--output-dir "$(OBSERVABILITY_DATASET_DIR)" --parquet

observability-notebook:
	@test -f "$(OBSERVABILITY_DATASET_DIR)/dataset-manifest.json" || { \
		echo 'Dataset ausente; execute make observability-dataset primeiro.' >&2; exit 1; \
	}
	OBSERVABILITY_DATASET_DIR="$(abspath $(OBSERVABILITY_DATASET_DIR))" \
		$(ANALYTICS_PYTHON) -m jupyter lab analytics/notebooks/analyze_runtime_observability.ipynb

mopep-performance-workload:
	@test -n "$(MOPEP_PERF_TOKENIZER_REVISION)" || { echo 'Informe MOPEP_PERF_TOKENIZER_REVISION com revisão imutável.' >&2; exit 2; }
	$(ANALYTICS_PYTHON) -m analytics.src.build_mopep_runtime_workload \
		--dataset "$(MOPEP_PERF_DATASET)" --warmup-dataset "$(MOPEP_PERF_WARMUP_DATASET)" \
		--output-dir "$(MOPEP_PERF_WORKLOAD_DIR)" --split "$(MOPEP_PERF_SPLIT)" \
		--tokenizer "$(MOPEP_PERF_TOKENIZER)" --tokenizer-revision "$(MOPEP_PERF_TOKENIZER_REVISION)" \
		$(if $(strip $(MOPEP_PERF_THRESHOLD_MANIFEST)),--threshold-manifest "$(MOPEP_PERF_THRESHOLD_MANIFEST)") \
		$(if $(strip $(MOPEP_PERF_SAMPLE_PER_BUCKET)),--sample-per-bucket "$(MOPEP_PERF_SAMPLE_PER_BUCKET)" --sample-seed "$(MOPEP_PERF_SAMPLE_SEED)") \
		$(if $(strip $(MOPEP_PERF_SAMPLE_DATASET)),--sample-dataset-out "$(MOPEP_PERF_SAMPLE_DATASET)")

push-mopep-performance-workload:
	@test -n "$(REMOTE_HOST)" || { echo 'Informe REMOTE_HOST=runpod-qwen.' >&2; exit 1; }
	@for name in workload.jsonl warmup.jsonl workload-manifest.json SHA256SUMS; do \
		test -f "$(MOPEP_PERF_WORKLOAD_DIR)/$$name" || { echo "Arquivo ausente: $(MOPEP_PERF_WORKLOAD_DIR)/$$name" >&2; exit 1; }; \
	done
	@tar -C "$(MOPEP_PERF_WORKLOAD_DIR)" -czf - workload.jsonl warmup.jsonl workload-manifest.json SHA256SUMS | \
		ssh $(QUANT_SSH_OPTIONS) "$(REMOTE_HOST)" \
		"set -eu; umask 077; incoming=\$$(mktemp -d /tmp/mopep-runtime-workload.XXXXXX); trap 'rm -rf \"\$$incoming\"' 0 1 2 3 15; tar -xzf - -C \"\$$incoming\"; cd \"\$$incoming\"; sha256sum --check --strict SHA256SUMS; mkdir -p '$(MOPEP_PERF_REMOTE_DIR)'; chmod 600 workload.jsonl warmup.jsonl workload-manifest.json SHA256SUMS; cp workload.jsonl warmup.jsonl workload-manifest.json SHA256SUMS '$(MOPEP_PERF_REMOTE_DIR)/'"

mopep-performance-dataset: observability-dataset
	$(ANALYTICS_PYTHON) -m analytics.src.build_mopep_runtime_comparison \
		--results-dir "$(MOPEP_PERF_RESULTS_DIR)" --observability-dir "$(OBSERVABILITY_DATASET_DIR)" \
		--golden "$(MOPEP_PERF_GOLDEN)" --output-dir "$(MOPEP_PERF_DATASET_DIR)" \
		$(if $(strip $(MOPEP_PERF_CANONICAL_RESPONSES)),--canonical-responses "$(MOPEP_PERF_CANONICAL_RESPONSES)")

mopep-performance-notebook:
	@test -f "$(MOPEP_PERF_DATASET_DIR)/dataset-manifest.json" || { echo 'Execute make mopep-performance-dataset primeiro.' >&2; exit 1; }
	MOPEP_PERFORMANCE_DATASET_DIR="$(abspath $(MOPEP_PERF_DATASET_DIR))" \
		$(ANALYTICS_PYTHON) -m jupyter lab analytics/notebooks/analyze_mopep_runtime_benchmark.ipynb

benchmark-tags:
	$(ANALYTICS_PYTHON) -m analytics.src.generate_tags --input analytics/data/benchmark_dataset.csv --output analytics/data/benchmark_tagged.csv

predict-benchmark:
	$(ANALYTICS_PYTHON) -m analytics.src.predict_benchmark --input analytics/data/benchmark_tagged.csv --output analytics/models/ttft.joblib

notebook:
	$(ANALYTICS_PYTHON) -m jupyter lab analytics/notebooks/analyze_inference_benchmarks.ipynb

openai-silver-notebook:
	$(ANALYTICS_PYTHON) -m jupyter lab analytics/notebooks/generate_mopep_openai_silver.ipynb

silver-qwen:
	@test -n "$(SILVER_DATASET)" || { echo 'Informe SILVER_DATASET com o caminho do CSV.'; exit 1; }
	$(ANALYTICS_PYTHON) -m analytics.src.generate_mopep_silver \
		--base-url "$(SILVER_BASE_URL)" \
		--model "$(SILVER_MODEL)" \
		--dataset "$(SILVER_DATASET)" \
		--output-root "$(SILVER_OUTPUT_DIR)" \
		--seed "$(SILVER_SEED)" \
		--max-output-tokens "$(SILVER_MAX_OUTPUT_TOKENS)" \
		--timeout-seconds "$(SILVER_TIMEOUT_SECONDS)" \
		--max-attempts "$(SILVER_MAX_ATTEMPTS)" \
		--retry-delay-seconds "$(SILVER_RETRY_DELAY_SECONDS)" $(if $(SILVER_MODEL_REVISION),--model-revision "$(SILVER_MODEL_REVISION)") $(if $(SILVER_LIMIT),--limit "$(SILVER_LIMIT)")

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
