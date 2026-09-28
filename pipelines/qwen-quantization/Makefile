SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.NOTPARALLEL:

QUANT_ENV_FILE ?=
ifneq ($(strip $(QUANT_ENV_FILE)),)
include $(QUANT_ENV_FILE)
endif

WORKSPACE ?= /workspace
SYSTEM_PYTHON ?= python3
MAKE_BIN ?= make
QUANT_VENV ?= $(WORKSPACE)/venvs/qwen-quantization
QUANT_PYTHON ?= $(QUANT_VENV)/bin/python
HF ?= $(QUANT_VENV)/bin/hf
HF_HOME ?= $(WORKSPACE)/.cache/huggingface
PIP_CACHE_DIR ?= $(WORKSPACE)/.cache/pip
TMPDIR ?= $(WORKSPACE)/tmp
CUDA_VISIBLE_DEVICES ?= 0
CUDA_MIN_DRIVER_BRANCH ?= 580
QUANT_LOCK_FILE ?= $(WORKSPACE)/locks/qwen-quantization.lock

QUANT_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
QUANT_MODEL_REVISION ?= main
QUANT_MODEL_NAME = $(notdir $(QUANT_MODEL_ID))
QUANT_MODEL_DIR ?= $(WORKSPACE)/models/$(QUANT_MODEL_NAME)-original
QUANT_METHOD ?= gptq
QUANT_METHOD_LABEL = $(shell printf '%s' '$(QUANT_METHOD)' | tr '[:lower:]' '[:upper:]')
QUANT_BITS ?= 4
QUANT_GROUP_SIZE ?= 128
QUANT_BATCH_SIZE ?= 1
QUANT_DATASET ?= $(WORKSPACE)/data/mopep-golden/train.csv
QUANT_SAMPLES ?= 256
QUANT_SEED ?= 42
QUANT_OUTPUT_DIR ?= $(WORKSPACE)/models/$(QUANT_MODEL_NAME)-$(QUANT_METHOD_LABEL)-$(QUANT_BITS)bit
QUANT_SMOKE_OUTPUT_DIR ?= $(WORKSPACE)/models/Qwen2.5-0.5B-Instruct-GPTQ-4bit-smoke
QUANT_VERIFY_RELOAD ?= 1
QUANT_HF_REPO_ID ?= thiagoouverney/$(QUANT_MODEL_NAME)-$(QUANT_METHOD_LABEL)-$(QUANT_BITS)bit
QUANT_HF_PRIVATE ?= 1
QUANT_HF_ALLOW_PUBLIC ?= 0
QUANT_HF_REPLACE_EXISTING ?= 0

GGUF_VENV ?= $(WORKSPACE)/venvs/qwen-gguf-imatrix
GGUF_PYTHON ?= $(GGUF_VENV)/bin/python
GGUF_LLAMA_CPP_DIR ?= $(WORKSPACE)/llama.cpp-imatrix
GGUF_LLAMA_CPP_REVISION ?= ed7ac35e1ee49cb70e4dfa9f0a2ce39b0a5ec4ea
GGUF_LLAMA_CPP_BUILD_DIR ?= $(GGUF_LLAMA_CPP_DIR)/build
GGUF_CONVERT_REQUIREMENTS ?= $(GGUF_LLAMA_CPP_DIR)/requirements/requirements-convert_hf_to_gguf.txt
GGUF_MODEL_PREFIX ?= Qwen2.5-7B-Instruct
GGUF_OUTPUT_DIR ?= $(WORKSPACE)/models/$(GGUF_MODEL_PREFIX)-GGUF-comparison
GGUF_CORPUS_DIR ?= $(WORKSPACE)/data/quantization/mopep-imatrix
GGUF_CORPUS_FILE ?= $(GGUF_CORPUS_DIR)/train-chat-template.txt
GGUF_CORPUS_MANIFEST ?= $(GGUF_CORPUS_DIR)/train-chat-template.manifest.json
GGUF_SAMPLES ?= 256
GGUF_SEED ?= 42
GGUF_THREADS ?= $(shell nproc 2>/dev/null || echo 1)
GGUF_GPU_LAYERS ?= 99
GGUF_CONTEXT ?= 512
GGUF_BATCH ?= 512
GGUF_UBATCH ?= 512
GGUF_LOCK_FILE ?= $(QUANT_LOCK_FILE)
GGUF_SMOKE_MODEL_DIR ?= $(WORKSPACE)/models/Qwen2.5-0.5B-Instruct-original
GGUF_SMOKE_OUTPUT_DIR ?= $(WORKSPACE)/models/Qwen2.5-0.5B-Instruct-GGUF-imatrix-smoke
GGUF_SMOKE_CORPUS_DIR ?= $(WORKSPACE)/data/quantization/mopep-imatrix-smoke

export HF_HOME PIP_CACHE_DIR TMPDIR CUDA_VISIBLE_DEVICES

.DEFAULT_GOAL := help

.PHONY: help doctor setup preflight upload-preflight check-hf-token print-config download-model data-check \
        run-quantize-model smoke-quantize-model verify-quantized-model \
        verify-for-upload upload-quantized-model \
        gguf-doctor gguf-fetch-llama gguf-build-tools gguf-setup gguf-download-model \
        gguf-data-check gguf-export-corpus gguf-smoke-build gguf-build-comparison gguf-verify-comparison \
        _gguf-build-comparison-locked gguf-print-config test check

help:
	@printf '%s\n' \
		'Pipeline de quantizacao Qwen com golden MOPEP' \
		'' \
		'  make doctor                    Diagnostica GPU, memoria, disco, Python e token' \
		'  make setup                     Cria o venv persistente e instala dependencias' \
		'  make data-check                Valida o CSV e a selecao deterministica' \
		'  make smoke-quantize-model       Testa GPTQ 4-bit com Qwen 0.5B e 8 exemplos' \
		'  make run-quantize-model         Quantiza o modelo configurado; nao publica' \
		'  make verify-quantized-model     Verifica arquivos e recarrega o checkpoint' \
		'  make upload-quantized-model     Publica somente a pasta validada no HF' \
		'' \
		'Comparacao GGUF com iMatrix (llama.cpp):' \
		'  make gguf-doctor                Valida imagem CUDA devel e recursos do Pod' \
		'  make gguf-setup                 Fixa llama.cpp e instala o conversor em venv separado' \
		'  make gguf-smoke-build           Testa o fluxo inteiro com Qwen 0.5B e 8 exemplos' \
		'  make gguf-build-comparison      Gera Q4 plain, Q4+iMatrix e Q8 a partir do mesmo BF16' \
		'  make gguf-verify-comparison     Confere assinatura e SHA dos tres GGUF finais' \
		'  make gguf-print-config          Mostra a configuracao sem segredos' \
		'' \
		'Variaveis principais:' \
		'  QUANT_MODEL_ID, QUANT_MODEL_REVISION, QUANT_MODEL_DIR' \
		'  QUANT_METHOD=gptq|awq|exl3, QUANT_BITS, QUANT_GROUP_SIZE' \
		'  QUANT_DATASET, QUANT_SAMPLES, QUANT_SEED, QUANT_OUTPUT_DIR' \
		'  QUANT_HF_REPO_ID, QUANT_HF_PRIVATE=1|0, HF_TOKEN (somente ambiente)' \
		'  QUANT_HF_ALLOW_PUBLIC e QUANT_HF_REPLACE_EXISTING sao confirmacoes' \
		'  QUANT_ENV_FILE pode apontar para um arquivo Make local sem segredos' \
		'  GGUF_OUTPUT_DIR, GGUF_LLAMA_CPP_REVISION, GGUF_SAMPLES e GGUF_THREADS'

gguf-doctor:
	@command -v "$(SYSTEM_PYTHON)" >/dev/null || { echo 'Python ausente' >&2; exit 1; }
	@"$(SYSTEM_PYTHON)" -c 'import sys; assert (3, 11) <= sys.version_info < (3, 14), "Python 3.11-3.13 obrigatorio"; print("Python:", sys.version.split()[0])'
	@for command in git cmake g++ nvcc nvidia-smi free df sha256sum flock; do command -v "$$command" >/dev/null || { echo "$$command ausente; use uma imagem NVIDIA CUDA devel" >&2; exit 1; }; done
	@"$(SYSTEM_PYTHON)" -c 'import venv'
	@[[ "$(GGUF_SAMPLES)" =~ ^[1-9][0-9]*$$ ]] || { echo 'GGUF_SAMPLES deve ser inteiro positivo' >&2; exit 2; }
	@[[ "$(GGUF_SEED)" =~ ^[0-9]+$$ ]] || { echo 'GGUF_SEED deve ser inteiro nao negativo' >&2; exit 2; }
	@for value in "$(GGUF_THREADS)" "$(GGUF_GPU_LAYERS)" "$(GGUF_CONTEXT)" "$(GGUF_BATCH)" "$(GGUF_UBATCH)"; do [[ "$$value" =~ ^[1-9][0-9]*$$ ]] || { echo "Parametro GGUF invalido: $$value" >&2; exit 2; }; done
	@if test "$$(basename "$(QUANT_DATASET)")" = test.csv; then echo 'test.csv e reservado para avaliacao final' >&2; exit 2; fi
	@nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
	@nvcc --version | tail -n 1
	@printf 'RAM:\n'; free -h
	@printf 'Disco em %s:\n' "$(WORKSPACE)"; df -h "$(WORKSPACE)"

gguf-fetch-llama:
	@mkdir -p "$(dir $(GGUF_LLAMA_CPP_DIR))"
	@if test -e "$(GGUF_LLAMA_CPP_DIR)" && ! test -d "$(GGUF_LLAMA_CPP_DIR)/.git"; then echo 'GGUF_LLAMA_CPP_DIR existe e nao e um checkout Git valido' >&2; exit 1; fi
	@if ! test -d "$(GGUF_LLAMA_CPP_DIR)/.git"; then \
		mkdir -p "$(GGUF_LLAMA_CPP_DIR)"; \
		git -C "$(GGUF_LLAMA_CPP_DIR)" init; \
		git -C "$(GGUF_LLAMA_CPP_DIR)" remote add origin https://github.com/ggml-org/llama.cpp.git; \
	fi
	@origin="$$(git -C "$(GGUF_LLAMA_CPP_DIR)" remote get-url origin)"; \
	case "$$origin" in https://github.com/ggml-org/llama.cpp.git|git@github.com:ggml-org/llama.cpp.git) ;; *) echo "Remote llama.cpp inesperado: $$origin" >&2; exit 1 ;; esac
	@if ! test "$$(git -C "$(GGUF_LLAMA_CPP_DIR)" rev-parse HEAD 2>/dev/null || true)" = "$(GGUF_LLAMA_CPP_REVISION)"; then \
		git -C "$(GGUF_LLAMA_CPP_DIR)" diff --quiet && git -C "$(GGUF_LLAMA_CPP_DIR)" diff --cached --quiet || { echo 'Checkout llama.cpp tem mudancas rastreadas; use outro diretorio' >&2; exit 1; }; \
		git -C "$(GGUF_LLAMA_CPP_DIR)" fetch --depth=1 origin "$(GGUF_LLAMA_CPP_REVISION)"; \
		git -C "$(GGUF_LLAMA_CPP_DIR)" checkout --detach FETCH_HEAD; \
	fi
	@test "$$(git -C "$(GGUF_LLAMA_CPP_DIR)" rev-parse HEAD)" = "$(GGUF_LLAMA_CPP_REVISION)" || { echo 'Falha ao fixar a revisao do llama.cpp' >&2; exit 1; }
	@git -C "$(GGUF_LLAMA_CPP_DIR)" diff --quiet && git -C "$(GGUF_LLAMA_CPP_DIR)" diff --cached --quiet || { echo 'Checkout llama.cpp tem mudancas rastreadas; use outro diretorio' >&2; exit 1; }

gguf-build-tools: gguf-doctor gguf-fetch-llama
	@cmake -S "$(GGUF_LLAMA_CPP_DIR)" -B "$(GGUF_LLAMA_CPP_BUILD_DIR)" \
		-DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
	@cmake --build "$(GGUF_LLAMA_CPP_BUILD_DIR)" --config Release \
		--target llama-imatrix llama-quantize llama-server -j"$(GGUF_THREADS)"
	@test -x "$(GGUF_LLAMA_CPP_BUILD_DIR)/bin/llama-imatrix"
	@test -x "$(GGUF_LLAMA_CPP_BUILD_DIR)/bin/llama-quantize"
	@test -x "$(GGUF_LLAMA_CPP_BUILD_DIR)/bin/llama-server"

gguf-setup: gguf-fetch-llama
	@test -f "$(GGUF_CONVERT_REQUIREMENTS)" || { echo "Requirements do conversor ausente: $(GGUF_CONVERT_REQUIREMENTS)" >&2; exit 1; }
	@mkdir -p "$(dir $(GGUF_VENV))" "$(HF_HOME)" "$(PIP_CACHE_DIR)" "$(TMPDIR)" "$(GGUF_CORPUS_DIR)" "$(dir $(GGUF_LOCK_FILE))" "$(WORKSPACE)/models"
	@test -x "$(GGUF_PYTHON)" || "$(SYSTEM_PYTHON)" -m venv "$(GGUF_VENV)"
	@revision="$$(git -C "$(GGUF_LLAMA_CPP_DIR)" rev-parse HEAD)"; stamp="$(GGUF_VENV)/.llama-cpp-revision"; \
	if ! test -f "$$stamp" || ! test "$$(cat "$$stamp")" = "$$revision"; then \
		"$(GGUF_PYTHON)" -m pip install --disable-pip-version-check --upgrade pip; \
		"$(GGUF_PYTHON)" -m pip install --disable-pip-version-check -r "$(GGUF_CONVERT_REQUIREMENTS)"; \
		"$(GGUF_PYTHON)" -m pip check; \
		printf '%s\n' "$$revision" > "$$stamp"; \
	else echo 'Dependencias do conversor ja correspondem ao commit fixado do llama.cpp'; fi
	@"$(GGUF_PYTHON)" -c 'import huggingface_hub, transformers; print("huggingface_hub:", huggingface_hub.__version__); print("transformers:", transformers.__version__)'

gguf-download-model: gguf-setup
	@mkdir -p "$(QUANT_MODEL_DIR)"
	@resolved="$$($(GGUF_PYTHON) -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)' "$(QUANT_MODEL_ID)" "$(QUANT_MODEL_REVISION)")"; \
	test -n "$$resolved" || { echo 'Nao foi possivel resolver a revisao do modelo' >&2; exit 1; }; \
	if test -f "$(QUANT_MODEL_DIR)/.source_revision"; then existing="$$(cat "$(QUANT_MODEL_DIR)/.source_revision")"; test "$$existing" = "$$resolved" || { echo "Diretorio pertence a $$existing, mas a fonte resolveu para $$resolved" >&2; exit 1; }; fi; \
	"$(GGUF_PYTHON)" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' "$(QUANT_MODEL_ID)" "$$resolved" "$(QUANT_MODEL_DIR)"; \
	printf '%s\n' "$$resolved" > "$(QUANT_MODEL_DIR)/.source_revision"

gguf-data-check:
	@"$(SYSTEM_PYTHON)" calibration.py "$(QUANT_DATASET)" --limit "$(GGUF_SAMPLES)" --seed "$(GGUF_SEED)"

gguf-export-corpus: gguf-setup gguf-download-model gguf-data-check
	@if test -e "$(GGUF_CORPUS_FILE)" || test -e "$(GGUF_CORPUS_MANIFEST)"; then \
		test -f "$(GGUF_CORPUS_FILE)" && test -f "$(GGUF_CORPUS_MANIFEST)" || { echo 'Corpus imatrix parcial; inspecione os dois caminhos' >&2; exit 1; }; \
		action=--verify-only; \
	else action=; fi; \
	"$(GGUF_PYTHON)" build_imatrix_corpus.py \
		--dataset "$(QUANT_DATASET)" --tokenizer-dir "$(QUANT_MODEL_DIR)" \
		--output "$(GGUF_CORPUS_FILE)" --manifest "$(GGUF_CORPUS_MANIFEST)" \
		--limit "$(GGUF_SAMPLES)" --seed "$(GGUF_SEED)" $$action
	@chmod 600 "$(GGUF_CORPUS_FILE)" "$(GGUF_CORPUS_MANIFEST)"

gguf-build-comparison:
	@mkdir -p "$(dir $(GGUF_LOCK_FILE))"
	@exec 9>"$(GGUF_LOCK_FILE)"; \
	flock -n 9 || { echo 'Outra quantizacao/verificacao esta ativa neste Pod; aguarde.' >&2; exit 1; }; \
	"$(MAKE_BIN)" --no-print-directory -C "$(CURDIR)" _gguf-build-comparison-locked

gguf-smoke-build:
	@"$(MAKE_BIN)" --no-print-directory -C "$(CURDIR)" gguf-build-comparison \
		QUANT_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
		QUANT_MODEL_REVISION=main \
		QUANT_MODEL_DIR="$(GGUF_SMOKE_MODEL_DIR)" \
		GGUF_MODEL_PREFIX=Qwen2.5-0.5B-Instruct \
		GGUF_OUTPUT_DIR="$(GGUF_SMOKE_OUTPUT_DIR)" \
		GGUF_CORPUS_DIR="$(GGUF_SMOKE_CORPUS_DIR)" \
		GGUF_CORPUS_FILE="$(GGUF_SMOKE_CORPUS_DIR)/train-chat-template.txt" \
		GGUF_CORPUS_MANIFEST="$(GGUF_SMOKE_CORPUS_DIR)/train-chat-template.manifest.json" \
		GGUF_SAMPLES=8
	@"$(MAKE_BIN)" --no-print-directory -C "$(CURDIR)" gguf-verify-comparison \
		GGUF_MODEL_PREFIX=Qwen2.5-0.5B-Instruct \
		GGUF_OUTPUT_DIR="$(GGUF_SMOKE_OUTPUT_DIR)"

_gguf-build-comparison-locked: gguf-doctor gguf-build-tools gguf-setup gguf-download-model gguf-export-corpus
	@mkdir -p "$(GGUF_OUTPUT_DIR)"
	@"$(GGUF_PYTHON)" build_gguf_comparison.py \
		--llama-cpp-dir "$(GGUF_LLAMA_CPP_DIR)" --llama-cpp-build-dir "$(GGUF_LLAMA_CPP_BUILD_DIR)" \
		--llama-cpp-revision "$(GGUF_LLAMA_CPP_REVISION)" \
		--hf-model-dir "$(QUANT_MODEL_DIR)" --source-model-id "$(QUANT_MODEL_ID)" \
		--corpus "$(GGUF_CORPUS_FILE)" --corpus-manifest "$(GGUF_CORPUS_MANIFEST)" \
		--output-dir "$(GGUF_OUTPUT_DIR)" --prefix "$(GGUF_MODEL_PREFIX)" \
		--python "$(GGUF_PYTHON)" --threads "$(GGUF_THREADS)" --gpu-layers "$(GGUF_GPU_LAYERS)" \
		--context "$(GGUF_CONTEXT)" --batch "$(GGUF_BATCH)" --ubatch "$(GGUF_UBATCH)"

gguf-verify-comparison:
	@mkdir -p "$(dir $(GGUF_LOCK_FILE))"
	@exec 9>"$(GGUF_LOCK_FILE)"; \
	flock -n 9 || { echo 'Outra quantizacao/verificacao esta ativa neste Pod; aguarde.' >&2; exit 1; }; \
	"$(SYSTEM_PYTHON)" build_gguf_comparison.py --verify-only \
		--output-dir "$(GGUF_OUTPUT_DIR)" --prefix "$(GGUF_MODEL_PREFIX)"

gguf-print-config:
	@printf '%s\n' \
		"SOURCE_MODEL=$(QUANT_MODEL_ID)" "SOURCE_REVISION_SELECTOR=$(QUANT_MODEL_REVISION)" \
		"SOURCE_DIR=$(QUANT_MODEL_DIR)" "DATASET=$(QUANT_DATASET)" \
		"SAMPLES=$(GGUF_SAMPLES)" "SEED=$(GGUF_SEED)" \
		"LLAMA_CPP_DIR=$(GGUF_LLAMA_CPP_DIR)" "LLAMA_CPP_REVISION=$(GGUF_LLAMA_CPP_REVISION)" \
		"CORPUS=$(GGUF_CORPUS_FILE)" "OUTPUT_DIR=$(GGUF_OUTPUT_DIR)" \
		"THREADS=$(GGUF_THREADS)" "GPU_LAYERS=$(GGUF_GPU_LAYERS)"

doctor:
	@command -v "$(SYSTEM_PYTHON)" >/dev/null || { echo 'Python ausente' >&2; exit 1; }
	@"$(SYSTEM_PYTHON)" -c 'import sys; assert (3, 11) <= sys.version_info < (3, 14), "Python 3.11-3.13 obrigatorio"; print("Python:", sys.version.split()[0])'
	@command -v git >/dev/null || { echo 'git ausente' >&2; exit 1; }
	@command -v make >/dev/null || { echo 'make ausente' >&2; exit 1; }
	@command -v gcc >/dev/null || { echo 'gcc/build-essential ausente' >&2; exit 1; }
	@command -v free >/dev/null || { echo 'free ausente' >&2; exit 1; }
	@command -v df >/dev/null || { echo 'df ausente' >&2; exit 1; }
	@command -v sha256sum >/dev/null || { echo 'sha256sum ausente' >&2; exit 1; }
	@command -v flock >/dev/null || { echo 'flock/util-linux ausente' >&2; exit 1; }
	@"$(SYSTEM_PYTHON)" -c 'import venv' || { echo 'Modulo venv ausente' >&2; exit 1; }
	@command -v nvidia-smi >/dev/null || { echo 'nvidia-smi ausente; use um Pod com GPU NVIDIA' >&2; exit 1; }
	@driver="$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"; \
	major="$${driver%%.*}"; \
	[[ "$$major" =~ ^[0-9]+$$ ]] || { echo "Versao de driver NVIDIA invalida: $$driver" >&2; exit 1; }; \
	test "$$major" -ge "$(CUDA_MIN_DRIVER_BRANCH)" || { echo "Driver NVIDIA $$driver e anterior ao ramo $(CUDA_MIN_DRIVER_BRANCH) exigido pelo lock CUDA 13" >&2; exit 1; }
	@nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
	@printf 'RAM:\n'; free -h
	@printf 'Disco raiz:\n'; df -h /
	@printf 'Disco em %s:\n' "$(WORKSPACE)"; df -h "$(WORKSPACE)"
	@if test -n "$${HF_TOKEN:-}"; then echo 'HF_TOKEN: configurado'; else echo 'HF_TOKEN: ausente (necessario apenas para repos privados/upload)'; fi
	@printf 'CUDA_VISIBLE_DEVICES=%s\n' "$(CUDA_VISIBLE_DEVICES)"

setup:
	@"$(SYSTEM_PYTHON)" -c 'import sys; assert (3, 11) <= sys.version_info < (3, 14), "Python 3.11-3.13 obrigatorio"'
	@mkdir -p "$(dir $(QUANT_VENV))" "$(HF_HOME)" "$(PIP_CACHE_DIR)" "$(TMPDIR)" "$(dir $(QUANT_LOCK_FILE))" "$(WORKSPACE)/models" "$(WORKSPACE)/data/mopep-golden"
	@test -x "$(QUANT_PYTHON)" || "$(SYSTEM_PYTHON)" -m venv "$(QUANT_VENV)"
	@current="$$(sha256sum requirements.txt | awk '{print $$1}')"; \
	stamp="$(QUANT_VENV)/.requirements.sha256"; \
	if ! test -f "$$stamp" || ! test "$$(cat "$$stamp")" = "$$current"; then \
		"$(QUANT_PYTHON)" -m pip install --disable-pip-version-check --upgrade pip; \
		"$(QUANT_PYTHON)" -m pip install --disable-pip-version-check -r requirements.txt; \
		"$(QUANT_PYTHON)" -m pip check; \
		printf '%s\n' "$$current" > "$$stamp"; \
	else \
		echo 'Dependencias de quantizacao ja correspondem a requirements.txt'; \
	fi
	@"$(QUANT_PYTHON)" -c 'import importlib.metadata as metadata; import torch, gptqmodel; print("torch:", torch.__version__); print("gptqmodel:", metadata.version("GPTQModel")); print("cuda:", torch.cuda.is_available()); assert torch.cuda.is_available(), "PyTorch nao detectou CUDA"'

preflight:
	@case "$(QUANT_METHOD)" in gptq|awq|exl3) ;; *) echo 'QUANT_METHOD deve ser gptq, awq ou exl3 neste fluxo com golden' >&2; exit 2 ;; esac
	@if test "$(QUANT_METHOD)" = exl3; then \
		[[ "$(QUANT_BITS)" =~ ^([1-7]([.][0-9]+)?|8([.]0+)?)$$ ]] || { echo 'EXL3 exige QUANT_BITS entre 1.0 e 8.0' >&2; exit 2; }; \
	else \
		[[ "$(QUANT_BITS)" =~ ^[2-8]$$ ]] || { echo 'GPTQ/AWQ exigem QUANT_BITS inteiro entre 2 e 8' >&2; exit 2; }; \
	fi
	@[[ "$(QUANT_SAMPLES)" =~ ^[1-9][0-9]*$$ ]] || { echo 'QUANT_SAMPLES deve ser inteiro positivo' >&2; exit 2; }
	@[[ "$(QUANT_BATCH_SIZE)" =~ ^[1-9][0-9]*$$ ]] || { echo 'QUANT_BATCH_SIZE deve ser inteiro positivo' >&2; exit 2; }
	@[[ "$(QUANT_GROUP_SIZE)" =~ ^[1-9][0-9]*$$ ]] || { echo 'QUANT_GROUP_SIZE deve ser inteiro positivo' >&2; exit 2; }
	@[[ "$(QUANT_SEED)" =~ ^[0-9]+$$ ]] || { echo 'QUANT_SEED deve ser inteiro nao negativo' >&2; exit 2; }
	@if test "$(QUANT_METHOD)" = awq && test "$(QUANT_BITS)" != 4; then echo 'AWQ aceita somente QUANT_BITS=4' >&2; exit 2; fi
	@if test "$(QUANT_METHOD)" != gptq; then command -v nvcc >/dev/null || { echo 'AWQ/EXL3 exigem uma imagem CUDA devel com nvcc neste fluxo' >&2; exit 2; }; fi
	@if test "$$(basename "$(QUANT_DATASET)")" = test.csv; then echo 'test.csv e reservado para avaliacao final' >&2; exit 2; fi
	@if test -e "$(QUANT_OUTPUT_DIR)" && ! test -d "$(QUANT_OUTPUT_DIR)"; then echo 'QUANT_OUTPUT_DIR existe e nao e diretorio' >&2; exit 1; fi
	@if test -d "$(QUANT_OUTPUT_DIR)" && test -n "$$(find "$(QUANT_OUTPUT_DIR)" -mindepth 1 -print -quit)"; then \
		echo "Saida ja existe e nao esta vazia: $(QUANT_OUTPUT_DIR)" >&2; \
		echo 'Use outro QUANT_OUTPUT_DIR ou inspecione/mova a pasta parcial.' >&2; \
		exit 1; \
	fi

upload-preflight:
	@case "$(QUANT_HF_PRIVATE)" in 0|1) ;; *) echo 'QUANT_HF_PRIVATE deve ser 0 ou 1' >&2; exit 2 ;; esac
	@case "$(QUANT_HF_ALLOW_PUBLIC)" in 0|1) ;; *) echo 'QUANT_HF_ALLOW_PUBLIC deve ser 0 ou 1' >&2; exit 2 ;; esac
	@case "$(QUANT_HF_REPLACE_EXISTING)" in 0|1) ;; *) echo 'QUANT_HF_REPLACE_EXISTING deve ser 0 ou 1' >&2; exit 2 ;; esac
	@if test "$(QUANT_HF_PRIVATE)" = 0 && test "$(QUANT_HF_ALLOW_PUBLIC)" != 1; then \
		echo 'Upload publico exige QUANT_HF_ALLOW_PUBLIC=1' >&2; exit 2; \
	fi

check-hf-token: setup
	@test -n "$${HF_TOKEN:-}" || { echo 'HF_TOKEN ausente. Configure um RunPod Secret ou use read -rsp e export.' >&2; exit 1; }
	@"$(HF)" auth whoami

print-config:
	@printf '%s\n' \
		"MODEL_ID=$(QUANT_MODEL_ID)" \
		"MODEL_REVISION=$(QUANT_MODEL_REVISION)" \
		"MODEL_DIR=$(QUANT_MODEL_DIR)" \
		"METHOD=$(QUANT_METHOD)" \
		"BITS=$(QUANT_BITS)" \
		"GROUP_SIZE=$(QUANT_GROUP_SIZE)" \
		"BATCH_SIZE=$(QUANT_BATCH_SIZE)" \
		"DATASET=$(QUANT_DATASET)" \
		"SAMPLES=$(QUANT_SAMPLES)" \
		"SEED=$(QUANT_SEED)" \
		"OUTPUT_DIR=$(QUANT_OUTPUT_DIR)" \
		"TMPDIR=$(TMPDIR)" \
		"HF_REPO_ID=$(QUANT_HF_REPO_ID)" \
		"HF_PRIVATE=$(QUANT_HF_PRIVATE)"

download-model: setup
	@mkdir -p "$(QUANT_MODEL_DIR)"
	@resolved="$$($(QUANT_PYTHON) -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)' "$(QUANT_MODEL_ID)" "$(QUANT_MODEL_REVISION)")"; \
	test -n "$$resolved" || { echo 'Nao foi possivel resolver a revisao do modelo' >&2; exit 1; }; \
	if test -f "$(QUANT_MODEL_DIR)/.source_revision"; then \
		existing="$$(cat "$(QUANT_MODEL_DIR)/.source_revision")"; \
		test "$$existing" = "$$resolved" || { \
			echo "$(QUANT_MODEL_DIR) pertence a revisao $$existing, mas $(QUANT_MODEL_REVISION) resolveu para $$resolved" >&2; \
			echo 'Escolha outro QUANT_MODEL_DIR para nao misturar snapshots.' >&2; \
			exit 1; \
		}; \
	fi; \
	printf '%s\n' "$$resolved" > "$(QUANT_MODEL_DIR)/.source_revision"; \
	echo "Baixando $(QUANT_MODEL_ID) na revisao $$resolved"; \
	"$(HF)" download "$(QUANT_MODEL_ID)" --revision "$$resolved" --local-dir "$(QUANT_MODEL_DIR)"

data-check: setup
	@"$(QUANT_PYTHON)" calibration.py "$(QUANT_DATASET)" --limit "$(QUANT_SAMPLES)" --seed "$(QUANT_SEED)"

run-quantize-model: preflight doctor setup data-check download-model
	@exec 9>"$(QUANT_LOCK_FILE)"; \
	flock -n 9 || { echo 'Outra quantizacao/verificacao esta ativa neste Pod; aguarde a conclusao.' >&2; exit 1; }; \
	revision="$$(cat "$(QUANT_MODEL_DIR)/.source_revision")"; \
	"$(QUANT_PYTHON)" quantize.py \
		"$(QUANT_MODEL_DIR)" \
		"$(QUANT_OUTPUT_DIR)" \
		"$(QUANT_METHOD)" \
		"$(QUANT_BITS)" \
		--calibration-csv "$(QUANT_DATASET)" \
		--calibration-limit "$(QUANT_SAMPLES)" \
		--calibration-seed "$(QUANT_SEED)" \
		--group-size "$(QUANT_GROUP_SIZE)" \
		--batch-size "$(QUANT_BATCH_SIZE)" \
		--source-model-id "$(QUANT_MODEL_ID)" \
		--source-model-revision "$$revision"
	@echo "Quantizacao concluida em $(QUANT_OUTPUT_DIR)"

verify-quantized-model: setup
	@reload=''; \
	if test "$(QUANT_VERIFY_RELOAD)" = 1; then reload='--reload'; fi; \
	if test -n "$$reload"; then exec 9>"$(QUANT_LOCK_FILE)"; flock -n 9 || { echo 'Outra quantizacao/verificacao esta ativa neste Pod.' >&2; exit 1; }; fi; \
	"$(QUANT_PYTHON)" verify_artifact.py "$(QUANT_OUTPUT_DIR)" $$reload --device "cuda:0" \
		--expected-method "$(QUANT_METHOD)" \
		--expected-bits "$(QUANT_BITS)" \
		--expected-source-model-id "$(QUANT_MODEL_ID)"

smoke-quantize-model:
	@$(MAKE) run-quantize-model \
		QUANT_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
		QUANT_MODEL_REVISION=main \
		QUANT_MODEL_DIR="$(WORKSPACE)/models/Qwen2.5-0.5B-Instruct-original" \
		QUANT_METHOD=gptq \
		QUANT_BITS=4 \
		QUANT_GROUP_SIZE=128 \
		QUANT_BATCH_SIZE=1 \
		QUANT_DATASET="$(QUANT_DATASET)" \
		QUANT_SAMPLES=8 \
		QUANT_SEED="$(QUANT_SEED)" \
		QUANT_OUTPUT_DIR="$(QUANT_SMOKE_OUTPUT_DIR)"
	@$(MAKE) verify-quantized-model \
		QUANT_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
		QUANT_METHOD=gptq \
		QUANT_BITS=4 \
		QUANT_VERIFY_RELOAD=1 \
		QUANT_OUTPUT_DIR="$(QUANT_SMOKE_OUTPUT_DIR)"

verify-for-upload: setup
	@exec 9>"$(QUANT_LOCK_FILE)"; \
	flock -n 9 || { echo 'Outra quantizacao/verificacao esta ativa neste Pod.' >&2; exit 1; }; \
	"$(QUANT_PYTHON)" verify_artifact.py "$(QUANT_OUTPUT_DIR)" --reload --device "cuda:0" \
		--expected-method "$(QUANT_METHOD)" \
		--expected-bits "$(QUANT_BITS)" \
		--expected-source-model-id "$(QUANT_MODEL_ID)"

upload-quantized-model: upload-preflight check-hf-token verify-for-upload
	@"$(QUANT_PYTHON)" publish_model.py \
		"$(QUANT_OUTPUT_DIR)" \
		"$(QUANT_HF_REPO_ID)" \
		--private "$(QUANT_HF_PRIVATE)" \
		--allow-public "$(QUANT_HF_ALLOW_PUBLIC)" \
		--replace-existing "$(QUANT_HF_REPLACE_EXISTING)" \
		--commit-message "Add $(QUANT_METHOD_LABEL) $(QUANT_BITS)-bit quantized model"

test:
	@"$(SYSTEM_PYTHON)" -m unittest discover -s tests -p 'test_*.py'

check: test
	@"$(SYSTEM_PYTHON)" -m py_compile calibration.py quantize.py verify_artifact.py publish_model.py \
		build_imatrix_corpus.py build_gguf_comparison.py benchmark.py perplexity.py
