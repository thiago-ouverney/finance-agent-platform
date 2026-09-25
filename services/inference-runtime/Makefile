# O Make precisa de um executável único; não use "/usr/bin/env bash" aqui.
ifeq ($(filter oneshell,$(.FEATURES)),)
$(error GNU Make com .ONESHELL necessário; no macOS use gmake atualizado ou execute no pod)
endif
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.NOTPARALLEL:

# Caminhos podem ser sobrescritos na chamada:
# make quantize-q8 LLAMA_CPP_DIR=/mnt/llama.cpp HF_MODEL_DIR=/mnt/qwen
# O repositório é autocontido. Usar CURDIR evita assumir que ele foi clonado
# em um nome específico dentro de /workspace.
BENCH_VENV ?= $(CURDIR)/.venv
SYSTEM_PYTHON ?= python3
PYTHON ?= $(BENCH_VENV)/bin/python
HF ?= $(BENCH_VENV)/bin/hf
LLAMA_CPP_DIR ?= /workspace/llama.cpp
LLAMA_CPP_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build
LLAMA_VENV ?= /workspace/llama-cpp-venv
LLAMA_PYTHON ?= $(LLAMA_VENV)/bin/python
HF_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
HF_REVISION ?= main
HF_MODEL_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-original
HF_HUB_ENABLE_HF_TRANSFER ?= 0
HF_REPO_ID ?= yanwerneck/Qwen2.5-7B-Instruct-GGUF-Q8_0
HF_UPLOAD_FILENAME ?= Qwen2.5-7B-Instruct-Q8_0.gguf
GGUF_OUTPUT_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-GGUF-Q8_0
QUANTIZE_THREADS ?= $(shell nproc 2>/dev/null || echo 1)
MODEL_SIZE ?= 7B
MODEL_7B_GGUF ?= /workspace/models/Qwen2.5-7B-Instruct-Q8_0/Qwen2.5-7B-Instruct-Q8_0.gguf
MODEL_14B_GGUF ?= /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
MODEL_7B_TOKENIZER ?= /workspace/models/Qwen2.5-7B-Instruct-original
MODEL_14B_TOKENIZER ?= /workspace/models/Qwen2.5-14B-tokenizer
VLLM_EXTRA_ARGS ?=
LLAMA_EXTRA_ARGS ?=
OLLAMA_EXTRA_ARGS ?=
export VLLM_EXTRA_ARGS LLAMA_EXTRA_ARGS OLLAMA_EXTRA_ARGS
ifeq ($(MODEL_SIZE),7B)
GGUF_FILE := $(MODEL_7B_GGUF)
TOKENIZER_DIR ?= $(MODEL_7B_TOKENIZER)
HF_GGUF_REPO ?= arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf
HF_GGUF_FILENAME ?= Qwen2.5-7B-Instruct-Q8_0.gguf
HF_TOKENIZER_MODEL ?= Qwen/Qwen2.5-7B-Instruct
VLLM_MODEL_DIR ?= $(MODEL_7B_GGUF)
VLLM_CONFIG ?= configs/vllm-7b-gguf.json
VLLM_LAUNCH ?= configs/launch-vllm-7b-gguf.example.json
LLAMA_MODEL_DIR ?= $(MODEL_7B_GGUF)
LLAMA_CONFIG ?= configs/llamacpp-7b-gguf.json
LLAMA_LAUNCH ?= configs/launch-llamacpp-7b-gguf.json
OLLAMA_MODEL_DIR ?= $(MODEL_7B_GGUF)
OLLAMA_CONFIG ?= configs/ollama-7b-gguf.json
OLLAMA_LAUNCH ?= configs/launch-ollama-7b-gguf.json
OLLAMA_MODEL_NAME ?= qwen7b-q8-gguf
else ifeq ($(MODEL_SIZE),14B)
GGUF_FILE := $(MODEL_14B_GGUF)
TOKENIZER_DIR ?= $(MODEL_14B_TOKENIZER)
HF_GGUF_REPO ?= arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf
HF_GGUF_FILENAME ?= Qwen2.5-14B-Instruct-Q8_0.gguf
HF_TOKENIZER_MODEL ?= Qwen/Qwen2.5-14B-Instruct
VLLM_MODEL_DIR ?= $(MODEL_14B_GGUF)
VLLM_CONFIG ?= configs/vllm-14b-gguf.json
VLLM_LAUNCH ?= configs/launch-vllm-14b-gguf.json
LLAMA_MODEL_DIR ?= $(MODEL_14B_GGUF)
LLAMA_CONFIG ?= configs/llamacpp-14b-gguf.json
LLAMA_LAUNCH ?= configs/launch-llamacpp-14b-gguf.json
OLLAMA_MODEL_DIR ?= $(MODEL_14B_GGUF)
OLLAMA_CONFIG ?= configs/ollama-14b-gguf.json
OLLAMA_LAUNCH ?= configs/launch-ollama-14b-gguf.json
OLLAMA_MODEL_NAME ?= qwen14b-q8-gguf
else
$(error MODEL_SIZE deve ser 7B ou 14B)
endif
# Qwen2.5: KV em FP16/BF16: 2(K,V) × camadas × cabeças KV × dimensão da cabeça × bytes.
# O sweep usa este coeficiente para converter o passo de memória em tokens.
ifeq ($(MODEL_SIZE),7B)
KV_BYTES_PER_TOKEN ?= 57344
else
KV_BYTES_PER_TOKEN ?= 196608
endif
# A imagem do pod normalmente expõe vLLM no PATH. O launch JSON usa o nome
# `vllm`, evitando depender do caminho do venv usado para construir a imagem.
VLLM_BIN ?= $(shell command -v vllm 2>/dev/null || printf 'vllm')
VLLM_PYTHON ?= /app/.vllm_venv/bin/python
# A imagem do pod já fornece llama-server no PATH.  O build local continua
# disponível em `make build-llama`, mas não é pré-requisito do benchmark.
LLAMA_SERVER_BIN ?= $(shell p=$$(command -v llama-server 2>/dev/null || true); if test -x "$$p"; then printf '%s' "$$p"; else found=; for p in /usr/local/lib/ollama/llama-server /workspace/llama.cpp/build/bin/llama-server /app/llama.cpp/build/bin/llama-server; do if test -x "$$p"; then printf '%s' "$$p"; found=1; break; fi; done; test "$${found:-}" = 1 || printf 'llama-server'; fi)
LLAMA_BACKEND_PATH ?= $(firstword $(wildcard /usr/local/lib/ollama/cuda_v12/libggml-cuda.so /usr/local/lib/ollama/cuda_v13/libggml-cuda.so))
REQUIRE_GPU ?= 1
OLLAMA_BIN ?= $(shell command -v ollama 2>/dev/null || printf 'ollama')
BENCH_SCENARIOS ?= short medium long
BENCH_REQUESTS ?= 50
BENCH_REPETITIONS ?= 1
BENCH_WARMUP ?= 3
BENCH_MODE ?= replay
CONVERSATION_TURNS ?= 1
CONVERSATION_FIXTURE ?= workloads/conversations/qwen_chat_bench_v2.json
WARMUP_CONVERSATION_FIXTURE ?= workloads/conversations/qwen_chat_warmup_v1.json
BENCH_STARTUP_TIMEOUT ?= 1800
# Reutiliza dependências e pesos já preparados, sem pip/downloads.
PREPARE_OFFLINE ?= 0
SWEEP_MAX_CONTEXT ?= 16384
SWEEP_START ?= 1024
SWEEP_MEMORY_STEP_MB ?= 256
SWEEP_REQUESTS ?= $(BENCH_REQUESTS)
SWEEP_REPETITIONS ?= $(BENCH_REPETITIONS)
SWEEP_WARMUP ?= $(BENCH_WARMUP)
SWEEP_MODE ?= $(BENCH_MODE)
SWEEP_CONVERSATION_TURNS ?= $(CONVERSATION_TURNS)
SWEEP_CONVERSATION_FIXTURE ?= $(CONVERSATION_FIXTURE)
POD_SSH ?=
POD_PORT ?= 22
POD_SSH_KEY ?= $(HOME)/.ssh/id_ed25519
REMOTE_BENCH_DIR ?= /workspace/llm_local_inference_masters
REMOTE_RESULTS_DIR ?= $(REMOTE_BENCH_DIR)/results
LOCAL_RESULTS_DIR ?= results-from-pod

QUANTIZE_BIN := $(LLAMA_CPP_BUILD_DIR)/bin/llama-quantize
QUANTIZE_SCRIPT := scripts/quantize_hf_to_gguf_q8_0.sh

.PHONY: help check-tools clone-llama build-llama install-llama-python \
        download-source inspect-source quantize-q8 verify-gguf upload-hf \
        install-benchmark download-model download-tokenizer check-runtimes check-vllm-gguf prepare-benchmark prepare-vllm prepare-llama prepare-ollama prepare-all smoke-vllm smoke-llama smoke-ollama smoke-all bench-vllm bench-llama bench-ollama bench-all bench kv-sweep kv-sweep-vllm kv-sweep-llama kv-sweep-ollama quick-sweep-vllm quick-sweep-llama quick-sweep-ollama pull-results check-profilers docs clean-info

help:
	@printf '%s\n' \
		'Fluxo llama.cpp → GGUF Q8_0' \
		'' \
		'  make clone-llama          clona o llama.cpp em LLAMA_CPP_DIR' \
		'  make build-llama          clona/compila llama-server e llama-quantize com CUDA' \
		'  make install-llama-python instala dependências em venv separado' \
		'  make download-source      baixa o Qwen original para o SSD' \
		'  make inspect-source       mostra formato/metadados da fonte' \
		'  make quantize-q8          gera GGUF F16 e depois GGUF Q8_0' \
		'  make verify-gguf          confirma assinatura, tamanho e SHA-256' \
		'  make upload-hf            publica somente o GGUF no Hugging Face' \
		'  make install-benchmark    cria o venv do cliente e instala requirements.txt' \
		'  make check-runtimes       diagnostica vLLM, llama-server e Ollama instalados' \
		'  make check-vllm-gguf      confirma o plugin GGUF no mesmo ambiente do vLLM' \
		'  make prepare-vllm         prepara cliente/modelo e valida somente vLLM' \
		'  make prepare-llama        prepara cliente/modelo e valida somente llama-server' \
		'  make prepare-ollama       prepara cliente/modelo e importa o GGUF no Ollama' \
		'  make prepare-all          prepara os três runtimes em sequência' \
		'  make download-model       baixa o GGUF selecionado para o SSD' \
		'  make download-tokenizer   baixa somente os arquivos do tokenizer' \
		'  make prepare-benchmark    etapa comum interna: cliente, modelo, tokenizer e configs' \
		'  make prepare-ollama      cria o alias Ollama a partir do GGUF local, sem download' \
		'  make smoke-vllm           executa smoke do benchmark com GGUF local' \
		'  make smoke-llama          executa smoke equivalente com llama.cpp' \
		'  make smoke-ollama         executa smoke equivalente com Ollama' \
		'  make smoke-all            executa os três smokes e resume falhas' \
		'  make bench-vllm            bateria formal: short/medium/long, 50x3, com telemetria' \
		'  make bench-llama           mesma bateria usando llama-server' \
		'  make bench-ollama          mesma bateria usando Ollama (modelo já criado localmente)' \
		'  make bench-all             executa vLLM, llama.cpp e Ollama; continua e resume falhas' \
		'  make kv-sweep              compatibilidade: KV sweep do vLLM (já incluído em bench-vllm)' \
		'  make docs                 regenera os HTMLs dos documentos' \
		'  make pull-results POD_SSH=root@host  copia results via SCP (SSH TCP direto)' \
		'  make check-profilers      verifica nsys/ncu para profiling dedicado' \
		'' \
		'Variáveis úteis:' \
		'  MODEL_SIZE=7B|14B, MODEL_7B_GGUF, MODEL_14B_GGUF' \
		'  PREPARE_OFFLINE=1 reutiliza cliente/modelos locais; BENCH_STARTUP_TIMEOUT limita startup' \
		'  VLLM_EXTRA_ARGS, LLAMA_EXTRA_ARGS, OLLAMA_EXTRA_ARGS (sintaxe shell, convertida com shlex)' \
		'  VLLM_PYTHON, LLAMA_BACKEND_PATH, REQUIRE_GPU=1|0' \
		'  BENCH_MODE=replay|closed-loop|independent, CONVERSATION_TURNS, CONVERSATION_FIXTURE, WARMUP_CONVERSATION_FIXTURE' \
		'  SWEEP_START, SWEEP_MEMORY_STEP_MB, SWEEP_MAX_CONTEXT, SWEEP_REQUESTS, SWEEP_REPETITIONS, SWEEP_WARMUP, SWEEP_MODE, SWEEP_CONVERSATION_TURNS, SWEEP_CONVERSATION_FIXTURE' \
		'  LLAMA_CPP_DIR, HF_MODEL_DIR, GGUF_OUTPUT_DIR, QUANTIZE_THREADS' \
		'  VLLM_MODEL_DIR, VLLM_CONFIG, VLLM_LAUNCH' \
		'  LLAMA_MODEL_DIR, LLAMA_CONFIG, LLAMA_LAUNCH' \
		'  OLLAMA_MODEL_DIR, OLLAMA_CONFIG, OLLAMA_LAUNCH' \
		'  POD_SSH, POD_PORT, POD_SSH_KEY, REMOTE_RESULTS_DIR, LOCAL_RESULTS_DIR'


check-tools:
	@command -v $(PYTHON) >/dev/null || { echo 'Python ausente'; exit 1; }
	@command -v git >/dev/null || { echo 'Git ausente'; exit 1; }
	@command -v cmake >/dev/null || { echo 'CMake ausente'; exit 1; }

clone-llama: check-tools
	@test -d "$(LLAMA_CPP_DIR)/.git" || git clone https://github.com/ggml-org/llama.cpp.git "$(LLAMA_CPP_DIR)"

build-llama: clone-llama
	cmake -S "$(LLAMA_CPP_DIR)" -B "$(LLAMA_CPP_BUILD_DIR)" -DGGML_CUDA=ON
	cmake --build "$(LLAMA_CPP_BUILD_DIR)" --target llama-server llama-quantize -j"$(QUANTIZE_THREADS)"

install-llama-python: clone-llama
	$(PYTHON) -m venv "$(LLAMA_VENV)"
	"$(LLAMA_PYTHON)" -m pip install --upgrade pip
	"$(LLAMA_PYTHON)" -m pip install -r "$(LLAMA_CPP_DIR)/requirements.txt"

download-source:
	mkdir -p "$(HF_MODEL_DIR)"
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) download "$(HF_MODEL_ID)" --revision "$(HF_REVISION)" --local-dir "$(HF_MODEL_DIR)"

inspect-source:
	@test -f "$(HF_MODEL_DIR)/config.json" || { echo "config.json ausente em $(HF_MODEL_DIR)"; exit 1; }
	$(PYTHON) -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"config.json").read_text()); print("model_type:",c.get("model_type")); print("architectures:",c.get("architectures")); print("quantization_config:",c.get("quantization_config")); print("safetensors:",sorted(x.name for x in p.glob("*.safetensors"))); print("gguf:",sorted(x.name for x in p.glob("*.gguf")))' "$(HF_MODEL_DIR)"

quantize-q8: build-llama install-llama-python inspect-source
	chmod +x "$(QUANTIZE_SCRIPT)"
	PYTHON="$(LLAMA_PYTHON)" LLAMA_QUANTIZE_THREADS="$(QUANTIZE_THREADS)" "$(QUANTIZE_SCRIPT)" \
		--llama-cpp-dir "$(LLAMA_CPP_DIR)" \
		--hf-model-dir "$(HF_MODEL_DIR)" \
		--output-dir "$(GGUF_OUTPUT_DIR)" \
		--threads "$(QUANTIZE_THREADS)"

verify-gguf:
	@test -f "$(GGUF_FILE)" || { echo "GGUF ausente: $(GGUF_FILE)"; exit 1; }
	ls -lh "$(GGUF_FILE)"
	printf 'assinatura: '
	head -c 4 "$(GGUF_FILE)" | od -An -tc
	$(SYSTEM_PYTHON) -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); data=p.open("rb").read(4); sys.exit("assinatura GGUF inválida") if data != b"GGUF" else print("assinatura GGUF válida")' "$(GGUF_FILE)"
	sha256sum "$(GGUF_FILE)"

install-benchmark:
	@echo '[PREPARE] criando/validando venv do cliente: $(BENCH_VENV)'
	@test -x "$(PYTHON)" || "$(SYSTEM_PYTHON)" -m venv "$(BENCH_VENV)"
	"$(PYTHON)" -m pip install --disable-pip-version-check --upgrade pip
	"$(PYTHON)" -m pip install --disable-pip-version-check -r requirements.txt
	"$(PYTHON)" -m pip check

download-model: install-benchmark
	@echo '[PREPARE] baixando GGUF $(MODEL_SIZE): $(HF_GGUF_REPO)'
	mkdir -p "$(dir $(GGUF_FILE))"
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) download "$(HF_GGUF_REPO)" "$(HF_GGUF_FILENAME)" --revision "$(HF_REVISION)" \
		--local-dir "$(dir $(GGUF_FILE))"
	@test -f "$(GGUF_FILE)" || { echo "GGUF baixado em local inesperado; esperado: $(GGUF_FILE)"; exit 1; }

download-tokenizer: install-benchmark
	@echo '[PREPARE] baixando tokenizer local: $(HF_TOKENIZER_MODEL)'
	mkdir -p "$(TOKENIZER_DIR)"
	# Usamos snapshot_download com allow_patterns em vez da combinação de
	# argumentos posicionais/--include da CLI `hf`, que em algumas versões
	# interpreta os padrões como nomes de arquivo e baixa somente config.json.
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		"$(PYTHON)" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json", "chat_template.jinja"], local_files_only=False)' \
		"$(HF_TOKENIZER_MODEL)" "$(HF_REVISION)" "$(TOKENIZER_DIR)"
	@test -f "$(TOKENIZER_DIR)/config.json" || { echo "config.json do tokenizer não foi baixado" >&2; exit 1; }
	@test -f "$(TOKENIZER_DIR)/tokenizer_config.json" || { echo "tokenizer_config.json não foi baixado" >&2; exit 1; }
	@test -f "$(TOKENIZER_DIR)/tokenizer.json" || { echo "tokenizer.json não foi baixado" >&2; exit 1; }

ifeq ($(PREPARE_OFFLINE),1)
prepare-benchmark: verify-gguf
else ifeq ($(PREPARE_OFFLINE),0)
prepare-benchmark: install-benchmark download-model download-tokenizer verify-gguf
else
$(error PREPARE_OFFLINE deve ser 0 ou 1)
endif

prepare-benchmark:
	@echo '[PREPARE] criando diretório de resultados'
	mkdir -p results
	@echo '[PREPARE] validando imports do cliente'
	$(PYTHON) -c 'from importlib.metadata import version; import guidellm, httpx, psutil, transformers, huggingface_hub; print("guidellm", version("guidellm"), "httpx", version("httpx"), "psutil", version("psutil"), "transformers", version("transformers"), "huggingface_hub", version("huggingface_hub"))'
	@echo '[PREPARE] validando tokenizer e JSONs'
	test -f "$(TOKENIZER_DIR)/tokenizer_config.json"
	test -f "$(TOKENIZER_DIR)/config.json"
	$(PYTHON) -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"config.json").read_text()); mt=c.get("model_type"); print("[PREPARE] tokenizer model_type:", mt); sys.exit("config.json do tokenizer não contém model_type=qwen2; redownload do tokenizer necessário") if mt != "qwen2" else None' "$(TOKENIZER_DIR)"
	$(PYTHON) -m json.tool "$(VLLM_CONFIG)" >/dev/null
	$(PYTHON) -m json.tool "$(VLLM_LAUNCH)" >/dev/null
	$(PYTHON) -m json.tool "$(LLAMA_CONFIG)" >/dev/null
	$(PYTHON) -m json.tool "$(LLAMA_LAUNCH)" >/dev/null
	$(PYTHON) -m json.tool "$(OLLAMA_CONFIG)" >/dev/null
	$(PYTHON) -m json.tool "$(OLLAMA_LAUNCH)" >/dev/null
	@echo '[PREPARE] cliente, modelo e configs prontos; execute make check-runtimes para diagnosticar runtimes'

check-runtimes:
	missing=0
	if command -v "$(VLLM_BIN)" >/dev/null 2>&1 || test -x "$(VLLM_BIN)"; then echo "[RUNTIME] vLLM: $(VLLM_BIN)"; "$(VLLM_BIN)" --help >/dev/null; else echo "[RUNTIME] vLLM ausente: $(VLLM_BIN)"; missing=1; fi
	if test -x "$(LLAMA_SERVER_BIN)"; then echo "[RUNTIME] llama-server: $(LLAMA_SERVER_BIN)"; "$(LLAMA_SERVER_BIN)" --help >/dev/null; else echo "[RUNTIME] llama-server ausente: $(LLAMA_SERVER_BIN)"; missing=1; fi
	if command -v "$(OLLAMA_BIN)" >/dev/null 2>&1 || test -x "$(OLLAMA_BIN)"; then
		echo "[RUNTIME] Ollama: $(OLLAMA_BIN)"
		# `ollama --version` pode retornar 1 quando o daemon ainda não está ativo;
		# a checagem deste alvo é do executável, não da API.
		"$(OLLAMA_BIN)" --version 2>&1 || true
	else
		echo "[RUNTIME] Ollama ausente: $(OLLAMA_BIN)"
		missing=1
	fi
	if test "$$missing" -ne 0; then echo '[RUNTIME] instale o runtime ausente no ambiente próprio ou sobrescreva VLLM_BIN/LLAMA_SERVER_BIN/OLLAMA_BIN'; exit 1; fi

check-vllm-gguf:
	@set -eu; \
	bin="$$(command -v "$(VLLM_BIN)" 2>/dev/null || printf '%s' "$(VLLM_BIN)")"; \
	test -x "$$bin" || { echo "vLLM ausente ou não executável: $$bin" >&2; exit 1; }; \
	py="$(VLLM_PYTHON)"; \
	if test -z "$$py"; then \
		shebang="$$(head -1 "$$bin")"; \
		case "$$shebang" in '#!'*python*) py="$${shebang#\#!}"; py="$${py%% *}" ;; \
		*) echo "O executável vLLM é um wrapper ou não declara Python diretamente; informe VLLM_PYTHON=/caminho/do/python-do-vllm." >&2; exit 1 ;; \
		esac; \
	fi; \
	test -x "$$py" || { echo "Python do vLLM ausente ou não executável: $$py" >&2; exit 1; }; \
	"$$py" -c 'import vllm; import vllm_gguf_plugin; print("[PREPARE vLLM] vLLM e vllm-gguf-plugin carregados no mesmo ambiente")' || { \
		echo "vllm-gguf-plugin ausente no ambiente do executável $(VLLM_BIN)." >&2; \
		echo "Instale nesse ambiente (sem misturar com o venv do benchmark): $$py -m pip install vllm-gguf-plugin" >&2; \
		exit 1; \
	}

prepare-vllm: prepare-benchmark check-vllm-gguf
	@command -v "$(VLLM_BIN)" >/dev/null 2>&1 || test -x "$(VLLM_BIN)" || { echo "vLLM ausente: $(VLLM_BIN). Verifique o PATH ou use VLLM_BIN=/caminho/para/vllm"; exit 1; }
	"$(VLLM_BIN)" --help >/dev/null
	@echo '[PREPARE vLLM] cliente, modelo, tokenizer e executável validados'

prepare-llama: prepare-benchmark
	@test -x "$(LLAMA_SERVER_BIN)" || { echo "llama-server ausente: $(LLAMA_SERVER_BIN). Compile o llama.cpp ou use LLAMA_SERVER_BIN=..."; exit 1; }
	backend='$(LLAMA_BACKEND_PATH)'; if test -n "$$backend"; then export GGML_BACKEND_PATH="$$backend"; export LD_LIBRARY_PATH="$$(dirname "$$backend"):$${LD_LIBRARY_PATH:-}"; fi
	"$(LLAMA_SERVER_BIN)" --help >/dev/null
	devices="$$("$(LLAMA_SERVER_BIN)" --list-devices 2>&1)"; printf '%s\n' "$$devices"
	if test "$(REQUIRE_GPU)" = 1 && ! printf '%s\n' "$$devices" | grep -qi cuda; then echo 'llama-server não detectou dispositivo CUDA; informe LLAMA_BACKEND_PATH ou use REQUIRE_GPU=0 conscientemente.' >&2; exit 1; fi
	@echo '[PREPARE llama.cpp] cliente, modelo, tokenizer e executável validados'

prepare-ollama: prepare-benchmark
	@echo '[PREPARE Ollama] criando $(OLLAMA_MODEL_NAME) a partir do GGUF local; nenhum download será feito'
	"$(PYTHON)" scripts/prepare_ollama.py --binary "$(OLLAMA_BIN)" --model "$(OLLAMA_MODEL_NAME)" \
		--gguf "$(GGUF_FILE)" --context 16384 --base-url "$$("$(PYTHON)" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_url"])' "$(OLLAMA_CONFIG)")"
	@echo '[PREPARE Ollama] alias $(OLLAMA_MODEL_NAME) disponível'

prepare-all: prepare-vllm prepare-llama prepare-ollama
	@echo '[PREPARE ALL] vLLM, llama.cpp e Ollama preparados'

upload-hf: verify-gguf
	@test -n "$${HF_TOKEN:-}" || { echo 'HF_TOKEN não está definido; exporte-o sem colar o valor no repositório.' >&2; exit 1; }
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) repo create "$(HF_REPO_ID)" --repo-type model --exist-ok
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) upload "$(HF_REPO_ID)" "$(GGUF_FILE)" "$(HF_UPLOAD_FILENAME)" \
			--repo-type model --commit-message "Add llama.cpp Q8_0 GGUF"

smoke-vllm: prepare-vllm
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run \
		--config "$(VLLM_CONFIG)" \
		--local-model-path "$(VLLM_MODEL_DIR)" \
		--launch "$(VLLM_LAUNCH)" \
		--launch-executable "$(VLLM_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))))')" \
		--smoke --scenarios short --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"

smoke-llama: prepare-llama
	backend='$(LLAMA_BACKEND_PATH)'; if test -n "$$backend"; then export GGML_BACKEND_PATH="$$backend"; export LD_LIBRARY_PATH="$$(dirname "$$backend"):$${LD_LIBRARY_PATH:-}"; fi
	PATH="$(dir $(LLAMA_SERVER_BIN)):$${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run --config "$(LLAMA_CONFIG)" --local-model-path "$(LLAMA_MODEL_DIR)" \
		--launch "$(LLAMA_LAUNCH)" --launch-executable "$(LLAMA_SERVER_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))))')" \
		--smoke --scenarios short --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"

smoke-ollama: prepare-ollama
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run --config "$(OLLAMA_CONFIG)" --local-model-path "$(OLLAMA_MODEL_DIR)" \
		--launch "$(OLLAMA_LAUNCH)" --launch-executable "$(OLLAMA_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("OLLAMA_EXTRA_ARGS", ""))))')" \
		--smoke --scenarios short --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"

smoke-all:
	@set +e
	@echo '[SMOKE ALL] 1/3 vLLM'; $(MAKE) --no-print-directory smoke-vllm; vllm_status=$$?
	@echo '[SMOKE ALL] 2/3 llama.cpp'; $(MAKE) --no-print-directory smoke-llama; llama_status=$$?
	@echo '[SMOKE ALL] 3/3 Ollama'; $(MAKE) --no-print-directory smoke-ollama; ollama_status=$$?
	@echo "[SMOKE ALL] status: vLLM=$$vllm_status llama.cpp=$$llama_status Ollama=$$ollama_status"
	@test "$$vllm_status" -eq 0 -a "$$llama_status" -eq 0 -a "$$ollama_status" -eq 0

bench-vllm: prepare-vllm
	@echo '[BENCH vLLM] início: short/medium/long + telemetria + KV sweep embutido'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run \
		--config "$(VLLM_CONFIG)" \
		--local-model-path "$(VLLM_MODEL_DIR)" \
		--launch "$(VLLM_LAUNCH)" \
		--launch-executable "$(VLLM_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))))')" \
		--scenarios $(BENCH_SCENARIOS) \
		--requests "$(BENCH_REQUESTS)" \
		--repetitions "$(BENCH_REPETITIONS)" \
		--warmup "$(BENCH_WARMUP)" \
		--mode "$(BENCH_MODE)" --conversation-turns "$(CONVERSATION_TURNS)" \
		--conversation-fixture "$(CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--collect-kv-metrics \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)"
	@echo '[BENCH vLLM] bateria base concluída; iniciando sweep KV em passos de $(SWEEP_MEMORY_STEP_MB) MB'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label vllm \
		--config "$(VLLM_CONFIG)" --launch "$(VLLM_LAUNCH)" \
		--local-model-path "$(VLLM_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-executable "$(VLLM_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))))')" \
		--start "$(SWEEP_START)" --memory-step-mb "$(SWEEP_MEMORY_STEP_MB)" --max-context "$(SWEEP_MAX_CONTEXT)" \
		--kv-bytes-per-token "$(KV_BYTES_PER_TOKEN)" \
		--requests "$(SWEEP_REQUESTS)" --repetitions "$(SWEEP_REPETITIONS)" \
		--warmup "$(SWEEP_WARMUP)" --mode "$(SWEEP_MODE)" --conversation-turns "$(SWEEP_CONVERSATION_TURNS)" \
		--conversation-fixture "$(SWEEP_CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		--results results

bench-llama: prepare-llama
	@echo '[BENCH llama.cpp] início: short/medium/long + telemetria + KV sweep embutido'
	backend='$(LLAMA_BACKEND_PATH)'; if test -n "$$backend"; then export GGML_BACKEND_PATH="$$backend"; export LD_LIBRARY_PATH="$$(dirname "$$backend"):$${LD_LIBRARY_PATH:-}"; fi
	PATH="$(dir $(LLAMA_SERVER_BIN)):$${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run --config "$(LLAMA_CONFIG)" \
		--local-model-path "$(LLAMA_MODEL_DIR)" --launch "$(LLAMA_LAUNCH)" \
		--launch-executable "$(LLAMA_SERVER_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))))')" \
		--scenarios $(BENCH_SCENARIOS) --requests "$(BENCH_REQUESTS)" \
		--repetitions "$(BENCH_REPETITIONS)" --warmup "$(BENCH_WARMUP)" \
		--mode "$(BENCH_MODE)" --conversation-turns "$(CONVERSATION_TURNS)" \
		--conversation-fixture "$(CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--collect-kv-metrics --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"
	@echo '[BENCH llama.cpp] bateria base concluída; iniciando sweep KV em passos de $(SWEEP_MEMORY_STEP_MB) MB'
	backend='$(LLAMA_BACKEND_PATH)'; if test -n "$$backend"; then export GGML_BACKEND_PATH="$$backend"; export LD_LIBRARY_PATH="$$(dirname "$$backend"):$${LD_LIBRARY_PATH:-}"; fi
	PATH="$(dir $(LLAMA_SERVER_BIN)):$${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label llama.cpp \
		--config "$(LLAMA_CONFIG)" --launch "$(LLAMA_LAUNCH)" \
		--local-model-path "$(LLAMA_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-executable "$(LLAMA_SERVER_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))))')" \
		--context-flag=--ctx-size \
		--start "$(SWEEP_START)" --memory-step-mb "$(SWEEP_MEMORY_STEP_MB)" --max-context "$(SWEEP_MAX_CONTEXT)" \
		--kv-bytes-per-token "$(KV_BYTES_PER_TOKEN)" \
		--requests "$(SWEEP_REQUESTS)" --repetitions "$(SWEEP_REPETITIONS)" \
		--warmup "$(SWEEP_WARMUP)" --mode "$(SWEEP_MODE)" --conversation-turns "$(SWEEP_CONVERSATION_TURNS)" \
		--conversation-fixture "$(SWEEP_CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		--results results

bench-ollama: prepare-ollama
	@echo '[BENCH Ollama] início: short/medium/long + telemetria + KV sweep embutido'
	@echo '[BENCH Ollama] pré-condição: o alias qwen7b-q8-gguf já deve existir em ollama list; nenhum pull será feito'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run --config "$(OLLAMA_CONFIG)" \
		--local-model-path "$(OLLAMA_MODEL_DIR)" --launch "$(OLLAMA_LAUNCH)" \
		--launch-executable "$(OLLAMA_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("OLLAMA_EXTRA_ARGS", ""))))')" \
		--scenarios $(BENCH_SCENARIOS) --requests "$(BENCH_REQUESTS)" \
		--repetitions "$(BENCH_REPETITIONS)" --warmup "$(BENCH_WARMUP)" \
		--mode "$(BENCH_MODE)" --conversation-turns "$(CONVERSATION_TURNS)" \
		--conversation-fixture "$(CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--collect-kv-metrics --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"
	@echo '[BENCH Ollama] bateria base concluída; iniciando sweep KV em passos de $(SWEEP_MEMORY_STEP_MB) MB'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label ollama \
		--config "$(OLLAMA_CONFIG)" --launch "$(OLLAMA_LAUNCH)" \
		--local-model-path "$(OLLAMA_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-executable "$(OLLAMA_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("OLLAMA_EXTRA_ARGS", ""))))')" \
		--context-flag none \
		--start "$(SWEEP_START)" --memory-step-mb "$(SWEEP_MEMORY_STEP_MB)" --max-context "$(SWEEP_MAX_CONTEXT)" \
		--kv-bytes-per-token "$(KV_BYTES_PER_TOKEN)" \
		--requests "$(SWEEP_REQUESTS)" --repetitions "$(SWEEP_REPETITIONS)" \
		--warmup "$(SWEEP_WARMUP)" --mode "$(SWEEP_MODE)" --conversation-turns "$(SWEEP_CONVERSATION_TURNS)" \
		--conversation-fixture "$(SWEEP_CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		--results results

bench-all:
	@set +e
	@echo '[BENCH ALL] 1/3 vLLM'
	@$(MAKE) --no-print-directory bench-vllm; vllm_status=$$?
	@echo '[BENCH ALL] 2/3 llama.cpp'
	@$(MAKE) --no-print-directory bench-llama; llama_status=$$?
	@echo '[BENCH ALL] 3/3 Ollama'
	@$(MAKE) --no-print-directory bench-ollama; ollama_status=$$?
	@echo "[BENCH ALL] status: vLLM=$$vllm_status llama.cpp=$$llama_status Ollama=$$ollama_status"
	@test "$$vllm_status" -eq 0 -a "$$llama_status" -eq 0 -a "$$ollama_status" -eq 0

bench: bench-all

kv-sweep: kv-sweep-vllm

kv-sweep-vllm: prepare-vllm
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label vllm \
		--config "$(VLLM_CONFIG)" --launch "$(VLLM_LAUNCH)" \
		--local-model-path "$(VLLM_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-executable "$(VLLM_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))))')" \
		--start "$(SWEEP_START)" --memory-step-mb "$(SWEEP_MEMORY_STEP_MB)" --max-context "$(SWEEP_MAX_CONTEXT)" \
		--kv-bytes-per-token "$(KV_BYTES_PER_TOKEN)" \
		--requests "$(SWEEP_REQUESTS)" --repetitions "$(SWEEP_REPETITIONS)" \
		--warmup "$(SWEEP_WARMUP)" --mode "$(SWEEP_MODE)" --conversation-turns "$(SWEEP_CONVERSATION_TURNS)" \
		--conversation-fixture "$(SWEEP_CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)" --results results

kv-sweep-llama: prepare-llama
	backend='$(LLAMA_BACKEND_PATH)'; if test -n "$$backend"; then export GGML_BACKEND_PATH="$$backend"; export LD_LIBRARY_PATH="$$(dirname "$$backend"):$${LD_LIBRARY_PATH:-}"; fi
	PATH="$(dir $(LLAMA_SERVER_BIN)):$${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label llama.cpp --config "$(LLAMA_CONFIG)" --launch "$(LLAMA_LAUNCH)" \
		--local-model-path "$(LLAMA_MODEL_DIR)" --python "$(PYTHON)" --launch-executable "$(LLAMA_SERVER_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))))')" --context-flag=--ctx-size \
		--start "$(SWEEP_START)" --memory-step-mb "$(SWEEP_MEMORY_STEP_MB)" --max-context "$(SWEEP_MAX_CONTEXT)" \
		--kv-bytes-per-token "$(KV_BYTES_PER_TOKEN)" \
		--requests "$(SWEEP_REQUESTS)" --repetitions "$(SWEEP_REPETITIONS)" \
		--warmup "$(SWEEP_WARMUP)" --mode "$(SWEEP_MODE)" --conversation-turns "$(SWEEP_CONVERSATION_TURNS)" \
		--conversation-fixture "$(SWEEP_CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)" --results results

kv-sweep-ollama: prepare-ollama
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label ollama --config "$(OLLAMA_CONFIG)" --launch "$(OLLAMA_LAUNCH)" \
		--local-model-path "$(OLLAMA_MODEL_DIR)" --python "$(PYTHON)" --launch-executable "$(OLLAMA_BIN)" \
		--launch-extra-args-json "$$($(SYSTEM_PYTHON) -c 'import json,os,shlex; print(json.dumps(shlex.split(os.environ.get("OLLAMA_EXTRA_ARGS", ""))))')" --context-flag none \
		--start "$(SWEEP_START)" --memory-step-mb "$(SWEEP_MEMORY_STEP_MB)" --max-context "$(SWEEP_MAX_CONTEXT)" \
		--kv-bytes-per-token "$(KV_BYTES_PER_TOKEN)" \
		--requests "$(SWEEP_REQUESTS)" --repetitions "$(SWEEP_REPETITIONS)" \
		--warmup "$(SWEEP_WARMUP)" --mode "$(SWEEP_MODE)" --conversation-turns "$(SWEEP_CONVERSATION_TURNS)" \
		--conversation-fixture "$(SWEEP_CONVERSATION_FIXTURE)" --warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)" --results results

quick-sweep-vllm: SWEEP_REQUESTS=1
quick-sweep-vllm: SWEEP_REPETITIONS=1
quick-sweep-vllm: SWEEP_WARMUP=0
quick-sweep-vllm: SWEEP_START=1024
quick-sweep-vllm: SWEEP_MEMORY_STEP_MB=256
quick-sweep-vllm: SWEEP_MAX_CONTEXT=8192
quick-sweep-vllm: kv-sweep-vllm
quick-sweep-llama: SWEEP_REQUESTS=1
quick-sweep-llama: SWEEP_REPETITIONS=1
quick-sweep-llama: SWEEP_WARMUP=0
quick-sweep-llama: SWEEP_START=1024
quick-sweep-llama: SWEEP_MEMORY_STEP_MB=256
quick-sweep-llama: SWEEP_MAX_CONTEXT=8192
quick-sweep-llama: kv-sweep-llama
quick-sweep-ollama: SWEEP_REQUESTS=1
quick-sweep-ollama: SWEEP_REPETITIONS=1
quick-sweep-ollama: SWEEP_WARMUP=0
quick-sweep-ollama: SWEEP_START=1024
quick-sweep-ollama: SWEEP_MEMORY_STEP_MB=256
quick-sweep-ollama: SWEEP_MAX_CONTEXT=8192
quick-sweep-ollama: kv-sweep-ollama

pull-results:
	@test -n "$(POD_SSH)" || { echo 'Informe POD_SSH, por exemplo: POD_SSH=root@pod-host'; exit 1; }
	@test -f "$(POD_SSH_KEY)" || { echo "Chave SSH ausente: $(POD_SSH_KEY). Use POD_SSH_KEY=/caminho/chave"; exit 1; }
	mkdir -p "$(LOCAL_RESULTS_DIR)"
	echo "[DOWNLOAD] $(POD_SSH):$(REMOTE_RESULTS_DIR) -> $(LOCAL_RESULTS_DIR)"
	if [[ "$(POD_SSH)" == *ssh.runpod.io* ]]; then
		command -v runpodctl >/dev/null || { echo 'runpodctl ausente; instale com: brew install runpod/runpodctl/runpodctl'; exit 1; }
		tmp_dir=$$(mktemp -d)
		log_file="$$tmp_dir/send.log"
		archive_name="llm-local-results-$${BASHPID}.tar.gz"
		archive_path="/tmp/$$archive_name"
		remote_parent=$$(dirname "$(REMOTE_RESULTS_DIR)")
		remote_base=$$(basename "$(REMOTE_RESULTS_DIR)")
		sender_pid=''
		cleanup() { test -z "$$sender_pid" || kill "$$sender_pid" 2>/dev/null || true; rm -rf "$$tmp_dir"; }
		trap cleanup EXIT INT TERM
		ssh -i "$(POD_SSH_KEY)" -p "$(POD_PORT)" "$(POD_SSH)" \
			"tar -czf '$$archive_path' -C '$$remote_parent' '$$remote_base' && runpodctl send '$$archive_path'" \
			>"$$log_file" 2>&1 &
		sender_pid=$$!
		code=''
		for attempt in $$(seq 1 120); do
			code=$$(sed -nE 's/.*[Cc]ode is: ([[:alnum:]-]+).*/\1/p' "$$log_file" | head -n 1)
			if test -n "$$code"; then break; fi
			if ! kill -0 "$$sender_pid" 2>/dev/null; then
				tail -n 30 "$$log_file"; echo '[DOWNLOAD] o envio remoto terminou sem código'; exit 1
			fi
			sleep 1
		done
		test -n "$$code" || { cat "$$log_file"; echo '[DOWNLOAD] timeout esperando código runpodctl'; exit 1; }
		echo "[DOWNLOAD] recebendo via runpodctl (código $$code)"
		( cd "$$tmp_dir" && runpodctl receive "$$code" )
		tar -xzf "$$tmp_dir/$$archive_name" -C "$(LOCAL_RESULTS_DIR)" --strip-components=1
		wait "$$sender_pid"
	else
		command -v scp >/dev/null || { echo 'scp ausente no computador local'; exit 1; }
		scp -i "$(POD_SSH_KEY)" -r -P "$(POD_PORT)" "$(POD_SSH):$(REMOTE_RESULTS_DIR)/." "$(LOCAL_RESULTS_DIR)/"
	fi
	echo "[DOWNLOAD] concluído; arquivos em $(LOCAL_RESULTS_DIR)"

check-profilers:
	@command -v nsys >/dev/null 2>&1 && nsys --version || echo '[PROFILE] nsys não encontrado: instale Nsight Systems no pod ou use o pacote NVIDIA correspondente.'
	@command -v ncu >/dev/null 2>&1 && ncu --version || echo '[PROFILE] ncu não encontrado: instale Nsight Compute no pod ou use o pacote NVIDIA correspondente.'


docs:
	$(PYTHON) scripts/build_docs.py

clean-info:
	@printf 'Não há limpeza automática: pesos, GGUFs e resultados são evidências experimentais.\n'
