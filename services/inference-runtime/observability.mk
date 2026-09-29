# Fluxos interativo e headless de observabilidade.
# Incluído pelo Makefile principal; o benchmark continua sequencial.

OBS_RUNTIME ?= vllm
OBS_SYSTEM_PYTHON ?= python3
OBS_HOST ?= 127.0.0.1
OBS_CONTEXT ?= 8192
OBS_MODEL_ALIAS ?= qwen2.5-7b-test
OBS_HF_HOME ?= /workspace/huggingface
OBS_MODELS_DIR ?= /workspace/models
OBS_VENVS_DIR ?= /workspace/venvs
OBS_JOBS ?= $(shell nproc 2>/dev/null || echo 1)
OBS_CUDA_ARCH ?= 86

OBS_MODEL_SOURCE ?= $(if $(filter vllm,$(OBS_RUNTIME)),hf,gguf)
OBS_MODEL ?= $(if $(filter hf,$(OBS_MODEL_SOURCE)),$(OBS_VLLM_MODEL),$(if $(filter gguf,$(OBS_MODEL_SOURCE)),$(OBS_GGUF_REPO),))
OBS_REVISION ?= main
OBS_TOKENIZER_REVISION ?= main
OBS_PREPARED_MODELS_DIR ?= $(OBS_MODELS_DIR)/observability/$(OBS_MODEL_ALIAS)
OBS_STATE_DIR ?= $(CURDIR)/observability/.state/headless
OBS_MODEL_METADATA ?= $(OBS_STATE_DIR)/model-metadata.json
OBS_GENERATED_CONFIG ?= $(OBS_STATE_DIR)/runtime-config.json
OBS_GENERATED_LAUNCH ?= $(OBS_STATE_DIR)/runtime-launch.json
OBS_RESULTS_DIR ?= results
OBS_RESULT_NAME ?=
OBS_BENCH_SCENARIOS ?= short medium long
OBS_BENCH_INPUT_TOKENS ?=
OBS_BENCH_REQUESTS ?= 20
OBS_BENCH_REPETITIONS ?= 1
OBS_BENCH_WARMUP ?= 3
OBS_BENCH_MODE ?= replay
OBS_BENCH_PROFILE ?= generic
OBS_BENCH_CONVERSATION_TURNS ?= 1
OBS_BENCH_SMOKE ?= 0
OBS_REQUEST_DATASET ?=
OBS_WARMUP_REQUEST_DATASET ?=
OBS_REQUEST_MANIFEST ?=
OBS_REPLAY_RESPONSES ?=
OBS_RESPONSES_OUT ?=
OBS_REMOTE_HOST ?=
OBS_REMOTE_PORT ?=
OBS_REMOTE_KEY ?=
OBS_REMOTE_RESULTS_DIR ?= /workspace/finance-agent-platform/services/inference-runtime/results
OBS_LOCAL_RESULTS_DIR ?= results-from-pod

OBS_VLLM_FORMAT ?= hf
OBS_VLLM_MODEL ?= Qwen/Qwen2.5-7B-Instruct
OBS_VLLM_VENV ?= $(OBS_VENVS_DIR)/vllm
OBS_VLLM_PYTHON ?= $(OBS_VLLM_VENV)/bin/python
OBS_VLLM_BIN ?= $(OBS_VLLM_VENV)/bin/vllm
OBS_VLLM_ENV_MODE ?= managed
OBS_VLLM_PACKAGE ?= vllm==0.29.0
# Wheel x86-64 publicado no PyPI para vLLM 0.29.0, fixado também pelo SHA-256.
OBS_VLLM_WHEEL ?= https://files.pythonhosted.org/packages/ca/09/7f79450e21bd1c2a0897ab946a544816a4f7e04c54f0ca49849b14b12d6a/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl\#sha256=09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635
OBS_VLLM_TORCH_PACKAGE ?= torch==2.13.0
OBS_VLLM_TORCH_BACKEND ?= cu130
OBS_VLLM_EXPECTED_VERSION ?= 0.29.0
OBS_VLLM_EXPECTED_TORCH_VERSION ?= 2.13.0+cu130
OBS_VLLM_EXPECTED_CUDA_VERSION ?= 13.0
OBS_VLLM_CUDA_HOME ?= /usr/local/cuda-13.0
OBS_VLLM_CUDA_TOOLKIT_PACKAGE ?= cuda-toolkit-13-0
OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT ?= 1
OBS_VLLM_ENV_MANIFEST ?= $(OBS_STATE_DIR)/vllm-environment.json
OBS_VLLM_GGUF_PLUGIN_DIR ?= /workspace/runtimes/vllm-gguf-plugin
OBS_VLLM_GGUF_PLUGIN_REVISION ?= e2b8ad532b8b5ea175100202c30430c1d2b5e6a8
OBS_VLLM_PORT ?= 8000
OBS_VLLM_DTYPE ?= auto
OBS_VLLM_GPU_MEMORY_UTILIZATION ?= 0.90
OBS_VLLM_MAX_NUM_SEQS ?= 1
OBS_VLLM_EXTRA_ARGS ?=
OBS_VLLM_CUDA_RUNTIME_LIB ?=
OBS_VLLM_NEEDS_COMPILER = $(if $(and $(filter managed,$(OBS_VLLM_ENV_MODE)),$(filter gguf local-gguf,$(OBS_VLLM_FORMAT) $(OBS_MODEL_SOURCE))),1,)

# O plugin GGUF contém extensão nativa. O nvcc usado no build precisa pertencer
# à mesma família CUDA do PyTorch fixado no lock do vLLM.
define OBS_VLLM_ENSURE_CUDA_TOOLKIT
case "$(OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT)" in 0|1) ;; *) echo 'OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT deve ser 0 ou 1.' >&2; exit 2 ;; esac
vllm_cuda_home="$(OBS_VLLM_CUDA_HOME)"
if ! test -x "$$vllm_cuda_home/bin/nvcc"; then \
	if test "$(OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT)" != 1; then \
		echo "nvcc ausente em $$vllm_cuda_home; instale $(OBS_VLLM_CUDA_TOOLKIT_PACKAGE) ou ajuste OBS_VLLM_CUDA_HOME." >&2; exit 1; \
	fi; \
	command -v apt-get >/dev/null || { echo 'A instalação automática do toolkit exige Ubuntu/Debian; instale-o na imagem ou use OBS_VLLM_CUDA_HOME.' >&2; exit 1; }; \
	test "$$(id -u)" -eq 0 || { echo 'A instalação automática do toolkit exige root; instale-o antes ou use OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT=0.' >&2; exit 1; }; \
	echo '[OBSERVE] instalando $(OBS_VLLM_CUDA_TOOLKIT_PACKAGE) para compilar o plugin GGUF'; \
	DEBIAN_FRONTEND=noninteractive apt-get install -y "$(OBS_VLLM_CUDA_TOOLKIT_PACKAGE)"; \
fi
export CUDA_HOME="$$vllm_cuda_home"
export CUDACXX="$$vllm_cuda_home/bin/nvcc"
export PATH="$$vllm_cuda_home/bin:$$PATH"
export LD_LIBRARY_PATH="$$vllm_cuda_home/lib64:$${LD_LIBRARY_PATH:-}"
endef

# No modo managed, o loader precisa enxergar o libcudart privado do lock CUDA.
# No modo existing, o ambiente da imagem é preservado; só um caminho informado
# explicitamente pelo operador é adicionado ao LD_LIBRARY_PATH.
define OBS_VLLM_EXPORT_CUDA_RUNTIME
vllm_site_packages=$$("$(OBS_VLLM_PYTHON)" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
vllm_cuda_runtime_lib="$(OBS_VLLM_CUDA_RUNTIME_LIB)"
if test -n "$$vllm_cuda_runtime_lib"; then \
	test -d "$$vllm_cuda_runtime_lib" || { echo "OBS_VLLM_CUDA_RUNTIME_LIB inexistente: $$vllm_cuda_runtime_lib" >&2; exit 1; }; \
elif test "$(OBS_VLLM_ENV_MODE)" = managed && test -f "$$vllm_site_packages/nvidia/cu13/lib/libcudart.so.13"; then \
	vllm_cuda_runtime_lib="$$vllm_site_packages/nvidia/cu13/lib"; \
fi
if test -n "$$vllm_cuda_runtime_lib"; then \
	export VLLM_CUDA_RUNTIME_LIB="$$vllm_cuda_runtime_lib"; \
	export LD_LIBRARY_PATH="$$vllm_cuda_runtime_lib:$${LD_LIBRARY_PATH:-}"; \
fi
endef

define OBS_VLLM_RUN_PREFLIGHT
"$(OBS_VLLM_PYTHON)" scripts/vllm_preflight.py \
	--mode "$(OBS_VLLM_ENV_MODE)" --output "$(OBS_VLLM_ENV_MANIFEST)" \
	--vllm-package "$(OBS_VLLM_PACKAGE)" --vllm-wheel "$(OBS_VLLM_WHEEL)" --torch-package "$(OBS_VLLM_TORCH_PACKAGE)" \
	--torch-backend "$(OBS_VLLM_TORCH_BACKEND)" --expected-vllm-version "$(OBS_VLLM_EXPECTED_VERSION)" \
	--expected-torch-version "$(OBS_VLLM_EXPECTED_TORCH_VERSION)" --expected-cuda-version "$(OBS_VLLM_EXPECTED_CUDA_VERSION)" \
	--plugin-revision "$(OBS_VLLM_GGUF_PLUGIN_REVISION)" --plugin-state expected \
	--cuda-home "$(OBS_VLLM_CUDA_HOME)" $(if $(OBS_VLLM_NEEDS_COMPILER),--require-compiler,)
endef

define OBS_VLLM_RUN_DRIVER_PREFLIGHT
"$(OBS_SYSTEM_PYTHON)" scripts/vllm_preflight.py \
	--mode managed --output "$(OBS_VLLM_ENV_MANIFEST)" \
	--expected-cuda-version "$(OBS_VLLM_EXPECTED_CUDA_VERSION)" --driver-only
endef

OBS_GGUF_REPO ?= arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf
OBS_GGUF_REVISION ?= main
OBS_GGUF_FILENAME ?= Qwen2.5-7B-Instruct-Q8_0.gguf
OBS_GGUF_MODEL ?= $(OBS_MODELS_DIR)/Qwen2.5-7B-Instruct-Q8_0/$(OBS_GGUF_FILENAME)
OBS_TOKENIZER_MODEL ?= Qwen/Qwen2.5-7B-Instruct
OBS_TOOLS_VENV ?= $(OBS_VENVS_DIR)/observability-tools
OBS_TOOLS_PYTHON ?= $(OBS_TOOLS_VENV)/bin/python

OBS_LLAMA_DIR ?= /workspace/runtimes/llama.cpp
OBS_LLAMA_BUILD_DIR ?= $(OBS_LLAMA_DIR)/build
OBS_LLAMA_BIN ?= $(OBS_LLAMA_BUILD_DIR)/bin/llama-server
OBS_LLAMA_REVISION ?= ed7ac35e1ee49cb70e4dfa9f0a2ce39b0a5ec4ea
OBS_LLAMA_PORT ?= 8080
OBS_LLAMA_GPU_LAYERS ?= 999
OBS_LLAMA_EXTRA_ARGS ?=

OBS_OLLAMA_BIN ?= $(shell command -v ollama 2>/dev/null || printf 'ollama')
OBS_OLLAMA_VERSION ?= 0.34.0
OBS_OLLAMA_PORT ?= 11434
OBS_OLLAMA_MODELS ?= /workspace/ollama-models
OBS_OLLAMA_KEEP_ALIVE ?= -1
OBS_OLLAMA_KV_CACHE_TYPE ?= f16
OBS_OLLAMA_FLASH_ATTENTION ?= 1
OBS_OLLAMA_NUM_PARALLEL ?= 1
OBS_OLLAMA_MAX_LOADED_MODELS ?= 1
OBS_OLLAMA_EXTRA_ARGS ?=
export OBS_VLLM_EXTRA_ARGS OBS_LLAMA_EXTRA_ARGS OBS_OLLAMA_EXTRA_ARGS

OBS_JUPYTER_PORT ?= 8889
OBS_PROMETHEUS_URL ?= http://127.0.0.1:9090
OBS_RUNTIME_PORT = $(if $(filter vllm,$(OBS_RUNTIME)),$(OBS_VLLM_PORT),$(if $(filter llama,$(OBS_RUNTIME)),$(OBS_LLAMA_PORT),$(if $(filter ollama,$(OBS_RUNTIME)),$(OBS_OLLAMA_PORT),)))
OBS_BASE_URL = http://$(OBS_HOST):$(OBS_RUNTIME_PORT)
OBS_RUNTIME_BIN = $(if $(filter vllm,$(OBS_RUNTIME)),$(OBS_VLLM_BIN),$(if $(filter llama,$(OBS_RUNTIME)),$(OBS_LLAMA_BIN),$(OBS_OLLAMA_BIN)))
OBS_RUNTIME_EXTRA_ARGS = $(if $(filter vllm,$(OBS_RUNTIME)),$(OBS_VLLM_EXTRA_ARGS),$(if $(filter llama,$(OBS_RUNTIME)),$(OBS_LLAMA_EXTRA_ARGS),$(OBS_OLLAMA_EXTRA_ARGS)))
OBS_OLLAMA_ENV = OLLAMA_MODELS="$(OBS_OLLAMA_MODELS)" OLLAMA_KEEP_ALIVE="$(OBS_OLLAMA_KEEP_ALIVE)" OLLAMA_KV_CACHE_TYPE="$(OBS_OLLAMA_KV_CACHE_TYPE)" OLLAMA_FLASH_ATTENTION="$(OBS_OLLAMA_FLASH_ATTENTION)" OLLAMA_NUM_PARALLEL="$(OBS_OLLAMA_NUM_PARALLEL)" OLLAMA_MAX_LOADED_MODELS="$(OBS_OLLAMA_MAX_LOADED_MODELS)"

.PHONY: observe-help observe-config observe-check-runtime observe-check-model-source observe-system-install observe-tools-install \
	observe-download-gguf observe-install observe-install-vllm observe-vllm-preflight observe-install-llama observe-install-ollama \
	observe-prepare observe-prepare-vllm observe-prepare-llama observe-prepare-ollama observe-bootstrap \
	observe-serve observe-serve-vllm observe-serve-llama observe-serve-ollama \
	observe-prometheus-start observe-prometheus-stop observe-status \
	observe-notebook-install observe-jupyter observe-model-prepare observe-render-runtime \
	observe-prepare-headless-ollama observe-bench-prepare observe-bench-validate observe-bench observe-bench-all pull-observe-results

observe-help:
	@printf '%s\n' \
		'Observabilidade de inferência no Pod' \
		'' \
		'  make observe-bootstrap OBS_RUNTIME=vllm       prepara Pod, runtime, Prometheus e notebook' \
		'  make observe-serve OBS_RUNTIME=vllm           inicia o runtime em foreground' \
		'  make observe-prometheus-start OBS_RUNTIME=vllm inicia a coleta' \
		'  make observe-jupyter OBS_RUNTIME=vllm          inicia o notebook no Pod' \
		'  make observe-config OBS_RUNTIME=vllm           mostra a configuração efetiva' \
		'  make observe-vllm-preflight                     valida driver, CUDA e UVA em segundos' \
		'  make observe-status OBS_RUNTIME=vllm           verifica APIs' \
		'  make observe-prometheus-stop                   encerra a coleta' \
		'  make observe-bench OBS_RUNTIME=vllm            roda bateria CLI + Prometheus' \
		'  make observe-bench-all OBS_MODEL_SOURCE=gguf    smoke e bateria completa nos 3 runtimes' \
		'  make pull-observe-results OBS_REMOTE_HOST=alias baixa e verifica resultados' \
		'' \
		'Padrão RTX 3090: Qwen2.5-7B HF, contexto 8192, uma sequência, 90% da VRAM.' \
		'Para comparar runtimes, use o mesmo GGUF:' \
		'  make observe-bootstrap OBS_RUNTIME=vllm OBS_VLLM_FORMAT=gguf' \
		'  make observe-bootstrap OBS_RUNTIME=llama' \
		'  make observe-bootstrap OBS_RUNTIME=ollama' \
		'' \
		'Variáveis principais: OBS_RUNTIME, OBS_CONTEXT, OBS_MODEL_ALIAS, OBS_JUPYTER_PORT,' \
		'OBS_VLLM_MODEL, OBS_VLLM_FORMAT=hf|gguf, OBS_VLLM_ENV_MODE=managed|existing,' \
		'OBS_VLLM_WHEEL, OBS_VLLM_TORCH_PACKAGE, OBS_VLLM_TORCH_BACKEND, OBS_VLLM_CUDA_HOME,' \
		'OBS_VLLM_CUDA_TOOLKIT_PACKAGE, OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT, OBS_VLLM_GPU_MEMORY_UTILIZATION,' \
		'OBS_GGUF_REPO, OBS_GGUF_FILENAME, OBS_GGUF_MODEL e *_EXTRA_ARGS.' \
		'' \
		'Headless: OBS_MODEL_SOURCE=hf|local-hf|gguf|local-gguf, OBS_MODEL, OBS_REVISION,' \
		'OBS_TOKENIZER_MODEL, OBS_BENCH_REQUESTS, OBS_BENCH_WARMUP e OBS_RESULTS_DIR.'

observe-config: observe-check-runtime
	@printf '%s\n' \
		"runtime=$(OBS_RUNTIME)" \
		"model_source=$(OBS_MODEL_SOURCE)" \
		"model=$(OBS_MODEL)" \
		"revision=$(OBS_REVISION)" \
		"base_url=$(OBS_BASE_URL)" \
		"model_alias=$(OBS_MODEL_ALIAS)" \
		"context=$(OBS_CONTEXT)" \
		"vllm_env_mode=$(OBS_VLLM_ENV_MODE)" \
		"vllm_lock=$(OBS_VLLM_WHEEL),$(OBS_VLLM_TORCH_PACKAGE),$(OBS_VLLM_TORCH_BACKEND),plugin@$(OBS_VLLM_GGUF_PLUGIN_REVISION)" \
		"vllm_cuda_toolkit=$(OBS_VLLM_CUDA_HOME),$(OBS_VLLM_CUDA_TOOLKIT_PACKAGE),auto_install=$(OBS_VLLM_AUTO_INSTALL_CUDA_TOOLKIT)" \
		"vllm_format=$(OBS_VLLM_FORMAT)" \
		"vllm_model=$(OBS_VLLM_MODEL)" \
		"gguf_model=$(OBS_GGUF_MODEL)" \
		"prometheus=$(OBS_PROMETHEUS_URL)" \
		"jupyter_port=$(OBS_JUPYTER_PORT)"

observe-check-runtime:
	@case "$(OBS_RUNTIME)" in \
		vllm|llama|ollama) ;; \
		*) echo 'OBS_RUNTIME deve ser vllm, llama ou ollama.' >&2; exit 2 ;; \
	esac

observe-check-model-source: observe-check-runtime
	@case "$(OBS_MODEL_SOURCE):$(OBS_RUNTIME)" in \
		hf:vllm|local-hf:vllm|gguf:vllm|local-gguf:vllm|gguf:llama|local-gguf:llama|gguf:ollama|local-gguf:ollama) ;; \
		*) echo 'Combinação inválida: HF/local-HF só usa vLLM; GGUF/local-GGUF usa vLLM, llama ou Ollama.' >&2; exit 2 ;; \
	esac
	@test -n "$(OBS_MODEL)" || { echo 'Informe OBS_MODEL com repo HF ou caminho local.' >&2; exit 2; }
	@if test "$(OBS_RUNTIME)" = ollama; then test -n "$(OBS_OLLAMA_VERSION)" || { echo 'OBS_OLLAMA_VERSION exata é obrigatória.' >&2; exit 2; }; fi
	@case "$(OBS_BENCH_SMOKE)" in 0|1) ;; *) echo 'OBS_BENCH_SMOKE deve ser 0 ou 1.' >&2; exit 2 ;; esac
	@case "$(OBS_BENCH_PROFILE)" in generic|mopep-single|mopep-review-replay|mopep-review-closed-loop) ;; *) echo 'OBS_BENCH_PROFILE inválido.' >&2; exit 2 ;; esac
	@if test "$(OBS_BENCH_PROFILE)" != generic; then \
		test -n "$(OBS_REQUEST_DATASET)" -a -n "$(OBS_WARMUP_REQUEST_DATASET)" -a -n "$(OBS_REQUEST_MANIFEST)" || { echo 'Perfil MOPEP exige OBS_REQUEST_DATASET, OBS_WARMUP_REQUEST_DATASET e OBS_REQUEST_MANIFEST.' >&2; exit 2; }; \
	fi
	@if test "$(OBS_BENCH_PROFILE)" = mopep-review-replay; then \
		test -n "$(OBS_REPLAY_RESPONSES)" || { echo 'mopep-review-replay exige OBS_REPLAY_RESPONSES.' >&2; exit 2; }; \
	fi

observe-system-install:
	@command -v apt-get >/dev/null || { echo 'Este alvo requer uma imagem Ubuntu/Debian.' >&2; exit 1; }
	@test "$$(id -u)" -eq 0 || { echo 'Execute observe-system-install como root no Pod.' >&2; exit 1; }
	apt-get update
	DEBIAN_FRONTEND=noninteractive apt-get install -y \
		git make curl ca-certificates python3 python3-venv python3-pip \
		build-essential cmake ninja-build pkg-config tmux
	command -v nvidia-smi >/dev/null
	nvidia-smi

observe-tools-install:
	@mkdir -p "$(OBS_VENVS_DIR)"
	@test -x "$(OBS_TOOLS_PYTHON)" || "$(OBS_SYSTEM_PYTHON)" -m venv "$(OBS_TOOLS_VENV)"
	"$(OBS_TOOLS_PYTHON)" -m pip install --disable-pip-version-check --upgrade pip
	"$(OBS_TOOLS_PYTHON)" -m pip install --disable-pip-version-check 'huggingface_hub==1.32.0'

observe-download-gguf: observe-tools-install
	@mkdir -p "$$(dirname "$(OBS_GGUF_MODEL)")"
	"$(OBS_TOOLS_PYTHON)" -c 'from huggingface_hub import hf_hub_download; import sys; print(hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], revision=sys.argv[3], local_dir=sys.argv[4]))' \
		"$(OBS_GGUF_REPO)" "$(OBS_GGUF_FILENAME)" "$(OBS_GGUF_REVISION)" "$$(dirname "$(OBS_GGUF_MODEL)")"
	@test -f "$(OBS_GGUF_MODEL)" || { echo 'GGUF não encontrado no destino esperado: $(OBS_GGUF_MODEL)' >&2; exit 1; }
	"$(OBS_SYSTEM_PYTHON)" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.open("rb").read(4) == b"GGUF", "assinatura GGUF inválida"; print(p)' "$(OBS_GGUF_MODEL)"

observe-install: observe-check-runtime
	@$(MAKE) --no-print-directory "observe-install-$(OBS_RUNTIME)"

observe-install-vllm:
	@case "$(OBS_VLLM_ENV_MODE)" in managed|existing) ;; *) echo 'OBS_VLLM_ENV_MODE deve ser managed ou existing.' >&2; exit 2 ;; esac
	if test "$(OBS_VLLM_ENV_MODE)" = managed; then \
		$(OBS_VLLM_RUN_DRIVER_PREFLIGHT); \
		command -v ninja >/dev/null || { echo 'ninja ausente; execute make observe-system-install ou apt-get install -y ninja-build.' >&2; exit 1; }; \
		mkdir -p "$(OBS_VENVS_DIR)"; \
		test -x "$(OBS_VLLM_PYTHON)" || "$(OBS_SYSTEM_PYTHON)" -m venv "$(OBS_VLLM_VENV)"; \
		"$(OBS_VLLM_PYTHON)" -m pip install --disable-pip-version-check --upgrade pip uv; \
		"$(OBS_VLLM_PYTHON)" -m uv pip install "$(OBS_VLLM_WHEEL)" "$(OBS_VLLM_TORCH_PACKAGE)" --torch-backend="$(OBS_VLLM_TORCH_BACKEND)"; \
	else \
		test -x "$(OBS_VLLM_PYTHON)" || { echo 'Modo existing exige OBS_VLLM_PYTHON apontando para um Python executável da imagem.' >&2; exit 1; }; \
		test -x "$(OBS_VLLM_BIN)" || { echo 'Modo existing exige OBS_VLLM_BIN apontando para o vLLM executável da imagem.' >&2; exit 1; }; \
		echo '[OBSERVE] reutilizando runtime vLLM existente; nenhuma instalação pip/uv será executada.'; \
	fi
	$(if $(OBS_VLLM_NEEDS_COMPILER),$(OBS_VLLM_ENSURE_CUDA_TOOLKIT))
	$(OBS_VLLM_EXPORT_CUDA_RUNTIME)
	$(OBS_VLLM_RUN_PREFLIGHT)
	if test "$(OBS_VLLM_FORMAT)" = gguf || test "$(OBS_MODEL_SOURCE)" = gguf || test "$(OBS_MODEL_SOURCE)" = local-gguf; then \
		if test "$(OBS_VLLM_ENV_MODE)" = managed; then \
			if ! test -d "$(OBS_VLLM_GGUF_PLUGIN_DIR)/.git"; then \
				test ! -e "$(OBS_VLLM_GGUF_PLUGIN_DIR)" || test -z "$$(ls -A "$(OBS_VLLM_GGUF_PLUGIN_DIR)")" || { echo 'OBS_VLLM_GGUF_PLUGIN_DIR existe e não é um repositório vazio.' >&2; exit 1; }; \
				mkdir -p "$(OBS_VLLM_GGUF_PLUGIN_DIR)"; \
				git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" init -q; \
				git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" remote add origin https://github.com/vllm-project/vllm-gguf-plugin.git; \
				git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" fetch --depth 1 origin "$(OBS_VLLM_GGUF_PLUGIN_REVISION)"; \
				git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" checkout --detach FETCH_HEAD; \
			else \
				actual=$$(git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" rev-parse HEAD); \
				test "$$actual" = "$(OBS_VLLM_GGUF_PLUGIN_REVISION)" || { echo "vllm-gguf-plugin está em $$actual, esperado $(OBS_VLLM_GGUF_PLUGIN_REVISION)." >&2; exit 1; }; \
			fi; \
			test -z "$$(git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" status --porcelain --untracked-files=no)" || { echo 'vllm-gguf-plugin possui alterações locais.' >&2; exit 1; }; \
			"$(OBS_VLLM_PYTHON)" -m pip install --disable-pip-version-check --no-build-isolation "$(OBS_VLLM_GGUF_PLUGIN_DIR)"; \
		fi; \
		$(OBS_VLLM_EXPORT_CUDA_RUNTIME); \
		"$(OBS_VLLM_PYTHON)" scripts/vllm_preflight.py \
			--mode "$(OBS_VLLM_ENV_MODE)" --output "$(OBS_VLLM_ENV_MANIFEST)" \
			--vllm-package "$(OBS_VLLM_PACKAGE)" --vllm-wheel "$(OBS_VLLM_WHEEL)" --torch-package "$(OBS_VLLM_TORCH_PACKAGE)" \
			--torch-backend "$(OBS_VLLM_TORCH_BACKEND)" --expected-vllm-version "$(OBS_VLLM_EXPECTED_VERSION)" \
			--expected-torch-version "$(OBS_VLLM_EXPECTED_TORCH_VERSION)" --expected-cuda-version "$(OBS_VLLM_EXPECTED_CUDA_VERSION)" \
			--plugin-revision "$(OBS_VLLM_GGUF_PLUGIN_REVISION)" --plugin-state require \
			--cuda-home "$(OBS_VLLM_CUDA_HOME)" $(if $(OBS_VLLM_NEEDS_COMPILER),--require-compiler,); \
	fi

observe-vllm-preflight:
	@test -x "$(OBS_VLLM_PYTHON)" || { echo 'Python do vLLM ausente; informe OBS_VLLM_PYTHON ou instale o runtime.' >&2; exit 1; }
	$(if $(OBS_VLLM_NEEDS_COMPILER),$(OBS_VLLM_ENSURE_CUDA_TOOLKIT))
	$(OBS_VLLM_EXPORT_CUDA_RUNTIME)
	$(OBS_VLLM_RUN_PREFLIGHT)

observe-install-llama:
	@command -v cmake >/dev/null || { echo 'cmake ausente; execute make observe-system-install.' >&2; exit 1; }
	@command -v nvcc >/dev/null || { echo 'nvcc ausente; use uma imagem CUDA devel para compilar llama.cpp.' >&2; exit 1; }
	@mkdir -p "$$(dirname "$(OBS_LLAMA_DIR)")"
	@if ! test -d "$(OBS_LLAMA_DIR)/.git"; then \
		test ! -e "$(OBS_LLAMA_DIR)" || test -z "$$(ls -A "$(OBS_LLAMA_DIR)")" || { echo 'OBS_LLAMA_DIR existe e não é um repositório vazio.' >&2; exit 1; }; \
		mkdir -p "$(OBS_LLAMA_DIR)"; \
		git -C "$(OBS_LLAMA_DIR)" init -q; \
		git -C "$(OBS_LLAMA_DIR)" remote add origin https://github.com/ggml-org/llama.cpp.git; \
		git -C "$(OBS_LLAMA_DIR)" fetch --depth 1 origin "$(OBS_LLAMA_REVISION)"; \
		git -C "$(OBS_LLAMA_DIR)" checkout --detach FETCH_HEAD; \
	else \
		actual=$$(git -C "$(OBS_LLAMA_DIR)" rev-parse HEAD); \
		resolved=$$(git -C "$(OBS_LLAMA_DIR)" rev-parse --verify "$(OBS_LLAMA_REVISION)^{commit}" 2>/dev/null || true); \
		if test "$$actual" != "$(OBS_LLAMA_REVISION)" && test "$$actual" != "$$resolved"; then \
			echo "llama.cpp está em $$actual, mas OBS_LLAMA_REVISION=$(OBS_LLAMA_REVISION). Use outro OBS_LLAMA_DIR ou alinhe explicitamente a revisão." >&2; exit 1; \
		fi; \
	fi
	@test -z "$$(git -C "$(OBS_LLAMA_DIR)" status --porcelain --untracked-files=no)" || { echo 'llama.cpp possui alterações locais; use uma árvore limpa.' >&2; exit 1; }
	cmake -S "$(OBS_LLAMA_DIR)" -B "$(OBS_LLAMA_BUILD_DIR)" \
		-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$(OBS_CUDA_ARCH)"
	cmake --build "$(OBS_LLAMA_BUILD_DIR)" --config Release --target llama-server -j"$(OBS_JOBS)"
	"$(OBS_LLAMA_BIN)" --list-devices
	git -C "$(OBS_LLAMA_DIR)" rev-parse HEAD

observe-install-ollama:
	@if command -v "$(OBS_OLLAMA_BIN)" >/dev/null 2>&1; then \
		true; \
	else \
		installer=$$(mktemp); trap 'rm -f "$$installer"' EXIT INT TERM; \
		curl -fsSL https://ollama.com/install.sh -o "$$installer"; \
		OLLAMA_VERSION="$(OBS_OLLAMA_VERSION)" sh "$$installer"; \
	fi
	@installed_text=$$("$(OBS_OLLAMA_BIN)" --version 2>&1); printf '%s\n' "$$installed_text"
	installed_version=$$(printf '%s\n' "$$installed_text" | "$(OBS_SYSTEM_PYTHON)" -c 'import re,sys; values=re.findall(r"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+)(?![0-9])", sys.stdin.read()); print(values[-1] if values else "")')
	test "$$installed_version" = "$(OBS_OLLAMA_VERSION)" || { echo "Ollama instalado=$$installed_version, esperado OBS_OLLAMA_VERSION=$(OBS_OLLAMA_VERSION)." >&2; exit 1; }

observe-prepare: observe-check-runtime
	@$(MAKE) --no-print-directory "observe-prepare-$(OBS_RUNTIME)"

observe-prepare-vllm: observe-install-vllm
	@if test "$(OBS_VLLM_FORMAT)" = gguf; then \
		$(MAKE) --no-print-directory observe-download-gguf; \
	elif test "$(OBS_VLLM_FORMAT)" != hf; then \
		echo 'OBS_VLLM_FORMAT deve ser hf ou gguf.' >&2; exit 2; \
	fi

observe-prepare-llama: observe-install-llama observe-download-gguf
	@true

observe-prepare-ollama: observe-install-ollama observe-download-gguf
	@mkdir -p "$(OBS_OLLAMA_MODELS)"
	$(OBS_OLLAMA_ENV) "$(OBS_SYSTEM_PYTHON)" scripts/prepare_ollama.py \
		--binary "$(OBS_OLLAMA_BIN)" --model "$(OBS_MODEL_ALIAS)" --gguf "$(OBS_GGUF_MODEL)" \
		--context "$(OBS_CONTEXT)" --base-url "http://$(OBS_HOST):$(OBS_OLLAMA_PORT)" \
		--results results

observe-bootstrap: observe-system-install
	@$(MAKE) --no-print-directory observe-prepare OBS_RUNTIME="$(OBS_RUNTIME)"
	@$(MAKE) --no-print-directory prometheus-install
	@$(MAKE) --no-print-directory observe-notebook-install

observe-serve: observe-check-runtime
	@$(MAKE) --no-print-directory "observe-serve-$(OBS_RUNTIME)"

observe-serve-vllm:
	@test -x "$(OBS_VLLM_BIN)" || { echo 'vLLM ausente; execute make observe-prepare OBS_RUNTIME=vllm.' >&2; exit 1; }
	$(OBS_VLLM_EXPORT_CUDA_RUNTIME)
	model="$(OBS_VLLM_MODEL)"
	format_args=()
	if test "$(OBS_VLLM_FORMAT)" = gguf; then \
		model="$(OBS_GGUF_MODEL)"; format_args=(--tokenizer "$(OBS_TOKENIZER_MODEL)"); \
	elif test "$(OBS_VLLM_FORMAT)" != hf; then \
		echo 'OBS_VLLM_FORMAT deve ser hf ou gguf.' >&2; exit 2; \
	fi
	HF_HOME="$(OBS_HF_HOME)" exec "$(OBS_VLLM_BIN)" serve "$$model" \
		--served-model-name "$(OBS_MODEL_ALIAS)" --host "$(OBS_HOST)" --port "$(OBS_VLLM_PORT)" \
		--max-model-len "$(OBS_CONTEXT)" --max-num-seqs "$(OBS_VLLM_MAX_NUM_SEQS)" \
		--gpu-memory-utilization "$(OBS_VLLM_GPU_MEMORY_UTILIZATION)" --dtype "$(OBS_VLLM_DTYPE)" \
		"$${format_args[@]}" $(OBS_VLLM_EXTRA_ARGS)

observe-serve-llama:
	@test -x "$(OBS_LLAMA_BIN)" || { echo 'llama-server ausente; execute make observe-prepare OBS_RUNTIME=llama.' >&2; exit 1; }
	@test -f "$(OBS_GGUF_MODEL)" || { echo 'GGUF ausente; execute make observe-download-gguf.' >&2; exit 1; }
	exec "$(OBS_LLAMA_BIN)" -m "$(OBS_GGUF_MODEL)" --alias "$(OBS_MODEL_ALIAS)" \
		--host "$(OBS_HOST)" --port "$(OBS_LLAMA_PORT)" --ctx-size "$(OBS_CONTEXT)" \
		--n-gpu-layers "$(OBS_LLAMA_GPU_LAYERS)" --parallel 1 --jinja $(OBS_LLAMA_EXTRA_ARGS)

observe-serve-ollama:
	@command -v "$(OBS_OLLAMA_BIN)" >/dev/null 2>&1 || { echo 'Ollama ausente; execute make observe-prepare OBS_RUNTIME=ollama.' >&2; exit 1; }
	$(OBS_OLLAMA_ENV) OLLAMA_HOST="$(OBS_HOST):$(OBS_OLLAMA_PORT)" \
		exec "$(OBS_OLLAMA_BIN)" serve $(OBS_OLLAMA_EXTRA_ARGS)

observe-prometheus-start: observe-check-runtime
	@if test "$(OBS_RUNTIME)" = vllm; then \
		$(MAKE) --no-print-directory prometheus-start PROMETHEUS_RUNTIME_TARGET="$(OBS_HOST):$(OBS_VLLM_PORT)"; \
	else \
		$(MAKE) --no-print-directory prometheus-start; \
	fi

observe-prometheus-stop:
	@$(MAKE) --no-print-directory prometheus-stop

observe-status: observe-check-runtime
	@curl -fsS "$(OBS_BASE_URL)/v1/models"
	@printf '\n'
	@$(MAKE) --no-print-directory prometheus-status

observe-notebook-install:
	@$(MAKE) --no-print-directory -C ../.. setup-notebook

observe-jupyter: observe-check-runtime
	@test -x "$(CURDIR)/../../analytics/.venv/bin/jupyter" || { echo 'Notebook ausente; execute make observe-notebook-install.' >&2; exit 1; }
	INFERENCE_BASE_URL="$(OBS_BASE_URL)" INFERENCE_MODEL="$(OBS_MODEL_ALIAS)" \
	PROMETHEUS_URL="$(OBS_PROMETHEUS_URL)" \
		exec "$(CURDIR)/../../analytics/.venv/bin/jupyter" lab \
		"$(CURDIR)/../../analytics/notebooks/profile_runtime_with_prometheus.ipynb" \
		--no-browser --ip 127.0.0.1 --port "$(OBS_JUPYTER_PORT)" --port-retries 0 --allow-root

# A preparação materializa tudo no SSD. O bench subsequente executa com o Hub
# offline e falha se qualquer peso/tokenizer estiver ausente.
observe-model-prepare: observe-check-model-source observe-tools-install
	@mkdir -p "$(OBS_STATE_DIR)" "$(OBS_PREPARED_MODELS_DIR)"
	"$(OBS_TOOLS_PYTHON)" scripts/prepare_observe_model.py prepare \
		--source "$(OBS_MODEL_SOURCE)" --runtime "$(OBS_RUNTIME)" --model "$(OBS_MODEL)" \
		$(if $(filter hf gguf,$(OBS_MODEL_SOURCE)),--revision "$(OBS_REVISION)") \
		$(if $(filter gguf,$(OBS_MODEL_SOURCE)),--gguf-filename "$(OBS_GGUF_FILENAME)") \
		$(if $(filter gguf local-gguf,$(OBS_MODEL_SOURCE)),--tokenizer "$(OBS_TOKENIZER_MODEL)" --tokenizer-revision "$(OBS_TOKENIZER_REVISION)") \
		--output-dir "$(OBS_PREPARED_MODELS_DIR)" --metadata-out "$(OBS_MODEL_METADATA)"

observe-render-runtime: observe-model-prepare
	@runtime_version=
	case "$(OBS_RUNTIME)" in \
		vllm) \
			runtime_version=$$("$(OBS_VLLM_PYTHON)" -c 'import importlib.metadata as m, torch; p=["vLLM "+m.version("vllm"), "torch "+m.version("torch"), "CUDA "+str(torch.version.cuda)]; p += ["vllm-gguf-plugin "+m.version("vllm-gguf-plugin")] if "$(OBS_MODEL_SOURCE)" in {"gguf","local-gguf"} else []; print(", ".join(p))'); \
			test -f "$(OBS_VLLM_ENV_MANIFEST)" || { echo 'Manifesto do preflight vLLM ausente.' >&2; exit 1; }; \
			runtime_version="$$runtime_version; env_mode=$(OBS_VLLM_ENV_MODE); environment_sha256=$$(sha256sum "$(OBS_VLLM_ENV_MANIFEST)" | awk '{print $$1}'); executable_sha256=$$(sha256sum "$(OBS_VLLM_BIN)" | awk '{print $$1}')"; \
			if test "$(OBS_MODEL_SOURCE)" = gguf || test "$(OBS_MODEL_SOURCE)" = local-gguf; then \
				if test "$(OBS_VLLM_ENV_MODE)" = managed; then runtime_version="$$runtime_version; plugin_commit=$$(git -C "$(OBS_VLLM_GGUF_PLUGIN_DIR)" rev-parse HEAD)"; \
				else runtime_version="$$runtime_version; plugin_source=preinstalled"; fi; \
			fi ;; \
		llama) runtime_version="llama.cpp $$(git -C "$(OBS_LLAMA_DIR)" rev-parse HEAD); binary_sha256=$$(sha256sum "$(OBS_LLAMA_BIN)" | awk '{print $$1}'); GGML_CUDA=ON; cuda_arch=$(OBS_CUDA_ARCH)" ;; \
		ollama) runtime_version="Ollama $(OBS_OLLAMA_VERSION); binary_sha256=$$(sha256sum "$(OBS_OLLAMA_BIN)" | awk '{print $$1}')" ;; \
	esac
	test -n "$$runtime_version" || { echo 'Não foi possível identificar a versão do runtime.' >&2; exit 1; }
	"$(OBS_TOOLS_PYTHON)" scripts/render_observe_runtime.py \
		--metadata "$(OBS_MODEL_METADATA)" --config-out "$(OBS_GENERATED_CONFIG)" \
		--launch-out "$(OBS_GENERATED_LAUNCH)" --runtime "$(OBS_RUNTIME)" \
		--runtime-version "$$runtime_version" \
		--model-alias "$(OBS_MODEL_ALIAS)" --context "$(OBS_CONTEXT)" \
		--host "$(OBS_HOST)" --port "$(OBS_RUNTIME_PORT)" --runtime-bin "$(OBS_RUNTIME_BIN)" \
		--gpu-memory-utilization "$(OBS_VLLM_GPU_MEMORY_UTILIZATION)" \
		--max-num-seqs "$(OBS_VLLM_MAX_NUM_SEQS)" --dtype "$(OBS_VLLM_DTYPE)" \
		--gpu-layers "$(OBS_LLAMA_GPU_LAYERS)"

observe-prepare-headless-ollama: observe-render-runtime
	@if test "$(OBS_RUNTIME)" = ollama; then \
		model_path=$$("$(OBS_TOOLS_PYTHON)" -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_path"])' "$(OBS_MODEL_METADATA)"); \
		mkdir -p "$(OBS_OLLAMA_MODELS)" "$(OBS_STATE_DIR)/ollama-prepare"; \
		$(OBS_OLLAMA_ENV) "$(PYTHON)" scripts/prepare_ollama.py \
			--binary "$(OBS_OLLAMA_BIN)" --model "$(OBS_MODEL_ALIAS)" --gguf "$$model_path" \
			--context "$(OBS_CONTEXT)" --base-url "$(OBS_BASE_URL)" \
			--results "$(OBS_STATE_DIR)/ollama-prepare"; \
	fi

observe-bench-prepare: observe-check-model-source observe-install install-benchmark prometheus-install observe-render-runtime observe-prepare-headless-ollama
	@echo '[OBSERVE] preparação concluída; pesos, tokenizer, runtimes e coletores estão locais.'

observe-bench-validate: observe-bench-prepare
	@"$(PYTHON)" bench.py validate \
		--config "$(OBS_GENERATED_CONFIG)" \
		$(if $(strip $(OBS_BENCH_INPUT_TOKENS)),--input-tokens $(OBS_BENCH_INPUT_TOKENS),--scenarios $(OBS_BENCH_SCENARIOS)) \
		--benchmark-profile "$(OBS_BENCH_PROFILE)" \
		$(if $(filter 1,$(OBS_BENCH_SMOKE)),--smoke)

observe-bench: observe-bench-validate
	@prometheus_started=0
	if test "$(OBS_RUNTIME)" = vllm; then \
		$(OBS_VLLM_EXPORT_CUDA_RUNTIME)
	fi
	cleanup() { \
		status=$$?; \
		trap - EXIT INT TERM; \
		if test "$$prometheus_started" = 1; then \
			"$(OBS_SYSTEM_PYTHON)" scripts/prometheus_stack.py stop \
				--state-dir "$(PROMETHEUS_STATE_DIR)" \
				--prometheus-host "$(PROMETHEUS_HOST)" --prometheus-port "$(PROMETHEUS_PORT)" \
				--exporter-host "$(PROMETHEUS_EXPORTER_HOST)" --exporter-port "$(PROMETHEUS_EXPORTER_PORT)" || true; \
		fi; \
		exit "$$status"; \
	}
	trap cleanup EXIT INT TERM
	runtime_target=()
	if test "$(OBS_RUNTIME)" = vllm; then runtime_target=(--runtime-metrics-target "$(OBS_HOST):$(OBS_VLLM_PORT)"); fi
	"$(PYTHON)" scripts/prometheus_stack.py start \
		--prometheus-bin "$(PROMETHEUS_BIN)" --python "$(PYTHON)" --state-dir "$(PROMETHEUS_STATE_DIR)" \
		--prometheus-host "$(PROMETHEUS_HOST)" --prometheus-port "$(PROMETHEUS_PORT)" \
		--exporter-host "$(PROMETHEUS_EXPORTER_HOST)" --exporter-port "$(PROMETHEUS_EXPORTER_PORT)" \
		--scrape-interval "$(PROMETHEUS_SCRAPE_INTERVAL)" --scrape-timeout "$(PROMETHEUS_SCRAPE_TIMEOUT)" \
		--retention "$(PROMETHEUS_RETENTION)" "$${runtime_target[@]}" \
		--runtime-metrics-path "$(PROMETHEUS_RUNTIME_METRICS_PATH)"
	prometheus_started=1
	extra_args=$$("$(OBS_SYSTEM_PYTHON)" -c 'import json,os,shlex,sys; key={"vllm":"OBS_VLLM_EXTRA_ARGS","llama":"OBS_LLAMA_EXTRA_ARGS","ollama":"OBS_OLLAMA_EXTRA_ARGS"}[sys.argv[1]]; print(json.dumps(shlex.split(os.environ.get(key, ""))))' "$(OBS_RUNTIME)")
	$(OBS_OLLAMA_ENV) HF_HOME="$(OBS_HF_HOME)" \
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		"$(PYTHON)" bench.py run \
		--config "$(OBS_GENERATED_CONFIG)" --local-model-path "$$("$(OBS_TOOLS_PYTHON)" -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_path"])' "$(OBS_MODEL_METADATA)")" \
		--model-metadata "$(OBS_MODEL_METADATA)" --launch "$(OBS_GENERATED_LAUNCH)" \
		--launch-extra-args-json "$$extra_args" --startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		$(if $(strip $(OBS_BENCH_INPUT_TOKENS)),--input-tokens $(OBS_BENCH_INPUT_TOKENS),--scenarios $(OBS_BENCH_SCENARIOS)) \
		--requests "$(OBS_BENCH_REQUESTS)" --repetitions "$(OBS_BENCH_REPETITIONS)" \
		--warmup "$(OBS_BENCH_WARMUP)" --mode "$(OBS_BENCH_MODE)" \
		--benchmark-profile "$(OBS_BENCH_PROFILE)" \
		$(if $(strip $(OBS_REQUEST_DATASET)),--request-dataset "$(OBS_REQUEST_DATASET)") \
		$(if $(strip $(OBS_WARMUP_REQUEST_DATASET)),--warmup-request-dataset "$(OBS_WARMUP_REQUEST_DATASET)") \
		$(if $(strip $(OBS_REQUEST_MANIFEST)),--request-manifest "$(OBS_REQUEST_MANIFEST)") \
		$(if $(strip $(OBS_REPLAY_RESPONSES)),--replay-responses "$(OBS_REPLAY_RESPONSES)") \
		$(if $(strip $(OBS_RESPONSES_OUT)),--responses-out "$(OBS_RESPONSES_OUT)") \
		--conversation-turns "$(OBS_BENCH_CONVERSATION_TURNS)" \
		--conversation-fixture "$(CONVERSATION_FIXTURE)" \
		--warmup-conversation-fixture "$(WARMUP_CONVERSATION_FIXTURE)" \
		--results "$(OBS_RESULTS_DIR)" $(if $(strip $(OBS_RESULT_NAME)),--result-name "$(OBS_RESULT_NAME)") \
		$(if $(filter 1,$(OBS_BENCH_SMOKE)),--smoke) --disable-native-monitor \
		--prometheus-url "$(OBS_PROMETHEUS_URL)" --prometheus-step "$(PROMETHEUS_SCRAPE_INTERVAL)"

observe-bench-all:
	@case "$(OBS_MODEL_SOURCE)" in \
		gguf|local-gguf) ;; \
		*) echo 'observe-bench-all exige OBS_MODEL_SOURCE=gguf ou local-gguf.' >&2; exit 2 ;; \
	esac
	@test -n "$(OBS_MODEL)" || { echo 'Informe OBS_MODEL com repo HF de um GGUF ou caminho local.' >&2; exit 2; }
	@printf '%s\n' '[CAMPAIGN] etapa 1/2: smoke em vLLM, llama.cpp e Ollama'
	@for runtime in vllm llama ollama; do \
		printf '[CAMPAIGN] smoke: %s\n' "$$runtime"; \
		$(MAKE) --no-print-directory observe-bench OBS_RUNTIME="$$runtime" OBS_BENCH_SMOKE=1; \
	done
	@printf '%s\n' '[CAMPAIGN] etapa 2/2: bateria completa em vLLM, llama.cpp e Ollama'
	@for runtime in vllm llama ollama; do \
		printf '[CAMPAIGN] completo: %s\n' "$$runtime"; \
		$(MAKE) --no-print-directory observe-bench OBS_RUNTIME="$$runtime" OBS_BENCH_SMOKE=0; \
	done
	@printf '%s\n' '[CAMPAIGN] concluída: 3 smokes e 3 baterias completas'

pull-observe-results: POD_SSH = $(OBS_REMOTE_HOST)
pull-observe-results: POD_PORT = $(OBS_REMOTE_PORT)
pull-observe-results: POD_SSH_KEY = $(OBS_REMOTE_KEY)
pull-observe-results: REMOTE_RESULTS_DIR = $(OBS_REMOTE_RESULTS_DIR)
pull-observe-results: LOCAL_RESULTS_DIR = $(OBS_LOCAL_RESULTS_DIR)
pull-observe-results: pull-results
	"$(OBS_SYSTEM_PYTHON)" scripts/verify_observability_results.py "$(OBS_LOCAL_RESULTS_DIR)"
