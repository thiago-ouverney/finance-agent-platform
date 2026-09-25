# Fluxo completo do Makefile

Este documento explica o que acontece quando cada alvo é executado. O objetivo é separar preparação, lançamento, medição e diagnóstico.

Requer GNU Make com suporte a `.ONESHELL` (3.82 ou mais recente). O Make 3.81 fornecido pelo macOS não executa corretamente receitas que compartilham variáveis de shell; use GNU Make atualizado (`gmake`) ou execute no pod.

## 1. Seleção do experimento

`MODEL_SIZE` aceita `7B` ou `14B`. O Make deriva o GGUF, tokenizer, config JSON e launch JSON correspondentes. O padrão é `7B`, usando o artefato `arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`.

Os launch/configs do vLLM usam por padrão `--gpu-memory-utilization 0.95`, deixando 5% de margem na VRAM. É um orçamento de memória, não um teto de 95% para utilização computacional da GPU. No Pod observado, `1.0` falhou porque havia 23,30 GiB livres de 23,56 GiB, menos que o orçamento integral solicitado.

```bash
make prepare-vllm MODEL_SIZE=7B
make prepare-llama MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
```

Para preparar todos os runtimes em sequência:

```bash
make prepare-all MODEL_SIZE=7B
```

Os caminhos podem ser sobrescritos:

```bash
make prepare-all MODEL_SIZE=7B \
  MODEL_7B_GGUF=/workspace/models/arthur/qwen7b.gguf \
  MODEL_7B_TOKENIZER=/workspace/models/qwen7b-tokenizer
```

## 2. Preparação

Cada alvo `prepare-<runtime>` depende de `prepare-benchmark`, que é a etapa comum de `install-benchmark`, `download-model`, `download-tokenizer` e `verify-gguf`. Depois da primeira preparação, use `PREPARE_OFFLINE=1` para reutilizar o ambiente e os arquivos locais sem pip nem downloads. `prepare-all` encadeia os três runtimes. As verificações de GGUF, imports, tokenizer e JSONs continuam obrigatórias; dependências ausentes causam falha.

```bash
make smoke-vllm PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
make smoke-llama PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
make smoke-ollama PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
make smoke-all PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
```

`BENCH_STARTUP_TIMEOUT` vale para os três smokes. `VLLM_EXTRA_ARGS` é repassado tanto à bateria base quanto ao sweep.

### `install-benchmark`

Cria `BENCH_VENV` usando `SYSTEM_PYTHON` quando necessário, atualiza o pip e instala `requirements.txt`. O Python efetivamente usado pelos demais alvos é `PYTHON`. Esse ambiente é do cliente do benchmark; não o misture com o venv do vLLM ou com o ambiente de conversão do llama.cpp.

### `download-model`

Cria o diretório pai e executa `hf download` com `HF_HUB_OFFLINE=0`. O repositório e o nome do arquivo vêm de `HF_GGUF_REPO` e `HF_GGUF_FILENAME`. O alvo verifica que o arquivo chegou exatamente ao caminho esperado.

### `download-tokenizer`

Cria `TOKENIZER_DIR` e usa `huggingface_hub.snapshot_download` com `allow_patterns` para baixar `config.json`, `tokenizer*`, `special_tokens_map.json` e `chat_template.jinja`. Isso evita uma incompatibilidade da CLI `hf` em que a combinação de nomes posicionais e `--include` baixava somente `config.json`. O alvo exige `config.json`, `tokenizer_config.json` e `tokenizer.json`; não baixa pesos do checkpoint base.

### `verify-gguf`

Verifica presença, tamanho, assinatura binária `GGUF` e imprime SHA-256. Não interpreta os tensores e não inicia GPU.

### Validações finais

O alvo instala `requirements.txt` no `PYTHON` escolhido, importa as bibliotecas do cliente e valida JSONs. A presença do runtime é verificada separadamente por `prepare-vllm`, `prepare-llama` ou `prepare-ollama`; assim, preparar o modelo não é bloqueado por um runtime que ainda não foi instalado. No caso do llama.cpp, `prepare-llama` procura o `llama-server` já instalado no `PATH`, como ocorre na imagem do pod, e o launch JSON usa o comando `llama-server`. A compilação local fica disponível somente quando necessária, via `make build-llama`.

`prepare-llama` também exige CUDA por padrão (`REQUIRE_GPU=1`). O Make detecta automaticamente o plugin Ollama em `cuda_v12` ou, na ausência dele, `cuda_v13`. Quando a distribuição usa plugins de backend em outro local, `LLAMA_BACKEND_PATH` deve ser o caminho completo do `.so`, por exemplo `/caminho/cuda_v13/libggml-cuda.so`. O Make define `GGML_BACKEND_PATH` com esse arquivo e inclui seu diretório em `LD_LIBRARY_PATH` antes de executar `--list-devices`. Use `REQUIRE_GPU=0` somente para uma execução CPU intencional.
Os três templates seguem esse mesmo contrato: chamam `vllm`, `llama-server` e `ollama` pelo `PATH`, sem assumir caminhos internos de venv ou `/usr/local/bin`. Use `VLLM_BIN`, `LLAMA_SERVER_BIN` ou `OLLAMA_BIN` para sobrescrever o executável quando a imagem utilizar outro local.

`prepare-vllm` também executa `check-vllm-gguf`, que usa `VLLM_PYTHON` (ou tenta resolver o Python do executável encontrado) e importa `vllm_gguf_plugin` nesse mesmo ambiente. Para um executável wrapper, informe explicitamente `VLLM_PYTHON=/caminho/do/python-do-vllm`. Sem o plugin, o vLLM pode tentar interpretar o arquivo GGUF como JSON e produzir `config file ... .gguf is not a valid JSON file`. A correção é instalar `vllm-gguf-plugin` no ambiente do vLLM; não se instala esse pacote no venv do benchmark. llama.cpp e Ollama não precisam do plugin porque carregam GGUF nativamente.

## 3. Preparação do Ollama

`prepare-ollama` usa o arquivo GGUF local em um `Modelfile` temporário e executa `ollama create`. O alias é `qwen7b-q8-gguf` ou `qwen14b-q8-gguf`. Não há `ollama pull`: o modelo deve vir do SSD informado. Se a API configurada já estiver ativa, o script a reutiliza e a preserva. Se não estiver, inicia um daemon temporário próprio, aguarda a API, cria/verifica o alias e encerra apenas o processo que criou. Essa tolerância vale para a preparação; `bench-ollama`, como os demais launchers medidos, requer a porta livre para possuir e cronometrar o servidor.

## 4. Bateria formal

`bench` chama `bench-all`. O agregador executa `bench-vllm`, `bench-llama` e `bench-ollama`, guarda um código de saída para cada um, continua após falhas e retorna código diferente de zero no final se algum falhou.

Cada alvo individual:

1. verifica o mesmo GGUF local;
2. inicia o runtime pelo launcher JSON;
3. espera `/v1/models` responder com o alias esperado;
4. mede a primeira requisição sem aquecimento oculto;
5. executa warmup e medida para `short`, `medium` e `long`;
6. mede TTFT, Tokens/s e velocidade efetiva;
7. coleta GPU, CPU, RAM, SSD, eventos e KV quando disponível;
8. encerra somente o grupo de processos criado;
9. executa o sweep de contexto/KV.

São 50 requisições por cenário, 1 repetição e 3 aquecimentos. A bateria base usa `BENCH_MODE=replay`: as 50 requisições percorrem as 50 perguntas de `qwen_chat_bench_v2.json`, com histórico e respostas `assistant` fixos. Os três warmups usam `qwen_chat_warmup_v1.json`, que não é reutilizado na medição. O perfil é síncrono, portanto não é um benchmark de concorrência.

Por padrão, `BENCH_MODE=replay` percorre `workloads/conversations/qwen_chat_bench_v2.json`, com 50 perguntas e respostas `assistant` fixas. Cada request seleciona um turno distinto e envia o histórico até ele. O warmup usa `qwen_chat_warmup_v1.json`, separado da medição. `closed-loop` continua disponível para colocar respostas reais no histórico; `independent` deve ser solicitado explicitamente quando a intenção for medir mensagens isoladas. Os blocos conversacionais gravam `*-turns.json` e `*-conversation.json` além dos brutos e CSVs.

A espera de startup não aceita apenas um HTTP 200: o alias configurado precisa aparecer em `GET /v1/models`. Enquanto aguarda, o log periódico identifica o endpoint consultado, o modelo esperado e a última observação ou erro. Isso torna distinguível uma API inacessível de um servidor vivo que expõe o alias errado.

## 5. Smoke e sweep rápido de integração

Há smoke equivalente para cada runtime:

```bash
make smoke-vllm MODEL_SIZE=7B
make smoke-llama MODEL_SIZE=7B
make smoke-ollama MODEL_SIZE=7B
make smoke-all MODEL_SIZE=7B
```

Para exercitar também a variação de contexto sem aguardar a bateria formal:

```bash
make quick-sweep-vllm MODEL_SIZE=7B
make quick-sweep-llama MODEL_SIZE=7B
make quick-sweep-ollama MODEL_SIZE=7B
```

O sweep rápido testa os pontos de 1024 e 5489 tokens até o limite padrão de 8192, com 1 requisição, 1 repetição e nenhum aquecimento por ponto. É diagnóstico de integração, não resultado estatístico nem substituto do sweep formal.

## 6. Argumentos experimentais

Os launchers JSON fornecem a configuração base. Variáveis `VLLM_EXTRA_ARGS`, `LLAMA_EXTRA_ARGS` e `OLLAMA_EXTRA_ARGS` são acrescentadas ao argv sem serem sanitizadas ou corrigidas.

```bash
make bench-vllm MODEL_SIZE=7B \
  VLLM_EXTRA_ARGS='--gpu-memory-utilization 0.95 --enforce-eager'
```

Esses argumentos aparecem no `lifecycle.json`. Se o runtime rejeitar a combinação, a execução falha e o `server.log` é a evidência.

## 7. Sweep de KV

Depois da bateria base, `run_kv_sweep.py` copia o launcher para uma pasta temporária e chama `bench.py` novamente. O padrão começa em 1024 tokens e soma 256 MB de KV lógico por ponto; no Qwen2.5 7B isso produz aproximadamente 1024, 5489, 9954 e 14419 tokens. Em modo `replay`, cada ponto seleciona e cicla apenas os históricos da fixture que cabem no limite `ctx<N>`. O `max_model_len` permanece pelo menos no `context_window` configurado no runtime, sem o teto artificial `N + 128 + 256`.

O sweep repassa `SWEEP_MODE`, `SWEEP_CONVERSATION_TURNS` e `SWEEP_CONVERSATION_FIXTURE` para `bench.py`, permitindo varrer contexto com a mesma política conversacional da bateria base ou com uma política própria do sweep. O passo formal é `SWEEP_MEMORY_STEP_MB=256`: o script converte esse orçamento de memória em tokens com `KV_BYTES_PER_TOKEN` (57.344 bytes/token no Qwen2.5 7B em KV FP16; 196.608 no 14B) e registra os dois valores no manifesto.

- vLLM: altera `--max-model-len`;
- llama.cpp: altera `--ctx-size`;
- Ollama: mantém a política configurada externamente, pois `ollama serve` não possui uma flag universal equivalente.

O primeiro código de saída diferente de zero encerra o sweep. Nenhum ponto falho é convertido em zero ou substituído por outro modelo.

## 8. Onde estão os resultados

Cada execução fica em `results/<runtime>/<timestamp>/<nome-humano>/`, com subpastas por formato:

| Arquivo | Função |
|---|---|
| `html/summary.html` | leitura humana, tabelas e gráficos SVG |
| `html/telemetry-memory.svg` | memória ocupada da GPU ao longo do tempo |
| `html/telemetry-gpu-util.svg` | GPU-utilização ao longo do tempo |
| `json/summary.json` | métricas estruturadas |
| `csv/gpu.csv` | amostras NVIDIA |
| `csv/system.csv` | CPU/RAM/disco do host |
| `csv/events.csv` | fases e timestamps |
| `json/telemetry-summary.json` | agregação por fase |
| `csv/kv-cache.csv` | gauge KV do servidor, se existir |
| `logs/server.log` | log do runtime iniciado |
| `json/lifecycle.json` | readiness, primeira resposta, argv e encerramento |
| `csv/telemetry-timeseries.csv` | série temporal da GPU com `elapsed_s`, VRAM ocupada e utilização |

As métricas de throughput dos requests não são extraídas do `server.log`. O cliente abre SSE e registra relógios monotônicos por requisição: início, primeiro evento com conteúdo e fim. `decode_tokens_per_second` usa tokens de `usage` dividido pelo intervalo primeiro–último evento; `end_to_end_tokens_per_second` divide pela latência total. O `Avg generation throughput` periódico do vLLM permanece apenas como observabilidade agregada. Ausência de `usage` produz `null`, e uma resposta de um token não recebe inter-token latency zero.

## 9. O que não é medido diretamente

`nvidia-smi` mostra VRAM total usada pela GPU, não bytes exclusivamente do KV. CPU/RAM/SSD são contadores observacionais do host. O benchmark não mede o tempo de cada cópia PCIe ou RAM↔VRAM; isso exige Nsight Systems/Compute ou instrumentação CUDA no runtime.

Para copiar resultados, execute localmente: `make pull-results POD_SSH=root@HOST_DO_POD`. O alvo aceita `POD_PORT`, `POD_SSH_KEY`, `REMOTE_RESULTS_DIR` e `LOCAL_RESULTS_DIR`, e não apaga resultados locais. Para `ssh.runpod.io`, detecta o proxy, cria um tar temporário no pod, coordena `runpodctl send/receive` e extrai os resultados automaticamente; instale o cliente local com `brew install runpod/runpodctl/runpodctl`. Para SSH TCP direto, usa `scp`. `make check-profilers` verifica `nsys` e `ncu`; profiling é complementar e deve ser executado separadamente, pois altera a latência.

O profiling automatizado fica em `Makefile.profiling`, separado da bateria oficial. Execute `make -f Makefile.profiling profile-vllm-nsys MODEL_SIZE=7B` ou o alvo equivalente de `ncu`/llama.cpp em outro ambiente. O alvo inicia o servidor em primeiro plano; envie uma única requisição curta em outro terminal e encerre com `Ctrl-C`. Os relatórios ficam em `results/profiling/`. Não instale/remova Nsight durante `bench-vllm`, pois isso mudaria o ambiente e contaminaria a medição.

## 10. Estado atual da validação no Pod

A integração foi validada ao vivo em 22/09/2026 no RunPod com uma RTX 3090 e o GGUF 7B. Preparação offline, três smokes, três quick sweeps e `bench-all` reduzido passaram; o agregado terminou com `vLLM=0 llama.cpp=0 Ollama=0`. As métricas e os caminhos dos artefatos estão em [resultados-pod-20260922.md](../benchmark/resultados-pod-20260922.md).

O vLLM foi executado com `--enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.80`. O `EngineDeadError` visto no encerramento do sweep ocorre após respostas HTTP 200 e o SIGTERM intencional do harness; não foi OOM e o alvo retornou zero.

## 11. Sequência recomendada

```bash
make prepare-all MODEL_SIZE=7B
make smoke-vllm MODEL_SIZE=7B
make smoke-llama MODEL_SIZE=7B
make smoke-ollama MODEL_SIZE=7B
make quick-sweep-vllm MODEL_SIZE=7B
make quick-sweep-llama MODEL_SIZE=7B
make quick-sweep-ollama MODEL_SIZE=7B
make bench-all MODEL_SIZE=7B
```

Só altere parâmetros depois de obter uma execução base válida. Cada alteração deve ser registrada na linha do Make e no `lifecycle.json`.

### Profiling

`Makefile.profiling` resolve o cliente Python a partir do diretório do próprio Makefile e chama os wrappers com Bash, sem exigir bit executável no checkout. Cada execução cria o subdiretório do relatório antes de iniciar o profiler. Os alvos de profiling iniciam servidores interativos; envie uma requisição curta e encerre com Ctrl-C para finalizar o relatório.
