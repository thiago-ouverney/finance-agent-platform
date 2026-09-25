# Chatbot Runtime Bench

Benchmark acadêmico de inferência local para comparar **vLLM, llama.cpp e Ollama** usando o mesmo modelo GGUF, uma requisição por vez e uma carga de chatbot reproduzível.

O fluxo oficial é baseado no `Makefile`. O benchmark não baixa modelos, não troca artefatos automaticamente e não faz fallback quando um runtime falha. Download, criação de diretórios e instalação são preparação; não entram nos tempos medidos.

Documentação detalhada: [índice de documentação](docs/README.md) · [bench explicado](docs/benchmark/bench-explicado.md) · [fluxo completo do Make](docs/make/make-fluxo.md) · [metodologia](docs/benchmark/metodologia.md) · [código explicado](docs/engenharia/codigo-explicado.md). Resultados e diagnósticos datados ficam no arquivo histórico do monorepo.

## Preparar um pod novo

Em uma imagem Ubuntu/Debian do RunPod:

```bash
apt-get update
apt-get install -y git curl ca-certificates build-essential cmake pkg-config python3 python3-venv python3-pip
nvidia-smi
git --version
python3 --version
mkdir -p /workspace
cd /workspace
git clone https://github.com/thiago-ouverney/finance-agent-platform.git
cd /workspace/finance-agent-platform/services/inference-runtime
```

O repositório é público, portanto HTTPS é o caminho mais simples para clonar no Pod e não exige chave SSH. O `hf` é instalado dentro do venv pelo `make install-benchmark`; não é necessário instalar um `hf` separado no sistema. Depois da instalação, verifique com `./.venv/bin/hf --help` ou deixe o Make chamá-lo automaticamente.

### SSH no Pod (somente se precisar fazer push)

O erro `Permission denied (publickey)` significa que o Pod não tem uma chave SSH autorizada na sua conta GitHub. Para configurar uma chave própria no Pod:

```bash
ssh-keygen -t ed25519 -C "runpod-github-key"
cat ~/.ssh/id_ed25519.pub
```

Adicione o conteúdo exibido em **GitHub → Settings → SSH and GPG keys → New SSH key**. Depois valide:

```bash
ssh -T git@github.com
```

Se o repositório já foi clonado por HTTPS e você quer fazer push por SSH:

```bash
cd /workspace/finance-agent-platform
git remote set-url origin git@github.com:thiago-ouverney/finance-agent-platform.git
```

Não copie a chave privada do seu computador pessoal para um Pod descartável. Ao terminar, remova a chave do Pod ou revogue-a no GitHub.

## Fluxo mínimo no RunPod

```bash
git clone https://github.com/thiago-ouverney/finance-agent-platform.git
cd finance-agent-platform/services/inference-runtime
make prepare-all MODEL_SIZE=7B
make bench-all MODEL_SIZE=7B
```

Para preparar ou medir somente um runtime, use `prepare-vllm`, `prepare-llama`, `prepare-ollama`, `bench-vllm`, `bench-llama` ou `bench-ollama`. Os alvos por runtime já dependem da preparação comum; não é necessário chamar `prepare-benchmark` antes. `prepare-all` prepara os três, `smoke-all` testa os três rapidamente e `bench-all` executa as baterias completas. O alvo `bench` é um alias de `bench-all`.

## Como o benchmark funciona

Depois da preparação, o comando principal é `bench-all`:

```text
prepare-<runtime> → inicia o servidor do runtime → verifica readiness
→ mede primeira resposta → aquece → mede requisições → coleta GPU/CPU/RAM
→ encerra somente o processo criado → grava tabelas, JSONs, CSVs, logs e gráficos
```

`bench-vllm`, `bench-llama` e `bench-ollama` executam esse ciclo para um runtime. Cada um roda a bateria base (`short`, `medium`, `long`) e depois o sweep de contexto/KV configurado. `bench-all` executa os três em sequência, continua mesmo se um falhar e retorna código diferente de zero no final quando houver falha. `bench` é apenas um alias de `bench-all`.

Para diagnosticar rapidamente instalação e carregamento, use `smoke-vllm`, `smoke-llama`, `smoke-ollama` ou `smoke-all`. O smoke usa poucas requisições no cenário `short`; ele valida servidor, alias, tokenizer, API, coleta e limpeza, mas não é uma medição formal comparável.

```bash
# Todos os runtimes
make prepare-all MODEL_SIZE=7B
make smoke-all MODEL_SIZE=7B PREPARE_OFFLINE=1
make bench-all MODEL_SIZE=7B PREPARE_OFFLINE=1

# Um runtime específico
make bench-vllm MODEL_SIZE=7B PREPARE_OFFLINE=1
```

Cada execução registra readiness, primeira resposta, warmup, medições, fases, argumentos efetivos, status de cada requisição e telemetria aproximadamente a cada segundo. Os artefatos ficam em `results/<runtime>/<timestamp>/<nome-humano>/`; a descrição completa da árvore está em [Telemetria e resultados](#telemetria-e-resultados).

## Modelo usado

O modelo 7B oficial desta rodada é [`arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`](https://huggingface.co/arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf). O `make prepare-all MODEL_SIZE=7B` baixa esse GGUF e o tokenizer base para o SSD e prepara os três runtimes. O tokenizer não é o peso: ele é necessário para construir prompts e contar tokens comparavelmente.

O download do tokenizer é feito pelo `huggingface_hub.snapshot_download` com filtros explícitos, e não pela combinação ambígua de nomes posicionais e `--include` da CLI `hf`. A preparação falha se `config.json`, `tokenizer_config.json` ou `tokenizer.json` não estiverem presentes; isso evita iniciar o llama.cpp com um diretório de tokenizer incompleto.

As configurações vLLM deste repositório usam por padrão `--gpu-memory-utilization 0.95`, reservando uma margem de 5% da VRAM. Esse parâmetro limita o orçamento de memória do executor; não limita a utilização computacional da GPU a 95%. O valor `1.0` falhou no Pod observado porque havia 23,30 GiB livres de 23,56 GiB, menos que o orçamento integral solicitado.

Para a rodada 14B:

```bash
make prepare-all MODEL_SIZE=14B
make bench MODEL_SIZE=14B
```

Se o arquivo estiver em outro caminho:

```bash
make prepare-all MODEL_SIZE=14B \
  MODEL_14B_GGUF=/workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
```

O 14B pode exceder a VRAM da RTX 3090. Isso é uma falha experimental válida: preserve o log, sem substituir automaticamente pelo 7B.

## O que cada alvo faz

### Preparação comum e por runtime

O fluxo recomendado é:

```bash
make prepare-vllm MODEL_SIZE=7B
make prepare-llama MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
```

Ou, para preparar os três de uma vez:

```bash
make prepare-all MODEL_SIZE=7B
```

Cada alvo `prepare-<runtime>` chama automaticamente `prepare-benchmark`, que é a etapa interna comum de modelo, tokenizer, cliente e configurações. Portanto, `prepare-benchmark` não precisa ser executado manualmente antes de um runtime.

### `make prepare-benchmark` (etapa comum interna)

Executa a preparação completa, nesta ordem:

1. baixa o GGUF selecionado para o SSD;
2. cria as pastas necessárias;
3. baixa `config.json`, `tokenizer_config.json`, tokenizer e arquivos especiais;
4. verifica assinatura `GGUF`, tamanho e SHA-256;
5. cria `results/`;
6. instala `requirements.txt` no Python indicado por `PYTHON`;
7. valida imports do cliente (`guidellm`, `httpx`, `psutil`, `transformers`);
8. valida os JSONs do modelo selecionado;
9. deixa a validação dos executáveis para o alvo específico de cada runtime.

Não inicia servidores e não mede desempenho. Para validar apenas um runtime:

```bash
make prepare-vllm MODEL_SIZE=7B
make prepare-llama MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
```

Assim, a ausência de vLLM não impede preparar o modelo para llama.cpp ou Ollama.

`prepare-llama` assume que a imagem do pod já fornece `llama-server` no `PATH` e valida esse executável. O template chama simplesmente `llama-server`, portanto não depende de um caminho específico de `/workspace`. Se a imagem não tiver o binário, use `make build-llama` explicitamente (com CMake e toolkit CUDA) ou instale o runtime no ambiente do pod. Por padrão, `REQUIRE_GPU=1`: a preparação falha se `llama-server --list-devices` não listar CUDA. Em imagens nas quais os backends são carregados como plugins, informe o caminho completo do arquivo `.so`, não apenas seu diretório:

```bash
make prepare-llama MODEL_SIZE=7B \
  LLAMA_BACKEND_PATH=/caminho/cuda_v13/libggml-cuda.so
```

O Make exporta esse arquivo em `GGML_BACKEND_PATH` e acrescenta seu diretório a `LD_LIBRARY_PATH`. `REQUIRE_GPU=0` existe somente para uma execução CPU conscientemente escolhida; não deve mascarar uma GPU ausente.

O mesmo princípio vale para os três runtimes: os templates chamam `vllm`, `llama-server` e `ollama` pelo `PATH`, sem assumir `/workspace/vllm-runtime/.venv` ou `/usr/local/bin/ollama`. Se a imagem usar outro local, coloque o diretório no `PATH` ou sobrescreva `VLLM_BIN`, `LLAMA_SERVER_BIN` ou `OLLAMA_BIN`.

Antes de iniciar o vLLM com GGUF, `prepare-vllm` executa `make check-vllm-gguf`. Essa verificação importa `vllm` e `vllm_gguf_plugin` usando o Python pertencente ao executável `vllm`. Wrappers nem sempre revelam seu interpretador pelo shebang; nesse caso, ou sempre que a detecção não for confiável, passe-o explicitamente com `VLLM_PYTHON=/caminho/do/python-do-vllm`. Se o plugin não estiver nesse ambiente, a preparação para com a instrução explícita:

```bash
/caminho/do/venv-do-vllm/bin/python -m pip install vllm-gguf-plugin
```

O erro `config file ... .gguf is not a valid JSON file` significa que o plugin não foi carregado; não significa que a assinatura GGUF esteja inválida. llama.cpp não precisa desse plugin: `llama-server` lê GGUF nativamente. Ollama também lê GGUF nativamente, mas precisa do daemon e do alias criado por `prepare-ollama`.

Use `make prepare-all MODEL_SIZE=7B` ou `make prepare-all MODEL_SIZE=14B`; a etapa comum é chamada automaticamente pelos alvos por runtime.

O cliente usa por padrão `.venv` na raiz do repositório. A criação e instalação podem ser executadas isoladamente:

```bash
make install-benchmark
```

Para usar outro local ou Python:

```bash
make install-benchmark \
  BENCH_VENV=/workspace/meu-bench/.venv \
  PYTHON=/workspace/meu-bench/.venv/bin/python
```

Não instale `requirements.txt` no venv do vLLM: ele pode trocar `torch`, `transformers` e dependências CUDA do runtime.

Para executar partes isoladas:

```bash
make download-model MODEL_SIZE=7B
make download-tokenizer MODEL_SIZE=7B
make verify-gguf MODEL_SIZE=7B
```

### `make prepare-ollama`

Cria o alias correspondente ao tamanho escolhido (`qwen7b-q8-gguf` ou `qwen14b-q8-gguf`) a partir do GGUF local usando um `Modelfile` temporário. Não executa `ollama pull`. Se já houver uma API Ollama na URL configurada, ela é preservada. Caso contrário, o script inicia seu próprio `ollama serve`, importa e verifica o alias e encerra somente esse daemon temporário. Já o benchmark lança um servidor sob seu controle e, por segurança, exige que a porta configurada esteja livre.

### Benchmarks individuais e agregado

```bash
make bench-vllm MODEL_SIZE=7B
make bench-llama MODEL_SIZE=7B
make bench-ollama MODEL_SIZE=7B
make bench-all MODEL_SIZE=7B
```

Cada alvo individual executa a bateria base e, imediatamente depois, o sweep de KV. O agregador tenta os três runtimes, continua se um falhar e retorna erro ao final se houver falha. A explicação do ciclo, fases, métricas e artefatos está em [Como o benchmark funciona](#como-o-benchmark-funciona); esta seção mantém apenas os comandos específicos.

### Smoke test

```bash
make smoke-vllm MODEL_SIZE=7B
make smoke-llama MODEL_SIZE=7B
make smoke-ollama MODEL_SIZE=7B
make smoke-all MODEL_SIZE=7B
```

Os três smokes usam poucas requisições e somente `short`; servem para diagnosticar instalação, porta, tokenizer, readiness, telemetria e carregamento. Não são resultados finais nem substituem `bench-all`.

Com o ambiente e os pesos já preparados no SSD, acrescente `PREPARE_OFFLINE=1` para evitar instalações/downloads repetidos e `BENCH_STARTUP_TIMEOUT=300` para limitar a espera de startup. Os Makefiles exigem GNU Make com `.ONESHELL` (3.82+); o Make 3.81 padrão do macOS é recusado. Consulte o [diagnóstico histórico no RunPod](../../docs/archive/inference-runtime/diagnostico-makefiles.md) para causas, comandos e cobertura dos testes daquela execução.

### Sweep rápido de integração

```bash
make quick-sweep-vllm MODEL_SIZE=7B
make quick-sweep-llama MODEL_SIZE=7B
make quick-sweep-ollama MODEL_SIZE=7B
```

Cada alvo usa por padrão início em 1024, passo de 256 MB de KV lógico, máximo 16384 e uma repetição. O número equivalente de tokens é calculado pela arquitetura: aproximadamente 4465 tokens por salto no Qwen2.5 7B. Para uma verificação rápida, use `quick-sweep-<runtime>`; ela limita o contexto a 8192, usa uma requisição, uma repetição e nenhum aquecimento. Para validar apenas um ponto, use `SWEEP_MAX_CONTEXT=1024`. É uma verificação de integração do launcher, API e coleta; a amostra não sustenta comparação estatística nem substitui o sweep formal.

### Estado da validação no Pod

A integração foi validada ao vivo em 22/09/2026 no RunPod, com o mesmo GGUF 7B e uma RTX 3090. O comando `bench-all` reduzido terminou com:

```text
[BENCH ALL] status: vLLM=0 llama.cpp=0 Ollama=0
```

Os três smokes, os três quick sweeps de 1024 tokens, as preparações offline e a bateria agregada reduzida passaram. O relatório datado dessa rodada foi preservado em `docs/archive/inference-runtime/` na raiz do monorepo.

O vLLM precisou da configuração diagnóstica `--enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.80`; o startup levou aproximadamente 96 segundos. O `EngineDeadError` registrado no encerramento do sweep ocorreu depois de respostas HTTP 200 e do SIGTERM intencional de shutdown; o alvo terminou com código zero. Isso não foi classificado como OOM ou falha de inferência.

Durante uma inicialização demorada, o progresso informa o endpoint `GET /v1/models`, o ID de modelo esperado e a última observação/erro. Uma resposta HTTP sem o alias correto não conta como disponibilidade.

## O que o benchmark mede

Por runtime e cenário, são executadas 50 requisições de medição, 1 repetição e 3 aquecimentos. A bateria base usa `replay`: as 50 chamadas percorrem 50 perguntas técnicas distintas da fixture versionada, com histórico determinístico. O warmup usa outra fixture e nunca reutiliza os prompts medidos. A execução é síncrona: uma requisição ativa por vez.

| Cenário | Entrada aproximada | Saída máxima |
|---|---:|---:|
| `short` | 256 tokens | 128 tokens |
| `medium` | 2048 tokens | 128 tokens |
| `long` | 8192 tokens | 128 tokens |

Métricas principais:

- **Time To First Token (ms)**: do POST até o primeiro conteúdo/token observado; p50, p95 e p99 resumem requisições diferentes.
- **Tokens/s**: velocidade durante o decode, calculada pelos intervalos entre tokens; não inclui TTFT.
- **Effective tokens/s**: tokens de saída divididos pela duração total, incluindo TTFT.

As taxas oficiais são calculadas no cliente, por requisição, usando `time.perf_counter()` e a resposta SSE. O benchmark registra `request_start_time`, `first_token_time`, `request_end_time`, `time_to_first_token_seconds`, `generation_time_seconds`, `end_to_end_latency_seconds`, `completion_tokens`, `prompt_tokens`, `total_tokens`, `decode_tokens_per_second` e `end_to_end_tokens_per_second`. As fórmulas são: TTFT = primeiro evento de conteúdo − início; geração = último evento de conteúdo − primeiro; decode tokens/s = tokens de saída ÷ geração; tokens/s efetivos = tokens de saída ÷ latência total. O throughput periódico `Avg generation throughput` do vLLM é apenas telemetria agregada em janelas do servidor e nunca entra nessas métricas.

Como a API SSE não garante um evento por token, `first_token_time` e `request_end_time` são observados no cliente e `generation_time_seconds` usa o primeiro/último evento SSE com conteúdo. A contagem de tokens vem somente de `usage`; se `usage` faltar, contagens e taxas ficam `null`, sem estimativa. Para uma saída de um token, `inter_token_latency` também fica `null`.

Warmup, primeira resposta, medição formal e encerramento são fases distintas. Falhas e incompletas não entram nas velocidades válidas.

## Sweep de KV cache

O sweep faz parte de `bench-vllm`, `bench-llama` e `bench-ollama`. Para cada runtime, o servidor é reiniciado começando em 1024 tokens e avançando 256 MB de KV lógico por ponto. No Qwen2.5 7B isso equivale a aproximadamente 4465 tokens por salto: 1024, 5489, 9954 e 14419, até a primeira falha de inicialização ou memória.

No vLLM também é coletado o gauge de ocupação do pool KV em `/metrics`. Esse percentual não é percentual de VRAM nem bytes físicos; a VRAM é observada separadamente pelo `nvidia-smi`.

## Parâmetros dos runtimes

Argumentos extras são anexados ao `argv` e registrados no `lifecycle.json`:

```bash
make bench-vllm MODEL_SIZE=7B \
  VLLM_EXTRA_ARGS='--gpu-memory-utilization 0.95 --enforce-eager'

make bench-llama MODEL_SIZE=7B \
  LLAMA_EXTRA_ARGS='--flash-attn auto --threads 8'

make bench-ollama MODEL_SIZE=7B OLLAMA_EXTRA_ARGS=''
```

O Make não corrige flags inválidas. A falha deve aparecer no `server.log`.

Variáveis principais:

```text
MODEL_SIZE=7B|14B
MODEL_7B_GGUF / MODEL_14B_GGUF
HF_GGUF_REPO / HF_GGUF_FILENAME
HF_TOKENIZER_MODEL
VLLM_EXTRA_ARGS / LLAMA_EXTRA_ARGS / OLLAMA_EXTRA_ARGS
BENCH_REQUESTS / BENCH_REPETITIONS / BENCH_WARMUP
BENCH_MODE=replay|closed-loop|independent
CONVERSATION_TURNS / CONVERSATION_FIXTURE / WARMUP_CONVERSATION_FIXTURE
```

`replay` é o padrão da bateria base: envia histórico determinístico e respostas `assistant` fixas, percorrendo os turnos da fixture em vez de repetir sempre a primeira pergunta. `closed-loop` usa o histórico, mas coloca a resposta real do runtime no turno seguinte. `independent` envia mensagens isoladas e, quando solicitado, cria um prompt determinístico distinto por requisição, mantendo o mesmo comprimento-alvo e conjuntos disjuntos entre warmup e medição para reduzir viés de cache de prefixo. A medição conversacional usa `CONVERSATION_FIXTURE` (`qwen_chat_bench_v2.json`, 50 turnos); o warmup usa `WARMUP_CONVERSATION_FIXTURE` (`qwen_chat_warmup_v1.json`), deliberadamente diferente. Ambos são JSON com `system` e `turns`; replay exige `assistant` fixo nos turnos usados.

## Telemetria e resultados

Cada execução cria uma árvore legível, sem misturar formatos:

```text
results/<runtime>/<timestamp>/<nome-humano>/
├── html/   summary.html, lifecycle.html, telemetry-memory.svg, telemetry-gpu-util.svg
├── json/   manifestos, métricas, respostas brutas e artefatos de conversa
├── csv/    summary.csv, gpu.csv, system.csv, events.csv e requests.csv
├── logs/   server.log e telemetry.log
└── text/   artefatos sem extensão conhecida
```

O `summary.html` incorpora gráficos offline de memória ocupada da GPU (MiB) e de GPU-utilização (%) ao longo do tempo, além das tabelas por fase. `telemetry-timeseries.csv` mantém a série temporal para análise externa; `gpu-summary.json` e `telemetry-summary.json` mantêm as agregações. Ausência de amostra aparece como indisponível, nunca como zero.

O nome humano é derivado do alvo, por exemplo `smoke-short`, `benchmark-short-medium-long` ou `contexto-1024-tokens`. Para uma execução manual, use `--result-name nome-descritivo`.

Os eventos relacionam picos de VRAM/CPU a startup, primeira requisição, aquecimento, medição ou encerramento. O instrumento não mede diretamente o tempo de cada transferência PCIe/RAM↔VRAM; isso exige Nsight/CUDA instrumentation.

### Baixar resultados do pod

Execute no computador local, não dentro do pod:

```bash
make pull-results POD_SSH=root@HOST_DO_POD
```

Com uma porta SSH diferente:

```bash
make pull-results POD_SSH=root@HOST_DO_POD POD_PORT=2222 \
  REMOTE_RESULTS_DIR=/workspace/finance-agent-platform/services/inference-runtime/results \
  LOCAL_RESULTS_DIR=results-pod
```

O alvo detecta `ssh.runpod.io` automaticamente e usa o relay oficial `runpodctl`; para hosts SSH TCP diretos, usa `scp`. Em ambos os casos, copia todas as execuções para `LOCAL_RESULTS_DIR` e não apaga arquivos locais. A chave padrão é `~/.ssh/id_ed25519`; altere com `POD_SSH_KEY=/caminho/chave`.

Com o alvo `make pull-results`, a transferência pelo RunPod é automática: o Make cria um tar temporário no pod, inicia `runpodctl send`, captura o código de uso único e executa `runpodctl receive` na máquina local. Instale o cliente localmente com `brew install runpod/runpodctl/runpodctl`.

No pod:

```bash
runpodctl send /tmp/llm-local-inference-results.tar.gz
```

Na máquina local, entre em `~/Documents` e use o código exibido:

```bash
cd ~/Documents
runpodctl receive CODIGO_EXIBIDO
tar -xzf llm-local-inference-results.tar.gz \
  -C inference-runtime-results --strip-components=1
```

O `runpodctl send/receive` funciona por código de transferência e não depende de uma sessão SCP tradicional. Relatórios de rodadas anteriores ficam em `docs/archive/inference-runtime/` na raiz do monorepo.

### Profiling para memory-bound/compute-bound

`gpu.csv` é uma série temporal de baixa intrusão. Para um roofline quantitativo, use uma execução separada com Nsight Systems (`nsys`) e Nsight Compute (`ncu`), pois eles medem kernels, FLOPs, bytes e duração com overhead. Verifique a instalação no pod com:

```bash
make check-profilers
```

Não misture Nsight com a bateria oficial de TTFT/Tokens/s: o profiler altera o tempo observado. Registre esse profiling como experimento complementar, com o mesmo modelo, prompt e contexto.

#### Makefile de profiling separado

O arquivo `Makefile.profiling` é deliberadamente separado do Makefile principal. Use-o em outro ambiente/execução, com `nsys` ou `ncu` instalados. Ele não instala nem remove o Nsight automaticamente e não deve ser usado para produzir os números oficiais de latência.

Exemplos no pod:

```bash
make -f Makefile.profiling check
make -f Makefile.profiling profile-vllm-nsys MODEL_SIZE=7B
```

O servidor fica em primeiro plano. Em outro terminal do pod, envie uma única requisição curta:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen7b-q8-gguf","messages":[{"role":"user","content":"Explique RAM e VRAM em duas frases."}],"max_tokens":64,"temperature":0,"stream":false}'
```

Depois interrompa o servidor com `Ctrl-C`. O relatório fica em `results/profiling/`. Para Nsight Compute, use `profile-vllm-ncu` ou `profile-llama-ncu`, faça uma única requisição e interrompa após os kernels desejados. O resultado `.ncu-rep` contém os contadores para estimar FLOPs, bytes de DRAM, FLOP/s, GB/s e intensidade aritmética.

## Regras de validade

- mesmo GGUF, tokenizer, conjunto determinístico de prompts, temperatura e limite de saída nos três runtimes;
- uma GPU limpa e nenhum servidor concorrente;
- nenhum download durante `bench`;
- smoke não é comparável à bateria formal;
- OOM, erro de loader, API ou tokenizer são preservados como falhas;
- comparação somente entre cenário, repetição e fase equivalentes.

## Diagnóstico rápido

```bash
make prepare-vllm MODEL_SIZE=7B
make smoke-vllm MODEL_SIZE=7B
```

Se o servidor encerrar:

```bash
cat results/<runtime>/<timestamp>/<nome-humano>/logs/server.log
cat results/<runtime>/<timestamp>/<nome-humano>/json/lifecycle.json
nvidia-smi
```

Uma falha é diagnóstico, não desempenho. Corrija a causa e repita o mesmo protocolo.
