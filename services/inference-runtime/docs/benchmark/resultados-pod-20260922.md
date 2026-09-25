# Resultados do Pod — 22/09/2026

## Escopo

Rodada histórica de integração no RunPod `m87zg5huybxfa7`, com uma RTX 3090 de 24 GB, usando o GGUF local `Qwen2.5-7B-Instruct-Q8_0.gguf`, tokenizer local e modo `independent`. O objetivo foi validar o fluxo anterior do Makefile e a execução real dos três runtimes; não é uma bateria estatística nem representa o protocolo replay atual.

## Rodada atual após replay e warmup separado

Em 22/09/2026, a implementação atual foi validada no mesmo Pod, já no commit `f9ff2ad`. Primeiro passaram os testes unitários (`17/17`) e a integração mock usando `/workspace/llm_local_inference_masters/.venv/bin/python`. Depois passaram os três smokes reais e uma bateria reduzida:

```bash
make bench-all MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  BENCH_MODE=replay BENCH_SCENARIOS=short BENCH_REQUESTS=5 \
  BENCH_REPETITIONS=1 BENCH_WARMUP=2 \
  SWEEP_MODE=replay SWEEP_START=1024 SWEEP_MAX_CONTEXT=1024 \
  SWEEP_REQUESTS=1 SWEEP_REPETITIONS=1 SWEEP_WARMUP=0
```

Resultado real:

```text
[BENCH ALL] status: vLLM=0 llama.cpp=0 Ollama=0
```

Os artefatos confirmaram, nos três runtimes:

- `manifest.json` com `mode=replay`;
- hash da fixture de medição diferente do hash da fixture de warmup;
- 5 requests de medição distintos e sem interseção com os 2 warmups;
- sweep com `memory_step_mb=256`, `token_step=4465` e ponto `input_tokens=1024`;
- HTML, JSON, CSV, logs, gráfico de memória e gráfico de GPU-util preservados.

Diretórios principais no Pod:

```text
results/vllm/20260922T175441.343339Z/benchmark-short/
results/vllm/20260922T175637.324381Z/contexto-1024-tokens-step-256mb/
results/llama.cpp/20260922T175837.984919Z/benchmark-short/
results/llama.cpp/20260922T175910.471376Z/contexto-1024-tokens-step-256mb/
results/ollama/20260922T175950.513323Z/benchmark-short/
results/ollama/20260922T180043.241969Z/contexto-1024-tokens-step-256mb/
```

Essa foi uma validação funcional reduzida, não a bateria formal de 50 requests por cenário e quatro pontos de sweep. A bateria formal permanece disponível para execução posterior no Pod.

O comando agregado reduzido foi:

```bash
make bench-all MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  BENCH_SCENARIOS=short BENCH_REQUESTS=1 BENCH_REPETITIONS=1 BENCH_WARMUP=0 \
  SWEEP_MAX_CONTEXT=1024 SWEEP_REQUESTS=1 SWEEP_REPETITIONS=1 SWEEP_WARMUP=0 \
  BENCH_STARTUP_TIMEOUT=300 \
  VLLM_PYTHON=/app/.vllm_venv/bin/python \
  VLLM_EXTRA_ARGS='--enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.80' \
  LLAMA_BACKEND_PATH=/usr/local/lib/ollama/cuda_v12/libggml-cuda.so
```

Resultado agregado:

```text
[BENCH ALL] status: vLLM=0 llama.cpp=0 Ollama=0
```

## Métricas observadas

Cada linha abaixo tem uma única requisição válida (`successful_request_count=1`). Os valores são exploratórios e não devem ser tratados como p95/p99 estáveis.

| Runtime | Fase/cenário | Entrada | Saída | TTFT | Decode | Efetivo |
|---|---|---:|---:|---:|---:|---:|
| vLLM | `measure/short` | 285 tokens | 128 | 733,3 ms | 48,62 tok/s | 38,26 tok/s |
| llama.cpp | `measure/short` | 285 tokens | 128 | 192,3 ms | 42,03 tok/s | 39,82 tok/s |
| Ollama | `measure/short` | 285 tokens | 128 | 460,3 ms | 40,30 tok/s | 35,44 tok/s |
| vLLM | `measure/ctx1024` | 1053 tokens | 89 | 2629,4 ms | 51,43 tok/s | 20,41 tok/s |
| llama.cpp | `measure/ctx1024` | 1053 tokens | 88 | 282,1 ms | 41,95 tok/s | 37,15 tok/s |
| Ollama | `measure/ctx1024` | 1053 tokens | 88 | 517,5 ms | 45,28 tok/s | 35,20 tok/s |

As métricas foram lidas dos `summary.json` locais. A VRAM observada durante as requisições ficou aproximadamente em 8,3 GiB para llama.cpp, 8,5 GiB para Ollama e 18,9 GiB para vLLM na configuração diagnóstica acima.

## Artefatos

Os artefatos completos foram preservados fora do Git em:

```text
~/Documents/llm_local_inference_masters-results/
```

Esta foi a rodada executada antes da adoção do layout categorizado; novas execuções passam a separar runtime, timestamp, nome humano e tipo de arquivo conforme o README.

Principais diretórios da rodada:

```text
20260922T153020.635497Z/              # bateria base vLLM
kv-sweep-vllm/20260922T153214.761385Z/
20260922T153417.801452Z/              # bateria base llama.cpp
kv-sweep-llama/20260922T153437.688254Z/
20260922T153516.631067Z/              # bateria base Ollama
kv-sweep-ollama/20260922T153557.011691Z/
```

Cada diretório contém `summary.html`, `lifecycle.html`, `summary.json`, `manifest.json`, `server.log`, telemetria e os CSVs brutos.

## Interpretação do log do vLLM

O `EngineDeadError` observado no `server.log` do sweep vLLM ocorre depois de `POST /v1/chat/completions 200 OK` e depois de `[shutdown] ... SIGTERM`. É uma mensagem do vLLM 0.29.0 durante a desmontagem solicitada pelo harness; o resultado foi concluído e o código do alvo foi zero. Não foi observado OOM nem falha de carregamento.

## Validação do Makefile

Passaram no Pod nesta atualização: `python3 -m unittest tests/test_bench.py` (`17/17`), integração mock com `.venv/bin/python`, preparação offline dos três runtimes, os três smokes reais, `bench-all` reduzido em replay, validação dos manifests/fixtures e dry-run dos grupos de alvos. O `python3` do sistema não possui `guidellm`; a integração foi executada com o `.venv` oficial do benchmark.

`nsys` não está instalado no pod. O `ncu` existe, mas os contadores estão bloqueados pelo host (`ERR_NVGPUCTRPERM`); profiling real continua uma limitação externa.

## Transferência dos resultados

O endereço `ssh.runpod.io` usado pelo terminal do RunPod é um proxy com PTY e não oferece SCP/SFTP. O alvo `pull-results` continua útil quando há SSH TCP direto, mas para este pod a transferência foi feita com `runpodctl send` no pod e `runpodctl receive` na máquina local. O pacote compactado original está em `~/Documents/llm-local-inference-results.tar.gz`.
