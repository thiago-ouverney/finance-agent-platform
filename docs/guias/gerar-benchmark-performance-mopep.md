# Gerar o benchmark MOPEP entre runtimes

> **Objetivo:** comparar vLLM, llama.cpp e Ollama com o mesmo GGUF usando BMCs
> reais, coleta Prometheus e três perfis: classificação, revisão controlada e
> revisão closed-loop.

## 1. Gere a carga privada

Na máquina local, use `calibration.csv` durante o desenvolvimento. Fixe uma
revisão imutável do tokenizer:

```bash
make mopep-performance-workload \
  MOPEP_PERF_SPLIT=calibration \
  MOPEP_PERF_TOKENIZER_REVISION='<commit-do-tokenizer>'
```

O alvo cria em `/tmp/mopep-runtime-workload/calibration/` o workload medido,
um warm-up vindo de `train.csv`, manifesto e checksums. `short`, `medium` e
`heavy` são os tercis do número de tokens do prompt completo na calibração.
Nenhum JSONL contém as tags esperadas.

Envie o pacote ao Pod sem versioná-lo:

```bash
make push-mopep-performance-workload REMOTE_HOST=runpod-qwen
```

## 2. Gere o replay canônico

No Pod, em `services/inference-runtime`, execute primeiro llama.cpp com o mesmo
GGUF que será usado nos três runtimes:

```bash
make observe-bench \
  OBS_RUNTIME=llama \
  OBS_MODEL_SOURCE=local-gguf \
  OBS_MODEL=/workspace/models/modelo.gguf \
  OBS_BENCH_PROFILE=mopep-single \
  OBS_REQUEST_DATASET=/workspace/data/mopep-runtime-workload/calibration/workload.jsonl \
  OBS_WARMUP_REQUEST_DATASET=/workspace/data/mopep-runtime-workload/calibration/warmup.jsonl \
  OBS_REQUEST_MANIFEST=/workspace/data/mopep-runtime-workload/calibration/workload-manifest.json \
  OBS_RESPONSES_OUT=/workspace/data/mopep-runtime-workload/calibration/canonical-responses.jsonl \
  OBS_RESULT_NAME=mopep-single-llama
```

Essa execução também é a rodada oficial `single-pass` do llama.cpp. O arquivo
canônico é privado, tem uma resposta por BMC/turno e alimenta o replay dos três
runtimes. O sidecar `canonical-responses.manifest.json` prova runtime, workload,
GGUF e SHA-256; o replay recusa origem diferente de llama.cpp, outro workload
ou outro GGUF. Não o gere com o golden nem o altere depois da campanha.

## 3. Execute as nove combinações

Execute sequencialmente, nunca em paralelo:

| Perfil | vLLM | llama.cpp | Ollama |
|---|---:|---:|---:|
| `mopep-single` | executar | já executado/confirmar | executar |
| `mopep-review-replay` | executar | executar | executar |
| `mopep-review-closed-loop` | executar | executar | executar |

Repita o comando anterior alterando `OBS_RUNTIME`, `OBS_BENCH_PROFILE` e
`OBS_RESULT_NAME`. Para `mopep-review-replay`, acrescente:

```bash
OBS_REPLAY_RESPONSES=/workspace/data/mopep-runtime-workload/calibration/canonical-responses.jsonl
```

`single` faz uma chamada por BMC. `review-replay` faz somente a revisão com a
mesma resposta anterior canônica. `review-closed-loop` faz duas chamadas e usa
a resposta real do primeiro turno; por isso seu segundo request pode divergir
entre runtimes e representa produto, não isolamento puro do runtime.

## 4. Baixe e analise

Na máquina local:

```bash
cd services/inference-runtime
make pull-observe-results OBS_REMOTE_HOST=runpod-qwen
cd ../..
make mopep-performance-dataset \
  MOPEP_PERF_SPLIT=calibration \
  MOPEP_PERF_CANONICAL_RESPONSES=/caminho/canonical-responses.jsonl
make mopep-performance-notebook
```

O comparativo apresenta Macro-F1, exact match, parsing, melhoria/regressão,
TTFT, tokens/s, latência e telemetria por runtime, bucket e turno.

## 5. Faça a campanha final

Depois de congelar prompt, heurísticas, GGUF e parâmetros, gere o workload de
teste reutilizando os limites da calibração:

```bash
make mopep-performance-workload \
  MOPEP_PERF_SPLIT=test \
  MOPEP_PERF_TOKENIZER_REVISION='<mesmo-commit>' \
  MOPEP_PERF_THRESHOLD_MANIFEST=/tmp/mopep-runtime-workload/calibration/workload-manifest.json
```

Repita as nove combinações sem ajustar decisões após observar o teste.

## Como testar o fluxo

```bash
make -C services/inference-runtime install-benchmark prometheus-install
make test-analytics
make test-inference
```

No Pod, use primeiro `OBS_BENCH_SMOKE=1`. Uma campanha só é válida quando as
runs estão `complete`, hashes conferem, os três runtimes compartilham o mesmo
`comparison_id` dentro de cada perfil e não há labels no workload enviado.

Contrato técnico: [workload MOPEP](../../services/inference-runtime/docs/benchmark/workload-mopep.md).
