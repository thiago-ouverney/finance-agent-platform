# Gerar o benchmark MOPEP entre runtimes

> **Objetivo:** comparar vLLM, llama.cpp e Ollama com o mesmo GGUF usando BMCs
> reais, coleta Prometheus e três perfis: classificação, revisão controlada e
> revisão closed-loop.

## 1. Gere a carga privada

Na máquina local, use `calibration.csv` durante o desenvolvimento. Fixe uma
revisão imutável do tokenizer:

```bash
make setup-notebook
```

Esse preparo instala também a versão de `transformers` usada para materializar
os buckets com o tokenizer real. Em um ambiente já criado, execute novamente o
alvo para alinhar as dependências antes de gerar a amostra.

```bash
make mopep-performance-workload \
  MOPEP_PERF_SPLIT=calibration \
  MOPEP_PERF_TOKENIZER_REVISION='<commit-do-tokenizer>'
```

O alvo cria em `/tmp/mopep-runtime-workload/calibration/` o workload medido,
um warm-up vindo de `train.csv`, manifesto e checksums. `short`, `medium` e
`heavy` são os tercis do número de tokens do prompt completo na calibração.
`heavy` corresponde ao grupo chamado informalmente de `large` ou `high`.
Nenhum JSONL contém as tags esperadas.

Envie uma carga completa ao Pod sem versioná-la:

```bash
make push-mopep-performance-workload REMOTE_HOST=runpod-qwen
```

### Piloto reproduzível com 30 IDs do teste

Antes de usar o teste, congele prompt, GGUF, tokenizer e parâmetros. A amostra
piloto usa 10 IDs de cada bucket. Os limites continuam vindo da calibração;
eles nunca são recalculados sobre os 30 escolhidos.

Gere primeiro o manifesto completo da calibração com o comando anterior.
Depois gere a amostra de teste:

```bash
make mopep-performance-workload \
  MOPEP_PERF_SPLIT=test \
  MOPEP_PERF_WORKLOAD_DIR=/tmp/mopep-runtime-workload/test-30 \
  MOPEP_PERF_TOKENIZER_REVISION='<mesmo-commit>' \
  MOPEP_PERF_THRESHOLD_MANIFEST=/tmp/mopep-runtime-workload/calibration/workload-manifest.json \
  MOPEP_PERF_SAMPLE_PER_BUCKET=10 \
  MOPEP_PERF_SAMPLE_SEED=42 \
  MOPEP_PERF_SAMPLE_DATASET=/tmp/mopep-runtime-workload/test-30/test-30.csv
```

O resultado contém:

```text
/tmp/mopep-runtime-workload/test-30/
├── test-30.csv              # golden local dos 30 IDs, contém tags
├── workload.jsonl           # entrada privada do Pod, não contém tags
├── warmup.jsonl
├── workload-manifest.json
└── SHA256SUMS
```

Valide antes do envio:

```bash
jq '{request_count,bucket_counts,sampling,privacy}' \
  /tmp/mopep-runtime-workload/test-30/workload-manifest.json
(cd /tmp/mopep-runtime-workload/test-30 && sha256sum --check --strict SHA256SUMS)
```

O esperado é `request_count=30` e `bucket_counts` igual a `short=10`,
`medium=10`, `heavy=10`. O CSV local preserva as tags para a avaliação
posterior. O alvo de envio transfere somente workload, warm-up, manifesto e
checksums, todos sem labels.

Na raiz do repositório local, envie a amostra:

```bash
make push-mopep-performance-workload \
  REMOTE_HOST=runpod-qwen \
  MOPEP_PERF_SPLIT=test \
  MOPEP_PERF_WORKLOAD_DIR=/tmp/mopep-runtime-workload/test-30 \
  MOPEP_PERF_REMOTE_DIR=/workspace/data/mopep-runtime-workload/test-30
```

## 2. Execute em um Pod já preparado

No Pod usado anteriormente, confirme GPU, repositório e GGUF. Não recompile
nem baixe outro modelo durante a janela medida:

```bash
nvidia-smi
cd /workspace/finance-agent-platform/services/inference-runtime
git rev-parse HEAD
test -f /workspace/models/modelo.gguf
```

Exporte as condições comuns. Ajuste apenas o caminho do GGUF e use o mesmo
commit de tokenizer usado para gerar a amostra:

```bash
export OBS_MODEL_SOURCE=local-gguf
export OBS_MODEL=/workspace/models/modelo.gguf
export OBS_TOKENIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
export OBS_TOKENIZER_REVISION='<mesmo-commit-fixado-localmente>'
export OBS_CONTEXT=8192
export OBS_REQUEST_DATASET=/workspace/data/mopep-runtime-workload/test-30/workload.jsonl
export OBS_WARMUP_REQUEST_DATASET=/workspace/data/mopep-runtime-workload/test-30/warmup.jsonl
export OBS_REQUEST_MANIFEST=/workspace/data/mopep-runtime-workload/test-30/workload-manifest.json
export OBS_BENCH_REPETITIONS=1
```

Faça primeiro um smoke em cada runtime. Smoke valida loader e integração, mas
não entra na comparação formal:

```bash
make observe-bench OBS_RUNTIME=llama OBS_BENCH_PROFILE=mopep-single \
  OBS_BENCH_SMOKE=1 OBS_RESULT_NAME=smoke-test30-single-llama
make observe-bench OBS_RUNTIME=vllm OBS_BENCH_PROFILE=mopep-single \
  OBS_BENCH_SMOKE=1 OBS_RESULT_NAME=smoke-test30-single-vllm
make observe-bench OBS_RUNTIME=ollama OBS_BENCH_PROFILE=mopep-single \
  OBS_BENCH_SMOKE=1 OBS_RESULT_NAME=smoke-test30-single-ollama
```

## 3. Execute one-shot, replay e closed-loop

Execute primeiro o single-pass formal do llama.cpp. Ele também gera a resposta
canônica exigida pelo replay:

```bash
make observe-bench OBS_RUNTIME=llama OBS_BENCH_PROFILE=mopep-single \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-single-llama \
  OBS_RESPONSES_OUT=/workspace/data/mopep-runtime-workload/test-30/canonical-responses.jsonl
```

Complete o one-shot nos outros runtimes:

```bash
make observe-bench OBS_RUNTIME=vllm OBS_BENCH_PROFILE=mopep-single \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-single-vllm
make observe-bench OBS_RUNTIME=ollama OBS_BENCH_PROFILE=mopep-single \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-single-ollama
```

`mopep-single` é o one-shot. Cada comando percorre internamente `short`,
`medium` e `heavy`; não execute um comando separado por bucket.

Execute a revisão controlada usando a mesma resposta canônica:

```bash
make observe-bench OBS_RUNTIME=llama OBS_BENCH_PROFILE=mopep-review-replay \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-review-replay-llama \
  OBS_REPLAY_RESPONSES=/workspace/data/mopep-runtime-workload/test-30/canonical-responses.jsonl
make observe-bench OBS_RUNTIME=vllm OBS_BENCH_PROFILE=mopep-review-replay \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-review-replay-vllm \
  OBS_REPLAY_RESPONSES=/workspace/data/mopep-runtime-workload/test-30/canonical-responses.jsonl
make observe-bench OBS_RUNTIME=ollama OBS_BENCH_PROFILE=mopep-review-replay \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-review-replay-ollama \
  OBS_REPLAY_RESPONSES=/workspace/data/mopep-runtime-workload/test-30/canonical-responses.jsonl
```

Por último, execute o closed-loop. Ele realiza duas chamadas por ID:

```bash
make observe-bench OBS_RUNTIME=llama OBS_BENCH_PROFILE=mopep-review-closed-loop \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-review-closed-loop-llama
make observe-bench OBS_RUNTIME=vllm OBS_BENCH_PROFILE=mopep-review-closed-loop \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-review-closed-loop-vllm
make observe-bench OBS_RUNTIME=ollama OBS_BENCH_PROFILE=mopep-review-closed-loop \
  OBS_BENCH_SMOKE=0 OBS_RESULT_NAME=test30-mopep-review-closed-loop-ollama
```

Execute sequencialmente, nunca em paralelo:

| Perfil | Chamadas por runtime | Chamadas nos três runtimes |
|---|---:|---:|
| `mopep-single` | 30 | 90 |
| `mopep-review-replay` | 30 | 90 |
| `mopep-review-closed-loop` | 60 | 180 |

Com o Pod preparado, reserve de 40 a 70 minutos para as nove runs formais e
cerca de 90 minutos incluindo smokes, verificações e transferências.

## 4. Baixe e analise

Na máquina local:

```bash
cd services/inference-runtime
make pull-observe-results OBS_REMOTE_HOST=runpod-qwen
cd ../..
```

Baixe também o replay canônico. Ele fica fora da árvore de resultados:

```bash
mkdir -p /tmp/mopep-runtime-workload/test-30
ssh runpod-qwen \
  "tar -C /workspace/data/mopep-runtime-workload/test-30 -czf - canonical-responses.jsonl canonical-responses.manifest.json" \
  | tar -C /tmp/mopep-runtime-workload/test-30 -xzf -
```

Una qualidade e observabilidade usando o golden local dos 30 IDs:

```bash
make mopep-performance-dataset \
  MOPEP_PERF_SPLIT=test \
  MOPEP_PERF_GOLDEN=/tmp/mopep-runtime-workload/test-30/test-30.csv \
  MOPEP_PERF_CANONICAL_RESPONSES=/tmp/mopep-runtime-workload/test-30/canonical-responses.jsonl \
  MOPEP_PERF_DATASET_DIR=analytics/data/mopep-runtime-comparison-test-30

make mopep-performance-notebook \
  MOPEP_PERF_DATASET_DIR=analytics/data/mopep-runtime-comparison-test-30
```

Avalie nesta ordem:

1. confirme nove runs formais com `status=complete` e hashes válidos;
2. confirme 10 requisições por bucket em `single` e `review-replay`, e 20 por
   bucket em `closed-loop`;
3. dentro de cada perfil, compare somente runtimes com o mesmo `comparison_id`;
4. examine parsing, Macro-F1 e exact match por bucket e runtime;
5. compare o delta da revisão contra o single-pass;
6. avalie TTFT, decode, latência ponta a ponta e recursos por bucket e turno.

O comparativo apresenta Macro-F1, exact match, parsing, melhoria/regressão,
TTFT, tokens/s, latência e telemetria. Com 10 observações por bucket, use p50 e
os valores individuais. P95 e p99 são exploratórios, e Macro-F1 pode variar
muito quando uma tag rara aparece em poucos exemplos.

A campanha de 30 é um gate. A conclusão final continua exigindo os 150 IDs do
teste sem alterar decisões após observar a amostra.

## 5. Faça a campanha final

Depois do piloto, gere o workload completo de teste reutilizando os mesmos
limites da calibração:

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

Uma campanha só é válida quando as runs estão `complete`, hashes conferem, os
três runtimes compartilham o mesmo `comparison_id` dentro de cada perfil e não
há labels no workload enviado.

Contrato técnico: [workload MOPEP](../../services/inference-runtime/docs/benchmark/workload-mopep.md).
