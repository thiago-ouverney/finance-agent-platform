# Observabilidade headless e interativa com Prometheus

> **Objetivo:** documentar o contrato de coleta temporal do `observe-bench` e
> do perfil interativo `observe-*`.

Guia de execução: [gerar o benchmark de
performance](../../../../docs/guias/gerar-benchmark-performance.md). Para uma
inspeção manual: [perfilar uma inferência
interativamente](../../../../docs/guias/perfilar-inferencia-prometheus.md).

## Dois modos, a mesma semântica

| Modo | Uso | Onde ocorre a análise |
|---|---|---|
| `make observe-bench` | bateria sequencial reproduzível | notebook local, depois do download |
| `observe-serve` + `observe-prometheus-start` + `observe-jupyter` | uma ou poucas requisições exploratórias | notebook e kernel no Pod |

O modo headless é o padrão para formar distribuições e comparar runtimes. O
modo interativo serve para explorar uma hipótese; uma requisição isolada não é
uma comparação estatística.

## Ciclo headless

```text
preparar runtime e modelo
        -> iniciar exporter e Prometheus
        -> iniciar runtime e aguardar /v1/models
        -> warmup
        -> bench.py: uma requisição por vez
        -> consultar e associar séries às requisições/fases
        -> gravar dataset e manifesto
        -> encerrar apenas os processos criados
```

Downloads e instalação acontecem antes da janela medida. O carregamento dos
pesos pelo runtime ocorre depois que a coleta começa, portanto startup e
primeira resposta continuam observáveis. O Prometheus permanece ativo durante
toda a bateria. Concorrência é um experimento separado.

## Modelos suportados

| Fonte/formato | vLLM | llama.cpp | Ollama |
|---|---:|---:|---:|
| Hugging Face/Safetensors remoto | sim | não | não |
| checkpoint Hugging Face local | sim | não | não |
| GGUF remoto ou local | sim, experimental | sim | sim |

Uma comparação entre runtimes deve usar o mesmo GGUF, tokenizer, workload,
contexto, limite de saída, hardware e estado de aquecimento. vLLM/HF contra
llama.cpp/GGUF não isola o runtime.

## Alvos do Make

| Alvo | Efeito |
|---|---|
| `observe-help` | mostra o fluxo e as variáveis principais |
| `observe-config` | imprime a configuração efetiva |
| `observe-prepare` | instala e prepara o runtime/modelo selecionado |
| `observe-bench` | orquestra preparação, runtime, Prometheus e bateria sequencial |
| `observe-serve` | mantém o runtime em primeiro plano para exploração |
| `observe-prometheus-start` | inicia exporter e Prometheus em segundo plano |
| `observe-status` | valida runtime, exporter e Prometheus |
| `observe-jupyter` | abre o perfil interativo no Pod |
| `observe-prometheus-stop` | encerra somente a pilha de observação |
| `pull-observe-results` | copia pacotes do Pod para a máquina local |

O runtime é escolhido por `OBS_RUNTIME=vllm|llama|ollama`. O modelo é definido
por `OBS_MODEL_SOURCE=hf|local-hf|gguf|local-gguf`, `OBS_MODEL`,
`OBS_REVISION` e, para repositório GGUF, `OBS_GGUF_FILENAME`. Tokenizer e
revisão podem ser fixados com `OBS_TOKENIZER_MODEL` e
`OBS_TOKENIZER_REVISION`.

A carga usa `OBS_BENCH_SCENARIOS`, `OBS_BENCH_REQUESTS`,
`OBS_BENCH_REPETITIONS`, `OBS_BENCH_WARMUP` e `OBS_BENCH_MODE`.
`OBS_RESULTS_DIR` define a raiz dos pacotes.

`OBS_BENCH_PROFILE` mantém `generic` como padrão e aceita os perfis MOPEP
`mopep-single`, `mopep-review-replay` e `mopep-review-closed-loop`. Eles exigem
dataset, warm-up e manifesto; replay também exige respostas canônicas. O
[contrato MOPEP](workload-mopep.md) define hashes, turnos e comparabilidade.

No `generic`, `short`, `medium` e `long` são cenários sintéticos com alvos de
256, 2048 e 7680 tokens e fonte `qwen_chat_bench_v2.json`; nenhum BMC é lido.
No MOPEP, BMCs reais permanecem íntegros e recebem os buckets `short`, `medium`
e `heavy` pelos tercis calculados na calibração. Não consolide os dois perfis
como se representassem a mesma carga.

Tokens privados do Hugging Face são lidos do ambiente e não podem aparecer em
linha versionada, manifesto ou resultado.

## Componentes e portas padrão

| Componente | Porta | Função |
|---|---:|---|
| vLLM | 8000 | API de inferência e métricas do runtime |
| llama.cpp | 8080 | API de inferência |
| Ollama | 11434 | API de inferência |
| Prometheus | 9090 | armazenamento e consulta das séries |
| exporter do projeto | 9108 | métricas do host e GPU |
| JupyterLab, somente no modo interativo | 8889 | interface e kernel |

Os endpoints escutam em `127.0.0.1` por padrão. A análise headless local não
expõe nenhuma dessas portas e não requer túnel.
O start falha se 9090 ou 9108 já estiver ocupada; readiness só aceita os PIDs
criados pela própria rodada.

## Eventos e fases observadas

| Evento | Significado |
|---|---|
| `request_start` | início do POST no cliente |
| `first_content` | primeiro evento SSE com conteúdo |
| `last_content` | último evento SSE com conteúdo |
| `request_end` | fim da resposta HTTP |

```text
request_start ------- first_content ------- last_content ---- request_end
       TTFT/prefill observado       decode observado           fechamento
```

`Prefill observado` é o intervalo de TTFT visto pelo cliente. Ele pode incluir
fila, HTTP, serialização, tokenização e agendamento; não é a duração interna
pura do kernel de prefill. O decode também usa eventos SSE observados. Tokens/s
depende da contagem `usage` fornecida pela API; quando ela faltar, a taxa fica
ausente.

## Fontes e limites das métricas

| Dado | Fonte | Limitação principal |
|---|---|---|
| CPU | `psutil` | host/Pod visível, não apenas o runtime |
| RAM e swap | `psutil` | inclui os demais processos visíveis |
| leitura/escrita | `psutil.disk_io_counters` | contador convertido em variação/taxa |
| page faults | `/proc/vmstat` | não isola o processo |
| GPU, VRAM, temperatura e potência | `nvidia-smi` | GPU inteira |
| pool KV | `/metrics` do vLLM | ocupação percentual do pool, não bytes de VRAM |
| TTFT e tokens/s | cliente SSE | medição externa ao runtime |
| tráfego PCIe | não coletado | exige DCGM, Nsight ou instrumentação equivalente |

O intervalo de scrape é de aproximadamente 500 ms. Fases mais curtas podem não
receber amostra. Uma métrica não exposta fica ausente/`null`; ausência nunca é
convertida em zero. Em particular, llama.cpp e Ollama não fornecem ao fluxo
atual a série `vllm:kv_cache_usage_perc`. Prometheus, exporter e consultas a
`nvidia-smi` também têm overhead; mantenha a mesma instrumentação em todas as
rodadas comparadas.

A exportação consulta a janela ao fim da rodada. A duração total precisa caber
em `PROMETHEUS_RETENTION` (6 h por padrão); aumente esse valor antes de uma
campanha mais longa.

## Dataset da execução

O pacote mantém os artefatos existentes do `bench.py` e acrescenta:

| Arquivo | Granularidade |
|---|---|
| `all-requests.csv` | uma linha por requisição |
| `request-events.csv` | uma linha por evento observado |
| `prometheus-samples.csv` | uma linha por timestamp, série e conjunto de labels |
| `telemetry-by-request.csv` | amostra temporal vinculada à requisição e fase |
| `request-phase-metrics.csv` | métrica resumida por requisição e fase |
| `dataset-manifest.json` | identidade e configuração da execução e dos arquivos |
| `SHA256SUMS` | integridade dos CSVs e dos manifestos final e do dataset |

O manifesto registra o suficiente para separar campanhas: runtime e versão,
modelo e origem, revisão/hash quando disponível, tokenizer, hardware, software,
argumentos efetivos, workload, horários e status. Ele não contém token de
acesso, prompt privado ou peso.
Metadados de modelo passam por uma allowlist, credenciais do Hugging Face não
são herdadas pelo processo lançado e flags sensíveis em `launch-extra-args`
são recusadas.

KV cache, prefix caching e residência do modelo são campos distintos. O
primeiro é necessário ao decode autoregressivo; sua ocupação só é declarada
quando há uma série do runtime. O segundo é derivado das flags efetivas quando
explícito e permanece `runtime-default`/indisponível nos demais casos.
Para Ollama, versão, keep-alive, tipo do KV, flash attention, paralelismo e
limite de modelos carregados são fixados pelo Make e copiados ao manifesto;
residência continua sendo uma política configurada, não uma medição inferida.

Para médias e percentis entre rodadas, a unidade estatística é a requisição. Os
scrapes dentro de uma requisição são correlacionados; eles descrevem sua série
temporal e não contam como repetições independentes.

## Transferência e análise local

Execute no computador local:

```bash
cd services/inference-runtime
make pull-observe-results OBS_REMOTE_HOST=runpod-qwen
cd ../..
make setup-notebook       # somente na primeira vez
make observability-dataset
make observability-notebook
```

`pull-observe-results` não apaga o pacote remoto nem o local. O consolidado
preserva a proveniência de cada run. Sem `OBS_REMOTE_PORT` ou `OBS_REMOTE_KEY`,
o alvo deixa `ssh`/`scp` resolverem porta e identidade pelo alias em
`~/.ssh/config`; valores explícitos apenas sobrescrevem esse alias. Antes de comparar, valida os cinco CSVs,
`dataset-manifest.json`, o `manifest.json` final e seus hashes. Run completa
com artefato ausente, malformado ou alterado fica somente no inventário.

Cada tabela comparativa recebe um `comparison_id` calculado com SHA do modelo
e tokenizer, contexto, limite de saída, workload, modo e hardware. Runtime e
versão do runtime são a variável comparada e não entram no fingerprint. O
notebook filtra um grupo por vez, portanto configurações incompatíveis não são
agregadas silenciosamente.

O inventário e o resumo expõem o `argv` e ambiente redigidos efetivamente
usados, seus fingerprints, flags de cache com valores, estado de KV/prefix
cache e residência do modelo. Eles são dimensões do runtime; o
`server_command` genérico não é evidência do processo lançado.

O notebook usa kernel local e os CSVs já baixados; runtime, Prometheus e túnel
de Jupyter podem estar desligados. Use `OBS_REMOTE_RESULTS_DIR` e
`OBS_LOCAL_RESULTS_DIR` para sobrescrever os diretórios de origem e destino.

## Perfil interativo

No Pod, os processos podem ser iniciados separadamente:

```bash
make observe-serve OBS_RUNTIME=vllm
make observe-prometheus-start OBS_RUNTIME=vllm
make observe-jupyter OBS_RUNTIME=vllm
```

Nesse modo, navegador local acessa Jupyter por SSH, mas o kernel, a inferência e
a coleta ficam no Pod. A persistência manual gera `profile.json`,
`profile.png`, `request-summary.csv` e `telemetry.csv`; ela não substitui o
dataset multi-requisição do `observe-bench`.

## Regras de validade

- uma requisição ativa por vez;
- nenhuma instalação ou transferência de pesos durante a medição;
- mesma carga e artefato para comparar runtimes;
- falha e dado ausente preservados, sem fallback silencioso;
- hashes e artefatos obrigatórios válidos antes de entrar no comparativo;
- perfil DCGM/Nsight executado separadamente, pois adiciona overhead;
- credenciais, pesos e resultados brutos fora do Git.

Uma análise mínima identifica a fase e o recurso suspeito, altera uma variável
por vez e repete exatamente a mesma carga. O antes/depois deve mostrar TTFT,
tokens/s, latência, recursos e falhas.
