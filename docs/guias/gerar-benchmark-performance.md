# Gerar o benchmark de performance

> **Objetivo:** executar no Pod um benchmark sequencial de um runtime e modelo,
> coletar Prometheus durante toda a rodada e analisar o pacote localmente, sem
> depender de um notebook no Pod.
>
> **Resultado esperado:** um diretório de execução autocontido, com métricas por
> requisição, eventos, amostras temporais, agregações por fase e manifesto.

O comando principal deste fluxo é `make observe-bench`. Ele prepara o modelo e
o runtime, inicia o servidor e o Prometheus, chama o `bench.py` com **uma
requisição por vez**, salva os dados e encerra apenas os processos que criou.
Downloads e instalações terminam antes da janela medida.

## 1. Prepare um Pod vazio

No Pod:

```bash
apt-get update
apt-get install -y git make curl ca-certificates build-essential cmake ninja-build \
  pkg-config python3 python3-venv python3-pip
ninja --version
nvidia-smi
cd /workspace
git clone https://github.com/thiago-ouverney/finance-agent-platform.git
cd finance-agent-platform/services/inference-runtime
make observe-help
```

O `ninja-build` é necessário porque vLLM/FlashInfer pode compilar kernels CUDA
durante o primeiro aquecimento. Sem ele, o servidor encerra antes de abrir a
porta e o benchmark reporta `Connection refused`; o erro raiz no
`logs/server.log` será `No such file or directory: 'ninja'`.

Para llama.cpp com CUDA, use uma imagem *devel* e valide também
`nvcc --version`.

Use armazenamento persistente para evitar baixar os pesos novamente. Tokens do
Hugging Face ficam somente no ambiente do Pod, por exemplo `HF_TOKEN`; nunca no
Makefile, manifesto ou Git.

## 2. Escolha runtime e modelo

### Modelo Hugging Face remoto ou local

Checkpoint Hugging Face/Safetensors é suportado neste fluxo somente pelo vLLM.
Para um repositório público ou privado:

```bash
make observe-bench \
  OBS_RUNTIME=vllm \
  OBS_MODEL_SOURCE=hf \
  OBS_MODEL='<organizacao/modelo>' \
  OBS_REVISION='<revisao-ou-commit>'
```

Para um checkpoint já presente no Pod, use o diretório local no mesmo campo:

```bash
make observe-bench \
  OBS_RUNTIME=vllm \
  OBS_MODEL_SOURCE=local-hf \
  OBS_MODEL=/workspace/models/meu-checkpoint
```

Fixe uma revisão do Hugging Face para uma campanha comparável. `main` sozinho
não identifica de forma imutável os pesos usados.

### GGUF remoto ou local

GGUF pode ser usado com vLLM, llama.cpp ou Ollama. Para baixar do Hugging Face
durante a preparação:

```bash
make observe-bench \
  OBS_RUNTIME=llama \
  OBS_MODEL_SOURCE=gguf \
  OBS_MODEL='<organizacao/repositorio>' \
  OBS_REVISION='<revisao-ou-commit>' \
  OBS_GGUF_FILENAME='<modelo>.gguf'
```

Para um arquivo já presente no Pod:

```bash
make observe-bench \
  OBS_RUNTIME=ollama \
  OBS_MODEL_SOURCE=local-gguf \
  OBS_MODEL=/workspace/models/<modelo>.gguf
```

Para comparar os três runtimes, repita a rodada com
`OBS_RUNTIME=vllm|llama|ollama`, mantendo o mesmo `OBS_MODEL`, tokenizer,
contexto, workload e hardware.
Comparar vLLM/HF contra llama.cpp/GGUF muda runtime e formato simultaneamente.
Use `OBS_TOKENIZER_MODEL` e `OBS_TOKENIZER_REVISION` quando precisar fixar um
tokenizer diferente daquele inferido pelo modelo.

Para executar a campanha inteira com um único comando, use:

```bash
make observe-bench-all \
  OBS_MODEL_SOURCE=gguf \
  OBS_MODEL='<organizacao/repositorio>' \
  OBS_REVISION='<revisao-ou-commit>' \
  OBS_GGUF_FILENAME='<modelo>.gguf' \
  OBS_TOKENIZER_MODEL='<organizacao/tokenizer>'
```

O alvo executa os smokes na ordem vLLM, llama.cpp e Ollama. Somente depois que
os três passam ele repete essa ordem com `OBS_BENCH_SMOKE=0` e a carga completa.
Qualquer falha interrompe a campanha e preserva os resultados já produzidos.
Para arquivo presente no Pod, troque por `OBS_MODEL_SOURCE=local-gguf` e informe
o caminho absoluto em `OBS_MODEL`.

O fluxo fixa por padrão `vllm==0.29.0`, o commit do plugin GGUF, um commit do
llama.cpp e `OBS_OLLAMA_VERSION=0.34.0`. Sobrescreva somente com outra versão
exata. A versão realmente executada, hashes dos binários e o `argv` efetivo ficam no manifesto;
uma instalação existente que divergir da revisão solicitada do llama.cpp ou da
versão solicitada do Ollama é recusada.

No Ollama, a rodada também fixa e registra `KEEP_ALIVE=-1`, KV `f16`, flash
attention ligada, um modelo carregado e paralelismo 1. Esses valores podem ser
alterados pelas variáveis `OBS_OLLAMA_*`, mas passam a definir outro experimento.

Antes de uma campanha, confira os valores efetivos:

```bash
make observe-config \
  OBS_RUNTIME=vllm \
  OBS_MODEL_SOURCE=local-gguf \
  OBS_MODEL=/workspace/models/<modelo>.gguf
```

## 3. Execute a rodada

### 3.1 Escolha o perfil da carga

`OBS_BENCH_PROFILE` define **de onde vêm as requisições**. Se a variável não
for informada, o valor padrão é `generic`; portanto, `observe-bench` não usa
BMCs nem o dataset MOPEP implicitamente.

| Perfil | Entrada | Grupos de tamanho |
|---|---|---|
| `generic` (padrão) | conversas técnicas versionadas em `qwen_chat_bench_v2.json` | `short=256`, `medium=2048`, `long=7680` tokens-alvo |
| `mopep-single` | BMCs reais de `workload.jsonl`, uma chamada por BMC | `short`, `medium`, `heavy`, definidos pelos tercis da calibração |
| `mopep-review-replay` | revisão sobre a mesma resposta canônica | buckets MOPEP preservados |
| `mopep-review-closed-loop` | classificação real seguida de revisão da resposta produzida | buckets MOPEP preservados |

Perfil e cenário são conceitos diferentes: `generic` é o perfil; `short`,
`medium` e `long` são os cenários executados dentro dele. Para usar MOPEP é
obrigatório definir `OBS_BENCH_PROFILE` e informar dataset, warm-up e
manifesto. Consulte o [guia MOPEP](gerar-benchmark-performance-mopep.md).

### 3.2 Execute o perfil genérico

Uma execução completa precisa de apenas um comando:

```bash
make observe-bench \
  OBS_RUNTIME=vllm \
  OBS_MODEL_SOURCE=hf \
  OBS_MODEL='Qwen/Qwen2.5-7B-Instruct' \
  OBS_BENCH_PROFILE=generic
```

Não inicie Jupyter, outro runtime ou outra carga na mesma GPU. O Prometheus
permanece ativo durante toda a rodada; o `bench.py` faz warmup e medições
sequenciais. Concorrência é outro experimento. As portas 9090 e 9108 precisam
estar livres; o comando recusa serviços preexistentes para não coletar outra
instância por engano.

Para configurar a carga, use as variáveis do cliente:

```bash
make observe-bench \
  OBS_RUNTIME=vllm \
  OBS_MODEL_SOURCE=hf \
  OBS_MODEL='Qwen/Qwen2.5-7B-Instruct' \
  OBS_BENCH_PROFILE=generic \
  OBS_BENCH_SCENARIOS='short medium long' \
  OBS_BENCH_REQUESTS=20 \
  OBS_BENCH_REPETITIONS=1 \
  OBS_BENCH_WARMUP=3 \
  OBS_BENCH_MODE=replay
```

No perfil `generic`, os nomes têm alvos fixos: `short=256`, `medium=2048` e
`long=7680` tokens de entrada. Esses tamanhos não foram derivados de BMCs. Em
`replay`, o cliente escolhe o histórico versionado mais próximo,
remove pares antigos se necessário e completa abaixo do alvo com contexto
sintético determinístico. O CSV preserva a estimativa e também
`prompt_tokens` retornado pelo runtime; este último é a medida real. O alvo
`long` cabe no `OBS_CONTEXT=8192` padrão, incluindo a reserva para saída.

`OBS_RESULTS_DIR` altera a raiz dos resultados. Uma execução é inválida para
comparação se pesos ou dependências forem baixados durante a janela medida.

## 4. Confira o pacote gerado

Cada execução preserva os artefatos existentes do benchmark e acrescenta:

| Arquivo | Unidade e finalidade |
|---|---|
| `all-requests.csv` | uma linha por requisição, com TTFT, latência, tokens/s e status |
| `request-events.csv` | início, primeiro conteúdo, último conteúdo e fim de cada requisição |
| `prometheus-samples.csv` | amostras temporais normalizadas do Prometheus |
| `telemetry-by-request.csv` | amostras associadas à requisição e fase observada |
| `request-phase-metrics.csv` | resumo de cada métrica por requisição e fase |
| `dataset-manifest.json` | runtime, modelo, configuração, hardware, software e arquivos da execução |
| `SHA256SUMS` | hashes dos CSVs e dos manifestos final e do dataset |

Valores não expostos ficam ausentes/`null`; nunca são convertidos em zero.
`prefill` no dataset significa o intervalo observado entre o início da
requisição e o primeiro conteúdo: é um proxy de TTFT, não o tempo interno puro
do kernel. Tráfego PCIe não é coletado diretamente; requer uma rodada separada
com DCGM ou Nsight.

KV cache, reutilização de prefixo e residência do modelo são registrados como
conceitos separados. A ocupação do KV só aparece quando o runtime a expõe; não
é inferida a partir da VRAM.

Para distribuições, a unidade estatística é a requisição. Os vários scrapes de
uma mesma requisição são amostras correlacionadas, não novas repetições.

Na consolidação local, uma run só entra nas tabelas elegíveis quando o status
final é `complete`, todos os artefatos obrigatórios abrem corretamente e os
hashes coincidem. As excluídas permanecem em `run-inventory.csv`; grupos
distintos continuam preservados no mesmo dataset. O notebook recebe
explicitamente um `run_id` de vLLM, um de llama.cpp e um de Ollama. O
`comparison_id` fixa modelo, tokenizer, contexto, saída, workload e hardware e
é usado como gate: grupos incompatíveis nunca aparecem no mesmo gráfico.

O comparativo também mostra o `argv` e ambiente redigidos efetivamente usados,
seus fingerprints, flags de cache com valores, estado do KV/prefix cache e
residência do modelo. Esses campos são dimensões do tratamento; o
`server_command` genérico da configuração não comprova o processo lançado.

## 5. Baixe e analise na máquina local

Na máquina local, a partir de `services/inference-runtime`:

```bash
make pull-observe-results OBS_REMOTE_HOST=runpod-qwen
```

O alvo copia os pacotes sem apagar os resultados remotos ou locais. Host,
usuário, porta e chave devem permanecer em `~/.ssh/config` ou apenas na linha
de comando. Quando `OBS_REMOTE_PORT` e `OBS_REMOTE_KEY` forem omitidos, `ssh` e
`scp` usam integralmente o alias, inclusive porta e identidade. Informe essas
variáveis somente para sobrescrever o alias. Se os diretórios não forem os padrões, use
`OBS_REMOTE_RESULTS_DIR` e `OBS_LOCAL_RESULTS_DIR`.
Antes de aceitar o pacote, o alvo valida `SHA256SUMS`, incluindo o manifesto
final que define se a execução terminou como `complete`.

Depois, na raiz do repositório local:

```bash
make setup-notebook
make observability-dataset
make observability-notebook
```

`setup-notebook` só é necessário na primeira vez. O alvo de dataset consolida
as execuções válidas e mantém grupos incompatíveis separados. O último abre o
notebook de análise com **kernel local** e lê os CSVs já baixados; não precisa
de túnel, runtime nem Prometheus ativos.

Na primeira célula, preencha `RUN_IDS` com as três execuções formais. Como
alternativa, defina `OBSERVABILITY_RUN_ID_VLLM`,
`OBSERVABILITY_RUN_ID_LLAMA` e `OBSERVABILITY_RUN_ID_OLLAMA`. Cada seção
individual funciona isoladamente; a consolidação só é liberada quando as três
runs compartilham a identidade experimental e o pareamento das requisições.

## 6. Faça a comparação

Para cada runtime e cenário, reporte:

- distribuição de TTFT, tokens/s de decode e latência ponta a ponta;
- CPU, GPU, RAM, VRAM, swap, disco e page faults por fase;
- ocupação do pool KV quando o runtime expuser a métrica;
- falhas, valores ausentes e configuração registrada no manifesto.

Depois das três leituras individuais, use a tabela completa e apenas quatro
eixos executivos: sucesso, TTFT p95, decode p05 e E2E p95. O heatmap de
recursos é neutro — maior uso não significa melhor runtime — e deve mostrar a
cobertura junto do valor. Os gráficos comparativos ficam limitados a entrega,
pressão de recursos e energia quando houver cobertura suficiente. Não gere
score composto.

Localize um gargalo, altere uma variável por vez e repita a mesma carga. Compare
antes e depois; uma única requisição ou um gráfico isolado não sustenta uma
conclusão estatística.

O fluxo legado `make bench-all` continua sendo a bateria formal completa dos
três runtimes e o `make observe-bench` é a variante headless que agrega a série
Prometheus ao pacote. Para uma inspeção manual de uma ou poucas requisições,
use o [perfil interativo](perfilar-inferencia-prometheus.md).

Este guia usa a carga conversacional genérica. Para BMCs reais, buckets
`short/medium/heavy` e revisão por heurísticas, use o
[benchmark de performance MOPEP](gerar-benchmark-performance-mopep.md).

Contrato técnico: [observabilidade com
Prometheus](../../services/inference-runtime/docs/benchmark/observabilidade-prometheus.md).
