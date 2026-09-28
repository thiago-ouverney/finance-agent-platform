# Perfilar uma inferência interativamente com Prometheus

> **Objetivo:** executar uma requisição contra um runtime no Pod e relacionar
> TTFT e tokens/s com CPU, GPU, RAM, VRAM, disco, swap, page faults e, quando
> exposto, ocupação do pool de KV cache.
>
> **Resultado esperado:** um gráfico temporal e os dados brutos em
> `services/inference-runtime/results/notebook-prometheus/<timestamp>/`.

Este é um fluxo exploratório para localizar gargalos. Ele não substitui o
[benchmark formal entre runtimes](gerar-benchmark-performance.md), que repete
uma carga controlada e produz as estatísticas comparáveis.

Para executar a bateria no Pod sem Jupyter, use `make observe-bench`. Depois,
baixe o pacote com `make pull-observe-results` e analise-o localmente. Este guia
permanece útil para inspecionar manualmente uma ou poucas requisições.

## Onde cada componente executa

```text
navegador local
    -> túnel SSH na porta 8889
    -> Jupyter e kernel Python no Pod
       -> runtime no Pod
       -> Prometheus e exporter no Pod
```

A requisição de inferência e a coleta não atravessam a internet nem o túnel
SSH. O túnel transporta apenas a interface do Jupyter para o navegador local.

## 1. Prepare um Pod vazio

Use um Pod Linux com GPU NVIDIA e armazenamento suficiente para o modelo. Para
vLLM e Ollama, `nvidia-smi` deve funcionar. Para compilar llama.cpp com CUDA, a
imagem também precisa fornecer `nvcc`; prefira uma imagem CUDA *devel*.

No Pod:

```bash
nvidia-smi
apt-get update
apt-get install -y git make
cd /workspace
git clone https://github.com/thiago-ouverney/finance-agent-platform.git
cd finance-agent-platform/services/inference-runtime
```

Se o runtime for llama.cpp, valide também `nvcc --version` antes de continuar.
Essa ferramenta não é necessária para o fluxo padrão com vLLM.

O fluxo instala pacotes de sistema com `apt-get` e, portanto, deve ser
executado como `root` no Pod. Pesos e ferramentas são baixados nesta preparação,
nunca durante a execução medida.

## 2. Confira e prepare a configuração

O padrão foi ajustado para uma RTX 3090: Qwen2.5-7B-Instruct em formato Hugging
Face, contexto de 8192 tokens, uma sequência e 90% da VRAM disponível para o
vLLM.

```bash
make observe-help
make observe-config OBS_RUNTIME=vllm
make observe-bootstrap OBS_RUNTIME=vllm
```

`observe-bootstrap` instala dependências, prepara o runtime, instala Prometheus
e cria o ambiente do notebook. Downloads e instalação não fazem parte da
medição.

## 3. Inicie os três processos no Pod

Abra três terminais no mesmo Pod e mantenha os valores `OBS_*` iguais entre os
comandos.

Terminal 1 — runtime em primeiro plano:

```bash
cd /workspace/finance-agent-platform/services/inference-runtime
make observe-serve OBS_RUNTIME=vllm
```

Terminal 2 — coleta:

```bash
cd /workspace/finance-agent-platform/services/inference-runtime
make observe-prometheus-start OBS_RUNTIME=vllm
make observe-status OBS_RUNTIME=vllm
```

Terminal 3 — notebook:

```bash
cd /workspace/finance-agent-platform/services/inference-runtime
make observe-jupyter OBS_RUNTIME=vllm OBS_JUPYTER_PORT=8889
```

Copie a URL com token exibida pelo Jupyter. O alvo já usa `--allow-root` e não
publica a porta para a internet.

## 4. Abra o notebook pela máquina local

Na máquina local, mantenha o túnel ativo:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8889:127.0.0.1:8889 runpod-qwen
```

Abra no navegador a URL copiada, substituindo o host por `127.0.0.1:8889` se
necessário. O alias `runpod-qwen` é configurado pelo
[guia de SSH](configurar-ssh.md).

## 5. Execute e leia o notebook

Abra `analytics/notebooks/profile_runtime_with_prometheus.ipynb` e execute as
células em ordem:

1. carregue as funções do projeto;
2. ajuste `MAX_TOKENS` e `PROMPT`;
3. valide as APIs do runtime e do Prometheus;
4. envie uma requisição sequencial;
5. examine a resposta, a tabela e os cinco painéis;
6. salve os artefatos da execução.

O gráfico marca os seguintes eventos:

| Intervalo | Interpretação |
|---|---|
| início da requisição até o primeiro conteúdo | TTFT/*prefill observado* |
| primeiro até o último conteúdo | *decode observado* |
| último conteúdo até o fim HTTP | fechamento da resposta |

O primeiro intervalo inclui fila, HTTP, tokenização e outras despesas do
servidor. Ele é útil para correlação, mas não é uma medição interna pura do
kernel de prefill.

Relacione os eventos com os recursos:

- TTFT alto com GPU ocupada sugere custo de prefill ou disputa na GPU;
- TTFT alto com GPU ociosa e CPU alta sugere tokenização, preparação ou
  *offload*;
- tokens/s baixos com GPU saturada sugere limite computacional ou de memória;
- aumento de RAM, disco, swap ou major page faults indica pressão de memória;
- crescimento do pool KV mostra ocupação do cache, não percentual de VRAM.

Uma amostragem de 500 ms pode não capturar um prefill muito curto. Para tornar
essa fase visível, aumente o prompt sem exceder `OBS_CONTEXT`.

## 6. Preserve e baixe os resultados

A célula de persistência cria:

```text
results/notebook-prometheus/<timestamp>/
├── profile.json
├── profile.png
├── request-summary.csv
└── telemetry.csv
```

Na máquina local, não no Pod:

```bash
cd services/inference-runtime
make pull-results \
  POD_SSH='<usuario@host>' \
  POD_PORT='<porta-ssh>' \
  REMOTE_RESULTS_DIR=/workspace/finance-agent-platform/services/inference-runtime/results \
  LOCAL_RESULTS_DIR=results-from-pod
```

Resultados brutos não devem ser versionados.

Este diretório é o artefato do perfil interativo. A rodada headless cria o
dataset multi-requisição (`all-requests.csv`, `request-events.csv`,
`prometheus-samples.csv`, `telemetry-by-request.csv`,
`request-phase-metrics.csv` e `dataset-manifest.json`) e deve ser baixada com
`make pull-observe-results`.

## 7. Troque o modelo ou os limites

Exemplo com outro modelo Hugging Face no mesmo vLLM:

```bash
make observe-bootstrap \
  OBS_RUNTIME=vllm \
  OBS_VLLM_MODEL='<repositorio/modelo>' \
  OBS_MODEL_ALIAS='modelo-teste' \
  OBS_CONTEXT=4096
```

Repita exatamente essas variáveis em `observe-serve`,
`observe-prometheus-start`, `observe-status` e `observe-jupyter`. Variáveis
úteis:

```text
OBS_VLLM_MODEL
OBS_MODEL_ALIAS
OBS_CONTEXT
OBS_VLLM_DTYPE
OBS_VLLM_GPU_MEMORY_UTILIZATION
OBS_VLLM_PACKAGE
OBS_VLLM_EXTRA_ARGS
```

Antes de iniciar, use `make observe-config ...` para registrar a configuração
efetiva.

## 8. Compare runtimes com o mesmo GGUF

Comparar vLLM com pesos Hugging Face contra llama.cpp ou Ollama com GGUF muda
formato e runtime ao mesmo tempo. Para isolar o runtime, use o mesmo GGUF,
tokenizer, contexto, prompt, saída máxima e hardware.

Prepare um runtime por vez:

```bash
# vLLM com GGUF
make observe-bootstrap OBS_RUNTIME=vllm OBS_VLLM_FORMAT=gguf

# llama.cpp com o mesmo GGUF; requer nvcc
make observe-bootstrap OBS_RUNTIME=llama

# Ollama com o mesmo GGUF
make observe-bootstrap OBS_RUNTIME=ollama
```

Depois, para o runtime escolhido, execute os três processos das seções 3 e 4.
As portas padrão são 8000 para vLLM, 8080 para llama.cpp e 11434 para Ollama.
O notebook recebe a URL correta pelo ambiente.

Este fluxo mantém uma requisição ativa por vez. Para obter médias, percentis e
distribuições comparáveis com a mesma telemetria, use `make observe-bench`
conforme o [benchmark formal](gerar-benchmark-performance.md); não trate cada
amostra temporal do Prometheus como uma requisição independente.

## 9. Encerre e diagnostique falhas comuns

No Pod:

```bash
make observe-prometheus-stop
```

Encerre runtime e Jupyter com `Ctrl-C` nos terminais correspondentes.

- **Porta 8889 ocupada:** use `OBS_JUPYTER_PORT=8890` no Make e a mesma porta
  nos dois lados do túnel SSH.
- **Prometheus indisponível:** execute `make observe-status OBS_RUNTIME=...` e
  consulte `observability/.state/{prometheus,exporter}.log`.
- **GPU ou VRAM ausentes:** valide `nvidia-smi` e
  `curl -s http://127.0.0.1:9108/metrics | grep inference_gpu`.
- **Painel KV vazio em llama.cpp ou Ollama:** esperado; o perfil atual só
  consulta a métrica de pool KV exposta pelo vLLM.
- **PCIe ausente:** o exporter atual não mede tráfego PCIe. Use Nsight/DCGM em
  uma execução complementar e não misture o overhead desse profiler com os
  números oficiais de TTFT e tokens/s.

Contrato completo de métricas, variáveis e limitações:
[observabilidade com Prometheus](../../services/inference-runtime/docs/benchmark/observabilidade-prometheus.md).
