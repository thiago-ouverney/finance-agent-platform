# Finance Agent Platform

Monorepo da plataforma de agente financeiro via WhatsApp, inferência local e pipeline de quantização do Qwen.

## Componentes

- `apps/whatsapp-adapter`: chat entre WhatsApp e o runtime de inferência, com janela persistente por contato.
- `services/inference-runtime`: benchmark e operação dos runtimes locais de inferência.
- `pipelines/qwen-quantization`: quantização, empacotamento e avaliação do Qwen.
- `analytics`: dataset agregado, tags BMC, notebook e comparação de modelos preditivos.

Os repositórios anteriores permanecem disponíveis apenas como legado. O desenvolvimento integrado deve continuar neste repositório.

Consulte [a arquitetura](docs/architecture.md) para o fluxo entre os componentes.

Para reproduzir a quantização no RunPod, siga o
[guia do modelo quantizado](docs/guias/gerar-modelo-quantizado.md).

Para gerar `Q4_K_M`, `Q4_K_M` com iMatrix e `Q8_0` e compará-los nos três
runtimes, siga o [guia GGUF/iMatrix](docs/guias/gerar-gguf-imatrix.md).

## Comandos da raiz

Execute `make` sem argumentos para listar os targets disponíveis. A preparação
completa é explícita por meio de `make setup`.

```bash
make
make setup
make install
make test
make build
make verify
make setup-notebook
make benchmark-dataset RESULTS_DIR=/caminho/para/results
make benchmark-tags
make notebook
```

`make setup` instala o adaptador e cria `analytics/.venv` com JupyterLab e o kernel `finance-agent-analytics`. Se quiser preparar somente o ambiente analítico, use `make setup-notebook`.

Cada componente mantém também seus próprios comandos e documentação.

Use Node.js 20.19 ou superior no adaptador WhatsApp (`.nvmrc`).

Para executar o mesmo benchmark localmente, na UFF ou no RunPod, consulte [execução remota](docs/remote-execution.md).

## Avaliar tags MOPEP com Ollama

O fluxo abaixo executa, fora do notebook, a mesma avaliação de tags de
`analytics/notebooks/loading_bench_files.ipynb`. O dataset e os resultados ficam
na máquina local; as requisições chegam ao Ollama da UFF ou do RunPod por um
túnel SSH. As inferências são sequenciais.

### 1. Preparar o ambiente e o servidor

Prepare o Python local uma vez:

```bash
make setup-notebook
```

Na máquina remota, confirme que o Ollama está ativo em `127.0.0.1:11434` e que
o modelo desejado aparece em `ollama list`. Guarde host, usuário, porta e chave
em um alias do `~/.ssh/config`; não coloque esses dados no repositório.

### 2. Disponibilizar o dataset

Por padrão, os CSVs são procurados em `/tmp/mopep-bmc-tags`. Cada arquivo deve
ter as colunas `business_model` e `tags`; `example_id` é recomendado, mas será
gerado a partir do texto quando estiver ausente.

Liste os datasets encontrados:

```bash
make eval-datasets
```

Para usar outra pasta:

```bash
make eval-datasets EVAL_DATASETS_DIR=/caminho/para/datasets
```

O ID mostrado pelo comando é o nome do CSV sem a extensão, por exemplo
`calibration` para `calibration.csv`.

### 3. Abrir o túnel SSH

Em um terminal local, mantenha o comando abaixo em execução:

```bash
make eval-tunnel EVAL_SSH_HOST=uff-delta
```

O alias pode apontar para a UFF ou para um RunPod. O túnel padrão encaminha
`127.0.0.1:18000` local para `127.0.0.1:11434` remoto. Para usar outra porta
local, informe o mesmo valor nos comandos seguintes:

```bash
make eval-tunnel EVAL_SSH_HOST=runpod-qwen EVAL_LOCAL_PORT=18001
```

### 4. Verificar a ponte e escolher o modelo

Em outro terminal:

```bash
make eval-check
make eval-models
```

Se o túnel estiver na porta `18001`, use `EVAL_LOCAL_PORT=18001` em todos os
comandos seguintes, inclusive no `eval-run`. O nome passado na avaliação deve
ser exatamente o exibido por `make eval-models`.

### 5. Executar a avaliação

```bash
make eval-run \
  EVAL_MODEL='qwen2.5:7b-instruct-q4_K_M' \
  EVAL_DATASET=calibration
```

Uma execução curta pode ser feita com:

```bash
make eval-run \
  EVAL_MODEL='qwen2.5:7b-instruct-q4_K_M' \
  EVAL_DATASET=calibration \
  EVAL_LIMIT=10
```

`EVAL_DATASET` também aceita o caminho completo de um CSV. Antes de iniciar, o
comando confirma a conexão e verifica se o modelo existe. Ele faz uma requisição
de aquecimento e depois calcula precision, recall, F1 e exact match por exemplo,
além das métricas agregadas do notebook.

Cada execução cria uma pasta identificada por horário, modelo, dataset e um ID
curto em `analytics/results/mopep-tags/`:

```text
analytics/results/mopep-tags/<run-id>/
├── predictions.csv
├── performance.csv
└── run.json
```

Esses resultados são locais e ignorados pelo Git. Use o split de calibração
enquanto ajustar prompt ou parâmetros e reserve o split de teste para a medição
final.

### 6. Comparar as avaliações realizadas

Para ver uma tabela resumida dos runs completos:

```bash
make eval-results
```

Para gerar CSVs consolidados:

```bash
make eval-consolidate
```

O comando grava dois arquivos locais, também ignorados pelo Git:

```text
analytics/data/mopep_eval_runs.csv
analytics/data/mopep_eval_predictions.csv
```

O primeiro possui uma linha por run e serve para comparar Macro F1, F1 médio,
exact match, taxa de parsing e latência. O segundo possui uma linha por exemplo
avaliado e permite comparar a distribuição do F1 e inspecionar erros.

Cada run recebe ainda um `comparison_id`, calculado a partir do dataset, dos
exemplos selecionados, do prompt e das opções do Ollama. Compare modelos dentro
do mesmo `comparison_id` para não misturar amostras ou configurações diferentes.

O notebook de análise está em
`analytics/notebooks/mopep-eval-analysis.ipynb`. Ele lê diretamente as pastas de
resultados, concatena os runs pela mesma função usada por
`make eval-consolidate` e apresenta:

- resumo das métricas por modelo;
- Macro F1 e distribuição do F1 por exemplo;
- trade-off entre qualidade e latência;
- trade-off entre qualidade, tamanho e quantização;
- exemplos com menor F1 para análise de erro.

Não há um comando Make para abrir o notebook. Abra o arquivo diretamente pelo
editor ou pelo Jupyter depois de gerar os runs que deseja comparar.

## Segurança

Não versione `.env`, credenciais do WhatsApp, tokens, modelos, ambientes virtuais nem resultados brutos. Use os arquivos `.env.example` de cada componente como referência.
