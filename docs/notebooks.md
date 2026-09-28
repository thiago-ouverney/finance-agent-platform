# Índice central de notebooks

Este documento define a responsabilidade de cada notebook em
`analytics/notebooks`. Os notebooks coordenam exploração, geração de dados e
análise; a lógica reutilizável e testável deve permanecer em `analytics/src`.

## Convenção de nomes

Os arquivos seguem `verbo_dominio_objeto.ipynb`:

- o verbo informa a ação principal, como `prepare`, `generate`, `build`,
  `evaluate` ou `analyze`;
- o domínio separa os fluxos `mopep`, `eaas` e `inference`;
- o objeto identifica o artefato produzido ou analisado.

## Fluxo MOPEP e golden

```text
prepare_mopep_benchmark_splits
        |-- generate_mopep_openai_silver --------+
        |-- make silver-qwen --------------------+--> build_mopep_golden_consensus
        +-- baseline GPT-4o já registrado -------+            |
                                                              v
                                      evaluate_gpt4o_against_mopep_golden

make eval-run
        -> analyze_mopep_model_evaluations

avaliação exploratória isolada
        -> evaluate_mopep_models_with_ollama
```

### [`prepare_mopep_benchmark_splits.ipynb`](../analytics/notebooks/prepare_mopep_benchmark_splits.ipynb)

- **Responsabilidade:** consultar os BMCs e rótulos de origem, normalizar os
  dados, selecionar a taxonomia e criar os splits de treino, calibração e teste.
- **Entradas:** BigQuery, credenciais locais do Google Cloud e a taxonomia
  indicada por `MOPEP_TAXONOMY_PATH`.
- **Saídas:** `/tmp/mopep-bmc-tags/{train,calibration,test}.csv`.
- **Não faz:** inferência, votação entre modelos ou avaliação do golden.

### [`generate_mopep_openai_silver.ipynb`](../analytics/notebooks/generate_mopep_openai_silver.ipynb)

- **Responsabilidade:** gerar os votos do GPT-5 nano para os splits MOPEP e
  materializar as predições usadas no consenso.
- **Entradas:** `/tmp/mopep-bmc-tags/*.csv` e `OPENAI_API_KEY` no ambiente.
- **Saídas:** `analytics/results/openai-silver/{train,calibration,test}.csv`.
- **Dependências de código:**
  [`analytics/src/evaluate_mopep_tags.py`](../analytics/src/evaluate_mopep_tags.py)
  e o cliente OpenAI. O estado e as limitações atuais estão no
  [guia de avaliações](guias/gerar-avaliacao-modelos.md).
- **Não faz:** combinar os três votos nem publicar o golden.

### [`build_mopep_golden_consensus.ipynb`](../analytics/notebooks/build_mopep_golden_consensus.ipynb)

- **Responsabilidade:** unir os votos GPT-4o, Qwen e GPT-5 por `example_id` e
  aplicar `majority_vote_2_of_3` para cada tag.
- **Entradas:** splits de origem, campanhas Qwen completas em
  `analytics/results/mopep-silver/qwen/` e predições GPT-5 em
  `analytics/results/openai-silver/`.
- **Saída privada:** `/tmp/mopep-bmc-private/identity.csv`.
- **Saídas compartilháveis locais:**
  `analytics/results/mopep-golden-shareable/{train,calibration,test}.csv`.
- **Não faz:** chamar modelos ou avaliar candidatos.

### [`evaluate_gpt4o_against_mopep_golden.ipynb`](../analytics/notebooks/evaluate_gpt4o_against_mopep_golden.ipynb)

- **Responsabilidade:** medir a concordância do baseline GPT-4o com o golden
  por split, no agregado e por tag.
- **Entradas:** `/tmp/mopep-bmc-tags/*.csv` e os splits de
  `analytics/results/mopep-golden-shareable/`.
- **Saídas:** `comparison.csv`, `summary.csv` e `per_tag.csv` em
  `analytics/results/mopep-golden-evaluation/gpt4o/`.
- **Não faz:** gerar novos rótulos ou alterar o consenso.

### [`evaluate_mopep_models_with_ollama.ipynb`](../analytics/notebooks/evaluate_mopep_models_with_ollama.ipynb)

- **Responsabilidade:** executar uma avaliação interativa e exploratória de
  modelos disponíveis em um endpoint Ollama.
- **Entradas:** um CSV MOPEP, `OLLAMA_HOST` e os nomes dos modelos.
- **Saídas:** tabelas e métricas mantidas na sessão do notebook.
- **Relação com a CLI:** para campanhas reproduzíveis e persistidas, prefira
  `make eval-run`; este notebook serve para inspeção e experimentação.

### [`analyze_mopep_model_evaluations.ipynb`](../analytics/notebooks/analyze_mopep_model_evaluations.ipynb)

- **Responsabilidade:** comparar runs MOPEP já materializados, agrupando apenas
  execuções com o mesmo `comparison_id`.
- **Entradas:** `analytics/results/mopep-tags/`.
- **Dependência de código:**
  [`analytics/src/consolidate_mopep_evals.py`](../analytics/src/consolidate_mopep_evals.py).
- **Saídas:** tabelas e gráficos de qualidade, latência e análise de erro na
  sessão do notebook.
- **Não faz:** enviar requisições de inferência.

## Fluxo de performance de inferência

### [`profile_runtime_with_prometheus.ipynb`](../analytics/notebooks/profile_runtime_with_prometheus.ipynb)

- **Responsabilidade:** enviar uma ou poucas requisições sequenciais e
  correlacionar TTFT/prefill observado e decode com a telemetria temporal do
  Pod.
- **Entradas:** `INFERENCE_BASE_URL`, `INFERENCE_MODEL`, `PROMETHEUS_URL`,
  prompt e limite de saída definidos na sessão.
- **Saídas:** `profile.json`, `profile.png`, `request-summary.csv` e
  `telemetry.csv` em
  `services/inference-runtime/results/notebook-prometheus/<timestamp>/`.
- **Comando:** no Pod, `make observe-jupyter OBS_RUNTIME=<runtime>` a partir de
  `services/inference-runtime`.
- **Guia:** [perfil de inferência com Prometheus](guias/perfilar-inferencia-prometheus.md).
- **Não faz:** calcular uma comparação estatística formal entre runtimes nem
  medir tráfego PCIe ou tempo interno exato dos kernels.

### [`analyze_runtime_observability.ipynb`](../analytics/notebooks/analyze_runtime_observability.ipynb)

- **Responsabilidade:** analisar separadamente uma run de vLLM, uma de
  llama.cpp e uma de Ollama e, depois, comparar as três quando o experimento
  for controlado e pareado.
- **Entradas:** pacotes baixados com `make pull-observe-results` e o dataset
  produzido por `make observability-dataset`.
- **Integridade:** somente runs `complete` com os cinco CSVs obrigatórios, os
  dois manifestos e os hashes válidos entram nas tabelas comparativas; as
  demais permanecem em `run-inventory.csv` com o motivo da exclusão.
- **Seleção:** informe os três `run_id` na célula de configuração ou por
  `OBSERVABILITY_RUN_ID_VLLM`, `OBSERVABILITY_RUN_ID_LLAMA` e
  `OBSERVABILITY_RUN_ID_OLLAMA`. O notebook nunca escolhe automaticamente a
  primeira run ou o primeiro grupo encontrado.
- **Comparabilidade:** `comparison_id` é um gate, não um seletor. Heatmaps,
  gráficos e conclusão comparativa só aparecem quando as três runs são
  distintas, completas, íntegras, pertencem ao mesmo grupo e têm o mesmo
  multiconjunto de `scenario + request_sha256`. O fingerprint fixa artefato,
  tokenizer, contexto, saída, workload e hardware; runtime e versão continuam
  sendo as variáveis comparadas.
- **Tratamento efetivo:** `run-inventory.csv`, `complete-runs.csv` e o resumo
  expõem o `argv` e ambiente redigidos realmente usados, seus fingerprints,
  flags de cache com valores, estado de KV/prefix cache e residência do modelo.
  O comando genérico da configuração não é usado como prova do processo.
- **Saídas:** identidade, entrega, recursos, energia estimada, timeline e
  conclusão de cada runtime; com o gate aprovado, tabela completa, heatmaps e
  gráficos comparativos. Os CSVs de origem não são modificados.
- **Comando:** na raiz local, `make observability-notebook`.
- **Onde executa:** navegador e kernel na máquina local, sem túnel, runtime ou
  Prometheus ativos.
- **Não faz:** enviar requisições, tratar scrapes como repetições independentes
  nem preencher métricas ausentes com zero ou agregar grupos incompatíveis.

### [`analyze_inference_benchmarks.ipynb`](../analytics/notebooks/analyze_inference_benchmarks.ipynb)

- **Responsabilidade:** explorar o dataset consolidado dos benchmarks de
  vLLM, llama.cpp e Ollama e comparar runtimes e quantizações.
- **Entrada:** `analytics/data/benchmark_tagged.csv`, produzido por
  `make benchmark-dataset` e `make benchmark-tags`.
- **Saídas:** tabelas e gráficos na sessão do notebook.
- **Comando:** `make notebook`.
- **Não faz:** medir o runtime novamente.

### [`analyze_mopep_runtime_benchmark.ipynb`](../analytics/notebooks/analyze_mopep_runtime_benchmark.ipynb)

- **Responsabilidade:** unir qualidade MOPEP e performance por runtime,
  bucket, perfil e turno, incluindo melhoria e regressão após revisão.
- **Entradas:** dataset criado por `make mopep-performance-dataset` a partir
  do golden local, respostas privadas e consolidado Prometheus.
- **Comando:** `make mopep-performance-notebook`.
- **Não faz:** inferência, geração de replay ou recálculo dos tercis.

## Fluxo EaaS

### [`build_eaas_business_model_canvas.ipynb`](../analytics/notebooks/build_eaas_business_model_canvas.ipynb)

- **Responsabilidade:** transformar exports ou consultas EaaS em Business Model
  Canvas, tags rastreáveis e indicadores de cobertura e revisão humana.
- **Entradas:** CSV, Parquet, JSON ou JSONL local; alternativamente, API
  GraphQL em modo `QUERY_ONLY`.
- **Dependência de código:**
  [`analytics/src/eaas_business_model_canvas.py`](../analytics/src/eaas_business_model_canvas.py).
- **Saídas:** `analytics/data/eaas_business_model_canvas.csv`, o Parquet
  correspondente e `eaas_business_model_canvas_tags.csv`.
- **Não faz:** criar os splits MOPEP nem participar do consenso do golden.

## Regras comuns

- Dados, predições, resultados brutos e credenciais permanecem fora do Git.
- Notebooks de análise consomem artefatos existentes e não duplicam medições do
  runtime. A exceção explícita é `profile_runtime_with_prometheus.ipynb`, que
  orquestra uma medição exploratória e salva o resultado bruto antes da análise.
- Funções compartilhadas ou usadas por automação devem ser extraídas para
  `analytics/src` e cobertas pelos testes de `analytics/tests`.
- O split de teste só deve ser usado depois que modelo, prompt e critérios
  estiverem congelados.
