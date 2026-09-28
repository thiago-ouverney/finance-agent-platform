# Benchmark analytics

> **Objetivo:** transformar resultados de inferência e avaliações MOPEP em
> tabelas comparáveis, análises e modelos auxiliares.

Este módulo consome os artefatos produzidos por `services/inference-runtime`. Ele não mede o runtime novamente.

O [índice central de notebooks](../docs/notebooks.md) descreve a responsabilidade,
as entradas, as saídas e as dependências de cada arquivo interativo.

## Fluxo

```bash
make setup-notebook
make benchmark-dataset RESULTS_DIR=/caminho/para/results
make benchmark-tags
make notebook
```

Para treinar e comparar Random Forest e Extra Trees na previsão do TTFT p50:

```bash
make predict-benchmark
```

As tags BMC são regras explícitas derivadas de runtime, família do modelo, quantização, contexto, fase, completude e faixa de TTFT. Ajuste os limiares em `src/generate_tags.py` quando a definição formal de BMC do estudo for fechada.

Dados consolidados e modelos treinados são locais e ignorados pelo Git. Resultados brutos continuam sob responsabilidade do benchmark.

## Observabilidade de runtime

Após baixar uma campanha do Pod, a análise é inteiramente local:

```bash
make setup-notebook  # somente na primeira vez
make observability-dataset OBSERVABILITY_RESULTS_DIR=services/inference-runtime/results-from-pod
make observability-notebook
```

O consolidador verifica integridade e compatibilidade, preserva execuções
inválidas apenas no inventário e agrupa separadamente campanhas com controles
diferentes. O notebook não envia inferências nem consulta o Prometheus.

## Avaliações MOPEP

Os comandos e contratos para gerar a silver Qwen, os votos GPT-5, o golden e as
avaliações de candidatos ficam no guia [Gerar avaliações e o
golden](../docs/guias/gerar-avaliacao-modelos.md). O estado atual do notebook
GPT-5 e as limitações que ainda impedem declarar o fluxo completamente
reproduzível também estão registrados nesse guia.

## Business Model Canvas com dados EaaS

O notebook `notebooks/build_eaas_business_model_canvas.ipynb` normaliza exports CSV,
Parquet, JSON ou JSONL do `data-intelligence/eaas` e também pode consultar a API
GraphQL em modo `QUERY_ONLY`. Ele produz uma tabela com os nove blocos do BMC,
tags rastreáveis, cobertura das evidências e sinalização de revisão humana.

Para usar um export local:

```bash
EAAS_ENRICHMENT_INPUT=/caminho/enrichment.jsonl \
  analytics/.venv/bin/jupyter lab analytics/notebooks/build_eaas_business_model_canvas.ipynb
```

Para consultar a API, defina `EAAS_SOURCE_MODE=api`, `EAAS_API_URL`,
`EAAS_API_KEY`, `EAAS_USER_EMAIL` e forneça
`analytics/data/eaas_identifiers.csv` com `cnpj` e/ou `domain`. Credenciais e
dados de entrada devem permanecer apenas no ambiente local.
