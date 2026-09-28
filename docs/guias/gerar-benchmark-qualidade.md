# Gerar as tabelas do benchmark de qualidade

> **Objetivo:** extrair BMCs e tags MOPEP do BigQuery, validar os dados e gerar
> os splits usados nas etapas de curadoria e avaliação.
>
> **Resultado esperado:** `/tmp/mopep-bmc-tags/train.csv`,
> `calibration.csv` e `test.csv`.

## Pré-requisitos

- acesso de leitura à tabela configurada no notebook;
- credenciais locais do Google Cloud;
- taxonomia em `/tmp/mopep-tags.csv`, ou em outro caminho definido por
  `MOPEP_TAXONOMY_PATH`, com uma coluna chamada `Tag`.

O repositório não gera o arquivo de taxonomia; obtenha a versão aprovada com o
responsável pela taxonomia antes de iniciar.

## Passo a passo

Na raiz do repositório, prepare o ambiente:

```bash
make setup-notebook
gcloud auth application-default login
```

Abra o notebook com os caminhos de saída e taxonomia explícitos:

```bash
MOPEP_BMC_DATA_DIR=/tmp/mopep-bmc-tags \
MOPEP_TAXONOMY_PATH=/tmp/mopep-tags.csv \
analytics/.venv/bin/python -m jupyter lab \
  analytics/notebooks/prepare_mopep_benchmark_splits.ipynb
```

Execute todas as células, confira as validações de duplicação e vazamento e só
então grave os três splits.

## Verificar o resultado

```bash
make eval-datasets EVAL_DATASETS_DIR=/tmp/mopep-bmc-tags
```

Os arquivos devem conter, no mínimo, `example_id`, `business_model` e `tags`.
Use `calibration` enquanto ajustar o processo e preserve `test` para a medição
final. Os CSVs contêm dados privados e não devem ser versionados.

O racional e a interpretação das colunas estão no [guia central](../README.md#1-benchmark-de-qualidade).
