# Gerar avaliações e o golden

> **Objetivo:** reunir os votos do GPT-4o, Qwen 72B e GPT-5 nano, gerar o golden
> por consenso de dois entre três e comparar modelos com a mesma referência.
>
> **Resultado esperado:** golden em
> `analytics/results/mopep-golden-shareable/`, baseline GPT-4o registrado e
> runs de candidatos comparáveis pelo Macro-F1.

> **Estado atual:** o notebook GPT-5 ainda não fecha os três splits exigidos
> pelo golden. As etapas abaixo definem o contrato, mas o fluxo completo só deve
> ser considerado reproduzível após resolver as limitações listadas no final.

## Pré-requisitos

- os três splits em `/tmp/mopep-bmc-tags/`;
- voto do GPT-4o já presente na coluna `tags` desses arquivos;
- ambiente criado com `make setup-notebook`;
- pontes SSH configuradas pelo guia [Configurar SSH](configurar-ssh.md).

## 1. Gerar os votos Qwen

Sirva o Qwen 72B no RunPod com o nome público `qwen-silver` e abra a ponte do
vLLM. Confirme esse ID com `curl http://127.0.0.1:18000/v1/models` e faça
primeiro uma amostra:

```bash
make silver-qwen \
  SILVER_MODEL=qwen-silver \
  SILVER_MODEL_REVISION='<revisao-ou-digest>' \
  SILVER_DATASET=/tmp/mopep-bmc-tags/calibration.csv \
  SILVER_LIMIT=10
```

Depois execute sem `SILVER_LIMIT` para `train.csv`, `calibration.csv` e
`test.csv`. Cada campanha completa fica em
`analytics/results/mopep-silver/qwen/`. O builder atual procura especificamente
campanhas cujo nome de modelo seja `qwen-silver`.

## 2. Gerar os votos GPT-5 nano

Mantenha a chave fora do notebook e abra o fluxo:

```bash
export OPENAI_API_KEY='<token-do-projeto>'
make openai-silver-notebook
```

Revise custo, modelo e limites antes de enviar. Execute o fluxo para os três
splits e só prossiga quando existirem:

```text
analytics/results/openai-silver/train.csv
analytics/results/openai-silver/calibration.csv
analytics/results/openai-silver/test.csv
```

## 3. Gerar o golden por consenso

```bash
analytics/.venv/bin/python -m jupyter lab \
  analytics/notebooks/build_mopep_golden_consensus.ipynb
```

Execute todas as células. Uma tag entra quando recebe ao menos dois dos três
votos. Verifique os arquivos `train.csv`, `calibration.csv` e `test.csv` em
`analytics/results/mopep-golden-shareable/`. Antes de aceitar o resultado,
confirme que cada split preservou a contagem da fonte e as revisões esperadas.

## 4. Registrar o baseline GPT-4o

```bash
analytics/.venv/bin/python -m jupyter lab \
  analytics/notebooks/evaluate_gpt4o_against_mopep_golden.ipynb
```

Execute todas as células. O resumo é salvo em
`analytics/results/mopep-golden-evaluation/gpt4o/summary.csv`. Como o GPT-4o
participa do consenso, esse valor mede concordância, não uma referência humana
independente.

## 5. Avaliar um candidato no teste

Com o Ollama remoto acessível pela ponte `uff-delta`:

```bash
make eval-check
make eval-models
make eval-run \
  EVAL_MODEL='<nome-exato-do-modelo>' \
  EVAL_DATASET=analytics/results/mopep-golden-shareable/test.csv
make eval-results
make eval-consolidate
```

Compare apenas runs com a mesma referência e configuração. Qualquer ajuste após
olhar o teste exige uma nova campanha. O desenho completo está em [três
avaliações e golden por consenso](../silver-qwen-openai-golden.md).

## Limitações que impedem declarar o fluxo fechado

- `generate_mopep_openai_silver.ipynb` ainda fixa uma chave vazia, materializa somente
  `calibration` e `test` e possui células finais que não executam com `Run All`;
  o builder do golden também exige `train.csv`;
- o builder escolhe a campanha Qwen completa mais recente por nome e contagem;
  revise a proveniência antes de aceitar o join;
- a avaliação Ollama atual pode excluir exemplos com `tags=[]`, não preserva a
  coluna `id` do golden e calcula Macro-F1 sobre classes observadas, não sobre a
  lista fixa das 20 tags.

Até esses itens serem corrigidos, os comandos servem para desenvolvimento, mas
não para publicar uma comparação final como plenamente reproduzível.
