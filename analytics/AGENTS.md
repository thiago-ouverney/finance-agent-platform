# Instruções de analytics

Este guia complementa o [`AGENTS.md` raiz](../AGENTS.md).

## Papel e fronteira

`analytics` transforma datasets e resultados produzidos pelos demais
componentes em workloads, avaliações, tabelas e análises. Ele não implementa
serving nem mede novamente TTFT, tokens/s ou KV-cache. `src` contém lógica
reutilizável; notebooks orquestram e explicam fluxos interativos; `tests`
protege os contratos reproduzíveis.

O índice canônico de notebooks é [`docs/notebooks.md`](../docs/notebooks.md).
Uma investigação não trivial deve apontar para sua spec e manter status do
trabalho separado do outcome da hipótese.

## Limites

- Dados privados, associações de identidade, predições, resultados brutos,
  modelos treinados e ambientes virtuais permanecem ignorados.
- Versione somente fixtures mínimas, sintéticas ou sanitizadas quando um teste
  realmente depender delas.
- Não mova algoritmo reutilizável para notebook nem faça produção depender de
  output interativo, checkpoint ou estado local.
- Preserve splits, seeds, revisões, prompt/taxonomia e hashes que sustentam
  comparabilidade. `test.csv` não pode orientar tuning ou quantização.
- Consultas a BigQuery/APIs e chamadas a modelos são efeitos externos, mesmo
  quando o notebook é local; confirme modo, credenciais, custo e output antes.
- Outputs derivados devem registrar fonte, parâmetros e proveniência; não edite
  relatório gerado como se fosse fonte.

## Antes de alterar

Identifique dataset e consumidor, risco de leakage, unidade da amostra,
baseline, métrica, critério de invalidação e política de dados. Se o contrato
de workload ou resultado do runtime mudar, alinhe primeiro com o componente
produtor e atualize a documentação canônica na mesma mudança.

## Evidência

Execute o teste focal e, quando o ambiente existir:

```bash
analytics/.venv/bin/python -m unittest discover -s analytics/tests
```

Para notebooks, valide o código extraído ou execute de forma reproduzível sem
gravar dados privados no arquivo. O handoff informa dataset/revisão, seed,
parâmetros, outputs locais, células não executadas e chamadas externas feitas.

