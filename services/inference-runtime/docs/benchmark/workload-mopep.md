# Contrato do workload MOPEP

O benchmark MOPEP é uma extensão do `observe-bench`; o benchmark sintético
permanece como perfil `generic`.

## Artefatos de entrada

`workload.jsonl` e `warmup.jsonl` contêm `request_id`, bucket, contagem de
tokens, mensagens iniciais e instrução de revisão. Eles contêm BMCs privados,
mas nunca as tags esperadas. `workload-manifest.json` registra hashes, ordem,
prompt, heurísticas, tokenizer e os limites de tercis calculados na calibração.

O teste deve apontar para o manifesto da calibração. Recalcular tercis no teste
é inválido porque altera a definição dos grupos depois de observar a amostra.

## Perfis

- `mopep-single`: uma chamada independente por BMC, turno 1.
- `mopep-review-replay`: uma chamada de revisão, turno 2, sobre resposta
  canônica gerada uma vez pelo llama.cpp com o mesmo GGUF.
- `mopep-review-closed-loop`: turno 1 real seguido do turno 2 que revisa essa
  resposta. O segundo corpo pode divergir entre runtimes.

Todos preservam uma chamada ativa por vez. Warm-up usa IDs vindos de `train`
e não entra nas métricas formais.

## Saídas e identidade

Cada pacote acrescenta `text/responses.jsonl`. As respostas mantêm
`experiment_id`, runtime, `request_id`, bucket, perfil e turno; prompts e golden
não são copiados para os CSVs analíticos.

Quando `OBS_RESPONSES_OUT` é usado, também é criado um sidecar
`*.manifest.json` com runtime de origem, hash do GGUF, workload e respostas. O
perfil replay exige esse sidecar, exige origem llama.cpp e valida os hashes.

O `comparison_id` inclui hashes do modelo, tokenizer, workload, prompts e
replay, além das condições já usadas pelo benchmark. Runtime continua sendo a
variável comparada. `closed-loop` é agrupado como cenário de produto, mas o
manifesto declara a divergência possível após o primeiro turno.

## Validade e privacidade

- Não versione workload, respostas, pesos ou resultados brutos.
- Não use tags esperadas para formar o replay.
- Não misture calibração e teste nem refaça buckets no teste.
- Use o mesmo GGUF, tokenizer, hardware, parâmetros e ordem.
- Preserve falhas e dados ausentes; nunca converta ausência em zero.
- Só consolide runs `complete` com hashes válidos.

## Cobertura de testes

Os testes de analytics verificam buckets, determinismo, ausência de labels,
qualidade e deltas de revisão. Os testes do runtime verificam schema, hashes,
IDs duplicados e o número/ordem das chamadas nos três perfis. O smoke real no
Pod valida loaders GGUF, API, Prometheus e cleanup.
