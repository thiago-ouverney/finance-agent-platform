# Instruções do inference-gateway

Este guia complementa o [`AGENTS.md` raiz](../../AGENTS.md).

## Papel e fronteira

O gateway é o dono de admissão, fila limitada, justiça entre agentes, validação
de saída estruturada e métricas Prometheus na frente de um runtime
OpenAI-compatible. Ele fala com o runtime só por HTTP. Não importa
`inference-runtime`, `analytics` nem `apps`.

O adapter e os clientes de experimento continuam vendo o mesmo contrato
`/v1/chat/completions`. O benchmark sequencial oficial do runtime não passa
por aqui.

## Limites

- v1 sem Redis/RabbitMQ: fila in-process.
- `max_inflight` segue o joelho operacional (D17): 8.
- Passthrough SSE; não bufferizar a resposta completa antes do primeiro token.
- Pesos, prompts privados e resultados brutos permanecem fora do Git.
- Sem efeito remoto neste componente até um gate explícito no Pod.

## Antes de alterar

Identifique se a mudança afeta admissão, fila, justiça, schema, SSE ou métricas.
Comunique o contrato antes de alterar consumidores. Teste com backend mock
antes de apontar para llama-server.

## Evidência

```bash
cd services/inference-gateway
.venv/bin/python -m unittest discover -s tests -v
```
