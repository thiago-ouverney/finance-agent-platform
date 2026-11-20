# inference-gateway

Middleware OpenAI-compatible na frente do `llama-server`: admissão, fila
limitada, justiça por agente, validação de JSON Schema e métricas.

Contrato e joelho: [`specs/concurrent-agents-middleware/spec.md`](../../specs/concurrent-agents-middleware/spec.md).
`max_inflight` operacional: 8 (D17).

## Testes

```bash
cd services/inference-gateway
.venv/bin/python -m unittest discover -s tests -v
```

Sem GPU. Upstream real só em gates posteriores, com Pod autorizado.
