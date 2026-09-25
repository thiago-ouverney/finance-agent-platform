# Documentação

Documentos separados por função para não misturar instruções de execução, método, engenharia e diagnósticos históricos.

## Benchmark

Explica o que é medido e como interpretar os resultados.

- [Bench explicado — HTML](benchmark/bench-explicado.html) · [fonte Markdown](benchmark/bench-explicado.md)
- [Metodologia — HTML](benchmark/metodologia.html) · [fonte Markdown](benchmark/metodologia.md)
- [Resultados da rodada no Pod](benchmark/resultados-pod-20260922.md)

## Make e execução

Explica os alvos, dependências, preparação, smokes, bateria base e sweep.

- [Fluxo completo do Make — HTML](make/make-fluxo.html) · [fonte Markdown](make/make-fluxo.md)
- [Diagnóstico dos Makefiles](make/diagnostico-makefiles.md)
- [Plano e evidências da execução](make/plano-implementacao-testes-makefiles.md)

## Engenharia e operação

- [Código explicado — HTML](engenharia/codigo-explicado.html) · [fonte Markdown](engenharia/codigo-explicado.md)
- [Fontes numerados](engenharia/codigo-fontes.md) — gerado por `scripts/build_code_reference.py`
- [Git no RunPod — HTML](engenharia/git-runpod.html) · [fonte Markdown](engenharia/git-runpod.md)
- [Segurança antes de publicar](engenharia/seguranca-publicacao.md)

## Diagnósticos históricos

Estes documentos preservam falhas e limitações de experimentos específicos. Não são instruções para trocar o modelo padrão da rodada atual.

- [Diagnóstico Qwen2.5-14B/vLLM](diagnosticos/diagnostico-qwen14b-vllm.md)
- [Relatório de falhas](diagnosticos/relatorio-falhas.md)
- [Evidências de execução local](diagnosticos/evidencias-execucao-local-20260922.md)

## Regenerar HTMLs

```bash
python scripts/build_docs.py
python scripts/build_code_reference.py
```

Os testes funcionais e benchmarks devem ser executados no Pod. A geração de documentação é local e não inicia runtimes.
