# Documentação

> **Objetivo:** organizar os materiais do benchmark por execução, metodologia,
> engenharia e histórico.

Documentos separados por função para não misturar instruções de execução, método, engenharia e diagnósticos históricos.

## Benchmark

Explica o que é medido e como interpretar os resultados.

- [Bench explicado](benchmark/bench-explicado.md)
- [Metodologia](benchmark/metodologia.md)
- [Observabilidade headless e interativa com Prometheus](benchmark/observabilidade-prometheus.md)
- [Contrato do workload MOPEP](benchmark/workload-mopep.md)

## Make e execução

Explica os alvos, dependências, preparação, smokes, bateria base e sweep.

- [Fluxo completo do Make](make/make-fluxo.md)

## Engenharia e operação

- [Código explicado](engenharia/codigo-explicado.md)
- [Git no RunPod](engenharia/git-runpod.md)
- [Segurança antes de publicar](engenharia/seguranca-publicacao.md)

## Diagnósticos históricos

Resultados, diagnósticos e planos datados foram preservados em `docs/archive/inference-runtime/` na raiz do monorepo. Eles não são instruções para trocar o protocolo corrente.

Os testes funcionais e benchmarks devem ser executados no Pod ou host com o runtime apropriado.
