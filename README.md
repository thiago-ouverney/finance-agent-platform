# Finance Agent Platform

> **Objetivo:** apresentar o monorepo e direcionar cada operação para um único
> índice de documentação.

Plataforma de agente financeiro via WhatsApp, inferência local, avaliação de
qualidade e quantização do Qwen.

## Comece aqui

Use o [guia central do projeto](docs/README.md) para escolher o que deseja
fazer: configurar SSH, gerar tabelas, avaliar modelos, quantizar, medir
performance ou iniciar o WhatsApp.

A [arquitetura](docs/architecture.md) explica os limites entre os componentes.

Para reproduzir a quantização no RunPod, siga o
[guia do modelo quantizado](docs/guias/gerar-modelo-quantizado.md).

Para gerar `Q4_K_M`, `Q4_K_M` com iMatrix e `Q8_0` e compará-los nos três
runtimes, siga o [guia GGUF/iMatrix](docs/guias/gerar-gguf-imatrix.md).

## Componentes

- `apps/whatsapp-adapter`: conversa entre WhatsApp e o runtime, com histórico
  isolado por contato;
- `services/inference-runtime`: comparação de vLLM, llama.cpp e Ollama;
- `pipelines/qwen-quantization`: geração e avaliação de artefatos quantizados;
- `analytics`: datasets, avaliações MOPEP e análise dos benchmarks.

## Preparação e validação

Execute `make` sem argumentos para listar todos os alvos disponíveis.

```bash
make setup
make test
make build
make verify
```

`make setup` instala o adapter e prepara `analytics/.venv`. O adapter requer
Node.js 20.19 ou superior. Cada componente mantém seus comandos específicos no
guia central.

## Segurança

Não versione `.env`, credenciais, sessões do WhatsApp, chaves, pesos de
modelos, ambientes virtuais ou resultados brutos. Use os arquivos
`.env.example` apenas como referência.
