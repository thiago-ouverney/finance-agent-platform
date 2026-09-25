# Execução local, UFF e RunPod

O mesmo benchmark deve usar o mesmo alvo Make e as mesmas variáveis em todos os hosts. O runner da raiz apenas escolhe onde executar; ele não altera o protocolo.

## Local

```bash
make benchmark-local BENCH_TARGET=bench-ollama BENCH_VARS='MODEL_SIZE=7B BENCH_REQUESTS=50'
```

## UFF por SSH

Use preferencialmente um alias em `~/.ssh/config`:

```bash
make benchmark-remote \
  REMOTE_HOST=uff-gpu \
  REMOTE_DIR=/caminho/finance-agent-platform/services/inference-runtime \
  BENCH_TARGET=bench-all \
  BENCH_VARS='MODEL_SIZE=7B PREPARE_OFFLINE=1'
```

## RunPod por SSH

```bash
make benchmark-remote \
  REMOTE_HOST=runpod-qwen \
  REMOTE_DIR=/workspace/finance-agent-platform/services/inference-runtime \
  BENCH_TARGET=bench-ollama
```

Não versione hostname, IP, usuário, porta ou caminho de chave. Guarde esses valores em `~/.ssh/config` ou informe-os somente na linha de comando.

Depois da execução, use o `pull-results` de `services/inference-runtime` ou copie os resultados para uma pasta local ignorada pelo Git. Em seguida, gere o dataset analítico com `make benchmark-dataset`.
