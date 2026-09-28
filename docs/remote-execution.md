# Execução local, UFF e RunPod

> **Objetivo:** executar o mesmo alvo de benchmark localmente ou por SSH,
> preservando as variáveis e o protocolo da medição.

O mesmo benchmark deve usar o mesmo alvo Make e as mesmas variáveis em todos os hosts. O runner da raiz apenas escolhe onde executar; ele não altera o protocolo.

## Local

```bash
make benchmark-local BENCH_TARGET=bench-ollama BENCH_VARS='MODEL_SIZE=7B BENCH_REQUESTS=50'
```

## UFF por SSH

Use preferencialmente um alias em `~/.ssh/config`:

```bash
make benchmark-remote \
  REMOTE_HOST=uff-delta \
  REMOTE_DIR=/caminho/finance-agent-platform/services/inference-runtime \
  BENCH_TARGET=bench-all \
  BENCH_VARS='MODEL_SIZE=7B PREPARE_OFFLINE=1'
```

## RunPod por SSH

```bash
make benchmark-remote \
  REMOTE_HOST=runpod-qwen \
  REMOTE_PORT=<porta-ssh-do-pod> \
  REMOTE_DIR=/workspace/finance-agent-platform/services/inference-runtime \
  BENCH_TARGET=bench-ollama
```

`benchmark-remote` informa `REMOTE_PORT` diretamente ao SSH e usa `22` por
padrão. Portanto, repita a porta do alias quando o host usar uma porta não
padrão.

Não versione hostname, IP, usuário, porta ou caminho de chave. Guarde esses valores em `~/.ssh/config` ou informe-os somente na linha de comando.

Depois da execução, use o `pull-results` de `services/inference-runtime` ou copie os resultados para uma pasta local ignorada pelo Git. Em seguida, gere o dataset analítico com `make benchmark-dataset`.

## Benchmark Prometheus sem notebook no Pod

Para a rodada temporal, entre no Pod e execute o alvo no próprio checkout:

```bash
ssh runpod-qwen
cd /workspace/finance-agent-platform/services/inference-runtime
make observe-bench OBS_RUNTIME=vllm \
  OBS_MODEL_SOURCE=hf \
  OBS_MODEL='Qwen/Qwen2.5-7B-Instruct' \
  OBS_REVISION='<revisao-ou-commit>'
```

O alvo prepara modelo e runtime antes da janela medida, inicia a coleta e chama
o `bench.py` sequencialmente. Para comparar runtimes, use o mesmo GGUF e repita
com `OBS_RUNTIME=vllm`, `llama` e `ollama`, mantendo
`OBS_MODEL_SOURCE=local-gguf` e o mesmo `OBS_MODEL`.

Na máquina local:

```bash
cd services/inference-runtime
make pull-observe-results OBS_REMOTE_HOST=runpod-qwen
cd ../..
make observability-dataset
make observability-notebook
```

Essa análise é offline: navegador e kernel ficam locais e nenhum túnel de
Jupyter é necessário. O túnel continua existindo apenas para o perfil
interativo de uma ou poucas requisições.

Não versione credenciais, tokens do Hugging Face, pesos ou resultados brutos.
Valores ausentes nas séries permanecem `null`; eles não significam uso zero.
