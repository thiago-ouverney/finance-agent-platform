# Benchmark Ollama no RunPod - Terminal 2 (execução)

Este terminal executa o benchmark nativo do Ollama e o benchmark unificado do projeto `chatbot-runtime-bench`. Antes de começar, mantenha o `ollama serve` aberto conforme o [guia do Terminal 1](benchmark-ollama-terminal-1-servidor.md).

## Configuração usada

| Item | Valor |
|---|---|
| GPU | RTX 4090, 24 GiB |
| Modelo | Qwen2.5-14B-Instruct |
| Quantização | GGUF Q8_0 |
| GGUF | `/workspace/models/qwen14b/Qwen2.5-14B-Instruct-Q8_0.gguf` |
| Alias Ollama | `qwen2.5-14b-q8:latest` |
| Contexto inicial | 4096 tokens |
| Projeto | `/workspace/chatbot-runtime-bench` |

Não altere silenciosamente modelo, hash, contexto ou runtime entre rodadas.

## 1. Conectar e preparar

Na sua máquina:

```bash
ssh runpod-qwen
```

O alias `runpod-qwen` usa o proxy do RunPod, que já foi testado com sucesso. Não use o endereço TCP direto enquanto ele continuar pedindo senha.

No Pod:

```bash
cd /workspace/chatbot-runtime-bench
source .venv/bin/activate

export GGUF_PATH=/workspace/models/qwen14b/Qwen2.5-14B-Instruct-Q8_0.gguf
export MODEL_ID=qwen2.5-14b-q8:latest
export OLLAMA_HOST=127.0.0.1:11434

# Descomente somente se o modelo foi importado neste diretório.
# export OLLAMA_MODELS=/workspace/ollama-models
```

Valide os arquivos e a API:

```bash
test -s "$GGUF_PATH"
test -f configs/ollama-runpod.json

ls -lh "$GGUF_PATH"
df -h /workspace /
nvidia-smi

curl -s http://127.0.0.1:11434/api/version | jq
curl -s http://127.0.0.1:11434/v1/models | jq
```

## 2. Fixar modelo, hash e versão

Confirme que o alias existe:

```bash
curl -s http://127.0.0.1:11434/v1/models |
  jq -e --arg model "$MODEL_ID" 'any(.data[]; .id == $model)'
```

O comando deve imprimir `true`.

Registre a identidade:

```bash
export GGUF_SHA="$(sha256sum "$GGUF_PATH" | awk '{print $1}')"
export OLLAMA_VERSION="$(ollama --version | tail -n 1)"

printf 'MODEL_ID=%s\n' "$MODEL_ID"
printf 'GGUF_SHA=%s\n' "$GGUF_SHA"
printf 'OLLAMA_VERSION=%s\n' "$OLLAMA_VERSION"
```

Atualize a configuração:

```bash
TMP_CONFIG="$(mktemp)"

jq \
  --arg model "$MODEL_ID" \
  --arg version "$OLLAMA_VERSION" \
  --arg artifact "sha256:${GGUF_SHA}; GGUF Q8_0" \
  '.model = $model
   | .runtime_version = $version
   | .model_artifact = $artifact' \
  configs/ollama-runpod.json > "$TMP_CONFIG"

mv "$TMP_CONFIG" configs/ollama-runpod.json
jq . configs/ollama-runpod.json
```

Valide os campos obrigatórios:

```bash
jq -e '
  (.model | length > 0)
  and (.runtime_version | length > 0)
  and (.model_artifact | test("^sha256:[0-9a-f]{64}; GGUF Q8_0$"))
' configs/ollama-runpod.json
```

O comando deve imprimir `true`.

## 3. Benchmark nativo do Ollama

Este resultado serve como diagnóstico do runtime. A comparação entre runtimes deve usar o benchmark unificado da próxima seção.

```bash
mkdir -p /workspace/results/native

ollama run --verbose "$MODEL_ID" \
  "Escreva uma análise detalhada sobre os impactos da inteligência artificial na educação, economia e ciência." \
  2>&1 | tee /workspace/results/native/ollama-qwen14b-q8.txt
```

Confira no final:

- `load duration`;
- `prompt eval rate`;
- `eval rate`;
- `total duration`.

Confirme o uso da GPU:

```bash
ollama ps
nvidia-smi
```

## 4. Smoke do benchmark unificado

```bash
python bench.py run \
  --config configs/ollama-runpod.json \
  --local-model-path "$GGUF_PATH" \
  --smoke \
  --scenarios short
```

Espere a mensagem `Concluido`. Depois valide o manifesto:

```bash
LATEST="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

jq '{
  status,
  error,
  model: .config.model,
  runtime: .config.runtime,
  artifact: .config.model_artifact,
  model_availability
}' "$LATEST/manifest.json"
```

Prossiga somente com `status: "complete"`, sem erro e com o artefato correto.

## 5. Bateria formal: short e medium

Recomendação: execute dentro de `tmux` para sobreviver a quedas do SSH.

```bash
tmux new -s qwen-bench
```

Dentro do `tmux`, repita a preparação curta:

```bash
cd /workspace/chatbot-runtime-bench
source .venv/bin/activate

export GGUF_PATH=/workspace/models/qwen14b/Qwen2.5-14B-Instruct-Q8_0.gguf
```

Execute:

```bash
python bench.py run \
  --config configs/ollama-runpod.json \
  --local-model-path "$GGUF_PATH" \
  --scenarios short medium \
  --requests 30 \
  --repetitions 3
```

São 180 requisições medidas, além de aquecimento e referências. Para sair do `tmux` sem interromper, pressione `Ctrl+B` e depois `D`.

## 6. Ler o resultado mais recente

```bash
cd /workspace/chatbot-runtime-bench
LATEST="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
printf '%s\n' "$LATEST"
```

Valide a execução:

```bash
jq '{
  status,
  error,
  runtime: .config.runtime,
  model: .config.model,
  runtime_version: .config.runtime_version,
  model_artifact: .config.model_artifact,
  context_window: .config.context_window,
  scenarios,
  requests,
  repetitions
}' "$LATEST/manifest.json"
```

Extraia as métricas:

```bash
jq '
  .[]
  | select(.phase == "measure")
  | {
      scenario,
      repetition,
      successful: .successful_request_count,
      expected,
      errors: .errored_request_count,
      incomplete: .incomplete_request_count,
      missing: .missing_request_count,
      ttft_p50_ms: .request_first_token_latency_milliseconds_p50,
      latency_p50_s: .request_latency_seconds_p50,
      mean_itl_p50_ms: .within_response_next_token_latency_milliseconds_p50,
      decode_tokens_s_p50: .decode_generation_tokens_per_second_p50,
      effective_tokens_s_p50: .effective_output_tokens_per_second_p50,
      input_tokens_p50: .input_prompt_token_count_p50,
      output_tokens_p50: .output_completion_token_count_p50,
      exploratory: .percentiles_are_exploratory
    }
' "$LATEST/summary.json"
```

Cada bloco deve ter:

```text
successful == expected
errors == 0
incomplete == 0
missing == 0
```

## 7. Benchmark de startup

Antes desta seção, encerre `ollama serve` no Terminal 1 com `Ctrl+C`. Confirme que a porta está livre:

```bash
curl -sS http://127.0.0.1:11434/api/version
```

O resultado esperado é erro de conexão.

Crie o comando de inicialização:

```bash
cd /workspace/chatbot-runtime-bench
printf '%s\n' '["ollama", "serve"]' > configs/launch-ollama.json
jq . configs/launch-ollama.json
```

Faça primeiro um smoke gerenciado:

```bash
source .venv/bin/activate

export GGUF_PATH=/workspace/models/qwen14b/Qwen2.5-14B-Instruct-Q8_0.gguf
export OLLAMA_HOST=127.0.0.1:11434

# Descomente somente se aplicável.
# export OLLAMA_MODELS=/workspace/ollama-models

python bench.py run \
  --config configs/ollama-runpod.json \
  --local-model-path "$GGUF_PATH" \
  --launch configs/launch-ollama.json \
  --smoke \
  --scenarios short \
  --startup-timeout 1800 \
  --initial-state 'GGUF no SSD; Ollama parado; caches de SO/CUDA não limpos'
```

Se concluir, registre três partidas independentes:

```bash
for startup_run in 1 2 3; do
  python bench.py run \
    --config configs/ollama-runpod.json \
    --local-model-path "$GGUF_PATH" \
    --launch configs/launch-ollama.json \
    --scenarios short \
    --requests 30 \
    --repetitions 1 \
    --startup-timeout 1800 \
    --initial-state "startup-${startup_run}; GGUF no SSD; Ollama parado; caches de SO/CUDA não limpos"
done
```

Leia o ciclo de vida da execução mais recente:

```bash
LATEST="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
jq . "$LATEST/lifecycle.json"
```

Não confunda:

- processo até API disponível;
- processo até primeiro conteúdo;
- TTFT aquecido;
- throughput de decode.

## 8. Cenário longo opcional

Não rode `long` com contexto 4096. Primeiro, reinicie o servidor manual no Terminal 1.

Crie um alias com contexto 9216:

```bash
printf '%s\n' \
  'FROM /workspace/models/qwen14b/Qwen2.5-14B-Instruct-Q8_0.gguf' \
  'PARAMETER num_ctx 9216' \
  > /workspace/Modelfile-qwen14b-ctx9216

ollama create qwen2.5-14b-q8-ctx9216 \
  -f /workspace/Modelfile-qwen14b-ctx9216
```

Crie a configuração separada:

```bash
cp configs/ollama-runpod.json configs/ollama-runpod-ctx9216.json

jq '
  .model = "qwen2.5-14b-q8-ctx9216:latest"
  | .context_window = 9216
  | .server_command = "OLLAMA_HOST=127.0.0.1:11434 ollama serve; Modelfile num_ctx=9216"
  | .notes = "RunPod RTX 4090; contexto 9216; validar 100% GPU"
' configs/ollama-runpod-ctx9216.json > /tmp/ollama-runpod-ctx9216.json

mv /tmp/ollama-runpod-ctx9216.json configs/ollama-runpod-ctx9216.json
```

Teste antes da bateria:

```bash
python bench.py run \
  --config configs/ollama-runpod-ctx9216.json \
  --local-model-path "$GGUF_PATH" \
  --smoke \
  --scenarios long

ollama ps
```

Somente com smoke completo e `100% GPU`:

```bash
python bench.py run \
  --config configs/ollama-runpod-ctx9216.json \
  --local-model-path "$GGUF_PATH" \
  --scenarios short medium long \
  --requests 100 \
  --repetitions 3
```

## 9. Empacotar os resultados

```bash
cd /workspace/chatbot-runtime-bench
LATEST="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
ARCHIVE="/workspace/$(basename "$LATEST")-ollama-benchmark.tar.gz"

tar -czf "$ARCHIVE" "$LATEST"
sha256sum "$ARCHIVE"
ls -lh "$ARCHIVE"
```

No estado atual, preserve o arquivo em `/workspace`: a conexão funcional pelo proxy não oferece SCP/SFTP.

Depois de corrigir a autenticação do endereço TCP direto, na sua máquina e fora do Pod:

```bash
scp runpod-qwen-tcp:/workspace/NOME-DO-ARQUIVO.tar.gz ~/Downloads/
```

## Checklist curto

- [ ] Mesmo GGUF e SHA-256 em todas as rodadas comparadas.
- [ ] Modelo do JSON coincide com `/v1/models`.
- [ ] Mesmo tokenizer, contexto e número de tokens de saída.
- [ ] `ollama ps` mostra `100% GPU`.
- [ ] Smoke manual concluído.
- [ ] Bateria `short medium` concluída sem erros ou incompletas.
- [ ] Servidor manual parado antes de usar `--launch`.
- [ ] Três partidas independentes registradas para startup.
- [ ] Cenário longo executado somente com contexto suficiente.
- [ ] Resultados e logs preservados, inclusive em caso de falha.
