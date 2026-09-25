# Benchmark Ollama no RunPod - Terminal 1 (servidor)

Use este terminal somente para controlar o servidor Ollama. Os benchmarks e a validação ficam no [Terminal 2](benchmark-ollama-terminal-2-execucao.md).

## 1. Conectar ao Pod

Na sua máquina:

```bash
ssh runpod-qwen
```

Nesse guia, `runpod-qwen` aponta para o proxy do RunPod, que já foi testado com sucesso. Se o alias SSH não estiver configurado, consulte [runpod-ssh-multiplos-terminais.md](runpod-ssh-multiplos-terminais.md).

## 2. Conferir o ambiente

```bash
ollama --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Se ambos funcionarem, prossiga.

## 3. Iniciar o servidor manual

Use este modo para:

- benchmark nativo do Ollama;
- smoke contra servidor existente;
- bateria unificada `short medium`;
- cenário longo opcional.

```bash
export OLLAMA_HOST=127.0.0.1:11434

# Descomente somente se o modelo foi importado neste diretório.
# export OLLAMA_MODELS=/workspace/ollama-models

ollama serve
```

Deixe o processo aberto neste terminal. As mensagens do servidor aparecerão aqui.

## 4. Encerrar antes do teste de startup

Quando o Terminal 2 chegar à seção "Benchmark de startup", pressione:

```text
Ctrl+C
```

Confirme que o processo encerrou:

```bash
curl -sS http://127.0.0.1:11434/api/version
```

O resultado esperado é erro de conexão. Não inicie outro `ollama serve`: nessa etapa, o próprio benchmark iniciará e encerrará o servidor.

Enquanto o startup roda no Terminal 2, este terminal pode monitorar a GPU:

```bash
watch -n 1 nvidia-smi
```

Saia do `watch` com `Ctrl+C`.

## 5. Voltar ao modo manual

Depois dos testes de startup, inicie novamente:

```bash
export OLLAMA_HOST=127.0.0.1:11434

# Descomente somente se aplicável.
# export OLLAMA_MODELS=/workspace/ollama-models

ollama serve
```

## Problemas comuns

### Porta ocupada

```bash
pgrep -a ollama
curl -s http://127.0.0.1:11434/api/version
```

Se a API responder, já existe um servidor ativo. Não inicie um segundo processo.

### Modelo usando CPU

Depois que o Terminal 2 fizer uma inferência:

```bash
ollama ps
```

O resultado esperado é `100% GPU`. Se houver offload, registre a ocorrência e não trate a rodada como equivalente a uma execução integralmente na GPU.

### Log do servidor gerenciado

Durante uma execução com `--launch`:

```bash
cd /workspace/chatbot-runtime-bench
LATEST="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tail -f "$LATEST/server.log"
```
