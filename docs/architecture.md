# Arquitetura

## Fluxo principal

1. `pipelines/qwen-quantization` prepara e avalia os artefatos do Qwen.
2. `services/inference-runtime` executa e compara Ollama, llama.cpp e vLLM com protocolo compatível com OpenAI.
3. `analytics` consolida resultados, gera tags BMC e treina modelos preditivos.
4. `apps/whatsapp-adapter` recebe texto pelo Baileys, recupera o histórico recente do contato e chama o endpoint local de inferência.

```text
Qwen / Hugging Face
        |
        v
pipelines/qwen-quantization
        |
        v
services/inference-runtime
        |
        +------> analytics
        |
        v
apps/whatsapp-adapter
        |
        v
WhatsApp
```

## Limites

- Pesos e resultados brutos não pertencem ao Git.
- O prompt fixo pertence ao adapter WhatsApp.
- O adapter persiste uma janela curta por contato e reenvia, por padrão, as últimas 10 mensagens; o runtime aplica o limite final de contexto.
- O runtime expõe inferência; ele não deve conhecer contatos ou regras do canal.
- O pipeline produz artefatos versionáveis externamente, preferencialmente identificados por revisão imutável.
- Analytics consome artefatos do benchmark; não implementa uma segunda medição de TTFT ou KV-cache.

## Origem dos componentes

Os históricos dos três repositórios anteriores foram preservados como imports `git subtree`, sem squash. O desenvolvimento novo deve ocorrer neste monorepo.
