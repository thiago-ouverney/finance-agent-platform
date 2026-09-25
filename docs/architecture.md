# Arquitetura

## Fluxo principal

1. `pipelines/qwen-quantization` prepara e avalia os artefatos do Qwen.
2. `services/inference-runtime` executa e compara Ollama, llama.cpp e vLLM com protocolo compatível com OpenAI.
3. `apps/whatsapp-adapter` recebe mensagens pelo Baileys e consulta o endpoint local de inferência.

```text
Qwen / Hugging Face
        |
        v
pipelines/qwen-quantization
        |
        v
services/inference-runtime
        |
        v
apps/whatsapp-adapter
        |
        v
WhatsApp
```

## Limites

- Pesos e resultados brutos não pertencem ao Git.
- O prompt e o histórico conversacional pertencem ao adaptador WhatsApp.
- O runtime expõe inferência; ele não deve conhecer contatos ou regras financeiras.
- O pipeline produz artefatos versionáveis externamente, preferencialmente identificados por revisão imutável.

## Origem dos componentes

Os históricos dos três repositórios anteriores foram preservados como imports `git subtree`, sem squash. O desenvolvimento novo deve ocorrer neste monorepo.
