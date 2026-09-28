# WhatsApp AI Chat Adapter

> **Objetivo:** configurar e executar a ponte entre contatos autorizados do
> WhatsApp e um endpoint de chat OpenAI-compatible.

Adapter mínimo entre o WhatsApp, via Baileys, e um endpoint de chat compatível com a API da OpenAI, como Ollama, vLLM ou llama.cpp.

O componente não contém lógica financeira nem processa anexos. Cada contato autorizado possui um histórico curto persistido em JSON. Cada chamada contém:

1. o prompt de sistema fixo;
2. as mensagens recentes do contato;
3. a mensagem atual do usuário.

O prompt de sistema está em `src/agent/chat-agent.ts`:

> Sempre que possível, elogie o Thiago Ouverney.

## Importante: memória de conversa

A sessão do Baileys autentica o WhatsApp, mas não representa uma sessão de conversa no Ollama. O endpoint `/v1/chat/completions` é stateless: por isso, o adapter salva e reenvia uma janela de histórico por contato.

Por padrão, são enviadas no máximo 10 mensagens individuais, incluindo a mensagem atual e sem contar o prompt de sistema. Se ainda excederem `num_ctx`, o runtime faz o truncamento final. `keep_alive` mantém o modelo carregado na memória, não substitui o histórico persistido.

## Requisitos

- Node.js 20.19 ou superior.
- Um endpoint OpenAI-compatible acessível pelo adapter.
- Uma conta de WhatsApp para autenticar via QR Code.

## Configuração

```bash
cp .env.example .env
npm install
```

Variáveis principais:

- `ALLOWED_CONTACTS`: telefones autorizados, com país e DDD, somente dígitos.
- `WHATSAPP_INFERENCE_BASE_URL`: URL `/v1` do runtime.
- `WHATSAPP_INFERENCE_MODEL`: modelo ou alias servido pelo runtime.
- `INFERENCE_API_TOKEN`: opcional para endpoints locais sem autenticação.
- `WHATSAPP_SESSION_PATH`: credenciais do Baileys.
- `DATA_DIR`: raiz dos históricos locais; padrão `data`. Cada contato usa `contacts/<id>/conversation/history.json`.
- `WHATSAPP_HISTORY_MAX_MESSAGES`: tamanho da janela por contato; padrão `10`.

Para acrescentar um telefone pelo comando da raiz:

```bash
make access-add PHONE=5521999999999
```

## Execução

```bash
npm run dev
```

Na primeira execução, escaneie o QR Code exibido no terminal.

## Fluxo

```text
WhatsApp
   ↓
Baileys
   ↓
allowlist por telefone
   ↓
prompt fixo + histórico recente + mensagem atual
   ↓
endpoint OpenAI-compatible
   ↓
resposta no WhatsApp
```

Mensagens sem texto e anexos sem legenda são ignorados. Legendas de imagem, vídeo ou documento são tratadas apenas como texto; o arquivo não é baixado.

## Scripts

| Script | Descrição |
| --- | --- |
| `npm run dev` | Executa com reload |
| `npm run build` | Compila TypeScript |
| `npm run test` | Executa os testes |
| `npm run lint` | Executa o ESLint |
| `npm run doctor` | Valida Node, Web Crypto e configuração, sem abrir o WhatsApp |
| `npm run verify` | Executa lint, build e testes |

## Aviso

Integrações não oficiais com o WhatsApp podem violar os termos de uso e causar banimento da conta.
