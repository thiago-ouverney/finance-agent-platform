# Iniciar o atendimento pelo WhatsApp

> **Objetivo:** iniciar o adapter local do WhatsApp conectado diretamente ao
> Ollama da UFF pela ponte `uff-delta`.
>
> **Resultado esperado:** contatos autorizados conversam com o Qwen, com
> histórico isolado por contato e sem expor o Ollama na internet.

## 1. Verificar a UFF

```bash
ssh uff-delta
curl http://127.0.0.1:11434/api/tags
ollama list
exit
```

Se a API não responder, inicie o Ollama conforme o [runbook
completo](../uff-ollama-whatsapp.md). Não prossiga até o modelo aparecer.

## 2. Abrir a ponte local

Em um terminal local dedicado:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8000:127.0.0.1:11434 \
  uff-delta
```

Em outro terminal, confirme:

```bash
curl http://127.0.0.1:8000/v1/models
```

## 3. Configurar o adapter

Use Node.js 20.19 ou superior.

```bash
cd apps/whatsapp-adapter
test -f .env || cp .env.example .env
npm install
```

No `.env` local, use o nome exato exibido por `ollama list`:

```dotenv
WHATSAPP_INFERENCE_BASE_URL=http://127.0.0.1:8000/v1
WHATSAPP_INFERENCE_MODEL=<nome-exato-no-ollama>
WHATSAPP_INFERENCE_TIMEOUT_MS=120000
INFERENCE_API_TOKEN=ollama
```

Autorize um contato a partir da raiz do repositório:

```bash
cd ../..
make access-add PHONE=55<ddd><numero>
```

## 4. Iniciar e verificar

```bash
npm --prefix apps/whatsapp-adapter run doctor
npm --prefix apps/whatsapp-adapter run dev
```

Escaneie o QR Code se solicitado e envie uma mensagem por um contato autorizado.
Mantenha o terminal da ponte aberto. Sessão, `.env`, contatos e históricos não
devem ser versionados.

Para operação persistente e diagnóstico, consulte [WhatsApp com Ollama na
UFF](../uff-ollama-whatsapp.md).
