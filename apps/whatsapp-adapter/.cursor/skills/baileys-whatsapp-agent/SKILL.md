---
name: baileys-whatsapp-agent
description: Orienta integração com Baileys no projeto whatsapp-finance-agent—sessão multi-arquivo, QR no terminal, eventos connection.update e messages.upsert, download de mídia CSV e allowlist de remetentes. Use ao alterar cliente WhatsApp, auth ou fluxo de mensagens.
---

# Baileys no whatsapp-finance-agent

## Configuração

- **Sessão**: `useMultiFileAuthState(SESSION_PATH)`; persistir credenciais em disco (ex.: `.baileys_auth/`).
- **QR**: `connection.update` com `qr`; opcionalmente `qrcode-terminal` para exibir no terminal.
- **Browser tuple**: identificar o cliente de forma estável, ex.: `["whatsapp-finance-agent", "Chrome", "1.0.0"]`.

## Eventos principais

- `connection.update`: QR, `connection === "open"`, fechamento com `lastDisconnect` — usar `DisconnectReason` para decidir reconectar.
- `messages.upsert`: processar apenas `type === "notify"` para mensagens novas.
- `creds.update`: sempre salvar credenciais quando disparado.

## Mensagens

- Ignorar `msg.key.fromMe`.
- Texto: `conversation`, `extendedTextMessage.text`, legendas de mídia.
- **CSV anexo**: `documentMessage` com `.csv` — se não houver legenda, usar `downloadMediaMessage` com `reuploadRequest: sock.updateMediaMessage` e logger.

## Controle de acesso

- Usar `jidDecode` para ignorar sufixo de device nos JIDs; considerar `senderPn` quando a chave for `@lid`.
- Extrair **lista de candidatos** (participant, remoteJid, senderPn) e permitir se algum número normalizado (só dígitos) estiver em `ALLOWED_CONTACTS`.

## Mensagens encapsuladas

- Desembrulhar `documentWithCaptionMessage`, `ephemeralMessage`, `viewOnceMessage(V2)` antes de ler `documentMessage` ou legenda.

## Segurança

- WhatsApp Web não oficial pode violar termos; usar apenas em ambiente controlado.
- Nunca commitar pasta de sessão nem dados sensíveis.
