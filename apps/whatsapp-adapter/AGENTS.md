# Instruções do adapter WhatsApp

Este guia complementa o [`AGENTS.md` raiz](../../AGENTS.md).

## Papel e fronteira

O adapter é dono do canal WhatsApp, allowlist, histórico curto por contato,
montagem da conversa e chamada ao contrato OpenAI-compatible. O runtime é dono
de serving e contexto final; o adapter não escolhe estratégia de benchmark nem
incorpora implementação de inferência.

Para operação real, a fonte canônica é
[`docs/uff-ollama-whatsapp.md`](../../docs/uff-ollama-whatsapp.md). O fluxo atual
usa o adapter local diretamente contra o Ollama por túnel `uff-delta`.

## Limites

- `.env`, sessão Baileys, QR code, contatos autorizados e históricos são
  sensíveis e nunca entram no Git, logs de teste ou fixtures.
- Não abra sessão, autorize contato, envie mensagem nem teste número real sem
  autorização explícita e target confirmado.
- Não substitua Ollama por `llama-server` nem introduza gateway intermediário
  sem uma mudança de contrato aprovada.
- Preserve isolamento por contato e serialização das mensagens do mesmo
  contato. Concorrência entre contatos exige teste específico.
- Exemplos de ambiente documentam nomes e formatos, nunca valores reais.
- A integração é não oficial; mudanças de reconexão e envio devem considerar
  risco de banimento, duplicação e efeito no destinatário.

## Antes de alterar

Confirme se a mudança toca sessão, allowlist, persistência, truncamento,
reconexão ou provider. Para diagnóstico sem efeito externo, prefira testes e o
script `doctor`; leitura do runbook é obrigatória antes de conexão com a UFF.

## Evidência

O gate local do componente é:

```bash
npm --prefix apps/whatsapp-adapter run verify
```

Teste unitariamente falhas, retries, isolamento e persistência com diretórios
temporários. O handoff distingue testes locais de sessão real e lista qualquer
mensagem, contato ou escrita externa apenas de forma sanitizada.

