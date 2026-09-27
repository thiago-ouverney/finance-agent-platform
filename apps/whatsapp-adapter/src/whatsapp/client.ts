import {
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeWASocket,
  proto,
  useMultiFileAuthState,
  type BaileysEventMap,
} from "@whiskeysockets/baileys";
import type { Boom } from "@hapi/boom";
import Pino from "pino";
import qrcode from "qrcode-terminal";
import type { EnvConfig } from "../config/env.js";
import { handleUserMessage } from "../agent/chat-agent.js";
import type { ConversationStore } from "../storage/conversation-store.js";
import {
  checkAccessCandidates,
  primeAllowedPhoneMappings,
  resolveSenderCandidates,
} from "../whatsapp/access-control.js";
import {
  isPermanentFatalDisconnect,
  MAX_RECONNECT_SCHEDULES_DEFAULT,
  shouldReconnectAfterStatus,
  shouldScheduleAnotherReconnect,
} from "./reconnect-policy.js";
import { applyBaileysDecryptConsoleFilter } from "./libsignal-console-filter.js";

/** Último socket ativo — encerrado antes de criar outro (evita vazamento em loop de reconexão). */
let lastSocket: { end: (err?: Error | undefined) => void } | null = null;

function disposeLastSocket(): void {
  if (!lastSocket) return;
  try {
    lastSocket.end(undefined);
  } catch (e) {
    console.warn("[whatsapp] ao encerrar socket anterior:", e);
  }
  lastSocket = null;
}

/** Evita empilhar listeners em reconexões e aplica backoff exponencial */
let reconnectAttempt = 0;
const RECONNECT_MS = [1000, 2000, 5000, 10000, 30000];

function scheduleReconnect(cfg: EnvConfig, store: ConversationStore): void {
  if (!shouldScheduleAnotherReconnect(reconnectAttempt)) {
    console.error(
      `[whatsapp] Limite de reconexões (${MAX_RECONNECT_SCHEDULES_DEFAULT}) atingido sem conexão estável. Pare o processo, corrija a causa e suba de novo.`
    );
    process.exit(1);
  }
  const ms =
    RECONNECT_MS[Math.min(reconnectAttempt, RECONNECT_MS.length - 1)] ?? 30000;
  reconnectAttempt += 1;
  console.info(`[whatsapp] Reconexão em ${ms}ms (tentativa ${reconnectAttempt})`);
  setTimeout(() => {
    startWhatsAppBot(cfg, store).catch(console.error);
  }, ms);
}

/**
 * Desembrulha ephemeral, viewOnce e documentWithCaption para chegar ao conteúdo real.
 */
export function unwrapMessage(
  content: proto.IMessage | null | undefined,
  depth = 0
): proto.IMessage | undefined {
  if (!content || depth > 12) return content ?? undefined;

  const inner =
    content.ephemeralMessage?.message ??
    content.viewOnceMessage?.message ??
    content.viewOnceMessageV2?.message ??
    content.documentWithCaptionMessage?.message ??
    undefined;

  if (inner) {
    return unwrapMessage(inner, depth + 1);
  }
  return content;
}

export async function startWhatsAppBot(
  cfg: EnvConfig,
  store: ConversationStore
): Promise<void> {
  disposeLastSocket();

  applyBaileysDecryptConsoleFilter(cfg.baileysLogLevel === "silent");

  const baileysLogger = Pino({ level: cfg.baileysLogLevel });

  const { state, saveCreds } = await useMultiFileAuthState(cfg.sessionPath);
  const { version } = await fetchLatestBaileysVersion();

  const sock = makeWASocket({
    version,
    logger: baileysLogger,
    printQRInTerminal: cfg.printQrInTerminal,
    auth: state,
    browser: ["whatsapp-ai-chat", "Chrome", "1.0.0"],
    markOnlineOnConnect: false,
    syncFullHistory: false,
  });

  lastSocket = sock;

  let allowedMappingsReady: Promise<number> | undefined;
  const ensureAllowedMappings = (): Promise<number> => {
    allowedMappingsReady ??= primeAllowedPhoneMappings(
      cfg.allowedContacts,
      (phoneJid) => sock.signalRepository.lidMapping.getLIDForPN(phoneJid)
    );
    return allowedMappingsReady;
  };

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on(
    "connection.update",
    (update: BaileysEventMap["connection.update"]) => {
      const { connection, lastDisconnect, qr } = update;
      if (qr && cfg.printQrInTerminal) {
        qrcode.generate(qr, { small: true });
      }
      if (connection === "close") {
        const boom = lastDisconnect?.error as Boom | undefined;
        const statusCode = boom?.output?.statusCode;
        const reason = lastDisconnect?.error?.message ?? "";
        console.warn(
          `[whatsapp] Conexão encerrada: statusCode=${statusCode} reason=${reason}`
        );

        if (lastSocket === sock) {
          lastSocket = null;
        }
        try {
          sock.end(undefined);
        } catch (e) {
          console.warn("[whatsapp] ao encerrar socket após close:", e);
        }

        if (isPermanentFatalDisconnect(reason)) {
          console.error(
            "[whatsapp] Erro permanente de configuração/runtime — reconexão não ajuda. Corrija a configuração e suba de novo."
          );
          process.exit(1);
        }

        if (shouldReconnectAfterStatus(statusCode)) {
          scheduleReconnect(cfg, store);
        } else if (statusCode === DisconnectReason.connectionReplaced) {
          console.error(
            "[whatsapp] Sessão substituída por outra instância (status 440). Encerre a outra instância antes de iniciar esta."
          );
        }
      } else if (connection === "open") {
        reconnectAttempt = 0;
        console.info("[whatsapp] WhatsApp conectado.");
        void ensureAllowedMappings().then((resolved) => {
          console.info(
            `[whatsapp] Mapeamentos telefone→LID preparados: ${resolved}/${cfg.allowedContacts.length}`
          );
        });
      }
    }
  );

  sock.ev.on(
    "messages.upsert",
    async (event: BaileysEventMap["messages.upsert"]) => {
      const { messages, type } = event;
      if (type !== "notify") return;

      for (const msg of messages) {
        if (!msg.message || msg.key.fromMe) continue;

        const jid = msg.key.remoteJid;
        if (!jid) {
          console.info("[whatsapp] Ignorado: sem remoteJid");
          continue;
        }

        await ensureAllowedMappings();
        const candidates = await resolveSenderCandidates(
          msg,
          (lidJid) => sock.signalRepository.lidMapping.getPNForLID(lidJid)
        );
        const access = checkAccessCandidates(candidates, cfg);

        if (!access.allowed) {
          console.info(
            `[whatsapp] Acesso negado jid=${jid} candidatos=[${candidates.join(",")}]`
          );
          if (cfg.replyToDenied) {
            await sock.sendMessage(jid, { text: cfg.deniedMessage });
          }
          continue;
        }

        const unwrapped = unwrapMessage(msg.message);
        if (!unwrapped) {
          console.info(`[whatsapp] Ignorado: unwrap vazio jid=${jid}`);
          continue;
        }

        const body = extractText(unwrapped);

        if (!body?.trim()) {
          console.info(
            `[whatsapp] Ignorado: mensagem sem texto jid=${jid}`
          );
          continue;
        }

        try {
          const reply = await handleUserMessage(
            cfg,
            store,
            access.authorizedContactId,
            body
          );
          await sock.sendMessage(jid, { text: reply });
        } catch (e) {
          console.error("[whatsapp] Erro ao processar mensagem:", e);
          await sock.sendMessage(jid, {
            text: "Erro ao processar sua mensagem. Tente novamente.",
          });
        }
      }
    }
  );
}

function extractText(content: proto.IMessage): string | undefined {
  if (content.conversation) return content.conversation;
  if (content.extendedTextMessage?.text)
    return content.extendedTextMessage.text;
  if (content.imageMessage?.caption) return content.imageMessage.caption;
  if (content.videoMessage?.caption) return content.videoMessage.caption;
  if (content.documentMessage?.caption)
    return content.documentMessage.caption;
  return undefined;
}
