import {
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
  jidNormalizedUser,
  makeWASocket,
  proto,
  useMultiFileAuthState,
  type BaileysEventMap,
} from "@whiskeysockets/baileys";
import type { Boom } from "@hapi/boom";
import Pino from "pino";
import qrcode from "qrcode-terminal";
import type { EnvConfig } from "../config/env.js";
import { handleUserMessage } from "../agent/finance-agent.js";
import { LocalStore } from "../storage/store.js";
import { extractSenderCandidates } from "../whatsapp/access-control.js";
import {
  deniedMessageForReason,
  featureDisabledMessage,
  noFeaturesMessage,
} from "../policy/messages.js";
import type { PolicyService } from "../policy/resolver.js";
import type { EffectivePolicy } from "../policy/types.js";
import {
  isPermanentFatalDisconnect,
  MAX_RECONNECT_SCHEDULES_DEFAULT,
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

function scheduleReconnect(cfg: EnvConfig, policySvc: PolicyService): void {
  if (!shouldScheduleAnotherReconnect(reconnectAttempt)) {
    console.error(
      `[whatsapp] Limite de reconexões (${MAX_RECONNECT_SCHEDULES_DEFAULT}) atingido sem conexão estável. Pare o processo, corrija a causa (ex.: Node 20+, crypto global, native sqlite) e suba de novo.`
    );
    process.exit(1);
  }
  const ms =
    RECONNECT_MS[Math.min(reconnectAttempt, RECONNECT_MS.length - 1)] ?? 30000;
  reconnectAttempt += 1;
  console.info(`[whatsapp] Reconexão em ${ms}ms (tentativa ${reconnectAttempt})`);
  setTimeout(() => {
    startWhatsAppBot(cfg, policySvc).catch(console.error);
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
  policySvc: PolicyService
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
    browser: ["whatsapp-finance-agent", "Chrome", "1.0.0"],
    markOnlineOnConnect: false,
    syncFullHistory: false,
  });

  lastSocket = sock;

  const store = new LocalStore(cfg.dataDir);

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
            "[whatsapp] Erro permanente de configuração/runtime — reconexão não ajuda. Corrija (ex.: Node 20 LTS, não usar Bun para este app, `npm install` no mesmo Node) e suba de novo."
          );
          process.exit(1);
        }

        const shouldReconnect =
          statusCode !== DisconnectReason.loggedOut &&
          statusCode !== DisconnectReason.badSession;
        if (shouldReconnect) {
          scheduleReconnect(cfg, policySvc);
        }
      } else if (connection === "open") {
        reconnectAttempt = 0;
        console.info("[whatsapp] WhatsApp conectado.");
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

        const candidates = extractSenderCandidates(msg);
        const access = policySvc.resolveFromCandidates(candidates);

        if (access.kind === "denied") {
          console.info(
            `[whatsapp] Acesso negado jid=${jid} reason=${access.reason} candidatos=[${candidates.join(",")}]`
          );
          if (cfg.replyToDenied) {
            const text = deniedMessageForReason(
              policySvc.db,
              cfg,
              access.reason
            );
            await sock.sendMessage(jid, { text });
          }
          continue;
        }

        if (access.kind === "blocked_no_features") {
          console.info(
            `[whatsapp] Sem features jid=${jid} contact=${access.contactId}`
          );
          if (cfg.replyToDenied) {
            await sock.sendMessage(jid, {
              text: noFeaturesMessage(policySvc.db),
            });
          }
          continue;
        }

        const authorizedContactId = access.contactId;
        const effectivePolicy: EffectivePolicy = {
          model: access.model,
          features: access.features,
          identitySource: access.identitySource,
        };
        const featureDisabledReply = featureDisabledMessage(policySvc.db);

        const unwrapped = unwrapMessage(msg.message);
        if (!unwrapped) {
          console.info(`[whatsapp] Ignorado: unwrap vazio jid=${jid}`);
          continue;
        }

        let body = extractText(unwrapped);
        let filename: string | undefined;
        let rawCsvFromFile: string | undefined;
        const doc = unwrapped.documentMessage;
        const lowerName = doc?.fileName?.toLowerCase() ?? "";
        if (lowerName.endsWith(".csv")) {
          filename = doc?.fileName ?? undefined;
          try {
            const buf = await downloadMediaMessage(
              msg,
              "buffer",
              {},
              {
                logger: baileysLogger,
                reuploadRequest: sock.updateMediaMessage,
              }
            );
            rawCsvFromFile = buf.toString("utf8");
            if (!body?.trim()) {
              body = buf.toString("utf8");
            }
            console.info(`[whatsapp] CSV baixado jid=${jid} bytes=${buf.length}`);
          } catch (e) {
            if (!body?.trim()) {
              console.error("[whatsapp] Falha ao baixar CSV:", e);
              await sock.sendMessage(jid, {
                text: "Não consegui ler o arquivo CSV. Envie o texto colado ou tente de novo.",
              });
              continue;
            }
            console.warn(
              `[whatsapp] Falha ao baixar CSV, seguindo com texto/caption jid=${jid}`
            );
          }
        }

        if (!body?.trim()) {
          console.info(
            `[whatsapp] Ignorado: sem texto/caption após unwrap jid=${jid} temDoc=${Boolean(doc)}`
          );
          continue;
        }

        const normalized = jidNormalizedUser(jid);
        const sourceChatId = normalized ?? jid;

        try {
          const reply = await handleUserMessage(
            cfg,
            store,
            authorizedContactId,
            body,
            {
            filename,
            sourceChatId,
            sourceJid: jid,
            rawCsvFromFile,
            policy: effectivePolicy,
            featureDisabledReply,
          }
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
