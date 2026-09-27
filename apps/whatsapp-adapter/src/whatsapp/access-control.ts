import type { WAMessage } from "@whiskeysockets/baileys";
import { jidDecode } from "@whiskeysockets/baileys";
import type { EnvConfig } from "../config/env.js";
import { normalizeContactDigits } from "../config/env.js";

export type AccessDecision =
  | { allowed: true; authorizedContactId: string }
  | { allowed: false; reason: "not_in_allowlist" };

function digitsFromDecodedUser(user: string | undefined): string | undefined {
  if (!user) return undefined;
  return normalizeContactDigits(user);
}

function digitsFromAddress(address: string): string | undefined {
  const decoded = jidDecode(address);
  return digitsFromDecodedUser(decoded?.user ?? address);
}

type AddressingKey = WAMessage["key"] & {
  senderPn?: string | null;
  remoteJidAlt?: string | null;
  participantAlt?: string | null;
};

function addressingJids(msg: WAMessage): string[] {
  const key = msg.key as AddressingKey;
  return [
    key.senderPn,
    key.participantAlt,
    key.remoteJidAlt,
    key.participant,
    key.remoteJid,
  ].filter((jid): jid is string => Boolean(jid));
}

/**
 * Extrai candidatos de número (somente dígitos) para comparar com a allowlist.
 * Usa jidDecode para ignorar sufixo de device em JIDs como 5511...@s.whatsapp.net.
 * Quando a chave vier como @lid, usa senderPn (telefone) quando o Baileys expõe.
 */
export function extractSenderCandidates(msg: WAMessage): string[] {
  const candidates = new Set<string>();

  for (const jid of addressingJids(msg)) {
    const d = digitsFromAddress(jid);
    if (d) candidates.add(d);
  }

  return [...candidates];
}

/**
 * Resolve endereços LID para telefone usando o mapa persistido pelo Baileys.
 * Telefones resolvidos são retornados antes do LID, preservando o telefone
 * cadastrado como identidade canônica da política e dos dados do contato.
 */
export async function resolveSenderCandidates(
  msg: WAMessage,
  getPhoneForLid: (lidJid: string) => Promise<string | null>
): Promise<string[]> {
  const resolvedPhones = new Set<string>();

  for (const jid of addressingJids(msg)) {
    const decoded = jidDecode(jid);
    if (decoded?.server !== "lid") continue;
    const phoneJid = await getPhoneForLid(jid);
    const phone = phoneJid ? digitsFromAddress(phoneJid) : undefined;
    if (phone) resolvedPhones.add(phone);
  }

  return [...resolvedPhones, ...extractSenderCandidates(msg)].filter(
    (candidate, index, all) => all.indexOf(candidate) === index
  );
}

/**
 * Aquece o mapa persistente telefone↔LID apenas para telefones autorizados.
 * A execução sequencial evita disparar consultas simultâneas ao WhatsApp.
 */
export async function primeAllowedPhoneMappings(
  phones: string[],
  getLidForPhone: (phoneJid: string) => Promise<string | null>
): Promise<number> {
  const uniquePhones = [
    ...new Set(phones.map(normalizeContactDigits).filter(Boolean)),
  ];
  let resolved = 0;

  for (const phone of uniquePhones) {
    try {
      if (await getLidForPhone(`${phone}@s.whatsapp.net`)) {
        resolved += 1;
      }
    } catch {
      // Uma falha isolada não deve impedir os demais contatos nem o bot.
    }
  }

  return resolved;
}

/**
 * @deprecated Prefira extractSenderCandidates + checkAccessCandidates
 */
export function extractSenderDigits(msg: WAMessage): string | undefined {
  const c = extractSenderCandidates(msg);
  return c[0];
}

export function checkAccessCandidates(
  candidates: string[],
  cfg: EnvConfig
): AccessDecision {
  if (cfg.allowedContacts.length === 0) {
    return { allowed: false, reason: "not_in_allowlist" };
  }
  const normalized = candidates
    .map((c) => normalizeContactDigits(c))
    .filter((c) => c.length > 0);
  if (normalized.length === 0) {
    return { allowed: false, reason: "not_in_allowlist" };
  }
  const matched = normalized.find((n) => cfg.allowedContacts.includes(n));
  return matched
    ? { allowed: true, authorizedContactId: matched }
    : { allowed: false, reason: "not_in_allowlist" };
}

export function checkAccess(
  senderDigits: string | undefined,
  cfg: EnvConfig
): AccessDecision {
  if (!senderDigits) return { allowed: false, reason: "not_in_allowlist" };
  return checkAccessCandidates([senderDigits], cfg);
}
