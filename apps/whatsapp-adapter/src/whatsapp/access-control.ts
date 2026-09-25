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

/**
 * Extrai candidatos de número (somente dígitos) para comparar com a allowlist.
 * Usa jidDecode para ignorar sufixo de device em JIDs como 5511...@s.whatsapp.net.
 * Quando a chave vier como @lid, usa senderPn (telefone) quando o Baileys expõe.
 */
export function extractSenderCandidates(msg: WAMessage): string[] {
  const key = msg.key as {
    participant?: string | null;
    remoteJid?: string | null;
    senderPn?: string | null;
  };
  const candidates = new Set<string>();

  const pn = key.senderPn;
  if (pn) {
    const d = normalizeContactDigits(pn);
    if (d) candidates.add(d);
  }

  const participant = key.participant ?? undefined;
  const remote = key.remoteJid ?? undefined;

  if (participant) {
    const decoded = jidDecode(participant);
    const d = digitsFromDecodedUser(decoded?.user);
    if (d) candidates.add(d);
  }

  if (remote) {
    const decoded = jidDecode(remote);
    const d = digitsFromDecodedUser(decoded?.user);
    if (d) candidates.add(d);
  }

  return [...candidates];
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
