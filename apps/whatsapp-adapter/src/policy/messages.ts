import type { EnvConfig } from "../config/env.js";
import type { PolicyDatabase } from "./policy-db.js";
import type { MessageTemplateKey } from "./types.js";

export function resolveTemplateText(
  db: PolicyDatabase,
  key: MessageTemplateKey,
  fallback: string
): string {
  const t = db.getTemplateBody(key)?.trim();
  return t && t.length > 0 ? t : fallback;
}

export function deniedMessageForReason(
  db: PolicyDatabase,
  cfg: EnvConfig,
  reason: "not_registered" | "disabled"
): string {
  if (reason === "disabled") {
    return resolveTemplateText(
      db,
      "access_denied_disabled",
      cfg.deniedMessage
    );
  }
  return resolveTemplateText(
    db,
    "access_denied_not_registered",
    cfg.deniedMessage
  );
}

export function noFeaturesMessage(db: PolicyDatabase): string {
  return resolveTemplateText(
    db,
    "registered_no_features",
    "Seu cadastro existe, mas nenhuma funcionalidade está liberada no momento. Fale com o administrador."
  );
}

export function featureDisabledMessage(db: PolicyDatabase): string {
  return resolveTemplateText(
    db,
    "feature_disabled",
    "Esta funcionalidade não está liberada para o seu usuário. Fale com o administrador."
  );
}
