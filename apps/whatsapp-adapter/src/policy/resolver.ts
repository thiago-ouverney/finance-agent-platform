import type { EnvConfig } from "../config/env.js";
import { normalizeContactDigits } from "../config/env.js";
import { allFeatureKeys } from "./feature-registry.js";
import { intersectGrantedWithRegistry } from "./orchestrator.js";
import type { PolicyDatabase } from "./policy-db.js";
import type { AccessResolution, PolicySourceMode } from "./types.js";

const KNOWN = new Set<string>(allFeatureKeys());

function normalizeCandidates(candidates: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const c of candidates) {
    const d = normalizeContactDigits(c);
    if (!d || seen.has(d)) continue;
    seen.add(d);
    out.push(d);
  }
  return out;
}

function envAllowlist(cfg: EnvConfig): Set<string> {
  return new Set(cfg.allowedContacts.map((x) => normalizeContactDigits(x)).filter(Boolean));
}

/**
 * Resolve acesso para o primeiro candidato (ordem preservada) que produz decisão terminal.
 * Espelha a ideia de `find` na allowlist legada.
 */
export function resolveAccessForCandidates(
  candidates: string[],
  cfg: EnvConfig,
  db: PolicyDatabase,
  mode: PolicySourceMode
): AccessResolution {
  const normalized = normalizeCandidates(candidates);
  if (normalized.length === 0) {
    return { kind: "denied", reason: "not_registered" };
  }

  const envSet = envAllowlist(cfg);
  const allKeys = allFeatureKeys();
  let sawDisabled = false;

  for (const contactId of normalized) {
    const row = db.getIdentity(contactId);

    if (mode === "db_only") {
      if (!row) {
        continue;
      }
      if (!row.enabled) {
        sawDisabled = true;
        continue;
      }
      return resolveDbAllowed(contactId, cfg, db);
    }

    // hybrid — DB com enabled=0 apenas pula este candidato; tenta o próximo
    if (row) {
      if (!row.enabled) {
        sawDisabled = true;
        continue;
      }
      return resolveDbAllowed(contactId, cfg, db);
    }

    if (envSet.has(contactId)) {
      const model = cfg.openaiModel;
      return {
        kind: "allowed",
        contactId,
        identitySource: "env",
        model,
        features: new Set(allKeys),
      };
    }
  }

  if (sawDisabled) {
    return { kind: "denied", reason: "disabled" };
  }
  return { kind: "denied", reason: "not_registered" };
}

function resolveDbAllowed(
  contactId: string,
  cfg: EnvConfig,
  db: PolicyDatabase
): AccessResolution {
  const modelFromDb = db.getModelForContact(contactId);
  const model = modelFromDb?.trim() || cfg.openaiModel;
  const rawFeats = db.getEnabledFeatures(contactId, KNOWN);
  const features = intersectGrantedWithRegistry(rawFeats);
  if (features.size === 0) {
    return {
      kind: "blocked_no_features",
      contactId,
      identitySource: "db",
    };
  }
  return {
    kind: "allowed",
    contactId,
    identitySource: "db",
    model,
    features,
  };
}

/** Cache TTL por contato (última resolução bem-sucedida ou negada). */
type CacheEntry = { expiresAt: number; value: AccessResolution };

export class PolicyService {
  private cache = new Map<string, CacheEntry>();
  private readonly ttlMs: number;

  constructor(
    readonly db: PolicyDatabase,
    private readonly cfg: EnvConfig,
    private readonly mode: PolicySourceMode,
    ttlMs: number
  ) {
    this.ttlMs = Math.max(1000, ttlMs);
  }

  invalidateCache(): void {
    this.cache.clear();
  }

  private cacheKey(candidates: string[]): string {
    return normalizeCandidates(candidates).join("|");
  }

  resolveFromCandidates(candidates: string[]): AccessResolution {
    const key = this.cacheKey(candidates);
    if (!key) {
      return resolveAccessForCandidates(candidates, this.cfg, this.db, this.mode);
    }
    const now = Date.now();
    const hit = this.cache.get(key);
    if (hit && hit.expiresAt > now) {
      return hit.value;
    }
    const value = resolveAccessForCandidates(
      candidates,
      this.cfg,
      this.db,
      this.mode
    );
    this.cache.set(key, { expiresAt: now + this.ttlMs, value });
    return value;
  }
}
