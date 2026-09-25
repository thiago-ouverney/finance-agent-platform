import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import type { EnvConfig } from "../config/env.js";
import { normalizeContactDigits } from "../config/env.js";
import type { MessageTemplateKey } from "./types.js";
import { allFeatureKeys } from "./feature-registry.js";

export type IdentityRow = {
  contact_digits: string;
  display_label: string | null;
  enabled: number;
};

const MIGRATION_V1 = `
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS identities (
  contact_digits TEXT PRIMARY KEY,
  display_label TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_policy (
  contact_digits TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  FOREIGN KEY (contact_digits) REFERENCES identities(contact_digits) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feature_grants (
  contact_digits TEXT NOT NULL,
  feature_key TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (contact_digits, feature_key),
  FOREIGN KEY (contact_digits) REFERENCES identities(contact_digits) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message_templates (
  template_key TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
`;

const DEFAULT_TEMPLATES: Array<{ key: MessageTemplateKey; body: string }> = [
  {
    key: "access_denied_not_registered",
    body: "Este número não está autorizado a usar o agente financeiro.",
  },
  {
    key: "access_denied_disabled",
    body: "Seu acesso ao agente financeiro está desativado. Fale com o administrador.",
  },
  {
    key: "registered_no_features",
    body: "Seu cadastro existe, mas nenhuma funcionalidade está liberada no momento. Fale com o administrador.",
  },
  {
    key: "feature_disabled",
    body: "Esta funcionalidade não está liberada para o seu usuário. Fale com o administrador.",
  },
];

function ensureDirForFile(filePath: string): void {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
}

export class PolicyDatabase {
  readonly raw: Database.Database;

  constructor(filePath: string) {
    ensureDirForFile(filePath);
    this.raw = new Database(filePath);
    this.raw.pragma("journal_mode = WAL");
    this.raw.pragma("foreign_keys = ON");
    this.raw.exec(MIGRATION_V1);
    this.ensureDefaultTemplates();
  }

  close(): void {
    this.raw.close();
  }

  private ensureDefaultTemplates(): void {
    const ins = this.raw.prepare(
      `INSERT OR IGNORE INTO message_templates (template_key, body, updated_at)
       VALUES (?, ?, datetime('now'))`
    );
    for (const t of DEFAULT_TEMPLATES) {
      ins.run(t.key, t.body);
    }
  }

  countIdentities(): number {
    const row = this.raw
      .prepare("SELECT COUNT(*) AS c FROM identities")
      .get() as { c: number };
    return row.c;
  }

  /** Seed inicial a partir do .env quando o banco está vazio (paridade com deploy legado). */
  seedFromEnvIfEmpty(cfg: EnvConfig): void {
    if (this.countIdentities() > 0) return;
    const contacts = cfg.allowedContacts;
    if (contacts.length === 0) return;

    const insertId = this.raw.prepare(
      `INSERT INTO identities (contact_digits, display_label, enabled, created_at, updated_at)
       VALUES (?, ?, 1, datetime('now'), datetime('now'))`
    );
    const insertLlm = this.raw.prepare(
      `INSERT INTO llm_policy (contact_digits, model_id) VALUES (?, ?)`
    );
    const insertFeat = this.raw.prepare(
      `INSERT INTO feature_grants (contact_digits, feature_key, enabled) VALUES (?, ?, 1)`
    );

    const tx = this.raw.transaction(() => {
      for (const rawDigits of contacts) {
        const d = normalizeContactDigits(rawDigits);
        if (!d) continue;
        insertId.run(d, null);
        insertLlm.run(d, cfg.openaiModel);
        for (const fk of allFeatureKeys()) {
          insertFeat.run(d, fk);
        }
      }
    });
    tx();
  }

  getIdentity(contactDigits: string): IdentityRow | undefined {
    return this.raw
      .prepare(
        `SELECT contact_digits, display_label, enabled FROM identities WHERE contact_digits = ?`
      )
      .get(contactDigits) as IdentityRow | undefined;
  }

  getModelForContact(contactDigits: string): string | undefined {
    const row = this.raw
      .prepare(`SELECT model_id FROM llm_policy WHERE contact_digits = ?`)
      .get(contactDigits) as { model_id: string } | undefined;
    return row?.model_id;
  }

  /** Features habilitadas e conhecidas no registry (chaves órfãs no DB são ignoradas). */
  getEnabledFeatures(
    contactDigits: string,
    knownKeys: ReadonlySet<string>
  ): Set<string> {
    const rows = this.raw
      .prepare(
        `SELECT feature_key FROM feature_grants WHERE contact_digits = ? AND enabled = 1`
      )
      .all(contactDigits) as Array<{ feature_key: string }>;
    const out = new Set<string>();
    for (const r of rows) {
      if (knownKeys.has(r.feature_key)) out.add(r.feature_key);
    }
    return out;
  }

  getTemplateBody(key: MessageTemplateKey): string | undefined {
    const row = this.raw
      .prepare(`SELECT body FROM message_templates WHERE template_key = ?`)
      .get(key) as { body: string } | undefined;
    return row?.body;
  }

  setTemplateBody(key: MessageTemplateKey, body: string): void {
    this.raw
      .prepare(
        `INSERT INTO message_templates (template_key, body, updated_at)
         VALUES (?, ?, datetime('now'))
         ON CONFLICT(template_key) DO UPDATE SET body = excluded.body, updated_at = datetime('now')`
      )
      .run(key, body);
  }

  listIdentities(): IdentityRow[] {
    return this.raw
      .prepare(
        `SELECT contact_digits, display_label, enabled FROM identities ORDER BY contact_digits`
      )
      .all() as IdentityRow[];
  }

  upsertIdentity(
    contactDigits: string,
    opts: {
      displayLabel?: string | null;
      enabled?: boolean;
      defaultModelIfNew?: string;
    }
  ): void {
    const d = normalizeContactDigits(contactDigits);
    if (!d) throw new Error("contact_digits inválido");
    const enabled = opts.enabled !== false ? 1 : 0;
    const label = opts.displayLabel ?? null;
    const existed = Boolean(
      this.raw.prepare(`SELECT 1 FROM identities WHERE contact_digits = ?`).get(d)
    );
    this.raw
      .prepare(
        `INSERT INTO identities (contact_digits, display_label, enabled, created_at, updated_at)
         VALUES (?, ?, ?, datetime('now'), datetime('now'))
         ON CONFLICT(contact_digits) DO UPDATE SET
           display_label = COALESCE(excluded.display_label, identities.display_label),
           enabled = excluded.enabled,
           updated_at = datetime('now')`
      )
      .run(d, label, enabled);
    if (!existed) {
      const defaultModel = opts.defaultModelIfNew?.trim() || "gpt-4o-mini";
      this.raw
        .prepare(`INSERT OR IGNORE INTO llm_policy (contact_digits, model_id) VALUES (?, ?)`)
        .run(d, defaultModel);
    }
  }

  deleteIdentity(contactDigits: string): void {
    const d = normalizeContactDigits(contactDigits);
    this.raw.prepare(`DELETE FROM identities WHERE contact_digits = ?`).run(d);
  }

  setModel(contactDigits: string, modelId: string): void {
    const d = normalizeContactDigits(contactDigits);
    this.raw
      .prepare(
        `UPDATE llm_policy SET model_id = ? WHERE contact_digits = ?`
      )
      .run(modelId, d);
  }

  getFeatureGrantMap(contactDigits: string): Record<string, boolean> {
    const d = normalizeContactDigits(contactDigits);
    const out: Record<string, boolean> = {};
    for (const fk of allFeatureKeys()) {
      const row = this.raw
        .prepare(
          `SELECT enabled FROM feature_grants WHERE contact_digits = ? AND feature_key = ?`
        )
        .get(d, fk) as { enabled: number } | undefined;
      out[fk] = row ? row.enabled === 1 : false;
    }
    return out;
  }

  replaceFeatureGrants(
    contactDigits: string,
    grants: Record<string, boolean>
  ): void {
    const d = normalizeContactDigits(contactDigits);
    const del = this.raw.prepare(
      `DELETE FROM feature_grants WHERE contact_digits = ?`
    );
    const ins = this.raw.prepare(
      `INSERT INTO feature_grants (contact_digits, feature_key, enabled) VALUES (?, ?, ?)`
    );
    const tx = this.raw.transaction(() => {
      del.run(d);
      for (const [k, v] of Object.entries(grants)) {
        ins.run(d, k, v ? 1 : 0);
      }
    });
    tx();
  }

  getLlmPolicies(): Array<{ contact_digits: string; model_id: string }> {
    return this.raw
      .prepare(`SELECT contact_digits, model_id FROM llm_policy`)
      .all() as Array<{ contact_digits: string; model_id: string }>;
  }

  getFeatureMatrix(): Array<{
    contact_digits: string;
    feature_key: string;
    enabled: number;
  }> {
    return this.raw
      .prepare(
        `SELECT contact_digits, feature_key, enabled FROM feature_grants ORDER BY contact_digits, feature_key`
      )
      .all() as Array<{
      contact_digits: string;
      feature_key: string;
      enabled: number;
    }>;
  }

  listTemplates(): Array<{ template_key: string; body: string }> {
    return this.raw
      .prepare(`SELECT template_key, body FROM message_templates ORDER BY template_key`)
      .all() as Array<{ template_key: string; body: string }>;
  }
}

export function defaultPolicyDbPath(dataDir: string): string {
  return path.join(dataDir, "policy.sqlite");
}
