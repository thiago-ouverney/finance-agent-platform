import { describe, expect, it, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { EnvConfig } from "../config/env.js";
import { PolicyDatabase } from "./policy-db.js";
import { resolveAccessForCandidates } from "./resolver.js";

function baseCfg(over: Partial<EnvConfig> = {}): EnvConfig {
  return {
    allowedContacts: ["5511999999999"],
    openaiApiKey: undefined,
    openaiBaseUrl: undefined,
    openaiModel: "gpt-4o-mini",
    allowedOpenaiModels: ["gpt-4o-mini"],
    sessionPath: "/tmp",
    dataDir: "/tmp",
    policySource: "hybrid",
    policyDbPath: "/tmp/x.sqlite",
    policyCacheTtlMs: 60_000,
    adminToken: undefined,
    adminHost: "127.0.0.1",
    adminPort: 3847,
    replyToDenied: false,
    deniedMessage: "negado",
    printQrInTerminal: false,
    baileysLogLevel: "silent",
    llmMaxOutputTokens: 100,
    llmMaxRequestTokens: undefined,
    llmDailyTokenBudgetPerUser: 0,
    llmDailyBudgetTimezone: "UTC",
    llmTimeoutMs: 30_000,
    llmLogJson: false,
    ...over,
  };
}

describe("resolveAccessForCandidates", () => {
  let dbPath: string;
  let db: PolicyDatabase | undefined;

  function requireDb(): PolicyDatabase {
    if (!db) throw new Error("PolicyDatabase não inicializado (native addon falhou?)");
    return db;
  }

  beforeEach(() => {
    dbPath = path.join(os.tmpdir(), `policy_test_${Date.now()}.sqlite`);
    db = new PolicyDatabase(dbPath);
  });

  afterEach(() => {
    db?.close();
    db = undefined;
    try {
      fs.unlinkSync(dbPath);
      fs.unlinkSync(`${dbPath}-wal`);
      fs.unlinkSync(`${dbPath}-shm`);
    } catch {
      /* ignore */
    }
  });

  it("hybrid: número só no env, sem linha no DB → allowed env com todas as features", () => {
    const cfg = baseCfg({ allowedContacts: ["5511888888888"] });
    const r = resolveAccessForCandidates(["5511888888888"], cfg, requireDb(), "hybrid");
    expect(r.kind).toBe("allowed");
    if (r.kind === "allowed") {
      expect(r.identitySource).toBe("env");
      expect(r.features.size).toBe(3);
    }
  });

  it("hybrid: identidade no DB desativada e mesmo número no env → tenta outro candidato ou denied disabled", () => {
    const cfg = baseCfg({
      allowedContacts: ["5511999999999", "5511888888888"],
    });
    const pdb = requireDb();
    pdb.upsertIdentity("5511999999999", { enabled: false, defaultModelIfNew: "gpt-4o-mini" });
    pdb.replaceFeatureGrants("5511999999999", {
      "finance.local_commands": true,
      "finance.csv_import": true,
      "finance.llm_chat": true,
    });
    const r = resolveAccessForCandidates(
      ["5511999999999", "5511888888888"],
      cfg,
      pdb,
      "hybrid"
    );
    expect(r.kind).toBe("allowed");
    if (r.kind === "allowed") {
      expect(r.contactId).toBe("5511888888888");
      expect(r.identitySource).toBe("env");
    }
  });

  it("db_only: exige linha no DB", () => {
    const cfg = baseCfg({ allowedContacts: ["5511999999999"] });
    const pdb = requireDb();
    const r = resolveAccessForCandidates(["5511999999999"], cfg, pdb, "db_only");
    expect(r.kind).toBe("denied");
    if (r.kind === "denied") expect(r.reason).toBe("not_registered");

    pdb.upsertIdentity("5511999999999", { enabled: true, defaultModelIfNew: "gpt-4o-mini" });
    pdb.replaceFeatureGrants("5511999999999", {
      "finance.local_commands": true,
      "finance.csv_import": false,
      "finance.llm_chat": false,
    });
    const r2 = resolveAccessForCandidates(["5511999999999"], cfg, pdb, "db_only");
    expect(r2.kind).toBe("allowed");
    if (r2.kind === "allowed") {
      expect(r2.features.has("finance.local_commands")).toBe(true);
      expect(r2.features.has("finance.llm_chat")).toBe(false);
    }
  });

  it("db: sem grants habilitados → blocked_no_features", () => {
    const cfg = baseCfg();
    const pdb = requireDb();
    pdb.upsertIdentity("5511777777777", { enabled: true, defaultModelIfNew: "gpt-4o-mini" });
    pdb.replaceFeatureGrants("5511777777777", {
      "finance.local_commands": false,
      "finance.csv_import": false,
      "finance.llm_chat": false,
    });
    const r = resolveAccessForCandidates(["5511777777777"], cfg, pdb, "hybrid");
    expect(r.kind).toBe("blocked_no_features");
  });
});
