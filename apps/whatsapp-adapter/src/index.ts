import "./bootstrap-webcrypto.js";
import { loadEnv } from "./config/env.js";
import { startAdminServer } from "./admin/server.js";
import { PolicyDatabase } from "./policy/policy-db.js";
import { PolicyService } from "./policy/resolver.js";
import { startWhatsAppBot } from "./whatsapp/client.js";

async function main(): Promise<void> {
  try {
    await import("dotenv/config");
  } catch {
    /* dotenv opcional (ex.: só Bun injeta .env) */
  }

  const cfg = loadEnv();

  const policyDb = new PolicyDatabase(cfg.policyDbPath);
  policyDb.seedFromEnvIfEmpty(cfg);
  const policySvc = new PolicyService(
    policyDb,
    cfg,
    cfg.policySource,
    cfg.policyCacheTtlMs
  );

  if (cfg.policySource === "db_only" && policyDb.countIdentities() === 0) {
    console.warn(
      "[policy] POLICY_SOURCE=db_only e nenhuma identidade no banco — ninguém poderá usar o bot até cadastrar em policy.sqlite (via admin ou seed)."
    );
  } else if (
    cfg.policySource === "hybrid" &&
    cfg.allowedContacts.length === 0 &&
    policyDb.countIdentities() === 0
  ) {
    console.warn(
      "[policy] HYBRID com ALLOWED_CONTACTS vazio e banco sem identidades — nenhum contato poderá usar o bot."
    );
  }

  if (cfg.adminToken) {
    startAdminServer(cfg, policySvc);
  } else {
    console.info(
      "[admin] Servidor admin desligado (defina ADMIN_TOKEN para ligar em " +
        `${cfg.adminHost}:${cfg.adminPort}).`
    );
  }

  console.log("Iniciando agente financeiro (Baileys)...");
  await startWhatsAppBot(cfg, policySvc);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
