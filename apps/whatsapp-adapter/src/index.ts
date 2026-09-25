import { loadEnv } from "./config/env.js";
import { startWhatsAppBot } from "./whatsapp/client.js";

async function main(): Promise<void> {
  try {
    await import("dotenv/config");
  } catch {
    /* dotenv opcional (ex.: só Bun injeta .env) */
  }

  const cfg = loadEnv();

  if (cfg.allowedContacts.length === 0) {
    console.warn(
      "AVISO: ALLOWED_CONTACTS está vazio — nenhum contato poderá usar o bot."
    );
    console.warn(
      "Defina ALLOWED_CONTACTS com números apenas dígitos, ex: 5511999999999"
    );
  }

  console.log("Iniciando agente financeiro (Baileys)...");
  await startWhatsAppBot(cfg);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
