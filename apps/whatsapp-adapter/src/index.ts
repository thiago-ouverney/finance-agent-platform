import "./bootstrap-webcrypto.js";
import { loadEnv } from "./config/env.js";
import { ConversationStore } from "./storage/conversation-store.js";
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
      "[access] ALLOWED_CONTACTS está vazio — nenhum contato poderá usar o bot."
    );
  }
  console.log("Iniciando chat de IA no WhatsApp (Baileys)...");
  const store = new ConversationStore(
    cfg.dataDir,
    cfg.conversationHistoryLimit
  );
  await startWhatsAppBot(cfg, store);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
