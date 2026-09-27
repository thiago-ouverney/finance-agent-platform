/**
 * Diagnóstico local sem subir Baileys nem WebSocket.
 * Rode: npm run doctor
 */
import "../bootstrap-webcrypto.js";

function ok(msg: string): void {
  console.info(`[doctor] OK — ${msg}`);
}

function fail(msg: string): never {
  console.error(`[doctor] FALHA — ${msg}`);
  process.exit(1);
}

async function main(): Promise<void> {
  console.info("[doctor] Node", process.version);

  if (typeof globalThis.crypto === "undefined" || !globalThis.crypto.subtle) {
    fail(
      "globalThis.crypto.subtle indisponível (Node 18 precisa do bootstrap; use Node 20+ ou confirme import de bootstrap-webcrypto)."
    );
  }
  ok("Web Crypto (globalThis.crypto.subtle)");

  try {
    await import("dotenv/config");
  } catch {
    /* opcional */
  }
  const { loadEnv } = await import("../config/env.js");
  const cfg = loadEnv();
  ok(
    `loadEnv() — model=${cfg.openaiModel} endpoint=${cfg.openaiBaseUrl ?? "não configurado"} allowedContacts=${cfg.allowedContacts.length}`
  );

  console.info("[doctor] Concluído. Próximo passo: npm run dev (Node, não Bun).");
}

main().catch((e) => {
  console.error("[doctor] erro inesperado:", e);
  process.exit(1);
});
