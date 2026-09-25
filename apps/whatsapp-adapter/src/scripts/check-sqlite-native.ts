/**
 * Smoke test do addon nativo better-sqlite3.
 * Usa import dinâmico para que erros de ABI caiam no catch (import estático falharia antes).
 */
try {
  const { default: Database } = await import("better-sqlite3");
  const db = new Database(":memory:");
  const row = db.prepare("SELECT 1 AS x").get() as { x: number };
  if (row?.x !== 1) {
    throw new Error("SELECT 1 retornou valor inesperado");
  }
  db.close();
  console.info("[check:native] better-sqlite3 OK neste Node", process.version);
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  console.error("[check:native] Falha ao carregar better-sqlite3:", msg);
  console.error("");
  console.error(
    "Causa comum: incompatibilidade de ABI (node_modules instalado com outra versão do Node)."
  );
  console.error("Correção na raiz do projeto:");
  console.error("  rm -rf node_modules && npm install");
  console.error("(o postinstall executa `npm rebuild better-sqlite3`.)");
  console.error(
    "Não copie a pasta node_modules entre PCs ou entre versões diferentes do Node."
  );
  process.exit(1);
}
