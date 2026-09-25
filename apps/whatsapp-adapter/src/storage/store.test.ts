import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { LocalStore } from "../storage/store.js";

describe("LocalStore", () => {
  let dir: string;

  afterEach(async () => {
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  it("clearAll remove conversa e extratos", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-clear-"));
    const store = new LocalStore(dir);
    await store.appendMessage("u1", { role: "user", text: "a" });
    await store.appendInvoiceRows("u1", "x.csv", [{ a: "1" }]);
    await store.clearAll("u1");
    const conv = await store.loadConversation("u1");
    const stmts = await store.loadStatements("u1");
    expect(conv.messages.length).toBe(0);
    expect(stmts.invoices.length).toBe(0);
  });

  it("persiste e carrega conversa", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-test-"));
    const store = new LocalStore(dir);
    await store.appendMessage("5511999999999", {
      role: "user",
      text: "oi",
    });
    const conv = await store.loadConversation("5511999999999");
    expect(conv.messages.length).toBe(1);
    expect(conv.messages[0]?.text).toBe("oi");
  });

  it("cria templates de contexto ao inicializar contato", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-template-"));
    const store = new LocalStore(dir);
    await store.ensureContactScaffold("5511999999999");
    const ctx = await store.loadContactContext("5511999999999");
    expect(ctx.profileMd).toContain("Perfil da pessoa");
    expect(ctx.goalsCsv).toContain("meta_mensal");
    expect(ctx.communicationMd).toContain("Forma de comunicacao");
  });

  it("persiste artefato de CSV com indice e arquivos derivados", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-artifact-"));
    const store = new LocalStore(dir);
    const artifact = await store.persistCsvArtifact("5511888888888", {
      fileName: "fatura-20260110.csv",
      source: "test",
      rawCsv: "descricao,valor\nPadaria,20,00\n",
      parsedRows: [{ descricao: "Padaria", valor: "20,00" }],
      summaryMarkdown: "# resumo",
    });
    const idxPath = path.join(
      dir,
      "contacts",
      "5511888888888",
      "artifacts",
      "index.csv"
    );
    const idx = await readFile(idxPath, "utf8");
    expect(idx).toContain("fatura-20260110.csv");
    expect(idx).toContain(artifact.rawSha256);
    await expect(stat(artifact.rawPath)).resolves.toBeTruthy();
    await expect(stat(artifact.parsedJsonPath)).resolves.toBeTruthy();
    await expect(stat(artifact.parsedCsvPath)).resolves.toBeTruthy();
    await expect(stat(artifact.summaryMdPath)).resolves.toBeTruthy();
  });

  it("rebuildConsolidatedExpenses gera csv consolidado com data da transacao", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-consolidated-"));
    const store = new LocalStore(dir);
    await store.appendInvoiceRows("5511777777777", "fatura-20260110.csv", [
      { data: "05/01/2026", descricao: "Padaria", valor: "20,00" },
      { data_compra: "2026-01-08", descricao: "Uber", valor: "31,40" },
    ]);
    const res = await store.rebuildConsolidatedExpenses(
      "5511777777777",
      "pattern,categoria,kind,prioridade,observacao\npadaria,Lanchonete,consumo,100,ok\n"
    );
    const csv = await readFile(res.path, "utf8");
    expect(csv).toContain("expense_id,source_file,source_row_index");
    expect(csv).toContain("2026-01-05");
    expect(csv).toContain("2026-01-08");
    expect(csv).toContain("Lanchonete");
    expect(res.rows).toBe(2);
  });

  it("rebuildConsolidatedExpenses faz fallback sem data e combina multiplas faturas", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-consolidated-"));
    const store = new LocalStore(dir);
    await store.appendInvoiceRows("5511666666666", "fatura-20260210.csv", [
      { descricao: "Mercado XPTO", valor: "150,00" },
    ]);
    await store.appendInvoiceRows("5511666666666", "extrato-avulso.csv", [
      { descricao: "Juros", valor: "10,00" },
    ]);
    const res = await store.rebuildConsolidatedExpenses(
      "5511666666666",
      "pattern,categoria,kind,prioridade,observacao\njuros,Financeiro,financeiro_nao_consumo,100,ok\n"
    );
    const csv = await readFile(res.path, "utf8");
    expect(csv).toContain("fatura-20260210.csv");
    expect(csv).toContain("extrato-avulso.csv");
    expect(csv).toContain(",2026-02,");
    expect(csv).toContain("financeiro_nao_consumo");
    expect(res.rows).toBe(2);
  });
});
