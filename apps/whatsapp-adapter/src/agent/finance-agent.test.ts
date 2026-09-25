import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { handleUserMessage } from "../agent/finance-agent.js";
import { LocalStore } from "../storage/store.js";
import type { EnvConfig } from "../config/env.js";
import { testEnvConfig } from "../test/test-env-config.js";

function baseCfg(over: Partial<EnvConfig> = {}): EnvConfig {
  return testEnvConfig(over);
}

async function completeContext(store: LocalStore, contactId: string): Promise<void> {
  await store.ensureContactScaffold(contactId);
  const dir = store.getContactContextPath(contactId);
  await writeFile(
    path.join(dir, "profile.md"),
    "# Perfil da pessoa\n- nome: Maria\n- renda_mensal: 12000\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "communication.md"),
    "# Forma de comunicacao\n- tom: direto\n- formato_pos_print: resumo curto\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "rules.md"),
    "# Regras\najustar pix interno para nao consumo\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "rules_custom.csv"),
    "pattern,categoria,kind,prioridade,observacao\npix ajuste,Financeiro,financeiro_nao_consumo,100,ok\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "billing_cycle.csv"),
    "campo,valor,observacao\nfatura_vencimento_dia,10,x\nfatura_fechamento_dia,3,x\ncontrole_mensal_inicio_dia,1,x\ncontrole_mensal_fim_dia,30,x\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "goals.csv"),
    "tipo,categoria,meta_mensal,data_alvo,tolerancia_percentual,observacao\ninvestimento,geral,1500,2026-12-31,10,ok\nreserva_emergencia,geral,900,2026-12-31,10,ok\n",
    "utf8"
  );
}

describe("handleUserMessage modo determinístico", () => {
  let dir: string;

  afterEach(async () => {
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  it("responde com análise quando CSV colado e sem OPENAI_API_KEY", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-agent-"));
    const store = new LocalStore(dir);
    await completeContext(store, "user1");
    const csv = `descricao,valor
Supermercado X,100,50
PAGAMENTO FATURA,200,00`;

    const reply = await handleUserMessage(
      baseCfg(),
      store,
      "user1",
      csv,
      {}
    );
    expect(reply).toContain("Total consumo");
    expect(reply).toContain("Mercado");
    expect(reply).toContain("/help");
  });

  it("/help lista comandos sem LLM", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-agent-"));
    const store = new LocalStore(dir);
    const reply = await handleUserMessage(baseCfg(), store, "u", "/help");
    expect(reply).toContain("/status");
    expect(reply).toContain("/orcamento");
  });

  it("/orcamento com histórico calcula orçamento", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-agent-"));
    const store = new LocalStore(dir);
    await completeContext(store, "u");
    const csv = `descricao,valor
Loja,50,00`;
    await handleUserMessage(baseCfg(), store, "u", csv, {});
    const reply = await handleUserMessage(
      baseCfg(),
      store,
      "u",
      "/orcamento renda=12000 fixos=4700"
    );
    expect(reply).toContain("Orçamento familiar");
    expect(reply).toContain("12.000");
  });

  it("bloqueia analise quando contexto obrigatorio nao foi preenchido", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-agent-"));
    const store = new LocalStore(dir);
    const reply = await handleUserMessage(baseCfg(), store, "faltando", "analise meus gastos");
    expect(reply).toContain("preciso que voce complete o contexto");
    expect(reply).toContain("profile.md");
  });

  it("analisa fatura com decimal em ponto sem inflar totais", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-agent-"));
    const store = new LocalStore(dir);
    await completeContext(store, "fatura-us");
    const fixtureCsv = readFileSync(
      path.join(
        path.dirname(fileURLToPath(import.meta.url)),
        "../test/fixtures/invoice-en-us.csv"
      ),
      "utf8"
    );

    const reply = await handleUserMessage(
      baseCfg(),
      store,
      "fatura-us",
      fixtureCsv,
      { filename: "invoice-en-us.csv" }
    );

    expect(reply).toContain("2.578,79");
    expect(reply).not.toContain("187.310");
    expect(reply).toContain("decimal com ponto (en-US)");
    expect(reply).toContain("separador CSV: vírgula");
  });

  it("gera artefatos csv/json/md ao receber CSV com nome de arquivo", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-agent-"));
    const store = new LocalStore(dir);
    await completeContext(store, "artefato");
    const csv = `descricao,valor
Padaria,20,00`;
    await handleUserMessage(baseCfg(), store, "artefato", csv, {
      filename: "fatura-20260110.csv",
      sourceJid: "5511999999999@s.whatsapp.net",
      rawCsvFromFile: csv,
    });
    const indexPath = path.join(
      dir,
      "contacts",
      "artefato",
      "artifacts",
      "index.csv"
    );
    const indexRaw = await readFile(indexPath, "utf8");
    expect(indexRaw).toContain("fatura-20260110.csv");
    expect(indexRaw).toContain("contacts/artefato/invoices/raw/");
    const consolidatedPath = path.join(
      dir,
      "contacts",
      "artefato",
      "invoices",
      "consolidated_expenses.csv"
    );
    const consolidatedRaw = await readFile(consolidatedPath, "utf8");
    expect(consolidatedRaw).toContain("expense_id,source_file,source_row_index");
    expect(consolidatedRaw).toContain("fatura-20260110.csv");
  });
});
