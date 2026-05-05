import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { categorizeRows, invoiceMonthFromFilename } from "../finance/categorizer.js";
import {
  applyCustomRules,
  parseCustomRulesCsv,
} from "../finance/custom-rules.js";

export type StoredConversation = {
  chatId: string;
  messages: Array<{ role: "user" | "assistant"; text: string; at: string }>;
};

export type StoredStatementsFile = {
  invoices: Array<{
    label: string;
    rows: Array<Record<string, string>>;
    uploadedAt: string;
  }>;
};

export type ContactContextRead = {
  profileMd: string;
  rulesMd: string;
  rulesCustomCsv: string;
  communicationMd: string;
  billingCycleCsv: string;
  goalsCsv: string;
};

export type SavedCsvArtifact = {
  timestamp: string;
  rawPath: string;
  parsedJsonPath: string;
  parsedCsvPath: string;
  summaryMdPath: string;
  rawSha256: string;
};

export type ConsolidatedExpenseRow = {
  expense_id: string;
  source_file: string;
  source_row_index: number;
  uploaded_at: string;
  transaction_date: string;
  invoice_month: string;
  descricao: string;
  valor: string;
  categoria: string;
  kind: string;
};

const PROFILE_TEMPLATE = `# Perfil da pessoa

- nome: PREENCHER
- ocupacao: PREENCHER
- renda_mensal: PREENCHER
- composicao_familiar: PREENCHER
- prioridades_financeiras: PREENCHER
- restricoes_relevantes: PREENCHER
- tolerancia_risco: PREENCHER
- habitos_sazonais: PREENCHER
- observacoes: PREENCHER
`;

const RULES_TEMPLATE = `# Regras de categorizacao (IA + contexto do contato)

## Diretrizes gerais
- Tratar pagamento de fatura, juros, IOF e encargos como nao consumo.
- Tratar transferencias internas (PIX entre contas proprias) como nao consumo quando identificado.
- Explicitar cenarios quando a classificacao for ambigua.

## Regras especificas do contato
- PREENCHER regra 1
- PREENCHER regra 2
- PREENCHER regra 3
`;

const COMMUNICATION_TEMPLATE = `# Forma de comunicacao

- tom: PREENCHER
- nivel_detalhe: PREENCHER
- formato_pos_print: PREENCHER
- tamanho_resposta: PREENCHER
- termos_a_evitar: PREENCHER
- como_cobrar_proximo_passo: PREENCHER
`;

const BILLING_CYCLE_TEMPLATE = `campo,valor,observacao
fatura_vencimento_dia,PREENCHER,dia de vencimento da fatura
fatura_fechamento_dia,PREENCHER,dia de fechamento da fatura
controle_mensal_inicio_dia,PREENCHER,dia inicial do controle
controle_mensal_fim_dia,PREENCHER,dia final do controle
`;

const GOALS_TEMPLATE = `tipo,categoria,meta_mensal,data_alvo,tolerancia_percentual,observacao
investimento,geral,PREENCHER,PREENCHER,10,meta de investimento mensal
reserva_emergencia,geral,PREENCHER,PREENCHER,10,aporte para reserva
categoria,mercado,PREENCHER,PREENCHER,10,teto de gasto categoria mercado
categoria,lazer,PREENCHER,PREENCHER,10,teto de gasto categoria lazer
`;

const RULES_CUSTOM_TEMPLATE = `pattern,categoria,kind,prioridade,observacao
PREENCHER,PREENCHER,consumo,100,override custom do contato
`;

export class LocalStore {
  constructor(private readonly dataDir: string) {}

  private safeId(contactId: string): string {
    return contactId.replace(/[^a-zA-Z0-9_-]/g, "_");
  }

  private contactRoot(contactId: string): string {
    return path.join(this.dataDir, "contacts", this.safeId(contactId));
  }

  private conversationDir(contactId: string): string {
    return path.join(this.contactRoot(contactId), "conversation");
  }

  private invoicesDir(contactId: string): string {
    return path.join(this.contactRoot(contactId), "invoices");
  }

  private artifactsDir(contactId: string): string {
    return path.join(this.contactRoot(contactId), "artifacts");
  }

  private contextDir(contactId: string): string {
    return path.join(this.contactRoot(contactId), "context");
  }

  private convPath(contactId: string): string {
    return path.join(this.conversationDir(contactId), "history.json");
  }

  private statementsPath(contactId: string): string {
    return path.join(this.invoicesDir(contactId), "statements.json");
  }

  private artifactsIndexPath(contactId: string): string {
    return path.join(this.artifactsDir(contactId), "index.csv");
  }

  private consolidatedExpensesPath(contactId: string): string {
    return path.join(this.invoicesDir(contactId), "consolidated_expenses.csv");
  }

  async ensureDir(): Promise<void> {
    await fs.mkdir(this.dataDir, { recursive: true });
  }

  private async writeIfMissing(filePath: string, content: string): Promise<void> {
    try {
      await fs.access(filePath);
    } catch {
      await fs.writeFile(filePath, content, "utf8");
    }
  }

  async ensureContactScaffold(contactId: string): Promise<void> {
    await this.ensureDir();
    await fs.mkdir(this.conversationDir(contactId), { recursive: true });
    await fs.mkdir(this.contextDir(contactId), { recursive: true });
    await fs.mkdir(path.join(this.invoicesDir(contactId), "raw"), {
      recursive: true,
    });
    await fs.mkdir(path.join(this.invoicesDir(contactId), "parsed"), {
      recursive: true,
    });
    await fs.mkdir(this.artifactsDir(contactId), { recursive: true });

    await this.writeIfMissing(
      path.join(this.contextDir(contactId), "profile.md"),
      PROFILE_TEMPLATE
    );
    await this.writeIfMissing(
      path.join(this.contextDir(contactId), "rules.md"),
      RULES_TEMPLATE
    );
    await this.writeIfMissing(
      path.join(this.contextDir(contactId), "rules_custom.csv"),
      RULES_CUSTOM_TEMPLATE
    );
    await this.writeIfMissing(
      path.join(this.contextDir(contactId), "communication.md"),
      COMMUNICATION_TEMPLATE
    );
    await this.writeIfMissing(
      path.join(this.contextDir(contactId), "billing_cycle.csv"),
      BILLING_CYCLE_TEMPLATE
    );
    await this.writeIfMissing(
      path.join(this.contextDir(contactId), "goals.csv"),
      GOALS_TEMPLATE
    );
    await this.writeIfMissing(
      this.artifactsIndexPath(contactId),
      "timestamp,file_name,source,bytes,sha256,raw_path,parsed_json_path,parsed_csv_path,summary_md_path,status\n"
    );
  }

  async loadContactContext(contactId: string): Promise<ContactContextRead> {
    await this.ensureContactScaffold(contactId);
    const cdir = this.contextDir(contactId);
    const read = async (rel: string) => fs.readFile(path.join(cdir, rel), "utf8");
    return {
      profileMd: await read("profile.md"),
      rulesMd: await read("rules.md"),
      rulesCustomCsv: await read("rules_custom.csv"),
      communicationMd: await read("communication.md"),
      billingCycleCsv: await read("billing_cycle.csv"),
      goalsCsv: await read("goals.csv"),
    };
  }

  getContactContextPath(contactId: string): string {
    return this.contextDir(contactId);
  }

  async loadConversation(contactId: string): Promise<StoredConversation> {
    await this.ensureContactScaffold(contactId);
    const p = this.convPath(contactId);
    try {
      const raw = await fs.readFile(p, "utf8");
      return JSON.parse(raw) as StoredConversation;
    } catch {
      return { chatId: contactId, messages: [] };
    }
  }

  async appendMessage(
    contactId: string,
    msg: { role: "user" | "assistant"; text: string }
  ): Promise<void> {
    await this.ensureContactScaffold(contactId);
    const cur = await this.loadConversation(contactId);
    cur.messages.push({
      ...msg,
      at: new Date().toISOString(),
    });
    const trimmed = cur.messages.slice(-40);
    const next: StoredConversation = { chatId: contactId, messages: trimmed };
    await fs.writeFile(this.convPath(contactId), JSON.stringify(next, null, 2));
  }

  async loadStatements(contactId: string): Promise<StoredStatementsFile> {
    await this.ensureContactScaffold(contactId);
    const p = this.statementsPath(contactId);
    try {
      const raw = await fs.readFile(p, "utf8");
      return JSON.parse(raw) as StoredStatementsFile;
    } catch {
      return { invoices: [] };
    }
  }

  async appendInvoiceRows(
    contactId: string,
    label: string,
    rows: Array<Record<string, string>>
  ): Promise<void> {
    await this.ensureContactScaffold(contactId);
    const cur = await this.loadStatements(contactId);
    cur.invoices.push({
      label,
      rows,
      uploadedAt: new Date().toISOString(),
    });
    await fs.writeFile(this.statementsPath(contactId), JSON.stringify(cur, null, 2));
  }

  private csvEscape(value: string): string {
    if (value.includes(",") || value.includes('"') || value.includes("\n")) {
      return `"${value.replace(/"/g, '""')}"`;
    }
    return value;
  }

  private rowsToCsv(rows: Array<Record<string, string>>): string {
    if (rows.length === 0) return "";
    const headers = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    const headerLine = headers.join(",");
    const body = rows
      .map((row) => headers.map((h) => this.csvEscape(row[h] ?? "")).join(","))
      .join("\n");
    return `${headerLine}\n${body}\n`;
  }

  private buildExpenseId(input: {
    sourceFile: string;
    sourceRowIndex: number;
    uploadedAt: string;
    transactionDate: string;
    descricao: string;
    valor: string;
  }): string {
    return crypto
      .createHash("sha256")
      .update(
        [
          input.sourceFile,
          String(input.sourceRowIndex),
          input.uploadedAt,
          input.transactionDate,
          input.descricao,
          input.valor,
        ].join("|"),
        "utf8"
      )
      .digest("hex")
      .slice(0, 16);
  }

  private normalizeDateString(raw: string | undefined): string {
    if (!raw) return "";
    const v = raw.trim();
    if (!v) return "";
    const iso = v.match(/^(\d{4})[-/](\d{2})[-/](\d{2})$/);
    if (iso) return `${iso[1]}-${iso[2]}-${iso[3]}`;
    const br = v.match(/^(\d{2})[-/](\d{2})[-/](\d{4})$/);
    if (br) return `${br[3]}-${br[2]}-${br[1]}`;
    const compact = v.match(/^(\d{4})(\d{2})(\d{2})$/);
    if (compact) return `${compact[1]}-${compact[2]}-${compact[3]}`;
    return "";
  }

  private extractTransactionDate(row: Record<string, string>): string {
    const keys = Object.keys(row);
    const candidates = [
      "data",
      "date",
      "dt",
      "data_compra",
      "data compra",
      "data_movimento",
      "data movimento",
      "transaction_date",
      "posted_date",
    ];
    for (const key of keys) {
      const norm = key.toLowerCase().trim();
      if (candidates.includes(norm)) {
        const parsed = this.normalizeDateString(row[key]);
        if (parsed) return parsed;
      }
    }
    return "";
  }

  private consolidatedRowsToCsv(rows: ConsolidatedExpenseRow[]): string {
    const headers: Array<keyof ConsolidatedExpenseRow> = [
      "expense_id",
      "source_file",
      "source_row_index",
      "uploaded_at",
      "transaction_date",
      "invoice_month",
      "descricao",
      "valor",
      "categoria",
      "kind",
    ];
    const headerLine = headers.join(",");
    const body = rows
      .map((row) =>
        headers.map((h) => this.csvEscape(String(row[h] ?? ""))).join(",")
      )
      .join("\n");
    return `${headerLine}\n${body}\n`;
  }

  async rebuildConsolidatedExpenses(
    contactId: string,
    rulesCustomCsv: string
  ): Promise<{ path: string; rows: number }> {
    await this.ensureContactScaffold(contactId);
    const stmts = await this.loadStatements(contactId);
    const customRules = parseCustomRulesCsv(rulesCustomCsv);
    const out: ConsolidatedExpenseRow[] = [];

    for (const invoice of stmts.invoices) {
      const invoiceMonth =
        invoiceMonthFromFilename(invoice.label) ?? invoice.uploadedAt.slice(0, 7);
      for (let idx = 0; idx < invoice.rows.length; idx++) {
        const rawRow = invoice.rows[idx] ?? {};
        const categorized = applyCustomRules(categorizeRows([rawRow]), customRules)[0];
        if (!categorized) continue;
        const transactionDate = this.extractTransactionDate(rawRow);
        const valor = categorized.valor.toFixed(2);
        out.push({
          expense_id: this.buildExpenseId({
            sourceFile: invoice.label,
            sourceRowIndex: idx,
            uploadedAt: invoice.uploadedAt,
            transactionDate,
            descricao: categorized.descricao,
            valor,
          }),
          source_file: invoice.label,
          source_row_index: idx,
          uploaded_at: invoice.uploadedAt,
          transaction_date: transactionDate,
          invoice_month: invoiceMonth,
          descricao: categorized.descricao,
          valor,
          categoria: categorized.categoria,
          kind: categorized.kind,
        });
      }
    }

    out.sort((a, b) => {
      const ka = `${a.transaction_date || "9999-99-99"}|${a.uploaded_at}`;
      const kb = `${b.transaction_date || "9999-99-99"}|${b.uploaded_at}`;
      return ka.localeCompare(kb);
    });

    const p = this.consolidatedExpensesPath(contactId);
    await fs.writeFile(p, this.consolidatedRowsToCsv(out), "utf8");
    return { path: p, rows: out.length };
  }

  async persistCsvArtifact(
    contactId: string,
    input: {
      fileName: string;
      source: string;
      rawCsv: string;
      parsedRows: Array<Record<string, string>>;
      summaryMarkdown: string;
    }
  ): Promise<SavedCsvArtifact> {
    await this.ensureContactScaffold(contactId);
    const now = new Date();
    const iso = now.toISOString();
    const stamp = iso.replace(/[:.]/g, "-");
    const fileSafe = input.fileName.replace(/[^a-zA-Z0-9._-]/g, "_");
    const hash = crypto
      .createHash("sha256")
      .update(input.rawCsv, "utf8")
      .digest("hex");

    const ym = `${String(now.getUTCFullYear())}-${String(now.getUTCMonth() + 1).padStart(2, "0")}`;
    const rawDir = path.join(this.invoicesDir(contactId), "raw", ym);
    const parsedDir = path.join(this.invoicesDir(contactId), "parsed", ym);
    await fs.mkdir(rawDir, { recursive: true });
    await fs.mkdir(parsedDir, { recursive: true });

    const baseName = `${stamp}__${fileSafe.replace(/\.csv$/i, "")}`;
    const rawPath = path.join(rawDir, `${baseName}.csv`);
    const parsedJsonPath = path.join(parsedDir, `${baseName}.json`);
    const parsedCsvPath = path.join(parsedDir, `${baseName}.csv`);
    const summaryMdPath = path.join(parsedDir, `${baseName}.md`);

    await fs.writeFile(rawPath, input.rawCsv, "utf8");
    await fs.writeFile(parsedJsonPath, JSON.stringify(input.parsedRows, null, 2), "utf8");
    await fs.writeFile(parsedCsvPath, this.rowsToCsv(input.parsedRows), "utf8");
    await fs.writeFile(summaryMdPath, input.summaryMarkdown, "utf8");

    const relRaw = path.relative(this.dataDir, rawPath);
    const relParsedJson = path.relative(this.dataDir, parsedJsonPath);
    const relParsedCsv = path.relative(this.dataDir, parsedCsvPath);
    const relSummaryMd = path.relative(this.dataDir, summaryMdPath);

    const line = [
      iso,
      input.fileName,
      input.source,
      Buffer.byteLength(input.rawCsv, "utf8").toString(),
      hash,
      relRaw,
      relParsedJson,
      relParsedCsv,
      relSummaryMd,
      "ok",
    ]
      .map((v) => this.csvEscape(v))
      .join(",");

    await fs.appendFile(this.artifactsIndexPath(contactId), `${line}\n`, "utf8");

    return {
      timestamp: iso,
      rawPath,
      parsedJsonPath,
      parsedCsvPath,
      summaryMdPath,
      rawSha256: hash,
    };
  }

  /** Remove historico operacional (mantem templates de contexto) */
  async clearAll(contactId: string): Promise<void> {
    await this.ensureContactScaffold(contactId);
    await fs.unlink(this.convPath(contactId)).catch(() => {});
    await fs.unlink(this.statementsPath(contactId)).catch(() => {});
    await fs.unlink(this.artifactsIndexPath(contactId)).catch(() => {});
    await fs.rm(path.join(this.invoicesDir(contactId), "raw"), {
      recursive: true,
      force: true,
    });
    await fs.rm(path.join(this.invoicesDir(contactId), "parsed"), {
      recursive: true,
      force: true,
    });
    await this.writeIfMissing(
      this.artifactsIndexPath(contactId),
      "timestamp,file_name,source,bytes,sha256,raw_path,parsed_json_path,parsed_csv_path,summary_md_path,status\n"
    );
  }
}
