import type {
  CategorizedRow,
  ExpenseCategory,
  LedgerKind,
} from "./categorizer.js";

export type CustomRule = {
  pattern: string;
  categoria: ExpenseCategory;
  kind: LedgerKind;
};

const EXPENSE_CATEGORIES: ExpenseCategory[] = [
  "Mercado",
  "Assinatura",
  "Restaurante",
  "Lanchonete",
  "Farmácia",
  "Uber",
  "Don",
  "Lazer",
  "Outros",
  "Financeiro",
  "Transporte",
  "Pagamento fatura",
];

function parseCsvLines(text: string): Array<Record<string, string>> {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) return [];
  const delimiter =
    lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
  const headers = lines[0].split(delimiter).map((h) => h.trim().toLowerCase());
  return lines.slice(1).map((line) => {
    const cols = line.split(delimiter).map((c) => c.trim());
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = cols[i] ?? "";
    });
    return row;
  });
}

function normalizeCategory(raw: string): ExpenseCategory {
  const hit = EXPENSE_CATEGORIES.find(
    (cat) => cat.toLowerCase() === raw.toLowerCase()
  );
  return hit ?? "Outros";
}

function normalizeKind(raw: string): LedgerKind {
  return raw === "financeiro_nao_consumo" ? "financeiro_nao_consumo" : "consumo";
}

function isFilled(value: string | undefined): boolean {
  if (!value) return false;
  const v = value.trim();
  return v.length > 0 && !/^preencher$/i.test(v);
}

export function parseCustomRulesCsv(rulesCustomCsv: string): CustomRule[] {
  return parseCsvLines(rulesCustomCsv)
    .filter((r) => isFilled(r.pattern) && isFilled(r.categoria))
    .map((r) => ({
      pattern: r.pattern,
      categoria: normalizeCategory(r.categoria),
      kind: normalizeKind(r.kind),
    }));
}

export function applyCustomRules(
  rows: CategorizedRow[],
  customRules: CustomRule[]
): CategorizedRow[] {
  if (customRules.length === 0) return rows;
  return rows.map((row) => {
    const text = row.descricao.toLowerCase();
    const match = customRules.find((rule) =>
      text.includes(rule.pattern.toLowerCase())
    );
    if (!match) return row;
    return {
      ...row,
      categoria: match.categoria,
      kind: match.kind,
    };
  });
}
