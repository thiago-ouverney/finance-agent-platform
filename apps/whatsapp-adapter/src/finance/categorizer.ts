/**
 * Categorização determinística inspirada no fluxo de análise de cartão/fatura.
 */

export type ExpenseCategory =
  | "Mercado"
  | "Assinatura"
  | "Restaurante"
  | "Lanchonete"
  | "Farmácia"
  | "Uber"
  | "Don"
  | "Lazer"
  | "Outros"
  | "Financeiro"
  | "Transporte"
  | "Pagamento fatura";

export type LedgerKind = "consumo" | "financeiro_nao_consumo";

export type CategorizedRow = {
  descricao: string;
  valor: number;
  categoria: ExpenseCategory;
  kind: LedgerKind;
  motivo?: string;
};

const PAYMENT_FATURA_PATTERNS: RegExp[] = [
  /pagamento\s+(de\s+)?fatura/i,
  /pagamento\s+cart[aã]o/i,
  /pix\s+.*fatura/i,
  /^pagamento\s+banco/i,
];

const ENCARGO_PATTERNS: RegExp[] = [/encargo/i, /\biof\b/i, /juros/i, /multa/i, /anuidade/i];

const ASSINATURA_PATTERNS: RegExp[] = [
  /netflix|spotify|amazon\s*prime|prime\s*video|disney|hbo|apple\s*one|icloud|google\s*one/i,
  /openai|chatgpt|cursor|github|notion|slack/i,
  /wellhub|gympass|smart\s*fit|crunchyroll/i,
  /assinatura|mensalidade\s+app/i,
];

const MERCADO_PATTERNS: RegExp[] = [
  /carrefour|pao\s*de\s*acucar|extra|assai|atacadao|atacad[aã]o|wal\s*mart|walmart/i,
  /mercado|supermercado|hortifruti/i,
  /\bifood\s*market\b/i,
];

const RESTAURANTE_PATTERNS: RegExp[] = [/restaurante|ifood|uber\s*eats|rappi|zomato/i];

const LANCHONETE_PATTERNS: RegExp[] = [/lanchonete|padaria|padoca|bakery/i];

const FARMACIA_PATTERNS: RegExp[] = [/drogaria|farm[aá]cia|drogasil|pacheco|panvel/i];

const UBER_PATTERNS: RegExp[] = [/\buber\b(?!\.?\s*eats)?/i, /99\s*taxi|99pop/i];

const DON_PATTERNS: RegExp[] = [/petz|cobasi|dog\s*chow|ra[cç][aã]o|veterin[aá]rio|banho\s*e\s*tosa/i];

const LAZER_PATTERNS: RegExp[] = [/cinema|ingresso|show|viagem|hotel|booking|airbnb/i];

const TRANSPORTE_GERAL_PATTERNS: RegExp[] = [/\bmetro\b|estacionamento|ped[aá]gio|passagem/i];

export function parseBrazilianMoney(raw: string): number | undefined {
  const s = raw.trim().replace(/\s/g, "");
  if (!s) return undefined;
  const normalized = s.replace(/\./g, "").replace(",", ".");
  const n = Number.parseFloat(normalized);
  return Number.isFinite(n) ? n : undefined;
}

export function categorizeDescription(desc: string): {
  categoria: ExpenseCategory;
  kind: LedgerKind;
  motivo?: string;
} {
  const d = desc.trim();

  for (const re of PAYMENT_FATURA_PATTERNS) {
    if (re.test(d)) {
      return {
        categoria: "Pagamento fatura",
        kind: "financeiro_nao_consumo",
        motivo: "Pagamento de fatura",
      };
    }
  }
  for (const re of ENCARGO_PATTERNS) {
    if (re.test(d)) {
      return {
        categoria: "Financeiro",
        kind: "financeiro_nao_consumo",
        motivo: "Encargo/IOF/juros",
      };
    }
  }

  for (const re of MERCADO_PATTERNS) {
    if (re.test(d)) return { categoria: "Mercado", kind: "consumo" };
  }
  for (const re of ASSINATURA_PATTERNS) {
    if (re.test(d)) return { categoria: "Assinatura", kind: "consumo" };
  }
  for (const re of RESTAURANTE_PATTERNS) {
    if (re.test(d)) return { categoria: "Restaurante", kind: "consumo" };
  }
  for (const re of LANCHONETE_PATTERNS) {
    if (re.test(d)) return { categoria: "Lanchonete", kind: "consumo" };
  }
  for (const re of FARMACIA_PATTERNS) {
    if (re.test(d)) return { categoria: "Farmácia", kind: "consumo" };
  }
  for (const re of UBER_PATTERNS) {
    if (re.test(d)) return { categoria: "Uber", kind: "consumo" };
  }
  for (const re of DON_PATTERNS) {
    if (re.test(d)) return { categoria: "Don", kind: "consumo" };
  }
  for (const re of LAZER_PATTERNS) {
    if (re.test(d)) return { categoria: "Lazer", kind: "consumo" };
  }
  for (const re of TRANSPORTE_GERAL_PATTERNS) {
    if (re.test(d)) return { categoria: "Transporte", kind: "consumo" };
  }

  return { categoria: "Outros", kind: "consumo" };
}

export function rowsFromLooseCsv(text: string): Array<Record<string, string>> {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) return [];

  const delimiter =
    lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
  const header = lines[0].split(delimiter).map((h) => h.trim().toLowerCase());

  const valorIdx = header.findIndex((h) =>
    ["valor", "amount", "value"].includes(h)
  );
  const descIdx = header.findIndex((h) =>
    ["descricao", "descrição", "memo", "historico", "histórico", "establishment"].includes(
      h
    )
  );

  const rows: Array<Record<string, string>> = [];

  if (valorIdx >= 0 && descIdx >= 0) {
    for (let i = 1; i < lines.length; i++) {
      const cols = splitCsvLine(lines[i], delimiter);
      const row: Record<string, string> = {};
      header.forEach((key, j) => {
        row[key] = cols[j]?.trim() ?? "";
      });
      rows.push(row);
    }
    return rows;
  }

  for (const line of lines) {
    const cols = splitCsvLine(line, delimiter);
    if (cols.length < 2) continue;
    const valorStr = cols[cols.length - 1];
    const desc = cols.slice(0, -1).join(" ").trim();
    rows.push({ descricao: desc, valor: valorStr });
  }
  return rows;
}

function splitCsvLine(line: string, delimiter: string): string[] {
  const parts: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (!inQuotes && c === delimiter) {
      parts.push(cur);
      cur = "";
      continue;
    }
    cur += c;
  }
  parts.push(cur);
  return parts;
}

export function categorizeRows(rows: Array<Record<string, string>>): CategorizedRow[] {
  const out: CategorizedRow[] = [];
  for (const r of rows) {
    const descRaw =
      r.descricao ??
      r.descrição ??
      r.memo ??
      r.establishment ??
      r.estabelecimento ??
      "";
    const keys = Object.keys(r);
    let valorRaw =
      r.valor ?? r.amount ?? r.value ?? "";
    if (!valorRaw) {
      const vk = keys.find((k) => k.toLowerCase().includes("valor"));
      if (vk) valorRaw = r[vk] ?? "";
    }

    const desc = String(descRaw).trim();
    let valor = parseBrazilianMoney(String(valorRaw));
    if (valor === undefined) {
      const nums = String(valorRaw).match(/-?[\d.,]+/);
      if (nums) valor = parseBrazilianMoney(nums[0]);
    }
    if (valor === undefined || desc.length === 0) continue;

    const abs = Math.abs(valor);
    const cat = categorizeDescription(desc);
    out.push({
      descricao: desc,
      valor: abs,
      categoria: cat.categoria,
      kind: cat.kind,
      motivo: cat.motivo,
    });
  }
  return out;
}

export function sumByCategory(rows: CategorizedRow[]): Record<string, number> {
  const acc: Record<string, number> = {};
  for (const r of rows) {
    if (r.kind !== "consumo") continue;
    acc[r.categoria] = (acc[r.categoria] ?? 0) + r.valor;
  }
  return acc;
}

export function totalConsumo(rows: CategorizedRow[]): number {
  return rows.filter((r) => r.kind === "consumo").reduce((s, r) => s + r.valor, 0);
}

export function totalNaoConsumo(rows: CategorizedRow[]): number {
  return rows
    .filter((r) => r.kind === "financeiro_nao_consumo")
    .reduce((s, r) => s + r.valor, 0);
}

export function invoiceMonthFromFilename(name: string): string | undefined {
  const m = name.match(/fatura-(\d{8})\.csv$/i);
  if (!m?.[1]) return undefined;
  const y = m[1].slice(0, 4);
  const mo = m[1].slice(4, 6);
  const monthNum = Number.parseInt(mo, 10);
  if (monthNum < 1 || monthNum > 12) return undefined;
  return `${y}-${mo}`;
}
