import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  categorizeDescription,
  categorizeRows,
  invoiceMonthFromFilename,
  parseBrazilianMoney,
  rowsFromLooseCsv,
  sumByCategory,
  totalConsumo,
  totalNaoConsumo,
} from "../finance/categorizer.js";

const fixtureCsv = readFileSync(
  path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "../test/fixtures/invoice-en-us.csv"
  ),
  "utf8"
);

describe("parseBrazilianMoney", () => {
  it("interpreta formato BR", () => {
    expect(parseBrazilianMoney("1.234,56")).toBe(1234.56);
  });
});

describe("categorizeDescription", () => {
  it("marca pagamento de fatura como não consumo", () => {
    const r = categorizeDescription("PAGAMENTO FATURA CARTAO");
    expect(r.kind).toBe("financeiro_nao_consumo");
    expect(r.categoria).toBe("Pagamento fatura");
  });

  it("categoriza mercado", () => {
    const r = categorizeDescription("SUPERMERCADO EXTRA");
    expect(r.categoria).toBe("Mercado");
    expect(r.kind).toBe("consumo");
  });
});

describe("rowsFromLooseCsv + categorizeRows", () => {
  it("processa CSV com header", () => {
    const csv = `descricao,valor
Supermercado X,"123,45"
PAGAMENTO FATURA,"500,00"`;
    const rows = rowsFromLooseCsv(csv);
    const cat = categorizeRows(rows);
    expect(cat.length).toBe(2);
    expect(totalConsumo(cat)).toBeCloseTo(123.45, 2);
  });
});

describe("sumByCategory", () => {
  it("soma só consumo", () => {
    const csv = `descricao,valor
NETFLIX,50
PAGAMENTO FATURA,100`;
    const cat = categorizeRows(rowsFromLooseCsv(csv));
    const sums = sumByCategory(cat);
    expect(sums["Assinatura"]).toBe(50);
    expect(sums["Pagamento fatura"]).toBeUndefined();
  });
});

describe("invoiceMonthFromFilename", () => {
  it("extrai AAAA-MM", () => {
    expect(invoiceMonthFromFilename("fatura-20260415.csv")).toBe("2026-04");
  });

  it("retorna undefined para nome inválido", () => {
    expect(invoiceMonthFromFilename("dump.csv")).toBeUndefined();
  });
});

describe("fixture sintética de fatura (decimal com ponto)", () => {
  it("totaliza consumo e pagamento corretamente", () => {
    const cat = categorizeRows(rowsFromLooseCsv(fixtureCsv));
    expect(totalConsumo(cat)).toBeCloseTo(2578.79, 2);
    expect(totalNaoConsumo(cat)).toBeCloseTo(4745.82, 2);
  });

  it("usa lançamento como descrição sem prefixar data", () => {
    const cat = categorizeRows(rowsFromLooseCsv(fixtureCsv));
    const superm = cat.find((r) => r.descricao.includes("SUPERM PRINCESA"));
    expect(superm?.descricao).toBe("SUPERM PRINCESA ICARAI");
    expect(superm?.valor).toBeCloseTo(103.46, 2);
  });
});
