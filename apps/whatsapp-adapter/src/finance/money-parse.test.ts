import { describe, expect, it } from "vitest";
import {
  inferMoneyLocale,
  parseBrazilianMoney,
  parseMoneyAmount,
} from "./money-parse.js";

describe("parseMoneyAmount", () => {
  it("interpreta formato BR clássico", () => {
    expect(parseMoneyAmount("1.234,56", "br")).toBe(1234.56);
    expect(parseBrazilianMoney("1.234,56")).toBe(1234.56);
  });

  it("interpreta decimal com ponto (export cartão)", () => {
    expect(parseMoneyAmount("103.46", "us")).toBe(103.46);
    expect(parseMoneyAmount("18.3", "us")).toBe(18.3);
  });

  it("interpreta negativo e inteiro", () => {
    expect(parseMoneyAmount("-4745.82", "us")).toBe(-4745.82);
    expect(parseMoneyAmount("27", "us")).toBe(27);
  });

  it("interpreta US com milhar", () => {
    expect(parseMoneyAmount("1,234.56", "us")).toBe(1234.56);
  });

  it("infere US em amostras com ponto decimal", () => {
    expect(inferMoneyLocale(["103.46", "18.3", "25.99"])).toBe("us");
  });

  it("infere BR em amostras com vírgula decimal", () => {
    expect(inferMoneyLocale(["123,45", "20,00", "1.234,56"])).toBe("br");
  });

  it("detecta US sem hint em valor típico de fatura", () => {
    expect(parseMoneyAmount("103.46")).toBe(103.46);
  });
});
