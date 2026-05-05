import { describe, expect, it } from "vitest";
import {
  computeBudgetSummary,
  topCategories,
} from "../finance/budget-planner.js";

describe("computeBudgetSummary", () => {
  it("calcula média do cartão e sobra", () => {
    const s = computeBudgetSummary(
      {
        rendaMensal: 12000,
        fixosCasa: 4701,
      },
      [5970, 5970]
    );
    expect(s.mediaCartaoConsumo).toBe(5970);
    expect(s.totalPlanejadoBase).toBe(4701 + 5970);
    expect(s.sobraSobreRenda).toBe(12000 - (4701 + 5970));
  });
});

describe("topCategories", () => {
  it("ordena por valor", () => {
    const t = topCategories({ Mercado: 100, Uber: 50 }, 5);
    expect(t[0]?.categoria).toBe("Mercado");
  });
});
