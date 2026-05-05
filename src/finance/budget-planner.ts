import type { CategorizedRow } from "./categorizer.js";

export type BudgetInputs = {
  rendaMensal: number;
  fixosCasa: number;
  metaInvestimento?: number;
  metaReserva?: number;
};

export type BudgetSummary = {
  renda: number;
  fixosCasa: number;
  mediaCartaoConsumo: number;
  totalPlanejadoBase: number;
  sobraSobreRenda: number;
  metaInvestimento: number;
  metaReserva: number;
  tetoGastoSugerido: number;
};

export function computeBudgetSummary(
  inputs: BudgetInputs,
  cartaoPorMes: number[]
): BudgetSummary {
  const renda = inputs.rendaMensal;
  const fixos = inputs.fixosCasa;
  const metaInvestimento = inputs.metaInvestimento ?? 1200;
  const metaReserva = inputs.metaReserva ?? 800;

  const mediaCartaoConsumo =
    cartaoPorMes.length > 0
      ? cartaoPorMes.reduce((a, b) => a + b, 0) / cartaoPorMes.length
      : 0;

  const totalPlanejadoBase = fixos + mediaCartaoConsumo;
  const sobraSobreRenda = renda - totalPlanejadoBase;

  const tetoGastoSugerido = Math.min(
    renda - metaInvestimento - metaReserva,
    renda * 0.95
  );

  return {
    renda,
    fixosCasa: fixos,
    mediaCartaoConsumo,
    totalPlanejadoBase,
    sobraSobreRenda,
    metaInvestimento,
    metaReserva,
    tetoGastoSugerido,
  };
}

export function topCategories(
  sums: Record<string, number>,
  limit = 8
): Array<{ categoria: string; valor: number; pct: number }> {
  const total = Object.values(sums).reduce((a, b) => a + b, 0);
  const entries = Object.entries(sums).sort((a, b) => b[1] - a[1]);
  return entries.slice(0, limit).map(([categoria, valor]) => ({
    categoria,
    valor,
    pct: total > 0 ? (valor / total) * 100 : 0,
  }));
}

export function suggestCutsFromAverages(
  categoryAvg: Record<string, number>,
  targets: Record<string, number>
): string[] {
  const lines: string[] = [];
  for (const [cat, target] of Object.entries(targets)) {
    const cur = categoryAvg[cat];
    if (cur === undefined || target === undefined) continue;
    if (cur > target) {
      lines.push(
        `${cat}: média ~R$ ${cur.toFixed(0)} → meta R$ ${target.toFixed(0)}`
      );
    }
  }
  return lines;
}

export function aggregateRowsByInvoiceMonth(
  batches: Array<{ month: string; rows: CategorizedRow[] }>
): Array<{ month: string; consumo: number; porCategoria: Record<string, number> }> {
  return batches.map(({ month, rows }) => {
    const consumo = rows
      .filter((r) => r.kind === "consumo")
      .reduce((s, r) => s + r.valor, 0);
    const porCategoria: Record<string, number> = {};
    for (const r of rows) {
      if (r.kind !== "consumo") continue;
      porCategoria[r.categoria] = (porCategoria[r.categoria] ?? 0) + r.valor;
    }
    return { month, consumo, porCategoria };
  });
}
