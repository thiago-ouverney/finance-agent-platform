export type MoneyLocale = "br" | "us";

function stripMoneyDecorations(raw: string): { s: string; negative: boolean } {
  let s = raw.trim().replace(/\s/g, "");
  if (!s) return { s: "", negative: false };

  let negative = false;
  if (s.startsWith("-")) {
    negative = true;
    s = s.slice(1);
  } else if (s.startsWith("(") && s.endsWith(")")) {
    negative = true;
    s = s.slice(1, -1);
  }

  s = s.replace(/^R\$/i, "").trim();
  return { s, negative };
}

function parseWithLocale(s: string, locale: MoneyLocale): number | undefined {
  if (!s) return undefined;

  const lastComma = s.lastIndexOf(",");
  const lastDot = s.lastIndexOf(".");

  let normalized: string;

  if (lastComma >= 0 && lastDot >= 0) {
    if (lastComma > lastDot) {
      normalized = s.replace(/\./g, "").replace(",", ".");
    } else {
      normalized = s.replace(/,/g, "");
    }
  } else if (lastComma >= 0) {
    const after = s.slice(lastComma + 1);
    if (/^\d{1,2}$/.test(after)) {
      normalized = s.replace(/\./g, "").replace(",", ".");
    } else {
      normalized = s.replace(/,/g, "");
    }
  } else if (lastDot >= 0) {
    const parts = s.split(".");
    const lastPart = parts[parts.length - 1] ?? "";

    if (parts.length === 2 && /^\d{1,2}$/.test(lastPart)) {
      normalized = s;
    } else if (
      parts.length > 2 &&
      parts.slice(1).every((p) => p.length === 3)
    ) {
      normalized = s.replace(/\./g, "");
    } else if (parts.length === 2 && lastPart.length === 3) {
      normalized = locale === "br" ? s.replace(/\./g, "") : s;
    } else {
      normalized = locale === "br" ? s.replace(/\./g, "") : s;
    }
  } else {
    normalized = s;
  }

  const n = Number.parseFloat(normalized);
  if (!Number.isFinite(n)) return undefined;
  return n;
}

/** Parseia valor monetário com detecção BR (1.234,56) ou US/export (103.46). */
export function parseMoneyAmount(
  raw: string,
  localeHint?: MoneyLocale
): number | undefined {
  const { s, negative } = stripMoneyDecorations(raw);
  if (!s) return undefined;

  const locale = localeHint ?? inferMoneyLocale([raw]);
  const n = parseWithLocale(s, locale);
  if (n === undefined) return undefined;
  if (negative && n > 0) return -n;
  return n;
}

/** Mantém compatibilidade: formato BR explícito. */
export function parseBrazilianMoney(raw: string): number | undefined {
  return parseMoneyAmount(raw, "br");
}

function scoreSample(raw: string): { us: number; br: number } {
  const { s } = stripMoneyDecorations(raw);
  if (!s) return { us: 0, br: 0 };

  let us = 0;
  let br = 0;

  if (/^\d+\.\d{1,2}$/.test(s)) us += 2;
  if (/,\d{2}$/.test(s)) br += 2;
  if (/\.\d{3},/.test(s) || /,\d{2}$/.test(s)) br += 1;
  if (s.includes(",") && s.includes(".") && s.lastIndexOf(",") > s.lastIndexOf(".")) {
    br += 2;
  }
  if (s.includes(",") && s.includes(".") && s.lastIndexOf(".") > s.lastIndexOf(",")) {
    us += 2;
  }

  return { us, br };
}

/** Infere locale predominante a partir de amostras da coluna valor. */
export function inferMoneyLocale(samples: string[]): MoneyLocale {
  let usScore = 0;
  let brScore = 0;

  for (const raw of samples) {
    const { us, br } = scoreSample(raw);
    usScore += us;
    brScore += br;
  }

  return usScore > brScore ? "us" : "br";
}

export function describeMoneyLocale(locale: MoneyLocale): string {
  return locale === "us"
    ? "decimal com ponto (en-US)"
    : "decimal com vírgula (pt-BR)";
}
