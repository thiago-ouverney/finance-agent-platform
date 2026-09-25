/**
 * Catálogo canônico de features (contrato estável DB ↔ código).
 * O admin só pode conceder chaves listadas aqui.
 * Extensão futura: mapear cada chave para handlers / sub-agentes.
 */
export const FEATURE_CATALOG = [
  {
    key: "finance.local_commands",
    label: "Comandos locais (/help, /status, …)",
  },
  {
    key: "finance.csv_import",
    label: "Importação e análise de CSV de fatura",
  },
  {
    key: "finance.llm_chat",
    label: "Enriquecimento por modelo de linguagem",
  },
] as const;

export type FeatureKey = (typeof FEATURE_CATALOG)[number]["key"];

const KEY_SET = new Set<string>(FEATURE_CATALOG.map((f) => f.key));

export function isKnownFeatureKey(key: string): key is FeatureKey {
  return KEY_SET.has(key);
}

export function allFeatureKeys(): readonly FeatureKey[] {
  return FEATURE_CATALOG.map((f) => f.key) as unknown as FeatureKey[];
}

export function allFeaturesEnabledSet(): Set<FeatureKey> {
  return new Set<FeatureKey>(allFeatureKeys());
}
