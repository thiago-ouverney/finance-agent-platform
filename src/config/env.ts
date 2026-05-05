import path from "node:path";

export type EnvConfig = {
  allowedContacts: string[];
  openaiApiKey: string | undefined;
  openaiModel: string;
  sessionPath: string;
  dataDir: string;
  replyToDenied: boolean;
  deniedMessage: string;
  /** Baileys prints QR to terminal by default */
  printQrInTerminal: boolean;
  /** Teto de tokens de saída por chamada ao chat completions */
  llmMaxOutputTokens: number;
  /**
   * Teto por requisição: tokens de entrada estimados + llmMaxOutputTokens.
   * `undefined` desativa truncagem por teto (mantém só o teto de saída).
   */
  llmMaxRequestTokens: number | undefined;
  /** 0 = sem limite diário por contato (`chatId`) */
  llmDailyTokenBudgetPerUser: number;
  /** IANA, ex.: America/Sao_Paulo — usado só para a chave do dia do orçamento */
  llmDailyBudgetTimezone: string;
  llmTimeoutMs: number;
  /** Uma linha JSON por chamada LLM bem-sucedida */
  llmLogJson: boolean;
};

function parseAllowedContacts(raw: string | undefined): string[] {
  if (!raw?.trim()) return [];
  return raw
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(normalizeContactDigits);
}

/** Keeps only digits for comparison with remote Jid user part */
export function normalizeContactDigits(input: string): string {
  const digits = input.replace(/\D/g, "");
  return digits;
}

function parseIntEnv(
  raw: string | undefined,
  fallback: number,
  opts?: { min?: number }
): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number.parseInt(raw.trim(), 10);
  if (!Number.isFinite(n)) return fallback;
  const min = opts?.min ?? 0;
  return Math.max(min, n);
}

function parseOptionalPositiveInt(raw: string | undefined): number | undefined {
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number.parseInt(raw.trim(), 10);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  return n;
}

function parseBoolEnv(raw: string | undefined, fallback: boolean): boolean {
  if (raw === undefined || raw.trim() === "") return fallback;
  const v = raw.trim().toLowerCase();
  return v === "true" || v === "1" || v === "yes";
}

export function loadEnv(): EnvConfig {
  const allowedContacts = parseAllowedContacts(process.env.ALLOWED_CONTACTS);

  const sessionPath = process.env.SESSION_PATH ?? ".baileys_auth";
  const dataDir = process.env.DATA_DIR ?? "data";

  const replyRaw = process.env.REPLY_TO_DENIED?.toLowerCase();
  const replyToDenied = replyRaw === "true" || replyRaw === "1";

  return {
    allowedContacts,
    openaiApiKey: process.env.OPENAI_API_KEY?.trim() || undefined,
    openaiModel: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
    sessionPath: path.resolve(sessionPath),
    dataDir: path.resolve(dataDir),
    replyToDenied,
    deniedMessage:
      process.env.DENIED_MESSAGE ??
      "Este número não está autorizado a usar o agente financeiro.",
    printQrInTerminal:
      process.env.PRINT_QR_TERMINAL !== "false" &&
      process.env.PRINT_QR_TERMINAL !== "0",
    llmMaxOutputTokens: parseIntEnv(process.env.LLM_MAX_OUTPUT_TOKENS, 1800, {
      min: 1,
    }),
    llmMaxRequestTokens: parseOptionalPositiveInt(
      process.env.LLM_MAX_REQUEST_TOKENS
    ),
    llmDailyTokenBudgetPerUser: parseIntEnv(
      process.env.LLM_DAILY_TOKEN_BUDGET_PER_USER,
      0,
      { min: 0 }
    ),
    llmDailyBudgetTimezone: (
      process.env.LLM_DAILY_BUDGET_TIMEZONE ?? "UTC"
    ).trim(),
    llmTimeoutMs: parseIntEnv(process.env.LLM_TIMEOUT_MS, 30_000, {
      min: 1000,
    }),
    llmLogJson: parseBoolEnv(process.env.LLM_LOG_JSON, false),
  };
}
