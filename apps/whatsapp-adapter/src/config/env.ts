import path from "node:path";

/** Níveis aceitos por Pino para o logger interno do Baileys. */
export type BaileysLogLevel =
  | "silent"
  | "fatal"
  | "error"
  | "warn"
  | "info"
  | "debug"
  | "trace";

const BAILEYS_LOG_LEVELS: ReadonlySet<string> = new Set([
  "silent",
  "fatal",
  "error",
  "warn",
  "info",
  "debug",
  "trace",
]);

function parseBaileysLogLevel(raw: string | undefined): BaileysLogLevel {
  const value = raw?.trim().toLowerCase() ?? "";
  return BAILEYS_LOG_LEVELS.has(value)
    ? (value as BaileysLogLevel)
    : "silent";
}

export type EnvConfig = {
  allowedContacts: string[];
  openaiApiKey: string | undefined;
  /** Endpoint OpenAI-compatible (Ollama, vLLM, llama.cpp ou OpenAI). */
  openaiBaseUrl: string | undefined;
  openaiModel: string;
  sessionPath: string;
  dataDir: string;
  conversationHistoryLimit: number;
  replyToDenied: boolean;
  deniedMessage: string;
  printQrInTerminal: boolean;
  baileysLogLevel: BaileysLogLevel;
  llmMaxOutputTokens: number;
  llmTimeoutMs: number;
};

function parseAllowedContacts(raw: string | undefined): string[] {
  if (!raw?.trim()) return [];
  return raw
    .split(/[\s,;]+/)
    .map(normalizeContactDigits)
    .filter(Boolean);
}

export function normalizeContactDigits(input: string): string {
  return input.replace(/\D/g, "");
}

function parseIntEnv(
  raw: string | undefined,
  fallback: number,
  minimum: number
): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number.parseInt(raw.trim(), 10);
  return Number.isFinite(value) ? Math.max(minimum, value) : fallback;
}

function parseBoolEnv(raw: string | undefined, fallback: boolean): boolean {
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = raw.trim().toLowerCase();
  return value === "true" || value === "1" || value === "yes";
}

export function loadEnv(): EnvConfig {
  const sessionPath =
    process.env.WHATSAPP_SESSION_PATH ??
    process.env.SESSION_PATH ??
    ".baileys_auth";

  return {
    allowedContacts: parseAllowedContacts(process.env.ALLOWED_CONTACTS),
    openaiApiKey:
      process.env.INFERENCE_API_TOKEN?.trim() ||
      process.env.OPENAI_API_KEY?.trim() ||
      undefined,
    openaiBaseUrl:
      process.env.WHATSAPP_INFERENCE_BASE_URL?.trim() ||
      process.env.OPENAI_BASE_URL?.trim() ||
      undefined,
    openaiModel:
      process.env.WHATSAPP_INFERENCE_MODEL?.trim() ||
      process.env.OPENAI_MODEL?.trim() ||
      "qwen-whatsapp",
    sessionPath: path.resolve(sessionPath),
    dataDir: path.resolve(process.env.DATA_DIR?.trim() || "data"),
    conversationHistoryLimit: parseIntEnv(
      process.env.WHATSAPP_HISTORY_MAX_MESSAGES,
      10,
      1
    ),
    replyToDenied: parseBoolEnv(process.env.REPLY_TO_DENIED, false),
    deniedMessage:
      process.env.DENIED_MESSAGE ??
      "Este número não está autorizado a usar o chat.",
    printQrInTerminal: parseBoolEnv(process.env.PRINT_QR_TERMINAL, true),
    baileysLogLevel: parseBaileysLogLevel(process.env.BAILEYS_LOG_LEVEL),
    llmMaxOutputTokens: parseIntEnv(
      process.env.LLM_MAX_OUTPUT_TOKENS,
      1800,
      1
    ),
    llmTimeoutMs: parseIntEnv(
      process.env.WHATSAPP_INFERENCE_TIMEOUT_MS ?? process.env.LLM_TIMEOUT_MS,
      30_000,
      1000
    ),
  };
}

export function hasLlmEndpoint(cfg: EnvConfig): boolean {
  return Boolean(cfg.openaiApiKey || cfg.openaiBaseUrl);
}
