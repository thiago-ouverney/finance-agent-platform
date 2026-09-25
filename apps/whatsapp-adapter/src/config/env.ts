import path from "node:path";
import type { PolicySourceMode } from "../policy/types.js";

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
  const v = raw?.trim().toLowerCase() ?? "";
  if (v === "") return "silent";
  if (BAILEYS_LOG_LEVELS.has(v)) return v as BaileysLogLevel;
  return "silent";
}

export type EnvConfig = {
  allowedContacts: string[];
  openaiApiKey: string | undefined;
  /** Endpoint OpenAI-compatible (Ollama, vLLM, llama.cpp ou OpenAI). */
  openaiBaseUrl: string | undefined;
  openaiModel: string;
  /** Modelos que o admin pode atribuir (validação). Sempre inclui `openaiModel` ao carregar. */
  allowedOpenaiModels: string[];
  sessionPath: string;
  dataDir: string;
  /** `hybrid`: DB + fallback `ALLOWED_CONTACTS`. `db_only`: só identidades no SQLite. */
  policySource: PolicySourceMode;
  /** Caminho do SQLite de política (allowlist, modelo por ID, features, templates). */
  policyDbPath: string;
  /** TTL do cache de resolução de política no processo do bot (ms). */
  policyCacheTtlMs: number;
  /** Se definido, inicia servidor admin em `adminHost`:`adminPort` (ex.: 127.0.0.1). */
  adminToken: string | undefined;
  adminHost: string;
  adminPort: number;
  replyToDenied: boolean;
  deniedMessage: string;
  /** Baileys prints QR to terminal by default */
  printQrInTerminal: boolean;
  /** Nível Pino passado ao Baileys (`downloadMediaMessage`, socket). Default `silent`. */
  baileysLogLevel: BaileysLogLevel;
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

function parsePolicySource(raw: string | undefined): PolicySourceMode {
  const v = (raw ?? "hybrid").trim().toLowerCase();
  if (v === "db_only" || v === "db-only") return "db_only";
  return "hybrid";
}

function parseModelAllowlist(
  raw: string | undefined,
  currentModel: string
): string[] {
  const defaults = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
  ];
  const parts = raw
    ?.split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const base = parts && parts.length > 0 ? parts : defaults;
  const set = new Set(base);
  set.add(currentModel);
  return [...set];
}

export function loadEnv(): EnvConfig {
  const allowedContacts = parseAllowedContacts(process.env.ALLOWED_CONTACTS);

  const sessionPath =
    process.env.WHATSAPP_SESSION_PATH ??
    process.env.SESSION_PATH ??
    ".baileys_auth";
  const dataDir = process.env.DATA_DIR ?? "data";
  const dataDirResolved = path.resolve(dataDir);

  const openaiModel =
    process.env.WHATSAPP_INFERENCE_MODEL?.trim() ||
    process.env.OPENAI_MODEL?.trim() ||
    "gpt-4o-mini";
  const policyDbPath = path.resolve(
    process.env.POLICY_DB_PATH?.trim() ||
      path.join(dataDirResolved, "policy.sqlite")
  );

  const replyRaw = process.env.REPLY_TO_DENIED?.toLowerCase();
  const replyToDenied = replyRaw === "true" || replyRaw === "1";

  const adminTokenRaw = process.env.ADMIN_TOKEN?.trim();
  const adminToken =
    adminTokenRaw && adminTokenRaw.length > 0 ? adminTokenRaw : undefined;
  const adminHost = (process.env.ADMIN_HOST ?? "127.0.0.1").trim() || "127.0.0.1";
  const adminPort = parseIntEnv(process.env.ADMIN_PORT, 3847, { min: 1 });

  return {
    allowedContacts,
    openaiApiKey:
      process.env.INFERENCE_API_TOKEN?.trim() ||
      process.env.OPENAI_API_KEY?.trim() ||
      undefined,
    openaiBaseUrl:
      process.env.WHATSAPP_INFERENCE_BASE_URL?.trim() ||
      process.env.OPENAI_BASE_URL?.trim() ||
      undefined,
    openaiModel,
    allowedOpenaiModels: parseModelAllowlist(
      process.env.ALLOWED_OPENAI_MODELS,
      openaiModel
    ),
    sessionPath: path.resolve(sessionPath),
    dataDir: dataDirResolved,
    policySource: parsePolicySource(process.env.POLICY_SOURCE),
    policyDbPath,
    policyCacheTtlMs: parseIntEnv(process.env.POLICY_CACHE_TTL_MS, 60_000, {
      min: 1000,
    }),
    adminToken,
    adminHost,
    adminPort,
    replyToDenied,
    deniedMessage:
      process.env.DENIED_MESSAGE ??
      "Este número não está autorizado a usar o agente financeiro.",
    printQrInTerminal:
      process.env.PRINT_QR_TERMINAL !== "false" &&
      process.env.PRINT_QR_TERMINAL !== "0",
    baileysLogLevel: parseBaileysLogLevel(process.env.BAILEYS_LOG_LEVEL),
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
    llmTimeoutMs: parseIntEnv(
      process.env.WHATSAPP_INFERENCE_TIMEOUT_MS ?? process.env.LLM_TIMEOUT_MS,
      30_000,
      { min: 1000 }
    ),
    llmLogJson: parseBoolEnv(process.env.LLM_LOG_JSON, false),
  };
}

export function hasLlmEndpoint(cfg: EnvConfig): boolean {
  return Boolean(cfg.openaiApiKey || cfg.openaiBaseUrl);
}
