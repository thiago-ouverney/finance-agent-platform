import "./bootstrap-webcrypto.js";

/**
 * Os testes devem ser determinísticos e não depender do `.env` da máquina
 * (chaves de API, allowlist, tetos de LLM, etc.). O Vitest/Vite pode injetar
 * variáveis do `.env` no `process.env` antes dos arquivos de teste rodarem;
 * este setup roda cedo e neutraliza as que costumam quebrar asserções.
 */
const KEYS_TO_DELETE = [
  "OPENAI_API_KEY",
  "OPENAI_BASE_URL",
  "INFERENCE_API_TOKEN",
  "WHATSAPP_INFERENCE_BASE_URL",
  "WHATSAPP_INFERENCE_MODEL",
  "WHATSAPP_INFERENCE_TIMEOUT_MS",
  "WHATSAPP_SESSION_PATH",
  "ALLOWED_CONTACTS",
  "ADMIN_TOKEN",
  "POLICY_DB_PATH",
  "POLICY_SOURCE",
  "POLICY_CACHE_TTL_MS",
  "LLM_MAX_REQUEST_TOKENS",
  "LLM_MAX_OUTPUT_TOKENS",
  "LLM_DAILY_TOKEN_BUDGET_PER_USER",
  "LLM_DAILY_BUDGET_TIMEZONE",
  "LLM_TIMEOUT_MS",
  "LLM_LOG_JSON",
  "ALLOWED_OPENAI_MODELS",
] as const;

for (const k of KEYS_TO_DELETE) {
  delete process.env[k];
}

process.env.OPENAI_MODEL = "gpt-4o-mini";
