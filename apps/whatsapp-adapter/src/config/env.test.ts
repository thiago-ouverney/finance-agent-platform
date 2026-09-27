import { afterEach, describe, expect, it, vi } from "vitest";
import { hasLlmEndpoint, loadEnv } from "./env.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("configuração do runtime de inferência", () => {
  it("usa as variáveis do monorepo para Ollama/OpenAI-compatible", () => {
    vi.stubEnv("WHATSAPP_INFERENCE_BASE_URL", "http://127.0.0.1:8000/v1");
    vi.stubEnv("WHATSAPP_INFERENCE_MODEL", "qwen-whatsapp");
    vi.stubEnv("WHATSAPP_INFERENCE_TIMEOUT_MS", "120000");
    vi.stubEnv("WHATSAPP_SESSION_PATH", ".session-test");
    vi.stubEnv("DATA_DIR", ".data-test");
    vi.stubEnv("WHATSAPP_HISTORY_MAX_MESSAGES", "10");

    const cfg = loadEnv();

    expect(cfg.openaiBaseUrl).toBe("http://127.0.0.1:8000/v1");
    expect(cfg.openaiModel).toBe("qwen-whatsapp");
    expect(cfg.llmTimeoutMs).toBe(120000);
    expect(cfg.sessionPath.endsWith(".session-test")).toBe(true);
    expect(cfg.dataDir.endsWith(".data-test")).toBe(true);
    expect(cfg.conversationHistoryLimit).toBe(10);
    expect(hasLlmEndpoint(cfg)).toBe(true);
  });

  it("mantém compatibilidade com as variáveis OPENAI", () => {
    vi.stubEnv("OPENAI_BASE_URL", "https://example.test/v1");
    vi.stubEnv("OPENAI_API_KEY", "test-key");
    vi.stubEnv("OPENAI_MODEL", "legacy-model");

    const cfg = loadEnv();

    expect(cfg.openaiBaseUrl).toBe("https://example.test/v1");
    expect(cfg.openaiApiKey).toBe("test-key");
    expect(cfg.openaiModel).toBe("legacy-model");
  });
});
