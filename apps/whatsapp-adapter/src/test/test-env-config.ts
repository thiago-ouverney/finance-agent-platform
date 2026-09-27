import type { EnvConfig } from "../config/env.js";

/** EnvConfig mínimo para testes (evita drift ao acrescentar campos). */
export function testEnvConfig(over: Partial<EnvConfig> = {}): EnvConfig {
  const openaiModel = over.openaiModel ?? "gpt-4o-mini";
  return {
    allowedContacts: [],
    openaiApiKey: undefined,
    openaiBaseUrl: undefined,
    openaiModel,
    sessionPath: "/tmp",
    dataDir: "/tmp",
    conversationHistoryLimit: 10,
    replyToDenied: false,
    deniedMessage: "negado",
    printQrInTerminal: true,
    baileysLogLevel: "silent",
    llmMaxOutputTokens: 1800,
    llmTimeoutMs: 30_000,
    ...over,
  };
}
