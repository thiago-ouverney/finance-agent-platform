import type { EnvConfig } from "../config/env.js";

/** EnvConfig mínimo para testes (evita drift ao acrescentar campos). */
export function testEnvConfig(over: Partial<EnvConfig> = {}): EnvConfig {
  const openaiModel = over.openaiModel ?? "gpt-4o-mini";
  return {
    allowedContacts: [],
    openaiApiKey: undefined,
    openaiBaseUrl: undefined,
    openaiModel,
    allowedOpenaiModels: over.allowedOpenaiModels ?? [
      "gpt-4o-mini",
      "gpt-4o",
    ],
    sessionPath: "/tmp",
    dataDir: "/tmp",
    policySource: "hybrid",
    policyDbPath: "/tmp/policy-test.sqlite",
    policyCacheTtlMs: 60_000,
    adminToken: undefined,
    adminHost: "127.0.0.1",
    adminPort: 3847,
    replyToDenied: false,
    deniedMessage: "negado",
    printQrInTerminal: true,
    baileysLogLevel: "silent",
    llmMaxOutputTokens: 1800,
    llmMaxRequestTokens: undefined,
    llmDailyTokenBudgetPerUser: 0,
    llmDailyBudgetTimezone: "UTC",
    llmTimeoutMs: 30_000,
    llmLogJson: false,
    ...over,
  };
}
