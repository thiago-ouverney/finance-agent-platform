import OpenAI from "openai";
import { hasLlmEndpoint, type EnvConfig } from "../config/env.js";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export async function completeChat(
  cfg: EnvConfig,
  messages: ChatMessage[]
): Promise<string> {
  if (!hasLlmEndpoint(cfg)) {
    throw new Error(
      "completeChat exige um endpoint local ou INFERENCE_API_TOKEN/OPENAI_API_KEY"
    );
  }

  const client = new OpenAI({
    apiKey: cfg.openaiApiKey || "local-openai-compatible",
    baseURL: cfg.openaiBaseUrl,
  });
  const timeoutMs = cfg.llmTimeoutMs;
  const signal =
    typeof AbortSignal !== "undefined" && "timeout" in AbortSignal
      ? AbortSignal.timeout(timeoutMs)
      : undefined;

  const res = await client.chat.completions.create(
    {
      model: cfg.openaiModel,
      messages,
      temperature: 0.4,
      max_tokens: cfg.llmMaxOutputTokens,
    },
    signal ? { signal } : undefined
  );

  const text = res.choices[0]?.message?.content?.trim();
  return text || "(Sem resposta do modelo.)";
}
