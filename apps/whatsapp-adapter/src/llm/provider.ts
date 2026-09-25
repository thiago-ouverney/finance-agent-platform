import OpenAI from "openai";
import type { CompletionUsage } from "openai/resources/completions";
import { hasLlmEndpoint, type EnvConfig } from "../config/env.js";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type CompleteChatResult = {
  text: string;
  usage: CompletionUsage;
};

function emptyUsage(): CompletionUsage {
  return {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  };
}

export async function completeChat(
  cfg: EnvConfig,
  messages: ChatMessage[],
  opts?: { model?: string }
): Promise<CompleteChatResult> {
  if (!hasLlmEndpoint(cfg)) {
    throw new Error(
      "completeChat exige um endpoint local ou INFERENCE_API_TOKEN/OPENAI_API_KEY"
    );
  }

  const model = opts?.model?.trim() || cfg.openaiModel;

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
      model,
      messages,
      temperature: 0.4,
      max_tokens: cfg.llmMaxOutputTokens,
    },
    signal ? { signal } : undefined
  );

  const text = res.choices[0]?.message?.content?.trim();
  const usage = res.usage ?? emptyUsage();
  return {
    text: text || "(Sem resposta do modelo.)",
    usage,
  };
}
