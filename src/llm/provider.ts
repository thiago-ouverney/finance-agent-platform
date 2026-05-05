import OpenAI from "openai";
import type { CompletionUsage } from "openai/resources/completions";
import type { EnvConfig } from "../config/env.js";

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
  messages: ChatMessage[]
): Promise<CompleteChatResult> {
  if (!cfg.openaiApiKey) {
    throw new Error("completeChat exige OPENAI_API_KEY");
  }

  const client = new OpenAI({ apiKey: cfg.openaiApiKey });
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
  const usage = res.usage ?? emptyUsage();
  return {
    text: text || "(Sem resposta do modelo.)",
    usage,
  };
}
