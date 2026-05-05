import { encodeChat } from "gpt-tokenizer";

export type ChatMsg = {
  role: "system" | "user" | "assistant";
  content: string;
};

export function estimateChatInputTokens(
  messages: ChatMsg[],
  model: string
): number {
  try {
    return encodeChat(messages as never, model as never).length;
  } catch {
    try {
      return encodeChat(messages as never).length;
    } catch {
      const chars = messages.reduce((n, m) => n + m.content.length, 0);
      return Math.ceil(chars / 3);
    }
  }
}

/**
 * Garante `estimateChatInputTokens(…) + maxOutputTokens <= maxRequestTokens`
 * removendo primeiro mensagens antigas entre o prefixo de `system` e a última
 * mensagem (tipicamente o `user` atual).
 */
export function truncateChatForRequestLimit(
  messages: ChatMsg[],
  maxRequestTokens: number,
  maxOutputTokens: number,
  model: string
): { messages: ChatMsg[]; droppedHistoryMessages: number } {
  let dropped = 0;
  let cur = messages;

  const budgetIn = maxRequestTokens - maxOutputTokens;
  if (!Number.isFinite(budgetIn) || budgetIn <= 0) {
    return { messages: cur, droppedHistoryMessages: 0 };
  }

  while (estimateChatInputTokens(cur, model) > budgetIn) {
    let prefixEnd = 0;
    while (cur[prefixEnd]?.role === "system") prefixEnd += 1;
    const last = cur[cur.length - 1];
    const middle = cur.slice(prefixEnd, -1);
    if (middle.length > 0) {
      cur = [...cur.slice(0, prefixEnd), ...middle.slice(1), last];
      dropped += 1;
      continue;
    }
    if (last && last.content.length > 400) {
      const nextLen = Math.max(200, Math.floor(last.content.length * 0.85));
      cur = [
        ...cur.slice(0, -1),
        { ...last, content: last.content.slice(0, nextLen) + "…" },
      ];
      continue;
    }
    break;
  }

  return { messages: cur, droppedHistoryMessages: dropped };
}
