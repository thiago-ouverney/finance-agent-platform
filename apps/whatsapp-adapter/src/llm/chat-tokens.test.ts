import { describe, expect, it } from "vitest";
import {
  estimateChatInputTokens,
  truncateChatForRequestLimit,
} from "./chat-tokens.js";

describe("truncateChatForRequestLimit", () => {
  it("remove mensagens antigas do meio até caber no teto", () => {
    const sys = { role: "system" as const, content: "instrução curta" };
    const mid: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (let i = 0; i < 15; i++) {
      mid.push({ role: "user", content: "u".repeat(800) });
      mid.push({ role: "assistant", content: "a".repeat(800) });
    }
    const last = { role: "user" as const, content: "pergunta final" };
    const messages = [sys, ...mid, last];
    const maxOut = 500;
    const maxReq = 6000;
    const { messages: out, droppedHistoryMessages } =
      truncateChatForRequestLimit(messages, maxReq, maxOut, "gpt-4o-mini");
    expect(droppedHistoryMessages).toBeGreaterThan(0);
    expect(out[0]).toEqual(sys);
    expect(out[out.length - 1]?.role).toBe("user");
    expect(
      estimateChatInputTokens(out, "gpt-4o-mini") + maxOut
    ).toBeLessThanOrEqual(maxReq);
  });
});
