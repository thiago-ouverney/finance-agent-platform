import fs from "node:fs/promises";
import path from "node:path";

export type DailyTokenTotals = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  updatedAt: string;
};

const ZERO: DailyTokenTotals = {
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  updatedAt: "",
};

/** YYYY-MM-DD no fuso informado (ex.: America/Sao_Paulo). */
export function calendarDateKeyInTimeZone(
  timeZone: string,
  now = new Date()
): string {
  const tz = timeZone.trim() || "UTC";
  try {
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: tz,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(now);
  } catch {
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: "UTC",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(now);
  }
}

export function safeChatIdForFilename(chatId: string): string {
  return chatId.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function usagePath(
  dataDir: string,
  chatId: string,
  dateKey: string
): string {
  const safe = safeChatIdForFilename(chatId);
  return path.join(dataDir, "contacts", safe, "usage", `llm_usage_${dateKey}.json`);
}

export async function loadDailyUsage(
  dataDir: string,
  chatId: string,
  dateKey: string
): Promise<DailyTokenTotals> {
  const p = usagePath(dataDir, chatId, dateKey);
  try {
    const raw = await fs.readFile(p, "utf8");
    const j = JSON.parse(raw) as Partial<DailyTokenTotals>;
    return {
      prompt_tokens: Math.max(0, Number(j.prompt_tokens) || 0),
      completion_tokens: Math.max(0, Number(j.completion_tokens) || 0),
      total_tokens: Math.max(0, Number(j.total_tokens) || 0),
      updatedAt: typeof j.updatedAt === "string" ? j.updatedAt : "",
    };
  } catch {
    return { ...ZERO };
  }
}

export async function addDailyUsage(
  dataDir: string,
  chatId: string,
  dateKey: string,
  delta: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  }
): Promise<void> {
  await fs.mkdir(path.dirname(usagePath(dataDir, chatId, dateKey)), {
    recursive: true,
  });
  const cur = await loadDailyUsage(dataDir, chatId, dateKey);
  const next: DailyTokenTotals = {
    prompt_tokens: cur.prompt_tokens + Math.max(0, delta.prompt_tokens),
    completion_tokens:
      cur.completion_tokens + Math.max(0, delta.completion_tokens),
    total_tokens: cur.total_tokens + Math.max(0, delta.total_tokens),
    updatedAt: new Date().toISOString(),
  };
  await fs.writeFile(
    usagePath(dataDir, chatId, dateKey),
    JSON.stringify(next, null, 2),
    "utf8"
  );
}
