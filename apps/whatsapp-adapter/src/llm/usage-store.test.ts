import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  addDailyUsage,
  calendarDateKeyInTimeZone,
  loadDailyUsage,
} from "./usage-store.js";

describe("usage-store", () => {
  let dir: string;

  afterEach(async () => {
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  it("acumula total_tokens por dia e chatId", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-usage-"));
    const dk = "2099-01-15";
    await addDailyUsage(dir, "5511@s.whatsapp.net", dk, {
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
    await addDailyUsage(dir, "5511@s.whatsapp.net", dk, {
      prompt_tokens: 20,
      completion_tokens: 10,
      total_tokens: 30,
    });
    const u = await loadDailyUsage(dir, "5511@s.whatsapp.net", dk);
    expect(u.prompt_tokens).toBe(30);
    expect(u.completion_tokens).toBe(15);
    expect(u.total_tokens).toBe(45);
  });

  it("calendarDateKeyInTimeZone retorna YYYY-MM-DD", () => {
    const d = new Date("2026-06-15T03:00:00.000Z");
    const key = calendarDateKeyInTimeZone("UTC", d);
    expect(key).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});
