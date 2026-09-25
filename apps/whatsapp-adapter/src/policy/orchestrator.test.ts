import { describe, expect, it } from "vitest";
import { intersectGrantedWithRegistry } from "./orchestrator.js";

describe("intersectGrantedWithRegistry", () => {
  it("ignora chaves órfãs e mantém só registry", () => {
    const s = new Set(["finance.llm_chat", "legacy.unknown", "finance.csv_import"]);
    const out = intersectGrantedWithRegistry(s);
    expect(out.size).toBe(2);
    expect(out.has("finance.llm_chat")).toBe(true);
    expect(out.has("finance.csv_import")).toBe(true);
  });
});
