import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../llm/provider.js", () => ({
  completeChat: vi.fn(),
}));

import { handleUserMessage } from "./finance-agent.js";
import { completeChat } from "../llm/provider.js";
import type { EnvConfig } from "../config/env.js";
import { LocalStore } from "../storage/store.js";
import {
  addDailyUsage,
  calendarDateKeyInTimeZone,
  loadDailyUsage,
} from "../llm/usage-store.js";
import { testEnvConfig } from "../test/test-env-config.js";

function baseCfg(over: Partial<EnvConfig> = {}): EnvConfig {
  return testEnvConfig({ openaiApiKey: "sk-test", ...over });
}

async function completeContext(store: LocalStore, contactId: string): Promise<void> {
  await store.ensureContactScaffold(contactId);
  const dir = store.getContactContextPath(contactId);
  await writeFile(path.join(dir, "profile.md"), "- nome: Ana\n", "utf8");
  await writeFile(path.join(dir, "communication.md"), "- tom: direto\n", "utf8");
  await writeFile(path.join(dir, "rules.md"), "- regra: base\n", "utf8");
  await writeFile(
    path.join(dir, "rules_custom.csv"),
    "pattern,categoria,kind,prioridade,observacao\npix,Financeiro,financeiro_nao_consumo,100,ok\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "billing_cycle.csv"),
    "campo,valor,observacao\nfatura_vencimento_dia,10,x\nfatura_fechamento_dia,3,x\ncontrole_mensal_inicio_dia,1,x\ncontrole_mensal_fim_dia,30,x\n",
    "utf8"
  );
  await writeFile(
    path.join(dir, "goals.csv"),
    "tipo,categoria,meta_mensal,data_alvo,tolerancia_percentual,observacao\ninvestimento,geral,1000,2026-12-31,10,ok\nreserva_emergencia,geral,800,2026-12-31,10,ok\n",
    "utf8"
  );
}

describe("handleUserMessage limites LLM", () => {
  let dir: string;

  afterEach(async () => {
    vi.mocked(completeChat).mockReset();
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  it("não chama o modelo quando o orçamento diário já foi ultrapassado", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-llm-budget-"));
    const store = new LocalStore(dir);
    await completeContext(store, "userA");
    const dk = calendarDateKeyInTimeZone("UTC");
    await addDailyUsage(dir, "userA", dk, {
      prompt_tokens: 300,
      completion_tokens: 100,
      total_tokens: 400,
    });
    const reply = await handleUserMessage(
      baseCfg({
        dataDir: dir,
        llmDailyTokenBudgetPerUser: 500,
      }),
      store,
      "userA",
      "Olá, preciso de ajuda",
      {}
    );
    expect(reply).toContain("Limite diário");
    expect(completeChat).not.toHaveBeenCalled();
  });

  it("chama o modelo e persiste uso quando há orçamento diário configurado", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wf-llm-ok-"));
    const store = new LocalStore(dir);
    await completeContext(store, "userB");
    vi.mocked(completeChat).mockResolvedValue({
      text: "Resposta simulada",
      usage: {
        prompt_tokens: 12,
        completion_tokens: 8,
        total_tokens: 20,
      },
    } as Awaited<ReturnType<typeof completeChat>>);
    const reply = await handleUserMessage(
      baseCfg({
        dataDir: dir,
        llmDailyTokenBudgetPerUser: 50_000,
      }),
      store,
      "userB",
      "Olá",
      {}
    );
    expect(reply).toContain("Resposta simulada");
    expect(completeChat).toHaveBeenCalled();
    const dk = calendarDateKeyInTimeZone("UTC");
    const u = await loadDailyUsage(dir, "userB", dk);
    expect(u.total_tokens).toBe(20);
  });
});
