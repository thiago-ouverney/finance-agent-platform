import { hasLlmEndpoint, type EnvConfig } from "../config/env.js";
import { allFeatureKeys, type FeatureKey } from "../policy/feature-registry.js";
import type { EffectivePolicy } from "../policy/types.js";
import {
  estimateChatInputTokens,
  truncateChatForRequestLimit,
  type ChatMsg,
} from "../llm/chat-tokens.js";
import { completeChat, type ChatMessage } from "../llm/provider.js";
import {
  addDailyUsage,
  calendarDateKeyInTimeZone,
  loadDailyUsage,
} from "../llm/usage-store.js";
import type { LocalStore } from "../storage/store.js";
import {
  categorizeRows,
  inferMoneyLocaleFromRows,
  invoiceMonthFromFilename,
  rowsFromLooseCsv,
  sumByCategory,
  totalConsumo,
  totalNaoConsumo,
} from "../finance/categorizer.js";
import {
  describeMoneyLocale,
  parseMoneyAmount,
} from "../finance/money-parse.js";
import {
  applyCustomRules,
  parseCustomRulesCsv,
  type CustomRule,
} from "../finance/custom-rules.js";
import {
  aggregateRowsByInvoiceMonth,
  computeBudgetSummary,
  topCategories,
} from "../finance/budget-planner.js";
import type { ContactContextRead } from "../storage/store.js";

const SYSTEM_PROMPT = `Você é um planejador financeiro no WhatsApp, em português do Brasil.
Tom: direto, empático, sem julgamento, focado em ação.

Princípios:
- Separar pagamento de fatura, encargos e IOF do consumo principal para não distorcer análises.
- Preferir visão por mês da fatura quando o usuário enviar arquivos nomeados como fatura-AAAAMMDD.csv.
- Identificar gargalos comuns: Assinaturas, comer fora, pulverização no cartão.
- Sugerir tetos por categoria e rotina (dia do pagamento, conta conjunta vs pessoal).
- Não prometer retorno de investimento; não dar orientação fiscal/legal definitiva.
- Quando faltar dado, pergunte 1–3 perguntas objetivas.

Quando receber números estruturados do sistema (resumo de CSV), use-os como fonte principal.`;

const CATEGORIES_LIST = [
  "Mercado",
  "Assinatura",
  "Restaurante",
  "Lanchonete",
  "Farmácia",
  "Uber",
  "Don",
  "Lazer",
  "Outros",
  "Financeiro",
  "Transporte",
  "Pagamento fatura",
].join(", ");

type ContextCheck = {
  missing: string[];
  contextPath: string;
};

type GoalsConfig = {
  investimentoMensal?: number;
  reservaMensal?: number;
};

function parseContextCsv(text: string): Array<Record<string, string>> {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) return [];
  const delimiter =
    lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
  const headers = lines[0].split(delimiter).map((h) => h.trim().toLowerCase());
  return lines.slice(1).map((line) => {
    const cols = line.split(delimiter).map((c) => c.trim());
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = cols[i] ?? "";
    });
    return row;
  });
}

function isPlaceholder(value: string | undefined): boolean {
  if (!value) return true;
  const t = value.trim();
  return t.length === 0 || /^preencher$/i.test(t);
}

function validateContextReadiness(
  ctx: ContactContextRead,
  contextPath: string
): ContextCheck {
  const missing: string[] = [];
  if (/PREENCHER/i.test(ctx.profileMd)) {
    missing.push("perfil da pessoa (`profile.md`)");
  }
  if (/PREENCHER/i.test(ctx.communicationMd)) {
    missing.push("forma de comunicacao (`communication.md`)");
  }
  if (/PREENCHER/i.test(ctx.rulesMd)) {
    missing.push("regras base de categorizacao (`rules.md`)");
  }

  const cycleRows = parseContextCsv(ctx.billingCycleCsv);
  const requiredCycle = new Set([
    "fatura_vencimento_dia",
    "fatura_fechamento_dia",
    "controle_mensal_inicio_dia",
    "controle_mensal_fim_dia",
  ]);
  for (const row of cycleRows) {
    const field = row.campo;
    if (field && requiredCycle.has(field) && isPlaceholder(row.valor)) {
      missing.push(`ciclo de faturamento (${field} em \`billing_cycle.csv\`)`);
    }
  }
  for (const k of requiredCycle) {
    if (!cycleRows.some((r) => r.campo === k)) {
      missing.push(`ciclo de faturamento (${k} em \`billing_cycle.csv\`)`);
    }
  }

  const goalsRows = parseContextCsv(ctx.goalsCsv);
  const investimento = goalsRows.find((r) => r.tipo === "investimento");
  const reserva = goalsRows.find((r) => r.tipo === "reserva_emergencia");
  if (!investimento || isPlaceholder(investimento.meta_mensal)) {
    missing.push("meta de investimento (`goals.csv`, tipo=investimento)");
  }
  if (!reserva || isPlaceholder(reserva.meta_mensal)) {
    missing.push("meta de reserva (`goals.csv`, tipo=reserva_emergencia)");
  }
  if (/PREENCHER/i.test(ctx.rulesCustomCsv)) {
    missing.push("regra customizavel de categorizacao (`rules_custom.csv`)");
  }

  return { missing, contextPath };
}

function buildMissingContextReply(check: ContextCheck): string {
  const unique = [...new Set(check.missing)];
  return [
    "Antes de responder com precisao, preciso que voce complete o contexto deste contato.",
    "",
    `Pasta do contato: ${check.contextPath}`,
    "Campos faltantes:",
    ...unique.map((m) => `- ${m}`),
    "",
    "Depois de preencher, envie sua pergunta novamente.",
  ].join("\n");
}

function parseGoalsConfig(goalsCsv: string): GoalsConfig {
  const rows = parseContextCsv(goalsCsv);
  const parseMoney = (value: string | undefined): number | undefined => {
    if (!value || isPlaceholder(value)) return undefined;
    return parseMoneyAmount(value, "br");
  };
  return {
    investimentoMensal: parseMoney(
      rows.find((r) => r.tipo === "investimento")?.meta_mensal
    ),
    reservaMensal: parseMoney(
      rows.find((r) => r.tipo === "reserva_emergencia")?.meta_mensal
    ),
  };
}


type ParsedCommand =
  | { kind: "help" }
  | { kind: "status" }
  | { kind: "categorias" }
  | { kind: "orcamento"; renda?: number; fixos?: number }
  | { kind: "limpar"; confirm: boolean };

export function parseCommandLine(trimmed: string): ParsedCommand | null {
  const t = trimmed.trim();
  if (!t.startsWith("/")) return null;
  const lower = t.toLowerCase();
  if (/^\/(help|ajuda)(\s|$)/.test(lower)) return { kind: "help" };
  if (/^\/status(\s|$)/.test(lower)) return { kind: "status" };
  if (/^\/categorias(\s|$)/.test(lower)) return { kind: "categorias" };
  if (/^\/limpar\s+sim$/i.test(t)) return { kind: "limpar", confirm: true };
  if (/^\/limpar$/i.test(t)) return { kind: "limpar", confirm: false };
  if (/^\/orcamento\b/i.test(t)) {
    const { renda, fixos } = parseBudgetArgs(t);
    return { kind: "orcamento", renda, fixos };
  }
  return null;
}

function looksLikeCsv(text: string): boolean {
  const t = text.trim();
  if (t.split(/\r?\n/).length < 2) return false;
  return (
    (t.includes(",") || t.includes(";")) &&
    /valor|descri|amount|memo/i.test(t)
  );
}

/** Extrai renda e fixos (texto livre ou /orcamento renda=… fixos=…) */
export function parseBudgetArgs(text: string): {
  renda?: number;
  fixos?: number;
} {
  const hints = parseMoneyHints(text);
  const parseNum = (s: string) => parseMoneyAmount(s, "br");
  const rEq = text.match(/renda\s*=\s*([\d.,]+)/i);
  const fEq = text.match(/fixos?\s*=\s*([\d.,]+)/i);
  return {
    renda: rEq?.[1] ? parseNum(rEq[1]) : hints.renda,
    fixos: fEq?.[1] ? parseNum(fEq[1]) : hints.fixos,
  };
}

function parseMoneyHints(text: string): {
  renda?: number;
  fixos?: number;
} {
  const lower = text.toLowerCase();
  const out: { renda?: number; fixos?: number } = {};
  const rendaM = lower.match(/renda\s*[:\s]*([\d.,]+)/);
  const fixosM = lower.match(/fixos?\s*[:\s]*([\d.,]+)/);
  if (rendaM?.[1]) out.renda = parseMoneyAmount(rendaM[1], "br");
  if (fixosM?.[1]) out.fixos = parseMoneyAmount(fixosM[1], "br");
  return out;
}

function formatCurrency(n: number): string {
  return n.toLocaleString("pt-BR", {
    style: "currency",
    currency: "BRL",
    minimumFractionDigits: 2,
  });
}

function footerSuggestions(): string {
  return [
    "",
    "Próximos passos:",
    "• Envie um CSV de fatura (colado ou arquivo .csv) ou use `/orcamento renda=12000 fixos=4701` após ter dados salvos.",
    "• `/help` lista os comandos.",
  ].join("\n");
}

export function defaultEffectivePolicy(cfg: EnvConfig): EffectivePolicy {
  return {
    model: cfg.openaiModel,
    features: new Set<FeatureKey>(allFeatureKeys()),
    identitySource: "env",
  };
}

function defaultNoDataReply(): string {
  return [
    "Olá! Sou o planejador financeiro (modo local, sem modelo de linguagem ligado).",
    "",
    "Para começar:",
    "• Envie um CSV com colunas de descrição e valor, ou",
    "• Digite `/help` para ver comandos.",
    "",
    "Para conversas com IA, configure o endpoint local ou uma chave de API no servidor.",
  ].join("\n");
}

function mergeBlocks(deterministicBlock: string, llmPart: string): string {
  const d = deterministicBlock.trim();
  const l = llmPart.trim();
  if (d && l) return `${d}\n\n---\n\n${l}`;
  return d || l;
}

async function dispatchCommand(
  cmd: ParsedCommand,
  cfg: EnvConfig,
  store: LocalStore,
  contactId: string,
  ctxCheck: ContextCheck,
  goals: GoalsConfig,
  customRules: CustomRule[],
  effectiveModel: string,
  llmFeatureEnabled: boolean
): Promise<string> {
  switch (cmd.kind) {
    case "help":
      return [
        "Comandos:",
        "• `/help` ou `/ajuda` — esta ajuda",
        "• `/status` — histórico salvo e se o modelo (IA) está ativo",
        "• `/categorias` — categorias usadas na classificação",
        "• `/orcamento renda=12000 fixos=4701` — estimativa usando faturas já enviadas",
        "• `/limpar` — pede confirmação; `/limpar sim` apaga histórico e faturas deste contato",
        "",
        "Envie CSV (colado ou arquivo .csv) com descrição e valor para análise automática.",
      ].join("\n");

    case "categorias":
      return `Categorias suportadas (classificação automática):\n${CATEGORIES_LIST}`;

    case "status": {
      const stmts = await store.loadStatements(contactId);
      const n = stmts.invoices.length;
      const llm =
        hasLlmEndpoint(cfg) && llmFeatureEnabled
          ? `Modelo (IA): ativo (${effectiveModel})`
          : "Modelo (IA): desligado — apenas análise local";
      const lines = [
        "Status",
        llm,
        `Saída máx. por chamada: ${cfg.llmMaxOutputTokens} tokens`,
      ];
      if (cfg.llmMaxRequestTokens != null && cfg.llmMaxRequestTokens > 0) {
        lines.push(
          `Teto por requisição (entrada estimada + saída): ${cfg.llmMaxRequestTokens} tokens`
        );
      }
      if (cfg.llmDailyTokenBudgetPerUser > 0) {
        const dk = calendarDateKeyInTimeZone(cfg.llmDailyBudgetTimezone);
        const u = await loadDailyUsage(cfg.dataDir, contactId, dk);
        lines.push(
          `Orçamento modelo hoje: ${u.total_tokens}/${cfg.llmDailyTokenBudgetPerUser} total_tokens (fuso ${cfg.llmDailyBudgetTimezone})`
        );
      }
      lines.push(
        `Faturas/CSV salvos neste contato: ${n}`,
        n > 0
          ? `Último upload: ${stmts.invoices[n - 1]?.uploadedAt ?? "?"}`
          : "Envie um CSV para começar."
      );
      if (ctxCheck.missing.length > 0) {
        lines.push(
          `Contexto do contato: pendente (${ctxCheck.missing.length} item(ns) faltando)`,
          `Pasta para preencher: ${ctxCheck.contextPath}`
        );
      } else {
        lines.push("Contexto do contato: completo");
      }
      return lines.join("\n");
    }

    case "limpar": {
      if (!cmd.confirm) {
        return [
          "Isso apaga o histórico de mensagens e as faturas salvas **deste contato**.",
          "Para confirmar, envie: `/limpar sim`",
        ].join("\n");
      }
      return "Histórico e faturas deste contato foram apagados.";
    }

    case "orcamento": {
      if (ctxCheck.missing.length > 0) {
        return buildMissingContextReply(ctxCheck);
      }
      const stmts = await store.loadStatements(contactId);
      if (stmts.invoices.length === 0) {
        return [
          "Ainda não há faturas salvas. Envie pelo menos um CSV (colado ou arquivo) antes de usar `/orcamento`.",
          "Exemplo: `/orcamento renda=12000 fixos=4701`",
        ].join("\n");
      }
      const renda = cmd.renda;
      const fixos = cmd.fixos;
      if (renda === undefined || fixos === undefined) {
        return [
          "Informe renda e fixos, por exemplo:",
          "`/orcamento renda=12000 fixos=4701`",
          "Ou numa linha: `renda 12000 fixos 4701`",
        ].join("\n");
      }
      const batches: Array<{
        month: string;
        rows: ReturnType<typeof categorizeRows>;
      }> = [];
      for (const inv of stmts.invoices.slice(-6)) {
        const cat = applyCustomRules(categorizeRows(inv.rows), customRules);
        const m =
          invoiceMonthFromFilename(inv.label) ?? inv.uploadedAt.slice(0, 7);
        batches.push({ month: m, rows: cat });
      }
      const agg = aggregateRowsByInvoiceMonth(batches);
      const cartaoPorMes = agg.map((a) => a.consumo);
      const summary = computeBudgetSummary(
        {
          rendaMensal: renda,
          fixosCasa: fixos,
          metaInvestimento: goals.investimentoMensal,
          metaReserva: goals.reservaMensal,
        },
        cartaoPorMes
      );
      return [
        "[Orçamento familiar (estimativa)]",
        `Renda informada: ${formatCurrency(summary.renda)}`,
        `Fixos da casa: ${formatCurrency(summary.fixosCasa)}`,
        `Média consumo cartão (meses com dados): ${formatCurrency(summary.mediaCartaoConsumo)}`,
        `Fixos + média cartão: ${formatCurrency(summary.totalPlanejadoBase)}`,
        `Sobra estimada: ${formatCurrency(summary.sobraSobreRenda)}`,
        `Sugestão de teto total gastos (após investimento R$ ${summary.metaInvestimento.toFixed(0)} e reserva R$ ${summary.metaReserva.toFixed(0)}): ~${formatCurrency(summary.tetoGastoSugerido)}`,
      ].join("\n");
    }

    default:
      return defaultNoDataReply();
  }
}

async function buildCsvDeterministicBlock(
  trimmed: string,
  filename: string | undefined,
  contactId: string,
  store: LocalStore,
  customRules: CustomRule[],
  rulesCustomCsv: string,
  sourceLabel: string,
  rawCsvFromFile: string | undefined
): Promise<string> {
  const rawRows = rowsFromLooseCsv(trimmed);
  const rows = applyCustomRules(categorizeRows(rawRows), customRules);
  const label = filename ?? "colado";
  await store.appendInvoiceRows(contactId, label, rawRows);

  const month = filename ? invoiceMonthFromFilename(filename) : undefined;
  const moneyLocale = inferMoneyLocaleFromRows(rawRows);
  const firstLine = trimmed.split(/\r?\n/)[0] ?? "";
  const csvSeparator =
    firstLine.includes(";") && !firstLine.includes(",")
      ? "ponto e vírgula"
      : "vírgula";
  const sums = sumByCategory(rows);
  const tc = topCategories(sums, 8);
  const tConsumo = totalConsumo(rows);
  const tFin = totalNaoConsumo(rows);

  const block = [
    `[Análise CSV: ${label}${month ? ` | mês fatura: ${month}` : ""}]`,
    `Formato de valores detectado: ${describeMoneyLocale(moneyLocale)} | separador CSV: ${csvSeparator}`,
    `Total consumo (sem pagamentos/encargos): ${formatCurrency(tConsumo)}`,
    tFin > 0 ? `Pagamentos/encargos (fora do consumo): ${formatCurrency(tFin)}` : "",
    "Top categorias:",
    ...tc.map(
      (x) =>
        `- ${x.categoria}: ${formatCurrency(x.valor)} (${x.pct.toFixed(1)}%)`
    ),
    "",
  ]
    .filter(Boolean)
    .join("\n");

  await store.persistCsvArtifact(contactId, {
    fileName: label.endsWith(".csv") ? label : `${label}.csv`,
    source: sourceLabel,
    rawCsv: rawCsvFromFile ?? trimmed,
    parsedRows: rawRows,
    summaryMarkdown: [
      `# Resumo do arquivo ${label}`,
      "",
      `- timestamp: ${new Date().toISOString()}`,
      `- origem: ${sourceLabel}`,
      `- mes_fatura_inferido: ${month ?? "nao identificado"}`,
      `- total_consumo: ${formatCurrency(tConsumo)}`,
      `- total_nao_consumo: ${formatCurrency(tFin)}`,
      "",
      "## Top categorias",
      ...tc.map(
        (x) => `- ${x.categoria}: ${formatCurrency(x.valor)} (${x.pct.toFixed(1)}%)`
      ),
      "",
      "## Bloco retornado ao usuario",
      "```",
      block,
      "```",
    ].join("\n"),
  });
  await store.rebuildConsolidatedExpenses(contactId, rulesCustomCsv);

  return block;
}

/** Monta bloco de orçamento a partir de texto livre com renda/fixos */
async function appendBudgetFromHints(
  trimmed: string,
  contactId: string,
  store: LocalStore,
  deterministicBlock: string,
  goals: GoalsConfig,
  customRules: CustomRule[]
): Promise<string> {
  const hints = parseBudgetArgs(trimmed);
  const stmts = await store.loadStatements(contactId);
  if (
    hints.renda === undefined ||
    hints.fixos === undefined ||
    stmts.invoices.length === 0
  ) {
    return deterministicBlock;
  }
  const batches: Array<{
    month: string;
    rows: ReturnType<typeof categorizeRows>;
  }> = [];
  for (const inv of stmts.invoices.slice(-6)) {
    const cat = applyCustomRules(categorizeRows(inv.rows), customRules);
    const m =
      invoiceMonthFromFilename(inv.label) ?? inv.uploadedAt.slice(0, 7);
    batches.push({ month: m, rows: cat });
  }
  const agg = aggregateRowsByInvoiceMonth(batches);
  const cartaoPorMes = agg.map((a) => a.consumo);
  const summary = computeBudgetSummary(
    {
      rendaMensal: hints.renda,
      fixosCasa: hints.fixos,
      metaInvestimento: goals.investimentoMensal,
      metaReserva: goals.reservaMensal,
    },
    cartaoPorMes
  );
  return (
    deterministicBlock +
    [
      "",
      "[Orçamento familiar (estimativa)]",
      `Renda informada: ${formatCurrency(summary.renda)}`,
      `Fixos da casa: ${formatCurrency(summary.fixosCasa)}`,
      `Média consumo cartão (meses com dados): ${formatCurrency(summary.mediaCartaoConsumo)}`,
      `Fixos + média cartão: ${formatCurrency(summary.totalPlanejadoBase)}`,
      `Sobra estimada: ${formatCurrency(summary.sobraSobreRenda)}`,
      `Sugestão de teto total gastos (após investimento R$ ${summary.metaInvestimento.toFixed(0)} e reserva R$ ${summary.metaReserva.toFixed(0)}): ~${formatCurrency(summary.tetoGastoSugerido)}`,
    ].join("\n")
  );
}

export async function handleUserMessage(
  cfg: EnvConfig,
  store: LocalStore,
  contactId: string,
  body: string,
  opts?: {
    filename?: string;
    sourceChatId?: string;
    sourceJid?: string;
    rawCsvFromFile?: string;
    /** Política efetiva por ID (modelo + features). Omitido = comportamento legado (todas as features). */
    policy?: EffectivePolicy;
    /** Texto quando uma feature necessária está desligada (vem do DB de templates). */
    featureDisabledReply?: string;
  }
): Promise<string> {
  const effective = opts?.policy ?? defaultEffectivePolicy(cfg);
  const featureDisabled =
    opts?.featureDisabledReply?.trim() ||
    "Esta funcionalidade não está liberada para o seu usuário. Fale com o administrador.";

  await store.ensureContactScaffold(contactId);
  const context = await store.loadContactContext(contactId);
  const contextPath = store.getContactContextPath(contactId);
  const ctxCheck = validateContextReadiness(context, contextPath);
  const goals = parseGoalsConfig(context.goalsCsv);
  const customRules = parseCustomRulesCsv(context.rulesCustomCsv);

  const trimmed = body.trim();

  const cmd = parseCommandLine(trimmed);
  if (cmd) {
    if (!effective.features.has("finance.local_commands")) {
      await store.appendMessage(contactId, { role: "user", text: trimmed });
      await store.appendMessage(contactId, {
        role: "assistant",
        text: featureDisabled,
      });
      return featureDisabled;
    }
    if (cmd.kind === "limpar" && cmd.confirm) {
      await store.clearAll(contactId);
    }
    await store.appendMessage(contactId, { role: "user", text: trimmed });
    const llmFeat =
      hasLlmEndpoint(cfg) && effective.features.has("finance.llm_chat");
    const out = await dispatchCommand(
      cmd,
      cfg,
      store,
      contactId,
      ctxCheck,
      goals,
      customRules,
      effective.model,
      llmFeat
    );
    await store.appendMessage(contactId, { role: "assistant", text: out });
    return out;
  }

  const convBefore = await store.loadConversation(contactId);

  await store.appendMessage(contactId, { role: "user", text: trimmed });

  if (
    looksLikeCsv(trimmed) &&
    !effective.features.has("finance.csv_import")
  ) {
    await store.appendMessage(contactId, {
      role: "assistant",
      text: featureDisabled,
    });
    return featureDisabled;
  }

  let deterministicBlock = "";
  const sourceLabel = opts?.sourceJid ?? opts?.sourceChatId ?? "whatsapp";

  if (looksLikeCsv(trimmed)) {
    deterministicBlock = await buildCsvDeterministicBlock(
      trimmed,
      opts?.filename,
      contactId,
      store,
      customRules,
      context.rulesCustomCsv,
      sourceLabel,
      opts?.rawCsvFromFile
    );
  }

  deterministicBlock = await appendBudgetFromHints(
    trimmed,
    contactId,
    store,
    deterministicBlock,
    goals,
    customRules
  );

  if (ctxCheck.missing.length > 0) {
    const pending = buildMissingContextReply(ctxCheck);
    const reply = deterministicBlock.trim()
      ? `${deterministicBlock.trim()}\n\n---\n\n${pending}`
      : pending;
    await store.appendMessage(contactId, { role: "assistant", text: reply });
    return reply;
  }

  const llmAllowed =
    hasLlmEndpoint(cfg) && effective.features.has("finance.llm_chat");

  if (!llmAllowed) {
    let reply: string;
    if (deterministicBlock.trim()) {
      reply = deterministicBlock.trim() + footerSuggestions();
    } else {
      reply = defaultNoDataReply();
    }
    await store.appendMessage(contactId, { role: "assistant", text: reply });
    return reply;
  }

  const history: ChatMessage[] = [
    {
      role: "system",
      content: [
        SYSTEM_PROMPT,
        "",
        "Contexto do contato (preenchido em templates):",
        `- Perfil:\n${context.profileMd}`,
        `- Comunicacao:\n${context.communicationMd}`,
        `- Regras de categorizacao:\n${context.rulesMd}`,
        `- Regras custom csv:\n${context.rulesCustomCsv}`,
        `- Ciclo de faturamento:\n${context.billingCycleCsv}`,
        `- Metas:\n${context.goalsCsv}`,
      ].join("\n"),
    },
  ];

  if (deterministicBlock.trim()) {
    history.push({
      role: "system",
      content: `Contexto calculado localmente:\n${deterministicBlock}`,
    });
  }

  const recent = convBefore.messages.slice(-12);
  for (const m of recent) {
    history.push({
      role: m.role,
      content: m.text,
    });
  }

  history.push({ role: "user", content: trimmed });

  let historyForModel: ChatMsg[] = history as ChatMsg[];
  let droppedHistoryMessages = 0;
  if (cfg.llmMaxRequestTokens != null && cfg.llmMaxRequestTokens > 0) {
    const fit = truncateChatForRequestLimit(
      historyForModel,
      cfg.llmMaxRequestTokens,
      cfg.llmMaxOutputTokens,
      effective.model
    );
    historyForModel = fit.messages;
    droppedHistoryMessages = fit.droppedHistoryMessages;
  }

  const estimatedInput = estimateChatInputTokens(
    historyForModel,
    effective.model
  );

  let dateKey = "";
  let usedToday = { total_tokens: 0, prompt_tokens: 0, completion_tokens: 0 };
  if (cfg.llmDailyTokenBudgetPerUser > 0) {
    dateKey = calendarDateKeyInTimeZone(cfg.llmDailyBudgetTimezone);
    usedToday = await loadDailyUsage(cfg.dataDir, contactId, dateKey);
  }

  let llmPart: string;
  if (
    cfg.llmDailyTokenBudgetPerUser > 0 &&
    usedToday.total_tokens + estimatedInput + cfg.llmMaxOutputTokens >
      cfg.llmDailyTokenBudgetPerUser
  ) {
    llmPart =
      "[Limite diário de tokens do modelo atingido para este número. Volte amanhã ou peça ao administrador para aumentar o orçamento.]";
  } else {
    try {
      const { text, usage } = await completeChat(
        cfg,
        historyForModel as ChatMessage[],
        { model: effective.model }
      );
      llmPart = text;
      if (cfg.llmDailyTokenBudgetPerUser > 0 && dateKey) {
        await addDailyUsage(cfg.dataDir, contactId, dateKey, {
          prompt_tokens: usage.prompt_tokens,
          completion_tokens: usage.completion_tokens,
          total_tokens: usage.total_tokens,
        });
      }
      if (cfg.llmLogJson) {
        console.info(
          JSON.stringify({
            event: "llm_completion",
            contactId,
            model: effective.model,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            estimated_input_tokens: estimatedInput,
            truncated_history_messages: droppedHistoryMessages,
          })
        );
      }
    } catch (e) {
      llmPart = `[Erro ao chamar o modelo: ${e instanceof Error ? e.message : String(e)}]`;
    }
  }

  const reply = mergeBlocks(deterministicBlock, llmPart);
  await store.appendMessage(contactId, { role: "assistant", text: reply });
  return reply;
}
