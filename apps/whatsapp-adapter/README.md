# whatsapp-finance-agent

Agente conversacional de planejamento financeiro via WhatsApp usando [Baileys](https://github.com/WhiskeySockets/Baileys). O acesso e o modelo por número são definidos em **`DATA_DIR/policy.sqlite`** (com seed a partir de `ALLOWED_CONTACTS` na primeira execução) e podem ser editados pelo **painel admin** local.

Cada número autorizado ganha um contexto próprio em `DATA_DIR/contacts/<id>/`, com templates preenchíveis de perfil, regras de categorização, comunicação, ciclo de fatura e metas.

## Pré-requisitos

- **Node.js 18.18+** (recomendado **20+**) para rodar o bot e os testes — o `better-sqlite3` (política em `policy.sqlite`) é um **addon nativo do Node** e [ainda não é suportado pelo Bun](https://github.com/oven-sh/bun/issues/4290); use `npm run dev` (tsx + Node), não `npm run dev:bun`.
- Em **Node 18**, o Baileys precisa de `globalThis.crypto` (Web Crypto); o projeto carrega [`src/bootstrap-webcrypto.ts`](src/bootstrap-webcrypto.ts) no arranque e no Vitest.
- [Bun](https://bun.sh) pode ser usado só para `bun install`, se preferir.
- Após **trocar a versão do Node**, rode `npm install` de novo (o `postinstall` executa `npm rebuild better-sqlite3`) para evitar erro de `NODE_MODULE_VERSION`.

### Se o computador ficou lento ou travou

Em falhas **repetidas** de conexão (ex.: `crypto is not defined` antes do polyfill), o processo podia **reconectar em loop** e acumular recursos. Hoje há **teto de reconexões** e encerramento explícito do socket antigo em [`src/whatsapp/client.ts`](src/whatsapp/client.ts). Se ainda notar lentidão: mate o processo (`Ctrl+C` ou `kill`), corrija a causa (Node 20 LTS, `npm install` no mesmo Node) e suba de novo.

### Diagnóstico rápido (sem WhatsApp)

```bash
npm run doctor
```

Valida Node, Web Crypto, `better-sqlite3` e `loadEnv()` sem abrir WebSocket. Útil para anexar saída em relatórios em vez de só o stack trace.

## Configuração

1. Copie o exemplo de ambiente:

   ```bash
   cp .env.example .env
   ```

2. Edite `.env`:

   - `ALLOWED_CONTACTS`: em **`POLICY_SOURCE=hybrid`** (padrão), lista de números **somente dígitos** usada como fallback quando ainda não há linha em `policy.sqlite`. Em **`POLICY_SOURCE=db_only`**, a allowlist vem só do banco (após seed ou cadastro no admin).
   - `OPENAI_API_KEY`: opcional. **Sem chave**, o bot funciona em **modo determinístico**: analisa CSV localmente, comandos `/help`, `/orcamento`, etc. **Com chave**, o modelo enriquece a resposta; os números calculados localmente continuam aparecendo quando há CSV/orçamento.
   - `POLICY_SOURCE`, `POLICY_DB_PATH`, `POLICY_CACHE_TTL_MS`: ver `.env.example`.
   - `ADMIN_TOKEN` (opcional): se definido, sobe um painel em `http://ADMIN_HOST:ADMIN_PORT/` (padrão `127.0.0.1:3847`) com `Authorization: Bearer <token>` para CRUD de identidades, modelo por número, features e textos de negação.

3. Instale dependências:

   ```bash
   bun install
   # ou: npm install
   ```

4. Com **Node.js**, variáveis em `.env` são carregadas via pacote `dotenv` no bootstrap (`src/index.ts`). Com **Bun** como runtime, o `.env` pode ser injetado automaticamente, mas o processo principal do agente deve ser **Node**.

## Execução

```bash
npm run dev
```

O script `dev` usa **Node + `tsx --watch`** (TypeScript sem build prévio). O comando `npm run dev:bun` existe, mas **falha** hoje por causa do `better-sqlite3` no Bun.

Escaneie o QR Code no terminal na primeira conexão. A sessão fica em `.baileys_auth/` (não versionar).

## Uso

### Modo sem IA (`OPENAI_API_KEY` vazio)

- Cole ou envie **CSV** com colunas de descrição e valor (ex.: `descricao,valor`).
- Use os **comandos** abaixo (sempre começam com `/`).
- O agente continua funcionando em modo local, mas exige contexto mínimo preenchido para respostas analíticas precisas.

### Com IA

- Mensagens livres passam pelo modelo configurado (`OPENAI_MODEL`, padrão `gpt-4o-mini`).
- Quando houver análise local (CSV/orçamento), o texto inclui primeiro o bloco numérico e depois a parte do modelo.
- O contexto do contato (templates) é injetado no prompt para personalizar regras, metas e estilo de comunicação.

### Comandos

| Comando | Descrição |
| ------- | --------- |
| `/help` ou `/ajuda` | Lista comandos e dicas |
| `/status` | Indica se a IA está ativa, quantas faturas há salvas e se o contexto do contato está completo |
| `/categorias` | Lista categorias da classificação automática |
| `/orcamento renda=12000 fixos=4701` | Estimativa usando faturas já enviadas para este contato |
| `/limpar` | Avisa que vai apagar dados; confirme com `/limpar sim` |

Também é aceito texto livre com `renda 12000 fixos 4701` (com histórico de CSV).

### Contexto por contato (obrigatório para análise final)

No primeiro uso de um contato autorizado, o bot cria automaticamente:

- `data/contacts/<id>/context/profile.md`
- `data/contacts/<id>/context/rules.md`
- `data/contacts/<id>/context/rules_custom.csv`
- `data/contacts/<id>/context/communication.md`
- `data/contacts/<id>/context/billing_cycle.csv`
- `data/contacts/<id>/context/goals.csv`

Se houver campos pendentes (`PREENCHER`), o agente retorna checklist de lacunas e o caminho da pasta para completar.

### Artefatos por arquivo CSV enviado

Cada arquivo CSV recebido gera artefatos por contato:

- CSV bruto em `data/contacts/<id>/invoices/raw/<YYYY-MM>/...csv`
- Parse JSON em `data/contacts/<id>/invoices/parsed/<YYYY-MM>/...json`
- Parse CSV normalizado em `data/contacts/<id>/invoices/parsed/<YYYY-MM>/...csv`
- Resumo markdown em `data/contacts/<id>/invoices/parsed/<YYYY-MM>/...md`
- Índice em `data/contacts/<id>/artifacts/index.csv`
- Consolidado cronológico em `data/contacts/<id>/invoices/consolidated_expenses.csv`

### Consolidado de gastos (ledger único)

Após cada CSV processado, o bot reconstrói `consolidated_expenses.csv` com colunas:

- `expense_id`
- `source_file`
- `source_row_index`
- `uploaded_at`
- `transaction_date` (quando detectável no CSV original)
- `invoice_month`
- `descricao`
- `valor`
- `categoria`
- `kind`

`transaction_date` tenta mapear chaves comuns (`data`, `date`, `dt`, `data_compra`, `data_movimento`, etc.). Se não houver data parseável, o campo fica vazio e o fallback de período é `invoice_month` + `uploaded_at`.

### Limitação LID

Se o WhatsApp entregar apenas um identificador `@lid` sem `senderPn`, o número pode não bater com `ALLOWED_CONTACTS`. Os logs do servidor mostram os candidatos testados; nesse caso use o número completo em dígitos na lista ou verifique se o cliente WhatsApp envia o telefone associado.

## Scripts

| Script              | Descrição                                           |
| ------------------- | --------------------------------------------------- |
| `npm run dev`       | Dev com reload (**Node + tsx**; necessário para `better-sqlite3`) |
| `npm run dev:tsx`   | Igual ao `dev` (alias) |
| `npm run dev:bun`   | Bun + watch (não use: SQLite nativo ainda não suportado no Bun) |
| `bun run build` | Compila TS |
| `bun run start` | Roda `dist/index.js` |
| `bun run test` | Testes Vitest |
| `bun run lint` | ESLint |
| `npm run check:native` | Smoke test do `better-sqlite3` (falha cedo se ABI do Node não bater) |
| `npm run doctor` | Diagnóstico: Node + Web Crypto + SQLite nativo + `loadEnv()` (sem Baileys) |
| `npm run verify` | `lint` + `build` + `test` (o `npm test` já roda `check:native` antes do Vitest) |

O `vitest` usa `src/vitest.setup.ts` para **não depender do seu `.env`** (chave OpenAI, allowlist, tetos de LLM, etc.); os testes usam `testEnvConfig()` com valores fixos.

### Erro `NODE_MODULE_VERSION` / `better_sqlite3.node`

O `better-sqlite3` é **nativo**: o binário em `node_modules` tem de ser compilado para **a mesma versão do Node** com que você roda `npm test`.

Checklist:

1. Confira com `node -v`.
2. **Não copie** a pasta `node_modules` entre computadores ou entre versões diferentes do Node.
3. Após `nvm use`, upgrade do Node ou pull que mudou dependências: `rm -rf node_modules && npm install` (o `postinstall` roda `npm rebuild better-sqlite3`).
4. Rode `npm run check:native` — se falhar, siga a mensagem na tela.
5. Em Linux, se o rebuild compilar do zero, instale toolchain típica (`build-essential`, `python3`).

No CI (GitHub Actions), cada job faz `npm ci` e `npm run verify` em Node 18 e 20.

## Skills do Cursor

Em [`.cursor/skills/`](.cursor/skills/) há orientações para evolução do projeto (`financial-planner-whatsapp`, `baileys-whatsapp-agent`).

## Aviso

Integrações não oficiais ao WhatsApp podem violar os termos de uso e implicar banimento. Use por sua conta e risco.
