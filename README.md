# whatsapp-finance-agent

Agente conversacional de planejamento financeiro via WhatsApp usando [Baileys](https://github.com/WhiskeySockets/Baileys). Responde apenas a números configurados em `ALLOWED_CONTACTS`.

Cada número autorizado ganha um contexto próprio em `DATA_DIR/contacts/<id>/`, com templates preenchíveis de perfil, regras de categorização, comunicação, ciclo de fatura e metas.

## Pré-requisitos

- Node.js 20+ **ou** [Bun](https://bun.sh)

## Configuração

1. Copie o exemplo de ambiente:

   ```bash
   cp .env.example .env
   ```

2. Edite `.env`:

   - `ALLOWED_CONTACTS`: lista de números **somente dígitos** (ex.: código do país + DDD + número), separados por vírgula. O código normaliza JIDs com sufixo de device (`5511...@s.whatsapp.net`).
   - `OPENAI_API_KEY`: opcional. **Sem chave**, o bot funciona em **modo determinístico**: analisa CSV localmente, comandos `/help`, `/orcamento`, etc. **Com chave**, o modelo enriquece a resposta; os números calculados localmente continuam aparecendo quando há CSV/orçamento.

3. Instale dependências:

   ```bash
   bun install
   # ou: npm install
   ```

4. Com **Node.js** (sem Bun), variáveis em `.env` são carregadas via pacote `dotenv` no bootstrap. Com **Bun**, o runtime também pode injetar `.env` automaticamente.

## Execução

```bash
bun run dev
```

O script `dev` usa `bun --watch` para rodar TypeScript nativamente. Alternativa com Node + tsx: `npm run dev:tsx`.

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
| `bun run dev`       | Dev com reload (`bun --watch`)                     |
| `npm run dev:tsx`   | Alternativa com `tsx` (Node.js sem erro Bun/tsx)   |
| `bun run build` | Compila TS |
| `bun run start` | Roda `dist/index.js` |
| `bun run test` | Testes Vitest |
| `bun run lint` | ESLint |

## Skills do Cursor

Em [`.cursor/skills/`](.cursor/skills/) há orientações para evolução do projeto (`financial-planner-whatsapp`, `baileys-whatsapp-agent`).

## Aviso

Integrações não oficiais ao WhatsApp podem violar os termos de uso e implicar banimento. Use por sua conta e risco.
