# Playbook: criar e usar a visao por ID especifico

Este playbook mostra como ativar o contexto de um contato especifico da allow-list e usar no dia a dia.

## 1) Escolher o ID canonico do contato

Use apenas digitos, no formato que esta em `ALLOWED_CONTACTS`.

Exemplo:

- `5511999999999`

## 2) Garantir que o ID esta na allow-list

No `.env`:

```env
ALLOWED_CONTACTS=5511999999999,5511888888888
```

## 3) Criar a estrutura inicial do contato

Opcao pratica: enviar qualquer mensagem desse contato para o bot (ex.: `/status`).

O agente cria automaticamente:

- `data/contacts/5511999999999/context/profile.md`
- `data/contacts/5511999999999/context/rules.md`
- `data/contacts/5511999999999/context/rules_custom.csv`
- `data/contacts/5511999999999/context/communication.md`
- `data/contacts/5511999999999/context/billing_cycle.csv`
- `data/contacts/5511999999999/context/goals.csv`

## 4) Popular com o conhecimento migrado

Use o guia `GUIA_MIGRACAO_CONTEXTO_IA.md` e preencha os 6 arquivos de contexto.

Prioridade minima para desbloquear analise completa:

1. `profile.md`
2. `communication.md`
3. `rules.md`
4. `rules_custom.csv` (pelo menos 1 regra valida)
5. `billing_cycle.csv` (4 campos obrigatorios)
6. `goals.csv` (investimento + reserva)

## 5) Validar se ficou pronto

No WhatsApp, com o proprio contato:

- enviar `/status`

Se ainda faltar algo, o bot vai indicar pendencias e caminho da pasta.

## 6) Usar a visao no fluxo normal

Com contexto pronto, esse ID passa a receber:

- analise CSV com regras customizadas do contato;
- orcamento usando metas do `goals.csv`;
- respostas com estilo definido em `communication.md`;
- bloqueio de ambiguidade reduzido por causa das regras/contexto preenchidos.

## 7) Exemplo de rotina operacional por contato

1. Contato envia CSV da fatura
2. Bot salva artefatos em:
   - `data/contacts/<id>/invoices/raw/...`
   - `data/contacts/<id>/invoices/parsed/...`
   - `data/contacts/<id>/invoices/consolidated_expenses.csv`
   - `data/contacts/<id>/artifacts/index.csv`
3. Contato pede "resuma meu mes"
4. Bot responde usando:
   - dados do CSV atual + historico salvo
   - perfil/regras/comunicacao/metas do proprio ID

## 8) Atualizacao continua do contexto

Quando descobrir nova preferencia no chat:

- atualize diretamente os arquivos de `context/`;
- mantenha versao curta e objetiva das regras;
- evite contradicoes entre `rules.md` e `rules_custom.csv`.

## 9) Problemas comuns

- **Nao responde com analise final**
  - ainda ha `PREENCHER` em arquivo obrigatorio.
- **Classificacao estranha**
  - falta regra no `rules_custom.csv` para descricao recorrente.
- **Orcamento desalinhado com objetivo**
  - metas de investimento/reserva ausentes ou antigas no `goals.csv`.

- **Consolidado sem data de transacao**
  - o CSV original pode nao ter coluna de data reconhecivel; o consolidado cai para `invoice_month` + `uploaded_at`.
