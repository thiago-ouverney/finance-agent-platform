# Guia de migracao de contexto da outra IA

Este guia existe para transformar o conhecimento que esta em outro chat de IA em dados estruturados que o agente consegue usar automaticamente.

## Objetivo

Popular os arquivos em `data/contacts/<ID_ALLOWLIST>/context/` com informacoes suficientes para:

- personalizar a analise financeira por pessoa;
- aplicar regras de categorizacao consistentes;
- ajustar tom e formato da comunicacao;
- respeitar ciclo de fatura e janela de controle mensal;
- usar metas realistas no orcamento.

## Fontes para extrair do outro chat de IA

Antes de preencher os templates, extraia do outro chat:

1. Perfil e contexto de vida
2. Regras de classificacao de gastos
3. Preferencia de comunicacao
4. Datas financeiras (vencimento/fechamento/janela)
5. Metas e limites numericos
6. Excecoes importantes (ex.: PIX interno, gastos recorrentes)

## Mapeamento: conhecimento -> arquivo

### 1) Perfil da pessoa -> `profile.md`

Preencher:

- nome
- ocupacao
- renda_mensal
- composicao_familiar
- prioridades_financeiras
- restricoes_relevantes
- tolerancia_risco
- habitos_sazonais
- observacoes

Boas praticas:

- use valores objetivos quando existir numero (ex.: `renda_mensal: 12500`);
- se houver incerteza, marque como "estimado".

### 2) Regras gerais de analise -> `rules.md`

Preencher:

- regras globais de interpretacao do extrato;
- como tratar casos ambiguos;
- criterios para considerar despesa como essencial vs estilo de vida.

Exemplos de regra:

- "pix entre contas proprias deve ser nao consumo"
- "reembolso de empresa nao entra como renda recorrente"

### 3) Regras customizaveis por padrao -> `rules_custom.csv`

Colunas:

- `pattern`: texto a procurar na descricao
- `categoria`: categoria final
- `kind`: `consumo` ou `financeiro_nao_consumo`
- `prioridade`: inteiro (100+ para regras fortes)
- `observacao`: justificativa curta

Quando usar:

- quando a regra depende de texto recorrente da fatura;
- quando a regra precisa sobrescrever a classificacao padrao.

### 4) Estilo da resposta -> `communication.md`

Preencher:

- tom (direto, acolhedor, tecnico etc.)
- nivel_detalhe (curto, medio, profundo)
- formato_pos_print (resumo + bullets + proximo passo)
- tamanho_resposta (curta/media/longa)
- termos_a_evitar
- como_cobrar_proximo_passo

Dica:

- use instrucoes objetivas do tipo "sempre terminar com uma pergunta de acao".

### 5) Datas de ciclo financeiro -> `billing_cycle.csv`

Campos obrigatorios:

- `fatura_vencimento_dia`
- `fatura_fechamento_dia`
- `controle_mensal_inicio_dia`
- `controle_mensal_fim_dia`

Regras:

- dias devem ser numericos (1-31);
- manter coerencia com a rotina real da pessoa.

### 6) Metas financeiras -> `goals.csv`

Minimo obrigatorio:

- linha `tipo=investimento` com `meta_mensal`
- linha `tipo=reserva_emergencia` com `meta_mensal`

Recomendado:

- metas por categoria (`tipo=categoria`) para os principais grupos de gasto.

## Critico para nao quebrar o fluxo

- remover `PREENCHER` dos campos obrigatorios;
- manter nomes de colunas dos CSVs;
- nao apagar arquivos de contexto;
- manter valores monetarios em formato simples (ex.: `1500` ou `1500,00`).

## Checklist de conclusao da migracao

- [ ] `profile.md` sem placeholders
- [ ] `communication.md` sem placeholders
- [ ] `rules.md` sem placeholders
- [ ] `rules_custom.csv` com ao menos 1 regra valida
- [ ] `billing_cycle.csv` com os 4 campos obrigatorios preenchidos
- [ ] `goals.csv` com investimento e reserva preenchidos

## Consolidado de gastos para auditoria e outra IA

O sistema gera automaticamente:

- `data/contacts/<ID_ALLOWLIST>/invoices/consolidated_expenses.csv`

Esse arquivo consolida os ultimos meses em uma visao unica por linha de gasto.

### Colunas do consolidado

- `expense_id` (identificador estavel da linha)
- `source_file` (arquivo de origem)
- `source_row_index` (indice da linha no arquivo)
- `uploaded_at` (data ISO do upload)
- `transaction_date` (data da transacao quando detectada)
- `invoice_month` (mes de referencia)
- `descricao`
- `valor`
- `categoria`
- `kind`

### Regra de preenchimento para outra IA

- **Rotina normal:** a outra IA **nao precisa preencher** esse arquivo; ela so le o consolidado para analisar.
- **Migracao manual de historico antigo:** se necessario, use `consolidated_expenses.template.csv` como base e gere um CSV nesse mesmo formato.

## Exemplo rapido de consolidacao a partir do outro chat

Se no outro chat a IA dizia:

- "A pessoa quer reduzir ifood e assinatura"
- "Prefere respostas curtas e diretas"
- "Fecha fatura dia 3 e vence dia 10"
- "Meta: investir 1500 e reserva 800"

Entao voce deve refletir isso nos arquivos:

- `rules.md` e `rules_custom.csv` (ifood/assinaturas)
- `communication.md` (resposta curta e direta)
- `billing_cycle.csv` (3 e 10)
- `goals.csv` (1500 e 800)
