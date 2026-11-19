# Instruções de documentação

Este guia complementa o [`AGENTS.md` raiz](../AGENTS.md).

## Papel e fronteira

`docs/README.md` roteia a operação atual, `architecture.md` define fronteiras
estáveis, `guias` contém runbooks, documentos temáticos detalham contratos e
`archive` preserva contexto passado não normativo. README navega; documentação
não substitui comportamento executável nem a intenção aprovada em specs.

## Limites

- Atualize documentação canônica na mesma mudança que altera comportamento,
  contrato, ambiente ou convenção.
- Não copie números efêmeros ou estado remoto entre documentos; linke a fonte
  dona e marque evidência temporal com data e contexto.
- Comandos devem existir, declarar pré-requisitos e deixar claro se instalam,
  acessam rede, transferem dados, publicam ou iniciam serviços.
- `docs/archive` pode conter paths e conclusões stale; preserve seu aviso de
  não normatividade e não o use como runbook atual.
- Não versione hostname, IP, usuário, porta, chave, contato, sessão, token, PII,
  peso ou resultado bruto.
- Documento gerado precisa apontar para fonte, gerador e gate de freshness; não
  edite output manualmente para corrigir a fonte.

## Antes de alterar

Classifique o documento como canônico, runbook, evidência de experimento,
histórico ou gerado. Confirme o consumidor e a fonte dona. Se duas páginas
descrevem o mesmo contrato de formas incompatíveis, consolide por referência ou
escale; não escolha silenciosamente.

## Evidência

Execute `make test-governance` para links de governança e valide manualmente os
comandos alterados no nível proporcional ao efeito. Registre comandos não
executados, dependências remotas e data da última evidência em vez de alegar
sucesso. Revise o diff para secrets, PII, IDs e resultados brutos.

