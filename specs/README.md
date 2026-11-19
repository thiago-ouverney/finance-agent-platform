# Specs de mudança

Specs registram a intenção durável de mudanças que atravessam componentes,
alteram contratos, exigem rollout ou têm risco operacional. Código e testes
continuam sendo a fonte do comportamento executável; a spec explica por que a
mudança existe, suas fronteiras e como aceitar o resultado.

## Quando criar

Crie `specs/<id>/spec.md` antes da implementação substancial quando a mudança:

- atravessa `apps`, `services`, `pipelines` ou `analytics`;
- altera contrato público, schema, ambiente, segurança ou operação;
- envolve execução remota, custo relevante, dados privados ou publicação;
- precisa sobreviver ao chat, à branch e à memória de uma pessoa.

Correções locais e mecânicas podem dispensar spec se o comportamento esperado
já estiver definido por código, teste e documentação canônica.

## Contrato versionado e pack local

`spec.md` é autocontido, versionado e a única cópia de decisões duráveis. Use o
conteúdo de [`_template/spec.md`](_template/spec.md) como ponto de partida.

Os demais arquivos do template formam um pack de execução local:

- `implementation-plan.md`: incrementos, superfícies e validações;
- `tasks.md`: checklist temporal;
- `test-plan.md`: cenários e comandos planejados ou executados;
- `acceptance.md`: evidência observada contra os critérios;
- `decisions.md`: alternativas em análise antes de promoção à spec.

Esses arquivos são ignorados por padrão em packs reais. Uma decisão necessária
para um clone novo deve ser promovida para `spec.md`; requisito durável não pode
existir apenas no pack local.

## Ciclo de vida

O frontmatter de cada spec usa um destes estados:

- `draft`: em descoberta, sem autorização para implementação substancial;
- `approved`: intenção, fronteiras e aceite aprovados;
- `implemented`: critérios atendidos e resultado integrado;
- `superseded`: substituída por outra spec indicada no documento;
- `abandoned`: encerrada sem implementação, com motivo registrado.

Para investigações, mantenha `status` separado de `hypothesis_outcome`. Os
outcomes permitidos são `pending`, `accepted`, `rejected` e `inconclusive`;
use `not-applicable` para mudanças que não testam uma hipótese.

## Experimentos

Experimentos permanecem junto da área que possui a hipótese e os consumidores:
analytics, quantização ou runtime. Não crie uma árvore paralela por tipo de
arquivo. A spec é dona de hipótese, baseline, critério de parada, status e
outcome; código reproduzível fica no componente e dados, pesos e resultados
brutos permanecem locais.

Promover um resultado exige consumidor produtivo, contrato público, testes e
observabilidade na camada dona. Um outcome aceito, isoladamente, não transforma
um notebook ou artefato em produção.

