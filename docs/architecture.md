# Arquitetura

Este documento fixa responsabilidades, contratos públicos e direção de
dependências entre `pipelines`, `services`, `analytics` e `apps`. Operação
temporal pertence ao [guia central](README.md); intenção de mudança pertence a
[specs](../specs/README.md).

## Contexto e fluxo principal

```text
checkpoint + dataset privado
        |
        v
pipelines/qwen-quantization -- artefato + manifest/hash --> runtime
                                                          /       \
                                   OpenAI-compatible API /         \ resultados/manifests
                                                        v           v
                                           apps/whatsapp-adapter   analytics
                                                        ^           |
                                                        |           |
                                             WhatsApp/contato   workloads versionados
                                                                    |
                                                                    +----> runtime
```

O fluxo de artefatos pode ser bidirecional entre runtime e analytics: analytics
produz workloads e o runtime devolve resultados. Isso é troca por contratos de
arquivo, não licença para imports cruzados ou dependência em estado local.

## Responsabilidades

| Área | É dona de | Não é dona de |
|---|---|---|
| `pipelines/qwen-quantization` | calibração, build, integridade e proveniência do modelo | serving, canal ou análise de runtime |
| `services/inference-runtime` | lifecycle, requests, métricas e resultados dos runtimes | contato, lógica do canal ou avaliação analítica duplicada |
| `analytics` | datasets, workloads, métricas de qualidade, consolidação e análise | servir modelo ou medir novamente sinais do runtime |
| `apps/whatsapp-adapter` | allowlist, sessão, conversa por contato e provider de chat | modelo, benchmark ou operação do runtime |
| Makefile e `scripts` raiz | orquestração entre componentes | regra de negócio interna de um componente |

## Direção das dependências

- Código de `apps`, `services`, `pipelines` e `analytics` permanece importável e
  testável dentro da própria fronteira.
- Tooling raiz pode invocar componentes e transferir artefatos entre contratos;
  componentes não importam tooling raiz, notebooks, `docs/archive`, specs ou
  outputs ignorados como dependência produtiva.
- O adapter depende somente do contrato OpenAI-compatible do provider, não da
  implementação de Ollama, llama.cpp ou vLLM.
- O runtime consome modelo/tokenizer e workload identificados; não depende do
  notebook que os analisará.
- Analytics consome manifests/resultados e produz novos artefatos por interfaces
  explícitas; não lê diretórios internos ou memória do processo de benchmark.
- Promoção de um protótipo exige código na camada dona, consumidor real,
  contrato, testes e observabilidade. Resultado aceito não vira produção por
  movimento de arquivo.

Dependência nova entre fronteiras ou mudança de schema/protocolo exige spec,
owner único do contrato e migração dos consumidores na mesma entrega ou em
rollout compatível documentado.

## Contratos entre componentes

| Contrato | Produtor | Consumidor | Identidade mínima |
|---|---|---|---|
| artefato quantizado | pipeline ou origem externa aprovada | runtime | formato, revisão/digest, SHA-256, tokenizer e manifest |
| API de chat | runtime OpenAI-compatible | adapter | base URL local, modelo/alias e limites documentados |
| workload | analytics ou fixtures versionadas | runtime | schema, split, seed, prompt/taxonomia e hashes |
| resultado de benchmark | runtime | analytics | run id, configuração, requests, completude e proveniência |
| avaliação/golden | analytics | quantização e avaliação | versão, split, política de labels e hash |

Mudanças nesses contratos atualizam código, exemplos de ambiente, testes e a
documentação canônica correspondente. Um índice pode resumir estado; o manifest
ou arquivo produtor continua sendo a fonte dos dados da execução.

## Dados, outputs e ambientes

- Pesos, resultados brutos, dados privados, sessões, contatos, secrets e
  ambientes virtuais não pertencem ao Git.
- Fixtures versionadas são mínimas e sanitizadas. Artefatos grandes usam storage
  próprio e identidade imutável.
- Relatório gerado é output: deve apontar para inputs, gerador e versão. A fonte
  é corrigida e o output regenerado; não se edita o relatório como contrato.
- Máquina local, UFF e RunPod são ambientes operacionais distintos. Código local
  não prova o estado de nenhum deles.
- Benchmarks formais são sequenciais. Concorrência tem workload, hipótese e spec
  próprios.

## Fontes arquiteturais e evolução

Este arquivo é dono das fronteiras. [`AGENTS.md`](../AGENTS.md) define como
alterá-las; [`specs/README.md`](../specs/README.md) define como aprovar intenção;
[`docs/README.md`](README.md) roteia a operação corrente. Documentos em `archive`
preservam contexto, mas não alteram o contrato acima.

Os históricos dos três repositórios anteriores foram importados por `git
subtree`, sem squash. Preserve a história em movimentos e evite misturar mudança
de layout com regra de produto ou reformat massivo.
