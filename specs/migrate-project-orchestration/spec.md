---
id: migrate-project-orchestration
status: implemented
hypothesis_outcome: not-applicable
owner: repository-maintainers
created: 2026-11-06
updated: 2026-11-06
---

# Governança do monorepo para trabalho por agentes

## Estado atual e evidência

O monorepo já separa produção e análise em `apps`, `services`, `pipelines` e
`analytics`. O `README.md` apresenta o produto, `docs/README.md` é o índice
operacional, `docs/architecture.md` registra fronteiras e o Makefile raiz
orquestra os componentes.

A inspeção de 2026-11-06 encontrou:

- um guia raiz curto, sem precedência, workflow de specs, mapa de fontes de
  verdade, efeitos externos ou definition of done;
- um `AGENTS.md` no runtime que mistura regras locais de benchmark com regras
  genéricas de delegação;
- nenhum pack de specs, registro local de skills ou teste de governança;
- um workflow sob `apps/whatsapp-adapter/.github/workflows`, path que o GitHub
  não descobre a partir da raiz, configurado com Node 18 embora o pacote exija
  Node 20.19 ou superior;
- dados, sessões, pesos e resultados adequadamente locais em grande parte, mas
  artefatos de apresentação e diretórios temporários sem regra explícita no
  ignore raiz;
- trabalho preexistente não relacionado em notebooks, documentação e código de
  observabilidade, além de outputs locais não rastreados.

### Mapa atual

| Área | Responsabilidade | Consumidores | Efeitos externos possíveis |
|---|---|---|---|
| `apps/whatsapp-adapter` | canal, allowlist e histórico por contato | usuários do WhatsApp | sessão e mensagens reais |
| `services/inference-runtime` | servir e comparar runtimes | adapter e analytics | GPU/Pod, downloads e processos remotos |
| `pipelines/qwen-quantization` | construir e verificar artefatos | runtime | GPU, dados privados e Hugging Face |
| `analytics` | datasets, avaliação e análise | mantenedores e estudos | APIs, BigQuery/OpenAI/Ollama conforme fluxo |
| `docs` | contratos atuais, runbooks e histórico marcado | todos os mantenedores | nenhum por leitura; comandos podem ter efeitos |
| Makefile e `scripts` raiz | orquestração entre áreas | operadores locais | depende do alvo escolhido |

### Matriz de verdade

| Assunto | Fonte atual | Fonte dona resultante | Versão | Ação |
|---|---|---|---|---|
| comportamento | código, configuração e testes do componente | os mesmos arquivos | Git | manter |
| arquitetura | `docs/architecture.md` e convenções implícitas | `docs/architecture.md` | Git | consolidar |
| intenção de mudança | chat, commits e docs ad hoc | `specs/<id>/spec.md` | Git | criar |
| operação atual | `docs/README.md`, guias e READMEs de componente | os mesmos, roteados pelo guia central | Git | manter |
| regras para agentes | guia raiz e guia do runtime | raiz + guia mais próximo | Git | consolidar |
| experimento e evidência | componente, notebook e outputs locais | spec para intenção/outcome; componente para reprodução; outputs locais | mista | consolidar |
| histórico | `docs/archive` | `docs/archive` não normativo | Git | historizar |

### Mapa do legado

| Peça | Dono final | Ação | Risco controlado |
|---|---|---|---|
| `README.md` e `docs/README.md` | navegação humana e operação | manter e completar links | duplicação de regras |
| `docs/architecture.md` | fronteiras estáveis | consolidar dependências e contratos | drift com código |
| `AGENTS.md` raiz | kernel operacional | consolidar | instruções concorrentes |
| `services/inference-runtime/AGENTS.md` | riscos do benchmark | restringir ao path | repetição do kernel |
| workflow aninhado do adapter | `.github/workflows/ci.yml` | mover e corrigir | CI antes inerte |
| `docs/archive` | contexto consultivo | manter historizado | conteúdo stale parecer normativo |
| dados, sessões, pesos e resultados | armazenamento local/externo autorizado | manter local | exposição no Git |
| artefatos de apresentação temporários | workspace local | preservar e ignorar | perda de trabalho do usuário |
| investigações em analytics/runtime/pipeline | componente proprietário + spec | manter ownership; não criar árvore artificial | fragmentação por extensão |

## Objetivo observável

Um mantenedor ou agente parte da raiz e consegue identificar produto, estado,
fontes donas, instruções aplicáveis, contrato de mudança, efeitos externos e
gates. A estrutura é validada por um comando local e por CI sem depender de
credenciais, dados privados ou execução remota.

## Escopo

Incluído:

- kernel de governança e precedência;
- specs duráveis e templates do pack local;
- uma skill reutilizável para decidir operações remotas;
- guias escopados somente nas fronteiras com lifecycle ou risco próprio;
- navegação e arquitetura reconciliadas;
- política explícita para dados, outputs e histórico;
- testes de governança e workflow de CI descobrível.

Fora de escopo:

- regra de negócio, prompt, modelo, schema ou contrato de API;
- execução no WhatsApp, UFF, RunPod, Hugging Face, OpenAI ou BigQuery;
- reorganização do código produtivo ou criação retroativa de specs;
- edição, movimento ou remoção do trabalho preexistente identificado;
- validação remota das alegações operacionais documentadas.

## Topologia e fontes de verdade

```text
README.md                         entrada humana e estado verificável
AGENTS.md                         kernel operacional e precedência
docs/architecture.md              fronteiras e contratos estáveis
docs/README.md + docs/guias/       operação atual
specs/<mudança>/spec.md            intenção e aceite duráveis
.agents/skills/<skill>/SKILL.md    lente reutilizável de decisão
tests/governance/                  propriedades estruturais
<área>/AGENTS.md                   complemento local quando muda o risco
```

O guia mais próximo complementa o raiz e nunca substitui fontes de produto.
Código e testes continuam donos de comportamento. Specs não são runbooks;
runbooks não aprovam intenção; histórico não define estado atual.

## Compatibilidade, rollout e remoção

A migração é aditiva, exceto pelo movimento do workflow inerte para a raiz e
pela consolidação do guia do runtime. Não há mudança de import, layout de dados
ou comportamento de produto. O gate entra primeiro vermelho, passa após cada
fatia e é incorporado a `make test` e `make verify`.

Nenhum documento histórico ou output local será removido. Compatibilidade de
comandos existentes é mantida; o novo alvo `make test-governance` é aditivo.

## Dados, segurança e efeitos externos

Os testes usam somente biblioteca padrão e inspeção Git local. Não acessam rede,
serviços, dados privados ou credenciais. A migração não executa targets remotos.
Sessões, contatos, chaves, datasets privados, pesos, resultados brutos e
artefatos temporários permanecem fora do Git.

## Decisões aprovadas

1. Preservar as quatro fronteiras atuais; governança é uma camada sobre elas.
2. Versionar apenas `spec.md` por mudança; pack temporal fica local por padrão.
3. Manter status de execução separado de outcome de hipótese.
4. Não criar `experiments/`: a unidade real de ownership é o componente.
5. Criar uma única skill local, focada em decisão e evidência de operações
   remotas, sem duplicar placement ou runbooks.
6. Ativar CI apenas para governança e para a verificação já existente do
   adapter; suítes Python pesadas permanecem nos gates locais dos componentes.
7. Preservar integralmente o working tree preexistente.

## Riscos

| Risco | Mitigação | Residual |
|---|---|---|
| gate rígido impedir evolução legítima | validar propriedades sem snapshot da árvore inteira | novos tipos de fronteira exigem atualizar o gate |
| documentação duplicar contrato | matriz de verdade e links para a fonte dona | revisão humana ainda necessária |
| workflow corrigido revelar falhas antigas | job isolado e comandos existentes | primeira execução remota não ocorre nesta migração |
| outputs locais entrarem no diff | ignore explícito e teste com `git ls-files` | arquivos já rastreados exigiriam decisão própria |
| conflito com trabalho em andamento | paths preexistentes modificados ficam intocados | integração posterior pode exigir reconciliação manual |

## Critérios de aceite

- [x] `python3 -m unittest discover -s tests/governance -p 'test_*.py' -v` passa.
- [x] `make test-governance` passa e participa de `make test`.
- [x] todo link local nos artefatos de governança resolve.
- [x] specs e skills possuem metadata válida e registro único.
- [x] cada fronteira de risco tem guia local complementar, sem repetir o kernel.
- [x] workflow vive na raiz e usa a versão de Node suportada pelo adapter.
- [x] dados sensíveis, pesos, resultados e artefatos temporários não são rastreados.
- [x] testes focados dos documentos/Make/CI passam sem rede ou efeito externo.
- [x] o diff final não altera os 11 arquivos preexistentes identificados.
