# Instruções do monorepo

Este arquivo define o kernel operacional para trabalho humano e por agentes.
Documentação de produto pertence a `docs`, intenção de mudanças não triviais
pertence a `specs` e comportamento pertence ao código, configuração e testes.

## Precedência e fontes de verdade

Instruções da sessão e do ambiente prevalecem sobre este repositório. Dentro do
repositório, aplique este guia e depois o `AGENTS.md` mais próximo do arquivo
alterado. Guias aninhados complementam o raiz; não podem redefinir arquitetura,
contratos ou políticas de segurança. Se duas fontes normativas divergirem, pare
na fronteira afetada, apresente a evidência e peça a menor decisão necessária.

| Assunto | Fonte dona |
|---|---|
| comportamento executável | código, configuração e testes do componente |
| fronteiras e direção de dependências | [`docs/architecture.md`](docs/architecture.md) |
| operação atual e runbooks | [`docs/README.md`](docs/README.md) e guia do fluxo |
| intenção, escopo e aceite de mudança | [`specs/<id>/spec.md`](specs/README.md) |
| regras para agentes | este arquivo e o guia mais próximo |
| estado/outcome experimental | spec da investigação; evidência no componente dono |
| contexto passado | `docs/archive`, sempre não normativo |

README e índices servem para navegação. Eles apontam para a fonte dona em vez de
copiar contratos. Não infira estado remoto a partir de código, documentação ou
resultados locais.

## Início de sessão

1. Leia [`docs/README.md`](docs/README.md), este guia e os guias aplicáveis ao
   path antes de analisar, planejar ou editar.
2. Inspecione `git status`, alterações preexistentes, callers, testes, ignores e
   histórico relevante. Trabalho de autoria incerta deve ser preservado.
3. Classifique a mudança como local, entre componentes, experimental ou
   operacional e identifique efeitos externos antes de executar comandos.
4. Para operação UFF/Ollama/WhatsApp, leia
   [`docs/uff-ollama-whatsapp.md`](docs/uff-ollama-whatsapp.md) antes de testar
   conexão ou iniciar sessão.

Não altere remote, branch, identidade, credenciais ou ambiente externo sem que
a tarefa exija isso. Nunca versione credenciais, sessões do WhatsApp, contatos,
pesos, dados privados ou resultados brutos.

## Mapa do repositório

| Path | Papel | Guia local |
|---|---|---|
| `apps/whatsapp-adapter` | canal WhatsApp, allowlist e histórico | [`AGENTS.md`](apps/whatsapp-adapter/AGENTS.md) |
| `services/inference-runtime` | serving, benchmark e observabilidade | [`AGENTS.md`](services/inference-runtime/AGENTS.md) |
| `services/inference-gateway` | admissão, fila e passthrough OpenAI-compatible | [`AGENTS.md`](services/inference-gateway/AGENTS.md) |
| `pipelines/qwen-quantization` | construção e verificação de modelos | [`AGENTS.md`](pipelines/qwen-quantization/AGENTS.md) |
| `analytics` | datasets, avaliações e análise | [`AGENTS.md`](analytics/AGENTS.md) |
| `docs` | documentação canônica, runbooks e histórico | [`AGENTS.md`](docs/AGENTS.md) |
| `specs` | contrato durável de mudanças | [`README.md`](specs/README.md) |
| `scripts` e Makefile raiz | tooling e orquestração entre componentes | este guia |
| `tests/governance` | invariantes estruturais do repositório | este guia |

Preserve os limites entre `apps`, `services`, `pipelines` e `analytics`. Tooling
da raiz pode orquestrar componentes; componentes não devem importar tooling,
histórico ou estado local como dependência produtiva. Benchmarks oficiais do
runtime enviam uma requisição por vez; concorrência é uma investigação separada.

## Fluxo de mudanças

1. Resolva fatos pelo repositório antes de perguntar e registre escopo, non-goals
   e consumidores afetados.
2. Crie ou aprove uma spec antes de mudança substancial que atravesse fronteiras,
   altere contrato, ambiente, segurança ou operação. Siga
   [`specs/README.md`](specs/README.md).
3. Escreva primeiro o teste ou checagem que observa a lacuna; confirme que falha
   pelo motivo esperado.
4. Implemente em fatias pequenas. Atualize import, discovery, exemplos de
   ambiente, documentação canônica e testes na mesma fatia quando o contrato
   mudar.
5. Execute o gate focal do componente e depois a suíte proporcional ao risco.
6. Revise o diff completo contra o estado inicial e registre testes não
   executados, riscos e efeitos externos no handoff.

Migração estrutural não autoriza refactor de produto. Não mova ou remova conteúdo
legado sem auditar consumidores, autoria e valor histórico. Arquivos ignorados
não podem carregar a única cópia de requisito necessário a um clone novo.

## Skills locais

- Use `remote-workflow-safety` antes de executar ou propor operações em UFF,
  RunPod, Hugging Face, OpenAI, BigQuery ou WhatsApp quando target, autorização,
  custo, dados e evidência precisarem ser decididos. Fonte:
  [`.agents/skills/remote-workflow-safety/SKILL.md`](.agents/skills/remote-workflow-safety/SKILL.md).

Skills ajudam a tomar decisões recorrentes; não substituem este guia, a
arquitetura ou os runbooks. O guia raiz registra skills e o agente que coordena
a mudança decide quando combiná-las.

## Efeitos externos

| Classe | Exemplos | Regra |
|---|---|---|
| local, somente leitura | busca, diff, teste de links | permitido dentro do escopo |
| local, mutável | instalar dependência, gerar output, iniciar processo | declarar efeito e manter output fora do Git |
| remoto, leitura | listar modelo, consultar API ou estado do Pod | confirmar ambiente, identidade, timeout e custo |
| remoto, escrita | enviar dados, publicar modelo, abrir sessão, mandar mensagem | exige autorização explícita e target exato |

Targets `push-*`, `upload-*`, `*-remote`, túneis, notebooks com API e execução
do adapter podem acessar sistemas externos. Setup também pode instalar pacotes
ou registrar kernel local. Prefira preflight/dry-run quando existir, fixe revisão
imutável e preserve evidência sanitizada. Não troque silenciosamente Ollama por
outro runtime, identidade SSH ou namespace após uma falha.

## Colaboração multiagente

Use trabalho paralelo apenas quando houver investigação independente ou paths
de escrita disjuntos. Um coordenador mantém intenção, spec, decisões, plano e
integração. Cada delegação deve informar objetivo, paths permitidos, non-goals,
skills, contrato compartilhado e evidência esperada.

Um arquivo compartilhado tem um writer e revisores somente leitura. O dono de
interface ou schema comunica o contrato antes que consumidores mudem. Handoffs
distinguem fato, proposta, decisão e bloqueio e listam arquivos, testes,
invalidações downstream, riscos e trabalho restante. O coordenador revisa o diff
integrado e executa os gates completos; delegação não transfere responsabilidade.

## Gates e conclusão

Use o menor conjunto que prove a mudança e amplie conforme o risco:

```bash
make test-governance
make test-analytics
make test-inference
make check-quantization
npm --prefix apps/whatsapp-adapter run verify
make verify
```

Os comandos específicos exigem os ambientes descritos nos READMEs. Não alegue
execução se dependências, dados ou hardware impediram o comando.

Uma mudança está concluída quando:

- teste focal e suíte aplicável passam;
- contratos e documentação canônica concordam com o comportamento;
- links, metadata, ignore e segurança passam nos gates de governança;
- outputs gerados têm fonte e reprodução, ou permanecem explicitamente locais;
- efeitos externos realizados têm target, autorização e evidência; os não
  realizados estão explícitos;
- o diff foi revisto contra o estado inicial e não contém iniciativa fora do
  escopo.
