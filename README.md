# Finance Agent Platform

Monorepo para atendimento financeiro via WhatsApp, inferência local, avaliação
de qualidade/performance e quantização de modelos Qwen.

## Comece aqui

- [Guia central](docs/README.md): escolha e reproduza um fluxo operacional.
- [Arquitetura](docs/architecture.md): entenda fronteiras, contratos e direção
  das dependências.
- [Governança para agentes](AGENTS.md): descubra precedência, workflow, efeitos
  externos e gates.
- [Specs](specs/README.md): proponha mudança não trivial e registre aceite.

Execute `make` ou `make help` para ver os comandos disponíveis. Os READMEs de
cada componente detalham seus pré-requisitos.

## Trilhas e estado verificável

| Trilha | Código e evidência no clone | Limite que exige ambiente externo/local |
|---|---|---|
| WhatsApp | adapter TypeScript, testes e doctor | sessão Baileys, contato autorizado e Ollama da UFF |
| Inferência | lifecycle, benchmarks, observabilidade e testes | runtimes, modelo e GPU/Pod |
| Quantização | scripts, verificação, testes e manifests | modelo, dataset privado, GPU e eventual Hugging Face |
| Analytics | transformações, avaliações, notebooks e testes | dados privados e APIs conforme o fluxo |

Esse quadro descreve o conteúdo do checkout, não o estado atual de UFF, RunPod,
WhatsApp, Hugging Face ou APIs. Consulte e verifique o sistema dono antes de
declarar um estado remoto.

## Componentes

- [`apps/whatsapp-adapter`](apps/whatsapp-adapter/README.md): conversa entre
  WhatsApp e um endpoint OpenAI-compatible, com histórico isolado por contato;
- [`services/inference-runtime`](services/inference-runtime/README.md): serving
  e comparação sequencial de vLLM, llama.cpp e Ollama;
- [`pipelines/qwen-quantization`](pipelines/qwen-quantization/README.md): geração,
  verificação e publicação explícita de artefatos quantizados;
- [`analytics`](analytics/README.md): datasets, avaliações MOPEP e análise dos
  benchmarks;
- [`docs`](docs/README.md): contratos atuais, runbooks e histórico marcado.

## Preparação e validação

```bash
make setup
make test-governance
make test
make build
make verify
```

`make setup` instala dependências, cria `analytics/.venv` e registra um kernel
Jupyter no usuário local; pode acessar a rede e altera o ambiente local. Os
gates de produto exigem as dependências descritas nos componentes.
`make test-governance` usa apenas Python padrão e Git e não acessa serviços.

Alvos como `push-*`, `upload-*`, `*-remote`, túneis, notebooks com APIs e o
adapter em execução podem ler ou gravar sistemas externos. Use o runbook do
fluxo e confirme target, autorização, custo e dados antes de executá-los.

## Dados e artefatos

Credenciais, `.env`, sessões e contatos do WhatsApp, dados privados, pesos,
ambientes virtuais, resultados brutos e outputs temporários ficam fora do Git.
Fixtures versionadas devem ser mínimas e sanitizadas. O contrato completo está
no [guia de governança](AGENTS.md) e nos guias escopados de cada componente.
