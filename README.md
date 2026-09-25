# Finance Agent Platform

Monorepo da plataforma de agente financeiro via WhatsApp, inferência local e pipeline de quantização do Qwen.

## Componentes

- `apps/whatsapp-adapter`: integração com WhatsApp e lógica do agente financeiro.
- `services/inference-runtime`: benchmark e operação dos runtimes locais de inferência.
- `pipelines/qwen-quantization`: quantização, empacotamento e avaliação do Qwen.
- `analytics`: dataset agregado, tags BMC, notebook e comparação de modelos preditivos.

Os repositórios anteriores permanecem disponíveis apenas como legado. O desenvolvimento integrado deve continuar neste repositório.

Consulte [a arquitetura](docs/architecture.md) para o fluxo entre os componentes.

## Comandos da raiz

```bash
make setup
make install
make test
make build
make verify
make setup-notebook
make benchmark-dataset RESULTS_DIR=/caminho/para/results
make benchmark-tags
make notebook
```

`make setup` instala o adaptador e cria `analytics/.venv` com JupyterLab e o kernel `finance-agent-analytics`. Se quiser preparar somente o ambiente analítico, use `make setup-notebook`.

Cada componente mantém também seus próprios comandos e documentação.

Use Node.js 20.19 ou superior no adaptador WhatsApp (`.nvmrc`).

Para executar o mesmo benchmark localmente, na UFF ou no RunPod, consulte [execução remota](docs/remote-execution.md).

## Segurança

Não versione `.env`, credenciais do WhatsApp, tokens, modelos, ambientes virtuais nem resultados brutos. Use os arquivos `.env.example` de cada componente como referência.
