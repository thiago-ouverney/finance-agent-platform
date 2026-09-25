# Finance Agent Platform

Monorepo da plataforma de agente financeiro via WhatsApp, inferência local e pipeline de quantização do Qwen.

## Componentes

- `apps/whatsapp-adapter`: integração com WhatsApp e lógica do agente financeiro.
- `services/inference-runtime`: benchmark e operação dos runtimes locais de inferência.
- `pipelines/qwen-quantization`: quantização, empacotamento e avaliação do Qwen.

Os repositórios anteriores permanecem disponíveis apenas como legado. O desenvolvimento integrado deve continuar neste repositório.

## Segurança

Não versione `.env`, credenciais do WhatsApp, tokens, modelos, ambientes virtuais nem resultados brutos. Use os arquivos `.env.example` de cada componente como referência.
