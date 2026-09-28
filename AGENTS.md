# Instruções do monorepo

- Antes de analisar, planejar ou implementar qualquer alteração, leia `docs/README.md` como guia central do projeto.
- Se uma decisão alterar fluxos, contratos, ambientes, convenções ou qualquer outra informação registrada nesse guia, explicite o impacto e alinhe com o responsável a decisão tomada e se `docs/README.md` será atualizado; não deixe código e guia divergirem silenciosamente.
- Preserve os limites entre `apps`, `services` e `pipelines`.
- Nunca versione credenciais, sessões do WhatsApp, pesos de modelos ou resultados brutos.
- Execute os testes do componente alterado antes de integrar mudanças.
- Alterações em contratos entre componentes devem atualizar a documentação da raiz e os exemplos de ambiente aplicáveis.
- Os benchmarks em `services/inference-runtime` executam uma requisição por vez; concorrência é um experimento separado.

## Operação UFF, Ollama e WhatsApp

- Antes de iniciar, testar ou diagnosticar a conexão do WhatsApp com a UFF, leia e siga `docs/uff-ollama-whatsapp.md`.
- O fluxo operacional atual usa o adapter local diretamente contra a API OpenAI-compatible do Ollama por um túnel `uff-delta`; não substitua Ollama por `llama-server` nem introduza um gateway intermediário sem solicitação explícita.
- Mantenha hostname, IP, usuário, porta SSH, chaves, contatos autorizados e sessões fora da documentação versionada.
