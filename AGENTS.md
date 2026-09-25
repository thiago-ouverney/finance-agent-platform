# Instruções do monorepo

- Preserve os limites entre `apps`, `services` e `pipelines`.
- Nunca versione credenciais, sessões do WhatsApp, pesos de modelos ou resultados brutos.
- Execute os testes do componente alterado antes de integrar mudanças.
- Alterações em contratos entre componentes devem atualizar a documentação da raiz e os exemplos de ambiente aplicáveis.
- Os benchmarks em `services/inference-runtime` executam uma requisição por vez; concorrência é um experimento separado.
