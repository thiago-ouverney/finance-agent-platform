# Instruções do runtime de inferência

Este guia complementa o [`AGENTS.md` raiz](../../AGENTS.md).

## Papel e fronteira

O runtime inicia e compara Ollama, llama.cpp e vLLM, coleta observabilidade e
produz resultados de benchmark. Analytics consome os artefatos; o runtime não
deve implementar uma segunda camada de avaliação MOPEP nem conhecer contatos ou
regras do WhatsApp.

A metodologia é definida em
[`docs/benchmark/metodologia.md`](docs/benchmark/metodologia.md); o layout e os
comandos atuais estão no [`README.md`](README.md).

## Limites

- O benchmark oficial executa uma requisição por vez. Concorrência é outro
  experimento e exige spec e workload próprios.
- Pesos já devem estar no SSD; download, setup e preparação invalidam a janela
  medida quando ocorrem durante ela.
- Meça inicialização, primeira resposta, warmup e operação posterior
  separadamente.
- Diferencie TTFT, tokens/s de decode e tokens/s efetivos e documente fórmulas.
- Diferencie tokens de contexto, KV lógico estimado, ocupação observada do pool
  KV e memória total; proxy não é medição física.
- Dados ausentes não são zero e execução parcial não é sucesso. Cada runtime
  mantém conclusão própria até que identidade e requests sejam comparáveis.
- Pesos, ambientes, resultados, logs, tokens e chaves permanecem locais. Não
  publique branch, artefato ou resultado sem autorização explícita.

## Antes de alterar

Fixe modelo/artefato e hash, tokenizer, hardware, runtime, workload, temperatura,
limite de saída e concorrência. Identifique se a mudança afeta lifecycle,
protocolo de requests, schema de resultado, observabilidade ou análise
downstream; comunique o contrato antes de alterar consumidores.

## Evidência

Com o ambiente preparado, execute:

```bash
make test-inference
```

Mudanças no runtime também atualizam testes e documentação Markdown
proporcionais ao risco. Para campanhas, preserve manifests, requests pareadas,
timestamps, completude e falhas; o handoff distingue smoke, perfil exploratório
e benchmark formal e registra verificações remotas não executadas.
