# Benchmark analytics

Este módulo consome os artefatos produzidos por `services/inference-runtime`. Ele não mede o runtime novamente.

## Fluxo

```bash
make setup-notebook
make benchmark-dataset RESULTS_DIR=/caminho/para/results
make benchmark-tags
make notebook
```

Para treinar e comparar Random Forest e Extra Trees na previsão do TTFT p50:

```bash
make predict-benchmark
```

As tags BMC são regras explícitas derivadas de runtime, família do modelo, quantização, contexto, fase, completude e faixa de TTFT. Ajuste os limiares em `src/generate_tags.py` quando a definição formal de BMC do estudo for fechada.

Dados consolidados e modelos treinados são locais e ignorados pelo Git. Resultados brutos continuam sob responsabilidade do benchmark.
