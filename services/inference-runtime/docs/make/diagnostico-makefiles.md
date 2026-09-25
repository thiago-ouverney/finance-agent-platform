# Diagnóstico dos Makefiles no RunPod

> Estado atualizado em 22/09/2026: a validação GPU dos três runtimes foi concluída. Veja o [relatório completo da rodada](../benchmark/resultados-pod-20260922.md). Os caminhos `results/<timestamp>/...` na matriz abaixo são artefatos históricos da execução anterior ao novo layout; novas execuções usam `results/<runtime>/<timestamp>/<nome-humano>/{html,json,csv,logs}`.

O projeto executável fica em `/workspace/llm_local_inference_masters` no pod e em `llm_local_inference_masters` neste workspace. O checkout remoto estava atrás das alterações locais; a sincronização preservou um backup remoto antes de aplicar os arquivos.

## Causas confirmadas

- O wrapper `vllm` usa `/app/.vllm_venv/bin/python`; o teste anterior usava outro Python, que não contém `vllm`. A verificação do plugin deve usar o ambiente do runtime.
- `llama-server` existe em `/usr/local/lib/ollama/llama-server`, fora do PATH. Sem `GGML_BACKEND_PATH` apontando ao arquivo `cuda_v12/libggml-cuda.so`, `--list-devices` não reconhecia CUDA. O Make detecta os caminhos disponíveis e mantém a exigência de GPU.
- A bateria base `bench-vllm` ignorava `VLLM_EXTRA_ARGS`, apesar de o smoke e o sweep usarem esses argumentos. O smoke também ignorava `BENCH_STARTUP_TIMEOUT`. Ambos corrigidos.
- Os wrappers Nsight são arquivos `100644`: invocá-los diretamente causava `Permission denied`. O Makefile de profiling agora usa Bash, cria a pasta de saída e encontra os executáveis/backends da imagem.
- O Makefile de profiling fixava o venv em outro checkout. Agora o caminho é relativo ao próprio Makefile.
- GNU Make 3.81 do macOS não suporta `.ONESHELL`: variáveis desapareciam entre linhas e o profiling podia produzir `/trace` ou `/roofline`. Os Makefiles agora recusam explicitamente essa versão. Use GNU Make 3.82+ ou o GNU Make 4.3 do pod.
- Cada smoke repetia instalação e downloads. `PREPARE_OFFLINE=1` permite reutilizar a preparação existente, mantendo validações de imports, tokenizer, JSONs e assinatura/SHA-256 do GGUF.

## Comandos reproduzíveis

Após preparar uma vez as dependências e o GGUF 7B no SSD:

```bash
cd /workspace/llm_local_inference_masters
make check-runtimes
make check-vllm-gguf
make smoke-llama PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
make smoke-vllm PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
make smoke-ollama PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Os smokes executam três medições do cenário curto, além da primeira requisição e do aquecimento. Não são resultados da bateria formal. Execute os runtimes sequencialmente e mantenha livres as portas 8000, 8080 e 11434 antes de lançar cada processo.

## Cobertura

| Fluxo | Nível | Resultado |
|---|---|---|
| `check-vllm-gguf` | Runtime real | Passou no Python do vLLM |
| `smoke-llama` | GPU real, 7B | Passou; resultado final em `results/20260922T151943.141369Z` |
| `smoke-vllm` | GPU real, 7B | Passou com configuração diagnóstica; `results/20260922T152227.971684Z` |
| `smoke-ollama` | GPU real, 7B | Passou; `results/20260922T152059.788856Z` |
| `quick-sweep-*` | GPU real, 7B | Um ponto de 1024 tokens passou nos três runtimes |
| `bench-all` reduzido | GPU real, 7B | `vLLM=0 llama.cpp=0 Ollama=0` |
| Preparador Ollama | Testes locais com daemon HTTP de teste | 3 testes passaram, incluindo preservação de daemon existente |
| `Makefile.profiling check` | Ambiente real | Passou |
| Wrapper NCU | Kernel CUDA mínimo real | Bloqueado por `ERR_NVGPUCTRPERM`: contadores GPU exigem configuração do host RunPod |
| Nsight Systems | Inspeção real | `nsys` não instalado na imagem |
| Make macOS 3.81 | Execução real | Rejeitado com instrução clara; não executa receitas incompatíveis |

Dry-runs e mocks verificam a orquestração; os resultados acima demonstram inferência real. O erro NCU é uma restrição de permissão do host; mudar o Makefile ou executar como root dentro do pod não habilita esses contadores.

Não executar como smoke: `upload-hf` publica externamente; `quantize-q8`/`build-llama` e a bateria completa demandam preparação/carga longa. O modelo 14B não estava presente no SSD. Nenhum peso adicional foi baixado para validar esses caminhos.
