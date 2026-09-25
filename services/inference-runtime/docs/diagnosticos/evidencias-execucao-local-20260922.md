# Evidências da execução local — 2026-09-22

Escopo: validação local curta do plano de Makefiles. Nenhum runtime GPU foi iniciado e nenhuma dependência foi instalada.

## Estado preservado

- `git status --short` e `git diff --stat` foram registrados antes dos testes.
- O checkout já continha alterações rastreadas e arquivos novos; não houve `reset`, `checkout`, `clean` ou descarte de mudanças.
- `scripts/build_code_reference.py` foi executado com sucesso e regenerou `docs/engenharia/codigo-fontes.md` a partir dos módulos atuais.

## Passou

- `python3 -m unittest tests.test_makefiles -v`: 3 testes passaram; 4 foram pulados porque o GNU Make local é 3.81 e não oferece `.ONESHELL`.
- `python3 -m unittest tests.test_prepare_ollama -v`: 3 testes passaram, cobrindo daemon próprio, daemon preexistente e GGUF ausente.
- `python3 -m py_compile bench.py lifecycle.py scripts/run_kv_sweep.py scripts/prepare_ollama.py`.
- `git diff --check`.
- `python3 scripts/build_code_reference.py`.
- `test_make_requires_oneshell_on_old_versions`: passou; `make help` e `make clean-info` recusaram corretamente o Make 3.81.
- Wrappers `nsys`/`ncu` via mocks: passaram nos testes de argumentos e diretório de saída.

## Bloqueios locais de infraestrutura

- GNU Make local: `3.81`, sem `.ONESHELL`; os alvos reais precisam ser executados no pod com GNU Make compatível.
- Não existe `.venv/bin/python` neste checkout local.
- `python3 -m unittest discover -s tests -p 'test_*.py' -v` não completou: faltam `httpx`, `guidellm` e `tokenizers` no Python global. Isso não foi classificado como falha de código.
- `python3 scripts/build_docs.py` não completou porque o módulo Python `markdown` não está instalado. Isso não foi instalado para manter o smoke offline.
- `python3 tests/integration_mock.py` também não completou pela ausência de `tokenizers`.

## Não executado localmente

Smokes GPU, preparação de modelos, sweeps, vLLM/Ollama/llama.cpp reais e profiling dependem do ambiente do pod e permanecem sob responsabilidade da execução remota.
