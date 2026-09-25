# Plano de implementação e testes dos Makefiles

## Instrução pronta para o próximo modelo

> Execute este plano usando subagentes para toda inspeção, implementação e teste; o agente coordenador apenas delega, revisa relatos e mantém o usuário informado. Separe um agente para o pod e outro para patches locais/testes, com responsabilidade exclusiva por arquivo. Preserve todas as mudanças existentes. Comece pelo baseline e prossiga por fases; corrija apenas defeitos demonstrados, use testes smoke e não execute bateria completa. Registre evidências e limitações, sem confundir mocks com execução GPU. Não faça commit/push nem upload de pesos.

## Objetivo e estado do handoff

> **Estado final — 22/09/2026:** este plano foi executado no pod. A preparação offline, os três smokes, os três quick sweeps e o `bench-all` reduzido passaram. O agregado terminou com `vLLM=0 llama.cpp=0 Ollama=0`. Os resultados e limitações estão em [resultados-pod-20260922.md](../benchmark/resultados-pod-20260922.md).

Concluir o diagnóstico e a correção dos Makefiles do benchmark no RunPod, usando testes rápidos e preservando as alterações existentes. A execução prevista neste documento foi concluída no pod; os itens que permanecem são operações externas ou experimentos deliberadamente não executados, como upload, build/quantização e profiling bloqueado pelo host.

- Workspace local: `/Users/yanwerneck/projects/master_ai_uff/llm_inference_2026_02`.
- Repositório Git real local: `/Users/yanwerneck/projects/master_ai_uff/llm_inference_2026_02/chatbot-runtime-bench`.
- Repositório no pod: `/workspace/llm_local_inference_masters`.
- Acesso: `ssh -tt -o ConnectTimeout=15 -o BatchMode=yes m87zg5huybxfa7-64410ff1@ssh.runpod.io -i ~/.ssh/id_ed25519`.
- O gateway RunPod exige PTY. A tentativa sem `-tt` retornou `Your SSH client doesn't support PTY`. Não presumir suporte a SCP/SFTP; a sincronização desta sessão utilizou payload pela sessão SSH.
- Pod: GNU Make 4.3; RTX 3090 com cerca de 24 GB de VRAM; GGUF Qwen 7B disponível, 14B ausente.
- Cliente do benchmark: `/workspace/llm_local_inference_masters/.venv/bin/python`. Há um venv antigo em `/workspace/chatbot-runtime-bench/.venv`; não confundir os dois checkouts.
- vLLM: `/usr/local/bin/vllm` é wrapper Bash; ambiente real `/app/.vllm_venv/bin/python`.
- llama.cpp: `/usr/local/lib/ollama/llama-server`; backend `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`. Os plugins v12 e v13 reconheceram a RTX 3090, e o teste de inferência usou v12.
- Modelo: `/workspace/models/Qwen2.5-7B-Instruct-Q8_0/Qwen2.5-7B-Instruct-Q8_0.gguf` (~8,1 GB).
- Tokenizer: `/workspace/models/Qwen2.5-7B-Instruct-original`.

Ler `AGENTS.md` do workspace e do repositório antes de continuar. O usuário prefere subagentes para implementação/testes, com um responsável por cada arquivo. Não fazer commit/push nem publicar pesos, resultados ou credenciais sem instrução específica.

## 1. Preservação e reconciliação antes de editar

1. Executar `git status --short` e `git diff --stat` localmente e no pod.
2. Conferir os arquivos presentes no pod contra os locais e os backups da sessão. Backup remoto disponível em `/tmp/codex-make-backup` (Makefiles, módulos, scripts e configs originais). O pod estava limpo no commit `d5494a3` antes da sincronização, enquanto o checkout local já tinha alterações não commitadas do usuário.
3. Nunca usar `git reset --hard`, `git checkout -- .`, `git clean` ou sincronização com exclusão. Aplicar diferenças incrementais; preservar arquivos de evidência.
4. Os arquivos previamente alterados incluíam `Makefile`, `README.md`, `bench.py`, `lifecycle.py`, `scripts/run_kv_sweep.py`, configs de vLLM/llama.cpp, `docs/make/make-fluxo.md` e `tests/test_lifecycle.py`; `scripts/prepare_ollama.py` e `tests/test_prepare_ollama.py` já existiam como não rastreados. **Não atribuir todo o diff desta sessão ao agente.**
5. Durante a execução inicial, `Makefile`, `Makefile.profiling`, módulos/configs necessários e `docs/make/make-fluxo.md` foram sincronizados ao pod antes dos demais documentos. Essa reconciliação foi concluída depois da validação; README, referências de código, diagnóstico, plano e testes locais foram atualizados no checkout local e estão refletidos neste handoff.
6. Se um patch desta sessão causar regressão, reverter apenas seus trechos após comparar baseline, backup remoto e arquivo atual. Não restaurar cegamente o backup remoto inteiro: isso também retiraria as alterações anteriores do usuário que foram sincronizadas.

Critério: possuir uma lista explícita do que diverge, com backup dos arquivos que serão substituídos, sem apagar alterações ou resultados.

## 2. Causas confirmadas e correções já implementadas

| Problema observado | Estado da correção |
|---|---|
| Verificação de vLLM usava Python sem `vllm` | Alteração local anterior fixa o Python do runtime; arquivo sincronizado e `check-vllm-gguf` passou no pod |
| `llama-server` fora do PATH e CUDA não detectada | Descoberta do binário existente e default do plugin `.so` v12/v13; backend exportado antes dos checks; inferência real passou |
| `bench-vllm` ignorava `VLLM_EXTRA_ARGS` na bateria base | Argumentos JSON agora repassados como no smoke/sweep; alvo validado no `bench-all` reduzido |
| `smoke-vllm` fixava timeout em 1800s | Agora usa `BENCH_STARTUP_TIMEOUT` |
| Todo smoke reinstalava cliente e consultava downloads | `PREPARE_OFFLINE=1` elimina essas etapas; mantém checks de imports, GGUF, tokenizer e JSONs |
| Wrappers Nsight sem bit executável (`100644`) davam `Permission denied` | `Makefile.profiling` chama os wrappers via Bash |
| Profiling fixava Python em outro checkout e não criava pasta final | Python relativo ao Makefile e `mkdir -p` do diretório do relatório |
| Profiling chamava llama-server inexistente no PATH | Usa descoberta de binários e backend como o Makefile principal |
| GNU Make 3.81 do macOS ignorava `.ONESHELL`, perdendo variáveis de shell | Ambos Makefiles agora recusam versão sem feature `oneshell`; evita relatórios incorretos em `/trace` ou `/roofline` |

Arquivos editados nesta sessão: `Makefile`, `Makefile.profiling`, `README.md`, `docs/make/make-fluxo.md`; documentos novos `docs/make/diagnostico-makefiles.md` e este plano; teste novo `tests/test_makefiles.py`. `docs/engenharia/codigo-fontes.md` foi regenerado a partir dos módulos locais, incluindo mudanças anteriores do usuário. HTMLs são gerados por `scripts/build_docs.py`.

## 3. Evidências existentes e limites

| Verificação | Nível | Resultado confirmado |
|---|---|---|
| `check-vllm-gguf` | Ambiente real do pod | Passou |
| `smoke-llama` | GPU real / modelo 7B | Passou: `results/20260922T151943.141369Z` |
| `Makefile.profiling check` | Ambiente real | Passou |
| NCU wrapper com kernel CUDA mínimo compilado por nvcc | GPU real | Falhou com `ERR_NVGPUCTRPERM`; contadores bloqueados pelo host |
| Nsight Systems | Inventário real | `nsys` ausente |
| `tests/test_prepare_ollama.py` | HTTP/processos locais de teste | 3 testes passaram: daemon próprio, preservação de daemon preexistente, GGUF ausente |
| `tests/test_makefiles.py` | Mocks/dry-run/Make real local | 7 testes: 3 passaram e 4 pulados por falta de `.ONESHELL`; os 4 restantes precisam GNU Make do pod |
| Suíte original do pod | Unit tests reais, antes de sincronizar testes novos | 17 passaram em 0,744s; `/tmp/codex-tests-original.log` |
| `check-runtimes` | Binários reais | Passou |
| Receitas smoke/bench/sweep dos três runtimes | Dry-run + `bash -n`, antes da guarda Make3.81 | 9 receitas passaram em sintaxe; isso não comprova GPU |
| `git diff --check` e `py_compile` dos módulos modificados | Estático | Passaram |
| `make help` no macOS 3.81 após a guarda | Execução real | Recusou versão incompatível, como esperado |
| Smoke vLLM | GPU real, configuração diagnóstica | Passou: `results/20260922T152227.971684Z` |
| Smoke Ollama | GPU real / modelo 7B | Passou: `results/20260922T152059.788856Z` |
| Sweeps mínimos | GPU real | Um ponto de 1024 tokens passou nos três runtimes |
| `bench-all` reduzido | GPU real | Status `vLLM=0 llama.cpp=0 Ollama=0` |

Registro histórico: antes da rodada final, um smoke vLLM foi interrompido durante o diagnóstico e usou:

```bash
timeout --signal=TERM --kill-after=20 300 make smoke-vllm PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=180 VLLM_EXTRA_ARGS='--enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.80'
```

Log: `/tmp/codex-smoke-vllm.log`. SIGINT no cliente encerrou a árvore criada; PIDs 43706, 43827 e 44355 ficaram ausentes e a GPU voltou a ~1 MiB. O código 130 do cliente / 2 do make decorre da interrupção solicitada; não diagnosticar como falha do runtime.

Não confundir falha do sandbox de testes localhost com defeito do projeto: a primeira tentativa dos testes Ollama locais foi bloqueada por permissão de socket; a execução permitida passou.

## 4. Inventário dos alvos e cobertura pretendida

| Grupo | Alvos | Plano de validação |
|---|---|---|
| Informação | `help`, `clean-info`, `check-profilers` | Executar; verificar mensagens e ausência de mutações indesejadas |
| Cliente/modelo | `install-benchmark`, `download-model`, `download-tokenizer`, `prepare-benchmark`, `verify-gguf` | Primeiro validar arquivos locais; executar `prepare-benchmark PREPARE_OFFLINE=1`; só instalar/baixar se ausência real justificada |
| Runtimes | `check-runtimes`, `check-vllm-gguf`, `prepare-vllm`, `prepare-llama`, `prepare-ollama` | Checks reais e preparação offline; preservar processos que não foram criados pelo teste |
| Inferência rápida | `smoke-vllm`, `smoke-llama`, `smoke-ollama` | Três smokes reais sequenciais; reaproveitar evidência llama se nenhum código relevante mudou |
| Medição base | `bench-vllm`, `bench-llama`, `bench-ollama` | Opcional: reduzir para cenário short, 1 request, 1 repetição, 1 warmup e 1 ponto do sweep; não executar defaults formais |
| Agregação | `bench-all`, `bench` | Mock controlado para comprovar continuação após falha e status final; evitar triplicar inferências já validadas |
| Sweep | `kv-sweep`, `kv-sweep-vllm`, `kv-sweep-llama`, `kv-sweep-ollama`, `quick-sweep-vllm`, `quick-sweep-llama`, `quick-sweep-ollama` | Um ponto por runtime, `SWEEP_MAX_CONTEXT=1024`; aliases podem ser verificados por plano/mocks |
| Conversão | `check-tools`, `clone-llama`, `build-llama`, `install-llama-python`, `download-source`, `inspect-source`, `quantize-q8` | Inspeção/dry-run e fixtures; sem clone/build/quantização completa para esta validação rápida |
| Publicação/cópia | `upload-hf`, `pull-results` | Não publicar. `pull-results` foi validado estruturalmente; neste proxy RunPod a cópia real foi feita por `runpodctl send/receive` |
| Documentação | `docs` | Regenerar com Python do benchmark; checar fontes e links |
| Profiling | `Makefile.profiling`: `help`, `check`, `profile-vllm-nsys`, `profile-vllm-ncu`, `profile-llama-nsys`, `profile-llama-ncu` | Mocks reais das receitas para argv/pastas; GPU profiling condicionado a ferramentas/permissões do host |

## 5. Sequência de continuação

### Fase 1. Checagem barata do ambiente e dos processos

Na sessão SSH, executar:

```bash
cd /workspace/llm_local_inference_masters
git status --short
git diff --stat
make --version
command -v vllm ollama ncu nsys || true
test -x .venv/bin/python
test -x /app/.vllm_venv/bin/python
test -x /usr/local/lib/ollama/llama-server
test -f /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
nvidia-smi
ss -ltnp | grep -E ':(8000|8080|11434)\b' || true
make help
make clean-info
make check-runtimes
make check-vllm-gguf VLLM_PYTHON=/app/.vllm_venv/bin/python
make check-profilers
make -f Makefile.profiling check
```

Não imprimir `env`, chaves ou tokens. Identificar o dono de uma porta ocupada antes de qualquer ação. Não matar daemons externos só para liberar o smoke.

Aceite: executáveis corretos, GPU disponível, nenhuma execução anterior ainda consumindo recursos, plugin vLLM carregado no ambiente certo.

### Fase 2. Revisão das alterações e regressões rápidas

1. Revisar o diff incremental das correções listadas acima.
2. Conferir `tests/test_makefiles.py`: priorizar comportamento de receitas com mocks, não apenas `assertIn` do texto fonte.
3. Com GNU Make do pod, testar offline inválido, ausência de pip/download no plano offline, transmissão de extra args com espaços, timeout do smoke, profiling com saída em pasta temporária e agregador que continua após falha.
4. Executar:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tests/integration_mock.py
```

Revisar previamente o tempo da integração mock e impor limite externo se necessário. A suíte completa pode revelar falhas das alterações locais anteriores; não mascarar esses erros nem sobrescrever o trabalho do usuário.

Aceite: testes pertinentes passam; caminho `/trace` ou `/roofline` jamais aparece como saída acidental; args preservam aspas/espaços; erro parcial propaga código não zero.

### Fase 3. Preparação offline

```bash
make prepare-benchmark MODEL_SIZE=7B PREPARE_OFFLINE=1
make prepare-llama MODEL_SIZE=7B PREPARE_OFFLINE=1
make prepare-vllm MODEL_SIZE=7B PREPARE_OFFLINE=1 VLLM_PYTHON=/app/.vllm_venv/bin/python
```

O hash de um GGUF de 8 GB ainda lê o SSD; isso é esperado e ocorre fora da medição. `PREPARE_OFFLINE=1` não cria um ambiente ausente. Se imports falharem, reparar só o venv do cliente; jamais instalar GuideLLM dentro do venv do vLLM. Se tokenizer/config não existirem, fazer preparação explícita antes de qualquer medição.

### Fase 4. Smokes reais, um runtime por vez — concluída

Os três smokes finais foram executados no pod, com `PREPARE_OFFLINE=1`, e terminaram com exit 0. Os comandos abaixo permanecem como reprodução documentada; não são pendências desta sessão. O vLLM usou argumentos diagnósticos para caber na RTX 3090, conforme detalhado no [relatório da rodada](../benchmark/resultados-pod-20260922.md):

```bash
make smoke-vllm MODEL_SIZE=7B PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300 VLLM_PYTHON=/app/.vllm_venv/bin/python
make smoke-llama MODEL_SIZE=7B PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=120 LLAMA_BACKEND_PATH=/usr/local/lib/ollama/cuda_v12/libggml-cuda.so
make smoke-ollama MODEL_SIZE=7B PREPARE_OFFLINE=1 BENCH_STARTUP_TIMEOUT=300
```

O smoke tem 3 requisições medidas, 1 repetição e cenário short, mais primeira resposta e 3 aquecimentos. Antes de executar, conferir `WORKLOADS` e limites de geração em `bench.py`; manter o cenário curto e a geração limitada, sem aumentar prompts/tokens para diagnóstico. O sweep explícito de 1024 tokens abaixo fornece o teste maior ainda limitado. `BENCH_STARTUP_TIMEOUT` limita a prontidão do servidor, não o tempo total da execução. Para vLLM, observar `server.log`: carregamento/compilação inicial pode consumir minutos. Se necessário reduzir custo com `VLLM_EXTRA_ARGS='--enforce-eager'`, marcar esse smoke como configuração diagnóstica, não validação da configuração padrão nem resultado comparável à bateria formal.

Aceite por runtime: exit 0, manifest e relatórios completos, resposta do alias correto, GPU efetivamente usada quando exigida, resultados sem erro silencioso e processo criado encerrado. Não tratar memória zero/ausente como medição válida. Em falha, salvar trecho final do log e causa, corrigir apenas a causa demonstrada e repetir só o teste afetado.

### Fase 5. Sweeps de um ponto e ramo formal reduzido — concluída

Os três quick sweeps e o `bench-all` reduzido foram executados depois dos smokes. Os comandos abaixo permanecem como reprodução documentada; os artefatos e métricas efetivamente coletados estão no [relatório da rodada](../benchmark/resultados-pod-20260922.md).

```bash
make quick-sweep-vllm PREPARE_OFFLINE=1 SWEEP_MAX_CONTEXT=1024 BENCH_STARTUP_TIMEOUT=300
make quick-sweep-llama PREPARE_OFFLINE=1 SWEEP_MAX_CONTEXT=1024 BENCH_STARTUP_TIMEOUT=120
make quick-sweep-ollama PREPARE_OFFLINE=1 SWEEP_MAX_CONTEXT=1024 BENCH_STARTUP_TIMEOUT=300
```

O sweep de um ponto usa prompt de 1024 tokens e reserva de geração de 128 tokens no código atual; conferir `scripts/run_kv_sweep.py` antes de executar. Cada quick sweep fixa request/repetition/warmup em 1, mas sem `SWEEP_MAX_CONTEXT=1024` ainda percorreria vários pontos. Inspecionar `sweep-manifest.json`: exatamente um ponto e returncode 0. Verificar que o contexto e os extra args efetivos coincidem com o registrado.

Se for necessário validar o ramo formal alterado de vLLM, em vez de rodar a bateria completa:

```bash
make bench-vllm PREPARE_OFFLINE=1 BENCH_SCENARIOS=short BENCH_REQUESTS=1 BENCH_REPETITIONS=1 BENCH_WARMUP=1 SWEEP_MAX_CONTEXT=1024 SWEEP_REQUESTS=1 SWEEP_REPETITIONS=1 SWEEP_WARMUP=1 BENCH_STARTUP_TIMEOUT=300
```

Esse comando ainda inicia a base e um sweep, portanto custa mais que um smoke. A sessão usou uma variante reduzida em `bench-all`, com `BENCH_WARMUP=0`, um request e uma repetição; não foram executados `make bench` ou `make bench-all` com defaults.

### Fase 6. Profiling: validar o que o ambiente permite

- Nsight Systems está ausente. A instalação é trabalho pendente e opcional; não afirmar que os alvos nsys passaram.
- Nsight Compute 2025.3 existe em `/usr/local/cuda/bin/ncu`, mas um kernel mínimo falhou com `ERR_NVGPUCTRPERM`. Isso exige configuração do host/plataforma, não uma alteração de Makefile ou root dentro do container.
- Manter regressões por mocks para os quatro alvos: resolver binário, backend, argumentos, pasta de saída e código de falha.
- Somente após o host liberar contadores ou a ferramenta estar instalada, iniciar o alvo interativo, enviar uma requisição curta e encerrar graciosamente para finalizar o relatório. Não misturar profiling com latências oficiais.

### Fase 7. Documentação e entrega — concluída

```bash
.venv/bin/python scripts/build_code_reference.py
.venv/bin/python scripts/build_docs.py
git diff --check
```

Conferir `docs/make/make-fluxo.md`, README e diagnóstico. Regenerar HTMLs a partir dos fontes atualizados, reconciliando de volta ao checkout local sem sobrescrever fontes com versões antigas. Essa etapa foi concluída: a matriz final, os caminhos de resultados, os comandos exatos e as limitações estão no [relatório da rodada](../benchmark/resultados-pod-20260922.md).

## 6. Timeouts, interrupção e limpeza

- Não deixar cargas longas sem log/progresso. Observar sessões em intervalos curtos (até 60s) e relatar último estágio.
- Usar `BENCH_STARTUP_TIMEOUT` para readiness; se impor `timeout` externo, reservar margem suficiente para a limpeza do launcher e registrar que o limite é total.
- Interromper apenas o benchmark/processo criado pelo teste, preferindo SIGINT/terminação graciosa. Conferir o grupo de processos e aguardar limpeza antes de novo runtime.
- `lifecycle.py` encerra o grupo que lançou. `prepare_ollama.py` preserva daemon preexistente e encerra apenas o seu próprio daemon temporário. Testar essas garantias e não substituí-las por `pkill` genérico.
- Ao sair, conferir GPU/processos/portas. Preservar resultados, server.log e backups; não apagar pesos ou diretórios de ambiente.

## 7. Critério final de conclusão

Concluir somente quando os três runtimes disponíveis tiverem inferência smoke comprovada, regressões pertinentes passarem e os arquivos locais/remotos estiverem reconciliados. Sweeps relevantes devem ter um ponto válido se forem anunciados como testados. Relatar NCU bloqueado e nsys ausente como limitações externas abertas. Não declarar 14B, quantização, upload, bateria completa ou profiling real validados sem execução correspondente.
