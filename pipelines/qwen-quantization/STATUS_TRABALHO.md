# Referência do trabalho — quantização e inferência de LLMs

> Documento de alinhamento do grupo e base para a apresentação.
> Estado registrado em **22/09/2026** a partir das execuções e decisões compartilhadas no grupo.

## 1. Resumo executivo

### O problema

Queremos executar um LLM do tipo *instruct* para um **chatbot comercial** em uma única GPU, equilibrando:

- qualidade das respostas;
- uso de memória;
- latência para iniciar a resposta;
- velocidade de geração.

### O que já concluímos

1. Quantizamos Qwen2.5 com diferentes técnicas e precisões.
2. Comparamos a qualidade das quantizações principalmente com MMLU-Redux.
3. Identificamos que as versões de 8 bits preservaram melhor a qualidade.
4. Criamos um ambiente RunPod com `llama.cpp`, Ollama e vLLM.
5. Geramos um GGUF real do Qwen2.5-14B Q8_0 para Ollama e `llama.cpp`.
6. Adotamos um benchmark externo comum, via HTTP, para comparar os runtimes.
7. Validamos o fluxo do benchmark em `llama.cpp`; Ollama e vLLM também já foram exercitados, mas a matriz final ainda não está completa.

### Onde estamos agora

Estamos executando os testes finais na RTX 3090. A matriz de comparação foi fechada:

- **Qwen2.5-7B Q8_0 GGUF:** vLLM, `llama.cpp` e Ollama;
- **Qwen2.5-14B Q8_0 GGUF:** `llama.cpp` e Ollama;
- **Qwen2.5-14B no vLLM:** não será incluído nas métricas comparativas; as falhas de carregamento serão apresentadas como limitação de compatibilidade/formato.

Arthur publicou o GGUF real do 7B em `arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`. O bloqueio atual passou a ser operacional: concluir as execuções, preservar as configurações e consolidar os resultados.

---

## 2. O que a matéria pede

O trabalho atual não deve apenas mostrar qual runtime foi mais rápido. Precisamos explicar **por que** cada resultado ocorreu, relacionando-o a:

- VRAM, RAM, armazenamento e movimentação de dados;
- largura de banda e comunicação CPU–GPU;
- gargalos *memory-bound*, I/O-bound e *compute-bound*;
- fases de prefill e decode;
- KV cache, paginação e fragmentação de memória;
- formato do modelo e suporte de cada runtime;
- parâmetros de execução, como contexto, batch e concorrência.

O professor também pediu coerência experimental: as métricas usadas no relatório final devem ser produzidas no **mesmo hardware e sob condições comparáveis**.

### Caso de uso e metas iniciais

| Item | Definição |
|---|---|
| Caso de uso | Chatbot comercial |
| Hardware | RunPod com NVIDIA RTX 3090, 24 GB de VRAM |
| Meta de TTFT | <= 500 ms |
| Meta de geração | >= 50 tokens/s |
| Meta aproximada de TPOT | <= 20 ms/token |

Essas metas são hipóteses iniciais. Devem ser validadas à luz dos primeiros resultados e do comportamento esperado para um chatbot.

---

## 3. Racional e sequência do trabalho

### Fase 1 — reduzir memória sem perder qualidade

**Pergunta:** qual quantização permite executar o modelo com menor memória, preservando sua capacidade?

1. Validamos o processo com o Qwen2.5-0.5B-Instruct.
2. Adotamos o Qwen2.5-14B-Instruct como modelo principal e fizemos experimentos complementares com o 7B.
3. Testamos GPTQ, AWQ, EXL3 e RTN/GGUF, principalmente em 4 e 8 bits.
4. Para técnicas que exigem calibração, usamos `Brench/MMLU-Pro-CoT-Train-84K`, com seed 42.
5. Avaliamos qualidade principalmente com MMLU-Redux.
6. Como o benchmark completo excedia o tempo disponível, adotamos uma amostra de aproximadamente 5%, usando a mesma seed.
7. Medimos perplexidade no WikiText-2, com janela 4096 e stride 512.
8. Registramos também tamanho, TTFT e tokens/s para parte dos modelos.
9. Publicamos os modelos no Hugging Face e consolidamos os resultados do primeiro trabalho.
10. Apresentamos essa etapa em 01/09.

**Conclusão:** no recorte usado do MMLU-Redux, as versões de 8 bits preservaram melhor a qualidade. No 14B, GPTQ 8-bit e EXL3 8-bit alcançaram aproximadamente 81,24%, enquanto RTN/GGUF 8-bit ficou em aproximadamente 81,20%.

> Esses resultados orientam a escolha de precisão, mas não podem ser usados diretamente para comparar os runtimes: foram coletados em hardwares e condições diferentes.

### Fase 2 — comparar runtimes de inferência

**Pergunta:** fixado um modelo quantizado, como o runtime e suas configurações alteram latência, vazão e memória?

1. Escolhemos uma RTX 3090 de 24 GB no RunPod como hardware comum.
2. Dividimos os runtimes:
   - `llama.cpp` — Arthur;
   - Ollama — Thiago;
   - vLLM — Yan.
3. Arthur criou uma imagem e um template RunPod com os três runtimes.
4. Descobrimos que os antigos repositórios chamados `GGUF-4bit` e `GGUF-8bit` continham Safetensors, não arquivos `.gguf` reais.
5. Para Ollama e `llama.cpp`, voltamos ao modelo-base FP16, convertemos para GGUF e quantizamos com `llama.cpp`.
6. Arthur publicou o `Qwen2.5-14B-Instruct-Q8_0.gguf` real.
7. Rejeitamos o uso exclusivo dos benchmarks nativos, pois cada runtime mede de forma diferente.
8. Adotamos o benchmark comum do repositório `llm_local_inference_masters`, acessando cada runtime pela API HTTP.
9. Ajustamos o benchmark para separar cenários por tamanho de entrada e permitir múltiplas repetições.
10. Validamos o fluxo com `llama.cpp` e iniciamos as execuções nos demais runtimes.

---

## 4. Decisões tomadas e justificativas

| Decisão | Racional | Estado |
|---|---|---|
| Usar modelo *instruct* | Representa o caso de uso de assistente/chatbot | Fechada |
| Usar RTX 3090 de 24 GB | Disponível no RunPod, custo menor e memória suficiente para os testes planejados | Fechada |
| Priorizar 8 bits | Melhor preservação de qualidade na primeira etapa | Fechada |
| Comparar `llama.cpp`, Ollama e vLLM | Representam runtimes populares com arquiteturas e suportes distintos | Fechada |
| Usar benchmark externo comum | Evita comparar métricas produzidas por metodologias diferentes | Fechada |
| Separar resultados por tamanho de entrada | Prefill e uso de KV cache mudam com o contexto; uma agregação única esconderia isso | Fechada |
| Fixar o mesmo hardware | Elimina diferenças de GPU como explicação dos resultados | Fechada |
| Usar 7B Q8 nos três runtimes | É o artefato comum que cabe e permite a comparação principal | Fechada |
| Usar 14B Q8 em Ollama e `llama.cpp` | Preserva o modelo principal da primeira etapa onde há suporte | Fechada |
| Não substituir silenciosamente o 14B no vLLM | A falha de compatibilidade/OOM é um resultado técnico relevante | Fechada |

---

## 5. Artefatos já gerados

### Neste repositório

| Artefato | Finalidade |
|---|---|
| `quantize.py` e `quantize-qwen2.5.sh` | Quantização dos modelos |
| `benchmark.py` | Benchmark de qualidade |
| `performance.py` | Perplexidade e métricas básicas de geração |
| `Dockerfile` | Ambiente da etapa de quantização |
| `runpod-runtime/Dockerfile` | Ambiente RunPod com os três runtimes |
| `runpod-runtime/README.md` | Conversão, quantização e uso dos runtimes |

### Externos

| Artefato | Situação |
|---|---|
| Modelos quantizados no Hugging Face do Arthur | Publicados |
| `Qwen2.5-14B-Instruct-Q8_0.gguf` | GGUF real publicado |
| `Qwen2.5-7B-Instruct-Q8_0.gguf` | GGUF real publicado em 21/09 |
| Template RunPod com os três runtimes | Publicado e utilizado |
| `yanwerneck/llm_local_inference_masters` | Benchmark comum disponível e ajustado |
| Planilha/documentos da primeira etapa | Resultados parciais consolidados |

### Atenção à nomenclatura

Os repositórios antigos do Arthur chamados `Qwen2.5-*-Instruct-GGUF-*bit` armazenam Safetensors. O nome do repositório não prova que exista um arquivo `.gguf`. Para os testes finais, devemos registrar o **arquivo exato**, URL, hash, formato e tamanho.

---

## 6. Status atual por runtime

Legenda:

- **Confirmado:** execução ou artefato demonstrado no grupo.
- **Relatado:** execução informada, mas ainda sem configuração e resultado consolidados.
- **Pendente:** necessário para fechar a comparação.

| Runtime | Modelo | Estado | Evidência atual | Próxima ação |
|---|---|---|---|---|
| `llama.cpp` | 0.5B GGUF | **Confirmado** | Servidor e benchmark comum executados localmente | Repetir com o modelo final na RTX 3090 |
| `llama.cpp` | 7B e 14B Q8_0 GGUF | **Em execução** | Arthur iniciou a preparação/execução do 7B | Executar os dois modelos e salvar resultados completos |
| Ollama | 7B e 14B Q8_0 GGUF | **Pendente** | Runtime e fluxo do benchmark já foram validados | Executar os dois modelos com o benchmark padronizado |
| vLLM | 14B Q8 GGUF | **Falhou** | Carregamento atingiu OOM; suporte GGUF é experimental | Não usar como comparação final sem resolver o carregamento |
| vLLM | 14B Q8 Safetensors | **Falhou** | Incompatibilidade envolvendo `lm_head` | Investigar apenas se o grupo mantiver o 14B |
| vLLM | 7B Q8_0 GGUF | **Em execução** | Modelo carregou e Yan iniciou a bateria do vLLM | Concluir e salvar configuração/resultados |

### Leitura do estado atual

A infraestrutura e o escopo estão definidos. O trabalho agora é concluir a matriz, verificar a validade de cada execução e consolidar as métricas.

---

## 7. Matriz final de experimentos

| Modelo | vLLM | `llama.cpp` | Ollama | Objetivo |
|---|---:|---:|---:|---|
| Qwen2.5-7B Q8_0 GGUF | Sim | Sim | Sim | Comparação principal e direta entre os três runtimes |
| Qwen2.5-14B Q8_0 GGUF | Não | Sim | Sim | Comparar o impacto do runtime em um modelo maior |

As tentativas do 14B no vLLM devem aparecer na apresentação como diagnóstico:

- Safetensors: incompatibilidade entre o checkpoint quantizado e o loader esperado pelo vLLM (`lm_head`);
- GGUF: suporte experimental e OOM durante o carregamento na RTX 3090 de 24 GB.

Não devemos substituir o artefato do vLLM por outro modelo e misturar o resultado na mesma tabela como se a comparação fosse equivalente.

---

## 8. Protocolo experimental a congelar

Antes da coleta final, precisamos registrar uma configuração única:

| Dimensão | Valor/decisão |
|---|---|
| GPU | RTX 3090, 24 GB |
| Modelo e arquivo | 7B Q8_0 nos três; 14B Q8_0 em Ollama e `llama.cpp` |
| Quantização | Q8 |
| Cenários de entrada | Short, medium e long; atualmente 256, 2048 e 8192 tokens |
| Tokens de saída | 128 no benchmark atual |
| Temperatura e sampling | `temperature=0`, `top_p=1` |
| Warm-ups | 3 por cenário/repetição |
| Requisições | 50 por cenário/repetição, conforme o fluxo atualizado do grupo |
| Repetições por cenário | 3 |
| Concorrência | 1; concorrência pertence ao próximo trabalho |
| Limite de contexto | Deve comportar entrada + saída do maior cenário e ser igual entre runtimes |
| Seed | 42 no benchmark atual |
| Monitoramento | GPU, VRAM, CPU e RAM |

Além dos parâmetros comuns, cada execução deve registrar configurações específicas do runtime, como batch, número de camadas na GPU, tipo/tamanho do KV cache e opções de paralelismo.

### Procedimento por execução

1. Iniciar o Pod limpo e registrar GPU, imagem e versões.
2. Subir apenas um runtime e um modelo.
3. Confirmar que não existem processos concorrentes consumindo VRAM.
4. Registrar tempo e pico de memória durante o carregamento.
5. Fazer os warm-ups definidos.
6. Executar as repetições de cada cenário.
7. Salvar configuração, logs, resultados brutos e monitoramento.
8. Encerrar o runtime antes de iniciar o próximo.

### Setup de um Pod novo

O fluxo abaixo atende tanto o template completo quanto um Pod básico. Para a responsabilidade do Thiago, apenas o Ollama é obrigatório; vLLM e `llama-server` podem permanecer ausentes. Não instalar dependências do benchmark dentro de ambientes de outros runtimes.

#### 1. Validar GPU, espaço e ferramentas

```bash
nvidia-smi
df -h /workspace
git --version
python3 --version
make --version
ollama --version 2>&1 || true
command -v ollama
command -v llama-server
command -v vllm
```

Se Git, Python/venv ou ferramentas básicas estiverem ausentes:

```bash
apt-get update
apt-get install -y git curl ca-certificates zstd build-essential cmake pkg-config \
  python3 python3-venv python3-pip iproute2 procps tmux
```

Reservar espaço suficiente para os GGUFs 7B e 14B, os blobs importados pelo Ollama, tokenizers, ambiente Python e resultados. Verificar novamente o disco antes de preparar o 14B.

Se `command -v ollama` não retornar um caminho, instalar a distribuição oficial:

```bash
curl -fsSL https://ollama.com/install.sh | sh
hash -r
command -v ollama
ollama --version 2>&1 || true
```

Em containers RunPod sem `systemd`, um aviso sobre serviço pode ser ignorado. Não iniciar um serviço permanente: o benchmark executará `ollama serve` diretamente e controlará seu ciclo de vida.

#### 2. Colocar dados persistentes em `/workspace`

```bash
mkdir -p /workspace/models /workspace/ollama-models /workspace/.cache/huggingface

export HF_HOME=/workspace/.cache/huggingface
export OLLAMA_MODELS=/workspace/ollama-models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
unset OLLAMA_KEEP_ALIVE OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE
```

As variáveis exportadas precisam continuar definidas no terminal que executará o Make. Para o baseline, manter permanência do modelo, Flash Attention e tipo do KV cache nos defaults; as variações serão rodadas depois e identificadas separadamente.

#### 3. Clonar a revisão mais recente

```bash
cd /workspace
git clone https://github.com/yanwerneck/llm_local_inference_masters.git
cd /workspace/llm_local_inference_masters
git rev-parse --short HEAD
make help
```

Se o diretório já existir, usar `git pull --ff-only origin main` em vez de clonar novamente. Após as melhorias de 22/09, o HEAD verificado era `5662c30`; uma revisão posterior é aceitável, mas todos devem registrar o commit efetivamente usado.

#### 4. Garantir que a porta do Ollama esteja livre

```bash
ss -ltnp | grep ':11434' || true
ps -ef | grep '[o]llama serve' || true
```

O benchmark inicia e encerra o próprio `ollama serve`. Se houver outro daemon, identificar seu PID e encerrá-lo conscientemente antes do smoke. Não iniciar `ollama serve` manualmente durante a medição.

#### 5. Preparar e validar o 7B

```bash
make prepare-ollama MODEL_SIZE=7B
```

O alvo por runtime chama automaticamente toda a preparação comum: baixa GGUF e tokenizer, cria `.venv`, valida assinatura/SHA-256, verifica o executável e importa o GGUF no Ollama com o alias `qwen7b-q8-gguf`. Não é mais necessário chamar `prepare-benchmark` separadamente. O preparador pode encerrar o daemon temporário ao terminar, portanto `ollama list` fora de um servidor ativo pode falhar normalmente. O smoke test é a validação correta do alias e da API.

#### 6. Rodar smoke e integração rápida

```bash
make smoke-ollama MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  BENCH_STARTUP_TIMEOUT=300

make quick-sweep-ollama MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  BENCH_STARTUP_TIMEOUT=300
```

Só avançar se os dois comandos terminarem sem erro. Conferir `server.log`, `manifest.json`, `lifecycle.json` e se não há requisições incompletas.

Na revisão `5662c30`, o alvo quick sweep usa `SWEEP_START=1024`, `SWEEP_STEP=2048` e `SWEEP_MAX_CONTEXT=8192`, percorrendo 1024, 3072, 5120 e 7168 tokens. Isso diverge de uma página que diz “1024 e 2048”. Para uma validação mínima e inequívoca de um ponto, usar:

```bash
make quick-sweep-ollama MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  SWEEP_MAX_CONTEXT=1024 BENCH_STARTUP_TIMEOUT=300
```

#### 7. Rodar a bateria formal do 7B

```bash
make bench-ollama MODEL_SIZE=7B PREPARE_OFFLINE=1
```

Esse alvo já executa:

- short, medium e long;
- 50 requisições, 3 repetições e 3 aquecimentos;
- telemetria de GPU/CPU/RAM/SSD;
- sweep de contexto/KV de 1024 em 1024, até 16384 ou até a primeira falha.

Ele pode demorar bastante; executar em sessão persistente (`tmux`) se o terminal SSH puder cair.

#### 8. Repetir para o 14B

```bash
make prepare-ollama MODEL_SIZE=14B

make smoke-ollama MODEL_SIZE=14B PREPARE_OFFLINE=1 \
  BENCH_STARTUP_TIMEOUT=600

make quick-sweep-ollama MODEL_SIZE=14B PREPARE_OFFLINE=1 \
  BENCH_STARTUP_TIMEOUT=600

make bench-ollama MODEL_SIZE=14B PREPARE_OFFLINE=1
```

Se o 14B falhar por memória em algum contexto, preservar `server.log` e o ponto da falha. Não reduzir silenciosamente o modelo nem classificar execução parcial como sucesso.

#### 9. Registrar a execução

```bash
git rev-parse HEAD
ollama --version
nvidia-smi
sha256sum /workspace/models/Qwen2.5-7B-Instruct-Q8_0/Qwen2.5-7B-Instruct-Q8_0.gguf
sha256sum /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
```

Preservar toda a pasta `results/`. As novas execuções ficam em `results/ollama/<timestamp>/<nome-humano>/`, separadas em `html/`, `json/`, `csv/`, `logs/` e `text/`. Antes de enviar ao Drive, verificar se não há tokens, chaves ou outros segredos nos logs.

### Execuções do Thiago — Ollama

Thiago deve executar **os dois modelos**, sempre no mesmo tipo de Pod:

1. **7B Q8_0:** entra na comparação principal entre os três runtimes.
2. **14B Q8_0:** entra na comparação adicional entre Ollama e `llama.cpp`.

Seguir o setup acima na ordem **7B smoke → 7B quick sweep → 7B formal → 14B smoke → 14B quick sweep → 14B formal**. Não executar `make bench`, pois esse alias tenta os três runtimes; para a responsabilidade do Thiago, usar `make bench-ollama`.

Como o `Makefile` recebeu várias correções, conferir `make help`, o README da revisão clonada e registrar o commit. Não usar uma cópia local desatualizada como fonte final.

Antes da bateria completa:

- executar o smoke test disponibilizado pelo projeto;
- durante a inferência, conferir `ollama ps` em outro terminal e garantir `100% GPU`;
- confirmar que nenhum outro processo usa a GPU;
- conferir se o cenário long cabe no contexto configurado;
- verificar que cada bloco terminou com 50 sucessos e sem erros/incompletas.

### Parâmetros a testar no Ollama

Primeiro executar um **baseline sem tuning**. Depois alterar apenas um fator por vez:

| Experimento | Valores | Hipótese avaliada |
|---|---|---|
| Tamanho do modelo | 7B e 14B, ambos Q8_0 | Efeito do número de parâmetros sobre memória e desempenho |
| Contexto (`num_ctx`/`OLLAMA_CONTEXT_LENGTH`) | valor comum suficiente para cada cenário; testar uma grade controlada se houver tempo | Efeito do contexto e do KV cache sobre TTFT e VRAM |
| Flash Attention (`OLLAMA_FLASH_ATTENTION`) | `0` e `1` | Efeito sobre memória e prefill, especialmente em contexto longo |
| Tipo do KV cache (`OLLAMA_KV_CACHE_TYPE`) | `f16` e `q8_0`; `q4_0` apenas exploratório | Troca entre memória, velocidade e potencial perda de qualidade |
| Estado do modelo | primeira resposta após iniciar versus modelo aquecido | Separar custo de carregamento da operação normal |

Configurações que devem permanecer fixas nesta entrega:

- `OLLAMA_NUM_PARALLEL=1`;
- um único modelo carregado por vez;
- `temperature=0` e `top_p=1`;
- mesmos prompts, seed, tokens de saída, warm-ups e repetições;
- modelo mantido em GPU durante os blocos aquecidos;
- nenhum download durante a medição.

Para o cenário long, um contexto nominal de 8192 pode não comportar 8192 tokens de entrada mais template e 128 tokens de saída. O grupo deve alinhar um limite comum maior — por exemplo, 16384 — ou reduzir o cenário; não basta alterar apenas o JSON do cliente, pois o servidor Ollama precisa ser iniciado com o mesmo limite.

O teste mínimo recomendável é:

1. baseline dos dois modelos;
2. Flash Attention desligado versus ligado;
3. KV cache `f16` versus `q8_0`, mantendo Flash Attention ligado;
4. comparação cold/warm já capturada pelo fluxo de lifecycle.

Não fazer uma busca combinatória. Isso permite explicar causalmente cada mudança e mantém o custo do Pod controlado.

---

## 9. Métricas: o que existe e o que falta

### Resultados já existentes

| Métrica | Situação | Uso correto |
|---|---|---|
| MMLU-Redux | Gerado para as quantizações | Comparar preservação de qualidade |
| Perplexidade no WikiText-2 | Gerada para parte dos modelos | Comparar degradação do modelo |
| Tamanho do modelo | Gerado | Comparar redução de armazenamento/memória |
| TTFT e tokens/s antigos | Parciais e em hardwares diferentes | Referência exploratória, não comparação final |

### Métricas obrigatórias ainda não consolidadas nos três runtimes

| Métrica | O que responde |
|---|---|
| TTFT — média, mediana, p95 e p99 | Quanto o usuário espera até a primeira resposta |
| TPOT/latência entre tokens | Regularidade e velocidade do decode |
| Tokens/s de geração | Velocidade percebida após o primeiro token |
| Throughput de prefill | Capacidade de processar o prompt de entrada |
| Latência total | Tempo completo da requisição |
| Tempo de carregamento | Custo para disponibilizar o modelo |
| Pico e série temporal de VRAM | Pressão de memória e efeito do KV cache |
| RAM e offload CPU/GPU | Custo de não manter tudo na GPU |
| Utilização de GPU e CPU | Indício do recurso limitante |
| Requisições concluídas/erros | Estabilidade da configuração |

Todas devem ser separadas por tamanho de entrada e acompanhadas da configuração que as produziu.

### Análises que ainda precisamos construir

- comparar prefill e decode separadamente;
- identificar gargalo memory-, I/O- ou compute-bound;
- relacionar crescimento do contexto e KV cache ao consumo de VRAM;
- explicar paginação, fragmentação e offload quando ocorrerem;
- mostrar como alterações de batch, contexto e cache mudam o resultado;
- verificar se TTFT <= 500 ms e geração >= 50 tokens/s são alcançáveis;
- explicar diferenças arquiteturais entre os runtimes observadas na prática.

---

## 10. Plano de execução

### Bloqueadores imediatos

- [x] Publicar um GGUF Q8_0 real do 7B.
- [x] Definir a matriz 7B/14B por runtime.
- [ ] Confirmar no repositório atualizado os comandos `make` após as correções de 22/09.
- [ ] Registrar SHA-256 dos GGUFs, versões dos runtimes e configuração efetivamente usada.

### Coleta final

- [ ] Arthur: executar o modelo escolhido no `llama.cpp` na RTX 3090.
- [ ] Thiago: executar o 7B Q8_0 e o 14B Q8_0 no Ollama na RTX 3090.
- [ ] Yan: reproduzir a execução no vLLM e salvar configuração/resultados.
- [ ] Todos: armazenar resultados brutos em um local comum, com nomes padronizados.
- [ ] Todos: repetir testes após variar os parâmetros escolhidos para análise.

### Consolidação

- [ ] Montar tabela única de configuração e versões.
- [ ] Consolidar métricas por runtime e cenário.
- [ ] Produzir gráficos de latência, vazão e memória.
- [ ] Interpretar os gargalos, não apenas ranquear os runtimes.
- [ ] Validar conclusões contra o caso de uso do chatbot.

---

## 11. Roteiro sugerido para a apresentação

1. **Problema:** servir um chatbot comercial em uma única GPU.
2. **Restrição:** RTX 3090 com 24 GB e metas de experiência do usuário.
3. **Aprendizado da etapa 1:** 8 bits preservou melhor a qualidade; quantizar envolve compatibilidade de formato e runtime.
4. **Decisão experimental:** modelo Q8, três runtimes, mesmo hardware e benchmark comum.
5. **Metodologia:** mesmos cenários, repetições e parâmetros; separação de prefill e decode.
6. **Resultados:** TTFT, TPOT/tokens/s, carregamento, VRAM/RAM e utilização.
7. **Explicação:** arquitetura do runtime, KV cache, contexto, batch e gargalo observado.
8. **Decisão para o caso de uso:** runtime/configuração recomendados e limitações.

Evitar uma revisão genérica das técnicas. Cada conceito deve aparecer para explicar uma decisão ou um resultado observado.

---

## 12. Critério de conclusão

O trabalho estará pronto quando tivermos, para cada runtime:

- o mesmo modelo — ou uma diferença de formato claramente justificada;
- execução reproduzível na mesma RTX 3090;
- versões, comandos e configurações registrados;
- resultados brutos preservados;
- métricas de latência, vazão e memória por cenário;
- ao menos uma análise de impacto das configurações;
- explicação técnica dos gargalos;
- recomendação conectada ao chatbot comercial.

Até que esses pontos estejam completos, os resultados atuais devem ser tratados como **exploratórios**, não como a comparação final.
