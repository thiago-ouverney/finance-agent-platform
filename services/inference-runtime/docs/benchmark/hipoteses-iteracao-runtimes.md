# Hipóteses de iteração dos runtimes

> **Objetivo:** registrar, de forma curta, os sinais da rodada observada e os
> próximos testes de configuração na mesma máquina.

## Escopo da análise

A leitura usa as três runs aprovadas pelo gate do notebook
`analytics/notebooks/analyze_runtime_observability.ipynb`, com:

- NVIDIA GeForce RTX 3090;
- mesmo GGUF Q8_0 e tokenizer;
- contexto máximo de 8192 tokens e saída máxima de 128 tokens;
- uma requisição por vez;
- 20 requisições medidas por cenário (`short`, `medium` e `long`).

Os resultados localizam sinais por fase. Eles não distinguem sozinhos limite
computacional de limite de largura de banda e não medem tráfego PCIe.

## Leitura executiva

O primeiro cenário a iterar deve ser o `long` do vLLM. Seu TTFT p95 foi
19,16 s, contra aproximadamente 1,8 s nos outros runtimes. O llama.cpp já foi
o melhor resultado geral e merece apenas uma otimização dirigida ao prefill.
No Ollama, o prefill está próximo do llama.cpp; o espaço observado está no
decode e, secundariamente, no custo de memória do KV cache.

## vLLM: testar a saída do modo eager

### Sinal observado

- TTFT p95 em `long`: 19,16 s.
- Decode p05 em `long`: 65,51 tokens/s.
- GPU e controlador de memória próximos de 100% em `long`.
- CPU do host baixa, sem swap, leitura de disco ou major page faults relevantes.
- VRAM observada de 19,28 GiB, com pico de apenas 3,86% no pool KV.
- Energia mediana de 6,98 kJ por requisição em `long`.

O gargalo está no caminho de execução do GGUF na GPU, não na capacidade do KV
cache, RAM ou disco. A run efetiva usou `--enforce-eager`, além de
`--max-num-seqs 1` e `--gpu-memory-utilization 0.8`.

### Hipótese direta

Remover somente `--enforce-eager` pode habilitar CUDA Graphs e reduzir overhead
de lançamento, principalmente durante o decode. O teste também deve verificar
se o plugin GGUF suporta essa execução e quanto aumentam startup e VRAM.

Tratamentos:

- **A:** configuração atual, com `OBS_VLLM_EXTRA_ARGS='--enforce-eager'`;
- **B:** mesma configuração, com `OBS_VLLM_EXTRA_ARGS=''`.

Manter explicitamente `OBS_VLLM_GPU_MEMORY_UTILIZATION=0.80` nos dois lados.

Aceitar a variante se reduzir TTFT p95 ou aumentar decode p05 em pelo menos
10%, sem falhas e sem regressão relevante de E2E. Se o efeito for pequeno, a
hipótese seguinte deixa de ser configuração: o caminho GGUF/plugin do vLLM
provavelmente é o limite dominante.

## llama.cpp: aumentar o microbatch do prefill

### Sinal observado

- TTFT p95 em `long`: 1,81 s.
- Decode p05 em `long`: 86,13 tokens/s, o melhor da rodada.
- Durante o decode, GPU em 100% e potência próxima de 367 W.
- Durante o prefill `long`, medianas de 33% de GPU e 185 W.
- VRAM observada de 7,88 GiB, deixando ampla margem na RTX 3090.

O decode já está concentrado na GPU. O prefill, porém, mostra utilização bem
menor e pode estar alimentando a GPU em microbatches pequenos. Como o scrape é
aproximadamente de um segundo e o prefill dura menos de dois segundos, esse
sinal precisa ser confirmado pelo A/B.

### Hipótese direta

Aumentar `--ubatch-size` pode criar blocos maiores de trabalho no prefill,
elevar a ocupação da GPU e reduzir o TTFT longo, usando parte da folga de VRAM.

Tratamentos:

- **A:** valor padrão da revisão fixada do llama.cpp;
- **B:** `OBS_LLAMA_EXTRA_ARGS='--ubatch-size 1024'`.

Antes da rodada, confirmar no `llama-server --help` da revisão fixada que o
padrão atual é menor que 1024. Se já for 1024, usar o dobro do padrão como
variante, preservando apenas essa mudança.

Aceitar a variante se reduzir o TTFT p95 de `long` em pelo menos 10%, mantendo
decode p05 dentro de 3% do baseline e sem OOM. Sem ganho consistente, manter a
configuração atual.

## Ollama: quantizar o KV cache somente se memória for objetivo

### Sinal observado

- TTFT p95 em `long`: 1,84 s, praticamente empatado com llama.cpp.
- Decode p05 em `long`: 59,98 tokens/s.
- GPU em 91,5%, mas potência de 310 W e controlador de memória em 30%.
- VRAM observada de 8,03 GiB.
- `OLLAMA_FLASH_ATTENTION=1` e `OLLAMA_NUM_PARALLEL=1` já estavam ativos.

O principal espaço é o decode. Os dados são compatíveis com menor eficiência
dos kernels ou overhead de dispatch, mas a interface do Ollama não expõe um
controle direto para esses mecanismos. Aumentar paralelismo não responde a
esta carga, porque o benchmark é sequencial.

### Hipótese direta

Trocar o KV cache de `f16` para `q8_0` pode reduzir memória e tráfego do cache
em contexto longo. A hipótese de ganho de velocidade é secundária, pois o
controlador de memória não apareceu saturado.

Tratamentos:

- **A:** `OBS_OLLAMA_KV_CACHE_TYPE=f16`;
- **B:** `OBS_OLLAMA_KV_CACHE_TYPE=q8_0`.

Aceitar a variante se reduzir VRAM ou energia de forma material, mantendo
decode e E2E dentro de 5% do baseline e sem regressão de qualidade. Se o único
objetivo for velocidade, este teste tem prioridade menor que os anteriores.

## Protocolo mínimo da iteração

1. Executar A e B no mesmo hardware, alterando uma variável por vez.
2. Começar com `long` nos dois tratamentos; não comparar uma run `long` isolada
   com uma bateria de carga diferente.
3. Fazer ao menos três runs completas por tratamento, alternando a ordem
   (`A-B-B-A-A-B`) para reduzir viés de temperatura e estado do host.
4. Preservar prompt, hashes, warmup, limite de saída e parâmetros de geração.
5. Comparar TTFT p95, decode p05, E2E p95, falhas, VRAM, GPU, potência e energia.
6. Se a variante vencer, repetir a bateria completa `short medium long`.

Com 20 requisições, o p95 é sensível à cauda da amostra. A decisão deve usar a
consistência entre runs, não apenas um percentil isolado.

## Ordem recomendada

1. vLLM: com e sem `--enforce-eager`.
2. llama.cpp: microbatch atual versus microbatch maior.
3. Ollama: KV `f16` versus `q8_0`, somente se memória ou energia forem eixos de
   decisão.
