# Relatório de falhas experimentais

Este documento registra falhas observadas durante os experimentos. Uma falha não é substituída silenciosamente por outro modelo, formato ou runtime. O objetivo é preservar evidência para a análise acadêmica.

## F-001 — incompatibilidade do `lm_head` em safetensors quantizado

- **Data observada:** 2026-09-21
- **Runtime:** vLLM 0.29.0 com `vllm-gguf-plugin`
- **GPU:** NVIDIA GeForce RTX 3090, 24 GB
- **Artefato:** `arthuravianna/Qwen2.5-14B-Instruct-GGUF-8bit`
- **Caminho local:** `/workspace/models/Qwen2.5-14B-Instruct-GGUF-8bit`
- **Formato físico:** cinco shards `safetensors`
- **Tamanho aproximado:** 17 GB
- **Quantização declarada:** 8-bit, `method: gguf`, `format: q_0`
- **Fase da falha:** carregamento do modelo, antes da API ficar disponível e antes da primeira requisição
- **Resultado:** falha de inicialização; não há métricas de latência válidas

### Evidência

O `config.json` e o `quantize_config.json` declaram:

```json
{
  "lm_head": false,
  "method": "gguf",
  "quant_method": "gguf",
  "format": "q_0"
}
```

O índice dos pesos contém:

```text
lm_head.weight
```

Durante o carregamento, porém, o vLLM informou que o `ParallelLMHead` possuía apenas:

```text
lm_head.qweight
```

e falhou com:

```text
ValueError: There is no module or parameter named 'lm_head.weight' in Qwen2ForCausalLM
```

### Interpretação

O loader construiu uma cabeça de saída quantizada e esperou `lm_head.qweight`, mas o checkpoint fornece uma cabeça não quantizada em `lm_head.weight`. O layout do checkpoint e a interpretação da configuração de quantização não são compatíveis com esta combinação de vLLM e plugin.

Esta ocorrência **não foi um OOM**, não foi causada pelo benchmark e não indica falha do tokenizer. Renomear o tensor manualmente não é uma correção válida, pois não converte sua representação.

### Decisão metodológica

Classificar como:

```text
startup_failure_checkpoint_parameter_mismatch
```

Preservar o `server.log` bruto no diretório privado de resultados, mas não publicá-lo sem revisão. Não preencher TTFT, tokens/s ou KV cache com zero: essas métricas são **não observadas**.

## F-002 — OOM ao carregar GGUF Q8_0 no vLLM

- **Data observada:** 2026-09-21
- **Runtime:** vLLM 0.29.0 com `vllm-gguf-plugin`
- **GPU:** NVIDIA GeForce RTX 3090, 24 GB (23,56 GiB reportados pelo PyTorch)
- **Artefato:** `arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf`
- **Formato físico:** arquivo GGUF verdadeiro; o arquivo começa com a assinatura `GGUF`
- **Tamanho no SSD:** aproximadamente 17 GB
- **Configuração observada:** `max_model_len=4096`, `tensor_parallel_size=1`, `dtype=torch.bfloat16`
- **Fase da falha:** materialização/padding dos pesos, antes da API ficar disponível
- **Resultado:** falha de inicialização; TTFT, tokens/s e KV cache não observados

### Evidência

O processo reconheceu a arquitetura e iniciou o carregamento:

```text
Resolved architecture: Qwen2ForCausalLM
Loading model from scratch...
Using FlashAttention version 2
```

Depois, o allocator CUDA informou:

```text
GPU 0 has a total capacity of 23.56 GiB
Process ... has 23.46 GiB memory in use
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 144.00 MiB.
```

O traceback terminou em:

```text
vllm_gguf_plugin/quantization/linear.py
_create_padded_weight_param
torch.zeros(...)
```

### Interpretação

Os aproximadamente 17 GB do arquivo são tamanho de armazenamento, não um limite de VRAM. Durante o carregamento, o plugin precisa manter pesos quantizados, partes não quantizadas, escalas/metadados, alinhamento/padding e buffers temporários. O vLLM também pode reservar estruturas para execução CUDA e KV cache.

Neste log não apareceu uma etapa bem-sucedida de inicialização do KV cache. Portanto, não podemos atribuir este OOM ao KV cache: a falha ocorreu enquanto os pesos ainda estavam sendo preparados. `--gpu-memory-utilization`, `--max-model-len` e `--max-num-seqs` podem reduzir o orçamento do cache, mas não garantem que um buffer temporário de carregamento caiba.

O arquivo é um GGUF válido; a falha demonstra que **este GGUF Q8_0 não coube nesta combinação de vLLM, plugin, versão e RTX 3090**, não que todo GGUF seja inválido no vLLM. O suporte GGUF do vLLM é experimental e deve ser tratado como uma condição do backend.

### Decisão metodológica

Classificar como:

```text
startup_failure_cuda_oom_during_weight_materialization
```

Não reduzir a memória silenciosamente, não trocar para outro formato e não preencher métricas de geração com zero. Para a próxima tentativa, usar o modelo 7B de 8 bits como **novo artefato**, com configuração e resultados separados; não comparar seus números diretamente com os do 14B.

## F-003 — OOM após o carregamento do GGUF, durante profiling do vLLM

- **Data observada:** 2026-09-21
- **Runtime:** vLLM 0.29.0 com `vllm-gguf-plugin`
- **GPU:** NVIDIA GeForce RTX 3090, 24 GB (23,56 GiB reportados pelo CUDA)
- **Artefato:** `arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf`
- **Configuração observada:** `max_model_len=4096`, `max_num_seqs=1`, `gpu_memory_utilization=0.90`
- **Fase da falha:** profiling de memória, compilação/autotuning e preparação do KV cache
- **Resultado:** falha de inicialização; nenhuma requisição foi atendida

### Evidência

Nesta execução o plugin terminou o carregamento dos pesos e registrou:

```text
Model loading took 23.09 GiB memory and 71.283941 seconds
```

Em seguida, o engine entrou em:

```text
_initialize_kv_caches
determine_available_memory
profile_run
_dummy_run
```

Durante o autotuning do TorchInductor, uma alocação adicional de 108 MiB falhou quando havia apenas 73 MiB livres:

```text
torch._inductor.exc.InductorError: ... CUDA out of memory
GPU ... 23.48 GiB memory in use
73.00 MiB is free
```

### Interpretação

Esta ocorrência é diferente de F-002. O log mostra que os pesos e as estruturas do carregamento já ocupavam 23,09 GiB; depois ainda eram necessários espaço para profiling, grafos compilados, autotuning, workspaces e o pool de KV cache. O processo falhou antes de publicar uma capacidade observável de KV cache, portanto não é correto chamar os 23,09 GiB de “KV cache”.

O resultado confirma que `--gpu-memory-utilization 0.90` não limita o pico de carregamento do modelo: ele é aplicado ao orçamento operacional depois que o modelo já foi colocado na GPU. Também confirma que `max_num_seqs=1` reduz a demanda de execução, mas não remove o custo fixo dos pesos nem dos buffers de inicialização.

### Decisão metodológica

Classificar como:

```text
startup_failure_cuda_oom_during_memory_profiling_and_kv_initialization
```

Registrar F-002 e F-003 separadamente. F-002 prova um pico durante a materialização/padding dos pesos; F-003 prova que, mesmo quando essa etapa termina, não sobra memória suficiente para o profiling e a inicialização normal do engine. Em ambas, TTFT, tokens/s e capacidade de KV cache são **não observados**.
