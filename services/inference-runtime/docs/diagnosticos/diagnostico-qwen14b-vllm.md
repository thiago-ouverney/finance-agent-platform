# Diagnóstico: Qwen2.5-14B no vLLM

Este documento registra, sem apagar as falhas, as tentativas de executar o Qwen2.5-14B quantizado no vLLM em uma RTX 3090 de 24 GB. O objetivo é separar problemas de arquivo, tokenizer, configuração, loader e memória. Uma falha de inicialização não é tratada como métrica de desempenho.

## Resumo executivo

Foram testados dois artefatos:

1. `arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf`, um arquivo GGUF Q8_0.
2. Um checkpoint safetensors de 8 bits produzido com GPTQModel.

O GGUF funcionou no llama.cpp/Ollama, mas não iniciou no vLLM porque o plugin GGUF atingiu aproximadamente 23,46 GiB de uso em uma GPU com cerca de 23,56 GiB disponíveis durante o carregamento. A falha ocorreu ao criar um tensor auxiliar padded, antes de o engine iniciar e antes de haver uma medição real do KV cache.

O safetensors não falhou por falta de VRAM. Ele falhou por incompatibilidades sucessivas entre o layout dos tensores quantizados, os metadados GPTQ e o loader/backend selecionado pelo vLLM. Depois de corrigir os metadados mínimos, o loader chegou aos tensores e encontrou uma incompatibilidade de dimensões (`4736` versus `3584`).

Conclusão: os dois artefatos são problemas diferentes. O GGUF é um caso de pico de memória do caminho experimental GGUF do vLLM; o safetensors é um caso de formato/layout GPTQ incompatível com o loader instalado.

## Atualização: execução que carregou o modelo e falhou no profiling

Na execução de 21/09, o arquivo local foi finalmente reconhecido e o plugin terminou o carregamento dos pesos. O log registrou:

```text
Model loading took 23.09 GiB memory and 71.283941 seconds
```

Esse valor é a evidência mais forte obtida até agora. A GPU reportava cerca de 23,56 GiB disponíveis, portanto o modelo sozinho ocupou aproximadamente 98% da capacidade. Restaram menos de 0,5 GiB para o restante do engine.

O processo não caiu durante `load_model`. Em seguida, o vLLM entrou em:

```text
_initialize_kv_caches
determine_available_memory
profile_run
_dummy_run
```

Durante o profiling, TorchInductor tentou executar um bloco de autotuning e pediu mais 108 MiB, mas havia apenas 73 MiB livres:

```text
torch._inductor.exc.InductorError: ... CUDA out of memory
Tried to allocate 108.00 MiB
GPU ... 23.48 GiB memory in use
73.00 MiB is free
```

Assim, a falha mais recente é composta por duas fases: (a) 23,09 GiB de pesos/estruturas após o carregamento GGUF; (b) memória adicional para profiling, compilação/autotuning, workspaces e preparação do KV cache. O log não mostra uma reserva bem-sucedida de KV cache nem uma API pronta. O uso observado não deve ser chamado de “23,09 GiB de KV cache”.

O warning sobre `expandable_segments` é consequência da falta de espaço, não uma explicação alternativa: o allocator tentou mapear outro segmento quando a GPU já estava praticamente cheia.

## 1. Ambiente de referência

- GPU: NVIDIA GeForce RTX 3090.
- VRAM nominal: 24 GB; capacidade reportada pelo CUDA no log: aproximadamente 23,56 GiB.
- vLLM observado nos logs: 0.29.0.
- Python: 3.12.
- Execução com um processo e uma sequência por vez (`--max-num-seqs 1` quando aplicável).
- Tokenizer separado do arquivo GGUF.

O tamanho do arquivo no SSD não é igual ao pico de VRAM. A memória pode conter pesos, cópias convertidas, padding, escalas, workspaces CUDA, buffers e KV cache.

## 2. Tentativas com o GGUF

### 2.1 Arquivo usado

```text
/workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
```

O arquivo possuía assinatura GGUF válida (`GGUF`) e aproximadamente 15–17 GB no SSD.

### 2.2 Primeiros problemas: tokenizer e caminho local

Inicialmente o tokenizer local estava incompleto. O vLLM reportou que o diretório não tinha um `config.json` reconhecível. Em outra tentativa, quando o caminho local não foi reconhecido, o Hugging Face tentou interpretar o caminho como um `repo_id` e produziu:

```text
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
```

Esse erro não significa que o GGUF seja inválido. Significa que o caminho fornecido não foi aceito como diretório local completo pelo Transformers/vLLM. A correção é preparar um diretório de tokenizer contendo, no mínimo, `config.json`, `tokenizer_config.json`, `tokenizer.json` e `special_tokens_map.json`, e validá-lo com `AutoTokenizer.from_pretrained(..., local_files_only=True)`.

Também observamos que certas combinações de vLLM e `HF_HUB_OFFLINE=1` podem fazer o caminho local passar por uma validação de repositório antes do loader GGUF. Para o diagnóstico do arquivo local, o plugin precisa estar instalado no mesmo venv do vLLM e o caminho deve ser reconhecido como arquivo local.

### 2.3 Comando de teste

```bash
/app/.vllm_venv/bin/vllm serve \
  /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --load-format gguf \
  --quantization gguf \
  --tokenizer /workspace/models/Qwen2.5-14B-tokenizer-complete \
  --served-model-name qwen14b-q8-gguf \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90
```

O suporte GGUF do vLLM é experimental e depende do `vllm-gguf-plugin`. O fato de o mesmo arquivo funcionar em llama.cpp/Ollama não garante que ele caiba no caminho de carregamento do vLLM.

### 2.4 OOM observado

O erro essencial foi:

```text
torch.OutOfMemoryError: CUDA out of memory.
GPU total: aproximadamente 23,56 GiB
GPU usada: aproximadamente 23,46 GiB
Tentativa adicional: 144 MiB
Memória livre: aproximadamente 93 MiB
```

O stack trace aponta para:

```text
vllm_gguf_plugin/quantization/linear.py
_create_padded_weight_param
torch.zeros(...)
```

Isso prova que o processo estava criando um tensor auxiliar padded/alinhado quando a placa já estava praticamente cheia. A falha aconteceu em `load_model`/`process_weights_after_loading`, antes de o servidor ficar pronto.

Não há evidência de que 23,46 GiB fossem KV cache. O log falha antes de uma inicialização normal do engine e antes de uma linha confiável de tamanho do pool KV. Portanto, a explicação correta é pico de carregamento/materialização dos pesos e buffers, não “o KV cache consumiu toda a VRAM”.

### 2.5 Por que llama.cpp/Ollama funcionaram

llama.cpp e Ollama usam GGUF como formato nativo. Eles conseguem manter os pesos Q8_0 compactados e usar kernels que operam nessa representação. O caminho é aproximadamente:

```text
GGUF Q8_0 → pesos quantizados + escalas → kernels nativos → KV cache
```

O plugin GGUF do vLLM precisa adaptar o arquivo às estruturas de execução do vLLM. Durante essa adaptação podem coexistir:

```text
pesos lidos + representação interna + cópia padded + escalas + workspaces
```

Assim, o pico de VRAM pode ser muito maior que o tamanho persistente do arquivo no SSD. O arquivo não “mudou de tamanho” no disco; a representação temporária na GPU ficou maior.

`--gpu-memory-utilization 0.90` limita principalmente o orçamento operacional/KV. Ele não elimina as cópias temporárias criadas durante o carregamento. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` pode ajudar fragmentação, mas não resolve um pico estrutural que já ultrapassa a capacidade.

### 2.6 O que o código do plugin faz

Inspecionamos o código do plugin instalado e a implementação pública correspondente. A sequência relevante é:

1. `plugin.py` registra `GGUFModelLoader`, o formato `gguf` e o parser GGUF. Quando o argumento termina em `.gguf`, ele configura `load_format=gguf`, `quantization=gguf` e usa o tokenizer/configuração como fonte da arquitetura Hugging Face.
2. `loader.py::_prepare_weights` aceita um arquivo local quando `os.path.isfile(model_name_or_path)` é verdadeiro. Para uma referência remota, baixa o GGUF; para `diretório:quant_type`, resolve o arquivo local/remoto. Isso explica por que, quando o arquivo não existia no caminho informado, o plugin caiu no erro de `repo_id=/workspace/models`.
3. `loader.py::_prepare_adapter` lê os nomes dos tensores GGUF, constrói o mapa para nomes Hugging Face, detecta módulos não quantizados e registra os layouts lineares.
4. `loader.py::load_model` inicializa a arquitetura vLLM na GPU, carrega os pesos via `gguf_quant_weights_iterator_multi` e chama `process_weights_after_loading`.
5. `linear.py::process_weights_after_loading` materializa os parâmetros GGUF e chama `_create_padded_weight_param`.
6. `_create_padded_weight_param` calcula `padded_side = max(x.size(1) ...)` e `concat_side = sum(x.size(0) ...)`, aloca `torch.zeros((concat_side, padded_side), device=weight.device)` e copia os fragmentos para uma nova matriz. Só depois limpa os contêineres antigos e substitui o parâmetro da camada.

O item 6 explica mecanicamente o pico observado no erro anterior: durante a criação do padding, a matriz antiga e `padded_data` podem coexistir. O código não faz uma conversão integral automática para BF16; ele mantém os pesos GGUF quantizados e cria parâmetros alinhados ao layout do kernel, mas essa cópia também é uma alocação real na GPU.

Depois do carregamento, o vLLM executa o próprio profiling de memória (`determine_available_memory`/`profile_run`). Esse profiling pode compilar grafos TorchInductor, fazer autotuning e alocar workspaces antes de decidir o tamanho do pool KV. Foi nessa etapa que a execução mais recente falhou.

Referências diretas do código: [`loader.py`](https://github.com/vllm-project/vllm-gguf-plugin/blob/main/vllm_gguf_plugin/loader.py), [`linear.py`](https://github.com/vllm-project/vllm-gguf-plugin/blob/main/vllm_gguf_plugin/quantization/linear.py) e [`plugin.py`](https://github.com/vllm-project/vllm-gguf-plugin/blob/main/vllm_gguf_plugin/plugin.py).

## 3. Tentativas com safetensors quantizado

### 3.1 Confusão de nomenclatura

O repositório/artefato tinha nome relacionado a `GGUF-8bit`, mas os arquivos observados eram:

```text
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

O nome do repositório não determina o formato interno. GGUF é um contêiner próprio; safetensors é outro formato. Um checkpoint safetensors chamado “GGUF” não deve ser carregado como um arquivo GGUF.

### 3.2 Primeira inconsistência: loader GGUF contra safetensors

Em uma tentativa, o log mostrou simultaneamente:

```text
Loading safetensors checkpoint shards
quantization=gguf
```

O vLLM construiu uma camada quantizada que esperava:

```text
lm_head.qweight
lm_head.qweight_type
```

mas encontrou:

```text
lm_head.weight
```

Isso era incompatibilidade entre o loader/configuração GGUF e o layout real dos safetensors. Não era um erro de qualidade nem de KV cache.

### 3.3 Tentativa de remover a configuração de quantização

Ao remover `quantization_config`, o vLLM passou a construir camadas densas esperando:

```text
layers.0.mlp.down_proj.weight
```

mas o checkpoint forneceu:

```text
layers.0.mlp.down_proj.qweight
```

Essa tentativa demonstrou que o checkpoint não era FP16/BF16 comum. Ele realmente continha pesos quantizados e precisava de um loader GPTQ compatível. Remover a configuração não era uma correção; apenas trocou o tipo de erro.

### 3.4 O que o repositório de quantização fez

O script usado no repositório `arthuravianna/qwen2.5-quantization` importa `GPTQConfig` de `gptqmodel` e cria:

```python
quant_config = GPTQConfig(
    bits=bits,
    group_size=128,
)
```

Logo, `group_size=128` foi explicitamente usado. `desc_act`, `sym` e `damp_percent` não foram informados explicitamente no script; dependem dos padrões da versão do GPTQModel instalada. O README do projeto trata GPTQ e GGUF como métodos separados.

### 3.5 Metadados GPTQ ausentes

Ao tentar `--quantization gptq`/`gptq_marlin`, o vLLM exigiu campos ausentes:

```text
Cannot find any of ['group_size'] in the model's quantization config
Cannot find any of ['desc_act'] in the model's quantization config
```

Foi possível justificar `group_size=128` pelo script de quantização. Foram testados metadados GPTQ em uma cópia, incluindo `bits=8`, `group_size=128`, `desc_act=false` e `sym=true`. Isso permitiu que o vLLM avançasse além da validação de configuração.

Adicionar campos manualmente não converte os tensores. Só descreve para o loader um formato que precisa ser verdadeiro. Se os valores não corresponderem à quantização real, o modelo pode carregar incorretamente ou produzir resultados inválidos.

### 3.6 Falha final: dimensões/empacotamento

Com a configuração GPTQ preenchida, o vLLM chegou ao carregamento dos shards e falhou em:

```text
RuntimeError: start (0) + length (4736) exceeds dimension size (3584)
```

Os tensores observados no arquivo foram:

```text
model.layers.0.mlp.down_proj.qweight  [3584, 20128]
model.layers.0.mlp.gate_proj.qweight  [18944, 3808]
model.layers.0.mlp.up_proj.qweight    [18944, 3808]
model.layers.0.self_attn.q_proj.qweight [3584, 3808]
```

O vLLM esperava uma forma GPTQ compatível com o particionamento do Qwen2.5-7B/14B que estava carregando, mas recebeu tensores com orientação, padding ou empacotamento diferentes. O número `4736` é compatível com uma dimensão empacotada esperada em uma matriz de tamanho lógico `18944` para fator de empacotamento 4; o tensor fornecido não tinha essa forma esperada pelo loader.

Isso não é resolvido com `int8_per_channel_weight_only`. Esse parâmetro representa outro esquema de quantização; não é um decodificador genérico de `qweight` GPTQ.

Também não é OOM: a falha ocorre durante a interpretação/particionamento dos tensores, antes da inicialização do servidor e antes de qualquer geração.

## 4. O que cada erro significa

| Erro | Etapa | Interpretação |
|---|---|---|
| `HFValidationError` para caminho `.gguf` | Resolução do modelo | O caminho local não foi interceptado/reconhecido e foi tratado como repo ID; pode envolver plugin ausente ou regressão com modo offline. |
| `HFValidationError` para tokenizer | Resolução do tokenizer | A pasta local estava incompleta ou não continha arquivos/configuração reconhecíveis. |
| `lm_head.weight` versus `lm_head.qweight` | Construção/carregamento da camada | Loader e layout do checkpoint eram incompatíveis. |
| `group_size` ausente | Validação GPTQ | Metadados obrigatórios não foram exportados no `config.json`. |
| `desc_act` ausente | Validação GPTQ | Outro metadado GPTQ obrigatório não foi exportado. |
| `4736 exceeds dimension 3584` | Leitura dos tensores | Layout/empacotamento GPTQ incompatível com o loader vLLM. |
| `CUDA out of memory` em `_create_padded_weight_param` | Materialização GGUF | Pico de VRAM do plugin GGUF durante conversão/padding. |

## 5. O que não devemos concluir

Não é correto dizer que:

- o arquivo GGUF tinha 24 GB no SSD;
- todo o uso de 23,46 GiB era KV cache;
- reduzir `--gpu-memory-utilization` transformaria o GGUF em uma representação menor;
- renomear o repositório ou mudar uma string de quantização converteria os pesos;
- qualquer safetensors de 8 bits é automaticamente GPTQ compatível com vLLM;
- funcionar no llama.cpp/Ollama prova compatibilidade com o plugin GGUF do vLLM.

## 6. Caminho recomendado

Para a comparação dos runtimes:

1. Use o GGUF Q8_0 no llama.cpp e no Ollama, onde ele é um formato nativo.
2. Para vLLM, use um checkpoint safetensors FP16/BF16 ou uma quantização comprovadamente compatível com o loader/backend da versão instalada.
3. Se a exigência for exatamente o mesmo artefato quantizado, registre que a comparação vLLM não foi possível por incompatibilidade de carregamento; não troque silenciosamente o modelo.
4. Preserve os logs de cada falha como resultado diagnóstico.

O próximo teste limpo para vLLM deve usar um artefato safetensors oficialmente compatível com vLLM ou uma nova quantização/exportação que escreva corretamente o layout GPTQ e todos os metadados. O GGUF de 14B não deve ser usado como resultado formal de desempenho no vLLM enquanto o plugin não carregar o modelo sem OOM.

## Referências

- [Documentação de GGUF no vLLM](https://docs.vllm.ai/en/latest/features/quantization/gguf/): suporte experimental, plugin externo e uso de tokenizer separado.
- [Repositório do plugin vLLM-GGUF](https://github.com/vllm-project/vllm-gguf-plugin).
- [Repositório de quantização usado pelo grupo](https://github.com/arthuravianna/qwen2.5-quantization).
- [GPTQConfig e parâmetros padrão](https://huggingface.co/docs/transformers/main/quantization).
