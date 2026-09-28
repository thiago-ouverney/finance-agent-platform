# Guia do projeto

> **Objetivo:** orientar a reprodução do fluxo completo e apontar para um guia
> curto de execução, sem repetir os tutoriais de cada componente.

Esta é a porta de entrada da documentação. Ela apresenta o fluxo completo sem
repetir os tutoriais de cada componente.

## Escolha o que você quer fazer

- **CONFIGURAR SSH E ABRIR PONTES** — deixa `uff-delta` e `runpod-qwen`
  prontos para uso local. [Executar o guia](guias/configurar-ssh.md).
- **GERAR TABELAS DO BENCHMARK DE QUALIDADE** — produz `train.csv`,
  `calibration.csv` e `test.csv`. [Executar o guia](guias/gerar-benchmark-qualidade.md).
- **GERAR AVALIAÇÕES E O GOLDEN** — reúne os três votos, cria o consenso e
  mede Macro-F1; o guia registra as pendências atuais do fluxo. [Executar o
  guia](guias/gerar-avaliacao-modelos.md).
- **GERAR UM MODELO QUANTIZADO** — cria, verifica e publica um artefato Qwen.
  [Executar o guia](guias/gerar-modelo-quantizado.md).
- **GERAR GGUF COM E SEM IMATRIX** — cria três GGUFs comparáveis e executa a
  matriz sequencial nos três runtimes. [Executar o guia](guias/gerar-gguf-imatrix.md).
- **GERAR O BENCHMARK DE PERFORMANCE** — executa uma carga sequencial no Pod,
  coleta Prometheus e compara vLLM, llama.cpp e Ollama localmente.
  [Executar o guia](guias/gerar-benchmark-performance.md).
- **COMPARAR RUNTIMES COM BMCS MOPEP** — separa BMCs reais por tamanho, mede
  classificação e revisão e une qualidade com Prometheus.
  [Executar o guia](guias/gerar-benchmark-performance-mopep.md).
- **PERFILAR UMA INFERÊNCIA NO POD** — correlaciona prefill/TTFT e decode com
  CPU, GPU, memória, disco e KV cache em um notebook.
  [Executar o guia](guias/perfilar-inferencia-prometheus.md).
- **INICIAR O ATENDIMENTO NO WHATSAPP** — conecta o adapter local ao Ollama da
  UFF. [Executar o guia](guias/iniciar-whatsapp.md).
- **ENTENDER OS NOTEBOOKS** — mostra a responsabilidade, as entradas, as saídas
  e a ordem de uso de cada notebook. [Consultar o índice](notebooks.md).

```text
BMC + tags MOPEP
        -> dataset train/calibration/test
        -> golden por consenso de três modelos
        -> Qwen2.5-7B quantizado
        -> avaliação de qualidade e performance
        -> uso em tempo real pelo WhatsApp
```

## 0. Ambientes e pontes

| Ambiente | Uso principal |
|---|---|
| Máquina local | Notebooks, dados, adapter do WhatsApp e pontas locais dos túneis |
| UFF | Ollama, Qwen na GPU e atendimento contínuo do WhatsApp |
| RunPod | Quantização, modelos grandes e benchmarks que exigem GPU dedicada |

O alias oficial da UFF é `uff-delta`; para Pods, use `runpod-qwen`. Hosts,
usuários, portas e chaves ficam somente em `~/.ssh/config`.
Os alvos de download respeitam esse alias por padrão; porta e chave só são
passadas à ferramenta SSH quando houver uma sobrescrita explícita.

As pontes mantêm os serviços remotos privados:

```text
notebook de avaliação -> localhost:18000 -> SSH -> Ollama remoto:11434
notebook da silver    -> localhost:18000 -> SSH -> vLLM no RunPod:8000
adapter WhatsApp      -> localhost:8000  -> uff-delta -> Ollama:11434
```

Notebooks e adapter usam uma URL local, enquanto o túnel alcança o runtime
remoto sem publicar sua porta. Uma porta local atende apenas um túnel por vez.

Para reproduzir: [**CONFIGURAR SSH E ABRIR PONTES**](guias/configurar-ssh.md).
Detalhes dos ambientes: [execução local, UFF e RunPod](remote-execution.md).

## 1. Benchmark de qualidade

Para reproduzir: [**GERAR TABELAS DO BENCHMARK DE
QUALIDADE**](guias/gerar-benchmark-qualidade.md).

O Business Model Canvas (BMC) resume o modelo de negócio. O notebook consome o
BMC e as tags MOPEP do BigQuery, normaliza, remove duplicações e mantém as 20
tags escolhidas no treino. Queremos reduzir o custo variável do GPT-4o de
produção com um Qwen local, medindo a perda de qualidade, a latência e o custo
de infraestrutura.

O notebook
[`prepare_mopep_benchmark_splits.ipynb`](../analytics/notebooks/prepare_mopep_benchmark_splits.ipynb)
gera, por padrão, os seguintes arquivos fora do Git:

| Arquivo | Finalidade |
|---|---|
| `/tmp/mopep-bmc-tags/train.csv` | Desenvolvimento, análise e eventual treinamento |
| `/tmp/mopep-bmc-tags/calibration.csv` | Escolha de prompt, regras e processo de curadoria |
| `/tmp/mopep-bmc-tags/test.csv` | Avaliação final, somente após congelar as decisões |

Cada linha representa um BMC único. Os campos principais são:

| Campo | Como interpretar |
|---|---|
| `example_id` | Identificador estável do exemplo |
| `business_model` | Texto fornecido aos modelos |
| `tags` | Lista JSON com os rótulos esperados da taxonomia selecionada |
| `label_source` | Origem dos rótulos, incluindo o registro do GPT-4o de produção |
| `taxonomy_version` | Versão da taxonomia usada na campanha |
| `split` | `train`, `calibration` ou `test` |

Os demais campos dão rastreabilidade e ajudam a detectar vazamento. Dados e
predições permanecem privados.

Depois das três avaliações, o notebook
[`build_mopep_golden_consensus.ipynb`](../analytics/notebooks/build_mopep_golden_consensus.ipynb)
salva o golden compartilhável em
`analytics/results/mopep-golden-shareable/{train,calibration,test}.csv`. Cada
arquivo final contém apenas `id`, `business_model` e `tags`. A associação
privada entre `id` e domínio fica separada em
`/tmp/mopep-bmc-private/identity.csv`.

### 1.1 Como avaliamos um modelo

Para reproduzir: [**GERAR AVALIAÇÕES E O
GOLDEN**](guias/gerar-avaliacao-modelos.md).

O golden é construído após avaliar o mesmo conjunto com três referências:

1. GPT-4o, baseline atualmente usado em produção e já registrado nos dados;
2. Qwen 72B servido no RunPod;
3. GPT-5 nano executado pela Batch API.

As respostas são unidas por `example_id`. Uma tag entra quando recebe ao menos
dois votos (`majority_vote_2_of_3`). Correções manuais geram outra versão com
proveniência.

Depois de congelado o golden, GPT-4o e cada candidato são avaliados no mesmo
`analytics/results/mopep-golden-shareable/test.csv`, com a mesma taxonomia e
sem acesso aos rótulos esperados. A métrica principal é **Macro-F1**: calculamos
o F1 de cada uma das 20 tags e tiramos a média, dando o mesmo peso a tags
frequentes e raras. Também reportamos precisão, recall, exact match, taxa de
parsing, falhas, latência e custo.

Reporte o valor do candidato e sua diferença para o GPT-4o. Qualquer mudança
após olhar o teste exige nova campanha. Como o GPT-4o participa do consenso,
seu Macro-F1 mede acordo com esse consenso, não uma verdade humana independente.

Fluxo detalhado: [três avaliações e golden por consenso](silver-qwen-openai-golden.md).

## 2. Gerando quantizações

Para reproduzir: [**GERAR UM MODELO
QUANTIZADO**](guias/gerar-modelo-quantizado.md).

O modelo-base é
[`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct),
sempre identificado também pela revisão utilizada. Novos artefatos são
publicados no namespace Hugging Face do Thiago, com o
padrão `thiagoouverney/Qwen2.5-7B-Instruct-<TECNICA>-<PRECISAO>`. Use esse
caminho em `QUANT_HF_REPO_ID` para o fluxo GPTQ/AWQ/EXL3 (`HF_REPO_ID`
permanece no fluxo GGUF legado) e registre-o em
`MODEL_ARTIFACT_HF_REPO`. Até a migração, estes artefatos externos continuam
como referências de origem:

- [GGUF Q8_0 do Arthur](https://huggingface.co/arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf), usado hoje pelo benchmark;
- [GGUF Q8_0 do Yan](https://huggingface.co/yanwerneck/Qwen2.5-7B-Instruct-GGUF-Q8_0), destino histórico do alvo de publicação.

O repositório canônico registra origem e revisão, técnica, bits, parâmetros,
calibração, seed, ferramentas, formato, SHA-256, tokenizer e runtimes validados.
Execuções fixam uma revisão ou digest; `main` não é identidade experimental.

### 2.1 Fundamentos das técnicas

| Técnica/artefato | Calibração | Saída principal |
|---|---|---|
| GPTQ 4/8-bit | Sim | checkpoint quantizado com tensores Safetensors |
| AWQ 4-bit | Sim | checkpoint AWQ com tensores Safetensors |
| EXL3 | Sim | checkpoint EXL3 |
| llama.cpp Q4_K_M | Opcional: iMatrix do `train.csv` | arquivo GGUF único |
| llama.cpp Q8_0 | Não; referência de maior precisão | arquivo GGUF único |

O helper experimental legado usa 256 exemplos de
`Brench/MMLU-Pro-CoT-Train-84K`. O fluxo reproduzível do projeto usa uma seleção
determinística de 256 exemplos do golden `train.csv`, aplica o chat template do
tokenizer com o prompt completo da taxonomia MOPEP e registra hashes do prompt
e do dataset, seed e assinatura da seleção. `calibration.csv` fica
reservado para escolher técnica e parâmetros; `test.csv` é recusado como fonte
de quantização e permanece reservado para a avaliação final.

O Make atual verifica integridade, recarga via GPTQModel e uma inferência
mínima. A aprovação de qualidade por Macro-F1 em `calibration.csv`/`test.csv` e
a validação em vLLM ainda são etapas separadas, não automatizadas por esse alvo.

GPTQ minimiza o erro nas ativações, AWQ preserva canais importantes e EXL3
distribui taxas de bits pelo modelo. O fluxo GGUF converte uma única vez a fonte
para BF16 e produz `Q4_K_M` sem iMatrix, `Q4_K_M` com iMatrix calculada sobre o
`train.csv` renderizado e `Q8_0` como referência. A comparação entre os dois Q4
isola o efeito da calibração; Q8 contra Q4 também muda a precisão.

A *importance matrix* (`imatrix`) do `llama.cpp` mede ativações e orienta
algumas quantizações GGUF. Ela não treina o modelo, não é usada pelo Q8_0 e não
é a mesma calibração de GPTQ, AWQ ou EXL3. O arquivo da iMatrix é insumo de
build e não é necessário pelos runtimes depois que o GGUF calibrado foi gerado.

### 2.2 Formatos e runtimes

| Artefato | Runtime no projeto |
|---|---|
| GPTQ | GPTQModel ou vLLM compatível |
| AWQ | loader AWQ ou vLLM compatível |
| EXL3 | ExLlamaV3/GPTQModel; fora do trio principal |
| GGUF | llama.cpp, Ollama e vLLM experimental |

| Formato | Característica |
|---|---|
| Safetensors | Armazena tensores com leitura segura; configuração e tokenizer normalmente ficam em arquivos separados |
| GGUF | Reúne tensores e metadados para o ecossistema GGML/llama.cpp, geralmente em um único arquivo |

Ambos podem armazenar pesos em alta precisão ou quantizados. Um nome contendo
“GGUF” ou “8-bit” não prova o formato real: valide assinatura, metadados, tamanho
e hash do artefato.

Detalhes: [pipeline de quantização](../pipelines/qwen-quantization/README.md) e
[guia completo para Pod vazio](guias/gerar-modelo-quantizado.md),
[guia GGUF com iMatrix e matriz entre runtimes](guias/gerar-gguf-imatrix.md) e
[conversão HF para GGUF Q8_0 legada](../pipelines/qwen-quantization/scripts/quantize_hf_to_gguf_q8_0.sh).

## 3. Performance de inferência

Para reproduzir: [**GERAR O BENCHMARK DE
PERFORMANCE**](guias/gerar-benchmark-performance.md).

A comparação principal usa o mesmo GGUF, tokenizer, conjunto de prompts,
temperatura e limite de saída em vLLM, llama.cpp e Ollama. O benchmark oficial
envia **uma requisição por vez**; concorrência é outro experimento.

| Métrica | Pergunta respondida |
|---|---|
| Startup e primeira resposta | Quanto custa deixar o servidor utilizável? |
| TTFT | Quanto o usuário espera até começar a resposta? |
| Tokens/s de decode | Quão rápido o texto é gerado depois do primeiro token? |
| Latência ponta a ponta | Quanto dura a resposta completa? |
| VRAM, GPU e RAM | Qual é o custo operacional observado? |
| Falhas e completude | Em qual contexto o runtime deixa de atender corretamente? |

### 3.1 Diagnóstico e melhoria de um gargalo

Para analisar os runtimes, fixe o mesmo GGUF, hardware, prompt, limite de
saída e demais condições do experimento. Compare, ao longo das fases de
inicialização, prefill e decode, os tempos e os gráficos de CPU, GPU, RAM,
VRAM, PCIe, disco e swap. PCIe, disco e swap só podem sustentar conclusões
depois que a execução tiver instrumentação explícita para essas medidas.

A comparação deve localizar em que fase e recurso aparece o gargalo, não
apenas ordenar os runtimes por velocidade. Depois do diagnóstico, escolha um
ponto específico, altere uma variável por vez e repita exatamente a mesma carga.
O resultado deve apresentar o antes e o depois em TTFT, tokens/s e utilização
dos recursos, explicando se a mudança confirmou ou rejeitou a hipótese inicial.

`make bench-all MODEL_SIZE=7B` executa os runtimes em sequência e grava em
`services/inference-runtime/results/<runtime>/...`. Smoke valida integração,
mas não produz comparação final.

Detalhes: [runtime e comandos](../services/inference-runtime/README.md),
[bench explicado](../services/inference-runtime/docs/benchmark/bench-explicado.md)
e [metodologia](../services/inference-runtime/docs/benchmark/metodologia.md).

### 3.2 Benchmark headless com Prometheus

`make observe-bench OBS_RUNTIME=<runtime>` prepara o runtime e o modelo antes
da medição, mantém o Prometheus ativo durante a bateria sequencial e gera um
pacote portátil com requisições, eventos, amostras e agregações por fase. O
pacote é baixado com `make pull-observe-results` e analisado localmente com
`make observability-dataset` e `make observability-notebook`.

Na análise offline, cada runtime usa um `run_id` explícito e mantém sua própria
conclusão. A comparação só é exibida quando as três runs compartilham a mesma
identidade experimental e as mesmas requisições pareadas.

Para uma campanha GGUF completa, `make observe-bench-all` executa primeiro o
smoke de vLLM, llama.cpp e Ollama. A bateria completa dos três só começa se os
três smokes terminarem com sucesso.

Checkpoint Hugging Face remoto ou local é suportado pelo vLLM. O mesmo GGUF
remoto ou local pode ser usado nos três runtimes. Downloads não pertencem à
janela medida; segredos, pesos e resultados brutos não são versionados.

### 3.3 Perfil exploratório de uma requisição

Para localizar um gargalo antes de formular uma mudança, o fluxo interativo
inicia um runtime, Prometheus e Jupyter no mesmo Pod. O navegador fica na
máquina local e acessa o notebook por túnel SSH; a inferência e a coleta
permanecem no Pod.

O notebook marca início da requisição, primeiro token, último conteúdo e fim da
resposta sobre gráficos de CPU, GPU, RAM, VRAM, disco, swap, page faults e pool
KV quando disponível. Esse perfil produz evidência para diagnóstico, mas uma
requisição isolada não produz média, distribuição ou comparação final.

Para executar: [**PERFILAR UMA INFERÊNCIA NO
POD**](guias/perfilar-inferencia-prometheus.md). Para transformar a hipótese em
comparação, preserve o mesmo modelo e carga e volte ao benchmark formal.

### 3.4 Reutilização proposital de contexto

Além da bateria principal, será usado um cenário conversacional com pedidos
como “refine o resultado anterior”. Cada turno reenvia um prefixo crescente e
permite observar TTFT, tokens/s, memória e métricas de cache disponíveis.

Compare com um controle de comprimento semelhante e prefixo alterado. No teste
controlado, respostas anteriores ficam fixas em replay; respostas reais formam
um cenário secundário porque mudam o histórico de cada runtime.

Uma melhora pode vir de pesos carregados, compilação ou cache de prefixo, não
necessariamente do KV-cache. O sweep mede capacidade e memória; o refinamento
mede a experiência multi-turno.

### 3.5 Workload real MOPEP

O perfil MOPEP usa BMCs independentes agrupados por tercis de tokens da
calibração e mede classificação, revisão controlada e closed-loop. O runtime
recebe BMC e heurísticas, nunca as tags esperadas. Para executar e interpretar:
[benchmark MOPEP entre runtimes](guias/gerar-benchmark-performance-mopep.md).

## 4. Uso em tempo real

Para reproduzir: [**INICIAR O ATENDIMENTO NO
WHATSAPP**](guias/iniciar-whatsapp.md).

O fluxo operacional atual é:

```text
WhatsApp -> Baileys -> adapter local -> túnel uff-delta -> Ollama -> Qwen
```

Cada contato possui histórico próprio. Mensagens do mesmo contato são
serializadas; contatos diferentes ficam isolados e podem avançar
independentemente. A capacidade sob uma carga concorrente específica ainda
exige benchmark separado.

O adapter chama a API OpenAI-compatible do Ollama pela ponte SSH, sem gateway.
Sessões, contatos, chaves e históricos não são versionados. A integração do
WhatsApp não é oficial e está sujeita às regras da plataforma.

Operação completa: [WhatsApp com Ollama na UFF](uff-ollama-whatsapp.md) e
[README do adapter](../apps/whatsapp-adapter/README.md).
