# Três avaliações MOPEP e um dataset golden por consenso

> **Objetivo:** gerar as duas avaliações auxiliares, combiná-las com o GPT-4o
> registrado e produzir um golden auditável por consenso de dois entre três.

> **Estado atual:** este documento descreve o contrato pretendido. O notebook
> GPT-5 ainda não materializa os três splits necessários com `Run All`; consulte
> as limitações do [guia curto de execução](guias/gerar-avaliacao-modelos.md)
> antes de iniciar uma campanha final.

Este guia descreve o contrato para reunir três classificações independentes por
`example_id` e submetê-las à curadoria:

- **baseline GPT-4o**: resultado do modelo atualmente usado em produção e já
  registrado na origem como `mopep-gpt4o-pseudo-label`;
- **silver Qwen**: um checkpoint Qwen servido pelo vLLM em um Pod RunPod;
- **silver GPT**: `gpt-5-nano` na Responses API, por meio da Batch API;
- **golden**: rótulo final incluído quando recebe ao menos dois dos três votos.

Os textos e resultados brutos permanecem locais. Não versione datasets,
tokens, endereços, chaves SSH, pesos ou saídas das inferências.

## Desenho experimental

```text
business_model + example_id
          |
          +----> GPT-4o / produção ----> baseline_tags ---------+
          +----> Qwen 72B / RunPod ----> silver_qwen_tags ------+--> consenso 2/3 --> golden_tags
          +----> GPT-5 nano / Batch ---> silver_gpt_tags -------+

calibration: define e valida o processo de curadoria
test: mede uma vez depois que modelos, prompts e critérios estão congelados
```

Os três modelos devem receber o mesmo texto, a mesma taxonomia e nenhum rótulo
esperado. Guarde modelo, digest ou revisão, prompt, parâmetros e `input_sha256`
para tornar cada campanha auditável. A origem do GPT-4o já está registrada no
dataset; preserve esse registro quando materializar a tabela de comparação. O
notebook de consolidação é
[`analytics/notebooks/build_mopep_golden_consensus.ipynb`](../analytics/notebooks/build_mopep_golden_consensus.ipynb).

## 1. Subir o Qwen 72B na RunPod

### Escolha do Pod

Crie um Pod com:

- template oficial PyTorch/RunPod;
- **1 x A100 80 GB**;
- acesso SSH habilitado;
- ao menos 100 GB persistentes, preferencialmente em Network Volume;
- nenhuma exposição pública das portas do runtime (`8000` no vLLM ou `11434`
  no Ollama).

O artefato Ollama `qwen2.5:72b-instruct-q4_K_M` tem 47 GB. A A100 de 80 GB
deixa margem para o runtime e o KV cache; a RTX 3090 de 24 GB usada nos
experimentos menores não é adequada para esse artefato sem offload. A RunPod
monta o armazenamento persistente em `/workspace` por padrão. Network Volume
sobrevive à terminação do Pod; Volume Disk sobrevive a stop/start, mas é
apagado quando o Pod é terminado.

Referências atuais:

- [criação de Pods e armazenamento](https://docs.runpod.io/get-started);
- [preços de GPU da RunPod](https://www.runpod.io/pricing);
- [tag Qwen2.5 72B Q4_K_M no Ollama](https://ollama.com/library/qwen2.5/tags).

### Servir o modelo com vLLM

O alvo `make silver-qwen` usa a API OpenAI-compatible do vLLM. Inicie o modelo
com um nome estável, por exemplo `qwen-silver`, e mantenha o servidor vinculado
ao loopback do Pod. O checkpoint concreto e a quantização devem ser os que você
validou para a GPU escolhida:

```bash
vllm serve <CHECKPOINT_QWEN> \
  --served-model-name qwen-silver \
  --host 127.0.0.1 \
  --port 8000
```

Encaminhe a porta por SSH a partir da máquina local:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  runpod-qwen
```

Em outro terminal local, confirme o endpoint e copie o `id` exato do modelo:

```bash
curl http://127.0.0.1:18000/v1/models
```

Gere primeiro dez exemplos:

```bash
make silver-qwen \
  SILVER_BASE_URL=http://127.0.0.1:18000/v1 \
  SILVER_MODEL=qwen-silver \
  SILVER_MODEL_REVISION=<COMMIT_OU_DIGEST_DO_CHECKPOINT> \
  SILVER_DATASET=/caminho/calibration.csv \
  SILVER_LIMIT=10
```

Se a saída estiver correta, repita sem `SILVER_LIMIT`. O mesmo comando retoma
uma execução interrompida e não reenvia os exemplos que já têm resposta válida.

### Alternativa: avaliação com Ollama

O fluxo `make eval-run` abaixo continua disponível para comparar modelos que já
estejam no Ollama, mas ele é separado da campanha produzida por
`make silver-qwen`.

#### Instalar e iniciar o Ollama

No Pod:

```bash
nvidia-smi
curl -fsSL https://ollama.com/install.sh | sh
mkdir -p /workspace/ollama-models
export OLLAMA_MODELS=/workspace/ollama-models
ollama serve
```

Mantenha `ollama serve` em primeiro plano. Em um segundo terminal SSH no mesmo
Pod:

```bash
export OLLAMA_MODELS=/workspace/ollama-models
ollama pull qwen2.5:72b-instruct-q4_K_M
ollama list
ollama show qwen2.5:72b-instruct-q4_K_M
```

O comando de instalação e o servidor acima seguem a
[documentação Linux do Ollama](https://docs.ollama.com/linux). Registre a
versão do Ollama e o digest mostrado por `ollama list`; a tag sozinha não é
uma identificação imutável.

#### Conectar sem publicar o Ollama

Adicione localmente um alias ao `~/.ssh/config`. Use os valores exibidos pela
RunPod e nunca os coloque neste repositório:

```sshconfig
Host runpod-qwen
    HostName <IP_DO_POD>
    User root
    Port <PORTA_TCP_SSH>
    IdentityFile <CAMINHO_DA_CHAVE>
```

Em um terminal local, abra o túnel:

```bash
make eval-tunnel EVAL_SSH_HOST=runpod-qwen EVAL_LOCAL_PORT=18001
```

Em outro terminal, valide a ponta remota e confirme o nome exato do modelo:

```bash
make eval-check EVAL_LOCAL_PORT=18001
make eval-models EVAL_LOCAL_PORT=18001
```

#### Executar a avaliação no Ollama

Comece com 10 exemplos de calibração:

```bash
make eval-run \
  EVAL_LOCAL_PORT=18001 \
  EVAL_MODEL='qwen2.5:72b-instruct-q4_K_M' \
  EVAL_DATASET=calibration \
  EVAL_LIMIT=10
```

Se o smoke estiver correto, rode a calibração completa. Ajuste prompt e
parâmetros somente nela. Depois congele a configuração e gere treino e teste:

```bash
make eval-run EVAL_LOCAL_PORT=18001 EVAL_MODEL='qwen2.5:72b-instruct-q4_K_M' EVAL_DATASET=calibration
make eval-run EVAL_LOCAL_PORT=18001 EVAL_MODEL='qwen2.5:72b-instruct-q4_K_M' EVAL_DATASET=train
make eval-run EVAL_LOCAL_PORT=18001 EVAL_MODEL='qwen2.5:72b-instruct-q4_K_M' EVAL_DATASET=test
make eval-consolidate
```

As inferências permanecem sequenciais, conforme a metodologia do repositório.
Pare o Pod assim que a campanha acabar. Mantenha o volume apenas se o custo de
armazenamento for menor que o custo/tempo de baixar novamente os 47 GB.

## 2. Gerar a silver GPT com custo controlado

O notebook é
[`analytics/notebooks/generate_mopep_openai_silver.ipynb`](../analytics/notebooks/generate_mopep_openai_silver.ipynb).
Prepare o ambiente e abra-o:

```bash
make setup-notebook
make openai-silver-notebook
```

Crie um projeto separado na OpenAI Platform, gere uma chave desse projeto e
configure um limite mensal rígido e um alerta. Exporte a chave apenas no shell:

```bash
export OPENAI_API_KEY='<TOKEN_DO_PROJETO>'
```

O notebook usa:

- `gpt-5-nano`, a opção de menor custo da família GPT-5;
- endpoint `/v1/responses`;
- Batch API, com processamento assíncrono;
- `reasoning.effort=minimal`;
- Structured Outputs com enum fechado nas 20 tags;
- `max_output_tokens=256`;
- `store=false`;
- `SUBMIT_BATCHES=False` por padrão;
- trava local `MAX_TOTAL_USD=2.00`.

Com os datasets atuais, a estimativa prática e o teto local calculados em
27/09/2026 são:

| split | exemplos | estimativa | teto local |
|---|---:|---:|---:|
| train | 2.683 | US$ 0,5817 | US$ 1,1419 |
| calibration | 574 | US$ 0,1239 | US$ 0,2428 |
| test | 575 | US$ 0,1247 | US$ 0,2448 |
| total | 3.832 | US$ 0,8303 | US$ 1,6295 |

A estimativa de entrada usa `bytes/3`; o teto usa a hipótese deliberadamente
folgada de um token por byte. Ambas cobram todos os tokens de saída permitidos
e não presumem cache. Confira os preços novamente antes de submeter. O custo
real é recalculado com `usage` quando as respostas são baixadas.

Execute o notebook nesta ordem:

1. prepare os JSONL e revise a tabela de custo;
2. confira o hard spend limit do projeto;
3. mude `SUBMIT_BATCHES=True` e execute a célula de submissão uma vez;
4. volte `SUBMIT_BATCHES=False`;
5. reexecute a célula de status até os batches terminarem;
6. baixe e materialize os resultados;
7. confira o resumo dos três splits.

O registro local de `batch_id` permite retomar o notebook sem submeter as mesmas
requisições novamente. O ID de campanha inclui modelo, parâmetros e hashes dos
três CSVs, evitando reaproveitamento acidental quando uma entrada muda. A Batch
API pode levar até 24 horas.

> Atenção: `gpt-5-nano` está disponível na data deste guia, mas sua versão
> original tem desligamento anunciado para 11/12/2026. Preserve o modelo pedido
> para esta campanha e planeje uma campanha de migração separada; não troque o
> modelo no meio de uma comparação.

Referências oficiais:

- [GPT-5 nano e preços](https://developers.openai.com/api/docs/models/gpt-5-nano);
- [Batch API](https://developers.openai.com/api/docs/guides/batch);
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs);
- [limites rígidos de gasto](https://developers.openai.com/api/docs/guides/spend-limits);
- [deprecações](https://developers.openai.com/api/docs/deprecations).

## 3. Construir o golden por consenso dos três modelos

O notebook faz o `join` do baseline GPT-4o e das duas silvers por `example_id`.
Antes de publicar a campanha, verifique também o hash da entrada: um ID igual
com texto diferente é erro de dados, não um caso para o consenso resolver.

Para cada exemplo e cada uma das 20 tags, cada modelo concede no máximo um
voto. A tag entra em `golden_tags` quando recebe pelo menos dois votos. A regra
é identificada como `majority_vote_2_of_3`; a proveniência da campanha deve
preservar também as versões das três fontes.

O notebook gera:

- `/tmp/mopep-bmc-private/identity.csv`, ligação privada entre ID e domínio;
- `analytics/results/mopep-golden-shareable/train.csv`;
- `analytics/results/mopep-golden-shareable/calibration.csv`;
- `analytics/results/mopep-golden-shareable/test.csv`.

Os arquivos compartilháveis contêm somente `id`, `business_model` e `tags`.
Textos, identidades e predições brutas continuam fora do Git.

A calibração serve para validar o procedimento e alinhar critérios antes da
medição final. Depois de congelar modelos, prompts, taxonomia e processo de
curadoria, avalie GPT-4o e os candidatos uma única vez no **test**. Mudanças
feitas depois de olhar o teste exigem uma nova campanha e outro split final.

Consenso entre modelos não prova que o rótulo está correto. Ele é o contrato de
curadoria adotado pelo projeto; qualquer correção manual posterior precisa
produzir outra versão, com justificativa e proveniência.
