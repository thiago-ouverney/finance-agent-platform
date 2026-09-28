# Gerar um modelo quantizado no RunPod

> **Objetivo:** partir de um Pod vazio, transferir o golden privado, executar
> um smoke GPTQ com o Qwen 0.5B, quantizar o Qwen2.5-7B com dados do
> `train.csv`, verificar o checkpoint e publicá-lo de forma explícita no
> namespace `thiagoouverney` do Hugging Face.

Os builds GPTQ/AWQ/EXL3 e GGUF/iMatrix pertencem a
`pipelines/qwen-quantization`. O carregamento e a comparação dos GGUFs em
vLLM, llama.cpp e Ollama pertencem a `services/inference-runtime`.

Neste guia, “Pod vazio” significa `/workspace` vazio sobre uma imagem RunPod
NVIDIA/PyTorch já compatível com CUDA, não um sistema operacional mínimo sem
driver. O lock atual exige Python 3.11, 3.12 ou 3.13, driver NVIDIA ramo R580 ou
mais novo e instala Torch/CUDA 13. O `doctor` inspeciona versão do driver, GPU
e ferramentas, e o final do `setup`
confirma de forma decisiva que o Torch instalado enxerga CUDA.

Usar os rótulos durante a calibração ajuda a preservar as ativações do modelo
nesse domínio, mas não equivale a treinamento ou fine-tuning e não ensina tags
novas ao Qwen.

## 1. O que precisa existir fora do Pod

Na máquina local:

- o repositório com esta versão do pipeline já commitada e enviada ao remote
  que será clonado pelo Pod;
- `analytics/results/mopep-golden-shareable/train.csv`;
- `analytics/results/mopep-golden-shareable/calibration.csv`;
- `analytics/results/mopep-golden-shareable/test.csv`;
- o alias SSH `runpod-qwen` configurado;
- GNU Make, cliente `ssh`, `tar`, `sha256sum` e `mktemp` disponíveis.

Os CSVs são ignorados pelo Git e nunca devem ser adicionados ao repositório.
O clone feito no Pod não os conterá.

Restrinja também a leitura no host local antes da transferência:

```bash
chmod 600 \
  analytics/results/mopep-golden-shareable/train.csv \
  analytics/results/mopep-golden-shareable/calibration.csv \
  analytics/results/mopep-golden-shareable/test.csv
```

Um `git clone` também não recebe arquivos modificados ou novos que existam
somente no seu checkout local. Antes de abrir o Pod, confirme que o commit/branch
remoto contém `pipelines/qwen-quantization/Makefile`, `calibration.py`,
`verify_artifact.py` e `publish_model.py`; alternativamente sincronize o
checkout inteiro por um meio seguro, sem incluir credenciais ou os CSVs.

Use os splits desta forma:

| Arquivo | Uso |
|---|---|
| `train.csv` | calibração da quantização |
| `calibration.csv` | escolha de técnica e parâmetros |
| `test.csv` | avaliação final; o pipeline recusa usá-lo na quantização |

## 2. Criar o token do Hugging Face

Crie um [token fine-grained](https://huggingface.co/settings/tokens) com
permissão de escrita nos repositórios de modelo do usuário `thiagoouverney`. O
Qwen público pode ser baixado sem token; o token é obrigatório para
autenticação em repositórios privados e para upload.

Nunca coloque o token:

- no comando `make`;
- em `quantization.env`;
- no Git;
- em uma URL de clone;
- em mensagens ou logs compartilhados.

A opção preferencial é cadastrar no painel do RunPod um Secret/Environment
Variable chamado exatamente `HF_TOKEN`. Em uma sessão já aberta, configure sem
gravar o valor no histórico:

```bash
read -rsp 'Cole o HF_TOKEN com permissao de escrita: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

Como o Qwen usado neste guia é público, a opção de menor privilégio é executar
esses três comandos somente imediatamente antes da seção 9. Assim o token de
escrita não fica exposto aos subprocessos de instalação e quantização. Configure
o Secret desde a criação do Pod apenas se também precisar baixar uma fonte
privada.

Valide sem imprimir o segredo:

```bash
test -n "${HF_TOKEN:-}" && echo 'HF_TOKEN configurado'
```

O Makefile não carrega `.env` automaticamente. `HF_WRITE_TOKEN` e
`HF_READ_TOKEN` não substituem `HF_TOKEN` neste pipeline.

## 3. Preparar um Pod vazio

Use armazenamento persistente montado em `/workspace` e uma imagem com driver
NVIDIA R580+, Python 3.11–3.13, `git`, GNU Make, `gcc`, módulo
`venv`, certificados, `procps`, `tar` e `sha256sum`. Em uma imagem Ubuntu que já
tenha Python compatível, os utilitários ausentes podem ser instalados assim:

```bash
apt-get update
apt-get install -y git make build-essential python3-venv ca-certificates procps tar coreutils util-linux tmux
python3 --version
nvidia-smi
```

Como margem operacional inicial para o 7B, reserve pelo menos 80 GB livres em
`/workspace` para venv CUDA, modelo-base (cerca de 15 GB), saída, cache e
temporários/offload. É uma margem conservadora, não uma garantia; o `doctor`
mostra o espaço real antes do download.

Entre no Pod e clone a branch publicada para este fluxo:

```bash
cd /workspace
git clone --branch codex/runpod-quantization --single-branch \
  https://github.com/thiago-ouverney/finance-agent-platform.git
cd /workspace/finance-agent-platform
test -f pipelines/qwen-quantization/Makefile
git rev-parse HEAD
```

Depois que a branch for integrada, você pode trocar o seletor por `main`. Se o
repositório for privado, `HF_TOKEN` não autentica o GitHub: injete uma
chave/deploy key somente de leitura no Pod e use, por exemplo:

```bash
chmod 600 /workspace/secrets/github_deploy_key
GIT_SSH_COMMAND='ssh -i /workspace/secrets/github_deploy_key -o IdentitiesOnly=yes' \
  git clone --branch codex/runpod-quantization --single-branch \
  git@github.com:thiago-ouverney/finance-agent-platform.git
```

Não embuta um PAT na URL. Depois, diagnostique o host:

```bash
make quantization-doctor
```

O diagnóstico mostra GPU/VRAM, RAM, disco, dispositivo CUDA escolhido e apenas
se `HF_TOKEN` está definido.

Instale as dependências antes de copiar os dados privados. Como o lock CUDA é
grande, faça isso dentro de uma sessão persistente:

```bash
tmux new -s quantizacao
cd /workspace/finance-agent-platform
make quantization-setup
```

Ao reconectar, use `tmux attach -t quantizacao`. Essa ordem também evita expor
os CSVs aos subprocessos de instalação. O Qwen é público, portanto não exporte
o token de escrita nesta etapa.

## 4. Transferir o golden privado

Saia do Pod ou abra outro terminal local. Na raiz do repositório local:

```bash
make push-quantization-data \
  REMOTE_HOST=runpod-qwen
```

Se a porta já está em `~/.ssh/config` no alias `runpod-qwen`, não informe outra
variável. Caso precise sobrescrevê-la apenas para essa transferência, acrescente
`QUANT_REMOTE_PORT='<porta-ssh-do-pod>'`.

Sem alias, use os mesmos campos diretamente, sem colocar credenciais na linha
de comando:

```bash
make push-quantization-data \
  REMOTE_HOST='<usuario>@<host>' \
  QUANT_REMOTE_PORT='<porta>' \
  REMOTE_IDENTITY='<caminho-da-chave-ssh>'
```

O alvo envia somente os três CSVs por um stream SSH cifrado para:

```text
/workspace/data/mopep-golden/
```

O alvo monta um pacote temporário com os hashes locais, extrai primeiro em um
diretório de staging no Pod, valida os três SHA-256, aplica permissão `600` e só
então move os CSVs para o destino. Uma falha antes da validação não toca no
golden válido; a substituição final dos três arquivos ocorre somente depois dos
hashes passarem.

O dataset não precisa ser publicado no Hugging Face e o loader de calibração
não depende de pandas: ele lê um CSV UTF-8 local. Você pode baixar ou alterar os
dados na sua máquina, gerar os splits e transferi-los por esse alvo. O contrato
é `id,business_model,tags` (`example_id` também é aceito por compatibilidade),
com `tags` serializando uma lista de nomes exatos da taxonomia MOPEP. No Pod,
`QUANT_DATASET` deve apontar para o arquivo transferido; no fluxo padrão é
`/workspace/data/mopep-golden/train.csv`.

Volte ao terminal do Pod e confirme:

```bash
cd /workspace/finance-agent-platform
ls -lh /workspace/data/mopep-golden
sha256sum /workspace/data/mopep-golden/*.csv
```

## 5. Criar a configuração não secreta

No Pod:

```bash
cp \
  pipelines/qwen-quantization/quantization.env.example \
  /workspace/quantization.env
chmod 600 /workspace/quantization.env
export QUANT_ENV_FILE=/workspace/quantization.env
```

O arquivo usa sintaxe de variáveis Make e já aponta para:

```text
Qwen/Qwen2.5-7B-Instruct
/workspace/data/mopep-golden/train.csv
GPTQ 4-bit
256 exemplos
seed 42
thiagoouverney/Qwen2.5-7B-Instruct-GPTQ-4bit
repositório HF privado
```

Não adicione `HF_TOKEN` a esse arquivo.

## 6. Validar o ambiente e os dados

O ambiente já foi instalado antes da transferência. Ainda no Pod:

```bash
make quantization-data-check
```

O ambiente fica em `/workspace/venvs/qwen-quantization` e o cache do HF em
`/workspace/.cache/huggingface`. Cache do pip e temporários/offload do GPTQModel
também ficam no volume persistente (`/workspace/.cache/pip` e
`/workspace/tmp`), sem pressionar desnecessariamente o disco raiz. O check do
dataset exige pelo menos a quantidade solicitada e mostra somente contagem,
hash, hash do prompt, seed e assinatura da seleção; não imprime os BMCs. Em uma
nova sessão, restaure `export QUANT_ENV_FILE=/workspace/quantization.env`. Se
`HF_TOKEN` não for um Secret do RunPod, configure-o somente antes do upload com
`read -rsp`.

## 7. Primeiro teste: Qwen 0.5B, GPTQ 4-bit e 8 exemplos

Execute:

```bash
make smoke-quantize-model
```

Esse alvo:

1. resolve `main` para um commit imutável e registra o SHA;
2. baixa `Qwen/Qwen2.5-0.5B-Instruct`;
3. seleciona deterministicamente 8 exemplos do `train.csv`;
4. usa o prompt completo com as 20 definições MOPEP do avaliador Ollama,
   aplica o chat template do tokenizer e inclui a resposta golden como lista;
5. quantiza em GPTQ 4-bit, `group_size=128`, batch 1;
6. salva pesos, tokenizer e `quantization_manifest.json`;
7. confere configuração, inventário e SHA-256 de todos os shards, recarrega o
   checkpoint e executa uma inferência mínima de um token na GPU;
8. não publica nada.

Saída esperada:

```text
/workspace/models/Qwen2.5-0.5B-Instruct-GPTQ-4bit-smoke
```

Uma pasta de saída não vazia nunca é sobrescrita. Se um job anterior falhar,
inspecione e mova a pasta parcial ou defina outro caminho antes de tentar
novamente, por exemplo:

```bash
make smoke-quantize-model \
  QUANT_SMOKE_OUTPUT_DIR=/workspace/models/Qwen2.5-0.5B-Instruct-GPTQ-4bit-smoke-2
```

## 8. Quantizar o Qwen2.5-7B com 256 exemplos

Depois que o smoke passar, confira a configuração:

```bash
make -C pipelines/qwen-quantization print-config
```

Execute a quantização real, uma por vez na GPU:

```bash
make run-quantize-model
make verify-quantized-model
```

Saída padrão:

```text
/workspace/models/Qwen2.5-7B-Instruct-GPTQ-4bit
```

O manifesto registra o commit real do modelo-base, método, bits, group size,
batch, SHA-256 do CSV e do prompt, seed, quantidade e assinatura dos exemplos
selecionados, versões, CUDA/GPU, commit do pipeline e hashes dos pesos. O CSV
não é copiado para a pasta do modelo. O prompt escolhido é o contrato de lista
do avaliador Ollama; o fluxo OpenAI estruturado que responde
`{"tags": [...]}` continua sendo outro contrato.

Para mudar técnica ou GPU, altere `/workspace/quantization.env`, por exemplo:

```makefile
CUDA_VISIBLE_DEVICES=0
QUANT_METHOD=awq
QUANT_BITS=4
QUANT_OUTPUT_DIR=/workspace/models/Qwen2.5-7B-Instruct-AWQ-4bit
QUANT_HF_REPO_ID=thiagoouverney/Qwen2.5-7B-Instruct-AWQ-4bit
```

O caminho reproduzível e usado pelo smoke é GPTQ com backend Torch, que não
depende de compilação Marlin por `nvcc`. AWQ e EXL3 continuam experimentais,
exigem uma imagem CUDA *devel* com `nvcc`; AWQ aceita apenas 4 bits neste
pipeline.
Não execute duas quantizações simultaneamente no mesmo Pod, mesmo que ele tenha
mais de uma GPU. O GPTQModel 7.3.4 compartilha caches de extensões e processos
concorrentes podem disputar esses arquivos. Além do Make ser sequencial, o
pipeline usa o lock global `/workspace/locks/qwen-quantization.lock`; uma
segunda quantização ou verificação com reload falha até a primeira terminar.
Portanto, para este fluxo de 7B, paralelize apenas tarefas sem GPU e execute uma
quantização por vez.
O smoke de 0.5B comprova o encanamento, não que o 7B caberá na mesma VRAM. Use
o relatório de `make quantization-doctor` para decidir; em GPUs de 24 GB o 7B
pode ficar no limite por buffers e offload, portanto uma GPU maior é a opção
mais segura para a primeira execução.

Este fluxo aprova a integridade técnica do artefato (hash, configuração, reload
e uma inferência mínima), mas ainda não calcula Macro-F1 no `calibration.csv`
ou `test.csv`, nem compara a qualidade do modelo-base com a do quantizado. Até
esse alvo de avaliação ser integrado, trate o upload como artefato experimental
privado e não como promoção para produção. O reload deste guia usa GPTQModel;
compatibilidade com vLLM ou outro runtime deve ser validada separadamente antes
de mudar o runtime de produção.

## 9. Publicar somente depois da verificação

Se ainda não exportou `HF_TOKEN`, use agora o `read -rsp` da seção 2. Confirme a
identidade no Hub sem mostrar o token:

```bash
make -C pipelines/qwen-quantization check-hf-token
```

Publique a pasta completa em um repositório privado por padrão:

```bash
make upload-quantized-model
```

O alvo confere hashes, recarrega o checkpoint, executa a inferência mínima e
cria `thiagoouverney/Qwen2.5-7B-Instruct-GPTQ-4bit` como privado. Ele envia
somente a pasta do modelo e recusa CSVs privados ou logs inesperados. Golden,
caches e resultados externos não são enviados.

Por segurança, a publicação falha se o repositório já existir com visibilidade
diferente ou contiver arquivos. Prefira um repo novo. Para substituir
conscientemente um repo existente da mesma visibilidade e remover arquivos
remotos obsoletos, use:

```bash
make upload-quantized-model QUANT_HF_REPLACE_EXISTING=1
```

Para tornar um artefato público, tome essa decisão explicitamente e use
as duas confirmações abaixo. O pipeline nunca altera a visibilidade de um repo
existente:

```bash
make upload-quantized-model \
  QUANT_HF_PRIVATE=0 \
  QUANT_HF_ALLOW_PUBLIC=1
```

## 10. Executar com variáveis na linha de comando

O arquivo `QUANT_ENV_FILE` é recomendado para documentação, mas qualquer valor
pode ser sobrescrito de forma explícita:

```bash
make run-quantize-model \
  QUANT_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
  QUANT_MODEL_REVISION=main \
  QUANT_METHOD=gptq \
  QUANT_BITS=4 \
  QUANT_DATASET=/workspace/data/mopep-golden/train.csv \
  QUANT_SAMPLES=256 \
  QUANT_SEED=42 \
  QUANT_OUTPUT_DIR=/workspace/models/Qwen2.5-7B-Instruct-GPTQ-4bit
```

Mesmo quando `main` é informado como seletor, o download resolve primeiro o
commit atual e grava essa revisão imutável no modelo e no manifesto.

## 11. GGUF com e sem iMatrix

O pipeline agora possui um experimento GGUF separado que gera, a partir do
mesmo BF16, `Q4_K_M` sem iMatrix, `Q4_K_M` com iMatrix derivada do `train.csv`
e `Q8_0` como referência. Os três resultados comparados são arquivos GGUF
únicos e podem ser exercitados sequencialmente em vLLM, llama.cpp e Ollama.

Siga [o guia GGUF/iMatrix](gerar-gguf-imatrix.md). Não converta um GPTQ/AWQ
já quantizado para GGUF: o fluxo parte do checkpoint original BF16/FP16 para
evitar dupla quantização.

## 12. Encerrar a sessão com segurança

Se o token foi digitado apenas para este trabalho, remova-o da sessão:

```bash
unset HF_TOKEN
```

Um Secret do RunPod deve ser removido ou rotacionado no painel quando deixar de
ser necessário. Faça o mesmo com uma deploy key temporária do GitHub; ela é uma
credencial separada do `HF_TOKEN`. Os CSVs ficam no volume persistente para
novas execuções. Se o
trabalho terminou e você não quer mantê-los, remova somente os arquivos
conhecidos (ou destrua o volume pelo painel):

```bash
rm -f \
  /workspace/data/mopep-golden/train.csv \
  /workspace/data/mopep-golden/calibration.csv \
  /workspace/data/mopep-golden/test.csv
```
