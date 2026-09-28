# Gerar e comparar GGUFs com iMatrix no RunPod

> **Objetivo:** usar o `train.csv` privado para gerar uma importance matrix do
> llama.cpp e produzir três arquivos GGUF únicos do Qwen2.5-7B: `Q4_K_M` sem
> iMatrix, `Q4_K_M` com iMatrix e `Q8_0` como referência. Depois, carregar cada
> arquivo em vLLM, llama.cpp e Ollama, sempre uma execução por vez.

## 1. O desenho do experimento

| Variante | Usa iMatrix | Papel na comparação |
|---|---:|---|
| `Q4_K_M` | não | controle de mesma precisão |
| `Q4_K_M-imatrix` | sim | tratamento calibrado pelo `train.csv` |
| `Q8_0` | não | referência de maior precisão/tamanho |

A comparação que isola o efeito da calibração é `Q4_K_M` contra
`Q4_K_M-imatrix`. Comparar Q8 com Q4 mistura dois fatores: precisão e uso da
iMatrix. O quantizador Q8_0 do llama.cpp não aplica os pesos da iMatrix, por
isso não existe um quarto arquivo “Q8 com iMatrix” neste experimento.

A iMatrix mede ativações no corpus; ela não treina o modelo nem grava o CSV no
GGUF. O corpus contém o prompt MOPEP, o BMC e a resposta golden renderizados
pelo chat template do Qwen. Ele permanece privado em
`/workspace/data/quantization/mopep-imatrix` e não deve ser publicado.

Os três arquivos finais são GGUF. O checkpoint Hugging Face em Safetensors é
somente a fonte local da conversão; o intermediário de alta precisão também é
GGUF (`BF16.gguf`). Nunca converta um GPTQ, AWQ ou outro modelo já quantizado.

## 2. Requisitos do Pod

Neste guia, “Pod vazio” significa `/workspace` vazio sobre uma imagem de
runtime preparada; não significa um Ubuntu mínimo sem Python, CUDA, vLLM e
Ollama. Use preferencialmente Ubuntu 24.04 com Python 3.12 e uma imagem NVIDIA
**CUDA devel**, não apenas runtime. Ela precisa ter:

- Python 3.11, 3.12 ou 3.13;
- driver NVIDIA compatível com o toolkit da imagem;
- `nvcc`, `cmake`, `g++`, Git, GNU Make, `venv`, `flock` e `tmux`;
- vLLM, Ollama e espaço para instalar o cliente de benchmark;
- armazenamento persistente em `/workspace`.

Uma imagem CUDA devel genérica basta para **gerar** os três GGUFs, mas a matriz
3×3 também exige que vLLM e Ollama já estejam instalados. O template de runtime
do projeto é definido em
`pipelines/qwen-quantization/runpod-runtime/Dockerfile` e fornece esses dois
runtimes; o repositório não declara uma imagem pública pré-construída. Se o Pod
atual não os tiver, conclua o build e a verificação, mas não inicie a matriz até
recriar/preparar o Pod com esse Dockerfile.
Em Ubuntu 22.04, instalar apenas `python3-venv` normalmente mantém Python 3.10;
nesse caso troque a imagem em vez de seguir com uma versão incompatível.

Para o 7B, comece com pelo menos 24 GB de VRAM, 64 GB de RAM e cerca de
80–100 GB livres em `/workspace`. Isso é margem operacional, não garantia: o
checkpoint original, o BF16 GGUF, a iMatrix, três saídas e ambientes coexistem
durante o build. Se `nvcc` não existir, troque o Pod para uma imagem devel; não
instale às cegas um toolkit Ubuntu diferente do driver da imagem.

Em uma imagem Ubuntu devel, instale apenas os utilitários ausentes:

```bash
apt-get update
apt-get install -y \
  git make cmake build-essential python3-venv ca-certificates \
  procps tar coreutils util-linux tmux
python3 --version
nvcc --version
nvidia-smi
df -h /workspace
free -h
```

## 3. Obter esta versão do projeto

No Pod:

```bash
cd /workspace
git clone --branch codex/runpod-quantization --single-branch \
  https://github.com/thiago-ouverney/finance-agent-platform.git
cd /workspace/finance-agent-platform
git rev-parse HEAD
test -f pipelines/qwen-quantization/build_gguf_comparison.py
test -f services/inference-runtime/Makefile.gguf-comparison
```

Se o repositório for privado, `HF_TOKEN` não autentica o GitHub. Use uma
deploy key somente de leitura e clone por SSH; não coloque PAT em URL ou
histórico de shell.

## 4. Criar a configuração local

O Qwen usado aqui é público: nenhum token Hugging Face é necessário para o
build. No Pod:

```bash
cp pipelines/qwen-quantization/quantization.env.example \
  /workspace/quantization.env
chmod 600 /workspace/quantization.env
export QUANT_ENV_FILE=/workspace/quantization.env

make -C pipelines/qwen-quantization gguf-print-config
make -C pipelines/qwen-quantization gguf-doctor
```

O arquivo fixa o commit do llama.cpp e mantém estes parâmetros:

```text
modelo: Qwen/Qwen2.5-7B-Instruct
dataset: /workspace/data/mopep-golden/train.csv
amostras: 256
seed: 42
contexto da iMatrix: 512
saída: /workspace/models/Qwen2.5-7B-Instruct-GGUF-comparison
```

Não adicione `HF_TOKEN` ao arquivo. Se a fonte for privada, injete o token como
Secret do RunPod ou use `read -rsp`/`export` somente durante o download.

## 5. Preparar ferramentas antes de transferir o dado privado

Use uma sessão persistente:

```bash
tmux new -s gguf-imatrix
cd /workspace/finance-agent-platform
export QUANT_ENV_FILE=/workspace/quantization.env

make -C pipelines/qwen-quantization gguf-setup
```

Esse alvo cria `/workspace/venvs/qwen-gguf-imatrix`, busca exatamente o commit
fixado do llama.cpp e instala as dependências oficiais do conversor em um venv
separado. Ele ainda não compila nem quantiza. Para reconectar:

```bash
tmux attach -t gguf-imatrix
```

## 6. Enviar o golden novo para o Pod

Na máquina local, em outro terminal e na raiz do checkout:

```bash
chmod 600 \
  analytics/results/mopep-golden-shareable/train.csv

make push-imatrix-train REMOTE_HOST=runpod-qwen
```

Se a porta não estiver no alias SSH, acrescente
`QUANT_REMOTE_PORT='<porta>'`. O alvo transfere somente `train.csv`, com
staging, SHA-256 e permissão `600`. O loader recebe CSV UTF-8 local, então o
dataset não precisa estar no Hugging Face nem ser lido por pandas. Use
`make push-quantization-data` apenas quando também precisar dos splits de
calibração e teste para outras etapas.

Contrato do CSV: `id,business_model,tags`, com `tags` como lista JSON dos nomes
exatos da taxonomia. `example_id` também é aceito por compatibilidade. O alvo
recusa `test.csv` como calibração.

No Pod, confirme sem imprimir os BMCs:

```bash
cd /workspace/finance-agent-platform
export QUANT_ENV_FILE=/workspace/quantization.env
make -C pipelines/qwen-quantization gguf-data-check
sha256sum /workspace/data/mopep-golden/train.csv
```

## 7. Testar o encanamento com Qwen 0.5B

Antes de gastar horas no 7B, execute uma quantização completa pequena. Ela usa
o Qwen2.5-0.5B, 8 exemplos do mesmo `train.csv` e gera os mesmos três tipos de
GGUF em uma pasta isolada:

```bash
mkdir -p /workspace/logs
set -o pipefail
make -C pipelines/qwen-quantization gguf-smoke-build \
  2>&1 | tee /workspace/logs/gguf-imatrix-smoke.log
```

O alvo termina verificando os três arquivos em
`/workspace/models/Qwen2.5-0.5B-Instruct-GGUF-imatrix-smoke`. Esse smoke prova
download, conversor, corpus, `llama-imatrix`, quantizador e retomada; ele não
prova que o 7B cabe na máquina nem serve como resultado de qualidade.

## 8. Gerar os três GGUFs finais do 7B

Ainda no `tmux`:

```bash
mkdir -p /workspace/logs
set -o pipefail
make -C pipelines/qwen-quantization gguf-build-comparison \
  2>&1 | tee /workspace/logs/gguf-imatrix-build.log
```

O alvo executa, em sequência:

1. fixa o seletor do modelo em um commit imutável e baixa a fonte original;
2. compila `llama-imatrix`, `llama-quantize` e `llama-server` com CUDA;
3. seleciona 256 exemplos do `train.csv` de forma determinística;
4. aplica o chat template local ao prompt e à resposta golden;
5. converte a fonte uma única vez para BF16 GGUF;
6. calcula a iMatrix com `--parse-special`, contexto 512 e offload GPU;
7. gera os dois Q4 e o Q8 serialmente a partir do mesmo BF16;
8. grava progresso fail-closed por etapa, comandos, revisões, parâmetros,
   tamanhos e hashes no manifesto.

O lock global impede outra quantização/verificação simultânea no Pod. Não abra
dois builds, mesmo em GPUs diferentes.

Uma saída `.partial` é preservada quando uma ferramenta falha e o pipeline não
a sobrescreve. Inspecione o log e mova conscientemente esse arquivo para uma
pasta de diagnóstico antes de tentar novamente; não apague o diretório inteiro
sem verificar quais etapas concluíram. No raro caso de queda entre a publicação
de um GGUF e a gravação do manifesto de progresso, a retomada também para em
modo fail-closed; mova o arquivo “sem hash de progresso” para diagnóstico antes
de repetir a etapa.

## 9. Verificar e identificar os arquivos finais

```bash
make -C pipelines/qwen-quantization gguf-verify-comparison

GGUF_DIR=/workspace/models/Qwen2.5-7B-Instruct-GGUF-comparison
sha256sum \
  "$GGUF_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  "$GGUF_DIR/Qwen2.5-7B-Instruct-Q4_K_M-imatrix.gguf" \
  "$GGUF_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"

find "$GGUF_DIR" -maxdepth 1 -name '*.safetensors' -print
```

O último comando não deve imprimir nada. Os três modelos comparados são:

```text
/workspace/models/Qwen2.5-7B-Instruct-GGUF-comparison/Qwen2.5-7B-Instruct-Q4_K_M.gguf
/workspace/models/Qwen2.5-7B-Instruct-GGUF-comparison/Qwen2.5-7B-Instruct-Q4_K_M-imatrix.gguf
/workspace/models/Qwen2.5-7B-Instruct-GGUF-comparison/Qwen2.5-7B-Instruct-Q8_0.gguf
```

No mesmo diretório ficam o BF16 GGUF, a própria iMatrix e manifestos de
proveniência. Os runtimes recebem somente um dos três modelos finais; eles não
precisam do arquivo da iMatrix.

## 10. Preparar os três runtimes

O benchmark usa os pesos já no SSD e não baixa nada durante uma célula. Primeiro
instale o cliente:

```bash
cd /workspace/finance-agent-platform/services/inference-runtime
make install-benchmark
```

Use o `llama-server` compilado no mesmo commit que gerou os arquivos:

```bash
export LLAMA_SERVER_BIN=/workspace/llama.cpp-imatrix/build/bin/llama-server
export LLAMA_BACKEND_PATH=
test -x "$LLAMA_SERVER_BIN"
```

Confirme vLLM e Ollama:

```bash
export VLLM_BIN="$(command -v vllm)"
export VLLM_PYTHON=/app/.vllm_venv/bin/python
export OLLAMA_BIN="$(command -v ollama)"

test -x "$VLLM_BIN"
test -x "$VLLM_PYTHON"
test -x "$OLLAMA_BIN"
```

O template atual usa `/app/.vllm_venv`; se seu executável vier de outro venv,
aponte `VLLM_PYTHON` para o Python daquele mesmo ambiente. O suporte GGUF do
vLLM exige o plugin externo. Instale-o fora da medição, no ambiente do vLLM,
fixando o commit oficial usado por esta campanha:

```bash
PLUGIN_DIR=/workspace/vllm-gguf-plugin
PLUGIN_REV=e2b8ad532b8b5ea175100202c30430c1d2b5e6a8

git clone https://github.com/vllm-project/vllm-gguf-plugin.git "$PLUGIN_DIR"
git -C "$PLUGIN_DIR" checkout --detach "$PLUGIN_REV"
"$VLLM_PYTHON" -m pip install --no-build-isolation "$PLUGIN_DIR"
"$VLLM_PYTHON" -c \
  'import importlib.metadata as m; import vllm, vllm_gguf_plugin; print("vllm", m.version("vllm")); print("vllm-gguf-plugin", m.version("vllm-gguf-plugin"))'
```

Se `PLUGIN_DIR` já existir, confira o `git rev-parse HEAD` em vez de cloná-lo
novamente. A cobertura oficial conhecida do plugin testa Qwen2.5 com `Q6_K`,
não garante Qwen2.5-7B em `Q4_K_M` nem em `Q8_0`; o smoke é o gate. A instalação
pode falhar se Torch/vLLM do template forem incompatíveis com o commit pinado.
Registre as versões impressas e não faça upgrade durante a campanha. Uma falha
do vLLM deve permanecer como `failed/unsupported`, sem trocar o arquivo por
Safetensors.

Cheque os executáveis:

```bash
make check-runtimes \
  VLLM_BIN="$VLLM_BIN" VLLM_PYTHON="$VLLM_PYTHON" \
  LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" LLAMA_BACKEND_PATH= \
  OLLAMA_BIN="$OLLAMA_BIN"
make check-vllm-gguf VLLM_BIN="$VLLM_BIN" VLLM_PYTHON="$VLLM_PYTHON"
```

Antes da matriz, as portas `8000`, `8080` e `11434` precisam estar livres. Se
um servidor já estiver ativo, encerre-o pelo gerenciador que o iniciou; não use
um `pkill` amplo em um Pod compartilhado. Esta checagem somente informa o
estado:

```bash
python3 - <<'PY'
import socket

for port in (8000, 8080, 11434):
    with socket.socket() as client:
        status = "ocupada" if client.connect_ex(("127.0.0.1", port)) == 0 else "livre"
    print(f"porta {port}: {status}")
PY
```

## 11. Rodar primeiro a matriz smoke 3×3

```bash
make -f Makefile.gguf-comparison gguf-comparison-smoke \
  VLLM_BIN="$VLLM_BIN" VLLM_PYTHON="$VLLM_PYTHON" \
  LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" LLAMA_BACKEND_PATH= \
  OLLAMA_BIN="$OLLAMA_BIN"
```

O runner adquire primeiro um lock global do Pod e, dentro dele, manda o renderer
criar perfis exclusivos daquela campanha. O manifesto de origem é obrigatório;
cada perfil registra caminho absoluto, alias, SHA do GGUF e hash dos metadados
do tokenizer. Antes de cada campanha esses vínculos são revalidados. O runner
prepara os três aliases do Ollama e executa nove células estritamente em
sequência. Ele continua após uma falha para mostrar a matriz completa, mas só
marca uma célula como completa quando existe um `manifest.json` de resultado
com runtime, alias e hash esperados.

O manifesto da campanha fica em:

```text
services/inference-runtime/results/gguf-comparison/<timestamp>-smoke/campaign.json
```

Confira `status`, `gguf_sha256`, `runtime`, `variant` e os logs de cada célula.
Não trate smoke como resultado de performance.

## 12. Rodar a comparação formal

Somente depois do smoke:

```bash
make -f Makefile.gguf-comparison gguf-comparison-bench \
  VLLM_BIN="$VLLM_BIN" VLLM_PYTHON="$VLLM_PYTHON" \
  LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" LLAMA_BACKEND_PATH= \
  OLLAMA_BIN="$OLLAMA_BIN" \
  BENCH_REQUESTS=50 BENCH_REPETITIONS=1 BENCH_WARMUP=3
```

A matriz formal executa a bateria base `short/medium/long` para as nove células,
sem o sweep de KV embutido dos alvos legados. Isso mantém a primeira comparação
focada em artefato × runtime. Sweep de contexto/KV é uma campanha posterior e
deve usar os mesmos hashes.

A ordem das nove células é balanceada para não alinhar sempre um runtime ou
modelo à mesma posição temporal. Ainda assim, uma única campanha não elimina
efeitos térmicos e de cache; para rigor estatístico, repita a campanha inteira
com uma ordem previamente registrada.

## 13. O que a matriz permite concluir

Ela compara carregamento, TTFT, latência, throughput e uso de recursos do mesmo
arquivo entre runtimes e das três variantes sob a mesma carga. Ela não prova,
sozinha, que a iMatrix melhorou a classificação MOPEP.

Para qualidade, avalie cada variante no `test.csv` reservado, com o mesmo
prompt, parâmetros e métrica Macro-F1. Não ajuste a quantização depois de olhar
o teste; use `calibration.csv` para escolher parâmetros e reserve uma nova
campanha de teste para a decisão final.

## 14. Encerrar com segurança

Este fluxo não publica os três GGUFs no Hugging Face. Não use o alvo legado de
upload de um único Q8, pois ele não representa nem valida o conjunto desta
campanha. A publicação do trio deve ser implementada separadamente, com o
manifesto e os três hashes preservados.

O corpus renderizado contém dados de treino e respostas golden:

```text
/workspace/data/quantization/mopep-imatrix/train-chat-template.txt
```

Mantenha-o com permissão `600`, não o mova para a pasta de upload e destrua o
volume quando não precisar mais dos dados. Se usou um token temporário:

```bash
unset HF_TOKEN
```

Pesos, corpus, resultados, logs e credenciais não devem ser commitados no Git.
