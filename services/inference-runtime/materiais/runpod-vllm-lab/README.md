# Seu primeiro Qwen no RunPod com vLLM

**Artefato único deste roteiro:** [Qwen2.5-14B-Instruct-Q8_0.gguf](https://huggingface.co/arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf). No vLLM CUDA, instale `vllm-gguf-plugin` e use o arquivo GGUF local com `--tokenizer` do Qwen2.5. A documentação do vLLM classifica GGUF como experimental; se o servidor não iniciar, preserve o erro, investigue o ambiente e marque a execução como falha. Não troque o artefato.

**Como o benchmark mede o carregamento:** use `--launch` com o servidor parado. O intervalo processo → `/v1/models` mede prontidão HTTP; processo → primeiro conteúdo da primeira geração é o indicador de carregamento efetivo, incluindo pesos locais, alocação e kernels. Sem `--launch`, o carregamento não é conhecido. Para llama.cpp use `llama-server -m ...`; para Ollama use `ollama serve` com preload local mantido em foreground; para vLLM use `vllm serve ...`. O download precisa ocorrer antes do processo.

Guia do Yan · RTX 3090 / 24 GB · conferido em 16/09/2026.

**Objetivo de hoje: receber uma resposta do modelo.** Depois, observar a máquina e mudar uma coisa por vez. Não há uma configuração “vencedora” aqui: os limites iniciais só deixam o primeiro teste pequeno e previsível.

**Para estudar os conceitos:** abra o [caderno interativo de inferência](estudo.html). Ele explica RAM, VRAM, HBM, prefill/decode, Roofline, PCIe e KV cache, com simuladores e uma trilha de experimentos para este modelo no vLLM.

Abra `index.html` para a versão de leitura com navegação e botões de copiar. Este Markdown contém o mesmo roteiro. Os comandos abaixo são para você executar; nenhum Pod foi criado e nenhuma inferência foi executada durante a elaboração do guia.

**Atualização após o primeiro teste do Yan:** a importação falhou com `libcudart.so.13`. O comando original não selecionava a variante CUDA do vLLM corretamente. Se você já chegou a esse erro, vá à **seção 3.1** antes de instalar qualquer coisa novamente. A correção no Pod ainda não foi validada; faltam as saídas de diagnóstico.

### Como ler os comandos

O prefixo `(.venv) root@...:/workspace/yan-vllm#` é o prompt do terminal: não o copie como parte do comando. `.venv` indica o ambiente Python ativo; `root` é o usuário; o caminho é a pasta atual. Uma barra `\` no fim da linha continua o mesmo comando na linha seguinte. Não acrescente espaços depois dela. As explicações ficam fora dos blocos para você poder copiá-los inteiros.

## 1. O que você vai montar

Seu navegador abre um terminal Linux no RunPod. Nesse Linux, o vLLM carrega o modelo na GPU e oferece uma API HTTP. Um segundo terminal no mesmo Pod envia as perguntas. Seu Mac apenas controla a máquina remota.

Modelo deste roteiro: [Qwen2.5-14B-Instruct-Q8_0.gguf](https://huggingface.co/arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf). Registre o SHA-256 do arquivo, tokenizer e template de chat. Essa inspeção prepara o comando; o carregamento no Pod ainda precisa ser confirmado.

Este roteiro começa do zero com o artefato GGUF acima. Se já houver um servidor na porta 8000, encerre-o antes de iniciar o experimento. Um erro como `libcudart.so.13` deve ser registrado e investigado na seção 3.1; ele não deve ser mascarado por outra instalação ou modelo.

“8 bits” descreve os pesos quantizados. Ativações, alguns tensores e KV cache ainda ocupam memória em outra precisão. O parâmetro `--dtype half` que usaremos não transforma esses pesos em um modelo FP16 completo.

O 14B em FP16 precisa de aproximadamente 29,4 GB decimais só para 14,7 bilhões de parâmetros a 2 bytes. Portanto, não cabe integralmente na 3090 de 24 GB. O GGUF Q8_0 pode exigir mais memória que a disponível conforme contexto e buffers; se não carregar, registre OOM e não reduza a comparabilidade trocando de artefato.

## 2. Criar o Pod

**No navegador, no RunPod:**

1. Crie sua conta, adicione crédito e abra **Pods → Deploy**. Anote a tarifa exibida antes de iniciar.
2. Escolha **1 × RTX 3090, 24 GB**. Para aprender, prefira **On-Demand**; uma instância interrompível pode parar sua sessão.
3. Use um template oficial **RunPod PyTorch**, com Ubuntu recente e Python. Vamos instalar o vLLM num ambiente separado dentro dele.
4. Nos filtros, procure um host compatível com **CUDA 12.9 ou superior**, para seguir a instalação deste guia. A GPU ser 3090, por si só, não informa a versão do driver do host.
5. Sugestão de capacidade para este laboratório: **32 GB ou mais de RAM**, idealmente 4 ou mais vCPUs, **30 GB de container disk** e **100 GB de volume disk**, montado em `/workspace`. São margens práticas para instalação, cache e logs, não mínimos oficiais.
6. Dê um nome como `yan-vllm-3090`, revise GPU, armazenamento e custo, e faça o deploy.
7. Quando estiver pronto, entre em **Connect → Web Terminal**; habilite o terminal se a interface pedir. Alternativamente, use o terminal do JupyterLab do template.

Os nomes exatos dos botões podem mudar. Consulte [deploy e gerenciamento](https://docs.runpod.io/pods/manage-pods) e [opções de conexão](https://docs.runpod.io/pods/connect-to-a-pod). Nesta primeira sessão, não é necessário expor a porta 8000 à internet.

**Ponto de conferência:** você deve estar num terminal Linux remoto, não no terminal local do Mac. Execute:

```bash
uname -m
nvidia-smi
python3 --version
df -h /workspace /
free -h
```

| Comando | O que faz e por que usamos |
| --- | --- |
| `uname -m` | Mostra a arquitetura da máquina. `x86_64` corresponde ao pacote binário usado neste roteiro. |
| `nvidia-smi` | Consulta o driver NVIDIA: GPU, driver, memória, temperatura e processos visíveis. Não instala nem modifica nada. |
| `python3 --version` | Mostra a versão do Python que criará o ambiente virtual. |
| `df -h /workspace /` | Mostra espaço total, ocupado e livre nos sistemas de arquivos desses caminhos. `-h` usa unidades legíveis. Volume e container podem ter limites diferentes. |
| `free -h` | Mostra RAM e swap visíveis. Observe especialmente `available`; RAM é diferente da VRAM mostrada pelo NVIDIA. |

Espere `x86_64`, uma RTX 3090 e aproximadamente 24 GB de VRAM. A linha “CUDA Version” do `nvidia-smi` indica a versão máxima suportada pelo driver, não o toolkit instalado. Se estiver abaixo de 12.9, pare aqui e traga essa saída: ajustaremos a combinação de wheel/driver. Não tente instalar um driver NVIDIA dentro do Pod.

## 3. Instalar o ambiente

**Terminal A, dentro do Pod.** Copie um bloco por vez e só avance se ele terminar sem erro.

### 3.1. Se apareceu `libcudart.so.13`: diagnosticar primeiro

O traceback enviado pelo Yan termina ao importar uma extensão binária do vLLM. Ela procura `libcudart.so.13`, uma biblioteca do runtime CUDA 13, que o carregador não encontrou. Isso acontece antes de carregar o Qwen e não tem relação com tamanho do prompt ou falta de VRAM do modelo.

**Correção do guia:** o pacote padrão do vLLM 0.29.0 no PyPI usa CUDA 13.0, conforme a [release oficial](https://github.com/vllm-project/vllm/releases/tag/v0.29.0). Acrescentar um índice de PyTorch `cu129` não seleciona automaticamente um vLLM compilado para CUDA 12.9. A instrução anterior podia combinar variantes incompatíveis.

Execute estes comandos no ambiente que apresentou o erro e envie suas saídas antes de tentar recuperá-lo:

```bash
nvidia-smi
python -m pip list | grep -Ei 'vllm|torch|nvidia|cuda'
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA do PyTorch:", torch.version.cuda); print("GPU disponível:", torch.cuda.is_available())'
```

| Comando ou trecho | O que revela |
| --- | --- |
| `nvidia-smi` | Qual GPU e driver o Pod recebeu, independentemente de conseguir importar o vLLM. |
| `python -m pip list` | Lista nomes e versões dos pacotes do Python ativo. `python -m pip` evita chamar por engano o pip de outro ambiente. |
| `\| grep -Ei 'vllm\|torch\|nvidia\|cuda'` | O pipe envia a lista ao filtro. `-E` permite alternativas separadas por barras verticais; `-i` ignora maiúsculas. Mostra apenas as dependências relevantes. |
| `python -c '...'` | Executa um pequeno programa Python escrito entre aspas, sem criar arquivo. |
| `import torch` | Testa se o PyTorch sozinho carrega. Não importa o vLLM, que já sabemos que falhou. |
| `torch.__version__` | Informa a versão instalada do PyTorch. |
| `torch.version.cuda` | Informa a versão CUDA associada à compilação do PyTorch, não a versão máxima do driver. |
| `torch.cuda.is_available()` | Testa se o PyTorch consegue disponibilizar CUDA; esperamos `True`. |

Se o PyTorch funcionar mas o vLLM falhar, isso restringe a investigação à combinação de pacotes/extensões e à busca de bibliotecas. Não prova sozinho qual pacote precisa ser substituído. O erro também não prova que o driver é antigo. Não crie um link de `libcudart.so.13` para `libcudart.so.12`: nomes diferentes podem representar interfaces binárias incompatíveis.

**Estado atual:** diagnóstico pendente. Os comandos seguintes descrevem uma instalação nova, não uma ordem para reinstalar por cima do ambiente com erro.

### 3.2. Preparar um ambiente novo

```bash
mkdir -p /workspace/yan-vllm/logs /workspace/yan-vllm/resultados
cd /workspace/yan-vllm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

| Comando | Explicação |
| --- | --- |
| `mkdir -p .../logs .../resultados` | Cria as pastas e os diretórios intermediários. `-p` aceita pastas que já existam. |
| `cd /workspace/yan-vllm` | Define a pasta de trabalho; caminhos relativos como `.venv` passam a partir dela. |
| `python3 -m venv .venv` | Cria um ambiente Python isolado. Não isola o driver da GPU nem cria outra máquina. Use numa instalação nova; não conserta automaticamente um ambiente existente. |
| `source .venv/bin/activate` | Ajusta o terminal atual para usar o Python e os executáveis desse ambiente. Cada terminal novo precisa ser ativado separadamente. |
| `python -m pip install --upgrade pip` | Atualiza o instalador de pacotes dentro do ambiente ativo. Não atualiza o driver NVIDIA. |

Use Python 3.10–3.13 neste roteiro, preferencialmente 3.12. Se `venv` estiver ausente num template Ubuntu, execute `apt-get update` e `apt-get install -y python3-venv` como root, e repita a criação. Se o template tiver Python fora dessa faixa, escolha um template adequado antes de continuar.

`apt-get update` atualiza o catálogo de pacotes do Ubuntu; não atualiza todos os programas. `apt-get install -y python3-venv` instala o suporte a ambientes virtuais e aceita automaticamente as confirmações do instalador. Esses comandos alteram o container, não apenas a `.venv`.

### 3.3. Instalar a variante CUDA explicitamente

Para uma instalação nova em Linux `x86_64`, com driver compatível com CUDA 12.9, use o arquivo **`+cu129`** publicado na release. Um wheel é um pacote Python pré-compilado; aqui ele contém código nativo dependente de CUDA.

```bash
export HF_HOME=/workspace/yan-vllm/hf-cache
export PIP_CACHE_DIR=/workspace/yan-vllm/pip-cache
python -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu129
python -m pip check
```

A URL fixa a versão e a variante do **vLLM**, ao contrário do comando anterior. `%2B` é a representação de `+` na URL. O índice adicional oferece dependências PyTorch CUDA 12.9, mas não é um seletor universal de variante: confira o resultado antes de avançar. Deixe o instalador resolver as dependências declaradas pelo wheel, sem atualizar PyTorch separadamente. Fonte do arquivo: [artefatos da release 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0).

| Comando | Explicação |
| --- | --- |
| `export HF_HOME=...` | Direciona o cache do Hugging Face para o volume em `/workspace`; ajuda a preservar os modelos entre sessões. Vale para este terminal e seus processos filhos. |
| `export PIP_CACHE_DIR=...` | Direciona o cache de downloads do pip para o volume. Não muda onde o pacote é instalado: continua na `.venv`. |
| `python -m pip install URL ...` | Baixa e instala o wheel escolhido e suas dependências. Esta operação altera o ambiente e consome disco/rede. |
| `--extra-index-url .../cu129` | Acrescenta uma fonte de pacotes ao índice padrão; não obriga todos os pacotes a vir dela. |
| `python -m pip check` | Verifica requisitos de versões declarados pelos pacotes. Não testa carregamento de bibliotecas CUDA; pode passar mesmo com o erro relatado. |

Confira que o ambiente enxerga a GPU:

```bash
python -c 'import torch, vllm; print("vLLM:", vllm.__version__); print("PyTorch:", torch.__version__); print("CUDA do PyTorch:", torch.version.cuda); print("GPU disponível:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0))'
python -m pip freeze > resultados/pacotes.txt
nvidia-smi > resultados/gpu-inicial.txt
vllm serve --help > resultados/serve-help.txt
```

**Ponto de conferência:** GPU disponível deve ser `True`, com a 3090 identificada. Um erro aqui é de ambiente; ainda não é hora de mexer em parâmetros do modelo.

O programa de conferência importa PyTorch e vLLM e imprime suas versões; `torch.cuda.get_device_name(0)` consulta a primeira GPU, cujo índice é zero. Se `import vllm` falhar, o Python interrompe o programa antes dos `print`: por isso o diagnóstico da seção 3.1 testa apenas PyTorch.

Os três comandos seguintes guardam evidências: `pip freeze` salva os pacotes e versões; `nvidia-smi` salva o estado da GPU; `vllm serve --help` salva a ajuda da versão instalada. **`>` redireciona a saída normal e cria ou sobrescreve o arquivo.** Erros continuam no terminal. Se `vllm serve --help > ...` falhar, o arquivo pode ficar vazio — exatamente por isso a ausência de texto nele não comprova sucesso.

## 4. Carregar o modelo e deixar o servidor ligado

Ainda no **Terminal A**, execute:

```bash
cd /workspace/yan-vllm
source .venv/bin/activate
export HF_HOME=/workspace/yan-vllm/hf-cache
set -o pipefail
vllm serve /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --served-model-name qwen14b-int8 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  2>&1 | tee "logs/arthur-$(date +%Y%m%d-%H%M%S).log"
```

| Trecho | O que faz |
| --- | --- |
| `cd`, `source`, `export HF_HOME` | Retomam a pasta, o ambiente Python e o cache; são os mesmos preparativos explicados na seção 3. |
| `set -o pipefail` | Faz a sequência com pipe sinalizar falha se o servidor falhar, mesmo que `tee` consiga gravar o log. Não reinicia o servidor automaticamente. |
| `vllm serve /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf` | Inicia o servidor usando o arquivo GGUF local. |
| `--served-model-name qwen14b-int8` | Define o nome curto que o cliente deve enviar na API. Não altera o modelo. |
| `--host 127.0.0.1 --port 8000` | Escuta na porta 8000 apenas na interface local do Pod. |
| `--dtype half` | Seleciona a precisão de cálculo aplicável ao backend; confirme o valor aceito para GGUF na versão instalada. |
| `--max-model-len 2048` | Limita o contexto total por sequência, somando entrada e saída. |
| `--max-num-seqs 1` | Permite uma sequência em execução por vez no agendador. |
| `2>&1` | Encaminha a saída de erros para a mesma saída dos logs normais. |
| `tee "logs/..."` | Mostra a saída no terminal e simultaneamente grava uma cópia em arquivo. |
| `$(date +%Y%m%d-%H%M%S)` | Executa `date` e insere ano, mês, dia, hora, minuto e segundo no nome do log. |

O primeiro início baixa muitos gigabytes, carrega pesos e prepara kernels. Isso pode levar vários minutos. O repositório é público; normalmente não precisa de token do Hugging Face. Download e inicialização não são o TTFT de uma requisição com o servidor aquecido.

O vLLM lê os metadados do GGUF conforme o plugin/backend instalado; não force um quantizador diferente. Observe no log o backend realmente escolhido.

Os dois limites explícitos são uma escolha didática: contexto total de **2048 tokens**, contando entrada e saída, e **uma sequência em execução**. As demais decisões de execução ficam nos padrões da versão. Esses padrões já incluem mecanismos de desempenho do próprio vLLM; este é um ponto de partida, não um motor sem otimizações internas.

Deixe o Terminal A aberto. Quando o log indicar que a aplicação iniciou, siga para o teste. Detalhes dos argumentos: [vLLM serve](https://docs.vllm.ai/en/v0.29.0/cli/serve/).

## 5. Fazer a primeira pergunta

**Prefere conversar pelo navegador?** Depois de iniciar o servidor, siga o [guia do chat em Python](CHAT.md). A interface tem histórico, respostas em streaming e formatação de texto. Os arquivos `chat.py` e `requirements-chat.txt` estão nesta pasta; o vLLM continua rodando como descrito acima.

Abra um **Terminal B no mesmo Pod**. Pode ser outro terminal no JupyterLab ou outra conexão. `127.0.0.1` abaixo aponta para o Pod, não para seu Mac.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

O primeiro comando pode responder sem texto e ainda estar correto; o segundo deve listar `qwen14b-int8`.

`curl` faz uma requisição HTTP. `-f` retorna falha para respostas HTTP de erro; `-s` remove a barra de progresso; `-S` mantém as mensagens de erro mesmo com `-s`. `/health` consulta a saúde do processo; `/v1/models` consulta os nomes de modelos disponíveis. São consultas, sem geração de texto.

```bash
curl -fN http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen14b-int8",
    "messages": [{"role": "user", "content": "Explique em português, em três frases, o que é o KV cache de um LLM."}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

Aqui, `-N` desativa o buffer de saída do curl para mostrar o streaming conforme chega. `-H` informa que o conteúdo enviado é JSON. `-d` envia esse conteúdo e faz o curl usar POST. `model` escolhe o alias; `messages` contém a conversa; `role: user` identifica sua fala; `content` é a pergunta. `max_tokens` limita a saída, `temperature: 0` pede seleção gulosa e `stream: true` solicita entrega progressiva. As aspas simples externas protegem o JSON da interpretação pelo shell.

Você verá eventos `data:` com pedaços da resposta. Esse é o formato de streaming da API, não um erro. **Se apareceu texto coerente, a meta da primeira sessão foi atingida.** Não precisa rodar o restante hoje.

Para uma resposta em JSON único, troque `"stream": true` por `false`. O campo `usage` ajuda a ver os tokens de entrada e saída. `max_tokens` é um teto: o modelo pode encerrar antes.

## 6. Entender o que vale mexer

Existem dois grupos: parâmetros da **requisição**, que você muda no JSON, e parâmetros do **servidor**, que exigem parar e iniciar o vLLM novamente. Não são hiperparâmetros de treinamento.

### Primeiro: a carga de trabalho

| Experimento | Como começar | O que observar |
| --- | --- | --- |
| Tamanho do prompt | Mesma pergunta com contexto curto, médio e longo | Quanto a espera inicial cresce com a entrada? |
| `max_tokens` | 64 → 128 → 256 | Tempo total e quantidade realmente gerada; não confundir mais texto com servidor mais lento |
| `temperature` | 0 para observar desempenho; depois 0.7 para explorar respostas | Variação de conteúdo; não é a principal alavanca de velocidade |
| `top_p` | Deixe como está inicialmente | Só explore junto da amostragem, depois de entender `temperature` |

Intuição: **prefill** lê o prompt; **decode** gera a continuação. Entrada maior tende a pesar na espera inicial. Saída maior mantém o servidor trabalhando por mais tempo. Compare comprimentos em tokens, não em palavras.

### Depois: os poucos controles do servidor que merecem atenção

| Parâmetro | Intuição | Experimento futuro |
| --- | --- | --- |
| `--max-model-len` | Limite de entrada + saída por sequência; não é o comprimento real de toda pergunta | 2048 → 4096, se precisar de contexto e houver memória |
| `--gpu-memory-utilization` | Orçamento de VRAM do motor; não é porcentagem de uso computacional da GPU | Nesta rodada usamos `1.0` para disponibilizar toda a VRAM ao executor; registre OOMs |
| `--max-num-seqs` | Quantas sequências podem ser processadas simultaneamente | 1 → 2 → 4, enviando de fato requisições concorrentes |
| `--max-num-batched-tokens` | Orçamento de tokens processados numa iteração; afeta o equilíbrio entre prefill e decode | Deixe no padrão até investigar latência com concorrência |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Permite reutilizar trabalho de um prefixo já processado | Compare ligado/desligado com um prefixo longo idêntico |
| `--enforce-eager` | Caminho de execução sem CUDA graphs; útil para diagnóstico | Experimento isolado se a preparação dos graphs falhar ou consumir muita memória |

Essas sugestões são hipóteses para explorar, não promessas de ganho. Aumentar memória disponível pode permitir mais cache sem acelerar uma pergunta curta. Aumentar sequências sem enviar mais perguntas não cria carga. Diminuir contexto máximo não garante redução proporcional da VRAM reservada: o motor pode usar a folga para o KV cache.

Deixe `tensor-parallel-size`, offload para CPU, precisão do KV cache, kernels de atenção e quantizadores alternativos para depois. Com uma GPU e uma pergunta, eles adicionam variáveis antes de você conhecer a referência. Documentação de apoio: [configuração do motor](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) e [prefix caching](https://docs.vllm.ai/en/v0.29.0/features/automatic_prefix_caching/).

**Para mudar um argumento:** pressione Ctrl+C no Terminal A, espere o processo encerrar, confira `nvidia-smi` e execute novamente o comando completo com uma alteração. A primeira resposta após cada reinício pode ter custos extras. Registre o comando usado.

## 7. Saber se o Pod está saudável

No Terminal B, antes ou durante perguntas:

```bash
nvidia-smi
free -h
df -h /workspace /
df -h /dev/shm
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8000/health
```

Os três primeiros comandos repetem as consultas de GPU, RAM e disco. `df -h /dev/shm` consulta o espaço do filesystem de memória compartilhada, usado por processos que precisam trocar dados. No último comando, `-o /dev/null` descarta o corpo da resposta e `-w` imprime apenas o código HTTP; `\n` acrescenta uma quebra de linha. `200` indica sucesso HTTP; `000` geralmente significa que nenhuma resposta HTTP foi recebida. Aqui não usamos `-f`: o objetivo é enxergar o código inclusive em erros HTTP.

Para observar continuamente, use um comando por vez; Ctrl+C encerra o monitor:

```bash
watch -n 1 nvidia-smi
```

```bash
vmstat 1
```

```bash
top
```

`watch -n 1 nvidia-smi` repete a consulta NVIDIA a cada segundo. `vmstat 1` imprime estatísticas de processos, memória, swap, I/O e CPU a cada segundo. `top` mostra processos e seu consumo atualizado; pressione `q` para sair. São monitores de observação: não mudam parâmetros do vLLM.

Se `watch` ou `vmstat` não existirem no template Ubuntu, instale `procps` com `apt-get install -y procps` como root. A primeira linha de dados de `vmstat` é uma média desde o boot; observe as seguintes.

`apt-get install -y procps` instala as ferramentas de monitoramento do sistema; só é necessário quando estiverem ausentes.

| Sinal | Leitura útil |
| --- | --- |
| VRAM ocupada, GPU quase ociosa sem perguntas | Pode ser normal: o servidor mantém pesos e reserva cache |
| GPU ocupada durante a geração | Indica atividade, mas não prova uso eficiente nem distingue compute-bound de memory-bound |
| RAM com pouco `available` | Investigue pressão de memória; `free` baixo sozinho pode ser cache do Linux |
| `si`/`so` recorrentes em `vmstat` | Há atividade de swap visível; correlacione com a lentidão |
| CPU alta e GPU esperando | Hipótese de gargalo fora da GPU; verifique processos e carga do host |
| Disco quase cheio | Pode interromper download, instalação ou gravação de logs |
| `/health` retorna 200, mas pergunta falha | O processo responde; ainda é preciso validar uma inferência real |
| Queda de clock/potência acompanhando temperatura e lentidão | Investigue limitação térmica ou energética, sem concluir só pela temperatura |
| Logs com OOM, preempções ou erros repetidos | Salve o trecho e o comando; são pistas melhores que apenas “está lento” |

Num container, algumas leituras de RAM, CPU e swap podem refletir o host. Compare com os recursos contratados. Em sistemas com cgroup v2, estes arquivos ajudam a enxergar limites do container:

```bash
for f in memory.current memory.max memory.events cpu.max; do
  if [ -r "/sys/fs/cgroup/$f" ]; then
    printf '\n%s\n' "$f"
    cat "/sys/fs/cgroup/$f"
  fi
done
```

O `for` percorre quatro nomes de arquivos. `f` guarda o nome atual; `[ -r ... ]` testa se o arquivo é legível. `printf` imprime seu nome e `cat` mostra o conteúdo. `fi` fecha o teste e `done` fecha a repetição. É uma leitura, sem modificar os limites. `memory.current` mostra uso em bytes, `memory.max` mostra o limite (ou `max`), `memory.events` mostra contadores de eventos e `cpu.max` mostra quota/período de CPU (ou quota `max`).

Em `memory.events`, aumento de `oom_kill` é evidência de processo morto pelo limite de memória do container. Ausência desses arquivos não significa ausência de limite. Referência: [cgroup v2, documentação do kernel](https://docs.kernel.org/admin-guide/cgroup-v2.html).

**Uma rotina suficiente no começo:** olhar GPU, RAM e disco antes de carregar; observar GPU e logs durante uma pergunta; confirmar que a resposta termina sem erro. Não precisa montar Grafana para começar.

## 8. Medir quando você estiver pronto

Comece só guardando o prompt, o comando de inicialização e os tokens retornados. Quando quiser números, rode o cliente de medição **dentro do Pod**: assim, a conexão Brasil–datacenter não entra na comparação local.

TTFT é a espera até o primeiro token de conteúdo, não até os cabeçalhos HTTP. A velocidade de decode de uma resposta é aproximadamente `(tokens gerados − 1) / (tempo entre primeiro e último token)`. Throughput agregado soma o trabalho de todos os usuários; não representa a velocidade que cada um vê. Um evento de streaming pode carregar mais de um token, portanto contar linhas `data:` não resolve.

O vLLM oferece um benchmark pronto. No Terminal B:

```bash
cd /workspace/yan-vllm
source .venv/bin/activate
export HF_HOME=/workspace/yan-vllm/hf-cache
vllm bench serve --help
```

Os três primeiros comandos preparam o terminal como antes. `vllm bench serve --help` apenas mostra as opções do cliente de benchmark; não executa a carga e não inicia outro servidor.

Um teste opcional pequeno, com prompts sintéticos e uma requisição por vez:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model qwen14b-int8 \
  --tokenizer /workspace/models/Qwen2.5-14B-tokenizer \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 128 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf \
  --seed 42 \
  --save-result \
  --result-dir resultados
```

| Argumento | Explicação |
| --- | --- |
| `vllm bench serve` | Executa o cliente que envia requisições ao servidor já iniciado. |
| `--backend vllm` | Seleciona o adaptador de requisições do benchmark. |
| `--base-url ... --endpoint /v1/completions` | Define endereço e rota da API testada, dentro do Pod. |
| `--model qwen14b-int8` | Usa o alias anunciado pelo servidor. |
| `--tokenizer /workspace/models/Qwen2.5-14B-tokenizer` | Usa a pasta local do tokenizer Qwen2.5, preparada antes da medição. |
| `--dataset-name random` | Gera entradas sintéticas para medir desempenho. |
| `--random-input-len 256` | Configura o comprimento de entrada sintética em tokens. |
| `--random-output-len 128` | Configura o comprimento de saída solicitado; confira o comprimento efetivo no resultado. |
| `--num-prompts 10` | Envia dez solicitações na carga medida. |
| `--max-concurrency 1` | Mantém no máximo uma requisição ativa no cliente. É distinto do limite de sequências do servidor. |
| `--request-rate inf` | Não acrescenta um intervalo artificial entre chegadas; o limite de concorrência continua valendo. |
| `--seed 42` | Fixa a semente da geração de dados; não garante tempos idênticos nem determinismo absoluto de toda a execução. |
| `--save-result --result-dir resultados` | Grava o relatório na pasta indicada para você comparar depois. |

É uma sondagem de desempenho, não de qualidade de resposta. Rode uma vez para aquecer e repita para observar variação. Dez amostras ajudam a aprender, mas não sustentam conclusões fortes sobre percentis de cauda. O benchmark usa completions; seus testes manuais usaram chat com template. Mantenha o mesmo protocolo ao comparar resultados. Verifique os tokens efetivamente gerados e eventuais falhas no relatório.

Confira as opções da versão instalada em [bench serve](https://docs.vllm.ai/en/v0.29.0/cli/bench/serve/). Se uma flag mudar, guarde o erro e o help em vez de modificar várias coisas juntas. Para investigar prefix caching, reinicie ou controle o cache: repetir o mesmo prompt pode reaproveitar trabalho e alterar o TTFT.

## 9. Se não funcionar

| Sintoma | Primeiro passo |
| --- | --- |
| `Connection refused` | Veja se o Terminal A terminou de iniciar ou saiu com erro; confira que o cliente está no mesmo Pod |
| `model not found` na API | Use o alias `qwen14b-int8`, exibido por `/v1/models` |
| Erro de tamanho de contexto | Encurte entrada/saída; ambas precisam caber em 2048, incluindo o template de chat |
| Falta de espaço em disco | Veja `df -h`; confirme `HF_HOME` em `/workspace` antes de baixar novamente |
| Driver incompatível, importação CUDA falha | Traga `nvidia-smi`, versão do vLLM/PyTorch e o primeiro erro do log |
| OOM ao carregar pesos | Confira processos usando VRAM; reduzir contexto não encolhe os pesos |
| OOM ao preparar graphs | Após salvar o log, tente acrescentar somente `--enforce-eager` |
| Não sobra memória para KV cache | Confirme GPU livre; tente contexto 1024 como diagnóstico e uma pergunta curta |
| Processo termina apenas com `Killed` | Confira RAM e `memory.events`; pode ser OOM de RAM, não de VRAM |
| Erro de backend/kernel | Guarde versão e traceback; investigue sem trocar o artefato |

Se o 14B impedir o aprendizado por falta de memória, registre a condição como não suportada neste roteiro. Reduzir o modelo seria outro experimento, com novo artefato, tokenizer e protocolo; não misture seus números.

Quando me chamar, mande: etapa em que está, comando executado, `nvidia-smi`, versões e as últimas linhas do log, incluindo a causa inicial do erro. Não precisa mandar tokens de acesso ou chaves privadas.

## 10. Encerrar e retomar sem sustos

**Ctrl+C encerra o vLLM, mas não encerra a cobrança do Pod. Fechar a aba também não.**

1. Encerre o servidor e baixe logs/resultados pelo navegador de arquivos do JupyterLab. Guarde também o nome do template e a configuração do Pod.
2. No painel RunPod, use **Stop** quando estiver disponível e quiser manter o volume local. Confirme visualmente o estado parado; armazenamento continua sendo cobrado.
3. **Terminate** remove o Pod e os dados dos discos locais associados. Faça backup antes. Um network volume é separado e persiste, com cobrança própria.

Pods com network volume podem permitir apenas terminar, em vez de parar. A opção exata depende do tipo de armazenamento; confira [as regras de persistência](https://docs.runpod.io/pods/storage/types) antes da primeira sessão. Este roteiro propõe volume disk local para simplificar, não pressupõe que você tenha network volume.

Ao retomar um Pod parado, pode não haver a mesma GPU disponível. Confirme `nvidia-smi`. Se `/workspace` foi preservado, ative `.venv`, exporte `HF_HOME` e execute o comando da seção 4. Arquivos do container fora do volume podem ter sido perdidos; mantenha o mesmo template para favorecer compatibilidade do ambiente Python.

## 11. Ajustes pequenos no plano do grupo

Sua divisão faz sentido para explorar ferramentas. Este roteiro usa o mesmo arquivo GGUF nos runtimes; identifique o caminho e SHA-256 nos resultados para não misturar medições.

- **Não esperem escolher a melhor quantização para começar a medir.** Guardem uma referência funcionando agora; “melhor” envolve qualidade, memória e latência no runtime escolhido.
- **Use o mesmo GGUF** para comparar vLLM, llama.cpp e Ollama. Se um backend não aceitar o arquivo, preserve logs e marque falha; não atribua uma tabela incompleta a uma diferença de runtime. Consulte [importação no Ollama](https://docs.ollama.com/import).
- **O FP16 do 14B precisa de outra estratégia de memória.** Uma GPU maior pode servir à avaliação de qualidade de referência. Seus tempos nela não são uma referência equivalente aos da 3090. Offload na 3090 também muda o experimento ao introduzir transferências CPU–GPU. Quantizar pode exigir mais recursos do que servir o resultado; Arthur deve dimensionar essa etapa separadamente e partir do checkpoint original em alta precisão para cada quantização.
- **O enunciado parte de 12 GB.** A 3090 tem 24 GB. Registrem essa adaptação e alinhem com o professor se 12 GB for requisito, não apenas cenário ilustrativo. Limitar artificialmente o vLLM não torna a 3090 equivalente a uma GPU de 12 GB.

No PDF `Problema2_Latencia_IO.pdf`, as páginas 5 e 13 pedem diagnóstico e dados antes/depois; a página 24 sugere medir antes de alterar. Para você, isso pode nascer naturalmente: primeiro faça funcionar, depois registre a observação que motivou cada mudança. Não é necessário antecipar toda a otimização agora.

## 12. Caderno mínimo de experimentos

Copie este bloco para uma nota a cada descoberta:

```text
Data / Pod / GPU:
Template / driver / vLLM:
Modelo / revisão do checkpoint:
Comando completo do servidor:
Prompt / tokens de entrada / teto e tokens reais de saída:
Servidor recém-iniciado ou aquecido? Prefix cache ligado?
Uma mudança que fiz:
O que eu esperava:
O que observei: resposta, TTFT, decode, VRAM, RAM, erros
O que ainda não sei:
Próxima pergunta:
```

Para registrar o artefato efetivamente usado, calcule o SHA-256 do GGUF local e salve o caminho, tokenizer correspondente e versões dos pacotes.

**Próxima ação concreta:** criar o Pod, executar `nvidia-smi` e chegar à primeira resposta da seção 5. Todo o restante pode esperar sua curiosidade aparecer.
