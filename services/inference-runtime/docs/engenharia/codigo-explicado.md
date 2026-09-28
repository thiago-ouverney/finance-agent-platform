# O benchmark por dentro: guia detalhado do código

> **Objetivo:** explicar como o cliente de medição executa, observa e registra
> o benchmark sem confundi-lo com o runtime de inferência.

Este documento explica a versão 0.4 para quem conhece estatística, mas não necessariamente programação de servidores ou GPUs. Leia junto com o [bench explicado](../benchmark/bench-explicado.md) e com os módulos Python do runtime. As explicações seguem funções e blocos lógicos; não são uma promessa de explicar uma operação de GPU que o cliente não observa.

## 1. Primeiro: o que este programa é — e o que não é

É um **cliente de medição**: envia mensagens para um servidor de inferência, observa respostas e salva dados. Não implementa a rede neural, não quantiza pesos e não executa as multiplicações do Transformer. Essas tarefas pertencem ao runtime, por exemplo vLLM.

O GuideLLM controla as requisições sintéticas sequenciais. Nosso código controla a partida opcional do servidor, primeira resposta, referência final, telemetria e apresentação. Há duas implementações de medição HTTP: a própria nas requisições inicial/final e a do GuideLLM nos blocos. Não trate seus números como cronômetros internos idênticos; a comparação inicial versus final usa o mesmo cliente próprio.

```text
bench.py: configuração, sequência, arquivos, telemetria
    ├── lifecycle.py: processo → API → primeiro stream
    ├── GuideLLM + guidellm_compat.py: warmup e measure, uma requisição por vez
    └── reporting.py: fórmulas derivadas, agrupamento e HTML

HTTP/SSE ⇄ runtime ⇄ modelo/GPU
```

Uma unidade amostral é uma **requisição**, não cada amostra de GPU e não cada fragmento SSE. Repetições no mesmo processo não são partidas independentes. `--requests 30` não significa 30 usuários simultâneos.

## 2. Requisitos de entrada e preparação

Antes do benchmark: runtime instalado, pesos completos no SSD e tokenizer preparado. O artefato desta rodada é `arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf`; em vLLM exige `vllm-gguf-plugin` e é experimental. Se falhar, preserve o log e marque a execução como falha; não substitua o artefato. Download não é etapa medida e não deve ocorrer durante uma execução válida. Instalação também fica fora. Carregamento do SSD, alocação de memória e inicialização de kernels **continuam sendo parte do problema**.

O arquivo de configuração descreve servidor, modelo, contexto, cache, artefato e versões. O código exige exatamente os campos esperados, evitando erro silencioso de digitação. Ele não configura o contexto ou o cache do runtime: esses campos são metadados do experimento.

`--local-model-path` aponta para os pesos existentes. Uma pasta Hugging Face é adequada; para llama.cpp, um GGUF; para Ollama, um blob de pesos já presente. Não passe um README ou um arquivo de configuração como se fossem pesos: o teste verifica presença/tamanho, não formato binário interno.

No modo `--launch`, o processo filho recebe `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`. Isso impede o caminho normal de download nessas bibliotecas, mas não bloqueia a rede do sistema inteiro. Um wrapper com `curl` poderia baixar: **não o use**. Servidor preexistente não herda variáveis do cliente. Se ocorreu download, a execução não respeitou o protocolo; prepare o ambiente e repita.

## 3. `bench.py`: organização e validações

### Imports e constantes

`argparse` transforma opções de terminal em um objeto Python. `Path` manipula caminhos; `json` e `csv` gravam resultados; `subprocess` executa ferramentas locais; `threading` permite amostrar telemetria enquanto a inferência ocorre. `asyncio` roda a API assíncrona do GuideLLM. Isso **não implica** concorrência de inferência: o perfil escolhido continua síncrono.

`ROOT` é a pasta do script, não necessariamente o diretório do terminal. `VERSION` identifica nosso cliente; `GUIDELLM_VERSION` fixa a versão testada. `NAMED_WORKLOADS` mapeia `short`, `medium` e `long` para 256/2048/7680 tokens. Em `replay`, são alvos do prompt completo estimado; em `independent`, do conteúdo sintético antes do template. `OUTPUT_TOKENS=128` é o teto solicitado de saída, não uma garantia de 128 tokens produzidos.

### `local_model_check(value)`

1. `expanduser().resolve()` normaliza o caminho. Não baixa nem cria arquivos.
2. Se for arquivo, usa esse arquivo; se for pasta, procura extensões de pesos conhecidas.
3. Examina índices `*.index.json`: `weight_map` associa tensores a shards. Cada shard declarado precisa existir.
4. Rejeita ausência de pesos e arquivos vazios.
5. Retorna caminho, quantidade, soma de bytes e limitações da validação para o manifesto.

`stat().st_size` lê metadados, não o conteúdo inteiro dos pesos. Isso evita deliberadamente preencher o cache do sistema lendo 8–15 GB antes de medir a partida. Mesmo assim, não garantimos cache de disco frio. Esta checagem não prova que o volume físico é SSD, não valida os tensores e não atesta que um servidor remoto usa esses arquivos.

### `write_json(path, data)`

Serializa dicionários/listas em UTF-8, com indentação e acentos legíveis. É uma função auxiliar de persistência; não calcula métricas. Os arquivos de cada execução ficam numa pasta nova com timestamp, para não misturar baterias.

### `redact(value, secret)`

Percorre dicionários e listas recursivamente. Campos `api_key` e `authorization` são substituídos por `[REDACTED]`; ocorrências literais da chave conhecida em strings também. Não é um detector universal de dados sensíveis: outro token, caminho pessoal, prompt privado ou log do servidor pode permanecer. Revise antes de publicar.

### `capture(command)`

Executa uma lista de argumentos sem shell, captura stdout/stderr e limita a espera a 15 segundos. Retorna o exit code e os textos, ou um diagnóstico de indisponibilidade. É usado para Git e NVIDIA. Não transforma falha de `nvidia-smi` em utilização de GPU igual a zero.

### `load_config(path)`

Lê o JSON, verifica campos obrigatórios, strings não vazias e contexto inteiro de pelo menos 512. Analisa a URL: aceita HTTP/HTTPS, mas rejeita usuário/senha embutidos, query e fragmento. Permite raiz ou `/v1`, normalizando a raiz para que as funções adicionem o endpoint correto sem duplicação.

O alias em `model` deve corresponder ao ID em `/v1/models`. `server_command` é registro documental: não é executado. Só o array passado por `--launch` vira processo.

### `validate_run(cfg, scenarios, smoke)`

Para cada cenário, exige contexto declarado de pelo menos `entrada solicitada + 128 + 256`. Os últimos 256 são margem para template; não uma prova de que o tokenizer/template real cabem. O servidor continua sendo a autoridade sobre limite real e VRAM.

Fora do smoke, rejeita metadados `PREENCHER`/`SUBSTITUA` e cache não verificado. Smoke tolera rascunhos, mas não ignora erros de inferência. Não converta um smoke em resultado definitivo só porque terminou.

### `tokenizer_digest(path)`

Exige tokenizer local e calcula SHA-256 dos arquivos, depois um hash do mapa de hashes. Serve para conferir se o grupo usou os mesmos arquivos. Não é hash dos pesos nem prova de que o template aplicado pelo servidor seja idêntico. Guarde tokenizer em pasta separada dos pesos: a função lê os arquivos dessa pasta para gerar hashes.

### `prepare_tokenizer(args)`

É **preparação**, não medição. Resolve a revisão do Hugging Face para um commit, baixa apenas o tokenizer, salva origem e arquivos. Não sobrescreve pasta existente. `trust_remote_code=False` não aceita executar código customizado do repositório. Na execução medida, o tokenizer é aberto com `local_files_only=True`.

### `scenario_config(...)`

Constrói a especificação entregue ao GuideLLM:

| Campo | Por que está assim |
|---|---|
| `openai_http`, `/v1/chat/completions` | Protocolo comum aos runtimes. |
| `stream=True` | Permite observar início da resposta antes do término. |
| `validate_backend=False` | Evita geração de validação escondida antes da primeira requisição própria. |
| `verify=True`, `follow_redirects=False` | Verifica TLS e evita redirecionamentos automáticos. |
| `temperature=0`, `top_p=1` | Controla amostragem; não garante determinismo bit a bit entre runtimes. |
| `synchronous` | Espera resposta terminar antes da próxima. |
| `warmup=0`, `cooldown=0` | Nosso laço cria blocos de warmup explícitos; evita aquecimento interno adicional. |
| `max_requests=count`, `max_errors=1` | Limita bloco e interrompe após erro conforme política do instrumento. |
| `synthetic_text` | Referência de configuração do cenário; no modo `independent`, a carga medida é gerada localmente com um prompt determinístico único por requisição. |
| `num_workers=0`, `shuffle=False` | Evita workers adicionais e reordenação do carregador. |
| `prefer_response_metrics=True` | Prefere contagens fornecidas pela API; examine os brutos para compatibilidade. |
| `outputs=[]` | Salvamos o relatório retornado em nosso formato/pasta. |

Não usamos `ignore_eos`: não é opção portátil. Portanto, modelos podem encerrar antes do teto. Olhe a saída real antes de comparar latências.

### `guidellm_compat.completion_drain`

O módulo de compatibilidade é específico do GuideLLM 0.7.4. Ao redor da iteração do benchmark, ele pode drenar uma atualização terminal genuína que tenha chegado atrasada à fila local, somente quando o snapshot final ainda indica requisições incompletas e o encerramento já foi sinalizado. Não repete requisições nem fabrica métricas; se não chegar uma atualização real, a guarda normal de completude continua falhando. O patch é temporário e process-global, por isso execuções sobrepostas não são suportadas.

## 4. `Monitor`: telemetria sem confundir medição e inferência

### `__init__` e `start`

Armazena pasta, fase atual, configuração e evento de parada. Inicia uma thread para NVIDIA e, se solicitado, outra para `/metrics`. Isso não cria outro usuário do chatbot: as chamadas de métricas são GETs, sem geração. Ainda assim têm custo; mantenha a coleta igual entre configurações.

### `loop`: NVIDIA

Abre `gpu.csv`, grava cabeçalho e consulta `nvidia-smi` para índice, nome, memória ocupada/total, utilização, temperatura e potência. Guarda UTC e a fase capturada antes da consulta. Cada GPU tem sua linha. `flush()` disponibiliza amostras em disco durante a execução.

Após a consulta espera aproximadamente um segundo. Logo, o período real inclui também o tempo da ferramenta; não é amostragem exata de 1.000 Hz. Se a ferramenta falha, grava `gpu-unavailable.json` e encerra apenas esse coletor. Apple/Metal não é medido por `nvidia-smi`.

### `kv_loop`: ocupação do cache exposta pelo servidor

Consulta a raiz do servidor + `/metrics` com timeout de um segundo. Uma expressão regular seleciona gauges `vllm:kv_cache_usage_perc` ou `vllm:gpu_cache_usage_perc`. Se o nome moderno aparece, ignora o legado para não contar duas vezes a mesma grandeza.

Aceita frações finitas entre 0 e 1; preserva labels de cada série, por exemplo engine/modelo. Grava UTC, fase, série e fração em `kv-cache.csv`. Inicialização, endpoint ausente ou erro HTTP não inventam amostras. Um arquivo apenas com cabeçalho significa **sem observação válida**, não “pool vazio”.

O servidor pode atualizar seu gauge menos frequentemente que o cliente. A coleta não mede a ocupação a cada token. Também não a converte em bytes sem saber capacidade e representação. Isso está separado da estimativa lógica por contexto.

### `stop`

Sinaliza parada e espera as threads por prazos limitados. A espera por evento, em vez de um `sleep` longo, permite encerrar rapidamente. Não mata outras ferramentas de monitoramento do usuário.

## 5. Estatísticas em `bench.py`

### `percentile(values, q)`

Remove ausências e valores não finitos, ordena e calcula posição `(n−1)q`. Interpola entre os elementos vizinhos. Exemplo: em `[10,20,30]`, p50=20 e p95=29. Não fornece intervalo de confiança. Sem valores retorna `None`, não zero.

### `summarize(report)`

Exige exatamente um benchmark GuideLLM no bloco. Separa requisições bem-sucedidas, com erro e incompletas; percentis só usam sucessos. Grava nomes longos como `request_first_token_latency_milliseconds_p50`, `request_latency_seconds_p95` e `decode_generation_tokens_per_second_p99`, além de `sample_count` por métrica. Também grava `statistics.percentiles_by_metric`, com p05, p50, p95 e p99, a definição da interpolação e a indicação de que caudas com menos de 100 sucessos são exploratórias. O primeiro-token é medido uma vez em cada requisição; `within_response_next_token_latency_milliseconds` descreve os intervalos entre tokens sucessivos da mesma resposta. O agrupamento externo preserva `phase`, `scenario` e `repetition`, portanto os resultados de short, medium e long ficam separados.

Gera ainda `requests_sha256` a partir de mensagens e `max_tokens`, não do alias do modelo ou chave. Esse hash ajuda a comparar a carga realmente registrada, mas só usa requisições bem-sucedidas: falhas precisam ser examinadas antes. Marca `p95_exploratory=True` abaixo de 100 sucessos; 100 não é garantia estatística de precisão.

### `write_requests_csv(path, report)`

Grava uma linha por requisição, inclusive falhas e incompletas. Para sucessos, inclui TTFT, duração, ITL, contagens, duas taxas, contexto inicial/final e faixa. No modo `independent`, `workload_seed` identifica a semente efetiva do prompt. Taxas de falhas não são usadas como velocidades válidas. Os JSONs brutos preservam detalhes que o CSV compacto não inclui.

### `write_summary(output, rows)`

Salva resumo JSON e, havendo linhas, CSV. Chama `reporting.render` para HTML integrado e resumos auxiliares. Mesmo sem blocos concluídos, o HTML pode mostrar falha/inicialização; não precisa perder a primeira requisição só porque GuideLLM não começou.

## 6. `run(args)`: a execução principal em ordem

1. Importa dependências e verifica versão exata do GuideLLM.
2. Lê configuração, valida contexto e pesos locais; registra coeficiente opcional de KV.
3. Verifica tokenizer e seu hash. Define três requisições/uma repetição no smoke; caso contrário usa argumentos.
4. Lê chave da variável `BENCH_API_KEY`, nunca exige senha no comando. Lê prompt inicial padrão ou arquivo UTF-8, rejeitando vazio.
5. Cria pasta de resultados e obtém lock não bloqueante. O lock impede outro cliente usando **a mesma pasta base**, não protege a GPU de processos externos.
6. Cria subpasta com timestamp UTC e manifesto `running`: versões, configuração, sementes, SO, tokenizer e presença dos pesos. Salva pacotes do cliente e snapshot NVIDIA.
7. Inicia monitores. Se houver `--launch`, cria objeto `Launch`, registra argumentos e inicia relógio de processo.
8. Aguarda API por GET. Nenhum POST de inferência foi enviado por este cliente.
9. Envia primeira requisição própria cronometrada. Ela também valida stream e usage; é medição, não aquecimento descartado.
10. Importa GuideLLM e percorre repetições. Rotaciona ordem dos cenários para reduzir efeito fixo de posição, sem prometer eliminar tendências.
11. Para cada cenário: roda bloco `warmup`, depois `measure`. Seeds do aquecimento recebem deslocamento de 1.000.000 e o índice da requisição compõe a semente efetiva; assim warmup e medição usam conjuntos disjuntos, mantendo o mesmo tamanho-alvo.
12. Executa GuideLLM via `asyncio.run`, salva bruto, CSV individual e resumo parcial. Verifica contagem de sucessos, erros e incompletas; discrepância interrompe a bateria com falha.
13. Repete o prompt inicial com o observador próprio como referência aquecida. Pode aproveitar prefix caching se habilitado; não atribua toda melhora a kernels.
14. Marca `complete` apenas se todas essas etapas passaram.
15. Em exceção, marca `failed` ou `interrupted` e preserva resultados. O bloco `finally` executa também em erro: recupera requisições parciais, encerra só o grupo criado, finaliza monitores/manifesto e reconstrói HTML.

`--repetitions` não reinicia runtime. Para amostrar partidas, repita o comando completo com `--launch`. O modo sem lançamento registra `existing-server-state-unknown`: só sabemos que é a primeira mensagem deste cliente.

### `positive`, `main` e o bloco `__main__`

`positive` rejeita zero/negativos em argumentos que exigem inteiros positivos. `main` define subcomandos `prepare-tokenizer`, `run` e `report`, interpreta argumentos e encaminha à função adequada. `--input-tokens` cria cenários como `ctx1024` e substitui `--scenarios`; comprimentos repetidos são rejeitados.

Erros são apresentados no stderr e retornam código 1; Ctrl+C retorna 130. `raise SystemExit(main())` propaga esse código ao shell, permitindo automação distinguir sucesso de falha.

### `rebuild_report(args)`

Reabre o resumo e os brutos GuideLLM de uma execução antiga, recalcula taxas e CSVs e gera o novo HTML. Não roda modelo, não reescreve manifestos ou brutos e não inventa telemetria histórica. **Substitui os arquivos derivados**; copie-os antes se quiser manter a apresentação anterior.

```bash
python bench.py report --output results/PASTA_DA_EXECUCAO
```

## 7. `lifecycle.py`: partida e primeiro stream

### `Launch.__init__`

Só aceita `localhost`, `127.0.0.1` ou `::1`: iniciar um processo no cliente não lançaria servidor numa máquina remota. Lê array JSON de strings não vazias e prepara campos de estado. Não interpreta shell. Comandos como `source`, `&`, pipes e redirecionamentos não pertencem a esse array.

### `Launch.start`

Testa conexão na porta. Se já estiver ocupada, recusa sem matar ninguém. Abre `server.log`, obtém origem com `time.perf_counter()` e executa `Popen` com `start_new_session=True`, `shell=False` e variáveis HF offline. Retorna origem temporal.

`perf_counter` é relógio monotônico: serve para diferenças de tempo, não datas de calendário. Criar nova sessão/grupo permite depois encerrar também workers daquele servidor. Não passe um daemon que se desconecte do grupo nem um serviço de terceiros.

### `Launch.close`

Envia SIGTERM ao grupo criado, espera até 20 segundos; se necessário usa SIGKILL e aguarda mais cinco. Verifica também workers que sobreviveram ao líder e fecha log. Não procura/mata todos os processos vLLM. O escopo é exclusivamente o grupo que criou.

### `wait_models`

Faz GET `/v1/models`, valida autenticação e procura o ID configurado. Com `--launch`, repete até prontidão, timeout ou saída prematura do filho. Sem lançamento, não espera indefinidamente por um servidor que o cliente não iniciou. O intervalo entre tentativas é de até meio segundo, além do custo de rede.

Retorna tempo desde espera e, quando conhecido, desde origem do processo. **Modelo listado não comprova pesos residentes na GPU.** Alguns runtimes carregam na primeira geração; por isso também medimos processo → primeiro conteúdo.

### `timed_request`: leitura detalhada do stream

1. Monta corpo com modelo, uma mensagem de usuário, temperatura zero, `top_p=1`, teto 128, streaming e `include_usage`.
2. Prepara resultado com status `running` e métricas ausentes.
3. Abre cliente HTTP, marca instante imediatamente antes do POST e mede chegada dos cabeçalhos separadamente.
4. Exige HTTP de sucesso e `text/event-stream`.
5. Percorre linhas SSE. Ignora linhas que não começam em `data:`. `[DONE]` encerra; outros dados são JSON.
6. Se evento contém erro, falha. Guarda usage quando presente.
7. Concatena `delta.content`. Evento apenas com `role`, conteúdo vazio ou usage **não inicia TTFT**.
8. Para conteúdo não vazio, registra offset desde POST e acrescenta texto. No primeiro desses eventos, fixa TTFT e, se possível, tempo desde partida.
9. Ao fim, calcula duração HTTP e exige `[DONE]`, texto e contagens positivas inteiras. Falha de usage não vira contagem estimada nesta função.
10. Com mais de um token, calcula ITL médio a partir de primeiro/último conteúdo e contagem. Calcula taxas derivadas e marca sucesso.
11. Em erro, registra estado parcial e duração até saída. O `finally` grava arquivo mesmo se houver exceção.

Timeout HTTP é limite de espera de operações/leitura, não um cronômetro absoluto de toda bateria. Um stream que continua enviando dados pode durar mais que o valor informado. Eventos SSE podem conter múltiplos tokens: **não contamos eventos como tokens**. A observação HTTP não oferece timestamps internos exatos de cada token da GPU.

### `lifecycle_report`

Salva JSON do ciclo de vida e HTML com tempos de partida, primeira resposta e referência final, inclusive tokens/s. Valores ausentes aparecem como não medidos. Escapa textos ao gerar HTML, evitando que um valor seja interpretado como marcação executável.

## 8. `reporting.py`: fórmulas e apresentação

### `ratio(a, b)` e `derived(row)`

Divisão só ocorre para números finitos e positivos. Isso evita divisão por zero, infinito e médias fabricadas. Para `N` tokens de saída:

```text
TTFT = primeiro conteúdo observado − envio
ITL médio = (último conteúdo − primeiro conteúdo)/(N−1)
tokens/s de geração = 1000 / ITL_médio_em_ms
tokens/s efetivos = N / duração_total_em_segundos
```

Exemplo: 128 tokens, TTFT de 1 s e intervalo primeiro→último de 4 s. Geração = 127/4 = 31,75 tokens/s. Se a requisição toda dura 5 s, efetiva = 128/5 = 25,6 tokens/s. A segunda cai quando aumenta a espera inicial; a primeira isola melhor a fase de geração observada.

Uma saída de um token não tem intervalo de geração: taxa de decode é ausente, não infinita. Os percentis das taxas são calculados **depois de calcular a taxa de cada requisição**; não invertemos p95 de latência para chamar de p95 de velocidade. A taxa efetiva não é throughput agregado de sistema multiusuário.

`context_start_tokens` é entrada real. `context_end_tokens` é entrada + saída, um comprimento lógico; o último token emitido pode ainda não ter sido processado em K/V.

Essa distinção é importante para interpretar a execução: uma amostra faltante (por exemplo, 2 sucessos quando o bloco pediu 3) não é uma validação parcial que possa ser promovida a aprovada. O `run` marca a bateria como falha e preserva o bruto para diagnóstico; testes locais/simulados não demonstram que um runtime real no pod foi validado.

Se esse padrão aparecer em uma execução real, documente-o como problema ainda aberto de integração GuideLLM/servidor até haver uma reprodução e uma verificação completa; esta documentação não afirma que ele já foi resolvido em todos os backends.

### `context_band(p)` e `context_summary`

Faixas semiabertas: [0,512), [512,1024), [1024,2048), [2048,4096), [4096,8192), [8192,16384), [16384,+∞). Assim, 512 pertence à segunda; 2.077 pertence a [2048,4096). O template conta. Não é o nome “medium” que determina o grupo.

Reabre CSVs individuais, exclui falhas e agrupa por fase + repetição + faixa real inicial. Calcula medianas de TTFT, taxas e comprimentos. Cenários na mesma faixa podem ser agrupados nessa tabela; a tabela anterior mantém cada cenário separado. Não mistura warmup com measure nem repetições.

Com coeficiente `B = --kv-bytes-per-token`, estima `MiB = tokens × B / 1.048.576`. Para atenção completa convencional, uma sequência: `B = 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento`.

Exemplo **aritmético, não configuração inferida automaticamente**: 48 camadas, 8 cabeças KV, dimensão 128, KV de dois bytes → B=196.608 bytes/token. Para 256 tokens, 48 MiB lógicos. Verifique arquitetura e dtype reais antes de usar esse coeficiente.

Não confundir:

| Grandeza | Origem | Limite |
|---|---|---|
| Contexto em tokens | Usage do servidor/instrumento | Proxy da carga de KV, não bytes. |
| KV lógico em MiB | Fórmula + coeficiente informado | Estimativa; exclui reserva/padding e particularidades de arquitetura. |
| Ocupação do pool KV | Gauge vLLM em `/metrics` | Fração de blocos ocupados; não percentual de VRAM. |
| Memória usada NVIDIA | `nvidia-smi` | Total na GPU, não exclusivamente KV ou runtime. |

Portanto “tokens/s por faixa” aqui é velocidade média da requisição iniciada em uma faixa de contexto/KV lógico. **Não** é velocidade instantânea de cada token condicionada a bytes físicos exatos de cache. Isso exigiria instrumentação no servidor com resolução compatível; não fabricamos essa correlação.

### `gpu_summary`

Agrupa CSV por fase, índice e nome da GPU. Converte valores numéricos; ignora `N/A` e não finitos sem substituí-los por zero. Calcula média aritmética das amostras válidas e máximo por campo. Não é média ponderada pelo tempo nem pico contínuo garantido. `samples` conta linhas, não leituras válidas de cada campo individual.

O máximo de memória inclui reservas do runtime e outros processos. Utilização NVIDIA não mede banda de memória; VRAM quase cheia não prova memory-bound. Potência média não é energia total: não integramos potência no tempo.

### `table` e `render`

`table` recebe linhas e pares de chave/rótulo, formata números com três casas, escapa textos e usa container rolável. `render` lê manifesto/ciclo de vida, gera resumos de contexto/GPU e monta HTML autônomo com cinco seções:

1. Partida e primeira resposta.
2. Warmup e medida, TTFT e duas taxas explícitas.
3. Taxas por faixa de contexto e estimativa opcional de KV.
4. GPU por fase.
5. Ocupação observada do pool KV por série/fase, quando coletada.

O HTML não depende de internet, JavaScript externo ou servidor web. Relatórios antigos reconstruídos mantêm aviso de que a nova política offline não foi verificada retrospectivamente. Não inventamos telemetria que não foi coletada.

## 9. Como executar e interpretar com segurança

```bash
python bench.py run --config configs/vllm.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --launch configs/launch-vllm.example.json \
  --input-tokens 256 512 1024 2048 3072 \
  --collect-kv-metrics --smoke --warmup 1
```

Esse comando exige pesos locais, inicia runtime conforme array, mede partida e primeiro stream, percorre cinco comprimentos (um warmup + três medidas em cada), faz referência final e encerra o próprio servidor. O teto de contexto declarado precisa suportar maior entrada + margem + saída. Não acrescente um coeficiente KV sem verificá-lo. Para uma bateria posterior, retire `--smoke` e configure número de requisições/repetições de acordo com orçamento e precisão desejada.

No relatório, confira status, falhas, contagens reais, alias/modelo, tokenizer, cache e versão. Compare contexto equivalente e output comparável. TTFT maior com prompt longo não significa necessariamente decode mais lento. Separe a pergunta “quando começa?” da pergunta “a que velocidade continua?”.

## 10. Testes, rastreabilidade e limites

`tests/test_bench.py` verifica validações, percentis, taxas, limites de faixas, estimativas, telemetria e escape HTML. `tests/test_lifecycle.py` verifica porta ocupada, saída do subprocesso, readiness e primeiro conteúdo, sem GPU. `tests/integration_mock.py` usa GuideLLM real com servidor SSE simulado para testar caminho completo, ausência de concorrência, falhas e encerramento.

Testes simulados não demonstram suporte do modelo na RTX3090. O teste local Metal real é outra evidência, com outro backend e quantização. Não confunda ambos. Resultados brutos devem permanecer fora do Git até revisão de privacidade.

O código não mede qualidade semântica, energia total, FLOPs, banda física, tráfego PCIe ou uso de memória por processo. Não atribui causalidade de gargalo automaticamente. Esses limites são parte da validade do instrumento, não informações a preencher por estimativa sem evidência.

## 11. Mudanças da versão 0.4

`Monitor` agora mantém `events.csv`, amostra `gpu.csv` e `system.csv` em paralelo e escreve `telemetry-summary.json` agrupado por fase. CPU, RAM e I/O de disco são sinais do host; VRAM e utilização são leituras do `nvidia-smi`. Essa combinação permite observar que um pico ocorreu durante startup, primeiro POST ou medida, mas não é um cronômetro de cópia PCIe. Tempo por transferência exige Nsight/CUDA.

`summarize` mantém os campos históricos e acrescenta os aliases canônicos `time_to_first_token_milliseconds_*` e `tokens_per_second_*`. TTFT tem um valor por requisição; os percentis resumem a distribuição de requisições. Tokens/s é decode e não inclui TTFT. O Make usa short/medium/long por padrão e 50 requisições × 1 repetição fora do smoke.

`run_conversational_batch` ordena os turnos fixos pela proximidade do alvo do cenário. No `replay` nomeado, remove pares antigos quando o histórico passa do alvo e completa o `system` com contexto sintético determinístico quando fica abaixo. A estimativa, o alvo, a quantidade removida e o padding ficam nos artefatos; `usage.prompt_tokens` do runtime é a autoridade. A fase warmup recebe uma fixture separada, enquanto a medição usa `qwen_chat_bench_v2.json`; respostas produzidas pelo runtime nunca contaminam o replay. `scripts/run_kv_sweep.py` reinicia o servidor a cada ponto, converte `--memory-step-mb` para tokens usando `--kv-bytes-per-token`, altera o limite de contexto quando o runtime oferece uma flag (`--max-model-len` no vLLM, `--ctx-size` no llama-server) e guarda cada execução até a primeira falha. `make bench-vllm`, `make bench-llama` e `make bench-ollama` executam a bateria formal e o sweep; `make bench-all` coordena os três.

## 12. Referências e fontes numerados

- [GuideLLM 0.7.4: métricas por requisição](https://github.com/vllm-project/guidellm/blob/v0.7.4/src/guidellm/schemas/request_stats.py).
- [vLLM: métricas de KV e serving](https://docs.vllm.ai/en/v0.12.0/design/metrics/).
- [Metodologia do experimento](../benchmark/metodologia.md).
- Código-fonte versionado do runtime, incluindo `bench.py`, `lifecycle.py`, `reporting.py` e `guidellm_compat.py`.

O apêndice usa números reais e hashes dos fontes. Regenere após editar código para manter referências consistentes. Ele não substitui os arquivos executáveis: números à esquerda são para leitura, não para copiar e executar.
