# Um usuário. Três runtimes. Um instrumento.

O artefato da rodada validada é `arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`, mantido idêntico entre os runtimes quando o backend o aceitar. No vLLM, GGUF exige `vllm-gguf-plugin` e deve ser tratado como experimental/subotimizado. Se o carregamento falhar, a execução falha e preserva logs; não troque silenciosamente o modelo para preencher uma tabela.

Este estudo pergunta: **como o runtime e seus parâmetros mudam o tempo de resposta de um chatbot para uma pessoa?** A carga de múltiplos usuários fica para o próximo trabalho.

## 1. Quem faz o quê?

```text
Este projeto → GuideLLM → API HTTP do runtime → modelo na GPU
                            ↓ respostas em streaming
               tempos + contagens + relatórios
```

O runtime pode ser vLLM, Ollama ou llama-server. O GuideLLM organiza e mede os blocos sequenciais. Nosso código também observa a partida do processo (quando solicitada) e cronometra o primeiro stream HTTP com relógio monotônico. Isso evita aquecer o servidor antes de medir o primeiro contato. Essas métricas iniciais são próprias do cliente; não as apresentamos como métricas internas de kernels nem como resultados produzidos pelo GuideLLM.

Escolhemos [GuideLLM](https://github.com/vllm-project/guidellm) por ser uma ferramenta existente no ecossistema de serving de LLMs e usar um protocolo comum. Não há a pretensão de que este projeto seja um padrão industrial certificado. **Ferramenta padronizada não torna automaticamente o experimento comparável:** os controles continuam sendo responsabilidade do grupo.

## 2. O que “um usuário” significa aqui?

Uma mensagem é enviada; a resposta termina; só então enviamos a próxima. Não há requisições sobrepostas. Isso é uma carga sequencial fechada, sem tempo de reflexão entre mensagens. Não é um robô simulando toda a rotina humana, apenas um instrumento para medir serviço individual.

O perfil GuideLLM é explicitamente `synchronous`. Não usamos o default `sweep`, nem taxa de chegada Poisson, nem perfil throughput. A documentação do [perfil sequencial](https://vllm-project.github.io/guidellm/0.7.0/getting-started/benchmark/) explica essa distinção.

Uma execução do cliente por vez não impede interferência externa: outro terminal, processo ou usuário no mesmo hardware pode contaminar o experimento. A trava de arquivo evita duas execuções deste projeto usando a mesma pasta de resultados, mas não controla o pod inteiro.

## 3. Qual carga será usada?

O próprio GuideLLM gera texto sintético com um tokenizer fixado. Usamos aproximadamente 256, 2048 e, opcionalmente, 8192 tokens de conteúdo, com teto de 128 tokens na resposta. O [gerador de datasets](https://vllm-project.github.io/guidellm/0.7.0/guides/datasets/) fornece uma carga repetível por semente.

| Cenário | Conteúdo de entrada | Teto de saída | Pergunta experimental |
|---|---:|---:|---|
| short | ~256 tokens | 128 | Quanto demora uma interação curta? |
| medium | ~2048 tokens | 128 | Como aumenta o custo de processar contexto? |
| long | ~8192 tokens | 128 | O que acontece com uma entrada grande, se couber? |

Esses números são **nosso desenho**, não limites oficiais de classificação de chatbot. Tokens de template e mensagens de sistema podem aumentar a entrada total. O servidor devolve as contagens reais, que devem ser analisadas.

O benchmark atual usa uma fixture replay validada no repositório, com 50 perguntas técnicas em português e histórico determinístico crescente. Cada requisição é uma conversa nova reconstruída a partir dos turnos fixos; a resposta real do runtime não altera o próximo request. O cenário longo continua representando uma carga de contexto sintética separada do eixo conversacional. O warmup usa outra fixture e não entra na medição. Quando o modo `independent` é solicitado, cada amostra recebe um prompt determinístico distinto com o mesmo tamanho-alvo, e warmup e medição usam sementes disjuntas.

O modelo pode encerrar antes de 128 tokens. Não forçamos ignore-EOS porque não é um parâmetro portátil entre os três servidores. Registrar saída efetiva é obrigatório: uma configuração pode parecer mais rápida apenas por gerar menos texto. Textos sintéticos também podem provocar respostas curtas ou estranhas; confirme os comprimentos no piloto antes de investir na bateria inteira.

## 4. Como uma execução acontece

### O que chamamos de carregamento

Sem `--launch`, o servidor já existia antes do cronômetro: medimos a primeira requisição deste cliente, mas não o carregamento do processo/pesos. Com `--launch`, medimos processo → API, processo → primeiro conteúdo e processo → fim da primeira resposta. O primeiro conteúdo é o marco operacional mais importante para o carregamento, porque cobre pesos locais, alocação, inicialização de kernels e compilação tardia que ocorram antes ou durante a primeira geração. O GET `/v1/models` é apenas prontidão observada.

O protocolo é comum aos três servidores, mas o comando muda: `vllm serve` para vLLM; `llama-server -m caminho/model.gguf` para llama.cpp; e `ollama serve`/preload local mantido em foreground para Ollama. No Ollama, a API pode estar viva antes do modelo ser carregado; por isso não usamos processo → API como substituto do processo → primeiro conteúdo. Cada runtime deve usar o mesmo artefato local e um arquivo `--launch` separado. Downloads antes do processo ficam fora; download iniciado pelo servidor torna a execução inválida para comparação.

Documentação histórica sobre tentativas anteriores foi movida para `docs/archive/inference-runtime/` na raiz do monorepo. Esses diagnósticos não fazem parte da rodada formal atual e não devem ser misturados aos resultados do protocolo vigente.

1. Validamos configuração, versão do instrumento e arquivos do tokenizer, antes de iniciar o runtime.
2. Com `--launch`, lançamos o processo e medimos até a API listar o modelo. Sem essa opção, a partida anterior é desconhecida e não recebe um tempo inventado.
3. Enviamos **a primeira requisição já cronometrada**, que também valida streaming e usage. Não há teste de geração anterior.
4. Para cada cenário/repetição, medimos o bloco de aquecimento e o bloco posterior, identificados separadamente no resumo.
5. Repetimos o mesmo prompt inicial como referência final aquecida, salvamos os resultados e encerramos somente o processo iniciado por nós.

**Nada disso é descartado:** inicialização e primeira resposta ficam em `lifecycle.html`; aquecimento e blocos posteriores em `summary.html`, com a coluna `phase`. Separar fases não é excluí-las. Não as misturamos em uma única média porque respondem perguntas diferentes.

O critério de validade do bloco é estrito: o relatório precisa conter o número solicitado de sucessos e nenhum erro/incompleta. Se uma execução real devolver apenas 2 de 3 amostras, ela deve ser investigada e repetida; não é evidência de que a validação do runtime passou.

Uma amostra faltante permanece um diagnóstico do caminho GuideLLM/servidor; não deve ser descrita como uma correção universal já comprovada. Enquanto a causa não for reproduzida e verificada, mantenha esse bloco como falho.

### O papel de cada etapa — e a diferença entre warmup e TTFT

| Etapa | O que acontece | Entra na medição formal? |
|---|---|---|
| Preparação | Confere modelo, tokenizer, configs, imports e executáveis; `prepare-all` encadeia os três runtimes | Não |
| Partida/readiness | Inicia o processo próprio, quando `--launch` foi usado, e consulta `/v1/models` até o alias esperado aparecer | Os tempos ficam no lifecycle; não são TTFT |
| Primeira resposta | Envia o primeiro POST já cronometrado e mede o primeiro conteúdo e o fim do stream | Sim, mas como sonda separada em `first-request.json`; não é misturada ao p50 da fase `measure` |
| Warmup | Envia requisições reais para aquecer buffers, kernels, caches e caminhos do servidor | Não; a fase fica registrada como `warmup`, mas seus números não entram na fase `measure` |
| Medição | Envia as requisições oficiais, uma por vez, e calcula TTFT, geração, latência total e tokens/s | Sim |
| Referência final/cleanup | Repete o prompt inicial aquecido, encerra apenas o processo criado e fecha telemetria | A referência fica separada; cleanup não é desempenho |

**Warmup não é TTFT.** TTFT é uma métrica de uma requisição: o intervalo entre o envio do POST e o primeiro conteúdo/token observado. Warmup é uma fase de preparação do estado do processo. O warmup pode reduzir o TTFT das requisições seguintes, mas não é o valor de TTFT e não prova que o sistema atingiu estabilidade. A primeira resposta pode ter TTFT alto por cold start; as requisições após o warmup medem um estado mais próximo de operação contínua.

O warmup não limpa caches nem garante um estado frio. Ele pode aquecer compilação, alocadores, buffers, HTTP/SSE, tokenizer, prefixos e estruturas KV. Por isso, `phase=warmup` deve ser analisada separadamente. O padrão formal é `--warmup 3` por cenário e repetição; use o mesmo valor em todos os runtimes. `--warmup 0` é útil para diagnóstico barato, mas deixa a medição mais dependente do estado inicial.

TTFT também não é o tempo total de resposta: a geração começa depois do primeiro conteúdo, enquanto a taxa efetiva divide os tokens de saída pela latência total. Portanto, uma configuração pode melhorar TTFT e piorar decode, ou melhorar tokens/s sem melhorar a latência ponta a ponta.

Para a versão fixada 0.7.4, o cliente possui um dreno limitado de compatibilidade: depois de o encerramento ser sinalizado, espera até cinco segundos por uma atualização terminal genuína que tenha chegado atrasada à fila local. Ele não repete chamadas nem fabrica contagens; sem atualização real, a guarda de completude permanece falhando. Isso reduz uma condição de corrida conhecida do adaptador, mas não equivale a validar um runtime real.

```text
PARTIDA DO PROCESSO  (--launch)
        │ leitura dos pesos locais, inicialização e compilação
        ▼
API LISTA O MODELO   (não necessariamente já residente na GPU)
        │ primeiro POST medido; possível carregamento tardio
        ▼
PRIMEIRO CONTEÚDO → FIM DA PRIMEIRA RESPOSTA
        │
        ▼
AQUECIMENTO MEDIDO → BLOCOS POSTERIORES → REFERÊNCIA FINAL
```

### As métricas iniciais

| Métrica | O que cobre |
|---|---|
| `process_to_api_observed_s` | Da chamada de criação do processo até o primeiro GET válido que lista o modelo. |
| `first_request.ttft_ms` | Do envio do primeiro POST até conteúdo não vazio; não termina ao receber apenas o campo role. |
| `first_request.e2e_s` | Duração total do primeiro stream, incluindo custo inicial que ocorra ali. |
| `process_to_first_content_s` | Da partida até aparecer o primeiro texto: medida útil para comparar runtimes com carregamento antecipado ou tardio. |
| `process_to_response_end_s` | Da partida até terminar a primeira resposta. |
| `warm_reference` | O mesmo prompt inicial repetido ao final, com o mesmo cliente observador. |

O polling de disponibilidade usa intervalos de até 0,5 s, além do tempo de rede/timeout dos GETs. Portanto, medimos **quando observamos a API disponível**, não o instante interno exato da prontidão. Os intervalos processo → conteúdo/fim incluem essa detecção e pequenos custos do cliente entre etapas. Nenhuma diferença de poucos milissegundos nesse intervalo deve ser atribuída automaticamente ao runtime.

O tempo até a API responder não é sozinho uma comparação justa de carregamento. Alguns servidores só carregam o modelo quando recebem a primeira geração. Compare também processo → primeiro conteúdo. O Ollama oferece métricas nativas como `load_duration`, mas elas não são um campo comum a todas as APIs; veja a [documentação de métricas do Ollama](https://docs.ollama.com/api/usage). Não fazemos uma segunda chamada nativa antes da primeira medição, pois ela alteraria o estado a observar.

### Processo novo não significa todos os caches frios

Estudamos processo novo com pesos locais, servidor ativo com modelo não residente e servidor/modelo aquecidos. Processo precisando baixar pesos está fora do protocolo. Cache de disco e compilação podem sobreviver à reinicialização; registre as condições em `--initial-state`. O script não apaga caches, não descarrega modelos alheios e não reinicia o pod.

Sem `--launch`, a primeira requisição é apenas a primeira que este cliente observou. Não a rotule como cold start comprovado. Com `--launch`, sabemos o início do processo, mas não o estado de todos os caches. Uma porta já ocupada bloqueia o lançamento sem matar o processo existente.

Provisionamento, dependências e todos os downloads ficam fora do experimento. `--local-model-path` verifica arquivos não vazios e shards declarados antes do relógio, sem ler os pesos integralmente. `--launch` passa `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1` ao filho; configure caminho local no runtime. Isso não bloqueia downloads arbitrários nem modifica servidor preexistente: o operador deve garantir o mesmo artefato local. Execução com download é inválida. Leitura de pesos locais, transferência PCIe, compilação e captura de grafos continuam incluídas, mas sua decomposição exige logs/profiling.

### Repetição de cold start e referência aquecida

Uma chamada do programa produz uma amostra de partida. `--repetitions` repete cenários sem reiniciar; não transforma três blocos em três cold starts. Para estudar a variabilidade inicial, repita a chamada completa com `--launch` em condições documentadas, uma por vez. O processo iniciado é encerrado ao final de cada chamada.

A pergunta inicial é texto real em português, distinta da carga sintética dos cenários. Pode ser substituída por `--first-prompt-file`. Comparamos seu primeiro acesso com uma referência usando **o mesmo texto** no final; não comparamos diretamente essa pergunta curta com o cenário sintético longo. A referência final pode aproveitar cache de prefixo, portanto a diferença não isola automaticamente apenas inicialização de kernels. Ambos os resultados ficam disponíveis para interpretação.

Aquecer com três requisições é um ponto de partida, não prova de estabilidade. Observe se ainda há tendência na sequência temporal. Se necessário, aumente `--warmup` igualmente para todas as configurações. Não use o conjunto medido inteiro como aquecimento: isso pode popular caches com os mesmos prompts.

## 5. O relógio começa e termina onde?

```text
envio ───────── primeiro conteúdo ── conteúdo ── conteúdo ── fim HTTP
      ← TTFT →                  ← geração →
      ←──────────── latência ponta a ponta ───────────────→
```

As medições são observadas pelo cliente. Incluem custos do serviço e transporte HTTP; não são cronômetros internos de kernels CUDA. Rodar cliente e servidor no mesmo pod reduz variação de rede, mas pode introduzir competição pela CPU. Mantenha a topologia igual entre testes.

| Coluna do resumo | Interpretação | Unidade |
|---|---|---|
| `request_first_token_latency_milliseconds_p50/p95/p99` | Distribuição, por requisição, da espera pelo primeiro conteúdo/token observado | ms |
| `request_latency_seconds_p50/p95/p99` | Distribuição do tempo total de requisição | s |
| `within_response_next_token_latency_milliseconds_p50/p95/p99` | Distribuição, dentro de cada resposta, da média entre um token e o próximo | ms |
| `decode_generation_tokens_per_second_p50/p95/p99` | Geração após primeiro token: `1000 / mean_itl_ms` | tokens/s |
| `effective_output_tokens_per_second_p50/p95/p99` | Saída / tempo total da requisição, incluindo TTFT | tokens/s |
| `input_prompt_token_count_p50/p95/p99` | Comprimento real de entrada registrado | tokens |
| `output_completion_token_count_p50/p95/p99` | Comprimento real da saída registrada | tokens |
| `successful_request_count`, `errored_request_count`, `incomplete_request_count` | Contagens de resultados por status | requisições |

Cada linha de `summary.json` é um grupo independente identificado por `phase`, `scenario` e `repetition`. Portanto, p50/p95/p99 de `short`, `medium`, `long`, `warmup` e `measure` não são misturados. `request_first_token_latency` é medido uma vez por requisição: cada requisição tem seu próprio primeiro token. Já `within_response_next_token_latency` mede os intervalos entre tokens sucessivos dentro da mesma resposta. A chave `statistics.percentiles_by_metric` repete essa estrutura em formato aninhado, com nomes completos e `sample_size` explícito. Com apenas três requisições, p95 e p99 são estatísticas exploratórias, não caudas estáveis.

**Atenção ao nome TPOT.** Na versão fixada do GuideLLM, `time_per_output_token_ms` inclui o tempo inicial. Já `inter_token_latency_ms` é calculado como `(último token − primeiro token)/(número de tokens − 1)`. Usamos essa segunda medida no resumo como `mean_itl_ms`, evitando dar o mesmo nome a definições diferentes. Veja o [código da versão 0.7.4](https://github.com/vllm-project/guidellm/blob/v0.7.4/src/guidellm/schemas/request_stats.py).

Um evento SSE pode carregar mais de um token. Por isso exigimos suporte a usage na primeira requisição medida e mantemos as contagens do servidor no instrumento. Se usage estiver ausente, salvamos os tempos iniciais parciais e marcamos falha, sem fabricar contagens. Os timestamps continuam limitados pela granularidade dos eventos e buffering: não revelam exatamente quando cada token foi calculado na GPU. Verifique essa compatibilidade em cada versão do servidor.

O p95 de `mean_itl_ms` **não é o p95 de todas as pausas individuais do streaming**. Uma resposta pode ter uma pausa grande e ainda ter média aceitável. O relatório bruto preserva detalhes do instrumento para investigações posteriores; o resumo é deliberadamente menor.

## 6. O olhar estatístico

A unidade básica é a requisição. O resumo calcula percentis empíricos com interpolação linear, usando somente sucessos, separadamente por **fase**, cenário e repetição. Filtre `phase=measure` para analisar os blocos posteriores e `phase=warmup` para estudar o aquecimento. Falhas são mostradas, não convertidas em latência zero nem descartadas sem aviso. A primeira resposta tem valor individual, não um p95 calculado de uma amostra única.

A bateria formal padrão faz 50 requisições por cenário, em três repetições e com três warmups por cenário/repetição. O smoke reduz isso para três requisições e uma repetição. Ambos servem para detectar efeitos e regressões; p95/p99 com poucos sucessos são exploratórios. Isso não quer dizer que 100 garantam precisão: dependência temporal e variabilidade continuam importando.

Para a análise, examine séries temporais e dispersão, além da mediana. Não calcule o “p95 geral” tirando a média de p95 de blocos. Se combinar requisições, preserve os identificadores de cenário e repetição e explicite a população que está resumindo.

Repetir requisições no mesmo pod não cria réplicas independentes de hardware. Clocks, temperatura, estado de cache e vizinhos do host podem introduzir dependência. Para inferência mais rigorosa, desenhe blocos comparáveis, alterne a ordem das configurações e use métodos que respeitem essa estrutura. Não trate centenas de tokens da mesma resposta como centenas de réplicas independentes.

## 7. O que precisa ser mantido igual?

- GPU, quantidade de GPUs e condições do host; confira a CPU e a RAM também.
- Arquitetura, revisão do modelo e representação dos pesos, quando tecnicamente possível.
- Tokenizer, chat template e mensagens de sistema efetivas.
- Lista de cenários, sementes, quantidade de requisições e limite de saída.
- Temperatura e demais parâmetros de geração relevantes. Enviamos `temperature=0` e `top_p=1`; isso não garante igualdade bit a bit entre kernels.
- Estado de carregamento e política de reutilização de cache entre requisições.
- Versão do cliente e forma de acessar o servidor.

`requests_sha256` compara mensagens e limites de saída realmente enviados, desconsiderando o alias do modelo. Hashes diferentes significam que o teste não usou a mesma carga. Hashes iguais não provam que os runtimes aplicaram o mesmo template internamente.

**Artefato comum:** o nome `Q8_0` não basta para provar equivalência. Registre caminho local, hash do arquivo, tokenizer/template e versão do backend. Se um servidor não aceitar o GGUF, marque a condição como não suportada e não misture números de outro artefato.

## 8. Cache: a principal armadilha silenciosa

O modelo estar carregado na GPU é desejável neste protocolo. Já reaproveitar resultados de um prefixo anterior pode mudar radicalmente o prefill. São coisas diferentes. Também não estamos propondo desativar o KV cache normal usado para gerar os tokens de uma resposta.

Para uma comparação de requisições independentes, tentem desativar reutilização entre requisições, quando o runtime permitir, e registrem como fizeram. Se não for possível, mantenham essa limitação explícita. O cliente não possui um botão universal de limpeza de cache para todos os servidores.

As sementes de aquecimento diferem das sementes de medição, e os prompts variam. Isso reduz repetição exata, mas **não garante cache frio**: templates e prefixos comuns ainda podem ser reaproveitados. `cache_policy` é uma declaração do operador, não uma medição automática.

## 9. Memória e saúde do pod

O HTML incorpora telemetria por fase/GPU: memória máxima amostrada, capacidade total, utilização média/máxima, temperatura máxima e potência média/máxima. Sem NVIDIA ou permissões, mostra ausência, nunca zero. Este coletor não mede GPU Apple. Em hosts diferentes, os dados do cliente não descrevem o servidor.

### Tokens/s por faixa de KV: o que podemos afirmar?

Agrupamos requisições pela entrada real, incluindo template: [0,512), [512,1024), [1024,2048), [2048,4096), [4096,8192), [8192,16384), [16384,+∞). Mostramos contexto inicial/final, TTFT e duas velocidades. Escolha a grade sintética com `--input-tokens 256 512 1024 2048 3072`. Uma resposta que cruza fronteira permanece no grupo inicial; a unidade é a requisição, não o token individual.

Para atenção completa, uma sequência e cache K/V convencional: `bytes por token = 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento do KV`. O 2 representa K e V. O sweep formal usa `SWEEP_MEMORY_STEP_MB=256` de memória lógica e converte esse valor em tokens com `KV_BYTES_PER_TOKEN`: 57.344 bytes/token no Qwen2.5 7B (aprox. 4.465 tokens por passo) e 196.608 no 14B (aprox. 1.302 tokens por passo). Bits dos pesos não determinam dtype do KV. Sliding window, MLA, padding, prefixos compartilhados e sharding exigem outro tratamento; o comprimento final é lógico, não prova materialização do último token em KV.

`--collect-kv-metrics` consulta `/metrics` (~1 Hz): usa `vllm:kv_cache_usage_perc` ou o nome legado `vllm:gpu_cache_usage_perc`, sem somar ambos. É fração de blocos ocupados do pool, não bytes de VRAM. Veja a [documentação oficial](https://docs.vllm.ai/en/v0.12.0/design/metrics/). Engines/séries são separados. Ausência não vira zero; a atualização do servidor e a coleta podem perder transientes.

Assim, o HTML separa **velocidade por contexto**, **KV lógico estimado** e **ocupação observada do pool KV**. Velocidade instantânea por faixa exata de bytes físicos exigiria instrumentação adicional do servidor; a API HTTP portátil não oferece isso. Não fazemos uma correlação token a token que a amostragem não sustenta. O proxy de contexto/KV ajuda a organizar a carga, mas não autoriza concluir ocupação real do pool ou da VRAM sem o gauge/telemetria correspondente.

`gpu.csv` amostra memória ocupada, GPU utilization, temperatura e potência. Serve para procurar crescimento inesperado, aquecimento e interferência. O monitor é local ao cliente e não mede banda da GDDR6X, tráfego PCIe, FLOPs ou memória por processo.

O maior uso registrado é um **máximo amostrado**, que pode perder um pico entre leituras. A alocação reservada pelo runtime também não é igual ao volume de dados lidos. Não conclua “memory-bound” apenas porque a VRAM está quase cheia. Essa conclusão exige evidência adicional, por exemplo profiling e experimentos controlados.

## 10. Como explorar os parâmetros do vLLM

Primeiro execute a configuração que já funciona, sem tunar. Escolha um sintoma: TTFT alto com entrada longa, geração lenta ou falta de memória. Depois altere **uma decisão por vez**, mantendo o protocolo idêntico, e registre o comando exato do servidor.

Neste trabalho, priorize contexto máximo, política de cache e opções de execução que façam sentido para uma sequência. Reservar mais memória não aumenta fisicamente a banda da GPU. Não faça uma busca enorme em parâmetros de concorrência: eles são assunto do próximo trabalho.

Não incluímos flags de tuning automaticamente porque variam conforme versão e podem mudar o experimento. Consulte o `vllm serve --help` da instalação e a [referência oficial](https://docs.vllm.ai/en/latest/cli/serve/). Compare o baseline com uma mudança; se o efeito se mantiver, investigue por quê.

## 11. Capítulos por runtime

O protocolo comum fixa artefato, tokenizer, endpoint OpenAI compatível, `stream=true`, usage, uma requisição por vez, `--launch` em foreground e logs completos. O que muda é o comando de servidor e as métricas nativas.

### vLLM

Instale o vLLM em ambiente separado e valide `vllm serve --help`, `nvidia-smi` e `python -c 'import vllm'`. Sirva o GGUF local com o plugin/backend suportado pela versão instalada, tokenizer local correspondente e sem download implícito. Use `--launch` com argv direto, `--host 127.0.0.1`, `--port 8000`, alias fixo em `--served-model-name` e contexto documentado. Confirme `/v1/models`, `/v1/chat/completions` e `/metrics`.

Registre backend de atenção, dtype de KV, prefix caching, preempções e OOM. Falha de importação, CUDA ou carregamento é diagnóstico: preserve `server.log`, `serve --help`, versões e `nvidia-smi`, e não substitua o artefato.

### llama.cpp

Use `llama-server` compilado com CUDA compatível e valide `llama-server --help`. O servidor deve receber o GGUF local e seu SHA-256 documentado, por exemplo `llama-server -m /workspace/models/Qwen2.5-7B-Instruct-Q8_0.gguf --host 127.0.0.1 --port 8000`. Execute em foreground pelo `--launch`, sem daemonização, shell ou wrapper que se desprenda. Confirme `/v1/models` e `/v1/chat/completions` com streaming e usage.

Registre camadas na GPU, contexto, quantização do GGUF, slots e logs de carregamento. Se a API não fornecer usage/stream compatível, marque falha de protocolo e preserve a evidência; não estime tokens nem troque de modelo.

### Ollama

Prepare o blob local e registre digest, identificador e Modelfile. Execute `ollama serve` em foreground e carregue o artefato antes da medição; `ollama pull` durante o `run` invalida a comparação. Confirme `/v1/models` e `/v1/chat/completions`, streaming, usage e política de permanência do modelo.

Ollama pode responder à listagem antes de carregar pesos: compare processo → primeiro conteúdo e, quando disponível, `load_duration`. Se o blob não carregar, descarregar por memória ou divergir no usage, preserve o diagnóstico e marque falha; não declare sucesso parcial.

Uma falha de inicialização, readiness, stream ou contagem é resultado experimental válido como diagnóstico, mas não como desempenho comparável. Repare a causa e repita com o mesmo protocolo.

## 12. Checklist antes de chamar um resultado de melhor

1. Todas as fases terminaram sem erro? O manifest mostra `complete`?
2. Os comprimentos reais das respostas são comparáveis?
3. GPU, modelo, tokenizer, template, carga e cache estão documentados?
4. O ganho aparece em mais de uma repetição, sem tendência térmica óbvia?
5. A melhora é em TTFT, geração ou resposta total? Não são a mesma coisa.
6. Existe diferença de quantização impedindo atribuir o efeito só ao runtime?

Esse benchmark mede desempenho, **não qualidade**. Uma pequena bateria fixa de perguntas reais pode funcionar como checagem de regressão em outro relatório, sem misturar sua nota com as métricas de latência.

## Atualização 0.4: métricas e telemetria operacional

Antes da medição, `make prepare-all` encadeia `prepare-vllm`, `prepare-llama` e `prepare-ollama`. Cada alvo por runtime chama a etapa comum `prepare-benchmark`, que cria diretórios, baixa o GGUF e tokenizer, instala o cliente e valida configurações; depois valida somente o runtime escolhido. Essas etapas são preparação e não entram nos tempos do benchmark.

O protocolo formal usa `short`, `medium` e `long` automaticamente, com 50 requisições em `replay`, 1 repetição e 3 aquecimentos por cenário. A fixture de medição `qwen_chat_bench_v2.json` tem 50 perguntas técnicas variadas; cada request seleciona um turno diferente e envia o histórico determinístico até ele. A fixture de warmup `qwen_chat_warmup_v1.json` é separada e nunca entra nos percentis. As métricas principais têm nomes canônicos: **Time To First Token (ms)**, um valor por requisição do POST ao primeiro token, e **Tokens/s**, a taxa de decode sem o TTFT. Com uma repetição, não há estimativa de variabilidade entre blocos; aumente `BENCH_REPETITIONS` manualmente quando essa análise for necessária. Percentis são calculados sobre requisições bem-sucedidas.

O monitor escreve `gpu.csv`, `system.csv` e `events.csv` durante toda a vida do processo e gera `telemetry-summary.json` agrupado pelas fases. CPU/RAM/SSD são observações do host; VRAM/utilização/temperatura/potência vêm do `nvidia-smi`. Esses sinais relacionam mudanças a startup, primeiro POST, aquecimento, medida e encerramento, mas não medem o tempo de cada cópia PCIe: para isso, use Nsight/CUDA instrumentation.

Cada alvo formal (`make bench-vllm`, `make bench-llama`, `make bench-ollama`) inclui a bateria short/medium/long com uma repetição e depois reinicia o runtime para uma grade que começa em 1024 tokens e avança 256 MB de KV lógico por ponto. No Qwen2.5 7B, os pontos são aproximadamente 1024, 5489, 9954 e 14419 tokens. Nos pontos de sweep em `replay`, somente históricos que cabem no `ctx<N>` são selecionados; o servidor mantém pelo menos o `context_window` configurado, evitando transformar um teto artificial em falso OOM. `make bench-all` executa os três e resume os códigos de saída. Para diagnóstico rápido, `smoke-vllm`, `smoke-llama`, `smoke-ollama` são os alvos individuais e `smoke-all` encadeia os três. O percentual do pool KV exposto pelo vLLM e a VRAM usada são mantidos como séries distintas; nenhum deles é apresentado como “bytes de KV” sem coeficiente arquitetural verificado.

## Próximo passo

Volte ao [README](../../README.md) para instalar e rodar. Faça primeiro o smoke em cada servidor. Os testes automatizados usam um servidor simulado: validam o instrumento, não antecipam o resultado na RTX 3090.
