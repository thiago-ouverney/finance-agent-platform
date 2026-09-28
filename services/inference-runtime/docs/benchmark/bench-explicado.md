# Bench explicado

> **Objetivo:** explicar, em linguagem direta, o que o benchmark mede, como ele
> executa e quais conclusões seus resultados permitem.

## A ideia em uma frase

O benchmark mede a experiência de **uma pessoa usando um chatbot**, uma requisição por vez, observando o servidor por HTTP/SSE e registrando latência, tokens, memória e utilização da GPU.

Ele não é um teste de usuários simultâneos, não mede qualidade das respostas e não executa a rede neural diretamente. Quem executa a inferência é o runtime: vLLM, llama.cpp ou Ollama. O projeto prepara o ambiente, inicia o runtime quando solicitado, envia a carga e organiza as evidências.

## O mapa completo

```text
prepare-<runtime>
        │ modelo + tokenizer + configs + runtime validado
        ▼
smoke-<runtime> ── diagnóstico rápido
        │
        ▼
bench-<runtime>
        ├── bateria base: short / medium / long
        │     └── independent, replay ou closed-loop
        └── sweep de contexto/KV: 1024 tokens + 256 MB de KV por ponto
```

`bench-all` executa essa combinação para vLLM, llama.cpp e Ollama. `smoke-all` executa somente o diagnóstico rápido dos três.

## Não confunda os três conceitos

| Conceito | O que é | O que muda | Para que serve |
|---|---|---|---|
| Bateria base | `bench-<runtime>` chamando `bench.py` | cenários, repetições e fases no mesmo processo | medir TTFT, decode e latência ponta a ponta |
| `replay` | modo `BENCH_MODE=replay` da bateria base | envia histórico com respostas fixas do fixture | repetir exatamente uma conversa conhecida |
| `closed-loop` | modo conversacional da bateria base | coloca a resposta real no turno seguinte | estudar crescimento de histórico com respostas reais |
| Sweep de contexto/KV | `kv-sweep-<runtime>`, incluído no alvo formal | reinicia o servidor e aumenta o contexto a cada ponto | observar custo de contexto, VRAM, TTFT e falhas |

`replay` não é o sweep. Replay é uma política de mensagens; sweep é uma série de execuções com tamanhos de contexto diferentes. É possível usar `SWEEP_MODE=replay`, mas ainda serão dois eixos experimentais diferentes.

## Parâmetros efetivos do Makefile

Quando você executa `make bench-vllm`, `make bench-llama` ou `make bench-ollama` sem sobrescrever variáveis, estes são os valores usados:

| Variável | Valor padrão | Efeito prático |
|---|---:|---|
| `BENCH_SCENARIOS` | `short medium long` | três tamanhos de entrada na bateria base |
| `BENCH_REQUESTS` | `50` | 50 requisições de medição por cenário e repetição |
| `BENCH_REPETITIONS` | `1` | um bloco por cenário; aumente manualmente para estudar variabilidade |
| `BENCH_WARMUP` | `3` | três requisições descartadas da medição por cenário/repetição |
| `BENCH_MODE` | `replay` | histórico determinístico com respostas fixas da fixture |
| `CONVERSATION_TURNS` | `1` | só tem efeito quando o modo conversacional é usado |
| `CONVERSATION_FIXTURE` | `qwen_chat_bench_v2.json` | 50 perguntas técnicas variadas para a medição |
| `WARMUP_CONVERSATION_FIXTURE` | `qwen_chat_warmup_v1.json` | perguntas separadas, usadas somente no warmup |
| `BENCH_STARTUP_TIMEOUT` | `1800` s | limite para readiness, não para a execução inteira |
| saída máxima | `128` tokens | limite solicitado no corpo HTTP; o modelo pode parar antes |
| `SWEEP_START` | `1024` | primeiro contexto do sweep |
| `SWEEP_MEMORY_STEP_MB` | `256` | incremento de memória lógica de KV entre pontos formais |
| `KV_BYTES_PER_TOKEN` | `57344` (7B) | coeficiente arquitetural usado para converter MB de KV em tokens |
| `SWEEP_MAX_CONTEXT` | `16384` | último contexto permitido |
| `SWEEP_REQUESTS` | `50` | herdado de `BENCH_REQUESTS` |
| `SWEEP_REPETITIONS` | `1` | herdado de `BENCH_REPETITIONS` |
| `SWEEP_WARMUP` | `3` | herdado de `BENCH_WARMUP` |

### Quantas requisições isso realmente representa?

Na bateria base, cada runtime faz, por cenário e repetição:

```text
3 warmup + 50 measure = 53 requisições
3 cenários × 1 repetição × 53 = 159 requisições
+ 1 primeira resposta
 + 1 referência final aquecida
```

O sweep usa um cenário de contexto por ponto. Com `1024..16384` e passos de 256 MB de KV lógico, o Qwen2.5 7B usa aproximadamente 4 pontos: 1024, 5489, 9954 e 14419 tokens. Cada ponto faz:

```text
3 warmup + 50 measure = 53
1 repetição × 53 = 53
+ primeira resposta + referência final = 55 requisições/probes por ponto
4 pontos × 55 = 220 requisições/probes por runtime
```

É por isso que o `bench-<runtime>` formal é longo: ele combina a bateria base com um sweep que reinicia e carrega o servidor em cada ponto. `bench-all` executa isso uma vez em cada runtime.

## Exemplo prático do corpo HTTP

No padrão `replay`, uma requisição enviada pelo cliente se parece com:

```json
{
  "model": "qwen7b-q8-gguf",
  "messages": [
    {"role": "system", "content": "Voce e um assistente tecnico conciso..."},
    {"role": "user", "content": "Explique a diferenca entre RAM do sistema e VRAM..."},
    {"role": "assistant", "content": "RAM atende o sistema e a movimentacao de dados..."},
    {"role": "user", "content": "Por que um modelo quantizado ainda pode usar KV-cache em FP16?"}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 128,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

As 50 requisições de cada bloco percorrem os 50 turnos da fixture. A primeira carrega a primeira pergunta; a última carrega o histórico completo até a pergunta 50. As respostas `assistant` são fixas e determinísticas; a resposta nova do runtime não é inserida na próxima requisição.

O warmup usa um corpo parecido, mas vem de `qwen_chat_warmup_v1.json`, com system e perguntas diferentes. Ele é marcado como `phase=warmup` e não entra nos percentis.

```json
{
  "messages": [
    {"role": "system", "content": "Voce e um assistente tecnico conciso..."},
    {"role": "user", "content": "Explique em duas frases a diferenca entre RAM ..."},
    {"role": "assistant", "content": "RAM guarda dados e processos ..."},
    {"role": "user", "content": "Agora relacione essa diferenca com o KV-cache ..."}
  ],
  "max_tokens": 128,
  "stream": true
}
```

No `closed-loop`, a mensagem `assistant` do turno anterior seria a resposta real recebida do runtime, e não o texto fixo do fixture.

### Exemplo didático: três formas de enviar a mesma interação

Considere o primeiro turno do fixture:

```text
system: Voce e um assistente tecnico conciso. Responda em portugues...
user:   Explique em duas frases a diferenca entre RAM do sistema e VRAM da GPU...
```

No `independent`, cada requisição continua curta e sem histórico:

```text
POST 1 → [user: "word17 word83 ... Explique de forma objetiva..." ] → resposta 1
POST 2 → [user: "word42 word11 ... Explique de forma objetiva..." ] → resposta 2
POST 3 → [user: "word08 word65 ... Explique de forma objetiva..." ] → resposta 3
```

Mesmo que a resposta 1 fale sobre VRAM, ela não aparece no POST 2. Cada requisição recebe um marcador determinístico próprio, preservando exatamente o tamanho-alvo do cenário. Warmup e medição usam sementes disjuntas para evitar que o conjunto medido seja apresentado integralmente durante o aquecimento.

No `replay`, cada turno cresce com o histórico fixo:

```text
turno 1 → [system, user_1]
turno 2 → [system, user_1, assistant_1_fixo, user_2]
turno 3 → [system, user_1, assistant_1_fixo, user_2,
           assistant_2_fixo, user_3]
```

No `closed-loop`, a forma é igual, mas as respostas vêm do runtime:

```text
turno 1 → [system, user_1]                    → resposta_real_1
turno 2 → [system, user_1, resposta_real_1, user_2] → resposta_real_2
turno 3 → [system, user_1, resposta_real_1, user_2,
           resposta_real_2, user_3]
```

Assim, `replay` permite repetir a mesma conversa com entradas determinísticas; `closed-loop` mede uma conversa que evolui conforme as respostas produzidas. Em ambos, `prompt_tokens` é a contagem real do histórico enviado naquele turno, não apenas o tamanho do texto do último `user`.

### Exemplo didático: o que muda no sweep

O sweep não acrescenta `assistant` ao histórico. Ele mantém a política de carga escolhida e muda o tamanho do contexto alvo:

```text
ponto 1024 → prompt sintético de ~1024 tokens → servidor reinicia
ponto 2048 → prompt sintético de ~2048 tokens → servidor reinicia
ponto 3072 → prompt sintético de ~3072 tokens → servidor reinicia
```

Cada ponto gera seu próprio `manifest.json`, `first-request.json`, `summary.json`, telemetria e log. Se o modo do sweep for `replay`, então cada ponto combina as duas coisas: histórico conversacional reproduzido e novo limite de contexto. Isso aumenta o custo experimental e não transforma o sweep em uma simples repetição da bateria base.

## Exemplos de execução controlada

A configuração formal padrão é equivalente a:

```bash
make bench-vllm MODEL_SIZE=7B PREPARE_OFFLINE=1
```

Para um diagnóstico pequeno no Pod, reduza explicitamente todas as dimensões:

```bash
make bench-vllm MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  BENCH_SCENARIOS=short BENCH_REQUESTS=1 BENCH_REPETITIONS=1 BENCH_WARMUP=0 \
  SWEEP_MAX_CONTEXT=1024 SWEEP_REQUESTS=1 SWEEP_REPETITIONS=1 SWEEP_WARMUP=0
```

Para uma conversa reproduzível na bateria base:

```bash
make bench-vllm MODEL_SIZE=7B PREPARE_OFFLINE=1 \
  BENCH_MODE=replay CONVERSATION_TURNS=3 \
  BENCH_SCENARIOS=short BENCH_REQUESTS=10 BENCH_REPETITIONS=3 BENCH_WARMUP=3
```

Esses overrides mudam o experimento e precisam ficar registrados no `manifest.json`; não compare o resultado reduzido com a bateria formal sem declarar a diferença.

### Como funciona o modo `replay`

`replay` é o modo padrão da bateria base. Cada requisição começa uma conversa determinística nova, mas seleciona um turno diferente da fixture:

```text
requisição 1: [system, user 1]
requisição 2: [system, user 1, assistant 1, user 2]
requisição 3: [system, user 1, assistant 1, user 2, assistant 2, user 3]
...
requisição 50: histórico até user 50
```

O cliente não coloca a resposta produzida pelo runtime na requisição seguinte: usa somente o `assistant` fixo da fixture. Isso mantém a carga textual controlada e reproduzível, mas permite observar históricos crescentes. `independent` continua disponível para medir mensagens isoladas; não é o padrão.

Isso é independência **da conversa**, não garantia de estado frio. O servidor continua vivo durante warmup e measure e pode manter pesos na GPU, caches de prefixo, alocadores e estruturas internas. Assim, `replay` responde “quanto custa servir históricos controlados em um processo aquecido?”, não “quanto custa iniciar um runtime frio a cada requisição?”. Para cold starts, o processo precisa ser reiniciado entre execuções, como no sweep ou em chamadas separadas com `--launch`.

No `independent`, a resposta nunca altera a próxima entrada. No `replay`, o fixture fornece o histórico e respostas `assistant` fixas; no `closed-loop`, as respostas reais entram no turno seguinte. Essa é a diferença operacional entre os três modos.

## O que acontece em uma execução

### 1. Preparação — fora do relógio

`prepare-vllm`, `prepare-llama` e `prepare-ollama` dependem da etapa comum `prepare-benchmark`. Ela confere modelo, tokenizer, configs, imports, assinatura e arquivos locais. A preparação não inicia uma medição e não entra na latência.

```bash
make prepare-all MODEL_SIZE=7B
```

O alvo por runtime já chama a etapa comum. Não é necessário executar `prepare-benchmark` manualmente antes.

### 2. Partida e readiness

Com `--launch`, o benchmark inicia o processo do runtime em foreground e registra o grupo criado. Em seguida consulta `/v1/models` até encontrar o alias esperado. Isso mede `process_to_api_observed_s`, mas **API disponível não significa necessariamente modelo já residente na GPU**.

Se o servidor já existia e `--launch` não foi usado, a idade do processo e o carregamento dos pesos são desconhecidos. O cliente mede apenas a primeira requisição que ele observou.

### 3. Primeira resposta

O primeiro POST é enviado sem uma geração anterior escondida. O cliente mede:

- tempo do processo até o primeiro conteúdo;
- TTFT da requisição;
- tempo total até o fim do stream;
- `usage`, quando o servidor fornece contagem de tokens.

Essa é uma **sonda de primeira resposta**, registrada em `first-request.json` e no lifecycle. Ela não é misturada ao p50 da fase `measure`.

### 4. Warmup

O warmup envia requisições reais antes da medição formal. Nos modos conversacionais, usa `WARMUP_CONVERSATION_FIXTURE`, uma fixture separada da medição; no modo `independent`, usa sementes distintas das requisições medidas. Elas podem aquecer compilação, buffers, alocadores, HTTP/SSE, tokenizer, prefixos e estruturas KV. Por padrão são três por cenário e repetição.

Warmup não é TTFT:

```text
TTFT  = uma métrica: POST → primeiro conteúdo daquela requisição
warmup = uma fase: requisições usadas para preparar o estado do processo
```

O warmup não limpa caches, não prova estabilidade e não entra na fase `measure`. Seus dados ficam registrados como `phase=warmup` para investigar tendência. Com `BENCH_WARMUP=0`, a execução fica mais barata, mas mais dependente do estado inicial.

### 5. Medição oficial

Na fase `measure`, o cliente envia uma requisição por vez. Para cada resposta bem-sucedida calcula TTFT, intervalo entre tokens, tokens/s de decode, tokens/s efetivos e latência total. A bateria formal padrão usa 50 perguntas variadas em `replay`, uma repetição e três warmups com fixture separada por cenário.

`short`, `medium` e `long` representam alvos de 256, 2048 e 7680 tokens de
entrada, com teto de 128 tokens de saída. No `replay`, o alvo considera o prompt
completo: o cliente seleciona o histórico fixo mais próximo, remove pares
antigos acima do alvo e adiciona contexto sintético determinístico abaixo dele.
O servidor informa `usage.prompt_tokens`; essa contagem real deve ser usada na
comparação, mantendo a estimativa como diagnóstico.

### 6. Referência final e cleanup

O prompt inicial é repetido no final como referência aquecida. Depois, o launcher encerra somente o processo criado pelo benchmark e fecha os coletores. A referência ajuda a comparar o primeiro acesso com um estado posterior, mas pode aproveitar cache de prefixo; não isola sozinha um único custo.

## O que é TTFT, decode e latência total?

```text
envio do POST ───── primeiro conteúdo ───── últimos tokens ───── fim HTTP
                  ←──── TTFT ────→
                  ←──────── geração/decode ────────→
←──────────────── latência ponta a ponta ───────────────────────→
```

| Métrica | Fórmula conceitual | Inclui TTFT? |
|---|---|---:|
| TTFT | primeiro conteúdo − início do POST | é o próprio intervalo |
| Decode tokens/s | tokens de saída ÷ tempo entre primeiro e último token | não |
| Tokens/s efetivos | tokens de saída ÷ latência total | sim |
| Latência ponta a ponta | fim da resposta − início do POST | sim |

Uma configuração pode ter TTFT menor e decode mais lento. Também pode gerar menos tokens e parecer mais rápida na taxa efetiva. Por isso o relatório preserva contagens de entrada/saída e não compara uma métrica isolada.

## O segundo bench: sweep de contexto/KV

O sweep não continua a mesma amostra estatística da bateria base. Ele reinicia o runtime em cada ponto para alterar o limite de contexto e medir o efeito de cargas crescentes:

```text
contexto 1024  → ~0 MB adicional da grade → inicia servidor → executa bench.py → encerra
contexto 5489  → ~256 MB adicionais de KV → inicia servidor → executa bench.py → encerra
contexto 9954  → ~512 MB adicionais de KV → inicia servidor → executa bench.py → encerra
contexto 14419 → ~768 MB adicionais de KV → inicia servidor → executa bench.py → encerra
```

Cada ponto tem startup, primeira resposta, warmup, medição, telemetria e diretório próprio. Nos pontos `ctx<N>`, o replay seleciona e cicla somente os turnos cujo histórico real cabe em `N` tokens. O servidor mantém pelo menos o `context_window` configurado no runtime, em vez de receber um teto artificial `N + 128 + 256`; assim o limite do servidor não mascara a carga que o ponto pretende observar. Por isso o sweep é mais lento: o custo dominante é reiniciar e carregar o runtime repetidamente, não apenas aumentar o orçamento de KV em 256 MB.

Para exploração rápida:

```bash
make quick-sweep-vllm MODEL_SIZE=7B SWEEP_MAX_CONTEXT=1024
```

O padrão agora é `SWEEP_MEMORY_STEP_MB=256`. Para uma grade mais espaçada e rápida, altere esse valor para `512`, `1024` ou `2048`. O código converte o valor para tokens conforme `KV_BYTES_PER_TOKEN`; o ponto falho não vira zero nem é substituído por outro modelo.

`SWEEP_MEMORY_STEP_MB` representa 256 MB de KV lógico, não 256 tokens. Para o Qwen2.5 7B em KV FP16, o coeficiente usado é 57.344 bytes/token (`2 × 28 camadas × 4 cabeças KV × 128 dimensão × 2 bytes`), então 256 MB correspondem a `256.000.000 / 57.344 = 4.464,3` tokens; o código arredonda para cima e usa 4.465 tokens por salto. Partindo de 1024, os pontos ficam aproximadamente em 1024, 5489, 9954 e 14419 tokens. Isso estima KV lógico, não VRAM total: pesos, buffers, blocos, overhead e política de cada runtime ficam fora dessa conta. Para 14B, o Make usa 196.608 bytes/token e recalcula a grade.

## Smoke, base e sweep: quando usar cada um

| Comando | Duração | O que valida | É resultado formal? |
|---|---:|---|---:|
| `smoke-<runtime>` | curta | processo, readiness, API, tokenizer, streaming e cleanup | não |
| `bench-<runtime>` | longa | bateria base + sweep do runtime | sim, se todos os blocos forem completos |
| `quick-sweep-<runtime>` | curta por ponto | integração do launcher e contexto | não; é exploratório |
| `bench-all` | mais longa | os três runtimes, continuando após falhas | sim, por runtime e protocolo |

## Como ler os resultados

Cada execução nova fica em:

```text
results/<runtime>/<timestamp>/<nome-humano>/
├── html/   summary.html, lifecycle.html e gráficos SVG
├── json/   manifestos, métricas e respostas brutas
├── csv/    GPU, CPU/RAM, eventos e requests
└── logs/   server.log e telemetry.log
```

Comece por `html/summary.html`. Depois confira `json/manifest.json` para status, configuração e contagens; `csv/gpu.csv` para memória/utilização; e `logs/server.log` para startup e erros. Uma mensagem no log de shutdown depois de HTTP 200 não deve ser classificada automaticamente como falha: confira o status do manifesto e o código do alvo.

## Perguntas que o benchmark responde

- O runtime está pronto e serve o alias esperado?
- Qual é o TTFT da primeira resposta e das requisições aquecidas?
- Como o tempo entre tokens e a latência total mudam com o cenário?
- O histórico conversacional real ou reproduzido muda a carga?
- Como memória GPU, GPU-utilização e TTFT se comportam quando o contexto cresce?
- Em qual ponto o runtime falha por startup, contexto ou memória?

Ele não responde sozinho quanto tempo cada kernel CUDA levou, se a GPU está memory-bound, qual runtime tem melhor qualidade de resposta ou como o sistema atende muitos usuários simultâneos. Essas perguntas exigem profiling, avaliação de qualidade ou um benchmark concorrente separado.
