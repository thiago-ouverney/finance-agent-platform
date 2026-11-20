---
id: concurrent-agents-middleware
status: draft
hypothesis_outcome: pending
owner: thiago-ouverney
created: 2026-11-19
updated: 2026-11-19
---

# Middleware de concorrência para N agentes sobre um runtime llama.cpp

## Estado atual e evidência

- O benchmark oficial do runtime envia uma requisição por vez. Concorrência é
  declarada como "outro experimento, com spec e workload próprios" em
  [`docs/architecture.md`](../../docs/architecture.md),
  [`services/inference-runtime/AGENTS.md`](../../services/inference-runtime/AGENTS.md)
  e na metodologia do benchmark.
- O Problema 2 (apresentação de 2026-11, campanha `test30/mopep-single`,
  RunPod RTX 3090, Qwen2.5-7B-Instruct Q8_0 GGUF) concluiu que llama.cpp é o
  melhor runtime sequencial: TTFT p95 219–291 ms, decode ≈ 95 tok/s, E2E médio
  0,607 s. Fonte: `analytics/notebooks/analyze_runtime_observability.ipynb`.
- O lançador de llama.cpp usado no benchmark fixa `--parallel 1`
  ([`configs/launch-llamacpp-7b-gguf.json`](../../services/inference-runtime/configs/launch-llamacpp-7b-gguf.json)).
  llama-server possui fila interna por slot; o comportamento dessa fila sob N
  clientes simultâneos **não foi medido** neste projeto.
- A UFF hoje serve Ollama (`qwen2.5:7b-instruct-q4_K_M`) para o WhatsApp em
  `127.0.0.1:11434` ([runbook](../../docs/uff-ollama-whatsapp.md)). GPU, VRAM
  e presença de `llama-server`/GGUF na UFF **não são conhecidos pelo
  repositório** e precisam de inspeção.
- O motor de medição é GuideLLM 0.7.4 fixado, com workaround em
  [`guidellm_compat.py`](../../services/inference-runtime/guidellm_compat.py).
  O `bench.py` só emite o perfil `synchronous`; os perfis `concurrent` e
  `constant` do GuideLLM ainda não são usados.
- O adapter do WhatsApp serializa mensagens por contato e depende só do
  contrato OpenAI-compatible, não da implementação do runtime.

## Hipótese

Um middleware entre os agentes e um único `llama-server` — com controle de
admissão, fila limitada, justiça entre agentes, timeouts e saída estruturada
validada — mantém o runtime disponível e as respostas íntegras sob carga
simultânea, com degradação previsível (latência de fila explícita, rejeições
controladas) em vez de travamento, mistura de respostas ou colapso.

Hipótese nula a refutar primeiro: `llama-server` sozinho (com sua fila
interna) já sustenta N agentes sem falha observável. Se a baseline não
reproduzir falha nem degradação relevante, o middleware é justificado apenas
por justiça, observabilidade e backpressure, e a spec registra isso.

## Objetivo observável

Sistema multi-agente em closed-loop contra um `llama-server` compartilhado
no Pod RunPod desta iniciativa, sem travar nem misturar respostas, com dados
empíricos comparando o runtime nu e o mesmo runtime atrás do middleware.

Métricas por tratamento e nível de concorrência:

| Métrica | Pergunta |
|---|---|
| taxa sustentada (req/s) e throughput de tokens | quanto o runtime aguenta sem degradar? |
| TTFT, E2E e ITL p50/p95 | como a latência degrada com N? |
| tempo em fila vs. tempo de serviço | onde a espera acontece? |
| erros, timeouts, rejeições (429/503) | a falha é explícita ou silenciosa? |
| disponibilidade do `/health` durante a carga | o runtime travou? |
| `echo_mismatch_rate` | alguma resposta foi entregue ao agente errado? |
| taxa de JSON inválido | a saída estruturada se mantém sob carga? |
| GPU, VRAM, RAM e KV cache (Prometheus) | o hardware está eficiente ou só sobrecarregado? |

`echo_mismatch_rate`: cada requisição carrega `agent_id` e `request_id`
(nonce) no prompt e exige que o JSON de saída os ecoe; qualquer divergência é
evidência objetiva de mistura de estado, independente de inspeção manual.

## Escopo

Incluído:

- inventário do Pod (GPU, VRAM, disco, binários, modelo) antes de medir;
- duas baselines sequenciais de llama.cpp nesse Pod, com o protocolo do
  Problema 2;
- curva closed-loop do runtime nu com GuideLLM `concurrent`, nos streams e
  slots já decididos;
- middleware OpenAI-compatible com admissão, fila limitada, justiça por agente,
  timeouts, streaming passthrough, validação de JSON Schema e métricas;
- repetição da mesma matriz através do middleware;
- demonstração ao vivo com N processos, nos níveis escolhidos depois da curva;
- notebook de análise e documentação canônica.

Fora de escopo:

- broker externo (Redis/RabbitMQ) na primeira versão; fica registrado como
  alternativa, com critério para revisitar;
- alterar o benchmark oficial sequencial ou seus resultados;
- integrar o middleware ao adapter do WhatsApp em produção;
- Problema 4 (rede entre máquinas da equipe);
- comparação entre runtimes sob concorrência (Ollama/vLLM) — llama.cpp é o
  runtime fixado por decisão do Problema 2.

## Topologia, contratos e fontes de verdade

```text
agente 1 ─┐
agente 2 ─┼─ HTTP /v1/chat/completions ─> middleware (admissão, fila, justiça,
agente 3 ─┘                                 validação JSON, métricas)
GuideLLM ─┘                                        │ httpx streaming
                                                   v
                                           llama-server (--parallel k)
                                                   │
                                           GPU do Pod (RTX 4090)
```

- O middleware expõe o mesmo contrato OpenAI-compatible que o runtime; agentes,
  adapter e GuideLLM não precisam saber se falam com o runtime ou com o
  middleware. Isso mantém a regra de `docs/architecture.md`.
- Saída estruturada usa `response_format` com JSON Schema, que o
  `llama-server` converte em gramática GBNF nativamente; o middleware valida a
  resposta e registra falhas, sem reescrevê-la.
- Placement: novo componente `services/inference-gateway` (dono de admissão,
  fila e validação). Direção de dependência: gateway -> API HTTP do runtime;
  nenhum import cruzado com `inference-runtime`, `analytics` ou `apps`.
- Workloads e resultados seguem os contratos de arquivo do runtime; a análise
  fica em `analytics`.

## Compatibilidade, rollout e remoção

Mudança aditiva. Nenhum alvo Make ou resultado existente é alterado. O
benchmark sequencial continua sendo o oficial; o experimento de concorrência
tem perfis, alvos e diretórios de resultado próprios.

## Dados, segurança e efeitos externos

- Target remoto: o Pod RunPod desta iniciativa, alcançado pelo alias
  `runpod-qwen` depois de apontá-lo para esse Pod. Serviços em `127.0.0.1`,
  sem porta publicada. A UFF e o WhatsApp ficam fora da medição.
- Hardware declarado em 2026-11-19: 1× RTX 4090, 16 vCPU, 83 GB de RAM,
  50 GB de disco de container, uptime de 15 s no momento do relato. Não há
  volume de rede informado. O disco é efêmero: resultados saem do Pod antes
  de ele ser encerrado.
- Workload MOPEP contém BMCs privados: prompts, respostas e resultados brutos
  permanecem locais e ignorados pelo Git; só agregados sanitizados vão para
  docs/notebooks.
- Modelo fixado por revisão/SHA-256 no manifest da run, nunca `latest`.
- Condições de parada: erro de autenticação SSH, VRAM insuficiente para o
  modelo escolhido, ou qualquer run que exija publicar porta/host.

## Decisões aprovadas

Aprovadas em 2026-11-19 pelo owner. O detalhe e as alternativas ficam em
`decisions.md` (pack local).

1. **Execução remota pelo usuário, no Pod RunPod.** O agente prepara
   comandos, código local, testes e análise, e só trata como fato a evidência
   devolvida. O alias `runpod-qwen` precisa apontar para este Pod antes do G0.
2. **Medição fora da UFF.** O Ollama e o WhatsApp não são pausados nem usados.
   O Pod é dedicado a este experimento.
3. **Modelo Qwen2.5-7B-Instruct Q8_0 GGUF**, o mesmo artefato do Problema 2,
   com SHA-256 no manifest. VRAM insuficiente reabre o modelo; não há troca
   silenciosa.
4. **Middleware em `services/inference-gateway`.** FastAPI + httpx, passthrough
   SSE, mesmo contrato OpenAI-compatible. Sem Redis/RabbitMQ na v1.
5. **Workload MOPEP `mopep-single` na calibração**, com JSON Schema. Sintético
   256/128 do GuideLLM só no smoke. O split de teste permanece fechado.
6. **Joelho definido depois dos dados.** Antes de medir ficam congelados os
   streams `[1, 2, 4, 6, 8, 12, 16, 24, 32]`, 3 repetições, 128 requisições por
   repetição, as colunas (req/s, tok/s, TTFT p50/p95, E2E p50/p95, erros,
   `/health`) e o corte por bucket. Não há limiar numérico prévio.
7. **Slots e contexto.** Contexto por requisição fixo em 8.192, igual ao
   protocolo do Problema 2 ("contexto 8K"). Baseline `--parallel 1 --ctx-size
   8192`; 8 slots com `--ctx-size 65536`; 16 slots com `--ctx-size 131072`.
   Assim a única variável entre os tratamentos é o número de slots. Se 16 não
   couber na VRAM, o recuo é 4.096 por requisição, declarado como outra
   variável. `GET /props` registra o `n_ctx` de cada slot, que nesta revisão é
   `--ctx-size` dividido por `--parallel`. Contexto curto demais produz
   HTTP 400, não colapso.
   (Correção: a versão anterior, 16384 na baseline, veio de um launcher legado.)
8. **Cache ligado na curva principal**, igual ao Problema 2. Um braço com
   `--no-cache-prompt`, uma repetição, em `--parallel` 1, 8 e 16, só para
   análise do TTFT. Fora do joelho.
9. **Só closed-loop.** Cada agente espera a resposta antes do próximo pedido.
   Open-loop (`constant`) fica fora. O relatório não afirma colapso por chegada
   sem espera.
10. **Dois instrumentos.** GuideLLM mede tempo. Um cliente de N processos mede
    conteúdo: eco de `agent_id`/`request_id`, acordo com a rodada sequencial e
    Macro-F1 na calibração. O ruído é a diferença entre duas rodadas
    sequenciais. O defeito injetado que troca respostas fica no teste de código.
11. **Admissão do gateway sai do G2.** `max_inflight` igual ao número de slots;
    o tamanho da fila e o 429 com `Retry-After` usam a latência observada, sem
    limiar congelado antes dos dados.

## Riscos

| Risco | Mitigação | Residual |
|---|---|---|
| RTX 4090 difere da RTX 3090 do Problema 2 | esta run é a baseline; números absolutos não se comparam | a leitura entre problemas fica qualitativa |
| disco de container de 50 GB, sem volume | G0 mede o espaço livre antes de baixar o GGUF; não baixar Safetensors nem outros runtimes | o Pod novo pode não ter folga para o modelo |
| disco efêmero | copiar resultados para a máquina local antes de encerrar o Pod | perder a evidência se o Pod for destruído |
| contexto por slot pequeno demais com `--parallel` alto | subir `--ctx-size` junto com os slots; conferir `n_ctx` em `/props` | HTTP 400 se o GGUF da UFF dividir o contexto de outro jeito |
| 16 slots não cabem na VRAM | recuar para 4.096 por slot; falha de subida encerra o tratamento | o ponto de 16 pode ficar de fora |
| gramática JSON altera custo de decode | medir com e sem `response_format`, uma variável por vez | overhead permanece como custo do requisito |
| middleware mascarar TTFT se não fizer streaming | passthrough SSE desde a primeira versão; teste focal | — |
| GuideLLM 0.7.4 com perfil `concurrent` ter outra corrida de condição | smoke com poucos requests e checar completude antes da matriz | pode exigir workaround adicional |

## Critérios de aceite

- [x] Inventário do Pod registrado em 2026-11-19: RTX 4090, 24564 MiB,
      50 GB livres no overlay, `llama-server` e GGUF ausentes. Evidência em
      `acceptance.md`.
- [ ] Duas baselines sequenciais na UFF, com manifest, como piso de ruído.
- [ ] Curva closed-loop nos streams congelados, em `--parallel` 1, 8 e 16, com
      o contexto por slot da decisão 7, mais o braço sem cache. O joelho é
      conclusão do notebook.
- [ ] Cliente de conteúdo com eco de ids, acordo de tags e Macro-F1 na
      calibração, comparado às duas rodadas sequenciais.
- [x] Middleware com testes locais verdes para admissão, fila limitada,
      justiça, timeout, rejeição explícita, streaming, schema e o defeito
      injetado de troca de respostas. Evidência em `acceptance.md`, G3.1–G3.7.
- [ ] A mesma curva através do middleware, com fila e 429 explícitos.
- [ ] Demonstração ao vivo nos níveis escolhidos depois da curva, com
      telemetria Prometheus e resumo agregado sanitizado.
- [ ] Notebook de análise com conclusão por tratamento e `hypothesis_outcome`
      atualizado nesta spec.
- [ ] `make test-governance` e gates do componente dono passam.
