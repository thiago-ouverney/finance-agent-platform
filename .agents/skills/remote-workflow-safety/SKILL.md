---
name: remote-workflow-safety
description: Decide e documenta operações remotas do projeto quando ambiente, autorização, custo, dados privados, revisão imutável e evidência precisam ser confirmados antes da execução.
---

# Segurança para workflows remotos

Use esta lente antes de executar ou propor um passo que alcance UFF, RunPod,
Hugging Face, OpenAI, BigQuery ou uma sessão real do WhatsApp.

## Questions

- Qual sistema, conta, namespace, host e ambiente são o target exato?
- A ação é somente leitura ou grava estado, publica artefato, transfere dado ou
  envia mensagem? Quem autorizou a escrita?
- Que custo, timeout, volume, concorrência e condição de parada se aplicam?
- Quais dados, identificadores, logs ou artefatos são privados e onde podem ser
  armazenados?
- Modelo, dataset, imagem, ferramenta e código estão fixados por revisão, digest
  ou hash imutável?
- Existe preflight, smoke, dry-run ou amostra mínima que reduz o risco?
- Qual evidência sanitizada prova o resultado sem expor segredo ou PII?

## Avoid

- Não deduza estado remoto de arquivos locais ou de uma execução passada.
- Não alterne silenciosamente identidade, host, runtime, namespace ou dataset
  depois de falha.
- Não inclua secret em argumento, Make variable, log, spec ou relatório.
- Não publique peso, dado, resultado bruto ou repositório como público por
  conveniência.
- Não misture download, setup ou warmup à janela medida de um benchmark.
- Não introduza concorrência em benchmark sequencial sem uma spec própria.

## Stop or escalate

Pare antes da ação quando o target ou owner for ambíguo, faltar autorização de
escrita, o custo não tiver limite, um secret apareceria em evidência, a revisão
for mutável, o destino puder ser público ou o único avanço exigir substituir o
runtime/ambiente aprovado.

Falha de autenticação não autoriza tentar outra identidade. Diferencie rede,
autenticação, autorização, quota, compatibilidade e erro do workload antes de
propor recuperação.

## Evidence

Registre, sem valores sensíveis:

- data, operador e target lógico;
- commit local e revisões/digests dos insumos;
- comando ou target executado com parâmetros não secretos;
- preflight, exit code, timestamps e condição de parada;
- hashes/manifests e local dos outputs;
- efeitos observados no sistema dono e verificações não executadas.

## Handoff

Informe fatos observados, efeitos externos realizados, dados transferidos,
artefatos criados/publicados, IDs ou hashes sanitizados, custo conhecido,
rollback, riscos restantes e próxima ação. Separe claramente estado verificado
de proposta ou reprodução ainda não executada.
