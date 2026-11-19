# Instruções do pipeline de quantização

Este guia complementa o [`AGENTS.md` raiz](../../AGENTS.md).

## Papel e fronteira

O pipeline seleciona calibração, constrói, verifica e opcionalmente publica
artefatos quantizados. O runtime mede serving/performance e analytics avalia
datasets e resultados; o pipeline não deve absorver essas responsabilidades.

O contrato operacional atual está no
[`README.md`](README.md) e nos guias de quantização referenciados por ele.

## Limites

- Fixe modelo, revisão, dataset, seleção, seed, tokenizer e ferramenta; `main`
  não é identidade experimental.
- `train.csv` pode alimentar calibração/iMatrix; `calibration.csv` escolhe
  técnica e parâmetros; `test.csv` fica reservado à avaliação final.
- Dados privados, corpus renderizado, pesos, intermediários, manifests de
  execução e outputs grandes permanecem locais ou no storage autorizado.
- Download, build, benchmark e upload são fases separadas. Download e setup não
  pertencem à janela medida.
- Publicação no Hugging Face é escrita externa: confirme namespace, revisão,
  visibilidade e autorização. Público exige decisão explícita.
- Não reutilize artefato quantizado como fonte de nova quantização sem contrato
  próprio; verifique formato, hash e proveniência em vez de confiar no nome.

## Antes de alterar

Confirme técnica, bits, fonte imutável, hardware/runtime suportado, dataset,
risco de leakage, storage, custo e destino. Use smoke menor antes da execução
completa e mantenha publicação como passo separado.

## Evidência

O gate local dos scripts é:

```bash
make check-quantization
```

Para execução remota, registre commit, revisões, hashes, seed, ambiente, exit
code e manifests sanitizados. O handoff declara claramente o que foi apenas
validado localmente, construído, verificado ou publicado.

