# Fontes do caderno interativo

Abra **../estudo.html** no navegador. Esse arquivo é autônomo e pode ser compartilhado sozinho: CSS, JavaScript e gráficos SVG estão incorporados, sem bibliotecas externas.

Os arquivos nesta pasta são a versão editável:

- `conteudo.html`: capítulos, referências e estrutura das atividades.
- `style.css`: apresentação, layout adaptável e impressão.
- `interacoes.js`: simuladores, guia de parâmetros, diagnóstico, quiz e notas.
- `build.py`: monta o HTML final com Python padrão, sem instalar dependências.

Para reconstruir, a partir da raiz do projeto:

```bash
python3 runpod-vllm-lab/estudo/build.py
```

Os laboratórios não consultam o Pod. Os parâmetros numéricos são didáticos, salvo as dimensões declaradas da arquitetura do Qwen usadas na fórmula do KV cache. Não use saídas dos simuladores como resultados experimentais.

As notas e marcações usam localStorage quando permitido. Podem ser bloqueadas em arquivos locais ou apagadas pelo navegador; exporte as notas em Markdown para preservá-las. A página funciona sem salvar progresso.
