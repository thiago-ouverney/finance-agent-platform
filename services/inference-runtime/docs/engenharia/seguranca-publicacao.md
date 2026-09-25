# Revisão de privacidade para publicação

Escopo: código do benchmark, testes, configurações de exemplo e materiais autorais de RunPod/vLLM, arquitetura, memórias e chat. Não foram incluídos pesos, ambientes virtuais, resultados experimentais, caches, dados privados, o PDF da disciplina ou repositórios de terceiros.

## Verificações

- Varredura automatizada com `detect-secrets 1.5.0`, sem verificação de credenciais em serviços externos.
- Inspeção de nomes de arquivos, padrões de chaves/tokens, caminhos pessoais e autoria Git.
- Inspeção do histórico local anterior e, separadamente, do snapshot destinado à publicação.
- Revisão dos alertas: os valores fictícios usados nos testes de remoção de chaves e rejeição de URLs com credenciais não são credenciais reais.

Não foram identificadas credenciais reais no conjunto revisado. Isso **não é garantia de ausência absoluta de dados sensíveis**: detectores têm falsos negativos, e novos arquivos exigem outra revisão.

## Ajustes de privacidade

- Caminhos absolutos do computador pessoal foram substituídos por caminhos genéricos na documentação publicada.
- A autoria da branch pública usa o endereço noreply do GitHub. O histórico local anterior permanece reservado em uma branch local que não deve ser enviada.
- Resultados, logs, tokenizers, ambientes virtuais, chaves e pesos ficam fora do controle de versão pelas regras de `.gitignore`.
- O diretório `/workspace/yan-vllm` nos exemplos identifica um caminho operacional do pod, não um segredo. Adapte-o à sua instalação.

## O que ainda exige cuidado

Os relatórios do benchmark incluem prompts, respostas, argumentos e detalhes de ambiente. O valor de `BENCH_API_KEY` é removido dos relatórios gravados pelo cliente, mas isso não é um anonimizador geral. Não coloque outros segredos em campos livres. `server.log` contém a saída do runtime e pode expor dados: revise antes de compartilhar.

Nunca faça `git add -f results/`, publique `.env` ou use `git push --all` sem conferir o conteúdo e os históricos. A branch local reservada existe justamente para não publicar a autoria anterior. `.gitignore` não protege um arquivo já rastreado nem apaga dados de commits antigos.

O nome público da conta GitHub e os identificadores públicos de modelos utilizados nas instruções permanecem no material. Não foram publicados resultados reais do pod nem alegações de desempenho obtidas em GPU.
