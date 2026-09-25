# Instruções para agentes neste projeto

## Avaliação obrigatória antes de executar

Antes de cada ação ou bloco de execução, avaliar se o trabalho deve ser feito pelo agente principal ou delegado. Considerar independência da tarefa, dificuldade, risco, custo de repassar contexto e possibilidade de progresso paralelo útil. Não delegar automaticamente só porque há uma ferramenta de agentes disponível.

O usuário solicita e autoriza subagentes com modelos menores para implementações delimitadas, testes, HTML, documentação, formatação e verificações mecânicas. Preferir o menor modelo disponível capaz de cumprir a tarefa com qualidade; não depender de um nome de modelo que pode não existir em outra sessão.

O agente principal concentra decisões arquiteturais e metodológicas, definição de contratos, diagnóstico complexo, revisão, integração e validação final. Pode executar diretamente ações curtas, sequenciais ou de alto acoplamento quando delegar acrescentaria custo sem benefício.

Comunicar a divisão de trabalho de forma breve quando relevante; não emitir um monólogo de raciocínio ou uma mensagem antes de cada comando trivial. Registrar decisões de trabalho e seus motivos em resumo, não raciocínio interno detalhado.

## Contrato de delegação

- Dar ao subagente uma tarefa concreta, contexto mínimo suficiente, arquivos sob responsabilidade, resultado esperado e critérios de aceite.
- Evitar dois agentes editando os mesmos arquivos. Reutilizar agentes quando adequado.
- Enquanto um subagente trabalha, avançar em trabalho independente útil.
- Revisar o resultado e executar validações pertinentes antes de afirmar conclusão; delegação não transfere responsabilidade pela correção.
- Se não houver modelos menores ou ferramenta de delegação, informar a limitação e continuar localmente, sem fingir uso de subagentes.

## Regras do benchmark

- Uma requisição de inferência por vez; concorrência é outro trabalho.
- Pesos já disponíveis no SSD. Downloads são preparação e invalidam a execução medida se ocorrerem durante ela.
- Medir inicialização, primeira resposta, aquecimento e operação posterior separadamente.
- Distinguir TTFT, tokens/s de geração e tokens/s efetivos; documentar as fórmulas.
- Distinguir contexto em tokens, KV lógico estimado, ocupação real do pool KV e memória total da GPU. Não apresentar proxies como medição física.
- Nunca apresentar dados ausentes como zero nem uma execução parcial como sucesso.
- Não publicar pesos, ambientes, resultados, logs ou credenciais. Revisar os arquivos antes de qualquer push autorizado; usar autoria noreply e preservar a branch de histórico privado sem publicá-la.
- Mudanças no código devem atualizar documentação e testes proporcionais ao risco. Regenerar fontes numerados e HTML quando necessário.

## Documentação gerada

`python scripts/build_code_reference.py` gera `docs/codigo-fontes.md`.
`python scripts/build_docs.py` gera os HTMLs a partir dos Markdown.
Não editar arquivos gerados como fonte primária.

Estas instruções têm escopo do projeto. Não alteram configurações globais, permissões de ferramentas ou instruções de maior prioridade.
