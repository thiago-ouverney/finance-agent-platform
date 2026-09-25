# Chat simples para o modelo do Arthur

Esta interface Python conversa com o vLLM existente. Não carrega outra cópia do modelo e não requer outra GPU. Tem histórico por sessão, streaming, Markdown para listas/código, botão de limpar e controles opcionais de temperatura e limite de saída. Não força respostas JSON: a apresentação é de um chatbot comum.

**Você fará três coisas:** copiar dois arquivos do Mac para o Pod, iniciar o programa Python no Pod e abrir a página do chat no navegador. Abrir este HTML no Mac mostra apenas as instruções; não inicia o aplicativo.

| Onde | O que você faz |
| --- | --- |
| Finder no Mac | Localiza os dois arquivos que serão enviados. |
| Painel RunPod no navegador | Abre o JupyterLab e configura o acesso à porta 7860. |
| JupyterLab do Pod | Recebe os arquivos e oferece os terminais Linux para executar os programas. |
| Terminal A do Pod | Mantém o vLLM rodando na porta 8000. |
| Terminal B do Pod | Mantém o chat Python rodando na porta 7860. |

Você pode continuar usando a mesma RTX 3090 e o mesmo Pod. Se o erro `libcudart.so.13` ainda não foi resolvido, volte ao [guia principal, seção 3.1](index.html): a interface não corrige a instalação do vLLM.

## 1. Colocar os arquivos no Pod

### No Mac: localizar os arquivos

1. Abra o **Finder**.
2. Pressione **Command + Shift + G** para abrir “Ir para a pasta”.
3. Cole o caminho abaixo e pressione Enter:

```text
/CAMINHO/LOCAL/runpod-vllm-lab
```

4. Localize **chat.py** e **requirements-chat.txt**. São esses dois arquivos que você vai enviar; deixe a janela do Finder aberta.

Eles também estão ao lado deste HTML: [abrir chat.py](chat.py) e [abrir requirements-chat.txt](requirements-chat.txt). Se o navegador mostrar o código ao clicar, isso é normal; o upload será feito pelo Finder.

### No navegador: abrir o JupyterLab do Pod

1. Entre no painel RunPod e abra **Pods**.
2. Localize seu Pod ligado e clique em **Connect**.
3. Em **HTTP Services**, abra **JupyterLab**, geralmente na porta **8888**. Se pedir autenticação, use as informações disponibilizadas pelo template/RunPod; não é a senha do chat que criaremos depois.
4. No painel de arquivos à esquerda do JupyterLab, localize a pasta `yan-vllm`. Muitos templates já abrem na raiz `/workspace`, então ela pode estar visível diretamente.
5. Abra `yan-vllm`. Arraste **chat.py** e **requirements-chat.txt** do Finder para o painel de arquivos do JupyterLab. Alternativamente, use o botão de **Upload Files**, com ícone de seta para cima, e selecione os dois arquivos.
6. Espere o upload concluir e confira que os dois nomes aparecem na pasta.

**Destino esperado:** `/workspace/yan-vllm/chat.py` e `/workspace/yan-vllm/requirements-chat.txt`. Não é preciso enviar o HTML ou os pesos do modelo. Copiar o caminho do seu Mac para um terminal remoto não transfere o arquivo: o upload é a etapa que faz essa transferência.

Se `yan-vllm` não aparecer, abra um terminal no JupyterLab por **File → New → Terminal** (ou pelo botão **+ → Terminal**) e execute:

```bash
ls -ld /workspace/yan-vllm
```

`ls -ld` verifica se a pasta existe e mostra seus dados, sem listar todos os arquivos dentro dela. Se existir, procure-a a partir da raiz de arquivos exposta pelo JupyterLab. Se o JupyterLab não estiver disponível no seu template, me diga quais opções aparecem em Connect para adaptarmos a transferência.

Mantenha o vLLM do guia rodando no Terminal A, com o alias `qwen14b-int8`. O erro de CUDA precisa estar resolvido antes de o chat conseguir gerar respostas.

## 2. Instalar a interface em outro ambiente

Abra **outro terminal**, sem interromper aquele em que o vLLM está rodando: no JupyterLab, use **File → New → Terminal** (ou **+ → Terminal**). Vamos chamá-lo de **Terminal B**. Todos os comandos abaixo são executados nele, não no terminal do Mac.

Primeiro confirme o upload:

```bash
cd /workspace/yan-vllm
ls -l chat.py requirements-chat.txt
```

`cd` entra na pasta remota. `ls -l` deve mostrar os dois arquivos com tamanho maior que zero. Se aparecer `No such file or directory`, confira a pasta em que fez o upload antes de continuar.

Depois instale:

```bash
cd /workspace/yan-vllm
python3 -m venv .venv-chat
source .venv-chat/bin/activate
python -m pip install -r requirements-chat.txt
```

`cd` entra na pasta onde você enviou os arquivos. `venv` cria um ambiente só para a interface, preservando as dependências do vLLM na outra `.venv`. `source` ativa esse ambiente no terminal atual. `pip install -r` instala as versões declaradas no arquivo. Faça a criação/instalação uma vez; nas próximas sessões, basta entrar na pasta e ativar `.venv-chat`.

## 3. Iniciar

No mesmo Terminal B, primeiro verifique o servidor:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

O comando deve mostrar um JSON contendo `qwen14b-int8`. Se der `Connection refused`, volte ao Terminal A e inicie o vLLM conforme a seção 4 do [guia principal](index.html). Aguarde a inicialização e repita esta consulta. A interface só consegue responder quando esse servidor está disponível.

Com o servidor respondendo, inicie o chat:

```bash
python chat.py
```

O curl confirma que o vLLM responde e lista `qwen14b-int8`. O Python inicia a interface e pede que você **crie uma senha** no terminal; nada aparece enquanto você digita. O usuário de login é **yan**. Essa senha é do seu chat, não da conta RunPod.

Digite uma senha e pressione Enter. Aguarde uma mensagem de que o aplicativo está disponível na porta **7860**. O terminal fica ocupado enquanto o programa roda; isso é esperado. O endereço `0.0.0.0:7860` exibido ali descreve onde o programa escuta no Pod — não é o link para abrir no Mac.

Deixe os dois terminais abertos: A serve o modelo, B serve a interface. O chat se conecta a `127.0.0.1:8000/v1` dentro do Pod; seu navegador acessa a interface na porta 7860.

## 4. Abrir no navegador

1. Volte à aba do painel **RunPod → Pods** e expanda seu Pod.
2. Abra o menu de opções do Pod e escolha **Edit Pod**. A posição/nome do menu pode variar na interface.
3. Encontre **Expose HTTP Ports** e adicione **7860**, preservando as portas existentes. Por exemplo, se já houver `8888`, mantenha-a e acrescente `7860` no formato aceito pelo campo. Use a seção **HTTP**, não a de portas TCP.
4. Salve a alteração. Se o painel avisar que o Pod será reiniciado, os programas precisam ser iniciados de novo depois; os comandos para retomar o chat estão na seção 6 abaixo.
5. Aguarde o Pod estar pronto e confirme que o vLLM e o chat voltaram a rodar, caso tenha ocorrido reinício.
6. Abra **Connect → HTTP Services** e clique no serviço da porta **7860**. Pode aparecer como um serviço HTTP genérico, sem o nome “chat”.
7. Na tela de login, entre com usuário **yan** e a senha que você acabou de criar no Terminal B.
8. Você deve ver **Chat com Qwen · vLLM**, uma área de conversa e um campo para digitar.

O endereço tem o formato `https://SEU_POD_ID-7860.proxy.runpod.net`. A interface escuta em `0.0.0.0` para o proxy alcançá-la; o vLLM pode continuar restrito a `127.0.0.1`. Não precisa expor a porta 8000. Referência: [portas HTTP no RunPod](https://docs.runpod.io/pods/configuration/expose-ports).

## 5. Usar e encerrar

Para conferir o primeiro funcionamento, envie: **“Explique KV cache em três tópicos e dê um exemplo curto.”** Depois pergunte **“Resuma sua resposta em uma frase.”** A segunda pergunta usa o histórico da primeira.

Digite uma pergunta e clique em **Enviar**. A resposta aparece aos poucos e preserva a formatação Markdown. Use **Parar** durante a geração e o botão de limpar do chat para iniciar outra conversa. O histórico é reenviado ao modelo em cada pergunta e não é salvo em disco pelo aplicativo; não conte com persistência após recarregar a página.

O limite inicial do servidor é **2048 tokens, somando histórico, pergunta e resposta**. Uma conversa longa pode ultrapassá-lo. Nesse caso, limpe o histórico ou diminua o máximo de tokens. O aplicativo mostra o erro e não apaga mensagens silenciosamente. Temperatura começa em 0 e saída em 256; os ajustes ficam no painel adicional do Gradio.

Um erro de conexão significa que o chat não encontrou o vLLM; confira o Terminal A. Um erro do proxy/502 normalmente pede conferir se `python chat.py` está ativo e se a porta 7860 foi exposta.

Ctrl+C no Terminal B encerra a interface. Isso não encerra o servidor do Terminal A nem a cobrança do Pod. Para parar a cobrança de GPU, use o painel RunPod conforme o guia principal.

## 6. Retomar em outra sessão

Depois de reiniciar o Pod, confirme que o vLLM voltou a rodar no Terminal A. Para o chat, abra outro terminal e execute:

```bash
cd /workspace/yan-vllm
source .venv-chat/bin/activate
python chat.py
```

Não precisa reinstalar as dependências se o ambiente no volume foi preservado. O programa pede uma senha ao iniciar; você pode escolher a mesma de antes. Abra novamente o serviço HTTP 7860 no RunPod.

## 7. Se alguma etapa falhar

| O que você vê | O que conferir |
| --- | --- |
| `can't open file ... chat.py` | O arquivo não está na pasta atual; execute `cd /workspace/yan-vllm` e confira o upload. |
| `No module named gradio` | Ative `.venv-chat` e execute a instalação com `python -m pip install -r requirements-chat.txt`. |
| A senha não aparece enquanto digito | É o comportamento normal do prompt de senha; digite e pressione Enter. |
| `Address already in use` na porta 7860 | Pode haver outra cópia do chat rodando. Volte ao terminal dela; não é necessário iniciar duas interfaces. |
| Serviço 7860 não aparece em Connect | Confira se a porta foi adicionada em Expose HTTP Ports e se a alteração foi salva. |
| Página com 502 ou indisponível | Confira se o Pod está ligado e se `python chat.py` está ativo no Terminal B após uma eventual reinicialização. |
| Chat abre, mas falha ao enviar | Veja se o vLLM está ativo no Terminal A e se `/v1/models` lista `qwen14b-int8`. |
| Erro mencionando contexto/tokens | Limpe a conversa e envie uma pergunta menor; histórico e saída compartilham o limite de 2048 tokens. |

Se continuar falhando, envie a etapa, o erro da tela e a saída do terminal correspondente. Não envie sua senha.

## Configuração opcional

O arquivo lê `VLLM_BASE_URL` e `VLLM_MODEL` caso você mude endereço ou alias. `VLLM_API_KEY` só é necessária se você habilitar autenticação no próprio vLLM. `CHAT_USER` muda o usuário de login, e `CHAT_PASSWORD` permite fornecer a senha por variável de ambiente em vez do prompt. Para a primeira execução, nenhum desses ajustes é necessário.

Implementação com [Gradio ChatInterface](https://www.gradio.app/docs/gradio/chatinterface). O chat é uma interface de uso; seus tempos incluem navegador/proxy e não substituem o benchmark dentro do Pod.
