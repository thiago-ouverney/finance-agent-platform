# Operação do WhatsApp com Ollama na UFF

Este runbook descreve como iniciar e testar o fluxo operacional do chat:

```text
WhatsApp
  -> adapter na máquina pessoal
  -> http://127.0.0.1:8000/v1
  -> túnel SSH uff-delta
  -> Ollama na UFF em 127.0.0.1:11434
  -> Qwen na GPU
```

Este é o fluxo de atendimento do WhatsApp. Ele não é o fluxo de benchmark de
`services/inference-runtime` e não usa `llama-server` nem um gateway HTTP
intermediário. O adapter chama diretamente a API OpenAI-compatible do Ollama
através do túnel SSH.

## Responsabilidade de cada máquina

| Máquina | Responsabilidade |
| --- | --- |
| Pessoal | sessão do WhatsApp, adapter Node.js, histórico por contato e túnel SSH |
| UFF | Ollama, modelo Qwen e GPU |

Um prompt com o usuário remoto e o hostname da UFF indica que o comando está
sendo executado na VM. Um prompt com o usuário local e um caminho em `Projects`
indica a máquina pessoal. Para sair da UFF e voltar ao terminal local, use
`exit`.

## 1. Configurar o alias SSH `uff-delta`

A configuração fica em `~/.ssh/config` na máquina pessoal e não deve ser
versionada no repositório. Crie ou edite o arquivo com os dados fornecidos pela
UFF:

```sshconfig
Host uff-delta
    HostName <host-ou-ip-da-uff>
    User <usuario-da-uff>
    Port <porta-ssh>
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Proteja o arquivo e teste o alias:

```bash
chmod 600 ~/.ssh/config
ssh uff-delta
```

Se a conexão pedir senha, o túnel pode ser usado manualmente mesmo assim. Para
evitar a senha em cada conexão, cadastre uma chave pública. Se já existir uma
chave adequada, não gere nem sobrescreva outra:

```bash
ls -l ~/.ssh/*.pub
ssh-copy-id -i ~/.ssh/id_ed25519.pub uff-delta
ssh uff-delta
```

Se não houver uma chave, crie uma dedicada:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_uff -C "uff-delta"
ssh-copy-id -i ~/.ssh/id_ed25519_uff.pub uff-delta
```

Nesse caso, acrescente ao bloco `Host uff-delta`:

```sshconfig
    IdentityFile ~/.ssh/id_ed25519_uff
    IdentitiesOnly yes
```

Nunca coloque a senha SSH, uma chave privada ou os dados reais do host em um
arquivo versionado.

## 2. Verificar Ollama, modelo e GPU na UFF

Na máquina pessoal, conecte-se:

```bash
ssh uff-delta
```

Na UFF, confira o ambiente:

```bash
hostname
nvidia-smi
command -v ollama
ollama list
```

O nome do modelo precisa ser idêntico ao usado pelo adapter. O modelo validado
neste fluxo é:

```text
qwen2.5:7b-instruct-q4_K_M
```

Confirme que ele existe:

```bash
ollama show qwen2.5:7b-instruct-q4_K_M
```

Se `ollama list` mostrar outro nome, use esse nome exato tanto nos testes quanto
em `WHATSAPP_INFERENCE_MODEL`.

## 3. Iniciar o servidor Ollama na UFF

Primeiro verifique se o servidor já está ativo:

```bash
curl http://127.0.0.1:11434/api/tags
```

Se responder, não inicie outro `ollama serve`. Se não responder e não houver um
serviço do sistema administrando o Ollama, abra uma sessão persistente:

```bash
tmux new -s ollama-whatsapp
```

Dentro do `tmux`:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

O endereço `127.0.0.1` é intencional: o Ollama permanece privado e só pode ser
alcançado localmente na UFF ou pelo túnel SSH.

Para desacoplar sem encerrar o Ollama, pressione `Ctrl+B` e depois `D`. Para
voltar à sessão:

```bash
tmux attach -t ollama-whatsapp
```

## 4. Testar a inferência dentro da UFF

Ainda na UFF, teste primeiro a API nativa:

```bash
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "messages": [
      {"role": "user", "content": "Responda apenas: Ollama funcionando"}
    ],
    "stream": false
  }'
```

Depois confirme a API OpenAI-compatible usada pelo adapter:

```bash
curl http://127.0.0.1:11434/v1/models
```

Só prossiga para o túnel quando os dois testes responderem. Isso separa um
problema do runtime de um problema de rede ou do WhatsApp.

## 5. Abrir o túnel SSH na máquina pessoal

Abra um terminal local dedicado e execute:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8000:127.0.0.1:11434 \
  uff-delta
```

O comando ficará sem prompt enquanto o túnel estiver ativo. Isso é esperado:

```text
127.0.0.1:8000 na máquina pessoal
  -> SSH uff-delta
  -> 127.0.0.1:11434 na UFF
  -> Ollama
```

Mantenha o terminal aberto. `Ctrl+C` encerra o túnel.

Se outra porta local for escolhida, por exemplo `11435`, todos os testes e
`WHATSAPP_INFERENCE_BASE_URL` também precisarão usar `11435`. Abrir `11435` e
testar `8000` sempre resultará em conexão recusada.

## 6. Testar a ponte na máquina pessoal

Em outro terminal local:

```bash
curl http://127.0.0.1:8000/api/tags
curl http://127.0.0.1:8000/v1/models
```

Teste a geração pela mesma interface usada pelo adapter:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "messages": [
      {"role": "user", "content": "Responda apenas: ponte funcionando"}
    ],
    "max_tokens": 30
  }'
```

## 7. Configurar o adapter WhatsApp

Na máquina pessoal:

```bash
cd apps/whatsapp-adapter
test -f .env || cp .env.example .env
npm install
```

Confira estas variáveis em `.env`:

```dotenv
WHATSAPP_INFERENCE_BASE_URL=http://127.0.0.1:8000/v1
WHATSAPP_INFERENCE_MODEL=qwen2.5:7b-instruct-q4_K_M
WHATSAPP_INFERENCE_TIMEOUT_MS=120000
INFERENCE_API_TOKEN=ollama
```

`ollama` funciona como uma chave local fictícia para o cliente compatível com
OpenAI; o servidor Ollama nesse fluxo não valida o valor.

Para autorizar um telefone, execute na raiz do monorepo:

```bash
cd /caminho/para/finance-agent-platform
make access-add PHONE=5521999999999
```

O formato correto do target exige `PHONE=`. O comando abaixo está errado porque
o número vira um argumento solto e a variável permanece vazia:

```bash
make access-add 5521999999999
```

Use país, DDD e número, somente com dígitos. O comando atualiza o `.env` local;
esse arquivo não deve ser versionado. Reinicie o adapter depois de alterar a
lista.

## 8. Abrir o WhatsApp

Com Ollama e túnel ativos:

```bash
cd apps/whatsapp-adapter
npm run doctor
npm run dev
```

Se a sessão existente do Baileys ainda for válida, a conta reconectará sem QR
Code. Caso contrário, no celular abra:

```text
WhatsApp -> Aparelhos conectados -> Conectar aparelho
```

Escaneie o QR Code do terminal e envie uma mensagem usando um telefone presente
em `ALLOWED_CONTACTS`.

O adapter mantém o histórico lógico por contato e reenvia as mensagens recentes
ao Ollama. A sessão do Baileys autentica o WhatsApp; ela não é uma sessão de
conversa do modelo.

## 9. Ordem operacional diária

1. Na UFF, confirme `curl http://127.0.0.1:11434/api/tags` e `ollama list`.
2. Na máquina pessoal, abra o túnel `8000 -> uff-delta -> 11434`.
3. Localmente, confirme `curl http://127.0.0.1:8000/v1/models`.
4. Execute `npm run doctor` no adapter.
5. Execute `npm run dev` e teste pelo WhatsApp.

Para encerrar:

1. interrompa o adapter com `Ctrl+C`;
2. interrompa o túnel com `Ctrl+C`;
3. mantenha o Ollama no `tmux` ou volte à sessão e encerre-o conscientemente.

## 10. Diagnóstico rápido

### `curl: (7) Failed to connect` na porta 8000

O túnel não está ativo, foi encerrado com `Ctrl+C` ou foi criado em outra porta.
Confira o lado esquerdo de `-L` e mantenha o terminal do SSH aberto.

### `curl: (56) Recv failure`

Primeiro confirme a API diretamente na UFF. Se ela responder, reabra o túnel com
`ExitOnForwardFailure=yes` e teste `/api/tags` antes de testar a geração.

### `Could not resolve hostname uff-delta`

O alias não existe em `~/.ssh/config` da máquina pessoal ou o comando foi
executado dentro da própria UFF. O alias é local; use `exit` e teste novamente na
máquina pessoal.

### Modelo não encontrado

Compare literalmente `ollama list` com `WHATSAPP_INFERENCE_MODEL`. Tags como
`3b`, `7b`, `q4_K_M` e `latest` fazem parte do identificador.

### `make access-add` pede `PHONE`

Use a atribuição nomeada:

```bash
make access-add PHONE=5521999999999
```

### `tsx watch` não encerra

O processo pode manter o socket do Baileys ou um timer de reconexão ativo. Não
pressione `Ctrl+C` repetidamente. Em outro terminal, identifique o PID exato:

```bash
pgrep -af 'tsx watch src/index.ts'
```

Tente encerrá-lo normalmente com `kill PID`. Use `kill -KILL PID` apenas se esse
PID específico continuar ativo após alguns segundos.

### Monitorar a GPU

Na UFF:

```bash
watch -n 1 nvidia-smi
```

Para conferir se o modelo continua carregado:

```bash
ollama ps
```

## Segurança

- Não configure `OLLAMA_HOST=0.0.0.0:11434` para este fluxo.
- Não abra a porta `11434` no firewall ou roteador.
- Não versione `.env`, `~/.ssh/config`, chaves, `.baileys_auth`, pesos ou
  históricos dos contatos.
- Não copie uma chave privada para a UFF; instale somente a chave pública em
  `authorized_keys`.
- O túnel SSH protege o transporte, mas não substitui a allowlist do adapter.
