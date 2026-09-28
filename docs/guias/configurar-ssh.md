# Configurar SSH e abrir pontes

> **Objetivo:** configurar os aliases `uff-delta` e `runpod-qwen` e abrir uma
> ponte local para um serviço remoto sem expor sua porta na internet.
>
> **Resultado esperado:** `ssh <alias>` conecta sem informar host e usuário, e
> a API remota responde por uma porta em `127.0.0.1`.

## Pré-requisitos

- host, usuário e porta fornecidos pelo responsável do ambiente;
- uma chave SSH existente ou permissão para cadastrar uma nova chave pública;
- OpenSSH instalado na máquina local.

Não coloque host, usuário, porta, senha ou chave privada no repositório.

## 1. Criar os aliases

Edite `~/.ssh/config` na máquina local:

```sshconfig
Host uff-delta
    HostName <host-da-uff>
    User <usuario-da-uff>
    Port <porta-ssh>
    IdentityFile <caminho-da-chave-privada>
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3

Host runpod-qwen
    HostName <host-do-pod>
    User <usuario-do-pod>
    Port <porta-ssh>
    IdentityFile <caminho-da-chave-privada>
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Proteja o arquivo:

```bash
chmod 600 ~/.ssh/config
```

Se a UFF ainda não tiver uma chave pública sua, gere uma chave dedicada sem
sobrescrever as existentes e cadastre apenas o arquivo `.pub`:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_uff -C "uff-delta"
ssh-copy-id -i ~/.ssh/id_ed25519_uff.pub uff-delta
```

Nesse caso, use `~/.ssh/id_ed25519_uff` como `IdentityFile`. Para RunPod, use o
método de chave indicado pelo provedor.

## 2. Validar a conexão

```bash
ssh uff-delta
ssh runpod-qwen
```

Se cada comando abrir o host correto, encerre a sessão com `exit`.

## 3. Abrir a ponte necessária

Execute apenas a ponte que será usada e mantenha seu terminal aberto.

| Uso | Comando local | Verificação em outro terminal |
|---|---|---|
| Avaliação no Ollama/UFF | `make eval-tunnel EVAL_SSH_HOST=uff-delta` | `make eval-check` |
| WhatsApp no Ollama/UFF | `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:11434 uff-delta` | `curl http://127.0.0.1:8000/v1/models` |
| Silver no vLLM/RunPod | `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18000:127.0.0.1:8000 runpod-qwen` | `curl http://127.0.0.1:18000/v1/models` |

Uma porta local só pode atender uma ponte por vez. `Ctrl+C` encerra a ponte.

Para detalhes de cada ambiente, consulte [execução local, UFF e
RunPod](../remote-execution.md) e [operação do WhatsApp na
UFF](../uff-ollama-whatsapp.md).
