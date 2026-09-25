# Acessar seu RunPod por SSH no Ubuntu e usar vários terminais

Este guia é específico para a sua máquina Linux e para o Pod atualmente ativo. Ele prepara o acesso necessário para executar o benchmark dividido entre [Terminal 1 - servidor](benchmark-ollama-terminal-1-servidor.md) e [Terminal 2 - execução](benchmark-ollama-terminal-2-execucao.md).

## 1. Estado já verificado na sua máquina

Não é necessário criar outra chave SSH nem instalar o cliente SSH.

| Item | Estado verificado |
|---|---|
| Sistema local | Ubuntu 24.04.3 LTS, GNOME/Wayland |
| Cliente SSH | OpenSSH 9.6p1 |
| Chave privada | `/home/THIAGO.OUVERNEY/.ssh/id_ed25519` |
| Permissão da chave privada | `600`, correta |
| Chave pública | `/home/THIAGO.OUVERNEY/.ssh/id_ed25519.pub` |
| Fingerprint | `SHA256:EGfeuHaFFnEjuy3UOh4QxTnTiK/exUpwAzVYirD8/cM` |
| `ssh`, `scp`, `sftp`, `rsync` | já instalados |
| Terminal gráfico | GNOME Terminal instalado |
| `tmux` local | não instalado; não é necessário |

Nunca copie ou compartilhe o conteúdo de `id_ed25519`. O arquivo com final `.pub` é a chave que pode ser cadastrada no RunPod.

## 2. Dados do Pod atual

O painel forneceu estas duas conexões:

| Modo | Endereço atual | Transferência de arquivos |
|---|---|---|
| SSH pelo proxy do RunPod | `gucvzvtx1kf5ld-64410d0b@ssh.runpod.io` | **funcionando**; não oferece SCP/SFTP |
| SSH direto por TCP | `root@213.192.2.93`, porta `40154` | pede senha porque a chave ainda não foi aceita |

Para o seu caso atual, use **o proxy do RunPod como conexão principal**. Essa é a conexão que já foi testada com sucesso e é suficiente para abrir vários terminais, usar `tmux`, executar o Ollama e rodar o benchmark completo.

O acesso TCP direto fica como uma correção posterior. Enquanto ele pedir senha, não tente adivinhar uma senha: a chave pública não está sendo aceita por esse `sshd`.

Esses dados pertencem ao Pod atual. Se ele for removido e outro Pod for criado, confira novamente IP, porta e identificador no botão **Connect** do RunPod.

O endereço completo do terminal web não é registrado neste arquivo porque contém um caminho de acesso temporário. Use **Open web terminal** diretamente no painel somente como recuperação.

## 3. Testar agora a conexão principal

Execute este comando no GNOME Terminal da sua máquina, não no terminal web do Pod:

```bash
ssh -o IdentitiesOnly=yes \
  gucvzvtx1kf5ld-64410d0b@ssh.runpod.io \
  -i ~/.ssh/id_ed25519
```

Na primeira conexão, o SSH mostrará a chave do servidor e perguntará se deseja continuar. Confirme que IP e porta ainda coincidem com o painel do RunPod antes de responder `yes`.

Depois de entrar, o prompt deverá ser semelhante a:

```text
root@5a397eec3b8b:/workspace#
```

Valide que entrou no Pod correto:

```bash
hostname
pwd
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -d /workspace/chatbot-runtime-bench
```

Saia apenas desta sessão com:

```bash
exit
```

## 4. Conexão TCP direta ainda pendente

O painel também informa este comando:

```bash
ssh root@213.192.2.93 -p 40154 -i ~/.ssh/id_ed25519
```

No estado atual ele pede a senha remota de `root`, o que confirma que a autenticação pela chave falhou. O RunPod não fornece uma senha padrão para esse fluxo. Continue pelo proxy e só use esse endereço depois que a chave correta estiver em `/root/.ssh/authorized_keys` e o comando entrar sem senha.

## 5. Criar atalhos específicos no SSH

Você já possui `/home/THIAGO.OUVERNEY/.ssh/config`, e o nome `runpod-qwen` ainda não está configurado. Abra esse arquivo no seu editor e acrescente o bloco abaixo, de preferência antes de um eventual bloco `Host *`:

```sshconfig
Host runpod-qwen
    HostName ssh.runpod.io
    User gucvzvtx1kf5ld-64410d0b
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 6

Host runpod-qwen-tcp
    HostName 213.192.2.93
    User root
    Port 40154
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

Mantenha a permissão do arquivo:

```bash
chmod 600 ~/.ssh/config
```

Confira a resolução sem se conectar:

```bash
ssh -G runpod-qwen | grep -E '^(hostname|user|port|identityfile) '
```

O resultado deve apontar para `ssh.runpod.io`, usuário `gucvzvtx1kf5ld-64410d0b`, porta `22` e sua chave Ed25519.

Depois disso, os comandos ficam curtos:

```bash
ssh runpod-qwen
ssh runpod-qwen-tcp  # somente depois de corrigir o acesso TCP direto
```

## 6. Abrir três terminais no seu Ubuntu

No GNOME Terminal, abra uma janela e pressione `Ctrl+Shift+T` duas vezes. Isso cria três abas locais.

Em **cada aba**, conecte-se separadamente:

```bash
ssh runpod-qwen
```

Use esta organização:

| Aba | Título sugerido | Função |
|---|---|---|
| 1 | `RunPod - Ollama` | servidor `ollama serve`, quando iniciado manualmente |
| 2 | `RunPod - Benchmark` | ambiente Python e `bench.py` |
| 3 | `RunPod - GPU` | `nvidia-smi`, logs e espaço em disco |

As três conexões acessam o mesmo Pod, os mesmos arquivos e os mesmos processos. Entretanto, cada aba possui seu próprio shell e suas próprias variáveis de ambiente.

### Aba 1: servidor Ollama

Use esta aba somente em execuções que **não** passam `--launch`:

```bash
export OLLAMA_HOST=127.0.0.1:11434

# Deixe comentado, exceto se o alias foi criado explicitamente nesse diretório.
# export OLLAMA_MODELS=/workspace/ollama-models

ollama serve
```

O comando fica em primeiro plano. Não feche a aba. Antes de executar um benchmark com `--launch`, volte a esta aba e encerre o servidor com `Ctrl+C`.

### Aba 2: benchmark

```bash
cd /workspace/chatbot-runtime-bench
source .venv/bin/activate

export GGUF_PATH=/workspace/models/qwen14b/Qwen2.5-14B-Instruct-Q8_0.gguf
export OLLAMA_HOST=127.0.0.1:11434

test -s "$GGUF_PATH"
python --version
```

Execute os testes e a bateria formal seguindo [benchmark-ollama-terminal-2-execucao.md](benchmark-ollama-terminal-2-execucao.md).

### Aba 3: monitoramento

Para observar GPU e VRAM:

```bash
watch -n 1 nvidia-smi
```

Saia do `watch` com `Ctrl+C`. Para acompanhar o log do servidor iniciado por `--launch`:

```bash
cd /workspace/chatbot-runtime-bench
LATEST="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tail -f "$LATEST/server.log"
```

Saia do `tail -f` com `Ctrl+C`.

## 7. Evitar perder o benchmark se o SSH cair

Abrir várias abas resolve a organização, mas não protege o processo contra uma desconexão. Para a bateria longa, use `tmux` **dentro do Pod**.

Na Aba 2, depois de conectar ao Pod:

```bash
if ! command -v tmux >/dev/null; then
  apt-get update
  apt-get install -y tmux
fi

tmux new -s qwen-bench
```

Dentro do `tmux`, prepare o ambiente e execute o benchmark. Para sair sem encerrar o processo, pressione `Ctrl+B` e depois `D`.

Após uma queda do SSH, reconecte e retome:

```bash
ssh runpod-qwen
tmux attach -t qwen-bench
```

Atalhos úteis dentro do `tmux`:

| Atalho | Ação |
|---|---|
| `Ctrl+B`, depois `C` | criar uma janela |
| `Ctrl+B`, depois `N` | próxima janela |
| `Ctrl+B`, depois `P` | janela anterior |
| `Ctrl+B`, depois `D` | desanexar preservando os processos |

O `tmux` protege contra queda da conexão SSH, mas não contra encerramento, interrupção ou falta de crédito do Pod.

## 8. Baixar os resultados para sua máquina

No estado atual, deixe o arquivo empacotado em `/workspace`: o proxy funcional não oferece SCP/SFTP. `rsync` também não resolve essa limitação porque depende do transporte SSH completo.

Depois de corrigir o acesso TCP direto, execute `scp` na sua máquina Ubuntu, fora do Pod. Por exemplo, após gerar `/workspace/RESULTADO-ollama-benchmark.tar.gz`:

```bash
scp -P 40154 -i ~/.ssh/id_ed25519 \
  root@213.192.2.93:/workspace/RESULTADO-ollama-benchmark.tar.gz \
  ~/Downloads/
```

Com o atalho TCP configurado, o equivalente é:

```bash
scp runpod-qwen-tcp:/workspace/RESULTADO-ollama-benchmark.tar.gz ~/Downloads/
```

Confirme localmente:

```bash
ls -lh ~/Downloads/RESULTADO-ollama-benchmark.tar.gz
sha256sum ~/Downloads/RESULTADO-ollama-benchmark.tar.gz
```

O `scp` deve usar o IP direto/alias `runpod-qwen-tcp`. O alias principal `runpod-qwen` usa o proxy e não oferece SCP/SFTP.

## 9. Acessar a API do Ollama sem expô-la publicamente

Ollama continua escutando em `127.0.0.1:11434` no Pod. Depois de corrigir o SSH TCP direto, você poderá consultar essa API a partir da sua máquina abrindo uma quarta aba local e criando um túnel para a porta local `11435`:

```bash
ssh -N -L 11435:127.0.0.1:11434 runpod-qwen-tcp
```

Enquanto essa aba permanecer aberta, em outra aba **local** execute:

```bash
curl http://127.0.0.1:11435/api/version
curl http://127.0.0.1:11435/v1/models
```

Isso não é necessário para o benchmark. É apenas uma forma segura de inspecionar a API sem publicar a porta 11434.

## 10. Diagnóstico específico

### `Permission denied (publickey)`

Sua chave local existe e já está com permissão correta. Confira no painel se a chave pública cadastrada possui este fingerprint:

```text
SHA256:EGfeuHaFFnEjuy3UOh4QxTnTiK/exUpwAzVYirD8/cM
```

Confira o fingerprint local sem mostrar a chave:

```bash
ssh-keygen -lf ~/.ssh/id_ed25519.pub
```

### Aviso de chave do host alterada

IP e porta do RunPod podem ser reutilizados. Primeiro confira no painel se `213.192.2.93:40154` ainda pertence ao seu Pod. Somente depois remova a entrada antiga:

```bash
ssh-keygen -R '[213.192.2.93]:40154'
```

Reconecte e aceite a nova chave apenas depois de validar o endereço no painel.

### Timeout ou senha no IP direto

Continue pela conexão funcional:

```bash
ssh runpod-qwen
```

Se o proxy funcionar, confira no painel se a porta TCP externa `40154` ainda está associada à porta interna `22`.

### A porta 11434 já está ocupada

No Pod:

```bash
pgrep -af 'ollama serve'
curl -sS http://127.0.0.1:11434/api/version
```

Servidor ativo é esperado para os comandos sem `--launch`. Para usar `--launch`, encerre o servidor manual com `Ctrl+C` e confirme que o `curl` passa a falhar.

## 11. Sequência recomendada

1. Na sua máquina, conecte pelo comando funcional do proxy com `IdentitiesOnly=yes`.
2. Acrescente os aliases ao `~/.ssh/config`; `runpod-qwen` será o proxy funcional.
3. Abra três abas com `Ctrl+Shift+T` e conecte todas com `ssh runpod-qwen`.
4. Use a Aba 1 para Ollama, a Aba 2 para o benchmark e a Aba 3 para GPU/logs.
5. Execute a bateria longa dentro de `tmux` no Pod.
6. Empacote os resultados conforme o guia do benchmark.
7. Preserve o arquivo em `/workspace`; depois de corrigir o TCP direto, baixe com `scp runpod-qwen-tcp:/workspace/ARQUIVO.tar.gz ~/Downloads/`.

Referências oficiais:

- [SSH no RunPod](https://docs.runpod.io/pods/configuration/use-ssh)
- [Opções de conexão de Pods](https://docs.runpod.io/pods/connect-to-a-pod)
