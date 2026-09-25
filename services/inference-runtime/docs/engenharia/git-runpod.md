# Git no RunPod, sem mistério

**Git** é o programa que mantém o histórico do código. **GitHub** é um serviço que hospeda repositórios. Não existe “instalar minha conta Git” no pod: você instala Git e baixa seu repositório. Para clonar um repositório público, nem login é necessário.

## 1. Instalar Git na VM

No terminal de uma imagem Ubuntu/Debian, como root:

```bash
apt-get update
apt-get install -y git ca-certificates openssh-client
git --version
```

`update` atualiza a lista de pacotes; `install` instala Git, certificados HTTPS e cliente SSH; `--version` confirma a instalação. Se não estiver como root e sua imagem permitir, prefixe os comandos de instalação com `sudo`.

São comandos de instalação Linux. Não os execute no macOS. Outras distribuições podem usar outro gerenciador; veja a [documentação oficial de instalação do Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## 2. Clonar um repositório público

O repositório deste projeto é `yanwerneck/llm_local_inference_masters`. O último argumento de `clone` escolhe o nome da pasta local:

```bash
cd /workspace
git clone https://github.com/yanwerneck/llm_local_inference_masters.git chatbot-runtime-bench
cd chatbot-runtime-bench
git status
```

`clone` baixa código e histórico para uma pasta nova. `status` mostra a situação local; não envia nada à internet. Use uma pasta persistente do pod, mas confirme as regras de retenção e cobrança do volume: código e resultados podem desaparecer ao encerrar um pod sem armazenamento persistente.

Você **não precisa configurar nome/e-mail para apenas baixar e executar**. Agora siga a instalação do [README](../../README.md), criando um venv separado.

## 3. Atualizar depois

```bash
cd /workspace/llm_local_inference_masters
git status
git pull --ff-only
```

`pull --ff-only` traz atualizações sem criar um merge automático. Se houver alterações locais conflitantes, pare e preserve-as: não use `reset --hard` para “resolver rápido”. Copie suas configurações antes de reorganizar qualquer coisa.

Se as dependências mudaram, ative o venv e reinstale `requirements.txt`. Não atualize o código no meio da bateria comparativa. Registre o commit usado:

```bash
git rev-parse HEAD
```

Esse hash identifica uma versão do código, não dos pesos do modelo.

## 4. Se o repositório for privado

Para um pod descartável, prefira uma **deploy key somente leitura**, específica deste repositório. Não copie sua chave SSH pessoal do laptop para a VM. A [documentação de deploy keys do GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys) explica permissões e limitações.

```bash
mkdir -p /workspace/chatbench-ssh
chmod 700 /workspace/chatbench-ssh
ssh-keygen -t ed25519 -C 'runpod-chatbench-readonly' -f /workspace/chatbench-ssh/id_ed25519
cat /workspace/chatbench-ssh/id_ed25519.pub
```

O diretório guarda a chave; `chmod 700` limita seu acesso ao usuário proprietário. `ssh-keygen` cria o par de chaves e pergunta uma senha opcional. Se o arquivo já existir, **não sobrescreva**: escolha outro nome. O `cat` mostra somente a chave **pública**, terminada em `.pub`.

No GitHub, abra o repositório → Settings → Deploy keys → Add deploy key. Cole a chave pública e deixe **Allow write access desmarcado**. Nunca cole ou compartilhe `id_ed25519` sem `.pub`.

```bash
git -c core.sshCommand='ssh -i /workspace/chatbench-ssh/id_ed25519 -o IdentitiesOnly=yes' clone git@github.com:yanwerneck/llm_local_inference_masters.git /workspace/llm_local_inference_masters
```

Na primeira conexão, confira a impressão digital apresentada com a [lista oficial do GitHub](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) antes de aceitar. Não desative a verificação do host.

Para que os próximos pulls usem a mesma chave:

```bash
cd /workspace/llm_local_inference_masters
git config core.sshCommand 'ssh -i /workspace/chatbench-ssh/id_ed25519 -o IdentitiesOnly=yes'
```

Isso modifica apenas a configuração desse repositório. Se a chave tiver senha, o SSH poderá solicitá-la. Quando terminar de usar o pod, revogue a deploy key no GitHub. Um volume persistente também pode reter a chave privada: trate-o como sensível.

## 5. Publicar o projeto a partir do seu Mac

O repositório remoto deste projeto já foi criado e é **público**. Os comandos abaixo servem para conferir a publicação e enviar mudanças futuras. Não crie outro repositório com o mesmo propósito nem publique resultados/chaves inadvertidamente. Para projetos diferentes, considere começar com um repositório privado.

No terminal do Mac:

```bash
cd /CAMINHO/LOCAL/chatbot-runtime-bench
git status
git log -1 --oneline
git remote -v
```

Substitua `/CAMINHO/LOCAL` pela localização do projeto no seu computador. Confira quais arquivos serão publicados. `results/`, `tokenizer/`, `.venv/` e arquivos de credenciais são ignorados; `.gitignore` não substitui revisão manual. Se já existe um commit inicial, não é necessário criá-lo novamente.

Se ainda não houver commit, configure sua identidade **localmente** e comite somente os arquivos do projeto:

```bash
git config user.name 'Seu Nome'
git config user.email 'SEU_EMAIL_OU_NOREPLY_DO_GITHUB'
git add README.md bench.py lifecycle.py requirements.txt .gitignore configs docs tests scripts
git diff --cached --stat
git commit -m 'Initial single-user chatbot benchmark'
```

Nome/e-mail são autoria dos commits, não credenciais de login. Use o endereço noreply da sua conta se preferir privacidade.

Se ainda não há `origin`, associe o repositório criado:

```bash
git remote add origin git@github.com:yanwerneck/llm_local_inference_masters.git
git push -u origin main
```

Esse push exige que o SSH do **Mac** já esteja autenticado no GitHub. Se `origin` já existir, confira seu destino em vez de sobrescrevê-lo. Não use force push.

### Autoria não é autenticação

Para não publicar seu e-mail pessoal, use nas configurações locais do Git o endereço noreply mostrado em Settings → Emails da sua conta GitHub. Isso só afeta commits novos: um commit antigo pode continuar expondo a autoria anterior. Revise também o histórico que será enviado, não apenas os arquivos atuais. Não use `git push --all` se houver branches locais reservadas para histórico privado.

## 6. Rodar sem GitHub por enquanto

Transfira a pasta do projeto ou `chatbot-runtime-bench.zip` para `/workspace` pelo cliente SSH/SFTP que você usa para acessar o pod. Descompacte com a interface de arquivos ou `unzip` e siga o README. Não é necessário publicar o código para executar o benchmark.

Depois, baixe `results/` de volta para o computador **antes de encerrar o pod**. Não coloque chaves ou resultados brutos em um repositório público por conveniência.
