"""Contrato versionado dos prompts de classificação e revisão MOPEP."""
from __future__ import annotations

import hashlib


PROMPT_VERSION = "mopep-tags-v1"
REVIEW_PROMPT_VERSION = "mopep-tags-review-v1"

TAG_DEFINITIONS = [
    ("empresa estabelecida", "As evidências indicam que a empresa já é estabelecida, com mais de 5 anos de existência."),
    ("estabelecimento físico", "As evidências apontam que a empresa possui um estabelecimento físico ou escritório onde a atividade principal é desempenhada."),
    ("b2b", "A empresa vende produtos ou serviços para outras empresas."),
    ("vendas remotas", "As negociações e vendas são realizadas remotamente, por telefone, e-mail, WhatsApp, videoconferência ou outros canais digitais, sem necessidade de visita presencial."),
    ("serviço", "A empresa oferece soluções intangíveis, como consultoria, manutenção, suporte técnico ou treinamentos, entregando valor por meio de conhecimento e experiência."),
    ("atuação nacional", "A empresa realiza operações ou atende clientes em todo o território brasileiro, com presença ou cobertura em múltiplas regiões ou estados."),
    ("tecnologia", "A empresa se apresenta como inovadora no uso de tecnologia em sua atuação, produtos ou serviços."),
    ("produto", "A empresa comercializa bens tangíveis ou digitais, como equipamentos, roupas, alimentos, softwares, e-books ou outros produtos."),
    ("whatsapp", "O site possui indícios de que a empresa realiza atendimento ou vendas por WhatsApp."),
    ("b2c", "A empresa vende produtos ou serviços diretamente para o consumidor final."),
    ("instagram", "O site possui link apontando para o perfil da empresa no Instagram."),
    ("compra planejada", "Os clientes costumam pesquisar, comparar alternativas ou planejar antes de comprar os produtos ou contratar os serviços."),
    ("blog", "O site possui um blog ou área dedicada à publicação de artigos, notícias ou conteúdos informativos."),
    ("público: empresas médias e grandes", "A empresa B2B ou B2B2C se posiciona com ênfase no atendimento a empresas médias e grandes, com maior faturamento, estrutura ou número de funcionários."),
    ("linkedin", "O site possui link apontando para o perfil da empresa no LinkedIn."),
    ("atuação regional", "A empresa atua predominantemente em uma região geográfica específica, como um estado, grupo de municípios ou área delimitada."),
    ("facebook", "O site possui link apontando para o perfil da empresa no Facebook."),
    ("atuação internacional", "A empresa possui operações, clientes ou negócios em mais de um país, atuando em mercados estrangeiros ou globalmente."),
    ("público: micro e pequenas empresas", "A empresa B2B ou B2B2C se posiciona com ênfase no atendimento a MEIs, microempresas ou pequenas empresas."),
    ("segmento: tecnologia da informação", "A empresa tem como atividade principal a oferta de soluções de tecnologia da informação, como desenvolvimento de software, infraestrutura, redes, segurança da informação ou suporte técnico."),
]
ALLOWED_TAGS = [name for name, _ in TAG_DEFINITIONS]


def build_prompt(business_model: str) -> str:
    tag_list = "\n".join(f"- {name}: {description}" for name, description in TAG_DEFINITIONS)
    return f"""
Seu objetivo é analisar o modelo de negócios de uma empresa e classificá-la usando somente as tags da lista abaixo:

{tag_list}

O modelo de negócios da empresa é:

{business_model}

Retorne somente uma lista Python com as tags aplicáveis.

Exemplo de resposta:

['empresa estabelecida', 'b2b', 'serviço', 'vendas remotas']

Procedimento:
- Avalie individualmente cada uma das 20 tags antes de responder.
- Considere todas as informações presentes no texto, mesmo quando algumas seções do Business Model Canvas estiverem vazias.
- Inclua uma tag quando houver evidência direta ou uma inferência razoável sustentada pelo texto.
- Não deixe de classificar a empresa apenas porque o documento está incompleto.
- Use [] somente depois de avaliar todas as tags e concluir que nenhuma delas pode ser sustentada.
- Retorne somente a lista final, sem mostrar a análise.

Regras:
- Use somente as tags apresentadas na lista.
- Copie exatamente o nome das tags, incluindo acentos.
- Não crie tags novas.
- Não inclua explicações ou qualquer texto fora da lista.
# - Não inclua uma tag quando não houver evidência suficiente.
# - Caso nenhuma tag seja aplicável, retorne [].
""".strip()


def build_review_prompt() -> str:
    return """
Revise a classificação anterior usando as regras e definições da taxonomia já apresentadas.
Procure tags ausentes sustentadas pelo BMC, tags incluídas sem evidência suficiente,
nomes fora da taxonomia, duplicações e violações do formato.
Retorne somente a lista Python final corrigida, sem explicações e sem mostrar a análise.
""".strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_contract() -> dict[str, str]:
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(build_prompt("")),
        "review_prompt_version": REVIEW_PROMPT_VERSION,
        "review_prompt_sha256": sha256_text(build_review_prompt()),
    }
