"""Prepare deterministic calibration text from the private MOPEP golden CSV."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


PROMPT_VERSION = "mopep-ollama-list-v1"
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
ALLOWED_TAGS = frozenset(name for name, _ in TAG_DEFINITIONS)


def build_prompt(business_model: str) -> str:
    """Render the versioned MOPEP list-output prompt used by the Ollama evaluator."""
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


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True)
class GoldenExample:
    example_id: str
    business_model: str
    tags: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_tags(value: str, row_number: int) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Linha {row_number}: tags nao sao JSON/lista valida") from exc

    if not isinstance(parsed, list) or any(not isinstance(tag, str) for tag in parsed):
        raise ValueError(f"Linha {row_number}: tags deve ser uma lista de strings")
    normalized = tuple(" ".join(tag.strip().split()) for tag in parsed)
    if any(not tag for tag in normalized):
        raise ValueError(f"Linha {row_number}: tag vazia")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Linha {row_number}: tags duplicadas")
    unknown = sorted(set(normalized) - ALLOWED_TAGS)
    if unknown:
        raise ValueError(f"Linha {row_number}: tags fora da taxonomia: {', '.join(unknown)}")
    return normalized


def load_golden_csv(path: Path, *, allow_test_split: bool = False) -> list[GoldenExample]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset de calibracao nao encontrado: {path}")
    if path.name.casefold() == "test.csv" and not allow_test_split:
        raise ValueError("test.csv e reservado para avaliacao final e nao pode calibrar a quantizacao")

    examples: list[GoldenExample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing = {"business_model", "tags"} - columns
        id_column = "id" if "id" in columns else "example_id" if "example_id" in columns else None
        if missing or id_column is None:
            required = sorted(missing | ({"id"} if id_column is None else set()))
            raise ValueError(f"Colunas ausentes em {path.name}: {', '.join(required)}")

        for row_number, row in enumerate(reader, start=2):
            example_id = str(row.get(id_column, "")).strip()
            business_model = str(row.get("business_model", "")).strip()
            if not example_id:
                raise ValueError(f"Linha {row_number}: id vazio")
            if example_id in seen_ids:
                raise ValueError(f"Linha {row_number}: id duplicado: {example_id}")
            if not business_model:
                raise ValueError(f"Linha {row_number}: business_model vazio")
            tags = _parse_tags(str(row.get("tags", "")), row_number)
            examples.append(GoldenExample(example_id, business_model, tags))
            seen_ids.add(example_id)

    if not examples:
        raise ValueError(f"Dataset vazio: {path}")
    return examples


def select_calibration_examples(
    examples: Sequence[GoldenExample],
    *,
    limit: int,
    seed: int,
) -> list[GoldenExample]:
    if limit < 1:
        raise ValueError("O limite de calibracao deve ser maior que zero")

    def rank(example: GoldenExample) -> tuple[str, str]:
        value = f"{seed}\0{example.example_id}".encode("utf-8")
        return hashlib.sha256(value).hexdigest(), example.example_id

    return sorted(examples, key=rank)[: min(limit, len(examples))]


def messages_for_example(example: GoldenExample) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_prompt(example.business_model),
        },
        {
            "role": "assistant",
            "content": json.dumps(list(example.tags), ensure_ascii=False),
        },
    ]


def render_calibration_texts(
    examples: Sequence[GoldenExample],
    tokenizer: ChatTokenizer,
) -> list[str]:
    rendered: list[str] = []
    for example in examples:
        text = tokenizer.apply_chat_template(
            messages_for_example(example),
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chat template produziu texto vazio para {example.example_id}")
        rendered.append(text)
    return rendered


def selection_sha256(examples: Sequence[GoldenExample]) -> str:
    payload = json.dumps(
        [example.example_id for example in examples],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_metadata(
    path: Path,
    selected: Sequence[GoldenExample],
    *,
    available_count: int,
    limit: int,
    seed: int,
) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {
        "source_filename": path.name,
        "source_sha256": sha256_file(path),
        "available_examples": available_count,
        "selected_examples": len(selected),
        "requested_limit": limit,
        "selection_seed": seed,
        "selection_sha256": selection_sha256(selected),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(build_prompt("").encode("utf-8")).hexdigest(),
        "includes_expected_answer": True,
    }


def inspect_dataset(path: Path, *, limit: int, seed: int) -> dict[str, object]:
    examples = load_golden_csv(path)
    if len(examples) < limit:
        raise ValueError(
            f"Dataset tem {len(examples)} exemplos, menos que os {limit} solicitados"
        )
    selected = select_calibration_examples(examples, limit=limit, seed=seed)
    return calibration_metadata(
        path.expanduser().resolve(),
        selected,
        available_count=len(examples),
        limit=limit,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(inspect_dataset(args.dataset, limit=args.limit, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
