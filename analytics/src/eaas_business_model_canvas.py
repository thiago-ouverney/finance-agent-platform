"""Normaliza enriquecimentos do EaaS e cria uma tabela-base de BMC.

O modulo aceita tanto o payload persistido no BigQuery quanto a resposta aninhada
da API GraphQL. O Canvas gerado aqui e uma hipotese inicial, baseada apenas em
evidencias presentes no enriquecimento; blocos sem evidencia nao sao inventados.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping


CANVAS_BLOCKS = (
    "customer_segments",
    "value_propositions",
    "channels",
    "customer_relationships",
    "revenue_streams",
    "key_resources",
    "key_activities",
    "key_partners",
    "cost_structure",
)

UNKNOWN = "Nao identificado nos dados EaaS"


def _jsonish(value: Any) -> Any:
    """Converte JSON (ou listas Python legadas) sem alterar valores simples."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def _mapping(value: Any) -> dict[str, Any]:
    parsed = _jsonish(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _list(value: Any) -> list[Any]:
    parsed = _jsonish(value)
    if parsed is None or parsed == "":
        return []
    if isinstance(parsed, (list, tuple, set)):
        return [item for item in parsed if item is not None and item != ""]
    return [parsed]


def _boolean(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "sim"}:
        return True
    if normalized in {"false", "0", "no", "nao", "não"}:
        return False
    return None


def _path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first(data: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _path(data, path)
        if value is not None and value != "" and value != [] and value != {}:
            return value
    return None


def _unique_strings(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _list(value):
            clean = str(item).strip()
            marker = clean.casefold()
            if clean and marker not in seen:
                seen.add(marker)
                result.append(clean)
    return result


def _slug(value: Any) -> str:
    if value is None or value == "":
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")


def _unwrap(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove envelopes GraphQL/BigQuery preservando colunas de identidade."""
    outer = dict(record)
    data = _mapping(outer.get("data"))
    company = _mapping(data.get("company")) if data else {}
    if not company:
        company = _mapping(outer.get("company"))
    base = company or outer

    payload = _mapping(outer.get("payload")) or _mapping(base.get("payload"))
    if payload:
        # O payload tem a representacao rica; colunas nativas completam lacunas.
        return {**outer, **payload}
    return dict(base)


def normalize_eaas_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Converte formatos EaaS conhecidos para um contrato analitico estavel."""
    raw = _unwrap(record)
    identity = _mapping(raw.get("identity"))
    digital = _mapping(
        raw.get("digitalPresence") or raw.get("digital_presence")
    )
    domain_details = _mapping(
        digital.get("domainDetails") or digital.get("domain_details")
    )
    social = _mapping(digital.get("socialMedia") or digital.get("social_media"))
    business = _mapping(
        raw.get("businessIntelligence") or raw.get("business_intelligence")
    )
    estimated = _mapping(business.get("estimated"))
    scores = _mapping(raw.get("scores"))
    mopep = _mapping(raw.get("mopep")) or _mapping(scores.get("mopep"))
    porte = _mapping(raw.get("company_size")) or _mapping(scores.get("porte"))
    crm = _mapping(raw.get("crmContext") or raw.get("crm_context"))
    receita = _mapping(raw.get("receita_federal")) or _mapping(
        identity.get("receitaFederal") or identity.get("receita_federal")
    )

    technologies = _unique_strings(
        raw.get("technologies"), domain_details.get("technologies")
    )
    tags = _unique_strings(
        raw.get("tags"),
        raw.get("cb_tags"),
        business.get("tags"),
        mopep.get("company_tags"),
        mopep.get("companyTags"),
    )
    competitors = _unique_strings(raw.get("competitors"), business.get("competitors"))

    social_fields: dict[str, Any] = {}
    for channel in (
        "linkedin",
        "facebook",
        "instagram",
        "tiktok",
        "pinterest",
        "youtube",
    ):
        social_fields[channel] = _first(raw, channel) or social.get(channel)
    social_fields["twitter_x"] = (
        _first(raw, "twitter_x", "twitterX")
        or social.get("twitterX")
        or social.get("twitter_x")
    )

    segment = _first(raw, "segment") or business.get("segment")
    estimated_segment = (
        _first(raw, "estimated_segment")
        or estimated.get("segment")
    )
    business_model = (
        _first(raw, "business_model")
        or mopep.get("business_model")
        or mopep.get("businessModel")
    )
    is_ecommerce = _boolean(_first(raw, "is_ecommerce"))
    if is_ecommerce is None and str(estimated_segment).casefold() == "ecommerce":
        is_ecommerce = True

    has_blog = _boolean(_first(raw, "has_blog"))
    if has_blog is None:
        has_blog = _boolean(domain_details.get("hasBlog"))

    cnpj = _first(raw, "cnpj")
    domain = _first(raw, "domain")
    name = _first(raw, "name") or identity.get("name")
    normalized = {
        "company_id": str(cnpj or domain or name or "").strip(),
        "cnpj": cnpj,
        "domain": domain,
        "name": name,
        "description": _first(raw, "description") or identity.get("description"),
        "main_cnae": (
            receita.get("main_cnae")
            or receita.get("mainCnae")
            or receita.get("cnae_principal")
        ),
        "segment": segment,
        "estimated_segment": estimated_segment,
        "business_model": business_model,
        "company_size": (
            porte.get("company_size_category") or porte.get("category")
        ),
        "employees": (
            _first(raw, "employees")
            or business.get("employees")
            or estimated.get("employees")
        ),
        "sales_team_size": (
            _first(raw, "sales_team_size") or business.get("salesTeamSize")
        ),
        "annual_revenue": (
            _first(raw, "annual_revenue", "estimated_revenue")
            or estimated.get("annualRevenue")
            or estimated.get("annual_revenue")
        ),
        "is_ecommerce": is_ecommerce,
        "has_blog": has_blog,
        "technologies": technologies,
        "tags": tags,
        "competitors": competitors,
        "social_media": social_fields,
        "relationship_rd": (
            _first(raw, "relationship_rd") or crm.get("relationshipRd")
        ),
        "relationship_totvs": (
            _first(raw, "relationship_totvs") or crm.get("relationshipTotvs")
        ),
        "sources_metadata": _mapping(raw.get("sources_metadata")),
        "warnings": _unique_strings(raw.get("_warnings"), raw.get("warnings")),
        "updated_at": _first(raw, "updated_at", "enriched_at"),
    }
    return normalized


def normalize_eaas_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_eaas_record(record) for record in records]


def load_eaas_records(path: Path) -> list[dict[str, Any]]:
    """Le CSV, Parquet, JSON ou JSONL exportado pelo EaaS/BigQuery."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        frame = pd.read_parquet(path)
        return frame.where(frame.notna(), None).to_dict(orient="records")
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            loaded = json.load(stream)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, Mapping) and isinstance(loaded.get("records"), list):
            return loaded["records"]
        if isinstance(loaded, Mapping):
            return [dict(loaded)]
    raise ValueError(f"Formato nao suportado: {path.suffix}")


def tags_for_company(company: Mapping[str, Any]) -> list[str]:
    """Cria tags rastreaveis, separando taxonomia EaaS de campos derivados."""
    tags: list[str] = []

    def add(prefix: str, value: Any) -> None:
        slug = _slug(value)
        if slug:
            tags.append(f"{prefix}:{slug}")

    add("segment", company.get("segment"))
    add("estimated-segment", company.get("estimated_segment"))
    add("business-model", company.get("business_model"))
    add("company-size", company.get("company_size"))
    add("relationship-rd", company.get("relationship_rd"))
    add("relationship-totvs", company.get("relationship_totvs"))
    if company.get("is_ecommerce") is not None:
        add("ecommerce", str(bool(company.get("is_ecommerce"))).lower())
    for value in company.get("tags") or []:
        add("eaas", value)
    for value in company.get("technologies") or []:
        add("technology", value)
    for channel, url in (company.get("social_media") or {}).items():
        if url:
            add("channel", channel)
    return list(dict.fromkeys(tags))


def _items_text(items: Iterable[str]) -> str:
    clean = [str(item).strip() for item in items if item]
    return "; ".join(clean) if clean else UNKNOWN


def baseline_canvas(company: Mapping[str, Any]) -> dict[str, Any]:
    """Gera um BMC conservador; inferencias sao marcadas para revisao."""
    segment = company.get("segment") or company.get("estimated_segment")
    business_model = company.get("business_model")
    description = company.get("description")
    is_ecommerce = company.get("is_ecommerce") is True
    raw_tags = " ".join(company.get("tags") or []).casefold()

    customer_segments = []
    model_text = str(business_model or "").casefold()
    has_b2b = "b2b" in model_text
    has_b2c = "b2c" in model_text
    if has_b2b and has_b2c:
        customer_segments.append(
            "Empresas e consumidores finais (hipotese baseada no modelo B2B/B2C)"
        )
    elif has_b2b:
        customer_segments.append("Empresas (hipotese baseada no modelo B2B)")
    elif has_b2c:
        customer_segments.append(
            "Consumidores finais (hipotese baseada no modelo B2C)"
        )
    elif "marketplace" in model_text:
        customer_segments.append(
            "Dois ou mais lados da plataforma (hipotese de marketplace)"
        )
    elif business_model:
        customer_segments.append(
            f"Orientacao comercial: {business_model}; publico-alvo requer validacao"
        )

    channels = []
    if company.get("domain"):
        channels.append(f"Website: {company['domain']}")
    for channel, url in (company.get("social_media") or {}).items():
        if url:
            channels.append(channel.replace("_", " ").title())
    if company.get("has_blog") is True:
        channels.append("Blog")

    relationships = []
    if is_ecommerce:
        relationships.append("Autosservico digital (hipotese)")
    if company.get("sales_team_size"):
        relationships.append("Venda assistida (hipotese; ha time de vendas informado)")

    revenue_streams = []
    if "saas" in model_text or "saas" in raw_tags or "subscription" in raw_tags:
        revenue_streams.append("Assinaturas recorrentes (hipotese)")
    if is_ecommerce:
        revenue_streams.append("Venda online (hipotese)")
    if business_model and not revenue_streams:
        revenue_streams.append(
            f"Modelo {business_model}; mecanismo de receita requer validacao"
        )

    resources = []
    technologies = company.get("technologies") or []
    if technologies:
        resources.append(f"Stack digital: {', '.join(technologies)}")
    if company.get("employees"):
        resources.append(f"Equipe: {company['employees']}")
    if company.get("domain"):
        resources.append("Presenca digital propria")

    activities = []
    if segment:
        activities.append(f"Operacao no segmento {segment} (hipotese)")
    if is_ecommerce:
        activities.append("Operacao de comercio eletronico (hipotese)")
    if company.get("has_blog") is True:
        activities.append("Producao de conteudo (hipotese)")

    evidence_fields = [
        key
        for key in (
            "description",
            "segment",
            "estimated_segment",
            "business_model",
            "technologies",
            "employees",
            "annual_revenue",
            "is_ecommerce",
        )
        if company.get(key) not in (None, "", [], {})
    ]
    coverage = round(len(evidence_fields) / 8, 2)

    return {
        "customer_segments": _items_text(customer_segments),
        "value_propositions": (
            f"Oferta descrita: {str(description).strip()}; validar valor percebido"
            if description
            else UNKNOWN
        ),
        "channels": _items_text(channels),
        "customer_relationships": _items_text(relationships),
        "revenue_streams": _items_text(revenue_streams),
        "key_resources": _items_text(resources),
        "key_activities": _items_text(activities),
        "key_partners": UNKNOWN,
        "cost_structure": UNKNOWN,
        "bmc_source": "eaas-baseline",
        "evidence_coverage": coverage,
        "evidence_fields": "|".join(evidence_fields),
        "needs_review": True,
    }


def build_canvas_table(companies: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retorna linhas com os nove blocos BMC e tags auditaveis."""
    rows: list[dict[str, Any]] = []
    for company in companies:
        canvas = baseline_canvas(company)
        tags = tags_for_company(company)
        rows.append(
            {
                "company_id": company.get("company_id"),
                "cnpj": company.get("cnpj"),
                "domain": company.get("domain"),
                "name": company.get("name"),
                "segment": company.get("segment"),
                "estimated_segment": company.get("estimated_segment"),
                "business_model": company.get("business_model"),
                **{block: canvas[block] for block in CANVAS_BLOCKS},
                "tags": "|".join(tags),
                "bmc_source": canvas["bmc_source"],
                "evidence_coverage": canvas["evidence_coverage"],
                "evidence_fields": canvas["evidence_fields"],
                "needs_review": canvas["needs_review"],
                "updated_at": company.get("updated_at"),
            }
        )
    return rows


def build_bmc_prompt(company: Mapping[str, Any]) -> str:
    """Cria prompt sem contatos pessoais para completar o BMC via LLM."""
    allowed = {
        key: company.get(key)
        for key in (
            "name",
            "domain",
            "description",
            "main_cnae",
            "segment",
            "estimated_segment",
            "business_model",
            "company_size",
            "employees",
            "sales_team_size",
            "annual_revenue",
            "is_ecommerce",
            "has_blog",
            "technologies",
            "tags",
            "competitors",
            "social_media",
        )
    }
    schema = {block: "string" for block in CANVAS_BLOCKS}
    schema.update({"tags": ["string"], "assumptions": ["string"], "confidence": 0.0})
    return (
        "Crie uma hipotese de Business Model Canvas em portugues do Brasil usando "
        "somente as evidencias abaixo. Nao trate concorrentes como parceiros. Quando "
        "nao houver evidencia, escreva 'Nao identificado nos dados EaaS'. Retorne "
        "apenas JSON valido no schema indicado.\n\n"
        f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"EVIDENCIAS:\n{json.dumps(allowed, ensure_ascii=False, default=str)}"
    )


def parse_bmc_response(content: str) -> dict[str, Any]:
    """Valida a estrutura minima de uma resposta JSON de BMC."""
    clean = content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.I)
    data = json.loads(clean)
    if not isinstance(data, dict):
        raise ValueError("A resposta do modelo deve ser um objeto JSON.")
    missing = [block for block in CANVAS_BLOCKS if not data.get(block)]
    if missing:
        raise ValueError(f"Blocos BMC ausentes: {', '.join(missing)}")
    return data
