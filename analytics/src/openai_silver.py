"""Gera a silver MOPEP com a OpenAI Batch API sem enviar os rótulos esperados."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from analytics.src.evaluate_mopep_tags import (
    ALLOWED_TAGS,
    TAG_DEFINITIONS,
    compare_tags,
    csv_ready,
    load_dataset,
    normalize_tags,
    slug,
    summarize,
)


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MAX_OUTPUT_TOKENS = 256
MAX_BATCH_REQUESTS = 50_000
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass(frozen=True)
class BatchPricing:
    """Preços em USD por 1 milhão de tokens; revise antes de cada campanha."""

    input_per_million: float = 0.05
    cached_input_per_million: float = 0.005
    output_per_million: float = 0.40
    as_of: str = "2026-09-27"


def build_structured_prompt(business_model: str) -> str:
    tag_list = "\n".join(
        f"- {name}: {description}" for name, description in TAG_DEFINITIONS
    )
    return f"""
Seu objetivo é analisar o modelo de negócios de uma empresa e classificá-la
usando somente as tags da lista abaixo:

{tag_list}

Procedimento:
- Avalie individualmente cada uma das 20 tags.
- Considere todas as informações presentes no texto, mesmo quando algumas
  seções do Business Model Canvas estiverem vazias.
- Inclua uma tag quando houver evidência direta ou inferência razoável
  sustentada pelo texto.
- Não crie tags e copie os nomes exatamente como aparecem na lista.
- Retorne uma lista vazia somente depois de avaliar todas as tags.

Modelo de negócios da empresa:

{business_model}
""".strip()


def output_schema() -> dict[str, object]:
    return {
        "type": "json_schema",
        "name": "mopep_tags",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": ALLOWED_TAGS},
                    "maxItems": len(ALLOWED_TAGS),
                }
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
    }


def response_body(
    business_model: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, object]:
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens deve ser maior que zero")
    return {
        "model": model,
        "input": build_structured_prompt(business_model),
        "reasoning": {"effort": "minimal"},
        "max_output_tokens": max_output_tokens,
        "text": {"format": output_schema()},
        "store": False,
    }


def select_dataset(dataset_path: Path, limit: int | None, seed: int) -> pd.DataFrame:
    dataset = load_dataset(dataset_path)
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("limit deve ser maior que zero")
    return dataset.sample(
        n=min(limit, len(dataset)), random_state=seed
    ).reset_index(drop=True)


def custom_id(split: str, row_number: int, example_id: str) -> str:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:16]
    return f"{slug(split)}-{row_number:06d}-{digest}"


def build_batch_records(
    dataset: pd.DataFrame,
    split: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if len(dataset) > MAX_BATCH_REQUESTS:
        raise ValueError(
            f"O split possui {len(dataset)} exemplos; fragmente em lotes de no "
            f"máximo {MAX_BATCH_REQUESTS}."
        )

    records: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for row_number, example in dataset.reset_index(drop=True).iterrows():
        example_id = str(example["example_id"])
        request_id = custom_id(split, row_number, example_id)
        business_model = str(example["business_model"])
        records.append({
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": response_body(business_model, model, max_output_tokens),
        })
        manifest = {
            "custom_id": request_id,
            "example_id": example_id,
            "input_sha256": hashlib.sha256(
                business_model.encode("utf-8")
            ).hexdigest(),
            "expected_tags": example["expected_tags"],
        }
        for column in (
            "domain", "split", "label_source", "taxonomy_version",
            "business_model_chars",
        ):
            if column in example.index:
                manifest[column] = example[column]
        manifest_rows.append(manifest)
    return records, pd.DataFrame(manifest_rows)


def estimated_input_tokens(record: dict[str, object]) -> int:
    """Estimativa prática local; inclui prompt, schema e envelope JSON."""
    payload = json.dumps(record["body"], ensure_ascii=False, separators=(",", ":"))
    return math.ceil(len(payload.encode("utf-8")) / 3) + 200


def input_tokens_hard_ceiling(record: dict[str, object]) -> int:
    """Teto local deliberadamente folgado: no máximo um token por byte."""
    payload = json.dumps(record["body"], ensure_ascii=False, separators=(",", ":"))
    return len(payload.encode("utf-8")) + 200


def estimate_batch_cost(
    records: Iterable[dict[str, object]],
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    pricing: BatchPricing = BatchPricing(),
) -> dict[str, object]:
    materialized = list(records)
    input_tokens = sum(estimated_input_tokens(record) for record in materialized)
    input_ceiling = sum(input_tokens_hard_ceiling(record) for record in materialized)
    output_tokens = len(materialized) * max_output_tokens
    input_usd = input_tokens * pricing.input_per_million / 1_000_000
    input_ceiling_usd = input_ceiling * pricing.input_per_million / 1_000_000
    output_usd = output_tokens * pricing.output_per_million / 1_000_000
    return {
        "requests": len(materialized),
        "estimated_input_tokens": input_tokens,
        "input_tokens_hard_ceiling": input_ceiling,
        "max_output_tokens": output_tokens,
        "estimated_input_usd": input_usd,
        "input_usd_hard_ceiling": input_ceiling_usd,
        "max_output_usd": output_usd,
        "estimated_total_usd": input_usd + output_usd,
        "max_total_usd": input_ceiling_usd + output_usd,
        "pricing": asdict(pricing),
    }


def enforce_budget(estimates: Iterable[dict[str, object]], max_total_usd: float) -> float:
    if max_total_usd <= 0:
        raise ValueError("max_total_usd deve ser maior que zero")
    ceiling = sum(float(item["max_total_usd"]) for item in estimates)
    if ceiling > max_total_usd:
        raise ValueError(
            f"Teto local de US$ {ceiling:.4f} excede o orçamento "
            f"de US$ {max_total_usd:.4f}. Reduza os splits/limites ou aumente o "
            "orçamento conscientemente."
        )
    return ceiling


def write_jsonl(records: Iterable[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_manifest(manifest: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_ready(manifest).to_csv(path, index=False, encoding="utf-8")
    return path


def submit_batch(client: object, input_path: Path, split: str) -> object:
    with input_path.open("rb") as stream:
        uploaded = client.files.create(file=stream, purpose="batch")
    return client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"pipeline": "mopep-openai-silver", "split": split},
    )


def response_output_text(body: dict[str, object]) -> str:
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                return str(content.get("text") or "")
    return ""


def parse_structured_output(raw_response: str) -> tuple[list[str], bool]:
    try:
        parsed = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError):
        return [], False
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tags"), list):
        return [], False
    tags = normalize_tags(parsed["tags"])
    if not set(tags).issubset(ALLOWED_TAGS):
        return [], False
    return tags, True


def _cached_tokens(usage: dict[str, object]) -> int:
    details = usage.get("input_tokens_details") or {}
    return int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0


def usage_cost(usage: dict[str, object], pricing: BatchPricing) -> float:
    input_tokens = int(usage.get("input_tokens") or 0)
    cached_tokens = min(input_tokens, _cached_tokens(usage))
    output_tokens = int(usage.get("output_tokens") or 0)
    return (
        (input_tokens - cached_tokens) * pricing.input_per_million
        + cached_tokens * pricing.cached_input_per_million
        + output_tokens * pricing.output_per_million
    ) / 1_000_000


def parse_batch_output(
    output_jsonl: str,
    manifest: pd.DataFrame,
    model: str,
    pricing: BatchPricing = BatchPricing(),
) -> pd.DataFrame:
    result_by_id: dict[str, dict[str, object]] = {}
    for line in output_jsonl.splitlines():
        if line.strip():
            item = json.loads(line)
            result_by_id[str(item["custom_id"])] = item

    predictions: list[dict[str, object]] = []
    for _, expected in manifest.iterrows():
        request_id = str(expected["custom_id"])
        item = result_by_id.get(request_id)
        response = item.get("response") if item else None
        body = response.get("body", {}) if isinstance(response, dict) else {}
        status_code = response.get("status_code") if isinstance(response, dict) else None
        raw_response = response_output_text(body) if isinstance(body, dict) else ""
        generated_tags, parse_ok = parse_structured_output(raw_response)
        request_ok = status_code == 200 and body.get("status") == "completed"
        error_value = item.get("error") if item else "resultado ausente"
        if not request_ok and not error_value and isinstance(body, dict):
            error_value = body.get("error") or body.get("incomplete_details")
        usage = body.get("usage") or {} if isinstance(body, dict) else {}
        expected_tags = expected["expected_tags"]
        if isinstance(expected_tags, str):
            expected_tags = json.loads(expected_tags)
        prediction = {
            "example_id": expected["example_id"],
            "input_sha256": expected["input_sha256"],
            "model": model,
            "provider": "openai",
            "expected_tags": expected_tags,
            "generated_tags": generated_tags,
            "raw_response": raw_response,
            "request_ok": request_ok,
            "parse_ok": parse_ok,
            "latency_ms": None,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "cached_input_tokens": _cached_tokens(usage),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cost_usd": usage_cost(usage, pricing),
            "error": "" if request_ok else json.dumps(error_value, ensure_ascii=False),
        }
        prediction.update(compare_tags(expected_tags, generated_tags))
        for column in (
            "domain", "split", "label_source", "taxonomy_version",
            "business_model_chars",
        ):
            if column in expected.index:
                prediction[column] = expected[column]
        predictions.append(prediction)
    return pd.DataFrame(predictions)


def make_run_id(model: str, split: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}__openai__{slug(model)}__{slug(split)}__{uuid.uuid4().hex[:8]}"


def write_run(
    predictions: pd.DataFrame,
    dataset_path: Path,
    split: str,
    model: str,
    output_root: Path,
    batch_id: str,
    pricing: BatchPricing = BatchPricing(),
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Path:
    if predictions.empty:
        raise ValueError("Nenhuma predição para materializar")
    run_id = make_run_id(model, split)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    performance = summarize(predictions)
    performance["input_tokens"] = predictions["input_tokens"].sum()
    performance["cached_input_tokens"] = predictions["cached_input_tokens"].sum()
    performance["output_tokens"] = predictions["output_tokens"].sum()
    performance["cost_usd"] = predictions["cost_usd"].sum()
    csv_ready(predictions).to_csv(
        run_dir / "predictions.csv", index=False, encoding="utf-8"
    )
    performance.to_csv(run_dir / "performance.csv", index=False, encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "dataset": str(dataset_path.resolve()),
        "dataset_rows": len(predictions),
        "split": split,
        "batch_id": batch_id,
        "prompt_sha256": hashlib.sha256(
            build_structured_prompt("").encode("utf-8")
        ).hexdigest(),
        "generation_options": {
            "reasoning_effort": "minimal",
            "max_output_tokens": max_output_tokens,
            "structured_outputs": True,
        },
        "pricing": asdict(pricing),
        "cost_usd": float(predictions["cost_usd"].sum()),
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir
