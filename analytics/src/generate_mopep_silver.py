"""Gera uma silver MOPEP por uma API OpenAI-compatible, como o vLLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from analytics.src.evaluate_mopep_tags import (
    ALLOWED_TAGS,
    compare_tags,
    normalize_tags,
    parse_tags,
    slug,
)
from analytics.src.openai_silver import (
    build_structured_prompt as build_prompt,
    output_schema,
)


DEFAULT_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_MODEL = "qwen-silver"
DEFAULT_OUTPUT_ROOT = Path("analytics/results/mopep-silver/qwen")
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
GENERATION_OPTIONS = {
    "temperature": 0.0,
    "top_k": 20,
    "enable_thinking": False,
}
PASSTHROUGH_COLUMNS = (
    "domain",
    "split",
    "taxonomy_version",
    "business_model_chars",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def response_format() -> dict[str, object]:
    structured = output_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": structured["name"],
            "strict": structured["strict"],
            "schema": structured["schema"],
        },
    }


def parse_structured_output(raw_response: str) -> tuple[list[str], bool, str]:
    try:
        parsed = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        return [], False, f"JSON invalido: {exc}"
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tags"), list):
        return [], False, "Resposta sem o objeto esperado {\"tags\": [...]}"
    tags = normalize_tags(parsed["tags"])
    unknown = sorted(set(tags) - set(ALLOWED_TAGS))
    if unknown:
        return [], False, f"Tags fora da taxonomia: {', '.join(unknown)}"
    return tags, True, ""


def load_source_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    if "business_model" not in frame:
        raise ValueError(f"Coluna ausente em {path.name}: business_model")

    frame = frame.loc[frame["business_model"].notna()].copy()
    frame["business_model"] = frame["business_model"].astype(str).str.strip()
    frame = frame.loc[frame["business_model"].str.len().gt(0)].copy()
    if frame.empty:
        raise ValueError(f"Nenhuma linha valida encontrada em {path}")

    generated_ids = frame["business_model"].map(sha256_text)
    if "example_id" not in frame:
        frame["example_id"] = generated_ids
    else:
        missing_id = frame["example_id"].isna() | (
            frame["example_id"].astype(str).str.strip().eq("")
        )
        frame.loc[missing_id, "example_id"] = generated_ids.loc[missing_id]
        frame["example_id"] = frame["example_id"].astype(str)

    duplicated = frame["example_id"].duplicated(keep=False)
    if duplicated.any():
        examples = ", ".join(sorted(frame.loc[duplicated, "example_id"].unique())[:5])
        raise ValueError(f"example_id duplicado no dataset: {examples}")

    frame["input_sha256"] = frame["business_model"].map(sha256_text)
    if "tags" in frame:
        frame["expected_tags"] = frame["tags"].apply(parse_tags)
    if "business_model_chars" not in frame:
        frame["business_model_chars"] = frame["business_model"].str.len()
    return frame.reset_index(drop=True)


def select_dataset(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame.reset_index(drop=True)
    if limit < 1:
        raise ValueError("--limit deve ser maior que zero")
    return (
        frame.sample(n=min(limit, len(frame)), random_state=seed)
        .sort_values("example_id")
        .reset_index(drop=True)
    )


def campaign_config(
    dataset: pd.DataFrame,
    *,
    model: str,
    model_revision: str,
    seed: int,
    max_output_tokens: int,
) -> dict[str, object]:
    inputs = [
        {"example_id": str(row.example_id), "input_sha256": str(row.input_sha256)}
        for row in dataset[["example_id", "input_sha256"]].itertuples(index=False)
    ]
    return {
        "provider": "openai-compatible",
        "model": model,
        "model_revision": model_revision,
        "seed": seed,
        "max_output_tokens": max_output_tokens,
        "generation_options": GENERATION_OPTIONS,
        "prompt_sha256": sha256_text(build_prompt("")),
        "schema_sha256": canonical_sha256(response_format()),
        "input_signature_sha256": canonical_sha256(inputs),
        "example_count": len(inputs),
    }


def resolve_run_dir(
    output_root: Path,
    dataset_path: Path,
    config: dict[str, object],
) -> Path:
    fingerprint = canonical_sha256(config)[:16]
    run_name = (
        f"{slug(dataset_path.stem)}__"
        f"{slug(str(config['model']))}__"
        f"{fingerprint}"
    )
    return output_root / run_name


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_run(
    run_dir: Path,
    dataset_path: Path,
    dataset: pd.DataFrame,
    config: dict[str, object],
    base_url: str,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run.json"
    fingerprint = canonical_sha256(config)
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("campaign_fingerprint") != fingerprint:
            raise ValueError(f"Configuracao divergente no run existente: {run_dir}")
        existing["updated_at"] = utc_now()
        existing["status"] = "running"
        _write_json_atomic(metadata_path, existing)
        return existing

    manifest_columns = ["example_id", "input_sha256"]
    manifest_columns.extend(column for column in PASSTHROUGH_COLUMNS if column in dataset)
    if "expected_tags" in dataset:
        manifest_columns.append("expected_tags")
    manifest = dataset[manifest_columns].copy()
    for column in ("expected_tags",):
        if column in manifest:
            manifest[column] = manifest[column].apply(
                lambda values: json.dumps(values, ensure_ascii=False)
            )
    manifest.to_csv(run_dir / "manifest.csv", index=False, encoding="utf-8")

    now = utc_now()
    metadata: dict[str, object] = {
        "run_id": run_dir.name,
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "campaign_fingerprint": fingerprint,
        "dataset": str(dataset_path.resolve()),
        "base_url": base_url,
        **config,
    }
    _write_json_atomic(metadata_path, metadata)
    return metadata


def append_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_result_history(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    results: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL corrompido em {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict) or "example_id" not in item:
            raise ValueError(f"Registro invalido em {path}:{line_number}")
        results.append(item)
    return results


def completed_example_ids(history: list[dict[str, object]]) -> set[str]:
    return {
        str(item["example_id"])
        for item in history
        if item.get("request_ok") is True and item.get("parse_ok") is True
    }


def _response_content(response: object) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return str(getattr(message, "content", "") or "")


def _usage_values(response: object) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def predict_one(
    client: object,
    *,
    model: str,
    example: pd.Series,
    seed: int,
    max_output_tokens: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    if max_attempts < 1:
        raise ValueError("max_attempts deve ser maior que zero")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds nao pode ser negativo")

    started = perf_counter()
    raw_response = ""
    error = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    effective_model = model
    response_id = ""
    request_succeeded = False
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": build_prompt(str(example["business_model"])),
                }],
                response_format=response_format(),
                temperature=GENERATION_OPTIONS["temperature"],
                seed=seed,
                max_tokens=max_output_tokens,
                extra_body={
                    "top_k": GENERATION_OPTIONS["top_k"],
                    "chat_template_kwargs": {
                        "enable_thinking": GENERATION_OPTIONS["enable_thinking"],
                    },
                },
            )
            raw_response = _response_content(response)
            request_succeeded = True
            usage = _usage_values(response)
            effective_model = str(getattr(response, "model", model) or model)
            response_id = str(getattr(response, "id", "") or "")
            generated_tags, parse_ok, parse_error = parse_structured_output(raw_response)
            if not parse_ok:
                raise ValueError(parse_error)
            return _result_row(
                example=example,
                model=model,
                effective_model=effective_model,
                response_id=response_id,
                generated_tags=generated_tags,
                raw_response=raw_response,
                request_ok=True,
                parse_ok=True,
                attempts=attempt,
                latency_ms=(perf_counter() - started) * 1000,
                usage=usage,
                error="",
            )
        except Exception as exc:  # A API pode expor varias subclasses transitórias.
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))

    return _result_row(
        example=example,
        model=model,
        effective_model=effective_model,
        response_id=response_id,
        generated_tags=[],
        raw_response=raw_response,
        request_ok=request_succeeded,
        parse_ok=False,
        attempts=max_attempts,
        latency_ms=(perf_counter() - started) * 1000,
        usage=usage,
        error=error,
    )


def _result_row(
    *,
    example: pd.Series,
    model: str,
    effective_model: str,
    response_id: str,
    generated_tags: list[str],
    raw_response: str,
    request_ok: bool,
    parse_ok: bool,
    attempts: int,
    latency_ms: float,
    usage: dict[str, int],
    error: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "example_id": str(example["example_id"]),
        "input_sha256": str(example["input_sha256"]),
        "provider": "openai-compatible",
        "label_source": "qwen",
        "model": model,
        "effective_model": effective_model,
        "response_id": response_id,
        "generated_tags": generated_tags,
        "raw_response": raw_response,
        "request_ok": request_ok,
        "parse_ok": parse_ok,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "error": error,
        "created_at": utc_now(),
        **usage,
    }
    if "expected_tags" in example.index:
        expected_tags = example["expected_tags"]
        result["expected_tags"] = expected_tags
        result.update(compare_tags(expected_tags, generated_tags))
    for column in PASSTHROUGH_COLUMNS:
        if column in example.index:
            result[column] = example[column]
    return result


def materialize_results(
    run_dir: Path,
    dataset: pd.DataFrame,
    history: list[dict[str, object]],
) -> dict[str, object]:
    latest_by_id: dict[str, dict[str, object]] = {}
    for item in history:
        latest_by_id[str(item["example_id"])] = item
    ordered = [
        latest_by_id[str(example_id)]
        for example_id in dataset["example_id"]
        if str(example_id) in latest_by_id
    ]
    predictions = pd.DataFrame(ordered)
    csv_frame = predictions.copy()
    for column in (
        "generated_tags",
        "expected_tags",
        "correct_tags",
        "missing_tags",
        "extra_tags",
    ):
        if column in csv_frame:
            csv_frame[column] = csv_frame[column].apply(
                lambda values: json.dumps(values, ensure_ascii=False)
            )
    csv_frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")

    total = len(dataset)
    successful = int(predictions["request_ok"].eq(True).sum()) if not predictions.empty else 0
    parsed = int(predictions["parse_ok"].eq(True).sum()) if not predictions.empty else 0
    summary: dict[str, object] = {
        "examples": total,
        "materialized": len(predictions),
        "successful": successful,
        "parsed": parsed,
        "failed": len(predictions) - successful,
        "pending": total - parsed,
        "request_success_rate": successful / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "mean_latency_ms": (
            float(predictions.loc[predictions["request_ok"].eq(True), "latency_ms"].mean())
            if successful else None
        ),
    }
    if "expected_tags" in predictions and parsed:
        comparable = predictions.loc[predictions["parse_ok"].eq(True)]
        summary["mean_individual_f1"] = float(comparable["f1"].mean())
        summary["exact_match_rate"] = float(comparable["exact_match"].mean())
    pd.DataFrame([summary]).to_csv(
        run_dir / "performance.csv", index=False, encoding="utf-8"
    )
    return summary


def available_models(client: object) -> list[str]:
    response = client.models.list()
    return sorted(
        str(getattr(model, "id", ""))
        for model in getattr(response, "data", [])
        if getattr(model, "id", None)
    )


def run_generation(
    client: object,
    *,
    dataset_path: Path,
    output_root: Path,
    base_url: str,
    model: str,
    model_revision: str,
    limit: int | None,
    seed: int,
    max_output_tokens: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> tuple[Path, dict[str, object]]:
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens deve ser maior que zero")
    try:
        models = available_models(client)
    except Exception as exc:
        raise ValueError(
            f"Nao foi possivel consultar {base_url.rstrip('/')}/models. "
            "Confirme se o vLLM e o tunel SSH estao ativos. "
            f"Erro: {type(exc).__name__}: {exc}"
        ) from exc
    if model not in models:
        rendered = ", ".join(models) if models else "nenhum"
        raise ValueError(f"Modelo {model!r} nao encontrado no endpoint. Disponiveis: {rendered}")

    dataset = select_dataset(load_source_dataset(dataset_path), limit, seed)
    config = campaign_config(
        dataset,
        model=model,
        model_revision=model_revision,
        seed=seed,
        max_output_tokens=max_output_tokens,
    )
    run_dir = resolve_run_dir(output_root, dataset_path, config)
    metadata = prepare_run(run_dir, dataset_path, dataset, config, base_url)
    results_path = run_dir / "predictions.jsonl"
    history = read_result_history(results_path)
    completed = completed_example_ids(history)
    pending = dataset.loc[~dataset["example_id"].astype(str).isin(completed)]

    interrupted = False
    try:
        for _, example in tqdm(
            pending.iterrows(),
            total=len(pending),
            desc=f"silver {model}",
        ):
            result = predict_one(
                client,
                model=model,
                example=example,
                seed=seed,
                max_output_tokens=max_output_tokens,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
            append_result(results_path, result)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        history = read_result_history(results_path)
        summary = materialize_results(run_dir, dataset, history)
        metadata["updated_at"] = utc_now()
        metadata["status"] = (
            "interrupted"
            if interrupted
            else "completed" if summary["pending"] == 0 else "incomplete"
        )
        metadata["summary"] = summary
        _write_json_atomic(run_dir / "run.json", metadata)

    if interrupted:
        raise KeyboardInterrupt
    return run_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.getenv("SILVER_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    try:
        run_dir, summary = run_generation(
            client,
            dataset_path=args.dataset,
            output_root=args.output_root,
            base_url=args.base_url,
            model=args.model,
            model_revision=args.model_revision,
            limit=args.limit,
            seed=args.seed,
            max_output_tokens=args.max_output_tokens,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt as exc:
        raise SystemExit("Execucao interrompida; o progresso foi salvo para retomada.") from exc

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Resultados: {run_dir}")
    if summary["pending"]:
        raise SystemExit(
            f"Execucao incompleta: {summary['pending']} exemplo(s) pendente(s). "
            "Repita o mesmo comando para tentar novamente."
        )


if __name__ == "__main__":
    main()
