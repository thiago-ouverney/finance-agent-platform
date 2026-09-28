"""Exporta a janela do Prometheus e a correlaciona com requisições do bench.

O módulo usa somente a biblioteca padrão no processamento. ``httpx`` é
importado apenas para consultar a API HTTP do Prometheus.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from results_layout import artifact


SCHEMA_VERSION = 1
DEFAULT_QUERY = '{__name__=~"inference_(host|gpu)_.*|inference_exporter_up|vllm:kv_cache_usage_perc|vllm:gpu_cache_usage_perc"}'


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def normalize_prometheus_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("URL do Prometheus deve ser http(s), sem credenciais, query, fragmento ou path.")
    return f"{parsed.scheme}://{parsed.netloc}"


def prometheus_ready(prometheus_url: str, timeout: float = 5) -> bool:
    import httpx

    try:
        response = httpx.get(normalize_prometheus_url(prometheus_url) + "/-/ready", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def query_range(prometheus_url: str, query: str, start: float, end: float,
                step: str | float = "500ms", timeout: float = 120) -> list[dict]:
    """Consulta uma matrix e devolve amostras longas com labels preservados."""
    import httpx

    response = httpx.get(
        normalize_prometheus_url(prometheus_url) + "/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Consulta Prometheus falhou: {payload}")
    rows = []
    for series in payload.get("data", {}).get("result", []):
        labels = dict(series.get("metric") or {})
        metric = labels.pop("__name__", None)
        if not metric:
            continue
        for timestamp, raw in series.get("values", []):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                rows.append({
                    "timestamp_s": float(timestamp),
                    "metric": metric,
                    "value": value,
                    "labels": labels,
                })
    return sorted(rows, key=lambda row: (row["timestamp_s"], row["metric"], json.dumps(row["labels"], sort_keys=True)))


def metric_metadata(metric: str) -> tuple[str, str]:
    """Retorna unidade e tipo sem inventar semântica para séries desconhecidas."""
    if metric in {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}:
        return "fraction", "gauge"
    if metric.endswith("_bytes_total"):
        return "bytes", "counter"
    if metric.endswith("_bytes"):
        return "bytes", "gauge"
    if metric.endswith("_total"):
        return "count", "counter"
    if metric.endswith("_ratio"):
        return "fraction", "gauge"
    if metric.endswith("_celsius"):
        return "celsius", "gauge"
    if metric.endswith("_watts"):
        return "watts", "gauge"
    return "unknown", "gauge"


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def request_events(requests: list[dict]) -> list[dict]:
    rows = []
    names = (
        ("request_start", "request_start_epoch_s"),
        ("first_content", "first_content_epoch_s"),
        ("last_content", "last_content_epoch_s"),
        ("request_end", "request_end_epoch_s"),
    )
    for request in requests:
        started = request.get("request_start_epoch_s")
        for name, field in names:
            timestamp = request.get(field)
            if timestamp is None:
                continue
            rows.append({
                "experiment_id": request.get("experiment_id"),
                "request_uid": request.get("request_uid"),
                "block": request.get("block"),
                "benchmark_phase": request.get("benchmark_phase"),
                "event": name,
                "timestamp_s": timestamp,
                "timestamp_utc": utc_iso(timestamp),
                "request_elapsed_s": timestamp - started if started is not None else None,
            })
    return rows


def observed_phase(request: dict, timestamp: float) -> str:
    first = request.get("first_content_epoch_s")
    last = request.get("last_content_epoch_s")
    if first is None:
        return "request_without_first_content"
    if timestamp < first:
        return "prefill_observed"
    if last is None or timestamp <= last:
        return "decode_observed"
    return "response_close"


def correlate_samples(samples: list[dict], requests: list[dict], run_start: float) -> list[dict]:
    intervals = sorted(
        (request for request in requests
         if request.get("request_start_epoch_s") is not None and request.get("request_end_epoch_s") is not None),
        key=lambda row: row["request_start_epoch_s"],
    )
    rows = []
    for sample in samples:
        timestamp = sample["timestamp_s"]
        request = next((candidate for candidate in intervals
                        if candidate["request_start_epoch_s"] <= timestamp <= candidate["request_end_epoch_s"]), None)
        labels_json = json.dumps(sample["labels"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        unit, kind = metric_metadata(sample["metric"])
        rows.append({
            "experiment_id": request.get("experiment_id") if request else (requests[0].get("experiment_id") if requests else None),
            "request_uid": request.get("request_uid") if request else None,
            "runtime": request.get("runtime") if request else (requests[0].get("runtime") if requests else None),
            "model": request.get("model") if request else (requests[0].get("model") if requests else None),
            "block": request.get("block") if request else None,
            "benchmark_phase": request.get("benchmark_phase") if request else None,
            "scenario": request.get("scenario") if request else None,
            "workload_request_id": request.get("workload_request_id") if request else None,
            "workload_profile": request.get("workload_profile") if request else None,
            "bucket": request.get("bucket") if request else None,
            "turn_index": request.get("turn_index") if request else None,
            "repetition": request.get("repetition") if request else None,
            "observed_phase": observed_phase(request, timestamp) if request else "outside_request",
            "timestamp_s": timestamp,
            "timestamp_utc": utc_iso(timestamp),
            "run_elapsed_s": timestamp - run_start,
            "request_elapsed_s": timestamp - request["request_start_epoch_s"] if request else None,
            "metric": sample["metric"],
            "value": sample["value"],
            "unit": unit,
            "kind": kind,
            "labels_json": labels_json,
            "job": sample["labels"].get("job"),
            "instance": sample["labels"].get("instance"),
        })
    return rows


def phase_metrics(correlated: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in correlated:
        if not row.get("request_uid"):
            continue
        key = (
            row.get("experiment_id"), row.get("request_uid"), row.get("runtime"), row.get("model"),
            row.get("block"), row.get("benchmark_phase"), row.get("scenario"), row.get("repetition"),
            row.get("workload_request_id"), row.get("workload_profile"), row.get("bucket"), row.get("turn_index"),
            row.get("observed_phase"), row.get("metric"), row.get("unit"), row.get("kind"), row.get("labels_json"),
        )
        groups[key].append(row)
    result = []
    columns = (
        "experiment_id", "request_uid", "runtime", "model", "block", "benchmark_phase",
        "scenario", "repetition", "workload_request_id", "workload_profile", "bucket", "turn_index",
        "observed_phase", "metric", "unit", "kind", "labels_json",
    )
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        rows.sort(key=lambda row: row["timestamp_s"])
        values = [row["value"] for row in rows]
        duration = rows[-1]["timestamp_s"] - rows[0]["timestamp_s"] if len(rows) > 1 else None
        delta = values[-1] - values[0] if len(values) > 1 else None
        kind = rows[0]["kind"]
        result.append({
            **dict(zip(columns, key)),
            "sample_count": len(values),
            "mean": (sum(values) / len(values)) if kind == "gauge" else None,
            "p50": percentile(values, .50) if kind == "gauge" else None,
            "p95": percentile(values, .95) if kind == "gauge" else None,
            "min": min(values) if kind == "gauge" else None,
            "max": max(values) if kind == "gauge" else None,
            "first": values[0],
            "last": values[-1],
            "delta": delta,
            "rate_mean": delta / duration if kind == "counter" and duration and duration > 0 else None,
        })
    return result


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_checksums(output: str | Path) -> Path:
    """Reescreve o inventário externo incluindo o manifesto final autoritativo."""
    output = Path(output)
    dataset_path = artifact(output, "dataset-manifest.json")
    final_manifest_path = artifact(output, "manifest.json")
    if not dataset_path.is_file() or not final_manifest_path.is_file():
        raise RuntimeError("dataset-manifest.json e manifest.json são obrigatórios no pacote final")
    dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    files = dataset_manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("dataset-manifest.json não contém inventário de arquivos")
    checksum_rows = []
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("dataset-manifest.json contém entrada de arquivo inválida")
        path = output / record["path"]
        if not path.is_file():
            raise RuntimeError(f"arquivo do dataset ausente: {record['path']}")
        checksum_rows.append({"path": record["path"], "sha256": _hash(path)})
    checksum_rows.extend((
        {"path": str(dataset_path.relative_to(output)), "sha256": _hash(dataset_path)},
        {"path": str(final_manifest_path.relative_to(output)), "sha256": _hash(final_manifest_path)},
    ))
    checksum_path = artifact(output, "SHA256SUMS")
    checksum_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows),
        encoding="utf-8",
    )
    return checksum_path


def export_dataset(*, output: str | Path, manifest: dict, requests: list[dict],
                   prometheus_url: str, query: str = DEFAULT_QUERY, step: str = "500ms",
                   run_start_epoch_s: float, run_end_epoch_s: float,
                   margin_seconds: float = 1.0) -> dict:
    """Exporta dados portáteis; Prometheus informado é uma dependência obrigatória."""
    output = Path(output)
    prometheus_url = normalize_prometheus_url(prometheus_url)
    if not prometheus_ready(prometheus_url):
        raise RuntimeError(f"Prometheus indisponível em {prometheus_url}")
    query_start = run_start_epoch_s - max(0.0, margin_seconds)
    query_end = run_end_epoch_s + max(0.0, margin_seconds)
    remaining = query_end - time.time() + 0.2
    if remaining > 0:
        time.sleep(remaining)
    samples = query_range(prometheus_url, query, query_start, query_end, step)
    if not samples:
        raise RuntimeError("Prometheus não retornou amostras para a janela do benchmark.")

    events = request_events(requests)
    correlated = correlate_samples(samples, requests, run_start_epoch_s)
    summaries = phase_metrics(correlated)
    sample_rows = []
    for sample in samples:
        unit, kind = metric_metadata(sample["metric"])
        labels_json = json.dumps(sample["labels"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        sample_rows.append({
            "experiment_id": manifest.get("experiment_id"),
            "timestamp_s": sample["timestamp_s"],
            "timestamp_utc": utc_iso(sample["timestamp_s"]),
            "run_elapsed_s": sample["timestamp_s"] - run_start_epoch_s,
            "metric": sample["metric"],
            "value": sample["value"],
            "unit": unit,
            "kind": kind,
            "labels_json": labels_json,
            "job": sample["labels"].get("job"),
            "instance": sample["labels"].get("instance"),
        })

    files = {
        "all_requests": artifact(output, "all-requests.csv"),
        "request_events": artifact(output, "request-events.csv"),
        "prometheus_samples": artifact(output, "prometheus-samples.csv"),
        "telemetry_by_request": artifact(output, "telemetry-by-request.csv"),
        "request_phase_metrics": artifact(output, "request-phase-metrics.csv"),
    }
    request_fields = [
        "experiment_id", "request_uid", "runtime", "model", "block", "benchmark_phase",
        "scenario", "repetition", "request_index", "status", "error", "mode",
        "workload_request_id", "workload_profile", "bucket", "rendered_prompt_tokens",
        "conversation_index", "turn_index", "request_sha256", "prompt_tokens", "completion_tokens",
        "total_tokens", "request_start_epoch_s", "first_content_epoch_s", "last_content_epoch_s",
        "request_end_epoch_s", "time_to_first_token_seconds", "time_to_first_token_ms",
        "generation_time_seconds", "end_to_end_latency_seconds", "decode_tokens_per_second",
        "end_to_end_tokens_per_second", "stream_content_event_count", "usage_observed",
    ]
    _write_csv(files["all_requests"], requests, request_fields)
    _write_csv(files["request_events"], events)
    _write_csv(files["prometheus_samples"], sample_rows)
    _write_csv(files["telemetry_by_request"], correlated)
    _write_csv(files["request_phase_metrics"], summaries)

    inventory = []
    for name, path in files.items():
        inventory.append({
            "name": name,
            "path": str(path.relative_to(output)),
            "sha256": _hash(path),
            "bytes": path.stat().st_size,
        })
    response_metadata = manifest.get("responses")
    if isinstance(response_metadata, dict) and isinstance(response_metadata.get("path"), str):
        response_path = output / response_metadata["path"]
        if not response_path.is_file():
            raise RuntimeError("Manifesto declara responses.jsonl ausente no pacote.")
        inventory.append({
            "name": "responses",
            "path": str(response_path.relative_to(output)),
            "sha256": _hash(response_path),
            "bytes": response_path.stat().st_size,
        })
        response_manifest_raw = response_metadata.get("manifest_path")
        if isinstance(response_manifest_raw, str):
            response_manifest_path = output / response_manifest_raw
            if not response_manifest_path.is_file():
                raise RuntimeError("Manifesto de responses.jsonl ausente no pacote.")
            inventory.append({
                "name": "responses_manifest",
                "path": str(response_manifest_path.relative_to(output)),
                "sha256": _hash(response_manifest_path),
                "bytes": response_manifest_path.stat().st_size,
            })
    dataset_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": manifest.get("experiment_id"),
        "status": manifest.get("status"),
        "runtime": manifest.get("config", {}).get("runtime"),
        "model": manifest.get("config", {}).get("model"),
        "config": manifest.get("config"),
        "model_identity": manifest.get("model_identity"),
        "tokenizer": manifest.get("tokenizer"),
        "runtime_process": manifest.get("runtime_process"),
        "cache_policy_details": manifest.get("cache_policy_details"),
        "benchmark": {
            "benchmark_profile": manifest.get("benchmark_profile"),
            "smoke": manifest.get("smoke"),
            "requests": manifest.get("requests"),
            "repetitions": manifest.get("repetitions"),
            "warmup_requests_per_case": manifest.get("warmup_requests_per_case"),
            "scenarios": manifest.get("scenarios"),
            "input_tokens": manifest.get("input_tokens"),
            "output_tokens": manifest.get("output_tokens"),
            "mode": manifest.get("mode"),
            "conversation_turns": manifest.get("conversation_turns"),
            "seed": manifest.get("seed"),
            "profile": manifest.get("profile"),
            "generation_parameters": manifest.get("generation_parameters"),
            "workload": manifest.get("workload"),
            "request_dataset": manifest.get("request_dataset"),
            "responses": manifest.get("responses"),
            "conversation_fixture": manifest.get("conversation_fixture"),
            "warmup_conversation_fixture": manifest.get("warmup_conversation_fixture"),
        },
        "environment": {
            "platform": manifest.get("platform"),
            "python": manifest.get("python"),
            "git": manifest.get("git"),
            "hardware": manifest.get("hardware"),
            "runtime_environment": manifest.get("runtime_environment"),
            "telemetry": manifest.get("telemetry"),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request_count": len(requests),
        "telemetry_sample_count": len(samples),
        "correlated_sample_count": sum(bool(row.get("request_uid")) for row in correlated),
        "observed_phase_definition": {
            "prefill_observed": "request_start <= sample < first_content; inclui HTTP, fila e scheduling, não é tempo puro de kernel prefill",
            "decode_observed": "first_content <= sample <= last_content",
            "response_close": "last_content < sample <= request_end",
        },
        "prometheus": {
            "url": prometheus_url,
            "query": query,
            "step": step,
            "start_epoch_s": query_start,
            "end_epoch_s": query_end,
            "metrics_observed": sorted({row["metric"] for row in samples}),
            "absence_policy": "série ausente permanece ausente; não é convertida em zero",
        },
        "files": inventory,
    }
    dataset_path = artifact(output, "dataset-manifest.json")
    dataset_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalize_checksums(output)
    return dataset_manifest
