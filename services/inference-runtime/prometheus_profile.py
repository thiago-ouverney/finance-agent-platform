"""Requisição sequencial OpenAI-compatible correlacionada com Prometheus."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from bench import stream_messages_request


DEFAULT_QUERY = '{__name__=~"inference_(host|gpu)_.*|inference_exporter_up|vllm:kv_cache_usage_perc"}'


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def list_models(base_url: str, api_key: str = "", timeout: float = 10) -> list[str]:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(base_url.rstrip("/") + "/v1/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]


def prometheus_ready(prometheus_url: str, timeout: float = 5) -> bool:
    import httpx

    try:
        response = httpx.get(prometheus_url.rstrip("/") + "/-/ready", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def query_range(prometheus_url: str, query: str, start: float, end: float,
                step: str | float = "1s", timeout: float = 30) -> list[dict]:
    """Consulta matriz do Prometheus e devolve linhas normalizadas."""
    import httpx

    response = httpx.get(
        prometheus_url.rstrip("/") + "/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Consulta Prometheus falhou: {payload}")
    rows: list[dict] = []
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
            rows.append({"metric": metric, "timestamp": float(timestamp), "value": value, "labels": labels})
    return rows


def profile_request(*, base_url: str, model: str, messages: list[dict],
                    prometheus_url: str = "http://127.0.0.1:9090", max_tokens: int = 256,
                    timeout: float = 180, api_key: str = "", query: str = DEFAULT_QUERY,
                    margin_before_seconds: float = 2, margin_after_seconds: float = 2,
                    prometheus_step: str | float = "500ms") -> dict:
    """Executa uma requisição e recupera a telemetria do mesmo intervalo."""
    if not messages:
        raise ValueError("messages não pode ser vazio.")
    if max_tokens <= 0:
        raise ValueError("max_tokens deve ser positivo.")
    if not prometheus_ready(prometheus_url):
        raise RuntimeError(f"Prometheus indisponível em {prometheus_url}")

    cfg = {"base_url": base_url.rstrip("/"), "model": model}
    started_epoch = time.time()
    request = stream_messages_request(cfg, messages, timeout, api_key, max_tokens=max_tokens)
    request_end_epoch = started_epoch + request["end_to_end_latency_seconds"]
    first_epoch = (
        started_epoch + request["time_to_first_token_seconds"]
        if request.get("time_to_first_token_seconds") is not None else None
    )
    last_content_epoch = (
        first_epoch + request["generation_time_seconds"]
        if first_epoch is not None and request.get("generation_time_seconds") is not None else None
    )
    query_start = started_epoch - max(0, margin_before_seconds)
    query_end = request_end_epoch + max(0, margin_after_seconds)
    remaining = query_end - time.time() + 0.2
    if remaining > 0:
        time.sleep(remaining)
    telemetry = query_range(prometheus_url, query, query_start, query_end, prometheus_step)
    if not telemetry:
        raise RuntimeError(
            "O Prometheus não retornou telemetria no intervalo da requisição; "
            "confirme make prometheus-status e aguarde o primeiro scrape."
        )
    events = [
        {"name": "request_start", "timestamp": started_epoch, "elapsed_seconds": 0.0},
    ]
    if first_epoch is not None:
        events.append({"name": "first_token", "timestamp": first_epoch,
                       "elapsed_seconds": first_epoch - started_epoch})
    if last_content_epoch is not None:
        events.append({"name": "last_content", "timestamp": last_content_epoch,
                       "elapsed_seconds": last_content_epoch - started_epoch})
    events.append({"name": "request_end", "timestamp": request_end_epoch,
                   "elapsed_seconds": request_end_epoch - started_epoch})
    return {
        "schema_version": 1,
        "metadata": {
            "base_url": cfg["base_url"],
            "model": model,
            "prometheus_url": prometheus_url.rstrip("/"),
            "started_utc": utc_iso(started_epoch),
            "ended_utc": utc_iso(request_end_epoch),
            "max_tokens": max_tokens,
            "messages_sha256": hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "telemetry_query": query,
            "telemetry_window": {"start": query_start, "end": query_end, "step": prometheus_step},
        },
        "request": request,
        "events": events,
        "telemetry": telemetry,
    }


def profile_prompts(prompts: list[str], **kwargs) -> list[dict]:
    """Executa prompts estritamente em sequência."""
    results = []
    for prompt in prompts:
        results.append(profile_request(messages=[{"role": "user", "content": prompt}], **kwargs))
    return results


def save_profile(profile: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def telemetry_frame(profile: dict):
    import pandas as pd

    started = next(event["timestamp"] for event in profile["events"] if event["name"] == "request_start")
    rows = []
    for row in profile.get("telemetry", []):
        labels = {f"label_{key}": value for key, value in row.get("labels", {}).items()}
        rows.append({**row, "elapsed_seconds": row["timestamp"] - started, **labels})
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["metric", "timestamp", "value", "labels", "elapsed_seconds"])


def summary_frame(profiles: list[dict] | dict):
    import pandas as pd

    profiles = [profiles] if isinstance(profiles, dict) else profiles
    rows = []
    for index, profile in enumerate(profiles, 1):
        request = profile["request"]
        rows.append({
            "run": index,
            "model": profile["metadata"]["model"],
            "ttft_ms": request.get("time_to_first_token_ms"),
            "decode_tokens_s": request.get("decode_tokens_per_second"),
            "effective_tokens_s": request.get("end_to_end_tokens_per_second"),
            "e2e_seconds": request.get("end_to_end_latency_seconds"),
            "prompt_tokens": request.get("prompt_tokens"),
            "completion_tokens": request.get("completion_tokens"),
        })
    return pd.DataFrame(rows)


def _counter_rate(frame, metric: str):
    import pandas as pd

    if frame.empty or "metric" not in frame:
        return frame.copy()
    subset = frame[frame["metric"] == metric].sort_values("timestamp").copy()
    if subset.empty:
        return subset
    label_columns = [column for column in subset if column.startswith("label_")]
    parts = []
    iterator = subset.groupby(label_columns, dropna=False) if label_columns else [(None, subset)]
    for _labels, part in iterator:
        part = part.copy()
        part["rate"] = part["value"].diff() / part["timestamp"].diff()
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else subset


def plot_profile(profile: dict):
    """Plota recursos no tempo com as regiões TTFT/prefill e decode observadas."""
    import matplotlib.pyplot as plt

    frame = telemetry_frame(profile)
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    specs = [
        ("inference_host_cpu_utilization_ratio", "CPU", 100),
        ("inference_gpu_utilization_ratio", "GPU", 100),
    ]
    for metric, label, scale in specs:
        rows = frame[frame.get("metric") == metric] if not frame.empty else frame
        for key, part in _series(rows):
            axes[0].plot(part["elapsed_seconds"], part["value"] * scale, marker="o", label=label + key)
    axes[0].set_ylabel("Utilização (%)")

    memory_specs = [
        ("inference_host_memory_used_bytes", "RAM"),
        ("inference_host_swap_used_bytes", "Swap"),
        ("inference_gpu_memory_used_bytes", "VRAM"),
    ]
    for metric, label in memory_specs:
        rows = frame[frame.get("metric") == metric] if not frame.empty else frame
        for key, part in _series(rows):
            axes[1].plot(part["elapsed_seconds"], part["value"] / 1073741824, marker="o", label=label + key)
    axes[1].set_ylabel("Memória (GiB)")

    for metric, label in (
        ("inference_host_disk_read_bytes_total", "Leitura"),
        ("inference_host_disk_write_bytes_total", "Escrita"),
    ):
        rates = _counter_rate(frame, metric)
        for key, part in _series(rates):
            axes[2].plot(part["elapsed_seconds"], part["rate"] / 1048576, marker="o", label=label + key)
    axes[2].set_ylabel("Disco (MiB/s)")

    for metric, label in (
        ("inference_host_page_faults_total", "Page faults"),
        ("inference_host_major_page_faults_total", "Major faults"),
    ):
        rates = _counter_rate(frame, metric)
        for key, part in _series(rates):
            axes[3].plot(part["elapsed_seconds"], part["rate"], marker="o", label=label + key)
    axes[3].set_ylabel("Falhas/s")

    rows = frame[frame.get("metric") == "vllm:kv_cache_usage_perc"] if not frame.empty else frame
    for key, part in _series(rows):
        axes[4].plot(part["elapsed_seconds"], part["value"] * 100, marker="o", label="KV" + key)
    axes[4].set_ylabel("Pool KV (%)")
    axes[4].set_xlabel("Tempo relativo ao início da requisição (s)")

    for axis in axes:
        if not axis.lines:
            axis.text(
                0.5, 0.5, "métrica não exposta",
                ha="center", va="center", transform=axis.transAxes, color="gray",
            )

    event_map = {event["name"]: event["elapsed_seconds"] for event in profile["events"]}
    first = event_map.get("first_token")
    end = event_map.get("request_end")
    decode_end = event_map.get("last_content", end)
    for axis in axes:
        if first is not None:
            axis.axvspan(0, first, color="#f0ad4e", alpha=0.18, label="TTFT/prefill observado")
            axis.axvspan(first, decode_end, color="#5bc0de", alpha=0.14, label="Decode observado")
            axis.axvline(first, color="#c27a00", linestyle="--", linewidth=1)
            axis.axvline(decode_end, color="#148ca6", linestyle="--", linewidth=1)
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.axvline(end, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    fig.suptitle(
        f"{profile['metadata']['model']} | TTFT={profile['request'].get('time_to_first_token_ms')} ms | "
        f"decode={profile['request'].get('decode_tokens_per_second')} tok/s"
    )
    fig.tight_layout()
    return fig


def _series(frame):
    if frame.empty:
        return []
    label_columns = [column for column in frame if column.startswith("label_")]
    if not label_columns:
        return [("", frame.sort_values("timestamp"))]
    values = []
    for labels, part in frame.groupby(label_columns, dropna=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        suffix = " [" + ", ".join(str(value) for value in labels if str(value) != "nan") + "]"
        values.append((suffix, part.sort_values("timestamp")))
    return values


def default_api_key() -> str:
    return os.environ.get("BENCH_API_KEY", "")
