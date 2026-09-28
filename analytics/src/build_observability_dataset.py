"""Consolida resultados de observabilidade de inferencia para analise offline.

O consolidador nunca consulta o runtime nem o Prometheus. Ele le os artefatos
baixados do Pod, mantem execucoes incompletas apenas no inventario e usa
somente execucoes com ``status=complete`` nas tabelas comparativas.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


MANIFEST_NAMES = ("dataset-manifest.json", "manifest.json")
TABLE_NAMES = (
    "all-requests.csv",
    "request-events.csv",
    "request-phase-metrics.csv",
    "telemetry-by-request.csv",
    "prometheus-samples.csv",
)
PROVENANCE_COLUMNS = (
    "source_run_key",
    "source_run_id",
    "comparison_id",
    "comparison_fingerprint",
    "source_runtime",
    "source_model",
    "source_status",
    "source_run_dir",
    "source_manifest",
    "source_table",
)

REQUIRED_TABLE_COLUMNS = {
    "all-requests.csv": {
        "experiment_id",
        "request_uid",
        "runtime",
        "model",
        "block",
        "benchmark_phase",
        "scenario",
        "status",
        "time_to_first_token_ms",
        "decode_tokens_per_second",
    },
    "request-events.csv": {
        "experiment_id",
        "request_uid",
        "event",
        "timestamp_s",
        "request_elapsed_s",
    },
    "prometheus-samples.csv": {
        "experiment_id",
        "timestamp_s",
        "metric",
        "value",
        "unit",
        "kind",
    },
    "telemetry-by-request.csv": {
        "experiment_id",
        "request_uid",
        "benchmark_phase",
        "observed_phase",
        "metric",
        "value",
    },
    "request-phase-metrics.csv": {
        "experiment_id",
        "request_uid",
        "benchmark_phase",
        "observed_phase",
        "metric",
        "sample_count",
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# A ausencia destas series continua explicita em metric-coverage.csv. O zero e
# usado apenas para a contagem de amostras, nunca como valor da metrica.
EXPECTED_PROMETHEUS_METRICS = (
    "inference_exporter_up",
    "inference_host_cpu_utilization_ratio",
    "inference_host_memory_used_bytes",
    "inference_host_swap_used_bytes",
    "inference_host_disk_read_bytes_total",
    "inference_host_disk_write_bytes_total",
    "inference_host_page_faults_total",
    "inference_host_major_page_faults_total",
    "inference_gpu_scrape_success",
    "inference_gpu_utilization_ratio",
    "inference_gpu_memory_used_bytes",
    "inference_gpu_memory_controller_utilization_ratio",
    "inference_gpu_temperature_celsius",
    "inference_gpu_power_watts",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)

REQUEST_METRICS = {
    "ttft_ms": (
        ("time_to_first_token_ms", 1.0),
        ("ttft_ms", 1.0),
        ("request_first_token_latency_milliseconds", 1.0),
        ("time_to_first_token_seconds", 1000.0),
    ),
    "decode_tokens_s": (
        ("decode_tokens_s", 1.0),
        ("decode_tokens_per_second", 1.0),
        ("tokens_per_second", 1.0),
    ),
    "effective_tokens_s": (
        ("effective_tokens_s", 1.0),
        ("end_to_end_tokens_per_second", 1.0),
    ),
    "e2e_seconds": (
        ("end_to_end_latency_seconds", 1.0),
        ("e2e_seconds", 1.0),
        ("request_latency", 1.0),
    ),
    "generation_seconds": (
        ("generation_time_seconds", 1.0),
        ("generation_seconds", 1.0),
    ),
    "prompt_tokens": (("prompt_tokens", 1.0), ("input_tokens", 1.0)),
    "completion_tokens": (
        ("completion_tokens", 1.0),
        ("output_tokens", 1.0),
    ),
}


@dataclass
class RunArtifact:
    """Uma execucao encontrada no diretorio de resultados."""

    key: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    status: str
    runtime: str | None
    model: str | None
    manifest_error: str | None = None
    dataset_manifest_path: Path | None = None
    final_manifest_path: Path | None = None
    final_manifest: dict[str, Any] | None = None
    tables: dict[str, Path | None] = field(default_factory=dict)
    table_errors: dict[str, str] = field(default_factory=dict)
    parsed_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    integrity_errors: list[str] = field(default_factory=list)
    comparison_identity: dict[str, Any] = field(default_factory=dict)
    comparison_fingerprint: str | None = None
    comparison_id: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    @property
    def included_in_comparison(self) -> bool:
        return self.complete and not self.integrity_errors and bool(self.comparison_id)


def _nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for item in path:
        if not isinstance(value, dict) or item not in value:
            return None
        value = value[item]
    return value


def _first_scalar(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _nested(mapping, path)
        if value is not None and not isinstance(value, (dict, list)):
            return value
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label} invalido: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} invalido: esperado objeto JSON")
        return {}
    return value


def _first_from_manifests(run: RunArtifact, *paths: tuple[str, ...]) -> Any:
    value = _first_scalar(run.manifest, *paths)
    if value is None and run.final_manifest is not None:
        return _first_scalar(run.final_manifest, *paths)
    return value


def _canonical_value(value: Any) -> Any:
    """Normaliza tipos JSON para um fingerprint independente de ordem."""

    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _find_table(
    run_dir: Path,
    filename: str,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    candidates = (
        run_dir / "csv" / filename,
        run_dir / filename,
        run_dir / "tables" / filename,
        run_dir / "data" / filename,
    )
    direct = next((path for path in candidates if path.is_file()), None)
    if direct is not None:
        return direct

    files = (manifest or {}).get("files")
    entries: list[Any] = []
    if isinstance(files, dict):
        if filename in files:
            entries.append(files[filename])
        entries.extend(files.values())
    elif isinstance(files, list):
        entries.extend(files)
    for entry in entries:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if not raw_path or Path(str(raw_path)).name != filename:
            continue
        path = Path(str(raw_path))
        for candidate in (path, run_dir / path, run_dir / path.name):
            if candidate.is_file():
                return candidate
    return None


def discover_runs(results_dir: Path) -> list[RunArtifact]:
    """Encontra runs e calcula integridade/identidade antes da consolidacao."""

    results_dir = Path(results_dir).expanduser().resolve()
    run_dirs: set[Path] = set()
    for name in MANIFEST_NAMES:
        for path in sorted(results_dir.glob(f"**/json/{name}")):
            run_dirs.add(path.parent.parent)

    runs: list[RunArtifact] = []
    for run_dir in sorted(run_dirs, key=str):
        errors: list[str] = []
        dataset_manifest_path = run_dir / "json" / "dataset-manifest.json"
        final_manifest_path = run_dir / "json" / "manifest.json"
        payload = (
            _read_json_object(dataset_manifest_path, "dataset-manifest.json", errors)
            if dataset_manifest_path.is_file()
            else {}
        )
        final_payload = (
            _read_json_object(final_manifest_path, "manifest.json", errors)
            if final_manifest_path.is_file()
            else None
        )
        manifest_path = (
            dataset_manifest_path if dataset_manifest_path.is_file() else final_manifest_path
        )
        declared_status = _first_scalar(payload, ("status",), ("dataset", "status"))
        final_status = (
            _first_scalar(final_payload, ("status",), ("run", "status"))
            if final_payload is not None
            else None
        )
        non_complete = next(
            (
                str(value).lower()
                for value in (final_status, declared_status)
                if value is not None and str(value).lower() != "complete"
            ),
            None,
        )
        status = non_complete or str(final_status or declared_status or "unknown").lower()
        run_id = str(
            _first_scalar(
                payload,
                ("run_id",),
                ("experiment_id",),
                ("execution_id",),
                ("metadata", "run_id"),
            )
            or _first_scalar(
                final_payload or {},
                ("run_id",),
                ("experiment_id",),
                ("execution_id",),
            )
            or run_dir.name
        )
        runtime = _first_scalar(
            payload,
            ("runtime",),
            ("runtime_label",),
            ("config", "runtime"),
            ("metadata", "runtime"),
        )
        if runtime is None:
            runtime = _first_scalar(final_payload or {}, ("runtime",), ("config", "runtime"))
        model = _first_scalar(
            payload,
            ("model",),
            ("model_id",),
            ("config", "model"),
            ("config", "model_id"),
            ("metadata", "model"),
        )
        if model is None:
            model = _first_scalar(final_payload or {}, ("model",), ("config", "model"))
        try:
            relative_key = run_dir.relative_to(results_dir).as_posix()
        except ValueError:
            relative_key = str(run_dir)
        key = relative_key if relative_key not in ("", ".") else run_id
        run = RunArtifact(
            key=key,
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            manifest=payload,
            status=status,
            runtime=str(runtime) if runtime is not None else None,
            model=str(model) if model is not None else None,
            manifest_error="; ".join(errors) if errors else None,
            dataset_manifest_path=(
                dataset_manifest_path if dataset_manifest_path.is_file() else None
            ),
            final_manifest_path=(final_manifest_path if final_manifest_path.is_file() else None),
            final_manifest=final_payload,
            tables={name: _find_table(run_dir, name, payload) for name in TABLE_NAMES},
            integrity_errors=errors,
        )
        _validate_run_integrity(run)
        _derive_comparison_identity(run)
        runs.append(run)
    return runs


def _add_integrity_error(run: RunArtifact, message: str) -> None:
    if message not in run.integrity_errors:
        run.integrity_errors.append(message)


def _manifest_file_records(run: RunArtifact) -> dict[str, tuple[Path, dict[str, Any]]]:
    files = run.manifest.get("files")
    if not isinstance(files, list):
        _add_integrity_error(run, "dataset-manifest.json: campo files ausente ou invalido")
        return {}
    records: dict[str, tuple[Path, dict[str, Any]]] = {}
    root = run.run_dir.resolve()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            _add_integrity_error(run, f"dataset files[{index}]: entrada nao e objeto")
            continue
        raw_path = record.get("path")
        expected_hash = str(record.get("sha256") or "").lower()
        if not isinstance(raw_path, str) or not raw_path.strip():
            _add_integrity_error(run, f"dataset files[{index}]: path ausente ou invalido")
            continue
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            _add_integrity_error(run, f"dataset files[{index}]: caminho fora da run: {raw_path}")
            continue
        filename = candidate.name
        if filename in records:
            _add_integrity_error(run, f"dataset files: artefato duplicado: {filename}")
            continue
        records[filename] = (candidate, record)
        if not candidate.is_file():
            _add_integrity_error(run, f"artefato obrigatorio ausente: {raw_path}")
            continue
        if not _SHA256_RE.fullmatch(expected_hash):
            _add_integrity_error(run, f"SHA-256 ausente ou invalido: {raw_path}")
        elif _sha256(candidate) != expected_hash:
            _add_integrity_error(run, f"SHA-256 divergente: {raw_path}")
        expected_bytes = record.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            _add_integrity_error(run, f"tamanho ausente ou invalido: {raw_path}")
        elif candidate.stat().st_size != expected_bytes:
            _add_integrity_error(run, f"tamanho divergente: {raw_path}")
    return records


def _validate_sha256sums(run: RunArtifact) -> None:
    path = next(
        (
            candidate
            for candidate in (
                run.run_dir / "text" / "SHA256SUMS",
                run.run_dir / "SHA256SUMS",
            )
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        _add_integrity_error(run, "artefato obrigatorio ausente: text/SHA256SUMS")
        return
    entries: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                _add_integrity_error(run, "SHA256SUMS malformado")
                continue
            relative = parts[1].strip().lstrip("*")
            entries[relative] = parts[0].lower()
    except OSError as exc:
        _add_integrity_error(run, f"SHA256SUMS ilegivel: {exc}")
        return
    required = {
        "json/dataset-manifest.json": run.dataset_manifest_path,
        "json/manifest.json": run.final_manifest_path,
    }
    for relative, artifact_path in required.items():
        expected = entries.get(relative)
        if expected is None:
            _add_integrity_error(run, f"SHA256SUMS sem {relative}")
        elif not _SHA256_RE.fullmatch(expected):
            _add_integrity_error(run, f"SHA256SUMS com hash invalido para {relative}")
        elif artifact_path is not None and _sha256(artifact_path) != expected:
            _add_integrity_error(run, f"SHA-256 divergente: {relative}")


def _validate_run_integrity(run: RunArtifact) -> None:
    """Valida o pacote antes que qualquer linha entre no comparativo."""

    if run.dataset_manifest_path is None:
        _add_integrity_error(run, "artefato obrigatorio ausente: json/dataset-manifest.json")
    if run.final_manifest_path is None:
        _add_integrity_error(run, "artefato obrigatorio ausente: json/manifest.json")
    if not run.manifest:
        _add_integrity_error(run, "dataset-manifest.json ausente ou malformado")
    if not run.final_manifest:
        _add_integrity_error(run, "manifest.json ausente ou malformado")

    for label, payload in (
        ("dataset-manifest.json", run.manifest),
        ("manifest.json", run.final_manifest or {}),
    ):
        experiment_id = _first_scalar(payload, ("experiment_id",), ("run_id",))
        if not experiment_id:
            _add_integrity_error(run, f"{label}: experiment_id ausente")
        elif str(experiment_id) != run.run_id:
            _add_integrity_error(run, f"{label}: experiment_id diverge da run")
        declared_status = _first_scalar(payload, ("status",), ("run", "status"))
        if declared_status is None:
            _add_integrity_error(run, f"{label}: status ausente")

    records = _manifest_file_records(run)
    for filename, required_columns in REQUIRED_TABLE_COLUMNS.items():
        record = records.get(filename)
        if record is None:
            _add_integrity_error(run, f"dataset manifest nao inventaria {filename}")
            continue
        table_path = record[0]
        direct_path = run.tables.get(filename)
        if direct_path is not None and direct_path.resolve() != table_path.resolve():
            _add_integrity_error(run, f"caminho ambiguo para {filename}")
        run.tables[filename] = table_path
        if not table_path.is_file():
            continue
        try:
            frame = pd.read_csv(table_path)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            run.table_errors[filename] = message
            _add_integrity_error(run, f"CSV malformado: {filename}: {message}")
            continue
        run.parsed_tables[filename] = frame
        if frame.empty:
            _add_integrity_error(run, f"CSV obrigatorio vazio: {filename}")
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            _add_integrity_error(
                run,
                f"CSV {filename} sem colunas obrigatorias: {', '.join(missing_columns)}",
            )
        if "experiment_id" in frame:
            observed_ids = set(frame["experiment_id"].dropna().astype(str))
            if observed_ids != {run.run_id}:
                _add_integrity_error(run, f"CSV {filename}: experiment_id divergente")

    requests = run.parsed_tables.get("all-requests.csv")
    if requests is not None and {"benchmark_phase", "status"} <= set(requests.columns):
        measured = requests.loc[requests["benchmark_phase"].astype(str) == "measure"]
        successful = measured["status"].astype(str).str.lower().isin(
            {"successful", "success", "ok"}
        )
        if measured.empty or not successful.any():
            _add_integrity_error(run, "all-requests.csv sem medicao bem-sucedida")
    _validate_sha256sums(run)


def _load_measurement_configs(run: RunArtifact) -> list[dict[str, Any]]:
    blocks = (run.final_manifest or {}).get("blocks")
    if not isinstance(blocks, list):
        return []
    normalized: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("phase") != "measure":
            continue
        prefix = block.get("prefix")
        if not isinstance(prefix, str) or Path(prefix).name != prefix:
            continue
        path = run.run_dir / "json" / f"{prefix}-config.json"
        if not path.is_file():
            continue
        errors: list[str] = []
        payload = _read_json_object(path, path.name, errors)
        if errors:
            continue
        spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
        data = spec.get("data") if isinstance(spec.get("data"), list) else []
        sample = data[0] if data and isinstance(data[0], dict) else {}
        backend = spec.get("backend") if isinstance(spec.get("backend"), dict) else {}
        extras = backend.get("extras") if isinstance(backend.get("extras"), dict) else {}
        body = extras.get("body") if isinstance(extras.get("body"), dict) else {}
        seed = spec.get("seed") if isinstance(spec.get("seed"), dict) else {}
        profile = spec.get("profile") if isinstance(spec.get("profile"), dict) else {}
        normalized.append(
            {
                "scenario": block.get("scenario"),
                "repetition": block.get("repetition"),
                "prompt_tokens": sample.get("prompt_tokens"),
                "output_tokens": sample.get("output_tokens"),
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "seed": seed.get("value"),
                "profile": profile.get("kind"),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            str(item.get("scenario")),
            int(item.get("repetition") or 0),
        ),
    )


def _measurement_blocks_from_manifest(final: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = final.get("blocks")
    if not isinstance(blocks, list):
        return []
    normalized = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("phase") != "measure":
            continue
        normalized.append(
            {
                key: block.get(key)
                for key in (
                    "scenario",
                    "repetition",
                    "seed",
                    "expected_requests",
                    "requests_sha256",
                    "mode",
                    "conversations",
                    "conversation_turns",
                    "conversation_artifact",
                )
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            str(item.get("scenario")),
            int(item.get("repetition") or 0),
        ),
    )


def _explicit_hardware(run: RunArtifact) -> dict[str, Any] | None:
    hardware = _nested(run.manifest, ("environment", "hardware"))
    if not isinstance(hardware, dict):
        hardware = _nested(run.final_manifest or {}, ("hardware",))
    return hardware if isinstance(hardware, dict) and hardware else None


def _queried_gpu_inventory(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("returncode") != 0:
        return None
    stdout = payload.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    devices = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) != 5 or not parts[1] or not parts[2]:
            return None
        try:
            index = int(parts[0])
            memory_total_mib = int(parts[3])
        except ValueError:
            return None
        if index < 0 or memory_total_mib <= 0:
            return None
        devices.append(
            {
                "index": index,
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": memory_total_mib,
                "driver_version": parts[4] or None,
            }
        )
    return {"devices": devices} if devices else None


def _structured_gpu_identity(hardware: dict[str, Any]) -> dict[str, Any] | None:
    queried = _queried_gpu_inventory(hardware.get("gpu_inventory"))
    if queried is not None:
        return queried
    # Se houve tentativa explicita e ela falhou, nao use campos de host como
    # evidencia de GPU nem esconda a falha com outro fallback.
    if "gpu_inventory" in hardware:
        return None

    name = hardware.get("gpu_name")
    memory = hardware.get("gpu_memory_total_mib")
    if isinstance(name, str) and name.strip() and type(memory) is int and memory > 0:
        identity = {
            "name": name.strip(),
            "memory_total_mib": memory,
        }
        count = hardware.get("gpu_count")
        if type(count) is int and count > 0:
            identity["count"] = count
        for source, target in (
            ("gpu_uuid", "uuid"),
            ("driver_version", "driver_version"),
            ("cuda_version", "cuda_version"),
        ):
            value = hardware.get(source)
            if isinstance(value, str) and value.strip():
                identity[target] = value.strip()
        return identity
    return None


def _host_identity(run: RunArtifact) -> dict[str, Any] | None:
    hardware = _explicit_hardware(run)
    if hardware is None:
        return None
    gpu_only = {
        "gpu_inventory",
        "gpu_name",
        "gpu_memory_total_mib",
        "gpu_count",
        "gpu_uuid",
        "driver_version",
        "cuda_version",
    }
    host = {key: value for key, value in hardware.items() if key not in gpu_only}
    return _canonical_value(host) if host else None


def _gpu_identity(run: RunArtifact) -> dict[str, Any] | None:
    explicit = _explicit_hardware(run)
    if explicit is not None:
        return _canonical_value(_structured_gpu_identity(explicit))

    path = run.run_dir / "json" / "gpu-before.json"
    if not path.is_file():
        return None
    errors: list[str] = []
    payload = _read_json_object(path, "gpu-before.json", errors)
    stdout = payload.get("stdout") if isinstance(payload.get("stdout"), str) else ""
    if not stdout:
        return None
    name_match = re.search(r"\|\s*\d+\s+(.+?)\s{2,}(?:On|Off)\s*\|", stdout)
    memory_matches = re.findall(r"(\d+)MiB\s*/\s*(\d+)MiB", stdout)
    driver_match = re.search(r"Driver Version:\s*([^\s|]+)", stdout)
    cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", stdout)
    identity: dict[str, Any] = {}
    if name_match:
        identity["gpu_name"] = " ".join(name_match.group(1).split())
    if memory_matches:
        identity["gpu_memory_total_mib"] = int(memory_matches[0][1])
        identity["gpu_count"] = len(memory_matches)
    if driver_match:
        identity["driver_version"] = driver_match.group(1)
    if cuda_match:
        identity["cuda_version"] = cuda_match.group(1)
    return identity or None


def _derive_comparison_identity(run: RunArtifact) -> None:
    model_identity = _nested(run.manifest, ("model_identity",))
    if not isinstance(model_identity, dict):
        model_identity = _nested(run.final_manifest or {}, ("model_identity",))
    if not isinstance(model_identity, dict):
        model_identity = {}
    tokenizer = _nested(run.manifest, ("tokenizer",))
    if not isinstance(tokenizer, dict):
        tokenizer = _nested(run.final_manifest or {}, ("tokenizer",))
    if not isinstance(tokenizer, dict):
        tokenizer = {}

    benchmark = _nested(run.manifest, ("benchmark",))
    if not isinstance(benchmark, dict):
        benchmark = {}
    final = run.final_manifest or {}
    measure_configs = _load_measurement_configs(run)
    generation_parameters = benchmark.get("generation_parameters")
    generation_source = "manifest"
    if not isinstance(generation_parameters, dict):
        generation_parameters = final.get("generation_parameters")
    if not isinstance(generation_parameters, dict):
        generation_source = "measure-config-fallback-unhashed"
        temperatures = {item.get("temperature") for item in measure_configs}
        top_ps = {item.get("top_p") for item in measure_configs}
        output_limits = {item.get("output_tokens") for item in measure_configs}
        generation_parameters = {
            "temperature": next(iter(temperatures)) if len(temperatures) == 1 else None,
            "top_p": next(iter(top_ps)) if len(top_ps) == 1 else None,
            "max_output_tokens": next(iter(output_limits)) if len(output_limits) == 1 else None,
            "stream": True,
        }
    platform_value = _first_from_manifests(
        run,
        ("environment", "platform"),
        ("platform",),
    )
    workload = {
        "benchmark_profile": benchmark.get(
            "benchmark_profile", final.get("benchmark_profile", "generic")
        ),
        "smoke": benchmark.get("smoke", final.get("smoke")),
        "requests": benchmark.get("requests", final.get("requests")),
        "repetitions": benchmark.get("repetitions", final.get("repetitions")),
        "warmup_requests_per_case": benchmark.get(
            "warmup_requests_per_case", final.get("warmup_requests_per_case")
        ),
        "scenarios": benchmark.get("scenarios", final.get("scenarios")),
        "input_tokens": benchmark.get("input_tokens", final.get("input_tokens")),
        "output_tokens": benchmark.get("output_tokens", final.get("output_tokens")),
        "mode": benchmark.get("mode", final.get("mode")),
        "conversation_turns": benchmark.get(
            "conversation_turns", final.get("conversation_turns")
        ),
        "seed": benchmark.get("seed", final.get("seed")),
        "profile": benchmark.get("profile", final.get("profile")),
        "workload_contract": benchmark.get("workload", final.get("workload")),
        "request_dataset": benchmark.get("request_dataset", final.get("request_dataset")),
        "conversation_fixture_sha256": _first_scalar(
            benchmark,
            ("conversation_fixture", "sha256"),
        )
        or _nested(final, ("conversation_fixture", "sha256")),
        "warmup_conversation_fixture_sha256": _first_scalar(
            benchmark,
            ("warmup_conversation_fixture", "sha256"),
        )
        or _nested(final, ("warmup_conversation_fixture", "sha256")),
        "measurement_blocks": _measurement_blocks_from_manifest(final),
    }
    if workload["output_tokens"] is None:
        workload["output_tokens"] = generation_parameters.get("max_output_tokens")
    if workload["output_tokens"] is None:
        observed_output_limits = sorted(
            {
                item["output_tokens"]
                for item in measure_configs
                if item.get("output_tokens") is not None
            }
        )
        workload["output_tokens"] = (
            observed_output_limits[0] if len(observed_output_limits) == 1 else observed_output_limits
        )
    config = _nested(run.manifest, ("config",))
    if not isinstance(config, dict):
        config = _nested(final, ("config",))
    if not isinstance(config, dict):
        config = {}
    hardware = {
        "platform": platform_value,
        "host": _host_identity(run),
        "gpu": _gpu_identity(run),
    }
    identity = {
        "schema_version": 1,
        "model": {
            "sha256": model_identity.get("sha256"),
        },
        "tokenizer": {"sha256": tokenizer.get("sha256")},
        "config": {
            "context_window": config.get("context_window"),
            "profile": final.get("profile"),
            "request_mode": workload["mode"],
            "sequential_requests": True,
            "generation_parameters": generation_parameters,
            "generation_parameters_source": generation_source,
        },
        "workload": workload,
        "hardware": hardware,
    }
    missing: list[str] = []
    if not _SHA256_RE.fullmatch(str(identity["model"]["sha256"] or "").lower()):
        missing.append("model.sha256")
    if not _SHA256_RE.fullmatch(str(identity["tokenizer"]["sha256"] or "").lower()):
        missing.append("tokenizer.sha256")
    if not isinstance(identity["config"]["context_window"], int):
        missing.append("config.context_window")
    if not workload["scenarios"]:
        missing.append("workload.scenarios")
    if workload["output_tokens"] is None or workload["output_tokens"] == []:
        missing.append("workload.output_tokens")
    if not hardware["platform"]:
        missing.append("hardware.platform")
    if not hardware["host"]:
        missing.append("hardware.host")
    if not hardware["gpu"]:
        missing.append("hardware.gpu")
    identity["identity_missing"] = missing
    # Sem identidade suficiente, o run continua inventariado mas recebe escopo
    # proprio: dois desconhecidos nunca sao tratados como comparaveis.
    if missing:
        identity["isolation_key"] = run.key
    run.comparison_identity = _canonical_value(identity)
    run.comparison_fingerprint = _fingerprint(run.comparison_identity)
    run.comparison_id = run.comparison_fingerprint[:16]


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PROVENANCE_COLUMNS))


def _annotate(frame: pd.DataFrame, run: RunArtifact, table_path: Path) -> pd.DataFrame:
    frame = frame.copy()
    values = {
        "source_run_key": run.key,
        "source_run_id": run.run_id,
        "comparison_id": run.comparison_id,
        "comparison_fingerprint": run.comparison_fingerprint,
        "source_runtime": run.runtime,
        "source_model": run.model,
        "source_status": run.status,
        "source_run_dir": str(run.run_dir),
        "source_manifest": str(run.manifest_path),
        "source_table": str(table_path),
    }
    for column, value in values.items():
        frame[column] = value
    for column, value in (
        ("run_id", run.run_id),
        ("experiment_id", run.run_id),
        ("runtime", run.runtime),
        ("model", run.model),
        ("run_status", run.status),
    ):
        if column not in frame:
            frame[column] = value
        else:
            frame[column] = frame[column].where(frame[column].notna(), value)
    ordered = [column for column in PROVENANCE_COLUMNS if column in frame]
    ordered.extend(column for column in frame if column not in ordered)
    return frame[ordered]


def collect_table(runs: Iterable[RunArtifact], filename: str) -> pd.DataFrame:
    """Concatena uma tabela somente para runs completos."""

    frames: list[pd.DataFrame] = []
    for run in runs:
        if not run.included_in_comparison:
            continue
        table_path = run.tables.get(filename)
        if table_path is None:
            continue
        frame = run.parsed_tables.get(filename)
        if frame is None:
            continue
        frames.append(_annotate(frame, run, table_path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_table()


def _manifest_value(run: RunArtifact, *paths: tuple[str, ...]) -> Any:
    value = _first_scalar(run.manifest, *paths)
    if value is None and run.final_manifest is not None:
        return _first_scalar(run.final_manifest, *paths)
    return value


def _runtime_metadata(run: RunArtifact) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_process = _nested(run.manifest, ("runtime_process",))
    if not isinstance(runtime_process, dict):
        runtime_process = _nested(run.final_manifest or {}, ("runtime_process",))
    if not isinstance(runtime_process, dict):
        runtime_process = {}
    cache_details = _nested(run.manifest, ("cache_policy_details",))
    if not isinstance(cache_details, dict):
        cache_details = _nested(run.final_manifest or {}, ("cache_policy_details",))
    if not isinstance(cache_details, dict):
        cache_details = {}
    return runtime_process, cache_details


def _cache_flags_with_values(argv: Any) -> list[dict[str, Any]]:
    """Extrai flags efetivas sem perder o valor que vem no argumento seguinte."""

    if not isinstance(argv, list):
        return []
    arguments = [str(value) for value in argv]
    records: list[dict[str, Any]] = []
    markers = ("cach", "keepalive", "keep-alive")
    cache_controls = {
        "-ctk",
        "-ctv",
        "--block-size",
        "--cpu-offload-gb",
        "--gpu-memory-utilization",
        "--num-gpu-blocks-override",
        "--swap-space",
    }
    for index, argument in enumerate(arguments):
        lowered = argument.lower()
        normalized_flag = lowered.split("=", 1)[0]
        if (
            not any(marker in lowered for marker in markers)
            and normalized_flag not in cache_controls
        ):
            continue
        if not argument.startswith("-") and "=" not in argument:
            continue
        if "=" in argument:
            flag, value = argument.split("=", 1)
        else:
            flag, value = argument, True
            boolean_flag = normalized_flag.startswith(
                ("--enable-", "--disable-", "--no-")
            )
            if (
                not boolean_flag
                and index + 1 < len(arguments)
                and not arguments[index + 1].startswith("-")
            ):
                value = arguments[index + 1]
        record = {"flag": flag, "value": value}
        if record not in records:
            records.append(record)
    return records


def _runtime_treatment_fields(run: RunArtifact) -> dict[str, Any]:
    runtime_process, cache_details = _runtime_metadata(run)
    argv = runtime_process.get("argv")
    normalized_argv = [str(value) for value in argv] if isinstance(argv, list) else None
    runtime_environment = _nested(run.manifest, ("environment", "runtime_environment"))
    if not isinstance(runtime_environment, dict):
        runtime_environment = _nested(run.final_manifest or {}, ("runtime_environment",))
    if not isinstance(runtime_environment, dict):
        runtime_environment = cache_details.get("runtime_environment")
    if not isinstance(runtime_environment, dict):
        runtime_environment = {}
    runtime_environment = _canonical_value(runtime_environment)
    prefix_cache = cache_details.get("prefix_cache")
    if not isinstance(prefix_cache, dict):
        prefix_cache = {}
    kv_cache = cache_details.get("kv_cache")
    if not isinstance(kv_cache, dict):
        kv_cache = {}
    cache_flags = _cache_flags_with_values(normalized_argv)
    effective_cache_argv = cache_details.get("effective_cache_argv")
    if not isinstance(effective_cache_argv, list) and normalized_argv is not None:
        effective_cache_argv = []
        for record in cache_flags:
            effective_cache_argv.append(record["flag"])
            if record["value"] is not True:
                effective_cache_argv.append(str(record["value"]))
    prefix_cache_flags = [
        record
        for record in cache_flags
        if "prefix" in record["flag"].lower()
        or "cache-reuse" in record["flag"].lower()
    ]
    cache_environment = {
        key: value
        for key, value in runtime_environment.items()
        if any(
            marker in key.upper()
            for marker in (
                "CACHE",
                "KEEP_ALIVE",
                "FLASH_ATTENTION",
                "MAX_LOADED_MODELS",
                "NUM_PARALLEL",
            )
        )
    }
    model_residency = cache_details.get("model_residency")
    model_residency_state = (
        model_residency.get("state")
        if isinstance(model_residency, dict)
        else model_residency
    )
    return {
        "runtime_process_ownership": runtime_process.get("ownership"),
        "runtime_process_argv_json": json.dumps(
            normalized_argv,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if normalized_argv is not None
        else None,
        "runtime_process_argv_fingerprint": _fingerprint({"argv": normalized_argv})
        if normalized_argv is not None
        else None,
        "runtime_environment_json": json.dumps(
            runtime_environment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if runtime_environment
        else None,
        "runtime_environment_fingerprint": _fingerprint(runtime_environment)
        if runtime_environment
        else None,
        "effective_cache_flags_json": json.dumps(
            cache_flags,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if normalized_argv is not None
        else None,
        "effective_cache_argv_json": json.dumps(
            effective_cache_argv,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if isinstance(effective_cache_argv, list)
        else None,
        "effective_cache_environment_json": json.dumps(
            cache_environment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if cache_environment
        else None,
        "prefix_cache_state": prefix_cache.get("state"),
        "prefix_cache_effective_flags_json": json.dumps(
            prefix_cache_flags,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if normalized_argv is not None
        else None,
        "kv_cache_state": kv_cache.get("state"),
        "kv_cache_configuration_json": json.dumps(
            kv_cache.get("configuration"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if isinstance(kv_cache.get("configuration"), dict)
        else None,
        "model_residency": model_residency_state,
        "model_residency_json": json.dumps(
            model_residency,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if isinstance(model_residency, dict)
        else None,
        "runtime_process_json": json.dumps(
            runtime_process, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if runtime_process
        else None,
        "runtime_cache_policy_details_json": json.dumps(
            cache_details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if cache_details
        else None,
    }


def _compatibility_reason(
    run: RunArtifact,
    identity_missing: list[str],
    treatment: dict[str, Any],
) -> str:
    if not run.included_in_comparison:
        return "excluded: status or integrity contract failed"
    if identity_missing:
        return (
            "run isolated because comparison identity is incomplete: "
            + ", ".join(identity_missing)
        )
    if treatment["runtime_process_argv_fingerprint"] is None:
        return (
            "eligible control fingerprint; effective runtime argv is unavailable, so launch "
            "and cache equivalence for the existing server is not established"
        )
    return (
        "eligible control fingerprint; runtime, runtime_version, effective argv/environment, "
        "cache flags, KV/prefix-cache state and model residency remain treatment dimensions"
    )


def run_inventory(runs: Iterable[RunArtifact]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        config = _nested(run.manifest, ("config",))
        if not isinstance(config, dict):
            config = _nested(run.final_manifest or {}, ("config",))
        if not isinstance(config, dict):
            config = {}
        model_identity = _nested(run.manifest, ("model_identity",))
        if not isinstance(model_identity, dict):
            model_identity = _nested(run.final_manifest or {}, ("model_identity",))
        identity_missing = run.comparison_identity.get("identity_missing", [])
        comparison_model = run.comparison_identity.get("model", {})
        comparison_tokenizer = run.comparison_identity.get("tokenizer", {})
        comparison_config = run.comparison_identity.get("config", {})
        comparison_workload = run.comparison_identity.get("workload", {})
        comparison_hardware = run.comparison_identity.get("hardware", {})
        treatment = _runtime_treatment_fields(run)
        compatibility_reason = _compatibility_reason(run, identity_missing, treatment)
        row: dict[str, Any] = {
            "source_run_key": run.key,
            "experiment_id": run.run_id,
            "run_id": run.run_id,
            "status": run.status,
            "included_in_comparison": run.included_in_comparison,
            "integrity_ok": not run.integrity_errors,
            "integrity_errors": json.dumps(
                run.integrity_errors, ensure_ascii=False, sort_keys=True
            )
            if run.integrity_errors
            else None,
            "comparison_id": run.comparison_id,
            "comparison_fingerprint": run.comparison_fingerprint,
            "comparison_identity_complete": not bool(
                run.comparison_identity.get("identity_missing")
            ),
            "comparison_identity_missing": json.dumps(
                run.comparison_identity.get("identity_missing", []),
                ensure_ascii=False,
            ),
            "comparison_identity_json": json.dumps(
                run.comparison_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "model_sha256": comparison_model.get("sha256"),
            "tokenizer_sha256": comparison_tokenizer.get("sha256"),
            "context_window": comparison_config.get("context_window"),
            "output_tokens": comparison_workload.get("output_tokens"),
            "generation_parameters_json": json.dumps(
                comparison_config.get("generation_parameters"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if comparison_config.get("generation_parameters") is not None
            else None,
            "hardware_json": json.dumps(
                comparison_hardware,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if comparison_hardware
            else None,
            "compatibility_reason": compatibility_reason,
            "runtime_version": config.get("runtime_version"),
            "runtime_cache_policy": config.get("cache_policy"),
            **treatment,
            "model_identity_json": json.dumps(
                model_identity, ensure_ascii=False, sort_keys=True
            )
            if isinstance(model_identity, dict)
            else None,
            "runtime": run.runtime,
            "model": run.model,
            "model_revision": _manifest_value(
                run,
                ("model_identity", "resolved_revision"),
                ("model_revision",),
                ("revision",),
                ("config", "model_revision"),
                ("config", "revision"),
            ),
            "started_utc": _manifest_value(
                run, ("started_utc",), ("started_at",), ("metadata", "started_utc")
            ),
            "ended_utc": _manifest_value(
                run, ("ended_utc",), ("ended_at",), ("metadata", "ended_utc")
            ),
            "run_dir": str(run.run_dir),
            "manifest_path": str(run.manifest_path),
            "manifest_sha256": _sha256(run.manifest_path)
            if run.manifest_path.is_file()
            else None,
            "dataset_manifest_path": str(run.dataset_manifest_path)
            if run.dataset_manifest_path
            else None,
            "final_manifest_path": str(run.final_manifest_path) if run.final_manifest_path else None,
            "final_manifest_sha256": _sha256(run.final_manifest_path)
            if run.final_manifest_path
            else None,
            "manifest_error": run.manifest_error,
            "table_errors": json.dumps(run.table_errors, ensure_ascii=False, sort_keys=True)
            if run.table_errors
            else None,
            "exclusion_reasons": json.dumps(
                ([f"status={run.status}"] if not run.complete else [])
                + run.integrity_errors,
                ensure_ascii=False,
            )
            if not run.included_in_comparison
            else None,
        }
        for filename in TABLE_NAMES:
            stem = filename.removesuffix(".csv").replace("-", "_")
            path = run.tables.get(filename)
            row[f"has_{stem}"] = path is not None
            row[f"path_{stem}"] = str(path) if path else None
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_groups(runs: Iterable[RunArtifact]) -> pd.DataFrame:
    grouped: dict[str, list[RunArtifact]] = {}
    for run in runs:
        if run.included_in_comparison and run.comparison_id:
            grouped.setdefault(run.comparison_id, []).append(run)
    rows = []
    for comparison_id, members in sorted(grouped.items()):
        identity = members[0].comparison_identity
        model_identity = identity.get("model", {})
        tokenizer_identity = identity.get("tokenizer", {})
        config_identity = identity.get("config", {})
        workload_identity = identity.get("workload", {})
        hardware_identity = identity.get("hardware", {})
        runtimes = sorted({run.runtime for run in members if run.runtime})
        models = sorted({run.model for run in members if run.model})
        rows.append(
            {
                "comparison_id": comparison_id,
                "comparison_fingerprint": members[0].comparison_fingerprint,
                "run_count": len(members),
                "runtime_count": len(runtimes),
                "runtimes": json.dumps(runtimes, ensure_ascii=False),
                "models": json.dumps(models, ensure_ascii=False),
                "cross_runtime_ready": len(runtimes) >= 2,
                "model_sha256": model_identity.get("sha256"),
                "tokenizer_sha256": tokenizer_identity.get("sha256"),
                "context_window": config_identity.get("context_window"),
                "output_tokens": workload_identity.get("output_tokens"),
                "generation_parameters_json": json.dumps(
                    config_identity.get("generation_parameters"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "hardware_json": json.dumps(
                    hardware_identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "comparison_identity_complete": not bool(identity.get("identity_missing")),
                "comparison_identity_missing": json.dumps(
                    identity.get("identity_missing", []), ensure_ascii=False
                ),
                "comparison_identity_json": json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison_id",
            "comparison_fingerprint",
            "run_count",
            "runtime_count",
            "runtimes",
            "models",
            "cross_runtime_ready",
            "model_sha256",
            "tokenizer_sha256",
            "context_window",
            "output_tokens",
            "generation_parameters_json",
            "hardware_json",
            "comparison_identity_complete",
            "comparison_identity_missing",
            "comparison_identity_json",
        ],
    )


def _successful_requests(frame: pd.DataFrame) -> tuple[pd.DataFrame, int | None, int | None, int | None]:
    if frame.empty or "status" not in frame:
        return frame, None, None, None
    status = frame["status"].astype("string").str.lower()
    successful = status.isin({"successful", "success", "ok", "complete", "completed"})
    failed = status.isin({"errored", "error", "failed", "incomplete", "timeout"})
    other = ~(successful | failed)
    return frame.loc[successful], int(successful.sum()), int(failed.sum()), int(other.sum())


def _metric_values(frame: pd.DataFrame, aliases: tuple[tuple[str, float], ...]) -> tuple[pd.Series, str | None]:
    for column, factor in aliases:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce") * factor
        if values.notna().any():
            return values, column
    return pd.Series(dtype="float64"), None


def experiment_summary(runs: Iterable[RunArtifact], requests: pd.DataFrame) -> pd.DataFrame:
    """Resume distribuicoes por bloco/cenario; estatistica ausente fica NaN."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.included_in_comparison:
            continue
        config = _nested(run.manifest, ("config",))
        if not isinstance(config, dict):
            config = _nested(run.final_manifest or {}, ("config",))
        if not isinstance(config, dict):
            config = {}
        treatment = _runtime_treatment_fields(run)
        compatibility_reason = _compatibility_reason(
            run,
            run.comparison_identity.get("identity_missing", []),
            treatment,
        )
        if "source_run_key" in requests:
            selected = requests.loc[requests["source_run_key"] == run.key]
        else:
            selected = requests.iloc[0:0]
        phase_column = "benchmark_phase" if "benchmark_phase" in selected else "phase"
        grouping = [
            column
            for column in ("block", phase_column, "scenario")
            if column in selected
        ]
        groups = (
            selected.groupby(grouping, dropna=False, sort=False)
            if grouping and not selected.empty
            else [((), selected)]
        )
        for labels, group in groups:
            labels = labels if isinstance(labels, tuple) else (labels,)
            dimensions = dict(zip(grouping, labels))
            measured, successful, failed, other = _successful_requests(group)
            if successful is None:
                measured = group
            row: dict[str, Any] = {
                "source_run_key": run.key,
                "comparison_id": run.comparison_id,
                "comparison_fingerprint": run.comparison_fingerprint,
                "experiment_id": run.run_id,
                "run_id": run.run_id,
                "runtime": run.runtime,
                "runtime_version": config.get("runtime_version"),
                "model": run.model,
                "compatibility_reason": compatibility_reason,
                **treatment,
                **dimensions,
                "phase": dimensions.get(phase_column),
                "request_rows": len(group),
                "successful_request_rows": successful,
                "failed_request_rows": failed,
                "unknown_status_request_rows": other,
            }
            for metric, aliases in REQUEST_METRICS.items():
                values, source = _metric_values(measured, aliases)
                clean = values.dropna()
                row[f"{metric}_source_column"] = source
                row[f"{metric}_sample_count"] = int(clean.size)
                row[f"{metric}_mean"] = clean.mean() if not clean.empty else None
                row[f"{metric}_p50"] = clean.quantile(0.50) if not clean.empty else None
                row[f"{metric}_p95"] = clean.quantile(0.95) if not clean.empty else None
                row[f"{metric}_min"] = clean.min() if not clean.empty else None
                row[f"{metric}_max"] = clean.max() if not clean.empty else None
            rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "source_run_key",
            "comparison_id",
            "comparison_fingerprint",
            "experiment_id",
            "run_id",
            "runtime",
            "runtime_version",
            "model",
            "compatibility_reason",
            "runtime_process_ownership",
            "runtime_process_argv_json",
            "runtime_process_argv_fingerprint",
            "runtime_environment_json",
            "runtime_environment_fingerprint",
            "effective_cache_flags_json",
            "effective_cache_argv_json",
            "effective_cache_environment_json",
            "prefix_cache_state",
            "prefix_cache_effective_flags_json",
            "kv_cache_state",
            "kv_cache_configuration_json",
            "model_residency",
            "model_residency_json",
            "block",
            "phase",
            "scenario",
            "request_rows",
        ]
    )


def _long_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    metric_column = next(
        (column for column in ("metric", "metric_name", "__name__") if column in frame),
        None,
    )
    value_column = next(
        (
            column
            for column in ("value", "mean", "value_mean", "avg", "average")
            if column in frame
        ),
        None,
    )
    return metric_column, value_column


def _observed_metric_names(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    metric_column, _value_column = _long_columns(frame)
    if metric_column:
        return set(frame[metric_column].dropna().astype(str))
    ignored = set(PROVENANCE_COLUMNS) | {
        "run_id",
        "experiment_id",
        "runtime",
        "model",
        "run_status",
        "request_id",
        "request_uid",
        "block",
        "benchmark_phase",
        "scenario",
        "repetition",
        "observed_phase",
        "phase",
        "timestamp",
        "timestamp_s",
        "timestamp_utc",
        "timestamp_seconds",
        "elapsed_seconds",
        "run_elapsed_s",
        "request_elapsed_s",
        "unit",
        "kind",
        "labels_json",
        "job",
        "instance",
        "sample_count",
    }
    names = set()
    for column in frame.columns:
        if column in ignored:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            names.add(column)
    return names


def _coverage_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    metric_column, value_column = _long_columns(frame)
    if metric_column:
        if value_column is None:
            return pd.Series(dtype="float64")
        selected = frame.loc[frame[metric_column].astype("string") == metric, value_column]
        return pd.to_numeric(selected, errors="coerce")
    if metric in frame:
        return pd.to_numeric(frame[metric], errors="coerce")
    return pd.Series(dtype="float64")


def metric_coverage(
    runs: Iterable[RunArtifact],
    prometheus_samples: pd.DataFrame,
    telemetry_by_request: pd.DataFrame,
) -> pd.DataFrame:
    """Explicita cobertura por run sem imputar zero a series ausentes."""

    runs = [run for run in runs if run.included_in_comparison]
    expected = set(EXPECTED_PROMETHEUS_METRICS)
    for run in runs:
        observed = _nested(run.manifest, ("prometheus", "metrics_observed"))
        if isinstance(observed, list):
            expected.update(str(metric) for metric in observed if metric)
    expected.update(_observed_metric_names(prometheus_samples))
    expected.update(_observed_metric_names(telemetry_by_request))
    rows: list[dict[str, Any]] = []
    for run in runs:
        raw = (
            prometheus_samples.loc[prometheus_samples["source_run_key"] == run.key]
            if "source_run_key" in prometheus_samples
            else prometheus_samples.iloc[0:0]
        )
        aggregated = (
            telemetry_by_request.loc[telemetry_by_request["source_run_key"] == run.key]
            if "source_run_key" in telemetry_by_request
            else telemetry_by_request.iloc[0:0]
        )
        observed_raw = _observed_metric_names(raw)
        observed_aggregated = _observed_metric_names(aggregated)
        observed = observed_raw | observed_aggregated
        for metric in sorted(expected):
            raw_values = _coverage_values(raw, metric).dropna()
            aggregated_values = _coverage_values(aggregated, metric).dropna()
            if not raw_values.empty:
                values = raw_values
                source_name = "prometheus-samples.csv"
            else:
                values = aggregated_values
                source_name = "telemetry-by-request.csv" if not values.empty else None
            rows.append(
                {
                    "source_run_key": run.key,
                    "comparison_id": run.comparison_id,
                    "comparison_fingerprint": run.comparison_fingerprint,
                    "run_id": run.run_id,
                    "runtime": run.runtime,
                    "model": run.model,
                    "metric": metric,
                    "available": bool(not values.empty),
                    "sample_count": int(values.size),
                    "mean": values.mean() if not values.empty else None,
                    "min": values.min() if not values.empty else None,
                    "max": values.max() if not values.empty else None,
                    "table_source": source_name,
                    "missing_reason": None
                    if not values.empty
                    else ("metric_not_collected" if metric not in observed else "values_are_null"),
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "source_run_key",
            "comparison_id",
            "comparison_fingerprint",
            "run_id",
            "runtime",
            "model",
            "metric",
            "available",
            "sample_count",
            "mean",
            "min",
            "max",
            "table_source",
            "missing_reason",
        ]
    )


def _parquet_engine() -> str | None:
    for engine in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(engine) is not None:
            return engine
    return None


def _write_frame(
    frame: pd.DataFrame,
    output_dir: Path,
    filename: str,
    *,
    write_parquet: bool,
    warnings: list[str],
) -> dict[str, Any]:
    path = output_dir / filename
    frame.to_csv(path, index=False, encoding="utf-8")
    result: dict[str, Any] = {"csv": str(path), "rows": len(frame)}
    if not write_parquet:
        return result
    engine = _parquet_engine()
    if engine is None:
        warnings.append(f"Parquet ignorado para {filename}: pyarrow/fastparquet indisponivel.")
        return result
    parquet_path = path.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False, engine=engine)
    except Exception as exc:  # CSV e o contrato obrigatorio; Parquet nunca bloqueia
        warnings.append(f"Parquet ignorado para {filename}: {type(exc).__name__}: {exc}")
    else:
        result["parquet"] = str(parquet_path)
    return result


def build_observability_dataset(
    results_dir: Path,
    output_dir: Path,
    *,
    write_parquet: bool = False,
) -> dict[str, Any]:
    """Descobre, consolida e grava o dataset portavel de observabilidade."""

    results_dir = Path(results_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    runs = discover_runs(results_dir)
    if not runs:
        raise ValueError(
            f"Nenhum json/dataset-manifest.json ou json/manifest.json encontrado em {results_dir}."
        )

    requests = collect_table(runs, "all-requests.csv")
    request_events = collect_table(runs, "request-events.csv")
    phase_metrics = collect_table(runs, "request-phase-metrics.csv")
    telemetry_by_request = collect_table(runs, "telemetry-by-request.csv")
    prometheus_samples = collect_table(runs, "prometheus-samples.csv")
    inventory = run_inventory(runs)
    complete_runs = inventory.loc[inventory["included_in_comparison"]].copy()
    groups = comparison_groups(runs)
    summary = experiment_summary(runs, requests)
    coverage = metric_coverage(runs, prometheus_samples, telemetry_by_request)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "run-inventory.csv": inventory,
        "complete-runs.csv": complete_runs,
        "comparison-groups.csv": groups,
        "all-requests.csv": requests,
        "request-events.csv": request_events,
        "request-phase-metrics.csv": phase_metrics,
        "telemetry-by-request.csv": telemetry_by_request,
        "prometheus-samples.csv": prometheus_samples,
        "experiment-summary.csv": summary,
        "metric-coverage.csv": coverage,
    }
    warnings: list[str] = []
    files = {
        filename: _write_frame(
            frame,
            output_dir,
            filename,
            write_parquet=write_parquet,
            warnings=warnings,
        )
        for filename, frame in frames.items()
    }
    status_counts = inventory["status"].value_counts(dropna=False).to_dict()
    comparable_count = int(inventory["included_in_comparison"].sum())
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "runs_discovered": len(runs),
        "runs_complete": int((inventory["status"] == "complete").sum()),
        "runs_comparable": comparable_count,
        "runs_excluded": int(len(inventory) - comparable_count),
        "comparison_groups": len(groups),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "comparison_policy": (
            "Only runs with final status complete, valid mandatory artifacts and matching "
            "SHA-256 enter comparison tables; comparison_id excludes runtime/runtime_version."
        ),
        "runtime_treatment_policy": (
            "Effective redacted runtime argv/environment and their fingerprints, cache flags "
            "with values, KV/prefix-cache state and model residency are reported as treatment "
            "dimensions; generic config.server_command is not used as process evidence."
        ),
        "integrity_policy": (
            "json/manifest.json, json/dataset-manifest.json, text/SHA256SUMS and all five "
            "CSV artifacts are mandatory and validated before consolidation."
        ),
        "missing_metric_policy": "Missing measurements remain null; they are never imputed as zero.",
        "files": files,
        "warnings": warnings,
    }
    manifest_path = output_dir / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["dataset_manifest"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Tambem tenta gravar Parquet; ausencia do engine nao falha o processo.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = build_observability_dataset(
            args.results_dir,
            args.output_dir,
            write_parquet=args.parquet,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Dataset: {manifest['dataset_manifest']} | "
        f"comparable={manifest['runs_comparable']} excluded={manifest['runs_excluded']}"
    )
    for warning in manifest["warnings"]:
        print(f"AVISO: {warning}")


if __name__ == "__main__":
    main()
