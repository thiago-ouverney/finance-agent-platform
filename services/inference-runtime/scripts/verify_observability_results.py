#!/usr/bin/env python3
"""Verifica hashes dos datasets de observabilidade depois da transferência."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
REQUIRED_DATASET_PATHS = frozenset({
    "csv/all-requests.csv",
    "csv/request-events.csv",
    "csv/prometheus-samples.csv",
    "csv/telemetry-by-request.csv",
    "csv/request-phase-metrics.csv",
    "json/dataset-manifest.json",
    "json/manifest.json",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_candidate(root: Path, raw_path: str) -> Path | None:
    if not raw_path or Path(raw_path).is_absolute() or "\x00" in raw_path:
        return None
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def read_external_checksums(root: Path) -> tuple[dict[str, str], list[str]]:
    """Lê e valida o inventário externo antes de confiar no manifesto JSON."""
    checksum_path = root / "text" / "SHA256SUMS"
    if not checksum_path.is_file():
        return {}, [f"{checksum_path}: arquivo estrutural ausente"]
    records: dict[str, str] = {}
    errors = []
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256.fullmatch(parts[0]):
            errors.append(f"{checksum_path}:{line_number}: linha SHA-256 inválida")
            continue
        digest, raw_path = parts[0].lower(), parts[1].lstrip("*")
        candidate = _safe_candidate(root, raw_path)
        if candidate is None:
            errors.append(f"{checksum_path}:{line_number}: caminho fora da execução: {raw_path}")
            continue
        if raw_path in records:
            errors.append(f"{checksum_path}:{line_number}: caminho duplicado: {raw_path}")
            continue
        records[raw_path] = digest
        if not candidate.is_file():
            errors.append(f"{checksum_path}: arquivo ausente: {raw_path}")
        elif sha256(candidate) != digest:
            errors.append(f"{checksum_path}: SHA-256 divergente: {raw_path}")
    missing = sorted(REQUIRED_DATASET_PATHS - records.keys())
    errors.extend(f"{checksum_path}: entrada estrutural ausente: {item}" for item in missing)
    return records, errors


def verify_manifest(path: Path) -> list[str]:
    root = path.parent.parent if path.parent.name == "json" else path.parent
    external, errors = read_external_checksums(root)
    # Não interpreta um manifesto cuja própria integridade ainda não foi
    # estabelecida pelo inventário externo.
    if errors:
        return errors

    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        return [f"{path}: campo files ausente ou inválido"]
    internal_paths = set()
    for record in files:
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            errors.append(f"{path}: entrada de arquivo inválida: {record!r}")
            continue
        raw_path = record["path"]
        digest = record["sha256"]
        if not isinstance(raw_path, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            errors.append(f"{path}: entrada de arquivo inválida: {record!r}")
            continue
        candidate = _safe_candidate(root, raw_path)
        if candidate is None:
            errors.append(f"{path}: caminho fora da execução: {record['path']}")
            continue
        if raw_path in internal_paths:
            errors.append(f"{path}: caminho duplicado no inventário interno: {raw_path}")
            continue
        internal_paths.add(raw_path)
        if not candidate.is_file():
            errors.append(f"{path}: arquivo ausente: {record['path']}")
            continue
        observed = sha256(candidate)
        if observed != digest.lower():
            errors.append(f"{path}: SHA-256 divergente: {record['path']}")
        external_digest = external.get(raw_path)
        if external_digest is None:
            errors.append(f"{path}: arquivo ausente do SHA256SUMS externo: {raw_path}")
        elif external_digest != digest.lower():
            errors.append(f"{path}: hash interno diverge do SHA256SUMS: {raw_path}")
    # O manifesto final é protegido pelo inventário externo porque só é fechado
    # depois da exportação; os cinco CSVs pertencem também ao inventário interno.
    required_internal = REQUIRED_DATASET_PATHS - {
        "json/dataset-manifest.json", "json/manifest.json"
    }
    missing_internal = sorted(required_internal - internal_paths)
    errors.extend(f"{path}: arquivo estrutural ausente do inventário interno: {item}"
                  for item in missing_internal)
    return errors


def verify_tree(root: Path) -> tuple[int, list[str]]:
    manifests = sorted(root.glob("**/json/dataset-manifest.json"))
    errors = []
    for manifest in manifests:
        try:
            errors.extend(verify_manifest(manifest))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest}: {type(exc).__name__}: {exc}")
    if not manifests:
        errors.append(f"nenhum json/dataset-manifest.json encontrado em {root}")
    return len(manifests), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    count, errors = verify_tree(args.root.expanduser().resolve())
    for error in errors:
        print(f"ERRO: {error}")
    if errors:
        return 1
    print(f"Verificados {count} dataset(s) de observabilidade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
