#!/usr/bin/env python3
"""Materializa um modelo local e grava sua identidade antes do benchmark."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Iterable


SOURCES = ("hf", "local-hf", "gguf", "local-gguf")
RUNTIMES = ("vllm", "llama", "ollama")
REMOTE_SOURCES = frozenset({"hf", "gguf"})
HF_SOURCES = frozenset({"hf", "local-hf"})
GGUF_SOURCES = frozenset({"gguf", "local-gguf"})

_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt"})
_NON_WEIGHT_BIN_PREFIXES = (
    "optimizer",
    "scheduler",
    "training_args",
    "rng_state",
    "scaler",
)
_TOKENIZER_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
    }
)
_TOKENIZER_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab*",
    "merges.txt",
    "*.model",
    "*.tiktoken",
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def validate_request(
    *,
    source: str,
    runtime: str,
    model: str,
    output_dir: Path | None,
    gguf_filename: str | None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
) -> None:
    """Valida a combinação sem acessar rede ou modificar o filesystem."""
    if source not in SOURCES:
        raise ValueError(f"source inválido: {source!r}")
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime inválido: {runtime!r}")
    if source in HF_SOURCES and runtime != "vllm":
        raise ValueError(f"source {source!r} só pode ser usada com runtime 'vllm'")
    if not model.strip():
        raise ValueError("model não pode ser vazio")
    if source in REMOTE_SOURCES and output_dir is None:
        raise ValueError(f"--output-dir é obrigatório para source {source!r}")
    if source == "gguf" and not gguf_filename:
        raise ValueError("--gguf-filename é obrigatório para source 'gguf'")
    if source in GGUF_SOURCES and not tokenizer:
        raise ValueError("--tokenizer é obrigatório para fontes GGUF")
    if tokenizer_revision and not tokenizer:
        raise ValueError("--tokenizer-revision exige --tokenizer")
    if tokenizer and output_dir is None and not _tokenizer_value_is_local(tokenizer):
        raise ValueError("--output-dir é obrigatório quando --tokenizer é um repo remoto")
    if gguf_filename:
        filename = PurePosixPath(gguf_filename)
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("--gguf-filename deve ser relativo e não pode conter '..'")


def resolve_hf_revision(repo_id: str, revision: str) -> str:
    """Resolve uma referência mutável do Hub para o commit imutável."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError(f"Hugging Face não retornou commit para {repo_id}@{revision}")
    if not _COMMIT_PATTERN.fullmatch(resolved):
        raise RuntimeError(
            f"Hugging Face retornou commit inválido para {repo_id}@{revision}: {resolved!r}"
        )
    return resolved.lower()


def _commit_directory(root: Path, kind: str, revision: str) -> Path:
    if kind not in {"model", "tokenizer"}:
        raise ValueError(f"tipo de snapshot inválido: {kind!r}")
    if not _COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(f"commit remoto inválido: {revision!r}")
    return root.expanduser().resolve() / kind / revision.lower()


def download_hf_snapshot(repo_id: str, revision: str, output_dir: Path) -> Path:
    """Baixa um snapshot já fixado por commit para um diretório explícito."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(output_dir),
        repo_type="model",
    )
    return Path(path)


def download_hf_file(repo_id: str, revision: str, filename: str, output_dir: Path) -> Path:
    """Baixa um único GGUF já fixado por commit para um diretório explícito."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        local_dir=str(output_dir),
        repo_type="model",
    )
    return Path(path)


def download_tokenizer_snapshot(repo_id: str, revision: str, output_dir: Path) -> Path:
    """Baixa apenas os arquivos necessários ao tokenizer, sem pesos do modelo."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(output_dir),
        repo_type="model",
        allow_patterns=list(_TOKENIZER_ALLOW_PATTERNS),
    )
    return Path(path)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    expected = path.stat().st_size
    if size != expected:
        raise RuntimeError(f"arquivo mudou durante o hash: {path}")
    return digest.hexdigest(), size


def _file_records(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        sha256, size = _sha256_file(path)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256,
            }
        )
    return records


def _aggregate_records(records: list[dict[str, Any]]) -> str:
    """Gera digest estável incluindo caminho, tamanho e digest de cada arquivo."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(record["sha256"]))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_weight(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in _WEIGHT_SUFFIXES:
        return False
    return not name.startswith(_NON_WEIGHT_BIN_PREFIXES)


def _is_tokenizer_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in _TOKENIZER_NAMES
        or name.startswith("tokenizer")
        or name.startswith("vocab.")
        or name.endswith(".tiktoken")
    )


def _directory_identity(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    weight_paths = [path for path in root.rglob("*") if path.is_file() and _is_weight(path)]
    if not weight_paths:
        raise ValueError(f"nenhum arquivo de pesos HF encontrado em {root}")
    weights = _file_records(root, weight_paths)
    tokenizer_paths = [
        path for path in root.rglob("*") if path.is_file() and _is_tokenizer_file(path)
    ]
    tokenizer_files = _file_records(root, tokenizer_paths)
    artifact = {
        "kind": "hf_directory",
        "bytes": sum(record["bytes"] for record in weights),
        "sha256": _aggregate_records(weights),
        "files": weights,
    }
    tokenizer = {
        "present": bool(tokenizer_files),
        "path": str(root) if tokenizer_files else None,
        "bytes": sum(record["bytes"] for record in tokenizer_files),
        "sha256": _aggregate_records(tokenizer_files) if tokenizer_files else None,
        "files": tokenizer_files,
    }
    return artifact, tokenizer


def _tokenizer_identity(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"diretório de tokenizer local não encontrado: {root}")
    tokenizer_paths = [
        path for path in root.rglob("*") if path.is_file() and _is_tokenizer_file(path)
    ]
    tokenizer_files = _file_records(root, tokenizer_paths)
    if not tokenizer_files:
        raise ValueError(f"nenhum arquivo de tokenizer encontrado em {root}")
    return {
        "present": True,
        "path": str(root),
        "bytes": sum(record["bytes"] for record in tokenizer_files),
        "sha256": _aggregate_records(tokenizer_files),
        "files": tokenizer_files,
    }


def _gguf_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature != b"GGUF":
        raise ValueError(f"assinatura GGUF inválida: {path}")
    sha256, size = _sha256_file(path)
    return {
        "kind": "gguf",
        "bytes": size,
        "sha256": sha256,
        "signature": signature.decode("ascii"),
        "files": [{"path": path.name, "bytes": size, "sha256": sha256}],
    }


def _local_snapshot_revision(path: Path) -> str | None:
    candidates = (path, *path.parents)
    for candidate in candidates:
        if candidate.parent.name == "snapshots" and _COMMIT_PATTERN.fullmatch(candidate.name):
            return candidate.name
    return None


def _tokenizer_value_is_local(value: str) -> bool:
    expanded = Path(value).expanduser()
    return expanded.exists() or value.startswith(("/", "./", "../", "~"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare_model(
    *,
    source: str,
    runtime: str,
    model: str,
    revision: str | None,
    gguf_filename: str | None,
    output_dir: Path | None,
    metadata_out: Path,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    revision_resolver: Callable[[str, str], str] = resolve_hf_revision,
    snapshot_downloader: Callable[[str, str, Path], Path] = download_hf_snapshot,
    file_downloader: Callable[[str, str, str, Path], Path] = download_hf_file,
    tokenizer_downloader: Callable[[str, str, Path], Path] = download_tokenizer_snapshot,
) -> dict[str, Any]:
    """Materializa, valida e identifica um modelo para uso posterior pelo runtime."""
    validate_request(
        source=source,
        runtime=runtime,
        model=model,
        output_dir=output_dir,
        gguf_filename=gguf_filename,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
    )

    requested_revision = revision
    resolved_revision: str | None = None
    tokenizer_identity: dict[str, Any] | None = None
    resolved_tokenizer_revision: str | None = None

    if source in REMOTE_SOURCES:
        requested = revision or "main"
        resolved_revision = revision_resolver(model, requested)
        destination = _commit_directory(
            output_dir, "model", resolved_revision  # type: ignore[arg-type]
        )
        destination.mkdir(parents=True, exist_ok=True)
        if source == "hf":
            local_path = snapshot_downloader(model, resolved_revision, destination)
        else:
            local_path = file_downloader(
                model, resolved_revision, gguf_filename or "", destination
            )
    else:
        local_path = Path(model).expanduser()

    local_path = local_path.resolve()
    if source in HF_SOURCES:
        if not local_path.is_dir():
            raise FileNotFoundError(f"diretório HF local não encontrado: {local_path}")
        artifact, model_tokenizer_identity = _directory_identity(local_path)
        model_path = local_path
    else:
        if not local_path.is_file():
            raise FileNotFoundError(f"GGUF local não encontrado: {local_path}")
        artifact = _gguf_identity(local_path)
        model_path = local_path

    if source not in REMOTE_SOURCES:
        resolved_revision = (
            revision
            or _local_snapshot_revision(local_path)
            or f"sha256:{artifact['sha256']}"
        )

    if tokenizer:
        if _tokenizer_value_is_local(tokenizer):
            prepared_tokenizer_path = Path(tokenizer).expanduser().resolve()
            tokenizer_identity = _tokenizer_identity(prepared_tokenizer_path)
            resolved_tokenizer_revision = (
                tokenizer_revision
                or _local_snapshot_revision(prepared_tokenizer_path)
                or f"sha256:{tokenizer_identity['sha256']}"
            )
            tokenizer_source = "local"
        else:
            if output_dir is None:
                raise ValueError(
                    "--output-dir é obrigatório quando --tokenizer é um repo remoto"
                )
            requested_tokenizer_revision = tokenizer_revision or "main"
            resolved_tokenizer_revision = revision_resolver(
                tokenizer, requested_tokenizer_revision
            )
            tokenizer_destination = _commit_directory(
                output_dir, "tokenizer", resolved_tokenizer_revision
            )
            tokenizer_destination.mkdir(parents=True, exist_ok=True)
            prepared_tokenizer_path = tokenizer_downloader(
                tokenizer, resolved_tokenizer_revision, tokenizer_destination
            ).expanduser().resolve()
            tokenizer_identity = _tokenizer_identity(prepared_tokenizer_path)
            tokenizer_source = "hf"
        tokenizer_identity.update(
            {
                "source": tokenizer_source,
                "requested_model": tokenizer,
                "requested_revision": tokenizer_revision,
                "resolved_revision": resolved_tokenizer_revision,
            }
        )
    elif source in HF_SOURCES:
        if not model_tokenizer_identity["present"]:
            raise ValueError(f"nenhum arquivo de tokenizer encontrado em {model_path}")
        prepared_tokenizer_path = model_path
        tokenizer_identity = model_tokenizer_identity
        resolved_tokenizer_revision = resolved_revision
        tokenizer_identity.update(
            {
                "source": "model_snapshot",
                "requested_model": model,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
            }
        )
    else:  # pragma: no cover - validate_request exige tokenizer para GGUF
        raise AssertionError("tokenizer ausente para fonte GGUF")

    tokenizer_path = str(prepared_tokenizer_path)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "prepared_at_utc": _utc_now(),
        "source": source,
        "runtime": runtime,
        "requested_model": model,
        "model": model,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "gguf_filename": gguf_filename,
        "paths": {
            "model": str(model_path),
            "tokenizer": tokenizer_path,
        },
        "local_path": str(model_path),
        "tokenizer_path": tokenizer_path,
        "requested_tokenizer": tokenizer,
        "requested_tokenizer_revision": tokenizer_revision,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        # Campos redundantes no topo simplificam o manifesto do benchmark.
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
        "artifact": artifact,
        "tokenizer": tokenizer_identity,
    }
    _write_json_atomic(metadata_out, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "prepare", help="baixa quando necessário, valida e grava metadata JSON"
    )
    command.add_argument("--source", required=True, choices=SOURCES)
    command.add_argument("--runtime", required=True, choices=RUNTIMES)
    command.add_argument("--model", required=True, help="repo HF ou caminho local")
    command.add_argument("--revision", help="referência HF; remoto usa 'main' por padrão")
    command.add_argument("--gguf-filename", help="arquivo no repo para source=gguf")
    command.add_argument(
        "--tokenizer",
        help="repo HF ou diretório local; obrigatório para fontes GGUF",
    )
    command.add_argument("--tokenizer-revision", help="referência do repo do tokenizer")
    command.add_argument("--output-dir", type=Path, help="destino local para fontes remotas")
    command.add_argument("--metadata-out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "prepare":  # pragma: no cover - protegido pelo argparse
        raise AssertionError(args.command)
    metadata = prepare_model(
        source=args.source,
        runtime=args.runtime,
        model=args.model,
        revision=args.revision,
        gguf_filename=args.gguf_filename,
        output_dir=args.output_dir,
        metadata_out=args.metadata_out,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
