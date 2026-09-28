#!/usr/bin/env python3
"""Renderiza config do benchmark e argv do runtime a partir do modelo preparado."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any


RUNTIMES = ("vllm", "llama", "ollama")
HF_SOURCES = frozenset({"hf", "local-hf"})
GGUF_SOURCES = frozenset({"gguf", "local-gguf"})
SOURCES = HF_SOURCES | GGUF_SOURCES
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_CONFIG_RUNTIME = {"vllm": "vllm", "llama": "llama.cpp", "ollama": "ollama"}
_CACHE_POLICY = {
    "vllm": "KV, prefix cache e residência registrados separadamente no manifesto",
    "llama": "KV, prefix cache e residência registrados separadamente no manifesto",
    "ollama": "KV, prefix cache e residência registrados separadamente no manifesto",
}
def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number


def _context(value: str) -> int:
    number = int(value)
    if number < 512:
        raise argparse.ArgumentTypeError("context deve ser >= 512")
    return number


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port deve estar entre 1 e 65535")
    return number


def _gpu_memory(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("gpu-memory-utilization deve estar em (0, 1]")
    return number


def _gpu_layers(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("gpu-layers deve ser >= 0")
    return number


def _nonempty(value: str, name: str, *, allow_spaces: bool = False) -> str:
    if not value.strip():
        raise ValueError(f"{name} deve ser uma string não vazia")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} contém caractere de controle")
    if not allow_spaces and any(char.isspace() for char in value):
        raise ValueError(f"{name} não pode conter espaços")
    return value


def _loopback_url(host: str, port: int) -> str:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--host deve ser loopback: 127.0.0.1, localhost ou ::1")
    authority = f"[{host}]" if host == "::1" else host
    return f"http://{authority}:{port}"


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata JSON inválido: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata deve conter um objeto JSON")
    return value


def _require_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata.{key} deve ser uma string não vazia")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"metadata.{key} contém caractere de controle")
    return value


def validate_metadata(metadata: dict[str, Any], runtime: str) -> dict[str, Any]:
    """Seleciona e valida somente os campos necessários, sem propagar segredos."""
    source = _require_text(metadata, "source")
    metadata_runtime = _require_text(metadata, "runtime")
    local_path_text = _require_text(metadata, "local_path")
    tokenizer_path_text = _require_text(metadata, "tokenizer_path")
    requested_model = _require_text(metadata, "requested_model")
    resolved_revision = _require_text(metadata, "resolved_revision")
    sha256 = _require_text(metadata, "sha256")
    size = metadata.get("bytes")

    if source not in SOURCES:
        raise ValueError(f"metadata.source inválido: {source!r}")
    if metadata_runtime != runtime:
        raise ValueError(
            f"metadata.runtime={metadata_runtime!r} diverge de --runtime={runtime!r}"
        )
    if runtime in {"llama", "ollama"} and source not in GGUF_SOURCES:
        raise ValueError(f"runtime {runtime!r} exige source GGUF")
    if not _SHA256.fullmatch(sha256):
        raise ValueError("metadata.sha256 deve conter 64 dígitos hexadecimais")
    if type(size) is not int or size <= 0:
        raise ValueError("metadata.bytes deve ser um inteiro positivo")

    local_path = Path(local_path_text).expanduser()
    tokenizer_path = Path(tokenizer_path_text).expanduser()
    if not local_path.is_absolute():
        raise ValueError("metadata.local_path deve ser absoluto")
    if not tokenizer_path.is_absolute():
        raise ValueError("metadata.tokenizer_path deve ser absoluto")
    local_path = local_path.resolve()
    tokenizer_path = tokenizer_path.resolve()

    if source in HF_SOURCES:
        if not local_path.is_dir():
            raise FileNotFoundError(f"snapshot HF local ausente: {local_path}")
    else:
        if not local_path.is_file():
            raise FileNotFoundError(f"GGUF local ausente: {local_path}")
        with local_path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError(f"assinatura GGUF inválida: {local_path}")

    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"tokenizer local ausente: {tokenizer_path}")
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(
            f"tokenizer_config.json ausente; o benchmark exige tokenizer local: {tokenizer_path}"
        )

    requested_revision = metadata.get("requested_revision")
    if requested_revision is not None and not isinstance(requested_revision, str):
        raise ValueError("metadata.requested_revision deve ser string ou null")

    return {
        "source": source,
        "runtime": runtime,
        "local_path": str(local_path),
        "tokenizer_path": str(tokenizer_path),
        "requested_model": requested_model,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "sha256": sha256.lower(),
        "bytes": size,
    }


def build_launch(
    *,
    identity: dict[str, Any],
    runtime: str,
    model_alias: str,
    context: int,
    host: str,
    port: int,
    runtime_bin: str,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    dtype: str,
    gpu_layers: int,
) -> list[str]:
    _nonempty(model_alias, "model-alias")
    _nonempty(runtime_bin, "runtime-bin", allow_spaces=True)
    _nonempty(dtype, "dtype")
    _loopback_url(host, port)
    if context < 512:
        raise ValueError("context deve ser >= 512")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization deve estar em (0, 1]")
    if max_num_seqs <= 0:
        raise ValueError("max-num-seqs deve ser positivo")
    if gpu_layers < 0:
        raise ValueError("gpu-layers deve ser >= 0")

    local_path = identity["local_path"]
    tokenizer_path = identity["tokenizer_path"]
    source = identity["source"]
    if runtime == "vllm":
        command = [runtime_bin, "serve", local_path]
        if source in GGUF_SOURCES:
            command.extend(["--load-format", "gguf", "--quantization", "gguf"])
        command.extend(
            [
                "--tokenizer",
                tokenizer_path,
                "--served-model-name",
                model_alias,
                "--host",
                host,
                "--port",
                str(port),
                "--max-model-len",
                str(context),
                "--max-num-seqs",
                str(max_num_seqs),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--dtype",
                dtype,
            ]
        )
        return command
    if runtime == "llama":
        return [
            runtime_bin,
            "-m",
            local_path,
            "--alias",
            model_alias,
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(context),
            "--n-gpu-layers",
            str(gpu_layers),
            "--parallel",
            "1",
            "--jinja",
        ]
    if runtime == "ollama":
        if host not in {"127.0.0.1", "localhost"} or port != 11434:
            raise ValueError(
                "ollama serve sem wrapper usa o endpoint padrão local 127.0.0.1:11434"
            )
        return [runtime_bin, "serve"]
    raise ValueError(f"runtime inválido: {runtime!r}")


def build_config(
    *,
    identity: dict[str, Any],
    runtime: str,
    model_alias: str,
    context: int,
    host: str,
    port: int,
    launch: list[str],
    runtime_version: str | None = None,
) -> dict[str, Any]:
    base_url = _loopback_url(host, port)
    requested_revision = identity["requested_revision"] or "não informada"
    model_artifact = (
        f"source={identity['source']}; model={identity['requested_model']}; "
        f"requested_revision={requested_revision}; "
        f"resolved_revision={identity['resolved_revision']}; "
        f"sha256={identity['sha256']}; bytes={identity['bytes']}"
    )
    notes = {
        "vllm": (
            "Pesos e tokenizer são locais; uma requisição por vez. "
            + (
                "GGUF no vLLM é experimental; não fazer fallback de formato."
                if identity["source"] in GGUF_SOURCES
                else "Snapshot HF fixado antes da medição."
            )
        ),
        "llama": "GGUF local; uma sequência por vez e offload GPU registrado no comando.",
        "ollama": (
            f"Alias {model_alias} deve estar criado a partir do GGUF local antes da medição; "
            "ollama pull não faz parte do benchmark."
        ),
    }[runtime]
    # `bench.load_config` exige exatamente estes dez campos.
    return {
        "runtime": _CONFIG_RUNTIME[runtime],
        "base_url": base_url,
        "model": model_alias,
        "tokenizer": identity["tokenizer_path"],
        "context_window": context,
        "cache_policy": _CACHE_POLICY[runtime],
        "runtime_version": _nonempty(
            runtime_version or "", "runtime-version", allow_spaces=True
        ),
        "model_artifact": model_artifact,
        "server_command": shlex.join(launch),
        "notes": notes,
    }


def _stage_json(path: Path, payload: object) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_outputs_atomic(
    *, config_out: Path, config: dict[str, Any], launch_out: Path, launch: list[str]
) -> None:
    config_out = config_out.expanduser().resolve()
    launch_out = launch_out.expanduser().resolve()
    if config_out == launch_out:
        raise ValueError("--config-out e --launch-out devem ser arquivos diferentes")
    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((_stage_json(config_out, config), config_out))
        staged.append((_stage_json(launch_out, launch), launch_out))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)


def render_runtime(
    *,
    metadata_path: Path,
    config_out: Path,
    launch_out: Path,
    runtime: str,
    model_alias: str,
    context: int,
    host: str,
    port: int,
    runtime_bin: str,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 1,
    dtype: str = "auto",
    gpu_layers: int = 999,
    runtime_version: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime inválido: {runtime!r}")
    identity = validate_metadata(_read_metadata(metadata_path), runtime)
    launch = build_launch(
        identity=identity,
        runtime=runtime,
        model_alias=model_alias,
        context=context,
        host=host,
        port=port,
        runtime_bin=runtime_bin,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        gpu_layers=gpu_layers,
    )
    config = build_config(
        identity=identity,
        runtime=runtime,
        model_alias=model_alias,
        context=context,
        host=host,
        port=port,
        launch=launch,
        runtime_version=runtime_version,
    )
    _write_outputs_atomic(
        config_out=config_out, config=config, launch_out=launch_out, launch=launch
    )
    return config, launch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--launch-out", required=True, type=Path)
    parser.add_argument("--runtime", required=True, choices=RUNTIMES)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--context", required=True, type=_context)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=_port)
    parser.add_argument("--runtime-bin", required=True)
    parser.add_argument(
        "--gpu-memory-utilization", type=_gpu_memory, default=0.9
    )
    parser.add_argument("--max-num-seqs", type=_positive, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-layers", type=_gpu_layers, default=999)
    parser.add_argument("--runtime-version", required=True,
                        help="versão/commit efetivo capturado após instalar o runtime")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_runtime(
        metadata_path=args.metadata,
        config_out=args.config_out,
        launch_out=args.launch_out,
        runtime=args.runtime,
        model_alias=args.model_alias,
        context=args.context,
        host=args.host,
        port=args.port,
        runtime_bin=args.runtime_bin,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        gpu_layers=args.gpu_layers,
        runtime_version=args.runtime_version,
    )
    print(args.config_out)
    print(args.launch_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
