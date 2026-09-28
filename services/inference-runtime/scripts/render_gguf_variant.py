#!/usr/bin/env python3
"""Renderiza configs reproduzíveis para uma variante GGUF do benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex


SCHEMA_VERSION = 1
VARIANTS = {
    "q8_0": {
        "alias": "qwen25-7b-q8-0",
        "quantization": "Q8_0",
        "imatrix": False,
        "label": "Q8_0",
        "manifest_key": "q8_0",
    },
    "q4_k_m_base": {
        "alias": "qwen25-7b-q4-k-m-base",
        "quantization": "Q4_K_M",
        "imatrix": False,
        "label": "Q4_K_M sem imatrix",
        "manifest_key": "q4_k_m",
    },
    "q4_k_m_imatrix": {
        "alias": "qwen25-7b-q4-k-m-imatrix",
        "quantization": "Q4_K_M",
        "imatrix": True,
        "label": "Q4_K_M com imatrix",
        "manifest_key": "q4_k_m_imatrix",
    },
}
TOKENIZER_METADATA_NAMES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gguf_signature(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read(4)


def tokenizer_metadata_sha256(path: Path) -> tuple[str, int]:
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and (
            item.name in TOKENIZER_METADATA_NAMES
            or item.name.startswith(("tokenizer", "chat_template"))
            or item.suffix == ".model"
        )
    )
    if not files:
        raise FileNotFoundError(f"Metadados do tokenizer ausentes: {path}")
    digest = hashlib.sha256()
    for item in files:
        name = item.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _config(
    *,
    runtime: str,
    base_url: str,
    alias: str,
    tokenizer: Path,
    context: int,
    artifact_description: str,
    server_command: list[str],
    notes: str,
) -> dict[str, object]:
    cache_policy = {
        "vllm": "enabled-recorded (vLLM default; verify in server.log)",
        "llama.cpp": "enabled-recorded (llama-server KV slots; verify in server.log)",
        "ollama": "enabled-recorded (Ollama resident-model policy; verify in server.log)",
    }[runtime]
    runtime_version = {
        "vllm": "vLLM + vllm-gguf-plugin; exact versions captured by the execution",
        "llama.cpp": "llama.cpp local; exact build captured by server.log",
        "ollama": "Ollama local; exact version captured by preparation/server logs",
    }[runtime]
    return {
        "runtime": runtime,
        "base_url": base_url,
        "model": alias,
        "tokenizer": str(tokenizer),
        "context_window": context,
        "cache_policy": cache_policy,
        "runtime_version": runtime_version,
        "model_artifact": artifact_description,
        "server_command": shlex.join(server_command),
        "notes": notes,
    }


def render_variant(
    *,
    variant: str,
    gguf: Path,
    tokenizer: Path,
    output_dir: Path,
    context: int = 16384,
    gpu_memory_utilization: float = 0.95,
    source_manifest: Path,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Variante inválida: {variant}; use {', '.join(VARIANTS)}")
    if context < 512:
        raise ValueError("context deve ser >= 512")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization deve estar em (0, 1]")

    gguf = gguf.expanduser().resolve()
    tokenizer = tokenizer.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve() / variant
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF ausente: {gguf}")
    if gguf.stat().st_size < 4 or gguf_signature(gguf) != b"GGUF":
        raise ValueError(f"Assinatura GGUF inválida: {gguf}")
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"Diretório do tokenizer ausente: {tokenizer}")

    spec = VARIANTS[variant]
    alias = str(spec["alias"])
    digest = sha256_file(gguf)
    stat = gguf.stat()
    tokenizer_digest, tokenizer_file_count = tokenizer_metadata_sha256(tokenizer)

    source_manifest = source_manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Manifesto de origem ausente: {source_manifest}")
    source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source_data, dict) or not isinstance(source_data.get("artifacts"), dict):
        raise ValueError("Manifesto de origem não contém artifacts")
    manifest_key = str(spec["manifest_key"])
    artifact = source_data["artifacts"].get(manifest_key)
    if not isinstance(artifact, dict):
        raise ValueError(f"Manifesto de origem não contém artifacts.{manifest_key}")
    expected = {
        "filename": gguf.name,
        "sha256": digest,
        "quantization": spec["quantization"],
        "uses_imatrix": spec["imatrix"],
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"Manifesto diverge da variante {variant}: {key}={artifact.get(key)!r}, esperado={value!r}"
            )
    source = {
        "path": str(source_manifest),
        "sha256": sha256_file(source_manifest),
        "artifact_key": manifest_key,
    }

    calibration = "imatrix" if spec["imatrix"] else "none"
    artifact_description = (
        f"{spec['label']}; arquivo local={gguf}; sha256={digest}; "
        f"calibracao_declarada={calibration}"
    )
    provenance_note = (
        "A declaração de imatrix vem do perfil da variante; valide o manifesto de origem."
        if spec["imatrix"]
        else "A variante declara quantização sem imatrix."
    )

    vllm_launch = [
        "vllm",
        "serve",
        str(gguf),
        "--load-format",
        "gguf",
        "--quantization",
        "gguf",
        "--tokenizer",
        str(tokenizer),
        "--served-model-name",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        str(context),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    llama_launch = [
        "llama-server",
        "-m",
        str(gguf),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--ctx-size",
        str(context),
        "--n-gpu-layers",
        "999",
        "--parallel",
        "1",
        "--jinja",
        "--alias",
        alias,
    ]
    ollama_launch = ["ollama", "serve"]

    configs = {
        "vllm": _config(
            runtime="vllm",
            base_url="http://127.0.0.1:8000",
            alias=alias,
            tokenizer=tokenizer,
            context=context,
            artifact_description=artifact_description,
            server_command=vllm_launch,
            notes=f"Variante {variant}. {provenance_note} GGUF no vLLM é experimental; não fazer fallback.",
        ),
        "llama": _config(
            runtime="llama.cpp",
            base_url="http://127.0.0.1:8080",
            alias=alias,
            tokenizer=tokenizer,
            context=context,
            artifact_description=artifact_description,
            server_command=llama_launch,
            notes=f"Variante {variant}. {provenance_note} Uma sequência e offload GPU registrado no log.",
        ),
        "ollama": _config(
            runtime="ollama",
            base_url="http://127.0.0.1:11434",
            alias=alias,
            tokenizer=tokenizer,
            context=context,
            artifact_description=artifact_description,
            server_command=ollama_launch,
            notes=f"Variante {variant}. {provenance_note} Alias criado previamente a partir deste GGUF.",
        ),
    }
    launches = {"vllm": vllm_launch, "llama": llama_launch, "ollama": ollama_launch}
    files: dict[str, dict[str, str]] = {}
    for runtime in ("vllm", "llama", "ollama"):
        config_path = output_dir / f"config-{runtime}.json"
        launch_path = output_dir / f"launch-{runtime}.json"
        write_json(config_path, configs[runtime])
        write_json(launch_path, launches[runtime])
        files[runtime] = {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "launch": str(launch_path),
            "launch_sha256": sha256_file(launch_path),
        }

    profile = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "label": spec["label"],
        "alias": alias,
        "quantization": spec["quantization"],
        "imatrix": spec["imatrix"],
        "imatrix_claim": "verified-by-source-manifest" if spec["imatrix"] else "not-used",
        "gguf_path": str(gguf),
        "gguf_sha256": digest,
        "gguf_bytes": stat.st_size,
        "gguf_mtime_ns": stat.st_mtime_ns,
        "tokenizer_path": str(tokenizer),
        "tokenizer_metadata_sha256": tokenizer_digest,
        "tokenizer_metadata_file_count": tokenizer_file_count,
        "context_window": context,
        "source_manifest": source,
        "files": files,
    }
    profile_path = output_dir / "profile.json"
    write_json(profile_path, profile)
    return profile_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--source-manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    profile = render_variant(
        variant=args.variant,
        gguf=args.gguf,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        context=args.context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        source_manifest=args.source_manifest,
    )
    print(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
