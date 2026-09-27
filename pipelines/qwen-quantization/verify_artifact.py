"""Verify that a saved quantized checkpoint is complete and optionally reload it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _saved_quantization_config(path: Path, config: dict[str, object]) -> dict[str, object]:
    embedded = config.get("quantization_config")
    if isinstance(embedded, dict) and embedded:
        return embedded
    for name in ("quantize_config.json", "quant_config.json"):
        separate = path / name
        if separate.is_file():
            loaded = json.loads(separate.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
    raise ValueError("Metadados de quantizacao ausentes do checkpoint")


def _verify_manifest_matches_config(
    manifest: dict[str, object],
    quantization_config: dict[str, object],
) -> None:
    manifest_bits = _normalized_number(manifest.get("bits"))
    saved_bits = _normalized_number(quantization_config.get("bits"))
    if manifest_bits is None:
        raise ValueError("Manifesto sem bits validos")
    if saved_bits is None:
        raise ValueError("Checkpoint sem bits validos na configuracao de quantizacao")
    if manifest_bits != saved_bits:
        raise ValueError("Bits do manifesto divergem do checkpoint")

    manifest_method = str(manifest.get("quantization_method", "")).casefold()
    if not manifest_method:
        raise ValueError("Manifesto sem metodo de quantizacao")
    saved_method = quantization_config.get("quant_method") or quantization_config.get("method")
    if not saved_method:
        raise ValueError("Checkpoint sem metodo na configuracao de quantizacao")
    if str(saved_method).casefold() != manifest_method:
        raise ValueError("Metodo do manifesto diverge do checkpoint")

    manifest_group = _normalized_number(manifest.get("group_size"))
    saved_group = _normalized_number(quantization_config.get("group_size"))
    if manifest_method == "gptq" and (manifest_group is None or saved_group is None):
        raise ValueError("Checkpoint GPTQ sem group_size verificavel")
    if manifest_group is not None and saved_group is not None and manifest_group != saved_group:
        raise ValueError("group_size do manifesto diverge do checkpoint")


def _verify_weight_inventory(path: Path, manifest: dict[str, object]) -> list[Path]:
    inventory = manifest.get("weight_files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Manifesto sem inventario SHA-256 dos pesos")

    verified: list[Path] = []
    for entry in inventory:
        if not isinstance(entry, dict):
            raise ValueError("Inventario de pesos invalido")
        name = entry.get("name")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Nome de peso invalido no manifesto")
        candidate = path / name
        if not candidate.is_file():
            raise ValueError(f"Peso registrado ausente: {name}")
        if candidate.stat().st_size < 1 or candidate.stat().st_size != expected_size:
            raise ValueError(f"Tamanho do peso diverge: {name}")
        if _sha256_file(candidate) != expected_hash:
            raise ValueError(f"SHA-256 do peso diverge: {name}")
        verified.append(candidate)
    actual = set(path.glob("*.safetensors")) | set(path.glob("*.bin"))
    if actual != set(verified):
        extras = sorted(item.name for item in actual - set(verified))
        raise ValueError(f"Pesos nao registrados no manifesto: {', '.join(extras)}")
    return verified


def _verify_weight_index(path: Path) -> None:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Indice de shards invalido: {index_name}")
        missing = sorted({str(name) for name in weight_map.values() if not (path / str(name)).is_file()})
        if missing:
            raise ValueError(f"Shards ausentes no indice: {', '.join(missing)}")


def inspect_artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Diretorio do modelo quantizado ausente: {path}")

    required = ["config.json", "quantization_manifest.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Arquivos obrigatorios ausentes: {', '.join(missing)}")

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "quantization_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise ValueError("Config ou manifesto invalido")

    weights = _verify_weight_inventory(path, manifest)
    _verify_weight_index(path)

    tokenizer_files = [
        candidate
        for candidate in ("tokenizer.json", "tokenizer.model", "vocab.json")
        if (path / candidate).is_file()
    ]
    if not (path / "tokenizer_config.json").is_file() or not tokenizer_files:
        raise ValueError("Tokenizer ausente no artefato")

    quantization_config = _saved_quantization_config(path, config)
    _verify_manifest_matches_config(manifest, quantization_config)
    if manifest.get("source_model_id"):
        if not manifest.get("source_model_revision"):
            raise ValueError("Manifesto sem revisao imutavel do modelo-base")
        calibration = manifest.get("calibration")
        if not isinstance(calibration, dict) or not calibration.get("source_sha256"):
            raise ValueError("Manifesto sem SHA-256 do dataset de calibracao")
    if str(manifest.get("source_model_id", "")).startswith("Qwen/Qwen2.5-"):
        license_path = path / "LICENSE"
        if not license_path.is_file() or license_path.stat().st_size < 1:
            raise ValueError("LICENSE do modelo-base Qwen ausente do artefato")

    return {
        "path": str(path),
        "method": manifest.get("quantization_method"),
        "bits": manifest.get("bits"),
        "weights": [item.name for item in weights],
        "weight_bytes": sum(item.stat().st_size for item in weights),
        "tokenizer_files": ["tokenizer_config.json", *tokenizer_files],
        "source_model_id": manifest.get("source_model_id"),
        "source_model_revision": manifest.get("source_model_revision"),
        "calibration_sha256": (manifest.get("calibration") or {}).get("source_sha256"),
    }


def verify_expectations(
    summary: dict[str, object],
    *,
    method: str | None,
    bits: str | None,
    source_model_id: str | None,
) -> None:
    if method and str(summary.get("method", "")).casefold() != method.casefold():
        raise ValueError("Metodo do artefato diverge do esperado pelo Make")
    if bits:
        expected_bits = _normalized_number(bits)
        actual_bits = _normalized_number(summary.get("bits"))
        if expected_bits is None or actual_bits != expected_bits:
            raise ValueError("Bits do artefato divergem do esperado pelo Make")
    if source_model_id and summary.get("source_model_id") != source_model_id:
        raise ValueError("Modelo-base do artefato diverge do esperado pelo Make")


def reload_artifact(path: Path, method: str | None, device: str) -> None:
    from gptqmodel import BACKEND, GPTQModel
    from transformers import AutoTokenizer

    kwargs: dict[str, object] = {"device": device}
    normalized_method = str(method).casefold()
    if normalized_method == "gptq":
        kwargs["backend"] = BACKEND.GPTQ_TORCH
    if normalized_method == "exl3":
        import torch

        kwargs["dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = GPTQModel.from_quantized(str(path), **kwargs)
    inputs = tokenizer("Responda somente: OK", return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    generated = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    if generated is None:
        raise RuntimeError("Reload ocorreu, mas a inferencia de smoke nao retornou tokens")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-method")
    parser.add_argument("--expected-bits")
    parser.add_argument("--expected-source-model-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = inspect_artifact(args.model_dir)
    verify_expectations(
        summary,
        method=args.expected_method,
        bits=args.expected_bits,
        source_model_id=args.expected_source_model_id,
    )
    print(json.dumps(summary, indent=2))
    if args.reload:
        reload_artifact(args.model_dir.resolve(), summary.get("method"), args.device)
        print(f"Reload concluido em {args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
