import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path, PurePath

try:
    from .calibration import (
        calibration_metadata,
        load_golden_csv,
        render_calibration_texts,
        select_calibration_examples,
    )
except ImportError:
    from calibration import (
        calibration_metadata,
        load_golden_csv,
        render_calibration_texts,
        select_calibration_examples,
    )

QUANTIZATION_METHODS = ["gptq", "gguf", "awq", "exl3"]
DEFAULT_BITS = 4


def _tool_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("GPTQModel", "datasets", "transformers", "torch", "huggingface_hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _runtime_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata.update(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except (ImportError, RuntimeError):
        metadata["cuda_available"] = False

    repository = Path(__file__).resolve().parents[2]
    try:
        metadata["pipeline_git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        metadata["pipeline_git_dirty"] = bool(dirty.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        metadata["pipeline_git_commit"] = None
        metadata["pipeline_git_dirty"] = None
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_inventory(path: Path) -> list[dict[str, object]]:
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    return [
        {
            "name": item.name,
            "size_bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in weights
    ]


def _compute_path_size_bytes(path):
    if not os.path.exists(path):
        return None

    if os.path.isfile(path):
        return os.path.getsize(path)

    total_size = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def _size_breakdown(size_bytes):
    if size_bytes is None:
        return {
            "bytes": None,
            "mb": None,
            "gb": None,
        }

    mb = size_bytes / (1024 * 1024)
    gb = mb / 1024
    return {
        "bytes": int(size_bytes),
        "mb": round(mb, 2),
        "gb": round(gb, 2),
    }


def build_quantization_summary(
    model_path,
    quant_path,
    quant_method,
    bits,
    *,
    source_model_id=None,
    source_model_revision=None,
    calibration=None,
    group_size=None,
    batch_size=None,
    backend=None,
):
    pre_size_bytes = _compute_path_size_bytes(model_path)
    quant_size_bytes = _compute_path_size_bytes(quant_path)

    size_diff_bytes = None
    percent_reduction = None
    if pre_size_bytes is not None and quant_size_bytes is not None:
        size_diff_bytes = pre_size_bytes - quant_size_bytes
        if pre_size_bytes > 0:
            percent_reduction = round((size_diff_bytes / pre_size_bytes) * 100, 2)

    summary = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "quantization_method": quant_method,
        "bits": bits,
        "source_model_path": model_path,
        "quantized_model_path": quant_path,
        "pre_quantized_model_size": _size_breakdown(pre_size_bytes),
        "quantized_model_size": _size_breakdown(quant_size_bytes),
        "size_difference": _size_breakdown(size_diff_bytes),
        "size_reduction_percent": percent_reduction,
        "source_model_id": source_model_id,
        "source_model_revision": source_model_revision,
        "calibration": calibration,
        "group_size": group_size,
        "batch_size": batch_size,
        "quantization_backend": backend,
        "tool_versions": _tool_versions(),
        "runtime": _runtime_metadata(),
        "weight_files": _weight_inventory(Path(quant_path)),
    }

    if pre_size_bytes is None:
        summary["notes"] = "Source model path not found locally. Size computed only for local paths."

    return summary


def prepare_calibration_dataset():
    """Return the legacy MMLU calibration corpus used by the original helper."""
    from datasets import load_dataset

    ds = load_dataset(
            "Brench/MMLU-Pro-CoT-Train-84K",
            split="train"
        )
    ds = ds.map(
        lambda x: {
            "text": "Question: " + x["question"] + "\nAnswer: " + x["answer"] # "Question: <question> \nAnswer: <answer>"
        }
    )
    return ds.select(range(256))["text"]


def prepare_golden_calibration_dataset(
    model_path: str,
    calibration_csv: Path,
    *,
    limit: int,
    seed: int,
):
    from transformers import AutoTokenizer

    calibration_csv = calibration_csv.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    examples = load_golden_csv(calibration_csv)
    if len(examples) < limit:
        raise ValueError(
            f"Dataset tem {len(examples)} exemplos, menos que os {limit} solicitados"
        )
    selected = select_calibration_examples(examples, limit=limit, seed=seed)
    texts = render_calibration_texts(selected, tokenizer)
    metadata = calibration_metadata(
        calibration_csv.resolve(),
        selected,
        available_count=len(examples),
        limit=limit,
        seed=seed,
    )
    return texts, metadata


def _copy_license(model_path: str, quant_path: str, *, required: bool) -> None:
    source = Path(model_path) / "LICENSE"
    destination = Path(quant_path) / "LICENSE"
    if source.is_file():
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("A saida ja contem uma LICENSE diferente da origem")
        if not destination.exists():
            shutil.copy2(source, destination)
    elif required:
        raise FileNotFoundError("LICENSE do modelo-base Qwen ausente; publicacao interrompida")


def _require_source_license(model_path: str, *, required: bool) -> None:
    if required and not (Path(model_path) / "LICENSE").is_file():
        raise FileNotFoundError("LICENSE do modelo-base Qwen ausente; quantizacao interrompida")


def _require_unquantized_source(model_path: str) -> None:
    source = Path(model_path)
    separate_configs = [source / "quantize_config.json", source / "quant_config.json"]
    config_path = source / "config.json"
    embedded = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config, dict):
            embedded = config.get("quantization_config")
    if any(path.is_file() for path in separate_configs) or embedded:
        raise ValueError("O modelo-base ja possui configuracao de quantizacao; evite dupla quantizacao")


def _write_model_card(path: Path, summary: dict[str, object]) -> None:
    readme = path / "README.md"
    if readme.exists():
        return
    source_model = summary.get("source_model_id") or summary.get("source_model_path")
    method = str(summary.get("quantization_method", "quantized")).upper()
    bits = summary.get("bits")
    license_lines = (
        ["license: apache-2.0"]
        if str(source_model).startswith("Qwen/Qwen2.5-")
        else []
    )
    readme.write_text(
        "\n".join(
            [
                "---",
                f"base_model: {source_model}",
                *license_lines,
                "library_name: gptqmodel",
                "pipeline_tag: text-generation",
                "tags:",
                "- quantized",
                f"- {method.casefold()}",
                "---",
                "",
                f"# {PurePath(str(source_model)).name} {method} {bits}-bit",
                "",
                f"Quantized from `{source_model}` with GPTQModel.",
                "The calibration CSV is not included. Its hash, deterministic selection",
                "parameters, source-model revision, and tool versions are recorded in",
                "`quantization_manifest.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def quantize_gptq(
    model_path,
    quant_path,
    bits: int,
    *,
    calibration_dataset=None,
    group_size: int = 128,
    batch_size: int = 1,
):
    from gptqmodel import BACKEND, GPTQConfig, GPTQModel

    print(f"Quantizing {model_path} to {quant_path} using GPTQ method.")
    # If model_id is a Hugging Face repo name, it will:
    #   use the local HF cache if already present
    #   otherwise download tokenizer files
    # If model_id is a local folder, it loads from that folder
    # Simple calibration dataset, following GPTQModel's docs pattern
    # Downloads the calibration dataset if it is not already cached
    
    if calibration_dataset is None:
        calibration_dataset = prepare_calibration_dataset()

    # Then tune if needed:

    # bits: usually 4; 8 keeps more quality but saves less memory
    # group_size: usually 128; sometimes 64 helps quality
    # desc_act: activation-order option; can affect quality/speed
    # sym: symmetric quantization
    # damp_percent: GPTQ damping
    # mse: MSE-based fitting option
    # act_group_aware=True: GPTQModel docs mention GAR; typically used with desc_act=False
    # dynamic={...}: per-module overrides/skips
    
    quant_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
    )

    # Loads the original full-precision model
    #  if model_id is a repo name, it may download model weights into the local HF cache
    #  If model_id is a local directory, it loads from there
    model = GPTQModel.load(model_path, quant_config)

    # Start with batch_size=1 on limited VRAM
    # Runs calibration samples through the model
    #  Measures quantization error/sensitivity
    #  Produces quantized weights according to GPTQConfig
    quantization_result = model.quantize(
        calibration_dataset,
        batch_size=batch_size,
        backend=BACKEND.GPTQ_TORCH,
    )

    # Writes the quantized model to quant_path
    model.save(quant_path)

    return quantization_result


def quantize_gguf(model_path, quant_path, bits: int):
    from gptqmodel import GGUFConfig, GPTQModel
    from gptqmodel.models._const import DEVICE

    print(f"Quantizing {model_path} to {quant_path} using GGUF method.")

    qcfg = GGUFConfig(
        bits=bits,
        device=DEVICE.CPU, # CPU due to GGUF's current lack of GPU support
    )

    model = GPTQModel.load(model_path, qcfg)
    quantization_result = model.quantize(calibration=None)
    model.save(quant_path)

    return quantization_result


def quantize_awq(
    model_path,
    quant_path,
    bits: int,
    *,
    calibration_dataset=None,
    batch_size: int = 1,
):
    from gptqmodel import AWQConfig, GPTQModel

    print(f"Quantizing {model_path} to {quant_path} using AWQ method.")

    if calibration_dataset is None:
        calibration_dataset = prepare_calibration_dataset()

    qcfg = AWQConfig(
        bits=bits
    )

    model = GPTQModel.load(model_path, qcfg)
    quantization_result = model.quantize(
        calibration=calibration_dataset,
        batch_size=batch_size,
    )
    model.save(quant_path)

    return quantization_result


def quantize_exl3(
    model_path,
    quant_path,
    bits: float,
    *,
    calibration_dataset=None,
    batch_size: int = 1,
):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import EXL3Config

    print(f"Quantizing {model_path} to {quant_path} using EXL3 method.")

    if calibration_dataset is None:
        calibration_dataset = prepare_calibration_dataset()

    qcfg = EXL3Config(
        bits=bits,        # target average bits-per-weight
        head_bits=6.0,   # optional higher bitrate for attention heads / sensitive tensors
        codebook="mcg",  # one of: mcg, mul1, 3inst
    )

    model = GPTQModel.load(model_path, qcfg)
    quantization_result = model.quantize(
        calibration_dataset,
        batch_size=batch_size,
        backend=BACKEND.EXL3_EXLLAMA_V3,
    )
    model.save(quant_path)

    return quantization_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize a Qwen checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("quant_path")
    parser.add_argument("quantization_method", type=str.casefold, choices=QUANTIZATION_METHODS)
    parser.add_argument("bits", nargs="?", default=str(DEFAULT_BITS))
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--calibration-limit", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-model-id")
    parser.add_argument("--source-model-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = args.model_path
    quant_path = args.quant_path
    quant_method = args.quantization_method.lower()
    try:
        bits: int | float
        bits = float(args.bits) if quant_method == "exl3" else int(args.bits)
    except ValueError as exc:
        raise ValueError("bits deve ser numerico") from exc

    if args.batch_size < 1:
        raise ValueError("--batch-size deve ser maior que zero")
    if args.group_size < 1:
        raise ValueError("--group-size deve ser maior que zero")
    if args.calibration_limit < 1:
        raise ValueError("--calibration-limit deve ser maior que zero")
    if bits <= 0:
        raise ValueError("bits deve ser maior que zero")
    if quant_method == "gptq" and not 2 <= bits <= 8:
        raise ValueError("GPTQ exige bits entre 2 e 8")
    if quant_method == "exl3" and not 1.0 <= bits <= 8.0:
        raise ValueError("EXL3 exige bits entre 1.0 e 8.0")
    if quant_method == "awq" and bits != 4:
        raise ValueError("AWQ quantization currently only supports 4 bits")
    if quant_method == "gguf" and args.calibration_csv:
        raise ValueError("GGUF e weight-only neste fluxo; --calibration-csv nao seria usado")
    output = Path(quant_path)
    if output.exists() and not output.is_dir():
        raise ValueError(f"Caminho de saida existe e nao e diretorio: {output}")
    if output.is_dir() and any(output.iterdir()):
        raise ValueError(f"Diretorio de saida existe e nao esta vazio: {output}")
    qwen_source = str(args.source_model_id or "").startswith("Qwen/Qwen2.5-")
    _require_source_license(model_path, required=qwen_source)
    _require_unquantized_source(model_path)

    calibration_dataset = None
    if args.calibration_csv:
        calibration_dataset, calibration_info = prepare_golden_calibration_dataset(
            model_path,
            args.calibration_csv,
            limit=args.calibration_limit,
            seed=args.calibration_seed,
        )
    elif quant_method in {"gptq", "awq", "exl3"}:
        calibration_dataset = prepare_calibration_dataset()
        calibration_info = {
            "source": "Brench/MMLU-Pro-CoT-Train-84K",
            "selected_examples": len(calibration_dataset),
            "selection": "first-256-legacy",
        }
    else:
        calibration_info = None

    quantization_result = None
    if quant_method == "gptq":
        quantization_result = quantize_gptq(
            model_path,
            quant_path,
            int(bits),
            calibration_dataset=calibration_dataset,
            group_size=args.group_size,
            batch_size=args.batch_size,
        )
    elif quant_method == "gguf":
        quantization_result = quantize_gguf(model_path, quant_path, int(bits))
    elif quant_method == "awq":
        quantization_result = quantize_awq(
            model_path,
            quant_path,
            int(bits),
            calibration_dataset=calibration_dataset,
            batch_size=args.batch_size,
        )
    elif quant_method == "exl3":
        quantization_result = quantize_exl3(
            model_path,
            quant_path,
            float(bits),
            calibration_dataset=calibration_dataset,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unknown quantization method: {quant_method}")

    _copy_license(
        model_path,
        quant_path,
        required=qwen_source,
    )

    # Save quantization result to a JSON file
    model_name = PurePath(model_path).parts[-1]
    result_dir = f"data/quantization/{model_name}/{quant_method.upper()}"
    os.makedirs(result_dir, exist_ok=True)

    #result_file = os.path.join(result_dir, f"{quant_method}-{bits}.json")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(result_dir, f"{timestamp}.json")
    result_summary = build_quantization_summary(
        model_path,
        quant_path,
        quant_method,
        bits,
        source_model_id=args.source_model_id,
        source_model_revision=args.source_model_revision,
        calibration=calibration_info,
        group_size=args.group_size if quant_method == "gptq" else None,
        batch_size=args.batch_size if quant_method in {"gptq", "awq", "exl3"} else None,
        backend={
            "gptq": "gptq_torch",
            "exl3": "exl3_exllama_v3",
        }.get(quant_method),
    )
    with open(filename, "w") as f:
        json.dump(result_summary, f, indent=4)

    output_manifest = Path(quant_path) / "quantization_manifest.json"
    output_manifest.write_text(json.dumps(result_summary, indent=2) + "\n", encoding="utf-8")
    _write_model_card(Path(quant_path), result_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
