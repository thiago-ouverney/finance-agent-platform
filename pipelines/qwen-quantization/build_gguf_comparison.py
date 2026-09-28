"""Build and verify three single-file GGUF artifacts from one BF16 source.

The calibrated variant uses a llama.cpp importance matrix produced from a
previously rendered private corpus.  This program never copies the corpus or
the source Safetensors into the final model files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Sequence


SCHEMA_VERSION = 1
FINAL_ARTIFACTS = {
    "q8_0": ("Q8_0", False),
    "q4_k_m": ("Q4_K_M", False),
    "q4_k_m_imatrix": ("Q4_K_M", True),
}


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON deve ser objeto: {path}")
    return value


def require_gguf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 4:
        raise ValueError(f"GGUF ausente ou vazio: {path}")
    with path.open("rb") as source:
        if source.read(4) != b"GGUF":
            raise ValueError(f"Assinatura GGUF invalida: {path}")


def source_revision(model_dir: Path) -> str:
    marker = model_dir / ".source_revision"
    if not marker.is_file() or not marker.read_text(encoding="utf-8").strip():
        raise ValueError(f"Revisao imutavel ausente: {marker}")
    return marker.read_text(encoding="utf-8").strip()


def llama_revision(llama_cpp_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def artifact_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "bf16": output_dir / f"{prefix}-BF16.gguf",
        "imatrix": output_dir / "mopep-train-imatrix.gguf",
        "q8_0": output_dir / f"{prefix}-Q8_0.gguf",
        "q4_k_m": output_dir / f"{prefix}-Q4_K_M.gguf",
        "q4_k_m_imatrix": output_dir / f"{prefix}-Q4_K_M-imatrix.gguf",
        "state": output_dir / "gguf-build-inputs.json",
        "progress": output_dir / "gguf-build-progress.json",
        "manifest": output_dir / "gguf-comparison-manifest.json",
        "commands": output_dir / "gguf-build-commands.jsonl",
    }


def command_record(path: Path, command: Sequence[str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"argv": list(command)}, ensure_ascii=False) + "\n")


def run_to_gguf(command: list[str], output: Path, commands_log: Path) -> None:
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ValueError(f"Saida parcial encontrada; inspecione/remova conscientemente: {partial}")
    rewritten = [str(partial) if item == str(output) else item for item in command]
    command_record(commands_log, rewritten)
    try:
        subprocess.run(rewritten, check=True)
        require_gguf(partial)
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            print(f"Saida parcial preservada para diagnostico: {partial}", file=sys.stderr)
        raise


def run_checked(command: list[str], commands_log: Path) -> None:
    command_record(commands_log, command)
    subprocess.run(command, check=True)


def initialize_state(
    *,
    paths: dict[str, Path],
    model_dir: Path,
    model_id: str,
    corpus: Path,
    corpus_manifest: Path,
    llama_cpp_dir: Path,
    expected_llama_revision: str,
    prefix: str,
    threads: int,
    gpu_layers: int,
    context: int,
    batch: int,
    ubatch: int,
) -> dict[str, Any]:
    config = model_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json ausente: {config}")
    config_data = read_json(config)
    if config_data.get("quantization_config"):
        raise ValueError("A fonte ja esta quantizada; use o checkpoint BF16/FP16 original")
    weights = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError(f"Pesos HF originais ausentes/vazios em {model_dir}")
    if not corpus.is_file() or not corpus_manifest.is_file():
        raise FileNotFoundError("Corpus ou manifesto de corpus ausente")
    if corpus.stat().st_size == 0:
        raise ValueError("Corpus imatrix vazio")

    actual_llama_revision = llama_revision(llama_cpp_dir)
    if actual_llama_revision != expected_llama_revision:
        raise ValueError(
            f"llama.cpp em {actual_llama_revision}, esperado {expected_llama_revision}"
        )
    corpus_data = read_json(corpus_manifest)
    if corpus_data.get("corpus_sha256") != sha256_file(corpus):
        raise ValueError("Hash do corpus diverge do manifesto")
    for field in ("dataset_sha256", "selection_sha256", "prompt_sha256", "tokenizer_sha256"):
        value = corpus_data.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Campo obrigatorio ausente no manifesto do corpus: {field}")
    if not isinstance(corpus_data.get("selected_examples"), int) or corpus_data["selected_examples"] < 1:
        raise ValueError("Quantidade de exemplos invalida no manifesto do corpus")
    if not isinstance(corpus_data.get("selection_seed"), int) or corpus_data["selection_seed"] < 0:
        raise ValueError("Seed invalida no manifesto do corpus")
    if corpus_data.get("includes_expected_answer") is not True:
        raise ValueError("O corpus imatrix deve incluir as respostas golden esperadas")

    state = {
        "schema_version": SCHEMA_VERSION,
        "source_model_id": model_id,
        "source_model_revision": source_revision(model_dir),
        "source_config_sha256": sha256_file(config),
        "source_weight_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weights
        ],
        "corpus_sha256": sha256_file(corpus),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "dataset_sha256": corpus_data.get("dataset_sha256"),
        "selection_sha256": corpus_data.get("selection_sha256"),
        "prompt_sha256": corpus_data.get("prompt_sha256"),
        "tokenizer_sha256": corpus_data.get("tokenizer_sha256"),
        "selected_examples": corpus_data.get("selected_examples"),
        "selection_seed": corpus_data.get("selection_seed"),
        "llama_cpp_revision": actual_llama_revision,
        "prefix": prefix,
        "threads": threads,
        "imatrix": {
            "gpu_layers": gpu_layers,
            "context": context,
            "batch": batch,
            "ubatch": ubatch,
            "parse_special": True,
            "compute_perplexity": False,
            "process_output_tensor": False,
        },
    }

    if paths["state"].is_file():
        existing = read_json(paths["state"])
        if existing != state:
            raise ValueError(
                f"Entradas mudaram desde o inicio da construcao: {paths['state']}"
            )
    else:
        unexpected = [
            paths[name]
            for name in (
                "bf16",
                "imatrix",
                *FINAL_ARTIFACTS,
                "progress",
                "manifest",
                "commands",
            )
            if paths[name].exists()
        ]
        if unexpected:
            raise ValueError(
                "Artefatos existem sem manifesto de entradas: "
                + ", ".join(str(path) for path in unexpected)
            )
        atomic_json(paths["state"], state)
    return state


def load_or_initialize_progress(paths: dict[str, Path]) -> dict[str, Any]:
    artifact_names = ("bf16", "imatrix", *FINAL_ARTIFACTS)
    progress_path = paths["progress"]
    if progress_path.is_file():
        progress = read_json(progress_path)
        artifacts = progress.get("artifacts")
        if progress.get("schema_version") != SCHEMA_VERSION or not isinstance(artifacts, dict):
            raise ValueError(f"Manifesto de progresso invalido: {progress_path}")
        unknown = set(artifacts) - set(artifact_names)
        if unknown:
            raise ValueError(f"Progresso contem artefatos desconhecidos: {sorted(unknown)}")
        return progress

    existing = [paths[name] for name in artifact_names if paths[name].exists()]
    if existing:
        raise ValueError(
            "Artefatos retomaveis existem sem manifesto de progresso: "
            + ", ".join(str(path) for path in existing)
        )
    progress: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "artifacts": {}}
    atomic_json(progress_path, progress)
    return progress


def ensure_progress_artifact(
    *,
    name: str,
    path: Path,
    progress_path: Path,
    progress: dict[str, Any],
    build_action: Callable[[], None],
) -> None:
    artifacts = progress["artifacts"]
    expected = artifacts.get(name)
    if path.exists():
        require_gguf(path)
        if not isinstance(expected, dict):
            raise ValueError(f"Artefato existe sem hash de progresso; nao sera legitimado: {path}")
        if (
            expected.get("filename") != path.name
            or expected.get("bytes") != path.stat().st_size
            or expected.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Integridade do artefato diverge do progresso: {path}")
        return
    if expected is not None:
        raise ValueError(f"Artefato registrado no progresso esta ausente: {path}")

    build_action()
    require_gguf(path)
    artifacts[name] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    atomic_json(progress_path, progress)


def build(args: argparse.Namespace) -> dict[str, Any]:
    llama_cpp_dir = args.llama_cpp_dir.expanduser().resolve()
    llama_cpp_build_dir = (
        args.llama_cpp_build_dir.expanduser().resolve()
        if getattr(args, "llama_cpp_build_dir", None) is not None
        else llama_cpp_dir / "build"
    )
    model_dir = args.hf_model_dir.expanduser().resolve()
    corpus = args.corpus.expanduser().resolve()
    corpus_manifest = args.corpus_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.prefix):
        raise ValueError("prefix deve conter apenas letras, numeros, ponto, hifen ou sublinhado")
    paths = artifact_paths(output_dir, args.prefix)

    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    imatrix_bin = llama_cpp_build_dir / "bin" / "llama-imatrix"
    quantize_bin = llama_cpp_build_dir / "bin" / "llama-quantize"
    for required in (converter, imatrix_bin, quantize_bin):
        if not required.is_file() or (required != converter and not os.access(required, os.X_OK)):
            raise FileNotFoundError(f"Ferramenta llama.cpp ausente: {required}")

    state = initialize_state(
        paths=paths,
        model_dir=model_dir,
        model_id=args.source_model_id,
        corpus=corpus,
        corpus_manifest=corpus_manifest,
        llama_cpp_dir=llama_cpp_dir,
        expected_llama_revision=args.llama_cpp_revision,
        prefix=args.prefix,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        context=args.context,
        batch=args.batch,
        ubatch=args.ubatch,
    )
    progress = load_or_initialize_progress(paths)

    if paths["manifest"].is_file():
        existing_manifest = verify(
            argparse.Namespace(output_dir=output_dir, prefix=args.prefix)
        )
        if existing_manifest.get("inputs") != state:
            raise ValueError("Manifesto final diverge das entradas fixadas do build")
        intermediates = existing_manifest.get("intermediates")
        if not isinstance(intermediates, dict):
            raise ValueError("Manifesto final nao contem os intermediarios")
        for name in ("bf16", "imatrix"):
            expected = intermediates.get(name)
            if not isinstance(expected, dict):
                raise ValueError(f"Intermediario ausente no manifesto: {name}")
            path = paths[name]
            require_gguf(path)
            if path.name != expected.get("filename"):
                raise ValueError(f"Nome do intermediario diverge: {path}")
            if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
                raise ValueError(f"Integridade do intermediario diverge: {path}")
        return existing_manifest

    ensure_progress_artifact(
        name="bf16",
        path=paths["bf16"],
        progress_path=paths["progress"],
        progress=progress,
        build_action=lambda: run_to_gguf(
            [
                str(args.python),
                str(converter),
                str(model_dir),
                "--outfile",
                str(paths["bf16"]),
                "--outtype",
                "bf16",
            ],
            paths["bf16"],
            paths["commands"],
        ),
    )

    ensure_progress_artifact(
        name="imatrix",
        path=paths["imatrix"],
        progress_path=paths["progress"],
        progress=progress,
        build_action=lambda: run_to_gguf(
            [
                str(imatrix_bin),
                "-m",
                str(paths["bf16"]),
                "-f",
                str(corpus),
                "-o",
                str(paths["imatrix"]),
                "--output-format",
                "gguf",
                "--parse-special",
                "--no-ppl",
                "-c",
                str(args.context),
                "-b",
                str(args.batch),
                "-ub",
                str(args.ubatch),
                "-ngl",
                str(args.gpu_layers),
                "-t",
                str(args.threads),
            ],
            paths["imatrix"],
            paths["commands"],
        ),
    )
    run_checked(
        [str(imatrix_bin), "--in-file", str(paths["imatrix"]), "--show-statistics"],
        paths["commands"],
    )

    quantize_commands = {
        "q4_k_m": [
            str(quantize_bin),
            str(paths["bf16"]),
            str(paths["q4_k_m"]),
            "Q4_K_M",
            str(args.threads),
        ],
        "q4_k_m_imatrix": [
            str(quantize_bin),
            "--imatrix",
            str(paths["imatrix"]),
            str(paths["bf16"]),
            str(paths["q4_k_m_imatrix"]),
            "Q4_K_M",
            str(args.threads),
        ],
        "q8_0": [
            str(quantize_bin),
            str(paths["bf16"]),
            str(paths["q8_0"]),
            "Q8_0",
            str(args.threads),
        ],
    }
    for name in ("q4_k_m", "q4_k_m_imatrix", "q8_0"):
        ensure_progress_artifact(
            name=name,
            path=paths[name],
            progress_path=paths["progress"],
            progress=progress,
            build_action=lambda name=name: run_to_gguf(
                quantize_commands[name], paths[name], paths["commands"]
            ),
        )

    if sha256_file(paths["q4_k_m"]) == sha256_file(paths["q4_k_m_imatrix"]):
        raise ValueError("Q4_K_M com e sem imatrix produziram o mesmo SHA-256")

    artifacts: dict[str, Any] = {}
    for name, (quantization, uses_imatrix) in FINAL_ARTIFACTS.items():
        path = paths[name]
        require_gguf(path)
        artifacts[name] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "quantization": quantization,
            "uses_imatrix": uses_imatrix,
            "imatrix_sha256": sha256_file(paths["imatrix"]) if uses_imatrix else None,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": state,
        "build_progress": progress,
        "intermediates": {
            "bf16": {
                "filename": paths["bf16"].name,
                "bytes": paths["bf16"].stat().st_size,
                "sha256": sha256_file(paths["bf16"]),
            },
            "imatrix": {
                "filename": paths["imatrix"].name,
                "bytes": paths["imatrix"].stat().st_size,
                "sha256": sha256_file(paths["imatrix"]),
            },
        },
        "artifacts": artifacts,
        "comparison_contract": {
            "imatrix_effect": "compare q4_k_m versus q4_k_m_imatrix",
            "high_precision_reference": "q8_0",
            "single_file_gguf": True,
            "runtime_requires_imatrix_file": False,
        },
    }
    atomic_json(paths["manifest"], manifest)
    return manifest


def verify(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    paths = artifact_paths(output_dir, args.prefix)
    manifest = read_json(paths["manifest"])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Versao de schema inesperada no manifesto GGUF")
    if manifest.get("inputs") != read_json(paths["state"]):
        raise ValueError("Manifesto final diverge do manifesto de entradas")
    progress = read_json(paths["progress"])
    if manifest.get("build_progress") != progress:
        raise ValueError("Manifesto final diverge do manifesto de progresso")
    progress_artifacts = progress.get("artifacts")
    expected_progress_names = {"bf16", "imatrix", *FINAL_ARTIFACTS}
    if not isinstance(progress_artifacts, dict) or set(progress_artifacts) != expected_progress_names:
        raise ValueError("Manifesto de progresso nao contem exatamente todos os artefatos")
    for name, expected in progress_artifacts.items():
        path = paths[name]
        require_gguf(path)
        if (
            not isinstance(expected, dict)
            or expected.get("filename") != path.name
            or expected.get("bytes") != path.stat().st_size
            or expected.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Integridade diverge do progresso: {path}")
    if list(output_dir.rglob("*.safetensors")):
        raise ValueError("A pasta final GGUF contem Safetensors inesperado")
    intermediates = manifest.get("intermediates")
    if not isinstance(intermediates, dict) or set(intermediates) != {"bf16", "imatrix"}:
        raise ValueError("Manifesto nao contem exatamente os intermediarios esperados")
    for name in ("bf16", "imatrix"):
        expected = intermediates[name]
        if not isinstance(expected, dict) or expected.get("filename") != paths[name].name:
            raise ValueError(f"Intermediario divergente no manifesto: {name}")
        path = paths[name]
        require_gguf(path)
        if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"Integridade do intermediario diverge: {path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Manifesto nao contem um objeto artifacts")
    for name, expected in artifacts.items():
        if name not in FINAL_ARTIFACTS:
            raise ValueError(f"Artefato desconhecido no manifesto: {name}")
        quantization, uses_imatrix = FINAL_ARTIFACTS[name]
        if not isinstance(expected, dict) or expected.get("filename") != paths[name].name:
            raise ValueError(f"Nome do artefato diverge no manifesto: {name}")
        if expected.get("quantization") != quantization or expected.get("uses_imatrix") is not uses_imatrix:
            raise ValueError(f"Identidade do artefato diverge no manifesto: {name}")
        expected_imatrix = intermediates["imatrix"]["sha256"] if uses_imatrix else None
        if expected.get("imatrix_sha256") != expected_imatrix:
            raise ValueError(f"Vinculo com a imatrix diverge no manifesto: {name}")
        path = paths[name]
        require_gguf(path)
        if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"Integridade divergente: {path}")
    if set(artifacts) != set(FINAL_ARTIFACTS):
        raise ValueError("Manifesto nao contem exatamente os tres GGUF finais")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--llama-cpp-dir", type=Path)
    parser.add_argument("--llama-cpp-build-dir", type=Path)
    parser.add_argument("--llama-cpp-revision")
    parser.add_argument("--hf-model-dir", type=Path)
    parser.add_argument("--source-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--threads", type=positive, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--gpu-layers", type=positive, default=99)
    parser.add_argument("--context", type=positive, default=512)
    parser.add_argument("--batch", type=positive, default=512)
    parser.add_argument("--ubatch", type=positive, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_only:
        manifest = verify(args)
    else:
        missing = [
            name
            for name in ("llama_cpp_dir", "llama_cpp_revision", "hf_model_dir", "corpus", "corpus_manifest")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            parser.error("argumentos obrigatorios para build: " + ", ".join(missing))
        manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
