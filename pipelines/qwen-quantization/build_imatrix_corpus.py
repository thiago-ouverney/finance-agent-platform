"""Build a deterministic llama.cpp imatrix corpus from the MOPEP golden train split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    from .calibration import (
        PROMPT_VERSION,
        calibration_metadata,
        load_golden_csv,
        render_calibration_texts,
        select_calibration_examples,
    )
except ImportError:
    from calibration import (
        PROMPT_VERSION,
        calibration_metadata,
        load_golden_csv,
        render_calibration_texts,
        select_calibration_examples,
    )


CORPUS_FORMAT_VERSION = "mopep-imatrix-corpus-v1"
_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_files(tokenizer_dir: Path) -> list[Path]:
    """Return tokenizer inputs without accidentally hashing model weights or caches."""
    files = []
    for item in tokenizer_dir.iterdir():
        if not item.is_file() or item.name.startswith("."):
            continue
        if (
            item.name in _TOKENIZER_FILENAMES
            or item.name.startswith("tokenizer")
            or item.name.startswith("chat_template")
        ):
            files.append(item)
    return sorted(files, key=lambda item: item.name)


def tokenizer_sha256(tokenizer_dir: Path) -> tuple[str, int]:
    """Hash tokenizer filenames and bytes in a stable order."""
    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Diretorio do tokenizer nao encontrado: {tokenizer_dir}")
    files = _tokenizer_files(tokenizer_dir)
    if not files:
        raise FileNotFoundError(f"Arquivos do tokenizer nao encontrados em: {tokenizer_dir}")

    digest = hashlib.sha256()
    for item in files:
        name = item.relative_to(tokenizer_dir).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def load_local_tokenizer(tokenizer_dir: Path) -> Any:
    """Load Transformers lazily and prohibit network fallback."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=False,
    )


def serialize_corpus(rendered_texts: list[str]) -> bytes:
    """Preserve chat-template special tokens and add deterministic boundaries."""
    if not rendered_texts:
        raise ValueError("Nenhum texto foi renderizado para o corpus imatrix")
    corpus = "\n\n".join(text.rstrip("\n") for text in rendered_texts) + "\n"
    return corpus.encode("utf-8")


def _target_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _write_temporary_sibling(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_pair_without_overwrite(
    corpus_path: Path,
    corpus_payload: bytes,
    manifest_path: Path,
    manifest_payload: bytes,
) -> None:
    """Atomically publish each complete file and roll back an incomplete pair."""
    corpus_temp = _write_temporary_sibling(corpus_path, corpus_payload)
    try:
        manifest_temp = _write_temporary_sibling(manifest_path, manifest_payload)
    except BaseException:
        corpus_temp.unlink(missing_ok=True)
        raise
    corpus_created = False
    manifest_created = False
    try:
        # Hard-linking a temporary sibling publishes complete bytes atomically and,
        # unlike os.replace(), fails when another process created the destination.
        os.link(corpus_temp, corpus_path)
        corpus_created = True
        os.link(manifest_temp, manifest_path)
        manifest_created = True
    except FileExistsError as exc:
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        if corpus_created:
            corpus_path.unlink(missing_ok=True)
        raise FileExistsError(f"Saida ja existe e nao sera sobrescrita: {exc.filename}") from exc
    except BaseException:
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        if corpus_created:
            corpus_path.unlink(missing_ok=True)
        raise
    finally:
        corpus_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)


def build_imatrix_corpus(
    dataset: Path,
    tokenizer_dir: Path,
    output: Path,
    manifest: Path,
    *,
    limit: int,
    seed: int,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """Render, hash, and atomically publish a private-data-free corpus manifest."""
    dataset = dataset.expanduser().resolve()
    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()

    if output == manifest:
        raise ValueError("O corpus e o manifesto devem usar caminhos diferentes")
    for target in (output, manifest):
        if _target_exists(target):
            raise FileExistsError(f"Saida ja existe e nao sera sobrescrita: {target}")
    if limit < 1:
        raise ValueError("O limite de calibracao deve ser maior que zero")
    if seed < 0:
        raise ValueError("A seed deve ser um inteiro nao negativo")

    corpus_payload, result = prepare_imatrix_corpus(
        dataset,
        tokenizer_dir,
        limit=limit,
        seed=seed,
        tokenizer=tokenizer,
    )
    manifest_payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_pair_without_overwrite(output, corpus_payload, manifest, manifest_payload)
    return result


def prepare_imatrix_corpus(
    dataset: Path,
    tokenizer_dir: Path,
    *,
    limit: int,
    seed: int,
    tokenizer: Any | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Render the corpus and its non-sensitive provenance without publishing it."""
    examples = load_golden_csv(dataset)
    if len(examples) < limit:
        raise ValueError(
            f"Dataset tem {len(examples)} exemplos, menos que os {limit} solicitados"
        )
    selected = select_calibration_examples(examples, limit=limit, seed=seed)
    tokenizer_digest, tokenizer_file_count = tokenizer_sha256(tokenizer_dir)
    active_tokenizer = tokenizer if tokenizer is not None else load_local_tokenizer(tokenizer_dir)
    rendered = render_calibration_texts(selected, active_tokenizer)
    corpus_payload = serialize_corpus(rendered)
    calibration = calibration_metadata(
        dataset,
        selected,
        available_count=len(examples),
        limit=limit,
        seed=seed,
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "encoding": "utf-8",
        "source_filename": dataset.name,
        "dataset_sha256": calibration["source_sha256"],
        "corpus_sha256": _sha256_bytes(corpus_payload),
        "tokenizer_sha256": tokenizer_digest,
        "tokenizer_file_count": tokenizer_file_count,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": calibration["prompt_sha256"],
        "selection_sha256": calibration["selection_sha256"],
        "available_examples": len(examples),
        "requested_examples": limit,
        "selected_examples": len(selected),
        "selection_seed": seed,
        "includes_expected_answer": True,
    }
    return corpus_payload, metadata


def verify_imatrix_corpus(
    dataset: Path,
    tokenizer_dir: Path,
    output: Path,
    manifest: Path,
    *,
    limit: int,
    seed: int,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """Recompute the corpus contract and reject stale or modified artifacts."""
    dataset = dataset.expanduser().resolve()
    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if not output.is_file() or not manifest.is_file():
        raise FileNotFoundError("Corpus ou manifesto imatrix ausente")
    expected_payload, expected = prepare_imatrix_corpus(
        dataset,
        tokenizer_dir,
        limit=limit,
        seed=seed,
        tokenizer=tokenizer,
    )
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or stored != expected:
        raise ValueError("Manifesto imatrix diverge do dataset, tokenizer ou parametros atuais")
    if output.read_bytes() != expected_payload:
        raise ValueError("Corpus imatrix diverge do manifesto e das entradas atuais")
    return expected


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("use zero ou um inteiro positivo")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=_nonnegative, default=42)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = verify_imatrix_corpus if args.verify_only else build_imatrix_corpus
    metadata = action(
        args.dataset,
        args.tokenizer_dir,
        args.output,
        args.manifest,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
