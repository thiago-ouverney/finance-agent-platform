"""Create a model repo with the requested visibility and upload one artifact folder."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_flag(value: str, name: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{name} deve ser 1 ou 0")


def _check_upload_contents(folder: Path) -> None:
    forbidden_names = {".env", "quantization.env", "train.csv", "calibration.csv", "test.csv"}
    unsafe = sorted(
        str(path.relative_to(folder))
        for path in folder.rglob("*")
        if path.is_symlink()
        or path.is_file()
        and (
            path.name in forbidden_names
            or path.name.casefold().startswith(".env")
            or path.name.casefold()
            in {"credentials.json", "private.json", "hf-token.txt", "hf_token.txt"}
            or path.suffix.casefold() in {".key", ".pem"}
            or path.suffix.casefold() == ".log"
            or (path.suffix.casefold() == ".csv" and path.name != "quant_log.csv")
        )
    )
    if unsafe:
        raise ValueError(f"Arquivos privados/inesperados na pasta de upload: {', '.join(unsafe)}")


def publish(
    folder: Path,
    repo_id: str,
    *,
    private: bool,
    allow_public: bool,
    replace_existing: bool,
    commit_message: str,
) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN ausente")
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Pasta do modelo ausente: {folder}")
    if not private and not allow_public:
        raise ValueError("Upload publico exige --allow-public 1")
    _check_upload_contents(folder)

    api = HfApi(token=token)
    existing_files: list[str] = []
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=False)
    else:
        current_private = bool(info.private)
        if current_private != private:
            expected = "privado" if private else "publico"
            actual = "privado" if current_private else "publico"
            raise ValueError(
                f"O repositorio existente e {actual}, mas QUANT_HF_PRIVATE pede {expected}. "
                "Altere a visibilidade no Hub conscientemente antes do upload."
            )
        existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        meaningful_existing = [name for name in existing_files if name != ".gitattributes"]
        if meaningful_existing and not replace_existing:
            raise ValueError(
                "O repositorio HF ja contem arquivos. Use um repo novo ou confirme "
                "conscientemente com QUANT_HF_REPLACE_EXISTING=1."
            )

    local_files = {
        str(path.relative_to(folder)).replace(os.sep, "/")
        for path in folder.rglob("*")
        if path.is_file()
    }
    stale_files = sorted(set(existing_files) - local_files - {".gitattributes"})
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message,
        delete_patterns=stale_files or None,
    )
    print(f"Upload concluido: {commit.commit_url}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--private", default="1")
    parser.add_argument("--allow-public", default="0")
    parser.add_argument("--replace-existing", default="0")
    parser.add_argument("--commit-message", default="Upload quantized model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publish(
        args.model_dir,
        args.repo_id,
        private=parse_flag(args.private, "--private"),
        allow_public=parse_flag(args.allow_public, "--allow-public"),
        replace_existing=parse_flag(args.replace_existing, "--replace-existing"),
        commit_message=args.commit_message,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
