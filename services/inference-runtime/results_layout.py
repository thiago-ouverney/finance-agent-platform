"""Layout legível e estável para artefatos de uma execução."""
from __future__ import annotations

import re
from pathlib import Path


DIRECTORIES = {
    ".html": "html",
    ".svg": "html",
    ".json": "json",
    ".csv": "csv",
    ".log": "logs",
}


def slug(value: str) -> str:
    """Converte labels de runtime/execução em nomes seguros e legíveis."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip().lower())
    return re.sub(r"-{2,}", "-", value).strip("-._") or "execucao"


def prepare(root: Path) -> Path:
    root = Path(root)
    for directory in {"html", "json", "csv", "logs", "text"}:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def artifact(root: Path, name: str) -> Path:
    """Retorna o caminho categorizado de um artefato.

    O lock fica na raiz da execução-base; os demais arquivos são separados por
    formato para facilitar navegação/download sem depender de um file browser.
    """
    root = Path(root)
    if name == ".benchmark.lock":
        return root / name
    suffix = Path(name).suffix.lower()
    directory = DIRECTORIES.get(suffix, "text")
    return root / directory / name


def locate(root: Path, name: str) -> Path:
    """Lê o layout novo e, para compatibilidade, o layout flat antigo."""
    candidate = artifact(root, name)
    return candidate if candidate.exists() else Path(root) / name


def href(name: str) -> str:
    """Link relativo a partir da pasta html."""
    suffix = Path(name).suffix.lower()
    return f"../{DIRECTORIES.get(suffix, 'text')}/{name}"


def execution_label(*, smoke: bool, scenarios: list[str], input_tokens: list[int] | None,
                    result_name: str | None = None) -> str:
    if result_name:
        return slug(result_name)
    if input_tokens:
        return "contexto-" + "-".join(str(item) for item in input_tokens) + "-tokens"
    prefix = "smoke" if smoke else "benchmark"
    return prefix + "-" + "-".join(slug(item) for item in scenarios)


def runtime_root(base: Path, runtime: str, stamp: str, label: str) -> Path:
    return Path(base) / slug(runtime) / stamp / slug(label)
