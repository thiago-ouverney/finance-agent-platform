#!/usr/bin/env python3
"""Baixa e verifica um binário oficial do Prometheus para Linux amd64."""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_VERSION = "3.13.3"
DEFAULT_SHA256 = "b349c732d8a853e657d0e7ae1bbad4d11b586615fb65fdc59d896b9f869c001e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install(version: str, expected_sha256: str, destination: Path, url: str | None = None) -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("O instalador experimental suporta somente Linux amd64, que é o alvo do Pod.")
    destination.mkdir(parents=True, exist_ok=True)
    prometheus = destination / "prometheus"
    promtool = destination / "promtool"
    if prometheus.is_file() and os.access(prometheus, os.X_OK) and promtool.is_file():
        print(f"Prometheus já instalado em {destination}")
        return
    archive_url = url or (
        f"https://github.com/prometheus/prometheus/releases/download/v{version}/"
        f"prometheus-{version}.linux-amd64.tar.gz"
    )
    with tempfile.TemporaryDirectory(prefix="prometheus-install-") as temp:
        archive = Path(temp) / "prometheus.tar.gz"
        print(f"Baixando {archive_url}", flush=True)
        with urllib.request.urlopen(archive_url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        observed = sha256(archive)
        if observed.lower() != expected_sha256.lower():
            raise RuntimeError(f"SHA-256 inesperado: esperado={expected_sha256} observado={observed}")
        prefix = f"prometheus-{version}.linux-amd64/"
        with tarfile.open(archive, "r:gz") as bundle:
            members = {member.name: member for member in bundle.getmembers()}
            for name in ("prometheus", "promtool"):
                member = members.get(prefix + name)
                if member is None or not member.isfile():
                    raise RuntimeError(f"Arquivo {name} ausente no pacote oficial.")
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Não foi possível extrair {name}.")
                target = destination / name
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755)
    print(f"Prometheus {version} instalado em {destination}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", default=DEFAULT_VERSION)
    result.add_argument("--sha256", default=DEFAULT_SHA256)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--url")
    return result


def main() -> None:
    args = parser().parse_args()
    install(args.version, args.sha256, args.destination, args.url)


if __name__ == "__main__":
    main()
