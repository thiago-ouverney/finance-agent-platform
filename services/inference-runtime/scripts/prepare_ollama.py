#!/usr/bin/env python3
"""Cria um alias Ollama a partir de um GGUF local, sem baixar ou carregar o modelo."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number


def api_ready(base_url: str, timeout: float = 1.0) -> bool:
    """Retorna verdadeiro somente quando a API Ollama responde com sucesso."""
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def wait_for_api(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if api_ready(base_url):
            return
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"ollama serve encerrou antes de ficar pronto (código {returncode})")
        time.sleep(0.1)
    raise TimeoutError(f"API Ollama não ficou pronta em {timeout:g}s: {base_url}")


def stop_owned_server(process: subprocess.Popen[bytes]) -> None:
    """Encerra somente o grupo criado por este programa."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_command(command: list[str], env: dict[str, str], log) -> None:
    log.write(("$ " + " ".join(json.dumps(part) for part in command) + "\n").encode())
    log.flush()
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def prepare(
    binary: str,
    model: str,
    gguf: Path,
    context: int,
    base_url: str,
    timeout: float,
    results: Path = Path("results"),
) -> Path:
    gguf = gguf.expanduser().resolve()
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF local não encontrado: {gguf}")

    results.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = results / f"prepare-ollama-{stamp}.log"
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    owned_process: subprocess.Popen[bytes] | None = None

    with log_path.open("ab", buffering=0) as log:
        try:
            if not api_ready(base_url):
                log.write(f"$ {json.dumps(binary)} \"serve\"\n".encode())
                owned_process = subprocess.Popen(
                    [binary, "serve"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_for_api(base_url, owned_process, timeout)
            else:
                log.write(b"API Ollama preexistente detectada; daemon preservado.\n")

            with tempfile.TemporaryDirectory(prefix="prepare-ollama-") as temp_dir:
                modelfile = Path(temp_dir) / "Modelfile"
                modelfile.write_text(
                    f"FROM {json.dumps(str(gguf))}\nPARAMETER num_ctx {context}\n",
                    encoding="utf-8",
                )
                run_command([binary, "create", model, "-f", str(modelfile)], env, log)
                # `show` consulta metadados; não faz inferência nem pré-carrega os pesos.
                run_command([binary, "show", model], env, log)
        finally:
            if owned_process is not None:
                stop_owned_server(owned_process)
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="Executável ou caminho do cliente Ollama")
    parser.add_argument("--model", required=True, help="Alias local a criar ou atualizar")
    parser.add_argument("--gguf", required=True, type=Path, help="Arquivo GGUF local; nenhum download é feito")
    parser.add_argument("--context", required=True, type=positive, help="Valor de num_ctx gravado no alias")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=positive, default=30, help="Limite para o daemon temporário ficar pronto")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Diretório do log de preparação")
    args = parser.parse_args(argv)
    log_path = prepare(
        args.binary, args.model, args.gguf, args.context, args.base_url, args.timeout, args.results
    )
    print(f"Alias Ollama preparado sem inferência. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
