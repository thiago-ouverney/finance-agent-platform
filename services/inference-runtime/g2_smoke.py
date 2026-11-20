#!/usr/bin/env python3
"""Smoke closed-loop: quatro pedidos, dois em voo, um slot.

Sobe o llama-server com o mesmo comando da baseline, mede e encerra o processo.
Não baixa modelo nem tokenizer. Não é uma célula da curva.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import bench
import concurrent_load

ROOT = Path(__file__).resolve().parent
MODEL = Path("/workspace/models/Qwen2.5-7B-Instruct-Q8_0/Qwen2.5-7B-Instruct-Q8_0.gguf")
WORKLOAD = Path("/workspace/data/mopep-runtime-workload/calibration-30/workload.jsonl")
LLAMA = Path("/workspace/runtimes/llama.cpp/build/bin/llama-server")
HOST = "127.0.0.1"
PORT = 8080
ALIAS = "qwen2.5-7b-test"
SMOKE_REQUESTS = 4
SMOKE_STREAMS = 2
STARTUP_TIMEOUT_S = 180
REQUEST_TIMEOUT_S = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def port_is_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_http(url: str, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    last_error = "sem tentativa"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise TimeoutError(f"{url} não ficou pronto em {timeout_s}s: {last_error}")


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{url} não devolveu um objeto JSON")
    return payload


def launch_args() -> list[str]:
    return [
        str(LLAMA), "-m", str(MODEL), "--alias", ALIAS,
        "--host", HOST, "--port", str(PORT), "--ctx-size", "8192",
        "--n-gpu-layers", "999", "--parallel", "1", "--jinja",
    ]


def main() -> int:
    repo = ROOT.parents[1]
    print("git", subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
    print("cliente", sha256_file(Path(concurrent_load.__file__)))
    print("gguf", sha256_file(MODEL))
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip())
    if not port_is_free(HOST, PORT):
        print(f"porta {PORT} ocupada; o smoke não mata processo alheio")
        return 2

    log_path = Path("/tmp/g2-smoke-llama.log")
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                launch_args(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        base = f"http://{HOST}:{PORT}"
        status = wait_http(f"{base}/health", STARTUP_TIMEOUT_S)
        props = read_json(f"{base}/props")
        contexts = concurrent_load.collect_scalars(props, "n_ctx")
        slots = concurrent_load.collect_scalars(props, "total_slots")
        print("health", status)
        print("n_ctx", contexts)
        print("total_slots", slots)
        records = concurrent_load.cycle_records(
            concurrent_load.load_jsonl(WORKLOAD), SMOKE_REQUESTS
        )
        cfg = {"base_url": base, "model": ALIAS}

        def request(record, index):
            del index
            return bench.stream_messages_request(
                cfg, record["initial_messages"], REQUEST_TIMEOUT_S, "",
                metadata={"request_id": record["request_id"], "bucket": record["bucket"]},
            )

        report = concurrent_load.run_closed_loop(records, request, SMOKE_STREAMS)
        for index, (row, error) in enumerate(zip(report["successful"], report["errored"])):
            if error is not None:
                print(f"erro index={index} {type(error).__name__}: {error}")
                continue
            print(
                "ok",
                f"index={index}",
                f"bucket={row.get('bucket')}",
                f"e2e={row.get('request_latency')}",
                f"prompt_tokens={row.get('prompt_tokens')}",
            )
        print(
            "smoke",
            f"streams={SMOKE_STREAMS}",
            f"requests={SMOKE_REQUESTS}",
            f"ok={report['successful_count']}",
            f"erro={report['errored_count']}",
        )
        return 0 if report["errored_count"] == 0 and report["successful_count"] == SMOKE_REQUESTS else 1
    except Exception:
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("log-final")
            print("\n".join(lines[-30:]))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        print("servidor-encerrado", process.poll() if process is not None else "ausente")


if __name__ == "__main__":
    sys.exit(main())
