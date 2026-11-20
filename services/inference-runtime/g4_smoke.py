#!/usr/bin/env python3
"""Smoke do joelho através do gateway: llama parallel 8 + gateway 8090."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import bench
import concurrent_load
import g2_smoke

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GATEWAY_ROOT = REPO / "services" / "inference-gateway"
GATEWAY_PYTHON = GATEWAY_ROOT / ".venv" / "bin" / "python"
GATEWAY_PORT = 8090
PARALLEL = 8
CTX = 65536
STREAMS = 8
REQUESTS = 8


def read_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8", errors="replace")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "ausente (extract sem .git)"


def main() -> int:
    print("git", git_head())
    print("cliente", g2_smoke.sha256_file(Path(concurrent_load.__file__)))
    print("gguf", g2_smoke.sha256_file(g2_smoke.MODEL))
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip())
    if not GATEWAY_PYTHON.is_file():
        print(f"python do gateway ausente: {GATEWAY_PYTHON}")
        return 2
    if not g2_smoke.port_is_free(g2_smoke.HOST, g2_smoke.PORT):
        print(f"porta {g2_smoke.PORT} ocupada")
        return 2
    if not g2_smoke.port_is_free(g2_smoke.HOST, GATEWAY_PORT):
        print(f"porta {GATEWAY_PORT} ocupada")
        return 2

    llama_log = Path("/tmp/g4-smoke-llama.log")
    gateway_log = Path("/tmp/g4-smoke-gateway.log")
    llama = gateway = None
    try:
        with llama_log.open("w", encoding="utf-8") as log:
            llama = subprocess.Popen(
                [
                    str(g2_smoke.LLAMA), "-m", str(g2_smoke.MODEL),
                    "--alias", g2_smoke.ALIAS, "--host", g2_smoke.HOST,
                    "--port", str(g2_smoke.PORT), "--ctx-size", str(CTX),
                    "--n-gpu-layers", "999", "--parallel", str(PARALLEL), "--jinja",
                ],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        llama_base = f"http://{g2_smoke.HOST}:{g2_smoke.PORT}"
        print("health_llama", g2_smoke.wait_http(f"{llama_base}/health", g2_smoke.STARTUP_TIMEOUT_S))
        props = g2_smoke.read_json(f"{llama_base}/props")
        print("n_ctx", concurrent_load.collect_scalars(props, "n_ctx"))
        print("n_ctx_slot_esperado", CTX // PARALLEL)
        print("total_slots", concurrent_load.collect_scalars(props, "total_slots"))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(GATEWAY_ROOT)
        with gateway_log.open("w", encoding="utf-8") as log:
            gateway = subprocess.Popen(
                [
                    str(GATEWAY_PYTHON), str(GATEWAY_ROOT / "run_gateway.py"),
                    "--upstream", llama_base,
                    "--host", g2_smoke.HOST, "--port", str(GATEWAY_PORT),
                    "--max-inflight", str(PARALLEL), "--max-queue", str(PARALLEL),
                ],
                cwd=str(GATEWAY_ROOT), env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        gateway_base = f"http://{g2_smoke.HOST}:{GATEWAY_PORT}"
        print("health_gateway", g2_smoke.wait_http(f"{gateway_base}/health", 60))
        print("metrics_before")
        print(read_text(f"{gateway_base}/metrics").strip())

        records = concurrent_load.cycle_records(concurrent_load.load_jsonl(g2_smoke.WORKLOAD), REQUESTS)
        cfg = {"base_url": gateway_base, "model": g2_smoke.ALIAS}

        def request(record, index):
            row = bench.stream_messages_request(
                cfg, record["initial_messages"], g2_smoke.REQUEST_TIMEOUT_S, "",
                metadata={
                    "request_id": record["request_id"],
                    "bucket": record["bucket"],
                    "curve_index": index,
                },
            )
            row["curve_index"] = index
            return row

        started = time.perf_counter()
        report = concurrent_load.run_closed_loop(records, request, STREAMS)
        wall = time.perf_counter() - started
        ok = report["successful_count"]
        err = report["errored_count"]
        print(f"smoke streams={STREAMS} requests={REQUESTS} ok={ok} erro={err} wall_s={wall:.3f}")
        print("health_after", g2_smoke.wait_http(f"{gateway_base}/health", 10))
        print("metrics_after")
        print(read_text(f"{gateway_base}/metrics").strip())
        return 0 if err == 0 and ok == REQUESTS else 1
    except Exception:
        for path in (llama_log, gateway_log):
            if path.exists():
                print(f"log-final {path}")
                print("\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
        raise
    finally:
        for process in (gateway, llama):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        print("gateway-encerrado", gateway.poll() if gateway is not None else "ausente")
        print("servidor-encerrado", llama.poll() if llama is not None else "ausente")


if __name__ == "__main__":
    sys.exit(main())
