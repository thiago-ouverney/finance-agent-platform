#!/usr/bin/env python3
"""Uma célula da curva G4: llama + gateway, medição pelo gateway."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import bench
import concurrent_load
import g2_cell
import g2_smoke

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GATEWAY_ROOT = REPO / "services" / "inference-gateway"
GATEWAY_PYTHON = GATEWAY_ROOT / ".venv" / "bin" / "python"
GATEWAY_PORT = 8090
KNEE = (8, 65536)
KNEE_STREAMS = (1, 2, 4, 6, 8)


def result_dir(streams: int, repetition: int) -> Path:
    return ROOT / "results" / "g4" / "p8-ctx65536-cache-on-via-gateway" / f"s{streams}-r{repetition}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--requests", type=int, default=concurrent_load.REQUESTS_PER_REPETITION)
    parser.add_argument("--repetition", type=int, required=True)
    args = parser.parse_args(argv)
    if args.streams not in KNEE_STREAMS:
        parser.error("streams fora do joelho G4 (1,2,4,6,8)")
    if args.requests < 1 or args.repetition < 1:
        parser.error("requests e repetition devem ser positivos")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    parallel, ctx_size = KNEE
    if not GATEWAY_PYTHON.is_file():
        print(f"python do gateway ausente: {GATEWAY_PYTHON}")
        return 2
    print("cliente", g2_smoke.sha256_file(Path(concurrent_load.__file__)))
    print("celula", g2_smoke.sha256_file(Path(__file__)))
    print("gguf", g2_smoke.sha256_file(g2_smoke.MODEL))
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip())
    print(
        f"plano parallel={parallel} ctx={ctx_size} streams={args.streams} "
        f"requests={args.requests} repetition={args.repetition} via=gateway:{GATEWAY_PORT}"
    )
    if not g2_smoke.port_is_free(g2_smoke.HOST, g2_smoke.PORT):
        print(f"porta {g2_smoke.PORT} ocupada")
        return 2
    if not g2_smoke.port_is_free(g2_smoke.HOST, GATEWAY_PORT):
        print(f"porta {GATEWAY_PORT} ocupada")
        return 2

    destination = result_dir(args.streams, args.repetition)
    try:
        g2_cell.discard_incomplete(destination)
    except FileExistsError:
        print("célula já tem manifesto ou pedidos; a medição não recomeçou")
        return 2
    destination.mkdir(parents=True, exist_ok=False)
    llama_log = destination / "llama.log"
    gateway_log = destination / "gateway.log"
    llama = gateway = None
    try:
        with llama_log.open("w", encoding="utf-8") as log:
            llama = subprocess.Popen(
                g2_cell.cell_launch_args(
                    str(g2_smoke.LLAMA), str(g2_smoke.MODEL), parallel, ctx_size, True
                ),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        llama_base = f"http://{g2_smoke.HOST}:{g2_smoke.PORT}"
        health = g2_smoke.wait_http(f"{llama_base}/health", g2_smoke.STARTUP_TIMEOUT_S)
        props = g2_smoke.read_json(f"{llama_base}/props")
        contexts = concurrent_load.collect_scalars(props, "n_ctx")
        slots = concurrent_load.collect_scalars(props, "total_slots")
        print("health_llama", health)
        print("n_ctx", contexts)
        print("n_ctx_slot_esperado", ctx_size // parallel)
        print("total_slots", slots)
        if not g2_cell.props_match(parallel, ctx_size, contexts, slots):
            print("props divergem do plano; a medição não começou")
            return 3

        env = os.environ.copy()
        env["PYTHONPATH"] = str(GATEWAY_ROOT)
        with gateway_log.open("w", encoding="utf-8") as log:
            gateway = subprocess.Popen(
                [
                    str(GATEWAY_PYTHON), str(GATEWAY_ROOT / "run_gateway.py"),
                    "--upstream", llama_base,
                    "--host", g2_smoke.HOST, "--port", str(GATEWAY_PORT),
                    "--max-inflight", str(parallel), "--max-queue", str(parallel),
                ],
                cwd=str(GATEWAY_ROOT), env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        gateway_base = f"http://{g2_smoke.HOST}:{GATEWAY_PORT}"
        print("health_gateway", g2_smoke.wait_http(f"{gateway_base}/health", 60))

        records = concurrent_load.cycle_records(concurrent_load.load_jsonl(g2_smoke.WORKLOAD), args.requests)
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
        report = concurrent_load.run_closed_loop(records, request, args.streams)
        wall_s = time.perf_counter() - started
        health_after = g2_smoke.wait_http(f"{gateway_base}/health", 10)
        successful = [row for row in report["successful"] if row is not None]
        summary = g2_cell.cell_summary(successful, report["errored_count"], wall_s)
        manifest = {
            "via": "gateway",
            "gateway_port": GATEWAY_PORT,
            "parallel": parallel,
            "ctx_size": ctx_size,
            "streams": args.streams,
            "requests": args.requests,
            "repetition": args.repetition,
            "cache": "on",
            "n_ctx": contexts,
            "total_slots": slots,
            "health_before": health,
            "health_after": health_after,
            "client_sha256": g2_smoke.sha256_file(Path(concurrent_load.__file__)),
            "cell_sha256": g2_smoke.sha256_file(Path(__file__)),
            "gguf_sha256": g2_smoke.sha256_file(g2_smoke.MODEL),
            "summary": summary,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (destination / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for row, error in zip(report["successful"], report["errored"]):
                if error is not None:
                    handle.write(json.dumps(
                        {"status": "errored", "error": f"{type(error).__name__}: {error}"},
                        ensure_ascii=False,
                    ) + "\n")
                else:
                    handle.write(json.dumps(
                        {
                            "status": "ok",
                            "curve_index": row.get("curve_index"),
                            "request_id": row.get("request_id"),
                            "bucket": row.get("bucket"),
                            "request_latency": row.get("request_latency"),
                            "time_to_first_token_ms": row.get("time_to_first_token_ms"),
                            "prompt_tokens": row.get("prompt_tokens"),
                            "output_tokens": row.get("output_tokens"),
                            "output": row.get("output"),
                        },
                        ensure_ascii=False,
                    ) + "\n")
        print("health_after", health_after)
        print(
            "celula",
            f"ok={summary['successful_count']}",
            f"erro={summary['errored_count']}",
            f"wall_s={summary['wall_seconds']:.3f}",
            f"req_s={summary['requests_per_second']:.3f}",
            f"tok_s={summary['output_tokens_per_second']:.3f}" if summary["output_tokens_per_second"] is not None else "tok_s=ausente",
            f"ttft_p50={summary['ttft_ms_p50']}",
            f"ttft_p95={summary['ttft_ms_p95']}",
            f"e2e_p50={summary['e2e_s_p50']}",
            f"e2e_p95={summary['e2e_s_p95']}",
            f"prompt_tokens_ausentes={summary['prompt_tokens_missing']}",
        )
        print("resultado", destination)
        return 0 if (
            summary["errored_count"] == 0
            and summary["successful_count"] == args.requests
            and summary["prompt_tokens_missing"] == 0
        ) else 1
    except Exception:
        for path in (llama_log, gateway_log):
            if path.exists():
                print(f"log-final {path.name}")
                print("\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
        raise
    finally:
        for process, label in ((gateway, "gateway-encerrado"), (llama, "servidor-encerrado")):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            print(label, process.poll() if process is not None else "ausente")


if __name__ == "__main__":
    sys.exit(main())
