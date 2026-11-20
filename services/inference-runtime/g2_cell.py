#!/usr/bin/env python3
"""Uma célula da curva closed-loop.

Sobe o llama-server, mede N pedidos com no máximo ``streams`` em voo e encerra
o processo. A janela medida começa depois do health e termina no último pedido.
"""
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
import g2_smoke

ROOT = Path(__file__).resolve().parent
FROZEN_SLOTS = {(1, 8192), (8, 65536), (16, 131072)}


def cell_launch_args(binary: str, model: str, parallel: int, ctx_size: int, cache: bool) -> list[str]:
    args = [
        binary, "-m", model, "--alias", g2_smoke.ALIAS,
        "--host", g2_smoke.HOST, "--port", str(g2_smoke.PORT),
        "--ctx-size", str(ctx_size), "--n-gpu-layers", "999",
        "--parallel", str(parallel), "--jinja",
    ]
    if not cache:
        args.append("--no-cache-prompt")
    return args


def cell_summary(successful: list[dict], errored_count: int, wall_s: float) -> dict:
    if wall_s <= 0:
        raise ValueError("a janela medida precisa ter duração positiva")
    e2e = [row.get("request_latency") for row in successful]
    ttft = [row.get("time_to_first_token_ms") for row in successful]
    tokens = [row.get("output_tokens") for row in successful if isinstance(row.get("output_tokens"), int)]
    return {
        "successful_count": len(successful),
        "errored_count": errored_count,
        "wall_seconds": wall_s,
        "requests_per_second": len(successful) / wall_s,
        "output_tokens_per_second": (sum(tokens) / wall_s) if tokens else None,
        "ttft_ms_p50": bench.percentile(ttft, 0.50),
        "ttft_ms_p95": bench.percentile(ttft, 0.95),
        "e2e_s_p50": bench.percentile(e2e, 0.50),
        "e2e_s_p95": bench.percentile(e2e, 0.95),
        "prompt_tokens_missing": sum(not isinstance(row.get("prompt_tokens"), int) for row in successful),
        "percentile_definition": "Empirical linear interpolation over successful requests in this cell.",
    }


def result_dir(parallel: int, ctx_size: int, cache: bool, streams: int, repetition: int) -> Path:
    cache_label = "cache-on" if cache else "cache-off"
    return ROOT / "results" / "g2" / f"p{parallel}-ctx{ctx_size}-{cache_label}" / f"s{streams}-r{repetition}"


def props_match(parallel: int, ctx_size: int, contexts: list, slots: list) -> bool:
    """`/props` desta revisão publica o contexto de cada slot, não o total do `--ctx-size`."""
    if parallel not in slots or not contexts or parallel < 1 or ctx_size % parallel != 0:
        return False
    per_slot = ctx_size // parallel
    return all(value == per_slot for value in contexts)


def discard_incomplete(destination: Path) -> None:
    if not destination.exists():
        return
    protected = {"manifest.json", "requests.jsonl"}
    names = {child.name for child in destination.iterdir()}
    if names & protected or any(not (destination / name).is_file() for name in names):
        raise FileExistsError(destination)
    for name in names:
        (destination / name).unlink()
    destination.rmdir()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--ctx-size", type=int, required=True)
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--requests", type=int, default=concurrent_load.REQUESTS_PER_REPETITION)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--cache", choices=("on", "off"), default="on")
    args = parser.parse_args(argv)
    if (args.parallel, args.ctx_size) not in FROZEN_SLOTS:
        parser.error("a combinação de slots e contexto não está congelada")
    if args.streams not in concurrent_load.STREAMS:
        parser.error("streams fora da lista congelada")
    if args.requests < 1:
        parser.error("requests deve ser positivo")
    if args.repetition < 1:
        parser.error("repetition deve ser positiva")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cache = args.cache == "on"
    repo = ROOT.parents[1]
    print("git", subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
    print("cliente", g2_smoke.sha256_file(Path(concurrent_load.__file__)))
    print("celula", g2_smoke.sha256_file(Path(__file__)))
    print("gguf", g2_smoke.sha256_file(g2_smoke.MODEL))
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip())
    print(
        "plano",
        f"parallel={args.parallel}",
        f"ctx={args.ctx_size}",
        f"streams={args.streams}",
        f"requests={args.requests}",
        f"repetition={args.repetition}",
        f"cache={args.cache}",
    )
    if not g2_smoke.port_is_free(g2_smoke.HOST, g2_smoke.PORT):
        print(f"porta {g2_smoke.PORT} ocupada; a célula não mata processo alheio")
        return 2

    destination = result_dir(args.parallel, args.ctx_size, cache, args.streams, args.repetition)
    try:
        discard_incomplete(destination)
    except FileExistsError:
        print("célula já tem manifesto ou pedidos; a medição não recomeçou")
        return 2
    destination.mkdir(parents=True, exist_ok=False)
    log_path = destination / "server.log"
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                cell_launch_args(str(g2_smoke.LLAMA), str(g2_smoke.MODEL), args.parallel, args.ctx_size, cache),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        base = f"http://{g2_smoke.HOST}:{g2_smoke.PORT}"
        health = g2_smoke.wait_http(f"{base}/health", g2_smoke.STARTUP_TIMEOUT_S)
        props = g2_smoke.read_json(f"{base}/props")
        contexts = concurrent_load.collect_scalars(props, "n_ctx")
        slots = concurrent_load.collect_scalars(props, "total_slots")
        print("health", health)
        print("n_ctx", contexts)
        print("n_ctx_slot_esperado", args.ctx_size // args.parallel)
        print("total_slots", slots)
        if not props_match(args.parallel, args.ctx_size, contexts, slots):
            print("props divergem do plano; a medição não começou")
            return 3
        records = concurrent_load.cycle_records(concurrent_load.load_jsonl(g2_smoke.WORKLOAD), args.requests)
        cfg = {"base_url": base, "model": g2_smoke.ALIAS}

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
        health_after = g2_smoke.wait_http(f"{base}/health", 10)
        successful = [row for row in report["successful"] if row is not None]
        summary = cell_summary(successful, report["errored_count"], wall_s)
        manifest = {
            "git": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "client_sha256": g2_smoke.sha256_file(Path(concurrent_load.__file__)),
            "cell_sha256": g2_smoke.sha256_file(Path(__file__)),
            "gguf_sha256": g2_smoke.sha256_file(g2_smoke.MODEL),
            "parallel": args.parallel,
            "ctx_size": args.ctx_size,
            "streams": args.streams,
            "requests": args.requests,
            "repetition": args.repetition,
            "cache": args.cache,
            "n_ctx": contexts,
            "total_slots": slots,
            "health_before": health,
            "health_after": health_after,
            "summary": summary,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (destination / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for row, error in zip(report["successful"], report["errored"]):
                if error is not None:
                    handle.write(json.dumps({"status": "errored", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False) + "\n")
                else:
                    kept = {
                        "status": "ok",
                        "curve_index": row.get("curve_index"),
                        "request_id": row.get("request_id"),
                        "bucket": row.get("bucket"),
                        "request_latency": row.get("request_latency"),
                        "time_to_first_token_ms": row.get("time_to_first_token_ms"),
                        "prompt_tokens": row.get("prompt_tokens"),
                        "output_tokens": row.get("output_tokens"),
                        "output": row.get("output"),
                    }
                    handle.write(json.dumps(kept, ensure_ascii=False) + "\n")
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
        return 0 if summary["errored_count"] == 0 and summary["successful_count"] == args.requests and summary["prompt_tokens_missing"] == 0 else 1
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
