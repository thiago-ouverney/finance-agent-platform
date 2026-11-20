#!/usr/bin/env python3
"""Roda as células restantes de um plano e imprime o tempo que falta."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import concurrent_load
import g2_cell
import g2_smoke

ROOT = Path(__file__).resolve().parent
PRIOR_CELL_S = 100.0
PORT_WAIT_S = 60.0


def pending_cells(streams: tuple[int, ...], repetitions: int, done: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (streams_count, repetition)
        for streams_count in streams
        for repetition in range(1, repetitions + 1)
        if (streams_count, repetition) not in done
    ]


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def estimate(done_durations: list[float], remaining: int, prior_s: float = PRIOR_CELL_S) -> tuple[float, float]:
    average = sum(done_durations) / len(done_durations) if done_durations else prior_s
    return average, average * remaining


def cell_is_complete(directory: Path, requests: int) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    summary = json.loads(manifest_path.read_text(encoding="utf-8")).get("summary") or {}
    return (
        summary.get("successful_count") == requests
        and summary.get("errored_count") == 0
        and summary.get("prompt_tokens_missing") == 0
    )


def progress_line(index: int, total: int, streams: int, repetition: int, phase: str, elapsed_s: float, remaining: int, average_s: float) -> str:
    _, left_s = estimate([average_s], remaining, average_s)
    return (
        f"[{index}/{total}] {phase} streams={streams} repetição={repetition} "
        f"| decorrido {format_duration(elapsed_s)} "
        f"| média {format_duration(average_s)} "
        f"| faltam {remaining} "
        f"| estimativa {format_duration(left_s)}"
    )


def wait_port_free(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if g2_smoke.port_is_free(host, port):
            return True
        time.sleep(1)
    return False


def progress_log(parallel: int, ctx_size: int, cache: bool) -> Path:
    label = "cache-on" if cache else "cache-off"
    return ROOT / "results" / "g2" / f"p{parallel}-ctx{ctx_size}-{label}-progress.log"


def parse_plan(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--cache", choices=("on", "off"), default="on")
    args = parser.parse_args(argv)
    if (args.parallel, args.ctx_size) not in g2_cell.FROZEN_SLOTS:
        parser.error("a combinação de slots e contexto não está congelada")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_plan(sys.argv[1:] if argv is None else argv)
    parallel, ctx_size = args.parallel, args.ctx_size
    cache = args.cache == "on"
    cache_label = "on" if cache else "off"
    requests = concurrent_load.REQUESTS_PER_REPETITION
    done = {
        (streams, repetition)
        for streams in concurrent_load.STREAMS
        for repetition in range(1, concurrent_load.REPETITIONS + 1)
        if cell_is_complete(g2_cell.result_dir(parallel, ctx_size, cache, streams, repetition), requests)
    }
    cells = pending_cells(concurrent_load.STREAMS, concurrent_load.REPETITIONS, done)
    total = len(cells)
    log_path = progress_log(parallel, ctx_size, cache)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    started_arm = time.perf_counter()

    def emit(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    average_s, left_s = estimate(durations, total)
    emit(
        f"[0/{total}] plano parallel={parallel} ctx={ctx_size} cache={cache_label} "
        f"| {total} células "
        f"| estimativa inicial {format_duration(left_s)}"
    )
    emit(f"progresso {log_path}")
    for index, (streams, repetition) in enumerate(cells, 1):
        average_s, _ = estimate(durations, total - index + 1)
        if not wait_port_free(g2_smoke.HOST, g2_smoke.PORT, PORT_WAIT_S):
            emit(
                f"[{index}/{total}] parada streams={streams} repetição={repetition} "
                f"motivo=porta {g2_smoke.PORT} ocupada por mais de {format_duration(PORT_WAIT_S)} "
                f"| decorrido {format_duration(time.perf_counter() - started_arm)}"
            )
            return 2
        emit(progress_line(index, total, streams, repetition, "iniciando", time.perf_counter() - started_arm, total - index + 1, average_s))
        cell_started = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable, str(ROOT / "g2_cell.py"),
                "--parallel", str(parallel),
                "--ctx-size", str(ctx_size),
                "--streams", str(streams),
                "--requests", str(requests),
                "--repetition", str(repetition),
                "--cache", cache_label,
            ],
            cwd=ROOT,
            check=False,
        )
        elapsed = time.perf_counter() - cell_started
        if completed.returncode != 0:
            emit(
                f"[{index}/{total}] parada streams={streams} repetição={repetition} "
                f"status={completed.returncode} | decorrido {format_duration(time.perf_counter() - started_arm)}"
            )
            return completed.returncode
        durations.append(elapsed)
        average_s, _ = estimate(durations, total - index)
        emit(progress_line(index, total, streams, repetition, "concluída", time.perf_counter() - started_arm, total - index, average_s))
    emit(f"[{total}/{total}] plano concluído | decorrido {format_duration(time.perf_counter() - started_arm)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
