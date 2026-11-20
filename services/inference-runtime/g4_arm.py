#!/usr/bin/env python3
"""Plano curto G4 no joelho: parallel 8 via gateway, streams 1..8 × 3 reps."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import concurrent_load
import g2_arm
import g4_cell
import g2_smoke

ROOT = Path(__file__).resolve().parent
STREAMS = g4_cell.KNEE_STREAMS
REPETITIONS = concurrent_load.REPETITIONS
PORT_WAIT_S = 60.0


def main() -> int:
    requests = concurrent_load.REQUESTS_PER_REPETITION
    done = {
        (streams, repetition)
        for streams in STREAMS
        for repetition in range(1, REPETITIONS + 1)
        if g2_arm.cell_is_complete(g4_cell.result_dir(streams, repetition), requests)
    }
    cells = g2_arm.pending_cells(STREAMS, REPETITIONS, done)
    total = len(cells)
    log_path = ROOT / "results" / "g4" / "p8-ctx65536-cache-on-via-gateway-progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    started_arm = time.perf_counter()

    def emit(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    average_s, left_s = g2_arm.estimate(durations, total)
    emit(
        f"[0/{total}] plano g4 parallel=8 ctx=65536 via=gateway "
        f"| {total} células | estimativa inicial {g2_arm.format_duration(left_s)}"
    )
    emit(f"progresso {log_path}")
    for index, (streams, repetition) in enumerate(cells, 1):
        average_s, _ = g2_arm.estimate(durations, total - index + 1)
        if not g2_arm.wait_port_free(g2_smoke.HOST, g2_smoke.PORT, PORT_WAIT_S):
            emit(f"[{index}/{total}] parada motivo=porta {g2_smoke.PORT} ocupada")
            return 2
        if not g2_arm.wait_port_free(g2_smoke.HOST, g4_cell.GATEWAY_PORT, PORT_WAIT_S):
            emit(f"[{index}/{total}] parada motivo=porta {g4_cell.GATEWAY_PORT} ocupada")
            return 2
        emit(g2_arm.progress_line(
            index, total, streams, repetition, "iniciando",
            time.perf_counter() - started_arm, total - index + 1, average_s,
        ))
        cell_started = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable, str(ROOT / "g4_cell.py"),
                "--streams", str(streams),
                "--requests", str(requests),
                "--repetition", str(repetition),
            ],
            cwd=ROOT,
            check=False,
        )
        elapsed = time.perf_counter() - cell_started
        if completed.returncode != 0:
            emit(
                f"[{index}/{total}] parada streams={streams} repetição={repetition} "
                f"status={completed.returncode} | decorrido {g2_arm.format_duration(time.perf_counter() - started_arm)}"
            )
            return completed.returncode
        durations.append(elapsed)
        average_s, _ = g2_arm.estimate(durations, total - index)
        emit(g2_arm.progress_line(
            index, total, streams, repetition, "concluída",
            time.perf_counter() - started_arm, total - index, average_s,
        ))
    emit(f"[{total}/{total}] plano concluído | decorrido {g2_arm.format_duration(time.perf_counter() - started_arm)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
