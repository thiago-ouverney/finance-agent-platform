#!/usr/bin/env python3
"""Acompanha, sem alterar arquivos, uma campanha observe-bench-all em execução."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any


RUNTIMES = ("vllm", "llama", "ollama")


def normalize_runtime(value: object) -> str:
    text = str(value or "").lower()
    if text in {"llama", "llama.cpp", "llamacpp", "llama-cpp"}:
        return "llama"
    return text


def parse_utc(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z") and "T" in value and "-" not in value[:8]:
        return datetime.strptime(value, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def read_last_event(run_dir: Path) -> str | None:
    path = run_dir / "csv" / "events.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (FileNotFoundError, OSError, csv.Error):
        return None
    return rows[-1].get("phase") if rows else None


def request_progress(run_dir: Path) -> tuple[int, int, int, int]:
    """Retorna concluídas, esperadas, falhas e blocos persistidos."""
    path = run_dir / "csv" / "summary.csv"
    complete = expected = failed = blocks = 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                blocks += 1
                complete += int(float(row.get("successful_request_count") or 0))
                failed += int(float(row.get("errored_request_count") or 0))
                failed += int(float(row.get("incomplete_request_count") or 0))
                expected += int(float(row.get("expected") or 0))
    except (FileNotFoundError, OSError, csv.Error, ValueError):
        return 0, 0, 0, 0
    return complete, expected, failed, blocks


def expected_blocks(manifest: dict[str, Any]) -> int | None:
    repetitions = int(manifest.get("repetitions") or 1)
    profile = manifest.get("benchmark_profile", "generic")
    if profile == "generic":
        return len(manifest.get("scenarios") or []) * repetitions * 2
    request_dataset = manifest.get("request_dataset") or {}
    bucket_counts = request_dataset.get("bucket_counts") or {}
    measured = sum(1 for value in bucket_counts.values() if int(value or 0) > 0)
    return repetitions * (1 + measured) if measured else None


@dataclass(frozen=True)
class Run:
    runtime: str
    smoke: bool
    profile: str
    status: str
    started: datetime
    run_id: str
    run_dir: Path
    phase: str | None
    completed_requests: int
    persisted_expected_requests: int
    failed_requests: int
    completed_blocks: int
    planned_blocks: int | None


def run_from_manifest(path: Path) -> Run | None:
    manifest = read_json(path)
    if not manifest or "experiment_id" not in manifest or "smoke" not in manifest:
        return None
    config = manifest.get("config") or {}
    runtime = normalize_runtime(config.get("runtime") or manifest.get("runtime"))
    if runtime not in RUNTIMES:
        return None
    try:
        started = parse_utc(str(manifest.get("started_utc")))
    except (TypeError, ValueError):
        return None
    run_dir = path.parent.parent if path.parent.name == "json" else path.parent
    complete, expected, failed, blocks = request_progress(run_dir)
    return Run(
        runtime=runtime,
        smoke=bool(manifest.get("smoke")),
        profile=str(manifest.get("benchmark_profile") or "generic"),
        status=str(manifest.get("status") or "unknown").lower(),
        started=started,
        run_id=str(manifest.get("experiment_id")),
        run_dir=run_dir,
        phase=read_last_event(run_dir),
        completed_requests=complete,
        persisted_expected_requests=expected,
        failed_requests=failed,
        completed_blocks=blocks,
        planned_blocks=expected_blocks(manifest),
    )


def discover(results: Path, cutoff: datetime, profile: str) -> dict[tuple[bool, str], Run]:
    latest: dict[tuple[bool, str], Run] = {}
    if not results.is_dir():
        return latest
    for path in results.rglob("manifest.json"):
        run = run_from_manifest(path)
        if not run or run.started < cutoff or run.profile != profile:
            continue
        key = (run.smoke, run.runtime)
        if key not in latest or run.started > latest[key].started:
            latest[key] = run
    return latest


def active_processes() -> list[str]:
    labels: set[str] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    for cmdline in proc.glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "observe-bench-all" in command:
            labels.add("orquestrador observe-bench-all")
        if "bench.py run" in command:
            labels.add("cliente bench.py")
        if "vllm serve" in command:
            labels.add("servidor vLLM")
        if "llama-server" in command:
            labels.add("servidor llama.cpp")
        if "ollama serve" in command:
            labels.add("servidor Ollama")
    return sorted(labels)


def stage_order() -> list[tuple[bool, str]]:
    return [(smoke, runtime) for smoke in (True, False) for runtime in RUNTIMES]


def stage_line(smoke: bool, runtime: str, run: Run | None) -> str:
    kind = "smoke " if smoke else "formal"
    if run is None:
        return f"· {kind:<7} {runtime:<6} pendente"
    symbol = {"complete": "✓", "running": "▶", "failed": "✗"}.get(run.status, "?")
    details = []
    if run.phase and run.status == "running":
        details.append(f"fase={run.phase}")
    block_total = str(run.planned_blocks) if run.planned_blocks is not None else "?"
    details.append(f"blocos={run.completed_blocks}/{block_total}")
    if run.persisted_expected_requests:
        details.append(
            f"req={run.completed_requests}/{run.persisted_expected_requests}"
        )
    if run.failed_requests:
        details.append(f"falhas={run.failed_requests}")
    details.append(f"run_id={run.run_id}")
    return f"{symbol} {kind:<7} {runtime:<6} {run.status:<8} " + "  ".join(details)


def render(results: Path, cutoff: datetime, profile: str) -> tuple[str, bool, bool]:
    runs = discover(results, cutoff, profile)
    lines = [
        f"Campanha observe-bench-all · perfil={profile}",
        f"Resultados: {results.resolve()}",
        f"Considerando runs desde: {cutoff.isoformat()}",
    ]
    processes = active_processes()
    lines.append("Processos ativos: " + (", ".join(processes) if processes else "nenhum reconhecido"))
    lines.append("")
    ordered = stage_order()
    for smoke, runtime in ordered:
        lines.append(stage_line(smoke, runtime, runs.get((smoke, runtime))))
    complete = sum(run.status == "complete" for run in runs.values())
    failed = sum(run.status == "failed" for run in runs.values())
    running = sum(run.status == "running" for run in runs.values())
    lines.extend(["", f"Resumo: completos={complete}/6  rodando={running}  falhos={failed}  pendentes={6-len(runs)}"])
    next_stage = next((key for key in ordered if key not in runs or runs[key].status != "complete"), None)
    if next_stage:
        run = runs.get(next_stage)
        state = run.status if run else "pendente"
        lines.append(f"Próximo estágio não concluído: {'smoke' if next_stage[0] else 'formal'} {next_stage[1]} ({state})")
    else:
        lines.append("Campanha concluída: os seis estágios terminaram com status complete.")
    all_complete = all(runs.get(key) and runs[key].status == "complete" for key in ordered)
    any_failed = any(run.status == "failed" for run in runs.values())
    return "\n".join(lines), all_complete, any_failed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results", type=Path, default=Path("results"))
    result.add_argument(
        "--profile",
        default="generic",
        choices=("generic", "mopep-single", "mopep-review-replay", "mopep-review-closed-loop"),
    )
    time_group = result.add_mutually_exclusive_group()
    time_group.add_argument("--since", help="Início UTC/ISO da campanha, por exemplo 2026-09-28T21:00:00Z.")
    time_group.add_argument("--since-minutes", type=float, default=240)
    result.add_argument("--watch", type=float, default=0, metavar="SEGUNDOS")
    result.add_argument("--no-clear", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.watch < 0 or args.since_minutes is not None and args.since_minutes <= 0:
        raise SystemExit("--watch deve ser >= 0 e --since-minutes deve ser > 0.")
    now = datetime.now(timezone.utc)
    cutoff = parse_utc(args.since) if args.since else now - timedelta(minutes=args.since_minutes)
    first = True
    try:
        while True:
            output, complete, failed = render(args.results, cutoff, args.profile)
            if args.watch and not args.no_clear and sys.stdout.isatty() and not first:
                print("\033[2J\033[H", end="")
            print(output, flush=True)
            first = False
            if not args.watch or complete or failed:
                return 1 if failed else 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nMonitor encerrado; a campanha não foi alterada.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
