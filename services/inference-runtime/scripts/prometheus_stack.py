#!/usr/bin/env python3
"""Inicia e encerra Prometheus + exporter local sem depender de Docker."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def render_config(exporter_target: str, interval: str, runtime_target: str | None = None,
                  runtime_metrics_path: str = "/metrics", timeout: str = "450ms") -> str:
    jobs = [f'''  - job_name: inference-host
    static_configs:
      - targets: ["{exporter_target}"]
''']
    if runtime_target:
        jobs.append(f'''  - job_name: inference-runtime
    metrics_path: "{runtime_metrics_path}"
    static_configs:
      - targets: ["{runtime_target}"]
''')
    return f'''global:
  scrape_interval: {interval}
  scrape_timeout: {timeout}
  evaluation_interval: {interval}

scrape_configs:
{''.join(jobs)}'''


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_matches(pid: int, expected: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return expected in command


def wait_url(url: str, timeout: float = 20) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    return False


def assert_port_available(host: str, port: int, service: str) -> None:
    """Falha antes do launch quando o endpoint solicitado já está reservado."""
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RuntimeError(f"Host inválido para {service}: {host}: {exc}") from exc
    if not addresses:
        raise RuntimeError(f"Nenhum endereço resolvido para {service}: {host}:{port}")

    checked = set()
    for family, socktype, protocol, _canonical_name, address in addresses:
        key = (family, socktype, protocol, address)
        if key in checked:
            continue
        checked.add(key)
        probe = socket.socket(family, socktype, protocol)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(address)
        except OSError as exc:
            raise RuntimeError(
                f"Porta de {service} indisponível em {host}:{port}; "
                "pare o serviço existente ou escolha outra porta."
            ) from exc
        finally:
            probe.close()


def wait_owned_url(url: str, process: subprocess.Popen, pid_path: Path,
                   service: str, timeout: float = 20) -> None:
    """Aceita readiness somente enquanto o filho e seu PID continuam válidos."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"{service} encerrou antes do readiness (exit={returncode})."
            )
        if read_pid(pid_path) != process.pid:
            raise RuntimeError(
                f"PID de {service} mudou durante o startup; readiness recusado."
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if 200 <= response.status < 300:
                    if process.poll() is not None or read_pid(pid_path) != process.pid:
                        raise RuntimeError(
                            f"{service} deixou de ser o processo ativo durante o readiness."
                        )
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{service} não ficou pronto em {timeout:g}s: {url}")


def launch(command: list[str], log_path: Path, pid_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    pid_path.write_text(str(process.pid) + "\n", encoding="utf-8")
    return process


def terminate_launched(process: subprocess.Popen | None, pid_path: Path) -> None:
    """Limpa somente o filho conhecido e remove seu PID mesmo após startup falho."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    pid_path.unlink(missing_ok=True)


def stop_process(pid_path: Path, expected: str) -> str:
    pid = read_pid(pid_path)
    if pid is None:
        return "not-running"
    if not process_matches(pid, expected):
        pid_path.unlink(missing_ok=True)
        return "stale-pid-not-signalled"
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process_matches(pid, expected):
        time.sleep(0.1)
    if process_matches(pid, expected):
        os.kill(pid, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)
    return "stopped"


def start(args) -> None:
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    exporter_pid = state / "exporter.pid"
    prometheus_pid = state / "prometheus.pid"
    if (read_pid(exporter_pid) and process_matches(read_pid(exporter_pid), "host_metrics_exporter.py")) or (
        read_pid(prometheus_pid) and process_matches(read_pid(prometheus_pid), "prometheus")
    ):
        raise RuntimeError("A pilha Prometheus já está ativa; use status ou stop.")
    if not args.prometheus_bin.is_file() or not os.access(args.prometheus_bin, os.X_OK):
        raise RuntimeError(f"Prometheus não encontrado: {args.prometheus_bin}")
    # Verifica as duas portas antes de criar qualquer processo. Além de tornar o
    # erro acionável, isso impede que o readiness de um serviço preexistente seja
    # confundido com o processo lançado abaixo.
    assert_port_available(args.exporter_host, args.exporter_port, "exporter")
    assert_port_available(args.prometheus_host, args.prometheus_port, "Prometheus")
    config = state / "prometheus.yml"
    config.write_text(render_config(
        f"{args.exporter_host}:{args.exporter_port}", args.scrape_interval,
        args.runtime_metrics_target, args.runtime_metrics_path, args.scrape_timeout,
    ), encoding="utf-8")
    exporter = launch([
        str(args.python), str(ROOT / "scripts" / "host_metrics_exporter.py"),
        "--host", args.exporter_host, "--port", str(args.exporter_port),
        "--nvidia-smi", args.nvidia_smi,
    ], state / "exporter.log", exporter_pid)
    prometheus = None
    try:
        wait_owned_url(
            f"http://{args.exporter_host}:{args.exporter_port}/-/healthy",
            exporter, exporter_pid, "Exporter",
        )
        prometheus = launch([
            str(args.prometheus_bin),
            f"--config.file={config}",
            f"--storage.tsdb.path={state / 'data'}",
            f"--storage.tsdb.retention.time={args.retention}",
            f"--web.listen-address={args.prometheus_host}:{args.prometheus_port}",
        ], state / "prometheus.log", prometheus_pid)
        wait_owned_url(
            f"http://{args.prometheus_host}:{args.prometheus_port}/-/ready",
            prometheus, prometheus_pid, "Prometheus",
        )
    except BaseException:
        terminate_launched(prometheus, prometheus_pid)
        terminate_launched(exporter, exporter_pid)
        raise
    metadata = {
        "prometheus_url": f"http://{args.prometheus_host}:{args.prometheus_port}",
        "exporter_url": f"http://{args.exporter_host}:{args.exporter_port}/metrics",
        "runtime_metrics_target": args.runtime_metrics_target,
        "scrape_interval": args.scrape_interval,
    }
    (state / "stack.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def stop(args) -> None:
    state = args.state_dir.resolve()
    result = {
        "prometheus": stop_process(state / "prometheus.pid", "prometheus"),
        "exporter": stop_process(state / "exporter.pid", "host_metrics_exporter.py"),
    }
    print(json.dumps(result, indent=2))


def status(args) -> None:
    state = args.state_dir.resolve()
    values = {
        "prometheus_process": bool((pid := read_pid(state / "prometheus.pid")) and process_matches(pid, "prometheus")),
        "exporter_process": bool((pid := read_pid(state / "exporter.pid")) and process_matches(pid, "host_metrics_exporter.py")),
        "prometheus_ready": wait_url(f"http://{args.prometheus_host}:{args.prometheus_port}/-/ready", timeout=1),
        "exporter_ready": wait_url(f"http://{args.exporter_host}:{args.exporter_port}/-/healthy", timeout=1),
    }
    print(json.dumps(values, indent=2))
    if not all(values.values()):
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", type=Path, default=ROOT / "observability" / ".state")
    common.add_argument("--prometheus-host", default="127.0.0.1")
    common.add_argument("--prometheus-port", type=int, default=9090)
    common.add_argument("--exporter-host", default="127.0.0.1")
    common.add_argument("--exporter-port", type=int, default=9108)
    start_parser = sub.add_parser("start", parents=[common])
    start_parser.add_argument("--prometheus-bin", type=Path, required=True)
    start_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    start_parser.add_argument("--nvidia-smi", default="nvidia-smi")
    start_parser.add_argument("--scrape-interval", default="1s")
    start_parser.add_argument("--scrape-timeout", default="450ms")
    start_parser.add_argument("--retention", default="6h")
    start_parser.add_argument("--runtime-metrics-target")
    start_parser.add_argument("--runtime-metrics-path", default="/metrics")
    sub.add_parser("stop", parents=[common])
    sub.add_parser("status", parents=[common])
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "start":
        start(args)
    elif args.command == "stop":
        stop(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
