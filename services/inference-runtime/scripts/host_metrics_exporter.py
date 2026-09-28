#!/usr/bin/env python3
"""Expõe telemetria leve do host/GPU no formato Prometheus."""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def sample(name: str, value, labels: dict[str, str] | None = None) -> str | None:
    number = finite_number(value)
    if number is None:
        return None
    suffix = ""
    if labels:
        encoded = ",".join(f'{key}="{escape_label(str(item))}"' for key, item in sorted(labels.items()))
        suffix = "{" + encoded + "}"
    return f"{name}{suffix} {number}\n"


def read_vmstat(path: str | Path = "/proc/vmstat") -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, raw = line.split(maxsplit=1)
            values[key] = int(raw)
    except (OSError, ValueError):
        return {}
    return values


def query_gpus(binary: str = "nvidia-smi") -> tuple[bool, list[list[str]]]:
    command = [
        binary,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False, []
    if result.returncode != 0:
        return False, []
    return True, list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def collect_metrics(nvidia_smi: str = "nvidia-smi") -> str:
    """Coleta valores atuais; ausências são omitidas, nunca convertidas em zero."""
    lines = [
        "# HELP inference_exporter_up Exporter conseguiu coletar métricas do host.\n",
        "# TYPE inference_exporter_up gauge\n",
        "inference_exporter_up 1\n",
    ]

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_io_counters()
    host_values = {
        "inference_host_cpu_utilization_ratio": psutil.cpu_percent(interval=None) / 100,
        "inference_host_memory_used_bytes": vm.used,
        "inference_host_memory_available_bytes": vm.available,
        "inference_host_memory_total_bytes": vm.total,
        "inference_host_swap_used_bytes": swap.used,
        "inference_host_swap_total_bytes": swap.total,
        "inference_host_swap_in_bytes_total": swap.sin,
        "inference_host_swap_out_bytes_total": swap.sout,
    }
    if disk is not None:
        host_values.update(
            inference_host_disk_read_bytes_total=disk.read_bytes,
            inference_host_disk_write_bytes_total=disk.write_bytes,
        )
    vmstat = read_vmstat()
    if "pgfault" in vmstat:
        host_values["inference_host_page_faults_total"] = vmstat["pgfault"]
    if "pgmajfault" in vmstat:
        host_values["inference_host_major_page_faults_total"] = vmstat["pgmajfault"]
    for name, value in host_values.items():
        rendered = sample(name, value)
        if rendered:
            lines.append(rendered)

    gpu_ok, rows = query_gpus(nvidia_smi)
    lines.append(sample("inference_gpu_scrape_success", 1 if gpu_ok else 0))
    for row in rows:
        if len(row) < 8:
            continue
        labels = {"gpu": row[0], "name": row[1]}
        values = {
            "inference_gpu_memory_used_bytes": finite_number(row[2]) * 1048576 if finite_number(row[2]) is not None else None,
            "inference_gpu_memory_total_bytes": finite_number(row[3]) * 1048576 if finite_number(row[3]) is not None else None,
            "inference_gpu_utilization_ratio": finite_number(row[4]) / 100 if finite_number(row[4]) is not None else None,
            "inference_gpu_memory_controller_utilization_ratio": finite_number(row[5]) / 100 if finite_number(row[5]) is not None else None,
            "inference_gpu_temperature_celsius": row[6],
            "inference_gpu_power_watts": row[7],
        }
        for name, value in values.items():
            rendered = sample(name, value, labels)
            if rendered:
                lines.append(rendered)
    return "".join(line for line in lines if line)


class MetricsHandler(BaseHTTPRequestHandler):
    nvidia_smi = "nvidia-smi"

    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path in {"/-/healthy", "/healthz"}:
            payload = b"ok\n"
        elif self.path == "/metrics":
            payload = collect_metrics(self.nvidia_smi).encode("utf-8")
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=9108)
    result.add_argument("--nvidia-smi", default="nvidia-smi")
    result.add_argument("--once", action="store_true", help="Imprime uma coleta e encerra.")
    return result


def main() -> None:
    args = parser().parse_args()
    psutil.cpu_percent(interval=None)
    if args.once:
        print(collect_metrics(args.nvidia_smi), end="")
        return
    MetricsHandler.nvidia_smi = args.nvidia_smi
    server = ThreadingHTTPServer((args.host, args.port), MetricsHandler)
    print(f"host_metrics_exporter=http://{args.host}:{args.port}/metrics", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
