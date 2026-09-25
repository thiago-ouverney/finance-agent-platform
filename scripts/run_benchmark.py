"""Executa um alvo do benchmark localmente ou por SSH."""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path


VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]*=.*$")


def make_arguments(target: str, values: list[str]) -> list[str]:
    invalid = [value for value in values if not VARIABLE.match(value)]
    if invalid:
        raise ValueError(f"Variável Make inválida: {invalid[0]}")
    return ["make", target, *values]


def remote_command(remote_dir: str, arguments: list[str]) -> str:
    return f"cd {shlex.quote(remote_dir)} && {shlex.join(arguments)}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("local", "ssh"), required=True)
    result.add_argument("--target", default="bench-all")
    result.add_argument(
        "--repo-dir",
        default="services/inference-runtime",
        help="Diretório local do runtime.",
    )
    result.add_argument("--host", help="Alias SSH, host UFF ou host RunPod.")
    result.add_argument("--port", type=int, default=22)
    result.add_argument("--identity", type=Path)
    result.add_argument(
        "--remote-dir",
        default="/workspace/finance-agent-platform/services/inference-runtime",
    )
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return result


def main() -> None:
    args = parser().parse_args()
    make = make_arguments(args.target, args.set)
    if args.mode == "local":
        subprocess.run(make, cwd=Path(args.repo_dir), check=True)
        return
    if not args.host:
        raise SystemExit("--host é obrigatório no modo ssh")
    ssh = ["ssh", "-p", str(args.port)]
    if args.identity:
        ssh.extend(["-i", str(args.identity)])
    ssh.extend([args.host, remote_command(args.remote_dir, make)])
    subprocess.run(ssh, check=True)


if __name__ == "__main__":
    main()
