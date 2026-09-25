#!/usr/bin/env python3
"""Executa uma bateria de contexto/KV, reiniciando o servidor a cada ponto."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

# Executado como `python scripts/run_kv_sweep.py`, o Python coloca `scripts/`
# no sys.path, não a raiz do checkout onde fica results_layout.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results_layout import artifact, prepare, runtime_root


def stop_process_group(process: subprocess.Popen, timeout: float = 20) -> None:
    """Encerra o ponto do sweep e todos os descendentes que ele criou."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("use zero ou um inteiro positivo")
    return number


def launch_with_context(source: Path, target: Path, context: int, context_flag: str, extra_args: list[str]) -> None:
    args = json.loads(source.read_text())
    if not isinstance(args, list):
        raise ValueError("launch JSON deve ser uma lista de argumentos")
    if context_flag != "none":
        if context_flag in args:
            index = args.index(context_flag) + 1
            args[index] = str(context)
        else:
            args.extend([context_flag, str(context)])
    args.extend(extra_args)
    target.write_text(json.dumps(args, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--start", type=positive, default=1024)
    parser.add_argument("--memory-step-mb", type=positive, default=256,
                        help="incremento de memória lógica do KV em MB decimais")
    parser.add_argument("--kv-bytes-per-token", type=positive, required=True,
                        help="bytes de KV lógico por token para a arquitetura/dtype do modelo")
    parser.add_argument("--max-context", type=positive, default=16384)
    parser.add_argument("--requests", type=positive, default=50)
    parser.add_argument("--repetitions", type=positive, default=1)
    parser.add_argument("--warmup", type=nonnegative, default=3)
    parser.add_argument("--mode", choices=["independent", "closed-loop", "replay"], default="replay")
    parser.add_argument("--conversation-turns", type=positive, default=1)
    parser.add_argument("--conversation-fixture", default="workloads/conversations/qwen_chat_bench_v2.json")
    parser.add_argument("--warmup-conversation-fixture", default="workloads/conversations/qwen_chat_warmup_v1.json")
    parser.add_argument("--startup-timeout", type=positive, default=1800)
    parser.add_argument("--results", default="results/kv-sweep")
    parser.add_argument("--runtime-label", default="runtime")
    parser.add_argument("--context-flag", default="--max-model-len",
                        help="Flag do servidor que controla contexto; use none quando ele é configurado fora do argv.")
    parser.add_argument("--launch-extra-args", nargs="*", default=[],
                        help="Argumentos adicionais acrescentados ao launch em todos os pontos.")
    parser.add_argument("--launch-executable", help="Substitui argv[0] do launch em todos os pontos.")
    parser.add_argument("--launch-extra-args-json", default="[]")
    args = parser.parse_args()
    memory_step_bytes = args.memory_step_mb * 1_000_000
    token_step = max(1, math.ceil(memory_step_bytes / args.kv_bytes_per_token))
    extra = json.loads(args.launch_extra_args_json)
    if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
        parser.error("--launch-extra-args-json deve ser array de strings")
    args.launch_extra_args.extend(extra)
    config = json.loads(Path(args.config).read_text())
    root = Path(args.results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    sweep_output = runtime_root(root, args.runtime_label, stamp,
                                f"sweep-contextos-{args.start}-{args.max_context}-tokens-step-{args.memory_step_mb}MB")
    prepare(sweep_output)
    manifest = {"runtime": args.runtime_label, "memory_step_mb": args.memory_step_mb,
                "kv_bytes_per_token": args.kv_bytes_per_token, "token_step": token_step,
                "start": args.start,
                "max_context": args.max_context, "mode": args.mode,
                "conversation_turns": args.conversation_turns,
                "conversation_fixture": args.conversation_fixture, "points": []}
    with tempfile.TemporaryDirectory(prefix="kv-sweep-") as temp:
        temp = Path(temp)
        for context in range(args.start, args.max_context + 1, token_step):
            # Não imponha ao servidor um teto artificial de `context + 128 + 256`.
            # O workload seleciona históricos que cabem no ponto; o runtime mantém
            # pelo menos o context_window já configurado no launcher/config.
            config_copy = dict(config)
            configured_context = int(config.get("context_window") or 0)
            server_context = max(configured_context, context + 128 + 256)
            config_copy["context_window"] = server_context
            if config.get("runtime") == "ollama":
                subprocess.run([args.python, str(Path(__file__).with_name("prepare_ollama.py")),
                                "--binary", args.launch_executable or "ollama",
                                "--model", config["model"], "--gguf", args.local_model_path,
                                "--context", str(server_context),
                                "--base-url", config["base_url"]], check=True)
            config_path = temp / f"config-{context}.json"
            launch_path = temp / f"launch-{context}.json"
            config_path.write_text(json.dumps(config_copy, indent=2) + "\n")
            launch_with_context(Path(args.launch), launch_path, server_context,
                                args.context_flag, args.launch_extra_args)
            command = [args.python, "bench.py", "run", "--config", str(config_path),
                       "--local-model-path", args.local_model_path, "--launch", str(launch_path),
                       "--input-tokens", str(context), "--requests", str(args.requests),
                       "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
                       "--mode", args.mode, "--conversation-turns", str(args.conversation_turns),
                       "--conversation-fixture", args.conversation_fixture,
                       "--warmup-conversation-fixture", args.warmup_conversation_fixture,
                       "--result-name", f"contexto-{context}-tokens-step-{args.memory_step_mb}MB",
                       "--collect-kv-metrics", "--startup-timeout", str(args.startup_timeout),
                       "--results", str(root)]
            if args.launch_executable:
                command.extend(["--launch-executable", args.launch_executable])
            estimated_mb = context * args.kv_bytes_per_token / 1_000_000
            print(f"\n[KV-SWEEP] runtime={args.runtime_label} input_tokens={context} "
                  f"estimated_logical_kv_mb={estimated_mb:.2f} "
                  f"server_max_model_len={server_context} "
                  f"requests={args.requests} repetitions={args.repetitions} "
                  f"mode={args.mode} turns={args.conversation_turns}", flush=True)
            child = subprocess.Popen(command, start_new_session=True)
            try:
                returncode = child.wait()
            except BaseException:
                # Ctrl-C/SIGTERM no make não pode deixar vLLM, workers ou
                # o bench vivos no Pod. O filho inteiro recebe o sinal.
                stop_process_group(child)
                raise
            finally:
                # Também limpa descendentes caso o bench tenha falhado antes
                # de executar seu próprio finally.
                stop_process_group(child)
            manifest["points"].append({"input_tokens": context, "returncode": returncode})
            artifact(sweep_output, "sweep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if returncode != 0:
                print(f"[KV-SWEEP] falha em input_tokens={context}; consulte server.log. Não classificada automaticamente como OOM.")
                return returncode
    print("Varredura concluída; consulte cada diretório e sweep-manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
