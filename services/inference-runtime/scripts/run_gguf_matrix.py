#!/usr/bin/env python3
"""Executa três GGUFs nos três runtimes, estritamente em sequência."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
from typing import Iterable

from render_gguf_variant import tokenizer_metadata_sha256


EXPECTED_VARIANTS = ("q8_0", "q4_k_m_base", "q4_k_m_imatrix")
EXPECTED_ALIASES = {
    "q8_0": "qwen25-7b-q8-0",
    "q4_k_m_base": "qwen25-7b-q4-k-m-base",
    "q4_k_m_imatrix": "qwen25-7b-q4-k-m-imatrix",
}
EXPECTED_IDENTITY = {
    "q8_0": ("Q8_0", False),
    "q4_k_m_base": ("Q4_K_M", False),
    "q4_k_m_imatrix": ("Q4_K_M", True),
}
RUNTIME_KEYS = ("vllm", "llama", "ollama")
# Três blocos ortogonais: em cada bloco aparecem os três runtimes e variantes,
# e cada fator ocupa uma posição diferente nos três blocos.
SCHEDULE = (
    ("vllm", "q8_0"),
    ("llama", "q4_k_m_base"),
    ("ollama", "q4_k_m_imatrix"),
    ("llama", "q4_k_m_imatrix"),
    ("ollama", "q8_0"),
    ("vllm", "q4_k_m_base"),
    ("ollama", "q4_k_m_base"),
    ("vllm", "q4_k_m_imatrix"),
    ("llama", "q8_0"),
)
CELL_TARGETS = {
    "smoke": {
        "vllm": "gguf-smoke-vllm",
        "llama": "gguf-smoke-llama",
        "ollama": "gguf-smoke-ollama",
    },
    "formal": {
        "vllm": "gguf-base-vllm",
        "llama": "gguf-base-llama",
        "ollama": "gguf-base-ollama",
    },
}
PROTECTED_MAKE_VARS = {
    "MODEL_SIZE",
    "PREPARE_OFFLINE",
    "MODEL_7B_GGUF",
    "MODEL_7B_TOKENIZER",
    "VLLM_MODEL_DIR",
    "VLLM_CONFIG",
    "VLLM_LAUNCH",
    "LLAMA_MODEL_DIR",
    "LLAMA_CONFIG",
    "LLAMA_LAUNCH",
    "OLLAMA_MODEL_DIR",
    "OLLAMA_CONFIG",
    "OLLAMA_LAUNCH",
    "OLLAMA_MODEL_NAME",
    "GGUF_VARIANT",
    "GGUF_RESULT_NAME",
    "GGUF_COMPARISON_CONTEXT",
    "GGUF_COMPARISON_RESULTS",
}
ALLOWED_MAKE_VARS = {
    "BENCH_SCENARIOS",
    "BENCH_REQUESTS",
    "BENCH_REPETITIONS",
    "BENCH_WARMUP",
    "BENCH_MODE",
    "CONVERSATION_TURNS",
    "CONVERSATION_FIXTURE",
    "WARMUP_CONVERSATION_FIXTURE",
    "BENCH_STARTUP_TIMEOUT",
    "SWEEP_START",
    "SWEEP_MEMORY_STEP_MB",
    "SWEEP_MAX_CONTEXT",
    "SWEEP_REQUESTS",
    "SWEEP_REPETITIONS",
    "SWEEP_WARMUP",
    "SWEEP_MODE",
    "SWEEP_CONVERSATION_TURNS",
    "SWEEP_CONVERSATION_FIXTURE",
    "VLLM_EXTRA_ARGS",
    "LLAMA_EXTRA_ARGS",
    "OLLAMA_EXTRA_ARGS",
    "VLLM_BIN",
    "VLLM_PYTHON",
    "LLAMA_SERVER_BIN",
    "LLAMA_BACKEND_PATH",
    "REQUIRE_GPU",
    "OLLAMA_BIN",
    "PYTHON",
    "SYSTEM_PYTHON",
}
IDENTITY_FLAGS = {
    "VLLM_EXTRA_ARGS": {"--model", "--tokenizer", "--served-model-name", "--load-format", "--quantization"},
    "LLAMA_EXTRA_ARGS": {"-m", "--model", "--alias"},
}
RENDER_MAKE_VAR_KEYS = {
    "GGUF_Q8_FILE",
    "GGUF_Q4_BASE_FILE",
    "GGUF_Q4_IMATRIX_FILE",
    "GGUF_COMPARISON_MANIFEST",
    "MODEL_7B_TOKENIZER",
    "GGUF_COMPARISON_CONTEXT",
    "GGUF_COMPARISON_GPU_MEMORY_UTILIZATION",
    "GGUF_COMPARISON_PYTHON",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gguf_signature(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read(4)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_profiles(profiles_dir: Path, verify_sha: bool = True) -> dict[str, dict[str, object]]:
    profiles_dir = profiles_dir.expanduser().resolve()
    profiles: dict[str, dict[str, object]] = {}
    for variant in EXPECTED_VARIANTS:
        path = profiles_dir / variant / "profile.json"
        if not path.is_file():
            raise FileNotFoundError(f"Perfil ausente: {path}")
        profile = json.loads(path.read_text(encoding="utf-8"))
        if profile.get("schema_version") != 1 or profile.get("variant") != variant:
            raise ValueError(f"Perfil inválido ou de outra variante: {path}")
        if profile.get("alias") != EXPECTED_ALIASES[variant]:
            raise ValueError(f"Alias inesperado no perfil {variant}: {profile.get('alias')!r}")
        quantization, uses_imatrix = EXPECTED_IDENTITY[variant]
        if (profile.get("quantization"), profile.get("imatrix")) != (quantization, uses_imatrix):
            raise ValueError(f"Quantização/imatrix inesperada no perfil {variant}")
        gguf = Path(str(profile.get("gguf_path", "")))
        tokenizer = Path(str(profile.get("tokenizer_path", "")))
        if not gguf.is_absolute() or not gguf.is_file():
            raise ValueError(f"GGUF deve existir e usar caminho absoluto em {path}")
        if not tokenizer.is_absolute() or not tokenizer.is_dir():
            raise ValueError(f"Tokenizer deve existir e usar caminho absoluto em {path}")
        tokenizer_digest, tokenizer_file_count = tokenizer_metadata_sha256(tokenizer)
        if profile.get("tokenizer_metadata_sha256") != tokenizer_digest:
            raise ValueError(f"Metadados do tokenizer divergiram desde a renderização: {tokenizer}")
        if profile.get("tokenizer_metadata_file_count") != tokenizer_file_count:
            raise ValueError(f"Inventário do tokenizer divergiu desde a renderização: {tokenizer}")
        if gguf_signature(gguf) != b"GGUF":
            raise ValueError(f"Assinatura GGUF inválida: {gguf}")
        declared = profile.get("gguf_sha256")
        if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
            raise ValueError(f"SHA-256 inválido no perfil {variant}")
        if verify_sha and sha256_file(gguf) != declared:
            raise ValueError(f"SHA-256 divergiu desde a renderização: {gguf}")
        source = profile.get("source_manifest")
        if not isinstance(source, dict) or set(source) != {"path", "sha256", "artifact_key"}:
            raise ValueError(f"Manifesto de origem obrigatório/inválido no perfil {variant}")
        source_path = Path(str(source["path"]))
        if not source_path.is_absolute() or not source_path.is_file():
            raise ValueError(f"Manifesto de origem ausente: {source_path}")
        if verify_sha and sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Manifesto de origem divergiu desde a renderização: {source_path}")
        source_data = json.loads(source_path.read_text(encoding="utf-8"))
        source_artifacts = source_data.get("artifacts") if isinstance(source_data, dict) else None
        source_artifact = (
            source_artifacts.get(source["artifact_key"])
            if isinstance(source_artifacts, dict)
            else None
        )
        expected_source = {
            "filename": gguf.name,
            "sha256": declared,
            "quantization": quantization,
            "uses_imatrix": uses_imatrix,
        }
        if not isinstance(source_artifact, dict) or any(
            source_artifact.get(key) != value for key, value in expected_source.items()
        ):
            raise ValueError(f"Artefato diverge do manifesto de origem no perfil {variant}")
        files = profile.get("files")
        if not isinstance(files, dict) or set(files) != set(RUNTIME_KEYS):
            raise ValueError(f"Perfil {variant} não contém os três runtimes")
        for runtime in RUNTIME_KEYS:
            pair = files[runtime]
            expected_keys = {"config", "config_sha256", "launch", "launch_sha256"}
            if not isinstance(pair, dict) or set(pair) != expected_keys:
                raise ValueError(f"Arquivos inválidos para {variant}/{runtime}")
            for name in ("config", "launch"):
                generated = Path(str(pair[name]))
                if not generated.is_absolute() or not generated.is_file():
                    raise ValueError(f"Arquivo gerado ausente ou não absoluto: {generated}")
                expected_sha = pair[f"{name}_sha256"]
                if verify_sha and sha256_file(generated) != expected_sha:
                    raise ValueError(f"Arquivo gerado divergiu desde a renderização: {generated}")
        validate_generated_files(profile)
        profiles[variant] = profile

    paths = [str(profile["gguf_path"]) for profile in profiles.values()]
    digests = [str(profile["gguf_sha256"]) for profile in profiles.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("As três variantes devem apontar para três arquivos distintos")
    if len(set(digests)) != len(digests):
        raise ValueError("As três variantes devem ter SHA-256 distintos; nenhum fallback é permitido")
    tokenizers = {str(profile["tokenizer_path"]) for profile in profiles.values()}
    tokenizer_digests = {profile["tokenizer_metadata_sha256"] for profile in profiles.values()}
    contexts = {profile["context_window"] for profile in profiles.values()}
    if len(tokenizers) != 1 or len(tokenizer_digests) != 1 or len(contexts) != 1:
        raise ValueError("As três variantes devem usar o mesmo tokenizer e contexto")
    return profiles


def validate_generated_files(profile: dict[str, object]) -> None:
    """Confirma que config e launch apontam para o artefato declarado, sem fallback."""
    files = profile["files"]
    assert isinstance(files, dict)
    alias = str(profile["alias"])
    gguf = str(profile["gguf_path"])
    tokenizer = str(profile["tokenizer_path"])
    digest = str(profile["gguf_sha256"])
    context = profile.get("context_window")
    expected_runtime = {"vllm": "vllm", "llama": "llama.cpp", "ollama": "ollama"}
    for runtime in RUNTIME_KEYS:
        pair = files[runtime]
        config = json.loads(Path(pair["config"]).read_text(encoding="utf-8"))
        launch = json.loads(Path(pair["launch"]).read_text(encoding="utf-8"))
        if config.get("runtime") != expected_runtime[runtime]:
            raise ValueError(f"Runtime divergente no config {runtime}")
        if config.get("model") != alias or config.get("tokenizer") != tokenizer:
            raise ValueError(f"Alias/tokenizer divergente no config {runtime}")
        if config.get("context_window") != context:
            raise ValueError(f"Contexto divergente no config {runtime}")
        if digest not in str(config.get("model_artifact", "")):
            raise ValueError(f"SHA do artefato ausente no config {runtime}")
        if not isinstance(launch, list) or not launch or any(not isinstance(item, str) for item in launch):
            raise ValueError(f"Launch inválido para {runtime}")
        if runtime in {"vllm", "llama"}:
            if gguf not in launch or alias not in launch:
                raise ValueError(f"Launch {runtime} não usa o GGUF/alias declarado")
            foreign_ggufs = [item for item in launch if item.lower().endswith(".gguf") and item != gguf]
            if foreign_ggufs:
                raise ValueError(f"Launch {runtime} contém outro GGUF: {foreign_ggufs}")
        elif launch != ["ollama", "serve"]:
            raise ValueError("Launch Ollama deve apenas iniciar o daemon local")


def parse_make_vars(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Variável Make inválida: {value!r}; use NOME=valor")
        key, _ = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Nome de variável Make inválido: {key!r}")
        if key in PROTECTED_MAKE_VARS:
            raise ValueError(f"{key} é controlada pelo perfil e não pode ser sobrescrita")
        if key not in ALLOWED_MAKE_VARS:
            raise ValueError(f"{key} não é permitida na matriz GGUF")
        if key in IDENTITY_FLAGS:
            arguments = shlex.split(value.split("=", 1)[1])
            normalized_flags = {argument.split("=", 1)[0] for argument in arguments}
            forbidden = IDENTITY_FLAGS[key].intersection(normalized_flags)
            if forbidden:
                raise ValueError(f"{key} não pode alterar identidade do modelo: {sorted(forbidden)}")
        parsed.append(value)
    return parsed


def parse_render_make_vars(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Variável de render inválida: {value!r}; use NOME=valor")
        key, _ = value.split("=", 1)
        if key not in RENDER_MAKE_VAR_KEYS:
            raise ValueError(f"{key} não é permitida na renderização da matriz")
        if key in seen:
            raise ValueError(f"Variável de render repetida: {key}")
        seen.add(key)
        parsed.append(value)
    missing = RENDER_MAKE_VAR_KEYS - seen
    if missing:
        raise ValueError(f"Variáveis de render ausentes: {sorted(missing)}")
    return parsed


def make_variables(profile: dict[str, object], results_root: Path) -> list[str]:
    files = profile["files"]
    assert isinstance(files, dict)
    gguf = str(profile["gguf_path"])
    tokenizer = str(profile["tokenizer_path"])
    return [
        "MODEL_SIZE=7B",
        "PREPARE_OFFLINE=1",
        f"MODEL_7B_GGUF={gguf}",
        f"MODEL_7B_TOKENIZER={tokenizer}",
        f"VLLM_MODEL_DIR={gguf}",
        f"VLLM_CONFIG={files['vllm']['config']}",
        f"VLLM_LAUNCH={files['vllm']['launch']}",
        f"LLAMA_MODEL_DIR={gguf}",
        f"LLAMA_CONFIG={files['llama']['config']}",
        f"LLAMA_LAUNCH={files['llama']['launch']}",
        f"OLLAMA_MODEL_DIR={gguf}",
        f"OLLAMA_CONFIG={files['ollama']['config']}",
        f"OLLAMA_LAUNCH={files['ollama']['launch']}",
        f"OLLAMA_MODEL_NAME={profile['alias']}",
        f"GGUF_VARIANT={profile['variant']}",
        f"GGUF_RESULT_NAME=gguf-{profile['variant']}-base",
        f"GGUF_COMPARISON_CONTEXT={profile['context_window']}",
        f"GGUF_COMPARISON_RESULTS={results_root}",
    ]


def make_command(
    *,
    make: str,
    base_makefile: Path,
    profile: dict[str, object],
    target: str,
    extra_make_vars: list[str],
    results_root: Path,
    skip_prepare: str | None = None,
) -> list[str]:
    command = [make, "--no-print-directory", "-f", str(base_makefile)]
    if skip_prepare:
        command.extend(["-o", skip_prepare])
    command.append(target)
    command.extend(make_variables(profile, results_root))
    command.extend(extra_make_vars)
    return command


def safe_command(command: list[str]) -> list[str]:
    redacted = []
    for value in command:
        key = value.split("=", 1)[0].upper() if "=" in value else ""
        sensitive = any(word in key for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
        redacted.append("[REDACTED]" if sensitive else value)
    return redacted


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + json.dumps(safe_command(command), ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
            return process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise


def runtime_directory(runtime: str) -> str:
    return "llama.cpp" if runtime == "llama" else runtime


def result_snapshot(results_root: Path, runtime: str) -> set[str]:
    root = results_root / runtime_directory(runtime)
    if not root.is_dir():
        return set()
    return {str(path.resolve()) for path in root.iterdir() if path.is_dir()}


def validate_cell_results(
    result_directories: list[str],
    *,
    profile: dict[str, object],
    runtime: str,
    mode: str,
) -> list[dict[str, str]]:
    if len(result_directories) != 1:
        raise ValueError(
            f"Esperado exatamente um diretório de resultado novo; obtidos {len(result_directories)}"
        )
    timestamp_dir = Path(result_directories[0])
    manifests = sorted(timestamp_dir.glob("*/json/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"Esperado exatamente um manifest.json no resultado; obtidos {len(manifests)}"
        )
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifesto de resultado deve ser um objeto JSON: {manifest_path}")
    config = manifest.get("config")
    if manifest.get("status") != "complete" or not isinstance(config, dict):
        raise ValueError(f"Resultado não está completo: {manifest_path}")
    if config.get("model") != profile["alias"]:
        raise ValueError(f"Alias do resultado diverge do perfil: {manifest_path}")
    if str(profile["gguf_sha256"]) not in str(config.get("model_artifact", "")):
        raise ValueError(f"SHA do GGUF não aparece no resultado: {manifest_path}")
    expected_runtime = {"vllm": "vllm", "llama": "llama.cpp", "ollama": "ollama"}[runtime]
    if config.get("runtime") != expected_runtime:
        raise ValueError(f"Runtime do resultado diverge da célula: {manifest_path}")
    if manifest.get("smoke") is not (mode == "smoke"):
        raise ValueError(f"Modo smoke/formal diverge no resultado: {manifest_path}")
    return [{"path": str(manifest_path), "sha256": sha256_file(manifest_path)}]


def run_matrix(
    *,
    mode: str,
    profiles_dir: Path,
    workdir: Path,
    base_makefile: Path,
    results_root: Path,
    lock_file: Path,
    make: str = "make",
    extra_make_vars: Iterable[str] = (),
    render_target: str | None = None,
    render_make_vars: Iterable[str] = (),
    verify_sha: bool = True,
) -> int:
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode deve ser smoke ou formal")
    workdir = workdir.expanduser().resolve()
    base_makefile = base_makefile.expanduser().resolve()
    results_root = results_root.expanduser().resolve()
    lock_file = lock_file.expanduser().resolve()
    if not base_makefile.is_file():
        raise FileNotFoundError(f"Makefile base ausente: {base_makefile}")
    make_vars = parse_make_vars(extra_make_vars)
    parsed_render_vars = parse_render_make_vars(render_make_vars) if render_target else []
    if render_target not in {None, "gguf-comparison-render"}:
        raise ValueError("render_target inesperado")
    results_root.mkdir(parents=True, exist_ok=True)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with lock_file.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Outra matriz GGUF está ativa: {lock_file}", file=sys.stderr)
            return 2

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        if render_target:
            profiles_dir = results_root / ".generated" / "gguf-comparison" / stamp
            render_command = [
                make,
                "--no-print-directory",
                "-f",
                str(base_makefile),
                render_target,
                f"GGUF_COMPARISON_OUTPUT={profiles_dir}",
                *parsed_render_vars,
            ]
            print("[GGUF MATRIX] renderizando perfis dentro do lock global", flush=True)
            render_status = run_logged(
                render_command,
                cwd=workdir,
                log_path=profiles_dir.parent / f"{stamp}-render.log",
            )
            if render_status != 0:
                print("Falha ao renderizar perfis GGUF; campanha não iniciada.", file=sys.stderr)
                return 2
        profiles = load_profiles(profiles_dir, verify_sha=verify_sha)
        campaign_dir = results_root / "gguf-comparison" / f"{stamp}-{mode}"
        campaign_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = campaign_dir / "campaign.json"
        manifest: dict[str, object] = {
            "schema_version": 1,
            "mode": mode,
            "started_utc": stamp,
            "status": "running",
            "strictly_sequential": True,
            "fallback_policy": "forbidden",
            "schedule": [{"runtime": runtime, "variant": variant} for runtime, variant in SCHEDULE],
            "profiles": {
                variant: {
                    "alias": profile["alias"],
                    "gguf_path": profile["gguf_path"],
                    "gguf_sha256": profile["gguf_sha256"],
                    "quantization": profile["quantization"],
                    "imatrix": profile["imatrix"],
                    "tokenizer_metadata_sha256": profile["tokenizer_metadata_sha256"],
                    "source_manifest": profile.get("source_manifest"),
                }
                for variant, profile in profiles.items()
            },
            "ollama_preparations": [],
            "cells": [],
        }
        write_json(manifest_path, manifest)

        preparation_status: dict[str, int] = {}
        for index, variant in enumerate(EXPECTED_VARIANTS, 1):
            profile = profiles[variant]
            command = make_command(
                make=make,
                base_makefile=base_makefile,
                profile=profile,
                target="gguf-prepare-ollama",
                extra_make_vars=make_vars,
                results_root=results_root,
            )
            print(f"[GGUF MATRIX] preparação Ollama {index}/3: {variant}", flush=True)
            returncode = run_logged(
                command,
                cwd=workdir,
                log_path=campaign_dir / f"prepare-ollama-{index:02d}-{variant}.log",
            )
            preparation_status[variant] = returncode
            manifest["ollama_preparations"].append(
                {"variant": variant, "alias": profile["alias"], "returncode": returncode}
            )
            write_json(manifest_path, manifest)

        targets = CELL_TARGETS[mode]
        for index, (runtime, variant) in enumerate(SCHEDULE, 1):
            profile = profiles[variant]
            before = result_snapshot(results_root, runtime)
            cell: dict[str, object] = {
                "index": index,
                "runtime": runtime,
                "variant": variant,
                "alias": profile["alias"],
                "gguf_path": profile["gguf_path"],
                "gguf_sha256": profile["gguf_sha256"],
                "status": "running",
                "returncode": None,
                "result_directories": [],
            }
            manifest["cells"].append(cell)
            write_json(manifest_path, manifest)
            print(f"[GGUF MATRIX] célula {index}/9: runtime={runtime} variante={variant}", flush=True)

            if runtime == "ollama" and preparation_status[variant] != 0:
                cell["status"] = "blocked-by-ollama-preparation"
                cell["error"] = "Alias não foi preparado; célula não executada para evitar usar artefato antigo."
                write_json(manifest_path, manifest)
                continue

            command = make_command(
                make=make,
                base_makefile=base_makefile,
                profile=profile,
                target=targets[runtime],
                extra_make_vars=make_vars,
                results_root=results_root,
                skip_prepare="gguf-prepare-ollama" if runtime == "ollama" else None,
            )
            returncode = run_logged(
                command,
                cwd=workdir,
                log_path=campaign_dir / f"cell-{index:02d}-{runtime}-{variant}.log",
            )
            after = result_snapshot(results_root, runtime)
            result_directories = sorted(after - before)
            cell["returncode"] = returncode
            cell["result_directories"] = result_directories
            if returncode != 0:
                cell["status"] = "failed"
            else:
                try:
                    cell["result_manifests"] = validate_cell_results(
                        result_directories,
                        profile=profile,
                        runtime=runtime,
                        mode=mode,
                    )
                    cell["status"] = "complete"
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    cell["status"] = "failed-invalid-result"
                    cell["error"] = str(exc)
            write_json(manifest_path, manifest)

        preparations_ok = all(code == 0 for code in preparation_status.values())
        cells = manifest["cells"]
        assert isinstance(cells, list)
        cells_ok = all(cell.get("status") == "complete" for cell in cells)
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete" if preparations_ok and cells_ok else "failed"
        manifest["summary"] = {
            "ollama_preparations_ok": sum(code == 0 for code in preparation_status.values()),
            "ollama_preparations_total": len(preparation_status),
            "cells_complete": sum(cell.get("status") == "complete" for cell in cells),
            "cells_failed_or_blocked": sum(cell.get("status") != "complete" for cell in cells),
            "cells_total": len(cells),
        }
        write_json(manifest_path, manifest)
        print(json.dumps(manifest["summary"], ensure_ascii=False), flush=True)
        print(f"Manifesto da campanha: {manifest_path}", flush=True)
        return 0 if manifest["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--makefile",
        "--base-makefile",
        dest="base_makefile",
        type=Path,
        required=True,
        help="Makefile com os targets base e gguf-base-*; use Makefile.gguf-comparison.",
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--make", default="make")
    parser.add_argument("--make-var", action="append", default=[])
    parser.add_argument("--render-target")
    parser.add_argument("--render-make-var", action="append", default=[])
    parser.add_argument(
        "--no-verify-sha",
        action="store_true",
        help="Somente para testes controlados; não use em campanha real.",
    )
    args = parser.parse_args(argv)
    try:
        return run_matrix(
            mode=args.mode,
            profiles_dir=args.profiles_dir,
            workdir=args.workdir,
            base_makefile=args.base_makefile,
            results_root=args.results_root,
            lock_file=args.lock_file,
            make=args.make,
            extra_make_vars=args.make_var,
            render_target=args.render_target,
            render_make_vars=args.render_make_var,
            verify_sha=not args.no_verify_sha,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Erro de configuração da matriz: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
