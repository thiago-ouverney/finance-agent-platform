#!/usr/bin/env python3
"""Um usuário: inicialização, primeira resposta, aquecimento e GuideLLM sequencial."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

from results_layout import artifact, execution_label, href, locate, prepare, runtime_root, slug

ROOT = Path(__file__).resolve().parent
VERSION = "0.5.0"
GUIDELLM_VERSION = "0.7.4"
# Alvos de entrada. ``long`` deixa 512 tokens no contexto padrão de 8192:
# 128 para saída e 384 para template/margem de segurança.
NAMED_WORKLOADS = {"short": 256, "medium": 2048, "long": 7680}
WORKLOADS = dict(NAMED_WORKLOADS)
OUTPUT_TOKENS = 128
DEFAULT_CONVERSATION_FIXTURE = ROOT / "workloads" / "conversations" / "qwen_chat_bench_v2.json"
DEFAULT_WARMUP_CONVERSATION_FIXTURE = ROOT / "workloads" / "conversations" / "qwen_chat_warmup_v1.json"


def _synthetic_prompt(tokenizer, target_tokens: int, seed: str | int) -> str:
    """Cria um prompt determinístico e distinto com exatamente o tamanho-alvo."""
    digest = hashlib.sha256(str(seed).encode()).digest()
    marker = " ".join(f"word{value % 100}" for value in digest[:8])
    sentence = "Explique de forma objetiva este conceito para um estudante de estatística. "
    text = f"{marker}. {sentence}"
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += sentence
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False)


def estimate_messages_tokens(tokenizer, messages) -> int:
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return len(encoded)
    except Exception:
        return sum(len(tokenizer.encode(m.get("content", ""), add_special_tokens=False)) for m in messages)


def request_sha256(body) -> str:
    payload = {k: body.get(k) for k in ("messages", "max_tokens")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stream_messages_request(cfg, messages, timeout, secret="", max_tokens=OUTPUT_TOKENS, metadata=None):
    """Mede uma requisição SSE com ``time.perf_counter`` por requisição.

    ``first_token_time``/``last_token_time`` são os instantes do primeiro/último
    evento SSE com conteúdo. A API OpenAI não promete um evento por token; por
    isso a resolução é a do evento de conteúdo, enquanto ``completion_tokens``
    vem exclusivamente de ``usage`` quando o servidor o fornece.
    """
    import httpx
    body = {"model": cfg["model"], "messages": messages,
            "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    started_epoch = time.time()
    started = time.perf_counter()
    first = last = None
    chunks = 0
    content = ""
    usage = None
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
        with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ValueError("A API não respondeu com SSE.")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                event = json.loads(value)
                if "error" in event:
                    raise ValueError(f"Erro no stream: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
                if text:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
                    content += text
    ended = time.perf_counter()
    ended_epoch = started_epoch + (ended - started)
    e2e = ended - started
    metrics = stream_metrics(started, first, last, ended, usage)
    metrics.update({
        "request_start_epoch_s": started_epoch,
        "first_content_epoch_s": started_epoch + (first - started) if first is not None else None,
        "last_content_epoch_s": started_epoch + (last - started) if last is not None else None,
        "request_end_epoch_s": ended_epoch,
    })
    metadata = dict(metadata or {})
    if "history_tokens_estimate" in metadata:
        metadata["history_tokens"] = metrics["prompt_tokens"] if metrics["prompt_tokens"] is not None else metadata["history_tokens_estimate"]
    return {**metrics,
            "stream_content_event_count": chunks, "output": content, "usage_observed": usage is not None,
            "request_sha256": request_sha256(body),
            "request_args": json.dumps({"body": body}, ensure_ascii=False), **metadata}


def stream_request(cfg, prompt, timeout, secret=""):
    return stream_messages_request(cfg, [{"role": "user", "content": prompt}], timeout, secret)


def stream_metrics(request_start_time, first_token_time, last_token_time, request_end_time, usage):
    """Calcula métricas exclusivamente de timestamps monotônicos do cliente."""
    usage = usage if isinstance(usage, dict) else {}
    completion = usage.get("completion_tokens") if type(usage.get("completion_tokens")) is int else None
    prompt_tokens = usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int else None
    total = usage.get("total_tokens") if type(usage.get("total_tokens")) is int else None
    ttft = first_token_time - request_start_time if first_token_time is not None else None
    e2e = request_end_time - request_start_time
    generation = (last_token_time - first_token_time) if first_token_time is not None and last_token_time is not None and completion is not None and completion > 1 else None
    inter = generation / (completion - 1) if generation is not None else None
    decode = completion / generation if completion is not None and generation and generation > 0 else None
    effective = completion / e2e if completion is not None and e2e > 0 else None
    return {"request_start_time": request_start_time, "first_token_time": first_token_time,
            "last_token_time": last_token_time, "request_end_time": request_end_time,
            "time_to_first_token_seconds": ttft,
            "generation_time_seconds": generation, "end_to_end_latency_seconds": e2e,
            "completion_tokens": completion, "prompt_tokens": prompt_tokens, "total_tokens": total,
            "decode_tokens_per_second": decode, "end_to_end_tokens_per_second": effective,
            "inter_token_latency_seconds": inter, "time_to_first_token_ms": ttft * 1000 if ttft is not None else None,
            "request_latency": e2e, "inter_token_latency_ms": inter * 1000 if inter is not None else None,
            "output_tokens": completion, "decode_tokens_s": decode, "effective_tokens_s": effective}


def run_stream_batch(cfg, tokenizer, scenario, count, timeout, secret="", seed=0):
    samples = [
        (f"{seed}:{index}", _synthetic_prompt(tokenizer, WORKLOADS[scenario], f"{seed}:{index}"))
        for index in range(count)
    ]
    successful, errored = [], []
    for index, (workload_seed, prompt) in enumerate(samples):
        try:
            row = stream_request(cfg, prompt, timeout, secret)
            row.update({"mode": "independent", "request_id": f"{scenario}-independent-{index+1}",
                        "workload_seed": workload_seed})
            row["status"] = "successful"
            successful.append(row)
        except Exception as exc:
            errored.append({"status": "errored", "mode": "independent",
                            "request_id": f"{scenario}-independent-{index+1}",
                            "workload_seed": workload_seed, "error": str(exc)})
    return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}


def load_conversation_fixture(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("system"), str) or not data["system"].strip():
        raise ValueError("Fixture conversacional precisa de campo system textual nao vazio.")
    if not isinstance(data.get("turns"), list) or not data["turns"]:
        raise ValueError("Fixture conversacional precisa de lista nao vazia em turns.")
    for index, turn in enumerate(data["turns"], 1):
        if not isinstance(turn, dict) or not isinstance(turn.get("user"), str) or not turn["user"].strip():
            raise ValueError(f"Turno {index} do fixture precisa de user textual.")
        if "assistant" in turn and (not isinstance(turn["assistant"], str) or not turn["assistant"].strip()):
            raise ValueError(f"Turno {index} do fixture tem assistant vazio/invalido.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "data": data}


def validate_conversation_fixture_for_mode(fixture, loop_mode, turns):
    if loop_mode == "replay":
        missing = [index for index, turn in enumerate(fixture["data"]["turns"], 1)
                   if "assistant" not in turn]
        if missing:
            raise ValueError(f"--mode replay exige assistant fixo em cada turno usado do fixture; ausente em: {missing}.")


def fixture_messages_for_turn(fixture, turn_index, prior_assistant_outputs, loop_mode):
    data = fixture["data"]
    messages = []
    if data.get("system"):
        messages.append({"role": "system", "content": data["system"]})
    for index, turn in enumerate(data["turns"][:turn_index], 1):
        messages.append({"role": "user", "content": turn["user"]})
        if index < turn_index:
            if loop_mode == "closed-loop":
                messages.append({"role": "assistant", "content": prior_assistant_outputs[index - 1]})
            else:
                if "assistant" not in turn:
                    raise ValueError("--mode replay exige assistant fixo nos turnos anteriores do fixture.")
                messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def _rank_replay_turns(tokenizer, fixture, target_tokens, minimum_turn=1):
    """Ordena turnos pelo histórico fixo mais próximo do alvo do cenário."""
    ranked = []
    for turn_index in range(minimum_turn, len(fixture["data"]["turns"]) + 1):
        messages = fixture_messages_for_turn(fixture, turn_index, [], "replay")
        estimate = estimate_messages_tokens(tokenizer, messages)
        ranked.append((abs(estimate - target_tokens), estimate > target_tokens,
                       turn_index, estimate))
    ranked.sort()
    return [turn_index for _distance, _over, turn_index, _estimate in ranked]


def _trim_oldest_replay_pairs(tokenizer, messages, target_tokens):
    """Remove pares user/assistant antigos, preservando system e user atual."""
    trimmed = [dict(message) for message in messages]
    dropped = 0
    history_start = 1 if trimmed and trimmed[0].get("role") == "system" else 0
    while estimate_messages_tokens(tokenizer, trimmed) > target_tokens:
        # Um replay válido termina em user. Antes dele há pares user/assistant.
        if len(trimmed) - history_start < 3:
            break
        if (trimmed[history_start].get("role") != "user"
                or trimmed[history_start + 1].get("role") != "assistant"):
            raise ValueError("Histórico replay inválido: esperado par user/assistant.")
        del trimmed[history_start:history_start + 2]
        dropped += 1
    estimate = estimate_messages_tokens(tokenizer, trimmed)
    if estimate > target_tokens:
        raise ValueError(
            f"O system prompt e o user atual usam {estimate} tokens estimados, "
            f"acima do alvo {target_tokens}; não é possível truncar sem alterar a pergunta."
        )
    return trimmed, dropped, estimate


def _pad_replay_messages(tokenizer, messages, target_tokens, seed):
    """Adiciona contexto sintético determinístico até o maior valor <= alvo."""
    base = [dict(message) for message in messages]
    base_estimate = estimate_messages_tokens(tokenizer, base)
    if base_estimate >= target_tokens:
        return base, 0, base_estimate
    if not base or base[0].get("role") != "system":
        raise ValueError("Fixture replay precisa começar com system para receber contexto sintético.")

    # Gera uma única sequência grande. O binary search só recorta e decodifica,
    # evitando reconstruir milhares de tokens em cada tentativa.
    filler = _synthetic_prompt(tokenizer, target_tokens, seed)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    original_system = base[0]["content"]
    heading = "\n\nContexto sintético determinístico do benchmark:\n"

    def candidate(filler_tokens):
        rendered = [dict(message) for message in base]
        if filler_tokens:
            rendered[0]["content"] = (
                original_system + heading
                + tokenizer.decode(filler_ids[:filler_tokens], skip_special_tokens=False)
            )
        return rendered, estimate_messages_tokens(tokenizer, rendered)

    best_messages, best_tokens, best_estimate = base, 0, base_estimate
    low, high = 1, len(filler_ids)
    while low <= high:
        middle = (low + high) // 2
        rendered, estimate = candidate(middle)
        if estimate <= target_tokens:
            if estimate > best_estimate:
                best_messages, best_tokens, best_estimate = rendered, middle, estimate
            low = middle + 1
        else:
            high = middle - 1

    # Tokenização na fronteira texto/texto pode não ser perfeitamente monótona.
    # Uma pequena busca local torna a escolha estável sem ultrapassar o alvo.
    for filler_tokens in range(max(1, high - 8), min(len(filler_ids), low + 8) + 1):
        rendered, estimate = candidate(filler_tokens)
        if best_estimate < estimate <= target_tokens:
            best_messages, best_tokens, best_estimate = rendered, filler_tokens, estimate
    return best_messages, best_tokens, best_estimate


def _target_replay_messages(tokenizer, fixture, turn_index, target_tokens, seed):
    raw = fixture_messages_for_turn(fixture, turn_index, [], "replay")
    raw_estimate = estimate_messages_tokens(tokenizer, raw)
    trimmed, dropped, _trimmed_estimate = _trim_oldest_replay_pairs(
        tokenizer, raw, target_tokens
    )
    messages, filler_tokens, estimate = _pad_replay_messages(
        tokenizer, trimmed, target_tokens, seed
    )
    return messages, {
        "fixture_history_tokens_estimate": raw_estimate,
        "history_pairs_dropped": dropped,
        "synthetic_context_tokens": filler_tokens,
        "history_tokens_estimate": estimate,
        "target_delta_tokens_estimate": target_tokens - estimate,
        "history_selection_policy": (
            "nearest fixture history; drop oldest complete pairs above target; "
            "deterministic system-context padding below target"
        ),
    }


def workload_contract(scenarios, mode):
    """Contrato serializável usado no manifesto e nos configs de cada bloco."""
    targets = {name: WORKLOADS[name] for name in scenarios}
    semantics = {}
    for name in scenarios:
        if mode == "independent":
            semantics[name] = "synthetic user-content tokens before chat template"
        elif mode == "replay" and name in NAMED_WORKLOADS:
            semantics[name] = (
                "estimated complete chat prompt tokens; nearest fixed fixture history, "
                "oldest-pair truncation and deterministic system-context padding"
            )
        elif mode == "replay":
            semantics[name] = (
                "maximum estimated complete chat prompt tokens; fixed fixture histories "
                "that fit, without target padding"
            )
        else:
            semantics[name] = (
                "reference ceiling; complete chat prompt evolves from real assistant outputs"
            )
    if mode == "independent":
        policy = "deterministic unique synthetic user prompt per request"
    elif mode == "replay" and any(name in NAMED_WORKLOADS for name in scenarios):
        policy = (
            "fixed replay fixture selected near each named token target; deterministic "
            "truncation/padding; server usage.prompt_tokens is authoritative"
        )
    elif mode == "replay":
        policy = "fixed replay fixture histories filtered by the requested context ceiling"
    else:
        policy = "closed-loop fixture with real assistant outputs accumulated in history"
    contract = {
        "schema_version": 1,
        "mode": mode,
        "named_scenario_defaults": dict(NAMED_WORKLOADS),
        "input_token_targets": targets,
        "input_token_target_semantics": semantics,
        "output_token_ceiling": OUTPUT_TOKENS,
        "prompt_policy": policy,
        "prompt_tokens_authority": (
            "OpenAI-compatible usage.prompt_tokens; null when the runtime omits usage"
        ),
    }
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return contract


def run_conversational_batch(cfg, tokenizer, scenario, conversations, turns, timeout, secret="", fixture=None, loop_mode="closed-loop"):
    if fixture is None:
        fixture = load_conversation_fixture(DEFAULT_CONVERSATION_FIXTURE)
    if turns > len(fixture["data"]["turns"]):
        raise ValueError(f"--conversation-turns={turns} excede turnos disponiveis no fixture ({len(fixture['data']['turns'])}).")
    validate_conversation_fixture_for_mode(fixture, loop_mode, turns)
    target = WORKLOADS[scenario]
    successful, errored = [], []
    available_turns = max(1, len(fixture["data"]["turns"]) - turns + 1)
    targeted_replay = loop_mode == "replay" and scenario in NAMED_WORKLOADS
    if targeted_replay:
        fitting_turns = _rank_replay_turns(tokenizer, fixture, target, turns)
        available_turns = len(fitting_turns)
    elif turns == 1 and scenario.startswith("ctx"):
        fitting_turns = []
        for candidate in range(1, available_turns + 1):
            candidate_messages = fixture_messages_for_turn(fixture, candidate, [], loop_mode)
            if estimate_messages_tokens(tokenizer, candidate_messages) <= target:
                fitting_turns.append(candidate)
        if not fitting_turns:
            raise ValueError(f"Nenhum turno da fixture cabe em ctx{target} tokens.")
        available_turns = len(fitting_turns)
    else:
        fitting_turns = list(range(1, available_turns + 1))
    for conversation_index in range(1, conversations + 1):
        assistant_outputs = []
        # Com turns=1, percorremos a fixture em vez de repetir sempre a primeira
        # pergunta. Cada request continua sendo uma conversa nova e determinística,
        # mas pode carregar um histórico de tamanho diferente.
        if turns == 1 or targeted_replay:
            final_turn = fitting_turns[(conversation_index - 1) % available_turns]
        else:
            final_turn = turns
        for turn_index in range(final_turn - turns + 1, final_turn + 1):
            messages = fixture_messages_for_turn(fixture, turn_index, assistant_outputs, loop_mode)
            targeting = {}
            if targeted_replay:
                messages, targeting = _target_replay_messages(
                    tokenizer, fixture, turn_index, target,
                    f"{fixture['sha256']}:{scenario}:{conversation_index}:{turn_index}",
                )
            estimate = estimate_messages_tokens(tokenizer, messages)
            metadata = {"mode": loop_mode, "conversation_index": conversation_index,
                        "fixture_sha256": fixture["sha256"],
                        "fixture_name": fixture["data"].get("name"),
                        "turn_index": turn_index, "history_tokens_estimate": estimate,
                        "target_history_tokens": target, "messages_count": len(messages),
                        "request_id": f"{scenario}-c{conversation_index}-t{turn_index}",
                        **targeting}
            try:
                row = stream_messages_request(cfg, messages, timeout, secret, metadata=metadata)
                row["status"] = "successful"
                successful.append(row)
                assistant_outputs.append(row.get("output") or "")
            except Exception as exc:
                errored.append({"status": "errored", **metadata, "history_tokens": estimate, "error": str(exc)})
                break
    return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}


def local_model_check(value):
    """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
            raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
        files = [path]
    elif path.is_dir():
        files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
        for index in path.glob("*.index.json"):
            data = json.loads(index.read_text())
            missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
            if missing:
                raise ValueError(f"Shards ausentes: {missing}")
    else:
        files = []
    if not files or any(f.stat().st_size == 0 for f in files):
        raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
    return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
            "files": len(files), "bytes": sum(f.stat().st_size for f in files),
            "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


_SENSITIVE_NAMES = {
    "api_key", "authorization", "access_token", "auth_token", "hf_token",
    "password", "secret", "token",
}


def _sensitive_name(value):
    normalized = str(value).strip().lower().lstrip("-").replace("-", "_")
    return normalized in _SENSITIVE_NAMES


def reject_sensitive_argv(values):
    for value in values or []:
        key = value.partition("=")[0] if isinstance(value, str) else value
        if _sensitive_name(key):
            raise ValueError(
                "Segredos não podem ser passados em --launch-extra-args; "
                "o benchmark usa loopback e não persiste credenciais de servidor."
            )


def redact(value, secret):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if _sensitive_name(k) else redact(v, secret))
                for k, v in value.items()}
    if isinstance(value, list):
        result = []
        redact_next = False
        for item in value:
            if redact_next:
                result.append("[REDACTED]")
                redact_next = False
                continue
            if isinstance(item, str):
                key, separator, _raw = item.partition("=")
                if separator and _sensitive_name(key):
                    result.append(f"{key}=[REDACTED]")
                    continue
                if _sensitive_name(item):
                    result.append(item)
                    redact_next = True
                    continue
            result.append(redact(item, secret))
        return result
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value


def sanitize_model_identity(value, secret=""):
    """Aceita somente o contrato público do preparador; campos extras não vazam."""
    if not isinstance(value, dict):
        raise ValueError("--model-metadata deve conter um objeto JSON")

    def file_records(records):
        if not isinstance(records, list):
            return []
        return [
            {key: record.get(key) for key in ("path", "bytes", "sha256")}
            for record in records if isinstance(record, dict)
        ]

    top_fields = (
        "schema_version", "prepared_at_utc", "source", "runtime",
        "requested_model", "model", "requested_revision", "resolved_revision",
        "gguf_filename", "local_path", "tokenizer_path", "requested_tokenizer",
        "requested_tokenizer_revision", "resolved_tokenizer_revision", "bytes", "sha256",
    )
    result = {key: value.get(key) for key in top_fields if key in value}
    if isinstance(value.get("paths"), dict):
        result["paths"] = {
            key: value["paths"].get(key) for key in ("model", "tokenizer")
            if key in value["paths"]
        }
    if isinstance(value.get("artifact"), dict):
        artifact_identity = value["artifact"]
        result["artifact"] = {
            key: artifact_identity.get(key)
            for key in ("kind", "bytes", "sha256", "signature")
            if key in artifact_identity
        }
        result["artifact"]["files"] = file_records(artifact_identity.get("files"))
    if isinstance(value.get("tokenizer"), dict):
        tokenizer_identity = value["tokenizer"]
        result["tokenizer"] = {
            key: tokenizer_identity.get(key)
            for key in (
                "present", "path", "bytes", "sha256", "source", "requested_model",
                "requested_revision", "resolved_revision",
            )
            if key in tokenizer_identity
        }
        result["tokenizer"]["files"] = file_records(tokenizer_identity.get("files"))
    if not isinstance(result.get("local_path"), str) or not result["local_path"].strip():
        raise ValueError("--model-metadata não contém local_path válido")
    for environment_secret in (
        secret,
        os.environ.get("HF_TOKEN", ""),
        os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
    ):
        result = redact(result, environment_secret)
    return result


def capture(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def hardware_snapshot():
    """Captura identidade estática suficiente para separar máquinas na análise."""
    import psutil

    cpu_model = platform.processor().strip()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    def read_optional(path):
        try:
            return Path(path).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_model": cpu_model or None,
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "cpu_affinity": affinity,
        "ram_total_bytes": psutil.virtual_memory().total,
        "cgroup_cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": read_optional("/sys/fs/cgroup/memory.max"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": capture([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]),
    }


def runtime_environment(cfg):
    common = {"CUDA_VISIBLE_DEVICES"}
    names = {
        "vllm": common | {
            "VLLM_ATTENTION_BACKEND", "VLLM_WORKER_MULTIPROC_METHOD",
            "VLLM_CUDA_RUNTIME_LIB",
        },
        "llama.cpp": common | {"GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_BACKEND_PATH"},
        "ollama": common | {
            "OLLAMA_KEEP_ALIVE", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION",
            "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_MODELS",
        },
    }.get(cfg.get("runtime"), common)
    return {name: os.environ[name] for name in sorted(names) if name in os.environ}


def _cache_argv(arguments):
    result = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        lowered = value.lower()
        if ("cache" in lowered or "caching" in lowered
                or "keepalive" in lowered or "keep-alive" in lowered):
            sensitive = any(marker in lowered for marker in (
                "api-key", "api_key", "authorization", "password", "secret", "token"
            ))
            if sensitive and "=" in value:
                result.append(value.partition("=")[0] + "=[REDACTED]")
                index += 1
                continue
            result.append(value)
            if "=" not in value and index + 1 < len(arguments):
                following = arguments[index + 1]
                if not following.startswith("--"):
                    result.append("[REDACTED]" if sensitive else following)
                    index += 1
        index += 1
    return result


def describe_cache_policy(cfg, argv, owns_runtime, environment=None):
    """Separa KV, reutilização de prefixo e residência sem inferir uma métrica."""
    arguments = [str(value) for value in (argv or [])]
    lowered = [value.lower() for value in arguments]
    environment = dict(environment or {})
    if not owns_runtime:
        prefix_state = "external-server-state-unknown"
    elif "--enable-prefix-caching" in lowered:
        prefix_state = "enabled-explicit-argv"
    elif "--no-enable-prefix-caching" in lowered:
        prefix_state = "disabled-explicit-argv"
    elif any(value == "--cache-reuse" or value.startswith("--cache-reuse=")
             for value in lowered):
        prefix_state = "configured-by-cache-reuse-argv"
    elif cfg.get("runtime") == "ollama":
        prefix_state = "not-exposed-by-runtime-interface"
    else:
        prefix_state = "runtime-default-not-overridden"
    kv_configuration = {
        key: environment[key] for key in ("OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION")
        if key in environment
    }
    cache_flags = _cache_argv(arguments)
    prefix_flags = [value for value in cache_flags if (
        "prefix" in value.lower() or "cache-reuse" in value.lower()
    )]
    keep_alive = environment.get("OLLAMA_KEEP_ALIVE")
    return {
        "declared": cfg.get("cache_policy"),
        "process_ownership": "owned-process-group" if owns_runtime else "external-server",
        "kv_cache": {
            "state": "enabled-required-for-autoregressive-decode",
            "configuration": kv_configuration,
            "note": "ocupação/capacidade só é observada quando o runtime exporta a métrica",
        },
        "effective_cache_argv": cache_flags,
        "prefix_cache": {"state": prefix_state, "effective_argv_flags": prefix_flags},
        "model_residency": {
            "state": (
                "policy-configured-not-observed" if owns_runtime and keep_alive is not None
                else "not-observed"
            ),
            "keep_alive_policy": keep_alive,
        },
        "runtime_environment": environment,
    }


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
                "runtime_version", "model_artifact", "server_command", "notes"}
    if set(cfg) != required:
        raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
    if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
        raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
    if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
        raise ValueError("context_window deve ser um inteiro >= 512.")
    url = urlsplit(cfg["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
    if url.path not in {"", "/", "/v1", "/v1/"}:
        raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
    cfg["base_url"] = f"{url.scheme}://{url.netloc}"
    return cfg


def validate_run(cfg, scenarios, smoke):
    for scenario in scenarios:
        # Margem para o template; o servidor continua sendo a autoridade final.
        needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
        if cfg["context_window"] < needed:
            raise ValueError(
                f"{scenario} exige context_window >= {needed}, mas a configuração "
                f"declara {cfg['context_window']}. No fluxo observe-bench, use "
                f"OBS_CONTEXT={needed} ou maior; para short/medium/long use "
                "OBS_CONTEXT=8192. Como alternativa, retire o cenário com "
                "OBS_BENCH_SCENARIOS."
            )
    if not smoke:
        if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
               for k in ("model", "runtime_version", "model_artifact", "server_command")):
            raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
        if cfg["cache_policy"] == "runtime-default-unverified":
            raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
                             "O script NÃO altera nem comprova a política de cache.")


def tokenizer_digest(path):
    path = Path(path)
    if not path.is_dir() or not (path / "tokenizer_config.json").exists():
        raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and not file.name.startswith("."):
            hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def prepare_tokenizer(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    path = Path(args.output)
    if path.exists():
        raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    path.mkdir(parents=True)
    tokenizer.save_pretrained(path)
    write_json(path / "source.json", {"model": args.model, "revision": revision})
    print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
    print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")


def scenario_config(cfg, name, count, seed, secret, timeout):
    return {
        "spec": {
            "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
                        "request_format": "/v1/chat/completions", "stream": True,
                        "validate_backend": False, "verify": True, "follow_redirects": False,
                        "timeout": timeout, "api_key": secret or None,
                        "extras": {"body": {"temperature": 0, "top_p": 1}}},
            "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
            "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
            "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
                          "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
            "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
                      "output_tokens": OUTPUT_TOKENS}],
            "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
            "seed": {"kind": "static", "value": seed},
            "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
            "outputs": [],
        }
    }


class Monitor:
    """Amostragem contínua e marcação de fases; não é um profiler PCIe."""
    def __init__(self, output, cfg=None, secret="", collect_kv=False, collect_resources=True):
        self.output = Path(output)
        self.stop_event = threading.Event()
        self.thread = None
        self.phase = "setup"
        self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
        self.collect_resources = collect_resources
        self.kv_thread = None
        self.started_monotonic = None

    def set_phase(self, phase, event=None):
        self.phase = phase
        utc = datetime.now(timezone.utc).isoformat()
        elapsed = (time.monotonic() - self.started_monotonic) if self.started_monotonic else 0.0
        self.log(f"[telemetria] utc={utc} decorrido={elapsed:.3f}s fase={phase} evento={event or 'phase_change'}")
        if hasattr(self, "events_handle"):
            writer = csv.writer(self.events_handle)
            writer.writerow([utc, time.monotonic(), elapsed, phase, event or "phase_change"])
            self.events_handle.flush()

    def start(self):
        self.started_monotonic = time.monotonic()
        self.telemetry_handle = artifact(self.output, "telemetry.log").open("a", encoding="utf-8")
        self.events_handle = artifact(self.output, "events.csv").open("w", newline="", encoding="utf-8")
        csv.writer(self.events_handle).writerow(["utc", "monotonic_s", "elapsed_s", "phase", "event"])
        self.set_phase(self.phase, "monitor_started")
        if self.collect_resources:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
        if self.collect_resources and self.collect_kv:
            self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
            self.kv_thread.start()

    def log(self, message):
        print(message, flush=True)
        if hasattr(self, "telemetry_handle"):
            self.telemetry_handle.write(message + "\n")
            self.telemetry_handle.flush()

    def kv_loop(self):
        import httpx
        headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
        pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
        with artifact(self.output, "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
            writer = csv.writer(handle)
            writer.writerow(["utc", "elapsed_s", "phase", "series", "fraction"])
            while not self.stop_event.is_set():
                phase = self.phase
                try:
                    response = client.get(self.cfg["base_url"] + "/metrics")
                    response.raise_for_status()
                    matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
                    modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
                    for m in matches:
                        if modern and m[1] != "vllm:kv_cache_usage_perc":
                            continue
                        value = float(m[3])
                        if math.isfinite(value) and 0 <= value <= 1:
                            writer.writerow([datetime.now(timezone.utc).isoformat(), time.monotonic() - self.started_monotonic,
                                             phase, redact(m[1] + (m[2] or ""), self.secret), value])
                    handle.flush()
                except (httpx.HTTPError, ValueError):
                    pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
                self.stop_event.wait(1)

    def loop(self):
        try:
            import psutil
        except ImportError:
            psutil = None
        with artifact(self.output, "gpu.csv").open("w", newline="", encoding="utf-8") as handle, \
             artifact(self.output, "system.csv").open("w", newline="", encoding="utf-8") as system_handle:
            writer = csv.writer(handle)
            writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "index", "name", "used_mib", "total_mib", "used_gib", "total_gib",
                             "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"])
            system_writer = csv.writer(system_handle)
            system_writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "cpu_util_pct", "ram_used_mib", "ram_available_mib", "ram_total_mib",
                                    "load1", "root_disk_used_mib", "root_disk_free_mib", "disk_read_bytes", "disk_write_bytes"])
            if psutil:
                psutil.cpu_percent(interval=None)
            sample_number = 0
            while not self.stop_event.is_set():
                sample_number += 1
                phase = self.phase
                utc = datetime.now(timezone.utc).isoformat()
                result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                                  "--format=csv,noheader,nounits"])
                if result.get("returncode") != 0:
                    write_json(artifact(self.output, "gpu-unavailable.json"), result)
                else:
                    gpu_rows = list(csv.reader(result["stdout"].splitlines(), skipinitialspace=True))
                    for row in gpu_rows:
                        used_mib, total_mib = float(row[2]), float(row[3])
                        vram_pct = (100 * used_mib / total_mib) if total_mib else None
                        writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, row[0], row[1], row[2], row[3],
                                         used_mib / 1024, total_mib / 1024, vram_pct, *row[4:]])
                    if sample_number == 1 or sample_number % 10 == 0:
                        compact = "; ".join(f"GPU{row[0]} VRAM={float(row[2]) / 1024:.2f}/{float(row[3]) / 1024:.2f} GiB "
                                             f"ocupada={100 * float(row[2]) / float(row[3]):.1f}% "
                                             f"atividade_gpu={row[4]}% atividade_leitura_escrita_memoria={row[5]}%" for row in gpu_rows)
                        elapsed = time.monotonic() - self.started_monotonic
                        utc_log = datetime.now(timezone.utc).isoformat()
                        self.log(f"[telemetria] utc={utc_log} decorrido={elapsed:.3f}s fase={phase} {compact or 'GPU sem amostra'}")
                handle.flush()
                if psutil:
                    vm = psutil.virtual_memory()
                    du = psutil.disk_usage(str(self.output.anchor or "/"))
                    io = psutil.disk_io_counters()
                    system_writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, psutil.cpu_percent(interval=None), vm.used / 1048576,
                                             vm.available / 1048576, vm.total / 1048576, os.getloadavg()[0],
                                             du.used / 1048576, du.free / 1048576,
                                             getattr(io, "read_bytes", None), getattr(io, "write_bytes", None)])
                    system_handle.flush()
                self.stop_event.wait(1)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=17)
        if self.kv_thread:
            self.kv_thread.join(timeout=3)
        self.set_phase("stopped", "monitor_stopped")
        if hasattr(self, "events_handle"):
            self.events_handle.close()
        if hasattr(self, "telemetry_handle"):
            self.telemetry_handle.close()
        self.write_telemetry_summary()

    def write_telemetry_summary(self):
        """Agrega telemetria por fase para relacionar picos com eventos do benchmark."""
        def read_rows(name):
            path = locate(self.output, name)
            if not path.exists():
                return []
            with path.open() as handle:
                return list(csv.DictReader(handle))
        sources = {"gpu": read_rows("gpu.csv"), "system": read_rows("system.csv"), "kv": read_rows("kv-cache.csv")}
        # Arquivo de entrada simples para gráficos: uma observação por linha,
        # com UTC, tempo desde o início do monitor e fase experimental.
        if sources["gpu"]:
            import shutil
            shutil.copyfile(artifact(self.output, "gpu.csv"), artifact(self.output, "telemetry-timeseries.csv"))
        phases = sorted({r.get("phase") for rows in sources.values() for r in rows if r.get("phase")})
        output = {"definition": "Amostras observadas por fase; não são bytes nem tempos de transferência PCIe.", "phases": {}}
        for phase in phases:
            entry = {"gpu_samples": 0, "system_samples": 0, "kv_samples": 0}
            grows = [r for r in sources["gpu"] if r.get("phase") == phase]
            for key in ("used_mib", "total_mib", "used_gib", "total_gib", "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"):
                vals = []
                for r in grows:
                    try: vals.append(float(r[key]))
                    except (ValueError, TypeError, KeyError): pass
                entry[f"gpu_{key}_mean"] = sum(vals) / len(vals) if vals else None
                entry[f"gpu_{key}_max"] = max(vals) if vals else None
                if key in {"used_mib", "used_gib", "vram_used_pct"}:
                    entry[f"gpu_{key}_min"] = min(vals) if vals else None
            entry["gpu_samples"] = len(grows)
            srows = [r for r in sources["system"] if r.get("phase") == phase]
            for key in ("cpu_util_pct", "ram_used_mib", "ram_available_mib", "root_disk_used_mib", "root_disk_free_mib"):
                vals = []
                for r in srows:
                    try: vals.append(float(r[key]))
                    except (ValueError, TypeError, KeyError): pass
                entry[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
                entry[f"{key}_max"] = max(vals) if vals else None
            entry["system_samples"] = len(srows)
            krows = [r for r in sources["kv"] if r.get("phase") == phase]
            vals = []
            for r in krows:
                try: vals.append(float(r["fraction"]) * 100)
                except (ValueError, TypeError, KeyError): pass
            entry["kv_occupancy_pct_mean"] = sum(vals) / len(vals) if vals else None
            entry["kv_occupancy_pct_max"] = max(vals) if vals else None
            entry["kv_samples"] = len(vals)
            output["phases"][phase] = entry
        write_json(artifact(self.output, "telemetry-summary.json"), output)


def percentile(values, q):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(report):
    """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
    benchmarks = report["benchmarks"]
    if len(benchmarks) != 1:
        raise ValueError("Esperado exatamente um benchmark sequencial.")
    requests = benchmarks[0]["requests"]
    from reporting import derived
    good = [{**r, **derived(r)} for r in requests["successful"]]
    result = {
        "successful_request_count": len(good),
        "errored_request_count": len(requests["errored"]),
        "incomplete_request_count": len(requests["incomplete"]),
    }
    # Os nomes são deliberadamente longos: summary.json é um artefato de
    # análise, e não uma API em que economizar alguns bytes melhora algo.
    metrics = {
        "request_first_token_latency_milliseconds": "time_to_first_token_ms",
        "request_latency_seconds": "request_latency",
        "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
        "output_completion_token_count": "output_tokens",
        "input_prompt_token_count": "prompt_tokens",
        "decode_generation_tokens_per_second": "decode_tokens_s",
        "effective_output_tokens_per_second": "effective_tokens_s",
    }
    for label, key in metrics.items():
        vals = [r.get(key) for r in good]
        result[label + "_sample_count"] = sum(v is not None for v in vals)
        result[label + "_p50"] = percentile(vals, .5)
        result[label + "_p95"] = percentile(vals, .95)
        result[label + "_p99"] = percentile(vals, .99)
    # Nomes canônicos para a comparação entre runtimes. Os campos históricos
    # acima permanecem para compatibilidade com relatórios já gerados.
    ttft = [r.get("time_to_first_token_ms") for r in good]
    tokens_s = [r.get("decode_tokens_s") for r in good]
    result["time_to_first_token_milliseconds_sample_count"] = sum(v is not None for v in ttft)
    result["tokens_per_second_sample_count"] = sum(v is not None for v in tokens_s)
    for suffix, q in (("p50", .50), ("p95", .95), ("p99", .99)):
        result[f"time_to_first_token_milliseconds_{suffix}"] = percentile(ttft, q)
        result[f"tokens_per_second_{suffix}"] = percentile(tokens_s, q)
    for label, key in {
        "time_to_first_token_seconds": "time_to_first_token_seconds",
        "generation_time_seconds": "generation_time_seconds",
        "end_to_end_latency_seconds": "end_to_end_latency_seconds",
        "decode_tokens_per_second": "decode_tokens_per_second",
        "end_to_end_tokens_per_second": "end_to_end_tokens_per_second",
    }.items():
        values = [r.get(key) for r in good]
        result[label + "_sample_count"] = sum(v is not None for v in values)
        result[label + "_p50"] = percentile(values, .50)
        result[label + "_p95"] = percentile(values, .95)
        result[label + "_p99"] = percentile(values, .99)
    result["metric_definitions"] = {
        "Time To First Token": "milissegundos entre o envio da requisição e o primeiro token/conteúdo observado; há um valor por requisição.",
        "Tokens/s": "tokens de saída por segundo durante o decode, calculado por requisição a partir do intervalo entre tokens; não inclui TTFT.",
    }
    result["requests_sha256"] = requests_digest(report)
    turns = sorted({r.get("turn_index") for r in good if r.get("turn_index") is not None})
    result["turn_indices"] = turns
    result["history_tokens_by_turn"] = [{"turn_index": turn,
                                         "history_tokens_p50": percentile([r.get("history_tokens") for r in good if r.get("turn_index") == turn], .50),
                                         "history_tokens_p95": percentile([r.get("history_tokens") for r in good if r.get("turn_index") == turn], .95)}
                                        for turn in turns]
    result["percentiles_are_exploratory"] = len(good) < 100
    result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
    return result


def requests_digest(report):
    # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
    bodies = []
    for row in report["benchmarks"][0]["requests"]["successful"]:
        args = json.loads(row["request_args"])
        body = args.get("body", {})
        bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
    return hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def turn_manifest(report):
    rows = report["benchmarks"][0]["requests"]["successful"]
    turns = []
    for row in rows:
        turns.append({"request_id": row.get("request_id"), "conversation_index": row.get("conversation_index"),
                      "turn_index": row.get("turn_index"), "mode": row.get("mode"),
                      "fixture_sha256": row.get("fixture_sha256"), "history_tokens": row.get("history_tokens"),
                      "history_tokens_estimate": row.get("history_tokens_estimate"),
                      "fixture_history_tokens_estimate": row.get("fixture_history_tokens_estimate"),
                      "target_history_tokens": row.get("target_history_tokens"),
                      "target_delta_tokens_estimate": row.get("target_delta_tokens_estimate"),
                      "history_pairs_dropped": row.get("history_pairs_dropped"),
                      "synthetic_context_tokens": row.get("synthetic_context_tokens"),
                      "prompt_tokens": row.get("prompt_tokens"), "total_tokens": row.get("total_tokens"),
                      "request_sha256": row.get("request_sha256")})
    return turns


def conversation_artifact(report, fixture=None):
    requests = report["benchmarks"][0]["requests"]
    rows = []
    for status in ("successful", "errored", "incomplete"):
        for row in requests[status]:
            body = {}
            if row.get("request_args"):
                body = json.loads(row["request_args"]).get("body", {})
            rows.append({"status": status, "request_id": row.get("request_id"),
                         "conversation_index": row.get("conversation_index"),
                         "turn_index": row.get("turn_index"), "mode": row.get("mode"),
                         "messages": body.get("messages"),
                         "assistant_output": row.get("output") if status == "successful" else None,
                         "error": row.get("error") if status != "successful" else None,
                         "request_sha256": row.get("request_sha256"),
                         "history_tokens": row.get("history_tokens"),
                         "history_tokens_estimate": row.get("history_tokens_estimate"),
                         "fixture_history_tokens_estimate": row.get("fixture_history_tokens_estimate"),
                         "target_history_tokens": row.get("target_history_tokens"),
                         "target_delta_tokens_estimate": row.get("target_delta_tokens_estimate"),
                         "history_pairs_dropped": row.get("history_pairs_dropped"),
                         "synthetic_context_tokens": row.get("synthetic_context_tokens")})
    artifact = {"mode": rows[0].get("mode") if rows else None, "turns": rows}
    if fixture:
        artifact["fixture"] = {"path": fixture["path"], "sha256": fixture["sha256"],
                               "name": fixture["data"].get("name"),
                               "version": fixture["data"].get("version")}
    return artifact


def write_requests_csv(path, report):
    """Amostras individuais para análise no R/Python, incluindo status de erro."""
    from reporting import derived
    fields = ["status", "error", "mode", "request_id", "conversation_index", "turn_index",
              "workload_request_id", "workload_profile", "bucket", "rendered_prompt_tokens",
              "fixture_name", "fixture_sha256", "workload_seed",
              "history_tokens", "history_tokens_estimate", "target_history_tokens", "messages_count",
              "fixture_history_tokens_estimate", "target_delta_tokens_estimate",
              "history_pairs_dropped", "synthetic_context_tokens", "history_selection_policy",
              "request_sha256", "request_start_time", "first_token_time", "last_token_time", "request_end_time",
              "request_start_epoch_s", "first_content_epoch_s", "last_content_epoch_s", "request_end_epoch_s",
              "time_to_first_token_seconds", "generation_time_seconds", "end_to_end_latency_seconds",
              "completion_tokens", "prompt_tokens", "total_tokens", "decode_tokens_per_second",
              "end_to_end_tokens_per_second", "inter_token_latency_seconds", "request_latency",
              "time_to_first_token_ms", "inter_token_latency_ms", "output_tokens", "decode_tokens_s", "effective_tokens_s",
              "context_start_tokens", "context_end_tokens", "context_band"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for status in ("successful", "errored", "incomplete"):
            for row in report["benchmarks"][0]["requests"][status]:
                if status == "successful":
                    row = {**row, **derived(row)}
                writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})


def observability_request_rows(report, *, experiment_id, runtime, model, block,
                               benchmark_phase, scenario, repetition):
    """Normaliza requisições sem copiar prompt nem resposta para o dataset analítico."""
    rows = []
    requests = report["benchmarks"][0]["requests"]
    for status in ("successful", "errored", "incomplete"):
        for index, request in enumerate(requests[status], 1):
            request_id = request.get("request_id") or f"request-{index}"
            rows.append({
                **request,
                "experiment_id": experiment_id,
                "request_uid": f"{experiment_id}:{block}:{request_id}:{index}",
                "runtime": runtime,
                "model": model,
                "block": block,
                "benchmark_phase": benchmark_phase,
                "scenario": scenario,
                "repetition": repetition,
                "request_index": index,
                "status": status,
            })
    return rows


def lifecycle_observability_row(request, *, experiment_id, runtime, model, phase):
    """Converte first/warm reference ao mesmo contrato das requisições do batch."""
    if not request:
        return None
    return {
        **request,
        "experiment_id": experiment_id,
        "request_uid": f"{experiment_id}:{phase}:1",
        "runtime": runtime,
        "model": model,
        "block": phase,
        "benchmark_phase": phase,
        "scenario": None,
        "repetition": None,
        "request_index": 1,
        "status": request.get("status", "unknown"),
        "request_sha256": request_sha256(request.get("body") or {}),
    }


def write_summary(output, rows):
    from reporting import render
    write_json(artifact(output, "summary.json"), rows)
    if not rows:
        render(output, rows)
        return
    with artifact(output, "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(output, rows)


def run(args):
    import fcntl
    from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
    import workload_dataset as dataset_workload
    if args.prometheus_url:
        from prometheus_dataset import prometheus_ready
        if not prometheus_ready(args.prometheus_url):
            raise ValueError(f"Prometheus indisponível em {args.prometheus_url}. Inicie a coleta antes do benchmark.")
    if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
        raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
    cfg = load_config(args.config)
    mopep_profile = args.benchmark_profile != "generic"
    if not mopep_profile:
        validate_run(cfg, args.scenarios, args.smoke)
    availability = local_model_check(args.local_model_path)
    availability["kv_bytes_per_token"] = args.kv_bytes_per_token
    availability["launch_hf_offline"] = bool(args.launch)
    secret = os.environ.get("BENCH_API_KEY", "")
    model_identity = None
    if args.model_metadata:
        raw_model_identity = json.loads(Path(args.model_metadata).read_text(encoding="utf-8"))
        model_identity = sanitize_model_identity(raw_model_identity, secret)
    digest = tokenizer_digest(cfg["tokenizer"])
    from transformers import AutoTokenizer
    measurement_tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True, trust_remote_code=False)
    count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
    prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
    if not prompt.strip():
        raise ValueError("O prompt inicial não pode estar vazio.")
    conversation_mode = not mopep_profile and args.mode in {"closed-loop", "replay"}
    if not conversation_mode and args.conversation_turns != 1:
        raise ValueError("--conversation-turns só pode ser maior que 1 com --mode closed-loop ou --mode replay.")
    conversation_fixture = load_conversation_fixture(args.conversation_fixture) if conversation_mode else None
    warmup_conversation_fixture = load_conversation_fixture(args.warmup_conversation_fixture) if conversation_mode else None
    if conversation_fixture:
        validate_conversation_fixture_for_mode(conversation_fixture, args.mode, args.conversation_turns)
    dataset_records = warmup_records = []
    dataset_manifest = None
    replay_responses, replay_sha256, replay_manifest = None, None, None
    if mopep_profile:
        if not args.request_dataset or not args.request_manifest or not args.warmup_request_dataset:
            raise ValueError("Perfis MOPEP exigem --request-dataset, --request-manifest e --warmup-request-dataset.")
        dataset_records, dataset_manifest = dataset_workload.load_workload(
            args.request_dataset, args.request_manifest
        )
        warmup_records = dataset_workload.load_jsonl(args.warmup_request_dataset)
        actual_warmup_sha = dataset_workload.sha256_file(args.warmup_request_dataset)
        if actual_warmup_sha != dataset_manifest.get("warmup_sha256"):
            raise ValueError("SHA-256 do workload de warm-up diverge do manifesto.")
        if args.smoke:
            dataset_records = dataset_records[:3]
            warmup_records = warmup_records[:1]
        largest_prompt = max(
            row.get("rendered_prompt_tokens") or 0
            for row in [*dataset_records, *warmup_records]
        )
        if largest_prompt + OUTPUT_TOKENS > cfg["context_window"]:
            raise ValueError(
                f"Workload MOPEP exige ao menos {largest_prompt + OUTPUT_TOKENS} tokens de contexto; "
                f"configuração declara {cfg['context_window']}."
            )
        count = len(dataset_records)
        if args.benchmark_profile == "mopep-review-replay":
            if not args.replay_responses:
                raise ValueError("mopep-review-replay exige --replay-responses.")
            replay_responses, replay_sha256, replay_manifest = dataset_workload.load_replay(args.replay_responses)
            replay_model_sha = replay_manifest.get("model_sha256")
            current_model_sha = (model_identity or {}).get("sha256")
            if replay_model_sha and current_model_sha and replay_model_sha != current_model_sha:
                raise ValueError("O replay canônico foi gerado com outro artefato de modelo.")
            if replay_manifest.get("workload_sha256") != dataset_manifest.get("workload_sha256"):
                raise ValueError("O replay canônico pertence a outro workload.")
    base = Path(args.results)
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        label = execution_label(smoke=args.smoke, scenarios=args.scenarios,
                                input_tokens=args.input_tokens, result_name=args.result_name)
        output = runtime_root(base, cfg["runtime"], stamp, label)
        prepare(output)
        experiment_id = f"{stamp}-{slug(cfg['runtime'])}-{label}"
        run_start_epoch_s = time.time()
        hardware = hardware_snapshot()
        effective_runtime_environment = runtime_environment(cfg)
        workload = workload_contract(args.scenarios, args.mode) if not mopep_profile else {
            "kind": "mopep-private-jsonl",
            "profile": args.benchmark_profile,
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            "workload_sha256": dataset_manifest["workload_sha256"],
            "warmup_sha256": dataset_manifest["warmup_sha256"],
            "request_order_sha256": dataset_manifest["request_order_sha256"],
            "prompt_sha256": dataset_manifest["prompt_sha256"],
            "review_prompt_sha256": dataset_manifest["review_prompt_sha256"],
            "bucket_thresholds": dataset_manifest["bucket_thresholds"],
            "replay_responses_sha256": replay_sha256,
            "closed_loop_inputs_diverge_after_turn_one": args.benchmark_profile == "mopep-review-closed-loop",
            "prompt_policy": "independent MOPEP BMC requests grouped by frozen calibration terciles",
        }
        effective_scenarios = ["short", "medium", "heavy"] if mopep_profile else args.scenarios
        manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
                    "experiment_id": experiment_id,
                    "results_layout": "runtime/timestamp/human-name/{html,json,csv,logs,text}",
                    "result_name": label, "runtime_directory": slug(cfg["runtime"]),
                    "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
                    "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
                    "scenarios": effective_scenarios, "input_tokens": args.input_tokens,
                    "output_tokens": OUTPUT_TOKENS, "seed": args.seed, "profile": "synchronous",
                    "generation_parameters": {
                        "temperature": 0, "top_p": 1,
                        "max_output_tokens": OUTPUT_TOKENS, "stream": True,
                    },
                    "mode": args.mode, "benchmark_profile": args.benchmark_profile,
                    "conversation_turns": args.conversation_turns,
                    "workload": workload,
                    "prompt_policy": workload["prompt_policy"],
                    "conversation_fixture": {"path": conversation_fixture["path"], "sha256": conversation_fixture["sha256"],
                                             "name": conversation_fixture["data"].get("name"),
                                             "version": conversation_fixture["data"].get("version")} if conversation_fixture else None,
                    "warmup_conversation_fixture": {"path": warmup_conversation_fixture["path"],
                                                     "sha256": warmup_conversation_fixture["sha256"],
                                                     "name": warmup_conversation_fixture["data"].get("name"),
                                                     "version": warmup_conversation_fixture["data"].get("version")} if warmup_conversation_fixture else None,
                    "request_dataset": ({
                        "manifest_sha256": dataset_workload.sha256_file(args.request_manifest),
                        **dataset_manifest,
                    } if mopep_profile else None),
                    "conversation_first_request": "first-request.json is a separate lifecycle probe and never enters formal MOPEP or conversational distributions",
                    "blocks": [],
                    "model_availability": availability,
                    "model_identity": model_identity,
                    "hardware": hardware,
                    "runtime_process": {
                        "ownership": "owned-process-group" if args.launch else "existing-server",
                        "argv": None,
                    },
                    "runtime_environment": effective_runtime_environment,
                    "cache_policy_details": describe_cache_policy(
                        cfg, [], bool(args.launch), effective_runtime_environment
                    ),
                    "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
                    "telemetry": {"gpu_source": "local nvidia-smi", "system_source": "psutil host CPU/RAM/disk counters",
                                  "event_source": "events.csv phase markers", "kv_metrics_requested": args.collect_kv_metrics,
                                  "native_monitor_enabled": not args.disable_native_monitor,
                                  "native_sampling": ("approximately 1 Hz" if not args.disable_native_monitor else None),
                                  "limits": "sampling is not per-token; PCIe copy time is not directly measured",
                                  "prometheus": {"enabled": bool(args.prometheus_url), "url": args.prometheus_url,
                                                 "query": args.prometheus_query if args.prometheus_url else None,
                                                 "step": args.prometheus_step if args.prometheus_url else None}},
                    "python": sys.version, "platform": platform.platform(),
                    "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
        write_json(artifact(output, "manifest.json"), redact(manifest, secret))
        write_json(artifact(output, "client-packages.json"), {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
        write_json(artifact(output, "gpu-before.json"), capture(["nvidia-smi"]))
        monitor = Monitor(output, cfg, secret, args.collect_kv_metrics,
                          collect_resources=not args.disable_native_monitor)
        rows = []
        observability_requests = []
        mopep_response_rows = []
        launch, origin, cleanup_error, dataset_error = None, None, None, None
        lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
                     "status": "running", "initial_state_note": args.initial_state,
                     "startup_timeout_s": args.startup_timeout}
        print(f"Resultados: {output.resolve()}", flush=True)
        try:
            monitor.start()
            if args.launch:
                reject_sensitive_argv(args.launch_extra_args)
                launch = Launch(cfg, args.launch, output, args.launch_extra_args, args.launch_executable)
                reject_sensitive_argv(launch.argv)
                lifecycle["argv"] = redact(launch.argv, secret)
                manifest["runtime_process"]["argv"] = lifecycle["argv"]
                manifest["cache_policy_details"] = describe_cache_policy(
                    cfg, launch.argv, True, effective_runtime_environment
                )
                monitor.set_phase("process_startup")
                origin = launch.start()
                lifecycle["pid"] = launch.process.pid
            lifecycle_report(output, redact(lifecycle, secret))
            monitor.set_phase("server_readiness")
            lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
            lifecycle_report(output, redact(lifecycle, secret))
            monitor.set_phase("first_request")
            print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
            lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
                                                       artifact(output, "first-request.json"), origin)
            observability_requests.append(lifecycle_observability_row(
                lifecycle["first_request"], experiment_id=experiment_id, runtime=cfg["runtime"],
                model=cfg["model"], phase="first_request"))
            lifecycle_report(output, redact(lifecycle, secret))
            if mopep_profile:
                def execute_mopep_block(records, phase, scenario, rep):
                    prefix = f"r{rep}-{scenario}-{phase}"
                    monitor.set_phase(prefix)
                    report = dataset_workload.run_batch(
                        cfg=cfg, records=records, profile=args.benchmark_profile,
                        timeout=args.timeout, secret=secret, request=stream_messages_request,
                        replay=replay_responses,
                    )
                    raw = redact(report, secret)
                    block_responses = dataset_workload.response_rows(raw)
                    for response in block_responses:
                        response.update({
                            "experiment_id": experiment_id,
                            "runtime": cfg["runtime"],
                            "model": cfg["model"],
                            "benchmark_phase": phase,
                            "repetition": rep,
                        })
                    mopep_response_rows.extend(block_responses)
                    observability_requests.extend(observability_request_rows(
                        raw, experiment_id=experiment_id, runtime=cfg["runtime"], model=cfg["model"],
                        block=prefix, benchmark_phase=phase, scenario=scenario, repetition=rep,
                    ))
                    write_json(artifact(output, f"{prefix}.json"), raw)
                    write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
                    summary = summarize(raw)
                    expected = len(records) * (2 if args.benchmark_profile == "mopep-review-closed-loop" else 1)
                    summary["expected"] = expected
                    summary["missing_request_count"] = max(0, expected - sum(
                        summary[key] for key in ("successful_request_count", "errored_request_count", "incomplete_request_count")
                    ))
                    manifest["blocks"].append({
                        "prefix": prefix, "mode": args.benchmark_profile, "phase": phase,
                        "scenario": scenario, "repetition": rep, "expected_requests": expected,
                        "requests_sha256": summary["requests_sha256"],
                    })
                    rows.append({
                        "runtime": cfg["runtime"], "model": cfg["model"],
                        "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
                        "mode": args.benchmark_profile, "phase": phase, "scenario": scenario,
                        "repetition": rep, **summary,
                    })
                    write_json(artifact(output, "manifest.json"), redact(manifest, secret))
                    write_summary(output, rows)
                    if summary["successful_request_count"] != expected or summary["errored_request_count"] or summary["incomplete_request_count"]:
                        raise RuntimeError(f"{prefix}: requisições MOPEP falharam ou execução incompleta.")

                for rep in range(1, repetitions + 1):
                    execute_mopep_block(warmup_records, "warmup", "warmup", rep)
                    for bucket in ("short", "medium", "heavy"):
                        selected = [row for row in dataset_records if row["bucket"] == bucket]
                        if selected:
                            execute_mopep_block(selected, "measure", bucket, rep)
            else:
                for rep in range(repetitions):
                    # Rotação balanceia parcialmente a posição dos cenários entre repetições.
                    names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
                    for name in names:
                        for phase, n in (("warmup", args.warmup), ("measure", count)):
                            if not n:
                                continue
                            seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
                            if phase == "warmup":
                                seed += 1_000_000
                            prefix = f"r{rep+1}-{name}-{phase}"
                            monitor.set_phase(prefix)
                            config = scenario_config(cfg, name, n, seed, secret, args.timeout)
                            config["mode"] = args.mode
                            active_fixture = warmup_conversation_fixture if phase == "warmup" else conversation_fixture
                            block_workload = workload_contract([name], args.mode)
                            config["workload"] = block_workload
                            config["prompt_policy"] = block_workload["prompt_policy"]
                            if conversation_mode:
                                config["conversation_turns"] = args.conversation_turns
                                config["conversation_fixture_sha256"] = active_fixture["sha256"]
                                if args.mode == "replay" and name in NAMED_WORKLOADS:
                                    config["spec_note"] = (
                                        "Warmup e medição usam fixtures separadas. O histórico fixo mais próximo "
                                        "do alvo é selecionado, pares antigos podem ser removidos e contexto "
                                        "sintético determinístico completa o prompt sem exceder o alvo. "
                                        "usage.prompt_tokens do runtime é a medida real."
                                    )
                                else:
                                    config["spec_note"] = (
                                        "Warmup usa fixture separada; medição usa a fixture versionada. "
                                        "usage.prompt_tokens do runtime é a medida real quando disponível."
                                    )
                            write_json(artifact(output, f"{prefix}-config.json"), redact(config, secret))
                            expected = n * args.conversation_turns if conversation_mode else n
                            unit = "conversas" if conversation_mode else "requisições"
                            print(f"{prefix}: {n} {unit}, uma chamada por vez", flush=True)
                            if conversation_mode:
                                report = run_conversational_batch(cfg, measurement_tokenizer, name, n, args.conversation_turns,
                                                                  args.timeout, secret, active_fixture, args.mode)
                            else:
                                report = run_stream_batch(cfg, measurement_tokenizer, name, n, args.timeout, secret, seed)
                            raw = redact(report, secret)
                            observability_requests.extend(observability_request_rows(
                                raw, experiment_id=experiment_id, runtime=cfg["runtime"], model=cfg["model"],
                                block=prefix, benchmark_phase=phase, scenario=name, repetition=rep + 1))
                            write_json(artifact(output, f"{prefix}.json"), raw)
                            turns = turn_manifest(raw) if conversation_mode else []
                            conversation_file = None
                            if conversation_mode:
                                write_json(artifact(output, f"{prefix}-turns.json"), turns)
                                conversation_file = f"{prefix}-conversation.json"
                                write_json(artifact(output, conversation_file), conversation_artifact(raw, conversation_fixture))
                            write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
                            summary = summarize(raw)
                            summary["expected"] = expected
                            summary["missing_request_count"] = max(0, expected - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
                            manifest["blocks"].append({"prefix": prefix, "mode": args.mode, "phase": phase,
                                                       "scenario": name, "repetition": rep+1, "seed": seed,
                                                       "conversations": n if conversation_mode else None,
                                                       "conversation_turns": args.conversation_turns if conversation_mode else None,
                                                       "expected_requests": expected,
                                                       "requests_sha256": summary["requests_sha256"],
                                                       "conversation_artifact": conversation_file,
                                                       "turns": turns if conversation_mode else None})
                            write_json(artifact(output, "manifest.json"), redact(manifest, secret))
                            rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
                                         "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
                                         "mode": args.mode, "phase": phase, "scenario": name, "repetition": rep+1, **summary})
                            write_summary(output, rows)
                            if summary["successful_request_count"] != expected or summary["errored_request_count"] or summary["incomplete_request_count"]:
                                raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
            monitor.set_phase("warm_reference")
            lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
                                                         artifact(output, "warm-reference.json"))
            observability_requests.append(lifecycle_observability_row(
                lifecycle["warm_reference"], experiment_id=experiment_id, runtime=cfg["runtime"],
                model=cfg["model"], phase="warm_reference"))
            manifest["status"] = lifecycle["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = redact(str(exc), secret)
            lifecycle["status"] = manifest["status"]
            lifecycle["error"] = manifest["error"]
            raise
        finally:
            for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
                if artifact(output, filename).exists():
                    lifecycle[key] = json.loads(artifact(output, filename).read_text(encoding="utf-8"))
                    uid = f"{experiment_id}:{key}:1"
                    if not any(row and row.get("request_uid") == uid for row in observability_requests):
                        observability_requests.append(lifecycle_observability_row(
                            lifecycle[key], experiment_id=experiment_id, runtime=cfg["runtime"],
                            model=cfg["model"], phase=key))
            observability_requests = [row for row in observability_requests if row is not None]
            if mopep_profile and mopep_response_rows:
                responses_path = artifact(output, "responses.jsonl")
                dataset_workload.write_responses(responses_path, mopep_response_rows)
                response_manifest = {
                    "schema_version": 1,
                    "source_runtime": cfg["runtime"],
                    "source_runtime_version": cfg.get("runtime_version"),
                    "model": cfg["model"],
                    "model_sha256": (model_identity or {}).get("sha256"),
                    "workload_sha256": dataset_manifest["workload_sha256"],
                    "prompt_sha256": dataset_manifest["prompt_sha256"],
                    "benchmark_profile": args.benchmark_profile,
                    "responses_sha256": dataset_workload.sha256_file(responses_path),
                    "row_count": len(mopep_response_rows),
                }
                response_manifest_path = artifact(output, "responses-manifest.json")
                write_json(response_manifest_path, response_manifest)
                manifest["responses"] = {
                    "path": str(responses_path.relative_to(output)),
                    "manifest_path": str(response_manifest_path.relative_to(output)),
                    "sha256": response_manifest["responses_sha256"],
                    "row_count": len(mopep_response_rows),
                }
                if args.responses_out:
                    external = Path(args.responses_out)
                    external.parent.mkdir(parents=True, exist_ok=True)
                    dataset_workload.write_responses(external, mopep_response_rows)
                    write_json(dataset_workload.replay_manifest_path(external), response_manifest)
            if args.prometheus_url:
                monitor.set_phase("prometheus_export")
                try:
                    from prometheus_dataset import export_dataset
                    manifest["observability_request_count"] = len(observability_requests)
                    manifest["dataset"] = export_dataset(
                        output=output, manifest=redact(manifest, secret), requests=observability_requests,
                        prometheus_url=args.prometheus_url, query=args.prometheus_query,
                        step=args.prometheus_step, run_start_epoch_s=run_start_epoch_s,
                        run_end_epoch_s=time.time(), margin_seconds=args.prometheus_margin_seconds,
                    )
                except Exception as exc:
                    dataset_error = redact(str(exc), secret)
                    manifest["prometheus_export_error"] = dataset_error
                    manifest["status"] = lifecycle["status"] = "failed"
            if launch is not None:
                monitor.set_phase("server_shutdown")
                try:
                    launch.close()
                    lifecycle["server_cleanup"] = "stopped owned process group only"
                except Exception as exc:
                    cleanup_error = redact(str(exc), secret)
                    lifecycle["server_cleanup_error"] = cleanup_error
                    lifecycle["status"] = manifest["status"] = "failed"
            lifecycle_report(output, redact(lifecycle, secret))
            monitor.stop()
            write_json(artifact(output, "gpu-after.json"), capture(["nvidia-smi"]))
            write_summary(output, rows)
            manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(artifact(output, "manifest.json"), redact(manifest, secret))
            if args.prometheus_url and artifact(output, "dataset-manifest.json").is_file():
                try:
                    from prometheus_dataset import finalize_checksums
                    finalize_checksums(output)
                except Exception as exc:
                    dataset_error = redact(str(exc), secret)
                    manifest["checksum_finalize_error"] = dataset_error
                    manifest["status"] = lifecycle["status"] = "failed"
                    write_json(artifact(output, "manifest.json"), redact(manifest, secret))
        if cleanup_error:
            raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
        if dataset_error:
            raise RuntimeError(f"Falha ao exportar dataset Prometheus: {dataset_error}")
        print(f"Concluído. Abra {artifact(output, 'lifecycle.html')} e {artifact(output, 'summary.html')}")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use um inteiro positivo.")
    return number


def validate_config(args):
    """Valida a configuração estática antes de iniciar runtime e coletores."""
    cfg = load_config(args.config)
    if args.benchmark_profile == "generic":
        validate_run(cfg, args.scenarios, args.smoke)
    print(json.dumps({
        "config": str(Path(args.config).resolve()),
        "context_window": cfg["context_window"],
        "benchmark_profile": args.benchmark_profile,
        "scenarios": args.scenarios,
        "smoke": args.smoke,
        "status": "valid",
    }, ensure_ascii=False, indent=2))


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Use zero ou um inteiro positivo.")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Use zero ou um número positivo.")
    return number


def rebuild_report(args):
    """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
    output = Path(args.output)
    rows = json.loads(locate(output, "summary.json").read_text())
    manifest = json.loads(locate(output, "manifest.json").read_text())
    for row in rows:
        prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
        raw = json.loads(locate(output, f"{prefix}.json").read_text())
        row.update(summarize(raw))
        expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
        row["expected"] = expected
        row["missing_request_count"] = max(0, expected - sum(row[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count"))) if expected is not None else None
        write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
    write_summary(output, rows)
    print(f"Relatório atualizado: {artifact(output, 'summary.html')}; dados brutos preservados.")


def main():
    from prometheus_dataset import DEFAULT_QUERY

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
    report.add_argument("--output", required=True)
    report.set_defaults(func=rebuild_report)
    prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
    prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    prep.add_argument("--revision", default="main")
    prep.add_argument("--output", default="tokenizer")
    prep.set_defaults(func=prepare_tokenizer)
    validate = commands.add_parser(
        "validate",
        help="Valida config, perfil e contexto sem iniciar runtime ou coletores.",
    )
    validate.add_argument("--config", required=True)
    validate.add_argument("--input-tokens", nargs="+", type=positive)
    validate.add_argument(
        "--scenarios", nargs="+", choices=list(WORKLOADS),
        default=["short", "medium", "long"],
    )
    validate.add_argument(
        "--benchmark-profile",
        choices=["generic", "mopep-single", "mopep-review-replay", "mopep-review-closed-loop"],
        default="generic",
    )
    validate.add_argument("--smoke", action="store_true")
    validate.set_defaults(func=validate_config)
    cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
    cmd.add_argument("--model-metadata", help="JSON de identidade gerado antes da medição; não pode conter credenciais.")
    cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
    cmd.add_argument("--disable-native-monitor", action="store_true",
                     help="Desativa o coletor CSV legado quando Prometheus já coleta os recursos.")
    cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
    cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
    cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium", "long"])
    cmd.add_argument("--requests", type=positive, default=50, help="Requisições de medição por cenário e repetição; replay percorre perguntas distintas da fixture.")
    cmd.add_argument("--repetitions", type=positive, default=1)
    cmd.add_argument("--warmup", type=nonnegative, default=3)
    cmd.add_argument("--mode", choices=["independent", "closed-loop", "replay"], default="replay",
                     help="independent preserva uma requisição sem histórico; closed-loop acumula respostas reais; replay usa assistant fixo do fixture.")
    cmd.add_argument("--conversation-turns", type=positive, default=1,
                     help="Turnos por conversa em --mode closed-loop ou replay; cada turno envia o histórico completo.")
    cmd.add_argument("--conversation-fixture", default=str(DEFAULT_CONVERSATION_FIXTURE),
                     help="Fixture JSON versionado com system e lista fixa de user turns; replay tambem exige assistant nos turnos anteriores.")
    cmd.add_argument("--warmup-conversation-fixture", default=str(DEFAULT_WARMUP_CONVERSATION_FIXTURE),
                     help="Fixture separada para warmup; nunca é usada na medição formal.")
    cmd.add_argument("--benchmark-profile", choices=["generic", "mopep-single", "mopep-review-replay", "mopep-review-closed-loop"], default="generic")
    cmd.add_argument("--request-dataset", help="workload.jsonl privado para perfis MOPEP.")
    cmd.add_argument("--warmup-request-dataset", help="warmup.jsonl privado e separado para perfis MOPEP.")
    cmd.add_argument("--request-manifest", help="workload-manifest.json com hashes e tercis congelados.")
    cmd.add_argument("--replay-responses", help="responses.jsonl canônico exigido por mopep-review-replay.")
    cmd.add_argument("--responses-out", help="Cópia explícita das respostas para alimentar replay ou análise privada.")
    cmd.add_argument("--seed", type=positive, default=42)
    cmd.add_argument("--timeout", type=positive, default=300)
    cmd.add_argument("--results", default="results")
    cmd.add_argument("--result-name", help="Nome humano da execução na pasta timestamp; sem isso é derivado do modo/cenário.")
    cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
    cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
    cmd.add_argument("--launch-extra-args", nargs="*", default=[],
                     help="Argumentos experimentais acrescentados ao argv do launch, sem editar o JSON; registrados no lifecycle.")
    cmd.add_argument("--launch-executable", help="Substitui argv[0] do launch pelo executável resolvido no ambiente do runtime.")
    cmd.add_argument("--launch-extra-args-json", default="[]", help="Array JSON de argumentos do runtime, preservando flags e espaços.")
    cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
    cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
    cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
    cmd.add_argument("--prometheus-url",
                     help="Ativa exportação obrigatória da janela do Prometheus ao fim da execução.")
    cmd.add_argument("--prometheus-query", default=DEFAULT_QUERY,
                     help="PromQL usada para o dataset temporal.")
    cmd.add_argument("--prometheus-step", default="500ms",
                     help="Resolução da query_range (deve acompanhar o scrape interval).")
    cmd.add_argument("--prometheus-margin-seconds", type=nonnegative_float, default=1.0,
                     help="Margem antes/depois da janela exportada.")
    cmd.set_defaults(func=run)
    args = parser.parse_args()
    if hasattr(args, "launch_extra_args_json"):
        extra = json.loads(args.launch_extra_args_json)
        if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
            parser.error("--launch-extra-args-json deve ser array de strings")
        args.launch_extra_args.extend(extra)
    if getattr(args, "input_tokens", None):
        if len(set(args.input_tokens)) != len(args.input_tokens):
            parser.error("Não repita comprimentos em --input-tokens.")
        args.scenarios = []
        for size in args.input_tokens:
            name = f"ctx{size}"
            WORKLOADS[name] = size
            args.scenarios.append(name)
    if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Não repita cenários na lista.")
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
