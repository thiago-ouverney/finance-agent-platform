# Código completo: referência linha a linha

Leia junto com [a explicação detalhada](codigo-explicado.md). Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.

## bench.py

SHA-256: `648fb2280cc7af9c2d6419fd8f5d0d57ac84aa4e83f825ace59adbf7942e599f`.

| Função/classe | Linhas |
|---|---|
| `_synthetic_prompt` | 34–43 |
| `estimate_messages_tokens` | 46–57 |
| `request_sha256` | 60–62 |
| `stream_messages_request` | 65–115 |
| `stream_request` | 118–119 |
| `stream_metrics` | 122–141 |
| `run_stream_batch` | 144–161 |
| `load_conversation_fixture` | 164–177 |
| `validate_conversation_fixture_for_mode` | 180–185 |
| `fixture_messages_for_turn` | 188–202 |
| `run_conversational_batch` | 205–251 |
| `local_model_check` | 254–274 |
| `write_json` | 277–278 |
| `redact` | 281–289 |
| `capture` | 292–297 |
| `load_config` | 300–316 |
| `validate_run` | 319–332 |
| `tokenizer_digest` | 335–344 |
| `prepare_tokenizer` | 347–359 |
| `scenario_config` | 362–381 |
| `Monitor` | 384–559 |
| `percentile` | 562–568 |
| `summarize` | 571–635 |
| `requests_digest` | 638–645 |
| `turn_manifest` | 648–658 |
| `conversation_artifact` | 661–683 |
| `write_requests_csv` | 686–705 |
| `write_summary` | 708–718 |
| `run` | 721–897 |
| `positive` | 900–904 |
| `nonnegative` | 907–911 |
| `rebuild_report` | 914–928 |
| `main` | 931–998 |
| `__init__` | 386–393 |
| `set_phase` | 395–403 |
| `start` | 405–415 |
| `log` | 417–421 |
| `kv_loop` | 423–447 |
| `loop` | 449–497 |
| `stop` | 499–510 |
| `write_telemetry_summary` | 512–559 |
| `read_rows` | 514–519 |

```text
0001 | #!/usr/bin/env python3
0002 | """Um usuário: inicialização, primeira resposta, aquecimento e GuideLLM sequencial."""
0003 | from __future__ import annotations
0004 |
0005 | import argparse
0006 | import asyncio
0007 | import csv
0008 | import hashlib
0009 | import importlib.metadata
0010 | import json
0011 | import math
0012 | import os
0013 | from pathlib import Path
0014 | import platform
0015 | import re
0016 | import subprocess
0017 | import sys
0018 | import threading
0019 | import time
0020 | from datetime import datetime, timezone
0021 | from urllib.parse import urlsplit
0022 |
0023 | from results_layout import artifact, execution_label, href, locate, prepare, runtime_root, slug
0024 |
0025 | ROOT = Path(__file__).resolve().parent
0026 | VERSION = "0.4.1"
0027 | GUIDELLM_VERSION = "0.7.4"
0028 | WORKLOADS = {"short": 256, "medium": 2048, "long": 8192}
0029 | OUTPUT_TOKENS = 128
0030 | DEFAULT_CONVERSATION_FIXTURE = ROOT / "workloads" / "conversations" / "qwen_chat_bench_v2.json"
0031 | DEFAULT_WARMUP_CONVERSATION_FIXTURE = ROOT / "workloads" / "conversations" / "qwen_chat_warmup_v1.json"
0032 |
0033 |
0034 | def _synthetic_prompt(tokenizer, target_tokens: int, seed: str | int) -> str:
0035 |     """Cria um prompt determinístico e distinto com exatamente o tamanho-alvo."""
0036 |     digest = hashlib.sha256(str(seed).encode()).digest()
0037 |     marker = " ".join(f"word{value % 100}" for value in digest[:8])
0038 |     sentence = "Explique de forma objetiva este conceito para um estudante de estatística. "
0039 |     text = f"{marker}. {sentence}"
0040 |     while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
0041 |         text += sentence
0042 |     ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
0043 |     return tokenizer.decode(ids, skip_special_tokens=False)
0044 |
0045 |
0046 | def estimate_messages_tokens(tokenizer, messages) -> int:
0047 |     try:
0048 |         encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
0049 |         if hasattr(encoded, "input_ids"):
0050 |             encoded = encoded.input_ids
0051 |         elif isinstance(encoded, dict):
0052 |             encoded = encoded["input_ids"]
0053 |         if encoded and isinstance(encoded[0], (list, tuple)):
0054 |             encoded = encoded[0]
0055 |         return len(encoded)
0056 |     except Exception:
0057 |         return sum(len(tokenizer.encode(m.get("content", ""), add_special_tokens=False)) for m in messages)
0058 |
0059 |
0060 | def request_sha256(body) -> str:
0061 |     payload = {k: body.get(k) for k in ("messages", "max_tokens")}
0062 |     return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0063 |
0064 |
0065 | def stream_messages_request(cfg, messages, timeout, secret="", max_tokens=OUTPUT_TOKENS, metadata=None):
0066 |     """Mede uma requisição SSE com ``time.perf_counter`` por requisição.
0067 |
0068 |     ``first_token_time``/``last_token_time`` são os instantes do primeiro/último
0069 |     evento SSE com conteúdo. A API OpenAI não promete um evento por token; por
0070 |     isso a resolução é a do evento de conteúdo, enquanto ``completion_tokens``
0071 |     vem exclusivamente de ``usage`` quando o servidor o fornece.
0072 |     """
0073 |     import httpx
0074 |     body = {"model": cfg["model"], "messages": messages,
0075 |             "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
0076 |             "stream": True, "stream_options": {"include_usage": True}}
0077 |     started = time.perf_counter()
0078 |     first = last = None
0079 |     chunks = 0
0080 |     content = ""
0081 |     usage = None
0082 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0083 |     with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0084 |         with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as response:
0085 |             response.raise_for_status()
0086 |             if "text/event-stream" not in response.headers.get("content-type", ""):
0087 |                 raise ValueError("A API não respondeu com SSE.")
0088 |             for line in response.iter_lines():
0089 |                 if not line.startswith("data:"):
0090 |                     continue
0091 |                 value = line[5:].strip()
0092 |                 if value == "[DONE]":
0093 |                     break
0094 |                 event = json.loads(value)
0095 |                 if "error" in event:
0096 |                     raise ValueError(f"Erro no stream: {event['error']}")
0097 |                 if event.get("usage"):
0098 |                     usage = event["usage"]
0099 |                 text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0100 |                 if text:
0101 |                     now = time.perf_counter()
0102 |                     first = first or now
0103 |                     last = now
0104 |                     chunks += 1
0105 |                     content += text
0106 |     ended = time.perf_counter()
0107 |     e2e = ended - started
0108 |     metrics = stream_metrics(started, first, last, ended, usage)
0109 |     metadata = dict(metadata or {})
0110 |     if "history_tokens_estimate" in metadata:
0111 |         metadata["history_tokens"] = metrics["prompt_tokens"] if metrics["prompt_tokens"] is not None else metadata["history_tokens_estimate"]
0112 |     return {**metrics,
0113 |             "stream_content_event_count": chunks, "output": content, "usage_observed": usage is not None,
0114 |             "request_sha256": request_sha256(body),
0115 |             "request_args": json.dumps({"body": body}, ensure_ascii=False), **metadata}
0116 |
0117 |
0118 | def stream_request(cfg, prompt, timeout, secret=""):
0119 |     return stream_messages_request(cfg, [{"role": "user", "content": prompt}], timeout, secret)
0120 |
0121 |
0122 | def stream_metrics(request_start_time, first_token_time, last_token_time, request_end_time, usage):
0123 |     """Calcula métricas exclusivamente de timestamps monotônicos do cliente."""
0124 |     usage = usage if isinstance(usage, dict) else {}
0125 |     completion = usage.get("completion_tokens") if type(usage.get("completion_tokens")) is int else None
0126 |     prompt_tokens = usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int else None
0127 |     total = usage.get("total_tokens") if type(usage.get("total_tokens")) is int else None
0128 |     ttft = first_token_time - request_start_time if first_token_time is not None else None
0129 |     e2e = request_end_time - request_start_time
0130 |     generation = (last_token_time - first_token_time) if first_token_time is not None and last_token_time is not None and completion is not None and completion > 1 else None
0131 |     inter = generation / (completion - 1) if generation is not None else None
0132 |     decode = completion / generation if completion is not None and generation and generation > 0 else None
0133 |     effective = completion / e2e if completion is not None and e2e > 0 else None
0134 |     return {"request_start_time": request_start_time, "first_token_time": first_token_time,
0135 |             "request_end_time": request_end_time, "time_to_first_token_seconds": ttft,
0136 |             "generation_time_seconds": generation, "end_to_end_latency_seconds": e2e,
0137 |             "completion_tokens": completion, "prompt_tokens": prompt_tokens, "total_tokens": total,
0138 |             "decode_tokens_per_second": decode, "end_to_end_tokens_per_second": effective,
0139 |             "inter_token_latency_seconds": inter, "time_to_first_token_ms": ttft * 1000 if ttft is not None else None,
0140 |             "request_latency": e2e, "inter_token_latency_ms": inter * 1000 if inter is not None else None,
0141 |             "output_tokens": completion, "decode_tokens_s": decode, "effective_tokens_s": effective}
0142 |
0143 |
0144 | def run_stream_batch(cfg, tokenizer, scenario, count, timeout, secret="", seed=0):
0145 |     samples = [
0146 |         (f"{seed}:{index}", _synthetic_prompt(tokenizer, WORKLOADS[scenario], f"{seed}:{index}"))
0147 |         for index in range(count)
0148 |     ]
0149 |     successful, errored = [], []
0150 |     for index, (workload_seed, prompt) in enumerate(samples):
0151 |         try:
0152 |             row = stream_request(cfg, prompt, timeout, secret)
0153 |             row.update({"mode": "independent", "request_id": f"{scenario}-independent-{index+1}",
0154 |                         "workload_seed": workload_seed})
0155 |             row["status"] = "successful"
0156 |             successful.append(row)
0157 |         except Exception as exc:
0158 |             errored.append({"status": "errored", "mode": "independent",
0159 |                             "request_id": f"{scenario}-independent-{index+1}",
0160 |                             "workload_seed": workload_seed, "error": str(exc)})
0161 |     return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}
0162 |
0163 |
0164 | def load_conversation_fixture(path):
0165 |     path = Path(path)
0166 |     data = json.loads(path.read_text(encoding="utf-8"))
0167 |     if not isinstance(data.get("system"), str) or not data["system"].strip():
0168 |         raise ValueError("Fixture conversacional precisa de campo system textual nao vazio.")
0169 |     if not isinstance(data.get("turns"), list) or not data["turns"]:
0170 |         raise ValueError("Fixture conversacional precisa de lista nao vazia em turns.")
0171 |     for index, turn in enumerate(data["turns"], 1):
0172 |         if not isinstance(turn, dict) or not isinstance(turn.get("user"), str) or not turn["user"].strip():
0173 |             raise ValueError(f"Turno {index} do fixture precisa de user textual.")
0174 |         if "assistant" in turn and (not isinstance(turn["assistant"], str) or not turn["assistant"].strip()):
0175 |             raise ValueError(f"Turno {index} do fixture tem assistant vazio/invalido.")
0176 |     digest = hashlib.sha256(path.read_bytes()).hexdigest()
0177 |     return {"path": str(path), "sha256": digest, "data": data}
0178 |
0179 |
0180 | def validate_conversation_fixture_for_mode(fixture, loop_mode, turns):
0181 |     if loop_mode == "replay":
0182 |         missing = [index for index, turn in enumerate(fixture["data"]["turns"], 1)
0183 |                    if "assistant" not in turn]
0184 |         if missing:
0185 |             raise ValueError(f"--mode replay exige assistant fixo em cada turno usado do fixture; ausente em: {missing}.")
0186 |
0187 |
0188 | def fixture_messages_for_turn(fixture, turn_index, prior_assistant_outputs, loop_mode):
0189 |     data = fixture["data"]
0190 |     messages = []
0191 |     if data.get("system"):
0192 |         messages.append({"role": "system", "content": data["system"]})
0193 |     for index, turn in enumerate(data["turns"][:turn_index], 1):
0194 |         messages.append({"role": "user", "content": turn["user"]})
0195 |         if index < turn_index:
0196 |             if loop_mode == "closed-loop":
0197 |                 messages.append({"role": "assistant", "content": prior_assistant_outputs[index - 1]})
0198 |             else:
0199 |                 if "assistant" not in turn:
0200 |                     raise ValueError("--mode replay exige assistant fixo nos turnos anteriores do fixture.")
0201 |                 messages.append({"role": "assistant", "content": turn["assistant"]})
0202 |     return messages
0203 |
0204 |
0205 | def run_conversational_batch(cfg, tokenizer, scenario, conversations, turns, timeout, secret="", fixture=None, loop_mode="closed-loop"):
0206 |     if fixture is None:
0207 |         fixture = load_conversation_fixture(DEFAULT_CONVERSATION_FIXTURE)
0208 |     if turns > len(fixture["data"]["turns"]):
0209 |         raise ValueError(f"--conversation-turns={turns} excede turnos disponiveis no fixture ({len(fixture['data']['turns'])}).")
0210 |     validate_conversation_fixture_for_mode(fixture, loop_mode, turns)
0211 |     target = WORKLOADS[scenario]
0212 |     successful, errored = [], []
0213 |     available_turns = max(1, len(fixture["data"]["turns"]) - turns + 1)
0214 |     if turns == 1 and scenario.startswith("ctx"):
0215 |         fitting_turns = []
0216 |         for candidate in range(1, available_turns + 1):
0217 |             candidate_messages = fixture_messages_for_turn(fixture, candidate, [], loop_mode)
0218 |             if estimate_messages_tokens(tokenizer, candidate_messages) <= target:
0219 |                 fitting_turns.append(candidate)
0220 |         if not fitting_turns:
0221 |             raise ValueError(f"Nenhum turno da fixture cabe em ctx{target} tokens.")
0222 |         available_turns = len(fitting_turns)
0223 |     else:
0224 |         fitting_turns = list(range(1, available_turns + 1))
0225 |     for conversation_index in range(1, conversations + 1):
0226 |         assistant_outputs = []
0227 |         # Com turns=1, percorremos a fixture em vez de repetir sempre a primeira
0228 |         # pergunta. Cada request continua sendo uma conversa nova e determinística,
0229 |         # mas pode carregar um histórico de tamanho diferente.
0230 |         if turns == 1:
0231 |             final_turn = fitting_turns[(conversation_index - 1) % available_turns]
0232 |         else:
0233 |             final_turn = turns
0234 |         for turn_index in range(final_turn - turns + 1, final_turn + 1):
0235 |             messages = fixture_messages_for_turn(fixture, turn_index, assistant_outputs, loop_mode)
0236 |             estimate = estimate_messages_tokens(tokenizer, messages)
0237 |             metadata = {"mode": loop_mode, "conversation_index": conversation_index,
0238 |                         "fixture_sha256": fixture["sha256"],
0239 |                         "fixture_name": fixture["data"].get("name"),
0240 |                         "turn_index": turn_index, "history_tokens_estimate": estimate,
0241 |                         "target_history_tokens": target, "messages_count": len(messages),
0242 |                         "request_id": f"{scenario}-c{conversation_index}-t{turn_index}"}
0243 |             try:
0244 |                 row = stream_messages_request(cfg, messages, timeout, secret, metadata=metadata)
0245 |                 row["status"] = "successful"
0246 |                 successful.append(row)
0247 |                 assistant_outputs.append(row.get("output") or "")
0248 |             except Exception as exc:
0249 |                 errored.append({"status": "errored", **metadata, "history_tokens": estimate, "error": str(exc)})
0250 |                 break
0251 |     return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}
0252 |
0253 |
0254 | def local_model_check(value):
0255 |     """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
0256 |     path = Path(value).expanduser().resolve()
0257 |     if path.is_file():
0258 |         if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
0259 |             raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
0260 |         files = [path]
0261 |     elif path.is_dir():
0262 |         files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
0263 |         for index in path.glob("*.index.json"):
0264 |             data = json.loads(index.read_text())
0265 |             missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
0266 |             if missing:
0267 |                 raise ValueError(f"Shards ausentes: {missing}")
0268 |     else:
0269 |         files = []
0270 |     if not files or any(f.stat().st_size == 0 for f in files):
0271 |         raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
0272 |     return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
0273 |             "files": len(files), "bytes": sum(f.stat().st_size for f in files),
0274 |             "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}
0275 |
0276 |
0277 | def write_json(path, data):
0278 |     Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0279 |
0280 |
0281 | def redact(value, secret):
0282 |     if isinstance(value, dict):
0283 |         return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization"} else redact(v, secret))
0284 |                 for k, v in value.items()}
0285 |     if isinstance(value, list):
0286 |         return [redact(v, secret) for v in value]
0287 |     if isinstance(value, str) and secret:
0288 |         return value.replace(secret, "[REDACTED]")
0289 |     return value
0290 |
0291 |
0292 | def capture(command):
0293 |     try:
0294 |         proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
0295 |         return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
0296 |     except (OSError, subprocess.TimeoutExpired) as exc:
0297 |         return {"unavailable": str(exc)}
0298 |
0299 |
0300 | def load_config(path):
0301 |     cfg = json.loads(Path(path).read_text(encoding="utf-8"))
0302 |     required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
0303 |                 "runtime_version", "model_artifact", "server_command", "notes"}
0304 |     if set(cfg) != required:
0305 |         raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
0306 |     if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
0307 |         raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
0308 |     if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
0309 |         raise ValueError("context_window deve ser um inteiro >= 512.")
0310 |     url = urlsplit(cfg["base_url"])
0311 |     if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
0312 |         raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
0313 |     if url.path not in {"", "/", "/v1", "/v1/"}:
0314 |         raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
0315 |     cfg["base_url"] = f"{url.scheme}://{url.netloc}"
0316 |     return cfg
0317 |
0318 |
0319 | def validate_run(cfg, scenarios, smoke):
0320 |     for scenario in scenarios:
0321 |         # Margem para o template; o servidor continua sendo a autoridade final.
0322 |         needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
0323 |         if cfg["context_window"] < needed:
0324 |             raise ValueError(f"{scenario} precisa de contexto declarado >= {needed}; "
0325 |                              "altere o servidor e depois o JSON, ou retire esse cenário.")
0326 |     if not smoke:
0327 |         if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
0328 |                for k in ("model", "runtime_version", "model_artifact", "server_command")):
0329 |             raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
0330 |         if cfg["cache_policy"] == "runtime-default-unverified":
0331 |             raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
0332 |                              "O script NÃO altera nem comprova a política de cache.")
0333 |
0334 |
0335 | def tokenizer_digest(path):
0336 |     path = Path(path)
0337 |     if not path.is_dir() or not (path / "tokenizer_config.json").exists():
0338 |         raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
0339 |     hashes = {}
0340 |     for file in sorted(path.rglob("*")):
0341 |         if file.is_file() and not file.name.startswith("."):
0342 |             hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
0343 |     digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
0344 |     return {"sha256": digest, "files": hashes}
0345 |
0346 |
0347 | def prepare_tokenizer(args):
0348 |     from huggingface_hub import HfApi
0349 |     from transformers import AutoTokenizer
0350 |     path = Path(args.output)
0351 |     if path.exists():
0352 |         raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
0353 |     revision = HfApi().model_info(args.model, revision=args.revision).sha
0354 |     tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
0355 |     path.mkdir(parents=True)
0356 |     tokenizer.save_pretrained(path)
0357 |     write_json(path / "source.json", {"model": args.model, "revision": revision})
0358 |     print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
0359 |     print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")
0360 |
0361 |
0362 | def scenario_config(cfg, name, count, seed, secret, timeout):
0363 |     return {
0364 |         "spec": {
0365 |             "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
0366 |                         "request_format": "/v1/chat/completions", "stream": True,
0367 |                         "validate_backend": False, "verify": True, "follow_redirects": False,
0368 |                         "timeout": timeout, "api_key": secret or None,
0369 |                         "extras": {"body": {"temperature": 0, "top_p": 1}}},
0370 |             "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
0371 |             "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
0372 |             "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
0373 |                           "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
0374 |             "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
0375 |                       "output_tokens": OUTPUT_TOKENS}],
0376 |             "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
0377 |             "seed": {"kind": "static", "value": seed},
0378 |             "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
0379 |             "outputs": [],
0380 |         }
0381 |     }
0382 |
0383 |
0384 | class Monitor:
0385 |     """Amostragem contínua e marcação de fases; não é um profiler PCIe."""
0386 |     def __init__(self, output, cfg=None, secret="", collect_kv=False):
0387 |         self.output = Path(output)
0388 |         self.stop_event = threading.Event()
0389 |         self.thread = None
0390 |         self.phase = "setup"
0391 |         self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
0392 |         self.kv_thread = None
0393 |         self.started_monotonic = None
0394 |
0395 |     def set_phase(self, phase, event=None):
0396 |         self.phase = phase
0397 |         utc = datetime.now(timezone.utc).isoformat()
0398 |         elapsed = (time.monotonic() - self.started_monotonic) if self.started_monotonic else 0.0
0399 |         self.log(f"[telemetria] utc={utc} decorrido={elapsed:.3f}s fase={phase} evento={event or 'phase_change'}")
0400 |         if hasattr(self, "events_handle"):
0401 |             writer = csv.writer(self.events_handle)
0402 |             writer.writerow([utc, time.monotonic(), elapsed, phase, event or "phase_change"])
0403 |             self.events_handle.flush()
0404 |
0405 |     def start(self):
0406 |         self.started_monotonic = time.monotonic()
0407 |         self.telemetry_handle = artifact(self.output, "telemetry.log").open("a", encoding="utf-8")
0408 |         self.events_handle = artifact(self.output, "events.csv").open("w", newline="", encoding="utf-8")
0409 |         csv.writer(self.events_handle).writerow(["utc", "monotonic_s", "elapsed_s", "phase", "event"])
0410 |         self.set_phase(self.phase, "monitor_started")
0411 |         self.thread = threading.Thread(target=self.loop, daemon=True)
0412 |         self.thread.start()
0413 |         if self.collect_kv:
0414 |             self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
0415 |             self.kv_thread.start()
0416 |
0417 |     def log(self, message):
0418 |         print(message, flush=True)
0419 |         if hasattr(self, "telemetry_handle"):
0420 |             self.telemetry_handle.write(message + "\n")
0421 |             self.telemetry_handle.flush()
0422 |
0423 |     def kv_loop(self):
0424 |         import httpx
0425 |         headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
0426 |         pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
0427 |         with artifact(self.output, "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
0428 |             writer = csv.writer(handle)
0429 |             writer.writerow(["utc", "elapsed_s", "phase", "series", "fraction"])
0430 |             while not self.stop_event.is_set():
0431 |                 phase = self.phase
0432 |                 try:
0433 |                     response = client.get(self.cfg["base_url"] + "/metrics")
0434 |                     response.raise_for_status()
0435 |                     matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
0436 |                     modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
0437 |                     for m in matches:
0438 |                         if modern and m[1] != "vllm:kv_cache_usage_perc":
0439 |                             continue
0440 |                         value = float(m[3])
0441 |                         if math.isfinite(value) and 0 <= value <= 1:
0442 |                             writer.writerow([datetime.now(timezone.utc).isoformat(), time.monotonic() - self.started_monotonic,
0443 |                                              phase, redact(m[1] + (m[2] or ""), self.secret), value])
0444 |                     handle.flush()
0445 |                 except (httpx.HTTPError, ValueError):
0446 |                     pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
0447 |                 self.stop_event.wait(1)
0448 |
0449 |     def loop(self):
0450 |         try:
0451 |             import psutil
0452 |         except ImportError:
0453 |             psutil = None
0454 |         with artifact(self.output, "gpu.csv").open("w", newline="", encoding="utf-8") as handle, \
0455 |              artifact(self.output, "system.csv").open("w", newline="", encoding="utf-8") as system_handle:
0456 |             writer = csv.writer(handle)
0457 |             writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "index", "name", "used_mib", "total_mib", "used_gib", "total_gib",
0458 |                              "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"])
0459 |             system_writer = csv.writer(system_handle)
0460 |             system_writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "cpu_util_pct", "ram_used_mib", "ram_available_mib", "ram_total_mib",
0461 |                                     "load1", "root_disk_used_mib", "root_disk_free_mib", "disk_read_bytes", "disk_write_bytes"])
0462 |             if psutil:
0463 |                 psutil.cpu_percent(interval=None)
0464 |             sample_number = 0
0465 |             while not self.stop_event.is_set():
0466 |                 sample_number += 1
0467 |                 phase = self.phase
0468 |                 utc = datetime.now(timezone.utc).isoformat()
0469 |                 result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
0470 |                                   "--format=csv,noheader,nounits"])
0471 |                 if result.get("returncode") != 0:
0472 |                     write_json(artifact(self.output, "gpu-unavailable.json"), result)
0473 |                 else:
0474 |                     gpu_rows = list(csv.reader(result["stdout"].splitlines(), skipinitialspace=True))
0475 |                     for row in gpu_rows:
0476 |                         used_mib, total_mib = float(row[2]), float(row[3])
0477 |                         vram_pct = (100 * used_mib / total_mib) if total_mib else None
0478 |                         writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, row[0], row[1], row[2], row[3],
0479 |                                          used_mib / 1024, total_mib / 1024, vram_pct, *row[4:]])
0480 |                     if sample_number == 1 or sample_number % 10 == 0:
0481 |                         compact = "; ".join(f"GPU{row[0]} VRAM={float(row[2]) / 1024:.2f}/{float(row[3]) / 1024:.2f} GiB "
0482 |                                              f"ocupada={100 * float(row[2]) / float(row[3]):.1f}% "
0483 |                                              f"atividade_gpu={row[4]}% atividade_leitura_escrita_memoria={row[5]}%" for row in gpu_rows)
0484 |                         elapsed = time.monotonic() - self.started_monotonic
0485 |                         utc_log = datetime.now(timezone.utc).isoformat()
0486 |                         self.log(f"[telemetria] utc={utc_log} decorrido={elapsed:.3f}s fase={phase} {compact or 'GPU sem amostra'}")
0487 |                 handle.flush()
0488 |                 if psutil:
0489 |                     vm = psutil.virtual_memory()
0490 |                     du = psutil.disk_usage(str(self.output.anchor or "/"))
0491 |                     io = psutil.disk_io_counters()
0492 |                     system_writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, psutil.cpu_percent(interval=None), vm.used / 1048576,
0493 |                                              vm.available / 1048576, vm.total / 1048576, os.getloadavg()[0],
0494 |                                              du.used / 1048576, du.free / 1048576,
0495 |                                              getattr(io, "read_bytes", None), getattr(io, "write_bytes", None)])
0496 |                     system_handle.flush()
0497 |                 self.stop_event.wait(1)
0498 |
0499 |     def stop(self):
0500 |         self.stop_event.set()
0501 |         if self.thread:
0502 |             self.thread.join(timeout=17)
0503 |         if self.kv_thread:
0504 |             self.kv_thread.join(timeout=3)
0505 |         self.set_phase("stopped", "monitor_stopped")
0506 |         if hasattr(self, "events_handle"):
0507 |             self.events_handle.close()
0508 |         if hasattr(self, "telemetry_handle"):
0509 |             self.telemetry_handle.close()
0510 |         self.write_telemetry_summary()
0511 |
0512 |     def write_telemetry_summary(self):
0513 |         """Agrega telemetria por fase para relacionar picos com eventos do benchmark."""
0514 |         def read_rows(name):
0515 |             path = locate(self.output, name)
0516 |             if not path.exists():
0517 |                 return []
0518 |             with path.open() as handle:
0519 |                 return list(csv.DictReader(handle))
0520 |         sources = {"gpu": read_rows("gpu.csv"), "system": read_rows("system.csv"), "kv": read_rows("kv-cache.csv")}
0521 |         # Arquivo de entrada simples para gráficos: uma observação por linha,
0522 |         # com UTC, tempo desde o início do monitor e fase experimental.
0523 |         if sources["gpu"]:
0524 |             import shutil
0525 |             shutil.copyfile(artifact(self.output, "gpu.csv"), artifact(self.output, "telemetry-timeseries.csv"))
0526 |         phases = sorted({r.get("phase") for rows in sources.values() for r in rows if r.get("phase")})
0527 |         output = {"definition": "Amostras observadas por fase; não são bytes nem tempos de transferência PCIe.", "phases": {}}
0528 |         for phase in phases:
0529 |             entry = {"gpu_samples": 0, "system_samples": 0, "kv_samples": 0}
0530 |             grows = [r for r in sources["gpu"] if r.get("phase") == phase]
0531 |             for key in ("used_mib", "total_mib", "used_gib", "total_gib", "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"):
0532 |                 vals = []
0533 |                 for r in grows:
0534 |                     try: vals.append(float(r[key]))
0535 |                     except (ValueError, TypeError, KeyError): pass
0536 |                 entry[f"gpu_{key}_mean"] = sum(vals) / len(vals) if vals else None
0537 |                 entry[f"gpu_{key}_max"] = max(vals) if vals else None
0538 |                 if key in {"used_mib", "used_gib", "vram_used_pct"}:
0539 |                     entry[f"gpu_{key}_min"] = min(vals) if vals else None
0540 |             entry["gpu_samples"] = len(grows)
0541 |             srows = [r for r in sources["system"] if r.get("phase") == phase]
0542 |             for key in ("cpu_util_pct", "ram_used_mib", "ram_available_mib", "root_disk_used_mib", "root_disk_free_mib"):
0543 |                 vals = []
0544 |                 for r in srows:
0545 |                     try: vals.append(float(r[key]))
0546 |                     except (ValueError, TypeError, KeyError): pass
0547 |                 entry[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
0548 |                 entry[f"{key}_max"] = max(vals) if vals else None
0549 |             entry["system_samples"] = len(srows)
0550 |             krows = [r for r in sources["kv"] if r.get("phase") == phase]
0551 |             vals = []
0552 |             for r in krows:
0553 |                 try: vals.append(float(r["fraction"]) * 100)
0554 |                 except (ValueError, TypeError, KeyError): pass
0555 |             entry["kv_occupancy_pct_mean"] = sum(vals) / len(vals) if vals else None
0556 |             entry["kv_occupancy_pct_max"] = max(vals) if vals else None
0557 |             entry["kv_samples"] = len(vals)
0558 |             output["phases"][phase] = entry
0559 |         write_json(artifact(self.output, "telemetry-summary.json"), output)
0560 |
0561 |
0562 | def percentile(values, q):
0563 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0564 |     if not values:
0565 |         return None
0566 |     pos = (len(values) - 1) * q
0567 |     lo, hi = math.floor(pos), math.ceil(pos)
0568 |     return values[lo] + (values[hi] - values[lo]) * (pos - lo)
0569 |
0570 |
0571 | def summarize(report):
0572 |     """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
0573 |     benchmarks = report["benchmarks"]
0574 |     if len(benchmarks) != 1:
0575 |         raise ValueError("Esperado exatamente um benchmark sequencial.")
0576 |     requests = benchmarks[0]["requests"]
0577 |     from reporting import derived
0578 |     good = [{**r, **derived(r)} for r in requests["successful"]]
0579 |     result = {
0580 |         "successful_request_count": len(good),
0581 |         "errored_request_count": len(requests["errored"]),
0582 |         "incomplete_request_count": len(requests["incomplete"]),
0583 |     }
0584 |     # Os nomes são deliberadamente longos: summary.json é um artefato de
0585 |     # análise, e não uma API em que economizar alguns bytes melhora algo.
0586 |     metrics = {
0587 |         "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0588 |         "request_latency_seconds": "request_latency",
0589 |         "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
0590 |         "output_completion_token_count": "output_tokens",
0591 |         "input_prompt_token_count": "prompt_tokens",
0592 |         "decode_generation_tokens_per_second": "decode_tokens_s",
0593 |         "effective_output_tokens_per_second": "effective_tokens_s",
0594 |     }
0595 |     for label, key in metrics.items():
0596 |         vals = [r.get(key) for r in good]
0597 |         result[label + "_sample_count"] = sum(v is not None for v in vals)
0598 |         result[label + "_p50"] = percentile(vals, .5)
0599 |         result[label + "_p95"] = percentile(vals, .95)
0600 |         result[label + "_p99"] = percentile(vals, .99)
0601 |     # Nomes canônicos para a comparação entre runtimes. Os campos históricos
0602 |     # acima permanecem para compatibilidade com relatórios já gerados.
0603 |     ttft = [r.get("time_to_first_token_ms") for r in good]
0604 |     tokens_s = [r.get("decode_tokens_s") for r in good]
0605 |     result["time_to_first_token_milliseconds_sample_count"] = sum(v is not None for v in ttft)
0606 |     result["tokens_per_second_sample_count"] = sum(v is not None for v in tokens_s)
0607 |     for suffix, q in (("p50", .50), ("p95", .95), ("p99", .99)):
0608 |         result[f"time_to_first_token_milliseconds_{suffix}"] = percentile(ttft, q)
0609 |         result[f"tokens_per_second_{suffix}"] = percentile(tokens_s, q)
0610 |     for label, key in {
0611 |         "time_to_first_token_seconds": "time_to_first_token_seconds",
0612 |         "generation_time_seconds": "generation_time_seconds",
0613 |         "end_to_end_latency_seconds": "end_to_end_latency_seconds",
0614 |         "decode_tokens_per_second": "decode_tokens_per_second",
0615 |         "end_to_end_tokens_per_second": "end_to_end_tokens_per_second",
0616 |     }.items():
0617 |         values = [r.get(key) for r in good]
0618 |         result[label + "_sample_count"] = sum(v is not None for v in values)
0619 |         result[label + "_p50"] = percentile(values, .50)
0620 |         result[label + "_p95"] = percentile(values, .95)
0621 |         result[label + "_p99"] = percentile(values, .99)
0622 |     result["metric_definitions"] = {
0623 |         "Time To First Token": "milissegundos entre o envio da requisição e o primeiro token/conteúdo observado; há um valor por requisição.",
0624 |         "Tokens/s": "tokens de saída por segundo durante o decode, calculado por requisição a partir do intervalo entre tokens; não inclui TTFT.",
0625 |     }
0626 |     result["requests_sha256"] = requests_digest(report)
0627 |     turns = sorted({r.get("turn_index") for r in good if r.get("turn_index") is not None})
0628 |     result["turn_indices"] = turns
0629 |     result["history_tokens_by_turn"] = [{"turn_index": turn,
0630 |                                          "history_tokens_p50": percentile([r.get("history_tokens") for r in good if r.get("turn_index") == turn], .50),
0631 |                                          "history_tokens_p95": percentile([r.get("history_tokens") for r in good if r.get("turn_index") == turn], .95)}
0632 |                                         for turn in turns]
0633 |     result["percentiles_are_exploratory"] = len(good) < 100
0634 |     result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
0635 |     return result
0636 |
0637 |
0638 | def requests_digest(report):
0639 |     # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
0640 |     bodies = []
0641 |     for row in report["benchmarks"][0]["requests"]["successful"]:
0642 |         args = json.loads(row["request_args"])
0643 |         body = args.get("body", {})
0644 |         bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
0645 |     return hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0646 |
0647 |
0648 | def turn_manifest(report):
0649 |     rows = report["benchmarks"][0]["requests"]["successful"]
0650 |     turns = []
0651 |     for row in rows:
0652 |         turns.append({"request_id": row.get("request_id"), "conversation_index": row.get("conversation_index"),
0653 |                       "turn_index": row.get("turn_index"), "mode": row.get("mode"),
0654 |                       "fixture_sha256": row.get("fixture_sha256"), "history_tokens": row.get("history_tokens"),
0655 |                       "history_tokens_estimate": row.get("history_tokens_estimate"),
0656 |                       "prompt_tokens": row.get("prompt_tokens"), "total_tokens": row.get("total_tokens"),
0657 |                       "request_sha256": row.get("request_sha256")})
0658 |     return turns
0659 |
0660 |
0661 | def conversation_artifact(report, fixture=None):
0662 |     requests = report["benchmarks"][0]["requests"]
0663 |     rows = []
0664 |     for status in ("successful", "errored", "incomplete"):
0665 |         for row in requests[status]:
0666 |             body = {}
0667 |             if row.get("request_args"):
0668 |                 body = json.loads(row["request_args"]).get("body", {})
0669 |             rows.append({"status": status, "request_id": row.get("request_id"),
0670 |                          "conversation_index": row.get("conversation_index"),
0671 |                          "turn_index": row.get("turn_index"), "mode": row.get("mode"),
0672 |                          "messages": body.get("messages"),
0673 |                          "assistant_output": row.get("output") if status == "successful" else None,
0674 |                          "error": row.get("error") if status != "successful" else None,
0675 |                          "request_sha256": row.get("request_sha256"),
0676 |                          "history_tokens": row.get("history_tokens"),
0677 |                          "history_tokens_estimate": row.get("history_tokens_estimate")})
0678 |     artifact = {"mode": rows[0].get("mode") if rows else None, "turns": rows}
0679 |     if fixture:
0680 |         artifact["fixture"] = {"path": fixture["path"], "sha256": fixture["sha256"],
0681 |                                "name": fixture["data"].get("name"),
0682 |                                "version": fixture["data"].get("version")}
0683 |     return artifact
0684 |
0685 |
0686 | def write_requests_csv(path, report):
0687 |     """Amostras individuais para análise no R/Python, incluindo status de erro."""
0688 |     from reporting import derived
0689 |     fields = ["status", "error", "mode", "request_id", "conversation_index", "turn_index",
0690 |               "fixture_name", "fixture_sha256", "workload_seed",
0691 |               "history_tokens", "history_tokens_estimate", "target_history_tokens", "messages_count",
0692 |               "request_sha256", "request_start_time", "first_token_time", "request_end_time",
0693 |               "time_to_first_token_seconds", "generation_time_seconds", "end_to_end_latency_seconds",
0694 |               "completion_tokens", "prompt_tokens", "total_tokens", "decode_tokens_per_second",
0695 |               "end_to_end_tokens_per_second", "inter_token_latency_seconds", "request_latency",
0696 |               "time_to_first_token_ms", "inter_token_latency_ms", "output_tokens", "decode_tokens_s", "effective_tokens_s",
0697 |               "context_start_tokens", "context_end_tokens", "context_band"]
0698 |     with Path(path).open("w", newline="", encoding="utf-8") as handle:
0699 |         writer = csv.DictWriter(handle, fieldnames=fields)
0700 |         writer.writeheader()
0701 |         for status in ("successful", "errored", "incomplete"):
0702 |             for row in report["benchmarks"][0]["requests"][status]:
0703 |                 if status == "successful":
0704 |                     row = {**row, **derived(row)}
0705 |                 writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})
0706 |
0707 |
0708 | def write_summary(output, rows):
0709 |     from reporting import render
0710 |     write_json(artifact(output, "summary.json"), rows)
0711 |     if not rows:
0712 |         render(output, rows)
0713 |         return
0714 |     with artifact(output, "summary.csv").open("w", newline="", encoding="utf-8") as handle:
0715 |         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
0716 |         writer.writeheader()
0717 |         writer.writerows(rows)
0718 |     render(output, rows)
0719 |
0720 |
0721 | def run(args):
0722 |     import fcntl
0723 |     from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
0724 |     if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
0725 |         raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
0726 |     cfg = load_config(args.config)
0727 |     validate_run(cfg, args.scenarios, args.smoke)
0728 |     availability = local_model_check(args.local_model_path)
0729 |     availability["kv_bytes_per_token"] = args.kv_bytes_per_token
0730 |     availability["launch_hf_offline"] = bool(args.launch)
0731 |     digest = tokenizer_digest(cfg["tokenizer"])
0732 |     from transformers import AutoTokenizer
0733 |     measurement_tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True, trust_remote_code=False)
0734 |     count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
0735 |     secret = os.environ.get("BENCH_API_KEY", "")
0736 |     prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
0737 |     if not prompt.strip():
0738 |         raise ValueError("O prompt inicial não pode estar vazio.")
0739 |     conversation_mode = args.mode in {"closed-loop", "replay"}
0740 |     if not conversation_mode and args.conversation_turns != 1:
0741 |         raise ValueError("--conversation-turns só pode ser maior que 1 com --mode closed-loop ou --mode replay.")
0742 |     conversation_fixture = load_conversation_fixture(args.conversation_fixture) if conversation_mode else None
0743 |     warmup_conversation_fixture = load_conversation_fixture(args.warmup_conversation_fixture) if conversation_mode else None
0744 |     if conversation_fixture:
0745 |         validate_conversation_fixture_for_mode(conversation_fixture, args.mode, args.conversation_turns)
0746 |     base = Path(args.results)
0747 |     base.mkdir(parents=True, exist_ok=True)
0748 |     with (base / ".benchmark.lock").open("a") as lock:
0749 |         try:
0750 |             fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
0751 |         except BlockingIOError:
0752 |             raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
0753 |         stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
0754 |         label = execution_label(smoke=args.smoke, scenarios=args.scenarios,
0755 |                                 input_tokens=args.input_tokens, result_name=args.result_name)
0756 |         output = runtime_root(base, cfg["runtime"], stamp, label)
0757 |         prepare(output)
0758 |         manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
0759 |                     "results_layout": "runtime/timestamp/human-name/{html,json,csv,logs,text}",
0760 |                     "result_name": label, "runtime_directory": slug(cfg["runtime"]),
0761 |                     "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
0762 |                     "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
0763 |                     "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
0764 |                     "mode": args.mode, "conversation_turns": args.conversation_turns,
0765 |                     "prompt_policy": ("deterministic unique prompt per request; warmup and measure use disjoint seeds"
0766 |                                       if not conversation_mode else "fixed conversation fixture"),
0767 |                     "conversation_fixture": {"path": conversation_fixture["path"], "sha256": conversation_fixture["sha256"],
0768 |                                              "name": conversation_fixture["data"].get("name"),
0769 |                                              "version": conversation_fixture["data"].get("version")} if conversation_fixture else None,
0770 |                     "warmup_conversation_fixture": {"path": warmup_conversation_fixture["path"],
0771 |                                                      "sha256": warmup_conversation_fixture["sha256"],
0772 |                                                      "name": warmup_conversation_fixture["data"].get("name"),
0773 |                                                      "version": warmup_conversation_fixture["data"].get("version")} if warmup_conversation_fixture else None,
0774 |                     "conversation_first_request": "first-request.json is a separate single-turn lifecycle probe; closed-loop/replay measured histories start empty per conversation and per block",
0775 |                     "blocks": [],
0776 |                     "model_availability": availability,
0777 |                     "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
0778 |                     "telemetry": {"gpu_source": "local nvidia-smi", "system_source": "psutil host CPU/RAM/disk counters",
0779 |                                   "event_source": "events.csv phase markers", "kv_metrics_requested": args.collect_kv_metrics,
0780 |                                   "sampling": "approximately 1 Hz; not per-token; PCIe copy time is not directly measured"},
0781 |                     "python": sys.version, "platform": platform.platform(),
0782 |                     "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
0783 |         write_json(artifact(output, "manifest.json"), redact(manifest, secret))
0784 |         write_json(artifact(output, "client-packages.json"), {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
0785 |         write_json(artifact(output, "gpu-before.json"), capture(["nvidia-smi"]))
0786 |         monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
0787 |         launch, origin, cleanup_error = None, None, None
0788 |         lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
0789 |                      "status": "running", "initial_state_note": args.initial_state,
0790 |                      "startup_timeout_s": args.startup_timeout}
0791 |         print(f"Resultados: {output.resolve()}", flush=True)
0792 |         try:
0793 |             monitor.start()
0794 |             if args.launch:
0795 |                 launch = Launch(cfg, args.launch, output, args.launch_extra_args, args.launch_executable)
0796 |                 lifecycle["argv"] = redact(launch.argv, secret)
0797 |                 monitor.set_phase("process_startup")
0798 |                 origin = launch.start()
0799 |                 lifecycle["pid"] = launch.process.pid
0800 |             lifecycle_report(output, redact(lifecycle, secret))
0801 |             monitor.set_phase("server_readiness")
0802 |             lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
0803 |             lifecycle_report(output, redact(lifecycle, secret))
0804 |             monitor.set_phase("first_request")
0805 |             print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
0806 |             lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
0807 |                                                        artifact(output, "first-request.json"), origin)
0808 |             lifecycle_report(output, redact(lifecycle, secret))
0809 |             for rep in range(repetitions):
0810 |                 # Rotação balanceia parcialmente a posição dos cenários entre repetições.
0811 |                 names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
0812 |                 for name in names:
0813 |                     for phase, n in (("warmup", args.warmup), ("measure", count)):
0814 |                         if not n:
0815 |                             continue
0816 |                         seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
0817 |                         if phase == "warmup":
0818 |                             seed += 1_000_000
0819 |                         prefix = f"r{rep+1}-{name}-{phase}"
0820 |                         monitor.set_phase(prefix)
0821 |                         config = scenario_config(cfg, name, n, seed, secret, args.timeout)
0822 |                         config["mode"] = args.mode
0823 |                         active_fixture = warmup_conversation_fixture if phase == "warmup" else conversation_fixture
0824 |                         config["prompt_policy"] = ("deterministic unique prompt per request; exact target content tokens; "
0825 |                                                    "block seed plus request index") if not conversation_mode else "fixed conversation fixture"
0826 |                         if conversation_mode:
0827 |                             config["conversation_turns"] = args.conversation_turns
0828 |                             config["conversation_fixture_sha256"] = active_fixture["sha256"]
0829 |                             config["spec_note"] = "Warmup usa fixture separada; medição usa fixture replay versionada com perguntas variadas e histórico determinístico."
0830 |                         write_json(artifact(output, f"{prefix}-config.json"), redact(config, secret))
0831 |                         expected = n * args.conversation_turns if conversation_mode else n
0832 |                         unit = "conversas" if conversation_mode else "requisições"
0833 |                         print(f"{prefix}: {n} {unit}, uma chamada por vez", flush=True)
0834 |                         if conversation_mode:
0835 |                             report = run_conversational_batch(cfg, measurement_tokenizer, name, n, args.conversation_turns,
0836 |                                                               args.timeout, secret, active_fixture, args.mode)
0837 |                         else:
0838 |                             report = run_stream_batch(cfg, measurement_tokenizer, name, n, args.timeout, secret, seed)
0839 |                         raw = redact(report, secret)
0840 |                         write_json(artifact(output, f"{prefix}.json"), raw)
0841 |                         turns = turn_manifest(raw) if conversation_mode else []
0842 |                         conversation_file = None
0843 |                         if conversation_mode:
0844 |                             write_json(artifact(output, f"{prefix}-turns.json"), turns)
0845 |                             conversation_file = f"{prefix}-conversation.json"
0846 |                             write_json(artifact(output, conversation_file), conversation_artifact(raw, conversation_fixture))
0847 |                         write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
0848 |                         summary = summarize(raw)
0849 |                         summary["expected"] = expected
0850 |                         summary["missing_request_count"] = max(0, expected - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
0851 |                         manifest["blocks"].append({"prefix": prefix, "mode": args.mode, "phase": phase,
0852 |                                                    "scenario": name, "repetition": rep+1, "seed": seed,
0853 |                                                    "conversations": n if conversation_mode else None,
0854 |                                                    "conversation_turns": args.conversation_turns if conversation_mode else None,
0855 |                                                    "expected_requests": expected,
0856 |                                                    "requests_sha256": summary["requests_sha256"],
0857 |                                                    "conversation_artifact": conversation_file,
0858 |                                                    "turns": turns if conversation_mode else None})
0859 |                         write_json(artifact(output, "manifest.json"), redact(manifest, secret))
0860 |                         rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
0861 |                                      "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
0862 |                                      "mode": args.mode, "phase": phase, "scenario": name, "repetition": rep+1, **summary})
0863 |                         write_summary(output, rows)
0864 |                         if summary["successful_request_count"] != expected or summary["errored_request_count"] or summary["incomplete_request_count"]:
0865 |                             raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
0866 |             monitor.set_phase("warm_reference")
0867 |             lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
0868 |                                                          artifact(output, "warm-reference.json"))
0869 |             manifest["status"] = lifecycle["status"] = "complete"
0870 |         except BaseException as exc:
0871 |             manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0872 |             manifest["error"] = redact(str(exc), secret)
0873 |             lifecycle["status"] = manifest["status"]
0874 |             lifecycle["error"] = manifest["error"]
0875 |             raise
0876 |         finally:
0877 |             for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
0878 |                 if artifact(output, filename).exists():
0879 |                     lifecycle[key] = json.loads(artifact(output, filename).read_text(encoding="utf-8"))
0880 |             if launch is not None:
0881 |                 monitor.set_phase("server_shutdown")
0882 |                 try:
0883 |                     launch.close()
0884 |                     lifecycle["server_cleanup"] = "stopped owned process group only"
0885 |                 except Exception as exc:
0886 |                     cleanup_error = redact(str(exc), secret)
0887 |                     lifecycle["server_cleanup_error"] = cleanup_error
0888 |                     lifecycle["status"] = manifest["status"] = "failed"
0889 |             lifecycle_report(output, redact(lifecycle, secret))
0890 |             monitor.stop()
0891 |             manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
0892 |             write_json(artifact(output, "manifest.json"), redact(manifest, secret))
0893 |             write_json(artifact(output, "gpu-after.json"), capture(["nvidia-smi"]))
0894 |             write_summary(output, rows)
0895 |         if cleanup_error:
0896 |             raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
0897 |         print(f"Concluído. Abra {artifact(output, 'lifecycle.html')} e {artifact(output, 'summary.html')}")
0898 |
0899 |
0900 | def positive(value):
0901 |     number = int(value)
0902 |     if number <= 0:
0903 |         raise argparse.ArgumentTypeError("Use um inteiro positivo.")
0904 |     return number
0905 |
0906 |
0907 | def nonnegative(value):
0908 |     number = int(value)
0909 |     if number < 0:
0910 |         raise argparse.ArgumentTypeError("Use zero ou um inteiro positivo.")
0911 |     return number
0912 |
0913 |
0914 | def rebuild_report(args):
0915 |     """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
0916 |     output = Path(args.output)
0917 |     rows = json.loads(locate(output, "summary.json").read_text())
0918 |     manifest = json.loads(locate(output, "manifest.json").read_text())
0919 |     for row in rows:
0920 |         prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
0921 |         raw = json.loads(locate(output, f"{prefix}.json").read_text())
0922 |         row.update(summarize(raw))
0923 |         expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
0924 |         row["expected"] = expected
0925 |         row["missing_request_count"] = max(0, expected - sum(row[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count"))) if expected is not None else None
0926 |         write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
0927 |     write_summary(output, rows)
0928 |     print(f"Relatório atualizado: {artifact(output, 'summary.html')}; dados brutos preservados.")
0929 |
0930 |
0931 | def main():
0932 |     parser = argparse.ArgumentParser(description=__doc__)
0933 |     commands = parser.add_subparsers(dest="command", required=True)
0934 |     report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
0935 |     report.add_argument("--output", required=True)
0936 |     report.set_defaults(func=rebuild_report)
0937 |     prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
0938 |     prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
0939 |     prep.add_argument("--revision", default="main")
0940 |     prep.add_argument("--output", default="tokenizer")
0941 |     prep.set_defaults(func=prepare_tokenizer)
0942 |     cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
0943 |     cmd.add_argument("--config", required=True)
0944 |     cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
0945 |     cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
0946 |     cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
0947 |     cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
0948 |     cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium", "long"])
0949 |     cmd.add_argument("--requests", type=positive, default=50, help="Requisições de medição por cenário e repetição; replay percorre perguntas distintas da fixture.")
0950 |     cmd.add_argument("--repetitions", type=positive, default=1)
0951 |     cmd.add_argument("--warmup", type=nonnegative, default=3)
0952 |     cmd.add_argument("--mode", choices=["independent", "closed-loop", "replay"], default="replay",
0953 |                      help="independent preserva uma requisição sem histórico; closed-loop acumula respostas reais; replay usa assistant fixo do fixture.")
0954 |     cmd.add_argument("--conversation-turns", type=positive, default=1,
0955 |                      help="Turnos por conversa em --mode closed-loop ou replay; cada turno envia o histórico completo.")
0956 |     cmd.add_argument("--conversation-fixture", default=str(DEFAULT_CONVERSATION_FIXTURE),
0957 |                      help="Fixture JSON versionado com system e lista fixa de user turns; replay tambem exige assistant nos turnos anteriores.")
0958 |     cmd.add_argument("--warmup-conversation-fixture", default=str(DEFAULT_WARMUP_CONVERSATION_FIXTURE),
0959 |                      help="Fixture separada para warmup; nunca é usada na medição formal.")
0960 |     cmd.add_argument("--seed", type=positive, default=42)
0961 |     cmd.add_argument("--timeout", type=positive, default=300)
0962 |     cmd.add_argument("--results", default="results")
0963 |     cmd.add_argument("--result-name", help="Nome humano da execução na pasta timestamp; sem isso é derivado do modo/cenário.")
0964 |     cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
0965 |     cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
0966 |     cmd.add_argument("--launch-extra-args", nargs="*", default=[],
0967 |                      help="Argumentos experimentais acrescentados ao argv do launch, sem editar o JSON; registrados no lifecycle.")
0968 |     cmd.add_argument("--launch-executable", help="Substitui argv[0] do launch pelo executável resolvido no ambiente do runtime.")
0969 |     cmd.add_argument("--launch-extra-args-json", default="[]", help="Array JSON de argumentos do runtime, preservando flags e espaços.")
0970 |     cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
0971 |     cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
0972 |     cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
0973 |     cmd.set_defaults(func=run)
0974 |     args = parser.parse_args()
0975 |     if hasattr(args, "launch_extra_args_json"):
0976 |         extra = json.loads(args.launch_extra_args_json)
0977 |         if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
0978 |             parser.error("--launch-extra-args-json deve ser array de strings")
0979 |         args.launch_extra_args.extend(extra)
0980 |     if getattr(args, "input_tokens", None):
0981 |         if len(set(args.input_tokens)) != len(args.input_tokens):
0982 |             parser.error("Não repita comprimentos em --input-tokens.")
0983 |         args.scenarios = []
0984 |         for size in args.input_tokens:
0985 |             name = f"ctx{size}"
0986 |             WORKLOADS[name] = size
0987 |             args.scenarios.append(name)
0988 |     if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
0989 |         parser.error("Não repita cenários na lista.")
0990 |     try:
0991 |         args.func(args)
0992 |     except KeyboardInterrupt:
0993 |         print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
0994 |         return 130
0995 |     except Exception as exc:
0996 |         print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
0997 |         return 1
0998 |     return 0
0999 |
1000 |
1001 | if __name__ == "__main__":
1002 |     raise SystemExit(main())
```

## lifecycle.py

SHA-256: `80ac8f54b43acc8e5192d8ee5a3b58f8cfd497e5371cd0f60cb42357461fbb06`.

| Função/classe | Linhas |
|---|---|
| `Launch` | 25–93 |
| `wait_models` | 96–137 |
| `timed_request` | 140–234 |
| `lifecycle_report` | 237–252 |
| `__init__` | 26–43 |
| `start` | 45–64 |
| `close` | 66–93 |

```text
0001 | """Cronometria de inicialização/primeiro stream, fora das fases GuideLLM.
0002 |
0003 | Nenhum POST de inferência é enviado antes da primeira requisição medida.
0004 | O lançamento é opt-in e usa argv sem shell. Só o processo criado é encerrado.
0005 | """
0006 | import json
0007 | import os
0008 | from pathlib import Path
0009 |
0010 | from results_layout import artifact, href, prepare
0011 | import signal
0012 | import socket
0013 | import subprocess
0014 | import time
0015 | from urllib.parse import urlsplit
0016 |
0017 | import httpx
0018 |
0019 | DEFAULT_PROMPT = (
0020 |     "Explique para um estudante de estatística a diferença entre memória RAM e VRAM. "
0021 |     "Inclua um exemplo de uso de um chatbot e conclua com um resumo em três itens."
0022 | )
0023 |
0024 |
0025 | class Launch:
0026 |     def __init__(self, cfg, command_file, output, extra_args=None, executable=None):
0027 |         url = urlsplit(cfg["base_url"])
0028 |         if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
0029 |             raise ValueError("--launch só aceita servidor local (localhost).")
0030 |         self.host, self.port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
0031 |         self.argv = json.loads(Path(command_file).read_text(encoding="utf-8"))
0032 |         if not isinstance(self.argv, list) or not self.argv or any(not isinstance(x, str) or not x for x in self.argv):
0033 |             raise ValueError("O arquivo --launch deve conter um array JSON não vazio de strings (argv).")
0034 |         if executable:
0035 |             self.argv[0] = executable
0036 |         if extra_args:
0037 |             if any(not isinstance(x, str) or not x for x in extra_args):
0038 |                 raise ValueError("--launch-extra-args aceita somente strings não vazias.")
0039 |             self.argv.extend(extra_args)
0040 |         self.output = Path(output)
0041 |         self.process = None
0042 |         self.log = None
0043 |         self.started = None
0044 |
0045 |     def start(self):
0046 |         # Não interrompe servidores existentes nem tenta tomar uma porta ocupada.
0047 |         try:
0048 |             connection = socket.create_connection((self.host, self.port), timeout=1)
0049 |         except OSError:
0050 |             pass
0051 |         else:
0052 |             connection.close()
0053 |             raise ValueError("A porta já está em uso. Pare o seu servidor manualmente antes de usar --launch.")
0054 |         prepare(self.output)
0055 |         self.log = artifact(self.output, "server.log").open("w", encoding="utf-8")
0056 |         self.started = time.perf_counter()
0057 |         try:
0058 |             self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
0059 |                                             start_new_session=True, shell=False,
0060 |                                             env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
0061 |         except BaseException:
0062 |             self.log.close()
0063 |             raise
0064 |         return self.started
0065 |
0066 |     def close(self):
0067 |         """Encerra exclusivamente o grupo criado por este objeto, inclusive filhos."""
0068 |         if self.process is not None and self.process.poll() is None:
0069 |             try:
0070 |                 os.killpg(self.process.pid, signal.SIGTERM)
0071 |             except ProcessLookupError:
0072 |                 pass
0073 |             try:
0074 |                 self.process.wait(timeout=20)
0075 |             except subprocess.TimeoutExpired:
0076 |                 os.killpg(self.process.pid, signal.SIGKILL)
0077 |                 self.process.wait(timeout=5)
0078 |             # Alguns workers podem sobreviver ao encerramento do processo líder.
0079 |             try:
0080 |                 os.killpg(self.process.pid, signal.SIGKILL)
0081 |             except ProcessLookupError:
0082 |                 pass
0083 |         elif self.process is not None:
0084 |             # O líder pode ter falhado, mas workers/servidor podem ainda
0085 |             # estar no mesmo grupo. Tenta limpar o grupo sem falhar se ele já
0086 |             # tiver desaparecido.
0087 |             try:
0088 |                 os.killpg(self.process.pid, signal.SIGKILL)
0089 |             except ProcessLookupError:
0090 |                 pass
0091 |         if self.log is not None:
0092 |             self.log.close()
0093 |             self.log = None
0094 |
0095 |
0096 | def wait_models(cfg, secret, timeout, launch=None):
0097 |     """Apenas GET. Modelo listado não comprova que seus pesos estão na GPU."""
0098 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0099 |     started = time.perf_counter()
0100 |     probes, last, next_update = 0, None, started + 15
0101 |     with httpx.Client(timeout=2, headers=headers, follow_redirects=False) as client:
0102 |         while True:
0103 |             probes += 1
0104 |             if launch is not None and launch.process.poll() is not None:
0105 |                 raise RuntimeError(f"Servidor encerrou antes de ficar disponível (exit={launch.process.returncode}). Veja server.log.")
0106 |             try:
0107 |                 response = client.get(cfg["base_url"] + "/v1/models")
0108 |                 if response.status_code in {401, 403}:
0109 |                     raise ValueError("API recusou autenticação; confira BENCH_API_KEY.")
0110 |                 response.raise_for_status()
0111 |                 models = [m["id"] for m in response.json().get("data", [])]
0112 |                 expected = cfg["model"]
0113 |                 accepted = {expected}
0114 |                 if cfg.get("runtime") == "ollama" and ":" not in expected:
0115 |                     accepted.add(expected + ":latest")
0116 |                 if accepted.intersection(models):
0117 |                     observed = time.perf_counter()
0118 |                     return {"models": models, "get_probes": probes,
0119 |                             "wait_wall_s": observed - started,
0120 |                             "process_to_api_observed_s": observed - launch.started if launch else None,
0121 |                             "criterion": "GET /v1/models retornou o ID; não é prova de pesos residentes"}
0122 |                 last = f"API respondeu, mas modelo esperado={expected!r}; disponíveis={models!r}"
0123 |                 if models:
0124 |                     raise RuntimeError(last + ". Confira --alias/--served-model-name e config.model.")
0125 |             except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
0126 |                 last = str(exc)
0127 |             elapsed = time.perf_counter() - started
0128 |             if not launch or elapsed >= timeout:
0129 |                 raise RuntimeError(f"API/modelo não disponível após {elapsed:.1f}s: {last}")
0130 |             if time.perf_counter() >= next_update:
0131 |                 print(f"[inicialização] runtime={cfg.get('runtime', 'não informado')} "
0132 |                       f"PID={launch.process.pid if launch else 'externo'} decorrido={elapsed:.1f}s "
0133 |                       f"limite={timeout}s; verificando GET {cfg['base_url']}/v1/models "
0134 |                       f"para modelo={cfg['model']!r}; última observação: {last}. "
0135 |                       f"Log do servidor: {artifact(launch.output, 'server.log') if launch else 'externo'}", flush=True)
0136 |                 next_update = time.perf_counter() + 15
0137 |             time.sleep(min(.5, max(0, timeout - elapsed)))
0138 |
0139 |
0140 | def timed_request(cfg, secret, timeout, prompt, output, process_origin=None):
0141 |     """Primeiro POST é simultaneamente medição e validação, sem pré-aquecimento oculto.
0142 |
0143 |     Grava resultado parcial inclusive em timeout, stream inválido ou usage ausente.
0144 |     TTFT aqui é primeiro conteúdo não vazio recebido (não mero cabeçalho/role).
0145 |     """
0146 |     path = Path(output)
0147 |     body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
0148 |             "temperature": 0, "top_p": 1, "max_tokens": 128, "stream": True,
0149 |             "stream_options": {"include_usage": True}}
0150 |     result = {"status": "running", "body": body, "stream_usage": None,
0151 |               "content_event_offsets_s": [], "output": "", "done": False,
0152 |               "ttft_ms": None, "e2e_s": None, "mean_itl_ms": None,
0153 |               "usage_observed": False, "request_start_time": None,
0154 |               "first_token_time": None, "request_end_time": None,
0155 |               "time_to_first_token_seconds": None, "generation_time_seconds": None,
0156 |               "end_to_end_latency_seconds": None, "completion_tokens": None,
0157 |               "prompt_tokens": None, "total_tokens": None,
0158 |               "decode_tokens_per_second": None, "end_to_end_tokens_per_second": None,
0159 |               "process_to_first_content_s": None, "process_to_response_end_s": None}
0160 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0161 |     started = None
0162 |     try:
0163 |         with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0164 |             started = time.perf_counter()
0165 |             result["request_start_time"] = started
0166 |             with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as stream:
0167 |                 result["headers_ms"] = (time.perf_counter() - started) * 1000
0168 |                 stream.raise_for_status()
0169 |                 if "text/event-stream" not in stream.headers.get("content-type", ""):
0170 |                     raise ValueError("A API não respondeu com SSE.")
0171 |                 for line in stream.iter_lines():
0172 |                     if not line.startswith("data:"):
0173 |                         continue
0174 |                     value = line[5:].strip()
0175 |                     if value == "[DONE]":
0176 |                         result["done"] = True
0177 |                         break
0178 |                     event = json.loads(value)
0179 |                     if "error" in event:
0180 |                         raise ValueError(f"Erro no stream: {event['error']}")
0181 |                     result["stream_usage"] = event.get("usage") or result["stream_usage"]
0182 |                     text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0183 |                     if text:
0184 |                         now = time.perf_counter()
0185 |                         result["content_event_offsets_s"].append(now - started)
0186 |                         result["output"] += text
0187 |                         result["first_token_time"] = result["first_token_time"] or now
0188 |                         result["last_token_time"] = now
0189 |                         if result["ttft_ms"] is None:
0190 |                             result["ttft_ms"] = (now - started) * 1000
0191 |                             if process_origin is not None:
0192 |                                 result["process_to_first_content_s"] = now - process_origin
0193 |             ended = time.perf_counter()
0194 |             result["request_end_time"] = ended
0195 |             result["e2e_s"] = ended - started
0196 |             result["end_to_end_latency_seconds"] = ended - started
0197 |             if process_origin is not None:
0198 |                 result["process_to_response_end_s"] = ended - process_origin
0199 |             if not result["done"] or not result["output"]:
0200 |                 raise ValueError("Stream incompleto ou sem conteúdo.")
0201 |             usage = result["stream_usage"]
0202 |             valid_usage = isinstance(usage, dict) and all(
0203 |                 type(usage.get(k)) is int and usage[k] >= 0
0204 |                 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
0205 |             result["usage_observed"] = valid_usage
0206 |             if valid_usage:
0207 |                 result["completion_tokens"] = usage["completion_tokens"]
0208 |                 result["prompt_tokens"] = usage["prompt_tokens"]
0209 |                 result["total_tokens"] = usage["total_tokens"]
0210 |                 result["time_to_first_token_seconds"] = ((result["first_token_time"] - started)
0211 |                                                            if result["first_token_time"] is not None else None)
0212 |                 if result["completion_tokens"] > 1 and result.get("last_token_time") is not None:
0213 |                     result["generation_time_seconds"] = result["last_token_time"] - result["first_token_time"]
0214 |                     result["mean_itl_ms"] = 1000 * result["generation_time_seconds"] / (result["completion_tokens"] - 1)
0215 |                 if result["generation_time_seconds"] and result["generation_time_seconds"] > 0:
0216 |                     result["decode_tokens_per_second"] = result["completion_tokens"] / result["generation_time_seconds"]
0217 |                 if result["end_to_end_latency_seconds"] > 0:
0218 |                     result["end_to_end_tokens_per_second"] = result["completion_tokens"] / result["end_to_end_latency_seconds"]
0219 |                 from reporting import derived
0220 |                 result.update(derived({"output_tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"],
0221 |                                        "inter_token_latency_ms": result["mean_itl_ms"], "request_latency": result["e2e_s"]}))
0222 |             result["status"] = "complete"
0223 |     except BaseException as exc:
0224 |         result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0225 |         result["error"] = str(exc)
0226 |         if started is not None:
0227 |             result["elapsed_until_exit_s"] = time.perf_counter() - started
0228 |         raise
0229 |     finally:
0230 |         serialized = json.dumps(result, ensure_ascii=False, indent=2)
0231 |         if secret:
0232 |             serialized = serialized.replace(secret, "[REDACTED]")
0233 |         path.write_text(serialized + "\n", encoding="utf-8")
0234 |     return result
0235 |
0236 |
0237 | def lifecycle_report(output, lifecycle):
0238 |     import html
0239 |     output = Path(output)
0240 |     artifact(output, "lifecycle.json").write_text(json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0241 |     rows = []
0242 |     readiness = lifecycle.get("readiness", {})
0243 |     rows.append(("Processo → API observada (s)", readiness.get("process_to_api_observed_s")))
0244 |     for phase in ("first_request", "warm_reference"):
0245 |         req = lifecycle.get(phase, {})
0246 |         for metric in ("ttft_ms", "decode_tokens_s", "effective_tokens_s", "e2e_s", "mean_itl_ms", "process_to_first_content_s", "process_to_response_end_s"):
0247 |             if phase == "warm_reference" and metric.startswith("process_"):
0248 |                 continue
0249 |             rows.append((phase + " · " + metric, req.get(metric)))
0250 |     table = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value)) if value is not None else 'Não medido'}</td></tr>" for label, value in rows)
0251 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciclo de vida</title><style>body{{font:17px/1.7 system-ui;background:#f6f3ec;color:#193835;margin:25px}}td,th{{padding:12px;border-bottom:1px solid #ccd6cc;text-align:left}}table{{width:100%;overflow-wrap:anywhere}}a{{color:#136d58}}</style><h1>Inicialização e primeira resposta</h1><p>Modo: {html.escape(lifecycle['mode'])}. Status: {html.escape(lifecycle['status'])}.</p><p>API disponível não implica modelo na GPU. O primeiro POST é cronometrado, sem teste de geração anterior. Processo novo não implica caches de disco/CUDA frios. A referência final repete o prompt e pode aproveitar prefix caching.</p><table>{table}</table><p><a href="{href('summary.html')}">Aquecimento e blocos GuideLLM</a> · <a href="{href('lifecycle.json')}">Dados do ciclo de vida</a></p></html>'''
0252 |     artifact(output, "lifecycle.html").write_text(page, encoding="utf-8")
```

## reporting.py

SHA-256: `42f9c070134b30a7cbb25e758cc25a6e11f675904096848c34b9457b611de02d`.

| Função/classe | Linhas |
|---|---|
| `ratio` | 12–15 |
| `percentile` | 18–25 |
| `derived` | 28–38 |
| `context_band` | 41–48 |
| `table` | 51–58 |
| `context_summary` | 61–100 |
| `gpu_summary` | 103–125 |
| `_timeseries` | 128–143 |
| `_chart` | 146–168 |
| `render` | 171–242 |
| `fmt` | 52–55 |

```text
0001 | """Métricas derivadas e relatório offline; não confunde contexto com VRAM."""
0002 | import csv
0003 | import html
0004 | import json
0005 | import math
0006 | from collections import defaultdict
0007 | from pathlib import Path
0008 |
0009 | from results_layout import artifact, href, locate
0010 |
0011 |
0012 | def ratio(a, b):
0013 |     if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
0014 |         return None
0015 |     return a / b if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 else None
0016 |
0017 |
0018 | def percentile(values, q):
0019 |     """Percentil empírico com interpolação linear; ausências não viram zero."""
0020 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0021 |     if not values:
0022 |         return None
0023 |     position = (len(values) - 1) * q
0024 |     lower, upper = math.floor(position), math.ceil(position)
0025 |     return values[lower] + (values[upper] - values[lower]) * (position - lower)
0026 |
0027 |
0028 | def derived(row):
0029 |     n, p = row.get("output_tokens"), row.get("prompt_tokens")
0030 |     itl = row.get("inter_token_latency_ms")
0031 |     return {
0032 |         "decode_tokens_s": ratio(1000, itl) if n is not None and n > 1 else None,
0033 |         "effective_tokens_s": ratio(n, row.get("request_latency")),
0034 |         "context_start_tokens": p,
0035 |         # Comprimento lógico final; não é o número exato de posições materializadas.
0036 |         "context_end_tokens": p + n if p is not None and n is not None else None,
0037 |         "context_band": context_band(p),
0038 |     }
0039 |
0040 |
0041 | def context_band(p):
0042 |     if p is None:
0043 |         return "não informado"
0044 |     for limit in (512, 1024, 2048, 4096, 8192, 16384):
0045 |         if p < limit:
0046 |             lower = 0 if limit == 512 else limit // 2
0047 |             return f"[{lower}, {limit})"
0048 |     return "[16384, +∞)"
0049 |
0050 |
0051 | def table(rows, columns):
0052 |     def fmt(v):
0053 |         if v is None:
0054 |             return "Não disponível"
0055 |         return f"{v:.3f}" if isinstance(v, float) else str(v)
0056 |     head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
0057 |     body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k, _ in columns) + "</tr>" for r in rows)
0058 |     return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
0059 |
0060 |
0061 | def context_summary(output, kv_bytes_per_token=None):
0062 |     groups = defaultdict(list)
0063 |     csv_dir = Path(output) / "csv"
0064 |     files = csv_dir.glob("r*-*-requests.csv") if csv_dir.exists() else Path(output).glob("r*-*-requests.csv")
0065 |     for file in sorted(files):
0066 |         rep, scenario, phase, _ = file.stem.split("-", 3)
0067 |         with file.open() as handle:
0068 |             for row in csv.DictReader(handle):
0069 |                 if row["status"] != "successful":
0070 |                     continue
0071 |                 for key in ("prompt_tokens", "output_tokens", "inter_token_latency_ms", "request_latency", "time_to_first_token_ms"):
0072 |                     row[key] = float(row[key]) if row.get(key) else None
0073 |                 row.update(derived(row))
0074 |                 # Preservamos o cenário mesmo quando dois cenários caem na
0075 |                 # mesma faixa de contexto. Isso evita que short/medium/long
0076 |                 # apareçam como uma única população no JSON derivado.
0077 |                 groups[(phase, rep, scenario, row["context_band"])].append(row)
0078 |     result = []
0079 |     for (phase, rep, scenario, band), rows in groups.items():
0080 |         entry = {"phase": phase, "repetition": rep, "scenario": scenario,
0081 |                  "context_band": band, "successful_request_count": len(rows)}
0082 |         context_metrics = {
0083 |             "decode_generation_tokens_per_second": "decode_tokens_s",
0084 |             "effective_output_tokens_per_second": "effective_tokens_s",
0085 |             "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0086 |             "initial_context_input_token_count": "context_start_tokens",
0087 |             "final_logical_context_token_count": "context_end_tokens",
0088 |         }
0089 |         for label, key in context_metrics.items():
0090 |             values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
0091 |             entry[label + "_sample_count"] = len(values)
0092 |             entry[label + "_p50"] = percentile(values, .50)
0093 |             entry[label + "_p95"] = percentile(values, .95)
0094 |             entry[label + "_p99"] = percentile(values, .99)
0095 |         result.append(entry)
0096 |         for edge in ("start", "end"):
0097 |             tokens_key = "initial_context_input_token_count_p50" if edge == "start" else "final_logical_context_token_count_p50"
0098 |             tokens = entry[tokens_key]
0099 |             entry[f"estimated_{edge}_logical_kv_cache_mebibytes"] = tokens * kv_bytes_per_token / 1048576 if tokens is not None and kv_bytes_per_token else None
0100 |     return result
0101 |
0102 |
0103 | def gpu_summary(output):
0104 |     groups = defaultdict(list)
0105 |     path = locate(Path(output), "gpu.csv")
0106 |     if path.exists():
0107 |         with path.open() as handle:
0108 |             for row in csv.DictReader(handle):
0109 |                 groups[(row["phase"], row["index"], row["name"])].append(row)
0110 |     result = []
0111 |     for (phase, index, name), rows in groups.items():
0112 |         entry = {"phase": phase, "gpu": index, "name": name, "samples": len(rows)}
0113 |         for key in ("used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"):
0114 |             values = []
0115 |             for row in rows:
0116 |                 try:
0117 |                     value = float(row[key])
0118 |                     if math.isfinite(value):
0119 |                         values.append(value)
0120 |                 except (ValueError, TypeError, KeyError):
0121 |                     pass
0122 |             entry[key + "_mean"] = sum(values) / len(values) if values else None
0123 |             entry[key + "_max"] = max(values) if values else None
0124 |         result.append(entry)
0125 |     return result
0126 |
0127 |
0128 | def _timeseries(output):
0129 |     path = locate(Path(output), "gpu.csv")
0130 |     if not path.exists():
0131 |         return []
0132 |     with path.open() as handle:
0133 |         rows = []
0134 |         for row in csv.DictReader(handle):
0135 |             try:
0136 |                 elapsed = float(row["elapsed_s"])
0137 |                 used = float(row["used_mib"])
0138 |                 util = float(row["gpu_util_pct"])
0139 |                 if all(math.isfinite(value) for value in (elapsed, used, util)):
0140 |                     rows.append({"elapsed_s": elapsed, "used_mib": used, "gpu_util_pct": util})
0141 |             except (KeyError, TypeError, ValueError):
0142 |                 continue
0143 |         return rows
0144 |
0145 |
0146 | def _chart(rows, key, title, ylabel, color, maximum=None):
0147 |     """Gera SVG autônomo para abrir no navegador sem dependências externas."""
0148 |     if not rows:
0149 |         return f'<p class="chart-empty">Sem amostras válidas para {html.escape(title.lower())}.</p>'
0150 |     width, height, pad = 920, 260, 42
0151 |     xmax = max(row["elapsed_s"] for row in rows) or 1.0
0152 |     ymax = maximum or max(row[key] for row in rows) or 1.0
0153 |     points = []
0154 |     for row in rows:
0155 |         x = pad + (width - 2 * pad) * row["elapsed_s"] / xmax
0156 |         y = height - pad - (height - 2 * pad) * row[key] / ymax
0157 |         points.append(f"{x:.1f},{y:.1f}")
0158 |     polyline = " ".join(points)
0159 |     svg = (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
0160 |            f'<rect width="100%" height="100%" fill="#fbfaf5"/><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#789"/>'
0161 |            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#789"/>'
0162 |            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
0163 |            f'<text x="{pad}" y="20" fill="#193835">{html.escape(title)}</text>'
0164 |            f'<text x="{pad}" y="{height-8}" fill="#526">0 s</text>'
0165 |            f'<text x="{width-pad-55}" y="{height-8}" fill="#526">{xmax:.1f} s</text>'
0166 |            f'<text x="6" y="{pad+5}" fill="#526">{ymax:.0f}</text>'
0167 |            f'<text x="6" y="{height-pad+5}" fill="#526">0</text></svg>')
0168 |     return svg
0169 |
0170 |
0171 | def render(output, rows):
0172 |     output = Path(output)
0173 |     lifecycle_path, manifest_path = locate(output, "lifecycle.json"), locate(output, "manifest.json")
0174 |     lifecycle = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else {}
0175 |     manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
0176 |     kv_bytes = manifest.get("model_availability", {}).get("kv_bytes_per_token")
0177 |     context, gpu = context_summary(output, kv_bytes), gpu_summary(output)
0178 |     artifact(output, "context-summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
0179 |     artifact(output, "gpu-summary.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2) + "\n")
0180 |     startup = lifecycle.get("readiness", {}).get("process_to_api_observed_s")
0181 |     initial = []
0182 |     for key, label in (("first_request", "Primeira resposta"), ("warm_reference", "Referência final")):
0183 |         req = lifecycle.get(key, {})
0184 |         usage = req.get("stream_usage") or {}
0185 |         d = derived({"output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
0186 |                      "inter_token_latency_ms": req.get("mean_itl_ms"), "request_latency": req.get("e2e_s")})
0187 |         initial.append({"phase": label, "ttft": req.get("ttft_ms"), "e2e": req.get("e2e_s"), **d})
0188 |     metrics = table(rows, [("phase", "Fase"), ("scenario", "Cenário"), ("repetition", "Repetição"),
0189 |         ("expected", "Requisições previstas"), ("successful_request_count", "Requisições bem-sucedidas"),
0190 |         ("errored_request_count", "Requisições com erro"), ("incomplete_request_count", "Requisições incompletas"),
0191 |         ("missing_request_count", "Requisições ausentes do relatório bruto"),
0192 |         ("time_to_first_token_milliseconds_p50", "Time To First Token p50 (ms)"),
0193 |         ("time_to_first_token_milliseconds_p95", "Time To First Token p95 (ms)"),
0194 |         ("time_to_first_token_milliseconds_p99", "Time To First Token p99 (ms)"),
0195 |         ("generation_time_seconds_p50", "Tempo de geração p50 (s)"),
0196 |         ("decode_tokens_per_second_p50", "Tokens/s de decodificação p50"),
0197 |         ("end_to_end_tokens_per_second_p50", "Tokens/s ponta a ponta p50"),
0198 |         ("end_to_end_latency_seconds_p50", "Latência ponta a ponta p50 (s)"),
0199 |         ("input_prompt_token_count_p50", "Tokens de entrada p50"),
0200 |         ("output_completion_token_count_p50", "Tokens de saída p50")])
0201 |     by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("scenario", "Cenário"), ("context_band", "Faixa de entrada (tokens)"),
0202 |         ("successful_request_count", "Requisições bem-sucedidas"),
0203 |         ("initial_context_input_token_count_p50", "Tokens de entrada inicial p50"),
0204 |         ("final_logical_context_token_count_p50", "Tokens de contexto lógico final p50"),
0205 |         ("estimated_start_logical_kv_cache_mebibytes", "KV lógico inicial estimado (MiB)"),
0206 |         ("estimated_end_logical_kv_cache_mebibytes", "KV lógico final estimado (MiB)"),
0207 |         ("request_first_token_latency_milliseconds_p50", "Latência da requisição até primeiro token p50 (ms)"),
0208 |         ("decode_generation_tokens_per_second_p50", "Velocidade de geração p50 (tokens/s)"),
0209 |         ("effective_output_tokens_per_second_p50", "Velocidade efetiva p50 (tokens/s)")])
0210 |     hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
0211 |         ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
0212 |         ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
0213 |         ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
0214 |     kvfile = locate(output, "kv-cache.csv")
0215 |     kvrows = []
0216 |     if kvfile.exists():
0217 |         with kvfile.open() as handle:
0218 |             groups = defaultdict(list)
0219 |             for r in csv.DictReader(handle):
0220 |                 groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
0221 |             kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
0222 |     kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
0223 |     first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
0224 |     policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
0225 |     failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
0226 |     series = _timeseries(output)
0227 |     memory_chart = _chart(series, "used_mib", "Memória da GPU ocupada (MiB)", "MiB", "#b54e27")
0228 |     util_chart = _chart(series, "gpu_util_pct", "Utilização computacional da GPU (%)", "%", "#136d58", 100)
0229 |     artifact(output, "telemetry-memory.svg").write_text(memory_chart, encoding="utf-8") if series else None
0230 |     artifact(output, "telemetry-gpu-util.svg").write_text(util_chart, encoding="utf-8") if series else None
0231 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
0232 | <style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}.chart{{width:100%;max-height:280px;border:1px solid #ccd6cc;margin:10px 0 22px}}.chart-empty{{padding:16px;background:#fff0de}}</style><main>
0233 | <h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
0234 | <p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
0235 | <h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
0236 | <h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
0237 | <h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
0238 | <p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
0239 | <h2>4. GPU, CPU, RAM e SSD por fase</h2><p>O monitor amostra aproximadamente 1 vez/s e relaciona cada amostra a eventos/fases. GPU vem do nvidia-smi; CPU, RAM e I/O de disco vêm do host. Máximos amostrados podem perder picos e não há atribuição por processo. N/A é ausência de dado, não zero.</p><h3>Memória e utilização ao longo da execução</h3>{memory_chart}{util_chart}<p><a href="{href('telemetry-memory.svg')}">SVG da memória</a> · <a href="{href('telemetry-gpu-util.svg')}">SVG da GPU-util</a> · <a href="{href('telemetry-summary.json')}">Resumo de telemetria</a> · <a href="{href('events.csv')}">Eventos</a> · <a href="{href('system.csv')}">CPU/RAM/SSD</a></p>{hardware}
0240 | <h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
0241 | <p><a href="{href('summary.json')}">Resumo JSON</a> · <a href="{href('context-summary.json')}">Faixas JSON</a> · <a href="{href('gpu-summary.json')}">GPU JSON</a> · <a href="{href('manifest.json')}">Manifesto</a></p></main></html>'''
0242 |     artifact(output, "summary.html").write_text(page, encoding="utf-8")
```

## guidellm_compat.py

SHA-256: `44ca9b6f443b04d7021323eb8584a6d29721d50c32b5f39bc549f371be70a5a7`.

| Função/classe | Linhas |
|---|---|
| `_is_incomplete` | 16–22 |
| `completion_drain` | 26–87 |

```text
0001 | """Narrow compatibility workarounds for the pinned GuideLLM release."""
0002 |
0003 | from __future__ import annotations
0004 |
0005 | import asyncio
0006 | from collections.abc import Iterator
0007 | from contextlib import contextmanager
0008 | from importlib.metadata import version
0009 | import time
0010 | from typing import Any
0011 |
0012 |
0013 | GUIDELLM_VERSION = "0.7.4"
0014 |
0015 |
0016 | def _is_incomplete(state: Any) -> bool:
0017 |     """Return whether a scheduler snapshot still has non-terminal requests."""
0018 |     return (
0019 |         state is not None
0020 |         and state.created_requests > state.processed_requests
0021 |         and state.processing_requests > 0
0022 |     )
0023 |
0024 |
0025 | @contextmanager
0026 | def completion_drain(timeout: float = 5.0) -> Iterator[None]:
0027 |     """Drain a late final scheduler update affected by GuideLLM 0.7.4's race.
0028 |
0029 |     GuideLLM 0.7.4 may set ``shutdown_event`` in its receive callback immediately
0030 |     before that callback's final update reaches the local receive buffer.  The
0031 |     stock iterator can observe the event during its timeout and return first.
0032 |
0033 |     This context manager temporarily wraps ``WorkerProcessGroup.request_updates``.
0034 |     After the stock iterator returns, the wrapper reads only genuine messages from
0035 |     GuideLLM's receive queue for at most ``timeout`` seconds, and only if the last
0036 |     yielded snapshot is demonstrably incomplete and shutdown has been signalled.
0037 |     If no update arrives, it returns the incomplete result unchanged so the
0038 |     caller's existing completeness guard remains authoritative.
0039 |
0040 |     This is a process-global monkey patch; do not overlap benchmark runs or use the
0041 |     context manager concurrently from multiple threads.
0042 |     """
0043 |     if timeout <= 0:
0044 |         raise ValueError("timeout must be greater than zero")
0045 |     installed = version("guidellm")
0046 |     if installed != GUIDELLM_VERSION:
0047 |         raise RuntimeError(
0048 |             f"completion_drain supports guidellm=={GUIDELLM_VERSION}, found {installed}"
0049 |         )
0050 |
0051 |     from guidellm.scheduler.worker_group import WorkerProcessGroup
0052 |
0053 |     original = WorkerProcessGroup.request_updates
0054 |
0055 |     async def request_updates_with_completion_drain(self):
0056 |         last_state = None
0057 |         async for update in original(self):
0058 |             last_state = update[3]
0059 |             yield update
0060 |
0061 |         shutdown_event = self.shutdown_event
0062 |         messaging = self.messaging
0063 |         if (
0064 |             not _is_incomplete(last_state)
0065 |             or shutdown_event is None
0066 |             or not shutdown_event.is_set()
0067 |             or messaging is None
0068 |         ):
0069 |             return
0070 |
0071 |         deadline = time.monotonic() + timeout
0072 |         while _is_incomplete(last_state):
0073 |             remaining = deadline - time.monotonic()
0074 |             if remaining <= 0:
0075 |                 return
0076 |             try:
0077 |                 update = await messaging.get(timeout=remaining)
0078 |             except asyncio.TimeoutError:
0079 |                 return
0080 |             last_state = update[3]
0081 |             yield update
0082 |
0083 |     WorkerProcessGroup.request_updates = request_updates_with_completion_drain
0084 |     try:
0085 |         yield
0086 |     finally:
0087 |         WorkerProcessGroup.request_updates = original
```
