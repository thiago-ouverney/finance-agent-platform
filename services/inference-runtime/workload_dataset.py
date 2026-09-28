"""Contrato e execução sequencial de workloads MOPEP privados."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable


PROFILES = ("mopep-single", "mopep-review-replay", "mopep-review-closed-loop")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL inválido em {path}:{line_number}: {exc.msg}") from exc
        request_id = row.get("request_id")
        messages = row.get("initial_messages")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(f"request_id inválido em {path}:{line_number}.")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"initial_messages ausente em {path}:{line_number}.")
        if row.get("bucket") not in {"short", "medium", "heavy"}:
            raise ValueError(f"bucket inválido em {path}:{line_number}.")
        if not isinstance(row.get("review_instruction"), str) or not row["review_instruction"].strip():
            raise ValueError(f"review_instruction ausente em {path}:{line_number}.")
        rows.append(row)
    ids = [row["request_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"request_id duplicado em {path}.")
    if not rows:
        raise ValueError(f"Workload vazio: {path}.")
    return rows


def load_workload(path: str | Path, manifest_path: str | Path) -> tuple[list[dict], dict]:
    path, manifest_path = Path(path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Versão do manifesto de workload não suportada.")
    actual = sha256_file(path)
    expected = manifest.get("workload_sha256")
    if actual != expected:
        raise ValueError(f"SHA-256 do workload diverge: atual={actual}, esperado={expected}.")
    return load_jsonl(path), manifest


def replay_manifest_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".manifest.json")


def load_replay(path: str | Path) -> tuple[dict[str, str], str, dict]:
    path = Path(path)
    manifest_path = replay_manifest_path(path)
    if not manifest_path.is_file():
        raise ValueError(f"Manifesto do replay ausente: {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    response_sha256 = sha256_file(path)
    if manifest.get("responses_sha256") != response_sha256:
        raise ValueError("SHA-256 das respostas canônicas diverge do manifesto.")
    if manifest.get("source_runtime") != "llama":
        raise ValueError("O replay canônico deve ter sido gerado pelo runtime llama.")
    rows = load_response_rows(path)
    mapping = {}
    for row in rows:
        request_id = row.get("workload_request_id") or row.get("request_id")
        output = row.get("assistant_output")
        if row.get("turn_index") != 1 or not isinstance(output, str) or not output.strip():
            continue
        if request_id in mapping:
            raise ValueError(f"Resposta canônica duplicada para {request_id}.")
        mapping[request_id] = output
    if not mapping:
        raise ValueError("O replay não contém respostas single-pass válidas.")
    return mapping, response_sha256, manifest


def load_response_rows(path: str | Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Resposta JSONL inválida na linha {line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Resposta inválida na linha {line_number}.")
        rows.append(row)
    return rows


def _metadata(item: dict, profile: str, turn: int) -> dict:
    return {
        "request_id": f"{item['request_id']}-t{turn}",
        "workload_request_id": item["request_id"],
        "workload_profile": profile,
        "bucket": item["bucket"],
        "turn_index": turn,
        "mode": profile,
        "rendered_prompt_tokens": item.get("rendered_prompt_tokens"),
    }


def run_batch(
    *, cfg: dict, records: list[dict], profile: str, timeout: int, secret: str,
    request: Callable, replay: dict[str, str] | None = None,
) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Perfil MOPEP inválido: {profile}.")
    successful, errored = [], []
    for item in records:
        try:
            if profile == "mopep-single":
                successful.append(request(
                    cfg, item["initial_messages"], timeout, secret,
                    metadata=_metadata(item, profile, 1),
                ))
                successful[-1]["status"] = "successful"
                continue

            if profile == "mopep-review-replay":
                if replay is None or item["request_id"] not in replay:
                    raise ValueError(f"Resposta canônica ausente para {item['request_id']}.")
                messages = [*item["initial_messages"],
                            {"role": "assistant", "content": replay[item["request_id"]]},
                            {"role": "user", "content": item["review_instruction"]}]
                successful.append(request(
                    cfg, messages, timeout, secret,
                    metadata=_metadata(item, profile, 2),
                ))
                successful[-1]["status"] = "successful"
                continue

            first = request(
                cfg, item["initial_messages"], timeout, secret,
                metadata=_metadata(item, profile, 1),
            )
            first["status"] = "successful"
            successful.append(first)
            review_messages = [*item["initial_messages"],
                               {"role": "assistant", "content": first.get("output") or "[]"},
                               {"role": "user", "content": item["review_instruction"]}]
            second = request(
                cfg, review_messages, timeout, secret,
                metadata=_metadata(item, profile, 2),
            )
            second["status"] = "successful"
            successful.append(second)
        except Exception as exc:
            errored.append({
                "status": "errored", "error": str(exc),
                **_metadata(item, profile, 1 if profile == "mopep-single" else 2),
            })
    return {"benchmarks": [{"requests": {
        "successful": successful, "errored": errored, "incomplete": [],
    }}]}


def response_rows(report: dict) -> list[dict]:
    result = []
    requests = report["benchmarks"][0]["requests"]
    for status in ("successful", "errored", "incomplete"):
        for row in requests[status]:
            result.append({
                "status": status,
                "request_id": row.get("request_id"),
                "workload_request_id": row.get("workload_request_id"),
                "workload_profile": row.get("workload_profile"),
                "bucket": row.get("bucket"),
                "turn_index": row.get("turn_index"),
                "request_sha256": row.get("request_sha256"),
                "assistant_output": row.get("output") if status == "successful" else None,
                "error": row.get("error") if status != "successful" else None,
            })
    return result


def write_responses(path: str | Path, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
