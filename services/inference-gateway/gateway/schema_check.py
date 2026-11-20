from __future__ import annotations

import json

import jsonschema


def request_schema(payload: dict) -> dict | None:
    fmt = payload.get("response_format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return None
    nested = fmt.get("json_schema")
    if not isinstance(nested, dict):
        return None
    schema = nested.get("schema")
    return schema if isinstance(schema, dict) else None


def assistant_text(body: bytes, content_type: str) -> str | None:
    if "text/event-stream" in content_type:
        return _stream_text(body)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    message = payload.get("choices", [{}])[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else None


def failure_reason(schema: dict, text: str | None) -> str | None:
    if text is None:
        return "conteúdo ausente"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"JSON inválido: {exc.msg}"
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def _stream_text(body: bytes) -> str | None:
    parts: list[str] = []
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return None
        delta = event.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)
