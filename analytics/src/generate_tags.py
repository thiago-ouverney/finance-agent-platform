"""Gera tags BMC transparentes a partir dos metadados e métricas do benchmark."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def quantization_tag(model: str) -> str:
    value = model.lower()
    for pattern, label in (
        (r"q4(?:_|\b)|4bit", "q4"),
        (r"q8(?:_|\b)|8bit", "q8"),
        (r"awq", "awq"),
        (r"gptq", "gptq"),
        (r"exl3", "exl3"),
        (r"fp16|f16", "fp16"),
        (r"bf16", "bf16"),
    ):
        if re.search(pattern, value):
            return label
    return "unknown"


def model_family_tag(model: str) -> str:
    value = model.lower()
    for family in ("qwen", "llama", "mistral", "gemma", "phi"):
        if family in value:
            return family
    return "other"


def context_tag(row: dict[str, str]) -> str:
    tokens = number(row, "input_prompt_token_count_p50")
    if tokens is None:
        return row.get("scenario", "unknown") or "unknown"
    if tokens < 512:
        return "short"
    if tokens < 4096:
        return "medium"
    return "long"


def performance_tag(row: dict[str, str]) -> str:
    ttft = number(row, "time_to_first_token_milliseconds_p50")
    if ttft is None:
        ttft = number(row, "request_first_token_latency_milliseconds_p50")
    if ttft is None:
        return "ttft-unavailable"
    if ttft < 500:
        return "ttft-fast"
    if ttft < 2000:
        return "ttft-medium"
    return "ttft-slow"


def tags_for(row: dict[str, str]) -> list[str]:
    model = row.get("model", "")
    runtime = (row.get("runtime", "unknown") or "unknown").lower()
    phase = (row.get("phase", "unknown") or "unknown").lower()
    errors = number(row, "errored_request_count") or 0
    incomplete = number(row, "incomplete_request_count") or 0
    status = "complete" if errors == 0 and incomplete == 0 else "degraded"
    return [
        f"runtime:{runtime}",
        f"model-family:{model_family_tag(model)}",
        f"quantization:{quantization_tag(model)}",
        f"context:{context_tag(row)}",
        f"phase:{phase}",
        f"status:{status}",
        f"performance:{performance_tag(row)}",
    ]


def tag_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**row, "bmc_tags": "|".join(tags_for(row))} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    tagged = tag_rows(rows)
    if not tagged:
        raise ValueError("Dataset vazio.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(tagged[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tagged)
    print(f"dataset={args.output} rows={len(tagged)}")


if __name__ == "__main__":
    main()
