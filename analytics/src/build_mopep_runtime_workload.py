"""Gera workloads privados e reproduzíveis para o benchmark MOPEP de runtimes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analytics.src.mopep_prompt import build_prompt, build_review_prompt, prompt_contract


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rendered_tokens(tokenizer: Any, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    try:
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(result, "input_ids"):
            result = result.input_ids
        if isinstance(result, dict):
            result = result["input_ids"]
        if result and isinstance(result[0], (list, tuple)):
            result = result[0]
        return len(result)
    except Exception:
        return len(tokenizer.encode(prompt, add_special_tokens=False))


def tercile_thresholds(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("Não há prompts válidos para calcular os tercis.")

    def nearest_rank(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {"short_max_tokens": nearest_rank(1 / 3), "medium_max_tokens": nearest_rank(2 / 3)}


def bucket_for(tokens: int, thresholds: dict[str, int]) -> str:
    if tokens <= thresholds["short_max_tokens"]:
        return "short"
    if tokens <= thresholds["medium_max_tokens"]:
        return "medium"
    return "heavy"


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "business_model" not in frame:
        raise ValueError(f"Coluna business_model ausente em {path}.")
    id_column = "example_id" if "example_id" in frame else "id" if "id" in frame else None
    if id_column is None:
        raise ValueError(f"Coluna example_id ou id ausente em {path}.")
    selected = frame[[id_column, "business_model"]].dropna().copy()
    selected[id_column] = selected[id_column].astype(str)
    selected["business_model"] = selected["business_model"].astype(str)
    if selected[id_column].duplicated().any():
        raise ValueError(f"IDs duplicados em {path}.")
    selected = selected.rename(columns={id_column: "request_id"}).sort_values("request_id")
    if selected.empty:
        raise ValueError(f"Nenhum BMC válido encontrado em {path}.")
    return selected.reset_index(drop=True)


def deterministic_bucket_sample(
    source: pd.DataFrame,
    *,
    tokenizer: Any,
    thresholds: dict[str, int],
    per_bucket: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if per_bucket <= 0:
        raise ValueError("--sample-per-bucket deve ser maior que zero.")
    ranked = source.copy()
    ranked["_rendered_prompt_tokens"] = [
        rendered_tokens(tokenizer, build_prompt(value))
        for value in ranked["business_model"]
    ]
    ranked["_bucket"] = ranked["_rendered_prompt_tokens"].map(
        lambda value: bucket_for(value, thresholds)
    )
    ranked["_sample_rank"] = ranked["request_id"].map(
        lambda request_id: hashlib.sha256(
            f"{seed}:{request_id}".encode("utf-8")
        ).hexdigest()
    )
    selected = []
    available_counts = {}
    bucket_order = {"short": 0, "medium": 1, "heavy": 2}
    for name in bucket_order:
        candidates = ranked.loc[ranked["_bucket"] == name].sort_values(
            ["_sample_rank", "request_id"]
        )
        available_counts[name] = len(candidates)
        if len(candidates) < per_bucket:
            raise ValueError(
                f"Bucket {name} possui {len(candidates)} exemplos; "
                f"a amostra exige {per_bucket}."
            )
        chosen = candidates.head(per_bucket).copy()
        chosen["_sample_position"] = range(len(chosen))
        chosen["_bucket_order"] = bucket_order[name]
        selected.append(chosen)
    sampled = pd.concat(selected, ignore_index=True).sort_values(
        ["_sample_position", "_bucket_order", "_sample_rank", "request_id"]
    )
    selected_ids = sampled["request_id"].tolist()
    metadata = {
        "strategy": "sha256-ranked-within-frozen-token-bucket",
        "seed": seed,
        "per_bucket": per_bucket,
        "source_request_count": len(source),
        "selected_request_count": len(sampled),
        "available_bucket_counts": available_counts,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
    }
    sampled = sampled.rename(columns={
        "_bucket": "bucket",
        "_rendered_prompt_tokens": "rendered_prompt_tokens",
    })
    return sampled[[
        "request_id", "business_model", "bucket", "rendered_prompt_tokens"
    ]].reset_index(drop=True), metadata


def write_sample_dataset(
    source_path: Path, sampled_source: pd.DataFrame, output_path: Path
) -> None:
    frame = pd.read_csv(source_path)
    id_column = "example_id" if "example_id" in frame else "id" if "id" in frame else None
    if id_column is None:
        raise ValueError(f"Coluna example_id ou id ausente em {source_path}.")
    frame[id_column] = frame[id_column].astype(str)
    request_ids = sampled_source["request_id"].tolist()
    order = {request_id: index for index, request_id in enumerate(request_ids)}
    sampled = frame.loc[frame[id_column].isin(order)].copy()
    sampled["_sample_order"] = sampled[id_column].map(order)
    sampled = sampled.sort_values("_sample_order").drop(columns="_sample_order")
    if len(sampled) != len(request_ids):
        raise ValueError("A amostra local não preservou todos os IDs selecionados.")
    details = sampled_source.set_index("request_id")
    sampled["bucket"] = sampled[id_column].map(details["bucket"])
    sampled["rendered_prompt_tokens"] = sampled[id_column].map(
        details["rendered_prompt_tokens"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)


def build_records(frame: pd.DataFrame, tokenizer: Any, thresholds: dict[str, int]) -> list[dict[str, Any]]:
    review = build_review_prompt()
    records = []
    for row in frame.itertuples(index=False):
        prompt = build_prompt(row.business_model)
        token_count = rendered_tokens(tokenizer, prompt)
        records.append({
            "schema_version": SCHEMA_VERSION,
            "request_id": row.request_id,
            "bucket": bucket_for(token_count, thresholds),
            "rendered_prompt_tokens": token_count,
            "initial_messages": [{"role": "user", "content": prompt}],
            "review_instruction": review,
        })
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )


def build_workload(
    *, dataset: Path, warmup_dataset: Path, output_dir: Path, tokenizer: Any,
    tokenizer_name: str, tokenizer_revision: str, split: str,
    threshold_manifest: Path | None = None, warmup_count: int = 3,
    limit: int | None = None,
    sample_per_bucket: int | None = None, sample_seed: int = 42,
    sample_dataset_out: Path | None = None,
) -> dict[str, Any]:
    source = load_source(dataset)
    if limit is not None and sample_per_bucket is not None:
        raise ValueError("Use --limit ou --sample-per-bucket, não ambos.")
    if sample_dataset_out is not None and sample_per_bucket is None:
        raise ValueError("--sample-dataset-out exige --sample-per-bucket.")
    threshold_source = source.head(limit) if limit is not None else source
    prompts = [build_prompt(value) for value in threshold_source["business_model"]]
    lengths = [rendered_tokens(tokenizer, prompt) for prompt in prompts]
    if threshold_manifest:
        reference = json.loads(threshold_manifest.read_text(encoding="utf-8"))
        thresholds = reference["bucket_thresholds"]
        threshold_source_sha256 = sha256_file(threshold_manifest)
    else:
        if split != "calibration":
            raise ValueError("Splits diferentes de calibration exigem --threshold-manifest.")
        thresholds = tercile_thresholds(lengths)
        threshold_source_sha256 = None

    sampling = None
    sampled_dataset_path = None
    if sample_per_bucket is not None:
        source, sampling = deterministic_bucket_sample(
            source,
            tokenizer=tokenizer,
            thresholds=thresholds,
            per_bucket=sample_per_bucket,
            seed=sample_seed,
        )
        sampled_dataset_path = sample_dataset_out or (
            output_dir / f"{split}-{sample_per_bucket * 3}.csv"
        )
        write_sample_dataset(
            dataset,
            source,
            sampled_dataset_path,
        )
        sampling["selected_dataset_file"] = sampled_dataset_path.name
        sampling["selected_dataset_sha256"] = sha256_file(sampled_dataset_path)
        sampling["selected_dataset_local_only"] = True
        sampling["selected_dataset_in_upload"] = False
    elif limit is not None:
        source = source.head(limit)

    warmup_source = load_source(warmup_dataset)
    selected_ids = set(source["request_id"])
    warmup_source = warmup_source.loc[~warmup_source["request_id"].isin(selected_ids)].head(warmup_count)
    if len(warmup_source) < warmup_count:
        raise ValueError("O dataset de warm-up não possui exemplos distintos suficientes.")

    records = build_records(source, tokenizer, thresholds)
    warmup_records = build_records(warmup_source, tokenizer, thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = output_dir / "workload.jsonl"
    warmup_path = output_dir / "warmup.jsonl"
    write_jsonl(workload_path, records)
    write_jsonl(warmup_path, warmup_records)
    order_sha256 = hashlib.sha256(
        "\n".join(row["request_id"] for row in records).encode("utf-8")
    ).hexdigest()
    tokenizer_contract = {
        "name": tokenizer_name,
        "revision": tokenizer_revision,
        "class": tokenizer.__class__.__name__,
    }
    tokenizer_contract["identity_sha256"] = hashlib.sha256(
        json.dumps(tokenizer_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "dataset_sha256": sha256_file(dataset),
        "warmup_dataset_sha256": sha256_file(warmup_dataset),
        "workload_sha256": sha256_file(workload_path),
        "warmup_sha256": sha256_file(warmup_path),
        "request_order_sha256": order_sha256,
        "request_count": len(records),
        "warmup_count": len(warmup_records),
        "bucket_thresholds": thresholds,
        "bucket_threshold_source_manifest_sha256": threshold_source_sha256,
        "bucket_counts": {
            name: sum(row["bucket"] == name for row in records)
            for name in ("short", "medium", "heavy")
        },
        "sampling": sampling,
        "tokenizer": tokenizer_contract,
        **prompt_contract(),
        "privacy": {
            "contains_expected_tags": False,
            "contains_business_models": True,
            "version_control_allowed": False,
        },
    }
    manifest_path = output_dir / "workload-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksums = [
        (workload_path, "workload.jsonl"),
        (warmup_path, "warmup.jsonl"),
        (manifest_path, "workload-manifest.json"),
    ]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {name}\n" for path, name in checksums),
        encoding="utf-8",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--warmup-dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--split", choices=("calibration", "test"), required=True)
    result.add_argument("--tokenizer", required=True)
    result.add_argument("--tokenizer-revision", required=True)
    result.add_argument("--threshold-manifest", type=Path)
    result.add_argument("--warmup-count", type=int, default=3)
    result.add_argument("--limit", type=int)
    result.add_argument("--sample-per-bucket", type=int)
    result.add_argument("--sample-seed", type=int, default=42)
    result.add_argument("--sample-dataset-out", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.tokenizer_revision == "main":
        raise ValueError("Fixe uma revisão imutável do tokenizer; main não é aceita.")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision, trust_remote_code=False
    )
    manifest = build_workload(
        dataset=args.dataset, warmup_dataset=args.warmup_dataset,
        output_dir=args.output_dir, tokenizer=tokenizer,
        tokenizer_name=args.tokenizer, tokenizer_revision=args.tokenizer_revision,
        split=args.split, threshold_manifest=args.threshold_manifest,
        warmup_count=args.warmup_count, limit=args.limit,
        sample_per_bucket=args.sample_per_bucket, sample_seed=args.sample_seed,
        sample_dataset_out=args.sample_dataset_out,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
