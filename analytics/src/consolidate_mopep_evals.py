"""Lista e consolida as avaliações de tags MOPEP."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("analytics/results/mopep-tags")
DEFAULT_OUTPUT_DIR = Path("analytics/data")
RUNS_FILENAME = "mopep_eval_runs.csv"
PREDICTIONS_FILENAME = "mopep_eval_predictions.csv"


def sample_sha256(predictions: pd.DataFrame) -> str:
    columns = ["example_id", "expected_tags"]
    if "input_sha256" in predictions:
        columns.append("input_sha256")
    sample = predictions[columns].fillna("").astype(str).sort_values("example_id")
    payload = "\n".join("|".join(row) for row in sample.itertuples(index=False, name=None))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parameter_size_b(parameter_size: object, model_name: str) -> float | None:
    text = str(parameter_size or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", text)
    if not match:
        match = re.search(r"(?:^|[^a-zA-Z0-9])(\d+(?:\.\d+)?)[bB](?:$|[^a-zA-Z])", model_name)
    return float(match.group(1)) if match else None


def quantization_level(value: object, model_name: str) -> str | None:
    if value:
        return str(value)
    match = re.search(r"q\d+(?:_[a-zA-Z0-9]+)*", model_name, flags=re.IGNORECASE)
    return match.group(0) if match else None


def comparison_id(metadata: dict[str, object], sample_hash: str) -> str:
    comparable = {
        "dataset": Path(str(metadata.get("dataset", ""))).name,
        "prompt_sha256": metadata.get("prompt_sha256"),
        "sample_sha256": sample_hash,
        "ollama_options": metadata.get("ollama_options", {}),
    }
    payload = json.dumps(comparable, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def collect_runs(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for run_json in sorted(results_root.glob("*/run.json")):
        run_dir = run_json.parent
        performance_path = run_dir / "performance.csv"
        predictions_path = run_dir / "predictions.csv"
        if not performance_path.is_file() or not predictions_path.is_file():
            continue

        metadata = json.loads(run_json.read_text(encoding="utf-8"))
        performance = pd.read_csv(performance_path)
        predictions = pd.read_csv(predictions_path)
        if performance.empty or predictions.empty:
            continue

        run_id = str(metadata.get("run_id") or run_dir.name)
        model_name = str(metadata.get("model") or performance.iloc[0].get("model") or "")
        dataset_path = str(metadata.get("dataset") or "")
        model_info = metadata.get("model_info") or {}
        options = metadata.get("ollama_options") or {}
        sample_hash = sample_sha256(predictions)
        group_id = comparison_id(metadata, sample_hash)
        size_bytes = model_info.get("size_bytes")
        size_gb = float(size_bytes) / 1_000_000_000 if size_bytes else None
        parameters_b = parameter_size_b(model_info.get("parameter_size"), model_name)
        quantization = quantization_level(model_info.get("quantization_level"), model_name)
        model_digest = model_info.get("digest")
        model_label = (
            f"{model_name} [{str(model_digest)[:8]}]"
            if model_digest else model_name
        )

        row = performance.iloc[0].to_dict()
        row.update({
            "run_id": run_id,
            "created_at": metadata.get("created_at"),
            "dataset": dataset_path,
            "dataset_name": Path(dataset_path).stem,
            "dataset_rows": metadata.get("dataset_rows"),
            "seed": metadata.get("seed"),
            "limit": metadata.get("limit"),
            "prompt_sha256": metadata.get("prompt_sha256"),
            "sample_sha256": sample_hash,
            "comparison_id": group_id,
            "temperature": options.get("temperature"),
            "num_predict": options.get("num_predict"),
            "ollama_seed": options.get("seed"),
            "model_digest": model_digest,
            "model_label": model_label,
            "model_format": model_info.get("format"),
            "model_family": model_info.get("family"),
            "parameter_size": model_info.get("parameter_size"),
            "parameter_size_b": parameters_b,
            "quantization_level": quantization,
            "model_size_bytes": size_bytes,
            "model_size_gb": size_gb,
            "run_dir": str(run_dir),
        })
        run_rows.append(row)

        predictions = predictions.copy()
        predictions["run_id"] = run_id
        predictions["created_at"] = metadata.get("created_at")
        predictions["dataset_name"] = Path(dataset_path).stem
        predictions["prompt_sha256"] = metadata.get("prompt_sha256")
        predictions["sample_sha256"] = sample_hash
        predictions["comparison_id"] = group_id
        predictions["model_label"] = model_label
        predictions["parameter_size_b"] = parameters_b
        predictions["quantization_level"] = quantization
        if size_gb is not None:
            predictions["model_size_gb"] = size_gb
        prediction_frames.append(predictions)

    runs = pd.DataFrame(run_rows)
    if not runs.empty:
        runs = runs.sort_values("created_at").reset_index(drop=True)
    all_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame()
    )
    return runs, all_predictions


def write_consolidated(runs: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    if runs.empty:
        raise ValueError("Nenhuma avaliação completa encontrada.")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / RUNS_FILENAME
    predictions_path = output_dir / PREDICTIONS_FILENAME
    runs.to_csv(runs_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    return runs_path, predictions_path


def show_results(runs: pd.DataFrame) -> None:
    if runs.empty:
        print("Nenhuma avaliação completa encontrada.")
        return
    columns = [
        "created_at", "model", "dataset_name", "requests",
        "macro_f1", "mean_individual_f1", "mean_latency_ms",
    ]
    print(runs[columns].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("results", help="Lista as avaliações completas.")
    commands.add_parser("consolidate", help="Grava os CSVs consolidados.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs, predictions = collect_runs(args.results_root)
    if args.command == "results":
        show_results(runs)
        return
    try:
        runs_path, predictions_path = write_consolidated(runs, predictions, args.output_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Runs: {runs_path} ({len(runs)} linhas)")
    print(f"Predições: {predictions_path} ({len(predictions)} linhas)")


if __name__ == "__main__":
    main()
