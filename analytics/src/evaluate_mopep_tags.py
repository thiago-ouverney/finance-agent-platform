"""Avalia um modelo Ollama no dataset de tags MOPEP."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
from ollama import Client
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

from analytics.src.mopep_prompt import ALLOWED_TAGS, TAG_DEFINITIONS, build_prompt


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:18000")
DEFAULT_DATASETS_DIR = Path(os.getenv("MOPEP_BMC_DATA_DIR", "/tmp/mopep-bmc-tags"))
DEFAULT_RESULTS_DIR = Path("analytics/results/mopep-tags")
OLLAMA_OPTIONS = {"temperature": 0.0, "num_predict": 512, "seed": 42}

def normalize_tags(tags: list[object]) -> list[str]:
    return sorted({
        " ".join(str(tag).strip().casefold().split())
        for tag in tags
        if str(tag).strip()
    })


def parse_tags(value: object) -> list[str]:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return []
    return normalize_tags(parsed) if isinstance(parsed, list) else []


def parse_model_output(raw_response: str) -> tuple[list[str], bool]:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [], False
    if not isinstance(parsed, list):
        return [], False
    return normalize_tags(parsed), True


def compare_tags(expected_tags: list[str], generated_tags: list[str]) -> dict[str, object]:
    expected = set(expected_tags)
    generated = set(generated_tags)
    correct = expected & generated
    precision = len(correct) / len(generated) if generated else 0.0
    recall = len(correct) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct_tags": sorted(correct),
        "missing_tags": sorted(expected - generated),
        "extra_tags": sorted(generated - expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": expected == generated,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"business_model", "tags"} - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["expected_tags"] = frame["tags"].apply(parse_tags)
    frame = frame.loc[
        frame["business_model"].notna()
        & frame["expected_tags"].str.len().gt(0)
    ].copy()

    if "example_id" not in frame:
        frame["example_id"] = frame["business_model"].map(
            lambda text: hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        )
    else:
        frame = frame.loc[frame["example_id"].notna()].copy()
        frame["example_id"] = frame["example_id"].astype(str)

    frame = frame.drop_duplicates("example_id").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Nenhuma linha válida encontrada em {path}")
    return frame


def resolve_dataset(value: str, datasets_dir: Path) -> Path:
    direct = Path(value).expanduser()
    candidates = [direct, datasets_dir / value, datasets_dir / f"{value}.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset não encontrado: {value}")


def model_names(client: Client) -> list[str]:
    return sorted(
        model.model
        for model in client.list().models
        if model.model
    )


def model_metadata(client: Client, model_name: str) -> dict[str, object]:
    for model in client.list().models:
        if model.model != model_name:
            continue
        details = model.details
        return {
            "digest": model.digest,
            "size_bytes": int(model.size) if isinstance(model.size, int) else None,
            "format": details.format if details else None,
            "family": details.family if details else None,
            "parameter_size": details.parameter_size if details else None,
            "quantization_level": details.quantization_level if details else None,
        }
    return {}


def response_content(response: object) -> str:
    message = getattr(response, "message", None)
    return getattr(message, "content", "") or ""


def predict_one(client: Client, model: str, example: pd.Series) -> dict[str, object]:
    started_at = perf_counter()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": build_prompt(str(example["business_model"]))}],
            think=False,
            options=OLLAMA_OPTIONS,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        raw_response = response_content(response)
        generated_tags, parse_ok = parse_model_output(raw_response)
        request_ok = True
        error = ""
    except Exception as exc:
        latency_ms = None
        raw_response = ""
        generated_tags = []
        parse_ok = False
        request_ok = False
        error = str(exc)

    prediction = {
        "example_id": example["example_id"],
        "input_sha256": hashlib.sha256(
            str(example["business_model"]).encode("utf-8")
        ).hexdigest(),
        "model": model,
        "expected_tags": example["expected_tags"],
        "generated_tags": generated_tags,
        "raw_response": raw_response,
        "request_ok": request_ok,
        "parse_ok": parse_ok,
        "latency_ms": latency_ms,
        "error": error,
    }
    prediction.update(compare_tags(prediction["expected_tags"], generated_tags))
    for column in ("domain", "split", "label_source", "taxonomy_version", "business_model_chars"):
        if column in example.index:
            prediction[column] = example[column]
    return prediction


def macro_f1(predictions: pd.DataFrame) -> float:
    label_space = sorted(
        {tag for tags in predictions["expected_tags"] for tag in tags}
        | {tag for tags in predictions["generated_tags"] for tag in tags}
    )
    binarizer = MultiLabelBinarizer(classes=label_space).fit([[]])
    y_true = binarizer.transform(predictions["expected_tags"])
    y_pred = binarizer.transform(predictions["generated_tags"])
    return float(f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=float("nan"),
    ))


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": predictions["model"].iloc[0],
        "requests": len(predictions),
        "request_success_rate": predictions["request_ok"].mean(),
        "parse_rate": predictions["parse_ok"].mean(),
        "mean_individual_f1": predictions["f1"].mean(),
        "exact_match_rate": predictions["exact_match"].mean(),
        "mean_latency_ms": predictions["latency_ms"].mean(),
        "macro_f1": macro_f1(predictions),
    }])


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def make_run_id(model: str, dataset: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}__{slug(model)}__{slug(dataset.stem)}__{uuid.uuid4().hex[:8]}"


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    list_columns = [
        "expected_tags", "generated_tags", "correct_tags",
        "missing_tags", "extra_tags",
    ]
    for column in list_columns:
        if column in result:
            result[column] = result[column].apply(
                lambda values: json.dumps(values, ensure_ascii=False)
            )
    return result


def run_evaluation(
    client: Client,
    model: str,
    dataset_path: Path,
    output_root: Path,
    limit: int | None,
    seed: int,
    ollama_host: str,
    model_info: dict[str, object] | None = None,
) -> Path:
    dataset = load_dataset(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit deve ser maior que zero")
        dataset = dataset.sample(n=min(limit, len(dataset)), random_state=seed).reset_index(drop=True)

    run_id = make_run_id(model, dataset_path)
    run_dir = output_root / run_id

    print(f"Warmup: {model}")
    try:
        client.chat(
            model=model,
            messages=[{"role": "user", "content": build_prompt(str(dataset.iloc[0]["business_model"]))}],
            think=False,
            options=OLLAMA_OPTIONS,
        )
    except Exception as exc:
        raise ValueError(f"Falha no warmup do modelo: {exc}") from exc

    run_dir.mkdir(parents=True, exist_ok=False)

    rows = [
        predict_one(client, model, example)
        for _, example in tqdm(dataset.iterrows(), total=len(dataset), desc=model)
    ]
    predictions = pd.DataFrame(rows)
    performance = summarize(predictions)

    csv_ready(predictions).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    performance.to_csv(run_dir / "performance.csv", index=False, encoding="utf-8")

    run_info = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_info": model_info or {},
        "dataset": str(dataset_path.resolve()),
        "dataset_rows": len(dataset),
        "ollama_host": ollama_host,
        "seed": seed,
        "limit": limit,
        "prompt_sha256": hashlib.sha256(build_prompt("").encode("utf-8")).hexdigest(),
        "ollama_options": OLLAMA_OPTIONS,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(performance.to_string(index=False))
    print(f"Resultados: {run_dir}")
    return run_dir


def show_datasets(datasets_dir: Path) -> None:
    paths = sorted(datasets_dir.glob("*.csv")) if datasets_dir.is_dir() else []
    if not paths:
        print(f"Nenhum dataset CSV encontrado em {datasets_dir}")
        return
    print(f"{'ID':<24} {'LINHAS':>8}  STATUS")
    for path in paths:
        try:
            rows = len(load_dataset(path))
            print(f"{path.stem:<24} {rows:>8}  OK")
        except ValueError as exc:
            print(f"{path.stem:<24} {'-':>8}  {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verifica a conexão com o Ollama.")
    commands.add_parser("models", help="Lista os modelos disponíveis.")
    commands.add_parser("datasets", help="Lista os datasets CSV disponíveis.")
    run = commands.add_parser("run", help="Executa a avaliação.")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "datasets":
        show_datasets(args.datasets_dir)
        return

    client = Client(host=args.ollama_host)
    try:
        available = model_names(client)
    except Exception as exc:
        raise SystemExit(
            f"Não foi possível acessar o Ollama em {args.ollama_host}: {exc}"
        ) from exc

    if args.command == "check":
        print(f"Conexão OK: {args.ollama_host} ({len(available)} modelos)")
    elif args.command == "models":
        print("Modelos disponíveis:")
        for model in available:
            print(f"- {model}")
    elif args.command == "run":
        try:
            if args.model not in available:
                raise ValueError(
                    f"Modelo não encontrado: {args.model}. "
                    "Use 'make eval-models' para listar os disponíveis."
                )
            dataset_path = resolve_dataset(args.dataset, args.datasets_dir)
            run_evaluation(
                client=client,
                model=args.model,
                dataset_path=dataset_path,
                output_root=args.output_root,
                limit=args.limit,
                seed=args.seed,
                ollama_host=args.ollama_host,
                model_info=model_metadata(client, args.model),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
