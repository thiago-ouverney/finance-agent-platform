"""Compara modelos de regressão para prever uma métrica do benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_TARGET = "time_to_first_token_milliseconds_p50"
CATEGORICAL_FEATURES = [
    "runtime", "model", "scenario", "phase", "mode", "bmc_tags"
]
NUMERIC_FEATURES = [
    "input_prompt_token_count_p50",
    "output_completion_token_count_p50",
    "initial_context_input_token_count_p50",
]


def train(dataset: Path, output: Path, target: str = DEFAULT_TARGET) -> dict[str, object]:
    frame = pd.read_csv(dataset)
    if target not in frame:
        raise ValueError(f"Métrica alvo ausente: {target}")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
    numeric = [column for column in NUMERIC_FEATURES if column in frame]
    features = [*categorical, *numeric]
    if not features:
        raise ValueError("Nenhuma feature de benchmark reconhecida no dataset.")
    usable = frame.dropna(subset=[target]).copy()
    for column in numeric:
        usable[column] = pd.to_numeric(usable[column], errors="coerce")
    if len(usable) < 8:
        raise ValueError("São necessárias ao menos 8 linhas completas para treinar e avaliar.")

    x_train, x_test, y_train, y_test = train_test_split(
        usable[features], usable[target], test_size=0.25, random_state=42
    )
    transformers = []
    if categorical:
        transformers.append(("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    candidates = {
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=250, random_state=42),
    }
    evaluations: dict[str, dict[str, float]] = {}
    trained: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("features", clone(preprocess)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        evaluations[name] = {
            "mae": float(mean_absolute_error(y_test, predicted)),
            "r2": float(r2_score(y_test, predicted)) if len(y_test) > 1 else 0.0,
        }
        trained[name] = pipeline

    best = min(evaluations, key=lambda name: evaluations[name]["mae"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained[best], output)
    report = {
        "target": target,
        "features": features,
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "models": evaluations,
        "selected_model": best,
        "artifact": str(output),
    }
    output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.target), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
