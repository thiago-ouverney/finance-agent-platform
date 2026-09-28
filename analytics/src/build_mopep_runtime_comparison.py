"""Une respostas MOPEP, golden local e métricas temporais dos runtimes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analytics.src.evaluate_mopep_tags import compare_tags, macro_f1, parse_model_output, parse_tags


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_golden(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_column = "example_id" if "example_id" in frame else "id" if "id" in frame else None
    if id_column is None or "tags" not in frame:
        raise ValueError("Golden precisa de id/example_id e tags.")
    result = frame[[id_column, "tags"]].rename(columns={id_column: "workload_request_id"})
    result["workload_request_id"] = result["workload_request_id"].astype(str)
    result["expected_tags"] = result["tags"].apply(parse_tags)
    if result["workload_request_id"].duplicated().any():
        raise ValueError("Golden contém IDs duplicados.")
    return result[["workload_request_id", "expected_tags"]]


def load_responses(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("**/text/responses.jsonl")):
        run_root = path.parent.parent
        final_manifest_path = run_root / "json" / "manifest.json"
        responses_manifest_path = run_root / "json" / "responses-manifest.json"
        if not final_manifest_path.is_file() or not responses_manifest_path.is_file():
            continue
        final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        responses_manifest = json.loads(responses_manifest_path.read_text(encoding="utf-8"))
        if final_manifest.get("status") != "complete":
            continue
        if responses_manifest.get("responses_sha256") != sha256_file(path):
            raise ValueError(f"SHA-256 de responses.jsonl divergente: {path}.")
        for row in read_jsonl(path):
            row["source_responses_path"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError(f"Nenhum responses.jsonl completo e íntegro encontrado em {results_dir}.")
    return pd.DataFrame(rows)


def score_responses(responses: pd.DataFrame, golden: pd.DataFrame) -> pd.DataFrame:
    measured = responses.loc[responses["benchmark_phase"] == "measure"].copy()
    measured = measured.merge(golden, on="workload_request_id", how="left", validate="many_to_one")
    if measured["expected_tags"].isna().any():
        missing = measured.loc[measured["expected_tags"].isna(), "workload_request_id"].unique()
        raise ValueError(f"IDs ausentes no golden: {missing[:5].tolist()}.")
    parsed = measured["assistant_output"].fillna("").apply(parse_model_output)
    measured["generated_tags"] = parsed.apply(lambda value: value[0])
    measured["parse_ok"] = parsed.apply(lambda value: value[1])
    metrics = measured.apply(
        lambda row: compare_tags(row["expected_tags"], row["generated_tags"]), axis=1
    )
    for column in ("correct_tags", "missing_tags", "extra_tags", "precision", "recall", "f1", "exact_match"):
        measured[column] = metrics.apply(lambda value: value[column])
    return measured


def quality_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["runtime", "workload_profile", "bucket", "turn_index"]
    for key, group in scored.groupby(group_columns, dropna=False):
        rows.append({
            **dict(zip(group_columns, key)),
            "requests": len(group),
            "parse_rate": group["parse_ok"].mean(),
            "macro_f1": macro_f1(group),
            "mean_individual_f1": group["f1"].mean(),
            "exact_match_rate": group["exact_match"].mean(),
        })
    return pd.DataFrame(rows)


def review_deltas(scored: pd.DataFrame, canonical_path: Path | None) -> pd.DataFrame:
    baselines = []
    closed = scored.loc[scored["workload_profile"] == "mopep-review-closed-loop"]
    if not closed.empty:
        baselines.append(closed.loc[closed["turn_index"] == 1, [
            "experiment_id", "runtime", "workload_request_id", "f1", "exact_match"
        ]].rename(columns={"f1": "baseline_f1", "exact_match": "baseline_exact_match"}))
    replay_baseline = None
    if canonical_path:
        canonical = pd.DataFrame(read_jsonl(canonical_path))
        replay_baseline = canonical.loc[canonical["turn_index"] == 1, [
            "workload_request_id", "assistant_output"
        ]].drop_duplicates("workload_request_id")
    rows = []
    for _, row in scored.loc[scored["turn_index"] == 2].iterrows():
        baseline_f1 = baseline_exact = None
        if row["workload_profile"] == "mopep-review-closed-loop":
            candidates = baselines[0]
            match = candidates.loc[
                (candidates["experiment_id"] == row["experiment_id"])
                & (candidates["workload_request_id"] == row["workload_request_id"])
            ]
            if not match.empty:
                baseline_f1 = match.iloc[0]["baseline_f1"]
                baseline_exact = match.iloc[0]["baseline_exact_match"]
        elif row["workload_profile"] == "mopep-review-replay" and replay_baseline is not None:
            match = replay_baseline.loc[replay_baseline["workload_request_id"] == row["workload_request_id"]]
            if not match.empty:
                tags, _ = parse_model_output(match.iloc[0]["assistant_output"])
                baseline = compare_tags(row["expected_tags"], tags)
                baseline_f1, baseline_exact = baseline["f1"], baseline["exact_match"]
        if baseline_f1 is None:
            continue
        rows.append({
            "experiment_id": row["experiment_id"], "runtime": row["runtime"],
            "workload_profile": row["workload_profile"], "bucket": row["bucket"],
            "workload_request_id": row["workload_request_id"],
            "baseline_f1": baseline_f1, "review_f1": row["f1"],
            "f1_delta": row["f1"] - baseline_f1,
            "baseline_exact_match": baseline_exact, "review_exact_match": row["exact_match"],
            "improved": row["f1"] > baseline_f1, "regressed": row["f1"] < baseline_f1,
        })
    return pd.DataFrame(rows)


def build(*, results_dir: Path, observability_dir: Path, golden_path: Path,
          output_dir: Path, canonical_responses: Path | None = None) -> dict:
    responses = load_responses(results_dir)
    scored = score_responses(responses, load_golden(golden_path))
    requests_path = observability_dir / "all-requests.csv"
    requests = pd.read_csv(requests_path) if requests_path.is_file() else pd.DataFrame()
    if not requests.empty:
        keys = ["experiment_id", "workload_request_id", "turn_index", "repetition"]
        performance = requests.loc[requests["benchmark_phase"] == "measure"].drop_duplicates(keys)
        keep = keys + [column for column in (
            "comparison_id", "time_to_first_token_ms", "end_to_end_latency_seconds",
            "decode_tokens_per_second", "prompt_tokens", "completion_tokens",
        ) if column in performance]
        scored = scored.merge(performance[keep], on=keys, how="left", validate="many_to_one")
    summary = quality_summary(scored)
    deltas = review_deltas(scored, canonical_responses)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = scored.copy()
    for column in ("expected_tags", "generated_tags", "correct_tags", "missing_tags", "extra_tags"):
        serializable[column] = serializable[column].apply(lambda value: json.dumps(value, ensure_ascii=False))
    serializable.to_csv(output_dir / "quality-by-request.csv", index=False)
    summary.to_csv(output_dir / "quality-summary.csv", index=False)
    deltas.to_csv(output_dir / "review-deltas.csv", index=False)
    manifest = {
        "results_dir": str(results_dir.resolve()),
        "observability_dir": str(observability_dir.resolve()),
        "golden_path": str(golden_path.resolve()),
        "canonical_responses": str(canonical_responses.resolve()) if canonical_responses else None,
        "rows": len(scored), "summary_rows": len(summary), "review_delta_rows": len(deltas),
    }
    (output_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--observability-dir", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-responses", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(
        results_dir=args.results_dir, observability_dir=args.observability_dir,
        golden_path=args.golden, output_dir=args.output_dir,
        canonical_responses=args.canonical_responses,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
