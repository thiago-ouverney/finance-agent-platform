"""Consolida os summary.csv produzidos pelo benchmark de runtimes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


IDENTITY_FIELDS = ("source_file", "experiment_id")


def find_summaries(roots: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "summary.csv":
            files.add(root.resolve())
        elif root.exists():
            files.update(path.resolve() for path in root.rglob("summary.csv"))
    return sorted(files)


def experiment_id(summary_path: Path) -> str:
    parent = summary_path.parent
    if parent.name == "csv":
        parent = parent.parent
    return parent.name


def collect_rows(roots: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in find_summaries(roots):
        with summary.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                clean = {str(key): value or "" for key, value in row.items() if key}
                clean["source_file"] = str(summary)
                clean["experiment_id"] = experiment_id(summary)
                rows.append(clean)
    return rows


def write_dataset(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        raise ValueError("Nenhum summary.csv encontrado nos caminhos informados.")
    fields = list(IDENTITY_FIELDS)
    fields.extend(
        sorted({key for row in rows for key in row if key not in IDENTITY_FIELDS})
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--results",
        action="append",
        required=True,
        type=Path,
        help="Pasta de resultados ou summary.csv; pode ser repetido.",
    )
    result.add_argument("--output", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    rows = collect_rows(args.results)
    write_dataset(rows, args.output)
    print(f"dataset={args.output} rows={len(rows)} summaries={len(find_summaries(args.results))}")


if __name__ == "__main__":
    main()
