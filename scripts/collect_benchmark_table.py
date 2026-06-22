#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def resolve_summary_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (path / "summary.csv", path / "final_summary.csv"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not find summary.csv under {path}")


def read_csv_rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return list(reader)


def normalize_cost(row: dict[str, str]) -> str:
    for key in ("cost", "final_cost", "final_objective", "mean_residual"):
        value = row.get(key, "")
        if value not in ("", None):
            return value
    return ""


def load_optimal_costs(optimal_csv: Path | None) -> dict[str, str]:
    if optimal_csv is None:
        return {}
    rows = read_csv_rows(optimal_csv)
    costs: dict[str, str] = {}
    for row in rows:
        dataset = (row.get("dataset") or row.get("name") or "").strip()
        if not dataset:
            continue
        optimal = row.get("optimal_cost") or row.get("cost") or row.get("final_cost") or ""
        costs[dataset] = optimal
    return costs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Combine benchmark summary CSV files.")
    parser.add_argument("result_dirs", nargs="+", help="Result directories or summary CSV files.")
    parser.add_argument("--optimal-csv", dest="optimal_csv", default=None, help="Optional CSV mapping datasets to optimal_cost.")
    parser.add_argument("-o", "--output", default="-", help="Output CSV path, or '-' for stdout.")
    args = parser.parse_args(argv)

    summary_paths = [resolve_summary_path(Path(item)) for item in args.result_dirs]
    optimal_costs = load_optimal_costs(Path(args.optimal_csv)) if args.optimal_csv else {}

    rows: list[dict[str, str]] = []
    field_order: list[str] = [
        "source_summary",
        "dataset",
        "method",
        "status",
        "cost",
        "optimal_cost",
        "cost_gap",
        "relative_cost_gap",
    ]
    seen_fields = set(field_order)

    for summary_path in summary_paths:
        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            for field in reader.fieldnames:
                if field not in seen_fields:
                    field_order.append(field)
                    seen_fields.add(field)
            for row in reader:
                normalized = dict(row)
                normalized["source_summary"] = str(summary_path)
                normalized["cost"] = normalize_cost(normalized)
                dataset = (normalized.get("dataset") or "").strip()
                normalized["optimal_cost"] = optimal_costs.get(dataset, "")
                try:
                    cost = float(normalized["cost"])
                    optimal = float(normalized["optimal_cost"])
                    gap = cost - optimal
                    normalized["cost_gap"] = f"{gap:.17g}"
                    normalized["relative_cost_gap"] = f"{gap / max(1.0, abs(optimal)):.17g}"
                except (TypeError, ValueError):
                    normalized["cost_gap"] = ""
                    normalized["relative_cost_gap"] = ""
                rows.append(normalized)

    for row in rows:
        for field in row.keys():
            if field not in seen_fields:
                field_order.append(field)
                seen_fields.add(field)

    output_handle = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    try:
        writer = csv.DictWriter(output_handle, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if output_handle is not sys.stdout:
            output_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
