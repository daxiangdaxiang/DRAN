#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable


DATASET_ALIASES = {
    "drone": "drone_object",
}


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    first_line = path.read_text(errors="ignore").splitlines()[0:1]
    delimiter = "\t" if first_line and "\t" in first_line[0] else ","
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def write_table(path: Path | None, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    seen = set()
    preferred = [
        "problem_type",
        "dataset",
        "method",
        "setting",
        "budget",
        "status",
        "cost",
        "optimal_cost",
        "cost_gap_abs",
        "cost_gap_rel",
        "log10_gap_abs",
        "final_iter",
        "total_comm_poses",
        "outer_comm_mb",
        "init_comm_mb",
        "total_comm_mb",
        "outer_time_sec",
        "init_time_sec",
        "total_time_sec",
        "trajectory_translation_rmse",
        "trajectory_rotation_rmse_deg",
        "object_mean_translation_rmse",
        "object_mean_rotation_rmse_deg",
        "object_copy_translation_rmse",
        "object_copy_rotation_rmse_deg",
        "source_summary",
        "source_iterations",
        "source_gt",
    ]
    for field in preferred:
        fields.append(field)
        seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)

    output = sys.stdout if path is None or str(path) == "-" else path.open("w", newline="")
    try:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if output is not sys.stdout:
            output.close()


def first_present(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return str(value)
    return ""


def parse_float(value: str | None) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.17g}"


def canonical_dataset(name: str) -> str:
    name = name.strip()
    return DATASET_ALIASES.get(name, name)


def resolve_manifest(result_root: Path | None, manifest: Path | None) -> Path | None:
    if manifest is not None:
        return manifest
    if result_root is None:
        return None
    candidate = result_root / "manifest.csv"
    return candidate if candidate.is_file() else None


def load_manifest(result_root: Path | None, manifest: Path | None) -> list[dict[str, str]]:
    manifest_path = resolve_manifest(result_root, manifest)
    if manifest_path is None:
        return []
    rows = read_table(manifest_path)
    for row in rows:
        row["_manifest_path"] = str(manifest_path)
    return rows


def is_result_manifest_row(row: dict[str, str]) -> bool:
    """Return true for rows that should appear in normalized metric tables."""
    setting = row.get("setting", "")
    method = row.get("method", "")
    if setting == "topology_generation":
        return False
    if method.startswith("topology-"):
        return False
    return True


def find_summary_path(run_root: Path) -> Path | None:
    for name in ("final_summary.csv", "summary.csv", "final_summary.tsv"):
        candidate = run_root / name
        if candidate.is_file():
            return candidate
    if (run_root / "iterations.csv").is_file():
        return run_root / "iterations.csv"
    return None


def find_iteration_path(run_root: Path, summary_row: dict[str, str]) -> Path | None:
    for key in ("iter_summary_path", "csv_path", "source_iterations"):
        value = summary_row.get(key, "")
        if value:
            candidate = Path(value)
            if candidate.is_file():
                return candidate
    for rel in (
        "iterations.csv",
        "iteration_summary.csv",
    ):
        candidate = run_root / rel
        if candidate.is_file():
            return candidate
    return None


def last_iteration_row(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_table(path)
    return rows[-1] if rows else {}


def load_gt(run_root: Path) -> dict[str, str]:
    path = run_root / "gt_eval.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    mapping = {
        "trajectory_translation_rmse": "trajectory_translation_rmse",
        "trajectory_rotation_rmse_deg": "trajectory_rotation_rmse_deg",
        "object_mean_translation_rmse": "object_mean_translation_rmse",
        "object_mean_rotation_rmse_deg": "object_mean_rotation_rmse_deg",
        "object_copy_translation_rmse": "object_copy_translation_rmse",
        "object_copy_rotation_rmse_deg": "object_copy_rotation_rmse_deg",
    }
    row = {out_key: format_float(parse_float(str(data.get(in_key, ""))))
           for in_key, out_key in mapping.items()}
    row["source_gt"] = str(path)
    return row


def load_init_comm(run_root: Path) -> tuple[str, str]:
    path = run_root / "init_summary.csv"
    rows = read_table(path)
    if not rows:
        return "", ""
    row = rows[-1]
    return (
        first_present(row, ("initialization_comm_mb", "init_comm_mb")),
        first_present(row, ("initialization_time_sec", "init_time_sec")),
    )


def load_optimal_costs(rows: list[dict[str, str]]) -> dict[str, float]:
    optimal: dict[str, float] = {}
    for row in rows:
        method = row.get("method", "")
        if "SE-Sync" not in method and row.get("mode", "") not in ("six", "drone"):
            continue
        dataset = canonical_dataset(first_present(row, ("dataset", "name")))
        value = parse_float(first_present(row, ("optimal_cost", "cost", "final_cost")))
        if dataset and value is not None:
            optimal[dataset] = value
    return optimal


def normalize_run_rows(
    manifest_row: dict[str, str], optimal_costs: dict[str, float]
) -> list[dict[str, str]]:
    run_root = Path(manifest_row.get("out_root", ""))
    summary_path = find_summary_path(run_root)
    if summary_path is None:
        return [{
            "problem_type": manifest_row.get("problem_type", ""),
            "dataset": "",
            "method": manifest_row.get("method", ""),
            "setting": manifest_row.get("setting", ""),
            "budget": manifest_row.get("budget", ""),
            "status": manifest_row.get("status", "missing_summary"),
            "source_summary": "",
        }]

    summary_rows = read_table(summary_path)
    if summary_path.name == "iterations.csv":
        summary_rows = summary_rows[-1:] if summary_rows else []
    if not summary_rows:
        return [{
            "problem_type": manifest_row.get("problem_type", ""),
            "dataset": "",
            "method": manifest_row.get("method", ""),
            "setting": manifest_row.get("setting", ""),
            "budget": manifest_row.get("budget", ""),
            "status": manifest_row.get("status", "empty_summary") or "empty_summary",
            "source_summary": str(summary_path),
        }]
    normalized: list[dict[str, str]] = []
    for source_row in summary_rows:
        dataset = canonical_dataset(first_present(source_row, ("dataset",)))
        if not dataset and manifest_row.get("problem_type") == "drone_object":
            dataset = "drone_object"
        method = manifest_row.get("method") or source_row.get("method", "")
        status = source_row.get("status") or manifest_row.get("status", "")

        if "SE-Sync" in method:
            cost_text = first_present(source_row, ("optimal_cost", "fxhat", "cost"))
        elif manifest_row.get("problem_type") == "drone_object":
            cost_text = first_present(source_row, (
                "chordal_measurement_cost",
                "measurement_cost",
                "global_cost",
                "final_cost",
            ))
        else:
            cost_text = first_present(source_row, (
                "final_cost",
                "global_cost",
                "final_objective",
                "cost",
            ))

        cost = parse_float(cost_text)
        optimal = optimal_costs.get(dataset)
        gap_abs = cost - optimal if cost is not None and optimal is not None else None
        gap_rel = (
            gap_abs / max(abs(optimal), 1e-12)
            if gap_abs is not None and optimal is not None
            else None
        )
        log_gap = (
            math.log10(max(abs(gap_abs), 1e-300))
            if gap_abs is not None
            else None
        )

        init_comm, init_time = load_init_comm(run_root)
        outer_comm = first_present(source_row, (
            "cumulative_comm_mb",
            "total_comm_mb",
        ))
        total_comm = outer_comm
        init_comm_float = parse_float(init_comm)
        outer_comm_float = parse_float(outer_comm)
        if init_comm_float is not None and outer_comm_float is not None:
            total_comm = format_float(init_comm_float + outer_comm_float)

        outer_time = first_present(source_row, (
            "cumulative_time_sec",
            "elapsed_sec",
            "total_time_sec",
        ))
        if not outer_time:
            ms = parse_float(source_row.get("cumulative_parallel_compute_ms", ""))
            if ms is not None:
                outer_time = format_float(ms / 1000.0)
        total_time = outer_time
        init_time_float = parse_float(init_time)
        outer_time_float = parse_float(outer_time)
        if init_time_float is not None and outer_time_float is not None:
            total_time = format_float(init_time_float + outer_time_float)

        iteration_path = find_iteration_path(run_root, source_row)
        final_iteration = last_iteration_row(iteration_path)
        final_iter = first_present(source_row, ("final_iter", "iter"))
        if not final_iter:
            final_iter = first_present(final_iteration, ("iter", "iteration"))

        out = {
            "problem_type": manifest_row.get("problem_type", ""),
            "dataset": dataset,
            "method": method,
            "setting": manifest_row.get("setting", ""),
            "budget": manifest_row.get("budget", ""),
            "status": status,
            "cost": format_float(cost),
            "optimal_cost": format_float(optimal),
            "cost_gap_abs": format_float(gap_abs),
            "cost_gap_rel": format_float(gap_rel),
            "log10_gap_abs": format_float(log_gap),
            "final_iter": final_iter,
            "total_comm_poses": first_present(source_row, (
                "total_comm_poses",
                "cumulative_object_comm_poses",
                "cum_comm_poses",
                "comm_pose_count",
            )),
            "outer_comm_mb": outer_comm,
            "init_comm_mb": init_comm,
            "total_comm_mb": total_comm,
            "outer_time_sec": outer_time,
            "init_time_sec": init_time,
            "total_time_sec": total_time,
            "source_summary": str(summary_path),
            "source_iterations": str(iteration_path) if iteration_path else "",
        }
        out.update(load_gt(run_root))
        normalized.append(out)
    return normalized


def discover_result_dirs(result_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(result_root.rglob("*")):
        if not path.is_dir():
            continue
        summary = find_summary_path(path)
        if summary is None:
            continue
        rows.append({
            "problem_type": "unknown",
            "setting": "",
            "method": "",
            "budget": "",
            "out_root": str(path),
            "status": "",
        })
    return rows


def build_curve_rows(
    normalized_rows: list[dict[str, str]], optimal_costs: dict[str, float]
) -> list[dict[str, str]]:
    curve_rows: list[dict[str, str]] = []
    for row in normalized_rows:
        path_text = row.get("source_iterations", "")
        if not path_text:
            continue
        path = Path(path_text)
        rows = read_table(path)
        if not rows:
            continue
        dataset = row.get("dataset", "")
        optimal = optimal_costs.get(dataset)
        for item in rows:
            cost = parse_float(first_present(item, (
                "chordal_measurement_cost",
                "measurement_cost",
                "global_cost",
                "final_cost",
                "cost",
                "local_model_cost",
            )))
            gap_abs = cost - optimal if cost is not None and optimal is not None else None
            curve_rows.append({
                "problem_type": row.get("problem_type", ""),
                "dataset": dataset,
                "method": row.get("method", ""),
                "setting": row.get("setting", ""),
                "budget": row.get("budget", ""),
                "iter": first_present(item, ("iter", "iteration")),
                "cost": format_float(cost),
                "optimal_cost": format_float(optimal),
                "cost_gap_abs": format_float(gap_abs),
                "cost_gap_rel": format_float(
                    gap_abs / max(abs(optimal), 1e-12)
                    if gap_abs is not None and optimal is not None
                    else None
                ),
                "cumulative_comm_mb": first_present(item, (
                    "cumulative_comm_mb",
                    "cum_comm_mb",
                    "cumulative_comm",
                    "total_comm_mb",
                )),
                "cumulative_time_sec": first_present(item, (
                    "cumulative_time_sec",
                    "time",
                    "elapsed_sec",
                )),
                "source_iterations": str(path),
            })
    return curve_rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize chordal DRAN fair benchmark outputs."
    )
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--curve-output", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_rows = load_manifest(args.result_root, args.manifest)
    if not manifest_rows:
        if args.result_root is None:
            parser.error("Provide --manifest or --result-root")
        manifest_rows = discover_result_dirs(args.result_root)

    provisional_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        if not is_result_manifest_row(row):
            continue
        summary = find_summary_path(Path(row.get("out_root", "")))
        if summary is None:
            continue
        for source in read_table(summary):
            source["method"] = row.get("method", source.get("method", ""))
            provisional_rows.append(source)
    optimal_costs = load_optimal_costs(provisional_rows)

    normalized_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        if not is_result_manifest_row(row):
            continue
        normalized_rows.extend(normalize_run_rows(row, optimal_costs))

    write_table(args.output, normalized_rows)

    if args.curve_output is not None:
        curve_rows = build_curve_rows(normalized_rows, optimal_costs)
        write_table(args.curve_output, curve_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
