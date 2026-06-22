#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


DATASETS = ("parking-garage", "sphere", "torus", "CSAIL", "inter", "manhattan")

METHOD_SPECS = {
    "DRAN": ("dran", "iterations/{dataset}.csv"),
    "DPGO-first": ("dpgo_first", "iterations/{dataset}.csv"),
    "DPGO-second": ("dpgo_second", "iterations/{dataset}.csv"),
    "DPGO-MM": ("dpgo_mm", "runs/{dataset}/iteration_summary.csv"),
    "Manual-reduced": ("manual_reduced", "runs/{dataset}/iteration_summary.csv"),
    "MESA": ("mesa", "runs/{dataset}/*/iteration_summary.csv"),
    "Distributed-Mapper": ("distributed_mapper", "iterations/{dataset}.tsv"),
}

MANUAL_REPLAY_METHOD_SPECS = {
    "DPGO-MM MM": ("dpgo_mm_mm", "{dataset}/runs/{dataset}/iteration_summary.csv"),
    "DPGO-MM AMM": ("dpgo_mm_amm", "{dataset}/runs/{dataset}/iteration_summary.csv"),
    "Manual-reduced": ("manual_reduced", "{dataset}/runs/{dataset}/iteration_summary.csv"),
}

COST_FIELDS = ("cost", "global_cost", "final_cost", "objective", "measurement_cost")
GRADIENT_FIELDS = (
    "gradient",
    "grad",
    "grad_norm",
    "gradient_norm",
    "final_gradient",
    "final_gradnorm",
)
ITER_FIELDS = ("iter", "iteration", "Iteration")
COMM_POSE_FIELDS = (
    "cumulative_comm_poses",
    "cumulative_comm_pose_count",
    "cum_comm_poses",
    "total_comm_poses",
)
COMM_MB_FIELDS = (
    "cumulative_comm_mb",
    "cum_comm_mb",
    "total_comm_mb",
    "cumulative_total_comm_mb",
)
TIME_FIELDS = ("time", "cumulative_time_sec", "cum_time_sec", "wall_time_sec")


def read_rows(path: Path) -> list[dict[str, str]]:
    first = path.read_text(errors="ignore").splitlines()[0:1]
    delimiter = "\t" if first and "\t" in first[0] else ","
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def first_present(row: dict[str, str], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field, "")
        if value not in ("", None):
            return str(value)
    return ""


def row_float(row: dict[str, str], fields: tuple[str, ...]) -> float | None:
    return as_float(first_present(row, fields))


def row_iter(row: dict[str, str], fallback: int) -> str:
    value = first_present(row, ITER_FIELDS)
    return value if value else str(fallback)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def find_convergence_index(
    rows: list[dict[str, str]],
    threshold: float,
    window: int,
    increase_tolerance: float,
) -> int | None:
    costs = [row_float(row, COST_FIELDS) for row in rows]
    streak = 0
    for idx in range(1, len(costs)):
        previous = costs[idx - 1]
        current = costs[idx]
        if previous is None or current is None:
            streak = 0
            continue
        improvement = previous - current
        if improvement >= -increase_tolerance and improvement <= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= window:
            return idx
    return None


def find_gradient_index(rows: list[dict[str, str]], threshold: float | None) -> int | None:
    if threshold is None:
        return None
    for idx, row in enumerate(rows):
        gradient = row_float(row, GRADIENT_FIELDS)
        if gradient is not None and gradient <= threshold:
            return idx
    return None


def select_convergence_index(
    cost_delta_idx: int | None,
    gradient_idx: int | None,
    rule: str,
    fallback_idx: int,
) -> tuple[int, str, bool]:
    if rule == "cost_delta":
        return (
            cost_delta_idx if cost_delta_idx is not None else fallback_idx,
            "cost_delta" if cost_delta_idx is not None else "available_final",
            cost_delta_idx is not None,
        )
    if rule == "gradient":
        return (
            gradient_idx if gradient_idx is not None else fallback_idx,
            "gradient" if gradient_idx is not None else "available_final",
            gradient_idx is not None,
        )
    if rule != "cost_delta_or_gradient":
        raise ValueError(f"Unknown convergence rule: {rule}")
    candidates: list[tuple[int, str]] = []
    if cost_delta_idx is not None:
        candidates.append((cost_delta_idx, "cost_delta"))
    if gradient_idx is not None:
        candidates.append((gradient_idx, "gradient"))
    if not candidates:
        return fallback_idx, "available_final", False
    candidates.sort(key=lambda item: item[0])
    return candidates[0][0], candidates[0][1], True


def update_row_from_iterations(
    row: dict[str, str],
    iter_rows: list[dict[str, str]],
    cost_delta_threshold: float,
    gradient_threshold: float | None,
    window: int,
    increase_tolerance: float,
    rule: str,
) -> None:
    cost_delta_idx = find_convergence_index(
        iter_rows, cost_delta_threshold, window, increase_tolerance
    )
    gradient_idx = find_gradient_index(iter_rows, gradient_threshold)
    selected_idx, reason, converged = select_convergence_index(
        cost_delta_idx, gradient_idx, rule, len(iter_rows) - 1
    )
    selected = iter_rows[selected_idx]
    final = iter_rows[-1]
    cost_delta_selected = iter_rows[cost_delta_idx] if cost_delta_idx is not None else None
    gradient_selected = iter_rows[gradient_idx] if gradient_idx is not None else None
    row.update(
        {
            "status": "ok",
            "converged": "1" if converged else "0",
            "convergence_rule": rule,
            "convergence_reason": reason,
            "convergence_iter": row_iter(selected, selected_idx),
            "convergence_cost": fmt(row_float(selected, COST_FIELDS)),
            "convergence_gradient": fmt(row_float(selected, GRADIENT_FIELDS)),
            "convergence_comm_poses": fmt(row_float(selected, COMM_POSE_FIELDS)),
            "convergence_comm_mb": fmt(row_float(selected, COMM_MB_FIELDS)),
            "convergence_time_sec": fmt(row_float(selected, TIME_FIELDS)),
            "cost_delta_converged": "1" if cost_delta_idx is not None else "0",
            "cost_delta_iter": row_iter(cost_delta_selected, cost_delta_idx)
            if cost_delta_selected is not None and cost_delta_idx is not None
            else "",
            "cost_delta_cost": fmt(row_float(cost_delta_selected, COST_FIELDS))
            if cost_delta_selected is not None
            else "",
            "cost_delta_gradient": fmt(row_float(cost_delta_selected, GRADIENT_FIELDS))
            if cost_delta_selected is not None
            else "",
            "cost_delta_comm_mb": fmt(row_float(cost_delta_selected, COMM_MB_FIELDS))
            if cost_delta_selected is not None
            else "",
            "gradient_converged": "1" if gradient_idx is not None else "0",
            "gradient_iter": row_iter(gradient_selected, gradient_idx)
            if gradient_selected is not None and gradient_idx is not None
            else "",
            "gradient_value": fmt(row_float(gradient_selected, GRADIENT_FIELDS))
            if gradient_selected is not None
            else "",
            "gradient_cost": fmt(row_float(gradient_selected, COST_FIELDS))
            if gradient_selected is not None
            else "",
            "gradient_comm_mb": fmt(row_float(gradient_selected, COMM_MB_FIELDS))
            if gradient_selected is not None
            else "",
            "available_final_iter": row_iter(final, len(iter_rows) - 1),
            "available_final_cost": fmt(row_float(final, COST_FIELDS)),
            "available_final_gradient": fmt(row_float(final, GRADIENT_FIELDS)),
            "available_final_comm_poses": fmt(row_float(final, COMM_POSE_FIELDS)),
            "available_final_comm_mb": fmt(row_float(final, COMM_MB_FIELDS)),
            "available_rows": str(len(iter_rows)),
        }
    )


def resolve_iteration_path(setting_root: Path, method_dir: str, pattern: str, dataset: str) -> Path | None:
    base = setting_root / method_dir
    glob_pattern = pattern.format(dataset=dataset)
    if "*" in glob_pattern:
        matches = sorted(base.glob(glob_pattern))
        return matches[-1] if matches else None
    candidate = base / glob_pattern
    return candidate if candidate.is_file() else None


def collect_fair_root(
    fair_root: Path,
    setting: str | None,
    datasets: tuple[str, ...],
    cost_delta_threshold: float,
    gradient_threshold: float | None,
    window: int,
    increase_tolerance: float,
    rule: str,
) -> list[dict[str, str]]:
    six_root = fair_root / "six"
    if setting is None:
        settings = sorted(path.name for path in six_root.glob("fixed_*") if path.is_dir())
        if not settings:
            raise FileNotFoundError(f"No fixed_* settings under {six_root}")
        setting = settings[-1]
    setting_root = six_root / setting
    rows: list[dict[str, str]] = []
    for method, (method_dir, pattern) in METHOD_SPECS.items():
        for dataset in datasets:
            path = resolve_iteration_path(setting_root, method_dir, pattern, dataset)
            row = {
                "result_root": str(fair_root),
                "setting": setting,
                "dataset": dataset,
                "method": method,
                "threshold": fmt(cost_delta_threshold),
                "cost_delta_threshold": fmt(cost_delta_threshold),
                "gradient_threshold": fmt(gradient_threshold),
                "window": str(window),
                "source_iterations": str(path) if path else "",
            }
            if path is None:
                row.update({"status": "missing_iterations", "converged": "0"})
                rows.append(row)
                continue
            iter_rows = read_rows(path)
            if not iter_rows:
                row.update({"status": "empty_iterations", "converged": "0"})
                rows.append(row)
                continue
            update_row_from_iterations(
                row,
                iter_rows,
                cost_delta_threshold,
                gradient_threshold,
                window,
                increase_tolerance,
                rule,
            )
            rows.append(row)
    return rows


def parse_dataset_filter(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return DATASETS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def collect_manual_replay_root(
    replay_root: Path,
    datasets: tuple[str, ...],
    cost_delta_threshold: float,
    gradient_threshold: float | None,
    window: int,
    increase_tolerance: float,
    rule: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method, (method_dir, pattern) in MANUAL_REPLAY_METHOD_SPECS.items():
        for dataset in datasets:
            path = resolve_iteration_path(replay_root, method_dir, pattern, dataset)
            row = {
                "result_root": str(replay_root),
                "setting": "manual_replay",
                "dataset": dataset,
                "method": method,
                "threshold": fmt(cost_delta_threshold),
                "cost_delta_threshold": fmt(cost_delta_threshold),
                "gradient_threshold": fmt(gradient_threshold),
                "window": str(window),
                "source_iterations": str(path) if path else "",
            }
            if path is None:
                row.update({"status": "missing_iterations", "converged": "0"})
                rows.append(row)
                continue
            iter_rows = read_rows(path)
            if not iter_rows:
                row.update({"status": "empty_iterations", "converged": "0"})
                rows.append(row)
                continue
            update_row_from_iterations(
                row,
                iter_rows,
                cost_delta_threshold,
                gradient_threshold,
                window,
                increase_tolerance,
                rule,
            )
            rows.append(row)
    return rows


def parse_method_root(value: str) -> tuple[str, Path, str]:
    if "=" not in value:
        raise ValueError("--method-root must use METHOD=ROOT[:PATTERN]")
    method, rest = value.split("=", 1)
    method = method.strip()
    if not method:
        raise ValueError("--method-root has an empty METHOD name")
    if ":" in rest:
        root_text, pattern = rest.split(":", 1)
    else:
        root_text = rest
        pattern = "runs/{dataset}/iteration_summary.csv"
    root = Path(root_text.strip())
    pattern = pattern.strip()
    if not pattern:
        raise ValueError("--method-root has an empty PATTERN")
    return method, root, pattern


def collect_method_roots(
    method_roots: list[str],
    datasets: tuple[str, ...],
    cost_delta_threshold: float,
    gradient_threshold: float | None,
    window: int,
    increase_tolerance: float,
    rule: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in method_roots:
        method, root, pattern = parse_method_root(spec)
        for dataset in datasets:
            path = resolve_iteration_path(root, "", pattern, dataset)
            row = {
                "result_root": str(root),
                "setting": "method_root",
                "dataset": dataset,
                "method": method,
                "threshold": fmt(cost_delta_threshold),
                "cost_delta_threshold": fmt(cost_delta_threshold),
                "gradient_threshold": fmt(gradient_threshold),
                "window": str(window),
                "source_iterations": str(path) if path else "",
            }
            if path is None:
                row.update({"status": "missing_iterations", "converged": "0"})
                rows.append(row)
                continue
            iter_rows = read_rows(path)
            if not iter_rows:
                row.update({"status": "empty_iterations", "converged": "0"})
                rows.append(row)
                continue
            update_row_from_iterations(
                row,
                iter_rows,
                cost_delta_threshold,
                gradient_threshold,
                window,
                increase_tolerance,
                rule,
            )
            rows.append(row)
    return rows


def write_rows(rows: list[dict[str, str]], output: Path | None) -> None:
    preferred = [
        "dataset",
        "method",
        "converged",
        "convergence_rule",
        "convergence_reason",
        "convergence_iter",
        "convergence_cost",
        "convergence_gradient",
        "convergence_comm_poses",
        "convergence_comm_mb",
        "convergence_time_sec",
        "cost_delta_converged",
        "cost_delta_iter",
        "cost_delta_cost",
        "cost_delta_gradient",
        "cost_delta_comm_mb",
        "gradient_converged",
        "gradient_iter",
        "gradient_value",
        "gradient_cost",
        "gradient_comm_mb",
        "available_final_iter",
        "available_final_cost",
        "available_final_gradient",
        "available_final_comm_poses",
        "available_final_comm_mb",
        "status",
        "setting",
        "threshold",
        "cost_delta_threshold",
        "gradient_threshold",
        "window",
        "source_iterations",
        "result_root",
    ]
    fields: list[str] = []
    seen = set()
    for field in preferred:
        fields.append(field)
        seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)

    stream = sys.stdout if output is None or str(output) == "-" else output.open("w", newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if stream is not sys.stdout:
            stream.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare six-dataset methods at a selected convergence row. "
            "Supported rules are cost-delta plateau, gradient threshold, or either."
        )
    )
    parser.add_argument("--fair-root", type=Path, default=None)
    parser.add_argument(
        "--manual-replay-root",
        type=Path,
        default=None,
        help="Root produced by compare_manual_dpgo_mm_replay.sh.",
    )
    parser.add_argument(
        "--method-root",
        action="append",
        default=[],
        help=(
            "Add an arbitrary method result root as METHOD=ROOT[:PATTERN]. "
            "The default pattern is runs/{dataset}/iteration_summary.csv."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset filter for --manual-replay-root.",
    )
    parser.add_argument("--setting", default=None, help="e.g. fixed_20. Defaults to latest fixed_* directory.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Backward-compatible alias for --cost-delta-threshold.",
    )
    parser.add_argument("--cost-delta-threshold", type=float, default=None)
    parser.add_argument("--gradient-threshold", type=float, default=None)
    parser.add_argument(
        "--rule",
        choices=("cost_delta", "gradient", "cost_delta_or_gradient"),
        default="cost_delta",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--increase-tolerance",
        type=float,
        default=0.0,
        help="Allowed cost increase when counting a small-improvement row. Default resets on any increase.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cost_delta_threshold = (
        args.cost_delta_threshold if args.cost_delta_threshold is not None else args.threshold
    )
    datasets = parse_dataset_filter(args.datasets)
    rows: list[dict[str, str]] = []
    if args.manual_replay_root is not None:
        rows.extend(
            collect_manual_replay_root(
            args.manual_replay_root,
            datasets,
            cost_delta_threshold,
            args.gradient_threshold,
            args.window,
            args.increase_tolerance,
            args.rule,
            )
        )
    if args.fair_root is not None:
        rows.extend(
            collect_fair_root(
            args.fair_root,
            args.setting,
            datasets,
            cost_delta_threshold,
            args.gradient_threshold,
            args.window,
            args.increase_tolerance,
            args.rule,
            )
        )
    if args.method_root:
        rows.extend(
            collect_method_roots(
                args.method_root,
                datasets,
                cost_delta_threshold,
                args.gradient_threshold,
                args.window,
                args.increase_tolerance,
                args.rule,
            )
        )
    if not rows:
        parser.error("one of --fair-root, --manual-replay-root, or --method-root is required")
    write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
