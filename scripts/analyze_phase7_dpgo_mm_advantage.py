#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_DATASETS = (
    "parking-garage",
    "sphere",
    "torus",
    "CSAIL",
    "inter",
    "manhattan",
)

DEFAULT_METHODS = ("DRAN", "DPGO-MM", "Manual-reduced")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def by_dataset_method(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("dataset", ""), row.get("method", ""))
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: parse_float(row.get("iter")) or 0.0)
    return grouped


def parse_extra_manual_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--extra-manual-summary must use METHOD=PATH, for example "
            "Manual-reduced-AMM=results/run/summary.csv"
        )
    method, path = value.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("extra manual summary method name is empty")
    return method, Path(path)


def append_manual_summary_curves(
    curve_rows: list[dict[str, str]],
    method: str,
    summary_path: Path,
) -> None:
    summary_rows = read_csv(summary_path)
    for summary in summary_rows:
        if summary.get("status") and summary.get("status") != "ok":
            continue
        dataset = summary.get("dataset", "")
        iter_summary = summary.get("iter_summary_path", "")
        if not dataset or not iter_summary:
            continue
        iter_path = Path(iter_summary)
        if not iter_path.is_absolute():
            iter_path = summary_path.parent / iter_path
        for row in read_csv(iter_path):
            iter_index = row.get("iter") or row.get("iteration") or ""
            curve_rows.append(
                {
                    "problem_type": "six",
                    "dataset": dataset,
                    "method": method,
                    "setting": "fixed_20",
                    "budget": summary.get("budget") or "",
                    "cost": row.get("global_cost") or row.get("cost") or "",
                    "gradient": row.get("gradient") or "",
                    "total_comm_poses": row.get("cumulative_comm_pose_count")
                    or row.get("cumulative_comm_poses")
                    or row.get("total_comm_poses")
                    or "",
                    "total_comm_mb": row.get("cumulative_comm_mb")
                    or row.get("total_comm_mb")
                    or "",
                    "cumulative_comm_mb": row.get("cumulative_comm_mb")
                    or row.get("total_comm_mb")
                    or "",
                    "source_iterations": str(iter_path),
                    "iter": iter_index,
                }
            )


def iter_summary_path_from_summary(summary_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = summary_path.parent / path
    return path


def append_manual_counter_sources(
    sources: list[tuple[str, str, Path]],
    method: str,
    summary_path: Path,
) -> None:
    for summary in read_csv(summary_path):
        if summary.get("status") and summary.get("status") != "ok":
            continue
        dataset = summary.get("dataset", "")
        iter_summary = summary.get("iter_summary_path", "")
        if dataset and iter_summary:
            sources.append(
                (dataset, method, iter_summary_path_from_summary(summary_path, iter_summary))
            )


def summarize_amm_counter_rows(
    sources: list[tuple[str, str, Path]]
) -> list[dict[str, str]]:
    counter_keys = (
        "amm_trace_count",
        "amm_refined_count",
        "amm_accelerated_accepted_count",
        "amm_restart_count",
        "amm_hard_restart_count",
        "amm_soft_restart_count",
        "amm_phi_fallback_count",
        "amm_local_merit_rejected_count",
        "amm_prox_reset_count",
    )
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for dataset, method, path in sources:
        key = (dataset, method, str(path))
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": "missing_iter_summary",
                    "iter_summary_path": str(path),
                }
            )
            continue
        iter_rows = read_csv(path)
        output = {
            "dataset": dataset,
            "method": method,
            "status": "ok",
            "iter_summary_path": str(path),
        }
        has_counter = False
        for counter in counter_keys:
            total = 0.0
            seen_counter = False
            for row in iter_rows:
                if counter not in row:
                    continue
                value = parse_float(row.get(counter))
                if value is not None:
                    has_counter = True
                    seen_counter = True
                    total += value
            output[counter] = fmt(total) if seen_counter else ""
        output["has_amm_counters"] = "1" if has_counter else "0"
        rows.append(output)
    return rows


def aggregate_amm_counters(counter_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    totals: dict[str, dict[str, float]] = {}
    for row in counter_rows:
        if row.get("status") != "ok" or row.get("has_amm_counters") != "1":
            continue
        method = row.get("method", "")
        if not method:
            continue
        bucket = totals.setdefault(method, {})
        for key, value in row.items():
            if not key.startswith("amm_"):
                continue
            parsed = parse_float(value)
            if parsed is None:
                continue
            bucket[key] = bucket.get(key, 0.0) + parsed
    output: list[dict[str, str]] = []
    for method, values in totals.items():
        output.append(
            {
                "method": method,
                "amm_trace_count": fmt(values.get("amm_trace_count")),
                "amm_refined_count": fmt(values.get("amm_refined_count")),
                "amm_accelerated_accepted_count": fmt(
                    values.get("amm_accelerated_accepted_count")
                ),
                "amm_restart_count": fmt(values.get("amm_restart_count")),
                "amm_phi_fallback_count": fmt(values.get("amm_phi_fallback_count")),
                "amm_local_merit_rejected_count": fmt(
                    values.get("amm_local_merit_rejected_count")
                ),
                "amm_prox_reset_count": fmt(values.get("amm_prox_reset_count")),
            }
        )
    return output


def row_cost(row: dict[str, str]) -> float | None:
    return parse_float(row.get("cost") or row.get("global_cost") or row.get("final_cost"))


def row_comm(row: dict[str, str]) -> float | None:
    return parse_float(
        row.get("cumulative_comm_mb")
        or row.get("total_comm_mb")
        or row.get("cumulative_total_comm_mb")
    )


def row_iter(row: dict[str, str]) -> str:
    return row.get("iter") or row.get("iteration") or ""


def build_optimal_costs(metric_rows: list[dict[str, str]]) -> dict[str, float]:
    optimal: dict[str, float] = {}
    for row in metric_rows:
        dataset = row.get("dataset", "")
        if not dataset:
            continue
        value = parse_float(row.get("optimal_cost"))
        if value is None and row.get("method") == "SE-Sync":
            value = parse_float(row.get("cost"))
        if value is not None:
            optimal[dataset] = value
    return optimal


def method_summary(
    dataset: str,
    method: str,
    rows: list[dict[str, str]],
    optimal: float | None,
    tail_window: int,
) -> dict[str, str]:
    if not rows:
        return {
            "dataset": dataset,
            "method": method,
            "status": "missing_curve",
        }
    first = rows[0]
    final = rows[-1]
    first_cost = row_cost(first)
    final_cost = row_cost(final)
    first_comm = row_comm(first) or 0.0
    final_comm = row_comm(final)
    comm_delta = None if final_comm is None else max(final_comm - first_comm, 0.0)
    initial_gap = None
    final_gap = None
    gap_closed = None
    gap_closed_ratio = None
    if optimal is not None:
        if first_cost is not None:
            initial_gap = first_cost - optimal
        if final_cost is not None:
            final_gap = final_cost - optimal
        if initial_gap is not None and final_gap is not None:
            gap_closed = initial_gap - final_gap
            if abs(initial_gap) > 1e-12:
                gap_closed_ratio = gap_closed / initial_gap
    decrease = None
    decrease_per_mb = None
    if first_cost is not None and final_cost is not None:
        decrease = first_cost - final_cost
        if comm_delta is not None and comm_delta > 0:
            decrease_per_mb = decrease / comm_delta
    tail_start = rows[max(0, len(rows) - 1 - tail_window)]
    tail_start_cost = row_cost(tail_start)
    tail_improvement = None
    if tail_start_cost is not None and final_cost is not None:
        tail_improvement = tail_start_cost - final_cost
    return {
        "dataset": dataset,
        "method": method,
        "status": "ok",
        "first_iter": row_iter(first),
        "first_cost": fmt(first_cost),
        "final_iter": row_iter(final),
        "final_cost": fmt(final_cost),
        "optimal_cost": fmt(optimal),
        "initial_gap": fmt(initial_gap),
        "final_gap": fmt(final_gap),
        "gap_closed": fmt(gap_closed),
        "gap_closed_ratio": fmt(gap_closed_ratio),
        "first_comm_mb": fmt(first_comm),
        "final_comm_mb": fmt(final_comm),
        "outer_comm_delta_mb": fmt(comm_delta),
        "cost_decrease": fmt(decrease),
        "cost_decrease_per_outer_mb": fmt(decrease_per_mb),
        "tail_window": str(tail_window),
        "tail_start_iter": row_iter(tail_start),
        "tail_start_cost": fmt(tail_start_cost),
        "tail_improvement": fmt(tail_improvement),
    }


def first_overtake_iter(
    baseline_rows: list[dict[str, str]],
    other_rows: list[dict[str, str]],
) -> tuple[str, float | None, float | None] | tuple[str, None, None]:
    count = min(len(baseline_rows), len(other_rows))
    for idx in range(count):
        baseline_cost = row_cost(baseline_rows[idx])
        other_cost = row_cost(other_rows[idx])
        if baseline_cost is not None and other_cost is not None and baseline_cost <= other_cost:
            return row_iter(baseline_rows[idx]), baseline_cost, other_cost
    return "", None, None


def pairwise_rows(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    datasets: tuple[str, ...],
    baseline: str,
    methods: tuple[str, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        baseline_rows = grouped.get((dataset, baseline), [])
        if not baseline_rows:
            continue
        baseline_final = row_cost(baseline_rows[-1])
        baseline_comm = row_comm(baseline_rows[-1])
        for method in methods:
            if method == baseline:
                continue
            other_rows = grouped.get((dataset, method), [])
            if not other_rows:
                rows.append({"dataset": dataset, "baseline": baseline, "method": method, "status": "missing_curve"})
                continue
            other_final = row_cost(other_rows[-1])
            other_comm = row_comm(other_rows[-1])
            overtake_iter, base_overtake_cost, other_overtake_cost = first_overtake_iter(
                baseline_rows, other_rows
            )
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "method": method,
                    "status": "ok",
                    "baseline_final_cost": fmt(baseline_final),
                    "method_final_cost": fmt(other_final),
                    "method_minus_baseline_cost": fmt(
                        None if baseline_final is None or other_final is None else other_final - baseline_final
                    ),
                    "baseline_final_comm_mb": fmt(baseline_comm),
                    "method_final_comm_mb": fmt(other_comm),
                    "method_comm_ratio_vs_baseline": fmt(
                        None
                        if baseline_comm is None or other_comm is None or abs(baseline_comm) < 1e-12
                        else other_comm / baseline_comm
                    ),
                    "baseline_first_overtake_iter": overtake_iter,
                    "baseline_overtake_cost": fmt(base_overtake_cost),
                    "method_overtake_cost": fmt(other_overtake_cost),
                }
            )
    return rows


def write_report(
    path: Path,
    summary_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    amm_counter_rows: list[dict[str, str]],
    baseline: str,
    methods: tuple[str, ...],
) -> None:
    datasets = tuple(dict.fromkeys(item["dataset"] for item in summary_rows if item.get("dataset")))
    lines = [
        "# Phase 7 DPGO-MM Advantage Analysis",
        "",
        "This report is generated from the current one-manifest curve artifacts.",
        "It is diagnostic evidence, not a new algorithm claim.",
        "",
        "## Final Cost And Communication",
        "",
        "| Dataset | Method | Final cost | Final MB | Tail improvement | Decrease / outer MB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for method in methods:
            item = next(
                (
                    candidate
                    for candidate in summary_rows
                    if candidate.get("dataset") == dataset
                    and candidate.get("method") == method
                ),
                None,
            )
            if item is None or item.get("status") != "ok":
                lines.append(f"| {dataset} | {method} |  |  |  |  |")
            else:
                lines.append(
                    "| {dataset} | {method} | {cost} | {mb} | {tail} | {per_mb} |".format(
                        dataset=dataset,
                        method=method,
                        cost=item.get("final_cost", ""),
                        mb=item.get("final_comm_mb", ""),
                        tail=item.get("tail_improvement", ""),
                        per_mb=item.get("cost_decrease_per_outer_mb", ""),
                    )
                )

    lines.extend(
        [
            "",
            "## Pairwise DPGO-MM Advantage",
            "",
            "| Dataset | Compared method | Final cost delta vs DPGO-MM | Comm ratio | First DPGO-MM overtake iter |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in pair_rows:
        lines.append(
            "| {dataset} | {method} | {delta} | {ratio} | {iter} |".format(
                dataset=item.get("dataset", ""),
                method=item.get("method", ""),
                delta=item.get("method_minus_baseline_cost", ""),
                ratio=item.get("method_comm_ratio_vs_baseline", ""),
                iter=item.get("baseline_first_overtake_iter", ""),
            )
        )

    amm_totals = aggregate_amm_counters(amm_counter_rows)
    if amm_totals:
        lines.extend(
            [
                "",
                "## AMM / Restart Counter Summary",
                "",
                "| Method | Trace | Refined | Accepted acceleration | Restart | Phi fallback | Local merit rejected | Prox reset |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in amm_totals:
            lines.append(
                "| {method} | {trace} | {refined} | {accepted} | {restart} | {phi} | {rejected} | {prox} |".format(
                    method=item.get("method", ""),
                    trace=item.get("amm_trace_count", ""),
                    refined=item.get("amm_refined_count", ""),
                    accepted=item.get("amm_accelerated_accepted_count", ""),
                    restart=item.get("amm_restart_count", ""),
                    phi=item.get("amm_phi_fallback_count", ""),
                    rejected=item.get("amm_local_merit_rejected_count", ""),
                    prox=item.get("amm_prox_reset_count", ""),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            f"- If a method starts lower than {baseline} but is overtaken later, the issue is not initialization alone.",
            f"- If a method uses more MB and still has a larger final gap, the issue is not insufficient pose payload alone.",
            "- If tail improvement is near zero while the gap remains large, the next improvement should target the coordination/update model rather than just more local computation.",
            "- Enabling an AMM switch outside the surrogate model it was designed for is not sufficient evidence of AMM's contribution; the useful mechanism must be tested with a surrogate-consistent local model and a topology-local communication policy.",
            "- A surrogate-consistent portfolio that improves most datasets but misses one should be treated as a direction signal, not a complete superiority claim; the next gate is all-dataset accuracy with bounded local compute.",
            "- The next DRAN candidate should borrow the principle of surrogate-level momentum/restart, but keep DRAN's decentralized boundary variables and avoid global multi-round cost or alpha synchronization.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="DPGO-MM")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--tail-window", type=int, default=5)
    parser.add_argument(
        "--extra-manual-summary",
        action="append",
        type=parse_extra_manual_summary,
        default=[],
        help="Append curves from a manual replay summary using METHOD=PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = read_csv(args.result_root / "normalized_metrics.csv")
    curve_rows = read_csv(args.result_root / "curves.csv")
    datasets = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    methods_list = [part.strip() for part in args.methods.split(",") if part.strip()]
    counter_sources: list[tuple[str, str, Path]] = []
    for row in metric_rows:
        if row.get("problem_type") != "six" or row.get("setting") != "fixed_20":
            continue
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        iter_summary = row.get("source_iterations", "")
        if dataset and method and iter_summary:
            counter_sources.append((dataset, method, Path(iter_summary)))
    for method, summary_path in args.extra_manual_summary:
        append_manual_summary_curves(curve_rows, method, summary_path)
        append_manual_counter_sources(counter_sources, method, summary_path)
        if method not in methods_list:
            methods_list.append(method)
    methods = tuple(methods_list)
    grouped = by_dataset_method(curve_rows)
    optimal_costs = build_optimal_costs(metric_rows)

    summary: list[dict[str, str]] = []
    for dataset in datasets:
        optimal = optimal_costs.get(dataset)
        for method in methods:
            summary.append(
                method_summary(
                    dataset,
                    method,
                    grouped.get((dataset, method), []),
                    optimal,
                    args.tail_window,
                )
            )

    pairs = pairwise_rows(grouped, datasets, args.baseline, methods)
    amm_counters = summarize_amm_counter_rows(counter_sources)
    write_csv(args.output_dir / "phase7_method_progress.csv", summary)
    write_csv(args.output_dir / "phase7_pairwise_vs_dpgo_mm.csv", pairs)
    write_csv(args.output_dir / "phase7_amm_counters.csv", amm_counters)
    write_report(
        args.output_dir / "report.md",
        summary,
        pairs,
        amm_counters,
        args.baseline,
        methods,
    )


if __name__ == "__main__":
    main()
