#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


NUMERIC_FIELDS = (
    "local_iter",
    "fobj",
    "F0",
    "F1",
    "gamma",
    "Gkh_initial",
    "minG",
    "Gk_after_accelerated",
    "Gkh_after_restart_check",
    "final_Gk",
    "phi_lhs",
    "phi_rhs",
    "soft_restart_hits0",
    "soft_restart_hits1",
    "num_oscillations",
)

BOOL_FIELDS = (
    "refined",
    "prox_reset",
    "hard_restart",
    "soft_restart",
    "restart_used_xakh",
    "phi_fallback",
)

SURROGATE_MANUAL_FIELDS = {
    "Gkh_initial": "surrogate_Gkh_initial",
    "Gk_after_accelerated": "surrogate_Gk_after_accelerated",
    "Gkh_after_restart_check": "surrogate_Gkh_after_restart_check",
    "final_Gk": "surrogate_final_Gk",
}


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


def parse_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return None


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def row_key(row: dict[str, str]) -> tuple[int, int] | None:
    iter_value = parse_float(row.get("iter"))
    robot_value = parse_float(row.get("robot"))
    if iter_value is None or robot_value is None:
        return None
    return (int(iter_value), int(robot_value))


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    indexed: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = row_key(row)
        if key is None:
            continue
        indexed[key] = row
    return indexed


def manual_field(field: str, mode: str) -> str:
    if mode == "surrogate":
        return SURROGATE_MANUAL_FIELDS.get(field, field)
    return field


def compare_rows(
    dpgo_rows: dict[tuple[int, int], dict[str, str]],
    manual_rows: dict[tuple[int, int], dict[str, str]],
    manual_cost_mode: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    pair_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    numeric_stats: dict[str, dict[str, float]] = {
        field: {"count": 0.0, "sum_abs": 0.0, "max_abs": 0.0}
        for field in NUMERIC_FIELDS
    }
    bool_stats: dict[str, dict[str, float]] = {
        field: {"count": 0.0, "mismatch": 0.0} for field in BOOL_FIELDS
    }

    all_keys = sorted(set(dpgo_rows) | set(manual_rows))
    for key in all_keys:
        dpgo = dpgo_rows.get(key)
        manual = manual_rows.get(key)
        if dpgo is None or manual is None:
            missing_rows.append(
                {
                    "iter": str(key[0]),
                    "robot": str(key[1]),
                    "missing_dpgo": "1" if dpgo is None else "0",
                    "missing_manual": "1" if manual is None else "0",
                }
            )
            continue

        base = {"iter": str(key[0]), "robot": str(key[1])}
        for field in NUMERIC_FIELDS:
            dpgo_value = parse_float(dpgo.get(field))
            manual_name = manual_field(field, manual_cost_mode)
            manual_value = parse_float(manual.get(manual_name))
            delta = None
            abs_delta = None
            if dpgo_value is not None and manual_value is not None:
                delta = manual_value - dpgo_value
                abs_delta = abs(delta)
                stats = numeric_stats[field]
                stats["count"] += 1.0
                stats["sum_abs"] += abs_delta
                stats["max_abs"] = max(stats["max_abs"], abs_delta)
            pair_rows.append(
                {
                    **base,
                    "field": field,
                    "kind": "numeric",
                    "dpgo_value": fmt(dpgo_value),
                    "manual_value": fmt(manual_value),
                    "manual_field": manual_name,
                    "delta_manual_minus_dpgo": fmt(delta),
                    "abs_delta": fmt(abs_delta),
                }
            )

        for field in BOOL_FIELDS:
            dpgo_value = parse_bool(dpgo.get(field))
            manual_value = parse_bool(manual.get(field))
            mismatch = ""
            if dpgo_value is not None and manual_value is not None:
                stats = bool_stats[field]
                stats["count"] += 1.0
                mismatch_bool = dpgo_value != manual_value
                stats["mismatch"] += 1.0 if mismatch_bool else 0.0
                mismatch = "1" if mismatch_bool else "0"
            pair_rows.append(
                {
                    **base,
                    "field": field,
                    "kind": "bool",
                    "dpgo_value": "" if dpgo_value is None else str(int(dpgo_value)),
                    "manual_value": "" if manual_value is None else str(int(manual_value)),
                    "manual_field": field,
                    "bool_mismatch": mismatch,
                }
            )

    summary_rows: list[dict[str, str]] = []
    for field, stats in numeric_stats.items():
        count = int(stats["count"])
        mean_abs = stats["sum_abs"] / stats["count"] if stats["count"] else None
        summary_rows.append(
            {
                "field": field,
                "kind": "numeric",
                "count": str(count),
                "mean_abs_delta": fmt(mean_abs),
                "max_abs_delta": fmt(stats["max_abs"] if count else None),
            }
        )
    for field, stats in bool_stats.items():
        count = int(stats["count"])
        rate = stats["mismatch"] / stats["count"] if stats["count"] else None
        summary_rows.append(
            {
                "field": field,
                "kind": "bool",
                "count": str(count),
                "mismatch_count": str(int(stats["mismatch"])),
                "mismatch_rate": fmt(rate),
            }
        )
    return pair_rows, summary_rows, missing_rows


def iter_sum_rows(
    dpgo_rows: dict[tuple[int, int], dict[str, str]],
    manual_rows: dict[tuple[int, int], dict[str, str]],
    manual_cost_mode: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    iters = sorted({key[0] for key in set(dpgo_rows) | set(manual_rows)})
    for iter_value in iters:
        dpgo_keys = {key for key in dpgo_rows if key[0] == iter_value}
        manual_keys = {key for key in manual_rows if key[0] == iter_value}
        common_keys = sorted(dpgo_keys & manual_keys)

        def sum_field(
            source_rows: dict[tuple[int, int], dict[str, str]],
            keys: set[tuple[int, int]] | list[tuple[int, int]],
            field: str,
            mode: str | None = None,
        ) -> float | None:
            total = 0.0
            count = 0
            for key in keys:
                source_field = manual_field(field, mode) if mode else field
                value = parse_float(source_rows[key].get(source_field))
                if value is None:
                    continue
                total += value
                count += 1
            return total if count else None

        dpgo_fobj = sum_field(dpgo_rows, dpgo_keys, "fobj")
        manual_fobj = sum_field(manual_rows, manual_keys, "fobj")
        dpgo_common_fobj = sum_field(dpgo_rows, common_keys, "fobj")
        manual_common_fobj = sum_field(manual_rows, common_keys, "fobj")
        dpgo_final = sum_field(dpgo_rows, common_keys, "final_Gk")
        manual_final = sum_field(
            manual_rows, common_keys, "final_Gk", manual_cost_mode)

        def delta(left: float | None, right: float | None) -> float | None:
            if left is None or right is None:
                return None
            return right - left

        rows.append(
            {
                "iter": str(iter_value),
                "dpgo_count": str(len(dpgo_keys)),
                "manual_count": str(len(manual_keys)),
                "common_count": str(len(common_keys)),
                "dpgo_fobj_sum": fmt(dpgo_fobj),
                "manual_fobj_sum": fmt(manual_fobj),
                "delta_manual_minus_dpgo_fobj_sum": fmt(
                    delta(dpgo_fobj, manual_fobj)),
                "dpgo_common_fobj_sum": fmt(dpgo_common_fobj),
                "manual_common_fobj_sum": fmt(manual_common_fobj),
                "delta_manual_minus_dpgo_common_fobj_sum": fmt(
                    delta(dpgo_common_fobj, manual_common_fobj)),
                "dpgo_common_final_Gk_sum": fmt(dpgo_final),
                "manual_common_final_Gk_sum": fmt(manual_final),
                "delta_manual_minus_dpgo_common_final_Gk_sum": fmt(
                    delta(dpgo_final, manual_final)),
            }
        )
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, str]],
    missing_rows: list[dict[str, str]],
    iter_rows: list[dict[str, str]],
) -> None:
    worst_numeric = [
        row for row in summary_rows if row.get("kind") == "numeric"
        and row.get("max_abs_delta")
    ]
    worst_numeric.sort(
        key=lambda row: parse_float(row.get("max_abs_delta")) or 0.0,
        reverse=True,
    )
    bool_rows = [row for row in summary_rows if row.get("kind") == "bool"]
    bool_rows.sort(
        key=lambda row: parse_float(row.get("mismatch_rate")) or 0.0,
        reverse=True,
    )
    iter_by_fobj = list(iter_rows)
    iter_by_fobj.sort(
        key=lambda row: abs(
            parse_float(row.get("delta_manual_minus_dpgo_fobj_sum")) or 0.0),
        reverse=True,
    )
    iter_by_final = list(iter_rows)
    iter_by_final.sort(
        key=lambda row: abs(
            parse_float(
                row.get("delta_manual_minus_dpgo_common_final_Gk_sum")) or 0.0
        ),
        reverse=True,
    )
    lines = [
        "# AMM Trace Parity Report",
        "",
        f"- DPGO trace: `{args.dpgo_trace}`",
        f"- Manual trace: `{args.manual_trace}`",
        f"- Manual cost mode: `{args.manual_cost_mode}`",
        f"- Missing keyed rows: `{len(missing_rows)}`",
        "",
        "## Largest Numeric Differences",
        "",
        "| Field | Count | Mean abs delta | Max abs delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in worst_numeric[:10]:
        lines.append(
            f"| {row['field']} | {row['count']} | "
            f"{row.get('mean_abs_delta', '')} | {row.get('max_abs_delta', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boolean Mismatches",
            "",
            "| Field | Count | Mismatches | Mismatch rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in bool_rows:
        lines.append(
            f"| {row['field']} | {row['count']} | "
            f"{row.get('mismatch_count', '')} | {row.get('mismatch_rate', '')} |"
        )
    lines.extend(
        [
            "",
            "## Iteration Sum Checks",
            "",
            "| Iter | DPGO rows | Manual rows | DPGO fobj sum | Manual fobj sum | Delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in iter_by_fobj[:10]:
        lines.append(
            f"| {row['iter']} | {row['dpgo_count']} | {row['manual_count']} | "
            f"{row.get('dpgo_fobj_sum', '')} | "
            f"{row.get('manual_fobj_sum', '')} | "
            f"{row.get('delta_manual_minus_dpgo_fobj_sum', '')} |"
        )
    lines.extend(
        [
            "",
            "| Iter | Common rows | DPGO final_Gk sum | Manual final_Gk sum | Delta |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in iter_by_final[:10]:
        lines.append(
            f"| {row['iter']} | {row['common_count']} | "
            f"{row.get('dpgo_common_final_Gk_sum', '')} | "
            f"{row.get('manual_common_final_Gk_sum', '')} | "
            f"{row.get('delta_manual_minus_dpgo_common_final_Gk_sum', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DPGO-MM and manual replay AMM trace CSVs.")
    parser.add_argument("--dpgo-trace", type=Path, required=True)
    parser.add_argument("--manual-trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--manual-cost-mode",
        choices=("surrogate", "true"),
        default="surrogate",
        help="Use manual surrogate_* cost fields for DPGO-MM surrogate parity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dpgo_rows = index_rows(read_csv(args.dpgo_trace))
    manual_rows = index_rows(read_csv(args.manual_trace))
    pair_rows, summary_rows, missing_rows = compare_rows(
        dpgo_rows, manual_rows, args.manual_cost_mode)
    iter_rows = iter_sum_rows(dpgo_rows, manual_rows, args.manual_cost_mode)
    write_csv(args.out_dir / "trace_pairwise.csv", pair_rows)
    write_csv(args.out_dir / "trace_summary.csv", summary_rows)
    write_csv(args.out_dir / "missing_rows.csv", missing_rows)
    write_csv(args.out_dir / "iter_sums.csv", iter_rows)
    write_report(args.out_dir / "summary.md", args, summary_rows, missing_rows,
                 iter_rows)
    print(f"Wrote {len(pair_rows)} pairwise rows")
    print(f"Wrote {len(summary_rows)} summary rows")
    print(f"Missing keyed rows: {len(missing_rows)}")


if __name__ == "__main__":
    main()
