#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SUMMARY_FIELDS = [
    "method",
    "track",
    "result_dir",
    "final_iter",
    "final_measurement_cost",
    "final_consensus_cost",
    "final_augmented_cost",
    "final_gradient",
    "final_comm_mb",
    "final_time_sec",
    "trajectory_translation_rmse",
    "trajectory_rotation_rmse_deg",
    "object_mean_translation_rmse",
    "object_mean_rotation_rmse_deg",
]


def _last_csv_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows[-1]


def _first_present(row, names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _iteration_row(result_dir):
    result = Path(result_dir)
    for name in ("iteration_summary.csv", "iterations.csv"):
        path = result / name
        if path.exists():
            return _last_csv_row(path)
    raise RuntimeError(f"missing iteration CSV in {result}")


def _gt_fields(result_dir):
    path = Path(result_dir) / "gt_eval.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def summarize_row(manifest_row):
    result_dir = manifest_row["result_dir"]
    iteration = _iteration_row(result_dir)
    gt = _gt_fields(result_dir)
    return {
        "method": manifest_row["method"],
        "track": manifest_row["track"],
        "result_dir": result_dir,
        "final_iter": _first_present(iteration, ["iter"]),
        "final_measurement_cost": _first_present(
            iteration,
            ["measurement_cost", "chordal_measurement_cost", "global_cost"],
        ),
        "final_consensus_cost": _first_present(iteration,
                                               ["consensus_cost"]),
        "final_augmented_cost": _first_present(iteration,
                                               ["augmented_cost"]),
        "final_gradient": _first_present(
            iteration,
            ["gradient", "total_gradnorm", "gradient_norm"],
        ),
        "final_comm_mb": _first_present(
            iteration,
            ["cumulative_comm_mb", "total_comm_mb", "comm_mb"],
        ),
        "final_time_sec": _first_present(
            iteration,
            ["cumulative_time_sec", "time", "elapsed_wall_time_sec"],
        ),
        "trajectory_translation_rmse": str(
            gt.get("trajectory_translation_rmse", "")),
        "trajectory_rotation_rmse_deg": str(
            gt.get("trajectory_rotation_rmse_deg", "")),
        "object_mean_translation_rmse": str(
            gt.get("object_mean_translation_rmse", "")),
        "object_mean_rotation_rmse_deg": str(
            gt.get("object_mean_rotation_rmse_deg", "")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize DRAN-object fair-comparison result directories.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.manifest).open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    summary_rows = [summarize_row(row) for row in manifest_rows]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
