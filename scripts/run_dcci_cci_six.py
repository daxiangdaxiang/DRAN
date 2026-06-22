#!/usr/bin/env python3
"""Compare centralized CCI and distributed initialization on default datasets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


DEFAULT_DATASETS = [
    ("parking-garage", "data/parking-garage.g2o"),
    ("sphere", "data/sphere2500.g2o"),
    ("torus", "data/torus3D.g2o"),
    ("CSAIL", "data/CSAIL.g2o"),
    ("inter", "data/input_INTEL_g2o.g2o"),
    ("manhattan", "data/input_M3500_g2o.g2o"),
    ("ais2klinik", "data/ais2klinik.g2o"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run D-CCI-PCG vs centralized CCI on the default datasets.")
  parser.add_argument("--bench-bin", default="build/bin/bench-dcci")
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--max-iters", type=int, default=5000)
  parser.add_argument("--rel-tol", type=float, default=1e-8)
  parser.add_argument("--abs-tol", type=float, default=1e-8)
  parser.add_argument("--initialization-mode", default="dpcg_cci")
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument(
      "--dataset",
      action="append",
      help="Dataset override as NAME=PATH. Can be repeated.")
  return parser.parse_args(argv)


def dataset_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
  if not args.dataset:
    return [(name, Path(path)) for name, path in DEFAULT_DATASETS]
  specs = []
  for token in args.dataset:
    if "=" not in token:
      path = Path(token)
      specs.append((path.stem, path))
      continue
    name, raw_path = token.split("=", 1)
    specs.append((name, Path(raw_path)))
  return specs


def parse_bench_line(line: str) -> dict[str, str]:
  parts = line.strip().split()
  if not parts or parts[0] != "BENCH_DCCI":
    raise ValueError(f"unexpected bench output line: {line!r}")
  row: dict[str, str] = {}
  for item in parts[1:]:
    if "=" not in item:
      continue
    key, value = item.split("=", 1)
    row[key] = value
  return row


def run_one(args: argparse.Namespace, name: str, path: Path) -> tuple[dict, str]:
  cmd = [
      args.bench_bin,
      "--g2o",
      str(path),
      "--dataset-name",
      name,
      "--robots",
      str(args.num_robots),
      "--max-iters",
      str(args.max_iters),
      "--rel-tol",
      f"{args.rel_tol:.17g}",
      "--abs-tol",
      f"{args.abs_tol:.17g}",
      "--initialization-mode",
      args.initialization_mode,
  ]
  if args.weighted:
    cmd.append("--use-measurement-weight")
  start = time.perf_counter()
  proc = subprocess.run(
      cmd, check=True, text=True, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  wall_time_sec = time.perf_counter() - start
  bench_line = ""
  for raw_line in proc.stdout.splitlines():
    if raw_line.startswith("BENCH_DCCI"):
      bench_line = raw_line
  if not bench_line:
    raise RuntimeError(f"bench output did not contain BENCH_DCCI: {proc.stdout}")
  row = parse_bench_line(bench_line)
  row["runner_wall_time_sec"] = f"{wall_time_sec:.9g}"
  row["command"] = " ".join(cmd)
  return row, proc.stderr


def write_outputs(output_dir: Path, rows: list[dict], stderr_by_dataset: dict):
  output_dir.mkdir(parents=True, exist_ok=True)
  for row in rows:
    row["comm_rounds"] = row.get(
        "method_comm_rounds", row.get("scalar_reductions", "0"))
    row["comm_mb"] = row.get(
        "method_comm_mb", row.get("estimated_scalar_reduction_mb", "0"))
    try:
      block_bytes = float(row.get("bytes_sent", "0"))
    except ValueError:
      block_bytes = 0.0
    row["block_comm_mb"] = f"{block_bytes / 1048576.0:.9g}"
  fieldnames = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  (output_dir / "report.json").write_text(
      json.dumps({
          "row_count": len(rows),
          "rows": rows,
          "stderr_by_dataset": stderr_by_dataset,
      }, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  with (output_dir / "commands.txt").open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(row["command"] + "\n")

  lines = [
      "# Distributed Initialization vs Centralized CCI Default-Dataset Comparison",
      "",
      "| Dataset | Mode | Poses | Edges | CCI cost | Method cost | Cost abs gap | Pose diff | Rot iters | Trans iters | Comm rounds | Comm MB | Block msgs | Block MB | CCI ms | Method ms |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for row in rows:
    lines.append(
        "| {dataset} | {initialization_mode} | {poses} | {edges} | "
        "{cci_cost} | {method_cost} | "
        "{cost_abs_gap} | {relative_pose_matrix_diff} | {rotation_iters} | "
        "{translation_iters} | {comm_rounds} | {comm_mb} | "
        "{block_messages} | {block_comm_mb} | {centralized_ms} | "
        "{method_ms} |".format(**row))
  lines.extend([
      "",
      "Notes:",
      "- Each method uses the same chordal objective and anchor as centralized CCI.",
      "- `Comm rounds` and `Comm MB` use the generic `method_comm_*` fields emitted by `bench-dcci`.",
      "- Default CCI behavior ignores `RelativeSEMeasurement::weight`; pass `--weighted` to enable `weight * kappa/tau`.",
      "",
  ])
  (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = Path("results") / (
        "dcci_cci_default_" + time.strftime("%Y%m%d_%H%M%S"))
  bench_path = Path(args.bench_bin)
  if not bench_path.exists():
    raise FileNotFoundError(f"bench binary not found: {bench_path}")

  rows = []
  stderr_by_dataset = {}
  for name, path in dataset_specs(args):
    print(f"[dcci-cci] {name}: {path}", flush=True)
    row, stderr = run_one(args, name, path)
    rows.append(row)
    if stderr.strip():
      stderr_by_dataset[name] = stderr
    print(
        "[dcci-cci] {dataset} diff={relative_pose_matrix_diff} "
        "cost_gap={cost_abs_gap} rot_iters={rotation_iters} "
        "trans_iters={translation_iters}".format(**row),
        flush=True)
  write_outputs(output_dir, rows, stderr_by_dataset)
  print(output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
