#!/usr/bin/env python3
"""Run PR14 RIFT-IF seven-dataset initialization benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from run_dcci_cci_six import DEFAULT_DATASETS, parse_bench_line  # noqa: E402


PR14_FIELD_ORDER = [
    "status",
    "dataset",
    "g2o",
    "dim",
    "poses",
    "edges",
    "robots",
    "initialization_mode",
    "selected_backend",
    "cci_cost",
    "rift_cost",
    "cost_abs_gap",
    "cost_rel_gap",
    "pose_diff",
    "rift_max_clique_blocks",
    "rift_max_separator_blocks",
    "rift_num_cliques",
    "rift_num_tree_edges",
    "directed_messages",
    "actual_message_bytes",
    "method_comm_rounds",
    "method_comm_mb",
    "symbolic_ms",
    "factor_transfer_ms",
    "message_qr_ms",
    "belief_solve_ms",
    "final_interface_residual",
    "used_global_matrix",
    "used_direct_solver",
    "used_collective",
    "centralized_ms",
    "method_ms",
    "runner_wall_time_sec",
    "command",
    "error",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run RIFT-IF PR14 seven-dataset benchmark and baselines.")
  parser.add_argument("--bench-bin", default="build/bin/bench-dcci")
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--max-iters", type=int, default=20000)
  parser.add_argument("--rel-tol", type=float, default=1e-10)
  parser.add_argument("--abs-tol", type=float, default=1e-10)
  parser.add_argument(
      "--methods",
      default="ted_cci_rift_if,ted_cci_sr_direct,dpcg_cci",
      help="Comma-separated initialization modes to run.")
  parser.add_argument("--interface-backend", default="rift_auto")
  parser.add_argument("--use-rotation-multi-rhs", default="true")
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--timeout-sec", type=float, default=0.0)
  parser.add_argument(
      "--dataset",
      action="append",
      help="Dataset override as NAME=PATH. Can be repeated.")
  return parser.parse_args(argv)


def dataset_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
  if not args.dataset:
    return [(name, Path(path)) for name, path in DEFAULT_DATASETS]
  specs: list[tuple[str, Path]] = []
  for token in args.dataset:
    if "=" not in token:
      path = Path(token)
      specs.append((path.stem, path))
      continue
    name, raw_path = token.split("=", 1)
    specs.append((name, Path(raw_path)))
  return specs


def methods(args: argparse.Namespace) -> list[str]:
  return [item.strip() for item in args.methods.split(",") if item.strip()]


def add_pr14_aliases(row: dict[str, str]) -> dict[str, str]:
  aliases = {
      "dim": "dimension",
      "selected_backend": "rift_selected_backend",
      "rift_cost": "method_cost",
      "pose_diff": "method_relative_pose_matrix_diff",
      "directed_messages": "rift_directed_messages_sent",
      "actual_message_bytes": "rift_actual_message_bytes",
      "symbolic_ms": "rift_symbolic_ms",
      "message_qr_ms": "rift_message_qr_ms",
      "belief_solve_ms": "rift_belief_solve_ms",
      "final_interface_residual": "rift_final_interface_residual",
      "used_global_matrix": "rift_used_global_matrix",
      "used_direct_solver": "rift_used_direct_solver",
      "used_collective": "rift_used_collective",
  }
  for dst, src in aliases.items():
    row.setdefault(dst, row.get(src, ""))
  row.setdefault("factor_transfer_ms", "0")
  row.setdefault("error", "")
  for key in PR14_FIELD_ORDER:
    row.setdefault(key, "")
  return row


def command_for(args: argparse.Namespace, name: str, path: Path,
                method: str) -> list[str]:
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
      method,
  ]
  if args.weighted:
    cmd.append("--use-measurement-weight")
  if method == "ted_cci_rift_if":
    cmd.extend([
        "--interface-backend",
        args.interface_backend,
        "--use-rotation-multi-rhs",
        args.use_rotation_multi_rhs,
        "--forbid-direct-interface-solver",
        "true",
        "--forbid-global-interface-matrix",
        "true",
        "--forbid-collectives",
        "true",
    ])
  return cmd


def run_one(args: argparse.Namespace, name: str, path: Path,
            method: str) -> tuple[dict[str, str], str]:
  cmd = command_for(args, name, path, method)
  start = time.perf_counter()
  try:
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_sec if args.timeout_sec > 0 else None)
  except subprocess.TimeoutExpired as exc:
    wall_time_sec = time.perf_counter() - start
    row = {
        "status": "timeout",
        "dataset": name,
        "g2o": str(path),
        "robots": str(args.num_robots),
        "initialization_mode": method,
        "runner_wall_time_sec": f"{wall_time_sec:.9g}",
        "command": " ".join(cmd),
        "error": str(exc),
    }
    return add_pr14_aliases(row), str(exc)

  wall_time_sec = time.perf_counter() - start
  bench_line = ""
  for raw_line in proc.stdout.splitlines():
    if raw_line.startswith("BENCH_DCCI"):
      bench_line = raw_line
  if proc.returncode != 0 or not bench_line:
    row = {
        "status": "failed",
        "dataset": name,
        "g2o": str(path),
        "robots": str(args.num_robots),
        "initialization_mode": method,
        "runner_wall_time_sec": f"{wall_time_sec:.9g}",
        "command": " ".join(cmd),
        "error": (proc.stderr.strip() or proc.stdout.strip()),
    }
    return add_pr14_aliases(row), proc.stderr

  row = parse_bench_line(bench_line)
  row["status"] = "ok"
  row["runner_wall_time_sec"] = f"{wall_time_sec:.9g}"
  row["command"] = " ".join(cmd)
  return add_pr14_aliases(row), proc.stderr


def ordered_fieldnames(rows: list[dict[str, str]]) -> list[str]:
  fieldnames = list(PR14_FIELD_ORDER)
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  return fieldnames


def write_outputs(output_dir: Path, rows: list[dict[str, str]],
                  stderr_by_run: dict[str, str]) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  fieldnames = ordered_fieldnames(rows)
  with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  (output_dir / "report.json").write_text(
      json.dumps({
          "row_count": len(rows),
          "rows": rows,
          "stderr_by_run": stderr_by_run,
      }, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  with (output_dir / "commands.txt").open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(row["command"] + "\n")

  lines = [
      "# RIFT-IF PR14 Seven-Dataset Benchmark",
      "",
      "| Dataset | Method | Status | Backend | Pose diff | Cost gap | Directed messages | Comm MB | Actual bytes | Method ms | Guard |",
      "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ]
  for row in rows:
    guard = "{}/{}/{}".format(
        row.get("used_global_matrix", ""),
        row.get("used_direct_solver", ""),
        row.get("used_collective", ""))
    lines.append(
        "| {dataset} | {initialization_mode} | {status} | {selected_backend} | "
        "{pose_diff} | {cost_abs_gap} | {directed_messages} | "
        "{method_comm_mb} | {actual_message_bytes} | {method_ms} | "
        "{guard} |".format(guard=guard, **row))
  lines.extend([
      "",
      "Guard column is `used_global_matrix/used_direct_solver/used_collective`; RIFT deployment rows should be `0/0/0`.",
      "",
  ])
  (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = Path("results") / (
        "rift_if_pr14_seven_" + time.strftime("%Y%m%d_%H%M%S"))
  bench_path = Path(args.bench_bin)
  if not bench_path.exists():
    raise FileNotFoundError(f"bench binary not found: {bench_path}")

  rows: list[dict[str, str]] = []
  stderr_by_run: dict[str, str] = {}
  for name, path in dataset_specs(args):
    for method in methods(args):
      print(f"[rift-if] {name}: {method}", flush=True)
      row, stderr = run_one(args, name, path, method)
      rows.append(row)
      run_key = f"{name}:{method}"
      if stderr.strip():
        stderr_by_run[run_key] = stderr
      print(
          "[rift-if] {dataset} {initialization_mode} status={status} "
          "backend={selected_backend} pose_diff={pose_diff} "
          "cost_gap={cost_abs_gap}".format(**row),
          flush=True)
  write_outputs(output_dir, rows, stderr_by_run)
  print(output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
