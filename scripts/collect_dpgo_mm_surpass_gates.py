#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASELINE_VARIANT = "dpgo_mm_amm"

NORMALIZED_PASSTHROUGH_FIELDS = [
    "timestamp",
    "run_root",
    "seed",
    "git_sha",
    "build",
    "command",
    "problem_type",
    "setting",
    "num_robots",
    "topology",
    "payload_breakdown",
    "init_time_sec",
    "outer_time_sec",
    "total_iter",
    "final_iter",
    "failure_marker",
    "gap_to_sesync",
    "baseline_cost",
    "convergence_iter",
    "convergence_comm_poses",
    "convergence_comm_mb",
    "trajectory_translation_rmse",
    "trajectory_rotation_rmse_deg",
    "object_mean_translation_rmse",
    "object_mean_rotation_rmse_deg",
    "source_summary",
    "source_iterations",
    "source_gt",
]


@dataclass(frozen=True)
class ComparisonSpec:
  gate: str
  label: str
  path: Path


def parse_spec(value: str) -> ComparisonSpec:
  parts = value.split(":", 2)
  if len(parts) != 3:
    raise argparse.ArgumentTypeError(
        "comparison spec must be GATE:LABEL:/path/to/comparison.csv")
  gate, label, path = parts
  gate = gate.strip().upper()
  if gate not in {"A", "B", "C"}:
    raise argparse.ArgumentTypeError("gate must be one of A, B, or C")
  label = label.strip()
  if not label:
    raise argparse.ArgumentTypeError("label must be nonempty")
  return ComparisonSpec(gate=gate, label=label, path=Path(path))


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
      return []
    return [dict(row) for row in reader]


def write_csv(
    path: Path | None, rows: list[dict[str, str]],
    fields: list[str] | None = None
) -> None:
  if fields is None:
    fields = preferred_fields(rows)
  output = sys.stdout if path is None or str(path) == "-" else path.open(
      "w", newline="")
  try:
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  finally:
    if output is not sys.stdout:
      output.close()


def preferred_fields(rows: Iterable[dict[str, str]]) -> list[str]:
  preferred = [
      "gate",
      "claim_scope",
      "label",
      "dataset",
      "updates",
      "method",
      "baseline",
      "status",
      "deployable_flag",
      "init_mode",
      "dist_init",
      "final_cost",
      "baseline_cost",
      "cost_delta",
      "cost_better",
      "gradient",
      "baseline_gradient",
      "gradient_delta",
      "init_comm_mb",
      "outer_comm_mb",
      "total_comm_mb",
      "payload_breakdown",
      "baseline_total_comm_mb",
      "comm_delta_mb",
      "comm_ratio",
      "same_or_lower_comm",
      "convergence_iter",
      "convergence_comm_poses",
      "convergence_comm_mb",
      "topology",
      "wall_time_sec",
      "baseline_wall_time_sec",
      "wall_time_delta_sec",
      "solver_time_per_node_sec",
      "baseline_solver_time_per_node_sec",
      "same_comm_cost_win",
      "run_root",
      "summary_csv",
      "source_comparison_csv",
      "num_datasets",
      "cost_win_count",
      "same_comm_cost_win_count",
      "mean_cost_delta",
      "mean_comm_ratio",
      "max_comm_ratio",
  ]
  fields: list[str] = []
  seen: set[str] = set()
  for field in preferred:
    fields.append(field)
    seen.add(field)
  for row in rows:
    for field in row:
      if field not in seen:
        fields.append(field)
        seen.add(field)
  return fields


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


def bool_text(value: bool | None) -> str:
  if value is None:
    return "unknown"
  return "true" if value else "false"


def gate_claim_scope(gate: str) -> str:
  return {
      "A": "same-init optimizer",
      "B": "deployment budget",
      "C": "object/topology scope",
  }[gate]


def default_init_mode(gate: str) -> str:
  return {
      "A": "centralized_chordal_same_init",
      "B": "deployment_init",
      "C": "scenario_specific",
  }[gate]


def default_dist_init(gate: str) -> str:
  return "false" if gate == "A" else "true"


def default_deployable_flag(gate: str, method: str) -> str:
  if gate == "A":
    return "diagnostic_same_init"
  if "global" in method:
    return "diagnostic_global_sync"
  return "candidate"


def normalize_rows(spec: ComparisonSpec) -> list[dict[str, str]]:
  source_rows = read_csv(spec.path)
  rows: list[dict[str, str]] = []
  for source in source_rows:
    method = source.get("variant", source.get("method", ""))
    total_comm = parse_float(source.get("total_comm_mb"))
    row = {
        "gate": spec.gate,
        "claim_scope": gate_claim_scope(spec.gate),
        "label": spec.label,
        "dataset": source.get("dataset", ""),
        "updates": source.get("updates", source.get("outer_updates", "")),
        "method": method,
        "baseline": BASELINE_VARIANT,
        "status": source.get("status", ""),
        "deployable_flag": default_deployable_flag(spec.gate, method),
        "init_mode": source.get("init_mode", default_init_mode(spec.gate)),
        "dist_init": source.get("dist_init", default_dist_init(spec.gate)),
        "final_cost": source.get("global_cost", source.get("final_cost", "")),
        "gradient": source.get("gradient", ""),
        "init_comm_mb": source.get(
            "init_comm_mb", "0" if spec.gate == "A" else ""),
        "outer_comm_mb": source.get(
            "outer_comm_mb", format_float(total_comm)),
        "total_comm_mb": source.get("total_comm_mb", ""),
        "wall_time_sec": source.get("wall_time_sec", ""),
        "solver_time_per_node_sec": source.get("solver_time_per_node_sec", ""),
        "run_root": source.get("run_root", ""),
        "summary_csv": source.get("summary_csv", ""),
        "source_comparison_csv": str(spec.path),
    }
    for field in NORMALIZED_PASSTHROUGH_FIELDS:
      if field in source and field not in row:
        row[field] = source.get(field, "")
    rows.append(row)
  return rows


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
  return (
      row.get("gate", ""),
      row.get("label", ""),
      row.get("dataset", ""),
      row.get("updates", ""),
  )


def pairwise_rows(
    rows: list[dict[str, str]], baseline_variant: str, comm_tol: float
) -> list[dict[str, str]]:
  baselines: dict[tuple[str, str, str, str], dict[str, str]] = {}
  for row in rows:
    if row.get("method") == baseline_variant:
      baselines[key(row)] = row

  output: list[dict[str, str]] = []
  for row in rows:
    method = row.get("method", "")
    if method == baseline_variant:
      continue
    base = baselines.get(key(row))
    if base is None:
      compared = dict(row)
      compared.update({
          "baseline": baseline_variant,
          "baseline_cost": "",
          "cost_delta": "",
          "cost_better": "unknown",
          "same_or_lower_comm": "unknown",
          "same_comm_cost_win": "unknown",
      })
      output.append(compared)
      continue

    cost = parse_float(row.get("final_cost"))
    base_cost = parse_float(base.get("final_cost"))
    gradient = parse_float(row.get("gradient"))
    base_gradient = parse_float(base.get("gradient"))
    comm = parse_float(row.get("total_comm_mb"))
    base_comm = parse_float(base.get("total_comm_mb"))
    wall = parse_float(row.get("wall_time_sec"))
    base_wall = parse_float(base.get("wall_time_sec"))
    solver = parse_float(row.get("solver_time_per_node_sec"))
    base_solver = parse_float(base.get("solver_time_per_node_sec"))

    cost_delta = None if cost is None or base_cost is None else cost - base_cost
    gradient_delta = (
        None if gradient is None or base_gradient is None else
        gradient - base_gradient)
    comm_delta = None if comm is None or base_comm is None else comm - base_comm
    comm_ratio = (
        None if comm is None or base_comm in (None, 0.0) else comm / base_comm)
    wall_delta = None if wall is None or base_wall is None else wall - base_wall
    solver_delta = (
        None if solver is None or base_solver is None else solver - base_solver)
    cost_better = None if cost_delta is None else cost_delta < 0.0
    same_or_lower_comm = (
        None if comm_delta is None else comm_delta <= comm_tol)
    same_comm_cost_win = (
        None if cost_better is None or same_or_lower_comm is None else
        cost_better and same_or_lower_comm)

    compared = dict(row)
    compared.update({
        "baseline": baseline_variant,
        "baseline_cost": base.get("final_cost", ""),
        "cost_delta": format_float(cost_delta),
        "cost_better": bool_text(cost_better),
        "baseline_gradient": base.get("gradient", ""),
        "gradient_delta": format_float(gradient_delta),
        "baseline_total_comm_mb": base.get("total_comm_mb", ""),
        "comm_delta_mb": format_float(comm_delta),
        "comm_ratio": format_float(comm_ratio),
        "same_or_lower_comm": bool_text(same_or_lower_comm),
        "baseline_wall_time_sec": base.get("wall_time_sec", ""),
        "wall_time_delta_sec": format_float(wall_delta),
        "baseline_solver_time_per_node_sec":
            base.get("solver_time_per_node_sec", ""),
        "solver_time_per_node_delta_sec": format_float(solver_delta),
        "same_comm_cost_win": bool_text(same_comm_cost_win),
    })
    output.append(compared)
  return sorted(output, key=lambda r: (
      r.get("gate", ""), r.get("label", ""), r.get("method", ""),
      r.get("dataset", ""), r.get("updates", "")))


def aggregate_rows(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
  groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
  for row in pairs:
    groups.setdefault((
        row.get("gate", ""),
        row.get("label", ""),
        row.get("method", ""),
        row.get("baseline", ""),
    ), []).append(row)

  output: list[dict[str, str]] = []
  for (gate, label, method, baseline), rows in groups.items():
    cost_deltas = [
        value for value in (parse_float(row.get("cost_delta")) for row in rows)
        if value is not None
    ]
    comm_ratios = [
        value for value in (parse_float(row.get("comm_ratio")) for row in rows)
        if value is not None
    ]
    output.append({
        "gate": gate,
        "claim_scope": gate_claim_scope(gate),
        "label": label,
        "method": method,
        "baseline": baseline,
        "num_datasets": str(len(rows)),
        "cost_win_count": str(sum(
            1 for row in rows if row.get("cost_better") == "true")),
        "same_comm_cost_win_count": str(sum(
            1 for row in rows if row.get("same_comm_cost_win") == "true")),
        "mean_cost_delta": format_float(
            None if not cost_deltas else sum(cost_deltas) / len(cost_deltas)),
        "mean_comm_ratio": format_float(
            None if not comm_ratios else sum(comm_ratios) / len(comm_ratios)),
        "max_comm_ratio": format_float(
            None if not comm_ratios else max(comm_ratios)),
    })
  return sorted(output, key=lambda r: (
      r.get("gate", ""), r.get("label", ""), r.get("method", "")))


def write_summary(path: Path, aggregate: list[dict[str, str]]) -> None:
  lines = [
      "# DPGO-MM AMM Surpass Gate Summary",
      "",
      "| Gate | Label | Method | Cost wins | Same-comm wins | Mean cost delta | Mean comm ratio | Max comm ratio |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for row in aggregate:
    wins = f"{row.get('cost_win_count', '')}/{row.get('num_datasets', '')}"
    same_comm = (
        f"{row.get('same_comm_cost_win_count', '')}/"
        f"{row.get('num_datasets', '')}")
    lines.append(
        "| {gate} | {label} | {method} | {wins} | {same_comm} | "
        "`{mean_delta}` | `{mean_comm}` | `{max_comm}` |".format(
            gate=row.get("gate", ""),
            label=row.get("label", ""),
            method=row.get("method", ""),
            wins=wins,
            same_comm=same_comm,
            mean_delta=row.get("mean_cost_delta", ""),
            mean_comm=row.get("mean_comm_ratio", ""),
            max_comm=row.get("max_comm_ratio", ""),
        ))
  path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(
      description="Normalize DPGO-MM AMM surpass evidence into Gate A/B/C tables.")
  parser.add_argument(
      "--comparison",
      action="append",
      type=parse_spec,
      required=True,
      help="Comparison spec: GATE:LABEL:/path/to/comparison.csv")
  parser.add_argument("--baseline", default=BASELINE_VARIANT)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--comm-tol", type=float, default=1e-9)
  args = parser.parse_args(argv)

  args.out_dir.mkdir(parents=True, exist_ok=True)

  normalized: list[dict[str, str]] = []
  for spec in args.comparison:
    if not spec.path.is_file():
      parser.error(f"missing comparison CSV: {spec.path}")
    normalized.extend(normalize_rows(spec))

  pairwise = pairwise_rows(normalized, args.baseline, args.comm_tol)
  aggregate = aggregate_rows(pairwise)

  write_csv(args.out_dir / "normalized_gate_rows.csv", normalized)
  write_csv(args.out_dir / "pairwise_to_dpgo_mm_amm.csv", pairwise)
  write_csv(args.out_dir / "aggregate_by_method.csv", aggregate, [
      "gate",
      "claim_scope",
      "label",
      "method",
      "baseline",
      "num_datasets",
      "cost_win_count",
      "same_comm_cost_win_count",
      "mean_cost_delta",
      "mean_comm_ratio",
      "max_comm_ratio",
  ])
  write_summary(args.out_dir / "summary.md", aggregate)

  print(f"Wrote {len(normalized)} normalized rows to {args.out_dir}")
  print(f"Wrote {len(pairwise)} pairwise rows")
  print(f"Wrote {len(aggregate)} aggregate rows")
  print(f"Wrote summary: {args.out_dir / 'summary.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
