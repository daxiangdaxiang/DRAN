#!/usr/bin/env python3
"""Run two-stage DCI Schur-gap certificates on one or more g2o graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
  from scripts import analyze_distributed_chordal_init as dci  # type: ignore
except ImportError:  # pragma: no cover - used when executed as a script.
  import analyze_distributed_chordal_init as dci  # type: ignore


def _parse_dataset_token(token: str) -> tuple[str, Path]:
  if "=" in token:
    name, raw_path = token.split("=", 1)
    name = name.strip()
    if not name:
      raise ValueError(f"dataset name is empty in {token!r}")
    return name, Path(raw_path)
  path = Path(token)
  return path.stem, path


def _robot_topology_edges(num_robots: int, mode: str):
  if mode == "none":
    return []
  if mode == "chain":
    return [(robot, robot + 1) for robot in range(max(0, num_robots - 1))]
  if mode == "complete":
    return [
        (first, second)
        for first in range(num_robots)
        for second in range(first + 1, num_robots)
    ]
  raise ValueError(f"unsupported robot topology mode: {mode}")


def _float_or_none(value):
  if value is None or value == "":
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  if not math.isfinite(result):
    return None
  return result


def read_reference_handoff_costs(reference_path: Path) -> dict[str, float]:
  references: dict[str, float] = {}
  with Path(reference_path).open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
      dataset = str(row.get("dataset", ""))
      if not dataset:
        continue
      for column in [
          "reference_handoff_cost",
          "cci_handoff_cost",
          "centralized_handoff_cost",
          "optimal_cost",
          "handoff_cost",
      ]:
        value = _float_or_none(row.get(column))
        if value is not None:
          references[dataset] = value
          break
  return references


def apply_reference_gate_to_rows(
    rows: list[dict],
    reference_handoff_costs: dict[str, float],
    handoff_reference_relative_slack: float,
    projection_cost_reference_relative_slack: float):
  for row in rows:
    dataset = str(row.get("dataset", ""))
    reference = reference_handoff_costs.get(dataset)
    handoff_cost = _float_or_none(row.get("handoff_cost"))
    projection_delta = _float_or_none(row.get("rotation_projection_cost_delta"))
    projection_abs_delta = _float_or_none(
        row.get("rotation_abs_projection_cost_delta"))
    if projection_delta is not None:
      projection_increase = max(0.0, projection_delta)
    else:
      projection_increase = projection_abs_delta
    reasons = []
    max_handoff = None
    max_projection = None
    if reference is None:
      reasons.append("missing_reference_handoff_cost")
    else:
      max_handoff = float(
          reference * (1.0 + float(handoff_reference_relative_slack)))
      max_projection = float(
          reference * float(projection_cost_reference_relative_slack))
      if handoff_cost is None or handoff_cost > max_handoff:
        reasons.append("handoff_cost")
      if (projection_increase is None or
          projection_increase > max_projection):
        reasons.append("rotation_projection_cost")
    row["reference_gate_source"] = "reference_summary_relative"
    row["reference_handoff_cost"] = "" if reference is None else float(reference)
    row["reference_gate_max_handoff_cost"] = (
        "" if max_handoff is None else float(max_handoff))
    row["reference_gate_max_rotation_projection_cost_delta"] = (
        "" if max_projection is None else float(max_projection))
    row["reference_gate_rotation_projection_cost_increase"] = (
        "" if projection_increase is None else float(projection_increase))
    row["reference_gate_feasible"] = bool(not reasons)
    row["reference_gate_failure_reasons"] = ",".join(reasons) or "none"
  return rows


def build_dataset_specs(dataset_tokens: list[str],
                        num_robots: int,
                        robot_topology: str = "chain"):
  specs = []
  topology_edges = _robot_topology_edges(num_robots, robot_topology)
  for token in dataset_tokens:
    dataset, graph_path = _parse_dataset_token(token)
    vertices, graph_edges = dci.parse_g2o_graph(graph_path)
    pose_ids = dci.graph_pose_ids(vertices, graph_edges)
    robot_of, ranges = dci.build_contiguous_robot_map(pose_ids, num_robots)
    specs.append({
        "dataset": dataset,
        "graph_path": str(graph_path),
        "graph_edges": graph_edges,
        "pose_ids": pose_ids,
        "robot_of": robot_of,
        "robot_ranges": ranges,
        "robot_topology_edges": topology_edges,
    })
  return specs


def dense_preflight_estimate(spec: dict) -> dict:
  dim = dci.pose_dimension_from_edges(spec["graph_edges"])
  variable_pose_count = max(0, len(spec["pose_ids"]) - 1)
  rotation_variables = variable_pose_count * dim * dim
  translation_variables = variable_pose_count * dim
  max_stage_variables = max(rotation_variables, translation_variables)
  local_system_count = len(set(spec["robot_of"].values()))
  dense_local_normal_bytes = (
      local_system_count *
      (rotation_variables * rotation_variables +
       translation_variables * translation_variables) *
      8)
  return {
      "dimension": int(dim),
      "pose_count": int(len(spec["pose_ids"])),
      "robot_count": int(local_system_count),
      "rotation_variables": int(rotation_variables),
      "translation_variables": int(translation_variables),
      "max_stage_variables": int(max_stage_variables),
      "estimated_dense_local_normal_bytes": int(dense_local_normal_bytes),
      "estimated_dense_local_normal_mb": (
          float(dense_local_normal_bytes) / float(1024 * 1024)),
  }


def dense_preflight_required(hessian_storage: str, certificate_mode: str) -> bool:
  """Returns whether the current run can still materialize dense diagnostics."""
  if hessian_storage == "dense":
    return True
  if certificate_mode == "dense":
    return True
  return False


def skipped_preflight_row(spec: dict, estimate: dict,
                          max_dense_variables: int,
                          hessian_storage: str = "dense",
                          certificate_mode: str = "dense") -> dict:
  return {
      "dataset": str(spec["dataset"]),
      "solver": "skipped",
      "status": "skipped_preflight",
      "skip_reason": "max_dense_variables_exceeded",
      "hessian_storage": hessian_storage,
      "certificate_mode": certificate_mode,
      "pose_count": int(estimate["pose_count"]),
      "dimension": int(estimate["dimension"]),
      "robot_count": int(estimate["robot_count"]),
      "max_stage_variables": int(estimate["max_stage_variables"]),
      "max_dense_variables": int(max_dense_variables),
      "estimated_dense_local_normal_bytes": int(
          estimate["estimated_dense_local_normal_bytes"]),
      "estimated_dense_local_normal_mb": float(
          estimate["estimated_dense_local_normal_mb"]),
      "rotation_schur_energy_gap": "",
      "translation_schur_energy_gap": "",
      "total_schur_energy_gap": "",
      "rotation_schur_residual_norm": "",
      "translation_schur_residual_norm": "",
      "total_schur_residual_norm": "",
      "handoff_cost": "",
      "all_edges_evaluated": "",
  }


def _write_mixed_rows(output_dir: Path, rows: list[dict], report: dict):
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / "two_stage_dci_certificate_summary.csv"
  if rows:
    fieldnames = []
    for row in rows:
      for key in row:
        if key not in fieldnames:
          fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)
  else:
    csv_path.write_text("", encoding="utf-8")
  (output_dir / "two_stage_dci_certificate_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description="Run two-stage DCI Schur-gap certificate sweeps.")
  parser.add_argument(
      "--dataset",
      action="append",
      required=True,
      help="Dataset graph, either NAME=PATH or PATH. Can be repeated.")
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--solver",
      choices=["reference", "pcg", "limited_hop_pcg", "fixed_step"],
      default="pcg")
  parser.add_argument("--rotation-iterations", type=int, default=100)
  parser.add_argument("--translation-iterations", type=int, default=100)
  parser.add_argument("--tolerance", type=float, default=1e-10)
  parser.add_argument("--damping", type=float, default=0.0)
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo")
  parser.add_argument(
      "--edge-owner-policy",
      choices=["lower_robot", "first_endpoint_robot", "second_endpoint_robot"],
      default="lower_robot")
  parser.add_argument(
      "--robot-topology",
      choices=["chain", "complete", "none"],
      default="chain")
  parser.add_argument("--hop-radius", type=int, default=1)
  parser.add_argument(
      "--communication-model",
      choices=["separator_owner_star", "separator_tree"],
      default="separator_owner_star")
  parser.add_argument(
      "--schur-preconditioner",
      choices=["none", "diagonal", "boundary_diagonal",
               "dirichlet_clipped_diagonal"],
      default="none")
  parser.add_argument(
      "--fixed-step-acceleration",
      choices=[
          "none",
          "chebyshev",
          "chebyshev_auto_gershgorin",
          "chebyshev_graph_normalized",
          "chebyshev_ritz_probe",
          "chebyshev_block_gershgorin_certificate",
          "chebyshev_block_gershgorin_budgeted_certificate",
      ],
      default="none")
  parser.add_argument("--chebyshev-lambda-min", type=float, default=None)
  parser.add_argument("--chebyshev-lambda-max", type=float, default=None)
  parser.add_argument("--dirichlet-diagonal-floor", type=float, default=0.25)
  parser.add_argument(
      "--chebyshev-safety-monitor",
      choices=["none", "residual_growth", "local_residual_envelope"],
      default="none")
  parser.add_argument(
      "--chebyshev-safety-growth-factor", type=float, default=10.0)
  parser.add_argument("--chebyshev-ritz-probe-iterations", type=int, default=8)
  parser.add_argument("--chebyshev-ritz-seed", type=int, default=0)
  parser.add_argument("--chebyshev-ritz-safety-factor", type=float, default=1.05)
  parser.add_argument("--chebyshev-certificate-iterations", type=int, default=128)
  parser.add_argument(
      "--chebyshev-certificate-vector-mode",
      choices=["lazy_power", "resolvent", "cg_resolvent"],
      default="cg_resolvent")
  parser.add_argument(
      "--chebyshev-certificate-resolvent-shift", type=float, default=0.0)
  parser.add_argument(
      "--chebyshev-certificate-max-payload-mb", type=float, default=None)
  parser.add_argument(
      "--chebyshev-certificate-preflight-hard-cap-mb",
      type=float,
      default=None)
  parser.add_argument("--reference-summary", type=Path, default=None)
  parser.add_argument(
      "--handoff-reference-relative-slack", type=float, default=0.0)
  parser.add_argument(
      "--projection-cost-reference-relative-slack", type=float, default=0.0)
  parser.add_argument(
      "--hessian-storage",
      choices=["dense", "sparse"],
      default="dense",
      help=(
          "Storage used for robot-local normal Hessians. Sparse reduces the "
          "solver-path dense memory footprint; exact certificates may still "
          "use dense Schur diagnostics."))
  parser.add_argument(
      "--certificate-mode",
      choices=["dense", "matrix_free_dual"],
      default="dense",
      help=(
          "Certificate backend. dense reports exact reference quantities by "
          "assembling diagnostic Schur blocks; matrix_free_dual estimates "
          "r^T S^{-1} r with a second local-Schur PCG solve."))
  parser.add_argument(
      "--max-dense-variables",
      type=int,
      default=20000,
      help=(
          "Skip datasets whose rotation/translation dense stage variable count "
          "exceeds this gate. Use 0 to disable the gate."))
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  dataset_specs = build_dataset_specs(
      dataset_tokens=args.dataset,
      num_robots=args.num_robots,
      robot_topology=args.robot_topology)
  max_dense_variables = int(args.max_dense_variables)
  preflight_active = dense_preflight_required(
      args.hessian_storage, args.certificate_mode)
  runnable_specs = []
  rows = []
  full_stats = []
  skipped = []
  reference_handoff_costs = {}
  if args.reference_summary is not None:
    reference_handoff_costs = read_reference_handoff_costs(
        args.reference_summary)
  for spec in dataset_specs:
    estimate = dense_preflight_estimate(spec)
    if (preflight_active and
        max_dense_variables > 0 and
        int(estimate["max_stage_variables"]) > max_dense_variables):
      row = skipped_preflight_row(
          spec,
          estimate,
          max_dense_variables,
          hessian_storage=args.hessian_storage,
          certificate_mode=args.certificate_mode)
      rows.append(row)
      skipped.append({
          "dataset": spec["dataset"],
          "graph_path": spec["graph_path"],
          "preflight": estimate,
          "row": row,
      })
    else:
      runnable_specs.append(spec)
  if runnable_specs:
    partial_dir = args.output_dir
    partial_report = dci.write_two_stage_dci_certificate_sweep(
        output_dir=partial_dir,
        dataset_specs=runnable_specs,
        weighted=args.weighted,
        cost_mode=args.cost_mode,
        solver=args.solver,
        rotation_iterations=args.rotation_iterations,
        translation_iterations=args.translation_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        edge_owner_policy=args.edge_owner_policy,
        hop_radius=args.hop_radius,
        communication_model=args.communication_model,
        schur_preconditioner=args.schur_preconditioner,
        hessian_storage=args.hessian_storage,
        certificate_mode=args.certificate_mode,
        fixed_step_acceleration=args.fixed_step_acceleration,
        chebyshev_lambda_min=args.chebyshev_lambda_min,
        chebyshev_lambda_max=args.chebyshev_lambda_max,
        dirichlet_diagonal_floor=args.dirichlet_diagonal_floor,
        chebyshev_safety_monitor=args.chebyshev_safety_monitor,
        chebyshev_safety_growth_factor=args.chebyshev_safety_growth_factor,
        chebyshev_ritz_probe_iterations=args.chebyshev_ritz_probe_iterations,
        chebyshev_ritz_seed=args.chebyshev_ritz_seed,
        chebyshev_ritz_safety_factor=args.chebyshev_ritz_safety_factor,
        chebyshev_certificate_iterations=args.chebyshev_certificate_iterations,
        chebyshev_certificate_vector_mode=(
            args.chebyshev_certificate_vector_mode),
        chebyshev_certificate_resolvent_shift=(
            args.chebyshev_certificate_resolvent_shift),
        chebyshev_certificate_max_payload_mb=(
            args.chebyshev_certificate_max_payload_mb),
        chebyshev_certificate_preflight_hard_cap_mb=(
            args.chebyshev_certificate_preflight_hard_cap_mb))
    partial_rows = list(partial_report["rows"])
    if args.reference_summary is not None:
      apply_reference_gate_to_rows(
          rows=partial_rows,
          reference_handoff_costs=reference_handoff_costs,
          handoff_reference_relative_slack=(
              args.handoff_reference_relative_slack),
          projection_cost_reference_relative_slack=(
              args.projection_cost_reference_relative_slack))
    rows.extend(partial_rows)
    full_stats.extend(partial_report["full_stats"])
  if args.reference_summary is not None and skipped:
    apply_reference_gate_to_rows(
        rows=rows,
        reference_handoff_costs=reference_handoff_costs,
        handoff_reference_relative_slack=(
            args.handoff_reference_relative_slack),
        projection_cost_reference_relative_slack=(
            args.projection_cost_reference_relative_slack))
  report = {
      "model": "two_stage_dci_certificate_sweep",
      "row_count": int(len(rows)),
      "weighted": bool(args.weighted),
      "cost_mode": args.cost_mode,
      "solver": args.solver,
      "rotation_iterations": int(args.rotation_iterations),
      "translation_iterations": int(args.translation_iterations),
      "tolerance": float(args.tolerance),
      "damping": float(args.damping),
      "edge_owner_policy": args.edge_owner_policy,
      "hop_radius": int(max(0, int(args.hop_radius))),
      "hessian_storage": args.hessian_storage,
      "certificate_mode": args.certificate_mode,
      "reference_gate_source": (
          "reference_summary_relative"
          if args.reference_summary is not None else "none"),
      "reference_summary_path": (
          None if args.reference_summary is None
          else str(args.reference_summary)),
      "reference_handoff_costs": reference_handoff_costs,
      "handoff_reference_relative_slack": float(
          args.handoff_reference_relative_slack),
      "projection_cost_reference_relative_slack": float(
          args.projection_cost_reference_relative_slack),
      "max_dense_variables": int(max_dense_variables),
      "dense_preflight_active": bool(preflight_active),
      "skipped_count": int(len(skipped)),
      "rows": rows,
      "full_stats": full_stats,
      "skipped": skipped,
  }
  _write_mixed_rows(args.output_dir, rows, report)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
