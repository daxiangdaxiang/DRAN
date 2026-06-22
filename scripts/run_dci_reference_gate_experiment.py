#!/usr/bin/env python3
"""Run CCI-reference-gated DCI initialization experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

try:
  from scripts import analyze_dci_stage_certificate_stop as cert_stop  # type: ignore
  from scripts import analyze_distributed_chordal_init as dci  # type: ignore
  from scripts import run_two_stage_dci_certificate_sweep as sweep  # type: ignore
  from scripts import write_cci_reference_summary as cci_ref  # type: ignore
except ImportError:  # pragma: no cover - used when executed as a script.
  import analyze_dci_stage_certificate_stop as cert_stop  # type: ignore
  import analyze_distributed_chordal_init as dci  # type: ignore
  import run_two_stage_dci_certificate_sweep as sweep  # type: ignore
  import write_cci_reference_summary as cci_ref  # type: ignore


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Generate CCI references, run matrix-free DCI stage-budget rows, "
          "apply reference-relative certificate gates, and write a compact "
          "basin-entry summary."))
  parser.add_argument(
      "--dataset",
      action="append",
      required=True,
      help="Dataset graph, either NAME=PATH or PATH. Can be repeated.")
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--rotation-budget",
      action="append",
      type=int,
      required=True,
      help="Rotation-stage solver iteration budget. Can be repeated.")
  parser.add_argument(
      "--translation-budget",
      action="append",
      type=int,
      required=True,
      help="Translation-stage solver iteration budget. Can be repeated.")
  parser.add_argument(
      "--solver",
      choices=["pcg", "limited_hop_pcg", "fixed_step"],
      default="pcg")
  parser.add_argument("--tolerance", type=float, default=1e-6)
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
  parser.add_argument(
      "--max-rotation-residual", type=float, default=None)
  parser.add_argument(
      "--max-translation-residual", type=float, default=None)
  parser.add_argument(
      "--max-rotation-projection-correction", type=float, default=None)
  parser.add_argument(
      "--handoff-reference-relative-slack", type=float, default=0.1)
  parser.add_argument(
      "--projection-cost-reference-relative-slack", type=float, default=0.1)
  parser.add_argument(
      "--checkpoint-rows",
      action="store_true",
      help=(
          "Write each completed stage-budget row to stage_budget_rows.jsonl "
          "and refresh the partial CSV after every row."))
  parser.add_argument(
      "--resume",
      action="store_true",
      help=(
          "When used with --checkpoint-rows, reuse completed stage-budget "
          "rows from stage_budget_rows.jsonl instead of recomputing them."))
  parser.add_argument(
      "--adaptive-reference-gate",
      action="store_true",
      help=(
          "Evaluate budget rows in the supplied order and stop each dataset "
          "after the first row satisfying the CCI-relative reference gate. "
          "This is a diagnostic mode; it still uses the centralized CCI "
          "reference generated by this runner."))
  parser.add_argument(
      "--adaptive-proxy-gate",
      action="store_true",
      help=(
          "Evaluate budget rows in the supplied order and stop each dataset "
          "after the first row satisfying only the configured reference-free "
          "proxy thresholds."))
  parser.add_argument("--proxy-max-rotation-residual", type=float, default=None)
  parser.add_argument(
      "--proxy-max-translation-residual", type=float, default=None)
  parser.add_argument(
      "--proxy-max-rotation-schur-energy-gap", type=float, default=None)
  parser.add_argument(
      "--proxy-max-translation-schur-energy-gap", type=float, default=None)
  parser.add_argument(
      "--proxy-max-rotation-schur-energy-ratio", type=float, default=None)
  parser.add_argument(
      "--proxy-max-translation-schur-energy-ratio", type=float, default=None)
  parser.add_argument(
      "--proxy-max-rotation-projection-correction",
      type=float,
      default=None)
  parser.add_argument(
      "--proxy-max-rotation-projection-cost-increase",
      type=float,
      default=None)
  parser.add_argument(
      "--write-selected-initialization",
      action="store_true",
      help=(
          "Recompute and write each selected DCI initialization pose set under "
          "selected_initializations/. Each selected row writes both the manual "
          "dense matrix pose-set format consumed by downstream diagnostics and "
          "a standard g2o vertex-only pose file for independent cost audits."))
  return parser.parse_args(argv)


def _finite(value) -> bool:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return False
  return result == result and math.isfinite(result)


def _write_csv(path: Path, rows: list[dict]):
  if rows:
    fieldnames = []
    for row in rows:
      for key in row:
        if key not in fieldnames:
          fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)
  else:
    path.write_text("", encoding="utf-8")


def _stage_checkpoint_key(args, dataset: str, rotation_budget: int,
                          translation_budget: int) -> str:
  """Key identifying a stage row under the current solver configuration."""
  key_parts = [
      str(dataset),
      f"rotation={int(rotation_budget)}",
      f"translation={int(translation_budget)}",
      f"solver={args.solver}",
      f"weighted={bool(args.weighted)}",
      f"cost_mode={args.cost_mode}",
      f"damping={float(args.damping)}",
      f"edge_owner={args.edge_owner_policy}",
      f"topology={args.robot_topology}",
      f"hop={int(max(0, int(args.hop_radius)))}",
      f"communication={args.communication_model}",
      f"preconditioner={args.schur_preconditioner}",
      f"fixed_step={args.fixed_step_acceleration}",
      f"lambda_min={args.chebyshev_lambda_min}",
      f"lambda_max={args.chebyshev_lambda_max}",
      f"dirichlet_floor={float(args.dirichlet_diagonal_floor)}",
      f"safety={args.chebyshev_safety_monitor}",
      f"safety_growth={float(args.chebyshev_safety_growth_factor)}",
      f"ritz_iters={int(args.chebyshev_ritz_probe_iterations)}",
      f"ritz_seed={int(args.chebyshev_ritz_seed)}",
      f"ritz_safety={float(args.chebyshev_ritz_safety_factor)}",
      f"cert_iters={int(args.chebyshev_certificate_iterations)}",
      f"cert_vector={args.chebyshev_certificate_vector_mode}",
      f"cert_shift={float(args.chebyshev_certificate_resolvent_shift)}",
      f"cert_budget={args.chebyshev_certificate_max_payload_mb}",
      f"cert_hard_cap={args.chebyshev_certificate_preflight_hard_cap_mb}",
  ]
  return "|".join(key_parts)


def _load_stage_checkpoints(checkpoint_path: Path) -> dict[str, dict]:
  checkpoints: dict[str, dict] = {}
  if not checkpoint_path.exists():
    return checkpoints
  with checkpoint_path.open(encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
      line = raw_line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid checkpoint JSON on line {line_number}: {exc}") from exc
      key = str(record.get("key", ""))
      if not key:
        raise ValueError(
            f"checkpoint line {line_number} is missing a nonempty key")
      checkpoints[key] = record
  return checkpoints


def _append_stage_checkpoint(checkpoint_path: Path, key: str, row: dict,
                             full_stats_entry: dict):
  checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
  record = {
      "key": key,
      "row": row,
      "full_stats": full_stats_entry,
  }
  with checkpoint_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def _reference_gates(args) -> dict:
  return {
      "max_rotation_residual": args.max_rotation_residual,
      "max_translation_residual": args.max_translation_residual,
      "max_rotation_projection_correction":
          args.max_rotation_projection_correction,
      "max_rotation_projection_cost_delta": None,
      "max_handoff_cost": None,
  }


def _reference_slacks(args) -> dict:
  return {
      "rotation_residual_slack": 0.0,
      "translation_residual_slack": 0.0,
      "rotation_projection_correction_slack": 0.0,
      "rotation_projection_cost_delta_slack": 0.0,
      "handoff_cost_slack": 0.0,
      "handoff_reference_relative_slack": float(
          args.handoff_reference_relative_slack),
      "projection_cost_reference_relative_slack": float(
          args.projection_cost_reference_relative_slack),
      "schur_preconditioner": args.schur_preconditioner,
  }


def _proxy_gate_thresholds(args) -> dict:
  return {
      "max_rotation_residual": args.proxy_max_rotation_residual,
      "max_translation_residual": args.proxy_max_translation_residual,
      "max_rotation_schur_energy_gap":
          args.proxy_max_rotation_schur_energy_gap,
      "max_translation_schur_energy_gap":
          args.proxy_max_translation_schur_energy_gap,
      "max_rotation_schur_energy_ratio": getattr(
          args, "proxy_max_rotation_schur_energy_ratio", None),
      "max_translation_schur_energy_ratio": getattr(
          args, "proxy_max_translation_schur_energy_ratio", None),
      "max_rotation_projection_correction":
          args.proxy_max_rotation_projection_correction,
      "max_rotation_projection_cost_increase":
          args.proxy_max_rotation_projection_cost_increase,
  }


def _row_float(row: dict, key: str) -> float:
  try:
    value = float(row.get(key, ""))
  except (TypeError, ValueError):
    return float("nan")
  if math.isfinite(value):
    return value
  return float("nan")


def _row_passes_reference_gate(row: dict, args,
                               reference_costs: dict[str, float]) -> bool:
  report = cert_stop.analyze_rows(
      [row],
      _reference_gates(args),
      gate_source="reference_summary_relative",
      slacks=_reference_slacks(args),
      reference_handoff_costs=reference_costs)
  return bool(report["datasets"][0]["feasible"])


def _row_passes_proxy_gate(row: dict, args) -> bool:
  thresholds = _proxy_gate_thresholds(args)
  if all(value is None for value in thresholds.values()):
    raise ValueError(
        "--adaptive-proxy-gate requires at least one proxy threshold")
  checks = [
      ("max_rotation_residual", "rotation_schur_residual_norm"),
      ("max_translation_residual", "translation_schur_residual_norm"),
      ("max_rotation_schur_energy_gap", "rotation_schur_energy_gap"),
      ("max_translation_schur_energy_gap", "translation_schur_energy_gap"),
      ("max_rotation_projection_correction",
       "rotation_max_projection_correction_norm"),
  ]
  for threshold_key, row_key in checks:
    threshold = thresholds[threshold_key]
    if threshold is None:
      continue
    value = _row_float(row, row_key)
    if not (math.isfinite(value) and value <= float(threshold)):
      return False
  ratio_checks = [
      ("max_rotation_schur_energy_ratio", "rotation_schur_energy_gap"),
      ("max_translation_schur_energy_ratio", "translation_schur_energy_gap"),
  ]
  for threshold_key, numerator_key in ratio_checks:
    threshold = thresholds[threshold_key]
    if threshold is None:
      continue
    numerator = _row_float(row, numerator_key)
    denominator = _row_float(row, "handoff_cost")
    if not (math.isfinite(numerator) and math.isfinite(denominator) and
            denominator > 0.0 and
            numerator / denominator <= float(threshold)):
      return False
  projection_threshold = thresholds[
      "max_rotation_projection_cost_increase"]
  if projection_threshold is not None:
    projection_delta = _row_float(row, "rotation_projection_cost_delta")
    projection_abs_delta = _row_float(
        row, "rotation_abs_projection_cost_delta")
    if math.isfinite(projection_delta):
      projection_increase = max(0.0, projection_delta)
    else:
      projection_increase = projection_abs_delta
    if not (math.isfinite(projection_increase) and
            projection_increase <= float(projection_threshold)):
      return False
  return True


def _row_triggers_adaptive_stop(
    row: dict,
    args,
    reference_costs: dict[str, float]) -> bool:
  active_gates = []
  if args.adaptive_reference_gate:
    active_gates.append(_row_passes_reference_gate(
        row, args, reference_costs))
  if args.adaptive_proxy_gate:
    active_gates.append(_row_passes_proxy_gate(row, args))
  return bool(active_gates and all(active_gates))


def _build_stage_rows(args, dataset_specs: list[dict],
                      stage_dir: Path | None = None,
                      reference_costs: dict[str, float] | None = None):
  rows = []
  full_stats = []
  computed_row_count = 0
  resumed_row_count = 0
  adaptive_stopped_dataset_count = 0
  reference_costs = reference_costs or {}
  checkpoint_path = None
  checkpoints: dict[str, dict] = {}
  if args.checkpoint_rows:
    if stage_dir is None:
      raise ValueError("stage_dir is required when checkpointing rows")
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = stage_dir / "stage_budget_rows.jsonl"
    if args.resume:
      checkpoints = _load_stage_checkpoints(checkpoint_path)
    elif checkpoint_path.exists():
      checkpoint_path.write_text("", encoding="utf-8")
  for spec in dataset_specs:
    dataset_stopped = False
    for rotation_budget in [int(value) for value in args.rotation_budget]:
      if dataset_stopped:
        break
      for translation_budget in [int(value) for value in args.translation_budget]:
        if dataset_stopped:
          break
        key = _stage_checkpoint_key(
            args,
            dataset=str(spec["dataset"]),
            rotation_budget=int(rotation_budget),
            translation_budget=int(translation_budget))
        if checkpoint_path is not None and key in checkpoints:
          record = checkpoints[key]
          row = dict(record["row"])
          rows.append(row)
          full_stats.append(dict(record["full_stats"]))
          resumed_row_count += 1
          if _row_triggers_adaptive_stop(row, args, reference_costs):
            adaptive_stopped_dataset_count += 1
            dataset_stopped = True
          continue
        start = time.perf_counter()
        row = dci.two_stage_dci_certificate_sweep_row(
            dataset=str(spec["dataset"]),
            graph_edges=spec["graph_edges"],
            pose_ids=spec["pose_ids"],
            robot_of=spec["robot_of"],
            weighted=args.weighted,
            cost_mode=args.cost_mode,
            anchor_pose=spec.get("anchor_pose"),
            solver=args.solver,
            rotation_iterations=int(rotation_budget),
            translation_iterations=int(translation_budget),
            tolerance=args.tolerance,
            damping=args.damping,
            edge_owner_policy=args.edge_owner_policy,
            robot_topology_edges=spec.get("robot_topology_edges"),
            hop_radius=spec.get("hop_radius", args.hop_radius),
            communication_model=args.communication_model,
            schur_preconditioner=args.schur_preconditioner,
            hessian_storage="sparse",
            certificate_mode="matrix_free_dual",
            fixed_step_acceleration=args.fixed_step_acceleration,
            chebyshev_lambda_min=args.chebyshev_lambda_min,
            chebyshev_lambda_max=args.chebyshev_lambda_max,
            dirichlet_diagonal_floor=args.dirichlet_diagonal_floor,
            chebyshev_safety_monitor=args.chebyshev_safety_monitor,
            chebyshev_safety_growth_factor=(
                args.chebyshev_safety_growth_factor),
            chebyshev_ritz_probe_iterations=(
                args.chebyshev_ritz_probe_iterations),
            chebyshev_ritz_seed=args.chebyshev_ritz_seed,
            chebyshev_ritz_safety_factor=args.chebyshev_ritz_safety_factor,
            chebyshev_certificate_iterations=(
                args.chebyshev_certificate_iterations),
            chebyshev_certificate_vector_mode=(
                args.chebyshev_certificate_vector_mode),
            chebyshev_certificate_resolvent_shift=(
                args.chebyshev_certificate_resolvent_shift),
            chebyshev_certificate_max_payload_mb=(
                args.chebyshev_certificate_max_payload_mb),
            chebyshev_certificate_preflight_hard_cap_mb=(
                args.chebyshev_certificate_preflight_hard_cap_mb))
        wall_time_sec = time.perf_counter() - start
        stats = row.pop("stats")
        row.update({
            "rotation_iteration_budget": int(rotation_budget),
            "translation_iteration_budget": int(translation_budget),
            "total_iteration_budget": int(rotation_budget + translation_budget),
            "stage_budget_pair": f"{rotation_budget}:{translation_budget}",
            "wall_time_sec": float(wall_time_sec),
            "hessian_storage": "sparse",
            "certificate_mode": "matrix_free_dual",
            "dense_preflight_active": False,
        })
        rows.append(row)
        full_stats.append({
            "dataset": row["dataset"],
            "rotation_iteration_budget": int(rotation_budget),
            "translation_iteration_budget": int(translation_budget),
            "wall_time_sec": float(wall_time_sec),
            "stats": stats,
        })
        computed_row_count += 1
        if checkpoint_path is not None:
          _append_stage_checkpoint(
              checkpoint_path,
              key,
              row,
              full_stats[-1])
          _write_csv(stage_dir / "stage_budget_summary.csv", rows)
        if _row_triggers_adaptive_stop(row, args, reference_costs):
          adaptive_stopped_dataset_count += 1
          dataset_stopped = True
  return rows, full_stats, {
      "computed_row_count": int(computed_row_count),
      "resumed_row_count": int(resumed_row_count),
      "adaptive_stopped_dataset_count": int(
          adaptive_stopped_dataset_count),
      "checkpoint_path": (
          None if checkpoint_path is None else str(checkpoint_path)),
  }


def _write_stage_outputs(output_dir: Path, args, rows: list[dict],
                         full_stats: list[dict], dataset_specs: list[dict]):
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / "stage_budget_summary.csv", rows)
  report = {
      "model": "dci_stage_budget_sweep",
      "row_count": int(len(rows)),
      "dataset_count": int(len(dataset_specs)),
      "rotation_iteration_budgets": [int(value) for value in args.rotation_budget],
      "translation_iteration_budgets": [
          int(value) for value in args.translation_budget
      ],
      "weighted": bool(args.weighted),
      "cost_mode": args.cost_mode,
      "edge_owner_policy": args.edge_owner_policy,
      "hop_radius": int(max(0, int(args.hop_radius))),
      "communication_model": args.communication_model,
      "solver": args.solver,
      "checkpoint_rows": bool(args.checkpoint_rows),
      "resume": bool(args.resume),
      "adaptive_reference_gate": bool(args.adaptive_reference_gate),
      "adaptive_proxy_gate": bool(args.adaptive_proxy_gate),
      "proxy_gate_thresholds": _proxy_gate_thresholds(args),
      "schur_preconditioner": args.schur_preconditioner,
      "hessian_storage": "sparse",
      "certificate_mode": "matrix_free_dual",
      "fixed_step_acceleration": args.fixed_step_acceleration,
      "chebyshev_certificate_max_payload_mb": (
          None if args.chebyshev_certificate_max_payload_mb is None
          else float(args.chebyshev_certificate_max_payload_mb)),
      "chebyshev_certificate_preflight_hard_cap_mb": (
          None if args.chebyshev_certificate_preflight_hard_cap_mb is None
          else float(args.chebyshev_certificate_preflight_hard_cap_mb)),
      "rows": rows,
      "full_stats": full_stats,
  }
  (output_dir / "stage_budget_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def _compact_summary_rows(gate_report: dict) -> list[dict]:
  summary_rows = []
  for dataset_report in gate_report["datasets"]:
    selected = dataset_report.get("selected_row")
    reference = dataset_report.get("reference_handoff_cost")
    row = {
        "dataset": dataset_report["dataset"],
        "feasible": bool(dataset_report["feasible"]),
        "feasible_count": int(dataset_report["feasible_count"]),
        "row_count": int(dataset_report["row_count"]),
        "selected_pair": dataset_report.get("selected_pair"),
        "reference_handoff_cost": reference,
        "selected_handoff_cost": None,
        "selected_handoff_gap_to_reference": None,
        "selected_total_comm_mb": None,
        "selected_rotation_residual": None,
        "selected_translation_residual": None,
        "selected_projection_cost_increase": None,
        "selected_rotation_solver_model": None,
        "selected_translation_solver_model": None,
        "selected_rotation_preconditioner": None,
        "selected_translation_preconditioner": None,
        "selected_rotation_fixed_step_acceleration": None,
        "selected_translation_fixed_step_acceleration": None,
        "selected_rotation_certificate_policy_decision": None,
        "selected_translation_certificate_policy_decision": None,
        "failure_counts": json.dumps(
            dataset_report.get("failure_counts", {}), sort_keys=True),
    }
    if selected is not None:
      selected_cost = selected.get("handoff_cost")
      row.update({
          "selected_handoff_cost": selected_cost,
          "selected_total_comm_mb": selected.get("total_comm_mb"),
          "selected_rotation_residual":
              selected.get("rotation_schur_residual_norm"),
          "selected_translation_residual":
              selected.get("translation_schur_residual_norm"),
          "selected_projection_cost_increase":
              selected.get("rotation_projection_cost_increase"),
          "selected_rotation_solver_model":
              selected.get("rotation_solver_model"),
          "selected_translation_solver_model":
              selected.get("translation_solver_model"),
          "selected_rotation_preconditioner":
              selected.get("rotation_schur_preconditioner"),
          "selected_translation_preconditioner":
              selected.get("translation_schur_preconditioner"),
          "selected_rotation_fixed_step_acceleration":
              selected.get("rotation_fixed_step_acceleration"),
          "selected_translation_fixed_step_acceleration":
              selected.get("translation_fixed_step_acceleration"),
          "selected_rotation_certificate_policy_decision":
              selected.get("rotation_chebyshev_certificate_policy_decision"),
          "selected_translation_certificate_policy_decision":
              selected.get(
                  "translation_chebyshev_certificate_policy_decision"),
      })
      if _finite(selected_cost) and _finite(reference):
        row["selected_handoff_gap_to_reference"] = (
            float(selected_cost) - float(reference))
    summary_rows.append(row)
  return summary_rows


def _safe_dataset_filename(dataset: str) -> str:
  safe = "".join(
      ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
      for ch in str(dataset))
  return safe or "dataset"


def _rotation_matrix_to_quaternion(rotation) -> tuple[float, float, float, float]:
  trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
  if trace > 0.0:
    scale = math.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * scale
    qx = (rotation[2, 1] - rotation[1, 2]) / scale
    qy = (rotation[0, 2] - rotation[2, 0]) / scale
    qz = (rotation[1, 0] - rotation[0, 1]) / scale
  elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
    scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] -
                      rotation[2, 2]) * 2.0
    qw = (rotation[2, 1] - rotation[1, 2]) / scale
    qx = 0.25 * scale
    qy = (rotation[0, 1] + rotation[1, 0]) / scale
    qz = (rotation[0, 2] + rotation[2, 0]) / scale
  elif rotation[1, 1] > rotation[2, 2]:
    scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] -
                      rotation[2, 2]) * 2.0
    qw = (rotation[0, 2] - rotation[2, 0]) / scale
    qx = (rotation[0, 1] + rotation[1, 0]) / scale
    qy = 0.25 * scale
    qz = (rotation[1, 2] + rotation[2, 1]) / scale
  else:
    scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] -
                      rotation[1, 1]) * 2.0
    qw = (rotation[1, 0] - rotation[0, 1]) / scale
    qx = (rotation[0, 2] + rotation[2, 0]) / scale
    qy = (rotation[1, 2] + rotation[2, 1]) / scale
    qz = 0.25 * scale
  norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
  if norm <= 0.0 or not math.isfinite(norm):
    return 0.0, 0.0, 0.0, 1.0
  return qx / norm, qy / norm, qz / norm, qw / norm


def _write_g2o_pose_set(path: Path, poses: dict[int, object], dim: int):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for pose_id in sorted(poses):
      pose = poses[pose_id]
      if dim == 2:
        theta = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
        handle.write(
            "VERTEX_SE2 "
            f"{pose_id} {pose[0, 3]:.17g} {pose[1, 3]:.17g} "
            f"{theta:.17g}\n")
      else:
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(pose[:3, :3])
        handle.write(
            "VERTEX_SE3:QUAT "
            f"{pose_id} {pose[0, 3]:.17g} {pose[1, 3]:.17g} "
            f"{pose[2, 3]:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
            f"{qw:.17g}\n")


def _write_selected_initializations(
    output_dir: Path,
    args,
    gate_report: dict,
    dataset_specs: list[dict]) -> list[dict]:
  output_dir.mkdir(parents=True, exist_ok=True)
  specs_by_dataset = {
      str(spec["dataset"]): spec
      for spec in dataset_specs
  }
  rows = []
  for dataset_report in gate_report["datasets"]:
    selected = dataset_report.get("selected_row")
    if selected is None:
      continue
    dataset = str(dataset_report["dataset"])
    spec = specs_by_dataset.get(dataset)
    if spec is None:
      continue
    rotation_budget = int(selected["rotation_iteration_budget"])
    translation_budget = int(selected["translation_iteration_budget"])
    poses, stats = dci.two_stage_graph_local_chordal_gap_decomposition(
        graph_edges=spec["graph_edges"],
        pose_ids=spec["pose_ids"],
        robot_of=spec["robot_of"],
        weighted=args.weighted,
        cost_mode=args.cost_mode,
        anchor_pose=spec.get("anchor_pose"),
        solver=args.solver,
        rotation_iterations=rotation_budget,
        translation_iterations=translation_budget,
        tolerance=args.tolerance,
        damping=args.damping,
        edge_owner_policy=args.edge_owner_policy,
        robot_topology_edges=spec.get("robot_topology_edges"),
        hop_radius=spec.get("hop_radius", args.hop_radius),
        communication_model=args.communication_model,
        schur_preconditioner=args.schur_preconditioner,
        hessian_storage="sparse",
        certificate_mode="matrix_free_dual",
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
        chebyshev_certificate_vector_mode=args.chebyshev_certificate_vector_mode,
        chebyshev_certificate_resolvent_shift=(
            args.chebyshev_certificate_resolvent_shift),
        chebyshev_certificate_max_payload_mb=(
            args.chebyshev_certificate_max_payload_mb),
        chebyshev_certificate_preflight_hard_cap_mb=(
            args.chebyshev_certificate_preflight_hard_cap_mb))
    pose_path = output_dir / f"{_safe_dataset_filename(dataset)}_init.txt"
    dci.write_manual_matrix_pose_set(
        pose_path, poses, int(stats["dimension"]))
    g2o_pose_path = output_dir / f"{_safe_dataset_filename(dataset)}_init.g2o"
    _write_g2o_pose_set(g2o_pose_path, poses, int(stats["dimension"]))
    rows.append({
        "dataset": dataset,
        "selected_pair": selected["stage_budget_pair"],
        "rotation_iteration_budget": rotation_budget,
        "translation_iteration_budget": translation_budget,
        "pose_path": str(pose_path),
        "g2o_pose_path": str(g2o_pose_path),
        "pose_count": int(stats["pose_count"]),
        "dimension": int(stats["dimension"]),
        "handoff_cost": float(stats["handoff_cost"]["total_cost"]),
        "total_schur_energy_gap": float(stats["total_schur_energy_gap"]),
        "total_schur_residual_norm": float(stats["total_schur_residual_norm"]),
    })
  _write_csv(output_dir / "selected_initialization_summary.csv", rows)
  return rows


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  if any(int(value) < 0 for value in args.rotation_budget + args.translation_budget):
    raise ValueError("stage budgets must be non-negative")
  output_dir = Path(args.output_dir)
  cci_dir = output_dir / "cci_reference"
  stage_dir = output_dir / "stage_budget"
  cert_dir = output_dir / "certificate_stop"
  output_dir.mkdir(parents=True, exist_ok=True)

  dataset_specs = sweep.build_dataset_specs(
      dataset_tokens=args.dataset,
      num_robots=args.num_robots,
      robot_topology=args.robot_topology)
  reference_rows, reference_full_stats = cci_ref.build_reference_rows(
      dataset_specs=dataset_specs,
      weighted=args.weighted,
      cost_mode=args.cost_mode,
      num_robots=args.num_robots)
  cci_ref._write_outputs(
      cci_dir,
      reference_rows,
      reference_full_stats,
      {
          "model": "cci_reference_summary",
          "row_count": int(len(reference_rows)),
          "dataset_count": int(len(dataset_specs)),
          "reference_method": "centralized_chordal_initialization",
          "weighted": bool(args.weighted),
          "cost_mode": str(args.cost_mode),
          "num_robots": int(args.num_robots),
      })

  reference_costs = {
      str(row["dataset"]): float(row["reference_handoff_cost"])
      for row in reference_rows
      if _finite(row.get("reference_handoff_cost"))
  }
  stage_rows, stage_full_stats, stage_checkpoint_info = _build_stage_rows(
      args,
      dataset_specs,
      stage_dir=stage_dir,
      reference_costs=reference_costs)
  _write_stage_outputs(stage_dir, args, stage_rows, stage_full_stats,
                       dataset_specs)

  gate_report = cert_stop.analyze_rows(
      stage_rows,
      _reference_gates(args),
      gate_source="reference_summary_relative",
      slacks=_reference_slacks(args),
      reference_handoff_costs=reference_costs)
  cert_stop.write_analysis(gate_report, cert_dir)

  summary_rows = _compact_summary_rows(gate_report)
  _write_csv(output_dir / "reference_gate_experiment_summary.csv",
             summary_rows)
  selected_initialization_rows = []
  selected_initialization_summary_path = None
  if args.write_selected_initialization:
    selected_initialization_dir = output_dir / "selected_initializations"
    selected_initialization_rows = _write_selected_initializations(
        selected_initialization_dir, args, gate_report, dataset_specs)
    selected_initialization_summary_path = str(
        selected_initialization_dir / "selected_initialization_summary.csv")
  experiment_report = {
      "model": "dci_reference_gate_experiment",
      "dataset_count": int(len(dataset_specs)),
      "stage_row_count": int(len(stage_rows)),
      "stage_computed_row_count": int(
          stage_checkpoint_info["computed_row_count"]),
      "stage_resumed_row_count": int(
          stage_checkpoint_info["resumed_row_count"]),
      "adaptive_stopped_dataset_count": int(
          stage_checkpoint_info["adaptive_stopped_dataset_count"]),
      "stage_checkpoint_path": stage_checkpoint_info["checkpoint_path"],
      "reference_row_count": int(len(reference_rows)),
      "summary_rows": summary_rows,
      "reference_summary_path": str(cci_dir / "cci_reference_summary.csv"),
      "stage_budget_summary_path": str(stage_dir / "stage_budget_summary.csv"),
      "certificate_report_path": str(
          cert_dir / "stage_certificate_stop_report.json"),
      "gate_report": gate_report,
      "weighted": bool(args.weighted),
      "cost_mode": str(args.cost_mode),
      "num_robots": int(args.num_robots),
      "solver": args.solver,
      "checkpoint_rows": bool(args.checkpoint_rows),
      "resume": bool(args.resume),
      "adaptive_reference_gate": bool(args.adaptive_reference_gate),
      "adaptive_proxy_gate": bool(args.adaptive_proxy_gate),
      "write_selected_initialization": bool(
          args.write_selected_initialization),
      "selected_initialization_summary_path": (
          selected_initialization_summary_path),
      "selected_initializations": selected_initialization_rows,
      "proxy_gate_thresholds": _proxy_gate_thresholds(args),
      "schur_preconditioner": args.schur_preconditioner,
      "fixed_step_acceleration": args.fixed_step_acceleration,
      "chebyshev_certificate_max_payload_mb": (
          None if args.chebyshev_certificate_max_payload_mb is None
          else float(args.chebyshev_certificate_max_payload_mb)),
      "chebyshev_certificate_preflight_hard_cap_mb": (
          None if args.chebyshev_certificate_preflight_hard_cap_mb is None
          else float(args.chebyshev_certificate_preflight_hard_cap_mb)),
      "rotation_iteration_budgets": [
          int(value) for value in args.rotation_budget
      ],
      "translation_iteration_budgets": [
          int(value) for value in args.translation_budget
      ],
      "handoff_reference_relative_slack": float(
          args.handoff_reference_relative_slack),
      "projection_cost_reference_relative_slack": float(
          args.projection_cost_reference_relative_slack),
  }
  (output_dir / "reference_gate_experiment_report.json").write_text(
      json.dumps(experiment_report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  print(output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
