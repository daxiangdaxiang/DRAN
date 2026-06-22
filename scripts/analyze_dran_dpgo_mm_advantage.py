#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


DATASET_G2O = {
  "parking-garage": "parking-garage.g2o",
  "sphere": "sphere2500.g2o",
  "torus": "torus3D.g2o",
  "CSAIL": "CSAIL.g2o",
  "inter": "input_INTEL_g2o.g2o",
  "manhattan": "input_M3500_g2o.g2o",
}

SUMMARY_FIELDS = [
  "dataset",
  "dimension",
  "edge_tag",
  "dpgo_mm_payload_bytes_per_pose",
  "dran_cost",
  "dpgo_mm_cost",
  "final_cost_dran_minus_dpgo_mm",
  "dran_gap_abs",
  "dpgo_mm_gap_abs",
  "final_gap_dran_minus_dpgo_mm",
  "dran_comm_poses",
  "dpgo_mm_comm_poses",
  "dran_outer_comm_mb_reported",
  "dpgo_mm_outer_comm_mb",
  "dran_total_comm_mb_reported",
  "dpgo_mm_total_comm_mb_reported",
  "dran_payload_normalized_comm_mb",
  "dpgo_mm_iter0_comm_poses",
  "dpgo_mm_iter0_comm_mb",
  "proxy_init_rounds",
  "proxy_init_comm_mb",
  "proxy_total_comm_mb",
  "reported_comm_mb_dran_minus_dpgo_mm",
  "payload_normalized_comm_mb_dran_minus_dpgo_mm",
  "proxy_total_comm_mb_dran_minus_dpgo_mm",
  "dpgo_mm_dominates_reported",
  "dpgo_mm_dominates_payload_normalized",
  "dpgo_mm_dominates_with_init_proxy",
  "dominance_note",
  "init_proxy_note",
]


def read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    return [dict(row) for row in reader]


def as_float(value: str | None, default: float = math.nan) -> float:
  if value is None or value == "":
    return default
  try:
    return float(value)
  except ValueError:
    return default


def as_int(value: str | None, default: int = 0) -> int:
  number = as_float(value)
  if math.isnan(number):
    return default
  return int(round(number))


def fmt(value: float | int | str) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, int):
    return str(value)
  if math.isnan(value):
    return ""
  text = f"{value:.9f}".rstrip("0").rstrip(".")
  return text if text else "0"


def bool_text(value: bool) -> str:
  return "true" if value else "false"


def latest_matching(pattern: str, root: Path) -> Path | None:
  matches = sorted(root.glob(pattern))
  return matches[-1] if matches else None


def default_normalized_metrics(results_dir: Path) -> Path:
  path = latest_matching("chordal_fair_b20_corrected_consolidated_*/normalized_metrics.csv", results_dir)
  if path is None:
    raise FileNotFoundError("could not find corrected consolidated normalized_metrics.csv")
  return path


def default_curve_paths(results_dir: Path) -> list[Path]:
  paths: list[Path] = []
  for pattern in (
      "chordal_fair_full_methods_b20_*/curves.csv",
      "chordal_fair_b20_corrections_*/curves.csv",
  ):
    path = latest_matching(pattern, results_dir)
    if path is not None:
      paths.append(path)
  return paths


def detect_dimension(g2o_path: Path) -> tuple[int, str]:
  with g2o_path.open(errors="ignore") as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      tag = line.split()[0]
      if tag.startswith("EDGE_SE2"):
        return 2, tag
      if tag.startswith("EDGE_SE3"):
        return 3, tag
  raise ValueError(f"could not find EDGE_SE2/EDGE_SE3 tag in {g2o_path}")


def dataset_g2o_path(dataset: str, data_dir: Path) -> Path:
  name = DATASET_G2O.get(dataset, f"{dataset}.g2o")
  path = data_dir / name
  if path.exists():
    return path
  candidates = sorted(data_dir.glob(f"*{dataset}*.g2o"))
  if candidates:
    return candidates[0]
  raise FileNotFoundError(f"could not locate g2o file for dataset {dataset!r} under {data_dir}")


def key_for(row: dict[str, str]) -> tuple[str, str, str, str]:
  return (
      row.get("dataset", ""),
      row.get("method", ""),
      row.get("setting", ""),
      row.get("budget", ""),
  )


def select_metric_rows(rows: list[dict[str, str]], setting: str, budget: str) -> dict[tuple[str, str], dict[str, str]]:
  selected: dict[tuple[str, str], dict[str, str]] = {}
  for row in rows:
    if row.get("problem_type") != "six":
      continue
    if row.get("setting") != setting or row.get("budget") != budget:
      continue
    method = row.get("method", "")
    if method not in {"DRAN", "DPGO-MM"}:
      continue
    selected[(row.get("dataset", ""), method)] = row
  return selected


def iter0_comm_from_source(row: dict[str, str], payload_bytes: int) -> tuple[int, float] | None:
  source = row.get("source_iterations", "")
  if not source:
    return None
  path = Path(source)
  if not path.exists():
    return None
  try:
    rows = read_rows(path)
  except OSError:
    return None
  iter0 = [item for item in rows if item.get("iter") == "0"]
  if not iter0:
    return None
  first = iter0[0]
  poses = as_int(first.get("comm_pose_count") or first.get("iter_comm_pose_count"), -1)
  mb = as_float(
      first.get("iter_comm_mb")
      or first.get("comm_mb")
      or first.get("cumulative_comm_mb")
  )
  if poses < 0 and not math.isnan(mb):
    poses = int(round(mb * 1024 * 1024 / payload_bytes))
  if poses < 0:
    poses = 0
  if math.isnan(mb):
    mb = poses * payload_bytes / (1024 * 1024)
  return poses, mb


def iter0_comm_from_curves(
    curve_paths: list[Path],
    dataset: str,
    setting: str,
    budget: str,
    payload_bytes: int,
) -> tuple[int, float] | None:
  for path in curve_paths:
    try:
      rows = read_rows(path)
    except OSError:
      continue
    for row in rows:
      if row.get("problem_type") != "six":
        continue
      if row.get("dataset") != dataset or row.get("method") != "DPGO-MM":
        continue
      if row.get("setting") != setting or row.get("budget") != budget:
        continue
      if row.get("iter") != "0":
        continue
      poses = as_int(row.get("comm_pose_count") or row.get("iter_comm_pose_count"), -1)
      mb = as_float(row.get("iter_comm_mb") or row.get("cumulative_comm_mb"))
      if poses < 0 and not math.isnan(mb):
        poses = int(round(mb * 1024 * 1024 / payload_bytes))
      if poses < 0:
        poses = 0
      if math.isnan(mb):
        mb = poses * payload_bytes / (1024 * 1024)
      return poses, mb
  return None


def dpgo_mm_iter0_comm(
    dpgo_row: dict[str, str],
    curve_paths: list[Path],
    payload_bytes: int,
) -> tuple[int, float]:
  source_comm = iter0_comm_from_source(dpgo_row, payload_bytes)
  if source_comm is not None:
    return source_comm
  curve_comm = iter0_comm_from_curves(
      curve_paths,
      dpgo_row.get("dataset", ""),
      dpgo_row.get("setting", ""),
      dpgo_row.get("budget", ""),
      payload_bytes,
  )
  if curve_comm is not None:
    return curve_comm
  total_poses = as_int(dpgo_row.get("total_comm_poses"), 0)
  final_iter = as_int(dpgo_row.get("final_iter"), 0)
  rounds = max(final_iter + 1, 1)
  poses = int(round(total_poses / rounds))
  return poses, poses * payload_bytes / (1024 * 1024)


def build_summary(
    normalized_metrics: Path,
    curve_paths: list[Path],
    data_dir: Path,
    setting: str,
    budget: str,
    init_proxy_rounds: int,
) -> list[dict[str, str]]:
  metrics = select_metric_rows(read_rows(normalized_metrics), setting, budget)
  datasets = sorted({dataset for dataset, method in metrics if method == "DRAN"})
  rows: list[dict[str, str]] = []
  for dataset in datasets:
    dran = metrics.get((dataset, "DRAN"))
    dpgo = metrics.get((dataset, "DPGO-MM"))
    if dran is None or dpgo is None:
      continue
    dimension, edge_tag = detect_dimension(dataset_g2o_path(dataset, data_dir))
    payload_bytes = dimension * (dimension + 1) * 8
    dran_poses = as_int(dran.get("total_comm_poses"), 0)
    dpgo_poses = as_int(dpgo.get("total_comm_poses"), 0)
    dran_payload_mb = dran_poses * payload_bytes / (1024 * 1024)
    iter0_poses, iter0_mb = dpgo_mm_iter0_comm(dpgo, curve_paths, payload_bytes)
    proxy_init_mb = iter0_poses * payload_bytes * init_proxy_rounds / (1024 * 1024)
    outer_mb = as_float(dpgo.get("outer_comm_mb") or dpgo.get("total_comm_mb"), 0.0)
    dran_gap_abs = as_float(dran.get("cost_gap_abs"))
    dpgo_gap_abs = as_float(dpgo.get("cost_gap_abs"))
    dran_total_mb = as_float(dran.get("total_comm_mb"))
    dpgo_total_mb = as_float(dpgo.get("total_comm_mb"))
    proxy_total_mb = proxy_init_mb + outer_mb
    gap_dominates = dpgo_gap_abs <= dran_gap_abs
    reported_dominates = gap_dominates and dpgo_total_mb <= dran_total_mb
    payload_dominates = gap_dominates and outer_mb <= dran_payload_mb
    proxy_dominates = gap_dominates and proxy_total_mb <= dran_total_mb
    dominance_notes = []
    if reported_dominates:
      dominance_notes.append("DPGO-MM dominates reported cost gap and MB")
    if payload_dominates:
      dominance_notes.append("DPGO-MM dominates after DRAN payload normalization")
    if gap_dominates and not proxy_dominates:
      dominance_notes.append("DPGO-MM no longer dominates when init proxy is charged")
    row = {
      "dataset": dataset,
      "dimension": str(dimension),
      "edge_tag": edge_tag,
      "dpgo_mm_payload_bytes_per_pose": str(payload_bytes),
      "dran_cost": fmt(as_float(dran.get("cost"))),
      "dpgo_mm_cost": fmt(as_float(dpgo.get("cost"))),
      "final_cost_dran_minus_dpgo_mm": fmt(as_float(dran.get("cost")) - as_float(dpgo.get("cost"))),
      "dran_gap_abs": fmt(dran_gap_abs),
      "dpgo_mm_gap_abs": fmt(dpgo_gap_abs),
      "final_gap_dran_minus_dpgo_mm": fmt(dran_gap_abs - dpgo_gap_abs),
      "dran_comm_poses": str(dran_poses),
      "dpgo_mm_comm_poses": str(dpgo_poses),
      "dran_outer_comm_mb_reported": fmt(as_float(dran.get("outer_comm_mb"))),
      "dpgo_mm_outer_comm_mb": fmt(outer_mb),
      "dran_total_comm_mb_reported": fmt(dran_total_mb),
      "dpgo_mm_total_comm_mb_reported": fmt(dpgo_total_mb),
      "dran_payload_normalized_comm_mb": fmt(dran_payload_mb),
      "dpgo_mm_iter0_comm_poses": str(iter0_poses),
      "dpgo_mm_iter0_comm_mb": fmt(iter0_mb),
      "proxy_init_rounds": str(init_proxy_rounds),
      "proxy_init_comm_mb": fmt(proxy_init_mb),
      "proxy_total_comm_mb": fmt(proxy_total_mb),
      "reported_comm_mb_dran_minus_dpgo_mm": fmt(dran_total_mb - dpgo_total_mb),
      "payload_normalized_comm_mb_dran_minus_dpgo_mm": fmt(dran_payload_mb - outer_mb),
      "proxy_total_comm_mb_dran_minus_dpgo_mm": fmt(dran_total_mb - proxy_total_mb),
      "dpgo_mm_dominates_reported": bool_text(reported_dominates),
      "dpgo_mm_dominates_payload_normalized": bool_text(payload_dominates),
      "dpgo_mm_dominates_with_init_proxy": bool_text(proxy_dominates),
      "dominance_note": "; ".join(dominance_notes),
      "init_proxy_note": "proxy: DPGO-MM iter=0 comm_pose_count * SE(d) payload bytes * DChordal init rounds; not real staged accounting",
    }
    rows.append(row)
  return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
  lines = []
  lines.append("| " + " | ".join(fields) + " |")
  lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
  for row in rows:
    lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
  return "\n".join(lines)


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    normalized_metrics: Path,
    curve_paths: list[Path],
    init_proxy_rounds: int,
) -> None:
  fields = [
    "dataset",
    "final_gap_dran_minus_dpgo_mm",
    "dran_comm_poses",
    "dpgo_mm_comm_poses",
    "dran_payload_normalized_comm_mb",
    "dpgo_mm_outer_comm_mb",
    "proxy_init_comm_mb",
    "proxy_total_comm_mb",
    "reported_comm_mb_dran_minus_dpgo_mm",
    "payload_normalized_comm_mb_dran_minus_dpgo_mm",
    "proxy_total_comm_mb_dran_minus_dpgo_mm",
    "dpgo_mm_dominates_reported",
    "dpgo_mm_dominates_payload_normalized",
    "dpgo_mm_dominates_with_init_proxy",
  ]
  lines = [
    "# DRAN vs DPGO-MM Advantage Diagnostics",
    "",
    f"- normalized_metrics: `{normalized_metrics}`",
    "- curves: " + ", ".join(f"`{path}`" for path in curve_paths),
    f"- DPGO-MM init proxy rounds: `{init_proxy_rounds}`",
    "- Proxy caveat: `proxy_init_comm_mb` is not a real staged DPGO-MM measurement. It is iter=0 communication times the configured DChordal round count.",
    "- Payload-normalized DRAN MB uses the DPGO-MM SE(d) pose payload, `d * (d + 1) * 8` bytes, with `d` detected from the first EDGE tag in each g2o file.",
    "- Dominance flags use cost gap plus communication: reported MB, DRAN payload-normalized MB, and DPGO-MM init-proxy total MB.",
    "",
    markdown_table(rows, fields),
    "",
  ]
  path.write_text("\n".join(lines))


def make_output_dir(results_dir: Path, output_dir: Path | None) -> Path:
  if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  path = results_dir / f"dran_dpgo_mm_advantage_analysis_{stamp}"
  path.mkdir(parents=True, exist_ok=False)
  return path


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(
      description="Analyze fair DRAN vs DPGO-MM communication/cost advantage diagnostics."
  )
  parser.add_argument("--results-dir", type=Path, default=Path("results"))
  parser.add_argument("--data-dir", type=Path, default=Path("data"))
  parser.add_argument("--normalized-metrics", type=Path, default=None)
  parser.add_argument("--curves", type=Path, action="append", default=[])
  parser.add_argument("--setting", default="fixed_20")
  parser.add_argument("--budget", default="20")
  parser.add_argument("--init-proxy-rounds", type=int, default=900)
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args(argv)

  normalized_metrics = args.normalized_metrics or default_normalized_metrics(args.results_dir)
  curve_paths = args.curves if args.curves else default_curve_paths(args.results_dir)
  if not curve_paths:
    print("warning: no curves.csv files found; falling back to normalized metric totals", file=sys.stderr)
  rows = build_summary(
      normalized_metrics=normalized_metrics,
      curve_paths=curve_paths,
      data_dir=args.data_dir,
      setting=args.setting,
      budget=args.budget,
      init_proxy_rounds=args.init_proxy_rounds,
  )
  if not rows:
    raise RuntimeError("no DRAN/DPGO-MM dataset pairs found")
  output_dir = make_output_dir(args.results_dir, args.output_dir)
  write_csv(output_dir / "summary.csv", rows)
  write_markdown(output_dir / "summary.md", rows, normalized_metrics, curve_paths, args.init_proxy_rounds)
  print(output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
