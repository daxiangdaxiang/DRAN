#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SUMMARY_COLUMNS = (
    "dataset",
    "ablation",
    "status",
    "baseline_cost",
    "cost",
    "delta_cost",
    "relative_delta_pct",
    "contribution_share",
    "baseline_gradient",
    "gradient",
    "delta_gradient",
    "comm_mb",
    "delta_comm_mb",
    "wall_time_sec",
    "delta_wall_time_sec",
    "solver_time_per_node_sec",
    "delta_solver_time_per_node_sec",
    "note",
)


def parse_float(value):
  if value is None or value == "":
    return None
  return float(value)


def load_rows(root):
  records = {}
  for summary_path in sorted(root.glob("*/summary.csv")):
    ablation = summary_path.parent.name
    with summary_path.open(newline="") as stream:
      reader = csv.DictReader(stream)
      for row in reader:
        dataset = row["dataset"]
        records[(ablation, dataset)] = row
  return records


def row_metric(row, key):
  return parse_float(row.get(key, ""))


def compute_contributions(records, baseline_name, share_ablations):
  datasets = sorted(
      dataset
      for ablation, dataset in records
      if ablation == baseline_name and records[(ablation, dataset)].get("status") == "ok"
  )
  if not datasets:
    raise RuntimeError(f"no ok baseline rows found for ablation '{baseline_name}'")

  ablations = sorted({ablation for ablation, _ in records if ablation != baseline_name})
  if not share_ablations:
    share_ablations = ablations

  rows = []
  for dataset in datasets:
    baseline = records[(baseline_name, dataset)]
    baseline_cost = row_metric(baseline, "global_cost")
    if baseline_cost is None:
      continue

    positive_total = 0.0
    deltas = {}
    for ablation in share_ablations:
      row = records.get((ablation, dataset))
      if row is None or row.get("status") != "ok":
        continue
      cost = row_metric(row, "global_cost")
      if cost is None:
        continue
      delta = cost - baseline_cost
      deltas[ablation] = delta
      positive_total += max(delta, 0.0)

    for ablation in ablations:
      row = records.get((ablation, dataset))
      if row is None:
        continue
      cost = row_metric(row, "global_cost")
      gradient = row_metric(row, "gradient")
      comm_mb = row_metric(row, "total_comm_mb")
      wall_time = row_metric(row, "wall_time_sec")
      solver_time = row_metric(row, "solver_time_per_node_sec")
      base_gradient = row_metric(baseline, "gradient")
      base_comm_mb = row_metric(baseline, "total_comm_mb")
      base_wall_time = row_metric(baseline, "wall_time_sec")
      base_solver_time = row_metric(baseline, "solver_time_per_node_sec")

      delta_cost = None if cost is None else cost - baseline_cost
      relative_delta_pct = None
      if delta_cost is not None and baseline_cost != 0:
        relative_delta_pct = 100.0 * delta_cost / abs(baseline_cost)

      contribution_share = None
      if ablation in share_ablations and delta_cost is not None:
        contribution_share = max(delta_cost, 0.0) / positive_total if positive_total > 0 else 0.0

      note = ""
      if row.get("status") != "ok":
        note = "run not ok"
      elif contribution_share == 0.0 and delta_cost is not None and delta_cost < 0:
        note = "ablation improved cost; advantage not confirmed in this budget"
      elif ablation == "centralized_init":
        note = "centralized-init replacement, not a removed mechanism"

      rows.append({
          "dataset": dataset,
          "ablation": ablation,
          "status": row.get("status", ""),
          "baseline_cost": baseline_cost,
          "cost": cost,
          "delta_cost": delta_cost,
          "relative_delta_pct": relative_delta_pct,
          "contribution_share": contribution_share,
          "baseline_gradient": base_gradient,
          "gradient": gradient,
          "delta_gradient": None if gradient is None or base_gradient is None else gradient - base_gradient,
          "comm_mb": comm_mb,
          "delta_comm_mb": None if comm_mb is None or base_comm_mb is None else comm_mb - base_comm_mb,
          "wall_time_sec": wall_time,
          "delta_wall_time_sec": None if wall_time is None or base_wall_time is None else wall_time - base_wall_time,
          "solver_time_per_node_sec": solver_time,
          "delta_solver_time_per_node_sec": None if solver_time is None or base_solver_time is None else solver_time - base_solver_time,
          "note": note,
      })
  return rows


def fmt(value):
  if value is None:
    return ""
  if isinstance(value, float):
    return f"{value:.12g}"
  return str(value)


def write_csv(path, rows):
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: fmt(row.get(key)) for key in SUMMARY_COLUMNS})


def write_markdown(path, rows, share_ablations):
  grouped = {}
  for row in rows:
    grouped.setdefault(row["ablation"], []).append(row)

  lines = [
      "# DPGO-MM Internal Ablation Contributions",
      "",
      "Contribution share is computed from positive final-cost degradation relative to baseline within each dataset.",
      "If an ablation improves cost, its share is zero and the mechanism is not confirmed by that budget.",
      "",
      "| Ablation | Mean delta cost | Mean relative delta % | Mean contribution share | Notes |",
      "| --- | ---: | ---: | ---: | --- |",
  ]

  for ablation in sorted(grouped):
    subset = grouped[ablation]
    deltas = [row["delta_cost"] for row in subset if row["delta_cost"] is not None]
    rels = [row["relative_delta_pct"] for row in subset if row["relative_delta_pct"] is not None]
    shares = [
        row["contribution_share"]
        for row in subset
        if row["contribution_share"] is not None and ablation in share_ablations
    ]
    notes = sorted({row["note"] for row in subset if row["note"]})
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    mean_rel = sum(rels) / len(rels) if rels else None
    mean_share = sum(shares) / len(shares) if shares else None
    lines.append(
        f"| `{ablation}` | {fmt(mean_delta)} | {fmt(mean_rel)} | "
        f"{fmt(mean_share)} | {'; '.join(notes)} |"
    )

  lines.extend([
      "",
      "Share ablations: " + ", ".join(f"`{item}`" for item in share_ablations),
      "",
  ])
  path.write_text("\n".join(lines))


def main():
  parser = argparse.ArgumentParser(
      description="Summarize DPGO-MM internal ablation contribution ratios.")
  parser.add_argument("--root", required=True, type=Path,
                      help="ablation matrix root containing <ablation>/summary.csv")
  parser.add_argument("--output-dir", type=Path,
                      help="directory for ablation_contributions.csv and summary.md")
  parser.add_argument("--baseline", default="baseline")
  parser.add_argument(
      "--share-ablations",
      default="no_amm,no_preconditioner,weak_init,no_tnt",
      help="comma-separated ablations included in positive-degradation share")
  args = parser.parse_args()

  output_dir = args.output_dir or args.root
  output_dir.mkdir(parents=True, exist_ok=True)
  share_ablations = [item for item in args.share_ablations.split(",") if item]
  records = load_rows(args.root)
  rows = compute_contributions(records, args.baseline, share_ablations)
  write_csv(output_dir / "ablation_contributions.csv", rows)
  write_markdown(output_dir / "summary.md", rows, share_ablations)
  print(f"Saved: {output_dir / 'ablation_contributions.csv'}")
  print(f"Saved: {output_dir / 'summary.md'}")


if __name__ == "__main__":
  main()
