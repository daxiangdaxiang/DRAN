#!/usr/bin/env python3
"""Summarize BC-DCI pre-gate and topology-coverage outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASSIFICATIONS = [
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "pre_gate_negative_unverified",
    "coverage_blocked",
    "value_skipped",
    "other_skip",
    "unknown",
]


def _truthy(value: object) -> bool:
  return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_or_none(value: object):
  text = str(value).strip()
  if not text:
    return None
  try:
    return float(text)
  except ValueError:
    return None


def _fmt(value) -> str:
  if value is None:
    return ""
  if isinstance(value, float):
    return f"{value:.12g}"
  return str(value)


def graph_key(row: dict[str, object]) -> str:
  graph = str(row.get("graph") or row.get("dataset") or "").strip()
  if not graph:
    return ""
  return Path(graph).name


def classify_row(row: dict[str, object]) -> str:
  decision = str(row.get("pre_gate_decision", "")).strip()
  selected = str(row.get("selected", "")).strip()
  solved = _truthy(row.get("candidate_solved", ""))
  accepted = _truthy(row.get("global_consensus_accept", ""))

  if decision == "skip_insufficient_topology_coverage":
    return "coverage_blocked"
  if decision == "skip_precert":
    return "pre_gate_negative_unverified"
  if decision == "skip":
    return "value_skipped"
  if decision.startswith("skip"):
    return "other_skip"
  if solved and (accepted or selected == "distributed_chordal_pcg"):
    return "true_positive"
  if solved and not accepted:
    return "false_positive"
  return "unknown"


def normalize_row(row: dict[str, object], source: str) -> dict[str, object]:
  baseline = _float_or_none(row.get("baseline_total_cost", ""))
  selected = _float_or_none(row.get("selected_total_cost", ""))
  cost_delta = None
  if baseline is not None and selected is not None:
    cost_delta = selected - baseline
  return {
      "source": source,
      "graph": str(row.get("graph") or row.get("dataset") or ""),
      "graph_key": graph_key(row),
      "classification": classify_row(row),
      "selected": str(row.get("selected", "")),
      "pre_gate_decision": str(row.get("pre_gate_decision", "")),
      "candidate_solved": str(row.get("candidate_solved", "")),
      "global_consensus_accept": str(row.get("global_consensus_accept", "")),
      "baseline_total_cost": _fmt(baseline),
      "selected_total_cost": _fmt(selected),
      "cost_delta": _fmt(cost_delta),
      "actual_dci_comm_mb": _fmt(_float_or_none(row.get("actual_dci_comm_mb", ""))),
      "candidate_precert_ratio": _fmt(_float_or_none(
          row.get("candidate_precert_sep_private_ratio", ""))),
      "topology_separator_coverage": _fmt(_float_or_none(
          row.get("topology_separator_coverage", ""))),
      "topology_separator_edges": str(row.get("topology_separator_edges", "")),
      "full_separator_edges": str(row.get("full_separator_edges", "")),
      "relay_selected": str(row.get("relay_scheduler_selected_relay_edges", "")),
      "relay_skipped": str(row.get("relay_scheduler_skipped_relay_edges", "")),
      "oracle_selected": "",
      "oracle_accept": "",
  }


def read_summary(path: Path, source: str | None = None) -> list[dict[str, object]]:
  label = source or path.parent.name
  with path.open(newline="", encoding="utf-8") as handle:
    return [normalize_row(dict(row), label) for row in csv.DictReader(handle)]


def _oracle_accept(row: dict[str, object]) -> bool:
  return (
      str(row.get("selected", "")) == "distributed_chordal_pcg" or
      _truthy(row.get("global_consensus_accept", ""))
  )


def apply_oracle_labels(rows: list[dict[str, object]],
                        oracle_rows: list[dict[str, object]]) -> None:
  oracle_by_graph = {str(row.get("graph_key", "")): row for row in oracle_rows}
  for row in rows:
    oracle = oracle_by_graph.get(str(row.get("graph_key", "")))
    if oracle is None:
      continue
    oracle_accept = _oracle_accept(oracle)
    row["oracle_selected"] = str(oracle.get("selected", ""))
    row["oracle_accept"] = "true" if oracle_accept else "false"
    if row["classification"] == "pre_gate_negative_unverified":
      row["classification"] = "false_negative" if oracle_accept else "true_negative"


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
  summary: dict[str, object] = {"total_rows": len(rows)}
  for name in CLASSIFICATIONS:
    summary[name] = sum(1 for row in rows if row.get("classification") == name)
  summary["candidate_solved"] = sum(
      1 for row in rows if _truthy(row.get("candidate_solved", "")))
  summary["accepted_cost_delta_sum"] = sum(
      _float_or_none(row.get("cost_delta", "")) or 0.0
      for row in rows
      if row.get("classification") == "true_positive")
  summary["actual_dci_comm_mb_sum"] = sum(
      _float_or_none(row.get("actual_dci_comm_mb", "")) or 0.0
      for row in rows)
  blocked_coverages = [
      _float_or_none(row.get("topology_separator_coverage", ""))
      for row in rows
      if row.get("classification") == "coverage_blocked"
  ]
  blocked_coverages = [value for value in blocked_coverages if value is not None]
  summary["coverage_blocked_min_coverage"] = (
      min(blocked_coverages) if blocked_coverages else None)
  return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  fields = [
      "source",
      "graph",
      "graph_key",
      "classification",
      "selected",
      "pre_gate_decision",
      "candidate_solved",
      "global_consensus_accept",
      "baseline_total_cost",
      "selected_total_cost",
      "cost_delta",
      "actual_dci_comm_mb",
      "candidate_precert_ratio",
      "topology_separator_coverage",
      "topology_separator_edges",
      "full_separator_edges",
      "relay_selected",
      "relay_skipped",
      "oracle_selected",
      "oracle_accept",
  ]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, object]],
                   summary: dict[str, object]) -> None:
  lines = [
      "# DCI Pre-Gate Generalization Summary",
      "",
      "## Counts",
      "",
      "| Metric | Value |",
      "| --- | ---: |",
  ]
  for key, value in summary.items():
    lines.append(f"| `{key}` | `{_fmt(value)}` |")
  lines.extend([
      "",
      "## Rows",
      "",
      "| Source | Graph | Class | Decision | Selected | Ratio | Coverage | DCI MB |",
      "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
  ])
  for row in rows:
    lines.append(
        f"| `{row.get('source', '')}` | `{row.get('graph_key', '')}` | "
        f"`{row.get('classification', '')}` | "
        f"`{row.get('pre_gate_decision', '')}` | "
        f"`{row.get('selected', '')}` | "
        f"`{row.get('candidate_precert_ratio', '')}` | "
        f"`{row.get('topology_separator_coverage', '')}` | "
        f"`{row.get('actual_dci_comm_mb', '')}` |"
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_labeled_path(spec: str) -> tuple[str | None, Path]:
  if "=" in spec:
    label, path = spec.split("=", 1)
    return label, Path(path)
  return None, Path(spec)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--summary", action="append", required=True,
                      help="Summary CSV, optionally label=path.")
  parser.add_argument("--oracle-summary", action="append", default=[],
                      help="Oracle summary CSV for false-negative labeling.")
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args(argv)

  rows: list[dict[str, object]] = []
  for spec in args.summary:
    label, path = _parse_labeled_path(spec)
    rows.extend(read_summary(path, label))

  oracle_rows: list[dict[str, object]] = []
  for spec in args.oracle_summary:
    label, path = _parse_labeled_path(spec)
    oracle_rows.extend(read_summary(path, label))
  if oracle_rows:
    apply_oracle_labels(rows, oracle_rows)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  summary = summarize(rows)
  write_csv(output_dir / "dci_pregate_generalization.csv", rows)
  (output_dir / "dci_pregate_generalization_summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  write_markdown(output_dir / "dci_pregate_generalization.md", rows, summary)
  print(output_dir)


if __name__ == "__main__":
  main()
