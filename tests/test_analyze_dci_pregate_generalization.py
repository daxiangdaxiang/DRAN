#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_dci_pregate_generalization.py"


def load_module():
  spec = importlib.util.spec_from_file_location("dci_pregate", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  fields: list[str] = []
  for row in rows:
    for key in row:
      if key not in fields:
        fields.append(key)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


class DciPregateGeneralizationAnalysisTest(unittest.TestCase):
  def test_classifies_candidate_outcomes_without_oracle(self) -> None:
    module = load_module()
    rows = [
        {
            "graph": "data/a.g2o",
            "selected": "distributed_chordal_pcg",
            "pre_gate_decision": "run",
            "candidate_solved": "True",
            "global_consensus_accept": "True",
            "baseline_total_cost": "10",
            "selected_total_cost": "5",
        },
        {
            "graph": "data/b.g2o",
            "selected": "baseline_gauge_selector",
            "pre_gate_decision": "run",
            "candidate_solved": "True",
            "global_consensus_accept": "False",
            "baseline_total_cost": "10",
            "selected_total_cost": "10",
        },
        {
            "graph": "data/c.g2o",
            "selected": "baseline_gauge_selector",
            "pre_gate_decision": "skip_precert",
            "candidate_solved": "False",
            "global_consensus_accept": "False",
        },
        {
            "graph": "data/d.g2o",
            "selected": "baseline_gauge_selector",
            "pre_gate_decision": "skip_insufficient_topology_coverage",
            "candidate_solved": "False",
            "topology_separator_coverage": "0.7",
        },
    ]

    classified = [module.normalize_row(row, "toy") for row in rows]
    labels = [row["classification"] for row in classified]

    self.assertEqual(labels, [
        "true_positive",
        "false_positive",
        "pre_gate_negative_unverified",
        "coverage_blocked",
    ])
    summary = module.summarize(classified)
    self.assertEqual(summary["total_rows"], 4)
    self.assertEqual(summary["true_positive"], 1)
    self.assertEqual(summary["false_positive"], 1)
    self.assertEqual(summary["pre_gate_negative_unverified"], 1)
    self.assertEqual(summary["coverage_blocked"], 1)
    self.assertEqual(summary["accepted_cost_delta_sum"], -5.0)

  def test_oracle_marks_skipped_accepted_candidate_as_false_negative(self) -> None:
    module = load_module()
    current = module.normalize_row({
        "graph": "data/sphere.g2o",
        "selected": "baseline_gauge_selector",
        "pre_gate_decision": "skip_precert",
        "candidate_solved": "False",
        "global_consensus_accept": "False",
    }, "current")
    oracle = module.normalize_row({
        "graph": "data/sphere.g2o",
        "selected": "distributed_chordal_pcg",
        "pre_gate_decision": "run",
        "candidate_solved": "True",
        "global_consensus_accept": "True",
    }, "oracle")

    module.apply_oracle_labels([current], [oracle])

    self.assertEqual(current["oracle_selected"], "distributed_chordal_pcg")
    self.assertEqual(current["oracle_accept"], "true")
    self.assertEqual(current["classification"], "false_negative")

  def test_oracle_marks_skipped_rejected_candidate_as_true_negative(self) -> None:
    module = load_module()
    current = module.normalize_row({
        "graph": "data/sphere.g2o",
        "selected": "baseline_gauge_selector",
        "pre_gate_decision": "skip_precert",
        "candidate_solved": "False",
        "global_consensus_accept": "False",
    }, "current")
    oracle = module.normalize_row({
        "graph": "data/sphere.g2o",
        "selected": "baseline_gauge_selector",
        "pre_gate_decision": "run",
        "candidate_solved": "True",
        "global_consensus_accept": "False",
    }, "oracle")

    module.apply_oracle_labels([current], [oracle])

    self.assertEqual(current["oracle_selected"], "baseline_gauge_selector")
    self.assertEqual(current["oracle_accept"], "false")
    self.assertEqual(current["classification"], "true_negative")

  def test_cli_writes_combined_csv_and_summary_json(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      summary_csv = tmp_path / "summary.csv"
      write_csv(summary_csv, [
          {
              "graph": "data/a.g2o",
              "selected": "distributed_chordal_pcg",
              "pre_gate_decision": "run",
              "candidate_solved": "True",
              "global_consensus_accept": "True",
              "baseline_total_cost": "4",
              "selected_total_cost": "1",
              "actual_dci_comm_mb": "0.5",
          }
      ])
      out_dir = tmp_path / "out"

      module.main([
          "--summary", str(summary_csv),
          "--output-dir", str(out_dir),
      ])

      combined = out_dir / "dci_pregate_generalization.csv"
      report = out_dir / "dci_pregate_generalization.md"
      summary = out_dir / "dci_pregate_generalization_summary.json"
      self.assertTrue(combined.is_file())
      self.assertTrue(report.is_file())
      self.assertTrue(summary.is_file())
      with combined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
      self.assertEqual(rows[0]["classification"], "true_positive")


if __name__ == "__main__":
  unittest.main()
