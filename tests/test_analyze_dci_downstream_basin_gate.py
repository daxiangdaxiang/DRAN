import csv
import json
from pathlib import Path

from scripts import analyze_dci_downstream_basin_gate as basin_gate


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _handoff_rows():
    return [
        {
            "dataset": "tiny",
            "case": "early",
            "selected_pair": "20:60",
            "final_cost_gap_to_baseline": "8.0",
            "gradient_gap_to_baseline": "4.0",
            "selected_handoff_gap_to_reference": "400.0",
            "selected_projection_cost_increase": "0.0",
            "selected_rotation_residual": "15.0",
            "selected_translation_residual": "30.0",
            "init_comm_mb": "1.0",
        },
        {
            "dataset": "tiny",
            "case": "translation_only",
            "selected_pair": "20:120",
            "final_cost_gap_to_baseline": "7.0",
            "gradient_gap_to_baseline": "3.5",
            "selected_handoff_gap_to_reference": "390.0",
            "selected_projection_cost_increase": "0.0",
            "selected_rotation_residual": "15.0",
            "selected_translation_residual": "0.0005",
            "init_comm_mb": "1.5",
        },
        {
            "dataset": "tiny",
            "case": "rotation_only",
            "selected_pair": "60:60",
            "final_cost_gap_to_baseline": "1.5",
            "gradient_gap_to_baseline": "1.0",
            "selected_handoff_gap_to_reference": "28.0",
            "selected_projection_cost_increase": "0.0",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "30.0",
            "init_comm_mb": "2.0",
        },
        {
            "dataset": "tiny",
            "case": "tight",
            "selected_pair": "60:120",
            "final_cost_gap_to_baseline": "0.009",
            "gradient_gap_to_baseline": "0.007",
            "selected_handoff_gap_to_reference": "0.17",
            "selected_projection_cost_increase": "0.0",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "0.0005",
            "init_comm_mb": "2.8",
        },
    ]


def test_basin_gate_selects_thresholds_that_reject_partial_stage_fixes():
    rows = basin_gate.annotate_downstream_basin_labels(
        _handoff_rows(),
        max_final_cost_gap=0.01,
        max_gradient_gap=0.01,
    )

    assert [row["downstream_basin_feasible"] for row in rows] == [
        False, False, False, True,
    ]

    candidates = basin_gate.candidate_thresholds_from_feasible_rows(rows)
    scored = [basin_gate.score_candidate(candidate, rows)
              for candidate in candidates]
    selected = basin_gate.select_candidate(scored, max_false_positive=0)

    assert selected is not None
    assert selected["true_positive"] == 1
    assert selected["false_positive"] == 0
    assert selected["false_negative"] == 0
    assert selected["max_rotation_residual"] == 0.04
    assert selected["max_translation_residual"] == 0.0005
    accepted = {
        (row["case"], row["selected_pair"])
        for row in selected["accepted_rows"]
    }
    assert accepted == {("tight", "60:120")}


def test_cli_writes_basin_gate_summary_and_report(tmp_path):
    handoff_csv = tmp_path / "handoff.csv"
    _write_csv(handoff_csv, _handoff_rows())
    output_dir = tmp_path / "basin_gate"

    exit_code = basin_gate.main([
        "--handoff-comparison", str(handoff_csv),
        "--output-dir", str(output_dir),
        "--max-final-cost-gap", "0.01",
        "--max-gradient-gap", "0.01",
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(
        (output_dir / "downstream_basin_gate_summary.csv").open(
            encoding="utf-8")))
    assert any(row["selected"] == "True" for row in rows)
    report = json.loads(
        (output_dir / "downstream_basin_gate_report.json").read_text(
            encoding="utf-8"))
    assert report["model"] == "dci_downstream_basin_gate"
    assert report["downstream_basin_feasible_count"] == 1
    assert report["selected_candidate"]["max_rotation_residual"] == 0.04
    markdown = (
        output_dir / "downstream_basin_gate_report.md").read_text(
            encoding="utf-8")
    assert "Downstream Basin Gate" in markdown
    assert "60:120" in markdown
