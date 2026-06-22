import csv
import json
from pathlib import Path

from scripts import apply_dci_downstream_basin_gate as apply_gate


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gate(path: Path):
    path.write_text(json.dumps({
        "model": "dci_downstream_basin_gate",
        "selected_candidate": {
            "max_handoff_gap_to_reference": 0.2,
            "max_rotation_residual": 0.05,
            "max_translation_residual": 0.001,
            "max_projection_cost_increase": 0.0,
        },
    }), encoding="utf-8")


def test_apply_gate_selects_min_comm_candidate_that_passes_all_stage_bounds(
        tmp_path):
    gate_path = tmp_path / "gate.json"
    _write_gate(gate_path)
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, [
        {
            "dataset": "tiny",
            "selected_pair": "20:120",
            "selected_handoff_cost": "100.1",
            "reference_handoff_cost": "100.0",
            "selected_handoff_gap_to_reference": "0.1",
            "selected_rotation_residual": "10.0",
            "selected_translation_residual": "0.0005",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "1.5",
        },
        {
            "dataset": "tiny",
            "selected_pair": "60:60",
            "selected_handoff_cost": "100.15",
            "reference_handoff_cost": "100.0",
            "selected_handoff_gap_to_reference": "0.15",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "30.0",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "2.0",
        },
        {
            "dataset": "tiny",
            "selected_pair": "60:120",
            "selected_handoff_cost": "100.17",
            "reference_handoff_cost": "100.0",
            "selected_handoff_gap_to_reference": "0.17",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "0.0005",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "2.8",
        },
    ])

    gate = apply_gate.load_gate(gate_path)
    rows = apply_gate.apply_gate_to_candidate_rows(
        apply_gate.read_candidate_rows([candidates]), gate)
    selected = apply_gate.select_min_comm_feasible(rows)

    assert selected is not None
    assert selected["selected_pair"] == "60:120"
    assert selected["gate_feasible"] is True
    assert selected["gate_failure_reasons"] == "none"
    failures = {row["selected_pair"]: row["gate_failure_reasons"]
                for row in rows if not row["gate_feasible"]}
    assert failures["20:120"] == "rotation_residual"
    assert failures["60:60"] == "translation_residual"


def test_apply_gate_supports_stage_budget_summary_fields(tmp_path):
    gate_path = tmp_path / "gate.json"
    _write_gate(gate_path)
    stage_summary = tmp_path / "stage_budget_summary.csv"
    _write_csv(stage_summary, [
        {
            "dataset": "tiny",
            "stage_budget_pair": "60:120",
            "handoff_cost": "100.17",
            "reference_handoff_cost": "100.0",
            "rotation_schur_residual_norm": "0.04",
            "translation_schur_residual_norm": "0.0005",
            "rotation_projection_cost_delta": "-5.0",
            "rotation_comm_mb": "1.2",
            "translation_comm_mb": "1.6",
        },
    ])

    rows = apply_gate.apply_gate_to_candidate_rows(
        apply_gate.read_candidate_rows([stage_summary]),
        apply_gate.load_gate(gate_path))

    assert rows[0]["gate_feasible"] is True
    assert rows[0]["selected_pair"] == "60:120"
    assert rows[0]["candidate_total_comm_mb"] == 2.8
    assert rows[0]["candidate_projection_cost_increase"] == 0.0


def test_cli_writes_application_outputs(tmp_path):
    gate_path = tmp_path / "gate.json"
    _write_gate(gate_path)
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, [
        {
            "dataset": "tiny",
            "selected_pair": "60:120",
            "selected_handoff_cost": "100.17",
            "reference_handoff_cost": "100.0",
            "selected_handoff_gap_to_reference": "0.17",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "0.0005",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "2.8",
        },
    ])
    output_dir = tmp_path / "application"

    exit_code = apply_gate.main([
        "--gate-report", str(gate_path),
        "--candidate-summary", str(candidates),
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(
        (output_dir / "downstream_basin_gate_application.csv").open(
            encoding="utf-8")))
    assert rows[0]["selected"] == "True"
    report = json.loads(
        (output_dir / "downstream_basin_gate_application.json").read_text(
            encoding="utf-8"))
    assert report["model"] == "dci_downstream_basin_gate_application"
    assert report["selected_row"]["selected_pair"] == "60:120"
    markdown = (
        output_dir / "downstream_basin_gate_application.md").read_text(
            encoding="utf-8")
    assert "Downstream Basin Gate Application" in markdown
