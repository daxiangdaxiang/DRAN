import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts import analyze_dci_grid_stability_gate as stability


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grid_rows():
    return [
        {
            "dataset": "tiny",
            "selected_pair": "20:60",
            "selected_handoff_cost": "250.0",
            "selected_rotation_residual": "15.0",
            "selected_translation_residual": "30.0",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "1.0",
        },
        {
            "dataset": "tiny",
            "selected_pair": "20:120",
            "selected_handoff_cost": "220.0",
            "selected_rotation_residual": "15.0",
            "selected_translation_residual": "0.0005",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "1.5",
        },
        {
            "dataset": "tiny",
            "selected_pair": "60:60",
            "selected_handoff_cost": "200.0",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "30.0",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "2.0",
        },
        {
            "dataset": "tiny",
            "selected_pair": "60:120",
            "selected_handoff_cost": "100.0",
            "selected_rotation_residual": "0.04",
            "selected_translation_residual": "0.0005",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "2.8",
        },
    ]


def test_grid_stability_rejects_rows_with_axis_improvements_available():
    rows = stability.analyze_rows(
        _grid_rows(),
        min_handoff_improvement=1.0,
        min_rotation_residual_improvement=0.01,
        min_translation_residual_improvement=0.01,
    )
    by_pair = {row["selected_pair"]: row for row in rows}

    assert not by_pair["20:60"]["grid_stability_feasible"]
    assert by_pair["20:60"]["grid_stability_failure_reasons"] == (
        "rotation_axis_improvement,translation_axis_improvement")
    assert not by_pair["20:120"]["grid_stability_feasible"]
    assert by_pair["20:120"]["grid_stability_failure_reasons"] == (
        "rotation_axis_improvement")
    assert not by_pair["60:60"]["grid_stability_feasible"]
    assert by_pair["60:60"]["grid_stability_failure_reasons"] == (
        "translation_axis_improvement")
    assert by_pair["60:120"]["grid_stability_feasible"]
    assert by_pair["60:120"]["grid_stability_on_budget_boundary"]

    selected = stability.select_min_comm_stable(rows)
    assert selected is not None
    assert selected["selected_pair"] == "60:120"


def test_grid_stability_supports_stage_budget_summary_fields():
    rows = stability.analyze_rows([
        {
            "dataset": "tiny",
            "stage_budget_pair": "20:60",
            "handoff_cost": "250.0",
            "rotation_schur_residual_norm": "15.0",
            "translation_schur_residual_norm": "30.0",
            "rotation_comm_mb": "0.4",
            "translation_comm_mb": "0.6",
        },
        {
            "dataset": "tiny",
            "stage_budget_pair": "60:60",
            "handoff_cost": "200.0",
            "rotation_schur_residual_norm": "0.04",
            "translation_schur_residual_norm": "30.0",
            "rotation_comm_mb": "1.4",
            "translation_comm_mb": "0.6",
        },
    ], min_handoff_improvement=1.0)

    assert rows[0]["selected_pair"] == "20:60"
    assert rows[0]["rotation_budget"] == 20
    assert rows[0]["translation_budget"] == 60
    assert rows[0]["candidate_total_comm_mb"] == 1.0
    assert rows[0]["grid_stability_failure_reasons"] == (
        "rotation_axis_improvement")


def test_cli_writes_grid_stability_outputs(tmp_path):
    candidate_csv = tmp_path / "candidates.csv"
    _write_csv(candidate_csv, _grid_rows())
    output_dir = tmp_path / "stability"

    exit_code = stability.main([
        "--candidate-summary", str(candidate_csv),
        "--output-dir", str(output_dir),
        "--min-handoff-improvement", "1.0",
        "--min-rotation-residual-improvement", "0.01",
        "--min-translation-residual-improvement", "0.01",
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(
        (output_dir / "dci_grid_stability_gate.csv").open(
            encoding="utf-8")))
    selected_rows = [row for row in rows if row["selected"] == "True"]
    assert len(selected_rows) == 1
    assert selected_rows[0]["selected_pair"] == "60:120"
    report = json.loads(
        (output_dir / "dci_grid_stability_gate.json").read_text(
            encoding="utf-8"))
    assert report["model"] == "dci_grid_stability_gate"
    assert report["selected_row"]["selected_pair"] == "60:120"
    markdown = (output_dir / "dci_grid_stability_gate.md").read_text(
        encoding="utf-8")
    assert "Grid Stability Gate" in markdown
    assert "budget boundary" in markdown


def test_script_can_run_directly_from_repository_root(tmp_path):
    candidate_csv = tmp_path / "candidates.csv"
    _write_csv(candidate_csv, _grid_rows())
    output_dir = tmp_path / "direct"
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "analyze_dci_grid_stability_gate.py")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run([
        sys.executable,
        str(script),
        "--candidate-summary", str(candidate_csv),
        "--output-dir", str(output_dir),
        "--min-handoff-improvement", "1.0",
    ], cwd=Path(__file__).resolve().parents[1], text=True, env=env,
       capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert (output_dir / "dci_grid_stability_gate.json").exists()
