import csv
import json
from pathlib import Path

from scripts import plan_dci_adaptive_grid_expansion as expansion


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


def test_expansion_plan_recommends_axes_with_large_boundary_gain():
    rows = expansion.analyze_candidate_rows(
        _grid_rows(),
        min_handoff_improvement=1.0,
        min_rotation_residual_improvement=0.01,
        min_translation_residual_improvement=0.01,
    )
    selected = expansion.select_min_comm_grid_stable(rows)

    assert selected["selected_pair"] == "60:120"
    recommendations = expansion.plan_expansions_for_selected(
        selected,
        rows,
        min_handoff_improvement=1.0,
        min_rotation_residual_improvement=0.01,
        min_translation_residual_improvement=0.01,
    )

    pairs = {row["recommended_pair"]: row["reason"]
             for row in recommendations}
    assert pairs == {
        "100:120": "rotation_boundary_gain",
        "60:180": "translation_boundary_gain",
        "100:180": "combined_boundary_gain",
    }
    rotation = next(row for row in recommendations
                    if row["recommended_pair"] == "100:120")
    assert rotation["previous_pair"] == "20:120"
    assert rotation["base_pair"] == "60:120"
    assert rotation["axis_step"] == 40


def test_expansion_plan_stops_when_boundary_gain_is_small():
    rows = expansion.analyze_candidate_rows([
        {
            "dataset": "tiny",
            "selected_pair": "20:60",
            "selected_handoff_cost": "100.05",
            "selected_rotation_residual": "0.05",
            "selected_translation_residual": "0.002",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "1.0",
        },
        {
            "dataset": "tiny",
            "selected_pair": "60:60",
            "selected_handoff_cost": "100.0",
            "selected_rotation_residual": "0.049",
            "selected_translation_residual": "0.002",
            "selected_projection_cost_increase": "0.0",
            "selected_total_comm_mb": "2.0",
        },
    ], min_handoff_improvement=1.0,
       min_rotation_residual_improvement=0.01)
    selected = next(row for row in rows if row["selected_pair"] == "60:60")
    assert expansion.plan_expansions_for_selected(
        selected, rows,
        min_handoff_improvement=1.0,
        min_rotation_residual_improvement=0.01,
        min_translation_residual_improvement=0.01,
    ) == []


def test_cli_writes_adaptive_expansion_plan(tmp_path):
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, _grid_rows())
    output_dir = tmp_path / "plan"

    exit_code = expansion.main([
        "--candidate-summary", str(candidates),
        "--output-dir", str(output_dir),
        "--min-handoff-improvement", "1.0",
        "--min-rotation-residual-improvement", "0.01",
        "--min-translation-residual-improvement", "0.01",
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(
        (output_dir / "dci_adaptive_grid_expansion_plan.csv").open(
            encoding="utf-8")))
    assert {row["recommended_pair"] for row in rows} == {
        "100:120", "60:180", "100:180",
    }
    report = json.loads(
        (output_dir / "dci_adaptive_grid_expansion_plan.json").read_text(
            encoding="utf-8"))
    assert report["model"] == "dci_adaptive_grid_expansion_plan"
    assert report["selected_row"]["selected_pair"] == "60:120"
    markdown = (
        output_dir / "dci_adaptive_grid_expansion_plan.md").read_text(
            encoding="utf-8")
    assert "Adaptive Grid Expansion Plan" in markdown
