import csv
from pathlib import Path

from scripts import analyze_dci_downstream_handoff as audit


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(root: Path, dataset: str, initial_cost: float,
               final_cost: float, final_gradient: float,
               downstream_comm_mb: float):
    iter_path = root / "runs" / dataset / "iteration_summary.csv"
    _write_csv(iter_path, [
        {
            "iter": "0",
            "global_cost": str(initial_cost),
            "gradient": "10",
            "cumulative_comm_mb": "0.1",
        },
        {
            "iter": "2",
            "global_cost": str(final_cost),
            "gradient": str(final_gradient),
            "cumulative_comm_mb": str(downstream_comm_mb),
        },
    ])
    _write_csv(root / "summary.csv", [
        {
            "dataset": dataset,
            "status": "ok",
            "global_cost": str(final_cost),
            "gradient": str(final_gradient),
            "total_comm_mb": str(downstream_comm_mb),
            "iter_summary_path": str(iter_path),
        }
    ])


def test_compare_handoff_roots_computes_init_and_final_gaps(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    _write_run(baseline, "tiny", 100.0, 90.0, 1.5, 0.25)
    _write_run(variant, "tiny", 110.0, 91.25, 2.0, 0.30)
    init_summary = tmp_path / "init_summary.csv"
    _write_csv(init_summary, [
        {
            "dataset": "tiny",
            "selected_pair": "20:60",
            "selected_handoff_cost": "110.0",
            "selected_total_comm_mb": "1.2",
        }
    ])

    rows = audit.compare_handoff_roots(
        baseline_root=baseline,
        case_roots={"dci20": variant},
        init_summary_paths={"dci20": init_summary},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "tiny"
    assert row["case"] == "dci20"
    assert row["selected_pair"] == "20:60"
    assert row["baseline_initial_cost"] == 100.0
    assert row["case_initial_cost"] == 110.0
    assert row["initial_cost_gap_to_baseline"] == 10.0
    assert row["final_cost_gap_to_baseline"] == 1.25
    assert row["gradient_gap_to_baseline"] == 0.5
    assert row["init_comm_mb"] == 1.2
    assert row["downstream_comm_mb"] == 0.30
    assert row["total_comm_with_init_mb"] == 1.5


def test_compare_handoff_roots_includes_reference_gate_diagnostics(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    _write_run(baseline, "tiny", 100.0, 90.0, 1.5, 0.25)
    _write_run(variant, "tiny", 100.2, 90.01, 1.6, 0.30)
    init_summary = tmp_path / "init_summary.csv"
    _write_csv(init_summary, [
        {
            "dataset": "tiny",
            "reference_handoff_cost": "100.0",
            "selected_pair": "60:120",
            "selected_handoff_cost": "100.2",
            "selected_handoff_gap_to_reference": "0.2",
            "selected_projection_cost_increase": "0.03",
            "selected_rotation_residual": "0.004",
            "selected_translation_residual": "0.07",
            "selected_total_comm_mb": "2.0",
        }
    ])

    rows = audit.compare_handoff_roots(
        baseline_root=baseline,
        case_roots={"dci60": variant},
        init_summary_paths={"dci60": init_summary},
    )

    assert rows[0]["reference_handoff_cost"] == 100.0
    assert rows[0]["selected_handoff_gap_to_reference"] == 0.2
    assert rows[0]["selected_projection_cost_increase"] == 0.03
    assert rows[0]["selected_rotation_residual"] == 0.004
    assert rows[0]["selected_translation_residual"] == 0.07


def test_cli_writes_csv_and_markdown(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    _write_run(baseline, "tiny", 100.0, 90.0, 1.5, 0.25)
    _write_run(variant, "tiny", 100.2, 90.01, 1.6, 0.30)
    init_summary = tmp_path / "init_summary.csv"
    _write_csv(init_summary, [
        {
            "dataset": "tiny",
            "selected_pair": "60:120",
            "selected_handoff_cost": "100.2",
            "selected_total_comm_mb": "2.0",
        }
    ])
    output_csv = tmp_path / "handoff.csv"
    output_md = tmp_path / "handoff.md"

    exit_code = audit.main([
        "--baseline-root", str(baseline),
        "--case", f"dci60={variant}",
        "--init-summary", f"dci60={init_summary}",
        "--output-csv", str(output_csv),
        "--output-md", str(output_md),
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert rows[0]["case"] == "dci60"
    assert rows[0]["selected_pair"] == "60:120"
    assert float(rows[0]["final_cost_gap_to_baseline"]) == 0.01
    report = output_md.read_text(encoding="utf-8")
    assert "DCI Downstream Handoff Audit" in report
    assert "dci60" in report
