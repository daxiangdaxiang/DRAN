import csv
import math

from scripts import analyze_dci_handoff_certificate as cert


def test_report_certificate_marks_complete_separator_summary_ready():
    report = {
        "dataset": "toy",
        "topology_separator_coverage": 1.0,
        "separator_residual_mass_coverage": 1.0,
        "separator_normal_summary_payload_coverage": 1.0,
        "separator_normal_summary_missing_bridge_block_edges": 0,
        "separator_normal_summary_component_count_delta": 0,
        "separator_normal_summary_structural_criticality": 0.0,
        "separator_normal_summary_block_graph": {
            "full_block_edge_count": 100,
        },
    }

    result = cert.evaluate_report_certificate(
        report,
        cert.CertificateThresholds(
            min_topology_coverage=0.95,
            min_residual_mass_coverage=0.99,
            min_payload_coverage=0.98,
            max_missing_bridge_ratio=0.02,
            max_component_delta_ratio=0.02,
            max_structural_criticality=0.05,
        ),
    )

    assert result["status"] == "ready"
    assert all(gate["passed"] for gate in result["gates"])


def test_report_certificate_flags_missing_bridge_structure_not_ready():
    report = {
        "dataset": "toy",
        "topology_separator_coverage": 0.99,
        "separator_residual_mass_coverage": 0.999,
        "separator_normal_summary_payload_coverage": 0.99,
        "separator_normal_summary_missing_bridge_block_edges": 3,
        "separator_normal_summary_component_count_delta": 1,
        "separator_normal_summary_structural_criticality": 0.01,
        "separator_normal_summary_block_graph": {
            "full_block_edge_count": 100,
        },
    }

    result = cert.evaluate_report_certificate(
        report,
        cert.CertificateThresholds(
            min_topology_coverage=0.95,
            min_residual_mass_coverage=0.99,
            min_payload_coverage=0.98,
            max_missing_bridge_ratio=0.02,
            max_component_delta_ratio=0.02,
            max_structural_criticality=0.05,
        ),
    )

    assert result["status"] == "not_ready"
    failed_gate_names = {
        gate["name"] for gate in result["gates"] if not gate["passed"]
    }
    assert failed_gate_names == {"missing_bridge_block_ratio"}


def test_report_certificate_does_not_fail_on_small_absolute_bridge_count():
    report = {
        "dataset": "toy",
        "topology_separator_coverage": 0.99,
        "separator_residual_mass_coverage": 0.999,
        "separator_normal_summary_payload_coverage": 0.99,
        "separator_normal_summary_missing_bridge_block_edges": 1,
        "separator_normal_summary_component_count_delta": 1,
        "separator_normal_summary_structural_criticality": 0.01,
        "separator_normal_summary_block_graph": {
            "full_block_edge_count": 200,
        },
    }

    result = cert.evaluate_report_certificate(
        report,
        cert.CertificateThresholds(
            min_topology_coverage=0.95,
            min_residual_mass_coverage=0.99,
            min_payload_coverage=0.98,
            max_missing_bridge_ratio=0.02,
            max_component_delta_ratio=0.02,
            max_structural_criticality=0.05,
        ),
    )

    assert result["status"] == "ready"
    gate_names = {gate["name"] for gate in result["gates"]}
    assert "missing_bridge_blocks" not in gate_names
    assert "component_count_delta" not in gate_names


def test_handoff_transfer_marks_strongly_attenuated_improvement(tmp_path):
    comparison_csv = tmp_path / "comparison.csv"
    with comparison_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dataset",
                "init_delta_adaptive_minus_selected",
                "b20_delta_adaptive_minus_selected",
                "b20_delta_adaptive_minus_no_external",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": "ais2klinik",
            "init_delta_adaptive_minus_selected": "-149.97203989232366",
            "b20_delta_adaptive_minus_selected": "-7.952186034715851",
            "b20_delta_adaptive_minus_no_external": "15893.146366428597",
        })

    rows = cert.evaluate_handoff_transfer_csv(
        comparison_csv,
        attenuation_ratio_threshold=0.2,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "ais2klinik"
    assert row["transfer_status"] == "attenuated"
    assert row["reference_gap_status"] == "worse_than_reference"
    assert math.isclose(row["transfer_ratio"], 0.05302445736168775)


def test_load_report_json_infers_dataset_from_parent_directory(tmp_path):
    run_dir = tmp_path / "runs" / "ais2klinik"
    run_dir.mkdir(parents=True)
    report_path = run_dir / "report.json"
    report_path.write_text('{"relay_scheduler": "pose_summary_budget"}\n')

    report = cert.load_report_json(report_path)

    assert report["dataset"] == "ais2klinik"
    assert report["source_path"] == str(report_path)


def test_load_report_json_infers_dataset_from_selector_result_filename(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    report_path = run_dir / "ais2klinik_projected_merit.json"
    report_path.write_text('{"relay_scheduler": "none"}\n')

    report = cert.load_report_json(report_path)

    assert report["dataset"] == "ais2klinik"
    assert report["source_path"] == str(report_path)


def test_report_certificate_can_gate_linear_solve_residuals():
    report = {
        "dataset": "toy",
        "topology_separator_coverage": 1.0,
        "separator_residual_mass_coverage": 1.0,
        "separator_normal_summary_payload_coverage": 1.0,
        "separator_normal_summary_missing_bridge_block_edges": 0,
        "separator_normal_summary_component_count_delta": 0,
        "separator_normal_summary_structural_criticality": 0.0,
        "separator_normal_summary_block_graph": {
            "full_block_edge_count": 100,
        },
        "distributed_stats": {
            "rotation_stats": {"final_normal_residual": 1.5},
            "translation_stats": {"final_normal_residual": 250.0},
        },
    }

    result = cert.evaluate_report_certificate(
        report,
        cert.CertificateThresholds(
            max_rotation_normal_residual=2.0,
            max_translation_normal_residual=200.0,
        ),
    )

    failed_gate_names = {
        gate["name"] for gate in result["gates"] if not gate["passed"]
    }
    assert "rotation_normal_residual" not in failed_gate_names
    assert "translation_normal_residual" in failed_gate_names
    assert result["status"] == "not_ready"


def test_metric_alignment_flags_residual_cost_conflict():
    reports = [
        {
            "dataset": "toy",
            "variant": "size",
            "distributed_total_cost": 100.0,
            "distributed_stats": {
                "rotation_stats": {"final_normal_residual": 70.0},
                "translation_stats": {"final_normal_residual": 600.0},
            },
        },
        {
            "dataset": "toy",
            "variant": "energy",
            "distributed_total_cost": 120.0,
            "distributed_stats": {
                "rotation_stats": {"final_normal_residual": 60.0},
                "translation_stats": {"final_normal_residual": 590.0},
            },
        },
    ]

    alignments = cert.evaluate_metric_alignment_reports(
        reports,
        residual_metric="rotation_normal_residual",
    )

    assert len(alignments) == 1
    item = alignments[0]
    assert item["dataset"] == "toy"
    assert item["status"] == "residual_cost_conflict"
    assert item["best_cost_variants"] == ["size"]
    assert item["best_residual_variants"] == ["energy"]
