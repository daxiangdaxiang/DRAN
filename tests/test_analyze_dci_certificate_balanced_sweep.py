from scripts import analyze_dci_certificate_balanced_sweep as balanced
from scripts import analyze_dci_handoff_certificate as cert


def _report(name, comm, cost, topology, payload, rot_res, trans_res):
    return {
        "dataset": "toy",
        "name": name,
        "relay_byte_budget_mb": None if name == "unbounded" else comm,
        "actual_dci_comm_mb": comm,
        "full_graph_distributed_total_cost": cost,
        "topology_separator_coverage": topology,
        "separator_residual_mass_coverage": topology,
        "separator_normal_summary_payload_coverage": payload,
        "separator_normal_summary_missing_bridge_block_edges": 0 if payload == 1.0 else 3,
        "separator_normal_summary_component_count_delta": 0 if payload == 1.0 else 3,
        "separator_normal_summary_structural_criticality": 0.0 if payload == 1.0 else 0.1,
        "separator_normal_summary_block_graph": {"full_block_edge_count": 100},
        "distributed_stats": {
            "rotation_stats": {"final_normal_residual": rot_res},
            "translation_stats": {"final_normal_residual": trans_res},
        },
    }


def test_certificate_balanced_sweep_selects_min_comm_ready_report():
    reports = [
        _report("structural_fail", 9.0, 90.0, 0.96, 0.97, 1.0, 100.0),
        _report("solve_fail", 10.0, 80.0, 1.0, 1.0, 3.0, 250.0),
        _report("ready_expensive", 26.0, 50.0, 1.0, 1.0, 1.0, 150.0),
        _report("ready_cheaper", 20.0, 60.0, 1.0, 1.0, 1.5, 180.0),
    ]

    rows, selected = balanced.evaluate_certificate_balanced_reports(
        reports,
        cert.CertificateThresholds(
            min_topology_coverage=0.95,
            min_residual_mass_coverage=0.99,
            min_payload_coverage=0.98,
            max_missing_bridge_ratio=0.02,
            max_component_delta_ratio=0.02,
            max_structural_criticality=0.05,
            max_rotation_normal_residual=2.0,
            max_translation_normal_residual=200.0,
        ),
    )

    assert [row["status"] for row in rows] == [
        "not_ready",
        "not_ready",
        "ready",
        "ready",
    ]
    assert selected["label"] == "ready_cheaper"
    assert selected["actual_dci_comm_mb"] == 20.0
    solve_fail = rows[1]
    assert "rotation_normal_residual" in solve_fail["failed_gates"]
    assert "translation_normal_residual" in solve_fail["failed_gates"]


def test_certificate_balanced_markdown_reports_selected_candidate():
    rows = [
        {
            "label": "a",
            "status": "not_ready",
            "actual_dci_comm_mb": 1.0,
            "cost": 10.0,
            "rotation_normal_residual": 3.0,
            "translation_normal_residual": 300.0,
            "failed_gates": ["rotation_normal_residual"],
        },
        {
            "label": "b",
            "status": "ready",
            "actual_dci_comm_mb": 2.0,
            "cost": 5.0,
            "rotation_normal_residual": 1.0,
            "translation_normal_residual": 100.0,
            "failed_gates": [],
        },
    ]

    markdown = balanced.render_certificate_balanced_markdown(rows, rows[1])

    assert "selected configuration: `b`" in markdown
    assert "| b | ready | 2.000000 | 5.000000 |" in markdown
