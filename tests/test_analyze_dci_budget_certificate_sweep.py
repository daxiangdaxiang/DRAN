from scripts import analyze_dci_budget_certificate_sweep as sweep
from scripts import analyze_dci_handoff_certificate as cert
from tests.test_budgeted_dci import info2, pose2
from scripts.evaluate_pgo import Edge


def test_budget_sweep_finds_first_ready_budget_for_relayed_bridge():
    edges = [
        Edge(0, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 2: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        2: pose2(1.0, 0.0, 0.0),
    }

    rows, first_ready = sweep.evaluate_budget_sweep_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_budgets_mb=[0.0, None],
        thresholds=cert.CertificateThresholds(
            min_topology_coverage=1.0,
            min_residual_mass_coverage=1.0,
            min_payload_coverage=1.0,
            max_missing_bridge_ratio=0.0,
            max_component_delta_ratio=0.0,
            max_structural_criticality=0.0,
        ),
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert [row["certificate_status"] for row in rows] == ["not_ready", "ready"]
    assert rows[0]["topology_separator_coverage"] == 0.0
    assert rows[0]["separator_normal_summary_payload_coverage"] == 0.0
    assert rows[1]["topology_separator_coverage"] == 1.0
    assert rows[1]["separator_normal_summary_payload_coverage"] == 1.0
    assert first_ready["budget_mb"] is None


def test_budget_sweep_renders_markdown_with_first_ready_budget():
    rows = [
        {
            "dataset": "toy",
            "relay_scheduler": "pose_summary_budget",
            "budget_mb": 0.0,
            "certificate_status": "not_ready",
            "topology_separator_coverage": 0.0,
            "separator_normal_summary_payload_coverage": 0.0,
            "missing_bridge_block_ratio": 1.0,
            "component_count_delta_ratio": 1.0,
            "structural_criticality": 1.0,
            "relay_scheduler_selected_relay_edges": 0,
            "relay_scheduler_predicted_selected_relay_mb": 0.0,
        },
        {
            "dataset": "toy",
            "relay_scheduler": "pose_summary_budget",
            "budget_mb": None,
            "certificate_status": "ready",
            "topology_separator_coverage": 1.0,
            "separator_normal_summary_payload_coverage": 1.0,
            "missing_bridge_block_ratio": 0.0,
            "component_count_delta_ratio": 0.0,
            "structural_criticality": 0.0,
            "relay_scheduler_selected_relay_edges": 1,
            "relay_scheduler_predicted_selected_relay_mb": 0.1,
        },
    ]

    markdown = sweep.render_budget_sweep_markdown(rows, rows[1])

    assert "first ready budget: `unbounded`" in markdown
    assert "| toy | unbounded | ready |" in markdown
