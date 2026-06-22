import math
from pathlib import Path

import numpy as np

from scripts import analyze_shape_drift_oracle as oracle
from scripts import run_budgeted_dci as bc_dci
from scripts.evaluate_pgo import Edge


def pose2(x, y, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    mat = np.eye(4, dtype=float)
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[1, 0] = s
    mat[1, 1] = c
    mat[0, 3] = x
    mat[1, 3] = y
    return mat


def info2(trans_weight, rot_weight):
    return np.diag([float(trans_weight), float(trans_weight), float(rot_weight)])


def test_budgeted_dci_skips_candidate_before_solve_when_value_bound_is_low(tmp_path: Path):
    edges = [Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2)]
    robot_of = {0: 0, 1: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=1e9,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )
    parsed = oracle.parse_manual_matrix_estimate(out)

    assert report["pre_gate_decision"] == "skip"
    assert report["candidate_solved"] is False
    assert report["selected"] == "baseline_gauge_selector"
    assert report["actual_dci_comm_mb"] == 0.0
    assert report["distributed_total_cost"] is None
    assert np.allclose(parsed[1], baseline[1], atol=1e-12)


def test_budgeted_dci_runs_candidate_and_accepts_certified_improvement(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(3.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )
    parsed = oracle.parse_manual_matrix_estimate(out)

    assert report["pre_gate_decision"] == "run"
    assert report["candidate_solved"] is True
    assert report["selected"] == "distributed_chordal_pcg"
    assert report["handoff_gate_mode"] == "evidence_cost"
    assert report["handoff_gate_accept"] is True
    assert report["evidence_global_consensus_accept"] is True
    assert report["global_consensus_accept"] is True
    assert report["actual_dci_comm_mb"] > 0.0
    assert report["distributed_total_cost"] < report["baseline_total_cost"]
    assert np.allclose(parsed[2][:2, 3], np.array([2.0, 0.0]), atol=1e-8)


def test_budgeted_dci_full_graph_handoff_gate_rejects_evidence_only_gain(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(10.0, 0.0, 0.0), info2(100.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(2.0, 0.0, 0.0),
        2: pose2(12.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        handoff_gate_mode="full_graph_cost",
    )
    parsed = oracle.parse_manual_matrix_estimate(out)

    assert report["certificate_total_delta"] < 0.0
    assert report["handoff_gate_mode"] == "full_graph_cost"
    assert report["handoff_gate_accept"] is False
    assert report["global_consensus_accept"] is False
    assert report["selected"] == "baseline_gauge_selector"
    assert report["full_graph_handoff_delta"] > 0.0
    assert np.allclose(parsed[1][:2, 3], baseline[1][:2, 3], atol=1e-12)


def test_mean_edge_residual_score_normalizes_total_cost_by_graph_size():
    baseline_cost = {
        "total_cost": 20.0,
        "total_edges": 10,
        "separator_edges": 2,
    }

    raw = bc_dci.compute_value_score(
        baseline_cost=baseline_cost,
        predicted_comm_mb=5.0,
        mode="raw_cost",
    )
    normalized = bc_dci.compute_value_score(
        baseline_cost=baseline_cost,
        predicted_comm_mb=5.0,
        mode="mean_edge_residual",
    )

    assert raw == 4.0
    assert normalized == 0.8


def test_separator_private_contrast_score_uses_relative_interface_residual():
    baseline_cost = {
        "private_cost": 10.0,
        "private_edges": 10,
        "separator_cost": 6.0,
        "separator_edges": 2,
    }

    score = bc_dci.compute_value_score(
        baseline_cost=baseline_cost,
        predicted_comm_mb=4.0,
        mode="separator_private_contrast",
    )

    assert score == 1.0


def test_separator_private_contrast_score_clips_uninformative_interfaces():
    baseline_cost = {
        "private_cost": 10.0,
        "private_edges": 10,
        "separator_cost": 1.0,
        "separator_edges": 2,
    }

    score = bc_dci.compute_value_score(
        baseline_cost=baseline_cost,
        predicted_comm_mb=4.0,
        mode="separator_private_contrast",
    )

    assert score == 0.0


def test_candidate_precert_separator_private_ratio_detects_cross_robot_specific_risk():
    weak = {
        "private_cost": 10.0,
        "private_edges": 10,
        "separator_cost": 11.0,
        "separator_edges": 10,
    }
    strong = {
        "private_cost": 10.0,
        "private_edges": 10,
        "separator_cost": 25.0,
        "separator_edges": 10,
    }

    weak_report = bc_dci.compute_candidate_precert(
        baseline_cost=weak,
        mode="separator_private_ratio",
        min_sep_private_ratio=1.2,
    )
    strong_report = bc_dci.compute_candidate_precert(
        baseline_cost=strong,
        mode="separator_private_ratio",
        min_sep_private_ratio=1.2,
    )

    assert weak_report["candidate_precert_accept"] is False
    assert weak_report["candidate_precert_sep_private_ratio"] == 1.1
    assert strong_report["candidate_precert_accept"] is True
    assert strong_report["candidate_precert_sep_private_ratio"] == 2.5


def test_budgeted_dci_uses_selected_value_score_mode_for_pre_gate(tmp_path: Path):
    edges = [Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2)]
    robot_of = {0: 0, 1: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=1.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        value_score_mode="mean_edge_residual",
        omitted_force_correction_curvature_rank_scheduler="projected_residual",
        omitted_force_correction_curvature_rank_max_condition=42.0,
        omitted_force_correction_curvature_rank_condition_policy="median_mad",
        omitted_force_correction_curvature_rank_condition_mad_scale=1.5,
    )

    assert report["value_score_mode"] == "mean_edge_residual"
    assert report["value_score"] < 1.0
    assert report["pre_gate_decision"] == "skip"
    assert report["omitted_force_correction_curvature_rank_scheduler"] == (
        "projected_residual"
    )
    assert report["omitted_force_correction_curvature_rank_max_condition"] == 42.0
    assert report[
        "omitted_force_correction_curvature_rank_condition_policy"
    ] == "median_mad"
    assert report[
        "omitted_force_correction_curvature_rank_condition_mad_scale"
    ] == 1.5
    report["graph"] = "unit"
    row = bc_dci._summary_row(report)
    assert row["omitted_force_correction_curvature_rank_scheduler"] == (
        "projected_residual"
    )
    assert row["omitted_force_correction_curvature_rank_max_condition"] == 42.0
    assert row[
        "omitted_force_correction_curvature_rank_condition_policy"
    ] == "median_mad"
    assert row[
        "omitted_force_correction_curvature_rank_condition_mad_scale"
    ] == 1.5


def test_budgeted_dci_precert_skips_candidate_before_chordal_solve(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(2.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        candidate_precert_mode="separator_private_ratio",
        candidate_precert_min_sep_private_ratio=1.2,
    )

    assert report["pre_gate_decision"] == "skip_precert"
    assert report["candidate_solved"] is False
    assert report["candidate_precert_accept"] is False
    assert report["candidate_precert_sep_private_ratio"] < 1.2
    assert report["actual_dci_comm_mb"] == 0.0


def test_topology_filter_keeps_private_edges_and_available_separator_edges():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 2}

    filtered = bc_dci.filter_edges_for_topology(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1)},
    )

    assert [(edge.i, edge.j) for edge in filtered] == [(0, 1), (1, 2)]


def test_load_topology_pairs_accumulates_contiguous_round_window(tmp_path: Path):
    topology = tmp_path / "topology_edges.csv"
    topology.write_text(
        "round,src,dst,weight\n"
        "0,0,1,0.5\n"
        "1,1,2,0.5\n"
        "2,2,3,0.5\n",
        encoding="utf-8",
    )

    assert bc_dci.load_topology_pairs(topology, round_index=0, round_window=2) == {
        (0, 1),
        (1, 2),
    }
    assert bc_dci.load_topology_pairs(topology, round_index=1, round_window=2) == {
        (1, 2),
        (2, 3),
    }


def test_load_topology_pairs_rejects_nonpositive_window(tmp_path: Path):
    topology = tmp_path / "topology_edges.csv"
    topology.write_text("round,src,dst,weight\n0,0,1,0.5\n", encoding="utf-8")

    try:
        bc_dci.load_topology_pairs(topology, round_index=0, round_window=0)
    except ValueError as exc:
        assert "round_window" in str(exc)
    else:
        raise AssertionError("expected round_window validation to reject zero")


def test_multi_round_union_does_not_double_count_repeated_robot_pair(tmp_path: Path):
    topology = tmp_path / "topology_edges.csv"
    topology.write_text(
        "round,src,dst,weight\n"
        "0,0,1,0.5\n"
        "1,1,0,0.5\n",
        encoding="utf-8",
    )

    assert bc_dci.load_topology_pairs(topology, round_index=0, round_window=2) == {
        (0, 1),
    }


def test_budgeted_dci_runs_when_multi_round_union_satisfies_coverage(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(3.0, 0.0, 0.0),
        3: pose2(5.0, 0.0, 0.0),
    }
    topology = tmp_path / "topology_edges.csv"
    topology.write_text(
        "round,src,dst,weight\n"
        "0,0,1,0.5\n"
        "1,1,2,0.5\n",
        encoding="utf-8",
    )
    available = bc_dci.load_topology_pairs(topology, round_index=0, round_window=2)
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs=available,
        min_topology_separator_coverage=1.0,
    )

    assert report["pre_gate_decision"] == "run"
    assert report["candidate_solved"] is True
    assert report["topology_separator_edges"] == 2
    assert report["full_separator_edges"] == 2
    assert report["topology_separator_coverage"] == 1.0


def test_relay_evidence_requires_explicit_hop_budget_for_indirect_separator():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 2}
    topology = {(0, 1), (1, 2)}

    direct_edges, direct_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs=topology,
        max_relay_hops=0,
    )
    relay_edges, relay_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs=topology,
        max_relay_hops=1,
    )

    assert direct_edges == []
    assert direct_evidence["covered_separator_edges"] == 0
    assert relay_edges == edges
    assert relay_evidence["covered_separator_edges"] == 1
    assert relay_evidence["relayed_separator_edges"] == 1
    assert relay_evidence["relay_extra_hop_count"] == 1
    assert relay_evidence["separator_paths"] == [
        {"edge_index": 0, "robots": [0, 1, 2], "path_hops": 2}
    ]


def test_topology_routed_normal_summary_counts_path_hops():
    report = bc_dci.estimate_topology_routed_normal_summary_communication(
        logical_payload_bytes=120,
        separator_paths=[
            {"edge_index": 0, "robots": [0, 1], "path_hops": 1},
            {"edge_index": 1, "robots": [0, 1, 2], "path_hops": 2},
            {"edge_index": 2, "robots": [0, 1, 2, 3], "path_hops": 3},
        ],
    )

    assert report["model"] == "equal_separator_edge_payload_path_hop_routing"
    assert report["logical_payload_bytes"] == 120
    assert report["separator_path_count"] == 3
    assert report["total_path_hops"] == 6
    assert report["routed_payload_bytes"] == 240
    assert report["extra_relay_payload_bytes"] == 120
    assert report["routed_payload_mb"] == 240 / (1024.0 * 1024.0)


def test_budgeted_dci_counts_relay_extra_hop_communication(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(2.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    direct = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1), (1, 2)},
        min_topology_separator_coverage=1.0,
        max_relay_hops=0,
    )
    relay = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1), (1, 2)},
        min_topology_separator_coverage=1.0,
        max_relay_hops=1,
    )

    assert direct["pre_gate_decision"] == "skip_no_separator_topology"
    assert direct["candidate_solved"] is False
    assert relay["pre_gate_decision"] == "run"
    assert relay["candidate_solved"] is True
    assert relay["relay_extra_hop_count"] == 1
    assert relay["predicted_relay_extra_comm_mb"] == relay["relay_extra_comm_mb"]
    assert relay["relay_extra_comm_mb"] > 0.0
    assert relay["predicted_candidate_comm_mb"] > relay["predicted_communication"]["linear_solve"]["linear_solve_total_estimated_mb"]
    assert relay["actual_dci_comm_mb"] > relay["actual_linear_communication"]["linear_solve_total_estimated_mb"]


def test_relay_scheduler_selects_high_value_relay_edges_under_byte_budget():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 2, 2: 0, 3: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(0.0, 0.0, 0.0),
        3: pose2(1.0, 0.0, 0.0),
    }
    single_edge_mb = bc_dci.estimate_relay_extra_communication(
        rotation_iterations=5,
        translation_iterations=5,
        relay_extra_hop_count=1,
        dimension=2,
    )["relay_extra_exchange_mb"]

    filtered, evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="edge_residual_budget",
        relay_byte_budget_mb=single_edge_mb * 1.01,
        baseline_poses=baseline,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert filtered == [edges[0]]
    assert evidence["covered_separator_edges"] == 1
    assert evidence["relayed_separator_edges"] == 1
    assert evidence["relay_scheduler"] == "edge_residual_budget"
    assert evidence["relay_scheduler_selected_relay_edges"] == 1
    assert evidence["relay_scheduler_skipped_relay_edges"] == 1
    assert evidence["relay_extra_hop_count"] == 1


def test_coverage_relay_scheduler_prefers_more_edges_under_byte_budget():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 3, 2: 0, 3: 2, 4: 1, 5: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(10.0, 0.0, 0.0),
        2: pose2(0.0, 0.0, 0.0),
        3: pose2(1.0, 0.0, 0.0),
        4: pose2(0.0, 0.0, 0.0),
        5: pose2(1.0, 0.0, 0.0),
    }
    single_hop_mb = bc_dci.estimate_relay_extra_communication(
        rotation_iterations=5,
        translation_iterations=5,
        relay_extra_hop_count=1,
        dimension=2,
    )["relay_extra_exchange_mb"]

    residual_filtered, residual_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="edge_residual_budget",
        relay_byte_budget_mb=single_hop_mb * 2.01,
        baseline_poses=baseline,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    coverage_filtered, coverage_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="edge_coverage_budget",
        relay_byte_budget_mb=single_hop_mb * 2.01,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert residual_filtered == [edges[0]]
    assert residual_evidence["covered_separator_edges"] == 1
    assert coverage_filtered == [edges[1], edges[2]]
    assert coverage_evidence["covered_separator_edges"] == 2
    assert coverage_evidence["relay_scheduler"] == "edge_coverage_budget"
    assert coverage_evidence["relay_scheduler_selected_relay_edges"] == 2
    assert coverage_evidence["relay_scheduler_skipped_relay_edges"] == 1
    assert coverage_evidence["relay_extra_hop_count"] == 2


def test_pose_summary_relay_scheduler_reuses_shared_endpoint_state():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 2, 2: 2}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)

    edgewise_filtered, edgewise_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="edge_coverage_budget",
        relay_byte_budget_mb=single_pose_unit_mb * 3.01,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    summary_filtered, summary_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_budget",
        relay_byte_budget_mb=single_pose_unit_mb * 3.01,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert edgewise_filtered == [edges[0]]
    assert edgewise_evidence["covered_separator_edges"] == 1
    assert summary_filtered == edges
    assert summary_evidence["covered_separator_edges"] == 2
    assert summary_evidence["relay_scheduler_selected_relay_edges"] == 2
    assert summary_evidence["relay_scheduler_pose_summary_units"] == 3
    assert (
        summary_evidence["relay_scheduler_predicted_selected_relay_mb"]
        <= single_pose_unit_mb * 3.01
    )


def test_pose_summary_criticality_scheduler_prefers_bridge_block_under_budget():
    edges = [
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    summary_filtered, summary_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    critical_filtered, critical_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_criticality_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert summary_filtered == [edges[0]]
    assert critical_filtered == [edges[3]]
    assert critical_evidence["relay_scheduler_selected_edge_indices"] == [3]
    assert critical_evidence["relay_scheduler_structural_bridge_gain"] == 1
    assert critical_evidence["relay_scheduler_structural_component_gain"] == 1


def test_pose_summary_weighted_criticality_prefers_high_information_bridge():
    edges = [
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    bridge_only_filtered, bridge_only_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_criticality_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    weighted_filtered, weighted_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_weighted_criticality_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert bridge_only_filtered == [edges[0]]
    assert weighted_filtered == [edges[1]]
    assert weighted_evidence["relay_scheduler_selected_edge_indices"] == [1]
    assert weighted_evidence["relay_scheduler_structural_weighted_gain"] > (
        bridge_only_evidence["relay_scheduler_structural_weighted_gain"]
    )


def test_pose_summary_leverage_prefers_weakly_constrained_bridge():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(2, 6, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(10.0, 10.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    weighted_filtered, weighted_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_weighted_criticality_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    leverage_filtered, leverage_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert weighted_filtered[-1] == edges[2]
    assert leverage_filtered[-1] == edges[3]
    assert leverage_evidence["relay_scheduler_selected_edge_indices"] == [3]
    assert leverage_evidence["relay_scheduler_structural_leverage_gain"] > (
        weighted_evidence["relay_scheduler_structural_leverage_gain"]
    )


def test_pose_summary_guarded_leverage_preserves_measurement_strength():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(2, 6, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(10.0, 10.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    guarded_filtered, guarded_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_guarded_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert guarded_filtered[-1] == edges[2]
    assert guarded_evidence["relay_scheduler_selected_edge_indices"] == [2]
    assert guarded_evidence["relay_scheduler_structural_guarded_leverage_gain"] > 0


def test_pose_summary_guarded_leverage_prefers_weak_endpoint_when_weights_are_close():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(2, 6, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(10.0, 10.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(9.0, 9.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    weighted_filtered, _ = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_weighted_criticality_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    guarded_filtered, guarded_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_guarded_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert weighted_filtered[-1] == edges[2]
    assert guarded_filtered[-1] == edges[3]
    assert guarded_evidence["relay_scheduler_selected_edge_indices"] == [3]


def test_pose_summary_normal_leverage_prefers_missing_selected_normal_blocks():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(50.0, 50.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(50.0, 50.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(11.0, 11.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(3.0, 3.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    weighted_filtered, _ = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_weighted_criticality_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    normal_filtered, normal_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert weighted_filtered[-1] == edges[2]
    assert normal_filtered[-1] == edges[3]
    assert normal_evidence["relay_scheduler_selected_edge_indices"] == [3]
    assert normal_evidence["relay_scheduler_structural_normal_leverage_gain"] > 0


def test_pose_summary_normal_leverage_preserves_large_measurement_strength():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(50.0, 50.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(50.0, 50.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(3.0, 3.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    normal_filtered, normal_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert normal_filtered[-1] == edges[2]
    assert normal_evidence["relay_scheduler_selected_edge_indices"] == [2]


def test_pose_summary_normal_leverage_does_not_compute_schur_scores():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 4, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(11.0, 11.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    _, normal_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert normal_evidence["relay_scheduler_selected_edge_indices"] == [4]
    assert normal_evidence["relay_scheduler_schur_eval_count"] == 0
    assert normal_evidence["relay_scheduler_structural_schur_leverage_gain"] == 0.0


def test_pose_summary_normal_density_prefers_gain_per_incremental_byte():
    edges = [
        Edge(10, 11, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(20, 21, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), info2(4.0, 4.0), 2),
        Edge(0, 3, pose2(1.0, 0.0, 0.0), info2(40.0, 40.0), 2),
    ]
    robot_of = {0: 0, 2: 2, 3: 3, 10: 1, 11: 1, 20: 2, 21: 2}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_long_edge_only = single_pose_unit_mb * 4.01

    normal_filtered, normal_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_long_edge_only,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    density_filtered, density_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_density_budget",
        relay_byte_budget_mb=budget_for_long_edge_only,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert normal_filtered[-1] == edges[2]
    assert normal_evidence["relay_scheduler_selected_edge_indices"] == [2]
    assert density_filtered[-1] == edges[3]
    assert density_evidence["relay_scheduler_selected_edge_indices"] == [3]
    assert density_evidence["relay_scheduler_schur_eval_count"] == 0


def test_pose_summary_bridge_normal_prefers_bridge_over_high_gain_nonbridge():
    edges = [
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(1.0, 0.0, 0.0), info2(50.0, 50.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    density_filtered, density_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_density_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    bridge_filtered, bridge_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_bridge_normal_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert density_filtered[-1] == edges[2]
    assert density_evidence["relay_scheduler_selected_edge_indices"] == [2]
    assert bridge_filtered[-1] == edges[3]
    assert bridge_evidence["relay_scheduler_selected_edge_indices"] == [3]
    assert bridge_evidence["relay_scheduler_structural_bridge_gain"] == 1
    assert bridge_evidence["relay_scheduler_schur_eval_count"] == 0


def test_adaptive_bridge_density_policy_requires_nonlower_bridge_gain():
    normal = {
        "relay_scheduler_structural_bridge_gain": 5,
        "relay_scheduler_structural_normal_leverage_gain": 10.0,
    }

    assert bc_dci.choose_adaptive_bridge_density_subscheduler(
        normal,
        {
            "relay_scheduler_structural_bridge_gain": 4,
            "relay_scheduler_structural_normal_leverage_gain": 20.0,
        },
    ) == "pose_summary_normal_leverage_budget"
    assert bc_dci.choose_adaptive_bridge_density_subscheduler(
        normal,
        {
            "relay_scheduler_structural_bridge_gain": 5,
            "relay_scheduler_structural_normal_leverage_gain": 20.0,
        },
    ) == "pose_summary_normal_density_budget"
    assert bc_dci.choose_adaptive_bridge_density_subscheduler(
        normal,
        {
            "relay_scheduler_structural_bridge_gain": 6,
            "relay_scheduler_structural_normal_leverage_gain": 9.0,
        },
    ) == "pose_summary_normal_leverage_budget"


def test_pose_summary_adaptive_bridge_density_uses_density_when_bridge_safe():
    edges = [
        Edge(10, 11, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(20, 21, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), info2(4.0, 4.0), 2),
        Edge(0, 3, pose2(1.0, 0.0, 0.0), info2(40.0, 40.0), 2),
    ]
    robot_of = {0: 0, 2: 2, 3: 3, 10: 1, 11: 1, 20: 2, 21: 2}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_long_edge_only = single_pose_unit_mb * 4.01

    adaptive_filtered, adaptive_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_adaptive_bridge_density_budget",
        relay_byte_budget_mb=budget_for_long_edge_only,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert adaptive_filtered[-1] == edges[3]
    assert adaptive_evidence["relay_scheduler_selected_edge_indices"] == [3]
    assert adaptive_evidence["relay_scheduler_adaptive_selected_subscheduler"] == (
        "pose_summary_normal_density_budget"
    )
    assert adaptive_evidence["relay_scheduler_schur_eval_count"] == 0


def test_pose_summary_jacobi_leverage_prefers_weak_selected_normal_diagonal():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(2, 6, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 8, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(11.0, 11.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    normal_filtered, _ = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    jacobi_filtered, jacobi_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_jacobi_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert normal_filtered[-1] == edges[4]
    assert jacobi_filtered[-1] == edges[5]
    assert jacobi_evidence["relay_scheduler_selected_edge_indices"] == [5]
    assert jacobi_evidence["relay_scheduler_structural_jacobi_leverage_gain"] > 0


def test_pose_summary_jacobi_leverage_preserves_large_measurement_strength():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(2, 6, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 8, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(3.0, 3.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    jacobi_filtered, jacobi_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2), (1, 3)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_jacobi_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert jacobi_filtered[-1] == edges[4]
    assert jacobi_evidence["relay_scheduler_selected_edge_indices"] == [4]


def test_pose_summary_interface_leverage_prefers_weak_selected_path():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    normal_filtered, _ = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    interface_filtered, interface_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_interface_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert normal_filtered[-1] == edges[4]
    assert interface_filtered[-1] == edges[5]
    assert interface_evidence["relay_scheduler_selected_edge_indices"] == [5]
    assert (
        interface_evidence["relay_scheduler_structural_interface_leverage_gain"]
        > 0
    )


def test_pose_summary_interface_leverage_preserves_large_measurement_strength():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(10000.0, 10000.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(3.0, 3.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    interface_filtered, interface_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_interface_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert interface_filtered[-1] == edges[4]
    assert interface_evidence["relay_scheduler_selected_edge_indices"] == [4]


def test_pose_summary_schur_leverage_detects_parallel_interface_redundancy():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    normal_filtered, _ = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    schur_filtered, schur_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert normal_filtered[-1] == edges[6]
    assert schur_filtered[-1] == edges[7]
    assert schur_evidence["relay_scheduler_selected_edge_indices"] == [7]
    assert schur_evidence["relay_scheduler_structural_schur_leverage_gain"] > 0


def test_pose_summary_schur_leverage_preserves_large_measurement_strength():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(3.0, 3.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    schur_filtered, schur_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert schur_filtered[-1] == edges[6]
    assert schur_evidence["relay_scheduler_selected_edge_indices"] == [6]


def test_pose_summary_guarded_schur_rejects_lower_selected_normal_gain():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(20.0, 20.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(20.0, 20.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(13.0, 13.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    schur_filtered, _ = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )
    guarded_filtered, guarded_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_guarded_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert schur_filtered[-1] == edges[7]
    assert guarded_filtered[-1] == edges[6]
    assert guarded_evidence["relay_scheduler_selected_edge_indices"] == [6]
    assert (
        guarded_evidence[
            "relay_scheduler_structural_guarded_schur_leverage_gain"
        ] > 0
    )


def test_pose_summary_guarded_schur_uses_schur_for_safe_ties():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    guarded_filtered, guarded_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_guarded_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert guarded_filtered[-1] == edges[7]
    assert guarded_evidence["relay_scheduler_selected_edge_indices"] == [7]
    assert guarded_evidence["relay_scheduler_schur_eval_count"] == 2


def test_pose_summary_lazy_guarded_schur_rejects_lower_selected_normal_without_schur_eval():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(20.0, 20.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(20.0, 20.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(13.0, 13.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    lazy_filtered, lazy_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_lazy_guarded_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert lazy_filtered[-1] == edges[6]
    assert lazy_evidence["relay_scheduler_selected_edge_indices"] == [6]
    assert lazy_evidence["relay_scheduler_schur_eval_count"] == 0
    assert lazy_evidence["relay_scheduler_wall_sec"] >= 0.0


def test_pose_summary_lazy_guarded_schur_uses_schur_for_safe_ties():
    edges = [
        Edge(1, 5, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(5, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 6, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(6, 2, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(3, 7, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(7, 4, pose2(1.0, 0.0, 0.0), info2(100.0, 100.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(12.0, 12.0), 2),
    ]
    robot_of = {1: 0, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 1}
    single_pose_unit_mb = (
        5 * 2 * 2 * 8 + 5 * 2 * 8
    ) / (1024.0 * 1024.0)
    budget_for_one_edge = single_pose_unit_mb * 2.01

    lazy_filtered, lazy_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_lazy_guarded_schur_leverage_budget",
        relay_byte_budget_mb=budget_for_one_edge,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    assert lazy_filtered[-1] == edges[7]
    assert lazy_evidence["relay_scheduler_selected_edge_indices"] == [7]
    assert lazy_evidence["relay_scheduler_schur_eval_count"] == 2
    assert (
        lazy_evidence[
            "relay_scheduler_structural_lazy_guarded_schur_leverage_gain"
        ] > 0
    )


def test_pose_summary_budget_uses_boundary_pose_linear_accounting(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_budget",
    )

    assert report["candidate_solved"] is True
    assert report["predicted_communication"]["linear_solve"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["actual_linear_communication"]["boundary_pose_unit_count"] == 3
    assert report["actual_linear_communication"]["separator_exchange_bytes"] < (
        report["distributed_stats"]["communication_estimate"]
        ["separator_exchange_bytes"]
    )
    assert report["normal_equation_summary_communication"]["model"] == (
        "separator_dense_block_normal_equation_summary"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0
    expected_accounted_mb = (
        report["actual_linear_communication"]["linear_solve_total_estimated_mb"] +
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["certificate_communication"]["residual_delta_consensus_mb"]
    )
    assert abs(report["actual_dci_comm_mb"] - expected_accounted_mb) < 1e-12


def test_guarded_leverage_budget_uses_boundary_pose_linear_accounting(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_guarded_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0


def test_normal_leverage_budget_uses_boundary_pose_linear_accounting(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_normal_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0


def test_jacobi_leverage_budget_uses_boundary_pose_linear_accounting(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_jacobi_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0


def test_interface_leverage_budget_uses_boundary_pose_linear_accounting(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_interface_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0


def test_schur_leverage_budget_uses_boundary_pose_linear_accounting(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_schur_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0


def test_guarded_schur_leverage_budget_uses_boundary_pose_linear_accounting(
    tmp_path: Path,
):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_guarded_schur_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0


def test_lazy_guarded_schur_leverage_budget_uses_boundary_pose_linear_accounting(
    tmp_path: Path,
):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=1.0,
        relay_scheduler="pose_summary_lazy_guarded_schur_leverage_budget",
    )

    assert report["candidate_solved"] is True
    assert report["actual_linear_communication"]["model"] == (
        "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles"
    )
    assert report["normal_equation_summary_comm_mb"] > 0.0
    assert report["relay_evidence"]["relay_scheduler_schur_eval_count"] == 0
    assert report["relay_evidence"]["relay_scheduler_wall_sec"] >= 0.0


def test_budgeted_dci_can_use_normal_summary_candidate_solver(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(3.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        candidate_solver="normal_summary",
    )

    assert report["candidate_solved"] is True
    assert report["global_consensus_accept"] is True
    assert report["distributed_stats"]["method"] == (
        "summary_chordal_normal_equation_solve"
    )
    assert report["candidate_solver"] == "normal_summary"


def test_evidence_mass_coverage_detects_low_residual_topology_subset(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 0, 3: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(0.0, 0.0, 0.0),
        3: pose2(10.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=1e9,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=0.0,
    )

    assert report["topology_separator_coverage"] == 0.5
    assert report["separator_residual_mass_coverage"] < 0.1
    assert report["missing_separator_residual_cost"] > 0.0
    assert report["separator_normal_summary_payload_coverage"] < 1.0
    assert report["missing_separator_normal_summary_payload_bytes"] > 0


def test_budgeted_dci_reports_missing_normal_summary_bridge_blocks(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=1e9,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=0.0,
    )

    assert report["candidate_solved"] is False
    assert report["topology_separator_coverage"] == 0.5
    assert report["separator_normal_summary_missing_block_edges"] > 0
    assert report["separator_normal_summary_missing_bridge_block_edges"] > 0
    assert report["separator_normal_summary_component_count_delta"] > 0
    assert report["separator_normal_summary_structural_criticality"] > 0.0


def test_budgeted_dci_uses_relay_scheduler_budget_in_runner(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 1, pose2(0.5, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(3.0, 0.0, 0.0),
    }
    single_edge_mb = bc_dci.estimate_relay_extra_communication(
        rotation_iterations=5,
        translation_iterations=5,
        relay_extra_hop_count=1,
        dimension=2,
    )["relay_extra_exchange_mb"]
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1), (1, 2)},
        min_topology_separator_coverage=0.5,
        max_relay_hops=1,
        relay_scheduler="edge_residual_budget",
        relay_byte_budget_mb=single_edge_mb * 1.01,
    )

    assert report["pre_gate_decision"] == "run"
    assert report["candidate_solved"] is True
    assert report["topology_separator_edges"] == 1
    assert report["full_separator_edges"] == 2
    assert report["topology_separator_coverage"] == 0.5
    assert report["relay_scheduler"] == "edge_residual_budget"
    assert report["relay_scheduler_selected_relay_edges"] == 1
    assert report["relay_scheduler_skipped_relay_edges"] == 1
    assert report["relay_extra_comm_mb"] <= single_edge_mb * 1.01


def test_budgeted_dci_skips_when_topology_has_no_available_separator_edges(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(3.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs=set(),
    )

    assert report["pre_gate_decision"] == "skip_no_separator_topology"
    assert report["candidate_solved"] is False
    assert report["actual_dci_comm_mb"] == 0.0


def test_budgeted_dci_skips_when_separator_topology_coverage_is_too_low(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(3.0, 0.0, 0.0),
        3: pose2(4.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        available_robot_pairs={(0, 1)},
        min_topology_separator_coverage=0.75,
    )

    assert report["pre_gate_decision"] == "skip_insufficient_topology_coverage"
    assert report["candidate_solved"] is False
    assert report["topology_separator_edges"] == 1
    assert report["full_separator_edges"] == 2
    assert report["topology_separator_coverage"] == 0.5


def test_budgeted_dci_can_decouple_relay_cost_iterations_from_solve_iterations(
        tmp_path: Path):
    edges = [
        Edge(0, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 2: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        2: pose2(1.0, 0.0, 0.0),
    }
    low_iter_relay_mb = bc_dci.estimate_relay_extra_communication(
        rotation_iterations=1,
        translation_iterations=1,
        relay_extra_hop_count=1,
        dimension=2,
    )["relay_extra_exchange_mb"]
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=50,
        translation_iterations=50,
        relay_cost_rotation_iterations=1,
        relay_cost_translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        candidate_solver="normal_summary",
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=low_iter_relay_mb * 1.01,
        min_topology_separator_coverage=1.0,
    )

    assert report["pre_gate_decision"] == "run"
    assert report["relay_cost_rotation_iterations"] == 1
    assert report["relay_cost_translation_iterations"] == 1
    assert report["relay_scheduler_selected_relay_edges"] == 1
    assert math.isclose(
        report["relay_scheduler_predicted_selected_relay_mb"],
        low_iter_relay_mb,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_budgeted_dci_compact_continuation_reports_compact_actual_comm(
        tmp_path: Path):
    edges = [
        Edge(0, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 2: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        2: pose2(1.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        relay_cost_rotation_iterations=1,
        relay_cost_translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
        candidate_solver="normal_summary",
        solve_communication_model="compact_continuation_diagnostic",
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    expected_actual = (
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["actual_linear_communication"]["pcg_global_reduction_mb"] +
        report["certificate_communication"]["residual_delta_consensus_mb"]
    )
    assert report["solve_communication_model"] == (
        "compact_continuation_diagnostic"
    )
    assert math.isclose(
        report["actual_dci_comm_mb"],
        expected_actual,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    expected_predicted = (
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["predicted_communication"]["linear_solve"]["pcg_global_reduction_mb"] +
        report["predicted_communication"]["certificate"][
            "residual_delta_consensus_mb"
        ]
    )
    assert math.isclose(
        report["predicted_candidate_comm_mb"],
        expected_predicted,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert report["actual_dci_comm_mb"] < (
        report["actual_linear_communication"]["linear_solve_total_estimated_mb"] +
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"]
    )


def test_budgeted_dci_compact_pcg_uses_solver_level_compact_comm(tmp_path: Path):
    edges = [
        Edge(0, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 2: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        2: pose2(1.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        relay_cost_rotation_iterations=1,
        relay_cost_translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    expected = (
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["actual_linear_communication"]["pcg_global_reduction_mb"] +
        report["certificate_communication"]["residual_delta_consensus_mb"]
    )
    assert report["actual_linear_communication"]["model"] == (
        "normal_summary_compact_pcg_global_reductions"
    )
    assert report["actual_linear_communication"]["separator_exchange_bytes"] == 0
    assert report["solve_communication_model"] == "boundary_pose_exchange"
    assert math.isclose(
        report["actual_dci_comm_mb"],
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    expected_predicted = (
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["predicted_communication"]["linear_solve"]["pcg_global_reduction_mb"] +
        report["predicted_communication"]["certificate"][
            "residual_delta_consensus_mb"
        ]
    )
    assert math.isclose(
        report["predicted_candidate_comm_mb"],
        expected_predicted,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert report["actual_dci_comm_mb"] <= report["predicted_candidate_comm_mb"]


def test_budgeted_dci_interface_schur_uses_interface_comm_accounting(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "interface_schur.txt",
        rotation_iterations=4,
        translation_iterations=4,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    linear = report["actual_linear_communication"]
    expected_bytes = (
        linear["rotation_interface_vector_bytes"] +
        linear["translation_interface_vector_bytes"] +
        linear["pcg_global_reduction_bytes"]
    )

    assert linear["model"] == "interface_schur_vector_exchange_plus_tree_allreduce"
    assert linear["linear_solve_total_estimated_bytes"] == expected_bytes
    assert report["effective_linear_solve_comm_mb"] == (
        linear["linear_solve_total_estimated_mb"]
    )


def test_budgeted_dci_can_select_two_level_interface_schur(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "two_level_interface_schur.txt",
        rotation_iterations=8,
        translation_iterations=8,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert distributed["interface_schur_preconditioner"] == (
        "block_jacobi+coarse"
    )
    assert distributed["rotation_stats"]["schur_preconditioner"] == (
        "block_jacobi+coarse"
    )
    assert distributed["translation_stats"]["schur_preconditioner"] == (
        "block_jacobi+coarse"
    )
    assert distributed["rotation_stats"]["coarse_basis_rank"] > 0
    assert distributed["translation_stats"]["coarse_basis_rank"] > 0


def test_budgeted_dci_can_select_balanced_coarse_interface_schur(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "balanced_coarse.txt",
        rotation_iterations=8,
        translation_iterations=8,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+balanced_coarse",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    linear = report["actual_linear_communication"]
    assert distributed["interface_schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert distributed["rotation_stats"]["schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert distributed["translation_stats"]["schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert distributed["rotation_stats"][
        "balanced_coarse_preconditioner_schur_matvec_count"] > 0
    assert linear["balanced_coarse_preconditioner_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_translation_only_balanced_coarse(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "translation_balanced.txt",
        rotation_iterations=8,
        translation_iterations=8,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi",
        interface_schur_translation_preconditioner=(
            "block_jacobi+balanced_coarse"
        ),
        interface_schur_coarse_basis_mode="component_coordinate",
        interface_schur_coarse_component_limit=1,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    linear = report["actual_linear_communication"]
    assert distributed["interface_schur_rotation_preconditioner"] == (
        "block_jacobi"
    )
    assert distributed["interface_schur_translation_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert distributed["rotation_stats"]["coarse_basis_rank"] == 0
    assert distributed["translation_stats"]["coarse_basis_rank"] > 0
    assert distributed["translation_stats"][
        "balanced_coarse_preconditioner_schur_matvec_count"] > 0
    assert linear["translation_balanced_coarse_preconditioner_schur_matvec_count"] > 0


def test_budgeted_dci_can_split_rotation_translation_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "split_coarse_basis.txt",
        rotation_iterations=8,
        translation_iterations=8,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+balanced_coarse",
        interface_schur_rotation_coarse_basis_mode="component_coordinate",
        interface_schur_rotation_coarse_component_limit=1,
        interface_schur_rotation_coarse_component_selection_mode=(
            "gradient_energy"
        ),
        interface_schur_translation_coarse_basis_mode="robot_coordinate",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert distributed["interface_schur_rotation_coarse_basis_mode"] == (
        "component_coordinate"
    )
    assert distributed["interface_schur_translation_coarse_basis_mode"] == (
        "robot_coordinate"
    )
    assert report["interface_schur_rotation_coarse_basis_mode"] == (
        "component_coordinate"
    )
    assert report["interface_schur_translation_coarse_basis_mode"] == (
        "robot_coordinate"
    )
    assert distributed[
        "interface_schur_rotation_coarse_component_selection_mode"] == (
            "gradient_energy"
        )
    assert report[
        "interface_schur_rotation_coarse_component_selection_mode"] == (
            "gradient_energy"
        )
    assert distributed["rotation_stats"]["component_coordinate_basis_rank"] > 0
    assert distributed["translation_stats"]["component_coordinate_basis_rank"] == 0
    assert distributed["translation_stats"]["robot_coordinate_basis_rank"] > 0


def test_budgeted_dci_can_select_overlap_schwarz_interface_schur(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "overlap_schwarz.txt",
        rotation_iterations=8,
        translation_iterations=8,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="overlap_schwarz",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    linear = report["actual_linear_communication"]
    assert distributed["interface_schur_preconditioner"] == "overlap_schwarz"
    assert distributed["rotation_stats"]["schur_preconditioner"] == (
        "overlap_schwarz")
    assert distributed["translation_stats"]["schur_preconditioner"] == (
        "overlap_schwarz")
    assert linear["overlap_schwarz_setup_interface_vector_mb"] == 0.0
    assert distributed["rotation_stats"][
        "overlap_schwarz_setup_local_schur_column_count"] > 0
    assert report["effective_linear_solve_comm_mb"] == (
        linear["linear_solve_total_estimated_mb"]
    )


def test_budgeted_dci_can_select_coarse_interface_initial_guess(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "coarse_initial_guess.txt",
        rotation_iterations=1,
        translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_coarse_initial_guess"]
    assert distributed["interface_schur_coarse_initial_guess"]
    assert distributed["rotation_stats"]["coarse_initial_guess_used"]
    assert distributed["translation_stats"]["coarse_initial_guess_used"]


def test_budgeted_dci_can_select_component_aware_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "component_coarse.txt",
        rotation_iterations=1,
        translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_basis_mode="robot_component_coordinate",
        interface_schur_coarse_component_limit=2,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_coarse_basis_mode"] == (
        "robot_component_coordinate")
    assert report["interface_schur_coarse_component_limit"] == 2
    assert distributed["interface_schur_coarse_basis_mode"] == (
        "robot_component_coordinate")
    assert distributed["rotation_stats"]["coarse_basis_mode"] == (
        "robot_component_coordinate")
    assert distributed["translation_stats"]["coarse_basis_mode"] == (
        "robot_component_coordinate")
    assert distributed["rotation_stats"]["component_coordinate_basis_rank"] > 0
    assert distributed["translation_stats"]["component_coordinate_basis_rank"] > 0


def test_budgeted_dci_can_select_residual_deflation_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "residual_deflation.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_residual_deflation_rank=1,
        interface_schur_residual_deflation_pilot_iterations=1,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_residual_deflation_rank"] == 1
    assert distributed["interface_schur_residual_deflation_rank"] == 1
    assert distributed["rotation_stats"]["residual_deflation_basis_rank"] > 0
    assert report["actual_linear_communication"][
        "residual_deflation_pilot_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_ritz_interface_coarse_basis(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "ritz.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_rank"] == 1
    assert distributed["interface_schur_ritz_rank"] == 1
    assert distributed["rotation_stats"]["ritz_deflation_basis_rank"] > 0
    assert report["actual_linear_communication"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_generalized_ritz_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "generalized_ritz.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="generalized",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_mode"] == "generalized"
    assert distributed["interface_schur_ritz_mode"] == "generalized"
    assert distributed["rotation_stats"]["ritz_mode"] == "generalized"
    assert report["actual_linear_communication"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_m_orthogonal_lanczos_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "m_orthogonal_lanczos.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="m_orthogonal_lanczos",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_mode"] == "m_orthogonal_lanczos"
    assert distributed["interface_schur_ritz_mode"] == "m_orthogonal_lanczos"
    assert distributed["rotation_stats"]["ritz_mode"] == "m_orthogonal_lanczos"
    assert report["actual_linear_communication"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_harmonic_ritz_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "harmonic_ritz.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="harmonic",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_mode"] == "harmonic"
    assert distributed["interface_schur_ritz_mode"] == "harmonic"
    assert distributed["rotation_stats"]["ritz_mode"] == "harmonic"
    assert report["actual_linear_communication"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_low_harmonic_hybrid_ritz_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "low_harmonic_hybrid.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="low_harmonic_hybrid",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_mode"] == "low_harmonic_hybrid"
    assert distributed["interface_schur_ritz_mode"] == "low_harmonic_hybrid"
    assert distributed["rotation_stats"]["ritz_mode"] == "low_harmonic_hybrid"
    assert distributed["rotation_stats"]["ritz_hybrid_candidate_count"] == 2
    assert report["actual_linear_communication"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_split_rotation_translation_ritz_modes(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "rot_low_trans_harmonic.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_mode"] == "rot_low_trans_harmonic"
    assert distributed["interface_schur_ritz_mode"] == "rot_low_trans_harmonic"
    assert distributed["rotation_stats"]["ritz_mode"] == "preconditioned_operator"
    assert distributed["translation_stats"]["ritz_mode"] == "harmonic"
    assert report["actual_linear_communication"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_use_split_ritz_rank_budget(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "split_ritz_budget.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_translation_ritz_rank=3,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_translation_ritz_probe_iterations=4,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_rank"] == 1
    assert report["interface_schur_translation_ritz_rank"] == 3
    assert distributed["interface_schur_rotation_ritz_rank"] == 1
    assert distributed["interface_schur_translation_ritz_rank"] == 3
    assert distributed["rotation_stats"]["ritz_deflation_requested_rank"] == 1
    assert distributed["translation_stats"]["ritz_deflation_requested_rank"] == 3
    assert distributed["translation_stats"][
        "ritz_probe_requested_iterations"] == 4


def test_budgeted_dci_can_select_translation_ritz_budget_portfolio(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "translation_budget_portfolio.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        interface_schur_translation_budget_candidates=[(1, 2), (2, 4)],
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    portfolio = report["translation_budget_portfolio"]
    distributed = report["distributed_stats"]
    assert portfolio["candidate_count"] == 2
    assert report["interface_schur_translation_ritz_rank"] == (
        portfolio["selected_rank"]
    )
    assert distributed["translation_budget_portfolio"][
        "selected_probe_iterations"
    ] == report["interface_schur_translation_ritz_probe_iterations"]
    assert report["actual_linear_communication"][
        "translation_budget_portfolio_candidate_count"
    ] == 2


def test_budgeted_dci_can_threshold_translation_ritz_rank(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "translation_ritz_threshold.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_translation_ritz_rank=3,
        interface_schur_translation_ritz_probe_iterations=4,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        interface_schur_translation_ritz_rank_selection_mode=(
            "value_threshold"),
        interface_schur_translation_ritz_value_threshold=-1.0,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_translation_ritz_rank"] == 3
    assert report[
        "interface_schur_translation_ritz_rank_selection_mode"
    ] == "value_threshold"
    assert distributed[
        "interface_schur_translation_ritz_rank_selection_mode"
    ] == "value_threshold"
    assert distributed["translation_stats"][
        "ritz_deflation_requested_rank_before_selection"] == 3
    assert distributed["translation_stats"]["ritz_deflation_selected_rank"] == 1
    row = bc_dci._summary_row({**report, "graph": "unit"})
    assert row[
        "interface_schur_translation_ritz_rank_selection_mode"
    ] == "value_threshold"


def test_budgeted_dci_can_use_translation_ritz_energy_capture(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "translation_ritz_energy.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_translation_ritz_rank=3,
        interface_schur_translation_ritz_probe_iterations=4,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        interface_schur_translation_ritz_rank_selection_mode=(
            "energy_capture"),
        interface_schur_translation_ritz_energy_capture_fraction=0.8,
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report[
        "interface_schur_translation_ritz_rank_selection_mode"
    ] == "energy_capture"
    assert report[
        "interface_schur_translation_ritz_energy_capture_fraction"] == 0.8
    assert distributed["translation_stats"][
        "ritz_rank_selection_mode"] == "energy_capture"
    row = bc_dci._summary_row({**report, "graph": "unit"})
    assert row[
        "interface_schur_translation_ritz_energy_capture_fraction"] == 0.8


def test_budgeted_dci_can_report_central_equivalence_diagnostic(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "central_equivalence.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
        central_equivalence_diagnostic=True,
    )

    distributed = report["distributed_stats"]
    assert report["central_equivalence_diagnostic"] is True
    assert distributed["central_equivalence_diagnostic"] is True
    assert "central_equivalence" in distributed["rotation_stats"]
    assert "central_equivalence" in distributed["translation_stats"]


def test_budgeted_dci_can_report_iterative_central_equivalence(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "iterative_equivalence.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
        central_equivalence_iterative_diagnostic_iterations=4,
    )

    distributed = report["distributed_stats"]
    assert report["central_equivalence_iterative_diagnostic_iterations"] == 4
    assert distributed[
        "central_equivalence_iterative_diagnostic_iterations"] == 4
    assert "central_equivalence_iterative" in distributed["rotation_stats"]
    assert distributed["rotation_stats"]["central_equivalence_iterative"][
        "materializes_dense_schur"] is False


def test_budgeted_dci_can_select_ritz_portfolio_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "ritz_portfolio.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="portfolio",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    assert report["interface_schur_ritz_mode"] == "portfolio"
    assert distributed["interface_schur_ritz_mode"] == "portfolio"
    assert distributed["rotation_stats"]["ritz_mode"] == "portfolio"
    assert distributed["rotation_stats"]["ritz_portfolio_candidate_count"] == 4
    assert report["actual_linear_communication"][
        "ritz_portfolio_scoring_interface_vector_mb"] > 0.0


def test_budgeted_dci_can_select_measurement_portfolio_interface_coarse_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "measurement_portfolio.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="measurement_portfolio",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    portfolio = distributed["measurement_portfolio"]
    assert report["interface_schur_ritz_mode"] == "measurement_portfolio"
    assert distributed["interface_schur_ritz_mode"] == "measurement_portfolio"
    assert portfolio["candidate_count"] == 4
    assert portfolio["selected_mode"] in {
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    }
    assert report["actual_linear_communication"][
        "measurement_portfolio_certificate_comm_mb"] > 0.0
    assert report["actual_linear_communication"][
        "measurement_portfolio_candidate_linear_mb"] > 0.0


def test_budgeted_dci_can_select_coarse_measurement_portfolio_interface_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "coarse_measurement_portfolio.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="coarse_measurement_portfolio",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    portfolio = distributed["coarse_measurement_portfolio"]
    assert report["interface_schur_ritz_mode"] == "coarse_measurement_portfolio"
    assert distributed["interface_schur_ritz_mode"] == (
        "coarse_measurement_portfolio"
    )
    assert portfolio["candidate_count"] == 4
    assert portfolio["selected_mode"] in {
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    }
    assert report["actual_linear_communication"][
        "coarse_measurement_portfolio_proxy_mb"] > 0.0
    assert report["actual_linear_communication"][
        "coarse_measurement_portfolio_selected_full_solve_mb"] > 0.0


def test_budgeted_dci_can_select_partial_measurement_portfolio_interface_basis(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    baseline = {
        pose_id: pose2(float(pose_id), 0.0, 0.0)
        for pose_id in robot_of
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=sorted(robot_of),
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "partial_measurement_portfolio.txt",
        rotation_iterations=2,
        translation_iterations=2,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="partial_measurement_portfolio",
        candidate_solver="normal_summary",
        summary_selection_mode="all",
        available_robot_pairs={(0, 1)},
        max_relay_hops=0,
        min_topology_separator_coverage=1.0,
    )

    distributed = report["distributed_stats"]
    portfolio = distributed["partial_measurement_portfolio"]
    assert report["interface_schur_ritz_mode"] == "partial_measurement_portfolio"
    assert distributed["interface_schur_ritz_mode"] == (
        "partial_measurement_portfolio"
    )
    assert portfolio["candidate_count"] == 4
    assert portfolio["selected_mode"] in {
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    }
    assert portfolio["proxy_rotation_iterations"] == 1
    assert portfolio["proxy_translation_iterations"] == 1
    assert report["actual_linear_communication"][
        "partial_measurement_portfolio_proxy_mb"] > 0.0
    assert report["actual_linear_communication"][
        "partial_measurement_portfolio_selected_full_solve_mb"] > 0.0


def test_budgeted_dci_can_account_topology_routed_normal_summary(
        tmp_path: Path):
    edges = [
        Edge(0, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        3: pose2(1.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=out,
        rotation_iterations=5,
        translation_iterations=5,
        relay_cost_rotation_iterations=1,
        relay_cost_translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        normal_summary_communication_model="topology_routed_equal_edge",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    routing = report["topology_routed_normal_summary_communication"]
    assert report["normal_summary_communication_model"] == (
        "topology_routed_equal_edge"
    )
    assert routing["separator_path_count"] == 1
    assert routing["total_path_hops"] == 3
    assert report["normal_equation_summary_comm_mb"] == (
        report["normal_equation_summary_routed_comm_mb"]
    )
    assert report["normal_equation_summary_comm_mb"] == (
        3.0 * report["normal_equation_summary_logical_comm_mb"]
    )
    expected = (
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["actual_linear_communication"]["pcg_global_reduction_mb"] +
        report["certificate_communication"]["residual_delta_consensus_mb"]
    )
    assert math.isclose(
        report["actual_dci_comm_mb"],
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_budgeted_dci_can_use_structural_spanning_summary_selection(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    full = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "full.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )
    selected = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "selected.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    assert selected["summary_selection_mode"] == "structural_spanning"
    assert selected["summary_max_offdiag_block_edges"] == 2
    assert selected["distributed_stats"]["rotation_separator_summary"][
        "summary_selection"
    ]["selected_offdiag_block_edges"] == 2
    assert selected["normal_equation_summary_comm_mb"] < (
        full["normal_equation_summary_comm_mb"]
    )
    assert selected["actual_dci_comm_mb"] < full["actual_dci_comm_mb"]


def test_budgeted_dci_reports_summary_model_representativeness(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    full = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "full.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )
    selected = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "selected.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    assert full["summary_model_representativeness"]["combined"][
        "normal_model_representative"
    ] is True
    assert selected["summary_model_representativeness"]["combined"][
        "omitted_model_residual_norm"
    ] > 0.0
    assert selected["summary_model_representativeness"]["combined"][
        "normal_model_representative"
    ] is False


def test_budgeted_dci_can_use_residual_force_summary_refinement(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    seed = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "seed.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )
    refined = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "refined.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="residual_force_refinement",
        summary_max_offdiag_block_edges=2,
        summary_refinement_max_offdiag_block_edges=3,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    assert refined["summary_selection_mode"] == "residual_force_refinement"
    assert refined["summary_refinement_max_offdiag_block_edges"] == 3
    assert refined["summary_model_representativeness"]["combined"][
        "omitted_model_residual_norm"
    ] < seed["summary_model_representativeness"]["combined"][
        "omitted_model_residual_norm"
    ]
    assert refined["normal_equation_summary_comm_mb"] >= (
        seed["normal_equation_summary_comm_mb"]
    )


def test_budgeted_dci_can_use_omitted_force_correction(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    seed = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "seed.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )
    corrected = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "corrected.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=3,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    assert corrected["summary_selection_mode"] == "omitted_force_correction"
    assert corrected["omitted_force_correction_rounds"] == 3
    assert corrected["summary_model_representativeness"]["combined"][
        "full_model_residual_norm"
    ] < seed["summary_model_representativeness"]["combined"][
        "full_model_residual_norm"
    ]
    force_mb = corrected["omitted_force_correction"]["communication_estimate"][
        "total_omitted_force_payload_mb"
    ]
    assert force_mb > 0.0
    assert corrected["actual_dci_comm_mb"] >= (
        corrected["normal_equation_summary_comm_mb"] + force_mb
    )


def test_budgeted_dci_can_use_omitted_force_correction_portfolio(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "portfolio.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=4,
        omitted_force_correction_portfolio=True,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    portfolio = report["omitted_force_correction_portfolio"]
    assert portfolio["enabled"] is True
    assert portfolio["candidate_count"] == 5
    assert 0 <= portfolio["selected_rounds"] <= 4
    assert report["distributed_stats"][
        "omitted_force_correction_portfolio"
    ] == portfolio


def test_budgeted_dci_summary_row_accepts_disabled_correction_portfolio():
    report = {
        "graph": "unit",
        "selected": "distributed_chordal_pcg",
        "pre_gate_decision": "run",
        "candidate_solved": True,
        "baseline_total_cost": 1.0,
        "distributed_total_cost": 0.5,
        "selected_total_cost": 0.5,
        "certificate_total_delta": -0.5,
        "global_consensus_accept": True,
        "predicted_candidate_comm_mb": 0.1,
        "actual_dci_comm_mb": 0.1,
        "value_upper_bound_cost_per_mb": 5.0,
        "value_score": 5.0,
        "value_score_mode": "raw_cost",
        "value_threshold": 0.0,
        "output_path": "selected.txt",
        "omitted_force_correction_portfolio": None,
        "omitted_force_correction_rank_portfolio": None,
    }

    row = bc_dci._summary_row(report)

    assert row["graph"] == "unit"
    assert row["omitted_force_correction_portfolio_selected_rounds"] is None
    assert row["omitted_force_correction_portfolio_selected_cost"] is None
    assert row["omitted_force_correction_portfolio_last_cost"] is None


def test_omitted_force_correction_portfolio_respects_zero_comm_budget(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "budgeted_portfolio.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=4,
        omitted_force_correction_portfolio=True,
        omitted_force_correction_max_comm_mb=0.0,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_curvature_payload_model="pairwise_projected",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    portfolio = report["omitted_force_correction_portfolio"]
    assert portfolio["comm_budget_enabled"] is True
    assert portfolio["comm_budget_mb"] == 0.0
    assert portfolio["selected_rounds"] == 0
    assert portfolio["selected_correction_comm_mb"] == 0.0
    assert report["omitted_force_correction_effective_comm_mb"] == 0.0


def test_omitted_force_correction_portfolio_respects_marginal_value_gate(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "value_gate_portfolio.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=4,
        omitted_force_correction_portfolio=True,
        omitted_force_correction_min_marginal_cost_per_mb=1e30,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_curvature_payload_model="pairwise_projected",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    portfolio = report["omitted_force_correction_portfolio"]
    assert portfolio["marginal_value_gate_enabled"] is True
    assert portfolio["min_marginal_cost_improvement_per_mb"] == 1e30
    assert portfolio["selected_rounds"] == 0
    assert portfolio["candidate_marginal_value_feasible"][0] is True
    assert any(
        not feasible
        for feasible in portfolio["candidate_marginal_value_feasible"][1:]
    )
    assert len(portfolio["candidate_marginal_cost_improvement_per_mb"]) == 5


def test_omitted_force_correction_portfolio_can_select_curvature_rank(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "rank_portfolio.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=4,
        omitted_force_correction_portfolio=True,
        omitted_force_correction_rank_portfolio=True,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_curvature_payload_model="pairwise_projected",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    rank_portfolio = report["omitted_force_correction_rank_portfolio"]
    selected_rank = rank_portfolio["selected_rank"]
    assert rank_portfolio["enabled"] is True
    assert report["omitted_force_correction_rank_portfolio_requested"] is True
    assert report["omitted_force_correction_rank_portfolio_enabled"] is True
    assert report["omitted_force_correction_requested_curvature_rank"] == 2
    assert rank_portfolio["candidate_count"] == 2
    assert rank_portfolio["rank_candidates"] == [1, 2]
    assert selected_rank in {1, 2}
    best_cost_index = min(
        range(rank_portfolio["candidate_count"]),
        key=lambda index: rank_portfolio["candidate_measurement_costs"][index],
    )
    assert rank_portfolio["selected_index"] == best_cost_index
    assert selected_rank == rank_portfolio["rank_candidates"][best_cost_index]
    assert report["omitted_force_correction_selected_curvature_rank"] == selected_rank
    assert report["omitted_force_correction_portfolio"][
        "selected_curvature_rank"
    ] == selected_rank


def test_rank_portfolio_enabled_reports_actual_activation(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "inactive_rank_portfolio.txt",
        rotation_iterations=5,
        translation_iterations=5,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=1,
        omitted_force_correction_rounds=2,
        omitted_force_correction_portfolio=True,
        omitted_force_correction_rank_portfolio=True,
        omitted_force_correction_curvature_model="block_gershgorin",
        omitted_force_correction_curvature_rank=2,
        available_robot_pairs={(0, 1), (1, 2)},
        max_relay_hops=1,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    assert report["omitted_force_correction_rank_portfolio_requested"] is True
    assert report["omitted_force_correction_rank_portfolio_enabled"] is False
    assert report["omitted_force_correction_rank_portfolio"] is None
    assert report["omitted_force_correction_requested_curvature_rank"] == 2
    assert report["omitted_force_correction_selected_curvature_rank"] == 2
    report["graph"] = "unit"
    row = bc_dci._summary_row(report)
    assert row["omitted_force_correction_rank_portfolio_enabled"] is False
    assert row["omitted_force_correction_rank_portfolio_selected_rank"] is None


def test_budgeted_dci_accounts_omitted_force_curvature_payload(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "curvature.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=2,
        omitted_force_correction_curvature_model="block_gershgorin",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    comm = report["omitted_force_correction"]["communication_estimate"]
    assert report["omitted_force_correction_curvature_model"] == "block_gershgorin"
    assert comm["total_curvature_payload_mb"] > 0.0
    assert comm["total_correction_payload_mb"] >= (
        comm["total_omitted_force_payload_mb"] + comm["total_curvature_payload_mb"]
    )
    assert report["actual_dci_comm_mb"] >= (
        report["normal_equation_summary_comm_mb"] +
        comm["total_correction_payload_mb"]
    )


def test_budgeted_dci_accounts_directional_secant_curvature_payload(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "directional.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=2,
        omitted_force_correction_curvature_model="directional_secant",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    comm = report["omitted_force_correction"]["communication_estimate"]
    assert report["omitted_force_correction_curvature_model"] == "directional_secant"
    assert comm["total_curvature_payload_mb"] > 0.0
    assert comm["total_correction_payload_mb"] >= (
        comm["total_omitted_force_payload_mb"] + comm["total_curvature_payload_mb"]
    )


def test_budgeted_dci_accounts_subspace_secant_curvature_payload(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "subspace.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=3,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    comm = report["omitted_force_correction"]["communication_estimate"]
    assert report["omitted_force_correction_curvature_model"] == "subspace_secant"
    assert report["omitted_force_correction_curvature_rank"] == 2
    assert comm["total_curvature_payload_mb"] > 0.0
    assert comm["total_correction_payload_mb"] >= (
        comm["total_omitted_force_payload_mb"] + comm["total_curvature_payload_mb"]
    )


def test_budgeted_dci_can_account_pairwise_projected_subspace_payload(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "pairwise_subspace.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=3,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_curvature_payload_model="pairwise_projected",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    correction = report["omitted_force_correction"]
    rotation_comm = correction["rotation"]["curvature_communication_estimate"]
    translation_comm = correction["translation"]["curvature_communication_estimate"]
    assert report[
        "omitted_force_correction_curvature_payload_model"
    ] == "pairwise_projected"
    assert rotation_comm["payload_model"] == "pairwise_projected"
    assert translation_comm["payload_model"] == "pairwise_projected"
    assert rotation_comm["projected_source"] == "pairwise_block_aggregation"
    assert translation_comm["projected_source"] == "pairwise_block_aggregation"
    assert rotation_comm["max_pairwise_projection_error"] < 1e-10
    assert translation_comm["max_pairwise_projection_error"] < 1e-10
    assert correction["communication_estimate"]["total_curvature_payload_mb"] > 0.0


def test_budgeted_dci_can_use_projected_residual_rank_scheduler(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "rank_scheduler.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=3,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_curvature_rank_scheduler="projected_residual",
        omitted_force_correction_curvature_rank_max_condition=10.0,
        omitted_force_correction_curvature_rank_condition_policy="median_mad",
        omitted_force_correction_curvature_rank_condition_mad_scale=2.0,
        omitted_force_correction_curvature_payload_model="pairwise_projected",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    assert report["omitted_force_correction_curvature_rank_scheduler"] == (
        "projected_residual"
    )
    assert report["omitted_force_correction_curvature_rank_max_condition"] == 10.0
    assert report[
        "omitted_force_correction_curvature_rank_condition_policy"
    ] == "median_mad"
    assert report[
        "omitted_force_correction_curvature_rank_condition_mad_scale"
    ] == 2.0
    correction = report["omitted_force_correction"]
    assert correction["rotation"]["curvature_rank_scheduler"] == (
        "projected_residual"
    )
    assert correction["rotation"]["curvature_rank_condition_policy"] == (
        "median_mad"
    )
    assert correction["rotation"]["curvature_rank_condition_mad_scale"] == 2.0
    assert correction["translation"]["curvature_rank_scheduler"] == (
        "projected_residual"
    )
    assert correction["translation"]["curvature_rank_condition_policy"] == (
        "median_mad"
    )
    assert correction["translation"]["curvature_rank_condition_mad_scale"] == 2.0


def test_budgeted_dci_can_report_subspace_miss_diagnostic(tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "subspace_miss.txt",
        rotation_iterations=10,
        translation_iterations=10,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=2,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_diagnose_subspace_miss=True,
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    correction = report["omitted_force_correction"]
    for component in ["rotation", "translation"]:
        cert = correction[component]["reference_subspace_error"]
        assert cert["model"] == "reference_error_subspace_projection"
        assert cert["reference_label"] == "full_normal_solution"
        assert 0.0 <= cert["missed_error_energy_fraction"] <= 1.0
    row = bc_dci._summary_row(report)
    assert row["omitted_force_correction_rotation_missed_error_fraction"] >= 0.0
    assert row["omitted_force_correction_translation_missed_error_fraction"] >= 0.0
    assert report["private_interface_residual_split"]["combined"][
        "full_model_interface_residual_norm"
    ] >= 0.0
    assert row["private_interface_full_interface_residual_norm"] >= 0.0
    assert 0.0 <= row[
        "private_interface_full_interface_energy_fraction"
    ] <= 1.0


def test_summary_row_tolerates_absent_omitted_force_component():
    report = {
        "graph": "synthetic.g2o",
        "selected": "distributed_chordal_pcg",
        "pre_gate_decision": "run",
        "candidate_solved": True,
        "baseline_total_cost": 10.0,
        "distributed_total_cost": 1.0,
        "selected_total_cost": 1.0,
        "certificate_total_delta": -9.0,
        "global_consensus_accept": True,
        "predicted_candidate_comm_mb": 0.1,
        "actual_dci_comm_mb": 0.1,
        "value_upper_bound_cost_per_mb": 90.0,
        "value_score": 90.0,
        "value_score_mode": "raw_cost",
        "value_threshold": 0.0,
        "output_path": "estimate.txt",
        "omitted_force_correction": {
            "rotation": None,
            "translation": None,
        },
        "private_interface_residual_split": {
            "combined": {
                "full_model_interface_residual_norm": 2.0,
                "full_model_interface_residual_energy_fraction": 1.0,
            },
        },
    }

    row = bc_dci._summary_row(report)

    assert row["omitted_force_correction_rotation_missed_error_fraction"] is None
    assert row["private_interface_full_interface_residual_norm"] == 2.0


def test_pairwise_projected_curvature_payload_is_topology_routed(
        tmp_path: Path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), info2(1.0, 1.0), 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), info2(1.0, 1.0), 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }

    report = bc_dci.run_budgeted_dci_for_graph_data(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        baseline_poses=baseline,
        output_selected_estimate=tmp_path / "routed_pairwise_subspace.txt",
        rotation_iterations=10,
        translation_iterations=10,
        relay_cost_rotation_iterations=1,
        relay_cost_translation_iterations=1,
        value_threshold=0.0,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        candidate_solver="normal_summary",
        normal_summary_communication_model="topology_routed_equal_edge",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=3,
        omitted_force_correction_curvature_model="subspace_secant",
        omitted_force_correction_curvature_rank=2,
        omitted_force_correction_curvature_payload_model="pairwise_projected",
        available_robot_pairs={(0, 1), (1, 2), (2, 3)},
        max_relay_hops=2,
        relay_scheduler="pose_summary_normal_leverage_budget",
        min_topology_separator_coverage=1.0,
    )

    correction_comm = report["omitted_force_correction"]["communication_estimate"]
    curvature_routing = report[
        "topology_routed_omitted_force_curvature_communication"
    ]
    assert curvature_routing["logical_payload_bytes"] == (
        correction_comm["total_curvature_payload_bytes"]
    )
    assert curvature_routing["total_path_hops"] == 10
    assert curvature_routing["separator_path_count"] == 6
    assert curvature_routing["routed_payload_bytes"] > (
        curvature_routing["logical_payload_bytes"]
    )
    assert report["omitted_force_correction_curvature_comm_mb"] == (
        curvature_routing["routed_payload_mb"]
    )
    expected_actual = (
        report["effective_linear_solve_comm_mb"] +
        report["certificate_communication"]["residual_delta_consensus_mb"] +
        report["relay_extra_comm_mb"] +
        report["normal_equation_summary_comm_mb"] +
        report["omitted_force_correction_force_comm_mb"] +
        report["omitted_force_correction_curvature_comm_mb"]
    )
    assert math.isclose(
        report["actual_dci_comm_mb"],
        expected_actual,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
