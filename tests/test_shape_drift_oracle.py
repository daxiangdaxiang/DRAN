import math
from pathlib import Path

import numpy as np

from scripts import analyze_shape_drift_oracle as oracle
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


def test_contiguous_robot_map_uses_dpgo_remainder_rule():
    robot_of, ranges = oracle.build_contiguous_robot_map(list(range(11)), 5)

    assert ranges == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11)]
    assert [robot_of[i] for i in range(11)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]


def test_active_pose_ids_expand_from_separator_over_private_edges():
    robot_of, _ = oracle.build_contiguous_robot_map(list(range(6)), 2)
    edges = [
        Edge(0, 1, np.eye(4), None, 2),
        Edge(1, 2, np.eye(4), None, 2),
        Edge(2, 3, np.eye(4), None, 2),
        Edge(3, 4, np.eye(4), None, 2),
        Edge(4, 5, np.eye(4), None, 2),
    ]

    assert oracle.active_pose_ids(edges, robot_of, hops=0) == {2, 3}
    assert oracle.active_pose_ids(edges, robot_of, hops=1) == {1, 2, 3, 4}


def test_graph_pose_ids_falls_back_to_edge_endpoints_when_vertices_missing():
    edges = [
        Edge(5, 6, np.eye(4), None, 2),
        Edge(6, 8, np.eye(4), None, 2),
    ]

    assert oracle.graph_pose_ids({}, edges) == [5, 6, 8]


def test_per_robot_shape_drift_is_zero_for_rigid_robot_gauge_difference():
    reference = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.1),
        2: pose2(10.0, 2.0, 0.3),
        3: pose2(11.0, 2.2, 0.4),
    }
    gauge0 = pose2(3.0, -1.0, 0.7)
    gauge1 = pose2(-2.0, 5.0, -0.2)
    estimate = {
        0: gauge0 @ reference[0],
        1: gauge0 @ reference[1],
        2: gauge1 @ reference[2],
        3: gauge1 @ reference[3],
    }
    robot_of = {0: 0, 1: 0, 2: 1, 3: 1}

    metrics = oracle.per_robot_shape_drift(estimate, reference, robot_of)

    assert metrics["pose_count"] == 4
    assert metrics["translation_rmse"] < 1e-10
    assert metrics["rotation_rmse_deg"] < 1e-6


def test_active_substitution_oracle_reduces_bad_separator_edge():
    measurement = pose2(1.0, 0.0, 0.0)
    edges = [Edge(0, 1, measurement, None, 2)]
    robot_of = {0: 0, 1: 1}
    reference = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
    }
    estimate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(2.0, 0.0, 0.0),
    }

    report = oracle.build_oracle_report_from_data(
        graph_edges=edges,
        estimate=estimate,
        reference=reference,
        robot_of=robot_of,
        active_hops=0,
        weighted=False,
        cost_mode="dpgo",
    )

    assert report["estimate_total_cost"] > 0.0
    assert report["active_substitution_total_cost"] == 0.0
    assert report["active_substitution_gap_explained_fraction"] == 1.0


def test_continuous_shape_oracle_reduces_separator_shape_error_without_copying():
    robot_of = {0: 0, 1: 0, 2: 1, 3: 1}
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    reference = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }
    estimate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
        3: pose2(5.0, 0.0, 0.0),
    }
    active_ids = oracle.active_pose_ids(edges, robot_of, hops=1)
    before = oracle.cost_breakdown(edges, estimate, robot_of, active_ids, False, "dpgo")

    updated, stats = oracle.continuous_fixed_reference_shape_oracle(
        graph_edges=edges,
        estimate=estimate,
        reference=reference,
        robot_of=robot_of,
        active_ids=active_ids,
        weighted=False,
        cost_mode="dpgo",
        damping_weight=1e-6,
    )

    after = oracle.cost_breakdown(edges, updated, robot_of, active_ids, False, "dpgo")
    assert stats["updated_pose_count"] > 0
    assert after["total_cost"] < before["total_cost"]
    assert after["separator_cost"] < before["separator_cost"]


def test_continuous_shape_oracle_keeps_inactive_poses_fixed():
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1}
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    reference = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(3.0, 0.0, 0.0),
    }
    estimate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
        3: pose2(5.0, 0.0, 0.0),
    }
    active_ids = oracle.active_pose_ids(edges, robot_of, hops=0)

    updated, stats = oracle.continuous_fixed_reference_shape_oracle(
        graph_edges=edges,
        estimate=estimate,
        reference=reference,
        robot_of=robot_of,
        active_ids=active_ids,
        weighted=False,
        cost_mode="dpgo",
        damping_weight=1e-6,
    )

    assert stats["active_pose_count"] == 2
    assert np.allclose(updated[0], estimate[0], atol=1e-12)
    assert np.allclose(updated[1], estimate[1], atol=1e-12)


def test_centralized_chordal_reference_from_tiny_se2_graph(tmp_path: Path):
    graph = tmp_path / "tiny.g2o"
    graph.write_text(
        "\n".join(
            [
                "VERTEX_SE2 0 0 0 0",
                "VERTEX_SE2 1 0 0 0",
                "EDGE_SE2 0 1 1 0 0 100 0 0 100 0 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reference = oracle.centralized_chordal_reference_from_graph(graph)

    assert sorted(reference) == [0, 1]
    assert np.allclose(reference[0], np.eye(4), atol=1e-10)
    assert np.allclose(reference[1][:3, 3], np.array([1.0, 0.0, 0.0]), atol=1e-8)


def test_parse_manual_interleaved_matrix_estimate(tmp_path: Path):
    estimate_path = tmp_path / "estimate.txt"
    estimate_path.write_text(
        "1 0 1 0 -1 2\n"
        "0 1 2 1 0 3\n",
        encoding="utf-8",
    )

    poses = oracle.parse_manual_matrix_estimate(estimate_path)

    assert sorted(poses) == [0, 1]
    assert np.allclose(poses[0][:2, :2], np.eye(2), atol=1e-12)
    assert np.allclose(poses[0][:3, 3], np.array([1.0, 2.0, 0.0]), atol=1e-12)
    assert np.allclose(
        poses[1][:2, :2],
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        atol=1e-12,
    )
    assert np.allclose(poses[1][:3, 3], np.array([2.0, 3.0, 0.0]), atol=1e-12)
