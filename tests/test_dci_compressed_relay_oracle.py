import math

import numpy as np

from scripts import analyze_dci_compressed_relay_oracle as oracle
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


def test_pose_summary_relay_reuses_shared_endpoint_state_under_budget():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 2, 2: 2}
    topology = {(0, 1), (1, 2)}
    pose_unit_mb = oracle.estimate_pose_summary_relay_unit_mb(
        rotation_iterations=5,
        translation_iterations=5,
        extra_hops=1,
        dimension=2,
    )

    edgewise_filtered, edgewise_evidence = bc_dci.topology_edge_evidence(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs=topology,
        max_relay_hops=1,
        relay_scheduler="edge_coverage_budget",
        relay_byte_budget_mb=pose_unit_mb * 3.01,
        rotation_iterations=5,
        translation_iterations=5,
    )
    compressed = oracle.compute_pose_summary_relay_oracle(
        graph_edges=edges,
        robot_of=robot_of,
        available_robot_pairs=topology,
        max_relay_hops=1,
        relay_byte_budget_mb=pose_unit_mb * 3.01,
        rotation_iterations=5,
        translation_iterations=5,
    )

    assert edgewise_filtered == [edges[0]]
    assert edgewise_evidence["covered_separator_edges"] == 1
    assert compressed["covered_separator_edges"] == 2
    assert compressed["selected_relay_edges"] == 2
    assert compressed["selected_pose_summary_units"] == 3
    assert compressed["predicted_relay_mb"] <= pose_unit_mb * 3.01
