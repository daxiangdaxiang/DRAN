import math

import numpy as np

from scripts import analyze_separator_gauge_sync as sync
from scripts.evaluate_pgo import Edge, edge_chordal_cost


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


def test_separator_gauge_sync_recovers_two_robot_left_gauge_error():
    measurement = pose2(1.0, 0.0, 0.0)
    edges = [Edge(0, 1, measurement, None, 2)]
    robot_of = {0: 0, 1: 1}
    bad_gauge = pose2(2.0, 1.0, math.pi / 2.0)
    estimate = {
        0: pose2(0.0, 0.0, 0.0),
        1: bad_gauge @ pose2(1.0, 0.0, 0.0),
    }

    before, _ = edge_chordal_cost(edges[0], estimate, weighted=False, cost_mode="dpgo")
    corrected, stats = sync.separator_graph_gauge_sync(
        graph_edges=edges,
        estimate=estimate,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
    )
    after, _ = edge_chordal_cost(edges[0], corrected, weighted=False, cost_mode="dpgo")

    assert before > 1.0
    assert after < 1e-10
    assert stats["separator_edge_count"] == 1
    assert stats["robot_correction_count"] == 2
    assert np.allclose(corrected[1], pose2(1.0, 0.0, 0.0), atol=1e-8)


def test_separator_gauge_sync_keeps_state_when_no_separator_edges():
    edges = [Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2)]
    robot_of = {0: 0, 1: 0}
    estimate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
    }

    corrected, stats = sync.separator_graph_gauge_sync(
        graph_edges=edges,
        estimate=estimate,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
    )

    assert stats["separator_edge_count"] == 0
    assert stats["robot_correction_count"] == 1
    assert np.allclose(corrected[0], estimate[0], atol=1e-12)
    assert np.allclose(corrected[1], estimate[1], atol=1e-12)
