import math

import torch

from scripts import run_bae_pypose_six as runner


def test_contiguous_partitions_match_dpgo_remainder_rule():
    assert runner.build_contiguous_partitions(11, 5) == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 11),
    ]


def test_chordal_residual_matches_dpgo_cost_for_se3_edge():
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.2, -0.1, 0.4, 0.0, 0.0, math.sin(0.1), math.cos(0.1)],
        ],
        dtype=torch.float64,
    )
    measurement = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, math.sin(0.05), math.cos(0.05)]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 1]], dtype=torch.long)
    weights = torch.tensor([[3.0, 7.0]], dtype=torch.float64)

    residual = runner.chordal_se3_residual(measurement, nodes[edge_index[:, 0]], nodes[edge_index[:, 1]], weights)
    direct_cost = runner.direct_chordal_cost_torch(measurement, nodes[edge_index[:, 0]], nodes[edge_index[:, 1]], weights)

    assert torch.allclose(torch.sum(residual * residual), direct_cost, atol=1e-10, rtol=1e-10)


def test_fixed_first_pose_is_not_a_robot_zero_variable():
    ids = [0, 1, 2, 3]
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    factors = runner.build_partitioned_factor_sets(ids, edges, num_robots=2, fixed_pose_ids={0})

    assert factors[0].owned_variable_ids == [1]
    assert factors[0].fixed_ids == [0, 2]
    assert factors[1].owned_variable_ids == [2, 3]
