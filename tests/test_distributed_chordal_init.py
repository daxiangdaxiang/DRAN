import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import analyze_distributed_chordal_init as dci
from scripts import run_two_stage_dci_certificate_sweep as dci_sweep
from scripts import analyze_shape_drift_oracle as oracle
from scripts.evaluate_pgo import (
    Edge,
    compute_chordal_cost,
    edge_chordal_cost,
    load_pose_set,
    parse_g2o_graph,
)


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


def total_chordal_cost(edges, poses, weighted=False, cost_mode="dpgo"):
    total = 0.0
    for edge in edges:
        cost, ok = edge_chordal_cost(edge, poses, weighted, cost_mode)
        assert ok
        total += cost
    return total


def test_distributed_chordal_init_solves_two_robot_separator_edge():
    edges = [Edge(0, 1, pose2(1.0, 0.0, math.pi / 2.0), None, 2)]
    robot_of = {0: 0, 1: 1}

    poses, stats = dci.solve_distributed_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1],
        robot_of=robot_of,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
    )

    cost, ok = edge_chordal_cost(edges[0], poses, weighted=False, cost_mode="dpgo")
    assert ok
    assert cost < 1e-10
    assert stats["separator_edge_count"] == 1
    assert stats["rotation_stats"]["iterations"] <= 5
    assert np.allclose(poses[1][:2, 3], np.array([1.0, 0.0]), atol=1e-8)


def test_distributed_chordal_init_matches_centralized_chain_solution():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}

    central, _ = dci.solve_centralized_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        weighted=False,
        cost_mode="dpgo",
    )
    distributed, stats = dci.solve_distributed_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        rotation_iterations=80,
        translation_iterations=80,
        weighted=False,
        cost_mode="dpgo",
    )

    assert stats["separator_edge_count"] == 1
    assert np.allclose(distributed[1], central[1], atol=1e-8)
    assert np.allclose(distributed[2], central[2], atol=1e-8)


def test_pcg_distributed_chordal_init_matches_centralized_chain_solution_fast():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}

    central, _ = dci.solve_centralized_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        weighted=False,
        cost_mode="dpgo",
    )
    distributed, stats = dci.solve_distributed_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )

    assert stats["rotation_stats"]["method"] == "block_pcg"
    assert stats["translation_stats"]["method"] == "block_pcg"
    assert np.allclose(distributed[1], central[1], atol=1e-8)
    assert np.allclose(distributed[2], central[2], atol=1e-8)


def test_pcg_distributed_chordal_init_reports_reduction_communication():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}

    _, stats = dci.solve_distributed_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        rotation_iterations=5,
        translation_iterations=5,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )

    comm = stats["communication_estimate"]
    rot_reductions = stats["rotation_stats"]["global_reduction_count"]
    trans_reductions = stats["translation_stats"]["global_reduction_count"]
    expected_rot_bytes = rot_reductions * 2 * (2 - 1) * 8
    expected_trans_bytes = trans_reductions * 2 * (2 - 1) * 8
    expected_rot_exchange_bytes = (
        stats["rotation_stats"]["iterations"] * stats["separator_edge_count"] *
        2 * (2 * 2) * 8
    )
    expected_trans_exchange_bytes = (
        stats["translation_stats"]["iterations"] * stats["separator_edge_count"] *
        2 * 2 * 8
    )

    assert comm["robot_count"] == 2
    assert comm["rotation_global_reduction_count"] == rot_reductions
    assert comm["translation_global_reduction_count"] == trans_reductions
    assert comm["rotation_global_reduction_bytes"] == expected_rot_bytes
    assert comm["translation_global_reduction_bytes"] == expected_trans_bytes
    assert comm["rotation_separator_exchange_bytes"] == expected_rot_exchange_bytes
    assert comm["translation_separator_exchange_bytes"] == expected_trans_exchange_bytes
    assert comm["pcg_global_reduction_mb"] == (
        (expected_rot_bytes + expected_trans_bytes) / (1024.0 * 1024.0)
    )
    assert comm["linear_solve_total_estimated_mb"] == (
        (
            expected_rot_bytes + expected_trans_bytes +
            expected_rot_exchange_bytes + expected_trans_exchange_bytes
        ) / (1024.0 * 1024.0)
    )


def test_boundary_pose_summary_communication_reuses_shared_separator_pose():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 1}
    rotation_stats = {"iterations": 5, "global_reduction_count": 10}
    translation_stats = {"iterations": 5, "global_reduction_count": 10}

    boundary_units = dci.separator_boundary_pose_unit_count(edges, robot_of)
    edgewise = dci.estimate_dci_linear_solve_communication(
        rotation_stats, translation_stats, robot_count=2,
        separator_edge_count=2, dimension=2)
    boundary_summary = dci.estimate_dci_boundary_pose_solve_communication(
        rotation_stats, translation_stats, robot_count=2,
        boundary_pose_unit_count=boundary_units, dimension=2)

    expected_rot_exchange_bytes = 5 * 3 * (2 * 2) * 8
    expected_trans_exchange_bytes = 5 * 3 * 2 * 8

    assert boundary_units == 3
    assert boundary_summary["boundary_pose_unit_count"] == 3
    assert boundary_summary["rotation_separator_exchange_bytes"] == (
        expected_rot_exchange_bytes
    )
    assert boundary_summary["translation_separator_exchange_bytes"] == (
        expected_trans_exchange_bytes
    )
    assert boundary_summary["separator_exchange_bytes"] < (
        edgewise["separator_exchange_bytes"]
    )


def test_interface_schur_communication_accounts_interface_vector_payload():
    rotation_stats = {
        "iterations": 3,
        "global_reduction_count": 6,
        "interface_variable_count": 8,
    }
    translation_stats = {
        "iterations": 5,
        "global_reduction_count": 10,
        "interface_variable_count": 4,
    }

    comm = dci.estimate_dci_interface_schur_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count=3,
    )

    expected_rot_reduction = 6 * 2 * (3 - 1) * 8
    expected_trans_reduction = 10 * 2 * (3 - 1) * 8
    expected_rot_interface = 3 * 8 * 8
    expected_trans_interface = 5 * 4 * 8

    assert comm["model"] == "interface_schur_vector_exchange_plus_tree_allreduce"
    assert comm["rotation_interface_vector_bytes"] == expected_rot_interface
    assert comm["translation_interface_vector_bytes"] == expected_trans_interface
    assert comm["rotation_global_reduction_bytes"] == expected_rot_reduction
    assert comm["translation_global_reduction_bytes"] == expected_trans_reduction
    assert comm["linear_solve_total_estimated_bytes"] == (
        expected_rot_reduction +
        expected_trans_reduction +
        expected_rot_interface +
        expected_trans_interface
    )


def test_interface_schur_communication_accounts_coarse_setup_payload():
    rotation_stats = {
        "iterations": 3,
        "global_reduction_count": 6,
        "interface_variable_count": 8,
        "coarse_basis_rank": 2,
    }
    translation_stats = {
        "iterations": 5,
        "global_reduction_count": 10,
        "interface_variable_count": 4,
        "coarse_basis_rank": 1,
    }

    comm = dci.estimate_dci_interface_schur_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count=3,
    )

    expected_rot_coarse_vec = 2 * 8 * 8
    expected_trans_coarse_vec = 1 * 4 * 8
    expected_rot_coarse_reduce = 3 * 2 * (3 - 1) * 8
    expected_trans_coarse_reduce = 1 * 2 * (3 - 1) * 8

    assert comm["rotation_coarse_setup_vector_bytes"] == expected_rot_coarse_vec
    assert comm["translation_coarse_setup_vector_bytes"] == (
        expected_trans_coarse_vec
    )
    assert comm["rotation_coarse_operator_reduction_bytes"] == (
        expected_rot_coarse_reduce
    )
    assert comm["translation_coarse_operator_reduction_bytes"] == (
        expected_trans_coarse_reduce
    )
    assert comm["coarse_setup_bytes"] == (
        expected_rot_coarse_vec +
        expected_trans_coarse_vec +
        expected_rot_coarse_reduce +
        expected_trans_coarse_reduce
    )


def test_interface_schur_communication_accounts_ritz_probe_payload():
    rotation_stats = {
        "iterations": 3,
        "global_reduction_count": 8,
        "interface_variable_count": 8,
        "coarse_basis_rank": 2,
        "ritz_probe_iterations": 2,
    }
    translation_stats = {
        "iterations": 5,
        "global_reduction_count": 13,
        "interface_variable_count": 4,
        "coarse_basis_rank": 1,
        "ritz_probe_iterations": 3,
    }

    comm = dci.estimate_dci_interface_schur_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count=3,
    )

    expected_rot_ritz = 2 * 8 * 8
    expected_trans_ritz = 3 * 4 * 8
    assert comm["rotation_ritz_probe_interface_vector_bytes"] == (
        expected_rot_ritz
    )
    assert comm["translation_ritz_probe_interface_vector_bytes"] == (
        expected_trans_ritz
    )
    assert comm["ritz_probe_interface_vector_mb"] > 0.0


def test_interface_schur_communication_accounts_ritz_portfolio_scoring_payload():
    rotation_stats = {
        "iterations": 3,
        "global_reduction_count": 8,
        "interface_variable_count": 8,
        "coarse_basis_rank": 2,
        "ritz_probe_iterations": 2,
        "ritz_portfolio_scoring_schur_matvec_count": 5,
    }
    translation_stats = {
        "iterations": 5,
        "global_reduction_count": 13,
        "interface_variable_count": 4,
        "coarse_basis_rank": 1,
        "ritz_probe_iterations": 3,
        "ritz_portfolio_scoring_schur_matvec_count": 7,
    }

    comm = dci.estimate_dci_interface_schur_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count=3,
    )

    expected_rot_scoring = 5 * 8 * 8
    expected_trans_scoring = 7 * 4 * 8
    assert comm["rotation_ritz_portfolio_scoring_interface_vector_bytes"] == (
        expected_rot_scoring
    )
    assert comm["translation_ritz_portfolio_scoring_interface_vector_bytes"] == (
        expected_trans_scoring
    )
    assert comm["ritz_portfolio_scoring_interface_vector_mb"] == (
        (expected_rot_scoring + expected_trans_scoring) / (1024.0 * 1024.0)
    )


def test_normal_equation_block_summary_reconstructs_rotation_system_exactly():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    matrix, rhs, meta = dci.assemble_rotation_system(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    summary = dci.normal_equation_block_summary(
        matrix, rhs, block_dim=meta["block_dim"])
    hessian, gradient = dci.reconstruct_normal_equation_from_summary(summary)
    expected_hessian = (matrix.T @ matrix).toarray()
    expected_gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)

    assert summary["block_dim"] == 4
    assert summary["variable_count"] == matrix.shape[1]
    assert np.allclose(hessian, expected_hessian, atol=1e-12)
    assert np.allclose(gradient, expected_gradient, atol=1e-12)
    assert summary["payload_bytes"] == (
        summary["hessian_scalar_count"] + summary["gradient_scalar_count"]
    ) * 8


def test_normal_summary_block_graph_detects_missing_bridge_block():
    def summary(block_count, offdiag_pairs):
        hessian_blocks = {}
        for block in range(block_count):
            hessian_blocks[(block, block)] = [[1.0]]
        for i, j in offdiag_pairs:
            hessian_blocks[(i, j)] = [[-1.0]]
            hessian_blocks[(j, i)] = [[-1.0]]
        return {
            "block_count": block_count,
            "block_dim": 1,
            "variable_count": block_count,
            "hessian_blocks": hessian_blocks,
            "gradient_blocks": {},
        }

    bridge_case = dci.normal_summary_block_graph_diagnostics(
        full_summary=summary(3, [(0, 1), (1, 2)]),
        selected_summary=summary(3, [(0, 1)]),
    )
    non_bridge_case = dci.normal_summary_block_graph_diagnostics(
        full_summary=summary(3, [(0, 1), (1, 2), (0, 2)]),
        selected_summary=summary(3, [(0, 1), (0, 2)]),
    )

    assert bridge_case["full_block_edge_count"] == 2
    assert bridge_case["covered_block_edge_count"] == 1
    assert bridge_case["missing_block_edge_count"] == 1
    assert bridge_case["missing_bridge_block_edge_count"] == 1
    assert bridge_case["full_component_count"] == 1
    assert bridge_case["selected_component_count"] == 2
    assert bridge_case["component_count_delta"] == 1
    assert bridge_case["structural_criticality"] > 0.0

    assert non_bridge_case["missing_block_edge_count"] == 1
    assert non_bridge_case["missing_bridge_block_edge_count"] == 0
    assert non_bridge_case["component_count_delta"] == 0
    assert non_bridge_case["structural_criticality"] == 0.0


def test_structural_spanning_summary_selection_preserves_connectivity_with_less_payload():
    full_summary = {
        "variable_count": 3,
        "block_dim": 1,
        "block_count": 3,
        "hessian_blocks": {
            (0, 0): [[2.0]],
            (1, 1): [[2.0]],
            (2, 2): [[2.0]],
            (0, 1): [[-1.0]],
            (1, 0): [[-1.0]],
            (1, 2): [[-1.0]],
            (2, 1): [[-1.0]],
            (0, 2): [[-0.5]],
            (2, 0): [[-0.5]],
        },
        "gradient_blocks": {
            0: [1.0],
            1: [2.0],
            2: [3.0],
        },
    }
    full_summary = dci.recount_normal_summary_payload(full_summary)

    selected = dci.select_normal_summary_blocks(
        full_summary,
        mode="structural_spanning",
        max_offdiag_block_edges=2,
    )
    diagnostics = dci.normal_summary_block_graph_diagnostics(
        full_summary,
        selected,
    )

    assert selected["summary_selection"]["mode"] == "structural_spanning"
    assert selected["summary_selection"]["selected_offdiag_block_edges"] == 2
    assert selected["hessian_block_count"] == 7
    assert selected["gradient_block_count"] == 3
    assert selected["payload_bytes"] < full_summary["payload_bytes"]
    assert diagnostics["component_count_delta"] == 0
    assert diagnostics["missing_block_edge_count"] == 1


def test_normal_model_representativeness_detects_pruned_residual_gap():
    full_hessian = np.asarray([[2.0, -1.0], [-1.0, 2.0]])
    full_gradient = np.asarray([0.0, 1.0])
    selected_hessian = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    selected_gradient = full_gradient.copy()
    selected_solution = np.asarray([0.0, 0.5])

    cert = dci.normal_model_representativeness_certificate(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        solution=selected_solution,
        relative_tolerance=1e-6,
    )

    assert cert["selected_model_residual_norm"] == 0.0
    assert cert["full_model_residual_norm"] > 0.0
    assert cert["omitted_model_residual_norm"] > 0.0
    assert cert["normal_model_representative"] is False


def test_residual_force_summary_refinement_adds_high_force_missing_block():
    full_summary = {
        "variable_count": 3,
        "block_dim": 1,
        "block_count": 3,
        "hessian_blocks": {
            (0, 0): [[2.0]],
            (1, 1): [[2.0]],
            (2, 2): [[2.0]],
            (0, 1): [[-0.1]],
            (1, 0): [[-0.1]],
            (1, 2): [[-10.0]],
            (2, 1): [[-10.0]],
            (0, 2): [[-0.2]],
            (2, 0): [[-0.2]],
        },
        "gradient_blocks": {0: [0.0], 1: [0.0], 2: [0.0]},
    }
    full_summary = dci.recount_normal_summary_payload(full_summary)
    seed = dci.select_normal_summary_blocks(
        full_summary,
        mode="structural_spanning",
        max_offdiag_block_edges=1,
    )

    refined = dci.refine_normal_summary_blocks_by_residual_force(
        full_summary=full_summary,
        selected_summary=seed,
        solution=np.asarray([0.0, 1.0, 1.0]),
        max_offdiag_block_edges=2,
    )

    selected_edges = {
        tuple(edge)
        for edge in refined["summary_selection"]["selected_offdiag_block_edge_keys"]
    }
    assert (1, 2) in selected_edges
    assert refined["summary_selection"]["refinement_added_offdiag_block_edges"] == 1
    assert refined["payload_bytes"] > seed["payload_bytes"]


def test_omitted_force_correction_reduces_full_normal_residual():
    full_hessian = np.asarray([[2.0, -1.0], [-1.0, 2.0]])
    full_gradient = np.asarray([0.0, 1.0])
    selected_hessian = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    selected_gradient = full_gradient.copy()
    selected_solution = np.asarray([0.0, 0.5])
    blocks = {0: np.asarray([0]), 1: np.asarray([1])}

    corrected, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
    )

    assert stats["model"] == "omitted_force_vector_correction"
    assert stats["before"]["full_model_residual_norm"] > (
        stats["after"]["full_model_residual_norm"]
    )
    assert np.linalg.norm(corrected - selected_solution) > 0.0
    assert stats["communication_estimate"]["omitted_force_payload_bytes"] > 0


def test_multi_round_omitted_force_correction_reduces_full_residual_further():
    full_hessian = np.asarray([[2.0, -1.0], [-1.0, 2.0]])
    full_gradient = np.asarray([0.0, 1.0])
    selected_hessian = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    selected_gradient = full_gradient.copy()
    selected_solution = np.asarray([0.0, 0.5])
    blocks = {0: np.asarray([0]), 1: np.asarray([1])}

    _, one_round = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        correction_rounds=1,
    )
    _, multi_round = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        correction_rounds=4,
    )

    assert multi_round["accepted_rounds"] > one_round["accepted_rounds"]
    assert multi_round["after"]["full_model_residual_norm"] < (
        one_round["after"]["full_model_residual_norm"]
    )
    assert multi_round["communication_estimate"]["omitted_force_payload_bytes"] > (
        one_round["communication_estimate"]["omitted_force_payload_bytes"]
    )


def test_omitted_force_correction_can_use_block_gershgorin_curvature_payload():
    full_hessian = np.asarray([[2.0, -1.0], [-1.0, 2.0]])
    full_gradient = np.asarray([0.0, 1.0])
    selected_hessian = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    selected_gradient = full_gradient.copy()
    selected_solution = np.asarray([0.0, 0.5])
    blocks = {0: np.asarray([0]), 1: np.asarray([1])}

    corrected, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=2,
        curvature_model="block_gershgorin",
    )

    comm = stats["communication_estimate"]
    assert stats["curvature_model"] == "block_gershgorin"
    assert comm["curvature_payload_bytes"] == 16
    assert comm["total_correction_payload_bytes"] == (
        comm["omitted_force_payload_bytes"] + comm["curvature_payload_bytes"]
    )
    assert stats["after"]["full_model_residual_norm"] < (
        stats["before"]["full_model_residual_norm"]
    )
    assert np.linalg.norm(corrected - selected_solution) > 0.0


def test_omitted_force_correction_can_use_directional_secant_curvature():
    full_hessian = np.asarray([[2.0, -1.0], [-1.0, 2.0]])
    selected_hessian = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    selected_solution = np.asarray([0.5, -0.5])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {0: np.asarray([0]), 1: np.asarray([1])}

    corrected, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=1,
        curvature_model="directional_secant",
    )

    first_round = stats["rounds"][0]
    comm = stats["communication_estimate"]
    assert stats["curvature_model"] == "directional_secant"
    assert first_round["directional_secant_scale"] < 1.0
    assert first_round["omitted_directional_curvature"] > 0.0
    assert comm["curvature_payload_bytes"] == 8
    assert comm["total_correction_payload_bytes"] == (
        comm["omitted_force_payload_bytes"] + comm["curvature_payload_bytes"]
    )
    assert stats["after"]["full_model_residual_norm"] < (
        stats["before"]["full_model_residual_norm"]
    )
    assert np.linalg.norm(corrected - selected_solution) > 0.0


def test_omitted_force_correction_can_use_rank_k_subspace_secant():
    full_hessian = np.asarray([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.5, -0.25, 0.75])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
        2: np.asarray([2]),
    }

    corrected, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
    )

    ranks = [round_report["subspace_rank"] for round_report in stats["rounds"]]
    comm = stats["communication_estimate"]
    assert stats["curvature_model"] == "subspace_secant"
    assert max(ranks) == 2
    assert comm["curvature_payload_bytes"] > 8
    assert stats["curvature_communication_estimate"]["max_subspace_rank"] == 2
    assert stats["after"]["full_model_residual_norm"] < (
        stats["before"]["full_model_residual_norm"]
    )
    assert np.linalg.norm(corrected - selected_solution) > 0.0


def test_subspace_secant_can_select_rank_by_projected_residual():
    full_hessian = np.asarray([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 1.0],
        [0.5, 1.0, 2.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.5, -0.25, 0.75])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
        2: np.asarray([2]),
    }

    _, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
        curvature_rank_scheduler="projected_residual",
    )

    scheduled_rounds = [
        report for report in stats["rounds"]
        if report["solve_stats"].get("subspace_rank_scheduler")
        == "projected_residual"
    ]
    active_selection_rounds = [
        report for report in scheduled_rounds
        if report["solve_stats"]["subspace_rank_selection"]["enabled"]
    ]
    assert stats["curvature_rank_scheduler"] == "projected_residual"
    assert active_selection_rounds
    for report in active_selection_rounds:
        selection = report["solve_stats"]["subspace_rank_selection"]
        assert selection["candidate_count"] >= 1
        assert selection["selected_rank"] == report["subspace_rank"]
        assert report["solve_stats"]["subspace_candidate_rank"] >= (
            report["subspace_rank"]
        )


def test_projected_residual_rank_scheduler_respects_condition_guard():
    full_hessian = np.asarray([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 1.0],
        [0.5, 1.0, 2.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.5, -0.25, 0.75])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
        2: np.asarray([2]),
    }

    _, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
        curvature_rank_scheduler="projected_residual",
        curvature_rank_max_condition=1.0,
    )

    active_selection_rounds = [
        report for report in stats["rounds"]
        if report["solve_stats"]["subspace_rank_selection"]["enabled"]
    ]
    assert active_selection_rounds
    for report in active_selection_rounds:
        selection = report["solve_stats"]["subspace_rank_selection"]
        selected_index = selection["selected_index"]
        assert selection["condition_guard_enabled"] is True
        assert selection["max_projected_condition"] == 1.0
        assert selection["candidate_projected_conditions"]
        assert selection["candidate_condition_feasible"][selected_index] is True


def test_projected_residual_rank_scheduler_falls_back_if_no_condition_feasible():
    full_hessian = np.asarray([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 1.0],
        [0.5, 1.0, 2.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.5, -0.25, 0.75])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
        2: np.asarray([2]),
    }

    _, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
        curvature_rank_scheduler="projected_residual",
        curvature_rank_max_condition=0.5,
    )

    fallback_rounds = []
    for report in stats["rounds"]:
        selection = report["solve_stats"]["subspace_rank_selection"]
        if selection["enabled"] and selection["eligible_candidate_count"] == 0:
            fallback_rounds.append(selection)

    assert fallback_rounds
    for selection in fallback_rounds:
        assert selection["condition_guard_enabled"] is True
        assert selection["max_projected_condition"] == 0.5
        assert not any(selection["candidate_condition_feasible"])
        assert selection["selected_rank"] in selection["candidate_ranks"]


def test_projected_residual_rank_scheduler_can_use_median_mad_condition_guard():
    full_hessian = np.asarray([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 1.0],
        [0.5, 1.0, 2.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.5, -0.25, 0.75])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
        2: np.asarray([2]),
    }

    _, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
        curvature_rank_scheduler="projected_residual",
        curvature_rank_condition_policy="median_mad",
        curvature_rank_condition_mad_scale=1.0,
    )

    active_selection_rounds = [
        report for report in stats["rounds"]
        if report["solve_stats"]["subspace_rank_selection"]["enabled"]
    ]
    assert active_selection_rounds
    for report in active_selection_rounds:
        selection = report["solve_stats"]["subspace_rank_selection"]
        conditions = np.asarray(
            selection["candidate_projected_conditions"],
            dtype=float,
        )
        expected = float(
            np.median(conditions) +
            np.median(np.abs(conditions - np.median(conditions)))
        )
        assert selection["condition_policy"] == "median_mad"
        assert selection["condition_mad_scale"] == 1.0
        assert np.isclose(selection["adaptive_projected_condition"], expected)
        assert np.isclose(selection["max_projected_condition"], expected)


def test_subspace_error_certificate_measures_unexplained_reference_error():
    solution = np.asarray([1.0, 0.0, 0.0])
    reference = np.asarray([1.0, 3.0, 4.0])
    subspace_vectors = [np.asarray([0.0, 1.0, 0.0])]

    cert = dci.subspace_error_certificate(
        reference_solution=reference,
        solution=solution,
        subspace_vectors=subspace_vectors,
        reference_label="unit_reference",
    )

    assert cert["model"] == "reference_error_subspace_projection"
    assert cert["reference_label"] == "unit_reference"
    assert cert["subspace_rank"] == 1
    assert np.isclose(cert["error_norm"], 5.0)
    assert np.isclose(cert["explainable_error_norm"], 3.0)
    assert np.isclose(cert["missed_error_norm"], 4.0)
    assert np.isclose(cert["explainable_error_energy_fraction"], 9.0 / 25.0)
    assert np.isclose(cert["missed_error_energy_fraction"], 16.0 / 25.0)


def test_private_interface_normal_residual_split_separates_block_energy():
    full_hessian = np.eye(4)
    full_gradient = np.zeros(4)
    selected_hessian = np.zeros((4, 4))
    selected_gradient = np.zeros(4)
    solution = np.asarray([3.0, 4.0, 12.0, 5.0])
    offsets = {10: 0, 20: 2}

    split = dci.normal_model_private_interface_residual_split(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        solution=solution,
        offsets=offsets,
        block_dim=2,
        interface_pose_ids={20},
    )

    assert split["model"] == "private_interface_normal_residual_split"
    assert split["private_block_count"] == 1
    assert split["interface_block_count"] == 1
    assert np.isclose(split["full_model_private_residual_norm"], 5.0)
    assert np.isclose(split["full_model_interface_residual_norm"], 13.0)
    assert np.isclose(split["full_model_residual_norm"], math.sqrt(194.0))
    assert np.isclose(
        split["full_model_interface_residual_energy_fraction"],
        169.0 / 194.0,
    )
    assert np.isclose(split["selected_model_residual_norm"], 0.0)
    assert np.isclose(split["omitted_model_interface_residual_norm"], 13.0)


def test_summary_chordal_init_reports_private_interface_residual_split():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.2, 0.1, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        rotation_iterations=20,
        translation_iterations=20,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=0,
        omitted_force_correction_rounds=1,
    )

    split = stats["private_interface_residual_split"]
    assert split["model"] == "combined_private_interface_normal_residual_split"
    assert split["rotation"]["interface_block_count"] > 0
    assert split["translation"]["interface_block_count"] > 0
    assert split["combined"]["full_model_interface_residual_norm"] >= 0.0
    assert 0.0 <= split["combined"][
        "full_model_interface_residual_energy_fraction"
    ] <= 1.0


def test_interface_schur_hessian_solve_matches_full_solution():
    hessian = np.asarray([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 1.0],
        [0.5, 1.0, 2.5],
    ])
    gradient = np.asarray([1.0, 2.0, -0.5])
    interface_indices = np.asarray([1])

    solution, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        iterations=20,
        damping=0.0,
        tolerance=1e-12,
    )

    expected = np.linalg.solve(hessian, gradient)
    assert stats["method"] == "interface_schur_pcg"
    assert stats["interface_variable_count"] == 1
    assert stats["private_variable_count"] == 2
    assert stats["final_normal_residual"] <= 1e-10
    assert np.allclose(solution, expected, atol=1e-10)


def test_interface_schur_hessian_solve_accepts_block_jacobi_preconditioner():
    hessian = np.asarray([
        [5.0, 0.5, 1.0, 0.2],
        [0.5, 4.0, 0.3, 1.1],
        [1.0, 0.3, 3.5, 0.4],
        [0.2, 1.1, 0.4, 3.2],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    solution, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=20,
        damping=0.0,
        tolerance=1e-12,
        schur_preconditioner="block_jacobi",
    )

    expected = np.linalg.solve(hessian, gradient)
    assert stats["schur_preconditioner"] == "block_jacobi"
    assert stats["schur_preconditioner_block_count"] == 1
    assert stats["final_normal_residual"] <= 1e-10
    assert np.allclose(solution, expected, atol=1e-10)


def test_interface_schur_hessian_solve_accepts_block_jacobi_plus_coarse():
    hessian = np.asarray([
        [7.0, 0.4, 1.2, 0.3, 0.1, 0.0],
        [0.4, 6.5, 0.2, 1.1, 0.0, 0.2],
        [1.2, 0.2, 4.5, 0.8, 1.4, 0.6],
        [0.3, 1.1, 0.8, 4.8, 0.7, 1.3],
        [0.1, 0.0, 1.4, 0.7, 3.9, 0.5],
        [0.0, 0.2, 0.6, 1.3, 0.5, 4.2],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25, -1.0, 0.75])
    interface_indices = np.asarray([2, 3, 4, 5])
    coarse_basis = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    solution, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3]), np.asarray([4, 5])],
        iterations=20,
        damping=0.0,
        tolerance=1e-12,
        schur_preconditioner="block_jacobi+coarse",
        coarse_basis=coarse_basis,
    )

    expected = np.linalg.solve(hessian, gradient)
    assert stats["schur_preconditioner"] == "block_jacobi+coarse"
    assert stats["coarse_basis_column_count"] == 2
    assert stats["coarse_basis_rank"] == 2
    assert stats["final_normal_residual"] <= 1e-10
    assert np.allclose(solution, expected, atol=1e-10)


def test_interface_component_coordinate_coarse_basis_uses_summary_components():
    offsets = {10: 0, 11: 2, 12: 4}
    interface_pose_ids = {10, 11, 12}
    interface_indices = np.arange(6)
    summary = {
        "variable_count": 6,
        "block_dim": 2,
        "hessian_blocks": {
            (0, 0): [[1.0, 0.0], [0.0, 1.0]],
            (1, 1): [[1.0, 0.0], [0.0, 1.0]],
            (2, 2): [[1.0, 0.0], [0.0, 1.0]],
            (0, 1): [[-1.0, 0.0], [0.0, -1.0]],
            (1, 0): [[-1.0, 0.0], [0.0, -1.0]],
        },
        "gradient_blocks": {},
    }

    basis = dci._interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=summary,
        component_limit=None,
    )

    assert basis.shape == (6, 4)
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    assert np.count_nonzero(np.abs(basis[0:4, 0]) > 1e-12) == 2
    assert np.count_nonzero(np.abs(basis[4:6, 2]) > 1e-12) == 1


def test_interface_endpoint_coordinate_coarse_basis_uses_component_endpoints():
    offsets = {10: 0, 11: 2, 12: 4}
    interface_pose_ids = {10, 11, 12}
    interface_indices = np.arange(6)
    summary = {
        "variable_count": 6,
        "block_dim": 2,
        "hessian_blocks": {
            (0, 0): [[1.0, 0.0], [0.0, 1.0]],
            (1, 1): [[1.0, 0.0], [0.0, 1.0]],
            (2, 2): [[1.0, 0.0], [0.0, 1.0]],
            (0, 1): [[-1.0, 0.0], [0.0, -1.0]],
            (1, 0): [[-1.0, 0.0], [0.0, -1.0]],
            (1, 2): [[-1.0, 0.0], [0.0, -1.0]],
            (2, 1): [[-1.0, 0.0], [0.0, -1.0]],
        },
        "gradient_blocks": {},
    }

    basis, info = dci._build_interface_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        robot_of={10: 0, 11: 0, 12: 1},
        interface_indices=interface_indices,
        separator_summary=summary,
        mode="component_endpoint_coordinate",
        component_limit=None,
    )

    assert info["endpoint_coordinate_basis_rank"] == 4
    assert basis.shape == (6, 4)
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    assert np.count_nonzero(np.abs(basis[0:2, :]) > 1e-12) == 2
    assert np.count_nonzero(np.abs(basis[2:4, :]) > 1e-12) == 0
    assert np.count_nonzero(np.abs(basis[4:6, :]) > 1e-12) == 2


def test_interface_component_coordinate_coarse_basis_zero_limit_disables_modes():
    offsets = {10: 0, 11: 2, 12: 4}
    interface_pose_ids = {10, 11, 12}
    interface_indices = np.arange(6)
    summary = {
        "variable_count": 6,
        "block_dim": 2,
        "hessian_blocks": {
            (0, 0): [[1.0, 0.0], [0.0, 1.0]],
            (1, 1): [[1.0, 0.0], [0.0, 1.0]],
            (2, 2): [[1.0, 0.0], [0.0, 1.0]],
            (0, 1): [[-1.0, 0.0], [0.0, -1.0]],
            (1, 0): [[-1.0, 0.0], [0.0, -1.0]],
        },
        "gradient_blocks": {},
    }

    basis = dci._interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=summary,
        component_limit=0,
    )

    assert basis.shape == (6, 0)


def test_interface_component_coordinate_coarse_basis_can_select_gradient_energy():
    offsets = {10: 0, 11: 2, 12: 4}
    interface_pose_ids = {10, 11, 12}
    interface_indices = np.arange(6)
    summary = {
        "variable_count": 6,
        "block_dim": 2,
        "hessian_blocks": {
            (0, 0): [[1.0, 0.0], [0.0, 1.0]],
            (1, 1): [[1.0, 0.0], [0.0, 1.0]],
            (2, 2): [[1.0, 0.0], [0.0, 1.0]],
            (0, 1): [[-1.0, 0.0], [0.0, -1.0]],
            (1, 0): [[-1.0, 0.0], [0.0, -1.0]],
        },
        "gradient_blocks": {
            0: [0.1, 0.0],
            1: [0.1, 0.0],
            2: [5.0, 0.0],
        },
    }

    size_basis = dci._interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=summary,
        component_limit=1,
        component_selection_mode="size",
    )
    energy_basis = dci._interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=summary,
        component_limit=1,
        component_selection_mode="gradient_energy",
    )

    assert size_basis.shape == (6, 2)
    assert energy_basis.shape == (6, 2)
    assert np.count_nonzero(np.abs(size_basis[0:4, 0]) > 1e-12) == 2
    assert np.count_nonzero(np.abs(size_basis[4:6, 0]) > 1e-12) == 0
    assert np.count_nonzero(np.abs(energy_basis[0:4, 0]) > 1e-12) == 0
    assert np.count_nonzero(np.abs(energy_basis[4:6, 0]) > 1e-12) == 1


def test_interface_component_coordinate_coarse_basis_can_select_projected_merit():
    offsets = {10: 0, 11: 2, 12: 4}
    interface_pose_ids = {10, 11, 12}
    interface_indices = np.arange(6)
    summary = {
        "variable_count": 6,
        "block_dim": 2,
        "hessian_blocks": {
            (0, 0): [[100.0, 0.0], [0.0, 100.0]],
            (1, 1): [[100.0, 0.0], [0.0, 100.0]],
            (2, 2): [[1.0, 0.0], [0.0, 1.0]],
            (0, 1): [[1.0, 0.0], [0.0, 1.0]],
            (1, 0): [[1.0, 0.0], [0.0, 1.0]],
        },
        "gradient_blocks": {
            0: [5.0, 0.0],
            1: [5.0, 0.0],
            2: [3.0, 0.0],
        },
    }

    energy_basis = dci._interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=summary,
        component_limit=1,
        component_selection_mode="gradient_energy",
    )
    merit_basis = dci._interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=2,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=summary,
        component_limit=1,
        component_selection_mode="projected_merit",
    )

    assert energy_basis.shape == (6, 2)
    assert merit_basis.shape == (6, 2)
    assert np.count_nonzero(np.abs(energy_basis[0:4, 0]) > 1e-12) == 2
    assert np.count_nonzero(np.abs(energy_basis[4:6, 0]) > 1e-12) == 0
    assert np.count_nonzero(np.abs(merit_basis[0:4, 0]) > 1e-12) == 0
    assert np.count_nonzero(np.abs(merit_basis[4:6, 0]) > 1e-12) == 1


def test_residual_deflation_basis_prefers_largest_residual_energy():
    residuals = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.1, 0.0]),
        np.asarray([0.0, 0.0, 5.0]),
    ]

    basis = dci._residual_deflation_basis_from_history(
        residuals, max_rank=1)

    assert basis.shape == (3, 1)
    assert abs(float(basis[:, 0] @ np.asarray([0.0, 0.0, 1.0]))) > 0.999


def test_ritz_low_mode_basis_prefers_small_preconditioned_eigenvalue():
    operator = np.diag([0.02, 2.0, 7.0])

    basis, stats = dci._ritz_low_mode_basis_from_operator(
        matvec=lambda vector: operator @ vector,
        size=3,
        max_rank=1,
        probe_iterations=3,
        seed_vectors=[np.ones(3)],
    )

    assert basis.shape == (3, 1)
    assert stats["ritz_basis_rank"] == 1
    assert stats["ritz_probe_iterations"] == 3
    assert abs(float(basis[:, 0] @ np.asarray([1.0, 0.0, 0.0]))) > 0.999


def test_generalized_ritz_low_mode_basis_uses_mass_matrix():
    stiffness = np.diag([1.0, 2.0, 3.0])
    mass = np.diag([1.0, 100.0, 1.0])
    mass_inverse = np.diag([1.0, 0.01, 1.0])

    basis, stats = dci._generalized_ritz_low_mode_basis_from_operators(
        stiffness_matvec=lambda vector: stiffness @ vector,
        mass_matvec=lambda vector: mass @ vector,
        probe_matvec=lambda vector: mass_inverse @ stiffness @ vector,
        size=3,
        max_rank=1,
        probe_iterations=3,
        seed_vectors=[np.ones(3)],
    )

    assert stats["ritz_mode"] == "generalized"
    assert stats["ritz_basis_rank"] == 1
    assert stats["ritz_probe_iterations"] == 3
    assert abs(float(basis[:, 0] @ np.asarray([0.0, 1.0, 0.0]))) > 0.999
    assert np.isclose(stats["ritz_selected_values"][0], 0.02)


def test_m_orthogonal_lanczos_ritz_probe_is_mass_orthonormal():
    stiffness = np.diag([1.0, 2.0, 3.0])
    mass = np.diag([1.0, 100.0, 1.0])
    mass_inverse = np.diag([1.0, 0.01, 1.0])

    basis, stats = dci._m_orthogonal_lanczos_ritz_low_mode_basis_from_operators(
        stiffness_matvec=lambda vector: stiffness @ vector,
        mass_matvec=lambda vector: mass @ vector,
        preconditioned_matvec=lambda vector: mass_inverse @ stiffness @ vector,
        size=3,
        max_rank=2,
        probe_iterations=3,
        seed_vectors=[np.ones(3)],
    )

    assert stats["ritz_mode"] == "m_orthogonal_lanczos"
    assert stats["ritz_basis_rank"] == 2
    assert stats["ritz_probe_iterations"] == 3
    assert stats["ritz_lanczos_m_orthogonality_error"] < 1e-10
    assert np.allclose(stats["ritz_selected_values"], [0.02, 1.0])
    assert basis.shape == (3, 2)
    assert abs(float(basis[:, 0] @ np.asarray([0.0, 1.0, 0.0]))) > 0.999


def test_harmonic_ritz_low_mode_basis_targets_near_zero_mode():
    operator = np.diag([0.02, 2.0, 7.0])

    basis, stats = dci._harmonic_ritz_low_mode_basis_from_operator(
        matvec=lambda vector: operator @ vector,
        size=3,
        max_rank=1,
        probe_iterations=3,
        seed_vectors=[np.ones(3)],
    )

    assert stats["ritz_mode"] == "harmonic"
    assert stats["ritz_basis_rank"] == 1
    assert stats["ritz_probe_iterations"] == 3
    assert abs(float(basis[:, 0] @ np.asarray([1.0, 0.0, 0.0]))) > 0.999
    assert np.isclose(stats["ritz_selected_values"][0], 0.02)


def test_ritz_value_threshold_rank_selection_truncates_basis():
    basis = np.eye(3)
    stats = {
        "ritz_basis_rank": 3,
        "ritz_selected_values": [0.1, 0.8, 1.5],
    }

    selected_basis, selected_stats = dci._apply_ritz_rank_selection(
        basis,
        stats,
        mode="value_threshold",
        value_threshold=1.0,
        min_rank=1,
    )

    assert selected_basis.shape == (3, 2)
    assert selected_stats["ritz_deflation_requested_rank_before_selection"] == 3
    assert selected_stats["ritz_deflation_selected_rank"] == 2
    assert selected_stats["ritz_rank_selection_mode"] == "value_threshold"
    assert selected_stats["ritz_selected_values"] == [0.1, 0.8]


def test_ritz_energy_capture_rank_selection_uses_projected_energy():
    basis = np.eye(3)
    stats = {
        "ritz_basis_rank": 3,
        "ritz_selected_values": [0.1, 0.2, 0.3],
        "ritz_energy_contributions": [9.0, 1.0, 0.1],
    }

    selected_basis, selected_stats = dci._apply_ritz_rank_selection(
        basis,
        stats,
        mode="energy_capture",
        energy_capture_fraction=0.95,
        min_rank=1,
    )

    assert selected_basis.shape == (3, 2)
    assert selected_stats["ritz_deflation_selected_rank"] == 2
    assert selected_stats["ritz_rank_selection_mode"] == "energy_capture"
    assert selected_stats["ritz_energy_capture_fraction"] == 0.95
    assert selected_stats["ritz_energy_capture_total"] == 10.1
    assert selected_stats["ritz_energy_capture_selected"] == 10.0
    assert selected_stats["ritz_energy_contributions"] == [9.0, 1.0]


def test_ritz_portfolio_selects_highest_quadratic_merit_candidate():
    stiffness = np.diag([10.0, 1.0])
    rhs = np.asarray([1.0, 8.0])
    candidates = [
        ("small_eigenvalue", np.asarray([[1.0], [0.0]]), {
            "ritz_probe_iterations": 1,
            "ritz_probe_global_reduction_count": 0,
        }),
        ("handoff_merit", np.asarray([[0.0], [1.0]]), {
            "ritz_probe_iterations": 1,
            "ritz_probe_global_reduction_count": 0,
        }),
    ]

    basis, stats = dci._select_ritz_portfolio_basis(
        candidates=candidates,
        stiffness_matvec=lambda vector: stiffness @ vector,
        rhs=rhs,
        base_coarse_basis=None,
        max_rank=1,
        damping=0.0,
    )

    assert stats["ritz_mode"] == "portfolio"
    assert stats["ritz_portfolio_selected_mode"] == "handoff_merit"
    assert stats["ritz_basis_rank"] == 1
    assert stats["ritz_portfolio_candidate_count"] == 2
    assert stats["ritz_portfolio_scoring_schur_matvec_count"] == 2
    assert abs(float(basis[:, 0] @ np.asarray([0.0, 1.0]))) > 0.999
    merits = {
        item["mode"]: item["incremental_merit"]
        for item in stats["ritz_portfolio_candidates"]
    }
    assert merits["handoff_merit"] > merits["small_eigenvalue"]


def test_interface_schur_hessian_solve_can_start_from_coarse_solution():
    hessian = np.asarray([
        [4.0, 0.6, 1.0, 0.2],
        [0.6, 5.0, 0.3, 1.1],
        [1.0, 0.3, 3.5, 0.4],
        [0.2, 1.1, 0.4, 3.2],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    solution, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=0,
        damping=0.0,
        tolerance=1e-12,
        schur_preconditioner="block_jacobi+coarse",
        coarse_basis=np.eye(2),
        use_coarse_initial_guess=True,
    )

    expected = np.linalg.solve(hessian, gradient)
    assert stats["coarse_initial_guess_used"]
    assert stats["iterations"] == 0
    assert stats["final_normal_residual"] <= 1e-10
    assert np.allclose(solution, expected, atol=1e-10)


def test_interface_schur_hessian_solve_can_add_residual_deflation_basis():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        residual_deflation_rank=1,
        residual_deflation_pilot_iterations=1,
    )

    assert stats["residual_deflation_requested_rank"] == 1
    assert stats["residual_deflation_basis_rank"] == 1
    assert stats["residual_deflation_pilot_iterations"] == 1
    assert stats["coarse_basis_rank"] == 1
    assert stats["global_reduction_count"] >= (
        stats["solve_global_reduction_count"] +
        stats["residual_deflation_pilot_global_reduction_count"])


def test_interface_schur_hessian_solve_can_add_ritz_low_modes():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=1,
        ritz_probe_iterations=2,
    )

    assert stats["ritz_deflation_requested_rank"] == 1
    assert stats["ritz_deflation_basis_rank"] == 1
    assert stats["ritz_probe_requested_iterations"] == 2
    assert 1 <= stats["ritz_probe_iterations"] <= 2
    assert stats["coarse_basis_rank"] == 1
    assert stats["global_reduction_count"] >= (
        stats["solve_global_reduction_count"] +
        stats["ritz_probe_global_reduction_count"])


def test_interface_schur_hessian_solve_can_add_generalized_ritz_low_modes():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=1,
        ritz_probe_iterations=2,
        ritz_mode="generalized",
    )

    assert stats["ritz_mode"] == "generalized"
    assert stats["ritz_deflation_requested_rank"] == 1
    assert stats["ritz_deflation_basis_rank"] == 1
    assert stats["ritz_probe_requested_iterations"] == 2
    assert 1 <= stats["ritz_probe_iterations"] <= 2


def test_interface_schur_hessian_solve_can_add_m_orthogonal_lanczos_modes():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=1,
        ritz_probe_iterations=2,
        ritz_mode="m_orthogonal_lanczos",
    )

    assert stats["ritz_mode"] == "m_orthogonal_lanczos"
    assert stats["ritz_deflation_requested_rank"] == 1
    assert stats["ritz_deflation_basis_rank"] == 1
    assert stats["ritz_probe_requested_iterations"] == 2
    assert 1 <= stats["ritz_probe_iterations"] <= 2
    assert stats["ritz_lanczos_m_orthogonality_error"] < 1e-10


def test_interface_schur_hessian_solve_can_add_harmonic_ritz_modes():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=1,
        ritz_probe_iterations=2,
        ritz_mode="harmonic",
    )

    assert stats["ritz_mode"] == "harmonic"
    assert stats["ritz_deflation_requested_rank"] == 1
    assert stats["ritz_deflation_basis_rank"] == 1
    assert stats["ritz_probe_requested_iterations"] == 2
    assert 1 <= stats["ritz_probe_iterations"] <= 2


def test_interface_schur_hessian_solve_can_add_low_harmonic_hybrid_ritz_modes():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(7, 7))
    hessian = base.T @ base + 0.5 * np.eye(7)
    gradient = rng.normal(size=7)
    interface_indices = np.asarray([3, 4, 5, 6])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[
            np.asarray([3]),
            np.asarray([4]),
            np.asarray([5]),
            np.asarray([6]),
        ],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=1,
        ritz_probe_iterations=4,
        ritz_mode="low_harmonic_hybrid",
    )

    assert stats["ritz_mode"] == "low_harmonic_hybrid"
    assert stats["ritz_deflation_requested_rank"] == 1
    assert stats["ritz_deflation_basis_rank"] >= 2
    assert stats["ritz_hybrid_candidate_count"] == 2
    assert stats["ritz_hybrid_candidate_modes"] == [
        "preconditioned_operator",
        "harmonic",
    ]
    assert stats["ritz_probe_iterations"] >= 2


def test_interface_schur_hessian_solve_can_damp_coarse_correction():
    hessian = np.asarray([
        [7.0, 0.5, 1.0, 0.2],
        [0.5, 6.0, 0.3, 1.1],
        [1.0, 0.3, 4.0, 0.4],
        [0.2, 1.1, 0.4, 3.0],
    ])
    gradient = np.asarray([1.0, -0.25, 2.0, -1.0])
    interface_indices = np.asarray([2, 3])
    coarse_basis = np.eye(2)

    _, undamped_stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=1,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        coarse_basis=coarse_basis,
        use_coarse_initial_guess=True,
        coarse_regularization=0.0,
    )
    _, damped_stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=1,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        coarse_basis=coarse_basis,
        use_coarse_initial_guess=True,
        coarse_regularization=10.0,
    )

    assert undamped_stats["coarse_regularization"] == 0.0
    assert damped_stats["coarse_regularization"] == 10.0
    assert damped_stats["coarse_initial_guess_norm"] < (
        undamped_stats["coarse_initial_guess_norm"])


def test_interface_schur_central_equivalence_diagnostic_matches_schur_energy():
    hessian = np.asarray([
        [7.0, 0.5, 1.0, 0.2],
        [0.5, 6.0, 0.3, 1.1],
        [1.0, 0.3, 4.0, 0.4],
        [0.2, 1.1, 0.4, 3.0],
    ])
    gradient = np.asarray([1.0, -0.25, 2.0, -1.0])
    interface_indices = np.asarray([2, 3])

    exact_solution = np.linalg.solve(hessian, gradient)
    exact_stats = dci.interface_schur_central_equivalence_diagnostic(
        hessian=hessian,
        gradient=gradient,
        solution=exact_solution,
        interface_indices=interface_indices,
        damping=0.0,
    )
    assert exact_stats["schur_energy_gap"] < 1e-20
    assert exact_stats["private_backsubstitution_error_norm"] < 1e-12

    private = np.asarray([0, 1])
    h_pp = hessian[private[:, None], private]
    h_pb = hessian[private[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private]
    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_p = gradient[private]
    g_b = gradient[interface_indices]
    schur = h_bb - h_bp @ np.linalg.solve(h_pp, h_pb)
    rhs = g_b - h_bp @ np.linalg.solve(h_pp, g_p)
    interface_reference = np.linalg.solve(schur, rhs)
    delta = np.asarray([0.125, -0.25])
    interface_solution = interface_reference + delta
    private_solution = np.linalg.solve(h_pp, g_p - h_pb @ interface_solution)
    perturbed_solution = np.zeros_like(gradient)
    perturbed_solution[private] = private_solution
    perturbed_solution[interface_indices] = interface_solution
    expected_energy = float(delta.T @ schur @ delta)

    stats = dci.interface_schur_central_equivalence_diagnostic(
        hessian=hessian,
        gradient=gradient,
        solution=perturbed_solution,
        interface_indices=interface_indices,
        damping=0.0,
    )

    assert np.isclose(stats["schur_energy_gap"], expected_energy, atol=1e-12)
    assert np.isclose(
        stats["schur_residual_dual_energy"], expected_energy, atol=1e-12)
    assert stats["private_backsubstitution_error_norm"] < 1e-12


def test_local_schur_contribution_sum_matches_centralized_schur():
    variable_count = 5
    interface_indices = np.asarray([3, 4])

    def embed(block_indices, block, vector):
        hessian = np.zeros((variable_count, variable_count), dtype=float)
        gradient = np.zeros(variable_count, dtype=float)
        for local_row, global_row in enumerate(block_indices):
            gradient[global_row] = vector[local_row]
            for local_col, global_col in enumerate(block_indices):
                hessian[global_row, global_col] = block[local_row, local_col]
        return hessian, gradient

    h0, g0 = embed(
        [0, 1, 3, 4],
        np.asarray([
            [6.0, 0.4, 1.2, 0.3],
            [0.4, 5.0, 0.2, 1.1],
            [1.2, 0.2, 3.0, 0.5],
            [0.3, 1.1, 0.5, 4.0],
        ]),
        np.asarray([1.0, -0.25, 0.5, -1.0]),
    )
    h1, g1 = embed(
        [2, 3, 4],
        np.asarray([
            [4.5, 0.7, 0.8],
            [0.7, 2.5, 0.2],
            [0.8, 0.2, 3.2],
        ]),
        np.asarray([0.75, -0.4, 1.25]),
    )

    schur, rhs, stats = dci.sum_local_interface_schur_contributions(
        local_systems=[
            {
                "hessian": h0,
                "gradient": g0,
                "private_indices": np.asarray([0, 1]),
            },
            {
                "hessian": h1,
                "gradient": g1,
                "private_indices": np.asarray([2]),
            },
        ],
        interface_indices=interface_indices,
        variable_count=variable_count,
        damping=0.0,
    )

    hessian = h0 + h1
    gradient = g0 + g1
    private = np.asarray([0, 1, 2])
    h_pp = hessian[private[:, None], private]
    h_pb = hessian[private[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private]
    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_p = gradient[private]
    g_b = gradient[interface_indices]
    expected_schur = h_bb - h_bp @ np.linalg.solve(h_pp, h_pb)
    expected_rhs = g_b - h_bp @ np.linalg.solve(h_pp, g_p)

    assert stats["model"] == "sum_local_interface_schur_contributions"
    assert stats["local_system_count"] == 2
    assert stats["interface_variable_count"] == 2
    assert stats["local_private_variable_count_sum"] == 3
    assert np.allclose(schur, expected_schur, atol=1e-12)
    assert np.allclose(rhs, expected_rhs, atol=1e-12)


def test_normalized_interface_schur_spectral_certificate_reports_bounds():
    hessian = np.asarray([
        [4.0, -1.0],
        [-1.0, 2.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.5]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    diagonal = np.diag(hessian)
    normalized = (hessian / np.sqrt(diagonal)[:, None]) / np.sqrt(diagonal)[None, :]
    eigenvalues = np.linalg.eigvalsh(normalized)

    cert = dci.normalized_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        lambda_min_bound=1e-4,
        lambda_max_bound=2.0,
    )

    assert cert["model"] == "normalized_interface_schur_spectral_certificate"
    assert cert["materializes_dense_schur"] is True
    assert cert["interface_variable_count"] == 2
    assert np.isclose(cert["normalized_lambda_min"], float(eigenvalues[0]))
    assert np.isclose(cert["normalized_lambda_max"], float(eigenvalues[-1]))
    assert cert["lambda_min_bound"] == 1e-4
    assert cert["lambda_max_bound"] == 2.0
    assert cert["lambda_max_covered"] is True
    assert cert["lambda_min_covered"] is True
    assert cert["interval_covered"] is True


def test_normalized_interface_schur_certificate_proves_sdd_m_matrix_bound():
    hessian = np.asarray([
        [3.0, -1.0, -0.5],
        [-1.0, 2.0, -0.25],
        [-0.5, -0.25, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.5, -0.25]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.normalized_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        lambda_min_bound=1e-8,
        lambda_max_bound=2.0,
    )

    assert cert["sdd_m_matrix_theorem_applies"] is True
    assert cert["sdd_m_matrix_is_symmetric"] is True
    assert cert["sdd_m_matrix_positive_diagonal"] is True
    assert cert["sdd_m_matrix_z_offdiag"] is True
    assert cert["sdd_m_matrix_diagonal_dominance"] is True
    assert cert["sdd_m_matrix_theorem_lambda_max_bound"] == 2.0
    assert cert["sdd_m_matrix_theorem_lambda_max_covered"] is True
    assert cert["lambda_max_bound_source"] == "sdd_m_matrix_theorem"


def test_normalized_interface_schur_certificate_does_not_overclaim_non_z_matrix():
    hessian = np.asarray([
        [2.0, 0.5],
        [0.5, 2.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.5]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.normalized_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        lambda_min_bound=1e-8,
        lambda_max_bound=2.0,
    )

    assert cert["lambda_max_covered"] is True
    assert cert["sdd_m_matrix_z_offdiag"] is False
    assert cert["sdd_m_matrix_theorem_applies"] is False
    assert cert["sdd_m_matrix_theorem_lambda_max_covered"] is False
    assert cert["lambda_max_bound_source"] == "numeric_eigendecomposition"


def test_dirichlet_schur_certificate_uses_boundary_diagonal_majorizer():
    hessian = np.asarray([
        [1.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([1], dtype=int),
        },
    ]

    cert = dci.dirichlet_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 2]),
        variable_count=3,
        lambda_max_bound=2.0,
    )

    assert cert["model"] == "dirichlet_interface_schur_spectral_certificate"
    assert cert["materializes_dense_schur"] is True
    assert cert["materializes_dense_full_hessian"] is True
    assert np.isclose(cert["schur_diagonal_normalized_lambda_max"], 2.0)
    assert np.isclose(cert["boundary_diagonal_normalized_lambda_max"], 1.0)
    assert cert["full_diagonal_normalized_lambda_max"] <= 2.0
    assert cert["dirichlet_transfer_bound_applies"] is True
    assert cert["lambda_max_bound_source"] == "dirichlet_full_diagonal_numeric"


def test_dirichlet_schur_certificate_does_not_apply_when_full_bound_fails():
    hessian = np.asarray([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([2], dtype=int),
        },
    ]

    cert = dci.dirichlet_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=3,
        lambda_max_bound=2.0,
    )

    assert cert["full_diagonal_normalized_lambda_max"] > 2.0
    assert cert["dirichlet_transfer_bound_applies"] is False
    assert cert["lambda_max_bound_source"] == "numeric_only"


def test_block_gershgorin_schur_certificate_uses_perron_weights():
    hessian = np.asarray([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.5],
        [0.0, 0.5, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.block_gershgorin_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
    )

    assert cert["model"] == "block_gershgorin_interface_schur_spectral_certificate"
    assert cert["materializes_dense_schur"] is True
    assert cert["block_count"] == 3
    assert cert["block_row_sum_max"] > 1.0
    assert cert["block_coupling_spectral_radius"] < 1.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is True
    assert cert["lambda_max_bound_source"] == "weighted_block_gershgorin"
    assert cert["weighted_block_gershgorin_lambda_max_bound"] <= 2.0
    assert cert["lambda_max_covered"] is True


def test_block_gershgorin_schur_certificate_does_not_overclaim_large_coupling():
    hessian = np.asarray([
        [1.0, 0.8, 0.8],
        [0.8, 1.0, 0.8],
        [0.8, 0.8, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.block_gershgorin_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
    )

    assert cert["block_coupling_spectral_radius"] > 1.0
    assert cert["normalized_lambda_max"] > 2.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is False
    assert cert["lambda_max_covered"] is False
    assert cert["lambda_max_bound_source"] == "numeric_only"


def test_distributed_block_perron_schur_certificate_uses_collatz_bound():
    hessian = np.asarray([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.5],
        [0.0, 0.5, 1.0],
    ])
    local_systems = [
        {
            "robot": "a",
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
        perron_iterations=80,
    )

    assert cert["model"] == "distributed_block_perron_schur_spectral_certificate"
    assert cert["materializes_dense_schur"] is False
    assert cert["materializes_dense_coupling"] is False
    assert cert["block_row_sum_max"] > 1.0
    assert cert["block_coupling_collatz_upper_bound"] < 1.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is True
    assert cert["lambda_max_bound_source"] == "distributed_collatz_perron"
    assert cert["weighted_block_gershgorin_lambda_max_bound"] <= 2.0
    assert cert["offdiag_block_packet_count"] == 4
    assert cert["perron_scalar_message_count"] > 0
    assert cert["certificate_payload_bytes"] > 0


def test_distributed_block_perron_schur_certificate_does_not_overclaim():
    hessian = np.asarray([
        [1.0, 0.8, 0.8],
        [0.8, 1.0, 0.8],
        [0.8, 0.8, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
        perron_iterations=80,
    )

    assert cert["block_coupling_collatz_upper_bound"] > 1.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is False
    assert cert["lambda_max_bound_source"] == "numeric_only"


def test_distributed_block_perron_resolvent_mode_certifies_near_critical_path():
    hessian = np.asarray([
        [1.0, 0.72, 0.0],
        [0.72, 1.0, 0.69],
        [0.0, 0.69, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
        perron_iterations=600,
        perron_vector_mode="resolvent",
        resolvent_shift=0.0,
    )

    assert cert["perron_vector_mode"] == "resolvent"
    assert cert["block_row_sum_max"] > 1.0
    assert cert["block_coupling_collatz_upper_bound"] < 1.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is True


def test_distributed_block_perron_cg_resolvent_certifies_near_critical_path():
    hessian = np.asarray([
        [1.0, 0.72, 0.0],
        [0.72, 1.0, 0.69],
        [0.0, 0.69, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
        perron_iterations=8,
        perron_vector_mode="cg_resolvent",
        resolvent_shift=0.0,
    )

    assert cert["perron_vector_mode"] == "cg_resolvent"
    assert cert["cg_resolvent_iterations"] <= 3
    assert cert["block_coupling_collatz_upper_bound"] < 1.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is True
    assert cert["perron_scalar_message_count"] < 100


def test_distributed_block_perron_cg_resolvent_does_not_overclaim():
    hessian = np.asarray([
        [1.0, 0.8, 0.8],
        [0.8, 1.0, 0.8],
        [0.8, 0.8, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    cert = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
        perron_iterations=8,
        perron_vector_mode="cg_resolvent",
        resolvent_shift=0.0,
    )

    assert cert["perron_vector_mode"] == "cg_resolvent"
    assert cert["block_coupling_collatz_upper_bound"] > 1.0
    assert cert["weighted_block_gershgorin_theorem_applies"] is False
    assert cert["lambda_max_bound_source"] == "numeric_only"


def test_distributed_block_packet_construction_uses_active_interface_blocks():
    local_systems = [
        {
            "hessian": np.asarray([
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]),
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "hessian": np.asarray([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    distributed = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
        perron_iterations=8,
        perron_vector_mode="cg_resolvent",
    )
    dense = dci.block_gershgorin_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        block_dim=1,
        variable_count=3,
    )

    assert distributed["block_packet_construction"] == (
        "component_dense_schur"
    )
    assert distributed["dense_candidate_block_pair_count"] == 18
    assert distributed["active_candidate_block_pair_count"] == 5
    assert distributed["candidate_block_pair_count"] == 5
    assert distributed["active_interface_block_visit_count"] == 3
    assert distributed["skipped_inactive_block_pair_count"] == 13
    assert distributed["skipped_cross_component_block_pair_count"] == 0
    assert distributed["component_schur_assembly_count"] == 2
    assert np.isclose(
        distributed["block_coupling_collatz_upper_bound"],
        dense["block_coupling_spectral_radius"],
        atol=1e-12,
    )
    assert distributed["weighted_block_gershgorin_theorem_applies"] is True


def test_distributed_block_packet_construction_skips_cross_component_pairs():
    local_systems = [
        {
            "hessian": np.asarray([
                [1.0, 0.0],
                [0.0, 1.0],
            ]),
            "gradient": np.zeros(2, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    distributed = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        block_dim=1,
        variable_count=2,
        perron_iterations=2,
        perron_vector_mode="cg_resolvent",
    )
    dense = dci.block_gershgorin_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        block_dim=1,
        variable_count=2,
    )

    assert distributed["block_packet_construction"] == (
        "component_dense_schur"
    )
    assert distributed["dense_candidate_block_pair_count"] == 4
    assert distributed["active_candidate_block_pair_count"] == 4
    assert distributed["candidate_block_pair_count"] == 2
    assert distributed["local_interface_component_count"] == 2
    assert distributed["skipped_cross_component_block_pair_count"] == 2
    assert distributed["component_schur_assembly_count"] == 2
    assert np.isclose(
        distributed["block_coupling_collatz_upper_bound"],
        dense["block_coupling_spectral_radius"],
        atol=1e-12,
    )
    assert distributed["weighted_block_gershgorin_theorem_applies"] is True


def test_distributed_block_packet_construction_assembles_component_once():
    hessian = np.asarray([
        [3.0, 0.25, 0.8],
        [0.25, 2.0, 0.5],
        [0.8, 0.5, 4.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.zeros(3, dtype=float),
            "private_indices": np.asarray([2], dtype=int),
        },
    ]

    distributed = dci.distributed_block_perron_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        block_dim=1,
        variable_count=3,
        perron_iterations=8,
        perron_vector_mode="cg_resolvent",
    )
    dense = dci.block_gershgorin_interface_schur_spectral_certificate(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        block_dim=1,
        variable_count=3,
    )

    assert distributed["block_packet_construction"] == (
        "component_dense_schur"
    )
    assert distributed["candidate_block_pair_count"] == 4
    assert distributed["component_schur_assembly_count"] == 1
    assert distributed["component_private_solve_count"] == 1
    assert distributed["component_schur_block_pair_loop_count"] == 4
    assert np.isclose(
        distributed["block_coupling_collatz_upper_bound"],
        dense["block_coupling_spectral_radius"],
        atol=1e-12,
    )


def test_matrix_free_schur_ritz_probe_bounds_known_normalized_spectrum():
    hessian = np.asarray([
        [4.0, -1.0, 0.25],
        [-1.0, 2.5, -0.4],
        [0.25, -0.4, 1.5],
    ])
    local_systems = [
        {
            "robot": 0,
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.5, -0.25]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    diagonal = np.maximum(np.abs(np.diag(hessian)), 1e-12)
    normalized = (hessian / np.sqrt(diagonal)[:, None]) / np.sqrt(diagonal)[None, :]
    exact_max = float(np.max(np.linalg.eigvalsh(normalized)))

    cert = dci.matrix_free_interface_schur_ritz_spectral_probe(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        schur_preconditioner="diagonal",
        probe_iterations=3,
        safety_factor=1.01,
        seed=7,
    )

    assert cert["model"] == "matrix_free_interface_schur_ritz_spectral_probe"
    assert cert["materializes_dense_schur"] is False
    assert cert["schur_preconditioner"] == "diagonal"
    assert cert["probe_basis_rank"] >= 2
    assert np.isclose(cert["ritz_lambda_max"], exact_max, rtol=1e-10, atol=1e-10)
    assert cert["safe_lambda_max"] >= exact_max
    assert cert["interface_matvec_count"] == cert["probe_basis_rank"]
    assert cert["global_reduction_count"] > 0


def test_local_interface_schur_residual_envelope_bounds_exact_residual():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.eye(2, dtype=float),
            "gradient": np.asarray([1.0, 0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.eye(2, dtype=float),
            "gradient": np.asarray([-0.5, 0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    report = dci.local_interface_schur_residual_envelope(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        interface_state=np.asarray([0.0, 0.0]),
        variable_count=2,
    )

    assert report["model"] == "local_interface_schur_residual_envelope"
    assert np.allclose(report["exact_residual"], np.asarray([0.5, 0.0]))
    assert np.allclose(report["local_residual_norms"], np.asarray([1.0, 0.5]))
    assert report["exact_residual_norm"] == 0.5
    assert report["envelope_residual_norm"] == 1.5
    assert report["envelope_residual_norm"] >= report["exact_residual_norm"]
    assert report["global_reduction_count"] == 0


def test_local_interface_schur_reference_solve_matches_full_solution():
    variable_count = 5
    interface_indices = np.asarray([3, 4])

    def embed(block_indices, block, vector):
        hessian = np.zeros((variable_count, variable_count), dtype=float)
        gradient = np.zeros(variable_count, dtype=float)
        for local_row, global_row in enumerate(block_indices):
            gradient[global_row] = vector[local_row]
            for local_col, global_col in enumerate(block_indices):
                hessian[global_row, global_col] = block[local_row, local_col]
        return hessian, gradient

    h0, g0 = embed(
        [0, 1, 3, 4],
        np.asarray([
            [6.0, 0.4, 1.2, 0.3],
            [0.4, 5.0, 0.2, 1.1],
            [1.2, 0.2, 3.0, 0.5],
            [0.3, 1.1, 0.5, 4.0],
        ]),
        np.asarray([1.0, -0.25, 0.5, -1.0]),
    )
    h1, g1 = embed(
        [2, 3, 4],
        np.asarray([
            [4.5, 0.7, 0.8],
            [0.7, 2.5, 0.2],
            [0.8, 0.2, 3.2],
        ]),
        np.asarray([0.75, -0.4, 1.25]),
    )

    solution, stats = dci.solve_local_interface_schur_reference(
        local_systems=[
            {
                "hessian": h0,
                "gradient": g0,
                "private_indices": np.asarray([0, 1]),
            },
            {
                "hessian": h1,
                "gradient": g1,
                "private_indices": np.asarray([2]),
            },
        ],
        interface_indices=interface_indices,
        variable_count=variable_count,
        damping=0.0,
    )

    expected = np.linalg.solve(h0 + h1, g0 + g1)
    assert stats["model"] == "local_interface_schur_reference_solve"
    assert stats["unassigned_variable_count"] == 0
    assert stats["final_normal_residual"] < 1e-12
    assert np.allclose(solution, expected, atol=1e-12)


def test_local_interface_schur_gap_certificate_zero_for_reference_solution():
    variable_count = 3
    interface_indices = np.asarray([2])
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [4.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 3.0],
            ]),
            "gradient": np.asarray([1.0, 0.0, 1.0]),
            "private_indices": np.asarray([0], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 2.0],
                [0.0, 2.0, 3.0],
            ]),
            "gradient": np.asarray([0.0, -1.0, 1.0]),
            "private_indices": np.asarray([1], dtype=int),
        },
    ]
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=variable_count,
        damping=0.0,
    )

    certificate = dci.local_interface_schur_gap_certificate(
        local_systems=local_systems,
        interface_indices=interface_indices,
        candidate_solution=reference_solution,
        variable_count=variable_count,
        damping=0.0,
    )

    assert certificate["model"] == "local_interface_schur_gap_certificate"
    assert certificate["schur_energy_gap"] < 1e-20
    assert certificate["schur_residual_norm"] < 1e-12
    assert certificate["interface_reference_error_norm"] < 1e-12
    assert certificate["full_reference_solution_error_norm"] < 1e-12
    assert certificate["local_private_consistency_error_norm"] < 1e-12


def test_local_interface_schur_gap_certificate_matches_interface_energy():
    variable_count = 3
    interface_indices = np.asarray([2])
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [4.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 3.0],
            ]),
            "gradient": np.asarray([1.0, 0.0, 1.0]),
            "private_indices": np.asarray([0], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 2.0],
                [0.0, 2.0, 3.0],
            ]),
            "gradient": np.asarray([0.0, -1.0, 1.0]),
            "private_indices": np.asarray([1], dtype=int),
        },
    ]
    schur, rhs, _ = dci.sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=variable_count,
        damping=0.0,
    )
    interface_reference = np.linalg.solve(schur, rhs)
    delta = np.asarray([0.25])
    candidate_interface = interface_reference + delta
    candidate_solution = np.zeros(variable_count, dtype=float)
    candidate_solution[interface_indices] = candidate_interface
    for system in local_systems:
        private_indices = np.asarray(system["private_indices"], dtype=int)
        hessian = np.asarray(system["hessian"], dtype=float)
        gradient = np.asarray(system["gradient"], dtype=float)
        h_pp = hessian[private_indices[:, None], private_indices]
        h_pb = hessian[private_indices[:, None], interface_indices]
        g_p = gradient[private_indices]
        candidate_solution[private_indices] = np.linalg.solve(
            h_pp, g_p - h_pb @ candidate_interface)
    expected_energy = float(delta.T @ schur @ delta)

    certificate = dci.local_interface_schur_gap_certificate(
        local_systems=local_systems,
        interface_indices=interface_indices,
        candidate_solution=candidate_solution,
        variable_count=variable_count,
        damping=0.0,
    )

    assert np.isclose(
        certificate["schur_energy_gap"], expected_energy, atol=1e-12)
    assert np.isclose(
        certificate["schur_residual_dual_energy"], expected_energy, atol=1e-12)
    assert certificate["schur_residual_norm"] > 0.0
    assert np.isclose(
        certificate["interface_reference_error_norm"], 0.25, atol=1e-12)
    assert certificate["local_private_consistency_error_norm"] < 1e-12


def test_fixed_step_boundary_diagonal_preconditioner_avoids_private_column_solves():
    hessian = np.asarray([
        [1.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.0, -1.0]),
            "private_indices": np.asarray([1], dtype=int),
        },
    ]

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="boundary_diagonal",
    )

    assert stats["schur_preconditioner"] == "boundary_diagonal"
    assert stats["schur_preconditioner_private_solve_count"] == 0
    assert stats["schur_preconditioner_diagonal_min"] == 1.0
    assert stats["schur_preconditioner_diagonal_max"] == 1.0


def test_fixed_step_dirichlet_clipped_diagonal_preserves_bound_floor():
    hessian = np.asarray([
        [1.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 1.0],
    ])
    local_systems = [
        {
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.0, -1.0]),
            "private_indices": np.asarray([1], dtype=int),
        },
    ]

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="dirichlet_clipped_diagonal",
        dirichlet_diagonal_floor=0.75,
        fixed_step_acceleration="chebyshev_graph_normalized",
    )

    assert stats["schur_preconditioner"] == "dirichlet_clipped_diagonal"
    assert stats["dirichlet_diagonal_floor"] == 0.75
    assert stats["dirichlet_clipped_diagonal_clipped_count"] == 2
    assert stats["schur_preconditioner_private_solve_count"] == 2
    assert stats["schur_preconditioner_diagonal_min"] == 0.75
    assert stats["schur_preconditioner_diagonal_max"] == 0.75
    assert stats["chebyshev_lambda_max"] == 2.0 / 0.75
    assert stats["chebyshev_bound_source"] == "dirichlet_clipped_graph_normalized"


def test_matrix_free_local_schur_dual_certificate_matches_dense_energy_without_dense_schur():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}
    sparse_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
            hessian_storage="sparse",
        )
    )
    dense_systems, _, _ = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
            hessian_storage="dense",
        )
    )
    reference_solution, _ = dci.solve_local_interface_schur_pcg(
        local_systems=sparse_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        iterations=20,
        tolerance=1e-12,
        damping=0.0,
    )
    candidate = reference_solution.copy()
    candidate[interface_indices] += np.linspace(
        0.01, 0.02, len(interface_indices))

    dense_certificate = dci.local_interface_schur_gap_certificate(
        local_systems=dense_systems,
        interface_indices=interface_indices,
        candidate_solution=candidate,
        variable_count=build_stats["variable_count"],
        damping=0.0,
    )
    matrix_free_certificate = (
        dci.matrix_free_local_interface_schur_dual_certificate(
            local_systems=sparse_systems,
            interface_indices=interface_indices,
            candidate_solution=candidate,
            variable_count=build_stats["variable_count"],
            damping=0.0,
            dual_iterations=30,
            dual_tolerance=1e-12,
        ))

    assert matrix_free_certificate["model"] == (
        "matrix_free_local_interface_schur_dual_certificate")
    assert matrix_free_certificate["uses_dense_schur_for_certificate"] is False
    assert matrix_free_certificate["materializes_dense_local_hessian"] is False
    assert matrix_free_certificate["uses_sparse_private_factorization"] is True
    assert np.isclose(
        matrix_free_certificate["schur_residual_norm"],
        dense_certificate["schur_residual_norm"],
        atol=1e-10,
    )
    assert np.isclose(
        matrix_free_certificate["schur_residual_dual_energy"],
        dense_certificate["schur_residual_dual_energy"],
        atol=1e-10,
    )


def test_graph_local_rotation_schur_reference_matches_centralized_normal_solve():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}

    local_systems, interface_indices, stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
        )
    )
    solution, solve_stats = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=stats["variable_count"],
        damping=0.0,
    )

    matrix, rhs, _ = dci.assemble_rotation_system(
        graph_edges=edges,
        pose_ids=pose_ids,
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )
    hessian = (matrix.T @ matrix).toarray()
    gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)
    expected = np.linalg.solve(hessian, gradient)

    assert stats["model"] == "graph_local_normal_systems_for_interface_schur"
    assert stats["stage"] == "rotation"
    assert stats["interface_pose_ids"] == [1, 2]
    assert stats["local_system_count"] == 2
    assert stats["assigned_edge_count"] == len(edges)
    assert solve_stats["final_normal_residual"] < 1e-12
    assert np.allclose(solution, expected, atol=1e-12)


def test_graph_local_normal_system_builder_can_keep_sparse_hessians():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}

    local_systems, interface_indices, stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
            hessian_storage="sparse",
        )
    )

    assert stats["hessian_storage"] == "sparse"
    assert stats["materializes_dense_local_hessian"] is False
    assert stats["local_hessian_nonzero_count"] > 0
    assert len(interface_indices) == stats["interface_variable_count"]
    for system in local_systems:
        assert hasattr(system["hessian"], "tocoo")


def test_graph_local_translation_schur_reference_matches_centralized_normal_solve():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}
    rotations = {pose_id: np.eye(2, dtype=float) for pose_id in pose_ids}

    local_systems, interface_indices, stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            rotations=rotations,
            edge_owner_policy="lower_robot",
        )
    )
    solution, solve_stats = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=stats["variable_count"],
        damping=0.0,
    )

    matrix, rhs, _ = dci.assemble_translation_system(
        graph_edges=edges,
        pose_ids=pose_ids,
        rotations=rotations,
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )
    hessian = (matrix.T @ matrix).toarray()
    gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)
    expected = np.linalg.solve(hessian, gradient)

    assert stats["model"] == "graph_local_normal_systems_for_interface_schur"
    assert stats["stage"] == "translation"
    assert stats["interface_pose_ids"] == [1, 2]
    assert stats["local_system_count"] == 2
    assert stats["assigned_edge_count"] == len(edges)
    assert solve_stats["final_normal_residual"] < 1e-12
    assert np.allclose(solution, expected, atol=1e-12)


def test_graph_local_rotation_schur_pcg_matches_dense_reference_without_dense_schur():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
        )
    )

    pcg_solution, pcg_stats = dci.solve_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        iterations=20,
        tolerance=1e-12,
        damping=0.0,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        damping=0.0,
    )

    assert pcg_stats["model"] == "local_interface_schur_pcg"
    assert pcg_stats["materializes_dense_schur"] is False
    assert pcg_stats["final_schur_residual"] < 1e-10
    assert pcg_stats["final_normal_residual"] < 1e-10
    assert pcg_stats["interface_matvec_count"] > 0
    assert pcg_stats["topology_reduction_count"] >= pcg_stats["iterations"]
    assert np.allclose(pcg_solution, reference_solution, atol=1e-10)


def test_local_schur_pcg_diagonal_preconditioner_solves_scaled_diagonal_system_fast():
    local_systems = [{
        "hessian": np.diag([1.0, 100.0]),
        "gradient": np.asarray([1.0, 1.0]),
        "private_indices": np.asarray([], dtype=int),
    }]
    interface_indices = np.asarray([0, 1], dtype=int)

    unpreconditioned_solution, unpreconditioned_stats = (
        dci.solve_local_interface_schur_pcg(
            local_systems=local_systems,
            interface_indices=interface_indices,
            variable_count=2,
            iterations=1,
            tolerance=1e-14,
            damping=0.0,
            schur_preconditioner="none",
        )
    )
    preconditioned_solution, preconditioned_stats = (
        dci.solve_local_interface_schur_pcg(
            local_systems=local_systems,
            interface_indices=interface_indices,
            variable_count=2,
            iterations=1,
            tolerance=1e-14,
            damping=0.0,
            schur_preconditioner="diagonal",
        )
    )

    assert unpreconditioned_stats["final_schur_residual"] > 1e-3
    assert unpreconditioned_stats["schur_preconditioner"] == "none"
    assert preconditioned_stats["schur_preconditioner"] == "diagonal"
    assert preconditioned_stats["schur_preconditioner_private_solve_count"] == 0
    assert preconditioned_stats["final_schur_residual"] < 1e-12
    assert np.allclose(preconditioned_solution, np.asarray([1.0, 0.01]),
                       atol=1e-12)


def test_sparse_graph_local_schur_pcg_avoids_dense_local_hessian_materialization():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}
    sparse_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
            hessian_storage="sparse",
        )
    )
    dense_systems, _, _ = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
            hessian_storage="dense",
        )
    )

    pcg_solution, pcg_stats = dci.solve_local_interface_schur_pcg(
        local_systems=sparse_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        iterations=20,
        tolerance=1e-12,
        damping=0.0,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=dense_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        damping=0.0,
    )

    assert pcg_stats["materializes_dense_local_hessian"] is False
    assert pcg_stats["uses_sparse_private_factorization"] is True
    assert pcg_stats["final_schur_residual"] < 1e-10
    assert pcg_stats["final_normal_residual"] < 1e-10
    assert np.allclose(pcg_solution, reference_solution, atol=1e-10)


def test_graph_local_translation_schur_pcg_matches_dense_reference_without_dense_schur():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}
    rotations = {pose_id: np.eye(2, dtype=float) for pose_id in pose_ids}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            rotations=rotations,
            edge_owner_policy="lower_robot",
        )
    )

    pcg_solution, pcg_stats = dci.solve_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        iterations=20,
        tolerance=1e-12,
        damping=0.0,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        damping=0.0,
    )

    assert pcg_stats["model"] == "local_interface_schur_pcg"
    assert pcg_stats["materializes_dense_schur"] is False
    assert pcg_stats["final_schur_residual"] < 1e-10
    assert pcg_stats["final_normal_residual"] < 1e-10
    assert pcg_stats["interface_matvec_count"] > 0
    assert pcg_stats["topology_reduction_count"] >= pcg_stats["iterations"]
    assert np.allclose(pcg_solution, reference_solution, atol=1e-10)


def test_graph_local_schur_pcg_reports_separator_owner_payload():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
        )
    )

    _, stats = dci.solve_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        iterations=20,
        tolerance=1e-12,
        damping=0.0,
    )

    expected_shared_pose2_scalars = 4
    expected_interface_bytes = (
        (stats["interface_matvec_count"] + stats["rhs_reduction_count"]) *
        expected_shared_pose2_scalars * 8
    )
    expected_global_scalar_bytes = stats["pcg_global_reduction_count"] * 8

    assert stats["communication_model"] == "separator_owner_star"
    assert stats["separator_owner_count"] == 1
    assert stats["separator_shared_scalar_count"] == expected_shared_pose2_scalars
    assert stats["separator_owner_pair_payload_bytes"] == {
        "1->0": expected_interface_bytes,
    }
    assert stats["separator_owner_interface_payload_bytes"] == (
        expected_interface_bytes
    )
    assert stats["global_scalar_reduction_payload_bytes"] == (
        expected_global_scalar_bytes
    )
    assert stats["estimated_comm_bytes"] == (
        expected_interface_bytes + expected_global_scalar_bytes
    )
    assert stats["estimated_comm_mb"] == (
        stats["estimated_comm_bytes"] / (1024.0 * 1024.0)
    )


def test_local_schur_pcg_separator_tree_payload_routes_over_topology():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([[0.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 2,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    stats = dci.estimate_local_interface_schur_pcg_communication(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        interface_matvec_count=3,
        pcg_global_reduction_count=4,
        rhs_reduction_count=1,
        scalar_bytes=8,
        communication_model="separator_tree",
        robot_topology_edges=[(0, 1), (1, 2)],
    )

    expected_exchange_rounds = 4
    expected_hop_payload = expected_exchange_rounds * 8
    expected_interface_payload = 2 * expected_hop_payload
    expected_global_scalar_payload = 2 * 4 * 8

    assert stats["communication_model"] == "separator_tree"
    assert stats["separator_owner_count"] == 1
    assert stats["separator_shared_scalar_count"] == 1
    assert stats["separator_tree_hop_count_per_round"] == 2
    assert stats["separator_owner_pair_payload_bytes"] == {
        "2->1": expected_hop_payload,
        "1->0": expected_hop_payload,
    }
    assert stats["separator_owner_interface_payload_bytes"] == (
        expected_interface_payload
    )
    assert stats["global_scalar_reduction_payload_bytes"] == (
        expected_global_scalar_payload
    )
    assert stats["global_scalar_reduction_hop_count_per_reduction"] == 2
    assert stats["global_scalar_reduction_pair_payload_bytes"] == {
        "2->1": 4 * 8,
        "1->0": 4 * 8,
    }
    assert stats["estimated_comm_bytes"] == (
        expected_interface_payload + expected_global_scalar_payload
    )


def test_separator_tree_global_reductions_are_routed_over_topology():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([[0.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 2,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    stats = dci.estimate_local_interface_schur_pcg_communication(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        interface_matvec_count=0,
        pcg_global_reduction_count=4,
        rhs_reduction_count=0,
        scalar_bytes=8,
        communication_model="separator_tree",
        robot_topology_edges=[(0, 1), (1, 2)],
    )

    expected_per_hop_payload = 4 * 8

    assert stats["global_scalar_reduction_hop_count_per_reduction"] == 2
    assert stats["global_scalar_reduction_pair_payload_bytes"] == {
        "2->1": expected_per_hop_payload,
        "1->0": expected_per_hop_payload,
    }
    assert stats["global_scalar_reduction_payload_bytes"] == (
        2 * expected_per_hop_payload
    )


def test_graph_local_schur_pcg_can_report_separator_tree_payload():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
        )
    )

    _, stats = dci.solve_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        iterations=20,
        tolerance=1e-12,
        damping=0.0,
        communication_model="separator_tree",
        robot_topology_edges=[(0, 1), (1, 2)],
    )

    assert stats["communication_model"] == "separator_tree"
    assert stats["materializes_dense_schur"] is False
    assert stats["separator_tree_hop_count_per_round"] > 0
    assert stats["estimated_comm_bytes"] > 0


def test_local_schur_fixed_step_solves_without_global_reductions():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([[2.0]]),
            "gradient": np.asarray([2.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([[2.0]]),
            "gradient": np.asarray([2.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    solution, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        variable_count=1,
        iterations=1,
        relaxation=1.0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        communication_model="separator_tree",
        robot_topology_edges=[(0, 1)],
    )

    assert np.allclose(solution, np.asarray([1.0]), atol=1e-12)
    assert stats["model"] == "local_interface_schur_fixed_step"
    assert stats["global_reduction_count"] == 0
    assert stats["global_scalar_reduction_payload_bytes"] == 0
    assert stats["interface_matvec_count"] == 1
    assert stats["final_schur_residual"] < 1e-12
    assert stats["final_normal_residual"] < 1e-12


def test_local_schur_fixed_step_coarse_initial_guess_removes_low_mode():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [2.0, -1.0],
                [-1.0, 2.0],
            ]),
            "gradient": np.asarray([1.0, 1.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    coarse_basis = np.asarray([[1.0], [1.0]])

    no_coarse_solution, no_coarse_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
    )
    coarse_solution, coarse_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        coarse_basis=coarse_basis,
    )

    assert not np.allclose(no_coarse_solution, np.asarray([1.0, 1.0]))
    assert no_coarse_stats["coarse_initial_guess_used"] is False
    assert np.allclose(coarse_solution, np.asarray([1.0, 1.0]), atol=1e-12)
    assert coarse_stats["coarse_initial_guess_used"] is True
    assert coarse_stats["coarse_basis_rank"] == 1
    assert coarse_stats["coarse_setup_matvec_count"] == 1
    assert coarse_stats["coarse_initial_residual"] < 1e-12
    assert coarse_stats["final_schur_residual"] < 1e-12


def test_local_schur_fixed_step_residual_krylov_coarse_basis_solves_two_modes():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [3.0, 1.0],
                [1.0, 2.0],
            ]),
            "gradient": np.asarray([1.0, 0.25]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    reference = np.linalg.solve(
        np.asarray([[3.0, 1.0], [1.0, 2.0]]),
        np.asarray([1.0, 0.25]),
    )

    rank1_solution, rank1_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        adaptive_coarse_basis="residual_krylov",
        adaptive_coarse_rank=1,
    )
    rank2_solution, rank2_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        adaptive_coarse_basis="residual_krylov",
        adaptive_coarse_rank=2,
    )

    assert rank1_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert rank1_stats["coarse_basis_rank"] == 1
    assert not np.allclose(rank1_solution, reference, atol=1e-8)
    assert rank2_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert rank2_stats["coarse_basis_rank"] == 2
    assert rank2_stats["coarse_setup_matvec_count"] >= 2
    assert np.allclose(rank2_solution, reference, atol=1e-12)
    assert rank2_stats["final_schur_residual"] < 1e-12


def test_local_schur_fixed_step_ritz_probe_coarse_basis_solves_two_modes():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [4.0, -1.0],
                [-1.0, 2.0],
            ]),
            "gradient": np.asarray([0.5, -1.25]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    reference = np.linalg.solve(
        np.asarray([[4.0, -1.0], [-1.0, 2.0]]),
        np.asarray([0.5, -1.25]),
    )

    solution, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        adaptive_coarse_basis="deterministic_ritz",
        adaptive_coarse_rank=2,
        adaptive_coarse_probe_count=2,
        adaptive_coarse_seed=7,
    )

    assert stats["adaptive_coarse_basis"] == "deterministic_ritz"
    assert stats["adaptive_coarse_rank"] == 2
    assert stats["adaptive_coarse_probe_count"] == 2
    assert stats["adaptive_coarse_ritz_selected_rank"] == 2
    assert stats["coarse_basis_rank"] == 2
    assert stats["coarse_initial_guess_used"]
    assert np.allclose(solution, reference, atol=1e-12)
    assert stats["final_schur_residual"] < 1e-12


def test_local_schur_ritz_probe_selects_rhs_coupled_mode_not_lowest_mode():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.0],
                [0.0, 10.0],
            ]),
            "gradient": np.asarray([0.0, 10.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    solution, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        adaptive_coarse_basis="deterministic_ritz",
        adaptive_coarse_rank=1,
        adaptive_coarse_probe_count=2,
        adaptive_coarse_seed=3,
    )

    assert stats["adaptive_coarse_basis"] == "deterministic_ritz"
    assert stats["adaptive_coarse_ritz_selected_rank"] == 1
    assert np.allclose(solution, np.asarray([0.0, 1.0]), atol=1e-12)
    assert stats["final_schur_residual"] < 1e-12


def test_local_schur_fixed_step_combines_explicit_and_adaptive_coarse_bases():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.0],
                [0.0, 2.0],
            ]),
            "gradient": np.asarray([1.0, 2.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    base_basis = np.asarray([[1.0], [0.0]], dtype=float)

    solution, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        coarse_basis=base_basis,
        adaptive_coarse_basis="residual_krylov",
        adaptive_coarse_rank=1,
    )

    assert stats["coarse_basis_column_count"] == 2
    assert stats["coarse_basis_rank"] == 2
    assert stats["adaptive_coarse_basis"] == "residual_krylov"
    assert stats["adaptive_coarse_generation_matvec_count"] >= 1
    assert np.allclose(solution, np.asarray([1.0, 1.0]), atol=1e-12)
    assert stats["final_schur_residual"] < 1e-12


def test_local_schur_chebyshev_fixed_step_accelerates_known_spectrum():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.0],
                [0.0, 50.0],
            ]),
            "gradient": np.asarray([1.0, 50.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, richardson_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=30,
        tolerance=1e-12,
        schur_preconditioner="none",
        relaxation=1.0 / 50.0,
    )
    chebyshev_solution, chebyshev_stats = (
        dci.solve_local_interface_schur_fixed_step(
            local_systems=local_systems,
            interface_indices=np.asarray([0, 1]),
            variable_count=2,
            iterations=30,
            tolerance=1e-12,
            schur_preconditioner="none",
            fixed_step_acceleration="chebyshev",
            chebyshev_lambda_min=1.0,
            chebyshev_lambda_max=50.0,
        )
    )

    assert chebyshev_stats["fixed_step_acceleration"] == "chebyshev"
    assert chebyshev_stats["global_reduction_count"] == 0
    assert chebyshev_stats["chebyshev_lambda_min"] == 1.0
    assert chebyshev_stats["chebyshev_lambda_max"] == 50.0
    assert chebyshev_stats["final_schur_residual"] < (
        0.25 * richardson_stats["final_schur_residual"]
    )
    assert np.allclose(chebyshev_solution, np.asarray([1.0, 1.0]), atol=1e-2)


def test_local_schur_chebyshev_fixed_step_rejects_invalid_spectral_bounds():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.eye(2, dtype=float),
            "gradient": np.ones(2, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    try:
        dci.solve_local_interface_schur_fixed_step(
            local_systems=local_systems,
            interface_indices=np.asarray([0, 1]),
            variable_count=2,
            iterations=2,
            schur_preconditioner="none",
            fixed_step_acceleration="chebyshev",
            chebyshev_lambda_min=2.0,
            chebyshev_lambda_max=1.0,
        )
    except ValueError as exc:
        assert "chebyshev" in str(exc).lower()
    else:
        raise AssertionError("invalid Chebyshev spectral bounds were accepted")


def test_local_schur_chebyshev_auto_gershgorin_bounds_cover_known_spectrum():
    hessian = np.asarray([
        [4.0, -1.0],
        [-1.0, 2.0],
    ])
    local_systems = [
        {
            "robot": 0,
            "hessian": hessian,
            "gradient": np.asarray([1.0, 0.5]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    diagonal = np.diag(hessian)
    normalized = (hessian / np.sqrt(diagonal)[:, None]) / np.sqrt(diagonal)[None, :]
    exact_max = float(np.max(np.linalg.eigvalsh(normalized)))

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration="chebyshev_auto_gershgorin",
    )

    assert stats["fixed_step_acceleration"] == "chebyshev_auto_gershgorin"
    assert stats["chebyshev_bound_source"] == "normalized_gershgorin"
    assert stats["chebyshev_lambda_max"] >= exact_max
    assert stats["chebyshev_gershgorin_lambda_max"] == stats["chebyshev_lambda_max"]
    assert 0.0 < stats["chebyshev_lambda_min"] < stats["chebyshev_lambda_max"]
    assert stats["chebyshev_bound_setup_matvec_count"] == 2
    assert stats["global_reduction_count"] == 0


def test_local_schur_chebyshev_graph_normalized_uses_no_setup_matvecs():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [4.0, -1.0],
                [-1.0, 2.0],
            ]),
            "gradient": np.asarray([1.0, 0.5]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration="chebyshev_graph_normalized",
    )

    assert stats["fixed_step_acceleration"] == "chebyshev_graph_normalized"
    assert stats["chebyshev_bound_source"] == "graph_normalized_heuristic"
    assert stats["chebyshev_lambda_min"] == 1e-8
    assert stats["chebyshev_lambda_max"] == 2.0
    assert stats["chebyshev_bound_setup_matvec_count"] == 0
    assert stats["interface_matvec_count"] == 0
    assert stats["global_reduction_count"] == 0


def test_local_schur_chebyshev_ritz_probe_reports_setup_cost():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [4.0, -1.0, 0.25],
                [-1.0, 2.5, -0.4],
                [0.25, -0.4, 1.5],
            ]),
            "gradient": np.asarray([1.0, 0.5, -0.25]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([
                [0.5, 0.0, 0.1],
                [0.0, 0.25, 0.0],
                [0.1, 0.0, 0.4],
            ]),
            "gradient": np.asarray([0.0, 0.25, -0.1]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, graph_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration="chebyshev_graph_normalized",
    )
    _, ritz_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration="chebyshev_ritz_probe",
        chebyshev_ritz_probe_iterations=3,
        chebyshev_ritz_safety_factor=1.01,
        chebyshev_ritz_seed=7,
    )

    assert ritz_stats["fixed_step_acceleration"] == "chebyshev_ritz_probe"
    assert ritz_stats["chebyshev_bound_source"] == "matrix_free_ritz_probe"
    assert ritz_stats["chebyshev_bound_setup_matvec_count"] > 0
    assert ritz_stats["chebyshev_bound_global_reduction_count"] > 0
    assert ritz_stats["global_reduction_count"] == (
        ritz_stats["chebyshev_bound_global_reduction_count"]
    )
    assert ritz_stats["chebyshev_ritz_probe_basis_rank"] >= 2
    assert ritz_stats["estimated_comm_bytes"] > graph_stats["estimated_comm_bytes"]


def test_local_schur_chebyshev_block_certificate_sets_safe_endpoint():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.5],
                [0.0, 0.5, 1.0],
            ]),
            "gradient": np.ones(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration="chebyshev_block_gershgorin_certificate",
        chebyshev_certificate_block_dim=1,
        chebyshev_certificate_iterations=8,
    )

    assert stats["fixed_step_acceleration"] == (
        "chebyshev_block_gershgorin_certificate"
    )
    assert stats["chebyshev_bound_source"] == (
        "distributed_block_gershgorin_certificate"
    )
    assert stats["chebyshev_block_certificate_theorem_applies"] is True
    assert stats["chebyshev_lambda_max"] <= 2.0
    assert stats["chebyshev_block_certificate_payload_bytes"] > 0
    assert stats["chebyshev_bound_global_reduction_count"] > 0


def test_local_schur_budgeted_certificate_skips_when_preflight_exceeds_budget():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.5],
                [0.0, 0.5, 1.0],
            ]),
            "gradient": np.ones(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration=(
            "chebyshev_block_gershgorin_budgeted_certificate"),
        chebyshev_certificate_block_dim=1,
        chebyshev_certificate_iterations=8,
        chebyshev_certificate_max_payload_mb=1e-12,
    )

    assert stats["fixed_step_acceleration"] == (
        "chebyshev_block_gershgorin_budgeted_certificate"
    )
    assert stats["chebyshev_certificate_policy_decision"] == "skip_budget"
    assert stats["chebyshev_bound_source"] == (
        "distributed_block_gershgorin_certificate_budget_skipped"
    )
    assert stats["chebyshev_lambda_max"] == 2.0
    assert stats["chebyshev_certificate_preflight_payload_upper_bytes"] > 0
    assert stats["chebyshev_block_certificate_payload_bytes"] == 0
    assert stats["chebyshev_bound_global_reduction_count"] == 0


def test_local_schur_budgeted_certificate_runs_when_preflight_fits_budget():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.5],
                [0.0, 0.5, 1.0],
            ]),
            "gradient": np.ones(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration=(
            "chebyshev_block_gershgorin_budgeted_certificate"),
        chebyshev_certificate_block_dim=1,
        chebyshev_certificate_iterations=8,
        chebyshev_certificate_max_payload_mb=1.0,
    )

    assert stats["chebyshev_certificate_policy_decision"] == "run_budget_ok"
    assert stats["chebyshev_bound_source"] == (
        "distributed_block_gershgorin_certificate"
    )
    assert stats["chebyshev_block_certificate_theorem_applies"] is True
    assert stats["chebyshev_block_certificate_payload_bytes"] > 0
    assert stats["chebyshev_certificate_preflight_payload_upper_bytes"] >= (
        stats["chebyshev_block_certificate_payload_bytes"]
    )


def test_local_schur_budgeted_certificate_can_run_under_hard_cap_and_postcheck():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.5],
                [0.0, 0.5, 1.0],
            ]),
            "gradient": np.ones(3, dtype=float),
            "private_indices": np.asarray([], dtype=int),
        },
    ]
    soft_budget_bytes = 300

    _, stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1, 2]),
        variable_count=3,
        iterations=0,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration=(
            "chebyshev_block_gershgorin_budgeted_certificate"),
        chebyshev_certificate_block_dim=1,
        chebyshev_certificate_iterations=8,
        chebyshev_certificate_vector_mode="cg_resolvent",
        chebyshev_certificate_max_payload_mb=(
            soft_budget_bytes / 1024.0 / 1024.0),
        chebyshev_certificate_preflight_hard_cap_mb=1.0,
    )

    assert stats["chebyshev_certificate_policy_decision"] == (
        "run_postcheck_budget_ok"
    )
    assert stats["chebyshev_certificate_preflight_payload_upper_bytes"] > (
        soft_budget_bytes
    )
    assert stats["chebyshev_block_certificate_payload_bytes"] <= (
        soft_budget_bytes
    )
    assert stats["chebyshev_bound_source"] == (
        "distributed_block_gershgorin_certificate"
    )


def test_local_schur_chebyshev_safety_monitor_rejects_residual_growth():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.0],
                [0.0, 50.0],
            ]),
            "gradient": np.asarray([1.0, 50.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, unsafe_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=6,
        tolerance=1e-12,
        schur_preconditioner="none",
        fixed_step_acceleration="chebyshev",
        chebyshev_lambda_min=1.0,
        chebyshev_lambda_max=5.0,
    )
    _, monitored_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=6,
        tolerance=1e-12,
        schur_preconditioner="none",
        fixed_step_acceleration="chebyshev",
        chebyshev_lambda_min=1.0,
        chebyshev_lambda_max=5.0,
        chebyshev_safety_monitor="residual_growth",
        chebyshev_safety_growth_factor=1.1,
    )

    assert monitored_stats["chebyshev_safety_monitor"] == "residual_growth"
    assert monitored_stats["chebyshev_safety_triggered"] is True
    assert monitored_stats["chebyshev_safety_trigger_iteration"] == 0
    assert monitored_stats["chebyshev_safety_rejected_residual"] > (
        monitored_stats["chebyshev_safety_accepted_residual"]
    )
    assert monitored_stats["final_schur_residual"] < (
        unsafe_stats["final_schur_residual"]
    )
    assert monitored_stats["global_reduction_count"] == 0


def test_local_schur_chebyshev_safety_monitor_can_use_local_envelope():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [1.0, 0.0],
                [0.0, 50.0],
            ]),
            "gradient": np.asarray([1.0, 50.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    _, unsafe_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=6,
        tolerance=1e-12,
        schur_preconditioner="none",
        fixed_step_acceleration="chebyshev",
        chebyshev_lambda_min=1.0,
        chebyshev_lambda_max=5.0,
    )
    _, monitored_stats = dci.solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=np.asarray([0, 1]),
        variable_count=2,
        iterations=6,
        tolerance=1e-12,
        schur_preconditioner="none",
        fixed_step_acceleration="chebyshev",
        chebyshev_lambda_min=1.0,
        chebyshev_lambda_max=5.0,
        chebyshev_safety_monitor="local_residual_envelope",
        chebyshev_safety_growth_factor=1.1,
    )

    assert monitored_stats["chebyshev_safety_monitor"] == "local_residual_envelope"
    assert monitored_stats["chebyshev_safety_metric"] == "local_residual_envelope"
    assert monitored_stats["chebyshev_safety_triggered"] is True
    assert monitored_stats["chebyshev_safety_trigger_iteration"] == 0
    assert monitored_stats["chebyshev_safety_rejected_metric"] > (
        monitored_stats["chebyshev_safety_accepted_metric"]
    )
    assert monitored_stats["chebyshev_safety_envelope_evaluation_count"] > 0
    assert monitored_stats["final_schur_residual"] < (
        unsafe_stats["final_schur_residual"]
    )
    assert monitored_stats["global_reduction_count"] == 0


def test_limited_hop_local_schur_matvec_reports_missing_remote_contribution():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([[0.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 2,
            "hessian": np.asarray([[3.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    one_hop = dci.limited_hop_local_interface_schur_matvec_diagnostic(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        vector=np.asarray([1.0]),
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=1,
    )
    two_hop = dci.limited_hop_local_interface_schur_matvec_diagnostic(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        vector=np.asarray([1.0]),
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=2,
    )

    assert one_hop["model"] == "limited_hop_local_interface_schur_matvec"
    assert np.allclose(one_hop["exact_matvec"], [4.0], atol=1e-12)
    assert np.allclose(one_hop["approx_matvec"], [1.0], atol=1e-12)
    assert one_hop["matvec_error_norm"] == 3.0
    assert one_hop["covered_participant_count"] == 1
    assert one_hop["total_participant_count"] == 2
    assert one_hop["limited_hop_payload_bytes"] == 0

    assert np.allclose(two_hop["exact_matvec"], [4.0], atol=1e-12)
    assert np.allclose(two_hop["approx_matvec"], [4.0], atol=1e-12)
    assert two_hop["matvec_error_norm"] == 0.0
    assert two_hop["covered_participant_count"] == 2
    assert two_hop["limited_hop_payload_bytes"] == 16


def test_limited_hop_local_schur_pcg_reports_reference_error_when_radius_is_too_small():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([1.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([[0.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 2,
            "hessian": np.asarray([[3.0]]),
            "gradient": np.asarray([3.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    approximate_solution, stats = dci.solve_limited_hop_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        variable_count=1,
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=0,
        iterations=5,
        tolerance=1e-12,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        variable_count=1,
        damping=0.0,
    )

    assert stats["model"] == "limited_hop_local_interface_schur_pcg"
    assert stats["hop_radius"] == 0
    assert stats["materializes_dense_schur"] is False
    assert stats["uses_exact_rhs"] is True
    assert stats["max_limited_hop_matvec_error_norm"] > 0.0
    assert stats["final_exact_schur_residual"] > 1.0
    assert stats["reference_solution_error_norm"] > 1.0
    assert stats["schur_energy_gap"] > 1.0
    assert stats["schur_residual_dual_energy"] > 1.0
    assert stats["limited_hop_payload_bytes"] == 0
    assert not np.allclose(approximate_solution, reference_solution, atol=1e-8)


def test_limited_hop_local_schur_pcg_matches_reference_when_radius_covers_topology():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([[1.0]]),
            "gradient": np.asarray([1.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.asarray([[0.0]]),
            "gradient": np.asarray([0.0]),
            "private_indices": np.asarray([], dtype=int),
        },
        {
            "robot": 2,
            "hessian": np.asarray([[3.0]]),
            "gradient": np.asarray([3.0]),
            "private_indices": np.asarray([], dtype=int),
        },
    ]

    solution, stats = dci.solve_limited_hop_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        variable_count=1,
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=2,
        iterations=5,
        tolerance=1e-12,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=np.asarray([0]),
        variable_count=1,
        damping=0.0,
    )

    assert stats["max_limited_hop_matvec_error_norm"] == 0.0
    assert stats["final_exact_schur_residual"] < 1e-12
    assert stats["reference_solution_error_norm"] < 1e-12
    assert stats["limited_hop_payload_bytes"] > 0
    assert np.allclose(solution, reference_solution, atol=1e-12)


def test_limited_hop_local_schur_pcg_applies_private_damping_in_matvec():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.asarray([
                [2.0, 1.0],
                [1.0, 3.0],
            ]),
            "gradient": np.asarray([2.0, 1.0]),
            "private_indices": np.asarray([0], dtype=int),
        },
    ]

    solution, stats = dci.solve_limited_hop_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=np.asarray([1]),
        variable_count=2,
        robot_topology_edges=[],
        hop_radius=0,
        iterations=5,
        tolerance=1e-12,
        damping=1.0,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=np.asarray([1]),
        variable_count=2,
        damping=1.0,
    )

    assert stats["max_limited_hop_matvec_error_norm"] < 1e-12
    assert stats["final_exact_schur_residual"] < 1e-12
    assert stats["reference_solution_error_norm"] < 1e-12
    assert np.allclose(solution, reference_solution, atol=1e-12)


def test_graph_local_limited_hop_matvec_matches_exact_when_radius_covers_topology():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
        )
    )
    vector = np.zeros(len(interface_indices), dtype=float)
    vector[0] = 1.0
    dense_schur, _, _ = dci.sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        damping=0.0,
    )
    exact_matvec = dense_schur @ vector

    zero_hop = dci.limited_hop_local_interface_schur_matvec_diagnostic(
        local_systems=local_systems,
        interface_indices=interface_indices,
        vector=vector,
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=0,
    )
    one_hop = dci.limited_hop_local_interface_schur_matvec_diagnostic(
        local_systems=local_systems,
        interface_indices=interface_indices,
        vector=vector,
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=1,
    )

    assert zero_hop["matvec_error_norm"] > 0.0
    assert zero_hop["coverage_fraction"] < 1.0
    assert np.allclose(one_hop["exact_matvec"], exact_matvec, atol=1e-12)
    assert np.allclose(one_hop["approx_matvec"], exact_matvec, atol=1e-12)
    assert one_hop["matvec_error_norm"] < 1e-12


def test_graph_local_limited_hop_schur_pcg_matches_reference_when_radius_covers_topology():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
            edge_owner_policy="lower_robot",
        )
    )

    solution, stats = dci.solve_limited_hop_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=1,
        iterations=20,
        tolerance=1e-12,
    )
    reference_solution, _ = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
        damping=0.0,
    )

    assert stats["model"] == "limited_hop_local_interface_schur_pcg"
    assert stats["max_limited_hop_matvec_error_norm"] < 1e-12
    assert stats["final_exact_schur_residual"] < 1e-10
    assert stats["reference_solution_error_norm"] < 1e-10
    assert np.allclose(solution, reference_solution, atol=1e-10)


def test_graph_local_schur_gap_decomposition_exact_pcg_has_zero_gap():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    solution, stats = dci.graph_local_interface_schur_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        edge_owner_policy="lower_robot",
        solver="pcg",
        iterations=20,
        tolerance=1e-12,
    )

    assert stats["model"] == "graph_local_interface_schur_gap_decomposition"
    assert stats["stage"] == "rotation"
    assert stats["solver"] == "pcg"
    assert stats["schur_energy_gap"] < 1e-20
    assert stats["schur_residual_norm"] < 1e-10
    assert stats["interface_reference_error_norm"] < 1e-10
    assert stats["local_private_consistency_error_norm"] < 1e-10
    assert stats["solver_stats"]["model"] == "local_interface_schur_pcg"
    assert len(solution) == stats["variable_count"]


def test_graph_local_schur_gap_decomposition_can_use_sparse_solver_storage():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}

    _, stats = dci.graph_local_interface_schur_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        edge_owner_policy="lower_robot",
        solver="pcg",
        hessian_storage="sparse",
        iterations=20,
        tolerance=1e-12,
    )

    assert stats["hessian_storage"] == "sparse"
    assert stats["build_stats"]["materializes_dense_local_hessian"] is False
    assert stats["solver_stats"]["materializes_dense_local_hessian"] is False
    assert stats["solver_stats"]["uses_sparse_private_factorization"] is True
    assert stats["certificate"]["uses_dense_schur_for_certificate"] is True
    assert stats["schur_energy_gap"] < 1e-20


def test_graph_local_schur_gap_decomposition_can_use_budgeted_fixed_step():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}

    _, stats = dci.graph_local_interface_schur_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        edge_owner_policy="lower_robot",
        solver="fixed_step",
        hessian_storage="sparse",
        iterations=3,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration=(
            "chebyshev_block_gershgorin_budgeted_certificate"),
        chebyshev_certificate_block_dim=4,
        chebyshev_certificate_iterations=8,
        chebyshev_certificate_max_payload_mb=1.0,
    )

    assert stats["solver"] == "fixed_step"
    assert stats["solver_stats"]["model"] == "local_interface_schur_fixed_step"
    assert stats["solver_stats"]["fixed_step_acceleration"] == (
        "chebyshev_block_gershgorin_budgeted_certificate"
    )
    assert stats["solver_stats"]["chebyshev_certificate_policy_decision"] in {
        "run_budget_ok",
        "run_postcheck_budget_ok",
        "run_unbounded_budget",
    }
    assert stats["solver_stats"]["chebyshev_block_certificate_payload_bytes"] > 0
    assert stats["estimated_comm_mb"] > 0.0


def test_graph_local_schur_gap_decomposition_can_use_matrix_free_dual_certificate():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(0.0, 1.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(0.0, 1.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2, 3, 4]
    robot_of = {0: 0, 1: 0, 4: 0, 2: 1, 3: 1}

    _, stats = dci.graph_local_interface_schur_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        edge_owner_policy="lower_robot",
        solver="pcg",
        hessian_storage="sparse",
        certificate_mode="matrix_free_dual",
        iterations=20,
        tolerance=1e-12,
    )

    assert stats["certificate_mode"] == "matrix_free_dual"
    assert stats["certificate"]["uses_dense_schur_for_certificate"] is False
    assert stats["certificate"]["materializes_dense_local_hessian"] is False
    assert stats["certificate"]["uses_sparse_private_factorization"] is True
    assert stats["schur_residual_dual_energy"] < 1e-18
    assert stats["schur_residual_norm"] < 1e-10


def test_graph_local_schur_gap_decomposition_limited_hop_reports_nonzero_gap():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, stats = dci.graph_local_interface_schur_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        edge_owner_policy="lower_robot",
        solver="limited_hop_pcg",
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=0,
        iterations=20,
        tolerance=1e-12,
    )

    assert stats["solver"] == "limited_hop_pcg"
    assert stats["schur_energy_gap"] > 0.0
    assert stats["schur_residual_norm"] > 0.0
    assert stats["interface_reference_error_norm"] > 0.0
    assert stats["solver_stats"]["coverage_fraction"] < 1.0
    assert stats["solver_stats"]["max_limited_hop_matvec_error_norm"] > 0.0


def test_graph_local_schur_gap_decomposition_supports_translation_stage():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    rotations = {pose_id: np.eye(2, dtype=float) for pose_id in pose_ids}

    _, stats = dci.graph_local_interface_schur_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        rotations=rotations,
        edge_owner_policy="lower_robot",
        solver="pcg",
        iterations=20,
        tolerance=1e-12,
    )

    assert stats["stage"] == "translation"
    assert stats["solver"] == "pcg"
    assert stats["schur_energy_gap"] < 1e-20
    assert stats["schur_residual_norm"] < 1e-10
    assert stats["local_private_consistency_error_norm"] < 1e-10


def test_two_stage_graph_local_chordal_gap_decomposition_exact_pcg_has_zero_stage_gaps():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    poses, stats = dci.two_stage_graph_local_chordal_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        solver="pcg",
        rotation_iterations=20,
        translation_iterations=20,
        tolerance=1e-12,
    )

    assert stats["model"] == "two_stage_graph_local_chordal_gap_decomposition"
    assert stats["solver"] == "pcg"
    assert stats["rotation"]["schur_energy_gap"] < 1e-20
    assert stats["translation"]["schur_energy_gap"] < 1e-20
    assert stats["rotation"]["schur_residual_norm"] < 1e-10
    assert stats["translation"]["schur_residual_norm"] < 1e-10
    assert stats["handoff_cost"]["all_edges_evaluated"] is True
    assert stats["handoff_cost"]["total_cost"] < 1e-10
    assert set(poses) == set(pose_ids)


def test_two_stage_graph_local_chordal_gap_decomposition_limited_hop_reports_stage_gap():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, stats = dci.two_stage_graph_local_chordal_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        solver="limited_hop_pcg",
        robot_topology_edges=[(0, 1), (1, 2)],
        hop_radius=0,
        rotation_iterations=20,
        translation_iterations=20,
        tolerance=1e-12,
    )

    assert stats["solver"] == "limited_hop_pcg"
    assert stats["rotation"]["solver_stats"]["coverage_fraction"] < 1.0
    assert stats["rotation"]["schur_energy_gap"] > 0.0
    assert stats["total_schur_energy_gap"] >= stats["rotation"]["schur_energy_gap"]


def test_two_stage_graph_local_chordal_gap_decomposition_uses_budgeted_fixed_step():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, stats = dci.two_stage_graph_local_chordal_gap_decomposition(
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
        solver="fixed_step",
        rotation_iterations=3,
        translation_iterations=3,
        tolerance=1e-12,
        schur_preconditioner="diagonal",
        fixed_step_acceleration=(
            "chebyshev_block_gershgorin_budgeted_certificate"),
        chebyshev_certificate_iterations=8,
        chebyshev_certificate_max_payload_mb=1.0,
    )

    assert stats["solver"] == "fixed_step"
    for stage in ["rotation", "translation"]:
        solver_stats = stats[stage]["solver_stats"]
        assert solver_stats["model"] == "local_interface_schur_fixed_step"
        assert solver_stats["fixed_step_acceleration"] == (
            "chebyshev_block_gershgorin_budgeted_certificate"
        )
        assert solver_stats["chebyshev_certificate_policy_decision"] in {
            "run_budget_ok",
            "run_postcheck_budget_ok",
            "run_unbounded_budget",
        }
        assert solver_stats["chebyshev_block_certificate_payload_bytes"] > 0


def test_rotation_projection_safety_certificate_reports_raw_so_gap():
    solution = np.asarray([1.2, 0.0, 0.0, 0.8], dtype=float)
    pose_ids = [0, 1]
    offsets = {1: 0}

    cert = dci.rotation_projection_safety_certificate(
        solution=solution,
        pose_ids=pose_ids,
        offsets=offsets,
        dim=2,
        anchor_pose=0,
    )

    assert cert["model"] == "rotation_projection_safety_certificate"
    assert cert["evaluated_pose_count"] == 1
    assert cert["max_projection_correction_norm"] > 0.0
    assert cert["mean_projection_correction_norm"] > 0.0
    assert cert["max_orthogonality_error"] > 0.0
    assert cert["max_determinant_deviation"] > 0.0


def test_rotation_projection_cost_certificate_reports_measurement_cost_delta():
    edges = [
        Edge(0, 1, pose2(0.0, 0.0, 0.0), None, 2),
    ]
    solution = np.asarray([1.2, 0.0, 0.0, 0.8], dtype=float)
    pose_ids = [0, 1]
    offsets = {1: 0}
    translations = {
        0: np.zeros(2),
        1: np.zeros(2),
    }

    cert = dci.rotation_projection_cost_certificate(
        graph_edges=edges,
        solution=solution,
        pose_ids=pose_ids,
        offsets=offsets,
        translations=translations,
        dim=2,
        anchor_pose=0,
        weighted=False,
        cost_mode="dpgo",
    )

    assert cert["model"] == "rotation_projection_cost_certificate"
    assert cert["raw_rotation_cost"]["all_edges_evaluated"] is True
    assert cert["projected_rotation_cost"]["all_edges_evaluated"] is True
    assert cert["absolute_projection_cost_delta"] > 0.0
    assert cert["raw_rotation_cost"]["total_cost"] > (
        cert["projected_rotation_cost"]["total_cost"])


def test_two_stage_dci_certificate_sweep_row_contains_stage_gaps():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    row = dci.two_stage_dci_certificate_sweep_row(
        dataset="tiny-chain",
        graph_edges=edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
        solver="pcg",
        rotation_iterations=20,
        translation_iterations=20,
        tolerance=1e-12,
    )

    assert row["dataset"] == "tiny-chain"
    assert row["solver"] == "pcg"
    assert row["pose_count"] == 3
    assert row["dimension"] == 2
    assert row["rotation_schur_energy_gap"] < 1e-20
    assert row["translation_schur_energy_gap"] < 1e-20
    assert row["total_schur_energy_gap"] < 1e-20
    assert row["handoff_cost"] < 1e-10
    assert row["all_edges_evaluated"] is True


def test_write_two_stage_dci_certificate_sweep_outputs_csv_and_json(tmp_path):
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    output_dir = tmp_path / "sweep"

    report = dci.write_two_stage_dci_certificate_sweep(
        output_dir=output_dir,
        dataset_specs=[
            {
                "dataset": "tiny-chain",
                "graph_edges": edges,
                "pose_ids": pose_ids,
                "robot_of": robot_of,
            }
        ],
        weighted=False,
        cost_mode="dpgo",
        solver="pcg",
        rotation_iterations=20,
        translation_iterations=20,
        tolerance=1e-12,
    )

    summary_csv = output_dir / "two_stage_dci_certificate_summary.csv"
    report_json = output_dir / "two_stage_dci_certificate_report.json"
    assert summary_csv.exists()
    assert report_json.exists()
    assert report["rows"][0]["dataset"] == "tiny-chain"
    csv_text = summary_csv.read_text(encoding="utf-8")
    assert "rotation_schur_energy_gap" in csv_text
    json_text = report_json.read_text(encoding="utf-8")
    assert "\"two_stage_dci_certificate_sweep\"" in json_text


def test_two_stage_dci_certificate_sweep_cli_writes_report(tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "pcg",
        "--rotation-iterations", "20",
        "--translation-iterations", "20",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    assert (output_dir / "two_stage_dci_certificate_summary.csv").exists()
    assert (output_dir / "two_stage_dci_certificate_report.json").exists()
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    assert "tiny" in summary
    assert "rotation_schur_energy_gap" in summary


def test_two_stage_dci_certificate_sweep_cli_accepts_budgeted_fixed_step(tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "fixed_step",
        "--rotation-iterations", "3",
        "--translation-iterations", "3",
        "--tolerance", "1e-12",
        "--schur-preconditioner", "diagonal",
        "--fixed-step-acceleration",
        "chebyshev_block_gershgorin_budgeted_certificate",
        "--chebyshev-certificate-iterations", "8",
        "--chebyshev-certificate-max-payload-mb", "1.0",
    ])

    assert exit_code == 0
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    report = (output_dir / "two_stage_dci_certificate_report.json").read_text(
        encoding="utf-8")
    assert "fixed_step" in summary
    assert "rotation_solver_model" in summary
    assert "local_interface_schur_fixed_step" in summary
    assert "chebyshev_block_gershgorin_budgeted_certificate" in report


def test_two_stage_dci_certificate_sweep_cli_reference_gates_budgeted_fixed_step(
        tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    reference = tmp_path / "reference_summary.csv"
    reference.write_text(
        "\n".join([
            "dataset,reference_handoff_cost",
            "tiny,1000.0",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "fixed_step",
        "--rotation-iterations", "3",
        "--translation-iterations", "3",
        "--tolerance", "1e-12",
        "--schur-preconditioner", "diagonal",
        "--fixed-step-acceleration",
        "chebyshev_block_gershgorin_budgeted_certificate",
        "--chebyshev-certificate-iterations", "8",
        "--chebyshev-certificate-max-payload-mb", "1.0",
        "--reference-summary", str(reference),
        "--handoff-reference-relative-slack", "0.0",
        "--projection-cost-reference-relative-slack", "1.0",
    ])

    assert exit_code == 0
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    report = (output_dir / "two_stage_dci_certificate_report.json").read_text(
        encoding="utf-8")
    assert "reference_handoff_cost" in summary
    assert "reference_gate_feasible" in summary
    assert "tiny" in summary
    assert "True" in summary
    assert "\"reference_gate_source\": \"reference_summary_relative\"" in report
    assert "\"reference_handoff_cost\": 1000.0" in report


def test_two_stage_dci_certificate_sweep_cli_can_skip_large_dense_graph(tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "pcg",
        "--max-dense-variables", "1",
    ])

    assert exit_code == 0
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    assert "skipped_preflight" in summary
    assert "estimated_dense_local_normal_bytes" in summary
    report = (output_dir / "two_stage_dci_certificate_report.json").read_text(
        encoding="utf-8")
    assert "\"skipped_preflight\"" in report


def test_two_stage_dci_certificate_sweep_cli_accepts_sparse_hessian_storage(tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "pcg",
        "--hessian-storage", "sparse",
        "--rotation-iterations", "20",
        "--translation-iterations", "20",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    assert "hessian_storage" in summary
    assert "sparse" in summary
    report = (output_dir / "two_stage_dci_certificate_report.json").read_text(
        encoding="utf-8")
    assert "\"materializes_dense_local_hessian\": false" in report
    assert "\"uses_dense_schur_for_certificate\": true" in report


def test_two_stage_dci_certificate_sweep_cli_accepts_matrix_free_certificate(tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "pcg",
        "--hessian-storage", "sparse",
        "--certificate-mode", "matrix_free_dual",
        "--rotation-iterations", "20",
        "--translation-iterations", "20",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    assert "certificate_mode" in summary
    assert "matrix_free_dual" in summary
    report = (output_dir / "two_stage_dci_certificate_report.json").read_text(
        encoding="utf-8")
    assert "\"uses_dense_schur_for_certificate\": false" in report
    assert "\"diagnostic_model\": \"matrix_free_local_schur_dual_energy_certificate\"" in report


def test_matrix_free_sparse_cli_does_not_use_dense_variable_skip_gate(tmp_path):
    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = dci_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "pcg",
        "--hessian-storage", "sparse",
        "--certificate-mode", "matrix_free_dual",
        "--max-dense-variables", "1",
        "--rotation-iterations", "20",
        "--translation-iterations", "20",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    summary = (output_dir / "two_stage_dci_certificate_summary.csv").read_text(
        encoding="utf-8")
    assert "skipped_preflight" not in summary
    assert "matrix_free_dual" in summary
    report = (output_dir / "two_stage_dci_certificate_report.json").read_text(
        encoding="utf-8")
    assert "\"dense_preflight_active\": false" in report
    assert "\"skipped_count\": 0" in report


def test_cci_reference_summary_cli_writes_chordal_handoff_reference(tmp_path):
    from scripts import write_cci_reference_summary as cci_ref

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "cci_reference"

    exit_code = cci_ref.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--cost-mode", "dpgo",
    ])

    assert exit_code == 0
    summary_path = output_dir / "cci_reference_summary.csv"
    report_path = output_dir / "cci_reference_report.json"
    assert summary_path.exists()
    assert report_path.exists()
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "tiny"
    assert rows[0]["reference_method"] == "centralized_chordal_initialization"
    assert rows[0]["cost_mode"] == "dpgo"
    assert rows[0]["all_edges_evaluated"] == "True"
    assert float(rows[0]["reference_handoff_cost"]) < 1e-10
    report = report_path.read_text(encoding="utf-8")
    assert "\"model\": \"cci_reference_summary\"" in report
    assert "\"row_count\": 1" in report


def test_reference_gate_experiment_runner_writes_selected_basin_summary(
        tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment"

    exit_code = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--rotation-budget", "20",
        "--translation-budget", "1",
        "--translation-budget", "20",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    assert (output_dir / "cci_reference" / "cci_reference_summary.csv").exists()
    assert (output_dir / "stage_budget" / "stage_budget_summary.csv").exists()
    assert (
        output_dir / "certificate_stop" / "stage_certificate_stop_report.json"
    ).exists()
    summary_path = output_dir / "reference_gate_experiment_summary.csv"
    report_path = output_dir / "reference_gate_experiment_report.json"
    assert summary_path.exists()
    assert report_path.exists()
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "tiny"
    assert rows[0]["selected_pair"] == "1:20"
    assert rows[0]["feasible"] == "True"
    assert float(rows[0]["selected_handoff_cost"]) < 1e-10
    assert float(rows[0]["selected_handoff_gap_to_reference"]) < 1e-10
    report = report_path.read_text(encoding="utf-8")
    assert "\"model\": \"dci_reference_gate_experiment\"" in report
    assert "\"selected_pair\": \"1:20\"" in report


def test_reference_gate_experiment_runner_writes_selected_initialization(
        tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment_init"

    exit_code = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--translation-budget", "20",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--write-selected-initialization",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    init_path = output_dir / "selected_initializations" / "tiny_init.txt"
    g2o_init_path = (
        output_dir / "selected_initializations" / "tiny_init.g2o"
    )
    summary_path = (
        output_dir / "selected_initializations" /
        "selected_initialization_summary.csv"
    )
    assert init_path.exists()
    assert g2o_init_path.exists()
    dense = np.loadtxt(init_path)
    assert dense.shape == (2, 9)
    _, graph_edges = parse_g2o_graph(graph)
    g2o_poses, fmt = load_pose_set(g2o_init_path, "g2o")
    assert fmt == "g2o"
    cost_report = compute_chordal_cost(
        graph_edges, g2o_poses, weighted=False, cost_mode="dpgo")
    assert cost_report["chordal_cost"] < 1e-10
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "tiny"
    assert rows[0]["selected_pair"] == "1:20"
    assert rows[0]["pose_path"] == str(init_path)
    assert rows[0]["g2o_pose_path"] == str(g2o_init_path)
    assert float(rows[0]["handoff_cost"]) < 1e-10
    report = json.loads((
        output_dir / "reference_gate_experiment_report.json"
    ).read_text(encoding="utf-8"))
    assert report["write_selected_initialization"] is True
    assert report["selected_initialization_summary_path"] == str(summary_path)


def test_reference_gate_experiment_runner_accepts_schur_preconditioner(
        tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment_preconditioned"

    exit_code = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "20",
        "--translation-budget", "20",
        "--schur-preconditioner", "diagonal",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(
        (output_dir / "reference_gate_experiment_summary.csv").open(
            encoding="utf-8")))
    assert rows[0]["selected_pair"] == "20:20"
    assert rows[0]["selected_rotation_preconditioner"] == "diagonal"
    assert rows[0]["selected_translation_preconditioner"] == "diagonal"


def test_reference_gate_experiment_runner_accepts_budgeted_fixed_step(
        tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 0 2 2.2 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment_fixed_step"

    exit_code = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--solver", "fixed_step",
        "--rotation-budget", "3",
        "--translation-budget", "3",
        "--schur-preconditioner", "diagonal",
        "--fixed-step-acceleration",
        "chebyshev_block_gershgorin_budgeted_certificate",
        "--chebyshev-certificate-iterations", "8",
        "--chebyshev-certificate-max-payload-mb", "1.0",
        "--handoff-reference-relative-slack", "100.0",
        "--projection-cost-reference-relative-slack", "100.0",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    stage_summary = (
        output_dir / "stage_budget" / "stage_budget_summary.csv"
    ).read_text(encoding="utf-8")
    assert "local_interface_schur_fixed_step" in stage_summary
    assert "chebyshev_block_gershgorin_budgeted_certificate" in stage_summary
    assert "run_budget_ok" in stage_summary
    report = (output_dir / "reference_gate_experiment_report.json").read_text(
        encoding="utf-8")
    assert "\"solver\": \"fixed_step\"" in report
    assert "\"fixed_step_acceleration\": " in report
    assert "\"chebyshev_block_gershgorin_budgeted_certificate\"" in report
    rows = list(csv.DictReader(
        (output_dir / "reference_gate_experiment_summary.csv").open(
            encoding="utf-8")))
    assert rows[0]["selected_pair"] == "3:3"
    assert rows[0]["selected_rotation_solver_model"] == (
        "local_interface_schur_fixed_step")
    assert rows[0]["selected_translation_solver_model"] == (
        "local_interface_schur_fixed_step")
    assert rows[0]["selected_rotation_fixed_step_acceleration"] == (
        "chebyshev_block_gershgorin_budgeted_certificate")
    assert rows[0]["selected_translation_fixed_step_acceleration"] == (
        "chebyshev_block_gershgorin_budgeted_certificate")
    assert rows[0]["selected_rotation_certificate_policy_decision"] == (
        "run_budget_ok")
    assert rows[0]["selected_translation_certificate_policy_decision"] == (
        "run_budget_ok")


def test_reference_gate_experiment_runner_checkpoints_and_resumes(tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment_checkpointed"

    first_exit = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--translation-budget", "3",
        "--translation-budget", "20",
        "--checkpoint-rows",
        "--tolerance", "1e-12",
    ])

    assert first_exit == 0
    checkpoint_path = output_dir / "stage_budget" / "stage_budget_rows.jsonl"
    assert checkpoint_path.exists()
    checkpoint_lines = checkpoint_path.read_text(
        encoding="utf-8").strip().splitlines()
    assert len(checkpoint_lines) == 2

    second_exit = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--translation-budget", "3",
        "--translation-budget", "20",
        "--checkpoint-rows",
        "--resume",
        "--tolerance", "1e-12",
    ])

    assert second_exit == 0
    resumed_lines = checkpoint_path.read_text(
        encoding="utf-8").strip().splitlines()
    assert len(resumed_lines) == 2
    stage_rows = list(csv.DictReader(
        (output_dir / "stage_budget" / "stage_budget_summary.csv").open(
            encoding="utf-8")))
    assert [row["stage_budget_pair"] for row in stage_rows] == ["1:3", "1:20"]
    report = json.loads((
        output_dir / "reference_gate_experiment_report.json"
    ).read_text(encoding="utf-8"))
    assert report["checkpoint_rows"] is True
    assert report["resume"] is True
    assert report["stage_resumed_row_count"] == 2
    assert report["stage_computed_row_count"] == 0


def test_reference_gate_experiment_runner_adaptive_stops_after_feasible_row(
        tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment_adaptive"

    exit_code = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--translation-budget", "1",
        "--translation-budget", "20",
        "--translation-budget", "40",
        "--adaptive-reference-gate",
        "--checkpoint-rows",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    stage_rows = list(csv.DictReader(
        (output_dir / "stage_budget" / "stage_budget_summary.csv").open(
            encoding="utf-8")))
    assert [row["stage_budget_pair"] for row in stage_rows] == ["1:1", "1:20"]
    summary_rows = list(csv.DictReader(
        (output_dir / "reference_gate_experiment_summary.csv").open(
            encoding="utf-8")))
    assert summary_rows[0]["selected_pair"] == "1:20"
    report = json.loads((
        output_dir / "reference_gate_experiment_report.json"
    ).read_text(encoding="utf-8"))
    assert report["adaptive_reference_gate"] is True
    assert report["adaptive_stopped_dataset_count"] == 1
    assert report["stage_computed_row_count"] == 2


def test_reference_gate_experiment_runner_proxy_gate_stops_without_reference(
        tmp_path):
    from scripts import run_dci_reference_gate_experiment as refgate

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_gate_experiment_proxy"

    exit_code = refgate.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--translation-budget", "1",
        "--translation-budget", "20",
        "--translation-budget", "40",
        "--adaptive-proxy-gate",
        "--proxy-max-rotation-residual", "1.0",
        "--proxy-max-translation-residual", "1e-12",
        "--proxy-max-rotation-projection-correction", "2.0",
        "--proxy-max-rotation-projection-cost-increase", "1e-12",
        "--checkpoint-rows",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    stage_rows = list(csv.DictReader(
        (output_dir / "stage_budget" / "stage_budget_summary.csv").open(
            encoding="utf-8")))
    assert [row["stage_budget_pair"] for row in stage_rows] == ["1:1", "1:20"]
    report = json.loads((
        output_dir / "reference_gate_experiment_report.json"
    ).read_text(encoding="utf-8"))
    assert report["adaptive_proxy_gate"] is True
    assert report["adaptive_reference_gate"] is False
    assert report["adaptive_stopped_dataset_count"] == 1
    assert report["stage_computed_row_count"] == 2
    assert report["proxy_gate_thresholds"][
        "max_translation_residual"] == 1e-12


def test_reference_gate_proxy_gate_supports_schur_energy_thresholds():
    from scripts import run_dci_reference_gate_experiment as refgate

    args = SimpleNamespace(
        proxy_max_rotation_residual=None,
        proxy_max_translation_residual=None,
        proxy_max_rotation_schur_energy_gap=0.2,
        proxy_max_translation_schur_energy_gap=5.0,
        proxy_max_rotation_projection_correction=1.0,
        proxy_max_rotation_projection_cost_increase=0.0,
    )
    safe_row = {
        "rotation_schur_residual_norm": "100.0",
        "translation_schur_residual_norm": "40.0",
        "rotation_schur_energy_gap": "0.1",
        "translation_schur_energy_gap": "5.0",
        "rotation_max_projection_correction_norm": "0.5",
        "rotation_projection_cost_delta": "0.0",
        "rotation_abs_projection_cost_delta": "0.0",
    }
    high_energy_row = dict(safe_row)
    high_energy_row["translation_schur_energy_gap"] = "6.0"

    assert refgate._row_passes_proxy_gate(safe_row, args)
    assert not refgate._row_passes_proxy_gate(high_energy_row, args)
    thresholds = refgate._proxy_gate_thresholds(args)
    assert thresholds["max_rotation_schur_energy_gap"] == 0.2
    assert thresholds["max_translation_schur_energy_gap"] == 5.0


def test_reference_gate_proxy_gate_supports_schur_energy_ratio_thresholds():
    from scripts import run_dci_reference_gate_experiment as refgate

    args = SimpleNamespace(
        proxy_max_rotation_residual=None,
        proxy_max_translation_residual=None,
        proxy_max_rotation_schur_energy_gap=None,
        proxy_max_translation_schur_energy_gap=None,
        proxy_max_rotation_schur_energy_ratio=0.1,
        proxy_max_translation_schur_energy_ratio=0.01,
        proxy_max_rotation_projection_correction=1.0,
        proxy_max_rotation_projection_cost_increase=0.0,
    )
    safe_row = {
        "handoff_cost": "1000.0",
        "rotation_schur_residual_norm": "100.0",
        "translation_schur_residual_norm": "40.0",
        "rotation_schur_energy_gap": "50.0",
        "translation_schur_energy_gap": "5.0",
        "rotation_max_projection_correction_norm": "0.5",
        "rotation_projection_cost_delta": "0.0",
        "rotation_abs_projection_cost_delta": "0.0",
    }
    high_ratio_row = dict(safe_row)
    high_ratio_row["handoff_cost"] = "10.0"

    assert refgate._row_passes_proxy_gate(safe_row, args)
    assert not refgate._row_passes_proxy_gate(high_ratio_row, args)
    thresholds = refgate._proxy_gate_thresholds(args)
    assert thresholds["max_rotation_schur_energy_ratio"] == 0.1
    assert thresholds["max_translation_schur_energy_ratio"] == 0.01


def test_proxy_gate_calibration_selects_safe_thresholds(tmp_path):
    from scripts import analyze_dci_proxy_gate_calibration as calibration

    stage_path = tmp_path / "stage_budget_summary.csv"
    reference_path = tmp_path / "cci_reference_summary.csv"
    output_dir = tmp_path / "proxy_calibration"
    fieldnames = [
        "dataset",
        "stage_budget_pair",
        "handoff_cost",
        "rotation_schur_residual_norm",
        "translation_schur_residual_norm",
        "rotation_max_projection_correction_norm",
        "rotation_projection_cost_delta",
        "rotation_abs_projection_cost_delta",
    ]
    with stage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                "dataset": "tiny",
                "stage_budget_pair": "1:1",
                "handoff_cost": "20.0",
                "rotation_schur_residual_norm": "0.1",
                "translation_schur_residual_norm": "0.5",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "tiny",
                "stage_budget_pair": "1:20",
                "handoff_cost": "10.5",
                "rotation_schur_residual_norm": "0.1",
                "translation_schur_residual_norm": "0.01",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "tiny",
                "stage_budget_pair": "2:20",
                "handoff_cost": "10.2",
                "rotation_schur_residual_norm": "0.2",
                "translation_schur_residual_norm": "0.02",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
        ])
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "reference_handoff_cost"])
        writer.writeheader()
        writer.writerow({
            "dataset": "tiny",
            "reference_handoff_cost": "10.0",
        })

    exit_code = calibration.main([
        "--stage-summary", str(stage_path),
        "--reference-summary", str(reference_path),
        "--output-dir", str(output_dir),
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--max-false-positive", "0",
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(
        (output_dir / "proxy_gate_calibration_summary.csv").open(
            encoding="utf-8")))
    selected = [row for row in rows if row["selected"] == "True"]
    assert len(selected) == 1
    assert selected[0]["source_pair"] == "2:20"
    assert selected[0]["true_positive"] == "2"
    assert selected[0]["false_positive"] == "0"
    assert float(selected[0]["max_translation_residual"]) == 0.02
    report = json.loads((
        output_dir / "proxy_gate_calibration_report.json"
    ).read_text(encoding="utf-8"))
    assert report["selected_candidate"]["source_pair"] == "2:20"
    assert report["reference_feasible_count"] == 2


def test_proxy_gate_calibration_searches_mixed_metric_thresholds(tmp_path):
    from scripts import analyze_dci_proxy_gate_calibration as calibration

    stage_path = tmp_path / "stage_budget_summary.csv"
    reference_path = tmp_path / "cci_reference_summary.csv"
    output_dir = tmp_path / "proxy_calibration_mixed"
    fieldnames = [
        "dataset",
        "stage_budget_pair",
        "handoff_cost",
        "rotation_schur_residual_norm",
        "translation_schur_residual_norm",
        "rotation_max_projection_correction_norm",
        "rotation_projection_cost_delta",
        "rotation_abs_projection_cost_delta",
    ]
    with stage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                "dataset": "tiny",
                "stage_budget_pair": "bad",
                "handoff_cost": "20.0",
                "rotation_schur_residual_norm": "1.5",
                "translation_schur_residual_norm": "1.5",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "tiny",
                "stage_budget_pair": "rot-hard",
                "handoff_cost": "10.5",
                "rotation_schur_residual_norm": "1.0",
                "translation_schur_residual_norm": "0.1",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "tiny",
                "stage_budget_pair": "trans-hard",
                "handoff_cost": "10.4",
                "rotation_schur_residual_norm": "0.1",
                "translation_schur_residual_norm": "1.0",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
        ])
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "reference_handoff_cost"])
        writer.writeheader()
        writer.writerow({
            "dataset": "tiny",
            "reference_handoff_cost": "10.0",
        })

    exit_code = calibration.main([
        "--stage-summary", str(stage_path),
        "--reference-summary", str(reference_path),
        "--output-dir", str(output_dir),
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--max-false-positive", "0",
    ])

    assert exit_code == 0
    report = json.loads((
        output_dir / "proxy_gate_calibration_report.json"
    ).read_text(encoding="utf-8"))
    selected = report["selected_candidate"]
    assert selected["source_pair"] == "metric_grid"
    assert selected["true_positive"] == 2
    assert selected["false_positive"] == 0
    assert selected["max_rotation_residual"] == 1.0
    assert selected["max_translation_residual"] == 1.0


def test_proxy_gate_calibration_schur_energy_profile_accepts_high_raw_residual(
    tmp_path,
):
    from scripts import analyze_dci_proxy_gate_calibration as calibration

    stage_path = tmp_path / "stage_budget_summary.csv"
    reference_path = tmp_path / "cci_reference_summary.csv"
    output_dir = tmp_path / "proxy_calibration_energy"
    fieldnames = [
        "dataset",
        "stage_budget_pair",
        "handoff_cost",
        "rotation_schur_residual_norm",
        "translation_schur_residual_norm",
        "rotation_schur_energy_gap",
        "translation_schur_energy_gap",
        "rotation_max_projection_correction_norm",
        "rotation_projection_cost_delta",
        "rotation_abs_projection_cost_delta",
    ]
    with stage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                "dataset": "tiny",
                "stage_budget_pair": "bad",
                "handoff_cost": "20.0",
                "rotation_schur_residual_norm": "0.1",
                "translation_schur_residual_norm": "50.0",
                "rotation_schur_energy_gap": "0.1",
                "translation_schur_energy_gap": "50.0",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "tiny",
                "stage_budget_pair": "high-raw-safe-energy",
                "handoff_cost": "10.5",
                "rotation_schur_residual_norm": "0.1",
                "translation_schur_residual_norm": "40.0",
                "rotation_schur_energy_gap": "0.1",
                "translation_schur_energy_gap": "5.0",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "tiny",
                "stage_budget_pair": "low-raw-safe-energy",
                "handoff_cost": "10.4",
                "rotation_schur_residual_norm": "0.1",
                "translation_schur_residual_norm": "1.0",
                "rotation_schur_energy_gap": "0.1",
                "translation_schur_energy_gap": "1.0",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
        ])
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "reference_handoff_cost"])
        writer.writeheader()
        writer.writerow({
            "dataset": "tiny",
            "reference_handoff_cost": "10.0",
        })

    exit_code = calibration.main([
        "--stage-summary", str(stage_path),
        "--reference-summary", str(reference_path),
        "--output-dir", str(output_dir),
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--max-false-positive", "0",
        "--proxy-metric-profile", "schur_energy",
    ])

    assert exit_code == 0
    report = json.loads((
        output_dir / "proxy_gate_calibration_report.json"
    ).read_text(encoding="utf-8"))
    selected = report["selected_candidate"]
    assert report["proxy_metric_profile"] == "schur_energy"
    assert selected["true_positive"] == 2
    assert selected["false_positive"] == 0
    assert selected["max_translation_schur_energy_gap"] == 5.0
    assert "max_translation_residual" not in selected


def test_proxy_gate_calibration_schur_energy_ratio_profile_normalizes_by_cost(
    tmp_path,
):
    from scripts import analyze_dci_proxy_gate_calibration as calibration

    stage_path = tmp_path / "stage_budget_summary.csv"
    reference_path = tmp_path / "cci_reference_summary.csv"
    output_dir = tmp_path / "proxy_calibration_energy_ratio"
    fieldnames = [
        "dataset",
        "stage_budget_pair",
        "handoff_cost",
        "rotation_schur_energy_gap",
        "translation_schur_energy_gap",
        "rotation_max_projection_correction_norm",
        "rotation_projection_cost_delta",
        "rotation_abs_projection_cost_delta",
    ]
    with stage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                "dataset": "large",
                "stage_budget_pair": "safe-large-cost",
                "handoff_cost": "1000.0",
                "rotation_schur_energy_gap": "10.0",
                "translation_schur_energy_gap": "5.0",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
            {
                "dataset": "small",
                "stage_budget_pair": "bad-small-cost",
                "handoff_cost": "20.0",
                "rotation_schur_energy_gap": "1.0",
                "translation_schur_energy_gap": "1.0",
                "rotation_max_projection_correction_norm": "0.5",
                "rotation_projection_cost_delta": "0.0",
                "rotation_abs_projection_cost_delta": "0.0",
            },
        ])
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "reference_handoff_cost"])
        writer.writeheader()
        writer.writerows([
            {"dataset": "large", "reference_handoff_cost": "1000.0"},
            {"dataset": "small", "reference_handoff_cost": "10.0"},
        ])

    exit_code = calibration.main([
        "--stage-summary", str(stage_path),
        "--reference-summary", str(reference_path),
        "--output-dir", str(output_dir),
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
        "--max-false-positive", "0",
        "--proxy-metric-profile", "schur_energy_ratio",
    ])

    assert exit_code == 0
    report = json.loads((
        output_dir / "proxy_gate_calibration_report.json"
    ).read_text(encoding="utf-8"))
    selected = report["selected_candidate"]
    assert report["proxy_metric_profile"] == "schur_energy_ratio"
    assert selected["true_positive"] == 1
    assert selected["false_positive"] == 0
    assert selected["max_rotation_schur_energy_ratio"] == 0.01
    assert selected["max_translation_schur_energy_ratio"] == 0.005
    assert "max_translation_schur_energy_gap" not in selected


def test_matrix_free_convergence_sweep_writes_budget_rows(tmp_path):
    from scripts import run_dci_matrix_free_convergence_sweep as convergence

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "convergence"

    exit_code = convergence.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--iteration-budget", "1",
        "--iteration-budget", "3",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    summary_path = output_dir / "matrix_free_convergence_summary.csv"
    report_path = output_dir / "matrix_free_convergence_report.json"
    assert summary_path.exists()
    assert report_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "iteration_budget" in summary
    assert "dense_preflight_active" in summary
    assert "matrix_free_dual" in summary
    assert summary.count("tiny,") == 2
    report = report_path.read_text(encoding="utf-8")
    assert "\"model\": \"matrix_free_dci_convergence_sweep\"" in report
    assert "\"row_count\": 2" in report


def test_matrix_free_convergence_analysis_flags_stage_and_nonmonotone_gap(tmp_path):
    from scripts import analyze_dci_matrix_free_convergence as analysis

    summary = tmp_path / "matrix_free_convergence_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,iteration_budget,rotation_schur_energy_gap,translation_schur_energy_gap,total_schur_energy_gap,handoff_cost,wall_time_sec",
            "tiny,1,1.0,4.0,5.0,100.0,0.1",
            "tiny,3,2.0,5.0,7.0,80.0,0.2",
            "tiny,5,0.5,1.0,1.5,60.0,0.3",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"

    exit_code = analysis.main([
        "--summary", str(summary),
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 0
    report_json = output_dir / "matrix_free_convergence_analysis.json"
    report_md = output_dir / "matrix_free_convergence_analysis.md"
    assert report_json.exists()
    assert report_md.exists()
    json_text = report_json.read_text(encoding="utf-8")
    assert "\"model\": \"matrix_free_convergence_analysis\"" in json_text
    assert "\"total_gap_monotone_nonincreasing\": false" in json_text
    assert "\"handoff_cost_monotone_nonincreasing\": true" in json_text
    assert "\"dominant_stage\": \"translation\"" in json_text
    md_text = report_md.read_text(encoding="utf-8")
    assert "nonmonotone_total_gap" in md_text
    assert "translation" in md_text


def test_stage_budget_sweep_writes_independent_stage_budget_rows(tmp_path):
    from scripts import run_dci_stage_budget_sweep as stage_sweep

    graph = tmp_path / "tiny_chain.g2o"
    graph.write_text(
        "\n".join([
            "VERTEX_SE2 0 0 0 0",
            "VERTEX_SE2 1 1 0 0",
            "VERTEX_SE2 2 2 0 0",
            "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
            "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage_budget"

    exit_code = stage_sweep.main([
        "--dataset", f"tiny={graph}",
        "--num-robots", "3",
        "--output-dir", str(output_dir),
        "--rotation-budget", "1",
        "--rotation-budget", "3",
        "--translation-budget", "1",
        "--translation-budget", "2",
        "--tolerance", "1e-12",
    ])

    assert exit_code == 0
    summary_path = output_dir / "stage_budget_summary.csv"
    report_path = output_dir / "stage_budget_report.json"
    assert summary_path.exists()
    assert report_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "rotation_iteration_budget" in summary
    assert "translation_iteration_budget" in summary
    assert summary.count("tiny,") == 4
    report = report_path.read_text(encoding="utf-8")
    assert "\"model\": \"dci_stage_budget_sweep\"" in report
    assert "\"row_count\": 4" in report
    assert "\"best_handoff_cost_pair\"" in report


def test_stage_budget_pareto_analysis_flags_non_equal_dominating_equal(tmp_path):
    from scripts import analyze_dci_stage_budget_pareto as pareto

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,total_schur_energy_gap,rotation_comm_mb,translation_comm_mb",
            "tiny,1,1,1:1,120.0,10.0,0.1,0.1",
            "tiny,3,3,3:3,100.0,8.0,0.3,0.3",
            "tiny,1,5,1:5,90.0,9.0,0.2,0.2",
            "tiny,5,5,5:5,80.0,7.0,0.5,0.5",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "pareto"

    exit_code = pareto.main([
        "--summary", str(summary),
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 0
    report_json = output_dir / "stage_budget_pareto_report.json"
    report_md = output_dir / "stage_budget_pareto_report.md"
    assert report_json.exists()
    assert report_md.exists()
    json_text = report_json.read_text(encoding="utf-8")
    assert "\"model\": \"dci_stage_budget_pareto_analysis\"" in json_text
    assert "\"non_equal_dominates_equal\": true" in json_text
    assert "\"dominated_pair\": \"3:3\"" in json_text
    assert "\"dominating_pair\": \"1:5\"" in json_text
    md_text = report_md.read_text(encoding="utf-8")
    assert "1:5 dominates equal-budget 3:3" in md_text


def test_stage_certificate_stop_selects_min_comm_feasible_pair(tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_comm_mb,translation_comm_mb",
            "tiny,1,3,1:3,100.0,1.0,9.0,0.2,0.1",
            "tiny,1,5,1:5,90.0,1.0,5.0,0.2,0.2",
            "tiny,3,5,3:5,85.0,0.5,4.0,0.3,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "certificate_stop"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--output-dir", str(output_dir),
        "--max-rotation-residual", "1.0",
        "--max-translation-residual", "5.0",
        "--max-handoff-cost", "95.0",
    ])

    assert exit_code == 0
    report_json = output_dir / "stage_certificate_stop_report.json"
    report_md = output_dir / "stage_certificate_stop_report.md"
    assert report_json.exists()
    assert report_md.exists()
    json_text = report_json.read_text(encoding="utf-8")
    assert "\"model\": \"dci_stage_certificate_stop_analysis\"" in json_text
    assert "\"selected_pair\": \"1:5\"" in json_text
    assert "\"feasible\": true" in json_text
    assert "\"selection_rule\": \"min_total_comm_then_handoff_cost\"" in json_text
    md_text = report_md.read_text(encoding="utf-8")
    assert "selected_pair: 1:5" in md_text


def test_stage_certificate_stop_can_derive_gates_from_best_observed(tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_comm_mb,translation_comm_mb",
            "tiny,1,3,1:3,100.0,1.0,9.0,0.2,0.1",
            "tiny,1,5,1:5,90.0,1.0,5.0,0.2,0.2",
            "tiny,3,5,3:5,85.0,0.5,4.0,0.3,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "derived_certificate_stop"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--output-dir", str(output_dir),
        "--derive-gates-from", "best_observed",
        "--rotation-residual-slack", "0.5",
        "--translation-residual-slack", "1.0",
        "--handoff-cost-slack", "5.0",
    ])

    assert exit_code == 0
    json_text = (output_dir / "stage_certificate_stop_report.json").read_text(
        encoding="utf-8")
    assert "\"gate_source\": \"best_observed_plus_slack\"" in json_text
    assert "\"selected_pair\": \"1:5\"" in json_text
    assert "\"max_rotation_residual\": 1.0" in json_text
    assert "\"max_translation_residual\": 5.0" in json_text
    assert "\"max_handoff_cost\": 90.0" in json_text


def test_stage_certificate_stop_can_gate_projection_safety(tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_max_projection_correction_norm,rotation_comm_mb,translation_comm_mb",
            "tiny,1,5,1:5,90.0,1.0,5.0,0.50,0.2,0.2",
            "tiny,3,5,3:5,85.0,0.5,4.0,0.05,0.3,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "projection_gate"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--output-dir", str(output_dir),
        "--max-rotation-residual", "1.0",
        "--max-translation-residual", "5.0",
        "--max-handoff-cost", "95.0",
        "--max-rotation-projection-correction", "0.1",
    ])

    assert exit_code == 0
    json_text = (output_dir / "stage_certificate_stop_report.json").read_text(
        encoding="utf-8")
    assert "\"selected_pair\": \"3:5\"" in json_text
    assert "\"rotation_projection\"" in json_text
    md_text = (output_dir / "stage_certificate_stop_report.md").read_text(
        encoding="utf-8")
    assert "rotation_projection" in md_text


def test_stage_certificate_stop_can_gate_projection_cost_delta(tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_abs_projection_cost_delta,rotation_comm_mb,translation_comm_mb",
            "tiny,1,5,1:5,90.0,1.0,5.0,30.0,0.2,0.2",
            "tiny,3,5,3:5,85.0,0.5,4.0,2.0,0.3,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "projection_cost_gate"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--output-dir", str(output_dir),
        "--max-rotation-residual", "1.0",
        "--max-translation-residual", "5.0",
        "--max-handoff-cost", "95.0",
        "--max-rotation-projection-cost-delta", "5.0",
    ])

    assert exit_code == 0
    json_text = (output_dir / "stage_certificate_stop_report.json").read_text(
        encoding="utf-8")
    assert "\"selected_pair\": \"3:5\"" in json_text
    assert "\"rotation_projection_cost\"" in json_text


def test_stage_certificate_stop_projection_gate_uses_positive_cost_increase(
        tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_projection_cost_delta,rotation_abs_projection_cost_delta,rotation_comm_mb,translation_comm_mb",
            "tiny,1,5,1:5,105.0,1.0,5.0,-25.0,25.0,0.2,0.2",
            "tiny,3,5,3:5,106.0,0.5,4.0,8.0,8.0,0.3,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    reference = tmp_path / "reference_summary.csv"
    reference.write_text(
        "\n".join([
            "dataset,reference_handoff_cost",
            "tiny,100.0",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "projection_increase_gate"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--reference-summary", str(reference),
        "--output-dir", str(output_dir),
        "--max-rotation-residual", "1.0",
        "--max-translation-residual", "5.0",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
    ])

    assert exit_code == 0
    json_text = (output_dir / "stage_certificate_stop_report.json").read_text(
        encoding="utf-8")
    assert "\"selected_pair\": \"1:5\"" in json_text
    assert "\"rotation_projection_cost_increase\": 0.0" in json_text


def test_stage_certificate_stop_can_gate_against_reference_summary(tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_abs_projection_cost_delta,rotation_comm_mb,translation_comm_mb",
            "tiny,1,5,1:5,112.0,1.0,5.0,25.0,0.2,0.2",
            "tiny,3,5,3:5,105.0,0.5,4.0,8.0,0.3,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    reference = tmp_path / "reference_summary.csv"
    reference.write_text(
        "\n".join([
            "dataset,reference_handoff_cost",
            "tiny,100.0",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reference_relative_gate"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--reference-summary", str(reference),
        "--output-dir", str(output_dir),
        "--max-rotation-residual", "1.0",
        "--max-translation-residual", "5.0",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
    ])

    assert exit_code == 0
    json_text = (output_dir / "stage_certificate_stop_report.json").read_text(
        encoding="utf-8")
    assert "\"gate_source\": \"reference_summary_relative\"" in json_text
    assert "\"reference_handoff_cost\": 100.0" in json_text
    assert "\"max_handoff_cost\": 110.00000000000001" in json_text
    assert "\"max_rotation_projection_cost_delta\": 10.0" in json_text
    assert "\"selected_pair\": \"3:5\"" in json_text
    assert "\"rotation_projection_cost\"" in json_text
    assert "\"handoff_cost\"" in json_text


def test_stage_certificate_stop_reference_summary_missing_dataset_fails_closed(
        tmp_path):
    from scripts import analyze_dci_stage_certificate_stop as cert_stop

    summary = tmp_path / "stage_budget_summary.csv"
    summary.write_text(
        "\n".join([
            "dataset,rotation_iteration_budget,translation_iteration_budget,stage_budget_pair,handoff_cost,rotation_schur_residual_norm,translation_schur_residual_norm,rotation_abs_projection_cost_delta,rotation_comm_mb,translation_comm_mb",
            "tiny,1,5,1:5,90.0,1.0,5.0,2.0,0.2,0.2",
            "",
        ]),
        encoding="utf-8",
    )
    reference = tmp_path / "reference_summary.csv"
    reference.write_text(
        "\n".join([
            "dataset,reference_handoff_cost",
            "other,100.0",
            "",
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "missing_reference_gate"

    exit_code = cert_stop.main([
        "--summary", str(summary),
        "--reference-summary", str(reference),
        "--output-dir", str(output_dir),
        "--max-rotation-residual", "1.0",
        "--max-translation-residual", "5.0",
        "--handoff-reference-relative-slack", "0.1",
        "--projection-cost-reference-relative-slack", "0.1",
    ])

    assert exit_code == 0
    json_text = (output_dir / "stage_certificate_stop_report.json").read_text(
        encoding="utf-8")
    assert "\"feasible_count\": 0" in json_text
    assert "\"selected_pair\": null" in json_text
    assert "\"missing_reference_handoff_cost\"" in json_text


def test_interface_schur_iterative_central_equivalence_matches_dense_energy():
    hessian = np.asarray([
        [7.0, 0.5, 1.0, 0.2],
        [0.5, 6.0, 0.3, 1.1],
        [1.0, 0.3, 4.0, 0.4],
        [0.2, 1.1, 0.4, 3.0],
    ])
    gradient = np.asarray([1.0, -0.25, 2.0, -1.0])
    interface_indices = np.asarray([2, 3])
    exact_solution = np.linalg.solve(hessian, gradient)
    perturbed_solution = exact_solution.copy()
    perturbed_solution[interface_indices] += np.asarray([0.125, -0.25])

    dense_stats = dci.interface_schur_central_equivalence_diagnostic(
        hessian=hessian,
        gradient=gradient,
        solution=perturbed_solution,
        interface_indices=interface_indices,
        damping=0.0,
    )
    iterative_stats = (
        dci.interface_schur_iterative_central_equivalence_diagnostic(
            hessian=hessian,
            gradient=gradient,
            solution=perturbed_solution,
            interface_indices=interface_indices,
            interface_blocks=[np.asarray([2, 3])],
            iterations=8,
            damping=0.0,
            tolerance=1e-14,
            schur_preconditioner="block_jacobi",
        ))

    assert iterative_stats["diagnostic_model"] == (
        "iterative_schur_dual_energy")
    assert iterative_stats["materializes_dense_schur"] is False
    assert np.isclose(
        iterative_stats["schur_residual_dual_energy_estimate"],
        dense_stats["schur_residual_dual_energy"],
        atol=1e-12,
    )
    assert iterative_stats["dual_solve_iterations"] <= 2


def test_interface_schur_hessian_solve_can_select_ritz_portfolio_mode():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])
    interface_indices = np.asarray([2, 3])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=interface_indices,
        interface_blocks=[np.asarray([2, 3])],
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=1,
        ritz_probe_iterations=2,
        ritz_mode="portfolio",
    )

    assert stats["ritz_mode"] == "portfolio"
    assert stats["ritz_portfolio_selected_mode"] in {
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    }
    assert stats["ritz_portfolio_candidate_count"] == 4
    assert stats["ritz_portfolio_scoring_schur_matvec_count"] >= 4
    assert stats["ritz_deflation_basis_rank"] == 1


def test_interface_schur_hessian_solve_can_threshold_ritz_rank():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=np.asarray([1, 2, 3]),
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=2,
        ritz_probe_iterations=2,
        ritz_mode="harmonic",
        ritz_rank_selection_mode="value_threshold",
        ritz_value_threshold=-1.0,
    )

    assert stats["ritz_deflation_requested_rank"] == 2
    assert stats["ritz_deflation_requested_rank_before_selection"] == 2
    assert stats["ritz_deflation_selected_rank"] == 1
    assert stats["ritz_deflation_basis_rank"] == 1
    assert stats["ritz_rank_selection_mode"] == "value_threshold"
    assert stats["ritz_value_threshold"] == -1.0


def test_interface_schur_hessian_solve_can_energy_capture_ritz_rank():
    hessian = np.asarray([
        [6.0, 0.5, 1.0, 0.1],
        [0.5, 5.5, 0.2, 1.2],
        [1.0, 0.2, 4.0, 0.3],
        [0.1, 1.2, 0.3, 3.5],
    ])
    gradient = np.asarray([1.0, -0.5, 2.0, 0.25])

    _, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=np.asarray([1, 2, 3]),
        iterations=4,
        damping=0.0,
        tolerance=1e-14,
        schur_preconditioner="block_jacobi+coarse",
        ritz_deflation_rank=2,
        ritz_probe_iterations=3,
        ritz_mode="harmonic",
        ritz_rank_selection_mode="energy_capture",
        ritz_energy_capture_fraction=0.8,
    )

    assert stats["ritz_rank_selection_mode"] == "energy_capture"
    assert stats["ritz_energy_capture_fraction"] == 0.8
    assert stats["ritz_deflation_requested_rank"] == 2
    assert 1 <= stats["ritz_deflation_selected_rank"] <= 2
    assert len(stats["ritz_energy_contributions"]) == (
        stats["ritz_deflation_selected_rank"])
    assert stats["ritz_energy_capture_selected"] <= (
        stats["ritz_energy_capture_total"] + 1e-12)


def test_interface_schur_hessian_solve_accepts_overlap_schwarz_preconditioner():
    hessian = np.asarray([
        [8.0, 1.0, 1.2, 0.2],
        [1.0, 7.0, 0.8, 1.4],
        [1.2, 0.8, 5.0, 0.7],
        [0.2, 1.4, 0.7, 4.5],
    ])
    gradient = np.asarray([0.5, -1.0, 2.0, 0.25])

    solution, stats = dci.interface_schur_hessian_solve(
        hessian=hessian,
        gradient=gradient,
        interface_indices=np.asarray([1, 2, 3]),
        iterations=8,
        damping=1e-12,
        tolerance=1e-12,
        schur_preconditioner="overlap_schwarz",
    )

    assert solution.shape == (4,)
    assert stats["schur_preconditioner"] == "overlap_schwarz"
    assert stats["schur_preconditioner_overlap_level"] == 1
    assert stats["schur_preconditioner_block_count"] == 1
    assert stats["overlap_schwarz_setup_schur_matvec_count"] == 0
    assert stats["overlap_schwarz_setup_local_schur_column_count"] > 0
    assert stats["final_normal_residual"] <= 1e-8


def test_interface_schur_hessian_solve_rejects_bad_coarse_basis_dimension():
    hessian = np.eye(4)
    gradient = np.ones(4)

    try:
        dci.interface_schur_hessian_solve(
            hessian=hessian,
            gradient=gradient,
            interface_indices=np.asarray([2, 3]),
            iterations=5,
            schur_preconditioner="block_jacobi+coarse",
            coarse_basis=np.ones((3, 1)),
        )
    except ValueError as exc:
        assert "coarse_basis row count" in str(exc)
    else:
        raise AssertionError("expected coarse basis dimension validation")


def test_summary_chordal_init_can_use_interface_schur_solver():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=30,
        translation_iterations=30,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        summary_selection_mode="all",
    )

    assert stats["rotation_stats"]["method"] == "interface_schur_pcg"
    assert stats["translation_stats"]["method"] == "interface_schur_pcg"
    assert stats["rotation_stats"]["schur_preconditioner"] == "block_jacobi"
    assert stats["translation_stats"]["schur_preconditioner"] == "block_jacobi"
    assert stats["private_interface_residual_split"]["combined"][
        "full_model_residual_norm"
    ] <= 1e-8


def test_summary_chordal_init_can_use_two_level_interface_schur_solver():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=30,
        translation_iterations=30,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        summary_selection_mode="all",
    )

    assert stats["rotation_stats"]["schur_preconditioner"] == (
        "block_jacobi+coarse"
    )
    assert stats["translation_stats"]["schur_preconditioner"] == (
        "block_jacobi+coarse"
    )
    assert stats["rotation_stats"]["coarse_basis_rank"] > 0
    assert stats["translation_stats"]["coarse_basis_rank"] > 0
    assert stats["private_interface_residual_split"]["combined"][
        "full_model_residual_norm"
    ] <= 1e-8


def test_summary_chordal_init_can_use_balanced_coarse_interface_schur_solver():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=30,
        translation_iterations=30,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+balanced_coarse",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert stats["rotation_stats"]["schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert stats["translation_stats"]["schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert stats["rotation_stats"]["coarse_basis_rank"] > 0
    assert stats["translation_stats"]["coarse_basis_rank"] > 0
    assert stats["rotation_stats"][
        "balanced_coarse_preconditioner_schur_matvec_count"] > 0
    assert stats["translation_stats"][
        "balanced_coarse_preconditioner_schur_matvec_count"] > 0
    assert stats["private_interface_residual_split"]["combined"][
        "full_model_residual_norm"
    ] <= 1e-8


def test_summary_chordal_init_can_use_translation_only_balanced_coarse():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=30,
        translation_iterations=30,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi",
        interface_schur_translation_preconditioner=(
            "block_jacobi+balanced_coarse"
        ),
        interface_schur_coarse_basis_mode="component_coordinate",
        interface_schur_coarse_component_limit=1,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_preconditioner"] == "block_jacobi"
    assert stats["interface_schur_rotation_preconditioner"] == "block_jacobi"
    assert stats["interface_schur_translation_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert stats["rotation_stats"]["schur_preconditioner"] == "block_jacobi"
    assert stats["translation_stats"]["schur_preconditioner"] == (
        "block_jacobi+balanced_coarse"
    )
    assert stats["rotation_stats"]["coarse_basis_rank"] == 0
    assert stats["translation_stats"]["coarse_basis_rank"] > 0
    assert stats["translation_stats"][
        "balanced_coarse_preconditioner_schur_matvec_count"] > 0
    assert stats["private_interface_residual_split"]["combined"][
        "full_model_residual_norm"
    ] <= 1e-8


def test_summary_chordal_init_can_split_rotation_translation_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=30,
        translation_iterations=30,
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
        summary_selection_mode="all",
    )

    assert stats["interface_schur_rotation_coarse_basis_mode"] == (
        "component_coordinate"
    )
    assert stats["interface_schur_translation_coarse_basis_mode"] == (
        "robot_coordinate"
    )
    assert stats["interface_schur_rotation_coarse_component_selection_mode"] == (
        "gradient_energy"
    )
    assert stats["rotation_stats"]["coarse_component_selection_mode"] == (
        "gradient_energy"
    )
    assert stats["rotation_stats"]["coarse_basis_mode"] == (
        "component_coordinate"
    )
    assert stats["translation_stats"]["coarse_basis_mode"] == (
        "robot_coordinate"
    )
    assert stats["rotation_stats"]["component_coordinate_basis_rank"] > 0
    assert stats["translation_stats"]["component_coordinate_basis_rank"] == 0
    assert stats["translation_stats"]["robot_coordinate_basis_rank"] > 0


def test_summary_chordal_init_can_use_overlap_schwarz_interface_solver():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=30,
        translation_iterations=30,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="overlap_schwarz",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_preconditioner"] == "overlap_schwarz"
    assert stats["rotation_stats"]["schur_preconditioner"] == (
        "overlap_schwarz")
    assert stats["translation_stats"]["schur_preconditioner"] == (
        "overlap_schwarz")
    assert stats["rotation_stats"][
        "overlap_schwarz_setup_schur_matvec_count"] == 0
    assert stats["rotation_stats"][
        "overlap_schwarz_setup_local_schur_column_count"] > 0
    assert stats["communication_estimate"][
        "overlap_schwarz_setup_interface_vector_mb"] == 0.0


def test_summary_chordal_init_can_use_coarse_interface_initial_guess():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=1,
        translation_iterations=1,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_coarse_initial_guess"]
    assert stats["rotation_stats"]["coarse_initial_guess_used"]
    assert stats["translation_stats"]["coarse_initial_guess_used"]


def test_summary_chordal_init_can_use_robot_component_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_basis_mode="robot_component_coordinate",
        interface_schur_coarse_component_limit=2,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_coarse_basis_mode"] == (
        "robot_component_coordinate"
    )
    assert stats["rotation_stats"]["coarse_basis_mode"] == (
        "robot_component_coordinate"
    )
    assert stats["translation_stats"]["coarse_basis_mode"] == (
        "robot_component_coordinate"
    )
    assert stats["rotation_stats"]["coarse_component_limit"] == 2
    assert stats["rotation_stats"]["coarse_basis_rank"] > 0
    assert stats["translation_stats"]["coarse_basis_rank"] > 0


def test_summary_chordal_init_can_use_residual_deflation_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_residual_deflation_rank=1,
        interface_schur_residual_deflation_pilot_iterations=1,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_residual_deflation_rank"] == 1
    assert stats["interface_schur_residual_deflation_pilot_iterations"] == 1
    assert stats["rotation_stats"]["residual_deflation_basis_rank"] > 0
    assert stats["translation_stats"]["residual_deflation_basis_rank"] > 0
    assert stats["communication_estimate"][
        "residual_deflation_pilot_interface_vector_mb"] > 0.0


def test_summary_chordal_init_can_use_ritz_low_mode_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_rank"] == 1
    assert stats["interface_schur_ritz_probe_iterations"] == 2
    assert stats["rotation_stats"]["ritz_deflation_basis_rank"] > 0
    assert stats["translation_stats"]["ritz_deflation_basis_rank"] > 0
    assert stats["communication_estimate"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_summary_chordal_init_can_use_generalized_ritz_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="generalized",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_mode"] == "generalized"
    assert stats["rotation_stats"]["ritz_mode"] == "generalized"
    assert stats["translation_stats"]["ritz_mode"] == "generalized"
    assert stats["rotation_stats"]["ritz_deflation_basis_rank"] > 0
    assert stats["translation_stats"]["ritz_deflation_basis_rank"] > 0


def test_summary_chordal_init_can_use_m_orthogonal_lanczos_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="m_orthogonal_lanczos",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_mode"] == "m_orthogonal_lanczos"
    assert stats["rotation_stats"]["ritz_mode"] == "m_orthogonal_lanczos"
    assert stats["translation_stats"]["ritz_mode"] == "m_orthogonal_lanczos"
    assert stats["rotation_stats"]["ritz_deflation_basis_rank"] > 0
    assert stats["translation_stats"]["ritz_deflation_basis_rank"] > 0


def test_summary_chordal_init_can_use_harmonic_ritz_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="harmonic",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_mode"] == "harmonic"
    assert stats["rotation_stats"]["ritz_mode"] == "harmonic"
    assert stats["translation_stats"]["ritz_mode"] == "harmonic"
    assert stats["rotation_stats"]["ritz_deflation_basis_rank"] > 0
    assert stats["translation_stats"]["ritz_deflation_basis_rank"] > 0


def test_summary_chordal_init_can_use_low_harmonic_hybrid_ritz_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="low_harmonic_hybrid",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_mode"] == "low_harmonic_hybrid"
    assert stats["rotation_stats"]["ritz_mode"] == "low_harmonic_hybrid"
    assert stats["translation_stats"]["ritz_mode"] == "low_harmonic_hybrid"
    assert stats["rotation_stats"]["ritz_hybrid_candidate_count"] == 2
    assert stats["translation_stats"]["ritz_hybrid_candidate_count"] == 2
    assert stats["communication_estimate"][
        "ritz_probe_interface_vector_mb"] > 0.0


def test_summary_chordal_init_can_split_rotation_translation_ritz_modes():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_mode"] == "rot_low_trans_harmonic"
    assert stats["rotation_stats"]["ritz_mode"] == "preconditioned_operator"
    assert stats["translation_stats"]["ritz_mode"] == "harmonic"

    _, reverse_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="rot_harmonic_trans_low",
        summary_selection_mode="all",
    )

    assert reverse_stats["interface_schur_ritz_mode"] == "rot_harmonic_trans_low"
    assert reverse_stats["rotation_stats"]["ritz_mode"] == "harmonic"
    assert reverse_stats["translation_stats"]["ritz_mode"] == (
        "preconditioned_operator"
    )


def test_summary_chordal_init_can_use_split_ritz_rank_budget():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_translation_ritz_rank=3,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_translation_ritz_probe_iterations=4,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_rank"] == 1
    assert stats["interface_schur_translation_ritz_rank"] == 3
    assert stats["rotation_stats"]["ritz_deflation_requested_rank"] == 1
    assert stats["translation_stats"]["ritz_deflation_requested_rank"] == 3
    assert stats["translation_stats"]["ritz_probe_requested_iterations"] == 4


def test_summary_chordal_init_can_select_translation_ritz_budget_portfolio():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        interface_schur_translation_budget_candidates=[(1, 2), (2, 4)],
        summary_selection_mode="all",
    )

    portfolio = stats["translation_budget_portfolio"]
    candidate_costs = portfolio["candidate_measurement_costs"]
    selected_index = min(
        range(len(candidate_costs)),
        key=lambda index: candidate_costs[index],
    )

    assert portfolio["candidate_count"] == 2
    assert portfolio["selected_index"] == selected_index
    assert stats["interface_schur_translation_ritz_rank"] == (
        portfolio["selected_rank"]
    )
    assert stats["interface_schur_translation_ritz_probe_iterations"] == (
        portfolio["selected_probe_iterations"]
    )
    assert stats["communication_estimate"][
        "translation_budget_portfolio_candidate_count"
    ] == 2


def test_summary_chordal_init_can_threshold_translation_ritz_rank():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_translation_ritz_rank=3,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_translation_ritz_probe_iterations=4,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        interface_schur_translation_ritz_rank_selection_mode=(
            "value_threshold"),
        interface_schur_translation_ritz_value_threshold=-1.0,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_translation_ritz_rank"] == 3
    assert stats["interface_schur_translation_ritz_rank_selection_mode"] == (
        "value_threshold")
    assert stats["interface_schur_translation_ritz_value_threshold"] == -1.0
    assert stats["translation_stats"]["ritz_deflation_requested_rank"] == 3
    assert stats["translation_stats"][
        "ritz_deflation_requested_rank_before_selection"] == 3
    assert stats["translation_stats"]["ritz_deflation_selected_rank"] == 1
    assert stats["translation_stats"]["ritz_deflation_basis_rank"] == 1


def test_summary_chordal_init_can_use_translation_ritz_energy_capture():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_translation_ritz_rank=3,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_translation_ritz_probe_iterations=4,
        interface_schur_ritz_mode="rot_low_trans_harmonic",
        interface_schur_translation_ritz_rank_selection_mode=(
            "energy_capture"),
        interface_schur_translation_ritz_energy_capture_fraction=0.8,
        summary_selection_mode="all",
    )

    assert stats["interface_schur_translation_ritz_rank_selection_mode"] == (
        "energy_capture")
    assert stats[
        "interface_schur_translation_ritz_energy_capture_fraction"] == 0.8
    assert stats["translation_stats"]["ritz_rank_selection_mode"] == (
        "energy_capture")
    assert stats["translation_stats"]["ritz_energy_capture_fraction"] == 0.8
    assert stats["translation_stats"]["ritz_energy_capture_total"] >= (
        stats["translation_stats"]["ritz_energy_capture_selected"])


def test_summary_chordal_init_can_report_central_equivalence_diagnostic():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        summary_selection_mode="all",
        central_equivalence_diagnostic=True,
    )

    assert stats["central_equivalence_diagnostic"] is True
    assert "central_equivalence" in stats["rotation_stats"]
    assert "central_equivalence" in stats["translation_stats"]
    assert stats["rotation_stats"]["central_equivalence"][
        "schur_energy_gap"] >= 0.0
    assert stats["translation_stats"]["central_equivalence"][
        "schur_energy_gap"] >= 0.0


def test_summary_chordal_init_can_report_iterative_central_equivalence():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        summary_selection_mode="all",
        central_equivalence_iterative_diagnostic_iterations=4,
    )

    assert stats["central_equivalence_iterative_diagnostic_iterations"] == 4
    assert "central_equivalence_iterative" in stats["rotation_stats"]
    assert "central_equivalence_iterative" in stats["translation_stats"]
    assert stats["rotation_stats"]["central_equivalence_iterative"][
        "materializes_dense_schur"] is False


def test_summary_chordal_init_can_use_ritz_portfolio_coarse_basis():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="portfolio",
        summary_selection_mode="all",
    )

    assert stats["interface_schur_ritz_mode"] == "portfolio"
    assert stats["rotation_stats"]["ritz_mode"] == "portfolio"
    assert stats["translation_stats"]["ritz_mode"] == "portfolio"
    assert stats["rotation_stats"]["ritz_portfolio_candidate_count"] == 4
    assert stats["translation_stats"]["ritz_portfolio_candidate_count"] == 4
    assert stats["communication_estimate"][
        "ritz_portfolio_scoring_interface_vector_mb"] > 0.0


def test_summary_chordal_init_measurement_portfolio_selects_min_cost_candidate():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    modes = [
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    ]
    candidate_costs = {}
    for mode in modes:
        poses, _ = dci.solve_summary_chordal_initialization(
            graph_edges=edges,
            pose_ids=[0, 1, 2, 3, 4, 5],
            robot_of=robot_of,
            rotation_iterations=2,
            translation_iterations=2,
            weighted=False,
            cost_mode="dpgo",
            linear_solver="interface_schur_pcg",
            interface_schur_preconditioner="block_jacobi+coarse",
            interface_schur_ritz_rank=1,
            interface_schur_ritz_probe_iterations=2,
            interface_schur_ritz_mode=mode,
            summary_selection_mode="all",
        )
        candidate_costs[mode] = total_chordal_cost(
            edges, poses, weighted=False, cost_mode="dpgo")

    poses, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="measurement_portfolio",
        summary_selection_mode="all",
    )

    selected_cost = total_chordal_cost(
        edges, poses, weighted=False, cost_mode="dpgo")
    best_mode = min(candidate_costs, key=candidate_costs.get)
    assert stats["interface_schur_ritz_mode"] == "measurement_portfolio"
    assert stats["measurement_portfolio_selected_mode"] == best_mode
    assert stats["measurement_portfolio_candidate_count"] == len(modes)
    assert np.isclose(
        stats["measurement_portfolio_selected_cost"],
        candidate_costs[best_mode])
    assert np.isclose(selected_cost, candidate_costs[best_mode])
    assert stats["measurement_portfolio_certificate_comm_mb"] > 0.0
    assert stats["measurement_portfolio_communication_estimate"][
        "linear_solve_total_estimated_mb"] >= stats["communication_estimate"][
            "linear_solve_total_estimated_mb"]


def test_summary_chordal_init_coarse_measurement_portfolio_uses_proxy_then_one_full_solve():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    modes = [
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    ]
    proxy_costs = {}
    for mode in modes:
        proxy_poses, _ = dci.solve_summary_chordal_initialization(
            graph_edges=edges,
            pose_ids=[0, 1, 2, 3, 4, 5],
            robot_of=robot_of,
            rotation_iterations=0,
            translation_iterations=0,
            weighted=False,
            cost_mode="dpgo",
            linear_solver="interface_schur_pcg",
            interface_schur_preconditioner="block_jacobi+coarse",
            interface_schur_coarse_initial_guess=True,
            interface_schur_ritz_rank=1,
            interface_schur_ritz_probe_iterations=2,
            interface_schur_ritz_mode=mode,
            summary_selection_mode="all",
        )
        proxy_costs[mode] = total_chordal_cost(
            edges, proxy_poses, weighted=False, cost_mode="dpgo")

    poses, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="coarse_measurement_portfolio",
        summary_selection_mode="all",
    )

    selected_mode = min(proxy_costs, key=proxy_costs.get)
    selected_cost = total_chordal_cost(
        edges, poses, weighted=False, cost_mode="dpgo")
    full_selected_poses, _ = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode=selected_mode,
        summary_selection_mode="all",
    )
    full_selected_cost = total_chordal_cost(
        edges, full_selected_poses, weighted=False, cost_mode="dpgo")

    assert stats["interface_schur_ritz_mode"] == "coarse_measurement_portfolio"
    assert stats["coarse_measurement_portfolio_selected_mode"] == selected_mode
    assert stats["coarse_measurement_portfolio_candidate_count"] == len(modes)
    assert np.isclose(
        stats["coarse_measurement_portfolio_selected_proxy_cost"],
        proxy_costs[selected_mode])
    assert np.isclose(selected_cost, full_selected_cost)
    assert stats["coarse_measurement_portfolio_proxy_comm_mb"] > 0.0
    assert stats["coarse_measurement_portfolio_selected_full_comm_mb"] > 0.0
    assert stats["coarse_measurement_portfolio_communication_estimate"][
        "coarse_measurement_portfolio_selected_full_solve_mb"] > 0.0


def test_summary_chordal_init_partial_measurement_portfolio_uses_pilot_then_one_full_solve():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(3, 4, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(4, 5, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 4, pose2(3.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    modes = [
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    ]
    pilot_costs = {}
    for mode in modes:
        pilot_poses, _ = dci.solve_summary_chordal_initialization(
            graph_edges=edges,
            pose_ids=[0, 1, 2, 3, 4, 5],
            robot_of=robot_of,
            rotation_iterations=1,
            translation_iterations=1,
            weighted=False,
            cost_mode="dpgo",
            linear_solver="interface_schur_pcg",
            interface_schur_preconditioner="block_jacobi+coarse",
            interface_schur_coarse_initial_guess=True,
            interface_schur_ritz_rank=1,
            interface_schur_ritz_probe_iterations=2,
            interface_schur_ritz_mode=mode,
            summary_selection_mode="all",
        )
        pilot_costs[mode] = total_chordal_cost(
            edges, pilot_poses, weighted=False, cost_mode="dpgo")

    poses, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode="partial_measurement_portfolio",
        summary_selection_mode="all",
    )

    selected_mode = min(pilot_costs, key=pilot_costs.get)
    selected_cost = total_chordal_cost(
        edges, poses, weighted=False, cost_mode="dpgo")
    full_selected_poses, _ = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3, 4, 5],
        robot_of=robot_of,
        rotation_iterations=2,
        translation_iterations=2,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="interface_schur_pcg",
        interface_schur_preconditioner="block_jacobi+coarse",
        interface_schur_coarse_initial_guess=True,
        interface_schur_ritz_rank=1,
        interface_schur_ritz_probe_iterations=2,
        interface_schur_ritz_mode=selected_mode,
        summary_selection_mode="all",
    )
    full_selected_cost = total_chordal_cost(
        edges, full_selected_poses, weighted=False, cost_mode="dpgo")

    assert stats["interface_schur_ritz_mode"] == "partial_measurement_portfolio"
    assert stats["partial_measurement_portfolio_selected_mode"] == selected_mode
    assert stats["partial_measurement_portfolio_candidate_count"] == len(modes)
    assert stats["partial_measurement_portfolio_proxy_rotation_iterations"] == 1
    assert stats["partial_measurement_portfolio_proxy_translation_iterations"] == 1
    assert np.isclose(
        stats["partial_measurement_portfolio_selected_proxy_cost"],
        pilot_costs[selected_mode])
    assert np.isclose(selected_cost, full_selected_cost)
    assert stats["partial_measurement_portfolio_proxy_comm_mb"] > 0.0
    assert stats["partial_measurement_portfolio_selected_full_comm_mb"] > 0.0


def test_omitted_force_correction_reports_reference_subspace_miss():
    full_hessian = np.asarray([
        [4.0, 1.0],
        [1.0, 3.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.0, 0.0])
    full_gradient = np.asarray([1.0, 2.0])
    selected_gradient = selected_hessian @ selected_solution
    reference_solution = np.linalg.solve(full_hessian, full_gradient)
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
    }

    _, stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=2,
        curvature_model="subspace_secant",
        curvature_rank=2,
        reference_solution=reference_solution,
        reference_label="full_normal_solution",
    )

    cert = stats["reference_subspace_error"]
    assert cert["reference_label"] == "full_normal_solution"
    assert cert["subspace_rank"] >= 1
    assert cert["error_norm"] >= 0.0
    assert 0.0 <= cert["missed_error_energy_fraction"] <= 1.0


def test_subspace_secant_can_account_pairwise_projected_curvature_payload():
    full_hessian = np.asarray([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 1.0],
        [0.5, 1.0, 2.0],
    ])
    selected_hessian = np.diag(np.diag(full_hessian))
    selected_solution = np.asarray([0.5, -0.25, 0.75])
    selected_gradient = selected_hessian @ selected_solution
    full_gradient = selected_gradient.copy()
    blocks = {
        0: np.asarray([0]),
        1: np.asarray([1]),
        2: np.asarray([2]),
    }

    _, global_stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
        curvature_payload_model="global_projected",
    )
    _, pairwise_stats = dci.apply_omitted_force_correction(
        full_hessian=full_hessian,
        full_gradient=full_gradient,
        selected_hessian=selected_hessian,
        selected_gradient=selected_gradient,
        blocks=blocks,
        solution=selected_solution,
        iterations=10,
        linear_solver="compact_pcg",
        relaxation=1.0,
        damping=1e-12,
        block_dim=1,
        correction_rounds=3,
        curvature_model="subspace_secant",
        curvature_rank=2,
        curvature_payload_model="pairwise_projected",
    )

    global_comm = global_stats["communication_estimate"]
    pairwise_comm = pairwise_stats["communication_estimate"]
    assert pairwise_stats["curvature_communication_estimate"][
        "payload_model"
    ] == "pairwise_projected"
    assert pairwise_stats["curvature_communication_estimate"][
        "projected_block_pair_count"
    ] > 0
    assert pairwise_stats["curvature_communication_estimate"][
        "projected_source"
    ] == "pairwise_block_aggregation"
    assert pairwise_stats["curvature_communication_estimate"][
        "max_pairwise_projection_error"
    ] < 1e-10
    assert pairwise_stats["curvature_communication_estimate"][
        "mean_pairwise_projection_error"
    ] < 1e-10
    first_round = pairwise_stats["rounds"][0]
    assert first_round["subspace_pairwise_projection_error"] < 1e-10
    assert pairwise_comm["curvature_payload_bytes"] > (
        global_comm["curvature_payload_bytes"]
    )
    assert np.isclose(
        pairwise_stats["after"]["full_model_residual_norm"],
        global_stats["after"]["full_model_residual_norm"],
    )


def test_summary_chordal_solver_residual_force_refinement_reduces_omitted_residual():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), None, 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    _, seed_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
    )
    _, refined_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="residual_force_refinement",
        summary_max_offdiag_block_edges=2,
        summary_refinement_max_offdiag_block_edges=3,
    )

    seed_omitted = seed_stats["summary_model_representativeness"]["combined"][
        "omitted_model_residual_norm"
    ]
    refined_omitted = refined_stats["summary_model_representativeness"]["combined"][
        "omitted_model_residual_norm"
    ]
    assert refined_omitted < seed_omitted
    assert refined_stats["rotation_separator_summary"]["summary_selection"][
        "mode"
    ] == "residual_force_refinement"


def test_summary_chordal_solver_omitted_force_correction_reduces_full_residual():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), None, 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    _, seed_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
    )
    _, corrected_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=3,
    )

    seed_full = seed_stats["summary_model_representativeness"]["combined"][
        "full_model_residual_norm"
    ]
    corrected_full = corrected_stats["summary_model_representativeness"][
        "combined"
    ]["full_model_residual_norm"]
    assert corrected_full < seed_full
    assert corrected_stats["omitted_force_correction"]["communication_estimate"][
        "total_omitted_force_payload_bytes"
    ] > 0
    assert corrected_stats["omitted_force_correction"]["rotation"][
        "accepted_rounds"
    ] >= 1


def test_summary_chordal_solver_omitted_force_portfolio_keeps_best_measured_round():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), None, 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    last_poses, _ = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=4,
    )
    portfolio_poses, portfolio_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=4,
        omitted_force_correction_portfolio=True,
    )

    portfolio = portfolio_stats["omitted_force_correction_portfolio"]
    assert portfolio["enabled"] is True
    assert portfolio["candidate_count"] == 5
    assert 0 <= portfolio["selected_rounds"] <= 4
    assert portfolio["selected_measurement_cost"] == min(
        portfolio["candidate_measurement_costs"]
    )
    assert total_chordal_cost(edges, portfolio_poses) <= (
        total_chordal_cost(edges, last_poses) + 1e-12
    )


def test_summary_chordal_solver_passes_omitted_force_curvature_model():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), None, 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(4.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    _, stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="omitted_force_correction",
        summary_max_offdiag_block_edges=2,
        omitted_force_correction_rounds=2,
        omitted_force_correction_curvature_model="block_gershgorin",
    )

    correction = stats["omitted_force_correction"]
    assert correction["rotation"]["curvature_model"] == "block_gershgorin"
    assert correction["translation"]["curvature_model"] == "block_gershgorin"
    assert correction["communication_estimate"][
        "total_curvature_payload_bytes"
    ] > 0
    assert correction["communication_estimate"][
        "total_correction_payload_bytes"
    ] >= correction["communication_estimate"][
        "total_omitted_force_payload_bytes"
    ]


def test_summary_chordal_solver_matches_edge_subgraph_solver():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(2.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 1}

    edge_solve, edge_stats = dci.solve_distributed_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=20,
        translation_iterations=20,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )
    summary_solve, summary_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=20,
        translation_iterations=20,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )

    assert summary_stats["method"] == "summary_chordal_normal_equation_solve"
    assert summary_stats["normal_equation_summary_communication"]["separator_edge_count"] == 2
    for pose_id in edge_solve:
        assert np.allclose(summary_solve[pose_id], edge_solve[pose_id], atol=1e-9)


def test_summary_chordal_solver_can_use_structural_spanning_summary_selection():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(0, 2, pose2(2.0, 0.0, 0.0), None, 2),
        Edge(0, 3, pose2(3.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(2.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    _, full_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
    )
    selected_poses, selected_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=10,
        translation_iterations=10,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
        summary_selection_mode="structural_spanning",
        summary_max_offdiag_block_edges=2,
    )

    assert set(selected_poses) == {0, 1, 2, 3}
    assert selected_stats["summary_selection_mode"] == "structural_spanning"
    assert selected_stats["rotation_separator_summary"]["summary_selection"][
        "selected_offdiag_block_edges"
    ] == 2
    assert selected_stats["normal_equation_summary_communication"][
        "total_summary_bytes"
    ] < full_stats["normal_equation_summary_communication"]["total_summary_bytes"]


def test_compact_pcg_summary_solver_matches_pcg_with_reduction_only_comm():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 3, pose2(2.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 1}

    pcg_solve, pcg_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=20,
        translation_iterations=20,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="pcg",
    )
    compact_solve, compact_stats = dci.solve_summary_chordal_initialization(
        graph_edges=edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
        rotation_iterations=20,
        translation_iterations=20,
        weighted=False,
        cost_mode="dpgo",
        linear_solver="compact_pcg",
    )

    for pose_id in pcg_solve:
        assert np.allclose(compact_solve[pose_id], pcg_solve[pose_id], atol=1e-12)

    compact_comm = compact_stats["communication_estimate"]
    pcg_comm = pcg_stats["communication_estimate"]
    assert compact_stats["linear_solver"] == "compact_pcg"
    assert compact_comm["model"] == "normal_summary_compact_pcg_global_reductions"
    assert compact_comm["separator_exchange_bytes"] == 0
    assert compact_comm["linear_solve_total_estimated_bytes"] == (
        compact_comm["pcg_global_reduction_bytes"]
    )
    assert compact_comm["pcg_global_reduction_bytes"] == (
        pcg_comm["pcg_global_reduction_bytes"]
    )
    assert compact_comm["linear_solve_total_estimated_bytes"] < (
        pcg_comm["linear_solve_total_estimated_bytes"]
    )


def test_residual_certificate_total_delta_matches_global_cost_delta():
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
    candidate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
    }

    cert = dci.compute_candidate_residual_certificate(
        graph_edges=edges,
        baseline=baseline,
        candidate=candidate,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
    )

    assert cert["candidate_total_cost"] < cert["baseline_total_cost"]
    assert abs(cert["total_delta"] - (
        cert["candidate_total_cost"] - cert["baseline_total_cost"])) < 1e-12
    assert cert["global_consensus_accept"] is True
    assert cert["strict_local_accept"] is True


def test_residual_certificate_reports_delta_consensus_communication():
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
    candidate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
    }

    cert = dci.compute_candidate_residual_certificate(
        graph_edges=edges,
        baseline=baseline,
        candidate=candidate,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
    )

    comm = cert["communication_estimate"]
    assert comm["robot_count"] == 2
    assert comm["residual_delta_allreduce_count"] == 1
    assert comm["residual_delta_consensus_bytes"] == 2 * (2 - 1) * 8
    assert comm["residual_delta_consensus_mb"] == (
        comm["residual_delta_consensus_bytes"] / (1024.0 * 1024.0)
    )


def test_strict_local_certificate_can_be_more_conservative_than_global_consensus():
    edges = [
        Edge(0, 1, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(1, 2, pose2(1.0, 0.0, 0.0), None, 2),
        Edge(2, 3, pose2(1.0, 0.0, 0.0), None, 2),
    ]
    robot_of = {0: 0, 1: 0, 2: 1, 3: 1}
    baseline = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(4.0, 0.0, 0.0),
        3: pose2(5.0, 0.0, 0.0),
    }
    candidate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
        3: pose2(4.2, 0.0, 0.0),
    }

    cert = dci.compute_candidate_residual_certificate(
        graph_edges=edges,
        baseline=baseline,
        candidate=candidate,
        robot_of=robot_of,
        weighted=False,
        cost_mode="dpgo",
    )

    assert cert["total_delta"] < 0.0
    assert cert["global_consensus_accept"] is True
    assert cert["strict_local_accept"] is False
    assert cert["private_delta_by_robot"]["1"] > 0.0


def test_manual_matrix_writer_round_trips_interleaved_pose_blocks(tmp_path: Path):
    poses = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 2.0, math.pi / 2.0),
    }
    out = tmp_path / "estimate.txt"

    dci.write_manual_matrix_pose_set(out, poses, dim=2)
    parsed = oracle.parse_manual_matrix_estimate(out)

    assert np.allclose(parsed[0], poses[0], atol=1e-12)
    assert np.allclose(parsed[1], poses[1], atol=1e-12)


def test_materialize_certified_hybrid_writes_certificate_selected_candidate(tmp_path: Path):
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
    candidate = {
        0: pose2(0.0, 0.0, 0.0),
        1: pose2(1.0, 0.0, 0.0),
        2: pose2(2.0, 0.0, 0.0),
    }
    out = tmp_path / "selected.txt"

    result = dci.materialize_certified_hybrid_estimate(
        graph_edges=edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
        baseline=baseline,
        candidate=candidate,
        weighted=False,
        cost_mode="dpgo",
        output_path=out,
    )
    parsed = oracle.parse_manual_matrix_estimate(out)

    assert result["selected"] == "distributed_chordal_pcg"
    assert np.allclose(parsed[2], candidate[2], atol=1e-12)
