import math
import json

import numpy as np

from scripts.evaluate_pgo import Edge
from scripts import dci_basin_skeleton as skeleton
from scripts import analyze_distributed_chordal_init as dci


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


def edge(i, j):
    return Edge(i, j, pose2(1.0, 0.0, 0.0), None, 2)


def test_basin_skeleton_marks_bridge_and_articulation_pose_candidates():
    graph_edges = [edge(1, 2), edge(0, 1)]
    robot_of = {0: 0, 1: 1, 2: 2}

    report = skeleton.build_basin_skeleton(
        graph_edges=graph_edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
    )

    assert report["robot_graph_edges"] == [
        {"robots": [0, 1], "edge_count": 1, "separator_pose_ids": [0, 1]},
        {"robots": [1, 2], "edge_count": 1, "separator_pose_ids": [1, 2]},
    ]
    assert report["bridge_robot_edges"] == [[0, 1], [1, 2]]
    assert report["articulation_robots"] == [1]
    assert report["bridge_pose_ids"] == [0, 1, 2]
    assert report["articulation_pose_ids"] == [1]
    assert report["shared_separator_pose_ids"] == [0, 1, 2]
    assert report["skeleton_pose_ids"] == [0, 1, 2]
    assert report["z_map"] == [
        {
            "block_index": 0,
            "pose_id": 0,
            "robot": 0,
            "reasons": ["bridge", "shared_separator"],
        },
        {
            "block_index": 1,
            "pose_id": 1,
            "robot": 1,
            "reasons": ["articulation", "bridge", "shared_separator"],
        },
        {
            "block_index": 2,
            "pose_id": 2,
            "robot": 2,
            "reasons": ["bridge", "shared_separator"],
        },
    ]


def test_initial_skeleton_policy_separates_full_shared_from_structural_seed():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
        edge(2, 3),
        edge(1, 3),
    ]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    report = skeleton.build_basin_skeleton(
        graph_edges=graph_edges,
        pose_ids=[0, 1, 2, 3],
        robot_of=robot_of,
    )

    all_shared = skeleton.select_initial_skeleton_pose_ids(
        report, policy="shared_separator")
    structural = skeleton.select_initial_skeleton_pose_ids(
        report, policy="bridge_articulation")

    assert all_shared == [0, 1, 2, 3]
    assert structural == [0, 1]
    assert len(structural) < len(all_shared)


def test_basin_skeleton_reports_deterministic_cycle_basis():
    graph_edges = [edge(2, 0), edge(1, 2), edge(0, 1)]
    robot_of = {0: 0, 1: 1, 2: 2}

    report = skeleton.build_basin_skeleton(
        graph_edges=graph_edges,
        pose_ids=[0, 1, 2],
        robot_of=robot_of,
    )

    assert report["bridge_robot_edges"] == []
    assert report["articulation_robots"] == []
    assert report["cycle_basis"] == [[0, 1, 2]]
    assert report["cycle_pose_ids"] == [0, 1, 2]
    assert report["skeleton_pose_ids"] == [0, 1, 2]
    assert [entry["reasons"] for entry in report["z_map"]] == [
        ["cycle", "shared_separator"],
        ["cycle", "shared_separator"],
        ["cycle", "shared_separator"],
    ]


def test_basin_skeleton_cli_writes_json_report(tmp_path):
    g2o_path = tmp_path / "chain.g2o"
    output_path = tmp_path / "skeleton.json"
    g2o_path.write_text(
        "\n".join(
            [
                "VERTEX_SE2 0 0 0 0",
                "VERTEX_SE2 1 1 0 0",
                "VERTEX_SE2 2 2 0 0",
                "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
                "EDGE_SE2 1 2 1 0 0 1 0 0 1 0 1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = skeleton.main(
        [
            "--g2o",
            str(g2o_path),
            "--num-robots",
            "3",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["dataset"] == "chain.g2o"
    assert report["robot_index_ranges"] == [[0, 1], [1, 2], [2, 3]]
    assert report["skeleton"]["bridge_robot_edges"] == [[0, 1], [1, 2]]
    assert report["skeleton"]["articulation_robots"] == [1]


def test_skeleton_reduced_schur_matches_full_interface_when_all_blocks_selected():
    graph_edges = [edge(0, 1), edge(1, 2)]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
        )
    )
    full_schur, full_rhs, _ = dci.sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )

    structural = skeleton.build_basin_skeleton(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
    )
    reduced = skeleton.project_local_interface_schur_to_skeleton(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        skeleton_pose_ids=structural["skeleton_pose_ids"],
    )

    assert reduced["selected_interface_pose_ids"] == [1, 2]
    assert reduced["basis_column_count"] == full_schur.shape[0]
    assert np.allclose(reduced["reduced_schur"], full_schur, atol=1e-12)
    assert np.allclose(reduced["reduced_rhs"], full_rhs, atol=1e-12)
    assert np.allclose(reduced["basis"], np.eye(full_schur.shape[0]), atol=1e-12)


def test_skeleton_reduced_solve_recovers_full_schur_solution_for_identity_basis():
    graph_edges = [edge(0, 1), edge(1, 2)]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )
    full_schur, full_rhs, _ = dci.sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    full_solution = np.linalg.pinv(full_schur, rcond=1e-12) @ full_rhs
    structural = skeleton.build_basin_skeleton(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
    )

    solved = skeleton.solve_skeleton_reduced_interface_schur(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        skeleton_pose_ids=structural["skeleton_pose_ids"],
    )

    assert solved["selected_interface_pose_ids"] == [1, 2]
    assert np.allclose(solved["lifted_interface_solution"], full_solution, atol=1e-12)
    assert np.allclose(solved["omitted_force"], np.zeros_like(full_solution), atol=1e-12)
    assert solved["omitted_force_norm"] < 1e-12


def test_strict_subset_skeleton_ranks_missing_pose_by_omitted_force():
    graph_edges = [edge(0, 1), edge(1, 2)]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )

    solved = skeleton.solve_skeleton_reduced_interface_schur(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        skeleton_pose_ids=[1],
    )
    ranking = skeleton.rank_omitted_force_blocks(
        omitted_force=solved["omitted_force"],
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        selected_interface_pose_ids=solved["selected_interface_pose_ids"],
    )

    assert solved["selected_interface_pose_ids"] == [1]
    assert solved["omitted_force_norm"] > 0.0
    assert ranking[0]["pose_id"] == 2
    assert ranking[0]["selected"] is False
    assert ranking[0]["block_norm"] > 0.0


def test_enrich_skeleton_by_omitted_force_adds_top_missing_pose():
    graph_edges = [edge(0, 1), edge(1, 2)]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )
    solved = skeleton.solve_skeleton_reduced_interface_schur(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        skeleton_pose_ids=[1],
    )

    enriched = skeleton.enrich_skeleton_by_omitted_force(
        skeleton_pose_ids=[1],
        omitted_force_blocks=solved["omitted_force_blocks"],
        max_add=1,
    )

    assert enriched["added_pose_ids"] == [2]
    assert enriched["skeleton_pose_ids"] == [1, 2]
    assert enriched["reason"] == "omitted_force"


def test_iterative_omitted_force_enrichment_reaches_full_interface_chain():
    graph_edges = [edge(0, 1), edge(1, 2)]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )

    result = skeleton.run_omitted_force_enrichment_iterations(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        initial_skeleton_pose_ids=[1],
        max_iterations=3,
        max_add_per_iteration=1,
    )

    assert result["final_skeleton_pose_ids"] == [1, 2]
    assert result["converged"] is True
    assert result["history"][0]["skeleton_pose_ids"] == [1]
    assert result["history"][0]["added_pose_ids"] == [2]
    assert result["history"][0]["interface_solution_error_norm"] > 0.0
    assert result["history"][1]["skeleton_pose_ids"] == [1, 2]
    assert result["history"][1]["added_pose_ids"] == []
    assert result["history"][1]["interface_solution_error_norm"] < 1e-12
    assert result["final_solve"]["omitted_force_norm"] < 1e-12


def test_graph_stage_omitted_force_diagnostic_starts_from_structural_subset():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
        edge(2, 3),
        edge(1, 3),
    ]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    result = skeleton.run_graph_stage_omitted_force_enrichment(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        initial_policy="bridge_articulation",
        max_iterations=4,
        max_add_per_iteration=1,
    )

    assert result["initial_policy"] == "bridge_articulation"
    assert result["initial_skeleton_pose_ids"] == [0, 1]
    assert result["variable_initial_skeleton_pose_ids"] == [1]
    assert result["full_interface_pose_ids"] == [0, 1, 2, 3]
    assert result["variable_interface_pose_ids"] == [1, 2, 3]
    assert len(result["variable_initial_skeleton_pose_ids"]) < len(
        result["variable_interface_pose_ids"])
    assert result["enrichment"]["converged"] is True
    assert result["enrichment"]["final_skeleton_pose_ids"] == [0, 1, 2, 3]
    assert result["enrichment"]["final_solve"]["selected_interface_pose_ids"] == [
        1, 2, 3]


def test_robot_coordinate_skeleton_basis_shares_columns_within_robot():
    basis_report = skeleton.robot_coordinate_skeleton_basis(
        interface_pose_ids=[0, 1, 2, 3],
        interface_variable_count=6,
        block_dim=2,
        robot_of={0: 0, 1: 1, 2: 1, 3: 2},
        anchor_pose=0,
    )

    basis = basis_report["basis"]

    assert basis_report["variable_interface_pose_ids"] == [1, 2, 3]
    assert basis_report["selected_robot_ids"] == [1, 2]
    assert basis_report["basis_row_count"] == 6
    assert basis_report["basis_column_count"] == 4
    assert basis.shape == (6, 4)
    assert basis_report["basis_column_count"] < 6

    # Poses 1 and 2 are owned by robot 1, so their x/y rows share columns 0/1.
    assert np.allclose(basis[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(basis[1], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(basis[2], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(basis[3], [0.0, 1.0, 0.0, 0.0])

    # Pose 3 is owned by robot 2 and receives its own robot-coordinate columns.
    assert np.allclose(basis[4], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(basis[5], [0.0, 0.0, 0.0, 1.0])


def test_project_local_interface_schur_with_robot_coordinate_basis():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )
    full_schur, full_rhs, _ = dci.sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    basis_report = skeleton.robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
    )

    projected = skeleton.project_local_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        basis=basis_report["basis"],
        basis_metadata=basis_report,
    )

    assert projected["basis_model"] == "robot_coordinate_skeleton_basis"
    assert projected["basis_row_count"] == full_schur.shape[0]
    assert projected["basis_column_count"] == 4
    assert projected["basis_column_count"] < full_schur.shape[0]
    assert np.allclose(
        projected["reduced_schur"],
        basis_report["basis"].T @ full_schur @ basis_report["basis"],
        atol=1e-12,
    )
    assert np.allclose(
        projected["reduced_rhs"],
        basis_report["basis"].T @ full_rhs,
        atol=1e-12,
    )


def test_robot_coordinate_solve_exposes_pose_specific_omitted_force():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )
    basis_report = skeleton.robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
    )

    solved = skeleton.solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        basis=basis_report["basis"],
        basis_metadata=basis_report,
    )

    assert solved["basis_model"] == "robot_coordinate_skeleton_basis"
    assert solved["basis_column_count"] == 4
    assert solved["lifted_interface_solution"].shape == (6,)
    assert solved["reduced_residual_norm"] < 1e-12
    assert solved["interface_solution_error_norm"] > 0.0
    assert solved["omitted_force_norm"] > 0.0
    assert len(solved["omitted_force_blocks"]) == 3
    assert solved["omitted_force_blocks"][0]["selected"] is False
    assert solved["omitted_force_blocks"][0]["block_norm"] > 0.0


def test_cycle_coordinate_skeleton_basis_shares_columns_across_cycle():
    basis_report = skeleton.cycle_coordinate_skeleton_basis(
        interface_pose_ids=[0, 1, 2, 3],
        interface_variable_count=6,
        block_dim=2,
        robot_of={0: 0, 1: 1, 2: 2, 3: 3},
        cycle_basis=[[1, 2, 3]],
        anchor_pose=0,
    )

    basis = basis_report["basis"]

    assert basis_report["variable_interface_pose_ids"] == [1, 2, 3]
    assert basis_report["selected_cycles"] == [[1, 2, 3]]
    assert basis_report["basis_row_count"] == 6
    assert basis_report["basis_column_count"] == 2
    assert basis.shape == (6, 2)
    assert basis_report["basis_column_count"] < 6

    # All three cycle-owned poses share the same x/y cycle-coordinate columns.
    assert np.allclose(basis[0], [1.0, 0.0])
    assert np.allclose(basis[1], [0.0, 1.0])
    assert np.allclose(basis[2], [1.0, 0.0])
    assert np.allclose(basis[3], [0.0, 1.0])
    assert np.allclose(basis[4], [1.0, 0.0])
    assert np.allclose(basis[5], [0.0, 1.0])


def test_separator_component_coordinate_basis_shares_columns_per_separator_component():
    graph_edges = [
        edge(0, 1),
        edge(1, 3),
        edge(2, 4),
    ]
    basis_report = skeleton.separator_component_coordinate_skeleton_basis(
        graph_edges=graph_edges,
        interface_pose_ids=[0, 1, 2, 3, 4],
        interface_variable_count=8,
        block_dim=2,
        robot_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
        anchor_pose=0,
    )

    basis = basis_report["basis"]

    assert basis_report["model"] == (
        "separator_component_coordinate_skeleton_basis"
    )
    assert basis_report["variable_interface_pose_ids"] == [1, 2, 3, 4]
    assert basis_report["separator_components"] == [[1, 3], [2, 4]]
    assert basis_report["basis_row_count"] == 8
    assert basis_report["basis_column_count"] == 4
    assert basis.shape == (8, 4)

    # Poses 1 and 3 are in the first separator component.
    assert np.allclose(basis[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(basis[1], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(basis[4], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(basis[5], [0.0, 1.0, 0.0, 0.0])

    # Poses 2 and 4 are in the second separator component.
    assert np.allclose(basis[2], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(basis[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(basis[6], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(basis[7], [0.0, 0.0, 0.0, 1.0])


def test_separator_component_geneo_basis_selects_rhs_coupled_component_modes():
    full_schur = np.diag([1.0, 10.0, 3.0, 20.0])
    full_rhs = np.asarray([0.0, 20.0, 3.0, 0.0], dtype=float)

    basis_report = skeleton.separator_component_geneo_skeleton_basis(
        graph_edges=[edge(1, 2)],
        interface_pose_ids=[1, 2],
        interface_variable_count=4,
        block_dim=2,
        robot_of={1: 1, 2: 2},
        full_schur=full_schur,
        full_rhs=full_rhs,
        max_modes_per_component=2,
    )

    basis = basis_report["basis"]
    active_rows = sorted({
        int(np.argmax(np.abs(basis[:, column])))
        for column in range(basis.shape[1])
    })

    assert basis_report["model"] == "separator_component_geneo_skeleton_basis"
    assert basis_report["separator_components"] == [[1, 2]]
    assert basis_report["basis_column_count"] == 2
    assert basis_report["component_records"][0]["selected_mode_rows"] == [1, 2]
    assert active_rows == [1, 2]
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-12)


def test_separator_component_merit_basis_selects_high_value_component_atoms():
    graph_edges = [
        edge(1, 2),
        edge(3, 4),
    ]
    full_schur = np.eye(8, dtype=float)
    full_rhs = np.asarray([1.0, 0.0, 1.0, 0.0, 5.0, 0.0, 5.0, 0.0])

    basis_report = skeleton.separator_component_merit_skeleton_basis(
        graph_edges=graph_edges,
        interface_pose_ids=[1, 2, 3, 4],
        interface_variable_count=8,
        block_dim=2,
        robot_of={1: 1, 2: 2, 3: 3, 4: 4},
        full_schur=full_schur,
        full_rhs=full_rhs,
        max_selected_components=1,
    )

    basis = basis_report["basis"]

    assert basis_report["model"] == "separator_component_merit_skeleton_basis"
    assert basis_report["selected_components"] == [[3, 4]]
    assert basis_report["rejected_components"] == [[1, 2]]
    assert basis_report["basis_column_count"] == 2
    assert basis.shape == (8, 2)
    assert np.allclose(basis[4], [1.0, 0.0])
    assert np.allclose(basis[5], [0.0, 1.0])
    assert np.allclose(basis[6], [1.0, 0.0])
    assert np.allclose(basis[7], [0.0, 1.0])
    assert basis_report["component_records"][0]["selected"] is True
    assert basis_report["component_records"][0]["projected_merit"] > (
        basis_report["component_records"][1]["projected_merit"]
    )


def test_cycle_coordinate_solve_uses_same_schur_certificate():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(3, 1)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}
    structural = skeleton.build_basin_skeleton(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
    )
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            weighted=False,
            cost_mode="dpgo",
            rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        )
    )
    basis_report = skeleton.cycle_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        cycle_basis=structural["cycle_basis"],
    )

    solved = skeleton.solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        basis=basis_report["basis"],
        basis_metadata=basis_report,
    )

    assert structural["cycle_basis"] == [[1, 2, 3]]
    assert solved["basis_model"] == "cycle_coordinate_skeleton_basis"
    assert solved["basis_column_count"] == 2
    assert solved["basis_column_count"] < len(interface_indices)
    assert solved["lifted_interface_solution"].shape == (6,)
    assert solved["reduced_residual_norm"] < 1e-12
    assert len(solved["omitted_force_blocks"]) == 3


def test_graph_stage_primal_skeleton_basis_diagnostic_reports_cycle_compression():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(3, 1)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    result = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        basis_mode="cycle_coordinate",
    )

    assert result["basis_mode"] == "cycle_coordinate"
    assert result["basis_model"] == "cycle_coordinate_skeleton_basis"
    assert result["structural_skeleton"]["cycle_basis"] == [[1, 2, 3]]
    assert result["full_interface_variable_count"] == 6
    assert result["basis_column_count"] == 2
    assert result["basis_compression_ratio"] == 2.0 / 6.0
    assert result["solve"]["reduced_residual_norm"] < 1e-12


def test_rank_aware_structural_span_selector_drops_dependent_and_zero_merit():
    identity = np.eye(4, dtype=float)
    rhs = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
    useful_and_zero = {
        "model": "candidate_a",
        "basis": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
        ),
    }
    duplicate = {
        "model": "candidate_b",
        "basis": np.asarray(
            [
                [1.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            dtype=float,
        ),
    }

    selected = skeleton.rank_aware_structural_span_basis(
        candidate_basis_reports=[useful_and_zero, duplicate],
        full_schur=identity,
        full_rhs=rhs,
    )

    assert selected["basis_model"] == "rank_aware_structural_span_basis"
    assert selected["basis_column_count"] == 1
    assert selected["accepted_columns"][0]["source_model"] == "candidate_a"
    assert selected["accepted_columns"][0]["source_column"] == 0
    rejected_reasons = {item["reason"] for item in selected["rejected_columns"]}
    assert "dependent" in rejected_reasons
    assert "zero_incremental_merit" in rejected_reasons
    assert selected["final_omitted_force_norm"] < 1e-12


def test_graph_stage_rank_aware_robot_cycle_span_reports_independent_columns():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(3, 1)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 2, 3: 3}

    result = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations={pose_id: np.eye(2) for pose_id in pose_ids},
        basis_mode="rank_aware_robot_cycle_span",
    )

    assert result["basis_mode"] == "rank_aware_robot_cycle_span"
    assert result["basis_model"] == "rank_aware_structural_span_basis"
    assert result["basis_column_count"] <= 8
    assert result["basis"]["candidate_column_count"] == 8
    assert result["basis"]["accepted_column_count"] == result["basis_column_count"]
    assert result["solve"]["reduced_singular"] is False
    assert result["solve"]["reduced_residual_norm"] < 1e-12


def test_pose_block_promotion_basis_selects_top_omitted_force_pose():
    omitted_blocks = [
        {"pose_id": 1, "selected": False, "block_norm": 0.2},
        {"pose_id": 2, "selected": False, "block_norm": 3.0},
        {"pose_id": 3, "selected": False, "block_norm": 1.0},
    ]

    report = skeleton.pose_block_promotion_basis(
        interface_pose_ids=[0, 1, 2, 3],
        interface_variable_count=6,
        block_dim=2,
        omitted_force_blocks=omitted_blocks,
        max_promoted_pose_blocks=1,
        anchor_pose=0,
    )

    assert report["model"] == "pose_block_promotion_basis"
    assert report["promoted_pose_ids"] == [2]
    assert report["basis_column_count"] == 2
    assert np.allclose(report["basis"][2], [1.0, 0.0])
    assert np.allclose(report["basis"][3], [0.0, 1.0])


def test_graph_stage_pose_promotion_reduces_omitted_force_after_robot_span():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    base = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
    )
    promoted = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span_plus_pose_promote",
        promoted_pose_block_count=1,
    )

    assert promoted["basis_mode"] == "rank_aware_robot_cycle_span_plus_pose_promote"
    assert promoted["promotion"]["promoted_pose_ids"]
    assert promoted["basis_column_count"] > base["basis_column_count"]
    assert promoted["solve"]["omitted_force_norm"] < base["solve"]["omitted_force_norm"]
    assert promoted["solve"]["reduced_singular"] is False


def test_value_per_byte_pose_promotion_skips_low_value_blocks():
    full_schur = np.eye(6, dtype=float)
    full_rhs = np.array([4.0, 0.0, 1.0, 0.0, 0.1, 0.0], dtype=float)
    omitted_blocks = [
        {"pose_id": 1, "selected": False, "block_norm": 4.0},
        {"pose_id": 2, "selected": False, "block_norm": 1.0},
        {"pose_id": 3, "selected": False, "block_norm": 0.1},
    ]

    report = skeleton.pose_block_value_per_byte_promotion_basis(
        interface_pose_ids=[1, 2, 3],
        interface_variable_count=6,
        block_dim=2,
        omitted_force_blocks=omitted_blocks,
        full_schur=full_schur,
        full_rhs=full_rhs,
        base_basis=np.zeros((6, 0), dtype=float),
        min_value_per_byte=0.1,
        max_promoted_pose_blocks=3,
        payload_bytes_per_coordinate=8,
    )

    assert report["model"] == "pose_block_value_per_byte_promotion_basis"
    assert report["promoted_pose_ids"] == [1]
    assert report["basis_column_count"] == 2
    assert report["accepted_promotions"][0]["pose_id"] == 1
    assert report["accepted_promotions"][0]["value_per_byte"] > 0.1
    rejected = {item["pose_id"]: item["reason"] for item in report["rejected_promotions"]}
    assert rejected[2] == "below_value_threshold"
    assert rejected[3] == "below_value_threshold"


def test_append_independent_columns_matches_full_orthonormalization():
    current = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    candidate = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    combined, added = skeleton._append_independent_columns_to_orthonormal_basis(
        current,
        candidate,
        row_count=4,
        tolerance=1e-12,
    )
    reference = skeleton._orthonormalize_basis_columns(
        np.column_stack([current, candidate]),
        row_count=4,
        tolerance=1e-12,
    )

    assert added == 2
    assert combined.shape == reference.shape
    assert np.allclose(combined @ combined.T, reference @ reference.T, atol=1e-12)


def test_value_per_byte_pose_promotion_relative_stop_is_scale_invariant():
    omitted_blocks = [
        {"pose_id": 1, "selected": False, "block_norm": 4.0},
        {"pose_id": 2, "selected": False, "block_norm": 1.0},
        {"pose_id": 3, "selected": False, "block_norm": 0.1},
    ]

    def run(scale):
      rhs = scale * np.array([4.0, 0.0, 1.0, 0.0, 0.1, 0.0], dtype=float)
      return skeleton.pose_block_value_per_byte_promotion_basis(
          interface_pose_ids=[1, 2, 3],
          interface_variable_count=6,
          block_dim=2,
          omitted_force_blocks=omitted_blocks,
          full_schur=np.eye(6, dtype=float),
          full_rhs=rhs,
          base_basis=np.zeros((6, 0), dtype=float),
          min_value_per_byte=0.0,
          min_relative_value_fraction=0.1,
          max_promoted_pose_blocks=3,
          payload_bytes_per_coordinate=8,
      )

    base = run(1.0)
    scaled = run(10.0)

    assert base["promoted_pose_ids"] == [1]
    assert scaled["promoted_pose_ids"] == [1]
    assert base["first_value_per_byte"] > 0.0
    assert scaled["first_value_per_byte"] > base["first_value_per_byte"]
    rejected = {item["pose_id"]: item["reason"] for item in base["rejected_promotions"]}
    assert rejected[2] == "below_relative_value_threshold"
    assert rejected[3] == "below_relative_value_threshold"


def test_basin_certificate_promotion_rejects_merit_step_that_worsens_omitted_force():
    full_schur = np.array([
        [4.08522644, 2.28004623, 1.19517536],
        [2.28004623, 5.06888221, 4.00640393],
        [1.19517536, 4.00640393, 7.00186192],
    ], dtype=float)
    full_rhs = np.array([-1.031863, 1.0090403, -0.60434096], dtype=float)
    base_basis = np.array([
        [-0.07267757],
        [-0.70576217],
        [-0.7047111],
    ], dtype=float)
    omitted_blocks = [
        {"pose_id": 0, "selected": False, "block_norm": 1.08787937},
        {"pose_id": 1, "selected": False, "block_norm": 0.8752022},
        {"pose_id": 2, "selected": False, "block_norm": 0.76431344},
    ]

    report = skeleton.pose_block_basin_certificate_promotion_basis(
        interface_pose_ids=[0, 1, 2],
        interface_variable_count=3,
        block_dim=1,
        omitted_force_blocks=omitted_blocks,
        full_schur=full_schur,
        full_rhs=full_rhs,
        base_basis=base_basis,
        min_value_per_byte=0.0,
        max_promoted_pose_blocks=1,
        payload_bytes_per_coordinate=8,
    )

    assert report["model"] == "pose_block_basin_certificate_promotion_basis"
    assert report["promoted_pose_ids"] == [0]
    assert report["accepted_promotions"][0]["omitted_force_delta"] < 0.0
    rejected = {item["pose_id"]: item["reason"] for item in report["rejected_promotions"]}
    assert rejected[1] == "omitted_force_worse"


def test_local_schur_residual_packets_sum_to_exact_omitted_force():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.array([
                [2.0, 0.0, 1.0],
                [0.0, 3.0, 2.0],
                [1.0, 2.0, 4.0],
            ], dtype=float),
            "gradient": np.array([1.0, 2.0, 3.0], dtype=float),
            "private_indices": np.array([2], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.array([
                [1.0, 0.2, 0.0],
                [0.2, 1.5, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=float),
            "gradient": np.array([-0.5, 0.4, 0.0], dtype=float),
            "private_indices": np.array([], dtype=int),
        },
    ]
    interface_indices = np.array([0, 1], dtype=int)
    interface_state = np.array([0.25, -0.5], dtype=float)
    full_schur, full_rhs, _ = dci.sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=3,
    )

    report = skeleton.local_schur_residual_packets(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_state=interface_state,
        interface_pose_ids=[10, 11],
        block_dim=1,
        variable_count=3,
    )

    expected = full_rhs - full_schur @ interface_state
    np.testing.assert_allclose(report["aggregate_residual"], expected, atol=1e-12)
    np.testing.assert_allclose(
        sum((packet["residual"] for packet in report["packets"]),
            np.zeros_like(expected)),
        expected,
        atol=1e-12,
    )
    assert report["model"] == "local_schur_residual_packets"
    assert report["packet_count"] == 2
    assert report["payload_bytes"] == 32
    assert [block["pose_id"] for block in report["aggregate_pose_blocks"]] == [10, 11]


def test_sparse_local_schur_residual_packets_keep_exact_nonzero_blocks_only():
    local_systems = [
        {
            "robot": 0,
            "hessian": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),
            "gradient": np.array([1.0, 0.0], dtype=float),
            "private_indices": np.array([], dtype=int),
        },
        {
            "robot": 1,
            "hessian": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float),
            "gradient": np.array([0.0, 2.0], dtype=float),
            "private_indices": np.array([], dtype=int),
        },
    ]
    interface_indices = np.array([0, 1], dtype=int)
    interface_state = np.zeros(2, dtype=float)
    dense = skeleton.local_schur_residual_packets(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_state=interface_state,
        interface_pose_ids=[10, 11],
        block_dim=1,
        variable_count=2,
    )

    sparse = skeleton.local_schur_sparse_residual_packets(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_state=interface_state,
        interface_pose_ids=[10, 11],
        block_dim=1,
        variable_count=2,
        block_norm_threshold=0.0,
        payload_index_bytes=4,
    )

    np.testing.assert_allclose(
        sparse["aggregate_residual"], dense["aggregate_residual"], atol=1e-12)
    assert sparse["sparse_residual_error_norm"] <= 1e-12
    assert sparse["selected_block_count"] == 2
    assert sparse["dropped_block_count"] == 2
    assert sparse["payload_bytes"] == 24
    assert sparse["payload_bytes"] < dense["payload_bytes"]


def test_sparse_local_schur_residual_packets_report_dropped_error_bound():
    local_systems = [{
        "robot": 0,
        "hessian": np.eye(2, dtype=float),
        "gradient": np.array([1.0, 2.0], dtype=float),
        "private_indices": np.array([], dtype=int),
    }]

    sparse = skeleton.local_schur_sparse_residual_packets(
        local_systems=local_systems,
        interface_indices=np.array([0, 1], dtype=int),
        interface_state=np.zeros(2, dtype=float),
        interface_pose_ids=[10, 11],
        block_dim=1,
        variable_count=2,
        block_norm_threshold=1.5,
        payload_index_bytes=4,
    )

    np.testing.assert_allclose(sparse["aggregate_residual"], [0.0, 2.0])
    assert sparse["selected_block_count"] == 1
    assert sparse["dropped_block_count"] == 1
    assert sparse["dropped_residual_norm_bound"] == 1.0
    assert sparse["sparse_residual_error_norm"] == 1.0


def test_sparse_basin_gate_decision_uses_error_bounds():
    before = {
        "aggregate_residual_norm": 10.0,
        "dropped_residual_norm_bound": 0.0,
    }
    after_safe = {
        "aggregate_residual_norm": 8.0,
        "dropped_residual_norm_bound": 1.0,
    }
    after_uncertain = {
        "aggregate_residual_norm": 8.0,
        "dropped_residual_norm_bound": 3.0,
    }

    safe = skeleton.sparse_basin_gate_decision(before, after_safe)
    uncertain = skeleton.sparse_basin_gate_decision(before, after_uncertain)

    assert safe["decision"] == "safe_accept"
    assert safe["after_upper_bound"] <= safe["before_lower_bound"]
    assert uncertain["decision"] == "uncertain"
    assert uncertain["after_upper_bound"] > uncertain["before_lower_bound"]


def test_streaming_sparse_basin_gate_stops_after_first_certifying_block():
    before_exact = {
        "interface_variable_count": 2,
        "packets": [{
            "robot": 0,
            "blocks": [
                {
                    "pose_id": 10,
                    "block_index": 0,
                    "residual_block": np.array([10.0], dtype=float),
                    "block_norm": 10.0,
                },
                {
                    "pose_id": 11,
                    "block_index": 1,
                    "residual_block": np.array([1.0], dtype=float),
                    "block_norm": 1.0,
                },
            ],
        }],
    }
    after_exact = {
        "interface_variable_count": 2,
        "packets": [{
            "robot": 0,
            "blocks": [{
                "pose_id": 10,
                "block_index": 0,
                "residual_block": np.array([1.0], dtype=float),
                "block_norm": 1.0,
            }],
        }],
    }

    report = skeleton.streaming_sparse_basin_gate_decision(
        before_exact,
        after_exact,
        block_dim=1,
        payload_bytes_per_coordinate=8,
        payload_index_bytes=4,
    )

    assert report["decision"]["decision"] == "safe_accept"
    assert report["selected_block_count"] == 1
    assert report["payload_bytes"] == 12
    assert report["selected_blocks"][0]["side"] == "before"
    assert report["selected_blocks"][0]["pose_id"] == 10


def test_topology_round_streaming_sends_one_local_block_per_robot_round():
    before_exact = {
        "interface_variable_count": 2,
        "packets": [
            {
                "robot": 0,
                "blocks": [{
                    "pose_id": 10,
                    "block_index": 0,
                    "residual_block": np.array([10.0], dtype=float),
                    "block_norm": 10.0,
                }],
            },
            {
                "robot": 1,
                "blocks": [{
                    "pose_id": 11,
                    "block_index": 1,
                    "residual_block": np.array([0.1], dtype=float),
                    "block_norm": 0.1,
                }],
            },
        ],
    }
    after_exact = {
        "interface_variable_count": 2,
        "packets": [{
            "robot": 0,
            "blocks": [{
                "pose_id": 10,
                "block_index": 0,
                "residual_block": np.array([1.0], dtype=float),
                "block_norm": 1.0,
            }],
        }],
    }

    report = skeleton.topology_round_streaming_sparse_basin_gate_decision(
        before_exact,
        after_exact,
        block_dim=1,
        payload_bytes_per_coordinate=8,
        payload_index_bytes=4,
    )

    assert report["decision"]["decision"] == "safe_accept"
    assert report["round_count"] == 1
    assert report["selected_block_count"] == 2
    assert report["payload_bytes"] == 24
    assert {block["robot"] for block in report["selected_blocks"]} == {0, 1}


def test_receiver_pulled_streaming_requests_only_certifying_queue_head():
    before_exact = {
        "interface_variable_count": 2,
        "packets": [
            {
                "robot": 0,
                "blocks": [
                    {
                        "pose_id": 10,
                        "block_index": 0,
                        "residual_block": np.array([10.0], dtype=float),
                        "block_norm": 10.0,
                    },
                    {
                        "pose_id": 12,
                        "block_index": 1,
                        "residual_block": np.array([1.0], dtype=float),
                        "block_norm": 1.0,
                    },
                ],
            },
            {
                "robot": 1,
                "blocks": [{
                    "pose_id": 11,
                    "block_index": 1,
                    "residual_block": np.array([0.1], dtype=float),
                    "block_norm": 0.1,
                }],
            },
        ],
    }
    after_exact = {
        "interface_variable_count": 2,
        "packets": [{
            "robot": 0,
            "blocks": [{
                "pose_id": 10,
                "block_index": 0,
                "residual_block": np.array([1.0], dtype=float),
                "block_norm": 1.0,
            }],
        }],
    }

    pulled = skeleton.receiver_pulled_streaming_sparse_basin_gate_decision(
        before_exact,
        after_exact,
        block_dim=1,
        payload_bytes_per_coordinate=8,
        payload_index_bytes=4,
    )
    round_based = skeleton.topology_round_streaming_sparse_basin_gate_decision(
        before_exact,
        after_exact,
        block_dim=1,
        payload_bytes_per_coordinate=8,
        payload_index_bytes=4,
    )

    assert pulled["decision"]["decision"] == "safe_accept"
    assert pulled["pull_count"] == 1
    assert pulled["selected_block_count"] == 1
    assert pulled["payload_bytes"] == 12
    assert pulled["selected_blocks"][0]["robot"] == 0
    assert pulled["payload_bytes"] < round_based["payload_bytes"]


def test_receiver_pulled_top_residual_block_certifies_before_full_round():
    sparse_report = {
        "interface_variable_count": 2,
        "full_aggregate_residual": np.array([10.0, 0.2], dtype=float),
        "packets": [
            {
                "robot": 0,
                "blocks": [{
                    "pose_id": 10,
                    "block_index": 0,
                    "residual_block": np.array([10.0], dtype=float),
                    "block_norm": 10.0,
                }],
                "dropped_blocks": [],
            },
            {
                "robot": 1,
                "blocks": [{
                    "pose_id": 11,
                    "block_index": 1,
                    "residual_block": np.array([0.2], dtype=float),
                    "block_norm": 0.2,
                }],
                "dropped_blocks": [],
            },
        ],
    }

    decision = skeleton.receiver_pulled_top_residual_block_decision(
        sparse_report,
        block_dim=1,
        payload_bytes_per_coordinate=8,
        payload_index_bytes=4,
    )

    assert decision["decision"]["decision"] == "safe_top"
    assert decision["decision"]["top_pose_id"] == 10
    assert decision["exact_top_pose_id"] == 10
    assert decision["pull_count"] == 1
    assert decision["payload_bytes"] == 12


def test_graph_stage_value_per_byte_promotion_reduces_omitted_force():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    base = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
    )
    promoted = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span_plus_value_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
    )

    assert promoted["basis_mode"] == "rank_aware_robot_cycle_span_plus_value_promote"
    assert promoted["promotion"]["promoted_pose_ids"]
    assert promoted["promotion"]["accepted_promotions"]
    assert promoted["solve"]["omitted_force_norm"] < base["solve"]["omitted_force_norm"]
    assert promoted["solve"]["reduced_singular"] is False


def test_graph_stage_value_promotion_reports_relative_certificate():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    promoted = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span_plus_value_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        promotion_min_relative_value_fraction=0.5,
    )

    assert promoted["promotion"]["min_relative_value_fraction"] == 0.5
    assert promoted["promotion"]["first_value_per_byte"] is not None
    assert promoted["promotion"]["final_effective_value_threshold"] >= 0.0


def test_graph_stage_basin_certificate_promotion_never_worsens_omitted_force():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    base = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
    )
    promoted = skeleton.run_graph_stage_primal_skeleton_basis_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
    )

    assert promoted["basis_mode"] == "rank_aware_robot_cycle_span_plus_basin_promote"
    assert promoted["promotion"]["accepted_promotions"]
    assert promoted["solve"]["omitted_force_norm"] <= base["solve"]["omitted_force_norm"] + 1e-12
    assert promoted["solve"]["reduced_singular"] is False


def test_graph_stage_residual_packets_match_skeleton_omitted_force():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_residual_packet_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
    )

    assert report["model"] == "graph_stage_residual_packet_diagnostic"
    assert report["residual_packet_error_norm"] <= 1e-12
    assert report["packet_report"]["packet_count"] == 3
    assert report["packet_report"]["payload_bytes"] > 0


def test_graph_stage_sparse_residual_packets_report_payload_and_error():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_sparse_residual_packet_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_sparse_residual_packet_diagnostic"
    assert report["sparse_packet_report"]["payload_bytes"] > 0
    assert report["sparse_packet_report"]["payload_bytes"] <= report["full_packet_payload_bytes"]
    assert report["sparse_residual_error_norm"] <= 1e-12


def test_graph_stage_sparse_basin_decision_certifies_exact_sparse_accept():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_sparse_basin_decision_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        before_basis_mode="rank_aware_robot_cycle_span",
        after_basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_sparse_basin_decision_diagnostic"
    assert report["decision"]["decision"] == "safe_accept"
    assert report["exact_decision"] == "accept"
    assert report["before_sparse"]["sparse_residual_error_norm"] <= 1e-12
    assert report["after_sparse"]["sparse_residual_error_norm"] <= 1e-12


def test_graph_stage_adaptive_sparse_basin_decision_picks_min_safe_top_k():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_adaptive_sparse_basin_decision_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        before_basis_mode="rank_aware_robot_cycle_span",
        after_basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_adaptive_sparse_basin_decision_diagnostic"
    assert report["selected_top_k_blocks_per_robot"] == 2
    assert report["selected_candidate"]["decision"]["decision"] == "safe_accept"
    assert report["selected_candidate"]["total_sparse_payload_bytes"] == 80
    assert [candidate["decision"]["decision"] for candidate in report["candidates"][:2]] == [
        "uncertain",
        "uncertain",
    ]


def test_graph_stage_streaming_sparse_basin_decision_stops_before_top_k():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_streaming_sparse_basin_decision_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        before_basis_mode="rank_aware_robot_cycle_span",
        after_basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_streaming_sparse_basin_decision_diagnostic"
    assert report["streaming_decision"]["decision"]["decision"] == "safe_accept"
    assert report["streaming_decision"]["payload_bytes"] == 40
    assert report["streaming_decision"]["payload_bytes"] < report["adaptive_top_k_payload_bytes"]
    assert report["streaming_decision"]["selected_block_count"] == 2


def test_graph_stage_topology_round_streaming_sparse_basin_decision_is_safe():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_topology_round_streaming_sparse_basin_decision_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        before_basis_mode="rank_aware_robot_cycle_span",
        after_basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_topology_round_streaming_sparse_basin_decision_diagnostic"
    assert report["topology_streaming_decision"]["decision"]["decision"] == "safe_accept"
    assert report["topology_streaming_decision"]["payload_bytes"] >= report["global_streaming_payload_bytes"]
    assert report["topology_streaming_decision"]["payload_bytes"] < report["total_full_payload_bytes"]


def test_graph_stage_receiver_pulled_streaming_matches_global_payload():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_streaming_sparse_basin_decision_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        before_basis_mode="rank_aware_robot_cycle_span",
        after_basis_mode="rank_aware_robot_cycle_span_plus_basin_promote",
        promoted_pose_block_count=1,
        promotion_min_value_per_byte=0.0,
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_receiver_pulled_streaming_sparse_basin_decision_diagnostic"
    assert report["receiver_pulled_decision"]["decision"]["decision"] == "safe_accept"
    assert report["receiver_pulled_payload_bytes"] == report["global_streaming_payload_bytes"]
    assert report["receiver_pulled_payload_bytes"] <= report["topology_streaming_payload_bytes"]


def test_graph_stage_receiver_pulled_top_residual_enrichment_matches_centralized_top():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_top_residual_enrichment_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="cycle_coordinate",
        sparse_block_norm_threshold=0.0,
    )

    assert report["model"] == "graph_stage_receiver_pulled_top_residual_enrichment_diagnostic"
    assert report["top_residual_decision"]["decision"]["decision"] == "safe_top"
    assert report["packet_selected_pose_id"] == report["centralized_top_pose_id"]
    assert report["after_omitted_force_norm"] < report["before_omitted_force_norm"]
    assert report["top_residual_payload_bytes"] < report["full_packet_payload_bytes"]


def test_graph_stage_receiver_pulled_top_residual_iterations_reach_full_interface():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="cycle_coordinate",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
    )

    assert report["model"] == "graph_stage_receiver_pulled_top_residual_enrichment_iterations"
    assert report["converged"] is True
    assert report["stopped_reason"] == "omitted_force_tolerance"
    assert report["promoted_pose_ids"] == [3, 2, 1]
    assert report["final_omitted_force_norm"] < 1e-12
    assert report["history"][-1]["after_omitted_force_norm"] < 1e-12
    assert [item["packet_selected_pose_id"] for item in report["history"]] == [3, 2, 1]
    assert report["total_top_residual_payload_bytes"] == 140


def test_graph_stage_enrichment_can_use_basin_certificate_policy():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        promotion_policy="basin_certificate",
    )

    assert report["promotion_policy"] == "basin_certificate"
    assert report["converged"] is True
    assert report["final_omitted_force_norm"] < 1e-12
    assert report["history"]
    for item in report["history"]:
        assert item["promotion_decision"]["model"] == "pose_block_basin_certificate_promotion_basis"
        assert item["after_omitted_force_norm"] <= item["before_omitted_force_norm"] + 1e-12


def test_graph_stage_enrichment_can_use_topk_batch_certificate_policy():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        promotion_policy="topk_basin_certificate",
        promotion_candidate_top_k=2,
    )

    assert report["promotion_policy"] == "topk_basin_certificate"
    assert report["history"]
    assert len(report["history"][0]["packet_selected_pose_ids"]) == 2
    for item in report["history"]:
        assert item["promotion_decision"]["model"] == "pose_block_topk_basin_certificate_promotion_basis"
        assert item["after_omitted_force_norm"] <= item["before_omitted_force_norm"] + 1e-12
    assert report["final_omitted_force_norm"] < 1e-12


def test_graph_stage_receiver_pulled_top_residual_iterations_stop_on_tie():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
        max_iterations=3,
        stop_omitted_force_norm=1e-12,
    )

    assert report["converged"] is False
    assert report["stopped_reason"] == "uncertain_top_residual"
    assert report["promoted_pose_ids"] == []
    assert report["history"][0]["top_residual_decision"]["decision"]["decision"] == "uncertain"
    assert report["final_omitted_force_norm"] == report["initial_omitted_force_norm"]


def test_graph_stage_receiver_pulled_top_residual_iterations_promote_safe_tie_set():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
        max_iterations=2,
        stop_omitted_force_norm=1e-12,
        allow_top_set_promotion=True,
    )

    assert report["converged"] is True
    assert report["stopped_reason"] == "omitted_force_tolerance"
    assert report["history"][0]["top_residual_decision"]["decision"]["decision"] == "safe_top_set"
    assert report["history"][0]["packet_selected_pose_ids"] == [1, 2]
    assert report["promoted_pose_ids"] == [1, 2]
    assert report["final_omitted_force_norm"] < 1e-12


def test_two_stage_receiver_pulled_top_residual_enrichment_reports_both_stages():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_two_stage_receiver_pulled_top_residual_enrichment_summary(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        rotations=rotations,
        basis_mode="cycle_coordinate",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        allow_top_set_promotion=True,
    )

    assert report["model"] == "two_stage_receiver_pulled_top_residual_enrichment_summary"
    assert report["stage_order"] == ["rotation", "translation"]
    assert report["all_stages_converged"] is True
    assert report["rotation_converged"] is True
    assert report["translation_converged"] is True
    assert report["rotation_final_omitted_force_norm"] < 1e-12
    assert report["translation_final_omitted_force_norm"] < 1e-12
    assert report["stages"]["rotation"]["promoted_pose_ids"] == [1, 2, 3]
    assert report["stages"]["translation"]["promoted_pose_ids"] == [3, 1, 2]
    assert report["total_top_residual_payload_bytes"] == (
        report["stages"]["rotation"]["total_top_residual_payload_bytes"]
        + report["stages"]["translation"]["total_top_residual_payload_bytes"]
    )
    assert report["total_full_packet_payload_bytes"] == (
        report["stages"]["rotation"]["total_full_packet_payload_bytes"]
        + report["stages"]["translation"]["total_full_packet_payload_bytes"]
    )
    assert report["total_sparse_packet_payload_bytes"] == (
        report["stages"]["rotation"]["total_sparse_packet_payload_bytes"]
        + report["stages"]["translation"]["total_sparse_packet_payload_bytes"]
    )


def test_two_stage_enrichment_propagates_basin_certificate_policy():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}

    report = skeleton.run_two_stage_receiver_pulled_top_residual_enrichment_summary(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        rotations=rotations,
        basis_mode="rank_aware_robot_cycle_span",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        promotion_policy="basin_certificate",
    )

    assert report["promotion_policy"] == "basin_certificate"
    assert report["all_stages_converged"] is True
    assert report["stages"]["rotation"]["promotion_policy"] == "basin_certificate"
    assert report["stages"]["translation"]["promotion_policy"] == "basin_certificate"
    assert report["total_promotion_payload_bytes"] > 0


def test_backsubstitute_full_solution_from_interface_matches_reference():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    rotations = {pose_id: np.eye(2) for pose_id in pose_ids}
    local_systems, interface_indices, build_stats = (
        dci.build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=2,
            rotations=rotations,
            weighted=False,
            cost_mode="dpgo",
            anchor_pose=0,
        )
    )
    reference_solution, reference_stats = dci.solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=build_stats["variable_count"],
    )

    recovered = skeleton.backsubstitute_full_solution_from_interface(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_solution=reference_solution[interface_indices],
        variable_count=build_stats["variable_count"],
    )

    assert recovered["model"] == "full_solution_from_interface_backsubstitution"
    assert recovered["private_backsubstitution_count"] == (
        reference_stats["private_backsubstitution_count"]
    )
    assert recovered["unassigned_variable_count"] == 0
    assert recovered["final_normal_residual"] < 1e-10
    assert np.allclose(recovered["solution"], reference_solution, atol=1e-10)


def test_two_stage_receiver_pulled_top_residual_handoff_matches_centralized_cost():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    poses, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        allow_top_set_promotion=True,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    assert report["model"] == "two_stage_receiver_pulled_top_residual_handoff_diagnostic"
    assert report["stage_summary"]["all_stages_converged"] is True
    assert report["handoff_cost"]["all_edges_evaluated"] is True
    assert report["centralized_handoff_cost"]["all_edges_evaluated"] is True
    assert report["centralized_handoff_cost"]["total_cost"] > 0.0
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert report["rotation_backsubstitution"]["final_normal_residual"] < 1e-10
    assert report["translation_backsubstitution"]["final_normal_residual"] < 1e-10
    assert set(poses) == set(pose_ids)


def test_two_stage_handoff_can_use_matrix_free_interface_pcg():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    poses, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="pcg",
        pcg_iterations=50,
        pcg_tolerance=1e-12,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    assert report["interface_solver"] == "pcg"
    assert report["stage_summary"]["all_stages_converged"] is True
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert report["stage_summary"]["stages"]["rotation"]["solver_stats"]["materializes_dense_schur"] is False
    assert report["stage_summary"]["stages"]["translation"]["solver_stats"]["materializes_dense_schur"] is False
    assert set(poses) == set(pose_ids)


def test_matrix_free_pcg_handoff_reuses_solver_private_recovery():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="pcg",
        pcg_iterations=50,
        pcg_tolerance=1e-12,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    assert report["rotation_backsubstitution"]["model"] == "pcg_full_solution_reuse"
    assert report["translation_backsubstitution"]["model"] == "pcg_full_solution_reuse"
    assert report["rotation_backsubstitution"]["uses_dense_backsubstitution"] is False
    assert report["translation_backsubstitution"]["uses_dense_backsubstitution"] is False
    assert report["rotation_backsubstitution"]["final_normal_residual"] < 1e-10
    assert report["translation_backsubstitution"]["final_normal_residual"] < 1e-10


def test_two_stage_handoff_can_use_reduction_free_fixed_step_interface_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    poses, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="fixed_step",
        pcg_iterations=10,
        pcg_tolerance=1e-12,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == "fixed_step"
    assert set(poses) == set(pose_ids)
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0
    assert rotation_stats["global_scalar_reduction_payload_bytes"] == 0
    assert translation_stats["global_scalar_reduction_payload_bytes"] == 0


def test_two_stage_handoff_can_use_coarse_fixed_step_interface_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="coarse_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == "coarse_fixed_step"
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["coarse_initial_guess_used"] is True
    assert translation_stats["coarse_initial_guess_used"] is True
    assert rotation_stats["coarse_basis_rank"] > 0
    assert translation_stats["coarse_basis_rank"] > 0
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_residual_krylov_fixed_step_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="residual_krylov_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == "residual_krylov_fixed_step"
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert translation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert rotation_stats["adaptive_coarse_rank"] == 2
    assert translation_stats["adaptive_coarse_rank"] == 2
    assert rotation_stats["coarse_initial_guess_used"] is True
    assert translation_stats["coarse_initial_guess_used"] is True
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_deterministic_ritz_fixed_step_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="deterministic_ritz_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=2,
        adaptive_coarse_probe_count=2,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == "deterministic_ritz_fixed_step"
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["adaptive_coarse_basis"] == "deterministic_ritz"
    assert translation_stats["adaptive_coarse_basis"] == "deterministic_ritz"
    assert rotation_stats["adaptive_coarse_probe_count"] == 2
    assert translation_stats["adaptive_coarse_probe_count"] == 2
    assert rotation_stats["adaptive_coarse_ritz_selected_rank"] == 2
    assert translation_stats["adaptive_coarse_ritz_selected_rank"] == 2
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_separator_component_fixed_step_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == "separator_component_fixed_step"
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["coarse_basis_mode"] == "separator_component_coordinate"
    assert translation_stats["coarse_basis_mode"] == "separator_component_coordinate"
    assert rotation_stats["coarse_initial_guess_used"] is True
    assert translation_stats["coarse_initial_guess_used"] is True
    assert rotation_stats["coarse_basis_rank"] > 0
    assert translation_stats["coarse_basis_rank"] > 0
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_separator_component_residual_krylov_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == (
        "separator_component_residual_krylov_fixed_step"
    )
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["coarse_basis_mode"] == "separator_component_coordinate"
    assert translation_stats["coarse_basis_mode"] == "separator_component_coordinate"
    assert rotation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert translation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert rotation_stats["coarse_initial_guess_used"] is True
    assert translation_stats["coarse_initial_guess_used"] is True
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_separator_component_geneo_residual_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_geneo_residual_krylov_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        component_modes_per_component=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == (
        "separator_component_geneo_residual_krylov_fixed_step"
    )
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["coarse_basis_mode"] == "separator_component_geneo"
    assert translation_stats["coarse_basis_mode"] == "separator_component_geneo"
    assert rotation_stats["component_modes_per_component"] == 1
    assert translation_stats["component_modes_per_component"] == 1
    assert rotation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert translation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_separator_component_merit_residual_solver():
    graph_edges = [edge(0, 1)]
    pose_ids = [0, 1]
    robot_of = {0: 0, 1: 1}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_merit_residual_krylov_fixed_step",
        pcg_iterations=0,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        component_max_components=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert report["interface_solver"] == (
        "separator_component_merit_residual_krylov_fixed_step"
    )
    assert abs(report["handoff_cost_delta_to_centralized"]) < 1e-10
    assert rotation_stats["coarse_basis_mode"] == "separator_component_merit"
    assert translation_stats["coarse_basis_mode"] == "separator_component_merit"
    assert rotation_stats["component_max_components"] == 1
    assert translation_stats["component_max_components"] == 1
    assert rotation_stats["component_merit_selected_component_count"] == 1
    assert translation_stats["component_merit_selected_component_count"] == 1
    assert rotation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert translation_stats["adaptive_coarse_basis"] == "residual_krylov"
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_chebyshev_component_residual_solver():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        fixed_step_acceleration="chebyshev",
        chebyshev_lambda_min=0.1,
        chebyshev_lambda_max=2.0,
        pcg_iterations=2,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert rotation_stats["fixed_step_acceleration"] == "chebyshev"
    assert translation_stats["fixed_step_acceleration"] == "chebyshev"
    assert rotation_stats["chebyshev_lambda_min"] == 0.1
    assert translation_stats["chebyshev_lambda_max"] == 2.0
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_auto_gershgorin_chebyshev_bounds():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        fixed_step_acceleration="chebyshev_auto_gershgorin",
        pcg_iterations=2,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert rotation_stats["fixed_step_acceleration"] == "chebyshev_auto_gershgorin"
    assert translation_stats["fixed_step_acceleration"] == "chebyshev_auto_gershgorin"
    assert rotation_stats["chebyshev_bound_source"] == "normalized_gershgorin"
    assert translation_stats["chebyshev_bound_source"] == "normalized_gershgorin"
    assert rotation_stats["chebyshev_lambda_max"] > 0.0
    assert translation_stats["chebyshev_lambda_max"] > 0.0
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_graph_normalized_chebyshev_bounds():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        fixed_step_acceleration="chebyshev_graph_normalized",
        pcg_iterations=2,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert rotation_stats["fixed_step_acceleration"] == "chebyshev_graph_normalized"
    assert translation_stats["fixed_step_acceleration"] == "chebyshev_graph_normalized"
    assert rotation_stats["chebyshev_bound_source"] == "graph_normalized_heuristic"
    assert translation_stats["chebyshev_bound_source"] == "graph_normalized_heuristic"
    assert rotation_stats["chebyshev_bound_setup_matvec_count"] == 0
    assert translation_stats["chebyshev_bound_setup_matvec_count"] == 0
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_ritz_probe_chebyshev_bounds():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        fixed_step_acceleration="chebyshev_ritz_probe",
        chebyshev_ritz_probe_iterations=3,
        chebyshev_ritz_safety_factor=1.01,
        pcg_iterations=2,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert rotation_stats["fixed_step_acceleration"] == "chebyshev_ritz_probe"
    assert translation_stats["fixed_step_acceleration"] == "chebyshev_ritz_probe"
    assert rotation_stats["chebyshev_bound_source"] == "matrix_free_ritz_probe"
    assert translation_stats["chebyshev_bound_source"] == "matrix_free_ritz_probe"
    assert rotation_stats["chebyshev_ritz_safety_factor"] == 1.01
    assert translation_stats["chebyshev_ritz_safety_factor"] == 1.01
    assert rotation_stats["chebyshev_bound_setup_matvec_count"] > 0
    assert translation_stats["chebyshev_bound_setup_matvec_count"] > 0
    assert rotation_stats["global_reduction_count"] > 0
    assert translation_stats["global_reduction_count"] > 0


def test_two_stage_handoff_can_use_chebyshev_safety_monitor():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        fixed_step_acceleration="chebyshev_graph_normalized",
        chebyshev_safety_monitor="residual_growth",
        chebyshev_safety_growth_factor=1.5,
        pcg_iterations=2,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert rotation_stats["chebyshev_safety_monitor"] == "residual_growth"
    assert translation_stats["chebyshev_safety_monitor"] == "residual_growth"
    assert rotation_stats["chebyshev_safety_growth_factor"] == 1.5
    assert translation_stats["chebyshev_safety_growth_factor"] == 1.5
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_two_stage_handoff_can_use_local_envelope_safety_monitor():
    graph_edges = [
        edge(0, 1),
        edge(1, 2),
    ]
    pose_ids = [0, 1, 2]
    robot_of = {0: 0, 1: 1, 2: 2}

    _, report = skeleton.run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="separator_component_residual_krylov_fixed_step",
        fixed_step_acceleration="chebyshev_graph_normalized",
        chebyshev_safety_monitor="local_residual_envelope",
        chebyshev_safety_growth_factor=1.5,
        pcg_iterations=2,
        pcg_tolerance=1e-12,
        adaptive_coarse_rank=1,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    rotation_stats = report["stage_summary"]["stages"]["rotation"]["solver_stats"]
    translation_stats = report["stage_summary"]["stages"]["translation"]["solver_stats"]

    assert rotation_stats["chebyshev_safety_monitor"] == "local_residual_envelope"
    assert translation_stats["chebyshev_safety_monitor"] == "local_residual_envelope"
    assert rotation_stats["chebyshev_safety_metric"] == "local_residual_envelope"
    assert translation_stats["chebyshev_safety_metric"] == "local_residual_envelope"
    assert rotation_stats["chebyshev_safety_growth_factor"] == 1.5
    assert translation_stats["chebyshev_safety_growth_factor"] == 1.5
    assert rotation_stats["global_reduction_count"] == 0
    assert translation_stats["global_reduction_count"] == 0


def test_packet_handoff_sweep_row_records_cost_delta_and_payload():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    row = skeleton.packet_handoff_sweep_row(
        dataset="toy-cycle",
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        allow_top_set_promotion=True,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    assert row["model"] == "packet_handoff_sweep_row"
    assert row["status"] == "ok"
    assert row["dataset"] == "toy-cycle"
    assert row["pose_count"] == 4
    assert row["edge_count"] == 4
    assert row["all_stages_converged"] is True
    assert row["handoff_cost_delta_to_centralized_abs"] < 1e-10
    assert row["rotation_promoted_pose_count"] == 3
    assert row["translation_promoted_pose_count"] == 3
    assert row["rotation_nonmonotone_iteration_count"] == 0
    assert row["translation_nonmonotone_iteration_count"] == 0
    assert row["rotation_final_basis_column_count"] == row["rotation_full_interface_variable_count"]
    assert row["translation_final_basis_column_count"] == row["translation_full_interface_variable_count"]
    assert row["total_top_residual_payload_bytes"] > 0
    assert row["total_top_residual_payload_bytes"] < row["total_full_packet_payload_bytes"]
    assert row["report"]["model"] == "two_stage_receiver_pulled_top_residual_handoff_diagnostic"


def test_packet_handoff_sweep_row_can_use_basin_certificate_policy():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    row = skeleton.packet_handoff_sweep_row(
        dataset="toy-cycle",
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="rank_aware_robot_cycle_span",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        promotion_policy="basin_certificate",
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    assert row["status"] == "ok"
    assert row["promotion_policy"] == "basin_certificate"
    assert row["all_stages_converged"] is True
    assert row["handoff_cost_delta_to_centralized_abs"] < 1e-10
    assert row["total_promotion_payload_bytes"] > 0
    assert row["report"]["promotion_policy"] == "basin_certificate"


def test_packet_handoff_sweep_row_can_use_matrix_free_pcg_solver():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    row = skeleton.packet_handoff_sweep_row(
        dataset="toy-cycle",
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        interface_solver="pcg",
        pcg_iterations=50,
        pcg_tolerance=1e-12,
        weighted=False,
        cost_mode="dpgo",
        anchor_pose=0,
    )

    assert row["status"] == "ok"
    assert row["interface_solver"] == "pcg"
    assert row["all_stages_converged"] is True
    assert row["handoff_cost_delta_to_centralized_abs"] < 1e-10
    assert row["total_pcg_comm_mb"] > 0.0
    assert row["report"]["interface_solver"] == "pcg"


def test_packet_handoff_sweep_row_can_skip_preflight_before_dense_solve():
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}

    row = skeleton.packet_handoff_sweep_row(
        dataset="toy-cycle",
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        dim=2,
        basis_mode="cycle_coordinate",
        max_interface_variables=1,
        weighted=False,
        cost_mode="dpgo",
    )

    assert row["status"] == "skipped_preflight"
    assert row["skip_reason"] == "interface_variable_count_exceeds_limit"
    assert row["estimated_rotation_interface_variables"] > 1
    assert row["estimated_translation_interface_variables"] > 1
    assert "report" not in row


def test_write_packet_handoff_sweep_outputs_csv_and_json(tmp_path):
    graph_edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(1, 3)]
    pose_ids = [0, 1, 2, 3]
    robot_of = {0: 0, 1: 1, 2: 1, 3: 2}
    output_dir = tmp_path / "packet_sweep"

    report = skeleton.write_packet_handoff_sweep(
        output_dir=output_dir,
        dataset_specs=[{
            "dataset": "toy-cycle",
            "graph_edges": graph_edges,
            "pose_ids": pose_ids,
            "robot_of": robot_of,
            "anchor_pose": 0,
        }],
        dim=2,
        basis_mode="cycle_coordinate",
        max_iterations=4,
        stop_omitted_force_norm=1e-12,
        allow_top_set_promotion=True,
        weighted=False,
        cost_mode="dpgo",
    )

    summary_csv = output_dir / "packet_handoff_sweep_summary.csv"
    report_json = output_dir / "packet_handoff_sweep_report.json"
    assert summary_csv.exists()
    assert report_json.exists()
    assert report["model"] == "packet_handoff_sweep"
    assert report["rows"][0]["status"] == "ok"
    assert report["rows"][0]["handoff_cost_delta_to_centralized_abs"] < 1e-10
    csv_text = summary_csv.read_text(encoding="utf-8")
    assert "handoff_cost_delta_to_centralized_abs" in csv_text
    json_text = report_json.read_text(encoding="utf-8")
    assert "\"packet_handoff_sweep\"" in json_text
