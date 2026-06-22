#!/usr/bin/env python3
"""Split a single g2o file into distributed-mapper robot subgraphs.

The split follows the same contiguous pose assignment used in dpgo_ICRA:
robot 0 gets the first chunk of poses, robot 1 the next chunk, and the last
robot gets the remainder. By default, vertex initial values are replaced by a
centralized chordal initialization computed on the original graph before
splitting.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def symbol_key(robot_idx: int, local_idx: int) -> int:
    return (ord("a") + robot_idx) << 56 | local_idx


def format_key(robot_idx: int, local_idx: int) -> str:
    return str(symbol_key(robot_idx, local_idx))


def yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quat_to_rot(qx: float, qy: float, qz: float, qw: float):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def rot_to_quat(rot):
    trace = float(np.trace(rot))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    return qx, qy, qz, qw


def project_to_rotation(rot):
    u, _, vt = np.linalg.svd(rot)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vt
    return projected


def compose_se2(pose, rel):
    x, y, theta = pose
    dx, dy, dtheta = rel
    c = math.cos(theta)
    s = math.sin(theta)
    return (
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        theta + dtheta,
    )


def se2_vertex_to_pose3(tokens):
    x = float(tokens[2])
    y = float(tokens[3])
    theta = float(tokens[4])
    qx, qy, qz, qw = yaw_to_quat(theta)
    return x, y, 0.0, qx, qy, qz, qw


def se3_vertex_to_pose3(tokens):
    x = float(tokens[2])
    y = float(tokens[3])
    z = float(tokens[4])
    qx = float(tokens[5])
    qy = float(tokens[6])
    qz = float(tokens[7])
    qw = float(tokens[8])
    return x, y, z, qx, qy, qz, qw


def se2_edge_to_se3(tokens):
    dx = float(tokens[3])
    dy = float(tokens[4])
    dtheta = float(tokens[5])
    i11 = float(tokens[6])
    i12 = float(tokens[7])
    i13 = float(tokens[8])
    i22 = float(tokens[9])
    i23 = float(tokens[10])
    i33 = float(tokens[11])
    qx, qy, qz, qw = yaw_to_quat(dtheta)
    pose = [dx, dy, 0.0, qx, qy, qz, qw]
    strong = 1e12
    info = [
        i11, i12, 0.0, i13, 0.0, 0.0,
        i22, 0.0, i23, 0.0, 0.0,
        strong, 0.0, 0.0, 0.0,
        strong, 0.0, 0.0,
        strong, 0.0,
        i33,
    ]
    return pose, info


def se3_edge_to_se3(tokens):
    pose = [float(tokens[i]) for i in range(3, 10)]
    if len(tokens) >= 31:
        info = [float(tokens[i]) for i in range(10, 31)]
    else:
        info = [1.0] * 21
    return pose, info


def pose3_vertex_line(key: str, pose) -> str:
    x, y, z, qx, qy, qz, qw = pose
    return f"VERTEX_SE3:QUAT {key} {x} {y} {z} {qx} {qy} {qz} {qw}"


def pose3_edge_line(key1: str, key2: str, pose, info) -> str:
    pose_str = " ".join(str(v) for v in pose)
    info_str = " ".join(str(v) for v in info)
    return f"EDGE_SE3:QUAT {key1} {key2} {pose_str} {info_str}"


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_g2o(path: Path):
    vertices = {}
    vertex_order = []
    edges = []
    passthrough = []
    vertex_kind = None
    edge_kind = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                passthrough.append(raw_line)
                continue
            tokens = line.split()
            if tokens[0].startswith("VERTEX"):
                gid = int(tokens[1])
                vertices[gid] = tokens
                vertex_order.append(gid)
                if vertex_kind is None:
                    vertex_kind = tokens[0]
            elif tokens[0].startswith("EDGE"):
                edges.append(tokens)
                if edge_kind is None:
                    edge_kind = tokens[0]
            else:
                passthrough.append(raw_line)
    return vertices, vertex_order, edges, passthrough, vertex_kind, edge_kind


def is_se3_g2o(vertex_kind, edge_kind) -> bool:
    first_kind = vertex_kind or edge_kind or ""
    return "SE3" in first_kind


def se2_measurement(tokens):
    dx = float(tokens[3])
    dy = float(tokens[4])
    dtheta = float(tokens[5])
    i11 = float(tokens[6])
    i12 = float(tokens[7])
    i22 = float(tokens[9])
    i33 = float(tokens[11])
    trans_info = np.array([[i11, i12], [i12, i22]], dtype=float)
    tau = 2.0 / np.trace(np.linalg.inv(trans_info))
    return np.array([dx, dy]), np.array([[math.cos(dtheta), -math.sin(dtheta)], [math.sin(dtheta), math.cos(dtheta)]]), i33, tau


def se3_measurement(tokens):
    t = np.array([float(tokens[3]), float(tokens[4]), float(tokens[5])], dtype=float)
    r = quat_to_rot(float(tokens[6]), float(tokens[7]), float(tokens[8]), float(tokens[9]))
    i11, i12, i13 = float(tokens[10]), float(tokens[11]), float(tokens[12])
    i22, i23, i33 = float(tokens[16]), float(tokens[17]), float(tokens[21])
    i44, i45, i46 = float(tokens[25]), float(tokens[26]), float(tokens[27])
    i55, i56, i66 = float(tokens[28]), float(tokens[29]), float(tokens[30])
    trans_info = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]], dtype=float)
    rot_info = np.array([[i44, i45, i46], [i45, i55, i56], [i46, i56, i66]], dtype=float)
    tau = 3.0 / np.trace(np.linalg.inv(trans_info))
    kappa = 3.0 / (2.0 * np.trace(np.linalg.inv(rot_info)))
    return t, r, kappa, tau


def chordal_initialized_vertices(edges, is3d: bool, ordered_vertices):
    dim = 3 if is3d else 2
    dim2 = dim * dim
    key_to_idx = {gid: idx for idx, gid in enumerate(ordered_vertices)}
    anchor_idx = 0
    variable_count = len(ordered_vertices) - 1

    rows, cols, data, rhs = [], [], [], []
    row = 0
    for tokens in edges:
        gid1, gid2 = int(tokens[1]), int(tokens[2])
        if gid1 not in key_to_idx or gid2 not in key_to_idx:
            continue
        if is3d:
            _, rel_rot, kappa, _ = se3_measurement(tokens)
        else:
            _, rel_rot, kappa, _ = se2_measurement(tokens)
        weight = math.sqrt(max(kappa, 0.0))
        idx1, idx2 = key_to_idx[gid1], key_to_idx[gid2]
        for r in range(dim):
            for c in range(dim):
                target = 0.0
                if idx1 == anchor_idx:
                    target -= weight * rel_rot[r, c]
                else:
                    base = (idx1 - 1) * dim2
                    for k in range(dim):
                        rows.append(row)
                        cols.append(base + r * dim + k)
                        data.append(weight * rel_rot[k, c])
                if idx2 == anchor_idx:
                    target += weight * (1.0 if r == c else 0.0)
                else:
                    rows.append(row)
                    cols.append((idx2 - 1) * dim2 + r * dim + c)
                    data.append(-weight)
                rhs.append(target)
                row += 1

    if variable_count > 0 and rows:
        rot_solution = lsqr(coo_matrix((data, (rows, cols)), shape=(row, variable_count * dim2)).tocsr(), np.array(rhs))[0]
    else:
        rot_solution = np.zeros(variable_count * dim2)

    rotations = [np.eye(dim) for _ in ordered_vertices]
    for idx in range(1, len(ordered_vertices)):
        raw = rot_solution[(idx - 1) * dim2 : idx * dim2].reshape((dim, dim))
        rotations[idx] = project_to_rotation(raw)

    rows, cols, data, rhs = [], [], [], []
    row = 0
    for tokens in edges:
        gid1, gid2 = int(tokens[1]), int(tokens[2])
        if gid1 not in key_to_idx or gid2 not in key_to_idx:
            continue
        if is3d:
            rel_t, _, _, tau = se3_measurement(tokens)
        else:
            rel_t, _, _, tau = se2_measurement(tokens)
        weight = math.sqrt(max(tau, 0.0))
        idx1, idx2 = key_to_idx[gid1], key_to_idx[gid2]
        offset = rotations[idx1] @ rel_t
        for c in range(dim):
            target = -weight * offset[c]
            if idx1 != anchor_idx:
                rows.append(row)
                cols.append((idx1 - 1) * dim + c)
                data.append(weight)
            if idx2 != anchor_idx:
                rows.append(row)
                cols.append((idx2 - 1) * dim + c)
                data.append(-weight)
            rhs.append(target)
            row += 1

    if variable_count > 0 and rows:
        trans_solution = lsqr(coo_matrix((data, (rows, cols)), shape=(row, variable_count * dim)).tocsr(), np.array(rhs))[0]
    else:
        trans_solution = np.zeros(variable_count * dim)

    translations = [np.zeros(dim) for _ in ordered_vertices]
    for idx in range(1, len(ordered_vertices)):
        translations[idx] = trans_solution[(idx - 1) * dim : idx * dim]

    output = {}
    for gid, idx in key_to_idx.items():
        if is3d:
            qx, qy, qz, qw = rot_to_quat(rotations[idx])
            output[gid] = (
                translations[idx][0],
                translations[idx][1],
                translations[idx][2],
                qx,
                qy,
                qz,
                qw,
            )
        else:
            theta = math.atan2(rotations[idx][1, 0], rotations[idx][0, 0])
            qx, qy, qz, qw = yaw_to_quat(theta)
            output[gid] = (translations[idx][0], translations[idx][1], 0.0, qx, qy, qz, qw)
    return output


def infer_se2_vertices_from_edges(edges):
    poses = {}
    poses[0] = (0.0, 0.0, 0.0)
    pending = True
    while pending:
        pending = False
        for tokens in edges:
            if tokens[0] != "EDGE_SE2":
                continue
            gid1 = int(tokens[1])
            gid2 = int(tokens[2])
            rel = (float(tokens[3]), float(tokens[4]), float(tokens[5]))
            if gid1 in poses and gid2 not in poses:
                poses[gid2] = compose_se2(poses[gid1], rel)
                pending = True
            elif gid2 in poses and gid1 not in poses:
                dx, dy, dtheta = rel
                inv = (
                    -(math.cos(dtheta) * dx + math.sin(dtheta) * dy),
                    -(-math.sin(dtheta) * dx + math.cos(dtheta) * dy),
                    -dtheta,
                )
                poses[gid1] = compose_se2(poses[gid2], inv)
                pending = True
    return poses


def split_dataset(input_path: Path, output_dir: Path, nr_robots: int, chordal_initialization: bool = True):
    vertices, vertex_order, edges, passthrough, vertex_kind, edge_kind = parse_g2o(input_path)
    if not edges:
        raise RuntimeError(f"{input_path} contains no edges")

    output_vertices = {}
    is3d = is_se3_g2o(vertex_kind, edge_kind)
    if vertex_order:
        ordered_vertices = sorted(vertex_order)
        if chordal_initialization:
            output_vertices = chordal_initialized_vertices(edges, is3d, ordered_vertices)
        elif vertex_kind == "VERTEX_SE2":
            for gid in ordered_vertices:
                output_vertices[gid] = se2_vertex_to_pose3(vertices[gid])
        elif vertex_kind == "VERTEX_SE3:QUAT":
            for gid in ordered_vertices:
                output_vertices[gid] = se3_vertex_to_pose3(vertices[gid])
        else:
            raise RuntimeError(f"Unsupported vertex type in {input_path}: {vertex_kind}")
    else:
        ordered_vertices = sorted({int(tokens[1]) for tokens in edges} | {int(tokens[2]) for tokens in edges})
        if chordal_initialization:
            output_vertices = chordal_initialized_vertices(edges, is3d, ordered_vertices)
        elif edge_kind == "EDGE_SE2":
            inferred = infer_se2_vertices_from_edges(edges)
            for gid in ordered_vertices:
                x, y, theta = inferred.get(gid, (0.0, 0.0, 0.0))
                qx, qy, qz, qw = yaw_to_quat(theta)
                output_vertices[gid] = (x, y, 0.0, qx, qy, qz, qw)
        else:
            for gid in ordered_vertices:
                output_vertices[gid] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    num_vertices = len(ordered_vertices)
    per_robot = num_vertices // nr_robots
    if per_robot == 0:
        raise RuntimeError(
            f"{input_path} has only {num_vertices} poses, fewer than robots {nr_robots}"
        )

    assignments = {}
    robot_vertices = defaultdict(list)

    for robot in range(nr_robots):
        start = robot * per_robot
        end = (robot + 1) * per_robot if robot < nr_robots - 1 else num_vertices
        for gid in ordered_vertices[start:end]:
            assignments[gid] = robot
            robot_vertices[robot].append(gid)

    local_map = {}
    for robot in range(nr_robots):
        for idx, gid in enumerate(robot_vertices[robot]):
            local_map[gid] = (robot, idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_dir / "fullGraph.g2o")
    mapping_path = output_dir / "mapping.tsv"
    with mapping_path.open("w", encoding="utf-8") as mapping:
        mapping.write("global_id\trobot\tlocal_id\tsymbol_key\n")
        for gid in ordered_vertices:
            robot, local_idx = local_map[gid]
            mapping.write(
                f"{gid}\t{robot}\t{local_idx}\t{format_key(robot, local_idx)}\n"
            )

    for robot in range(nr_robots):
        robot_path = output_dir / f"{robot}.g2o"
        with robot_path.open("w", encoding="utf-8") as out:
            for raw_line in passthrough:
                out.write(raw_line)

            for gid in robot_vertices[robot]:
                out.write(pose3_vertex_line(format_key(robot, local_map[gid][1]), output_vertices[gid]) + "\n")

            for tokens in edges:
                gid1 = int(tokens[1])
                gid2 = int(tokens[2])
                owner1 = assignments.get(gid1)
                owner2 = assignments.get(gid2)
                if owner1 is None or owner2 is None:
                    continue
                if owner1 != robot and owner2 != robot:
                    continue
                robot1, local1 = local_map[gid1]
                robot2, local2 = local_map[gid2]
                if tokens[0] == "EDGE_SE2":
                    pose, info = se2_edge_to_se3(tokens)
                elif tokens[0] == "EDGE_SE3:QUAT":
                    pose, info = se3_edge_to_se3(tokens)
                else:
                    raise RuntimeError(f"Unsupported edge type in {input_path}: {tokens[0]}")
                out.write(
                    pose3_edge_line(
                        format_key(robot1, local1),
                        format_key(robot2, local2),
                        pose,
                        info,
                    )
                    + "\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument(
        "--chordal-initialization",
        type=parse_bool,
        default=True,
        help="Compute a centralized chordal initialization on the original g2o before splitting.",
    )
    args = parser.parse_args()

    split_dataset(Path(args.input), Path(args.output_dir), args.robots, args.chordal_initialization)


if __name__ == "__main__":
    main()
