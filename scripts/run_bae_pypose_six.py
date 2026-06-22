#!/usr/bin/env python3
"""Run the PyPose/BAE PGO baseline on the six project datasets.

SE(3) graphs are optimized directly with BAE's sparse LM backend. SE(2) graphs
are embedded as planar SE(3) graphs so the same BAE sparse solver path can be
used, then evaluated on the original SE(2) graph with scripts/evaluate_pgo.py.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("BAE_USE_PYPOSE_AMBIENT_GRAD", "1")

import pypose as pp  # noqa: E402
from bae.autograd.function import TrackingTensor, map_transform  # noqa: E402
from bae.optim import LM  # noqa: E402
from bae.utils.pgo_dataset import G2OPGO  # noqa: E402
from bae.utils.pysolvers import cuSolverSP  # noqa: E402
from pypose.optim.scheduler import StopOnPlateau  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import split_g2o_to_distributed_mapper as dm_split  # noqa: E402


DATASETS = {
    "parking-garage": Path("data/parking-garage.g2o"),
    "sphere": Path("data/sphere2500.g2o"),
    "torus": Path("data/torus3D.g2o"),
    "CSAIL": Path("data/CSAIL.g2o"),
    "inter": Path("data/input_INTEL_g2o.g2o"),
    "manhattan": Path("data/input_M3500_g2o.g2o"),
}


@dataclass
class Edge2:
    i: int
    j: int
    measurement: np.ndarray
    info: np.ndarray


@dataclass
class PartitionFactorSet:
    robot: int
    owned_variable_ids: list[int]
    fixed_ids: list[int]
    edge_indices: list[int]
    edge_local_indices: torch.Tensor


@map_transform
def se3_residual(poses, node1, node2, infos):
    residual = (poses.Inv() @ node1.Inv() @ node2).Log().tensor()
    residual = infos @ residual[..., None]
    return residual[..., 0]


class SparseSE3PoseGraph(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nn.Parameter(TrackingTensor(nodes))

    def forward(self, edges, poses, infos):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        return se3_residual(poses, node1, node2, infos)


@map_transform
def chordal_se3_residual(poses, node1, node2, weights):
    node1 = pp.SE3(node1)
    node2 = pp.SE3(node2)
    poses = pp.SE3(poses)
    ri = node1.rotation().matrix()
    rj = node2.rotation().matrix()
    rij = poses.rotation().matrix()
    rt = node2.translation() - node1.translation() - node1.rotation().Act(poses.translation())
    rr = rj - torch.matmul(ri, rij)
    trans = torch.sqrt(weights[..., 0:1].clamp_min(0.0)) * rt
    rot = torch.sqrt(weights[..., 1:2].clamp_min(0.0)) * rr.reshape(rr.shape[:-2] + (9,))
    return torch.cat((trans, rot), dim=-1)


def direct_chordal_cost_torch(poses, node1, node2, weights):
    node1 = pp.SE3(node1)
    node2 = pp.SE3(node2)
    poses = pp.SE3(poses)
    ri = node1.rotation().matrix()
    rj = node2.rotation().matrix()
    rij = poses.rotation().matrix()
    rt = node2.translation() - node1.translation() - node1.rotation().Act(poses.translation())
    rr = rj - torch.matmul(ri, rij)
    return torch.sum(weights[..., 0] * torch.sum(rt * rt, dim=-1) + weights[..., 1] * torch.sum(rr * rr, dim=(-2, -1)))


class LocalChordalPoseGraph(nn.Module):
    def __init__(self, owned_nodes, fixed_nodes):
        super().__init__()
        self.nodes = nn.Parameter(TrackingTensor(owned_nodes))
        self.nodes.trim_SE3_grad = True
        self.register_buffer("fixed_nodes", fixed_nodes)

    def combined_nodes(self):
        if self.fixed_nodes.numel() == 0:
            return self.nodes
        return torch.cat([self.nodes, self.fixed_nodes], dim=0)

    def forward(self, edges, poses, weights):
        nodes = self.combined_nodes()
        node1 = nodes[edges[..., 0]]
        node2 = nodes[edges[..., 1]]
        return chordal_se3_residual(poses, node1, node2, weights)


def quat_from_yaw(theta: float) -> tuple[float, float, float, float]:
    half = 0.5 * theta
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def se2_to_mat(x: float, y: float, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    out = np.eye(3)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[0, 2] = x
    out[1, 2] = y
    return out


def mat_to_se2(mat: np.ndarray) -> tuple[float, float, float]:
    return float(mat[0, 2]), float(mat[1, 2]), math.atan2(float(mat[1, 0]), float(mat[0, 0]))


def unpack_info3(values: list[float]) -> np.ndarray:
    info = np.zeros((3, 3), dtype=float)
    idx = 0
    for row in range(3):
        for col in range(row, 3):
            info[row, col] = values[idx]
            info[col, row] = values[idx]
            idx += 1
    return info


def pack_info6(info: np.ndarray) -> str:
    values = []
    for row in range(6):
        for col in range(row, 6):
            values.append(info[row, col])
    return " ".join(f"{v:.17g}" for v in values)


def build_contiguous_partitions(num_poses: int, num_robots: int) -> list[tuple[int, int]]:
    if num_robots <= 0:
        raise ValueError("num_robots must be positive")
    poses_per_robot = num_poses // num_robots
    if poses_per_robot <= 0:
        raise ValueError(f"num_poses={num_poses} is smaller than num_robots={num_robots}")
    partitions = []
    for robot in range(num_robots):
        start = robot * poses_per_robot
        end = (robot + 1) * poses_per_robot if robot < num_robots - 1 else num_poses
        partitions.append((start, end))
    return partitions


def owner_from_sorted_index(index: int, partitions: list[tuple[int, int]]) -> int:
    for robot, (start, end) in enumerate(partitions):
        if start <= index < end:
            return robot
    raise ValueError(f"pose index {index} outside partitions")


def build_partitioned_factor_sets(
    ids: list[int], edge_global_ids: torch.Tensor, num_robots: int, fixed_pose_ids: set[int]
) -> list[PartitionFactorSet]:
    sorted_ids = sorted(int(gid) for gid in ids)
    id_to_sorted_index = {gid: idx for idx, gid in enumerate(sorted_ids)}
    partitions = build_contiguous_partitions(len(sorted_ids), num_robots)
    owner = {gid: owner_from_sorted_index(id_to_sorted_index[gid], partitions) for gid in sorted_ids}
    owned_by_robot: list[list[int]] = [[] for _ in range(num_robots)]
    for gid in sorted_ids:
        if gid not in fixed_pose_ids:
            owned_by_robot[owner[gid]].append(gid)

    factors = []
    edges_cpu = edge_global_ids.detach().cpu().tolist()
    for robot in range(num_robots):
        edge_indices = [
            idx
            for idx, (gid1, gid2) in enumerate(edges_cpu)
            if owner[int(gid1)] == robot or owner[int(gid2)] == robot
        ]
        owned_ids = owned_by_robot[robot]
        owned_set = set(owned_ids)
        fixed = sorted(
            {
                int(gid)
                for idx in edge_indices
                for gid in edges_cpu[idx]
                if int(gid) not in owned_set
            }
        )
        combined_ids = owned_ids + fixed
        combined_index = {gid: idx for idx, gid in enumerate(combined_ids)}
        local_edges = [[combined_index[int(edges_cpu[idx][0])], combined_index[int(edges_cpu[idx][1])]] for idx in edge_indices]
        edge_tensor = torch.tensor(local_edges, dtype=torch.long) if local_edges else torch.empty((0, 2), dtype=torch.long)
        factors.append(PartitionFactorSet(robot, owned_ids, fixed, edge_indices, edge_tensor))
    return factors


def communication_pose_count_per_round(ids: list[int], edge_global_ids: torch.Tensor, num_robots: int, fixed_pose_ids: set[int]) -> int:
    sorted_ids = sorted(int(gid) for gid in ids)
    id_to_sorted_index = {gid: idx for idx, gid in enumerate(sorted_ids)}
    partitions = build_contiguous_partitions(len(sorted_ids), num_robots)
    owner = {gid: owner_from_sorted_index(id_to_sorted_index[gid], partitions) for gid in sorted_ids}
    directed_boundary_poses = set()
    for gid1_raw, gid2_raw in edge_global_ids.detach().cpu().tolist():
        gid1 = int(gid1_raw)
        gid2 = int(gid2_raw)
        robot1 = owner[gid1]
        robot2 = owner[gid2]
        if robot1 == robot2:
            continue
        if gid1 not in fixed_pose_ids:
            directed_boundary_poses.add((robot1, robot2, gid1))
        if gid2 not in fixed_pose_ids:
            directed_boundary_poses.add((robot2, robot1, gid2))
    return len(directed_boundary_poses)


def detect_graph_dim(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tag = line.split()[0]
            if tag in {"VERTEX_SE3:QUAT", "EDGE_SE3:QUAT"}:
                return 3
            if tag in {"VERTEX_SE2", "EDGE_SE2"}:
                return 2
    raise ValueError(f"cannot detect graph dimension for {path}")


def parse_se2_graph(path: Path) -> tuple[dict[int, np.ndarray], list[Edge2]]:
    vertices: dict[int, np.ndarray] = {}
    edges: list[Edge2] = []
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            toks = raw.strip().split()
            if not toks:
                continue
            if toks[0] == "VERTEX_SE2":
                gid = int(toks[1])
                vertices[gid] = se2_to_mat(float(toks[2]), float(toks[3]), float(toks[4]))
                ids.add(gid)
            elif toks[0] == "EDGE_SE2":
                i = int(toks[1])
                j = int(toks[2])
                meas = se2_to_mat(float(toks[3]), float(toks[4]), float(toks[5]))
                info = unpack_info3([float(v) for v in toks[6:12]])
                edges.append(Edge2(i, j, meas, info))
                ids.update((i, j))
    if not vertices and ids:
        vertices[min(ids)] = np.eye(3)
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.i in vertices and edge.j not in vertices:
                vertices[edge.j] = vertices[edge.i] @ edge.measurement
                changed = True
            elif edge.j in vertices and edge.i not in vertices:
                vertices[edge.i] = vertices[edge.j] @ np.linalg.inv(edge.measurement)
                changed = True
    for gid in ids:
        vertices.setdefault(gid, np.eye(3))
    return vertices, edges


def write_planar_se3_graph(source: Path, target: Path, extra_weight: float) -> None:
    vertices, edges = parse_se2_graph(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for gid in sorted(vertices):
            x, y, theta = mat_to_se2(vertices[gid])
            qx, qy, qz, qw = quat_from_yaw(theta)
            f.write(f"VERTEX_SE3:QUAT {gid} {x:.17g} {y:.17g} 0 {qx:.17g} {qy:.17g} {qz:.17g} {qw:.17g}\n")
        for edge in edges:
            x, y, theta = mat_to_se2(edge.measurement)
            qx, qy, qz, qw = quat_from_yaw(theta)
            info6 = np.eye(6) * extra_weight
            idx = [0, 1, 5]
            for a in range(3):
                for b in range(3):
                    info6[idx[a], idx[b]] = edge.info[a, b]
            f.write(
                f"EDGE_SE3:QUAT {edge.i} {edge.j} {x:.17g} {y:.17g} 0 "
                f"{qx:.17g} {qy:.17g} {qz:.17g} {qw:.17g} {pack_info6(info6)}\n"
            )


def load_centralized_chordal_init(source: Path, ids: list[int], is3d: bool, device: str, dtype: torch.dtype) -> torch.Tensor:
    vertices, vertex_order, edges, _, vertex_kind, edge_kind = dm_split.parse_g2o(source)
    if vertex_order:
        ordered_vertices = sorted(vertex_order)
    else:
        ordered_vertices = sorted({int(tokens[1]) for tokens in edges} | {int(tokens[2]) for tokens in edges})
    init = dm_split.chordal_initialized_vertices(edges, is3d, ordered_vertices)
    rows = []
    for gid in ids:
        if int(gid) not in init:
            raise RuntimeError(f"chordal initialization did not produce pose {gid} for {source}")
        rows.append(init[int(gid)])
    return torch.tensor(rows, dtype=dtype, device=device)


def weights_from_se3_infos(infos: torch.Tensor) -> torch.Tensor:
    trans_info = infos[..., :3, :3]
    rot_info = infos[..., 3:6, 3:6]
    trans_cov = torch.linalg.inv(trans_info)
    rot_cov = torch.linalg.inv(rot_info)
    tau = 3.0 / torch.diagonal(trans_cov, dim1=-2, dim2=-1).sum(dim=-1)
    kappa = 3.0 / (2.0 * torch.diagonal(rot_cov, dim1=-2, dim2=-1).sum(dim=-1))
    return torch.stack((tau, kappa), dim=-1)


def weights_from_se2_graph(source: Path, device: str, dtype: torch.dtype) -> torch.Tensor:
    _, edges = parse_se2_graph(source)
    values = []
    for edge in edges:
        trans_info = torch.tensor(edge.info[:2, :2], dtype=dtype, device=device)
        trans_cov = torch.linalg.inv(trans_info)
        tau = 2.0 / torch.diagonal(trans_cov).sum()
        kappa = torch.tensor(edge.info[2, 2], dtype=dtype, device=device)
        values.append(torch.stack((tau, kappa)))
    return torch.stack(values, dim=0)


def write_estimate(path: Path, nodes, ids) -> None:
    tensor = pp.SE3(nodes.detach()).tensor().detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for gid, pose in zip(ids, tensor):
            f.write(
                "VERTEX_SE3:QUAT "
                f"{int(gid)} "
                + " ".join(f"{float(v):.17g}" for v in pose)
                + "\n"
            )


def evaluate_chordal(repo: Path, graph: Path, estimate: Path) -> dict:
    cmd = [
        sys.executable,
        str(repo / "scripts/evaluate_pgo.py"),
        "--graph",
        str(graph),
        "--estimate",
        str(estimate),
        "--reference",
        str(graph),
        "--weighted",
        "--cost-mode",
        "dpgo",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=repo, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def cholesky_with_relative_jitter(infos: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    infos = 0.5 * (infos + infos.transpose(-1, -2))
    eye = torch.eye(infos.shape[-1], dtype=infos.dtype, device=infos.device)
    diag_scale = infos.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp_min(1.0)
    last_jitter = 0.0
    regularized_count = 0
    for power in range(0, 13):
        rel_jitter = 0.0 if power == 0 else 10.0 ** (-14 + power)
        candidate = infos if rel_jitter == 0.0 else infos + (rel_jitter * diag_scale)[:, None, None] * eye
        chol, info = torch.linalg.cholesky_ex(candidate)
        if bool((info == 0).all()):
            return chol, regularized_count, rel_jitter
        regularized_count = max(regularized_count, int((info != 0).sum().item()))
        last_jitter = rel_jitter

    mats = infos.detach().cpu()
    fixed = []
    for mat in mats:
        vals, vecs = torch.linalg.eigh(mat)
        floor = max(float(vals.abs().max().item()) * max(last_jitter, 1e-10), 1e-9)
        vals = vals.clamp_min(floor)
        fixed.append((vecs @ torch.diag(vals) @ vecs.T).to(infos.device))
    repaired = torch.stack(fixed, dim=0)
    return torch.linalg.cholesky(repaired), regularized_count, max(last_jitter, 1e-10)


def run_partitioned_chordal(data, weights, source_graph: Path, dim: int, num_robots: int, fixed_pose_ids: set[int], args):
    ids = [int(v) for v in data.ids.detach().cpu().tolist()]
    id_to_row = {gid: idx for idx, gid in enumerate(ids)}
    global_nodes = load_centralized_chordal_init(source_graph, ids, dim == 3, args.device, args.dtype)
    factors = build_partitioned_factor_sets(ids, data.edges.detach().cpu(), num_robots, fixed_pose_ids)
    comm_pose_count = communication_pose_count_per_round(ids, data.edges.detach().cpu(), num_robots, fixed_pose_ids)
    pose_payload_bytes = 7 * 8
    iter_rows = []
    total_solver_steps = 0
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for outer in range(args.steps):
        snapshot = global_nodes.detach().clone()
        updates: list[tuple[list[int], torch.Tensor]] = []
        for factor in factors:
            if not factor.owned_variable_ids or not factor.edge_indices:
                continue
            owned_rows = [id_to_row[gid] for gid in factor.owned_variable_ids]
            fixed_rows = [id_to_row[gid] for gid in factor.fixed_ids]
            owned_nodes = snapshot[owned_rows].clone()
            fixed_nodes = snapshot[fixed_rows].clone() if fixed_rows else snapshot.new_empty((0, 7))
            local_model = LocalChordalPoseGraph(owned_nodes, fixed_nodes).to(args.device)
            local_edges = factor.edge_local_indices.to(args.device)
            local_poses = data.poses[factor.edge_indices]
            local_weights = weights[factor.edge_indices]
            optimizer = LM(local_model, solver=cuSolverSP(), strategy=pp.optim.strategy.Adaptive(), min=1e-10, reject=30)
            scheduler = StopOnPlateau(optimizer, steps=20, patience=3, decreasing=1e-7, verbose=False)
            for _ in range(args.local_steps):
                if args.verbose_solver:
                    loss = optimizer.step(input={"edges": local_edges, "poses": local_poses, "weights": local_weights})
                else:
                    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                        loss = optimizer.step(input={"edges": local_edges, "poses": local_poses, "weights": local_weights})
                scheduler.step(loss)
                total_solver_steps += 1
            updates.append((factor.owned_variable_ids, pp.SE3(local_model.nodes.detach()).tensor().detach()))
        for owned_ids, nodes in updates:
            rows = [id_to_row[gid] for gid in owned_ids]
            global_nodes[rows] = nodes
        current_cost = float(
            direct_chordal_cost_torch(
                data.poses,
                global_nodes[data.edges[..., 0]],
                global_nodes[data.edges[..., 1]],
                weights,
            ).detach().cpu().item()
        )
        iter_rows.append(
            {
                "iter": outer + 1,
                "global_chordal_cost": current_cost,
                "comm_pose_count": comm_pose_count,
                "iter_comm_mb": comm_pose_count * pose_payload_bytes / (1024.0 * 1024.0),
                "cumulative_comm_pose_count": comm_pose_count * (outer + 1),
                "cumulative_comm_mb": comm_pose_count * (outer + 1) * pose_payload_bytes / (1024.0 * 1024.0),
            }
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    wall_sec = time.perf_counter() - start
    return global_nodes, iter_rows, wall_sec, total_solver_steps, comm_pose_count


def run_one(repo: Path, name: str, graph: Path, output_dir: Path, args) -> dict:
    source_graph = repo / graph
    dim = detect_graph_dim(source_graph)
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if dim == 2:
        opt_graph = run_dir / f"{name}_planar_se3.g2o"
        write_planar_se3_graph(source_graph, opt_graph, args.se2_extra_weight)
    else:
        opt_graph = source_graph

    data = G2OPGO(str(opt_graph.parent), opt_graph.name, device=args.device, download=False)
    data.nodes = data.nodes.to(args.dtype)
    data.poses = data.poses.to(args.dtype)
    data.infos = data.infos.to(args.dtype)
    weights = weights_from_se3_infos(data.infos) if dim == 3 else weights_from_se2_graph(source_graph, args.device, args.dtype)
    regularized_count = 0
    whitening_jitter = 0.0
    global_nodes, iter_rows, wall_sec, total_solver_steps, comm_pose_count = run_partitioned_chordal(
        data, weights, source_graph, dim, args.num_robots, {0} if args.fix_first_pose else set(), args
    )
    iteration_path = run_dir / "iteration_summary.csv"
    if iter_rows:
        with iteration_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(iter_rows[0].keys()))
            writer.writeheader()
            writer.writerows(iter_rows)
    estimate_path = run_dir / "estimate.g2o"
    write_estimate(estimate_path, global_nodes, data.ids.detach().cpu().numpy())
    chordal = evaluate_chordal(repo, source_graph, estimate_path)
    return {
        "dataset": name,
        "dim": f"SE({dim})",
        "num_robots": args.num_robots,
        "outer_rounds": args.steps,
        "local_steps": args.local_steps,
        "solver_only_init": "centralized_chordal",
        "fixed_first_pose": int(args.fix_first_pose),
        "pypose_solver_steps": total_solver_steps,
        "partitioned_solver_chordal_cost": iter_rows[-1]["global_chordal_cost"] if iter_rows else float("nan"),
        "wall_sec": wall_sec,
        "total_comm_poses": comm_pose_count * args.steps,
        "total_comm_mb": comm_pose_count * args.steps * 7 * 8 / (1024.0 * 1024.0),
        "whitening_regularized_count": regularized_count,
        "whitening_max_relative_jitter": whitening_jitter,
        "iteration_summary": str(iteration_path),
        "estimate": str(estimate_path),
        "chordal_cost_dpgo": chordal["chordal_cost"],
        "chordal_cost_per_edge": chordal["chordal_cost_per_edge"],
        "chordal_edges_used": chordal["chordal_edges_used"],
        "chordal_edges_missing": chordal["chordal_edges_missing"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated dataset keys")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--num-robots", type=int, default=5)
    parser.add_argument("--fix-first-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--output-dir", default="results/bae_pypose_six_20260616")
    parser.add_argument("--se2-extra-weight", type=float, default=1.0)
    args = parser.parse_args()
    args.dtype = torch.float64 if args.dtype == "float64" else torch.float32

    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]

    rows = []
    for name in requested:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name}; choices: {sorted(DATASETS)}")
        print(f"BAE_DATASET_START {name}", flush=True)
        row = run_one(repo, name, DATASETS[name], output_dir, args)
        rows.append(row)
        print("BAE_DATASET_RESULT " + json.dumps(row, sort_keys=True), flush=True)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"BAE_SUMMARY {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
