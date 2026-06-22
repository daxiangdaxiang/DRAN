#!/usr/bin/env python3
"""Visualize g2o pose graphs and optimizer pose estimates.

The script supports a single graph from command-line arguments or a multi-panel
JSON manifest. Pose inputs reuse the permissive parsers in scripts/evaluate_pgo.py
and therefore accept g2o vertices, SE-Sync matrix outputs, and DPGO/DPGO-MM
matrix outputs such as estimates_trivial.txt.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from evaluate_pgo import load_pose_set, parse_g2o_graph


METHOD_COLORS = {
    "DRAN": "#0072B2",
    "SE-Sync": "#111111",
    "DPGO-MM AMM": "#D55E00",
    "DPGO-MM": "#D55E00",
    "MESA": "#009E73",
    "Distributed Mapper": "#CC79A7",
    "Initial": "#8A8A8A",
}

ROBOT_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2"]


@dataclass
class PoseSeries:
    label: str
    ids: np.ndarray
    xyz: np.ndarray


@dataclass
class PanelSpec:
    title: str
    graph: Path
    estimates: dict[str, Path]
    projection: str = "auto"
    view_elev: float | None = None
    view_azim: float | None = None
    metrics: list[str] | None = None
    trajectory_stride: int = 1
    robot_points_label: str | None = None
    robot_line_label: str | None = None
    robot_point_size: float = 3.0
    robot_point_alpha: float = 0.72
    robot_line_width: float = 0.9
    show_estimate_lines: bool = True


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def parse_estimate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("estimate must have form LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("estimate label must be nonempty")
    return label, Path(path)


def looks_like_rotation(block: np.ndarray, dim: int) -> bool:
    rot = np.asarray(block[:dim, :dim], dtype=float)
    if rot.shape != (dim, dim):
        return False
    eye_err = np.linalg.norm(rot.T @ rot - np.eye(dim), ord="fro")
    det = np.linalg.det(rot)
    return bool(eye_err < 0.2 and abs(det - 1.0) < 0.2)


def parse_dense_positions(dense: np.ndarray) -> dict[int, np.ndarray]:
    dense = np.asarray(dense, dtype=float)
    if dense.ndim == 1:
        dense = dense.reshape(1, -1)

    # DPGO-MM transposed matrix: first n rows are translations, remaining rows
    # store 3 x 3 rotations per pose.
    if dense.ndim == 2 and dense.shape[1] == 3 and dense.shape[0] % 4 == 0:
        num_poses = dense.shape[0] // 4
        first_rot = dense[num_poses:num_poses + 3, :]
        if looks_like_rotation(first_rot, 3):
            return {gid: dense[gid, :3].copy() for gid in range(num_poses)}

    if dense.ndim == 2 and dense.shape[1] == 2 and dense.shape[0] % 3 == 0:
        num_poses = dense.shape[0] // 3
        first_rot = dense[num_poses:num_poses + 2, :]
        if looks_like_rotation(first_rot, 2):
            poses = {}
            for gid in range(num_poses):
                xyz = np.zeros(3, dtype=float)
                xyz[:2] = dense[gid, :2]
                poses[gid] = xyz
            return poses

    # Interleaved full or lifted pose blocks: first d rows of [R | t] repeated n times.
    if dense.ndim == 2 and dense.shape[0] >= 3 and dense.shape[1] % 4 == 0:
        first_block = dense[:3, :4]
        if looks_like_rotation(first_block, 3):
            poses = {}
            for gid in range(dense.shape[1] // 4):
                block = dense[:3, 4 * gid:4 * (gid + 1)]
                poses[gid] = block[:3, 3].copy()
            return poses

    if dense.ndim == 2 and dense.shape[0] >= 2 and dense.shape[1] % 3 == 0:
        first_block = dense[:2, :3]
        if looks_like_rotation(first_block, 2):
            poses = {}
            for gid in range(dense.shape[1] // 3):
                block = dense[:2, 3 * gid:3 * (gid + 1)]
                xyz = np.zeros(3, dtype=float)
                xyz[:2] = block[:2, 2]
                poses[gid] = xyz
            return poses

    # Packed SE-Sync-style matrices: [t_1 ... t_n R_1 ... R_n].
    if dense.ndim == 2 and dense.shape[0] == 3 and dense.shape[1] % 4 == 0:
        num_poses = dense.shape[1] // 4
        return {gid: dense[:, gid].copy() for gid in range(num_poses)}
    if dense.ndim == 2 and dense.shape[0] == 2 and dense.shape[1] % 3 == 0:
        num_poses = dense.shape[1] // 3
        poses = {}
        for gid in range(num_poses):
            xyz = np.zeros(3, dtype=float)
            xyz[:2] = dense[:, gid]
            poses[gid] = xyz
        return poses

    # Position-only rows are accepted as a final fallback.
    if dense.ndim == 2 and dense.shape[1] in {2, 3} and dense.shape[0] > 4:
        poses = {}
        for gid, row in enumerate(dense):
            xyz = np.zeros(3, dtype=float)
            xyz[:dense.shape[1]] = row[:dense.shape[1]]
            poses[gid] = xyz
        return poses

    return {}


def load_pose_positions(path: Path) -> dict[int, np.ndarray]:
    try:
        dense = np.loadtxt(path)
        positions = parse_dense_positions(dense)
        if positions:
            return positions
    except ValueError:
        pass

    poses, _ = load_pose_set(path, "auto")
    return {gid: mat[:3, 3].copy() for gid, mat in poses.items()}


def load_series(path: Path, label: str, fmt: str = "auto") -> PoseSeries:
    if fmt != "auto":
        poses, _ = load_pose_set(path, fmt)
        positions = {gid: mat[:3, 3].copy() for gid, mat in poses.items()}
    else:
        positions = load_pose_positions(path)
    if not positions:
        raise ValueError(f"no poses parsed from {path}")
    ids = np.array(sorted(positions.keys()), dtype=int)
    xyz = np.array([positions[int(gid)] for gid in ids], dtype=float)
    return PoseSeries(label=label, ids=ids, xyz=xyz)


def make_partitions(ids: np.ndarray, num_robots: int) -> list[np.ndarray]:
    if num_robots <= 0:
        return [ids]
    n = len(ids)
    per_robot = n // num_robots
    if per_robot <= 0:
        return [ids]
    parts = []
    for robot in range(num_robots):
        start = robot * per_robot
        end = (robot + 1) * per_robot if robot < num_robots - 1 else n
        parts.append(ids[start:end])
    return parts


def common_aligned_positions(source: PoseSeries, reference: PoseSeries) -> np.ndarray:
    src_index = {int(gid): idx for idx, gid in enumerate(source.ids)}
    ref_index = {int(gid): idx for idx, gid in enumerate(reference.ids)}
    common = sorted(set(src_index).intersection(ref_index))
    if len(common) < 3:
        return source.xyz.copy()
    p = np.array([source.xyz[src_index[gid]] for gid in common], dtype=float)
    q = np.array([reference.xyz[ref_index[gid]] for gid in common], dtype=float)
    p_mean = p.mean(axis=0)
    q_mean = q.mean(axis=0)
    pc = p - p_mean
    qc = q - q_mean
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = u @ vt
    return (source.xyz - p_mean) @ rot + q_mean


def projection_for(series: Iterable[PoseSeries], requested: str) -> str:
    if requested != "auto":
        return requested
    max_z_span = 0.0
    max_xy_span = 0.0
    for item in series:
        span = np.ptp(item.xyz, axis=0)
        max_z_span = max(max_z_span, float(span[2]))
        max_xy_span = max(max_xy_span, float(max(span[0], span[1], 1e-12)))
    return "3d" if max_z_span > 0.08 * max_xy_span else "xy"


def edge_pairs(edges, edge_mode: str, edge_sample: int) -> list[tuple[int, int, bool]]:
    pairs = []
    sample = max(1, int(edge_sample))
    loop_idx = 0
    for edge in edges:
        is_odometry = abs(edge.i - edge.j) == 1
        if edge_mode == "none":
            continue
        if edge_mode == "odometry" and not is_odometry:
            continue
        if edge_mode == "loop" and is_odometry:
            continue
        if not is_odometry:
            loop_idx += 1
            if loop_idx % sample != 0:
                continue
        pairs.append((int(edge.i), int(edge.j), is_odometry))
    return pairs


def set_equal_axis_3d(ax, xyz_sets: list[np.ndarray]) -> None:
    xyz = np.vstack([points for points in xyz_sets if len(points) > 0])
    center = xyz.mean(axis=0)
    span = np.ptp(xyz, axis=0)
    radius = max(float(span.max()) * 0.52, 1e-9)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 0.72))
    except AttributeError:
        pass


def draw_edges(ax, graph_series: PoseSeries, edges, projection: str, edge_mode: str,
               edge_sample: int) -> None:
    if edge_mode == "none":
        return
    index = {int(gid): idx for idx, gid in enumerate(graph_series.ids)}
    for i, j, is_odometry in edge_pairs(edges, edge_mode, edge_sample):
        if i not in index or j not in index:
            continue
        pts = graph_series.xyz[[index[i], index[j]]]
        if projection == "3d":
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="#C8C8C8" if is_odometry else "#BDA0CB",
                linewidth=0.28 if is_odometry else 0.22,
                alpha=0.20 if is_odometry else 0.18,
                zorder=1,
            )
        else:
            xidx, yidx = projection_indices(projection)
            ax.plot(
                pts[:, xidx],
                pts[:, yidx],
                color="#C8C8C8" if is_odometry else "#BDA0CB",
                linewidth=0.28 if is_odometry else 0.22,
                alpha=0.24 if is_odometry else 0.18,
                zorder=1,
            )


def projection_indices(projection: str) -> tuple[int, int]:
    if projection == "xy":
        return 0, 1
    if projection == "xz":
        return 0, 2
    if projection == "yz":
        return 1, 2
    raise ValueError(f"unsupported 2D projection: {projection}")


def plot_series(ax, series: PoseSeries, xyz: np.ndarray, projection: str,
                linewidth: float, alpha: float, stride: int,
                method_colors: dict[str, str] | None = None) -> None:
    color = (method_colors or {}).get(series.label, METHOD_COLORS.get(series.label, "#666666"))
    stride = max(1, int(stride))
    if stride > 1 and len(xyz) > 1:
        idx = np.arange(0, len(xyz), stride)
        if idx[-1] != len(xyz) - 1:
            idx = np.append(idx, len(xyz) - 1)
        xyz = xyz[idx]
    if projection == "3d":
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            label=series.label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=4,
        )
    else:
        xidx, yidx = projection_indices(projection)
        ax.plot(
            xyz[:, xidx],
            xyz[:, yidx],
            label=series.label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=4,
        )


def draw_robot_partition_ticks(ax, graph_series: PoseSeries, num_robots: int,
                               projection: str, point_size: float = 1.5,
                               alpha: float = 0.22,
                               draw_lines: bool = False,
                               line_width: float = 0.9) -> None:
    if num_robots <= 1:
        return
    parts = make_partitions(graph_series.ids, num_robots)
    index = {int(gid): idx for idx, gid in enumerate(graph_series.ids)}
    for rid, ids in enumerate(parts):
        if len(ids) == 0:
            continue
        line_pts = np.array([graph_series.xyz[index[int(gid)]] for gid in ids])
        chosen = ids[np.linspace(0, len(ids) - 1, min(100, len(ids))).astype(int)]
        pts = np.array([graph_series.xyz[index[int(gid)]] for gid in chosen])
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        if projection == "3d":
            if draw_lines and len(line_pts) > 1:
                ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
                        color=color, linewidth=line_width, alpha=0.92, zorder=3)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, color=color, alpha=alpha, zorder=2)
        else:
            xidx, yidx = projection_indices(projection)
            if draw_lines and len(line_pts) > 1:
                ax.plot(line_pts[:, xidx], line_pts[:, yidx],
                        color=color, linewidth=line_width, alpha=0.92, zorder=3)
            ax.scatter(pts[:, xidx], pts[:, yidx], s=point_size, color=color, alpha=alpha, zorder=2)


def draw_panel(ax, panel: PanelSpec, num_robots: int, edge_mode: str,
               edge_sample: int, align_to: str | None,
               method_colors: dict[str, str] | None = None) -> None:
    graph_poses, graph_edges = parse_g2o_graph(panel.graph)
    estimates = [load_series(path, label) for label, path in panel.estimates.items()]
    if graph_poses:
        graph_series = PoseSeries(
            label="Initial",
            ids=np.array(sorted(graph_poses.keys()), dtype=int),
            xyz=np.array([graph_poses[gid][:3, 3] for gid in sorted(graph_poses.keys())], dtype=float),
        )
    elif estimates:
        graph_series = PoseSeries("Initial", estimates[0].ids.copy(), estimates[0].xyz.copy())
    else:
        raise ValueError(f"missing graph vertices and estimates for {panel.graph}")
    all_series = [graph_series] + estimates
    projection = projection_for(all_series, panel.projection)

    reference = None
    if align_to:
        for item in all_series:
            if item.label == align_to:
                reference = item
                break
    aligned_xyz = {}
    for item in all_series:
        aligned_xyz[item.label] = (
            common_aligned_positions(item, reference) if reference is not None and item is not reference
            else item.xyz.copy()
        )

    graph_for_edges = PoseSeries("Initial", graph_series.ids, aligned_xyz["Initial"])
    draw_edges(ax, graph_for_edges, graph_edges, projection, edge_mode, edge_sample)
    robot_label = panel.robot_line_label or panel.robot_points_label
    if robot_label and robot_label in aligned_xyz:
        source = next(item for item in all_series if item.label == robot_label)
        robot_series = PoseSeries(source.label, source.ids, aligned_xyz[source.label])
        draw_robot_partition_ticks(
            ax, robot_series, num_robots, projection,
            point_size=panel.robot_point_size,
            alpha=panel.robot_point_alpha,
            draw_lines=bool(panel.robot_line_label),
            line_width=panel.robot_line_width,
        )
    else:
        draw_robot_partition_ticks(ax, graph_for_edges, num_robots, projection)

    if panel.show_estimate_lines:
        for item in estimates:
            width = 0.85 if item.label == "DRAN" else 0.55
            alpha = 0.96 if item.label == "DRAN" else 0.78
            plot_series(
                ax, item, aligned_xyz[item.label], projection,
                width, alpha, panel.trajectory_stride, method_colors,
            )

    ax.set_title(panel.title, loc="left", pad=2, fontsize=8, fontweight="bold")
    if projection == "3d":
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        elev = 24.0 if panel.view_elev is None else panel.view_elev
        azim = -58.0 if panel.view_azim is None else panel.view_azim
        ax.view_init(elev=elev, azim=azim)
        set_equal_axis_3d(ax, [aligned_xyz[item.label] for item in all_series])
        ax.grid(False)
    else:
        labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
        ax.set_xlabel(labels[projection][0])
        ax.set_ylabel(labels[projection][1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(False)
    if panel.metrics:
        text_kwargs = {
            "ha": "left",
            "va": "top",
            "fontsize": 6.3,
            "color": "#333333",
            "bbox": {"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.86, "pad": 2.0},
        }
        if projection == "3d" and hasattr(ax, "text2D"):
            ax.text2D(0.02, 0.90, "\n".join(panel.metrics), transform=ax.transAxes, **text_kwargs)
        else:
            ax.text(0.02, 0.90, "\n".join(panel.metrics), transform=ax.transAxes, **text_kwargs)


def infer_panel_projection(panel: PanelSpec) -> str:
    graph_poses, _ = parse_g2o_graph(panel.graph)
    series = []
    if graph_poses:
        series.append(PoseSeries(
            label="Initial",
            ids=np.array(sorted(graph_poses.keys()), dtype=int),
            xyz=np.array([graph_poses[gid][:3, 3] for gid in sorted(graph_poses.keys())], dtype=float),
        ))
    series.extend(load_series(path, label) for label, path in panel.estimates.items())
    return projection_for(series, panel.projection)


def load_manifest(path: Path) -> tuple[list[PanelSpec], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    panels = []
    for raw in data.get("panels", []):
        estimates = {
            str(label): resolve_path(value, base)
            for label, value in raw.get("estimates", {}).items()
        }
        panels.append(PanelSpec(
            title=str(raw.get("title", "")),
            graph=resolve_path(raw["graph"], base),
            estimates=estimates,
            projection=str(raw.get("projection", data.get("projection", "auto"))),
            view_elev=float(raw["view"]["elev"]) if "view" in raw and "elev" in raw["view"] else None,
            view_azim=float(raw["view"]["azim"]) if "view" in raw and "azim" in raw["view"] else None,
            metrics=[str(item) for item in raw.get("metrics", [])],
            trajectory_stride=int(raw.get("trajectory_stride", data.get("trajectory_stride", 1))),
            robot_points_label=raw.get("robot_points_label", data.get("robot_points_label")),
            robot_line_label=raw.get("robot_line_label", data.get("robot_line_label")),
            robot_point_size=float(raw.get("robot_point_size", data.get("robot_point_size", 3.0))),
            robot_point_alpha=float(raw.get("robot_point_alpha", data.get("robot_point_alpha", 0.72))),
            robot_line_width=float(raw.get("robot_line_width", data.get("robot_line_width", 0.9))),
            show_estimate_lines=bool(raw.get("show_estimate_lines", data.get("show_estimate_lines", True))),
        ))
    return panels, data


def save_outputs(fig, output: Path, formats: list[str], dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = output.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"png", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(target, **kwargs)


def build_figure(panels: list[PanelSpec], *, output: Path, formats: list[str],
                 num_robots: int, edge_mode: str, edge_sample: int,
                 align_to: str | None, dpi: int,
                 method_colors: dict[str, str] | None = None) -> None:
    configure_matplotlib()
    cols = len(panels)
    if cols == 1:
        fig = plt.figure(figsize=(3.35, 3.0))
        projection = infer_panel_projection(panels[0])
        ax = fig.add_subplot(1, 1, 1, projection="3d" if projection == "3d" else None)
        draw_panel(ax, panels[0], num_robots, edge_mode, edge_sample, align_to, method_colors)
        axes = [ax]
    else:
        if cols <= 3:
            rows = 1
            grid_cols = cols
            fig = plt.figure(figsize=(6.95, 2.55))
        else:
            rows = 2
            grid_cols = int(np.ceil(cols / rows))
            fig = plt.figure(figsize=(6.95, 5.05))
        axes = []
        for idx, panel in enumerate(panels, start=1):
            projection = infer_panel_projection(panel)
            ax = fig.add_subplot(rows, grid_cols, idx, projection="3d" if projection == "3d" else None)
            draw_panel(ax, panel, num_robots, edge_mode, edge_sample, align_to, method_colors)
            axes.append(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if any(panel.robot_points_label for panel in panels):
        robot_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=ROBOT_COLORS[idx % len(ROBOT_COLORS)],
                   markeredgecolor="none", markersize=4.5,
                   label=f"robot {idx + 1}")
            for idx in range(num_robots)
        ]
        handles = robot_handles + handles
        labels = [handle.get_label() for handle in robot_handles] + labels
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(labels)),
            handlelength=1.8,
            columnspacing=1.2,
        )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_outputs(fig, output, formats, dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="JSON manifest for a multi-panel figure")
    parser.add_argument("--graph", type=Path, help="g2o graph for single-panel mode")
    parser.add_argument("--estimate", action="append", type=parse_estimate_spec, default=[],
                        help="single-panel estimate as LABEL=PATH; repeatable")
    parser.add_argument("--title", default="Pose graph", help="single-panel title")
    parser.add_argument("--output", type=Path, help="output path without extension")
    parser.add_argument("--formats", default="pdf,svg,png", help="comma-separated output formats")
    parser.add_argument("--num-robots", type=int, default=5)
    parser.add_argument("--edge-mode", choices=["none", "odometry", "loop", "all"], default="odometry")
    parser.add_argument("--edge-sample", type=int, default=25,
                        help="sample every Nth non-odometry edge when edge-mode includes loops")
    parser.add_argument("--projection", choices=["auto", "xy", "xz", "yz", "3d"], default="auto")
    parser.add_argument("--align-to", default=None, help="estimate label used as translation alignment reference")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)

    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        parser.error("at least one output format is required")

    if args.manifest:
        panels, manifest = load_manifest(args.manifest)
        if not panels:
            parser.error("manifest contains no panels")
        output = args.output or resolve_path(manifest.get("output", "posegraph_figure"), args.manifest.parent)
        build_figure(
            panels,
            output=Path(output),
            formats=formats,
            num_robots=int(manifest.get("num_robots", args.num_robots)),
            edge_mode=str(manifest.get("edge_mode", args.edge_mode)),
            edge_sample=int(manifest.get("edge_sample", args.edge_sample)),
            align_to=manifest.get("align_to", args.align_to),
            dpi=args.dpi,
            method_colors={str(k): str(v) for k, v in manifest.get("method_colors", {}).items()},
        )
        return 0

    if args.graph is None:
        parser.error("--graph is required in single-panel mode")
    if not args.estimate:
        parser.error("at least one --estimate LABEL=PATH is required")
    if args.output is None:
        parser.error("--output is required")

    panel = PanelSpec(
        title=args.title,
        graph=args.graph,
        estimates={label: path for label, path in args.estimate},
        projection=args.projection,
    )
    build_figure(
        [panel],
        output=args.output,
        formats=formats,
        num_robots=args.num_robots,
        edge_mode=args.edge_mode,
        edge_sample=args.edge_sample,
        align_to=args.align_to,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
