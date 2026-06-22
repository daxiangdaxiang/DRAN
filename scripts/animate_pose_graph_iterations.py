#!/usr/bin/env python3
"""Animate DRAN pose estimates saved at each optimizer iteration."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from evaluate_pgo import parse_g2o_graph
from visualize_pose_graph import (
    ROBOT_COLORS,
    PoseSeries,
    common_aligned_positions,
    edge_pairs,
    load_series,
    make_partitions,
    projection_for,
    projection_indices,
    set_equal_axis_3d,
)


@dataclass
class AnimationFrame:
    series: PoseSeries
    stage: str
    step: int | None
    metrics_iter: int | None = None
    metadata: dict[str, str] | None = None


def configure_rviz_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "figure.facecolor": "#F5F6F7",
        "axes.facecolor": "#F8F9FA",
        "legend.frameon": False,
    })


def configure_ffmpeg_writer() -> None:
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        return
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def load_iteration_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["iter"]): row for row in csv.DictReader(handle) if row.get("iter")}


def sorted_pose_files(directory: Path, prefix: str) -> list[Path]:
    files = sorted(directory.glob(f"{prefix}_*.txt"))
    if not files:
        raise ValueError(f"no {prefix}_*.txt files found in {directory}")
    return files


def load_stage_metadata(directory: Path, prefix: str) -> dict[int, dict[str, str]]:
    path = directory / f"{prefix}_metadata.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            try:
                frame = int(row["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[frame] = row
        return rows


def file_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_frame_series(path: Path, label: str, reference: PoseSeries | None) -> PoseSeries:
    frame = load_series(path, label)
    if reference is not None:
        frame = PoseSeries(frame.label, frame.ids, common_aligned_positions(frame, reference))
    return frame


def load_stage_frames(directory: Path, label: str, reference: PoseSeries | None,
                      stage: str, prefix: str, metrics_from_filename: bool) -> list[AnimationFrame]:
    frames = []
    metadata = load_stage_metadata(directory, prefix)
    pose_files = sorted_pose_files(directory, prefix)
    file_by_index = {file_index(path): path for path in pose_files}
    if metadata:
        frame_items = []
        for frame_index in sorted(metadata):
            if frame_index not in file_by_index:
                raise ValueError(
                    f"metadata references missing {prefix}_{frame_index:04d}.txt in {directory}")
            frame_items.append((frame_index, file_by_index[frame_index]))
    else:
        frame_items = [(file_index(path), path) for path in pose_files]
    for frame_index, path in frame_items:
        row = metadata.get(frame_index, {})
        step = frame_index
        if row.get("pcg_iter"):
            try:
                step = int(row["pcg_iter"])
            except ValueError:
                step = frame_index
        frame_stage = row.get("stage") or stage
        frames.append(AnimationFrame(
            series=load_frame_series(path, label, reference),
            stage=frame_stage,
            step=step,
            metrics_iter=frame_index if metrics_from_filename else None,
            metadata=row,
        ))
    return frames


def graph_pose_series(graph_poses: dict[int, np.ndarray], reference: PoseSeries | None) -> PoseSeries | None:
    if not graph_poses:
        return None
    ids = np.array(sorted(graph_poses.keys()), dtype=int)
    xyz = np.array([graph_poses[int(gid)][:3, 3] for gid in ids], dtype=float)
    series = PoseSeries("Raw", ids, xyz)
    if reference is not None:
        series = PoseSeries(series.label, series.ids, common_aligned_positions(series, reference))
    return series


def draw_edges(ax, ids: np.ndarray, xyz: np.ndarray, edges, projection: str,
               edge_mode: str, edge_sample: int) -> None:
    index = {int(gid): idx for idx, gid in enumerate(ids)}
    for i, j, is_odometry in edge_pairs(edges, edge_mode, edge_sample):
        if i not in index or j not in index:
            continue
        pts = xyz[[index[i], index[j]]]
        color = "#BFC6CF" if is_odometry else "#A78BBD"
        alpha = 0.24 if is_odometry else 0.18
        if projection == "3d":
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.25, alpha=alpha)
        else:
            xidx, yidx = projection_indices(projection)
            ax.plot(pts[:, xidx], pts[:, yidx], color=color, linewidth=0.25, alpha=alpha)


def draw_robot_tracks(ax, frame: PoseSeries, num_robots: int, projection: str,
                      line_width: float, point_size: float) -> None:
    index = {int(gid): idx for idx, gid in enumerate(frame.ids)}
    for rid, ids in enumerate(make_partitions(frame.ids, num_robots)):
        if len(ids) == 0:
            continue
        pts = np.array([frame.xyz[index[int(gid)]] for gid in ids])
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        if projection == "3d":
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=color, linewidth=line_width, alpha=0.94)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=point_size, color=color, alpha=0.84, depthshade=False)
        else:
            xidx, yidx = projection_indices(projection)
            if len(pts) > 1:
                ax.plot(pts[:, xidx], pts[:, yidx],
                        color=color, linewidth=line_width, alpha=0.94)
            ax.scatter(pts[:, xidx], pts[:, yidx],
                       s=point_size, color=color, alpha=0.84)


def fixed_limits(frames: list[PoseSeries], projection: str):
    xyz = np.vstack([frame.xyz for frame in frames])
    center = xyz.mean(axis=0)
    span = np.ptp(xyz, axis=0)
    if projection == "3d":
        radius = max(float(span.max()) * 0.55, 1e-9)
        return (
            (center[0] - radius, center[0] + radius),
            (center[1] - radius, center[1] + radius),
            (center[2] - radius, center[2] + radius),
        )
    xidx, yidx = projection_indices(projection)
    radius_x = max(float(span[xidx]) * 0.55, 1e-9)
    radius_y = max(float(span[yidx]) * 0.55, 1e-9)
    return (
        (center[xidx] - radius_x, center[xidx] + radius_x),
        (center[yidx] - radius_y, center[yidx] + radius_y),
    )


def format_metric(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    parts = []
    for label, key in [
        ("cost", "global_cost"),
        ("grad", "gradient"),
        ("comm", "cumulative_comm_mb"),
    ]:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if label == "comm":
            parts.append(f"{label}: {number:.3f} MB")
        elif abs(number) >= 1000:
            parts.append(f"{label}: {number:.1f}")
        else:
            parts.append(f"{label}: {number:.4g}")
    return "\n".join(parts)


def format_stage_metadata(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    parts = []
    pcg_iter = row.get("pcg_iter")
    if pcg_iter not in (None, ""):
        try:
            parts.append(f"D-CCI-PCG iter: {int(pcg_iter)}")
        except ValueError:
            parts.append(f"D-CCI-PCG iter: {pcg_iter}")
    residual = row.get("residual_norm")
    if residual not in (None, ""):
        try:
            parts.append(f"D-CCI residual: {float(residual):.3e}")
        except ValueError:
            pass
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--estimate-dir", type=Path, required=True)
    parser.add_argument("--initialization-dir", type=Path,
                        help="optional init_*.txt directory for decentralized initialization frames")
    parser.add_argument("--iteration-summary", type=Path)
    parser.add_argument("--reference", type=Path, help="optional reference pose file for visual alignment")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--title", default="DRAN optimization")
    parser.add_argument("--method-label", default="DRAN")
    parser.add_argument("--initialization-stage-label",
                        default="Decentralized initialization")
    parser.add_argument("--optimization-stage-label",
                        default="Distributed optimization")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-robots", type=int, default=5)
    parser.add_argument("--projection", choices=["auto", "xy", "xz", "yz", "3d"], default="auto")
    parser.add_argument("--edge-mode", choices=["none", "odometry", "loop", "all"], default="none")
    parser.add_argument("--edge-sample", type=int, default=25)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--view-elev", type=float, default=24.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    parser.add_argument("--raw-hold-frames", type=int, default=6)
    parser.add_argument("--line-width", type=float, default=1.25)
    parser.add_argument("--point-size", type=float, default=4.2)
    parser.add_argument("--no-raw", action="store_true",
                        help="do not prepend the raw g2o vertex estimate")
    args = parser.parse_args(argv)

    configure_rviz_style()
    configure_ffmpeg_writer()
    reference = load_series(args.reference, args.reference_label) if args.reference else None
    graph_poses, edges = parse_g2o_graph(args.graph)
    frames: list[AnimationFrame] = []
    raw_series = graph_pose_series(graph_poses, reference)
    if raw_series is not None and not args.no_raw:
        hold = max(1, args.raw_hold_frames)
        frames.extend(AnimationFrame(raw_series, "Raw input poses", None) for _ in range(hold))
    if args.initialization_dir:
        frames.extend(load_stage_frames(
            args.initialization_dir, args.method_label, reference,
            args.initialization_stage_label, "init", False,
        ))
    frames.extend(load_stage_frames(
        args.estimate_dir, args.method_label, reference,
        args.optimization_stage_label, "iter", True,
    ))
    iter_rows = load_iteration_rows(args.iteration_summary) if args.iteration_summary else {}
    projection = projection_for([frame.series for frame in frames], args.projection)
    limits = fixed_limits([frame.series for frame in frames], projection)

    fig = plt.figure(figsize=(7.0, 5.0) if projection == "3d" else (7.0, 4.7))
    ax = fig.add_subplot(1, 1, 1, projection="3d" if projection == "3d" else None)

    def draw(frame_idx: int):
        ax.clear()
        frame = frames[frame_idx]
        series = frame.series
        if args.edge_mode != "none":
            draw_edges(ax, series.ids, series.xyz, edges, projection, args.edge_mode, args.edge_sample)
        draw_robot_tracks(ax, series, args.num_robots, projection,
                          args.line_width, args.point_size)
        metric_parts = [
            text for text in [
                format_stage_metadata(frame.metadata),
                format_metric(iter_rows.get(frame.metrics_iter)),
            ] if text
        ]
        metric_text = "\n".join(metric_parts)
        if frame.step is None:
            stage_text = frame.stage
        elif frame.stage.startswith("Decentralized"):
            stage_text = f"{frame.stage} round {frame.step}"
        elif "PCG" in frame.stage:
            stage_text = f"{frame.stage} iter {frame.step}"
        else:
            stage_text = f"{frame.stage} iter {frame.step}"
        title = f"{args.title}  |  {stage_text}"
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        if metric_text:
            if projection == "3d" and hasattr(ax, "text2D"):
                ax.text2D(0.02, 0.92, metric_text, transform=ax.transAxes,
                          ha="left", va="top",
                          bbox={"facecolor": "white", "edgecolor": "#D5DADD", "alpha": 0.9})
            else:
                ax.text(0.02, 0.92, metric_text, transform=ax.transAxes,
                        ha="left", va="top",
                        bbox={"facecolor": "white", "edgecolor": "#D5DADD", "alpha": 0.9})
        if projection == "3d":
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_zlim(*limits[2])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=args.view_elev, azim=args.view_azim)
            try:
                ax.set_box_aspect((1, 1, 0.72))
            except AttributeError:
                pass
        else:
            labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_xlabel(labels[projection][0])
            ax.set_ylabel(labels[projection][1])
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#DDE2E6", linewidth=0.5)
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(1, args.fps), blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".gif":
        ani.save(args.output, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    else:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
        ani.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
