#!/usr/bin/env python3
"""Generate shared communication topologies and weighting matrices.

The primary machine-readable output is an edge schedule CSV:

  round,src,dst,weight

`src` and `dst` are zero-based robot ids and each undirected edge appears once
with `src < dst`.  `weight` is the off-diagonal entry W_ij of a symmetric
doubly-stochastic consensus matrix for that round.  The same file is intended to
be consumed by DRAN, MESA adapters, and other baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from pathlib import Path


def ring_edges(n: int, hops: int) -> set[tuple[int, int]]:
  edges = set()
  hops = max(1, min(hops, n - 1))
  for i in range(n):
    for offset in range(1, hops + 1):
      j = (i + offset) % n
      if i != j:
        edges.add(tuple(sorted((i, j))))
  return edges


def complete_edges(n: int) -> set[tuple[int, int]]:
  return {(i, j) for i in range(n) for j in range(i + 1, n)}


def path_edges(n: int) -> set[tuple[int, int]]:
  return {(i, i + 1) for i in range(n - 1)}


def random_edges(n: int, probability: float, rng: random.Random) -> set[tuple[int, int]]:
  edges = set()
  for i in range(n):
    for j in range(i + 1, n):
      if rng.random() < probability:
        edges.add((i, j))
  return edges


def is_connected(n: int, edges: set[tuple[int, int]]) -> bool:
  if n <= 1:
    return True
  adjacency = [[] for _ in range(n)]
  for i, j in edges:
    adjacency[i].append(j)
    adjacency[j].append(i)
  seen = [False] * n
  queue = deque([0])
  seen[0] = True
  while queue:
    node = queue.popleft()
    for nbr in adjacency[node]:
      if not seen[nbr]:
        seen[nbr] = True
        queue.append(nbr)
  return all(seen)


def graph_edges(args, rng: random.Random, round_idx: int) -> set[tuple[int, int]]:
  if args.topology == "ring":
    return ring_edges(args.num_robots, args.ring_hops)
  if args.topology == "complete":
    return complete_edges(args.num_robots)
  if args.topology == "path":
    return path_edges(args.num_robots)
  if args.topology == "random":
    local_rng = rng if args.time_varying else random.Random(args.seed)
    for _ in range(args.max_resample):
      edges = random_edges(args.num_robots, args.probability, local_rng)
      if not args.require_connected or is_connected(args.num_robots, edges):
        return edges
    raise RuntimeError(f"could not sample a connected random graph for round {round_idx}")
  raise ValueError(f"unsupported topology: {args.topology}")


def metropolis_matrix(n: int, edges: set[tuple[int, int]]) -> list[list[float]]:
  degrees = [0] * n
  for i, j in edges:
    degrees[i] += 1
    degrees[j] += 1
  matrix = [[0.0 for _ in range(n)] for _ in range(n)]
  for i, j in edges:
    weight = 1.0 / (1.0 + max(degrees[i], degrees[j]))
    matrix[i][j] = weight
    matrix[j][i] = weight
  for i in range(n):
    matrix[i][i] = 1.0 - sum(matrix[i][j] for j in range(n) if j != i)
  return matrix


def uniform_matrix(n: int, edges: set[tuple[int, int]]) -> list[list[float]]:
  if not edges:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
  degrees = [0] * n
  for i, j in edges:
    degrees[i] += 1
    degrees[j] += 1
  max_degree = max(degrees)
  offdiag = 1.0 / (1.0 + max_degree)
  matrix = [[0.0 for _ in range(n)] for _ in range(n)]
  for i, j in edges:
    matrix[i][j] = offdiag
    matrix[j][i] = offdiag
  for i in range(n):
    matrix[i][i] = 1.0 - sum(matrix[i][j] for j in range(n) if j != i)
  return matrix


def make_matrix(n: int, edges: set[tuple[int, int]], weighting: str) -> list[list[float]]:
  if weighting == "metropolis":
    return metropolis_matrix(n, edges)
  if weighting == "uniform":
    return uniform_matrix(n, edges)
  raise ValueError(f"unsupported weighting: {weighting}")


def diagnostics(matrix: list[list[float]]) -> dict[str, float]:
  n = len(matrix)
  row_errors = [abs(sum(row) - 1.0) for row in matrix]
  col_errors = [abs(sum(matrix[i][j] for i in range(n)) - 1.0) for j in range(n)]
  symmetry = max(
      abs(matrix[i][j] - matrix[j][i]) for i in range(n) for j in range(n)
  ) if n else 0.0
  return {
      "max_row_sum_error": max(row_errors) if row_errors else 0.0,
      "max_col_sum_error": max(col_errors) if col_errors else 0.0,
      "max_symmetry_error": symmetry,
  }


def write_matrix(path: Path, matrix: list[list[float]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(matrix)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-robots", type=int, required=True)
  parser.add_argument("--topology", choices=["ring", "complete", "path", "random"], default="ring")
  parser.add_argument("--ring-hops", type=int, default=1)
  parser.add_argument("--probability", type=float, default=0.2)
  parser.add_argument("--time-varying", action="store_true")
  parser.add_argument("--rounds", type=int, default=1)
  parser.add_argument("--weighting", choices=["metropolis", "uniform"], default="metropolis")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--require-connected", action="store_true", default=True)
  parser.add_argument("--max-resample", type=int, default=1000)
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args()

  if args.num_robots < 1:
    raise ValueError("--num-robots must be positive")
  if args.rounds < 1:
    raise ValueError("--rounds must be positive")

  output_dir = Path(args.output_dir)
  matrix_dir = output_dir / "matrices"
  matrix_dir.mkdir(parents=True, exist_ok=True)
  rng = random.Random(args.seed)

  edge_csv = output_dir / "topology_edges.csv"
  summaries = []
  with edge_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "src", "dst", "weight"])
    cached_edges = None
    for round_idx in range(args.rounds):
      if args.time_varying or cached_edges is None:
        cached_edges = graph_edges(args, rng, round_idx)
      edges = cached_edges
      matrix = make_matrix(args.num_robots, edges, args.weighting)
      for i, j in sorted(edges):
        writer.writerow([round_idx, i, j, matrix[i][j]])
      write_matrix(matrix_dir / f"weight_matrix_round_{round_idx:04d}.csv", matrix)
      info = diagnostics(matrix)
      info.update({
          "round": round_idx,
          "num_edges": len(edges),
          "connected": is_connected(args.num_robots, edges),
      })
      summaries.append(info)

  summary = {
      "num_robots": args.num_robots,
      "topology": args.topology,
      "ring_hops": args.ring_hops,
      "probability": args.probability,
      "time_varying": args.time_varying,
      "rounds": args.rounds,
      "weighting": args.weighting,
      "seed": args.seed,
      "edge_csv": str(edge_csv),
      "matrix_dir": str(matrix_dir),
      "rounds_summary": summaries,
  }
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
