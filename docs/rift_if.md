# RIFT-IF Interface Solver

RIFT-IF is the decentralized interface-solve backend for TED-CCI.  Its exact
backend treats the condensed TED-CCI interface least-squares problem as a
clique-tree factor graph and solves it with rootless directed square-root
messages.  The direct interface solver remains an oracle for tests and
regression checks only.

Current implementation status:

- `ted_cci_rift_if` is available through `bench-dcci --initialization-mode`.
- `--interface-backend rift_exact` selects the exact rootless message backend.
- `--interface-backend rift_auto` first builds the symbolic clique tree.  If
  the exact backend satisfies the separator/message-byte gate it selects
  `rift_exact`; otherwise the stable-network benchmark path selects
  `rift_cak` and keeps the exact symbolic clique statistics in the report.
  Dynamic-network tests can still explicitly request `rift_async_schur`.
- `--forbid-direct-interface-solver true`,
  `--forbid-global-interface-matrix true`, and
  `--forbid-collectives true` enable deployment guard checks.
- Rotation uses the row-wise multi-RHS interface representation by default in
  `ted_cci_rift_if`.  This solves rotation as `min_Y ||A_R Y - B_R||_F^2`
  with `block_dim=d` and `rhs_dim=d`, then converts the result back to the
  legacy column-major `vec(X)` layout for projection and local
  back-substitution.  Pass `--use-rotation-multi-rhs false` to run the legacy
  vectorized RIFT oracle path.
- The multi-RHS path keeps the same relaxed and projected rotations as the
  vectorized path on the covered 2D/3D regression graphs while reducing the
  3D rotation message payload estimator.
- Local multi-RHS condensation follows the same full-rank interior assumption
  as TED-CCI local QR condensation.  Directed clique messages remain
  rank-aware and retain the residual rows after QR when an eliminated message
  block is rank-deficient.
- Clique hosts are assigned by deterministic ownership majority with
  lowest-robot-id tie breaking.  The current single-process deployment
  accounting models same-host clique-tree edges as local and different-host
  edges as one-hop robot-to-robot routes, and reports host count, maximum hosted
  clique load, cross-host edges, route hops, and routed message bytes.
- `RIFTP2PNetworkSimulator` provides the first solver-neutral event-driven
  transport layer for RIFT messages.  It supports deterministic seeded latency,
  jitter, drop probability, reordering, per-directed-link bandwidth
  serialization, and explicit hold-or-drop behavior while a link is
  disconnected.  Current tests use it around `RIFTRootlessScheduler`; CAK,
  Async-Schur, component fallback, and reconnect merge will consume the same
  transport layer in later phases.
- `RIFTDirtyMessageTracker` provides the first incremental exact-cache slice.
  A changed interface factor marks its assigned clique dirty, invalidates that
  clique's outgoing messages, and then propagates invalidation only along
  directed message dependencies: if `u -> v` is dirty, then `v -> w` is dirty
  for `w != u`.  The reverse message `v -> u` remains reusable unless another
  dirty wave requires it, matching the square-root message definition that
  excludes the destination side from its dependencies.
- `RIFTExactSolver::SolveMatrix` can now accept a reusable directed-message
  cache plus a dirty tracker.  In that mode, unchanged messages are reused as
  local dependencies, dirty or missing messages are recomputed, and the updated
  full cache can be returned for the next update epoch.  The default call path
  still rebuilds all directed messages.
- `--interface-backend rift_cak` selects the first communication-avoiding
  Krylov fallback slice.  It exposes a matrix-free interface normal-operator
  apply, solves the normal equations with CG, and accounts dot products through
  a tree scalar reducer rather than an MPI collective.  This MVP is intended
  for high-treewidth fallback validation on small well-conditioned interface
  systems; s-step/pipelined CAK and stronger preconditioning are later phases.
- `--interface-backend rift_async_schur` selects the first asynchronous
  Schur/RAS fallback slice.  Factor-owner robots repeatedly send
  factor-local gradient and block-diagonal curvature contributions computed
  from cached interface states.  Variable-owner robots apply damped relaxed
  block updates, reject stale messages by sequence number, and retransmit
  current interface states.  When links are down, each live communication
  component only uses factors fully contained in that component and therefore
  reports component-wise consistency rather than global consistency.  Reconnect
  keeps the current interface state and resumes on the merged component as a
  warm start.
- Component fallback and reconnect merge are covered by the async Schur MVP
  tests.  Stronger asynchronous Schur preconditioning and production network
  integration remain follow-up work.

The RIFT exact path must not call `InterfaceDirectSolver::Solve`, must not call
`stackInterfaceSystem*`, and must not use MPI collectives.  Small 2D/3D tests
compare RIFT against centralized CCI, direct-interface oracle results, and the
legacy vectorized RIFT rotation representation.

PR14 seven-dataset benchmarking:

```sh
python scripts/run_rift_if_seven.py \
  --bench-bin build/bin/bench-dcci \
  --output-dir results/rift_if_pr14_seven \
  --num-robots 5 \
  --max-iters 20000 \
  --rel-tol 1e-10 \
  --abs-tol 1e-10
```

The runner covers `parking-garage`, `sphere`, `torus`, `CSAIL`, `inter`,
`manhattan`, and `ais2klinik`.  By default it compares `ted_cci_rift_if`,
`ted_cci_sr_direct`, and `dpcg_cci`; RIFT rows pass the deployment guard flags
and report PR14 alias columns such as `selected_backend`, `rift_cost`,
`pose_diff`, `directed_messages`, `actual_message_bytes`, `symbolic_ms`,
`message_qr_ms`, `belief_solve_ms`, `final_interface_residual`, and
`used_global_matrix/used_direct_solver/used_collective`.
