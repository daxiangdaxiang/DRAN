# RIFT-IF Interface Solver

RIFT-IF is the decentralized interface-solve backend for TED-CCI.  Its exact
backend treats the condensed TED-CCI interface least-squares problem as a
clique-tree factor graph and solves it with rootless directed square-root
messages.  The direct interface solver remains an oracle for tests and
regression checks only.

Current implementation status:

- `ted_cci_rift_if` is available through `bench-dcci --initialization-mode`.
- `--interface-backend rift_exact` selects the exact rootless message backend.
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
- CAK, Async-Schur, network-delay simulation, and incremental dirty-message
  updates are planned follow-up backends.

The RIFT exact path must not call `InterfaceDirectSolver::Solve`, must not call
`stackInterfaceSystem*`, and must not use MPI collectives.  Small 2D/3D tests
compare RIFT against centralized CCI, direct-interface oracle results, and the
legacy vectorized RIFT rotation representation.
