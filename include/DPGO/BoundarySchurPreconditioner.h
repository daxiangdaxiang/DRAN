/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef BOUNDARYSCHURPRECONDITIONER_H
#define BOUNDARYSCHURPRECONDITIONER_H

#include <DPGO/DPGO_types.h>

namespace DPGO {

/**
 * Solve a compact sender-side boundary block model:
 *
 *   step = -gradient * (Q_BB + damping I)^{-1}
 *
 * Rows are independent lifted coordinates and columns are the pose block
 * coordinates. Invalid inputs return a zero matrix with gradient dimensions.
 */
Matrix solveBoundaryBlockPreconditionedStep(const SparseMatrix &Q,
                                            const Matrix &gradient,
                                            unsigned colStart,
                                            unsigned blockCols,
                                            double damping);

/**
 * Solve a compact local Schur boundary model by eliminating the private columns
 * directly coupled to the requested boundary block.
 */
Matrix solveBoundaryLocalSchurPreconditionedStep(const SparseMatrix &Q,
                                                 const Matrix &fullGradient,
                                                 unsigned colStart,
                                                 unsigned blockCols,
                                                 double damping,
                                                 unsigned maxPrivateCols);

/**
 * Convert a sender-side Schur packet into an extra receiver-side damping term.
 *
 * schurSensitivity measures how strongly the sender boundary block couples to
 * private columns; reducedPreconditioner is the norm of the sender's compact
 * preconditioned response. A high sensitivity with little local mobility should
 * make the receiver's response more conservative.
 */
double boundaryPacketSchurExtraDamping(double schurSensitivity,
                                       double reducedPreconditioner,
                                       double gain,
                                       double maxExtraDamping);

}  // namespace DPGO

#endif
