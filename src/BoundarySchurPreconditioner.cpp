/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/BoundarySchurPreconditioner.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace DPGO {

namespace {

Matrix solveDenseBoundaryStep(const Matrix &AIn, const Matrix &gradient) {
  if (AIn.rows() != AIn.cols() || AIn.rows() != gradient.cols()) {
    return Matrix::Zero(gradient.rows(), gradient.cols());
  }

  Matrix A = AIn;
  A = 0.5 * (A + A.transpose());
  Matrix rhs = -gradient.transpose();
  Eigen::LDLT<Matrix> ldlt(A);
  Matrix solved;
  if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
    solved = ldlt.solve(rhs);
  } else {
    solved = A.colPivHouseholderQr().solve(rhs);
  }

  if (solved.rows() != A.rows() ||
      solved.cols() != gradient.rows() || !solved.allFinite()) {
    return Matrix::Zero(gradient.rows(), gradient.cols());
  }
  return solved.transpose();
}

Matrix solveDenseLinearSystem(const Matrix &AIn, const Matrix &rhs) {
  Matrix A = 0.5 * (AIn + AIn.transpose());
  Eigen::LDLT<Matrix> ldlt(A);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return Matrix::Zero(rhs.rows(), rhs.cols());
  }
  Matrix solved = ldlt.solve(rhs);
  if (solved.rows() != rhs.rows() || solved.cols() != rhs.cols() ||
      !solved.allFinite()) {
    return Matrix::Zero(rhs.rows(), rhs.cols());
  }
  return solved;
}

Matrix denseQBlock(const SparseMatrix &Q, const std::vector<unsigned> &rows,
                   const std::vector<unsigned> &cols) {
  Matrix block = Matrix::Zero(rows.size(), cols.size());
  for (unsigned r = 0; r < rows.size(); ++r) {
    for (unsigned c = 0; c < cols.size(); ++c) {
      block(static_cast<int>(r), static_cast<int>(c)) =
          Q.coeff(static_cast<int>(rows[r]), static_cast<int>(cols[c]));
    }
  }
  return block;
}

}  // namespace

Matrix solveBoundaryBlockPreconditionedStep(const SparseMatrix &Q,
                                            const Matrix &gradient,
                                            unsigned colStart,
                                            unsigned blockCols,
                                            double damping) {
  if (gradient.cols() != static_cast<int>(blockCols) || blockCols == 0 ||
      colStart + blockCols > static_cast<unsigned>(Q.rows()) ||
      colStart + blockCols > static_cast<unsigned>(Q.cols())) {
    return Matrix::Zero(gradient.rows(), gradient.cols());
  }

  Matrix A = Matrix::Zero(blockCols, blockCols);
  for (unsigned row = 0; row < blockCols; ++row) {
    for (unsigned col = 0; col < blockCols; ++col) {
      A(static_cast<int>(row), static_cast<int>(col)) =
          Q.coeff(static_cast<int>(colStart + row),
                  static_cast<int>(colStart + col));
    }
  }

  const double ridge = std::max(0.0, damping);
  for (unsigned col = 0; col < blockCols; ++col) {
    A(static_cast<int>(col), static_cast<int>(col)) += ridge;
  }
  return solveDenseBoundaryStep(A, gradient);
}

Matrix solveBoundaryLocalSchurPreconditionedStep(const SparseMatrix &Q,
                                                 const Matrix &fullGradient,
                                                 unsigned colStart,
                                                 unsigned blockCols,
                                                 double damping,
                                                 unsigned maxPrivateCols) {
  if (fullGradient.cols() != Q.cols() || blockCols == 0 ||
      colStart + blockCols > static_cast<unsigned>(Q.rows()) ||
      colStart + blockCols > static_cast<unsigned>(Q.cols())) {
    return Matrix::Zero(fullGradient.rows(), static_cast<int>(blockCols));
  }

  std::vector<unsigned> boundaryCols;
  boundaryCols.reserve(blockCols);
  for (unsigned col = 0; col < blockCols; ++col) {
    boundaryCols.push_back(colStart + col);
  }

  const unsigned colEnd = colStart + blockCols;
  std::map<unsigned, double> couplingByCol;
  for (unsigned row = colStart; row < colEnd; ++row) {
    for (SparseMatrix::InnerIterator it(Q, static_cast<int>(row)); it; ++it) {
      const unsigned col = static_cast<unsigned>(it.col());
      if (col < colStart || col >= colEnd) {
        couplingByCol[col] += std::abs(it.value());
      }
    }
  }
  if (couplingByCol.empty()) {
    return solveBoundaryBlockPreconditionedStep(
        Q, fullGradient.block(0, colStart, fullGradient.rows(), blockCols),
        colStart, blockCols, damping);
  }

  std::vector<std::pair<unsigned, double>> coupledCols(couplingByCol.begin(),
                                                       couplingByCol.end());
  std::sort(coupledCols.begin(), coupledCols.end(),
            [](const auto &a, const auto &b) {
              if (a.second == b.second) {
                return a.first < b.first;
              }
              return a.second > b.second;
            });
  if (maxPrivateCols > 0 && coupledCols.size() > maxPrivateCols) {
    coupledCols.resize(maxPrivateCols);
  }
  std::sort(coupledCols.begin(), coupledCols.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<unsigned> privateCols;
  privateCols.reserve(coupledCols.size());
  for (const auto &item : coupledCols) {
    privateCols.push_back(item.first);
  }

  Matrix Qbb = denseQBlock(Q, boundaryCols, boundaryCols);
  Matrix Qbp = denseQBlock(Q, boundaryCols, privateCols);
  Matrix Qpp = denseQBlock(Q, privateCols, privateCols);
  const double ridge = std::max(0.0, damping);
  for (int col = 0; col < Qpp.cols(); ++col) {
    Qpp(col, col) += ridge;
  }

  Matrix privateGradient(fullGradient.rows(), privateCols.size());
  for (unsigned col = 0; col < privateCols.size(); ++col) {
    privateGradient.col(static_cast<int>(col)) =
        fullGradient.col(static_cast<int>(privateCols[col]));
  }
  const Matrix boundaryGradient =
      fullGradient.block(0, colStart, fullGradient.rows(), blockCols);
  const Matrix qppInvQpb = solveDenseLinearSystem(Qpp, Qbp.transpose());
  const Matrix qppInvGp =
      solveDenseLinearSystem(Qpp, privateGradient.transpose());
  if ((Qbp.norm() > 1e-14 && qppInvQpb.norm() <= 1e-14) ||
      (privateGradient.norm() > 1e-14 && qppInvGp.norm() <= 1e-14)) {
    return Matrix::Zero(fullGradient.rows(), static_cast<int>(blockCols));
  }
  Matrix schur = Qbb - Qbp * qppInvQpb;
  for (int col = 0; col < schur.cols(); ++col) {
    schur(col, col) += ridge;
  }
  Eigen::LDLT<Matrix> schurLdlt(0.5 * (schur + schur.transpose()));
  if (schurLdlt.info() != Eigen::Success || !schurLdlt.isPositive()) {
    return Matrix::Zero(fullGradient.rows(), static_cast<int>(blockCols));
  }
  const Matrix effectiveGradient =
      boundaryGradient - (Qbp * qppInvGp).transpose();
  return solveDenseBoundaryStep(schur, effectiveGradient);
}

double boundaryPacketSchurExtraDamping(double schurSensitivity,
                                       double reducedPreconditioner,
                                       double gain,
                                       double maxExtraDamping) {
  if (!std::isfinite(schurSensitivity) || !std::isfinite(reducedPreconditioner) ||
      !std::isfinite(gain) || !std::isfinite(maxExtraDamping) ||
      schurSensitivity <= 0.0 || reducedPreconditioner <= 0.0 ||
      gain <= 0.0 || maxExtraDamping <= 0.0) {
    return 0.0;
  }
  const double extra = gain * schurSensitivity / reducedPreconditioner;
  if (!std::isfinite(extra) || extra <= 0.0) {
    return 0.0;
  }
  return std::min(extra, maxExtraDamping);
}

}  // namespace DPGO
