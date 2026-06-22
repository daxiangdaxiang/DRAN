/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/ReducedRotationQuadraticOptimizer.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SparseCholesky>

namespace DPGO {

using ColMajorSparseMatrix = Eigen::SparseMatrix<double>;

namespace {

bool envFlagEnabled(const char *name, bool defaultValue) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return defaultValue;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  if (value == "0" || value == "false" || value == "off" ||
      value == "no" || value == "disabled" || value == "full" ||
      value == "lifted") {
    return false;
  }
  if (value == "1" || value == "true" || value == "on" ||
      value == "yes" || value == "enabled" || value == "compact") {
    return true;
  }
  return defaultValue;
}

class CachedTranslationSystem {
 public:
  CachedTranslationSystem(const SparseMatrix &QIn, unsigned dIn, unsigned nIn,
                          double dampingIn, double proxDampingIn)
      : Q(QIn),
        d(dIn),
        n(nIn),
        damping(dampingIn),
        proxDamping(std::max(0.0, proxDampingIn)),
        valid(false),
        numericalRidgeUsed(0.0),
        reducedResponseMapReady(false) {
    translationCols.reserve(n);
    rotationCols.reserve(n * d);
    isTranslation.assign(Q.cols(), 0);
    translationIndexByColumn.assign(Q.cols(), -1);
    rotationIndexByColumn.assign(Q.cols(), -1);
    translationColumnEntries.assign(n, {});
    rotationColumnEntries.assign(Q.cols(), {});
    for (unsigned pose = 0; pose < n; ++pose) {
      const int col = static_cast<int>(pose * (d + 1) + d);
      translationCols.push_back(col);
      isTranslation[col] = 1;
      translationIndexByColumn[col] = static_cast<int>(pose);
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int rotationCol = static_cast<int>(pose * (d + 1) + localCol);
        rotationIndexByColumn[rotationCol] =
            static_cast<int>(rotationCols.size());
        rotationCols.push_back(rotationCol);
      }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<Eigen::Triplet<double>> rotationTriplets;
    std::vector<Eigen::Triplet<double>> translationToRotationTriplets;
    const bool precomputeResponseMap =
        envFlagEnabled("DRAN_REDUCED_ROTATION_COMPACT_HVP", false) &&
        envFlagEnabled("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP",
                       false);
    triplets.reserve(Q.nonZeros() + n);
    rotationTriplets.reserve(Q.nonZeros() + rotationCols.size());
    if (precomputeResponseMap) {
      translationToRotationTriplets.reserve(Q.nonZeros());
    }
    Matrix rotationToTranslation;
    if (precomputeResponseMap) {
      rotationToTranslation =
          Matrix::Zero(static_cast<int>(rotationCols.size()),
                       static_cast<int>(n));
    }
    double diagScale = 0.0;
    unsigned diagCount = 0;
    double rotationDiagScale = 0.0;
    unsigned rotationDiagCount = 0;
    std::vector<double> rotationRowAbsSums(rotationCols.size(), 0.0);
    for (int k = 0; k < Q.outerSize(); ++k) {
      for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
        const int rowIndex = static_cast<int>(it.row());
        const int colIndex = static_cast<int>(it.col());
        const bool rowIsTranslation = isTranslation[rowIndex] != 0;
        const bool colIsTranslation = isTranslation[colIndex] != 0;
        const int rotationRow = rotationIndexByColumn[rowIndex];
        const int rotationCol = rotationIndexByColumn[colIndex];
        if (rotationRow >= 0 && rotationCol >= 0) {
          rotationTriplets.emplace_back(rotationRow, rotationCol, it.value());
          rotationRowAbsSums[static_cast<std::size_t>(rotationRow)] +=
              std::abs(it.value());
          if (rotationRow == rotationCol) {
            rotationDiagScale += std::abs(it.value());
            ++rotationDiagCount;
          }
        }
        if (precomputeResponseMap && rowIsTranslation && rotationCol >= 0) {
          const int translationRow = translationIndexByColumn[rowIndex];
          if (translationRow >= 0) {
            translationToRotationTriplets.emplace_back(translationRow,
                                                       rotationCol,
                                                       it.value());
          }
        }
        if (precomputeResponseMap && rotationRow >= 0 && colIsTranslation) {
          const int translationCol = translationIndexByColumn[colIndex];
          if (translationCol >= 0) {
            rotationToTranslation(rotationRow, translationCol) += it.value();
          }
        }
        if (colIsTranslation) {
          const int translationIndex = translationIndexByColumn[colIndex];
          if (rowIsTranslation) {
            const int row = static_cast<int>(it.row() / (d + 1));
            const int col = static_cast<int>(it.col() / (d + 1));
            triplets.emplace_back(row, col, it.value());
            if (row == col) {
              diagScale += std::abs(it.value());
              ++diagCount;
            }
          } else if (translationIndex >= 0) {
            translationColumnEntries[static_cast<std::size_t>(
                translationIndex)].emplace_back(rowIndex, it.value());
          }
        } else {
          rotationColumnEntries[static_cast<std::size_t>(colIndex)]
              .emplace_back(rowIndex, it.value());
        }
      }
    }
    if (diagCount > 0) {
      diagScale /= static_cast<double>(diagCount);
    }
    if (!std::isfinite(diagScale) || diagScale <= 0.0) {
      diagScale = 1.0;
    }
    if (rotationDiagCount > 0) {
      rotationDiagScale /= static_cast<double>(rotationDiagCount);
    }
    if (!std::isfinite(rotationDiagScale) || rotationDiagScale <= 0.0) {
      rotationDiagScale = 1.0;
    }
    double rotationSpectralUpperBound = 0.0;
    for (const double rowAbsSum : rotationRowAbsSums) {
      rotationSpectralUpperBound =
          std::max(rotationSpectralUpperBound, rowAbsSum);
    }
    if (!std::isfinite(rotationSpectralUpperBound) ||
        rotationSpectralUpperBound <= 0.0) {
      rotationSpectralUpperBound = rotationDiagScale;
    }
    const double regularizedCholeskyShift =
        std::max(1e-12, rotationSpectralUpperBound / 1e6);

    for (unsigned attempt = 0; attempt < 6; ++attempt) {
      std::vector<Eigen::Triplet<double>> regularizedTriplets = triplets;
      numericalRidgeUsed = 0.0;
      if (attempt > 0) {
        numericalRidgeUsed =
            std::max(dampingIn, 1e-10 * std::max(1.0, diagScale)) *
            std::pow(10.0, attempt - 1);
      }
      const double diagonalShift = proxDamping + numericalRidgeUsed;
      if (diagonalShift > 0.0) {
        for (unsigned i = 0; i < n; ++i) {
          regularizedTriplets.emplace_back(static_cast<int>(i),
                                           static_cast<int>(i), diagonalShift);
        }
      }
      ColMajorSparseMatrix regularized(n, n);
      regularized.setFromTriplets(regularizedTriplets.begin(),
                                  regularizedTriplets.end());
      regularized.makeCompressed();
      solver.compute(regularized);
      if (solver.info() == Eigen::Success) {
        valid = true;
        break;
      }
    }
    if (!valid) {
      numericalRidgeUsed = 0.0;
    }

    if (precomputeResponseMap) {
      rotationBlock.resize(static_cast<int>(rotationCols.size()),
                           static_cast<int>(rotationCols.size()));
      rotationBlock.setFromTriplets(rotationTriplets.begin(),
                                    rotationTriplets.end());
      rotationBlock.makeCompressed();

      translationToRotationBlock.resize(static_cast<int>(n),
                                        static_cast<int>(rotationCols.size()));
      translationToRotationBlock.setFromTriplets(
          translationToRotationTriplets.begin(),
          translationToRotationTriplets.end());
      translationToRotationBlock.makeCompressed();
    }

    if (precomputeResponseMap && valid && n > 0 && !rotationCols.empty()) {
      Matrix solved = solver.solve(rotationToTranslation.transpose());
      if (solver.info() == Eigen::Success &&
          solved.rows() == static_cast<int>(n) &&
          solved.cols() == static_cast<int>(rotationCols.size()) &&
          solved.allFinite()) {
        reducedResponseMap = solved.transpose();
        reducedResponseMapReady =
            reducedResponseMap.rows() == static_cast<int>(rotationCols.size()) &&
            reducedResponseMap.cols() == static_cast<int>(n);
      }
    }

    for (unsigned attempt = 0; attempt < 6; ++attempt) {
      std::vector<Eigen::Triplet<double>> regularizedRotationTriplets =
          rotationTriplets;
      const double ridge = regularizedCholeskyShift * std::pow(10.0, attempt);
      for (std::size_t i = 0; i < rotationCols.size(); ++i) {
        regularizedRotationTriplets.emplace_back(static_cast<int>(i),
                                                 static_cast<int>(i), ridge);
      }
      ColMajorSparseMatrix regularizedRotation(
          static_cast<int>(rotationCols.size()),
          static_cast<int>(rotationCols.size()));
      regularizedRotation.setFromTriplets(
          regularizedRotationTriplets.begin(),
          regularizedRotationTriplets.end());
      regularizedRotation.makeCompressed();
      rotationSolver.compute(regularizedRotation);
      if (rotationSolver.info() == Eigen::Success) {
        break;
      }
    }
  }

  bool matches(const SparseMatrix &candidateQ, unsigned candidateD,
               unsigned candidateN, double candidateDamping,
               double candidateProxDamping) const {
    if (candidateD != d || candidateN != n ||
        std::abs(candidateDamping - damping) > 0.0 ||
        std::abs(std::max(0.0, candidateProxDamping) - proxDamping) > 0.0 ||
        candidateQ.rows() != Q.rows() || candidateQ.cols() != Q.cols() ||
        candidateQ.nonZeros() != Q.nonZeros()) {
      return false;
    }
    for (int k = 0; k < Q.outerSize(); ++k) {
      SparseMatrix::InnerIterator cachedIt(Q, k);
      SparseMatrix::InnerIterator candidateIt(candidateQ, k);
      for (; cachedIt && candidateIt; ++cachedIt, ++candidateIt) {
        if (cachedIt.row() != candidateIt.row() ||
            cachedIt.col() != candidateIt.col() ||
            cachedIt.value() != candidateIt.value()) {
          return false;
        }
      }
      if (cachedIt || candidateIt) {
        return false;
      }
    }
    return true;
  }

  Matrix recover(const Matrix &Y, const SparseMatrix &G,
                 const QuadraticProblem *problem, bool validateCost) const {
    if (!valid || n == 0) {
      return Y;
    }

    Matrix rhs(Y.rows(), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      rhs.col(static_cast<int>(pose)).setZero();
      for (const auto &entry : translationColumnEntries[pose]) {
        rhs.col(static_cast<int>(pose)).noalias() -=
            entry.second * Y.col(entry.first);
      }
      const int col = translationCols[pose];
      for (int row = 0; row < G.rows(); ++row) {
        rhs(row, static_cast<int>(pose)) -= G.coeff(row, col);
      }
      if (proxDamping > 0.0) {
        rhs.col(static_cast<int>(pose)).noalias() +=
            proxDamping * Y.col(col);
      }
    }
    return writeSolution(Y, rhs, problem, validateCost);
  }

  Matrix response(const Matrix &rotationEta) const {
    Matrix responseMatrix =
        Matrix::Zero(rotationEta.rows(), rotationEta.cols());
    if (!valid || n == 0) {
      return responseMatrix;
    }

    const Matrix solved = responseRows(rotationEta);
    if (solved.rows() != static_cast<int>(n) ||
        solved.cols() != rotationEta.rows()) {
      return responseMatrix;
    }
    for (unsigned pose = 0; pose < n; ++pose) {
      responseMatrix.col(pose * (d + 1) + d) =
          solved.row(static_cast<int>(pose)).transpose();
    }
    return responseMatrix;
  }

  Matrix responseRowsPacked(const Matrix &packedRotationEta) const {
    if (!valid || n == 0) {
      return Matrix();
    }

    Matrix rhs(packedRotationEta.rows(), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      rhs.col(static_cast<int>(pose)).setZero();
      for (const auto &entry : translationColumnEntries[pose]) {
        const int rotationIndex = rotationIndexByColumn[entry.first];
        if (rotationIndex >= 0 &&
            rotationIndex < packedRotationEta.cols()) {
          rhs.col(static_cast<int>(pose)).noalias() -=
              entry.second * packedRotationEta.col(rotationIndex);
        }
      }
    }

    Matrix solved = solver.solve(rhs.transpose());
    if (solver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(n) ||
        solved.cols() != packedRotationEta.rows() || !solved.allFinite()) {
      return Matrix();
    }
    return solved;
  }

  Matrix reducedRotationProductPacked(const Matrix &packedRotationEta) const {
    Matrix packedProduct = Matrix::Zero(
        packedRotationEta.rows(), static_cast<int>(rotationCols.size()));
    if (packedRotationEta.cols() != static_cast<int>(rotationCols.size())) {
      return packedProduct;
    }

    const Matrix translationRows = responseRowsPacked(packedRotationEta);
    const bool hasTranslationRows =
        translationRows.rows() == static_cast<int>(n) &&
        translationRows.cols() == packedRotationEta.rows();

    for (std::size_t packedCol = 0; packedCol < rotationCols.size();
         ++packedCol) {
      const int fullCol = rotationCols[packedCol];
      for (const auto &entry : rotationColumnEntries[fullCol]) {
        const int translationIndex = translationIndexByColumn[entry.first];
        if (translationIndex >= 0 && hasTranslationRows) {
          packedProduct.col(static_cast<int>(packedCol)).noalias() +=
              entry.second *
              translationRows.row(translationIndex).transpose();
          continue;
        }
        const int rotationIndex = rotationIndexByColumn[entry.first];
        if (rotationIndex >= 0 &&
            rotationIndex < packedRotationEta.cols()) {
          packedProduct.col(static_cast<int>(packedCol)).noalias() +=
              entry.second * packedRotationEta.col(rotationIndex);
        }
      }
    }
    return packedProduct;
  }

  Matrix reducedRotationProduct(const Matrix &rotationEta) const {
    if (envFlagEnabled("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP",
                       false) &&
        reducedResponseMapReady) {
      Matrix packedEta(rotationEta.rows(),
                       static_cast<int>(rotationCols.size()));
      for (std::size_t i = 0; i < rotationCols.size(); ++i) {
        packedEta.col(static_cast<int>(i)) = rotationEta.col(rotationCols[i]);
      }

      Matrix packedProduct = packedEta * rotationBlock;
      const Matrix translationRows = -(packedEta * reducedResponseMap);
      packedProduct.noalias() += translationRows * translationToRotationBlock;

      Matrix product = Matrix::Zero(rotationEta.rows(), rotationEta.cols());
      for (std::size_t i = 0; i < rotationCols.size(); ++i) {
        product.col(rotationCols[i]) = packedProduct.col(static_cast<int>(i));
      }
      return product;
    }

    Matrix product = Matrix::Zero(rotationEta.rows(), rotationEta.cols());
    const Matrix translationRows = responseRows(rotationEta);
    const bool hasTranslationRows =
        translationRows.rows() == static_cast<int>(n) &&
        translationRows.cols() == rotationEta.rows();

    for (int col : rotationCols) {
      for (const auto &entry : rotationColumnEntries[col]) {
        const int translationIndex = translationIndexByColumn[entry.first];
        if (translationIndex >= 0 && hasTranslationRows) {
          product.col(col).noalias() +=
              entry.second *
              translationRows.row(translationIndex).transpose();
        } else {
          product.col(col).noalias() +=
              entry.second * rotationEta.col(entry.first);
        }
      }
    }
    return product;
  }

  Matrix rotationProduct(const Matrix &Y) const {
    Matrix product = Matrix::Zero(Y.rows(), Y.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int col = static_cast<int>(pose * (d + 1) + localCol);
        for (const auto &entry : rotationColumnEntries[col]) {
          product.col(col).noalias() += entry.second * Y.col(entry.first);
        }
      }
    }
    return product;
  }

  Matrix rotationGradient(const Matrix &Y, const SparseMatrix &G) const {
    Matrix gradient = rotationProduct(Y);
    for (int k = 0; k < G.outerSize(); ++k) {
      for (SparseMatrix::InnerIterator it(G, k); it; ++it) {
        if (!isTranslation[it.col()]) {
          gradient.coeffRef(it.row(), it.col()) += it.value();
        }
      }
    }
    return gradient;
  }

  Matrix choleskyPrecondition(const Matrix &residual) const {
    Matrix preconditioned = Matrix::Zero(residual.rows(), residual.cols());
    if (rotationSolver.info() != Eigen::Success || rotationCols.empty()) {
      return residual;
    }

    Matrix rhs(residual.rows(), static_cast<int>(rotationCols.size()));
    for (std::size_t i = 0; i < rotationCols.size(); ++i) {
      rhs.col(static_cast<int>(i)) = residual.col(rotationCols[i]);
    }

    Matrix solved = rotationSolver.solve(rhs.transpose());
    if (rotationSolver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(rotationCols.size()) ||
        solved.cols() != residual.rows() || !solved.allFinite()) {
      return residual;
    }
    for (std::size_t i = 0; i < rotationCols.size(); ++i) {
      preconditioned.col(rotationCols[i]) =
          solved.row(static_cast<int>(i)).transpose();
    }
    return preconditioned;
  }

  Matrix choleskyPreconditionPacked(const Matrix &packedResidual) const {
    if (rotationSolver.info() != Eigen::Success || rotationCols.empty() ||
        packedResidual.cols() != static_cast<int>(rotationCols.size())) {
      return packedResidual;
    }

    Matrix solved = rotationSolver.solve(packedResidual.transpose());
    if (rotationSolver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(rotationCols.size()) ||
        solved.cols() != packedResidual.rows() || !solved.allFinite()) {
      return packedResidual;
    }

    Matrix packedPreconditioned(packedResidual.rows(),
                                packedResidual.cols());
    for (std::size_t i = 0; i < rotationCols.size(); ++i) {
      packedPreconditioned.col(static_cast<int>(i)) =
          solved.row(static_cast<int>(i)).transpose();
    }
    return packedPreconditioned;
  }

  bool choleskyPreconditionPackedProjected(const Matrix &base,
                                           const Matrix &packedResidual,
                                           Matrix &projected) const {
    if (rotationSolver.info() != Eigen::Success || rotationCols.empty() ||
        base.rows() != packedResidual.rows() ||
        base.cols() != static_cast<int>((d + 1) * n) ||
        packedResidual.cols() != static_cast<int>(rotationCols.size())) {
      return false;
    }

    Matrix solved = rotationSolver.solve(packedResidual.transpose());
    if (rotationSolver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(rotationCols.size()) ||
        solved.cols() != packedResidual.rows() || !solved.allFinite()) {
      return false;
    }

    projected.resize(packedResidual.rows(), packedResidual.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned fullColStart = pose * (d + 1);
      const unsigned packedColStart = pose * d;
      auto projectedBlock =
          projected.block(0, static_cast<int>(packedColStart),
                          projected.rows(), static_cast<int>(d));
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const std::size_t packedIndex =
            static_cast<std::size_t>(packedColStart + localCol);
        if (packedIndex >= rotationCols.size()) {
          return false;
        }
        projectedBlock.col(static_cast<int>(localCol)) =
            solved.row(static_cast<int>(packedIndex)).transpose();
      }
      const auto R =
          base.block(0, static_cast<int>(fullColStart), base.rows(), d);
      Matrix sym = R.transpose() * projectedBlock;
      sym = (0.5 * (sym + sym.transpose())).eval();
      projectedBlock.noalias() -= R * sym;
    }
    return true;
  }

  bool choleskyPreconditionProjected(const Matrix &base,
                                     const Matrix &residual,
                                     Matrix &projected) const {
    if (rotationSolver.info() != Eigen::Success || rotationCols.empty() ||
        base.rows() != residual.rows() || base.cols() != residual.cols()) {
      return false;
    }

    Matrix rhs(residual.rows(), static_cast<int>(rotationCols.size()));
    for (std::size_t i = 0; i < rotationCols.size(); ++i) {
      rhs.col(static_cast<int>(i)) = residual.col(rotationCols[i]);
    }

    Matrix solved = rotationSolver.solve(rhs.transpose());
    if (rotationSolver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(rotationCols.size()) ||
        solved.cols() != residual.rows() || !solved.allFinite()) {
      return false;
    }

    projected = Matrix::Zero(residual.rows(), residual.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      Matrix solvedBlock(residual.rows(), static_cast<int>(d));
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const std::size_t rotationIndex =
            static_cast<std::size_t>(pose * d + localCol);
        solvedBlock.col(static_cast<int>(localCol)) =
            solved.row(static_cast<int>(rotationIndex)).transpose();
      }
      const auto R = base.block(0, colStart, base.rows(), d);
      Matrix sym = R.transpose() * solvedBlock;
      sym = (0.5 * (sym + sym.transpose())).eval();
      projected.block(0, colStart, projected.rows(), static_cast<int>(d)) =
          solvedBlock - R * sym;
    }
    return true;
  }

  Matrix schurJacobiPrecondition(const Matrix &residual) const {
    Matrix preconditioned = Matrix::Zero(residual.rows(), residual.cols());
    if (!valid || n == 0) {
      return residual;
    }
    ensureSchurJacobiInverseBlocks();
    if (schurJacobiInverseBlocks.size() != n) {
      return residual;
    }

    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      const Matrix &inverseBlock = schurJacobiInverseBlocks[pose];
      if (inverseBlock.rows() != static_cast<int>(d) ||
          inverseBlock.cols() != static_cast<int>(d) ||
          !inverseBlock.allFinite()) {
        return residual;
      }
      const Matrix residualBlock = residual.block(
          0, static_cast<int>(colStart), residual.rows(), static_cast<int>(d));
      preconditioned.block(0, static_cast<int>(colStart),
                           preconditioned.rows(), static_cast<int>(d))
          .noalias() = residualBlock * inverseBlock;
    }

    return preconditioned;
  }

  std::size_t getRecoveryCostValidationCount() const {
    return recoveryCostValidationCount;
  }

 private:
  Matrix responseRows(const Matrix &rotationEta) const {
    if (!valid || n == 0) {
      return Matrix();
    }

    Matrix rhs(rotationEta.rows(), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      rhs.col(static_cast<int>(pose)).setZero();
      for (const auto &entry : translationColumnEntries[pose]) {
        rhs.col(static_cast<int>(pose)).noalias() -=
            entry.second * rotationEta.col(entry.first);
      }
    }

    Matrix solved = solver.solve(rhs.transpose());
    if (solver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(n) ||
        solved.cols() != rotationEta.rows() || !solved.allFinite()) {
      return Matrix();
    }
    return solved;
  }

  void ensureSchurJacobiInverseBlocks() const {
    if (schurJacobiInverseBlocksReady) {
      return;
    }
    schurJacobiInverseBlocks.assign(n, Matrix());
    if (!valid || n == 0) {
      schurJacobiInverseBlocksReady = true;
      return;
    }

    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      Matrix block = Matrix::Zero(static_cast<int>(d), static_cast<int>(d));
      Matrix translationToRotation =
          Matrix::Zero(static_cast<int>(n), static_cast<int>(d));
      Matrix rotationToTranslation =
          Matrix::Zero(static_cast<int>(d), static_cast<int>(n));

      for (unsigned localRow = 0; localRow < d; ++localRow) {
        const int rotationRow = static_cast<int>(colStart + localRow);
        for (unsigned localCol = 0; localCol < d; ++localCol) {
          const int rotationCol = static_cast<int>(colStart + localCol);
          block(localRow, localCol) = Q.coeff(rotationRow, rotationCol);
        }
        for (unsigned translationPose = 0; translationPose < n;
             ++translationPose) {
          const int translationCol = translationCols[translationPose];
          rotationToTranslation(static_cast<int>(localRow),
                                static_cast<int>(translationPose)) =
              Q.coeff(rotationRow, translationCol);
        }
      }
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int rotationCol = static_cast<int>(colStart + localCol);
        for (unsigned translationPose = 0; translationPose < n;
             ++translationPose) {
          const int translationCol = translationCols[translationPose];
          translationToRotation(static_cast<int>(translationPose),
                                static_cast<int>(localCol)) =
              Q.coeff(translationCol, rotationCol);
        }
      }

      Matrix solved = solver.solve(translationToRotation);
      if (solver.info() != Eigen::Success ||
          solved.rows() != translationToRotation.rows() ||
          solved.cols() != translationToRotation.cols() ||
          !solved.allFinite()) {
        schurJacobiInverseBlocks.clear();
        schurJacobiInverseBlocksReady = true;
        return;
      }
      block.noalias() -= rotationToTranslation * solved;
      block = 0.5 * (block + block.transpose());

      double diagScale = block.diagonal().cwiseAbs().mean();
      if (!std::isfinite(diagScale) || diagScale <= 0.0) {
        diagScale = 1.0;
      }

      bool solvedBlock = false;
      for (unsigned attempt = 0; attempt < 6; ++attempt) {
        Matrix regularized = block;
        const double ridge =
            (1e-10 * std::max(1.0, diagScale) + damping) *
            std::pow(10.0, attempt);
        regularized.diagonal().array() += ridge;
        Eigen::LDLT<Matrix> blockSolver(regularized);
        if (blockSolver.info() != Eigen::Success || !blockSolver.isPositive()) {
          continue;
        }
        Matrix inverseBlock =
            blockSolver.solve(Matrix::Identity(static_cast<int>(d),
                                               static_cast<int>(d)));
        if (inverseBlock.rows() == static_cast<int>(d) &&
            inverseBlock.cols() == static_cast<int>(d) &&
            inverseBlock.allFinite()) {
          schurJacobiInverseBlocks[pose] =
              0.5 * (inverseBlock + inverseBlock.transpose());
          solvedBlock = true;
          break;
        }
      }

      if (!solvedBlock) {
        Matrix inverseBlock =
            Matrix::Zero(static_cast<int>(d), static_cast<int>(d));
        for (unsigned localCol = 0; localCol < d; ++localCol) {
          double diagonal = std::abs(block(localCol, localCol));
          if (!std::isfinite(diagonal) || diagonal <= 1e-14) {
            diagonal = 1.0;
          }
          inverseBlock(localCol, localCol) =
              1.0 / (diagonal + damping + 1e-12);
        }
        schurJacobiInverseBlocks[pose] = inverseBlock;
      }
    }
    schurJacobiInverseBlocksReady = true;
  }

  SparseMatrix Q;
  unsigned d;
  unsigned n;
  double damping;
  double proxDamping;
  bool valid;
  double numericalRidgeUsed;
  std::vector<int> translationCols;
  std::vector<int> rotationCols;
  std::vector<char> isTranslation;
  std::vector<int> translationIndexByColumn;
  std::vector<int> rotationIndexByColumn;
  std::vector<std::vector<std::pair<int, double>>> translationColumnEntries;
  std::vector<std::vector<std::pair<int, double>>> rotationColumnEntries;
  ColMajorSparseMatrix rotationBlock;
  ColMajorSparseMatrix translationToRotationBlock;
  Matrix reducedResponseMap;
  bool reducedResponseMapReady;
  mutable Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
  mutable Eigen::SimplicialLDLT<ColMajorSparseMatrix> rotationSolver;
  mutable std::size_t recoveryCostValidationCount{0};
  mutable bool schurJacobiInverseBlocksReady{false};
  mutable std::vector<Matrix> schurJacobiInverseBlocks;

  Matrix writeSolution(const Matrix &Y, const Matrix &rhs,
                       const QuadraticProblem *problem,
                       bool validateCost) const {
    Matrix solved = solver.solve(rhs.transpose());
    if (solver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(n) || solved.cols() != Y.rows() ||
        !solved.allFinite()) {
      return Y;
    }
    Matrix recovered = Y;
    for (unsigned pose = 0; pose < n; ++pose) {
      recovered.col(pose * (d + 1) + d) =
          solved.row(static_cast<int>(pose)).transpose();
    }
    if (!validateCost && numericalRidgeUsed == 0.0) {
      return recovered;
    }
    ++recoveryCostValidationCount;
    const double recoveredCost = problem->f(recovered);
    if (std::isfinite(recoveredCost) &&
        (numericalRidgeUsed == 0.0 || recoveredCost <= problem->f(Y) + 1e-10)) {
      return recovered;
    }
    return Y;
  }

};

}  // namespace

struct ReducedRotationQuadraticOptimizer::TranslationCache {
  struct Entry {
    std::unique_ptr<CachedTranslationSystem> system;
    std::size_t lastUse{0};
  };

  std::vector<Entry> systems;
  std::size_t factorizationCount{0};
  std::size_t accessCounter{0};
  static constexpr std::size_t kMaxSystems = 4;

  const CachedTranslationSystem &get(const SparseMatrix &Q, unsigned d,
                                     unsigned n, double damping,
                                     double proxDamping) {
    ++accessCounter;
    for (Entry &entry : systems) {
      if (entry.system &&
          entry.system->matches(Q, d, n, damping, proxDamping)) {
        entry.lastUse = accessCounter;
        return *entry.system;
      }
    }
    if (systems.size() >= kMaxSystems) {
      auto oldest = std::min_element(
          systems.begin(), systems.end(), [](const Entry &lhs,
                                             const Entry &rhs) {
            return lhs.lastUse < rhs.lastUse;
          });
      systems.erase(oldest);
    }
    Entry entry;
    entry.system =
        std::make_unique<CachedTranslationSystem>(Q, d, n, damping, proxDamping);
    entry.lastUse = accessCounter;
    systems.push_back(std::move(entry));
    ++factorizationCount;
    return *systems.back().system;
  }
};

ReducedRotationQuadraticOptimizer::ReducedRotationQuadraticOptimizer(
    QuadraticProblem *p)
      : problem(p),
      verbose(false),
      recordResultStats(true),
      profileRuntime(false),
      validateTranslationRecoveryCost(true),
      useJacobiPreconditioner(false),
      useCholeskyPreconditioner(false),
      useSchurJacobiPreconditioner(false),
      useCoupledTrustRegionNorm(false),
      useDirectObjectiveEvaluation(false),
      useCurvatureCauchyCandidate(false),
      useCurvatureCauchyFallbackCandidate(false),
      useGradientBoundaryCandidate(true),
      useSurrogateTcgAccept(false),
      skipRedundantCandidateProjection(false),
      maxIterations(1),
      maxAcceptedIterations(1),
      maxCgIterations(50),
      gradientTolerance(1e-2),
      initialRadius(10.0),
      initialDamping(1e-8),
      truncatedCgRelativeTolerance(0.1),
      translationEliminationProxWeight(0.0),
      profileTranslationRecoverySeconds(0.0),
      profileTranslationFactorizationSeconds(0.0),
      profileTranslationCacheLookupSeconds(0.0),
      profileReducedHessianProductSeconds(0.0),
      profileReducedObjectiveSeconds(0.0),
      profileReducedGradientBuildSeconds(0.0),
      profileReducedPreconditionerSeconds(0.0),
      profileReducedProjectionSeconds(0.0),
      profileReducedRetractionSeconds(0.0),
      profileReducedStepNormSeconds(0.0),
      profileReducedTrustRegionScalingSeconds(0.0),
      profileReducedTranslationResponseSeconds(0.0),
      profileReducedRotationProductSeconds(0.0),
      profileTranslationRecoveryCount(0),
      profileTranslationFactorizationCount(0),
      profileTranslationCacheLookupCount(0),
      profileReducedHessianProductCount(0),
      profileReducedObjectiveCount(0),
      profileReducedGradientBuildCount(0),
      profileReducedPreconditionerCount(0),
      profileReducedProjectionCount(0),
      profileReducedRetractionCount(0),
      profileReducedStepNormCount(0),
      profileReducedTrustRegionScalingCount(0),
      profileReducedTranslationResponseCount(0),
      profileReducedRotationProductCount(0),
      profileCurvatureCauchyCandidateCount(0),
      profileCurvatureCauchyAcceptedCount(0),
      profileCurvatureCauchyFallbackCandidateCount(0),
      profileCurvatureCauchyFallbackAcceptedCount(0),
      profileReducedCandidateProjectionSkipCount(0),
      translationCache(std::make_unique<TranslationCache>()) {}

ReducedRotationQuadraticOptimizer::~ReducedRotationQuadraticOptimizer() =
    default;

std::size_t
ReducedRotationQuadraticOptimizer::getTranslationFactorizationCount() const {
  return translationCache ? translationCache->factorizationCount : 0u;
}

std::size_t ReducedRotationQuadraticOptimizer::
    getTranslationRecoveryCostValidationCount() const {
  if (!translationCache) {
    return 0u;
  }
  std::size_t count = 0;
  for (const auto &entry : translationCache->systems) {
    if (entry.system) {
      count += entry.system->getRecoveryCostValidationCount();
    }
  }
  return count;
}

Matrix ReducedRotationQuadraticOptimizer::recoverTranslationsForRotations(
    const Matrix &Y) {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  if (n == 0 || Y.cols() != static_cast<int>((d + 1) * n)) {
    return Y;
  }
  const SparseMatrix &Q = problem->getQRef();
  const SparseMatrix &G = problem->getGRef();
  if (!translationCache) {
    translationCache = std::make_unique<TranslationCache>();
  }
  const CachedTranslationSystem &translationSystem =
      translationCache->get(Q, d, n, initialDamping,
                            translationEliminationProxWeight);
  return translationSystem.recover(Y, G, problem,
                                   validateTranslationRecoveryCost);
}

double ReducedRotationQuadraticOptimizer::evaluateReducedObjective(
    const Matrix &Y) {
  const Matrix recovered = recoverTranslationsForRotations(Y);
  return useDirectObjectiveEvaluation ? evaluateQuadraticObjectiveDirect(recovered)
                                      : problem->f(recovered);
}

double ReducedRotationQuadraticOptimizer::evaluateQuadraticObjectiveDirect(
    const Matrix &Y) const {
  const SparseMatrix &Q = problem->getQRef();
  const SparseMatrix &G = problem->getGRef();
  if (Y.rows() != G.rows() || Y.cols() != Q.rows() ||
      Q.rows() != Q.cols() || G.cols() != Q.cols()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double quadratic = 0.0;
  for (int outer = 0; outer < Q.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(Q, outer); it; ++it) {
      quadratic += it.value() * Y.col(it.row()).dot(Y.col(it.col()));
    }
  }

  double linear = 0.0;
  for (int outer = 0; outer < G.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(G, outer); it; ++it) {
      linear += it.value() * Y(it.row(), it.col());
    }
  }
  return 0.5 * quadratic + linear;
}

Matrix ReducedRotationQuadraticOptimizer::buildReducedGradient(
    const Matrix &Y) {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  if (n == 0 || Y.cols() != static_cast<int>((d + 1) * n)) {
    return Matrix::Zero(Y.rows(), Y.cols());
  }
  const SparseMatrix &Q = problem->getQRef();
  const SparseMatrix &G = problem->getGRef();
  if (!translationCache) {
    translationCache = std::make_unique<TranslationCache>();
  }
  const CachedTranslationSystem &translationSystem =
      translationCache->get(Q, d, n, initialDamping,
                            translationEliminationProxWeight);
  const Matrix recovered =
      translationSystem.recover(Y, G, problem,
                                validateTranslationRecoveryCost);
  return projectRotationTangent(recovered, recovered * Q + G);
}

Matrix ReducedRotationQuadraticOptimizer::applyReducedSchurHv(
    const Matrix &Y, const Matrix &rotationEta) {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  if (n == 0 || Y.cols() != static_cast<int>((d + 1) * n) ||
      rotationEta.rows() != Y.rows() || rotationEta.cols() != Y.cols()) {
    return Matrix::Zero(rotationEta.rows(), rotationEta.cols());
  }
  const SparseMatrix &Q = problem->getQRef();
  if (!translationCache) {
    translationCache = std::make_unique<TranslationCache>();
  }
  const CachedTranslationSystem &translationSystem =
      translationCache->get(Q, d, n, initialDamping,
                            translationEliminationProxWeight);
  const Matrix base = recoverTranslationsForRotations(Y);
  const auto hvProfileStart =
      profileRuntime ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point();
  Matrix hEta;
  if (envFlagEnabled("DRAN_REDUCED_ROTATION_COMPACT_HVP", false)) {
    hEta = translationSystem.reducedRotationProduct(rotationEta);
  } else {
    const Matrix fullEta = rotationEta + translationSystem.response(rotationEta);
    hEta = fullEta * Q;
  }
  if (profileRuntime) {
    profileReducedHessianProductSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      hvProfileStart)
            .count();
    ++profileReducedHessianProductCount;
  }
  return projectRotationTangent(base, hEta);
}

double ReducedRotationQuadraticOptimizer::rotationInnerProduct(
    const Matrix &A, const Matrix &B, unsigned d) {
  double value = 0.0;
  const unsigned n = static_cast<unsigned>(A.cols()) / (d + 1);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    value +=
        (A.block(0, colStart, A.rows(), d)
             .cwiseProduct(B.block(0, colStart, B.rows(), d)))
            .sum();
  }
  return value;
}

Matrix ReducedRotationQuadraticOptimizer::projectRotationTangent(
    const Matrix &Y, const Matrix &Z) const {
  const unsigned r = problem->relaxation_rank();
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  Matrix projected = Matrix::Zero(Z.rows(), Z.cols());
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    const auto R = Y.block(0, colStart, r, d);
    const auto ZR = Z.block(0, colStart, r, d);
    Matrix sym = R.transpose() * ZR;
    sym = (0.5 * (sym + sym.transpose())).eval();
    projected.block(0, colStart, r, d) = ZR - R * sym;
  }
  return projected;
}

Matrix ReducedRotationQuadraticOptimizer::retractRotations(
    const Matrix &Y, const Matrix &eta) const {
  const unsigned r = problem->relaxation_rank();
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  Matrix candidate = Y + eta;

  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    const Matrix block = candidate.block(0, colStart, r, d);
    Eigen::HouseholderQR<Matrix> qr(block);
    Matrix qFull = qr.householderQ() * Matrix::Identity(r, d);
    Matrix rTri = qr.matrixQR().topLeftCorner(d, d)
                      .template triangularView<Eigen::Upper>();
    for (unsigned col = 0; col < d; ++col) {
      if (rTri(col, col) < 0.0) {
        qFull.col(col) *= -1.0;
      }
    }
    candidate.block(0, colStart, r, d) = qFull;
  }
  return candidate;
}

Matrix ReducedRotationQuadraticOptimizer::recoverTranslations(
    const Matrix &Y, const SparseMatrix &Q) const {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  if (n == 0 || Y.cols() != static_cast<int>((d + 1) * n)) {
    return Y;
  }

  std::vector<int> translationCols;
  translationCols.reserve(n);
  std::vector<char> isTranslation(Y.cols(), 0);
  for (unsigned pose = 0; pose < n; ++pose) {
    const int col = static_cast<int>(pose * (d + 1) + d);
    translationCols.push_back(col);
    isTranslation[col] = 1;
  }

  Matrix fixedY = Y;
  for (int col : translationCols) {
    fixedY.col(col).setZero();
  }

  const SparseMatrix G = problem->getG();
  const Matrix fixedTimesQ = fixedY * Q;
  Matrix rhs(Y.rows(), static_cast<int>(n));
  for (unsigned pose = 0; pose < n; ++pose) {
    const int col = translationCols[pose];
    rhs.col(static_cast<int>(pose)) = -fixedTimesQ.col(col);
    for (int row = 0; row < G.rows(); ++row) {
      rhs(row, static_cast<int>(pose)) -= G.coeff(row, col);
    }
  }

  ColMajorSparseMatrix translationQ(n, n);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(Q.nonZeros() + n);
  double diagScale = 0.0;
  unsigned diagCount = 0;
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
      if (!isTranslation[it.row()] || !isTranslation[it.col()]) {
        continue;
      }
      const int row = static_cast<int>(it.row() / (d + 1));
      const int col = static_cast<int>(it.col() / (d + 1));
      triplets.emplace_back(row, col, it.value());
      if (row == col) {
        diagScale += std::abs(it.value());
        ++diagCount;
      }
    }
  }
  if (diagCount > 0) {
    diagScale /= static_cast<double>(diagCount);
  }
  if (!std::isfinite(diagScale) || diagScale <= 0.0) {
    diagScale = 1.0;
  }

  Matrix recovered = Y;
  for (unsigned attempt = 0; attempt < 6; ++attempt) {
    std::vector<Eigen::Triplet<double>> regularizedTriplets = triplets;
    if (attempt > 0) {
      const double ridge =
          std::max(initialDamping, 1e-10 * std::max(1.0, diagScale)) *
          std::pow(10.0, attempt - 1);
      for (unsigned i = 0; i < n; ++i) {
        regularizedTriplets.emplace_back(static_cast<int>(i),
                                         static_cast<int>(i), ridge);
      }
    }

    ColMajorSparseMatrix regularized(n, n);
    regularized.setFromTriplets(regularizedTriplets.begin(),
                                regularizedTriplets.end());
    regularized.makeCompressed();

    Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
    solver.compute(regularized);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    Matrix solved = solver.solve(rhs.transpose());
    if (solver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(n) || solved.cols() != Y.rows() ||
        !solved.allFinite()) {
      continue;
    }
    for (unsigned pose = 0; pose < n; ++pose) {
      recovered.col(pose * (d + 1) + d) =
          solved.row(static_cast<int>(pose)).transpose();
    }
    if (std::isfinite(problem->f(recovered)) &&
        (attempt == 0 || problem->f(recovered) <= problem->f(Y) + 1e-10)) {
      return recovered;
    }
  }

  return Y;
}

Matrix ReducedRotationQuadraticOptimizer::translationResponse(
    const Matrix &rotationEta, const SparseMatrix &Q) const {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  Matrix response = Matrix::Zero(rotationEta.rows(), rotationEta.cols());
  if (n == 0) {
    return response;
  }

  std::vector<int> translationCols;
  translationCols.reserve(n);
  std::vector<char> isTranslation(rotationEta.cols(), 0);
  for (unsigned pose = 0; pose < n; ++pose) {
    const int col = static_cast<int>(pose * (d + 1) + d);
    translationCols.push_back(col);
    isTranslation[col] = 1;
  }

  const Matrix fixedTimesQ = rotationEta * Q;
  Matrix rhs(rotationEta.rows(), static_cast<int>(n));
  for (unsigned pose = 0; pose < n; ++pose) {
    rhs.col(static_cast<int>(pose)) =
        -fixedTimesQ.col(translationCols[pose]);
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(Q.nonZeros() + n);
  double diagScale = 0.0;
  unsigned diagCount = 0;
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
      if (!isTranslation[it.row()] || !isTranslation[it.col()]) {
        continue;
      }
      const int row = static_cast<int>(it.row() / (d + 1));
      const int col = static_cast<int>(it.col() / (d + 1));
      triplets.emplace_back(row, col, it.value());
      if (row == col) {
        diagScale += std::abs(it.value());
        ++diagCount;
      }
    }
  }
  if (diagCount > 0) {
    diagScale /= static_cast<double>(diagCount);
  }
  if (!std::isfinite(diagScale) || diagScale <= 0.0) {
    diagScale = 1.0;
  }

  for (unsigned attempt = 0; attempt < 6; ++attempt) {
    std::vector<Eigen::Triplet<double>> regularizedTriplets = triplets;
    if (attempt > 0) {
      const double ridge =
          std::max(initialDamping, 1e-10 * std::max(1.0, diagScale)) *
          std::pow(10.0, attempt - 1);
      for (unsigned i = 0; i < n; ++i) {
        regularizedTriplets.emplace_back(static_cast<int>(i),
                                         static_cast<int>(i), ridge);
      }
    }
    ColMajorSparseMatrix translationQ(n, n);
    translationQ.setFromTriplets(regularizedTriplets.begin(),
                                 regularizedTriplets.end());
    translationQ.makeCompressed();

    Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
    solver.compute(translationQ);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    Matrix solved = solver.solve(rhs.transpose());
    if (solver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(n) ||
        solved.cols() != rotationEta.rows() || !solved.allFinite()) {
      continue;
    }
    for (unsigned pose = 0; pose < n; ++pose) {
      response.col(pose * (d + 1) + d) =
          solved.row(static_cast<int>(pose)).transpose();
    }
    return response;
  }

  return response;
}

Matrix ReducedRotationQuadraticOptimizer::reducedHessianEta(
    const Matrix &Y, const Matrix &rotationEta,
    const SparseMatrix &Q) const {
  Matrix fullEta = rotationEta + translationResponse(rotationEta, Q);
  Matrix hEta = fullEta * Q;
  return projectRotationTangent(Y, hEta);
}

double ReducedRotationQuadraticOptimizer::reducedStepNorm(
    const Matrix &rotationEta, const SparseMatrix &Q) const {
  if (!useCoupledTrustRegionNorm) {
    return rotationEta.norm();
  }
  return (rotationEta + translationResponse(rotationEta, Q)).norm();
}

Matrix ReducedRotationQuadraticOptimizer::scaleToTrustRegion(
    const Matrix &rotationEta, double radius, const SparseMatrix &Q) const {
  const double norm = reducedStepNorm(rotationEta, Q);
  if (norm <= radius || norm <= 1e-14) {
    return rotationEta;
  }
  return (radius / norm) * rotationEta;
}

Matrix ReducedRotationQuadraticOptimizer::applyJacobiPreconditioner(
    const Matrix &Y, const Matrix &residual, const SparseMatrix &Q) const {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  Matrix preconditioned = Matrix::Zero(residual.rows(), residual.cols());

  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    Matrix block = Matrix::Zero(d, d);
    for (unsigned row = 0; row < d; ++row) {
      for (unsigned col = 0; col < d; ++col) {
        block(row, col) = Q.coeff(static_cast<int>(colStart + row),
                                  static_cast<int>(colStart + col));
      }
    }

    const int translationCol = static_cast<int>(colStart + d);
    const double translationDiagonal = Q.coeff(translationCol, translationCol);
    if (std::abs(translationDiagonal) > 1e-14) {
      for (unsigned row = 0; row < d; ++row) {
        for (unsigned col = 0; col < d; ++col) {
          block(row, col) -=
              Q.coeff(static_cast<int>(colStart + row), translationCol) *
              Q.coeff(translationCol, static_cast<int>(colStart + col)) /
              translationDiagonal;
        }
      }
    }

    block = 0.5 * (block + block.transpose());
    double diagScale = block.diagonal().cwiseAbs().mean();
    if (!std::isfinite(diagScale) || diagScale <= 0.0) {
      diagScale = 1.0;
    }

    const Matrix residualBlock = residual.block(
        0, static_cast<int>(colStart), residual.rows(), static_cast<int>(d));
    bool solved = false;
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
      Matrix regularized = block;
      const double ridge =
          (initialDamping + 1e-10 * std::max(1.0, diagScale)) *
          std::pow(10.0, attempt);
      regularized.diagonal().array() += ridge;
      Eigen::LDLT<Matrix> solver(regularized);
      if (solver.info() != Eigen::Success || !solver.isPositive()) {
        continue;
      }
      Matrix solvedBlock = solver.solve(residualBlock.transpose()).transpose();
      if (solvedBlock.rows() == residualBlock.rows() &&
          solvedBlock.cols() == residualBlock.cols() &&
          solvedBlock.allFinite()) {
        preconditioned.block(0, static_cast<int>(colStart),
                             preconditioned.rows(), static_cast<int>(d)) =
            solvedBlock;
        solved = true;
        break;
      }
    }

    if (!solved) {
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int rotationCol = static_cast<int>(colStart + localCol);
        double diagonal = std::abs(block(localCol, localCol));
        if (!std::isfinite(diagonal) || diagonal <= 1e-14) {
          diagonal = 1.0;
        }
        preconditioned.col(rotationCol) =
            residual.col(rotationCol) / (diagonal + initialDamping);
      }
    }
  }

  return projectRotationTangent(Y, preconditioned);
}

Matrix ReducedRotationQuadraticOptimizer::optimize(const Matrix &Y) {
  result = ROPTResult();
  auto exactObjective = [&](const Matrix &value) {
    const auto objectiveProfileStart =
        profileRuntime ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
    const double objective =
        useDirectObjectiveEvaluation ? evaluateQuadraticObjectiveDirect(value)
                                     : problem->f(value);
    if (profileRuntime) {
      profileReducedObjectiveSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        objectiveProfileStart)
              .count();
      ++profileReducedObjectiveCount;
    }
    return objective;
  };

  if (recordResultStats) {
    result.fInit = exactObjective(Y);
    result.gradNormInit = problem->RieGradNorm(Y);
  } else {
    result.fInit = std::numeric_limits<double>::quiet_NaN();
    result.gradNormInit = std::numeric_limits<double>::quiet_NaN();
  }
  const auto startTime = std::chrono::steady_clock::now();

  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  const SparseMatrix &Q = problem->getQRef();
  const SparseMatrix &G = problem->getGRef();
  const std::size_t factorizationCountBefore =
      profileRuntime && translationCache ? translationCache->factorizationCount
                                         : 0u;
  const auto cacheLookupProfileStart =
      profileRuntime ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point();
  const CachedTranslationSystem &translationSystem =
      translationCache->get(Q, d, n, initialDamping,
                            translationEliminationProxWeight);
  if (profileRuntime) {
    const double cacheLookupProfileSec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      cacheLookupProfileStart)
            .count();
    profileTranslationCacheLookupSeconds += cacheLookupProfileSec;
    ++profileTranslationCacheLookupCount;
    const std::size_t factorizationCountAfter =
        translationCache ? translationCache->factorizationCount : 0u;
    if (factorizationCountAfter > factorizationCountBefore) {
      profileTranslationFactorizationSeconds += cacheLookupProfileSec;
      profileTranslationFactorizationCount +=
          factorizationCountAfter - factorizationCountBefore;
    }
  }
  std::string reducedHvpMode = "euclidean";
  if (const char *hvpEnv = std::getenv("DRAN_REDUCED_ROTATION_HVP")) {
    reducedHvpMode = hvpEnv;
    std::transform(reducedHvpMode.begin(), reducedHvpMode.end(),
                   reducedHvpMode.begin(), ::tolower);
  }
  const bool useRiemannianHvp =
      reducedHvpMode == "riemannian" || reducedHvpMode == "roptlib" ||
      reducedHvpMode == "exact";
  const bool useCompactReducedHvp =
      envFlagEnabled("DRAN_REDUCED_ROTATION_COMPACT_HVP", false);
  const bool usePackedCholeskyProjection =
      envFlagEnabled("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT", false);
  const bool usePackedReducedTcg =
      envFlagEnabled("DRAN_REDUCED_ROTATION_PACKED_TCG", false);
  const bool usePackedRiemannianHvp =
      envFlagEnabled("DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP", false);
  auto recoverCached = [&](const Matrix &value) {
    if (!profileRuntime) {
      return translationSystem.recover(value, G, problem,
                                       validateTranslationRecoveryCost);
    }
    const auto recoveryProfileStart = std::chrono::steady_clock::now();
    Matrix recovered = translationSystem.recover(
        value, G, problem, validateTranslationRecoveryCost);
    profileTranslationRecoverySeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      recoveryProfileStart)
            .count();
	    ++profileTranslationRecoveryCount;
	    return recovered;
	  };
  auto projectCached = [&](const Matrix &base, const Matrix &value) {
    if (!profileRuntime) {
      return projectRotationTangent(base, value);
    }
    const auto projectProfileStart = std::chrono::steady_clock::now();
    Matrix projected = projectRotationTangent(base, value);
    profileReducedProjectionSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      projectProfileStart)
            .count();
    ++profileReducedProjectionCount;
    return projected;
  };
  auto packRotations = [&](const Matrix &value) {
    Matrix packed(value.rows(), static_cast<int>(d * n));
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned fullColStart = pose * (d + 1);
      const unsigned packedColStart = pose * d;
      packed.block(0, static_cast<int>(packedColStart), packed.rows(),
                   static_cast<int>(d)) =
          value.block(0, static_cast<int>(fullColStart), value.rows(),
                      static_cast<int>(d));
    }
    return packed;
  };
  auto unpackRotationStep = [&](const Matrix &packed) {
    Matrix full = Matrix::Zero(packed.rows(), static_cast<int>((d + 1) * n));
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned fullColStart = pose * (d + 1);
      const unsigned packedColStart = pose * d;
      full.block(0, static_cast<int>(fullColStart), full.rows(),
                 static_cast<int>(d)) =
          packed.block(0, static_cast<int>(packedColStart), packed.rows(),
                       static_cast<int>(d));
    }
    return full;
  };
  auto projectPackedCached = [&](const Matrix &base,
                                 const Matrix &packedValue) {
    if (!profileRuntime) {
      Matrix projected = Matrix::Zero(packedValue.rows(), packedValue.cols());
      for (unsigned pose = 0; pose < n; ++pose) {
        const unsigned fullColStart = pose * (d + 1);
        const unsigned packedColStart = pose * d;
        const auto R =
            base.block(0, static_cast<int>(fullColStart), base.rows(), d);
        const auto ZR = packedValue.block(
            0, static_cast<int>(packedColStart), packedValue.rows(),
            static_cast<int>(d));
        Matrix sym = R.transpose() * ZR;
        sym = (0.5 * (sym + sym.transpose())).eval();
        projected.block(0, static_cast<int>(packedColStart),
                        projected.rows(), static_cast<int>(d)) =
            ZR - R * sym;
      }
      return projected;
    }
    const auto projectProfileStart = std::chrono::steady_clock::now();
    Matrix projected = Matrix::Zero(packedValue.rows(), packedValue.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned fullColStart = pose * (d + 1);
      const unsigned packedColStart = pose * d;
      const auto R =
          base.block(0, static_cast<int>(fullColStart), base.rows(), d);
      const auto ZR =
          packedValue.block(0, static_cast<int>(packedColStart),
                            packedValue.rows(), static_cast<int>(d));
      Matrix sym = R.transpose() * ZR;
      sym = (0.5 * (sym + sym.transpose())).eval();
      projected.block(0, static_cast<int>(packedColStart),
                      projected.rows(), static_cast<int>(d)) =
          ZR - R * sym;
    }
    profileReducedProjectionSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      projectProfileStart)
            .count();
    ++profileReducedProjectionCount;
    return projected;
  };
  auto packedRiemannianCorrection =
      [&](const Matrix &base, const Matrix &packedEta,
          const Matrix &packedEuclideanGradient) {
    Matrix correction = Matrix::Zero(packedEta.rows(), packedEta.cols());
    if (packedEta.rows() != packedEuclideanGradient.rows() ||
        packedEta.cols() != packedEuclideanGradient.cols()) {
      return correction;
    }
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned fullColStart = pose * (d + 1);
      const unsigned packedColStart = pose * d;
      const auto R =
          base.block(0, static_cast<int>(fullColStart), base.rows(), d);
      const auto eta =
          packedEta.block(0, static_cast<int>(packedColStart),
                          packedEta.rows(), static_cast<int>(d));
      const auto euclideanGradient =
          packedEuclideanGradient.block(0, static_cast<int>(packedColStart),
                                        packedEuclideanGradient.rows(),
                                        static_cast<int>(d));
      Matrix sym = R.transpose() * euclideanGradient;
      sym = (0.5 * (sym + sym.transpose())).eval();
      correction.block(0, static_cast<int>(packedColStart),
                       correction.rows(), static_cast<int>(d)) = eta * sym;
    }
    return correction;
  };
  auto retractCached = [&](const Matrix &base, const Matrix &eta) {
    if (!profileRuntime) {
      return retractRotations(base, eta);
    }
    const auto retractProfileStart = std::chrono::steady_clock::now();
    Matrix retracted = retractRotations(base, eta);
    profileReducedRetractionSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      retractProfileStart)
            .count();
    ++profileReducedRetractionCount;
    return retracted;
  };
  auto stepNormCached = [&](const Matrix &rotationEta) {
    if (!profileRuntime) {
      return reducedStepNorm(rotationEta, Q);
    }
    const auto normProfileStart = std::chrono::steady_clock::now();
    const double norm = reducedStepNorm(rotationEta, Q);
    profileReducedStepNormSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      normProfileStart)
            .count();
    ++profileReducedStepNormCount;
    return norm;
  };
  auto packedInnerProduct = [&](const Matrix &A, const Matrix &B) {
    return (A.cwiseProduct(B)).sum();
  };
  auto stepNormPackedCached = [&](const Matrix &packedRotationEta) {
    if (useCoupledTrustRegionNorm) {
      return stepNormCached(unpackRotationStep(packedRotationEta));
    }
    if (!profileRuntime) {
      return packedRotationEta.norm();
    }
    const auto normProfileStart = std::chrono::steady_clock::now();
    const double norm = packedRotationEta.norm();
    profileReducedStepNormSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      normProfileStart)
            .count();
    ++profileReducedStepNormCount;
    return norm;
  };
  auto scaleToTrustRegionCached = [&](const Matrix &rotationEta,
                                      double radius) -> Matrix {
    const double norm = stepNormCached(rotationEta);
    if (norm <= radius || norm <= 1e-14) {
      return rotationEta;
    }
    if (!profileRuntime) {
      return (radius / norm) * rotationEta;
    }
    const auto scaleProfileStart = std::chrono::steady_clock::now();
    Matrix scaled = (radius / norm) * rotationEta;
    profileReducedTrustRegionScalingSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      scaleProfileStart)
            .count();
    ++profileReducedTrustRegionScalingCount;
    return scaled;
  };
  auto scalePackedToTrustRegionCached =
      [&](const Matrix &packedRotationEta, double radius) -> Matrix {
    const double norm = stepNormPackedCached(packedRotationEta);
    if (norm <= radius || norm <= 1e-14) {
      return packedRotationEta;
    }
    if (!profileRuntime) {
      return (radius / norm) * packedRotationEta;
    }
    const auto scaleProfileStart = std::chrono::steady_clock::now();
    Matrix scaled = (radius / norm) * packedRotationEta;
    profileReducedTrustRegionScalingSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      scaleProfileStart)
            .count();
    ++profileReducedTrustRegionScalingCount;
    return scaled;
  };
  auto hessianCached = [&](const Matrix &base, const Matrix &rotationEta) {
    const auto hvProfileStart =
        profileRuntime ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
    Matrix hEta;
    if (useRiemannianHvp) {
      Matrix translationEta;
      {
        const auto responseProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        translationEta = translationSystem.response(rotationEta);
        if (profileRuntime) {
          profileReducedTranslationResponseSeconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            responseProfileStart)
                  .count();
          ++profileReducedTranslationResponseCount;
        }
      }
      Matrix fullEta = rotationEta + translationEta;
      {
        const auto productProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        hEta = translationSystem.rotationProduct(fullEta);
        if (profileRuntime) {
          profileReducedRotationProductSeconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            productProfileStart)
                  .count();
          ++profileReducedRotationProductCount;
        }
      }
      LiftedSEVariable point(problem->relaxation_rank(), d, n);
      LiftedSEVector etaVec(problem->relaxation_rank(), d, n);
      LiftedSEVector gradVec(problem->relaxation_rank(), d, n);
      LiftedSEVector hVec(problem->relaxation_rank(), d, n);
      point.setData(base);
      etaVec.setData(fullEta);
      const ROPTLIB::Problem *baseProblem = problem;
      baseProblem->RieGrad(point.var(), gradVec.vec());
      baseProblem->HessianEta(point.var(), etaVec.vec(), hVec.vec());
      Matrix riemannianHeta = hVec.getData();
      if (riemannianHeta.rows() == base.rows() &&
          riemannianHeta.cols() == base.cols() &&
          riemannianHeta.allFinite()) {
        hEta = riemannianHeta;
      }
    } else if (useCompactReducedHvp) {
      const auto productProfileStart =
          profileRuntime ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point();
      hEta = translationSystem.reducedRotationProduct(rotationEta);
      if (profileRuntime) {
        profileReducedRotationProductSeconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          productProfileStart)
                .count();
        ++profileReducedRotationProductCount;
      }
    } else {
      Matrix translationEta;
      {
        const auto responseProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        translationEta = translationSystem.response(rotationEta);
        if (profileRuntime) {
          profileReducedTranslationResponseSeconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            responseProfileStart)
                  .count();
          ++profileReducedTranslationResponseCount;
        }
      }
      Matrix fullEta = rotationEta + translationEta;
      {
        const auto productProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        hEta = translationSystem.rotationProduct(fullEta);
        if (profileRuntime) {
          profileReducedRotationProductSeconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            productProfileStart)
                  .count();
          ++profileReducedRotationProductCount;
        }
      }
    }
    if (profileRuntime) {
      profileReducedHessianProductSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        hvProfileStart)
              .count();
      ++profileReducedHessianProductCount;
    }
    return projectCached(base, hEta);
  };
  Matrix packedCurrentEuclideanGradient;
  auto hessianPackedCached = [&](const Matrix &base,
                                 const Matrix &packedRotationEta) {
    if (useRiemannianHvp) {
      return packRotations(
          hessianCached(base, unpackRotationStep(packedRotationEta)));
    }
    const auto hvProfileStart =
        profileRuntime ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
    Matrix packedHEta;
    {
      const auto productProfileStart =
          profileRuntime ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point();
      packedHEta =
          translationSystem.reducedRotationProductPacked(packedRotationEta);
      if (usePackedRiemannianHvp) {
        packedHEta.noalias() -= packedRiemannianCorrection(
            base, packedRotationEta, packedCurrentEuclideanGradient);
      }
      if (profileRuntime) {
        profileReducedRotationProductSeconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          productProfileStart)
                .count();
        ++profileReducedRotationProductCount;
      }
    }
    if (profileRuntime) {
      profileReducedHessianProductSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        hvProfileStart)
              .count();
      ++profileReducedHessianProductCount;
    }
    return projectPackedCached(base, packedHEta);
  };
  auto solveCgCached = [&](const Matrix &base, const Matrix &grad,
                           double trustRadius) -> Matrix {
    Matrix eta = Matrix::Zero(grad.rows(), grad.cols());
    Matrix residual = grad;
    double residualNormSq = rotationInnerProduct(residual, residual, d);
    const double residualNorm0 = std::sqrt(std::max(0.0, residualNormSq));
    const double cgTol = std::max(
        1e-12, std::max(0.0, truncatedCgRelativeTolerance) * residualNorm0);
    if (residualNorm0 <= cgTol || trustRadius <= 0.0) {
      return eta;
    }

	    auto applySelectedPreconditioner = [&](const Matrix &value) {
	      if (useCholeskyPreconditioner) {
	        const auto preconditionerProfileStart =
	            profileRuntime ? std::chrono::steady_clock::now()
	                           : std::chrono::steady_clock::time_point();
	        Matrix preconditioned;
          const bool alreadyProjected =
              usePackedCholeskyProjection &&
              translationSystem.choleskyPreconditionProjected(base, value,
                                                              preconditioned);
          if (!alreadyProjected) {
            preconditioned = translationSystem.choleskyPrecondition(value);
          }
	        if (profileRuntime) {
	          profileReducedPreconditionerSeconds +=
	              std::chrono::duration<double>(
	                  std::chrono::steady_clock::now() -
	                  preconditionerProfileStart)
	                  .count();
	          ++profileReducedPreconditionerCount;
	        }
	        return alreadyProjected ? preconditioned
                                  : projectCached(base, preconditioned);
	      }
	      if (useSchurJacobiPreconditioner) {
	        const auto preconditionerProfileStart =
	            profileRuntime ? std::chrono::steady_clock::now()
	                           : std::chrono::steady_clock::time_point();
	        Matrix preconditioned =
	            translationSystem.schurJacobiPrecondition(value);
	        if (profileRuntime) {
	          profileReducedPreconditionerSeconds +=
	              std::chrono::duration<double>(
	                  std::chrono::steady_clock::now() -
	                  preconditionerProfileStart)
	                  .count();
	          ++profileReducedPreconditionerCount;
	        }
	        return projectCached(base, preconditioned);
	      }
	      if (useJacobiPreconditioner) {
	        const auto preconditionerProfileStart =
	            profileRuntime ? std::chrono::steady_clock::now()
	                           : std::chrono::steady_clock::time_point();
	        Matrix preconditioned = applyJacobiPreconditioner(base, value, Q);
	        if (profileRuntime) {
	          profileReducedPreconditionerSeconds +=
	              std::chrono::duration<double>(
	                  std::chrono::steady_clock::now() -
	                  preconditionerProfileStart)
	                  .count();
	          ++profileReducedPreconditionerCount;
	        }
	        return preconditioned;
	      }
	      return value;
	    };

    Matrix z = applySelectedPreconditioner(residual);
    double rz = rotationInnerProduct(residual, z, d);
    if (!std::isfinite(rz) || rz <= 1e-20) {
      return eta;
    }
    Matrix direction = -z;
    for (int cgIter = 0; cgIter < std::max(1, maxCgIterations); ++cgIter) {
      Matrix hDir = hessianCached(base, direction);
      hDir += initialDamping * direction;
	      const double curvature = rotationInnerProduct(direction, hDir, d);
	      if (curvature <= 1e-14) {
	        return scaleToTrustRegionCached(eta + direction, trustRadius);
	      }
	      const double alpha = rz / curvature;
	      Matrix nextEta = eta + alpha * direction;
	      if (stepNormCached(nextEta) >= trustRadius) {
	        return scaleToTrustRegionCached(nextEta, trustRadius);
	      }

      eta = nextEta;
      Matrix nextResidual = residual + alpha * hDir;
      const double nextResidualNormSq =
          rotationInnerProduct(nextResidual, nextResidual, d);
      if (std::sqrt(std::max(0.0, nextResidualNormSq)) <= cgTol) {
        return eta;
      }
      Matrix nextZ = applySelectedPreconditioner(nextResidual);
      const double nextRz = rotationInnerProduct(nextResidual, nextZ, d);
      if (!std::isfinite(nextRz) || nextRz <= 1e-20) {
        return eta;
      }
      const double beta = nextRz / rz;
      direction = -nextZ + beta * direction;
      residual = nextResidual;
      residualNormSq = nextResidualNormSq;
      rz = nextRz;
    }
    return eta;
  };
  auto solveCgPackedCached = [&](const Matrix &base, const Matrix &packedGrad,
                                 double trustRadius) -> Matrix {
    Matrix eta = Matrix::Zero(packedGrad.rows(), packedGrad.cols());
    Matrix residual = packedGrad;
    double residualNormSq = packedInnerProduct(residual, residual);
    const double residualNorm0 = std::sqrt(std::max(0.0, residualNormSq));
    const double cgTol = std::max(
        1e-12, std::max(0.0, truncatedCgRelativeTolerance) * residualNorm0);
    if (residualNorm0 <= cgTol || trustRadius <= 0.0) {
      return eta;
    }

    auto applySelectedPackedPreconditioner = [&](const Matrix &value) {
      if (useCholeskyPreconditioner) {
        const auto preconditionerProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        Matrix preconditioned;
        const bool alreadyProjected =
            usePackedCholeskyProjection &&
            translationSystem.choleskyPreconditionPackedProjected(
                base, value, preconditioned);
        if (!alreadyProjected) {
          preconditioned = translationSystem.choleskyPreconditionPacked(value);
        }
        if (profileRuntime) {
          profileReducedPreconditionerSeconds +=
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() -
                  preconditionerProfileStart)
                  .count();
          ++profileReducedPreconditionerCount;
        }
        return alreadyProjected ? preconditioned
                                : projectPackedCached(base, preconditioned);
      }
      if (useSchurJacobiPreconditioner) {
        const auto preconditionerProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        Matrix preconditioned = translationSystem.schurJacobiPrecondition(
            unpackRotationStep(value));
        if (profileRuntime) {
          profileReducedPreconditionerSeconds +=
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() -
                  preconditionerProfileStart)
                  .count();
          ++profileReducedPreconditionerCount;
        }
        return projectPackedCached(base, packRotations(preconditioned));
      }
      if (useJacobiPreconditioner) {
        const auto preconditionerProfileStart =
            profileRuntime ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        Matrix preconditioned =
            applyJacobiPreconditioner(base, unpackRotationStep(value), Q);
        if (profileRuntime) {
          profileReducedPreconditionerSeconds +=
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() -
                  preconditionerProfileStart)
                  .count();
          ++profileReducedPreconditionerCount;
        }
        return packRotations(preconditioned);
      }
      return value;
    };

    Matrix z = applySelectedPackedPreconditioner(residual);
    double rz = packedInnerProduct(residual, z);
    if (!std::isfinite(rz) || rz <= 1e-20) {
      return eta;
    }
    Matrix direction = -z;
    for (int cgIter = 0; cgIter < std::max(1, maxCgIterations); ++cgIter) {
      Matrix hDir = hessianPackedCached(base, direction);
      hDir += initialDamping * direction;
      const double curvature = packedInnerProduct(direction, hDir);
      if (curvature <= 1e-14) {
        return scalePackedToTrustRegionCached(eta + direction, trustRadius);
      }
      const double alpha = rz / curvature;
      Matrix nextEta = eta + alpha * direction;
      if (stepNormPackedCached(nextEta) >= trustRadius) {
        return scalePackedToTrustRegionCached(nextEta, trustRadius);
      }

      eta = nextEta;
      Matrix nextResidual = residual + alpha * hDir;
      const double nextResidualNormSq =
          packedInnerProduct(nextResidual, nextResidual);
      if (std::sqrt(std::max(0.0, nextResidualNormSq)) <= cgTol) {
        return eta;
      }
      Matrix nextZ = applySelectedPackedPreconditioner(nextResidual);
      const double nextRz = packedInnerProduct(nextResidual, nextZ);
      if (!std::isfinite(nextRz) || nextRz <= 1e-20) {
        return eta;
      }
      const double beta = nextRz / rz;
      direction = -nextZ + beta * direction;
      residual = nextResidual;
      residualNormSq = nextResidualNormSq;
      rz = nextRz;
    }
    return eta;
  };

  Matrix X = recoverCached(Y);
  double radius = std::max(1e-12, initialRadius);
  const double maxRadius = std::max(radius, 5.0 * radius);

  enum class CandidateSource {
    Tcg,
    CurvatureCauchy,
    GradientBoundary,
    CurvatureFallback,
  };

  unsigned acceptedIterations = 0;
	  for (unsigned iter = 0;
	       iter < maxIterations && acceptedIterations < maxAcceptedIterations;
	       ++iter) {
	    const double f = exactObjective(X);
	    Matrix gradientEuclidean;
	    {
	      const auto gradientProfileStart =
	          profileRuntime ? std::chrono::steady_clock::now()
	                         : std::chrono::steady_clock::time_point();
	      gradientEuclidean = translationSystem.rotationGradient(X, G);
	      if (profileRuntime) {
	        profileReducedGradientBuildSeconds +=
	            std::chrono::duration<double>(
	                std::chrono::steady_clock::now() - gradientProfileStart)
	                .count();
	        ++profileReducedGradientBuildCount;
	      }
	    }
	    Matrix grad = projectCached(X, gradientEuclidean);
    if (usePackedReducedTcg) {
      packedCurrentEuclideanGradient = packRotations(gradientEuclidean);
    }
    const double gradNorm =
        std::sqrt(std::max(0.0, rotationInnerProduct(grad, grad, d)));
    if (gradNorm < gradientTolerance) {
      break;
    }

    bool accepted = false;
    for (unsigned attempt = 0; attempt < 12 && radius >= 1e-12; ++attempt) {
      std::vector<std::pair<Matrix, CandidateSource>> candidates;
      if (usePackedReducedTcg) {
        const Matrix packedGrad = packRotations(grad);
        candidates.push_back(
            {unpackRotationStep(solveCgPackedCached(X, packedGrad, radius)),
             CandidateSource::Tcg});
      } else {
        candidates.push_back({solveCgCached(X, grad, radius),
                              CandidateSource::Tcg});
      }
      if (useCurvatureCauchyCandidate) {
        Matrix hGrad = hessianCached(X, grad);
        hGrad += initialDamping * grad;
        const double curvature = rotationInnerProduct(grad, hGrad, d);
        const double gradNormSq = rotationInnerProduct(grad, grad, d);
        if (std::isfinite(curvature) && curvature > 1e-14 &&
            std::isfinite(gradNormSq) && gradNormSq > 1e-20) {
          const double alpha = gradNormSq / curvature;
          Matrix cauchyEta = -alpha * grad;
          if (std::isfinite(alpha) && alpha > 0.0 && cauchyEta.allFinite() &&
              stepNormCached(cauchyEta) > 1e-14) {
            candidates.push_back({std::move(cauchyEta),
                                  CandidateSource::CurvatureCauchy});
            if (profileRuntime) {
              ++profileCurvatureCauchyCandidateCount;
            }
          }
        }
      }
      if (useGradientBoundaryCandidate) {
        candidates.push_back({-(radius / std::max(gradNorm, 1e-14)) * grad,
                              CandidateSource::GradientBoundary});
      }

      double bestCost = std::numeric_limits<double>::infinity();
      double bestRho = -std::numeric_limits<double>::infinity();
      Matrix bestCandidate = X;
      Matrix bestEta = Matrix::Zero(X.rows(), X.cols());
      bool hasCandidate = false;
      bool bestIsCurvatureCauchy = false;
      bool bestIsCurvatureFallback = false;
      bool hasFallbackCurvature = false;
      double fallbackCurvature = std::numeric_limits<double>::quiet_NaN();

      for (auto candidateEntry : candidates) {
        Matrix eta = std::move(candidateEntry.first);
        const CandidateSource source = candidateEntry.second;
        const bool candidateProjectionIsRedundant =
            source == CandidateSource::CurvatureCauchy ||
            source == CandidateSource::GradientBoundary;
        if (skipRedundantCandidateProjection &&
            candidateProjectionIsRedundant) {
          if (profileRuntime) {
            ++profileReducedCandidateProjectionSkipCount;
          }
        } else {
          eta = projectCached(X, eta);
        }
	        const double etaNorm = stepNormCached(eta);
	        if (etaNorm <= 1e-14) {
	          continue;
	        }
	        eta = scaleToTrustRegionCached(eta, radius);

        Matrix hEta = hessianCached(X, eta);
        if (source == CandidateSource::GradientBoundary) {
          const double gradNormSq = rotationInnerProduct(grad, grad, d);
          const double scale = -rotationInnerProduct(eta, grad, d) /
                               std::max(gradNormSq, 1e-20);
          if (std::isfinite(scale) && scale > 1e-14 &&
              std::isfinite(gradNormSq) && gradNormSq > 1e-20) {
            const double curvature =
                -rotationInnerProduct(grad, hEta, d) / scale +
                initialDamping * gradNormSq;
            if (std::isfinite(curvature) && curvature > 1e-14) {
              fallbackCurvature = curvature;
              hasFallbackCurvature = true;
            }
          }
        }
        double modelDecrease =
            -(rotationInnerProduct(grad, eta, d) +
              0.5 * rotationInnerProduct(eta, hEta, d));
        if (modelDecrease <= 1e-14) {
          modelDecrease = -rotationInnerProduct(grad, eta, d);
        }
        if (modelDecrease <= 1e-14) {
          continue;
        }

        Matrix candidate = recoverCached(retractCached(X, eta));
        if (useSurrogateTcgAccept && candidates.size() == 1 &&
            source == CandidateSource::Tcg &&
            !useCurvatureCauchyCandidate &&
            !useCurvatureCauchyFallbackCandidate &&
            !useGradientBoundaryCandidate && candidate.rows() == X.rows() &&
            candidate.cols() == X.cols() && candidate.allFinite()) {
          bestCost = f - modelDecrease;
          bestRho = 1.0;
          bestCandidate = candidate;
          bestEta = eta;
          hasCandidate = true;
          break;
        }
        const double candidateCost = exactObjective(candidate);
        const double actualDecrease = f - candidateCost;
        const double rho = actualDecrease / modelDecrease;
        if (std::isfinite(candidateCost) && actualDecrease > 0.0 &&
            rho > 0.05 && candidateCost < bestCost) {
          bestCost = candidateCost;
          bestRho = rho;
          bestCandidate = candidate;
          bestEta = eta;
          hasCandidate = true;
          bestIsCurvatureCauchy =
              source == CandidateSource::CurvatureCauchy;
          bestIsCurvatureFallback =
              source == CandidateSource::CurvatureFallback;
        }
      }

      if (!hasCandidate && useCurvatureCauchyFallbackCandidate &&
          hasFallbackCurvature) {
        const double gradNormSq = rotationInnerProduct(grad, grad, d);
        const double alpha = gradNormSq / fallbackCurvature;
        Matrix eta = -alpha * grad;
        if (std::isfinite(alpha) && alpha > 0.0 && eta.allFinite() &&
	            stepNormCached(eta) > 1e-14) {
          if (profileRuntime) {
            ++profileCurvatureCauchyFallbackCandidateCount;
          }
          if (skipRedundantCandidateProjection) {
            if (profileRuntime) {
              ++profileReducedCandidateProjectionSkipCount;
            }
          } else {
            eta = projectCached(X, eta);
          }
	          eta = scaleToTrustRegionCached(eta, radius);
          Matrix hEta = hessianCached(X, eta);
          double modelDecrease =
              -(rotationInnerProduct(grad, eta, d) +
                0.5 * rotationInnerProduct(eta, hEta, d));
          if (modelDecrease <= 1e-14) {
            modelDecrease = -rotationInnerProduct(grad, eta, d);
          }
          if (modelDecrease > 1e-14) {
	            Matrix candidate = recoverCached(retractCached(X, eta));
            const double candidateCost = exactObjective(candidate);
            const double actualDecrease = f - candidateCost;
            const double rho = actualDecrease / modelDecrease;
            if (std::isfinite(candidateCost) && actualDecrease > 0.0 &&
                rho > 0.05 && candidateCost < bestCost) {
              bestCost = candidateCost;
              bestRho = rho;
              bestCandidate = candidate;
              bestEta = eta;
              hasCandidate = true;
              bestIsCurvatureCauchy = false;
              bestIsCurvatureFallback = true;
            }
          }
        }
      }

      if (!hasCandidate) {
        result.rtrRejectedSteps++;
        radius *= 0.25;
        continue;
      }

      X = bestCandidate;
      if (bestIsCurvatureCauchy && profileRuntime) {
        ++profileCurvatureCauchyAcceptedCount;
      }
      if (bestIsCurvatureFallback && profileRuntime) {
        ++profileCurvatureCauchyFallbackAcceptedCount;
      }
      ++acceptedIterations;
	      if (bestRho > 0.75 && stepNormCached(bestEta) > 0.8 * radius) {
	        radius = std::min(maxRadius, 2.0 * radius);
	      }
      accepted = true;
      break;
    }

    if (!accepted) {
      double step = std::min(radius, 1.0) / std::max(gradNorm, 1e-14);
      for (unsigned attempt = 0; attempt < 12; ++attempt) {
	        Matrix candidate = recoverCached(retractCached(X, -step * grad));
        const double candidateCost = exactObjective(candidate);
        if (std::isfinite(candidateCost) && candidateCost < f) {
          X = candidate;
          ++acceptedIterations;
          accepted = true;
          radius = std::max(radius, step * gradNorm);
          break;
        }
        step *= 0.5;
      }
    }

    if (radius < 1e-12) {
      break;
    }
  }

  const auto elapsed =
      std::chrono::steady_clock::now() - startTime;
  result.elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  if (recordResultStats) {
    result.fOpt = exactObjective(X);
    result.gradNormOpt = problem->RieGradNorm(X);
  } else {
    result.fOpt = std::numeric_limits<double>::quiet_NaN();
    result.gradNormOpt = std::numeric_limits<double>::quiet_NaN();
  }
  result.relativeChange =
      std::sqrt((X - Y).squaredNorm() / problem->num_poses());
  result.rtrAcceptedIterations = acceptedIterations;
  result.rtrAcceptedRadius = radius;
  result.success = true;
  return X;
}

}  // namespace DPGO
