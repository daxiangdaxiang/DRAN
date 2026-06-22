/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#include <DPGO/FullEquivHybridOptimizer.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DPGO {
namespace {

using ColMajorSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

double matrixInnerProduct(const Matrix &A, const Matrix &B) {
  return (A.cwiseProduct(B)).sum();
}

void validatePlan(const FullEquivHybridEliminationPlan &plan,
                  int totalColumns) {
  if (plan.dimension == 0 || plan.numPoses == 0) {
    throw std::invalid_argument("FE-Hybrid Schur plan is empty");
  }
  const int expectedColumns =
      static_cast<int>(plan.numPoses * (plan.dimension + 1));
  if (expectedColumns != totalColumns) {
    throw std::invalid_argument("FE-Hybrid Schur plan dimension mismatch");
  }
  if (plan.keepColumns.empty() || plan.eliminateColumns.empty()) {
    throw std::invalid_argument(
        "FE-Hybrid Schur plan must keep and eliminate columns");
  }
}

Matrix dampedDenseSystem(const SparseMatrix &H, double damping) {
  Matrix dense = Matrix(H);
  if (dense.rows() != dense.cols()) {
    throw std::invalid_argument("FE-Hybrid linear system must be square");
  }
  if (damping != 0.0) {
    dense.diagonal().array() += damping;
  }
  dense = 0.5 * (dense + dense.transpose());
  return dense;
}

Matrix selectRowsCols(const Matrix &matrix, const std::vector<int> &rows,
                      const std::vector<int> &cols) {
  Matrix result(rows.size(), cols.size());
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t c = 0; c < cols.size(); ++c) {
      result(static_cast<int>(r), static_cast<int>(c)) =
          matrix(rows[r], cols[c]);
    }
  }
  return result;
}

Matrix selectRowsColsSparseDamped(const SparseMatrix &matrix, double damping,
                                  const std::vector<int> &rows,
                                  const std::vector<int> &cols) {
  Matrix result(rows.size(), cols.size());
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t c = 0; c < cols.size(); ++c) {
      double value = matrix.coeff(rows[r], cols[c]);
      if (rows[r] == cols[c]) {
        value += damping;
      }
      result(static_cast<int>(r), static_cast<int>(c)) = value;
    }
  }
  return result;
}

Matrix selectGradientColumns(const Matrix &gradient,
                             const std::vector<int> &cols) {
  Matrix result(gradient.rows(), cols.size());
  for (std::size_t c = 0; c < cols.size(); ++c) {
    result.col(static_cast<int>(c)) = gradient.col(cols[c]);
  }
  return result;
}

void scatterStepColumns(Matrix &step, const Matrix &block,
                        const std::vector<int> &cols) {
  if (block.cols() != static_cast<int>(cols.size())) {
    throw std::invalid_argument("FE-Hybrid step block/column mismatch");
  }
  for (std::size_t c = 0; c < cols.size(); ++c) {
    step.col(cols[c]) = block.col(static_cast<int>(c));
  }
}

Vector selectVectorEntries(const Vector &vector,
                           const std::vector<int> &entries) {
  Vector result(entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    result(static_cast<int>(i)) = vector(entries[i]);
  }
  return result;
}

void scatterVectorEntries(Vector &vector, const Vector &block,
                          const std::vector<int> &entries) {
  if (block.rows() != static_cast<int>(entries.size())) {
    throw std::invalid_argument("FE-Hybrid vector block/entry mismatch");
  }
  for (std::size_t i = 0; i < entries.size(); ++i) {
    vector(entries[i]) = block(static_cast<int>(i));
  }
}

Matrix solveWithLdlt(const Matrix &A, const Matrix &rhs) {
  Eigen::LDLT<Matrix> ldlt(A);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("FE-Hybrid LDLT factorization failed");
  }
  Matrix solution = ldlt.solve(rhs);
  if (ldlt.info() != Eigen::Success || !solution.allFinite()) {
    throw std::runtime_error("FE-Hybrid LDLT solve failed");
  }
  const double relativeResidual =
      (A * solution - rhs).norm() / (rhs.norm() + 1e-12);
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-8) {
    throw std::runtime_error("FE-Hybrid LDLT residual check failed");
  }
  return solution;
}

void computeLdltOrThrow(Eigen::LDLT<Matrix> &ldlt, const Matrix &A,
                        const std::string &label) {
  ldlt.compute(A);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("FE-Hybrid LDLT factorization failed: " +
                             label);
  }
}

Matrix solveWithLdltFactor(const Eigen::LDLT<Matrix> &ldlt, const Matrix &A,
                           const Matrix &rhs, const std::string &label) {
  Matrix solution = ldlt.solve(rhs);
  if (ldlt.info() != Eigen::Success || !solution.allFinite()) {
    throw std::runtime_error("FE-Hybrid LDLT solve failed: " + label);
  }
  const double relativeResidual =
      (A * solution - rhs).norm() / (rhs.norm() + 1e-12);
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-8) {
    throw std::runtime_error(
        "FE-Hybrid LDLT residual check failed: " + label);
  }
  return solution;
}

Vector buildJacobiInverseDiagonal(const Matrix &A) {
  Vector inverseDiagonal(A.rows());
  for (int i = 0; i < A.rows(); ++i) {
    const double value = A(i, i);
    if (std::isfinite(value) && value > 1e-14) {
      inverseDiagonal(i) = 1.0 / value;
    } else {
      inverseDiagonal(i) = 1.0;
    }
  }
  return inverseDiagonal;
}

struct JacobiPreconditioner {
  bool enabled{false};
  bool useBlocks{false};
  unsigned blockSize{0};
  Vector inverseDiagonal;
  std::vector<Matrix> inverseBlocks;
};

struct TranslationSchurPreconditioner {
  bool enabled{false};
  FullEquivHybridEliminationPlan plan;
  Matrix Hab;
  Matrix Hba;
  Matrix Hbb;
  Matrix S;
  Eigen::LDLT<Matrix> hbbLdlt;
  Eigen::LDLT<Matrix> sLdlt;
  std::size_t factorizationCount{0};
};

struct ReducedRotationPreconditioner {
  bool enabled{false};
  FullEquivHybridReducedRotationPreconditioner mode{
      FullEquivHybridReducedRotationPreconditioner::Cholesky};
  FullEquivHybridEliminationPlan plan;
  Matrix Hab;
  Matrix Hba;
  Matrix Hbb;
  Matrix rotationSystem;
  Eigen::LDLT<Matrix> hbbLdlt;
  Eigen::LDLT<Matrix> rotationLdlt;
  JacobiPreconditioner schurJacobi;
  std::size_t factorizationCount{0};
};

struct TranslationBlockPreconditioner {
  TranslationBlockPreconditioner() = default;
  TranslationBlockPreconditioner(const TranslationBlockPreconditioner &) =
      delete;
  TranslationBlockPreconditioner &operator=(
      const TranslationBlockPreconditioner &) = delete;
  TranslationBlockPreconditioner(TranslationBlockPreconditioner &&) = default;
  TranslationBlockPreconditioner &operator=(
      TranslationBlockPreconditioner &&) = default;

  bool enabled{false};
  FullEquivHybridEliminationPlan plan;
  JacobiPreconditioner jacobi;
  ColMajorSparseMatrix translationBlock;
  std::unique_ptr<Eigen::SimplicialLDLT<ColMajorSparseMatrix>> translationLdlt;
  std::size_t factorizationCount{0};
};

struct LocalChainPreconditioner {
  LocalChainPreconditioner() = default;
  LocalChainPreconditioner(const LocalChainPreconditioner &) = delete;
  LocalChainPreconditioner &operator=(const LocalChainPreconditioner &) =
      delete;
  LocalChainPreconditioner(LocalChainPreconditioner &&) = default;
  LocalChainPreconditioner &operator=(LocalChainPreconditioner &&) = default;

  bool enabled{false};
  ColMajorSparseMatrix chainMatrix;
  std::unique_ptr<Eigen::SimplicialLDLT<ColMajorSparseMatrix>> chainLdlt;
  std::size_t factorizationCount{0};
};

struct TranslationSparseSchurPreconditioner {
  TranslationSparseSchurPreconditioner() = default;
  TranslationSparseSchurPreconditioner(
      const TranslationSparseSchurPreconditioner &) = delete;
  TranslationSparseSchurPreconditioner &operator=(
      const TranslationSparseSchurPreconditioner &) = delete;
  TranslationSparseSchurPreconditioner(
      TranslationSparseSchurPreconditioner &&) = default;
  TranslationSparseSchurPreconditioner &operator=(
      TranslationSparseSchurPreconditioner &&) = default;

  bool enabled{false};
  FullEquivHybridEliminationPlan plan;
  ColMajorSparseMatrix Hrt;
  ColMajorSparseMatrix Htr;
  ColMajorSparseMatrix Htt;
  ColMajorSparseMatrix sparseSchur;
  std::unique_ptr<Eigen::SimplicialLDLT<ColMajorSparseMatrix>> httLdlt;
  std::unique_ptr<Eigen::SimplicialLDLT<ColMajorSparseMatrix>> schurLdlt;
  std::size_t factorizationCount{0};
};

struct TranslationLocalSchurPreconditioner {
  bool enabled{false};
  FullEquivHybridEliminationPlan activePlan;
  JacobiPreconditioner jacobi;
  Matrix Hkt;
  Matrix Htk;
  Matrix Htt;
  Matrix S;
  Eigen::LDLT<Matrix> httLdlt;
  Eigen::LDLT<Matrix> sLdlt;
  std::size_t factorizationCount{0};
  std::size_t activePoseCount{0};
  std::size_t activeColumnCount{0};
};

struct LaplacianDeflationPreconditioner {
  bool enabled{false};
  Matrix basis;
  Matrix coarseSystem;
  Eigen::LDLT<Matrix> coarseLdlt;
  std::size_t factorizationCount{0};
  std::size_t fallbackCount{0};
};

JacobiPreconditioner buildJacobiPreconditioner(const Matrix &A,
                                               bool enabled,
                                               unsigned blockSize);

JacobiPreconditioner buildSparseJacobiPreconditioner(const SparseMatrix &A,
                                                     double damping,
                                                     bool enabled,
                                                     unsigned blockSize);

Vector applyJacobiPreconditioner(const JacobiPreconditioner &preconditioner,
                                 const Vector &residual);

std::vector<unsigned> selectTranslationLocalSchurActivePoses(
    const SparseMatrix &A, const Matrix &rhs, unsigned dimension,
    unsigned maxActivePoses) {
  const unsigned blockSize = dimension + 1;
  if (dimension == 0 || A.rows() != A.cols() ||
      A.cols() % static_cast<int>(blockSize) != 0) {
    throw std::invalid_argument(
        "FE-Hybrid local Schur active-set dimension mismatch");
  }
  const unsigned numPoses = static_cast<unsigned>(A.cols()) / blockSize;
  const unsigned limit =
      maxActivePoses == 0 ? numPoses : std::min(maxActivePoses, numPoses);
  if (limit == 0) {
    return {};
  }
  if (limit == numPoses) {
    std::vector<unsigned> all;
    all.reserve(numPoses);
    for (unsigned pose = 0; pose < numPoses; ++pose) {
      all.push_back(pose);
    }
    return all;
  }

  std::vector<double> scores(numPoses, 0.0);
  if (rhs.rows() == A.rows()) {
    for (unsigned pose = 0; pose < numPoses; ++pose) {
      const int rowStart = static_cast<int>(pose * blockSize);
      const int rows = static_cast<int>(blockSize);
      scores[pose] += rhs.block(rowStart, 0, rows, rhs.cols()).norm();
    }
  }
  for (int outer = 0; outer < A.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(A, outer); it; ++it) {
      if (it.row() == it.col()) {
        continue;
      }
      const unsigned rowPose =
          static_cast<unsigned>(it.row()) / blockSize;
      const unsigned colPose =
          static_cast<unsigned>(it.col()) / blockSize;
      const unsigned rowLocal =
          static_cast<unsigned>(it.row()) % blockSize;
      const unsigned colLocal =
          static_cast<unsigned>(it.col()) % blockSize;
      const double weight =
          std::abs(it.value()) *
          ((rowLocal == dimension || colLocal == dimension) ? 2.0 : 1.0);
      if (rowPose < numPoses) {
        scores[rowPose] += weight;
      }
      if (colPose < numPoses && colPose != rowPose) {
        scores[colPose] += weight;
      }
    }
  }

  std::vector<std::pair<double, unsigned>> ranked;
  ranked.reserve(numPoses);
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    ranked.emplace_back(scores[pose], pose);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first > rhs.first;
              }
              return lhs.second < rhs.second;
            });
  std::vector<unsigned> active;
  active.reserve(limit);
  for (unsigned i = 0; i < limit; ++i) {
    active.push_back(ranked[i].second);
  }
  std::sort(active.begin(), active.end());
  return active;
}

TranslationLocalSchurPreconditioner buildTranslationLocalSchurPreconditioner(
    const SparseMatrix &A, double damping, const Matrix &rhs, unsigned dimension,
    unsigned maxActivePoses, bool useJacobi) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument(
        "FE-Hybrid translation local Schur preconditioner requires square system");
  }
  const unsigned blockSize = dimension + 1;
  const unsigned numPoses = static_cast<unsigned>(A.cols()) / blockSize;
  const std::vector<unsigned> activePoses =
      selectTranslationLocalSchurActivePoses(A, rhs, dimension, maxActivePoses);
  if (activePoses.empty()) {
    throw std::runtime_error(
        "FE-Hybrid translation local Schur active set is empty");
  }

  TranslationLocalSchurPreconditioner preconditioner;
  preconditioner.jacobi =
      buildSparseJacobiPreconditioner(A, damping, useJacobi, 0);
  preconditioner.activePlan.numPoses =
      static_cast<unsigned>(activePoses.size());
  preconditioner.activePlan.dimension = dimension;
  for (const unsigned pose : activePoses) {
    if (pose >= numPoses) {
      throw std::runtime_error(
          "FE-Hybrid translation local Schur active pose is invalid");
    }
    const unsigned base = pose * blockSize;
    for (unsigned col = 0; col < dimension; ++col) {
      preconditioner.activePlan.keepColumns.push_back(
          static_cast<int>(base + col));
    }
    preconditioner.activePlan.eliminateColumns.push_back(
        static_cast<int>(base + dimension));
  }
  preconditioner.activePoseCount = activePoses.size();
  preconditioner.activeColumnCount =
      preconditioner.activePlan.keepColumns.size() +
      preconditioner.activePlan.eliminateColumns.size();

  const Matrix Hkk = selectRowsColsSparseDamped(
      A, damping, preconditioner.activePlan.keepColumns,
      preconditioner.activePlan.keepColumns);
  preconditioner.Hkt = selectRowsColsSparseDamped(
      A, damping, preconditioner.activePlan.keepColumns,
      preconditioner.activePlan.eliminateColumns);
  preconditioner.Htk = selectRowsColsSparseDamped(
      A, damping, preconditioner.activePlan.eliminateColumns,
      preconditioner.activePlan.keepColumns);
  preconditioner.Htt = selectRowsColsSparseDamped(
      A, damping, preconditioner.activePlan.eliminateColumns,
      preconditioner.activePlan.eliminateColumns);
  preconditioner.Htt =
      0.5 * (preconditioner.Htt + preconditioner.Htt.transpose());
  computeLdltOrThrow(preconditioner.httLdlt, preconditioner.Htt,
                     "translation local Schur Htt");
  ++preconditioner.factorizationCount;
  if (!preconditioner.httLdlt.isPositive()) {
    throw std::runtime_error(
        "FE-Hybrid translation local Schur Htt is not SPD");
  }
  const Matrix httInvHtk =
      solveWithLdltFactor(preconditioner.httLdlt, preconditioner.Htt,
                          preconditioner.Htk,
                          "translation local Schur Htt/Htk");
  preconditioner.S = Hkk - preconditioner.Hkt * httInvHtk;
  preconditioner.S =
      0.5 * (preconditioner.S + preconditioner.S.transpose());
  computeLdltOrThrow(preconditioner.sLdlt, preconditioner.S,
                     "translation local Schur S");
  ++preconditioner.factorizationCount;
  if (!preconditioner.sLdlt.isPositive()) {
    throw std::runtime_error(
        "FE-Hybrid translation local Schur S is not SPD");
  }
  preconditioner.enabled = true;
  return preconditioner;
}

Vector applyTranslationLocalSchurPreconditioner(
    const TranslationLocalSchurPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled) {
    throw std::invalid_argument(
        "FE-Hybrid translation local Schur preconditioner is disabled");
  }
  Vector result = applyJacobiPreconditioner(preconditioner.jacobi, residual);
  const Vector keepResidual =
      selectVectorEntries(residual, preconditioner.activePlan.keepColumns);
  const Vector translationResidual =
      selectVectorEntries(residual,
                          preconditioner.activePlan.eliminateColumns);
  const Matrix httInvTranslation =
      solveWithLdltFactor(preconditioner.httLdlt, preconditioner.Htt,
                          Matrix(translationResidual),
                          "translation local Schur Htt residual");
  const Matrix reducedResidual =
      Matrix(keepResidual) - preconditioner.Hkt * httInvTranslation;
  const Matrix keepStep =
      solveWithLdltFactor(preconditioner.sLdlt, preconditioner.S,
                          reducedResidual,
                          "translation local Schur S residual");
  const Matrix translationRhs =
      Matrix(translationResidual) - preconditioner.Htk * keepStep;
  const Matrix translationStep =
      solveWithLdltFactor(preconditioner.httLdlt, preconditioner.Htt,
                          translationRhs,
                          "translation local Schur Htt backsolve");
  scatterVectorEntries(result, Vector(keepStep),
                       preconditioner.activePlan.keepColumns);
  scatterVectorEntries(result, Vector(translationStep),
                       preconditioner.activePlan.eliminateColumns);
  return result;
}

Matrix orthonormalizeBasisColumns(const Matrix &rawBasis) {
  Matrix basis = Matrix::Zero(rawBasis.rows(), rawBasis.cols());
  int accepted = 0;
  for (int col = 0; col < rawBasis.cols(); ++col) {
    Vector candidate = rawBasis.col(col);
    for (int prev = 0; prev < accepted; ++prev) {
      candidate -= basis.col(prev).dot(candidate) * basis.col(prev);
    }
    const double norm = candidate.norm();
    if (std::isfinite(norm) && norm > 1e-10) {
      basis.col(accepted) = candidate / norm;
      ++accepted;
    }
  }
  return basis.leftCols(accepted);
}

Matrix buildDctPoseBasis(unsigned numPoses, unsigned basisSize) {
  const unsigned k = std::min(numPoses, basisSize);
  Matrix raw = Matrix::Zero(static_cast<int>(numPoses), static_cast<int>(k));
  const double pi = 3.14159265358979323846;
  for (unsigned mode = 0; mode < k; ++mode) {
    for (unsigned pose = 0; pose < numPoses; ++pose) {
      if (mode == 0) {
        raw(static_cast<int>(pose), static_cast<int>(mode)) = 1.0;
      } else {
        raw(static_cast<int>(pose), static_cast<int>(mode)) =
            std::cos(pi * (static_cast<double>(pose) + 0.5) *
                     static_cast<double>(mode) /
                     static_cast<double>(numPoses));
      }
    }
  }
  return orthonormalizeBasisColumns(raw);
}

Matrix buildTranslationGraphLaplacianBasis(const SparseMatrix &A,
                                           unsigned dimension,
                                           unsigned basisSize,
                                           unsigned maxEigenPoses,
                                           bool *usedDctFallback) {
  if (usedDctFallback != nullptr) {
    *usedDctFallback = false;
  }
  auto fallbackToDct = [&](unsigned numPoses, unsigned k) {
    if (usedDctFallback != nullptr) {
      *usedDctFallback = true;
    }
    return buildDctPoseBasis(numPoses, k);
  };
  const unsigned blockSize = dimension + 1;
  if (dimension == 0 || A.rows() != A.cols() ||
      A.cols() % static_cast<int>(blockSize) != 0 || basisSize == 0) {
    throw std::invalid_argument(
        "FE-Hybrid Laplacian deflation basis dimension mismatch");
  }
  const unsigned numPoses = static_cast<unsigned>(A.cols()) / blockSize;
  const unsigned k = std::min(numPoses, basisSize);
  if (k == 0) {
    throw std::runtime_error("FE-Hybrid Laplacian deflation basis is empty");
  }
  if (numPoses > maxEigenPoses) {
    return fallbackToDct(numPoses, k);
  }

  Matrix laplacian = Matrix::Zero(static_cast<int>(numPoses),
                                  static_cast<int>(numPoses));
  double totalWeight = 0.0;
  for (int outer = 0; outer < A.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(A, outer); it; ++it) {
      const unsigned rowPose =
          static_cast<unsigned>(it.row()) / blockSize;
      const unsigned colPose =
          static_cast<unsigned>(it.col()) / blockSize;
      const unsigned rowLocal =
          static_cast<unsigned>(it.row()) % blockSize;
      const unsigned colLocal =
          static_cast<unsigned>(it.col()) % blockSize;
      if (rowPose >= numPoses || colPose >= numPoses ||
          rowPose >= colPose || rowLocal != dimension ||
          colLocal != dimension) {
        continue;
      }
      const double weight = std::abs(it.value());
      if (!std::isfinite(weight) || weight <= 0.0) {
        continue;
      }
      laplacian(static_cast<int>(rowPose), static_cast<int>(rowPose)) +=
          weight;
      laplacian(static_cast<int>(colPose), static_cast<int>(colPose)) +=
          weight;
      laplacian(static_cast<int>(rowPose), static_cast<int>(colPose)) -=
          weight;
      laplacian(static_cast<int>(colPose), static_cast<int>(rowPose)) -=
          weight;
      totalWeight += weight;
    }
  }
  if (totalWeight <= 0.0) {
    return fallbackToDct(numPoses, k);
  }

  Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(laplacian);
  if (eigensolver.info() != Eigen::Success) {
    return fallbackToDct(numPoses, k);
  }
  Matrix raw = eigensolver.eigenvectors().leftCols(static_cast<int>(k));
  Matrix basis = orthonormalizeBasisColumns(raw);
  if (basis.cols() == 0) {
    return fallbackToDct(numPoses, k);
  }
  return basis;
}

LaplacianDeflationPreconditioner buildLaplacianDeflationPreconditioner(
    const SparseMatrix &A, double damping, unsigned dimension,
    unsigned basisSize, unsigned maxEigenPoses) {
  LaplacianDeflationPreconditioner preconditioner;
  const unsigned blockSize = dimension + 1;
  const unsigned numPoses = static_cast<unsigned>(A.cols()) / blockSize;
  bool usedDctFallback = false;
  const Matrix poseBasis = buildTranslationGraphLaplacianBasis(
      A, dimension, basisSize, maxEigenPoses, &usedDctFallback);
  preconditioner.fallbackCount = usedDctFallback ? 1u : 0u;
  if (poseBasis.cols() == 0) {
    throw std::runtime_error("FE-Hybrid Laplacian deflation basis is empty");
  }

  preconditioner.basis =
      Matrix::Zero(A.cols(), poseBasis.cols());
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const int translationCol =
        static_cast<int>(pose * blockSize + dimension);
    preconditioner.basis.row(translationCol) =
        poseBasis.row(static_cast<int>(pose));
  }
  Matrix AZ = A * preconditioner.basis;
  if (damping != 0.0) {
    AZ.noalias() += damping * preconditioner.basis;
  }
  preconditioner.coarseSystem =
      preconditioner.basis.transpose() * AZ;
  preconditioner.coarseSystem =
      0.5 * (preconditioner.coarseSystem +
             preconditioner.coarseSystem.transpose());
  computeLdltOrThrow(preconditioner.coarseLdlt,
                     preconditioner.coarseSystem,
                     "Laplacian deflation coarse system");
  ++preconditioner.factorizationCount;
  if (!preconditioner.coarseLdlt.isPositive()) {
    throw std::runtime_error(
        "FE-Hybrid Laplacian deflation coarse system is not SPD");
  }
  preconditioner.enabled = true;
  return preconditioner;
}

Vector applyLaplacianDeflationPreconditioner(
    const LaplacianDeflationPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled) {
    return Vector::Zero(residual.rows());
  }
  const Matrix coarseRhs = preconditioner.basis.transpose() * residual;
  const Matrix coarseStep = solveWithLdltFactor(
      preconditioner.coarseLdlt, preconditioner.coarseSystem, coarseRhs,
      "Laplacian deflation coarse residual");
  return preconditioner.basis * coarseStep;
}

TranslationBlockPreconditioner buildTranslationBlockPreconditioner(
    const SparseMatrix &A, double damping,
    const FullEquivHybridEliminationPlan &plan, bool useJacobi,
    unsigned /*blockSize*/) {
  validatePlan(plan, A.cols());
  if (A.rows() != A.cols()) {
    throw std::invalid_argument(
        "FE-Hybrid translation block preconditioner requires square system");
  }

  TranslationBlockPreconditioner preconditioner;
  preconditioner.plan = plan;
  // Keep this preconditioner block diagonal: a sparse coupled translation
  // block plus diagonal scaling on the remaining columns. Mixing a full
  // pose-block inverse with an overwritten translation solve would make the
  // linear operator non-symmetric and can break PCG convergence.
  preconditioner.jacobi =
      buildSparseJacobiPreconditioner(A, damping, useJacobi, 0);

  std::vector<int> translationIndex(A.cols(), -1);
  for (std::size_t i = 0; i < plan.eliminateColumns.size(); ++i) {
    translationIndex[plan.eliminateColumns[i]] = static_cast<int>(i);
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(A.nonZeros()) +
                   plan.eliminateColumns.size());
  for (int outer = 0; outer < A.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(A, outer); it; ++it) {
      const int row = translationIndex[it.row()];
      const int col = translationIndex[it.col()];
      if (row >= 0 && col >= 0) {
        triplets.emplace_back(row, col, it.value());
      }
    }
  }
  for (std::size_t i = 0; i < plan.eliminateColumns.size(); ++i) {
    triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), damping);
  }

  preconditioner.translationBlock.resize(
      static_cast<int>(plan.eliminateColumns.size()),
      static_cast<int>(plan.eliminateColumns.size()));
  preconditioner.translationBlock.setFromTriplets(triplets.begin(),
                                                  triplets.end());
  preconditioner.translationBlock =
      0.5 * (preconditioner.translationBlock +
             ColMajorSparseMatrix(preconditioner.translationBlock.transpose()));
  preconditioner.translationLdlt =
      std::make_unique<Eigen::SimplicialLDLT<ColMajorSparseMatrix>>();
  preconditioner.translationLdlt->compute(preconditioner.translationBlock);
  ++preconditioner.factorizationCount;
  if (preconditioner.translationLdlt->info() != Eigen::Success) {
    throw std::runtime_error(
        "FE-Hybrid translation block preconditioner factorization failed");
  }
  preconditioner.enabled = true;
  return preconditioner;
}

Vector applyTranslationBlockPreconditioner(
    const TranslationBlockPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled) {
    throw std::invalid_argument(
        "FE-Hybrid translation block preconditioner is disabled");
  }
  Vector result = applyJacobiPreconditioner(preconditioner.jacobi, residual);
  const Vector translationResidual =
      selectVectorEntries(residual, preconditioner.plan.eliminateColumns);
  if (!preconditioner.translationLdlt) {
    throw std::runtime_error(
        "FE-Hybrid translation block preconditioner factor is missing");
  }
  Vector translationStep =
      preconditioner.translationLdlt->solve(translationResidual);
  if (preconditioner.translationLdlt->info() != Eigen::Success ||
      !translationStep.allFinite()) {
    throw std::runtime_error(
        "FE-Hybrid translation block preconditioner solve failed");
  }
  const double relativeResidual =
      (preconditioner.translationBlock * translationStep -
       translationResidual)
          .norm() /
      (translationResidual.norm() + 1e-12);
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-8) {
    throw std::runtime_error(
        "FE-Hybrid translation block preconditioner residual check failed");
  }
  scatterVectorEntries(result, translationStep,
                       preconditioner.plan.eliminateColumns);
  return result;
}

TranslationSparseSchurPreconditioner buildTranslationSparseSchurPreconditioner(
    const SparseMatrix &A, double damping,
    const FullEquivHybridEliminationPlan &plan) {
  validatePlan(plan, A.cols());
  if (A.rows() != A.cols()) {
    throw std::invalid_argument(
        "FE-Hybrid translation sparse Schur preconditioner requires square system");
  }

  TranslationSparseSchurPreconditioner preconditioner;
  preconditioner.plan = plan;

  std::vector<int> keepIndex(A.cols(), -1);
  std::vector<int> translationIndex(A.cols(), -1);
  for (std::size_t i = 0; i < plan.keepColumns.size(); ++i) {
    keepIndex[plan.keepColumns[i]] = static_cast<int>(i);
  }
  for (std::size_t i = 0; i < plan.eliminateColumns.size(); ++i) {
    translationIndex[plan.eliminateColumns[i]] = static_cast<int>(i);
  }

  std::vector<Eigen::Triplet<double>> hrrTriplets;
  std::vector<Eigen::Triplet<double>> hrtTriplets;
  std::vector<Eigen::Triplet<double>> htrTriplets;
  std::vector<Eigen::Triplet<double>> httTriplets;
  hrrTriplets.reserve(static_cast<std::size_t>(A.nonZeros()));
  hrtTriplets.reserve(static_cast<std::size_t>(A.nonZeros()));
  htrTriplets.reserve(static_cast<std::size_t>(A.nonZeros()));
  httTriplets.reserve(static_cast<std::size_t>(A.nonZeros()) +
                      plan.eliminateColumns.size());
  for (int outer = 0; outer < A.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(A, outer); it; ++it) {
      const int keepRow = keepIndex[it.row()];
      const int keepCol = keepIndex[it.col()];
      const int translationRow = translationIndex[it.row()];
      const int translationCol = translationIndex[it.col()];
      if (keepRow >= 0 && keepCol >= 0) {
        hrrTriplets.emplace_back(keepRow, keepCol, it.value());
      } else if (keepRow >= 0 && translationCol >= 0) {
        hrtTriplets.emplace_back(keepRow, translationCol, it.value());
      } else if (translationRow >= 0 && keepCol >= 0) {
        htrTriplets.emplace_back(translationRow, keepCol, it.value());
      } else if (translationRow >= 0 && translationCol >= 0) {
        httTriplets.emplace_back(translationRow, translationCol, it.value());
      }
    }
  }
  for (std::size_t i = 0; i < plan.keepColumns.size(); ++i) {
    hrrTriplets.emplace_back(static_cast<int>(i), static_cast<int>(i),
                             damping);
  }
  for (std::size_t i = 0; i < plan.eliminateColumns.size(); ++i) {
    httTriplets.emplace_back(static_cast<int>(i), static_cast<int>(i),
                             damping);
  }

  ColMajorSparseMatrix Hrr(static_cast<int>(plan.keepColumns.size()),
                           static_cast<int>(plan.keepColumns.size()));
  Hrr.setFromTriplets(hrrTriplets.begin(), hrrTriplets.end());
  Hrr = 0.5 * (Hrr + ColMajorSparseMatrix(Hrr.transpose()));
  preconditioner.Hrt.resize(static_cast<int>(plan.keepColumns.size()),
                            static_cast<int>(plan.eliminateColumns.size()));
  preconditioner.Hrt.setFromTriplets(hrtTriplets.begin(), hrtTriplets.end());
  preconditioner.Htr.resize(static_cast<int>(plan.eliminateColumns.size()),
                            static_cast<int>(plan.keepColumns.size()));
  preconditioner.Htr.setFromTriplets(htrTriplets.begin(), htrTriplets.end());
  preconditioner.Htt.resize(static_cast<int>(plan.eliminateColumns.size()),
                            static_cast<int>(plan.eliminateColumns.size()));
  preconditioner.Htt.setFromTriplets(httTriplets.begin(), httTriplets.end());
  preconditioner.Htt =
      0.5 * (preconditioner.Htt +
             ColMajorSparseMatrix(preconditioner.Htt.transpose()));

  Vector inverseTranslationDiagonal(preconditioner.Htt.rows());
  for (int i = 0; i < preconditioner.Htt.rows(); ++i) {
    const double value = preconditioner.Htt.coeff(i, i);
    if (std::isfinite(value) && value > 1e-14) {
      inverseTranslationDiagonal(i) = 1.0 / value;
    } else {
      throw std::runtime_error(
          "FE-Hybrid translation sparse Schur diagonal is not SPD");
    }
  }
  const DiagonalMatrix inverseDiagonal(inverseTranslationDiagonal);
  preconditioner.sparseSchur =
      Hrr - preconditioner.Hrt * inverseDiagonal * preconditioner.Htr;
  preconditioner.sparseSchur =
      0.5 * (preconditioner.sparseSchur +
             ColMajorSparseMatrix(preconditioner.sparseSchur.transpose()));

  preconditioner.httLdlt =
      std::make_unique<Eigen::SimplicialLDLT<ColMajorSparseMatrix>>();
  preconditioner.httLdlt->compute(preconditioner.Htt);
  ++preconditioner.factorizationCount;
  if (preconditioner.httLdlt->info() != Eigen::Success) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur Htt factorization failed");
  }
  preconditioner.schurLdlt =
      std::make_unique<Eigen::SimplicialLDLT<ColMajorSparseMatrix>>();
  preconditioner.schurLdlt->compute(preconditioner.sparseSchur);
  ++preconditioner.factorizationCount;
  if (preconditioner.schurLdlt->info() != Eigen::Success) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur factorization failed");
  }
  preconditioner.enabled = true;
  return preconditioner;
}

Vector applyTranslationSparseSchurPreconditioner(
    const TranslationSparseSchurPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled) {
    throw std::invalid_argument(
        "FE-Hybrid translation sparse Schur preconditioner is disabled");
  }
  if (!preconditioner.httLdlt || !preconditioner.schurLdlt) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur factor is missing");
  }

  const Vector keepResidual =
      selectVectorEntries(residual, preconditioner.plan.keepColumns);
  const Vector translationResidual =
      selectVectorEntries(residual, preconditioner.plan.eliminateColumns);
  const Vector httInvTranslation =
      preconditioner.httLdlt->solve(translationResidual);
  if (preconditioner.httLdlt->info() != Eigen::Success ||
      !httInvTranslation.allFinite()) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur Htt solve failed");
  }
  const Vector reducedResidual =
      keepResidual - preconditioner.Hrt * httInvTranslation;
  const Vector keepStep = preconditioner.schurLdlt->solve(reducedResidual);
  if (preconditioner.schurLdlt->info() != Eigen::Success ||
      !keepStep.allFinite()) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur solve failed");
  }
  const Vector translationRhs =
      translationResidual - preconditioner.Htr * keepStep;
  const Vector translationStep =
      preconditioner.httLdlt->solve(translationRhs);
  if (preconditioner.httLdlt->info() != Eigen::Success ||
      !translationStep.allFinite()) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur backsolve failed");
  }

  const double reducedResidualCheck =
      (preconditioner.sparseSchur * keepStep - reducedResidual).norm() /
      (reducedResidual.norm() + 1e-12);
  const double translationResidualCheck =
      (preconditioner.Htt * translationStep - translationRhs).norm() /
      (translationRhs.norm() + 1e-12);
  if (!std::isfinite(reducedResidualCheck) ||
      !std::isfinite(translationResidualCheck) ||
      reducedResidualCheck > 1e-8 || translationResidualCheck > 1e-8) {
    throw std::runtime_error(
        "FE-Hybrid translation sparse Schur residual check failed");
  }

  Vector result = Vector::Zero(residual.rows());
  scatterVectorEntries(result, keepStep, preconditioner.plan.keepColumns);
  scatterVectorEntries(result, translationStep,
                       preconditioner.plan.eliminateColumns);
  return result;
}

LocalChainPreconditioner buildLocalChainPreconditioner(
    const SparseMatrix &A, double damping, unsigned blockSize) {
  if (blockSize == 0 || A.rows() != A.cols() ||
      A.cols() % static_cast<int>(blockSize) != 0) {
    throw std::invalid_argument(
        "FE-Hybrid local-chain preconditioner dimension mismatch");
  }
  LocalChainPreconditioner preconditioner;
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(A.nonZeros()) +
                   static_cast<std::size_t>(A.cols()));
  for (int outer = 0; outer < A.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(A, outer); it; ++it) {
      const int rowBlock = it.row() / static_cast<int>(blockSize);
      const int colBlock = it.col() / static_cast<int>(blockSize);
      if (std::abs(rowBlock - colBlock) <= 1) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
    }
  }
  if (damping != 0.0) {
    for (int col = 0; col < A.cols(); ++col) {
      triplets.emplace_back(col, col, damping);
    }
  }
  preconditioner.chainMatrix.resize(A.rows(), A.cols());
  preconditioner.chainMatrix.setFromTriplets(triplets.begin(), triplets.end());
  preconditioner.chainMatrix =
      0.5 * (preconditioner.chainMatrix +
             ColMajorSparseMatrix(preconditioner.chainMatrix.transpose()));
  preconditioner.chainMatrix.makeCompressed();
  preconditioner.chainLdlt =
      std::make_unique<Eigen::SimplicialLDLT<ColMajorSparseMatrix>>();
  preconditioner.chainLdlt->compute(preconditioner.chainMatrix);
  ++preconditioner.factorizationCount;
  if (preconditioner.chainLdlt->info() != Eigen::Success) {
    throw std::runtime_error(
        "FE-Hybrid local-chain preconditioner factorization failed");
  }
  preconditioner.enabled = true;
  return preconditioner;
}

Vector applyLocalChainPreconditioner(
    const LocalChainPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled || !preconditioner.chainLdlt) {
    throw std::invalid_argument(
        "FE-Hybrid local-chain preconditioner is disabled");
  }
  const Vector step = preconditioner.chainLdlt->solve(residual);
  if (preconditioner.chainLdlt->info() != Eigen::Success ||
      !step.allFinite()) {
    throw std::runtime_error("FE-Hybrid local-chain solve failed");
  }
  const double relativeResidual =
      (preconditioner.chainMatrix * step - residual).norm() /
      (residual.norm() + 1e-12);
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-8) {
    throw std::runtime_error(
        "FE-Hybrid local-chain residual check failed");
  }
  return step;
}

TranslationSchurPreconditioner buildTranslationSchurPreconditioner(
    const Matrix &A, const FullEquivHybridEliminationPlan &plan) {
  validatePlan(plan, A.cols());
  if (A.rows() != A.cols()) {
    throw std::invalid_argument(
        "FE-Hybrid translation Schur preconditioner requires square system");
  }

  TranslationSchurPreconditioner preconditioner;
  preconditioner.enabled = true;
  preconditioner.plan = plan;
  const Matrix Haa = selectRowsCols(A, plan.keepColumns, plan.keepColumns);
  preconditioner.Hab =
      selectRowsCols(A, plan.keepColumns, plan.eliminateColumns);
  preconditioner.Hba =
      selectRowsCols(A, plan.eliminateColumns, plan.keepColumns);
  preconditioner.Hbb =
      selectRowsCols(A, plan.eliminateColumns, plan.eliminateColumns);
  computeLdltOrThrow(preconditioner.hbbLdlt, preconditioner.Hbb,
                     "translation Schur Hbb");
  ++preconditioner.factorizationCount;
  if (!preconditioner.hbbLdlt.isPositive()) {
    throw std::runtime_error(
        "FE-Hybrid translation Schur preconditioner Hbb is not SPD");
  }
  const Matrix hbbInvHba =
      solveWithLdltFactor(preconditioner.hbbLdlt, preconditioner.Hbb,
                          preconditioner.Hba, "translation Schur Hbb/Hba");
  preconditioner.S = Haa - preconditioner.Hab * hbbInvHba;
  preconditioner.S =
      0.5 * (preconditioner.S + preconditioner.S.transpose());
  computeLdltOrThrow(preconditioner.sLdlt, preconditioner.S,
                     "translation Schur S");
  ++preconditioner.factorizationCount;
  if (!preconditioner.sLdlt.isPositive()) {
    throw std::runtime_error(
        "FE-Hybrid translation Schur preconditioner S is not SPD");
  }
  return preconditioner;
}

ReducedRotationPreconditioner buildReducedRotationPreconditioner(
    const Matrix &A, const FullEquivHybridEliminationPlan &plan,
    FullEquivHybridReducedRotationPreconditioner mode) {
  validatePlan(plan, A.cols());
  if (A.rows() != A.cols()) {
    throw std::invalid_argument(
        "FE-Hybrid reduced-rotation preconditioner requires square system");
  }

  ReducedRotationPreconditioner preconditioner;
  preconditioner.enabled = true;
  preconditioner.mode = mode;
  preconditioner.plan = plan;
  const Matrix Haa = selectRowsCols(A, plan.keepColumns, plan.keepColumns);
  preconditioner.Hab =
      selectRowsCols(A, plan.keepColumns, plan.eliminateColumns);
  preconditioner.Hba =
      selectRowsCols(A, plan.eliminateColumns, plan.keepColumns);
  preconditioner.Hbb =
      selectRowsCols(A, plan.eliminateColumns, plan.eliminateColumns);
  computeLdltOrThrow(preconditioner.hbbLdlt, preconditioner.Hbb,
                     "reduced-rotation Hbb");
  ++preconditioner.factorizationCount;
  if (!preconditioner.hbbLdlt.isPositive()) {
    throw std::runtime_error(
        "FE-Hybrid reduced-rotation preconditioner Hbb is not SPD");
  }
  const Matrix hbbInvHba =
      solveWithLdltFactor(preconditioner.hbbLdlt, preconditioner.Hbb,
                          preconditioner.Hba, "reduced-rotation Hbb/Hba");
  Matrix schur = Haa - preconditioner.Hab * hbbInvHba;
  schur = 0.5 * (schur + schur.transpose());

  if (mode == FullEquivHybridReducedRotationPreconditioner::Jacobi) {
    preconditioner.schurJacobi = buildJacobiPreconditioner(schur, true, 0);
  } else if (mode ==
             FullEquivHybridReducedRotationPreconditioner::SchurJacobi) {
    preconditioner.schurJacobi =
        buildJacobiPreconditioner(schur, true, plan.dimension);
  } else if (mode ==
             FullEquivHybridReducedRotationPreconditioner::Cholesky) {
    preconditioner.rotationSystem = Haa;
    preconditioner.rotationSystem =
        0.5 * (preconditioner.rotationSystem +
               preconditioner.rotationSystem.transpose());
    computeLdltOrThrow(preconditioner.rotationLdlt,
                       preconditioner.rotationSystem,
                       "reduced-rotation Haa");
    ++preconditioner.factorizationCount;
    if (!preconditioner.rotationLdlt.isPositive()) {
      throw std::runtime_error(
          "FE-Hybrid reduced-rotation preconditioner Haa is not SPD");
    }
  }
  return preconditioner;
}

Vector applyTranslationSchurPreconditioner(
    const TranslationSchurPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled) {
    return residual;
  }
  const Vector keepResidual =
      selectVectorEntries(residual, preconditioner.plan.keepColumns);
  const Vector eliminateResidual =
      selectVectorEntries(residual, preconditioner.plan.eliminateColumns);
  const Matrix hbbInvEliminate =
      solveWithLdltFactor(preconditioner.hbbLdlt, preconditioner.Hbb,
                          Matrix(eliminateResidual),
                          "translation Schur Hbb residual");
  const Matrix schurRhs =
      keepResidual - preconditioner.Hab * hbbInvEliminate;
  const Matrix keepStep =
      solveWithLdltFactor(preconditioner.sLdlt, preconditioner.S,
                          Matrix(schurRhs),
                          "translation Schur S residual");
  const Matrix eliminateRhs =
      eliminateResidual - preconditioner.Hba * keepStep;
  const Matrix eliminateStep =
      solveWithLdltFactor(preconditioner.hbbLdlt, preconditioner.Hbb,
                          Matrix(eliminateRhs),
                          "translation Schur Hbb backsolve");

  Vector result = Vector::Zero(residual.rows());
  scatterVectorEntries(result, keepStep, preconditioner.plan.keepColumns);
  scatterVectorEntries(result, eliminateStep,
                       preconditioner.plan.eliminateColumns);
  return result;
}

Vector applyReducedRotationPreconditioner(
    const ReducedRotationPreconditioner &preconditioner,
    const Vector &residual) {
  if (!preconditioner.enabled) {
    return residual;
  }
  const Vector keepResidual =
      selectVectorEntries(residual, preconditioner.plan.keepColumns);
  const Vector eliminateResidual =
      selectVectorEntries(residual, preconditioner.plan.eliminateColumns);
  const Matrix hbbInvEliminate =
      solveWithLdltFactor(preconditioner.hbbLdlt, preconditioner.Hbb,
                          Matrix(eliminateResidual),
                          "reduced-rotation Hbb residual");
  const Matrix reducedResidual =
      keepResidual - preconditioner.Hab * hbbInvEliminate;

  Matrix keepStep;
  if (preconditioner.mode ==
      FullEquivHybridReducedRotationPreconditioner::Cholesky) {
    keepStep = solveWithLdltFactor(preconditioner.rotationLdlt,
                                   preconditioner.rotationSystem,
                                   reducedResidual,
                                   "reduced-rotation Haa residual");
  } else if (preconditioner.mode ==
                 FullEquivHybridReducedRotationPreconditioner::Jacobi ||
             preconditioner.mode ==
                 FullEquivHybridReducedRotationPreconditioner::SchurJacobi) {
    keepStep = applyJacobiPreconditioner(
        preconditioner.schurJacobi, Vector(reducedResidual));
  } else {
    keepStep = reducedResidual;
  }

  const Matrix eliminateRhs =
      eliminateResidual - preconditioner.Hba * keepStep;
  const Matrix eliminateStep =
      solveWithLdltFactor(preconditioner.hbbLdlt, preconditioner.Hbb,
                          Matrix(eliminateRhs),
                          "reduced-rotation Hbb backsolve");

  Vector result = Vector::Zero(residual.rows());
  scatterVectorEntries(result, keepStep, preconditioner.plan.keepColumns);
  scatterVectorEntries(result, eliminateStep,
                       preconditioner.plan.eliminateColumns);
  return result;
}

JacobiPreconditioner buildJacobiPreconditioner(const Matrix &A,
                                               bool enabled,
                                               unsigned blockSize) {
  JacobiPreconditioner preconditioner;
  preconditioner.enabled = enabled;
  preconditioner.blockSize = blockSize;
  preconditioner.inverseDiagonal = buildJacobiInverseDiagonal(A);
  if (!enabled || blockSize == 0 ||
      A.rows() % static_cast<int>(blockSize) != 0) {
    return preconditioner;
  }

  const unsigned numBlocks = static_cast<unsigned>(A.rows()) / blockSize;
  preconditioner.inverseBlocks.reserve(numBlocks);
  for (unsigned block = 0; block < numBlocks; ++block) {
    const int start = static_cast<int>(block * blockSize);
    Matrix local = A.block(start, start, static_cast<int>(blockSize),
                           static_cast<int>(blockSize));
    local = 0.5 * (local + local.transpose());
    try {
      const Matrix inverseBlock =
          solveWithLdlt(local, Matrix::Identity(static_cast<int>(blockSize),
                                               static_cast<int>(blockSize)));
      preconditioner.inverseBlocks.push_back(inverseBlock);
    } catch (const std::exception &) {
      preconditioner.inverseBlocks.clear();
      return preconditioner;
    }
  }
  preconditioner.useBlocks =
      preconditioner.inverseBlocks.size() == static_cast<std::size_t>(numBlocks);
  return preconditioner;
}

Vector applyJacobiPreconditioner(const JacobiPreconditioner &preconditioner,
                                 const Vector &residual) {
  if (!preconditioner.enabled) {
    return residual;
  }
  if (preconditioner.useBlocks && preconditioner.blockSize > 0) {
    Vector result = Vector::Zero(residual.rows());
    for (std::size_t block = 0; block < preconditioner.inverseBlocks.size();
         ++block) {
      const int start =
          static_cast<int>(block * preconditioner.blockSize);
      const int size = static_cast<int>(preconditioner.blockSize);
      result.segment(start, size).noalias() =
          preconditioner.inverseBlocks[block] * residual.segment(start, size);
    }
    return result;
  }
  return preconditioner.inverseDiagonal.array() * residual.array();
}

JacobiPreconditioner buildSparseJacobiPreconditioner(const SparseMatrix &A,
                                                     double damping,
                                                     bool enabled,
                                                     unsigned blockSize) {
  JacobiPreconditioner preconditioner;
  preconditioner.enabled = enabled;
  preconditioner.blockSize = blockSize;
  preconditioner.inverseDiagonal = Vector::Ones(A.rows());
  for (int i = 0; i < A.rows(); ++i) {
    const double value = A.coeff(i, i) + damping;
    if (std::isfinite(value) && value > 1e-14) {
      preconditioner.inverseDiagonal(i) = 1.0 / value;
    }
  }
  if (!enabled || blockSize == 0 ||
      A.rows() % static_cast<int>(blockSize) != 0) {
    return preconditioner;
  }

  const unsigned numBlocks = static_cast<unsigned>(A.rows()) / blockSize;
  preconditioner.inverseBlocks.reserve(numBlocks);
  for (unsigned block = 0; block < numBlocks; ++block) {
    const int start = static_cast<int>(block * blockSize);
    Matrix local = Matrix::Zero(static_cast<int>(blockSize),
                                static_cast<int>(blockSize));
    for (unsigned row = 0; row < blockSize; ++row) {
      for (unsigned col = 0; col < blockSize; ++col) {
        const int r = start + static_cast<int>(row);
        const int c = start + static_cast<int>(col);
        local(static_cast<int>(row), static_cast<int>(col)) = A.coeff(r, c);
      }
    }
    local.diagonal().array() += damping;
    local = 0.5 * (local + local.transpose());
    try {
      const Matrix inverseBlock =
          solveWithLdlt(local, Matrix::Identity(static_cast<int>(blockSize),
                                               static_cast<int>(blockSize)));
      preconditioner.inverseBlocks.push_back(inverseBlock);
    } catch (const std::exception &) {
      preconditioner.inverseBlocks.clear();
      return preconditioner;
    }
  }
  preconditioner.useBlocks =
      preconditioner.inverseBlocks.size() == static_cast<std::size_t>(numBlocks);
  return preconditioner;
}

struct PcgSolveState {
  Matrix solution;
  unsigned iterations{0};
  double initialResidual{0.0};
  double finalResidual{0.0};
  bool converged{true};
  std::size_t translationSchurPreconditionerApplicationCount{0};
  std::size_t translationSchurPreconditionerFactorizationCount{0};
  std::size_t translationSchurPreconditionerFallbackCount{0};
  std::size_t localChainPreconditionerApplicationCount{0};
  std::size_t localChainPreconditionerFactorizationCount{0};
  std::size_t localChainPreconditionerFallbackCount{0};
  std::size_t translationBlockPreconditionerApplicationCount{0};
  std::size_t translationBlockPreconditionerFactorizationCount{0};
  std::size_t translationBlockPreconditionerFallbackCount{0};
  std::size_t translationSparseSchurPreconditionerApplicationCount{0};
  std::size_t translationSparseSchurPreconditionerFactorizationCount{0};
  std::size_t translationSparseSchurPreconditionerFallbackCount{0};
  std::size_t translationLocalSchurPreconditionerApplicationCount{0};
  std::size_t translationLocalSchurPreconditionerFactorizationCount{0};
  std::size_t translationLocalSchurPreconditionerFallbackCount{0};
  std::size_t translationLocalSchurPreconditionerActivePoseCount{0};
  std::size_t translationLocalSchurPreconditionerActiveColumnCount{0};
  std::size_t laplacianDeflationPreconditionerApplicationCount{0};
  std::size_t laplacianDeflationPreconditionerFactorizationCount{0};
  std::size_t laplacianDeflationPreconditionerFallbackCount{0};
  std::size_t laplacianDeflationPreconditionerBasisDimension{0};
  std::size_t reducedRotationPreconditionerApplicationCount{0};
  std::size_t reducedRotationPreconditionerFactorizationCount{0};
  std::size_t rqnMemoryPreconditionerApplicationCount{0};
  std::size_t sparseMatrixVectorProductCount{0};
};

PcgSolveState solveSymmetricPositiveSystemPcg(
    const Matrix &A, const Matrix &rhs,
    const FullEquivHybridPcgOptions &options, unsigned blockSize,
    const FullEquivHybridEliminationPlan *translationPlan = nullptr,
    const Matrix *initialSolution = nullptr) {
  if (A.rows() != A.cols() || A.rows() != rhs.rows()) {
    throw std::invalid_argument("FE-Hybrid PCG system dimension mismatch");
  }
  if (options.maxIterations == 0) {
    throw std::invalid_argument("FE-Hybrid PCG requires positive maxIterations");
  }
  if (initialSolution != nullptr &&
      (initialSolution->rows() != rhs.rows() ||
       initialSolution->cols() != rhs.cols())) {
    throw std::invalid_argument("FE-Hybrid PCG initial guess mismatch");
  }

  TranslationSchurPreconditioner translationSchurPreconditioner;
  std::size_t translationSchurFallbacks = 0;
  if (options.useTranslationSchurPreconditioner) {
    if (translationPlan == nullptr) {
      ++translationSchurFallbacks;
    } else {
      try {
        translationSchurPreconditioner =
            buildTranslationSchurPreconditioner(A, *translationPlan);
      } catch (const std::exception &) {
        translationSchurPreconditioner.enabled = false;
        ++translationSchurFallbacks;
      }
    }
  }
  std::size_t translationSchurFactorizations =
      translationSchurPreconditioner.factorizationCount;
  ReducedRotationPreconditioner reducedRotationPreconditioner;
  if (!translationSchurPreconditioner.enabled &&
      options.useReducedRotationPreconditioner &&
      translationPlan != nullptr) {
    try {
      reducedRotationPreconditioner =
          buildReducedRotationPreconditioner(
              A, *translationPlan, options.reducedRotationPreconditioner);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("FE-Hybrid reduced-rotation preconditioner failed: ") +
          e.what());
    }
  }
  std::size_t reducedRotationFactorizations =
      reducedRotationPreconditioner.factorizationCount;
  JacobiPreconditioner jacobiPreconditioner = buildJacobiPreconditioner(
      A,
      options.useBlockJacobiPreconditioner &&
          !translationSchurPreconditioner.enabled &&
          !reducedRotationPreconditioner.enabled,
      blockSize);
  std::size_t translationSchurApplications = 0;
  std::size_t reducedRotationApplications = 0;
  std::size_t rqnMemoryApplications = 0;
  auto applyRqnMemoryPreconditioner = [&](int rhsCol,
                                          const Vector &residual,
                                          Vector *preconditioned) {
    if (!options.useRqnMemoryPreconditioner ||
        options.rqnMemoryPreconditioner == nullptr ||
        options.rqnMemoryPreconditioner->size() == 0) {
      return false;
    }
    try {
      Vector candidate =
          options.rqnMemoryPreconditioner->applyInverseHessianRow(
              static_cast<unsigned>(rhsCol), residual);
      if (candidate.rows() == residual.rows() && candidate.allFinite() &&
          residual.dot(candidate) > 0.0) {
        *preconditioned = candidate;
        ++rqnMemoryApplications;
        return true;
      }
    } catch (const std::exception &) {
    }
    return false;
  };
  auto applyPreconditioner = [&](const Vector &residual, int rhsCol) {
    Vector rqnPreconditioned;
    if (applyRqnMemoryPreconditioner(rhsCol, residual,
                                     &rqnPreconditioned)) {
      return rqnPreconditioned;
    }
    if (translationSchurPreconditioner.enabled) {
      ++translationSchurApplications;
      return applyTranslationSchurPreconditioner(
          translationSchurPreconditioner, residual);
    }
    if (reducedRotationPreconditioner.enabled) {
      ++reducedRotationApplications;
      return applyReducedRotationPreconditioner(
          reducedRotationPreconditioner, residual);
    }
    return applyJacobiPreconditioner(jacobiPreconditioner, residual);
  };
  Matrix solution = Matrix::Zero(rhs.rows(), rhs.cols());
  double maxInitialResidual = 0.0;
  double maxFinalResidual = 0.0;
  unsigned maxIterationsUsed = 0;
  bool allConverged = true;

  for (int col = 0; col < rhs.cols(); ++col) {
    const Vector b = rhs.col(col);
    Vector x;
    if (initialSolution == nullptr) {
      x = Vector::Zero(A.cols());
    } else {
      x = initialSolution->col(col);
    }
    Vector residual = b - A * x;
    const double initialResidual = residual.norm();
    const double referenceResidual = b.norm();
    maxInitialResidual = std::max(maxInitialResidual, initialResidual);
    const double tolerance = std::max(
        options.absoluteTolerance, options.relativeTolerance * referenceResidual);
    if (initialResidual <= tolerance) {
      maxFinalResidual = std::max(maxFinalResidual, initialResidual);
      solution.col(col) = x;
      continue;
    }

    Vector z = applyPreconditioner(residual, col);
    Vector direction = z;
    double rz = residual.dot(z);
    if (!std::isfinite(rz) || rz <= 0.0) {
      throw std::runtime_error("FE-Hybrid PCG invalid initial curvature");
    }

    double finalResidual = initialResidual;
    unsigned iter = 0;
    for (; iter < options.maxIterations; ++iter) {
      const Vector Adirection = A * direction;
      const double denom = direction.dot(Adirection);
      if (!std::isfinite(denom) || denom <= 0.0) {
        throw std::runtime_error("FE-Hybrid PCG non-positive curvature");
      }
      const double alpha = rz / denom;
      x += alpha * direction;
      residual -= alpha * Adirection;
      finalResidual = residual.norm();
      if (!std::isfinite(finalResidual)) {
        throw std::runtime_error("FE-Hybrid PCG residual became invalid");
      }
      if (finalResidual <= tolerance) {
        ++iter;
        break;
      }
      z = applyPreconditioner(residual, col);
      const double rzNext = residual.dot(z);
      if (!std::isfinite(rzNext) || rzNext <= 0.0) {
        throw std::runtime_error("FE-Hybrid PCG invalid recurrence curvature");
      }
      const double beta = rzNext / rz;
      direction = z + beta * direction;
      rz = rzNext;
    }
    if (finalResidual > tolerance) {
      allConverged = false;
    }
    solution.col(col) = x;
    maxFinalResidual = std::max(maxFinalResidual, finalResidual);
    maxIterationsUsed = std::max(maxIterationsUsed, iter);
  }

  PcgSolveState state;
  state.solution = solution;
  state.iterations = maxIterationsUsed;
  state.initialResidual = maxInitialResidual;
  state.finalResidual = maxFinalResidual;
  state.converged = allConverged;
  state.translationSchurPreconditionerApplicationCount =
      translationSchurApplications;
  state.translationSchurPreconditionerFactorizationCount =
      translationSchurFactorizations;
  state.translationSchurPreconditionerFallbackCount =
      translationSchurFallbacks;
  state.reducedRotationPreconditionerApplicationCount =
      reducedRotationApplications;
  state.reducedRotationPreconditionerFactorizationCount =
      reducedRotationFactorizations;
  state.rqnMemoryPreconditionerApplicationCount =
      rqnMemoryApplications;
  return state;
}

PcgSolveState solveSymmetricPositiveSparseSystemPcg(
    const SparseMatrix &A, double damping, const Matrix &rhs,
    const FullEquivHybridPcgOptions &options, unsigned blockSize,
    const Matrix *initialSolution = nullptr) {
  if (A.rows() != A.cols() || A.rows() != rhs.rows()) {
    throw std::invalid_argument("FE-Hybrid sparse PCG system dimension mismatch");
  }
  if (options.maxIterations == 0) {
    throw std::invalid_argument(
        "FE-Hybrid sparse PCG requires positive maxIterations");
  }
  if (initialSolution != nullptr &&
      (initialSolution->rows() != rhs.rows() ||
       initialSolution->cols() != rhs.cols())) {
    throw std::invalid_argument("FE-Hybrid sparse PCG initial guess mismatch");
  }

  JacobiPreconditioner jacobiPreconditioner =
      buildSparseJacobiPreconditioner(
          A, damping,
	          options.useBlockJacobiPreconditioner &&
	              !options.useLocalChainPreconditioner &&
	              !options.useTranslationBlockPreconditioner &&
	              !options.useTranslationSparseSchurPreconditioner &&
	              !options.useTranslationLocalSchurPreconditioner,
	          blockSize);
  TranslationLocalSchurPreconditioner translationLocalSchurPreconditioner;
  std::size_t translationLocalSchurFallbacks = 0;
  if (options.useTranslationLocalSchurPreconditioner) {
    try {
      const unsigned dimension = blockSize - 1;
      translationLocalSchurPreconditioner =
          buildTranslationLocalSchurPreconditioner(
              A, damping, rhs, dimension,
              options.translationLocalSchurMaxActivePoses,
              options.useBlockJacobiPreconditioner);
    } catch (const std::exception &) {
      translationLocalSchurPreconditioner.enabled = false;
      ++translationLocalSchurFallbacks;
      jacobiPreconditioner = buildSparseJacobiPreconditioner(
          A, damping, options.useBlockJacobiPreconditioner, blockSize);
    }
  }
  TranslationSparseSchurPreconditioner translationSparseSchurPreconditioner;
  std::size_t translationSparseSchurFallbacks = 0;
  if (!translationLocalSchurPreconditioner.enabled &&
      options.useTranslationSparseSchurPreconditioner) {
    try {
      const unsigned dimension = blockSize - 1;
      const unsigned numPoses =
          static_cast<unsigned>(A.cols()) / blockSize;
      const FullEquivHybridEliminationPlan plan =
          makeFullEquivHybridTranslationEliminationPlan(numPoses, dimension);
      translationSparseSchurPreconditioner =
          buildTranslationSparseSchurPreconditioner(A, damping, plan);
    } catch (const std::exception &) {
      translationSparseSchurPreconditioner.enabled = false;
      ++translationSparseSchurFallbacks;
      jacobiPreconditioner = buildSparseJacobiPreconditioner(
          A, damping, options.useBlockJacobiPreconditioner, blockSize);
    }
  }
	  TranslationBlockPreconditioner translationBlockPreconditioner;
	  std::size_t translationBlockFallbacks = 0;
	  LocalChainPreconditioner localChainPreconditioner;
	  std::size_t localChainFallbacks = 0;
	  if (!translationLocalSchurPreconditioner.enabled &&
	      !translationSparseSchurPreconditioner.enabled &&
	      options.useLocalChainPreconditioner) {
	    try {
	      localChainPreconditioner =
	          buildLocalChainPreconditioner(A, damping, blockSize);
	    } catch (const std::exception &) {
	      localChainPreconditioner.enabled = false;
	      ++localChainFallbacks;
	      jacobiPreconditioner = buildSparseJacobiPreconditioner(
	          A, damping, options.useBlockJacobiPreconditioner, blockSize);
	    }
	  }
	  if (!translationLocalSchurPreconditioner.enabled &&
	      !translationSparseSchurPreconditioner.enabled &&
	      !localChainPreconditioner.enabled &&
	      options.useTranslationBlockPreconditioner) {
    try {
      const unsigned dimension = blockSize - 1;
      const unsigned numPoses =
          static_cast<unsigned>(A.cols()) / blockSize;
      const FullEquivHybridEliminationPlan plan =
          makeFullEquivHybridTranslationEliminationPlan(numPoses, dimension);
      translationBlockPreconditioner =
          buildTranslationBlockPreconditioner(
              A, damping, plan, options.useBlockJacobiPreconditioner,
              blockSize);
    } catch (const std::exception &) {
      translationBlockPreconditioner.enabled = false;
      ++translationBlockFallbacks;
      jacobiPreconditioner = buildSparseJacobiPreconditioner(
          A, damping, options.useBlockJacobiPreconditioner, blockSize);
    }
  }
  LaplacianDeflationPreconditioner laplacianDeflationPreconditioner;
  std::size_t laplacianDeflationFallbacks = 0;
  if (options.useLaplacianDeflationPreconditioner) {
    try {
      const unsigned dimension = blockSize - 1;
      laplacianDeflationPreconditioner =
          buildLaplacianDeflationPreconditioner(
              A, damping, dimension, options.laplacianDeflationBasisSize,
              options.laplacianDeflationMaxEigenPoses);
    } catch (const std::exception &) {
      laplacianDeflationPreconditioner.enabled = false;
      ++laplacianDeflationFallbacks;
    }
  }
  Matrix solution = Matrix::Zero(rhs.rows(), rhs.cols());
  double maxInitialResidual = 0.0;
  double maxFinalResidual = 0.0;
  unsigned maxIterationsUsed = 0;
  bool allConverged = true;
  std::size_t sparseMatvecs = 0;
	  std::size_t translationLocalSchurApplications = 0;
	  std::size_t translationSparseSchurApplications = 0;
	  std::size_t localChainApplications = 0;
	  std::size_t translationBlockApplications = 0;
  std::size_t laplacianDeflationApplications = 0;
  std::size_t rqnMemoryApplications = 0;
  auto multiply = [&](const Vector &x) {
    ++sparseMatvecs;
    Vector result = A * x;
    if (damping != 0.0) {
      result.noalias() += damping * x;
    }
    return result;
  };
  auto applyRqnMemoryPreconditioner = [&](int rhsCol,
                                          const Vector &residual,
                                          Vector *preconditioned) {
    if (!options.useRqnMemoryPreconditioner ||
        options.rqnMemoryPreconditioner == nullptr ||
        options.rqnMemoryPreconditioner->size() == 0) {
      return false;
    }
    try {
      Vector candidate =
          options.rqnMemoryPreconditioner->applyInverseHessianRow(
              static_cast<unsigned>(rhsCol), residual);
      if (candidate.rows() == residual.rows() && candidate.allFinite() &&
          residual.dot(candidate) > 0.0) {
        *preconditioned = candidate;
        ++rqnMemoryApplications;
        return true;
      }
    } catch (const std::exception &) {
    }
    return false;
  };

  for (int col = 0; col < rhs.cols(); ++col) {
    const Vector b = rhs.col(col);
    Vector x;
    Vector residual;
    if (initialSolution == nullptr) {
      x = Vector::Zero(A.cols());
      residual = b;
    } else {
      x = initialSolution->col(col);
      residual = b - multiply(x);
    }
    const double initialResidual = residual.norm();
    const double referenceResidual = b.norm();
    maxInitialResidual = std::max(maxInitialResidual, initialResidual);
    const double tolerance = std::max(
        options.absoluteTolerance, options.relativeTolerance * referenceResidual);
    if (initialResidual <= tolerance) {
      maxFinalResidual = std::max(maxFinalResidual, initialResidual);
      solution.col(col) = x;
      continue;
    }

    auto applyPreconditioner = [&](const Vector &r) {
      Vector rqnPreconditioned;
      if (applyRqnMemoryPreconditioner(col, r, &rqnPreconditioned)) {
        return rqnPreconditioned;
      }
      Vector preconditioned;
      if (translationLocalSchurPreconditioner.enabled) {
        ++translationLocalSchurApplications;
        preconditioned = applyTranslationLocalSchurPreconditioner(
            translationLocalSchurPreconditioner, r);
	      } else if (translationSparseSchurPreconditioner.enabled) {
	        ++translationSparseSchurApplications;
	        preconditioned = applyTranslationSparseSchurPreconditioner(
	            translationSparseSchurPreconditioner, r);
	      } else if (localChainPreconditioner.enabled) {
	        ++localChainApplications;
	        preconditioned =
	            applyLocalChainPreconditioner(localChainPreconditioner, r);
	      } else if (translationBlockPreconditioner.enabled) {
	        ++translationBlockApplications;
        preconditioned = applyTranslationBlockPreconditioner(
            translationBlockPreconditioner, r);
      } else {
        preconditioned = applyJacobiPreconditioner(jacobiPreconditioner, r);
      }
      if (laplacianDeflationPreconditioner.enabled) {
        ++laplacianDeflationApplications;
        preconditioned += applyLaplacianDeflationPreconditioner(
            laplacianDeflationPreconditioner, r);
      }
      return preconditioned;
    };

    Vector z = applyPreconditioner(residual);
    Vector direction = z;
    double rz = residual.dot(z);
    if (!std::isfinite(rz) || rz <= 0.0) {
      throw std::runtime_error("FE-Hybrid sparse PCG invalid initial curvature");
    }

    double finalResidual = initialResidual;
    unsigned iter = 0;
    for (; iter < options.maxIterations; ++iter) {
      const Vector Adirection = multiply(direction);
      const double denom = direction.dot(Adirection);
      if (!std::isfinite(denom) || denom <= 0.0) {
        throw std::runtime_error("FE-Hybrid sparse PCG non-positive curvature");
      }
      const double alpha = rz / denom;
      x += alpha * direction;
      residual -= alpha * Adirection;
      finalResidual = residual.norm();
      if (!std::isfinite(finalResidual)) {
        throw std::runtime_error("FE-Hybrid sparse PCG residual became invalid");
      }
      if (finalResidual <= tolerance) {
        ++iter;
        break;
      }
      z = applyPreconditioner(residual);
      const double rzNext = residual.dot(z);
      if (!std::isfinite(rzNext) || rzNext <= 0.0) {
        throw std::runtime_error(
            "FE-Hybrid sparse PCG invalid recurrence curvature");
      }
      const double beta = rzNext / rz;
      direction = z + beta * direction;
      rz = rzNext;
    }
    if (finalResidual > tolerance) {
      allConverged = false;
    }
    solution.col(col) = x;
    maxFinalResidual = std::max(maxFinalResidual, finalResidual);
    maxIterationsUsed = std::max(maxIterationsUsed, iter);
  }

  PcgSolveState state;
  state.solution = solution;
  state.iterations = maxIterationsUsed;
  state.initialResidual = maxInitialResidual;
  state.finalResidual = maxFinalResidual;
  state.converged = allConverged;
	  state.sparseMatrixVectorProductCount = sparseMatvecs;
	  state.localChainPreconditionerApplicationCount = localChainApplications;
	  state.localChainPreconditionerFactorizationCount =
	      localChainPreconditioner.factorizationCount;
	  state.localChainPreconditionerFallbackCount = localChainFallbacks;
	  state.translationBlockPreconditionerApplicationCount =
      translationBlockApplications;
  state.translationBlockPreconditionerFactorizationCount =
      translationBlockPreconditioner.factorizationCount;
  state.translationBlockPreconditionerFallbackCount = translationBlockFallbacks;
  state.translationSparseSchurPreconditionerApplicationCount =
      translationSparseSchurApplications;
  state.translationSparseSchurPreconditionerFactorizationCount =
      translationSparseSchurPreconditioner.factorizationCount;
  state.translationSparseSchurPreconditionerFallbackCount =
      translationSparseSchurFallbacks;
  state.translationLocalSchurPreconditionerApplicationCount =
      translationLocalSchurApplications;
  state.translationLocalSchurPreconditionerFactorizationCount =
      translationLocalSchurPreconditioner.factorizationCount;
  state.translationLocalSchurPreconditionerFallbackCount =
      translationLocalSchurFallbacks;
  state.translationLocalSchurPreconditionerActivePoseCount =
      translationLocalSchurPreconditioner.activePoseCount;
  state.translationLocalSchurPreconditionerActiveColumnCount =
      translationLocalSchurPreconditioner.activeColumnCount;
  state.laplacianDeflationPreconditionerApplicationCount =
      laplacianDeflationApplications;
  state.laplacianDeflationPreconditionerFactorizationCount =
      laplacianDeflationPreconditioner.factorizationCount;
  state.laplacianDeflationPreconditionerFallbackCount =
      laplacianDeflationFallbacks +
      laplacianDeflationPreconditioner.fallbackCount;
  state.laplacianDeflationPreconditionerBasisDimension =
      static_cast<std::size_t>(
          laplacianDeflationPreconditioner.basis.cols());
  state.rqnMemoryPreconditionerApplicationCount = rqnMemoryApplications;
  return state;
}

}  // namespace

FullEquivHybridRqnMemory::FullEquivHybridRqnMemory(
    const FullEquivHybridRqnOptions &options)
    : options(options) {}

bool FullEquivHybridRqnMemory::addPair(
    const Matrix &step, const Matrix &gradientDifference) {
  auto reject = [&]() {
    ++rejectedPairs;
    return false;
  };
  if (options.memorySize == 0 || step.rows() == 0 || step.cols() == 0 ||
      step.rows() != gradientDifference.rows() ||
      step.cols() != gradientDifference.cols() || !step.allFinite() ||
      !gradientDifference.allFinite()) {
    return reject();
  }

  const double sy = matrixInnerProduct(step, gradientDifference);
  const double ss = matrixInnerProduct(step, step);
  const double yy = matrixInnerProduct(gradientDifference, gradientDifference);
  const double curvatureThreshold =
      std::max(options.minCurvatureAbsolute,
               options.minCurvatureRatio * std::sqrt(ss * yy));
  if (!std::isfinite(sy) || !std::isfinite(ss) || !std::isfinite(yy) ||
      ss <= 0.0 || yy <= 0.0 || sy <= curvatureThreshold) {
    return reject();
  }

  if (pairs.size() == options.memorySize) {
    pairs.erase(pairs.begin());
  }
  Pair pair;
  pair.step = step;
  pair.gradientDifference = gradientDifference;
  pair.rho = 1.0 / sy;
  pairs.push_back(pair);
  ++acceptedPairs;
  return true;
}

Matrix FullEquivHybridRqnMemory::applyInverseHessian(
    const Matrix &gradient) const {
  if (pairs.empty()) {
    return gradient;
  }
  const Pair &latest = pairs.back();
  if (gradient.rows() != latest.step.rows() ||
      gradient.cols() != latest.step.cols() || !gradient.allFinite()) {
    throw std::invalid_argument("FE-Hybrid RQN gradient shape mismatch");
  }

  std::vector<double> alpha(pairs.size(), 0.0);
  Matrix q = gradient;
  for (std::size_t reverse = pairs.size(); reverse > 0; --reverse) {
    const std::size_t i = reverse - 1;
    alpha[i] = pairs[i].rho * matrixInnerProduct(pairs[i].step, q);
    q -= alpha[i] * pairs[i].gradientDifference;
  }

  const double latestSy =
      matrixInnerProduct(latest.step, latest.gradientDifference);
  const double latestYy =
      matrixInnerProduct(latest.gradientDifference,
                         latest.gradientDifference);
  double gamma = 1.0;
  if (std::isfinite(latestSy) && std::isfinite(latestYy) &&
      latestSy > 0.0 && latestYy > 0.0) {
    gamma = latestSy / latestYy;
  }
  Matrix result = gamma * q;

  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const double beta =
        pairs[i].rho * matrixInnerProduct(pairs[i].gradientDifference,
                                          result);
    result += pairs[i].step * (alpha[i] - beta);
  }
  return result;
}

Vector FullEquivHybridRqnMemory::applyInverseHessianRow(
    unsigned row, const Vector &gradient) const {
  if (pairs.empty()) {
    return gradient;
  }
  const Pair &latest = pairs.back();
  if (row >= static_cast<unsigned>(latest.step.rows()) ||
      gradient.rows() != latest.step.cols() || !gradient.allFinite()) {
    throw std::invalid_argument("FE-Hybrid RQN row gradient shape mismatch");
  }

  struct ActivePair {
    Vector step;
    Vector gradientDifference;
    double rho{0.0};
  };
  std::vector<ActivePair> activePairs;
  activePairs.reserve(pairs.size());
  for (const Pair &pair : pairs) {
    if (row >= static_cast<unsigned>(pair.step.rows()) ||
        row >= static_cast<unsigned>(pair.gradientDifference.rows()) ||
        pair.step.cols() != gradient.rows() ||
        pair.gradientDifference.cols() != gradient.rows()) {
      continue;
    }
    ActivePair active;
    active.step = pair.step.row(static_cast<int>(row)).transpose();
    active.gradientDifference =
        pair.gradientDifference.row(static_cast<int>(row)).transpose();
    const double sy = active.step.dot(active.gradientDifference);
    const double ss = active.step.squaredNorm();
    const double yy = active.gradientDifference.squaredNorm();
    const double curvatureThreshold =
        std::max(options.minCurvatureAbsolute,
                 options.minCurvatureRatio * std::sqrt(ss * yy));
    if (std::isfinite(sy) && std::isfinite(ss) && std::isfinite(yy) &&
        ss > 0.0 && yy > 0.0 && sy > curvatureThreshold) {
      active.rho = 1.0 / sy;
      activePairs.push_back(std::move(active));
    }
  }
	  if (activePairs.empty()) {
	    throw std::runtime_error(
	        "FE-Hybrid RQN row has no valid curvature pair");
	  }

  std::vector<double> alpha(activePairs.size(), 0.0);
  Vector q = gradient;
  for (std::size_t reverse = activePairs.size(); reverse > 0; --reverse) {
    const std::size_t i = reverse - 1;
    alpha[i] = activePairs[i].rho * activePairs[i].step.dot(q);
    q -= alpha[i] * activePairs[i].gradientDifference;
  }

  const ActivePair &latestActive = activePairs.back();
  const double latestSy =
      latestActive.step.dot(latestActive.gradientDifference);
  const double latestYy = latestActive.gradientDifference.squaredNorm();
  double gamma = 1.0;
  if (std::isfinite(latestSy) && std::isfinite(latestYy) &&
      latestSy > 0.0 && latestYy > 0.0) {
    gamma = latestSy / latestYy;
  }
  Vector result = gamma * q;

  for (std::size_t i = 0; i < activePairs.size(); ++i) {
    const double beta =
        activePairs[i].rho *
        activePairs[i].gradientDifference.dot(result);
    result += activePairs[i].step * (alpha[i] - beta);
  }
  return result;
}

void FullEquivHybridRqnMemory::clear() {
  pairs.clear();
  acceptedPairs = 0;
  rejectedPairs = 0;
}

std::size_t FullEquivHybridRqnMemory::size() const {
  return pairs.size();
}

std::size_t FullEquivHybridRqnMemory::acceptedPairCount() const {
  return acceptedPairs;
}

std::size_t FullEquivHybridRqnMemory::rejectedPairCount() const {
  return rejectedPairs;
}

FullEquivHybridEliminationPlan
makeFullEquivHybridTranslationEliminationPlan(unsigned numPoses,
                                              unsigned dimension) {
  if (numPoses == 0 || dimension == 0) {
    throw std::invalid_argument(
        "FE-Hybrid translation elimination requires nonzero dimensions");
  }

  FullEquivHybridEliminationPlan plan;
  plan.numPoses = numPoses;
  plan.dimension = dimension;
  const unsigned blockWidth = dimension + 1;
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned base = pose * blockWidth;
    for (unsigned c = 0; c < dimension; ++c) {
      plan.keepColumns.push_back(static_cast<int>(base + c));
    }
    plan.eliminateColumns.push_back(static_cast<int>(base + dimension));
  }
  return plan;
}

FullEquivHybridSchurSystem buildFullEquivHybridExactSchurSystem(
    const SparseMatrix &H, const Matrix &gradient,
    const FullEquivHybridEliminationPlan &plan, double damping) {
  if (gradient.cols() != H.cols()) {
    throw std::invalid_argument("FE-Hybrid gradient/system column mismatch");
  }
  validatePlan(plan, static_cast<int>(H.cols()));

  const Matrix dense = dampedDenseSystem(H, damping);
  FullEquivHybridSchurSystem schur;
  schur.Haa = selectRowsCols(dense, plan.keepColumns, plan.keepColumns);
  schur.Hab = selectRowsCols(dense, plan.keepColumns, plan.eliminateColumns);
  schur.Hba = selectRowsCols(dense, plan.eliminateColumns, plan.keepColumns);
  schur.Hbb =
      selectRowsCols(dense, plan.eliminateColumns, plan.eliminateColumns);

  const Matrix gKeep = selectGradientColumns(gradient, plan.keepColumns);
  const Matrix gEliminate =
      selectGradientColumns(gradient, plan.eliminateColumns);
  const Matrix hbbInvHba = solveWithLdlt(schur.Hbb, schur.Hba);
  const Matrix hbbInvGb = solveWithLdlt(schur.Hbb, gEliminate.transpose());
  schur.S = schur.Haa - schur.Hab * hbbInvHba;
  schur.S = 0.5 * (schur.S + schur.S.transpose());
  schur.rhs = -gKeep + hbbInvGb.transpose() * schur.Hba;
  return schur;
}

Matrix solveFullEquivHybridFullDirect(const SparseMatrix &H,
                                      const Matrix &gradient,
                                      double damping) {
  if (gradient.cols() != H.cols()) {
    throw std::invalid_argument("FE-Hybrid gradient/system column mismatch");
  }
  const Matrix dense = dampedDenseSystem(H, damping);
  return solveWithLdlt(dense, -gradient.transpose()).transpose();
}

Matrix solveFullEquivHybridSchurDirect(
    const FullEquivHybridSchurSystem &schur, const Matrix &gradient,
    const FullEquivHybridEliminationPlan &plan) {
  validatePlan(plan, static_cast<int>(gradient.cols()));
  const Matrix gEliminate =
      selectGradientColumns(gradient, plan.eliminateColumns);

  Matrix step = Matrix::Zero(gradient.rows(), gradient.cols());
  const Matrix deltaKeep =
      solveWithLdlt(schur.S, schur.rhs.transpose()).transpose();
  const Matrix eliminatedRhs =
      -gEliminate.transpose() - schur.Hba * deltaKeep.transpose();
  const Matrix deltaEliminate =
      solveWithLdlt(schur.Hbb, eliminatedRhs).transpose();

  scatterStepColumns(step, deltaKeep, plan.keepColumns);
  scatterStepColumns(step, deltaEliminate, plan.eliminateColumns);
  return step;
}

FullEquivHybridPcgResult solveFullEquivHybridSchurPcg(
    const FullEquivHybridSchurSystem &schur, const Matrix &gradient,
    const FullEquivHybridEliminationPlan &plan,
    const FullEquivHybridPcgOptions &options) {
  validatePlan(plan, static_cast<int>(gradient.cols()));
	  if (options.useRqnMemoryPreconditioner) {
	    throw std::invalid_argument(
	        "FE-Hybrid RQN memory preconditioner requires PCG_FULL");
	  }
	  if (options.useLocalChainPreconditioner) {
	    throw std::invalid_argument(
	        "FE-Hybrid local-chain preconditioner requires PCG_FULL");
	  }
  const Matrix gEliminate =
      selectGradientColumns(gradient, plan.eliminateColumns);
  Matrix initialKeepGuess;
  const Matrix *initialKeepGuessPtr = nullptr;
  if (options.initialStepGuess.size() != 0) {
    if (options.initialStepGuess.rows() != gradient.rows() ||
        options.initialStepGuess.cols() != gradient.cols()) {
      throw std::invalid_argument(
          "FE-Hybrid Schur PCG initial step guess mismatch");
    }
    initialKeepGuess =
        selectGradientColumns(options.initialStepGuess, plan.keepColumns)
            .transpose();
    initialKeepGuessPtr = &initialKeepGuess;
  }

  PcgSolveState reducedSolve =
      solveSymmetricPositiveSystemPcg(
          schur.S, schur.rhs.transpose(), options, plan.dimension, nullptr,
          initialKeepGuessPtr);
  const Matrix deltaKeep = reducedSolve.solution.transpose();
  const Matrix eliminatedRhs =
      -gEliminate.transpose() - schur.Hba * deltaKeep.transpose();
  const Matrix deltaEliminate =
      solveWithLdlt(schur.Hbb, eliminatedRhs).transpose();

  FullEquivHybridPcgResult result;
  result.step = Matrix::Zero(gradient.rows(), gradient.cols());
  scatterStepColumns(result.step, deltaKeep, plan.keepColumns);
  scatterStepColumns(result.step, deltaEliminate, plan.eliminateColumns);
  result.iterations = reducedSolve.iterations;
  result.initialResidual = reducedSolve.initialResidual;
  result.finalResidual = reducedSolve.finalResidual;
  result.converged = reducedSolve.converged;
  result.translationSchurPreconditionerApplicationCount =
      reducedSolve.translationSchurPreconditionerApplicationCount;
  result.translationSchurPreconditionerFactorizationCount =
      reducedSolve.translationSchurPreconditionerFactorizationCount;
	  result.translationSchurPreconditionerFallbackCount =
	      reducedSolve.translationSchurPreconditionerFallbackCount;
	  result.localChainPreconditionerApplicationCount =
	      reducedSolve.localChainPreconditionerApplicationCount;
	  result.localChainPreconditionerFactorizationCount =
	      reducedSolve.localChainPreconditionerFactorizationCount;
	  result.localChainPreconditionerFallbackCount =
	      reducedSolve.localChainPreconditionerFallbackCount;
	  result.translationBlockPreconditionerApplicationCount =
	      reducedSolve.translationBlockPreconditionerApplicationCount;
  result.translationBlockPreconditionerFactorizationCount =
      reducedSolve.translationBlockPreconditionerFactorizationCount;
  result.translationBlockPreconditionerFallbackCount =
      reducedSolve.translationBlockPreconditionerFallbackCount;
  result.translationSparseSchurPreconditionerApplicationCount =
      reducedSolve.translationSparseSchurPreconditionerApplicationCount;
  result.translationSparseSchurPreconditionerFactorizationCount =
      reducedSolve.translationSparseSchurPreconditionerFactorizationCount;
  result.translationSparseSchurPreconditionerFallbackCount =
      reducedSolve.translationSparseSchurPreconditionerFallbackCount;
  result.translationLocalSchurPreconditionerApplicationCount =
      reducedSolve.translationLocalSchurPreconditionerApplicationCount;
  result.translationLocalSchurPreconditionerFactorizationCount =
      reducedSolve.translationLocalSchurPreconditionerFactorizationCount;
  result.translationLocalSchurPreconditionerFallbackCount =
      reducedSolve.translationLocalSchurPreconditionerFallbackCount;
  result.translationLocalSchurPreconditionerActivePoseCount =
      reducedSolve.translationLocalSchurPreconditionerActivePoseCount;
  result.translationLocalSchurPreconditionerActiveColumnCount =
      reducedSolve.translationLocalSchurPreconditionerActiveColumnCount;
  result.reducedRotationPreconditionerApplicationCount =
      reducedSolve.reducedRotationPreconditionerApplicationCount;
  result.reducedRotationPreconditionerFactorizationCount =
      reducedSolve.reducedRotationPreconditionerFactorizationCount;
  result.rqnMemoryPreconditionerApplicationCount =
      reducedSolve.rqnMemoryPreconditionerApplicationCount;
  result.sparseMatrixVectorProductCount =
      reducedSolve.sparseMatrixVectorProductCount;
  return result;
}

FullEquivHybridPcgResult solveFullEquivHybridFullPcg(
    const SparseMatrix &H, const Matrix &gradient, double damping,
    unsigned dimension, const FullEquivHybridPcgOptions &options) {
  if (gradient.cols() != H.cols()) {
    throw std::invalid_argument("FE-Hybrid gradient/system column mismatch");
  }
  if (dimension == 0 ||
      H.cols() % static_cast<int>(dimension + 1) != 0) {
    throw std::invalid_argument("FE-Hybrid full PCG block dimension mismatch");
  }
  if (options.useLaplacianDeflationPreconditioner &&
      (options.useTranslationSchurPreconditioner ||
       options.useReducedRotationPreconditioner)) {
    throw std::invalid_argument(
        "FE-Hybrid Laplacian deflation requires the sparse PCG_FULL path");
  }
  if (options.useLocalChainPreconditioner &&
      (options.useTranslationSchurPreconditioner ||
       options.useReducedRotationPreconditioner)) {
    throw std::invalid_argument(
        "FE-Hybrid local-chain preconditioner requires the sparse PCG_FULL path");
  }
  if (options.useRqnMemoryPreconditioner &&
      (options.useTranslationSchurPreconditioner ||
       options.useTranslationBlockPreconditioner ||
       options.useTranslationSparseSchurPreconditioner ||
       options.useTranslationLocalSchurPreconditioner ||
       options.useLocalChainPreconditioner ||
       options.useLaplacianDeflationPreconditioner ||
       options.useReducedRotationPreconditioner)) {
    throw std::invalid_argument(
        "FE-Hybrid RQN memory preconditioner is a standalone PCG_FULL base preconditioner");
  }

  const unsigned numPoses =
      static_cast<unsigned>(H.cols()) / (dimension + 1);
  const FullEquivHybridEliminationPlan plan =
      makeFullEquivHybridTranslationEliminationPlan(numPoses, dimension);
  Matrix initialFullGuess;
  const Matrix *initialFullGuessPtr = nullptr;
  if (options.initialStepGuess.size() != 0) {
    if (options.initialStepGuess.rows() != gradient.rows() ||
        options.initialStepGuess.cols() != gradient.cols()) {
      throw std::invalid_argument(
          "FE-Hybrid full PCG initial step guess mismatch");
    }
    initialFullGuess = options.initialStepGuess.transpose();
    initialFullGuessPtr = &initialFullGuess;
  }
  const Matrix rhs = -gradient.transpose();
  const bool sparseOnlyPreconditionerRequested =
      options.useTranslationBlockPreconditioner ||
      options.useTranslationSparseSchurPreconditioner ||
      options.useTranslationLocalSchurPreconditioner ||
      options.useLocalChainPreconditioner ||
      options.useLaplacianDeflationPreconditioner;
  const bool canUseSparseMatvec =
      (options.useSparseMatrixVectorProduct ||
       sparseOnlyPreconditionerRequested) &&
      !options.useTranslationSchurPreconditioner &&
      !options.useReducedRotationPreconditioner;
  PcgSolveState fullSolve;
  if (canUseSparseMatvec) {
    fullSolve = solveSymmetricPositiveSparseSystemPcg(
        H, damping, rhs, options, dimension + 1, initialFullGuessPtr);
  } else {
    const Matrix dense = dampedDenseSystem(H, damping);
    fullSolve = solveSymmetricPositiveSystemPcg(
        dense, rhs, options, dimension + 1, &plan, initialFullGuessPtr);
  }

  FullEquivHybridPcgResult result;
  result.step = fullSolve.solution.transpose();
  result.iterations = fullSolve.iterations;
  result.initialResidual = fullSolve.initialResidual;
  result.finalResidual = fullSolve.finalResidual;
  result.converged = fullSolve.converged;
  result.translationSchurPreconditionerApplicationCount =
      fullSolve.translationSchurPreconditionerApplicationCount;
  result.translationSchurPreconditionerFactorizationCount =
      fullSolve.translationSchurPreconditionerFactorizationCount;
	  result.translationSchurPreconditionerFallbackCount =
	      fullSolve.translationSchurPreconditionerFallbackCount;
	  result.localChainPreconditionerApplicationCount =
	      fullSolve.localChainPreconditionerApplicationCount;
	  result.localChainPreconditionerFactorizationCount =
	      fullSolve.localChainPreconditionerFactorizationCount;
	  result.localChainPreconditionerFallbackCount =
	      fullSolve.localChainPreconditionerFallbackCount;
	  result.translationBlockPreconditionerApplicationCount =
	      fullSolve.translationBlockPreconditionerApplicationCount;
  result.translationBlockPreconditionerFactorizationCount =
      fullSolve.translationBlockPreconditionerFactorizationCount;
  result.translationBlockPreconditionerFallbackCount =
      fullSolve.translationBlockPreconditionerFallbackCount;
  result.translationSparseSchurPreconditionerApplicationCount =
      fullSolve.translationSparseSchurPreconditionerApplicationCount;
  result.translationSparseSchurPreconditionerFactorizationCount =
      fullSolve.translationSparseSchurPreconditionerFactorizationCount;
  result.translationSparseSchurPreconditionerFallbackCount =
      fullSolve.translationSparseSchurPreconditionerFallbackCount;
  result.translationLocalSchurPreconditionerApplicationCount =
      fullSolve.translationLocalSchurPreconditionerApplicationCount;
  result.translationLocalSchurPreconditionerFactorizationCount =
      fullSolve.translationLocalSchurPreconditionerFactorizationCount;
  result.translationLocalSchurPreconditionerFallbackCount =
      fullSolve.translationLocalSchurPreconditionerFallbackCount;
  result.translationLocalSchurPreconditionerActivePoseCount =
      fullSolve.translationLocalSchurPreconditionerActivePoseCount;
  result.translationLocalSchurPreconditionerActiveColumnCount =
      fullSolve.translationLocalSchurPreconditionerActiveColumnCount;
  result.laplacianDeflationPreconditionerApplicationCount =
      fullSolve.laplacianDeflationPreconditionerApplicationCount;
  result.laplacianDeflationPreconditionerFactorizationCount =
      fullSolve.laplacianDeflationPreconditionerFactorizationCount;
  result.laplacianDeflationPreconditionerFallbackCount =
      fullSolve.laplacianDeflationPreconditionerFallbackCount;
  result.laplacianDeflationPreconditionerBasisDimension =
      fullSolve.laplacianDeflationPreconditionerBasisDimension;
  result.reducedRotationPreconditionerApplicationCount =
      fullSolve.reducedRotationPreconditionerApplicationCount;
  result.reducedRotationPreconditionerFactorizationCount =
      fullSolve.reducedRotationPreconditionerFactorizationCount;
  result.rqnMemoryPreconditionerApplicationCount =
      fullSolve.rqnMemoryPreconditionerApplicationCount;
  result.sparseMatrixVectorProductCount =
      fullSolve.sparseMatrixVectorProductCount;
  return result;
}

Matrix projectFullEquivHybridTangent(const Matrix &Y, const Matrix &Z,
                                     unsigned dimension) {
  if (dimension == 0 || Y.rows() != Z.rows() || Y.cols() != Z.cols() ||
      Y.cols() % static_cast<int>(dimension + 1) != 0) {
    throw std::invalid_argument("FE-Hybrid tangent projection shape mismatch");
  }
  const unsigned rank = static_cast<unsigned>(Y.rows());
  const unsigned numPoses =
      static_cast<unsigned>(Y.cols()) / (dimension + 1);
  Matrix projected = Matrix::Zero(Z.rows(), Z.cols());
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned colStart = pose * (dimension + 1);
    const Matrix R = Y.block(0, colStart, rank, dimension);
    const Matrix ZR = Z.block(0, colStart, rank, dimension);
    Matrix sym = R.transpose() * ZR;
    sym = 0.5 * (sym + sym.transpose());
    projected.block(0, colStart, rank, dimension) = ZR - R * sym;
    projected.col(colStart + dimension) = Z.col(colStart + dimension);
  }
  return projected;
}

Matrix retractFullEquivHybridByProjection(const Matrix &Y, const Matrix &eta,
                                          unsigned dimension) {
  if (dimension == 0 || Y.rows() != eta.rows() || Y.cols() != eta.cols() ||
      Y.cols() % static_cast<int>(dimension + 1) != 0) {
    throw std::invalid_argument("FE-Hybrid retraction shape mismatch");
  }
  const unsigned rank = static_cast<unsigned>(Y.rows());
  const unsigned numPoses =
      static_cast<unsigned>(Y.cols()) / (dimension + 1);
  Matrix candidate = Y + eta;
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned colStart = pose * (dimension + 1);
    const Matrix block = candidate.block(0, colStart, rank, dimension);
    Eigen::HouseholderQR<Matrix> qr(block);
    Matrix qFull = qr.householderQ() * Matrix::Identity(rank, dimension);
    Matrix rTri = qr.matrixQR().topLeftCorner(dimension, dimension)
                      .template triangularView<Eigen::Upper>();
    for (unsigned col = 0; col < dimension; ++col) {
      if (rTri(col, col) < 0.0) {
        qFull.col(col) *= -1.0;
      }
    }
    candidate.block(0, colStart, rank, dimension) = qFull;
  }
  return candidate;
}

double fullEquivHybridRelativeLinearResidual(const SparseMatrix &H,
                                             const Matrix &gradient,
                                             const Matrix &step,
                                             double damping) {
  if (gradient.rows() != step.rows() || gradient.cols() != step.cols() ||
      gradient.cols() != H.cols()) {
    throw std::invalid_argument("FE-Hybrid residual dimension mismatch");
  }
  const Matrix dense = dampedDenseSystem(H, damping);
  const Matrix residual = step * dense + gradient;
  return residual.norm() / (gradient.norm() + 1e-12);
}

}  // namespace DPGO
