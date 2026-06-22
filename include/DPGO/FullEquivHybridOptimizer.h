/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef FULL_EQUIV_HYBRID_OPTIMIZER_H
#define FULL_EQUIV_HYBRID_OPTIMIZER_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <vector>

namespace DPGO {

class FullEquivHybridRqnMemory;

struct FullEquivHybridEliminationPlan {
  unsigned numPoses{0};
  unsigned dimension{0};
  std::vector<int> keepColumns;
  std::vector<int> eliminateColumns;
};

struct FullEquivHybridSchurSystem {
  Matrix Haa;
  Matrix Hab;
  Matrix Hba;
  Matrix Hbb;
  Matrix S;
  Matrix rhs;
};

enum class FullEquivHybridReducedRotationPreconditioner {
  None,
  Jacobi,
  SchurJacobi,
  Cholesky
};

struct FullEquivHybridPcgOptions {
  double relativeTolerance{1e-6};
  double absoluteTolerance{1e-10};
  unsigned maxIterations{200};
  bool useBlockJacobiPreconditioner{true};
  bool useSparseMatrixVectorProduct{true};
  bool useTranslationSchurPreconditioner{false};
  bool useLocalChainPreconditioner{false};
  bool useTranslationBlockPreconditioner{false};
  bool useTranslationSparseSchurPreconditioner{false};
  bool useTranslationLocalSchurPreconditioner{false};
  unsigned translationLocalSchurMaxActivePoses{32};
  bool useLaplacianDeflationPreconditioner{false};
  unsigned laplacianDeflationBasisSize{4};
  unsigned laplacianDeflationMaxEigenPoses{512};
  bool useReducedRotationPreconditioner{false};
  FullEquivHybridReducedRotationPreconditioner reducedRotationPreconditioner{
      FullEquivHybridReducedRotationPreconditioner::Cholesky};
  bool useRqnMemoryPreconditioner{false};
  const FullEquivHybridRqnMemory *rqnMemoryPreconditioner{nullptr};
  Matrix initialStepGuess;
};

struct FullEquivHybridPcgResult {
  Matrix step;
  unsigned iterations{0};
  double initialResidual{0.0};
  double finalResidual{0.0};
  bool converged{false};
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

struct FullEquivHybridRqnOptions {
  unsigned memorySize{5};
  double minCurvatureRatio{1e-8};
  double minCurvatureAbsolute{1e-12};
};

class FullEquivHybridRqnMemory {
 public:
  explicit FullEquivHybridRqnMemory(
      const FullEquivHybridRqnOptions &options =
          FullEquivHybridRqnOptions());

  bool addPair(const Matrix &step, const Matrix &gradientDifference);
  Matrix applyInverseHessian(const Matrix &gradient) const;
  Vector applyInverseHessianRow(unsigned row, const Vector &gradient) const;
  void clear();

  std::size_t size() const;
  std::size_t acceptedPairCount() const;
  std::size_t rejectedPairCount() const;

 private:
  struct Pair {
    Matrix step;
    Matrix gradientDifference;
    double rho{0.0};
  };

  FullEquivHybridRqnOptions options;
  std::vector<Pair> pairs;
  std::size_t acceptedPairs{0};
  std::size_t rejectedPairs{0};
};

FullEquivHybridEliminationPlan
makeFullEquivHybridTranslationEliminationPlan(unsigned numPoses,
                                              unsigned dimension);

FullEquivHybridSchurSystem buildFullEquivHybridExactSchurSystem(
    const SparseMatrix &H, const Matrix &gradient,
    const FullEquivHybridEliminationPlan &plan, double damping);

Matrix solveFullEquivHybridFullDirect(const SparseMatrix &H,
                                      const Matrix &gradient,
                                      double damping);

Matrix solveFullEquivHybridSchurDirect(
    const FullEquivHybridSchurSystem &schur, const Matrix &gradient,
    const FullEquivHybridEliminationPlan &plan);

FullEquivHybridPcgResult solveFullEquivHybridSchurPcg(
    const FullEquivHybridSchurSystem &schur, const Matrix &gradient,
    const FullEquivHybridEliminationPlan &plan,
    const FullEquivHybridPcgOptions &options);

FullEquivHybridPcgResult solveFullEquivHybridFullPcg(
    const SparseMatrix &H, const Matrix &gradient, double damping,
    unsigned dimension, const FullEquivHybridPcgOptions &options);

Matrix projectFullEquivHybridTangent(const Matrix &Y, const Matrix &Z,
                                     unsigned dimension);

Matrix retractFullEquivHybridByProjection(const Matrix &Y, const Matrix &eta,
                                          unsigned dimension);

double fullEquivHybridRelativeLinearResidual(const SparseMatrix &H,
                                             const Matrix &gradient,
                                             const Matrix &step,
                                             double damping);

}  // namespace DPGO

#endif
