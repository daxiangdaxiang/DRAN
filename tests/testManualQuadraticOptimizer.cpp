#include <DPGO/BoundarySchurPreconditioner.h>
#include <DPGO/FullEquivHybridOptimizer.h>
#include <DPGO/ManualQuadraticOptimizer.h>
#include <DPGO/QuadraticOptimizer.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/ReducedRotationQuadraticOptimizer.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include "gtest/gtest.h"

#include <cstdlib>
#include <string>

using namespace DPGO;

TEST(testDPGO, BoundarySchurPreconditionerSolvesDenseBoundaryBlock) {
  SparseMatrix Q(6, 6);
  Q.insert(1, 1) = 4.0;
  Q.insert(1, 2) = 1.0;
  Q.insert(2, 1) = 1.0;
  Q.insert(2, 2) = 3.0;

  Matrix gradient(2, 2);
  gradient << 5.0, -2.0,
              -1.0, 4.0;

  const Matrix step = solveBoundaryBlockPreconditionedStep(Q, gradient, 1, 2, 0.5);

  Matrix A(2, 2);
  A << 4.5, 1.0,
       1.0, 3.5;
  const Matrix expected = -gradient * A.inverse();

  ASSERT_EQ(step.rows(), gradient.rows());
  ASSERT_EQ(step.cols(), gradient.cols());
  EXPECT_LE((step - expected).norm(), 1e-12);
}

TEST(testDPGO, FullEquivHybridTranslationSchurMatchesFullDirectSolve) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const unsigned cols = numPoses * (d + 1);
  Matrix dense = Matrix::Zero(cols, cols);
  dense << 8.0, 0.3, -0.2, 1.0, 0.4, 0.0, 0.2, -0.1,
           0.3, 7.0, 0.5, -0.4, 0.1, 0.3, -0.2, 0.0,
          -0.2, 0.5, 6.5, 0.2, -0.3, 0.2, 0.1, 0.4,
           1.0, -0.4, 0.2, 5.5, 0.0, -0.2, 0.3, 0.1,
           0.4, 0.1, -0.3, 0.0, 7.5, -0.1, 0.2, 0.5,
           0.0, 0.3, 0.2, -0.2, -0.1, 6.8, 0.4, -0.3,
           0.2, -0.2, 0.1, 0.3, 0.2, 0.4, 6.2, 0.2,
          -0.1, 0.0, 0.4, 0.1, 0.5, -0.3, 0.2, 5.8;
  dense = 0.5 * (dense + dense.transpose());
  dense += 2.0 * Matrix::Identity(cols, cols);

  SparseMatrix H = dense.sparseView();
  Matrix gradient(2, cols);
  gradient << 1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.75, -0.25,
             -0.2, 1.3, -2.1, 0.4, 3.2, -0.6, 1.1, -1.7;

  const double damping = 0.15;
  const FullEquivHybridEliminationPlan plan =
      makeFullEquivHybridTranslationEliminationPlan(numPoses, d);
  const FullEquivHybridSchurSystem schur =
      buildFullEquivHybridExactSchurSystem(H, gradient, plan, damping);
  const Matrix schurStep =
      solveFullEquivHybridSchurDirect(schur, gradient, plan);
  const Matrix fullStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  ASSERT_EQ(schurStep.rows(), fullStep.rows());
  ASSERT_EQ(schurStep.cols(), fullStep.cols());
  const double relativeStepError =
      (schurStep - fullStep).norm() / (fullStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-10);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, schurStep,
                                                  damping),
            1e-10);
}

TEST(testDPGO, FullEquivHybridPcgSchurMatchesDirectSchurSolve) {
  const unsigned d = 3;
  const unsigned numPoses = 3;
  const unsigned cols = numPoses * (d + 1);
  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned c = 0; c < cols; ++c) {
    dense(c, c) = 5.0 + static_cast<double>(c % 4);
  }
  for (unsigned c = 0; c + 1 < cols; ++c) {
    dense(c, c + 1) = 0.15;
    dense(c + 1, c) = 0.15;
  }
  for (unsigned c = 0; c + 4 < cols; ++c) {
    dense(c, c + 4) = -0.08;
    dense(c + 4, c) = -0.08;
  }

  SparseMatrix H = dense.sparseView();
  Matrix gradient(2, cols);
  gradient.setZero();
  for (int row = 0; row < gradient.rows(); ++row) {
    for (int col = 0; col < gradient.cols(); ++col) {
      gradient(row, col) =
          0.2 * static_cast<double>((row + 1) * (col + 2)) - 0.7;
    }
  }

  const double damping = 0.1;
  const FullEquivHybridEliminationPlan plan =
      makeFullEquivHybridTranslationEliminationPlan(numPoses, d);
  const FullEquivHybridSchurSystem schur =
      buildFullEquivHybridExactSchurSystem(H, gradient, plan, damping);
  const Matrix directStep =
      solveFullEquivHybridSchurDirect(schur, gradient, plan);

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 100;
  pcgOptions.useBlockJacobiPreconditioner = true;
  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridSchurPcg(schur, gradient, plan, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_GT(pcg.iterations, 0u);
  EXPECT_LT(pcg.finalResidual, pcg.initialResidual);
  const double relativeStepError =
      (pcg.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, pcg.step,
                                                  damping),
            1e-8);
}

TEST(testDPGO, FullEquivHybridPcgFullMatchesDirectFullSolve) {
  const unsigned d = 3;
  const unsigned numPoses = 3;
  const unsigned cols = numPoses * (d + 1);
  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned c = 0; c < cols; ++c) {
    dense(c, c) = 6.0 + static_cast<double>(c % 5);
  }
  for (unsigned c = 0; c + 1 < cols; ++c) {
    dense(c, c + 1) = -0.12;
    dense(c + 1, c) = -0.12;
  }
  for (unsigned c = 0; c + 4 < cols; ++c) {
    dense(c, c + 4) = 0.07;
    dense(c + 4, c) = 0.07;
  }

  SparseMatrix H = dense.sparseView();
  Matrix gradient(2, cols);
  gradient.setZero();
  for (int row = 0; row < gradient.rows(); ++row) {
    for (int col = 0; col < gradient.cols(); ++col) {
      gradient(row, col) =
          0.13 * static_cast<double>((row + 2) * (col + 1)) - 0.4;
    }
  }

  const double damping = 0.05;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 100;
  pcgOptions.useBlockJacobiPreconditioner = true;
  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, pcgOptions);
  pcgOptions.useSparseMatrixVectorProduct = false;
  const FullEquivHybridPcgResult densePcg =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  ASSERT_TRUE(densePcg.converged);
  EXPECT_GT(pcg.iterations, 0u);
  EXPECT_GT(pcg.sparseMatrixVectorProductCount, 0u);
  EXPECT_EQ(densePcg.sparseMatrixVectorProductCount, 0u);
  EXPECT_LT(pcg.finalResidual, pcg.initialResidual);
  EXPECT_LE((pcg.step - densePcg.step).norm() /
                (densePcg.step.norm() + 1e-12),
            1e-10);
  const double relativeStepError =
      (pcg.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, pcg.step,
                                                  damping),
            1e-8);
}

TEST(testDPGO, FullEquivHybridPcgSchurBlockJacobiUsesPoseBlocks) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const FullEquivHybridEliminationPlan plan =
      makeFullEquivHybridTranslationEliminationPlan(numPoses, d);

  FullEquivHybridSchurSystem schur;
  schur.S = Matrix::Zero(static_cast<int>(numPoses * d),
                         static_cast<int>(numPoses * d));
  schur.S.block(0, 0, 3, 3) << 4.0, 1.0, 0.5,
                                1.0, 3.0, 0.25,
                                0.5, 0.25, 2.0;
  schur.S.block(3, 3, 3, 3) << 5.0, -0.7, 0.2,
                                -0.7, 4.0, 0.3,
                                0.2, 0.3, 3.0;
  schur.rhs = Matrix(1, static_cast<int>(numPoses * d));
  schur.rhs << 1.0, -2.0, 0.5, 0.25, 1.5, -0.75;
  schur.Hbb = Matrix::Identity(static_cast<int>(numPoses),
                               static_cast<int>(numPoses));
  schur.Hba = Matrix::Zero(static_cast<int>(numPoses),
                           static_cast<int>(numPoses * d));

  Matrix gradient = Matrix::Zero(1, static_cast<int>(numPoses * (d + 1)));
  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 20;
  pcgOptions.useBlockJacobiPreconditioner = true;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridSchurPcg(schur, gradient, plan, pcgOptions);

  EXPECT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.iterations, 1u);
  EXPECT_LE(pcg.finalResidual, 1e-12);
}

TEST(testDPGO, FullEquivHybridPcgFullBlockJacobiUsesFullPoseBlocks) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const unsigned blockSize = d + 1;
  const unsigned cols = numPoses * blockSize;

  Matrix dense = Matrix::Zero(static_cast<int>(cols), static_cast<int>(cols));
  dense.block(0, 0, 4, 4) << 4.0, 1.0, 0.5, -0.2,
                              1.0, 3.0, 0.25, 0.4,
                              0.5, 0.25, 2.0, -0.1,
                              -0.2, 0.4, -0.1, 5.0;
  dense.block(4, 4, 4, 4) << 5.0, -0.7, 0.2, 0.3,
                              -0.7, 4.0, 0.3, -0.5,
                              0.2, 0.3, 3.0, 0.1,
                              0.3, -0.5, 0.1, 6.0;
  SparseMatrix H = dense.sparseView();

  Matrix gradient = Matrix::Zero(1, static_cast<int>(cols));
  gradient << -1.0, 2.0, -0.5, 0.75, -0.25, -1.5, 0.8, -0.6;

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 20;
  pcgOptions.useBlockJacobiPreconditioner = true;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, 0.0, d, pcgOptions);

  EXPECT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.iterations, 1u);
  EXPECT_LE(pcg.finalResidual, 1e-12);
}

TEST(testDPGO, FullEquivHybridPcgFullTranslationSchurPreconditionerIsExact) {
  const unsigned d = 3;
  const unsigned numPoses = 3;
  const unsigned blockSize = d + 1;
  const unsigned cols = numPoses * blockSize;

  Matrix dense = Matrix::Zero(static_cast<int>(cols), static_cast<int>(cols));
  for (unsigned c = 0; c < cols; ++c) {
    dense(static_cast<int>(c), static_cast<int>(c)) =
        8.0 + static_cast<double>(c % 7);
  }
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const int base = static_cast<int>(pose * blockSize);
    dense(base, base + 3) = 0.8;
    dense(base + 3, base) = 0.8;
    dense(base + 1, base + 3) = -0.4;
    dense(base + 3, base + 1) = -0.4;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const int lhs = static_cast<int>(pose * blockSize);
    const int rhs = static_cast<int>((pose + 1) * blockSize);
    dense(lhs + 2, rhs + 3) = 0.25;
    dense(rhs + 3, lhs + 2) = 0.25;
    dense(lhs + 3, rhs + 3) = -0.15;
    dense(rhs + 3, lhs + 3) = -0.15;
  }
  SparseMatrix H = dense.sparseView();

  Matrix gradient(3, static_cast<int>(cols));
  for (int row = 0; row < gradient.rows(); ++row) {
    for (int col = 0; col < gradient.cols(); ++col) {
      gradient(row, col) =
          0.17 * static_cast<double>((row + 1) * (col + 2)) - 0.9;
    }
  }

  const double damping = 0.01;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 5;
  pcgOptions.useBlockJacobiPreconditioner = false;
  pcgOptions.useTranslationSchurPreconditioner = true;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.iterations, 1u);
  EXPECT_GT(pcg.translationSchurPreconditionerApplicationCount, 0u);
  EXPECT_EQ(pcg.translationSchurPreconditionerFactorizationCount, 2u);
  EXPECT_LT(pcg.translationSchurPreconditionerFactorizationCount,
            pcg.translationSchurPreconditionerApplicationCount);
  EXPECT_EQ(pcg.translationSchurPreconditionerFallbackCount, 0u);
  EXPECT_EQ(pcg.reducedRotationPreconditionerApplicationCount, 0u);
  EXPECT_EQ(pcg.reducedRotationPreconditionerFactorizationCount, 0u);
  const double relativeStepError =
      (pcg.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-10);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, pcg.step,
                                                  damping),
            1e-10);
}

TEST(testDPGO,
     FullEquivHybridPcgFullReducedRotationPreconditionerAppliesInsidePcg) {
  const unsigned d = 3;
  const unsigned numPoses = 3;
  const unsigned blockSize = d + 1;
  const unsigned cols = numPoses * blockSize;

  Matrix dense = Matrix::Zero(static_cast<int>(cols), static_cast<int>(cols));
  for (unsigned c = 0; c < cols; ++c) {
    dense(static_cast<int>(c), static_cast<int>(c)) =
        9.0 + static_cast<double>(c % 5);
  }
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const int base = static_cast<int>(pose * blockSize);
    dense(base, base + 3) = 0.7;
    dense(base + 3, base) = 0.7;
    dense(base + 1, base + 3) = -0.35;
    dense(base + 3, base + 1) = -0.35;
    dense(base + 2, base + 3) = 0.25;
    dense(base + 3, base + 2) = 0.25;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const int lhs = static_cast<int>(pose * blockSize);
    const int rhs = static_cast<int>((pose + 1) * blockSize);
    dense(lhs, rhs + 1) = 0.18;
    dense(rhs + 1, lhs) = 0.18;
    dense(lhs + 2, rhs + 3) = -0.2;
    dense(rhs + 3, lhs + 2) = -0.2;
    dense(lhs + 3, rhs + 3) = 0.12;
    dense(rhs + 3, lhs + 3) = 0.12;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient(2, static_cast<int>(cols));
  for (int row = 0; row < gradient.rows(); ++row) {
    for (int col = 0; col < gradient.cols(); ++col) {
      gradient(row, col) =
          0.11 * static_cast<double>((row + 3) * (col + 1)) - 0.6;
    }
  }

  const double damping = 0.02;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 30;
  pcgOptions.useBlockJacobiPreconditioner = false;
  pcgOptions.useReducedRotationPreconditioner = true;
  pcgOptions.reducedRotationPreconditioner =
      FullEquivHybridReducedRotationPreconditioner::Cholesky;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_GT(pcg.reducedRotationPreconditionerApplicationCount, 0u);
  EXPECT_EQ(pcg.reducedRotationPreconditionerFactorizationCount, 2u);
  EXPECT_LT(pcg.reducedRotationPreconditionerFactorizationCount,
            pcg.reducedRotationPreconditionerApplicationCount);
  EXPECT_GT(pcg.iterations, 0u);
  const double relativeStepError =
      (pcg.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, pcg.step,
                                                  damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullTranslationBlockPreconditionerReducesChainWork) {
  const unsigned d = 3;
  const unsigned numPoses = 18;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;
  Matrix dense = Matrix::Zero(static_cast<int>(cols), static_cast<int>(cols));
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned base = pose * blockWidth;
    for (unsigned c = 0; c < d; ++c) {
      dense(static_cast<int>(base + c), static_cast<int>(base + c)) =
          4.0 + 0.1 * static_cast<double>(c);
    }
    dense(static_cast<int>(base + d), static_cast<int>(base + d)) = 0.01;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const unsigned current = pose * blockWidth + d;
    const unsigned next = (pose + 1) * blockWidth + d;
    dense(static_cast<int>(current), static_cast<int>(current)) += 1.0;
    dense(static_cast<int>(next), static_cast<int>(next)) += 1.0;
    dense(static_cast<int>(current), static_cast<int>(next)) -= 1.0;
    dense(static_cast<int>(next), static_cast<int>(current)) -= 1.0;
  }
  dense(static_cast<int>(d), static_cast<int>(d)) += 0.05;

  SparseMatrix H = dense.sparseView();
  Matrix gradient(2, static_cast<int>(cols));
  gradient.setZero();
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned translationCol = pose * blockWidth + d;
    const double x =
        static_cast<double>(pose + 1) / static_cast<double>(numPoses);
    gradient(0, static_cast<int>(translationCol)) =
        2.0 + std::sin(3.14159265358979323846 * x);
    gradient(1, static_cast<int>(translationCol)) =
        -1.0 + std::cos(2.0 * 3.14159265358979323846 * x);
  }

  const double damping = 0.001;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions baselineOptions;
  baselineOptions.relativeTolerance = 1e-10;
  baselineOptions.absoluteTolerance = 1e-12;
  baselineOptions.maxIterations = 100;
  baselineOptions.useBlockJacobiPreconditioner = true;
  const FullEquivHybridPcgResult baseline =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, baselineOptions);

  FullEquivHybridPcgOptions blockOptions = baselineOptions;
  blockOptions.useTranslationBlockPreconditioner = true;
  const FullEquivHybridPcgResult block =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, blockOptions);

  ASSERT_TRUE(baseline.converged);
  ASSERT_TRUE(block.converged);
  EXPECT_GT(baseline.sparseMatrixVectorProductCount, 0u);
  EXPECT_GT(block.sparseMatrixVectorProductCount, 0u);
  EXPECT_GT(block.translationBlockPreconditionerApplicationCount, 0u);
  EXPECT_EQ(block.translationBlockPreconditionerFactorizationCount, 1u);
  EXPECT_EQ(block.translationBlockPreconditionerFallbackCount, 0u);
  EXPECT_LT(block.iterations, baseline.iterations);
  const double relativeStepError =
      (block.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, block.step,
                                                  damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullTranslationBlockHandlesPoseTranslationCoupling) {
  const unsigned d = 3;
  const unsigned numPoses = 6;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;
  Matrix dense = 0.2 * Matrix::Identity(static_cast<int>(cols),
                                        static_cast<int>(cols));
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned base = pose * blockWidth;
    for (unsigned c = 0; c < d; ++c) {
      dense(static_cast<int>(base + c), static_cast<int>(base + c)) +=
          4.0 + 0.1 * static_cast<double>(c);
    }
    dense(static_cast<int>(base + d), static_cast<int>(base + d)) += 0.01;
    dense(static_cast<int>(base), static_cast<int>(base + d)) = 0.5;
    dense(static_cast<int>(base + d), static_cast<int>(base)) = 0.5;
    dense(static_cast<int>(base + 1), static_cast<int>(base + d)) = -0.25;
    dense(static_cast<int>(base + d), static_cast<int>(base + 1)) = -0.25;
    dense(static_cast<int>(base + 2), static_cast<int>(base + d)) = 0.18;
    dense(static_cast<int>(base + d), static_cast<int>(base + 2)) = 0.18;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const unsigned current = pose * blockWidth + d;
    const unsigned next = (pose + 1) * blockWidth + d;
    dense(static_cast<int>(current), static_cast<int>(current)) += 1.0;
    dense(static_cast<int>(next), static_cast<int>(next)) += 1.0;
    dense(static_cast<int>(current), static_cast<int>(next)) -= 1.0;
    dense(static_cast<int>(next), static_cast<int>(current)) -= 1.0;
    dense(static_cast<int>(pose * blockWidth + 2),
          static_cast<int>(next)) -= 0.2;
    dense(static_cast<int>(next),
          static_cast<int>(pose * blockWidth + 2)) -= 0.2;
    dense(static_cast<int>(pose * blockWidth),
          static_cast<int>((pose + 1) * blockWidth + 1)) += 0.18;
    dense(static_cast<int>((pose + 1) * blockWidth + 1),
          static_cast<int>(pose * blockWidth)) += 0.18;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient(1, static_cast<int>(cols));
  for (int col = 0; col < gradient.cols(); ++col) {
    gradient(0, col) =
        0.07 * static_cast<double>((col + 3) * ((col % 5) + 1)) - 0.6;
  }
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned translationCol = pose * blockWidth + d;
    const double x =
        static_cast<double>(pose + 1) / static_cast<double>(numPoses);
    gradient(0, static_cast<int>(translationCol)) +=
        3.0 * std::sin(3.14159265358979323846 * x);
  }

  const double damping = 0.001;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-10;
  pcgOptions.absoluteTolerance = 1e-12;
  pcgOptions.maxIterations = 80;
  pcgOptions.useBlockJacobiPreconditioner = true;
  pcgOptions.useTranslationBlockPreconditioner = true;
  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_LT(pcg.iterations, 30u);
  EXPECT_GT(pcg.translationBlockPreconditionerApplicationCount, 0u);
  EXPECT_EQ(pcg.translationBlockPreconditionerFactorizationCount, 1u);
  EXPECT_EQ(pcg.translationBlockPreconditionerFallbackCount, 0u);
  const double relativeStepError =
      (pcg.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient, pcg.step,
                                                  damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullTranslationSparseSchurReducesCoupledWork) {
  const unsigned d = 3;
  const unsigned numPoses = 12;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;
  Matrix dense = Matrix::Identity(static_cast<int>(cols),
                                  static_cast<int>(cols));
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned base = pose * blockWidth;
    dense.block(static_cast<int>(base), static_cast<int>(base), 3, 3) +=
        (Matrix(3, 3) << 2.8, 0.7, -0.2,
                         0.7, 2.5, 0.35,
                        -0.2, 0.35, 2.2)
            .finished();
    dense(static_cast<int>(base + d), static_cast<int>(base + d)) += 0.02;
    dense(static_cast<int>(base), static_cast<int>(base + d)) += 0.8;
    dense(static_cast<int>(base + d), static_cast<int>(base)) += 0.8;
    dense(static_cast<int>(base + 1), static_cast<int>(base + d)) -= 0.45;
    dense(static_cast<int>(base + d), static_cast<int>(base + 1)) -= 0.45;
    dense(static_cast<int>(base + 2), static_cast<int>(base + d)) += 0.35;
    dense(static_cast<int>(base + d), static_cast<int>(base + 2)) += 0.35;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const unsigned current = pose * blockWidth + d;
    const unsigned next = (pose + 1) * blockWidth + d;
    dense(static_cast<int>(current), static_cast<int>(current)) += 1.0;
    dense(static_cast<int>(next), static_cast<int>(next)) += 1.0;
    dense(static_cast<int>(current), static_cast<int>(next)) -= 0.95;
    dense(static_cast<int>(next), static_cast<int>(current)) -= 0.95;
    dense(static_cast<int>(pose * blockWidth),
          static_cast<int>(next)) += 0.55;
    dense(static_cast<int>(next),
          static_cast<int>(pose * blockWidth)) += 0.55;
    dense(static_cast<int>(pose * blockWidth + 1),
          static_cast<int>(next)) -= 0.25;
    dense(static_cast<int>(next),
          static_cast<int>(pose * blockWidth + 1)) -= 0.25;
    dense(static_cast<int>(pose * blockWidth + 2),
          static_cast<int>(next)) += 0.2;
    dense(static_cast<int>(next),
          static_cast<int>(pose * blockWidth + 2)) += 0.2;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient(1, static_cast<int>(cols));
  for (int col = 0; col < gradient.cols(); ++col) {
    gradient(0, col) =
        0.05 * static_cast<double>((col + 5) * ((col % 7) + 1)) - 0.7;
  }
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned translationCol = pose * blockWidth + d;
    const double x =
        static_cast<double>(pose + 1) / static_cast<double>(numPoses + 1);
    gradient(0, static_cast<int>(translationCol)) +=
        3.0 * std::sin(3.14159265358979323846 * x);
  }

  const double damping = 0.001;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions blockOptions;
  blockOptions.relativeTolerance = 1e-10;
  blockOptions.absoluteTolerance = 1e-12;
  blockOptions.maxIterations = 120;
  blockOptions.useBlockJacobiPreconditioner = true;
  blockOptions.useTranslationBlockPreconditioner = true;
  const FullEquivHybridPcgResult block =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, blockOptions);

  FullEquivHybridPcgOptions sparseSchurOptions = blockOptions;
  sparseSchurOptions.useTranslationBlockPreconditioner = false;
  sparseSchurOptions.useTranslationSparseSchurPreconditioner = true;
  const FullEquivHybridPcgResult sparseSchur =
      solveFullEquivHybridFullPcg(H, gradient, damping, d,
                                  sparseSchurOptions);

  ASSERT_TRUE(block.converged);
  ASSERT_TRUE(sparseSchur.converged);
  EXPECT_GT(sparseSchur.sparseMatrixVectorProductCount, 0u);
  EXPECT_GT(sparseSchur.translationSparseSchurPreconditionerApplicationCount,
            0u);
  EXPECT_EQ(sparseSchur.translationSparseSchurPreconditionerFactorizationCount,
            2u);
  EXPECT_EQ(sparseSchur.translationSparseSchurPreconditionerFallbackCount, 0u);
  EXPECT_LT(sparseSchur.iterations, block.iterations);
  const double relativeStepError =
      (sparseSchur.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient,
                                                  sparseSchur.step, damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullLocalChainPreconditionerReducesChainWork) {
  const unsigned d = 1;
  const unsigned blockWidth = d + 1;
  const unsigned numPoses = 8;
  const unsigned cols = numPoses * blockWidth;
  Matrix dense = Matrix::Zero(static_cast<int>(cols),
                              static_cast<int>(cols));
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    for (unsigned k = 0; k < blockWidth; ++k) {
      dense(static_cast<int>(pose * blockWidth + k),
            static_cast<int>(pose * blockWidth + k)) = 4.0;
    }
    if (pose + 1 < numPoses) {
      for (unsigned k = 0; k < blockWidth; ++k) {
        const int a = static_cast<int>(pose * blockWidth + k);
        const int b = static_cast<int>((pose + 1) * blockWidth + k);
        dense(a, b) = -1.25;
        dense(b, a) = -1.25;
      }
    }
  }
  SparseMatrix H = dense.sparseView();
  Matrix gradient(2, static_cast<int>(cols));
  for (int row = 0; row < gradient.rows(); ++row) {
    for (int col = 0; col < gradient.cols(); ++col) {
      gradient(row, col) =
          0.17 * static_cast<double>((row + 1) * (col + 2)) - 0.9;
    }
  }

  const double damping = 1e-6;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions blockOptions;
  blockOptions.relativeTolerance = 1e-10;
  blockOptions.absoluteTolerance = 1e-12;
  blockOptions.maxIterations = 100;
  blockOptions.useBlockJacobiPreconditioner = true;
  const FullEquivHybridPcgResult block =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, blockOptions);

  FullEquivHybridPcgOptions chainOptions = blockOptions;
  chainOptions.useLocalChainPreconditioner = true;
  const FullEquivHybridPcgResult chain =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, chainOptions);

  ASSERT_TRUE(block.converged);
  ASSERT_TRUE(chain.converged);
  EXPECT_GT(chain.localChainPreconditionerApplicationCount, 0u);
  EXPECT_EQ(chain.localChainPreconditionerFactorizationCount, 1u);
  EXPECT_EQ(chain.localChainPreconditionerFallbackCount, 0u);
  EXPECT_LT(chain.iterations, block.iterations);
  EXPECT_LE((chain.step - directStep).norm(), 1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullTranslationLocalSchurAllActiveMatchesDirect) {
  const unsigned d = 3;
  const unsigned numPoses = 8;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;

  Matrix dense = Matrix::Identity(static_cast<int>(cols),
                                  static_cast<int>(cols));
  dense *= 7.0;
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const unsigned t = pose * blockWidth + d;
    const unsigned nextR = (pose + 1) * blockWidth;
    dense(static_cast<int>(t), static_cast<int>(nextR)) = -0.35;
    dense(static_cast<int>(nextR), static_cast<int>(t)) = -0.35;
    dense(static_cast<int>(pose * blockWidth + 1),
          static_cast<int>(t)) = 0.45;
    dense(static_cast<int>(t),
          static_cast<int>(pose * blockWidth + 1)) = 0.45;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient(2, static_cast<int>(cols));
  for (int col = 0; col < gradient.cols(); ++col) {
    gradient(0, col) =
        0.03 * static_cast<double>((col + 3) * ((col % 5) + 1)) - 0.4;
    gradient(1, col) =
        -0.02 * static_cast<double>((col + 7) * ((col % 3) + 1)) + 0.2;
  }

  const double damping = 0.002;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions options;
  options.relativeTolerance = 1e-10;
  options.absoluteTolerance = 1e-12;
  options.maxIterations = 20;
  options.useBlockJacobiPreconditioner = true;
  options.useSparseMatrixVectorProduct = false;
  options.useTranslationLocalSchurPreconditioner = true;
  options.translationLocalSchurMaxActivePoses = 0;
  const FullEquivHybridPcgResult localSchur =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, options);

  ASSERT_TRUE(localSchur.converged);
  EXPECT_GT(localSchur.sparseMatrixVectorProductCount, 0u);
  EXPECT_GT(localSchur.translationLocalSchurPreconditionerApplicationCount,
            0u);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerFactorizationCount,
            2u);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerFallbackCount, 0u);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerActivePoseCount,
            numPoses);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerActiveColumnCount,
            cols);
  const double relativeStepError =
      (localSchur.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-9);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient,
                                                  localSchur.step, damping),
            1e-9);
}

TEST(testDPGO,
     FullEquivHybridPcgFullTranslationLocalSchurUsesBoundedActiveSet) {
  const unsigned d = 3;
  const unsigned numPoses = 14;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;

  Matrix dense = Matrix::Identity(static_cast<int>(cols),
                                  static_cast<int>(cols));
  dense *= 9.0;
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const unsigned t = pose * blockWidth + d;
    const unsigned nextR = (pose + 1) * blockWidth;
    const double strength = pose < 4 ? 0.75 : 0.12;
    dense(static_cast<int>(t), static_cast<int>(nextR)) = -strength;
    dense(static_cast<int>(nextR), static_cast<int>(t)) = -strength;
    dense(static_cast<int>(pose * blockWidth + 2),
          static_cast<int>(t)) = 0.5 * strength;
    dense(static_cast<int>(t),
          static_cast<int>(pose * blockWidth + 2)) = 0.5 * strength;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient(1, static_cast<int>(cols));
  for (int col = 0; col < gradient.cols(); ++col) {
    gradient(0, col) =
        0.04 * static_cast<double>((col + 2) * ((col % 7) + 1)) - 0.5;
  }

  const double damping = 0.001;
  FullEquivHybridPcgOptions blockOptions;
  blockOptions.relativeTolerance = 1e-10;
  blockOptions.absoluteTolerance = 1e-12;
  blockOptions.maxIterations = 160;
  blockOptions.useBlockJacobiPreconditioner = true;
  blockOptions.useTranslationBlockPreconditioner = true;
  const FullEquivHybridPcgResult block =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, blockOptions);

  FullEquivHybridPcgOptions localOptions = blockOptions;
  localOptions.useTranslationBlockPreconditioner = false;
  localOptions.useTranslationLocalSchurPreconditioner = true;
  localOptions.translationLocalSchurMaxActivePoses = 4;
  const FullEquivHybridPcgResult localSchur =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, localOptions);

  ASSERT_TRUE(block.converged);
  ASSERT_TRUE(localSchur.converged);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerActivePoseCount, 4u);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerActiveColumnCount,
            4u * blockWidth);
  EXPECT_GT(localSchur.translationLocalSchurPreconditionerApplicationCount,
            0u);
  EXPECT_EQ(localSchur.translationLocalSchurPreconditionerFallbackCount, 0u);
  EXPECT_GT(localSchur.sparseMatrixVectorProductCount, 0u);
  EXPECT_LE(localSchur.iterations, block.iterations);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient,
                                                  localSchur.step, damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullLaplacianDeflationReducesLowFrequencyTranslationWork) {
  const unsigned d = 3;
  const unsigned numPoses = 50;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;

  Matrix dense = Matrix::Identity(static_cast<int>(cols),
                                  static_cast<int>(cols));
  dense *= 20.0;
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    dense(static_cast<int>(pose * blockWidth + d),
          static_cast<int>(pose * blockWidth + d)) = 1e-4;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const int t0 = static_cast<int>(pose * blockWidth + d);
    const int t1 = static_cast<int>((pose + 1) * blockWidth + d);
    dense(t0, t0) += 1.0;
    dense(t1, t1) += 1.0;
    dense(t0, t1) -= 1.0;
    dense(t1, t0) -= 1.0;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient = Matrix::Zero(1, static_cast<int>(cols));
  const double pi = 3.14159265358979323846;
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const double theta =
        2.0 * pi * static_cast<double>(pose) /
        static_cast<double>(numPoses - 1);
    const double lowFrequencyResidual =
        std::sin(theta) + 0.2 * std::cos(2.0 * theta);
    gradient(0, static_cast<int>(pose * blockWidth + d)) =
        -lowFrequencyResidual;
  }

  const double damping = 1e-6;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions baselineOptions;
  baselineOptions.relativeTolerance = 1e-8;
  baselineOptions.absoluteTolerance = 1e-12;
  baselineOptions.maxIterations = 200;
  baselineOptions.useBlockJacobiPreconditioner = true;
  const FullEquivHybridPcgResult baseline =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, baselineOptions);

  FullEquivHybridPcgOptions deflationOptions = baselineOptions;
  deflationOptions.useLaplacianDeflationPreconditioner = true;
  deflationOptions.laplacianDeflationBasisSize = 6;
  deflationOptions.laplacianDeflationMaxEigenPoses = 128;
  const FullEquivHybridPcgResult deflated =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, deflationOptions);

  ASSERT_TRUE(baseline.converged);
  ASSERT_TRUE(deflated.converged);
  EXPECT_GT(deflated.sparseMatrixVectorProductCount, 0u);
  EXPECT_GT(deflated.laplacianDeflationPreconditionerApplicationCount, 0u);
  EXPECT_EQ(deflated.laplacianDeflationPreconditionerFactorizationCount, 1u);
  EXPECT_EQ(deflated.laplacianDeflationPreconditionerFallbackCount, 0u);
  EXPECT_EQ(deflated.laplacianDeflationPreconditionerBasisDimension, 6u);
  EXPECT_LT(deflated.iterations, baseline.iterations);
  const double relativeStepError =
      (deflated.step - directStep).norm() / (directStep.norm() + 1e-12);
  EXPECT_LE(relativeStepError, 1e-8);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient,
                                                  deflated.step, damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullLaplacianDeflationReportsDctFallback) {
  const unsigned d = 3;
  const unsigned numPoses = 32;
  const unsigned blockWidth = d + 1;
  const unsigned cols = numPoses * blockWidth;

  Matrix dense = Matrix::Identity(static_cast<int>(cols),
                                  static_cast<int>(cols));
  dense *= 10.0;
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    dense(static_cast<int>(pose * blockWidth + d),
          static_cast<int>(pose * blockWidth + d)) = 1e-3;
  }
  for (unsigned pose = 0; pose + 1 < numPoses; ++pose) {
    const int t0 = static_cast<int>(pose * blockWidth + d);
    const int t1 = static_cast<int>((pose + 1) * blockWidth + d);
    dense(t0, t0) += 0.5;
    dense(t1, t1) += 0.5;
    dense(t0, t1) -= 0.5;
    dense(t1, t0) -= 0.5;
  }
  dense = 0.5 * (dense + dense.transpose());
  SparseMatrix H = dense.sparseView();

  Matrix gradient = Matrix::Zero(1, static_cast<int>(cols));
  const double pi = 3.14159265358979323846;
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const double theta =
        pi * (static_cast<double>(pose) + 0.5) /
        static_cast<double>(numPoses);
    gradient(0, static_cast<int>(pose * blockWidth + d)) =
        -std::cos(theta);
  }

  const double damping = 1e-6;
  FullEquivHybridPcgOptions options;
  options.relativeTolerance = 1e-8;
  options.absoluteTolerance = 1e-12;
  options.maxIterations = 200;
  options.useBlockJacobiPreconditioner = true;
  options.useLaplacianDeflationPreconditioner = true;
  options.laplacianDeflationBasisSize = 5;
  options.laplacianDeflationMaxEigenPoses = 4;

  const FullEquivHybridPcgResult result =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, options);

  ASSERT_TRUE(result.converged);
  EXPECT_GT(result.sparseMatrixVectorProductCount, 0u);
  EXPECT_GT(result.laplacianDeflationPreconditionerApplicationCount, 0u);
  EXPECT_EQ(result.laplacianDeflationPreconditionerFactorizationCount, 1u);
  EXPECT_EQ(result.laplacianDeflationPreconditionerFallbackCount, 1u);
  EXPECT_EQ(result.laplacianDeflationPreconditionerBasisDimension, 5u);
  EXPECT_LE(fullEquivHybridRelativeLinearResidual(H, gradient,
                                                  result.step, damping),
            1e-8);
}

TEST(testDPGO,
     FullEquivHybridPcgFullLaplacianDeflationRejectsDenseOnlyPreconditionerConflict) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const unsigned cols = numPoses * (d + 1);
  SparseMatrix H =
      Matrix::Identity(static_cast<int>(cols), static_cast<int>(cols))
          .sparseView();
  Matrix gradient = Matrix::Ones(1, static_cast<int>(cols));

  FullEquivHybridPcgOptions options;
  options.useLaplacianDeflationPreconditioner = true;
  options.useTranslationSchurPreconditioner = true;
  EXPECT_THROW(solveFullEquivHybridFullPcg(H, gradient, 1e-6, d, options),
               std::invalid_argument);

  options.useTranslationSchurPreconditioner = false;
  options.useReducedRotationPreconditioner = true;
  EXPECT_THROW(solveFullEquivHybridFullPcg(H, gradient, 1e-6, d, options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridPcgFullUsesInitialStepGuess) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const unsigned cols = numPoses * (d + 1);

  Matrix dense = Matrix::Identity(static_cast<int>(cols),
                                  static_cast<int>(cols));
  dense *= 5.0;
  dense(0, 1) = 0.2;
  dense(1, 0) = 0.2;
  dense(2, 7) = -0.15;
  dense(7, 2) = -0.15;
  SparseMatrix H = dense.sparseView();

  Matrix gradient(2, static_cast<int>(cols));
  gradient << 1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.75, -0.25,
              -0.4, 1.2, -1.1, 0.6, 2.0, -0.8, 1.4, -1.6;

  const double damping = 0.02;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, damping);

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 1e-12;
  pcgOptions.absoluteTolerance = 1e-14;
  pcgOptions.maxIterations = 20;
  pcgOptions.useBlockJacobiPreconditioner = false;
  pcgOptions.initialStepGuess = directStep;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, damping, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.iterations, 0u);
  EXPECT_LE(pcg.initialResidual, 1e-12);
  EXPECT_LE((pcg.step - directStep).norm(), 1e-12);
}

TEST(testDPGO, FullEquivHybridPcgFullInitialGuessUsesRhsRelativeTolerance) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const unsigned cols = numPoses * (d + 1);

  const SparseMatrix H =
      Matrix::Identity(static_cast<int>(cols), static_cast<int>(cols))
          .sparseView();
  Matrix gradient = Matrix::Zero(1, static_cast<int>(cols));
  gradient(0, 0) = 10.0;
  const Matrix directStep = -gradient;
  Matrix warmStep = directStep;
  warmStep(0, 0) += 0.1;

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 0.02;
  pcgOptions.absoluteTolerance = 0.0;
  pcgOptions.maxIterations = 20;
  pcgOptions.useBlockJacobiPreconditioner = false;
  pcgOptions.initialStepGuess = warmStep;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, 0.0, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.iterations, 0u);
  EXPECT_NEAR(pcg.initialResidual, 0.1, 1e-12);
  EXPECT_LE((pcg.step - warmStep).norm(), 1e-12);
}

TEST(testDPGO, FullEquivHybridPcgFullRhsRelativeToleranceHasNoUnitFloor) {
  const unsigned d = 3;
  const unsigned numPoses = 2;
  const unsigned cols = numPoses * (d + 1);

  const SparseMatrix H =
      Matrix::Identity(static_cast<int>(cols), static_cast<int>(cols))
          .sparseView();
  Matrix gradient = Matrix::Zero(1, static_cast<int>(cols));
  gradient(0, 0) = 0.01;
  const Matrix directStep = -gradient;
  Matrix warmStep = directStep;
  warmStep(0, 0) += 0.005;

  FullEquivHybridPcgOptions pcgOptions;
  pcgOptions.relativeTolerance = 0.1;
  pcgOptions.absoluteTolerance = 0.0;
  pcgOptions.maxIterations = 20;
  pcgOptions.useBlockJacobiPreconditioner = false;
  pcgOptions.initialStepGuess = warmStep;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, 0.0, d, pcgOptions);

  ASSERT_TRUE(pcg.converged);
  EXPECT_GT(pcg.iterations, 0u);
  EXPECT_NEAR(pcg.initialResidual, 0.005, 1e-12);
  EXPECT_LE((pcg.step - directStep).norm(), 1e-12);
}

TEST(testDPGO, FullEquivHybridRqnMemoryRejectsBadCurvatureAndCapsSize) {
  FullEquivHybridRqnOptions options;
  options.memorySize = 2;
  options.minCurvatureRatio = 1e-8;
  FullEquivHybridRqnMemory memory(options);

  Matrix s1(1, 2);
  s1 << 1.0, 0.0;
  Matrix y1(1, 2);
  y1 << 2.0, 0.0;
  EXPECT_TRUE(memory.addPair(s1, y1));
  EXPECT_EQ(memory.size(), 1u);

  Matrix badY(1, 2);
  badY << -1.0, 0.0;
  EXPECT_FALSE(memory.addPair(s1, badY));
  EXPECT_EQ(memory.size(), 1u);

  Matrix s2(1, 2);
  s2 << 0.0, 1.0;
  Matrix y2(1, 2);
  y2 << 0.0, 3.0;
  Matrix s3(1, 2);
  s3 << 1.0, 1.0;
  Matrix y3(1, 2);
  y3 << 4.0, 4.0;
  EXPECT_TRUE(memory.addPair(s2, y2));
  EXPECT_TRUE(memory.addPair(s3, y3));
  EXPECT_EQ(memory.size(), 2u);
  EXPECT_EQ(memory.acceptedPairCount(), 3u);
  EXPECT_EQ(memory.rejectedPairCount(), 1u);
}

TEST(testDPGO, FullEquivHybridRqnMemoryTwoLoopAppliesInverseCurvature) {
  FullEquivHybridRqnOptions options;
  options.memorySize = 4;
  FullEquivHybridRqnMemory memory(options);

  Matrix step(1, 1);
  step << 2.0;
  Matrix gradientDiff(1, 1);
  gradientDiff << 6.0;
  ASSERT_TRUE(memory.addPair(step, gradientDiff));

  Matrix gradient(1, 1);
  gradient << 12.0;
  const Matrix inverseHessianGradient = memory.applyInverseHessian(gradient);

  ASSERT_EQ(inverseHessianGradient.rows(), gradient.rows());
  ASSERT_EQ(inverseHessianGradient.cols(), gradient.cols());
  EXPECT_NEAR(inverseHessianGradient(0, 0), 4.0, 1e-12);
}

TEST(testDPGO,
     FullEquivHybridPcgFullRqnMemoryPreconditionerAppliesInsidePcg) {
  const unsigned d = 1;
  Matrix dense = Matrix::Zero(2, 2);
  dense(0, 0) = 100.0;
  dense(1, 1) = 1.0;
  SparseMatrix H = dense.sparseView();

  Matrix gradient(1, 2);
  gradient << -100.0, -1.0;
  const Matrix directStep =
      solveFullEquivHybridFullDirect(H, gradient, 0.0);

  FullEquivHybridRqnMemory memory;
  Matrix s1(1, 2);
  s1 << 1.0, 0.0;
  Matrix y1(1, 2);
  y1 << 100.0, 0.0;
  Matrix s2(1, 2);
  s2 << 0.0, 1.0;
  Matrix y2(1, 2);
  y2 << 0.0, 1.0;
  ASSERT_TRUE(memory.addPair(s1, y1));
  ASSERT_TRUE(memory.addPair(s2, y2));

  FullEquivHybridPcgOptions options;
  options.relativeTolerance = 1e-12;
  options.absoluteTolerance = 1e-12;
  options.maxIterations = 1;
  options.useBlockJacobiPreconditioner = false;
  options.useRqnMemoryPreconditioner = true;
  options.rqnMemoryPreconditioner = &memory;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, 0.0, d, options);

  ASSERT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.iterations, 1u);
  EXPECT_EQ(pcg.rqnMemoryPreconditionerApplicationCount, 1u);
  EXPECT_LE((pcg.step - directStep).norm(), 1e-12);
}

TEST(testDPGO,
     FullEquivHybridPcgFullRqnMemoryPreconditionerFallsBackOnEmptyRowMemory) {
  const unsigned d = 1;
  Matrix dense = Matrix::Identity(2, 2);
  dense(0, 0) = 100.0;
  SparseMatrix H = dense.sparseView();

  Matrix gradient(2, 2);
  gradient << -100.0, 0.0,
              0.0, -1.0;

  FullEquivHybridRqnMemory memory;
  Matrix step(2, 2);
  step << 1.0, 0.0,
          0.0, 0.0;
  Matrix gradientDiff(2, 2);
  gradientDiff << 100.0, 0.0,
                  0.0, 0.0;
  ASSERT_TRUE(memory.addPair(step, gradientDiff));

  FullEquivHybridPcgOptions options;
  options.relativeTolerance = 1e-12;
  options.absoluteTolerance = 1e-12;
  options.maxIterations = 1;
  options.useBlockJacobiPreconditioner = true;
  options.useRqnMemoryPreconditioner = true;
  options.rqnMemoryPreconditioner = &memory;

  const FullEquivHybridPcgResult pcg =
      solveFullEquivHybridFullPcg(H, gradient, 0.0, d, options);

  ASSERT_TRUE(pcg.converged);
  EXPECT_EQ(pcg.rqnMemoryPreconditionerApplicationCount, 1u);
  EXPECT_LE((pcg.step - solveFullEquivHybridFullDirect(H, gradient, 0.0)).norm(),
            1e-12);
}

TEST(testDPGO, FullEquivHybridPcgSchurRejectsRqnMemoryPreconditioner) {
  const unsigned d = 1;
  const unsigned numPoses = 1;
  Matrix dense = Matrix::Identity(2, 2);
  SparseMatrix H = dense.sparseView();
  Matrix gradient(1, 2);
  gradient << -1.0, -2.0;

  const FullEquivHybridEliminationPlan plan =
      makeFullEquivHybridTranslationEliminationPlan(numPoses, d);
  const FullEquivHybridSchurSystem schur =
      buildFullEquivHybridExactSchurSystem(H, gradient, plan, 0.0);
  FullEquivHybridPcgOptions options;
  options.useRqnMemoryPreconditioner = true;

  EXPECT_THROW(solveFullEquivHybridSchurPcg(schur, gradient, plan, options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridRejectsSingularEliminatedBlock) {
  const unsigned d = 1;
  const unsigned numPoses = 1;
  SparseMatrix H(2, 2);
  H.insert(0, 0) = 1.0;

  Matrix gradient(1, 2);
  gradient << 0.0, 1.0;

  const FullEquivHybridEliminationPlan plan =
      makeFullEquivHybridTranslationEliminationPlan(numPoses, d);
  EXPECT_THROW(buildFullEquivHybridExactSchurSystem(H, gradient, plan, 0.0),
               std::runtime_error);
}

TEST(testDPGO, BoundarySchurPreconditionerEliminatesCoupledPrivateColumns) {
  SparseMatrix Q(4, 4);
  Q.insert(0, 0) = 5.0;
  Q.insert(0, 1) = 1.0;
  Q.insert(1, 0) = 1.0;
  Q.insert(1, 1) = 4.0;
  Q.insert(0, 2) = 2.0;
  Q.insert(2, 0) = 2.0;
  Q.insert(1, 3) = -1.0;
  Q.insert(3, 1) = -1.0;
  Q.insert(2, 2) = 6.0;
  Q.insert(2, 3) = 0.5;
  Q.insert(3, 2) = 0.5;
  Q.insert(3, 3) = 3.0;

  Matrix gradient(2, 4);
  gradient << 3.0, -2.0, 5.0, 1.0,
              -1.0, 4.0, -2.0, 3.0;

  const Matrix step =
      solveBoundaryLocalSchurPreconditionedStep(Q, gradient, 0, 2, 0.0, 8);

  Matrix Qbb(2, 2);
  Qbb << 5.0, 1.0,
         1.0, 4.0;
  Matrix Qbp(2, 2);
  Qbp << 2.0, 0.0,
         0.0, -1.0;
  Matrix Qpp(2, 2);
  Qpp << 6.0, 0.5,
         0.5, 3.0;
  Matrix gB = gradient.block(0, 0, 2, 2);
  Matrix gP = gradient.block(0, 2, 2, 2);
  const Matrix expected =
      -(gB - (Qbp * Qpp.ldlt().solve(gP.transpose())).transpose()) *
      (Qbb - Qbp * Qpp.ldlt().solve(Qbp.transpose())).inverse();

  ASSERT_EQ(step.rows(), gB.rows());
  ASSERT_EQ(step.cols(), gB.cols());
  EXPECT_LE((step - expected).norm(), 1e-12);
}

TEST(testDPGO, BoundarySchurPreconditionerRejectsIndefinitePrivateBlock) {
  SparseMatrix Q(3, 3);
  Q.insert(0, 0) = 2.0;
  Q.insert(0, 1) = 1.0;
  Q.insert(1, 0) = 1.0;
  Q.insert(1, 1) = -1.0;

  Matrix gradient(2, 3);
  gradient << 1.0, 2.0, 0.0,
              -3.0, 4.0, 0.0;

  const Matrix step =
      solveBoundaryLocalSchurPreconditionedStep(Q, gradient, 0, 1, 0.0, 2);

  ASSERT_EQ(step.rows(), gradient.rows());
  ASSERT_EQ(step.cols(), 1);
  EXPECT_LE(step.norm(), 1e-14);
}

TEST(testDPGO, BoundarySchurPacketDampingUsesSensitivityOverMobility) {
  EXPECT_DOUBLE_EQ(
      boundaryPacketSchurExtraDamping(0.0, 0.5, 0.1, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(
      boundaryPacketSchurExtraDamping(2.0, 4.0, 0.1, 1.0), 0.05);
  EXPECT_DOUBLE_EQ(
      boundaryPacketSchurExtraDamping(2.0, 0.1, 0.1, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(
      boundaryPacketSchurExtraDamping(2.0, 0.0, 0.1, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(
      boundaryPacketSchurExtraDamping(2.0, 4.0, 0.0, 1.0), 0.0);
}

TEST(testDPGO, ManualQuadraticOptimizerReducesSimpleTranslationQuadratic) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 1;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  Q.insert(d, d) = 1.0;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = 2.0;
  G.insert(1, d) = -1.0;
  G.insert(2, d) = 0.5;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, d + 1);
  X0.block(0, 0, d, d) = Matrix::Identity(d, d);
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ManualQuadraticOptimizer optimizer(&problem);
  optimizer.setMaxIterations(5);
  optimizer.setMaxCgIterations(20);
  optimizer.setGradientTolerance(1e-8);
  optimizer.setInitialDamping(1e-6);

  const double f0 = problem.f(X0);
  Matrix Xopt = optimizer.optimize(X0);
  const double f1 = problem.f(Xopt);

  EXPECT_LT(f1, f0 - 1.0);
  EXPECT_LT(problem.RieGradNorm(Xopt), problem.RieGradNorm(X0));
  EXPECT_NEAR(Xopt(0, d), -2.0, 1e-3);
  EXPECT_NEAR(Xopt(1, d), 1.0, 1e-3);
  EXPECT_NEAR(Xopt(2, d), -0.5, 1e-3);
  EXPECT_LE((Xopt.block(0, 0, d, d).transpose() *
                 Xopt.block(0, 0, d, d) -
             Matrix::Identity(d, d))
                .norm(),
            1e-9);
}

TEST(testDPGO, QuadraticProblemExposesConstSparseMatrixReferences) {
  QuadraticProblem problem(1, 3, 3);
  SparseMatrix Q(4, 4);
  Q.insert(0, 0) = 2.0;
  problem.setQ(Q);

  SparseMatrix G(3, 4);
  G.insert(1, 3) = -4.0;
  problem.setG(G);

  const SparseMatrix &qRef = problem.getQRef();
  const SparseMatrix &gRef = problem.getGRef();

  EXPECT_EQ(qRef.rows(), 4);
  EXPECT_EQ(gRef.cols(), 4);
  EXPECT_DOUBLE_EQ(qRef.coeff(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(gRef.coeff(1, 3), -4.0);
}

TEST(testDPGO, QuadraticProblemFusedCostGradientMatchesSeparateCalls) {
  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 3;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.2 * col;
  }
  Q.insert(0, 5) = -0.3;
  Q.insert(5, 0) = -0.3;
  Q.insert(2, 11) = 0.4;
  Q.insert(11, 2) = 0.4;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 1) = 0.7;
  G.insert(2, 6) = -1.1;
  G.insert(3, 9) = 0.2;
  problem.setG(G);

  Matrix Y(r, cols);
  for (unsigned row = 0; row < r; ++row) {
    for (unsigned col = 0; col < cols; ++col) {
      Y(row, col) = 0.03 * static_cast<double>((row + 1) * (col + 2)) -
                    0.2 * static_cast<double>((row + col) % 3 == 0);
    }
  }

  const auto fused = problem.fAndRieGradNorm(Y);
  EXPECT_DOUBLE_EQ(fused.first, problem.f(Y));
  EXPECT_NEAR(fused.second, problem.RieGradNorm(Y), 1e-12);
}

TEST(testDPGO, QuadraticProblemExplicitGradientProjectionMatchesRoptlib) {
  const unsigned d = 3;
  const unsigned r = 5;
  const unsigned n = 4;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.05 * col;
  }
  Q.insert(0, 6) = -0.15;
  Q.insert(6, 0) = -0.15;
  Q.insert(3, 12) = 0.21;
  Q.insert(12, 3) = 0.21;
  Q.insert(9, 15) = -0.08;
  Q.insert(15, 9) = -0.08;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 2) = 0.25;
  G.insert(1, 5) = -0.31;
  G.insert(3, 10) = 0.43;
  G.insert(4, 15) = -0.17;
  problem.setG(G);

  Matrix Y = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    Matrix raw = Matrix::Zero(r, d);
    for (unsigned row = 0; row < r; ++row) {
      for (unsigned col = 0; col < d; ++col) {
        raw(row, col) =
            0.13 * static_cast<double>((row + 1) * (col + 2 + pose)) -
            0.07 * static_cast<double>((row + col + pose) % 2);
      }
    }
    Eigen::HouseholderQR<Matrix> qr(raw);
    Y.block(0, colStart, r, d) =
        qr.householderQ() * Matrix::Identity(r, d);
    for (unsigned row = 0; row < r; ++row) {
      Y(row, colStart + d) =
          0.04 * static_cast<double>((row + 2) * (pose + 1));
    }
  }

  LiftedSEVariable var(r, d, n);
  LiftedSEVector roptGrad(r, d, n);
  var.setData(Y);
  static_cast<ROPTLIB::Problem &>(problem).RieGrad(var.var(),
                                                   roptGrad.vec());

  const Matrix explicitGrad = problem.RieGrad(Y);
  EXPECT_LE((explicitGrad - roptGrad.getData()).norm(), 1e-10);
  const auto fused = problem.fAndRieGradNorm(Y);
  EXPECT_NEAR(fused.second, roptGrad.getData().norm(), 1e-10);
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerDirectObjectiveMatchesProblemF) {
  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 3;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.1 * col;
  }
  Q.insert(0, 3) = 0.25;
  Q.insert(3, 0) = 0.25;
  Q.insert(2, 7) = -0.4;
  Q.insert(7, 2) = -0.4;
  Q.insert(4, 10) = 0.15;
  Q.insert(10, 4) = 0.15;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.7;
  G.insert(2, 3) = 0.9;
  G.insert(3, 8) = -1.2;
  problem.setG(G);

  Matrix Y(r, cols);
  for (unsigned row = 0; row < r; ++row) {
    for (unsigned col = 0; col < cols; ++col) {
      Y(row, col) = 0.05 * static_cast<double>((row + 2) * (col + 3)) -
                    0.3 * static_cast<double>(row == col % r);
    }
  }

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  EXPECT_NEAR(optimizer.evaluateQuadraticObjectiveDirect(Y), problem.f(Y),
              1e-12);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerCurvatureCauchyCandidateIsProfiled) {
  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 4;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.05 * col;
  }
  Q.insert(0, 4) = 0.2;
  Q.insert(4, 0) = 0.2;
  Q.insert(2, 10) = -0.1;
  Q.insert(10, 2) = -0.1;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.9;
  G.insert(1, 5) = 0.4;
  G.insert(2, 9) = -0.3;
  problem.setG(G);

  Matrix Y = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    for (unsigned col = 0; col < d; ++col) {
      Y(col, pose * (d + 1) + col) = 1.0;
    }
    Y(3, pose * (d + 1) + d) = 0.1 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  Y = manifold.project(Y);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setProfileRuntime(true);
  optimizer.setUseCurvatureCauchyCandidate(true);
  optimizer.setTrustRegionIterations(2);
  optimizer.setTrustRegionAcceptedIterations(1);
  optimizer.setTrustRegionMaxInnerIterations(3);
  const Matrix Xopt = optimizer.optimize(Y);

  EXPECT_TRUE(Xopt.allFinite());
  EXPECT_TRUE(std::isfinite(problem.f(Xopt)));
  EXPECT_GT(optimizer.getProfileCurvatureCauchyCandidateCount(), 0u);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerCurvatureFallbackCandidateCanRun) {
  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 4;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.04 * col;
  }
  Q.insert(1, 6) = 0.12;
  Q.insert(6, 1) = 0.12;
  Q.insert(4, 13) = -0.08;
  Q.insert(13, 4) = -0.08;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.8;
  G.insert(1, 6) = 0.3;
  G.insert(2, 10) = -0.5;
  problem.setG(G);

  Matrix Y = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    for (unsigned col = 0; col < d; ++col) {
      Y(col, pose * (d + 1) + col) = 1.0;
    }
    Y(3, pose * (d + 1) + d) = 0.05 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  Y = manifold.project(Y);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setProfileRuntime(true);
  optimizer.setUseCurvatureCauchyFallbackCandidate(true);
  optimizer.setTrustRegionIterations(2);
  optimizer.setTrustRegionAcceptedIterations(1);
  optimizer.setTrustRegionMaxInnerIterations(3);
  const Matrix Xopt = optimizer.optimize(Y);

  EXPECT_TRUE(Xopt.allFinite());
  EXPECT_TRUE(std::isfinite(problem.f(Xopt)));
  EXPECT_EQ(optimizer.getProfileCurvatureCauchyCandidateCount(), 0u);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerGradientBoundaryCandidateCanBeDisabled) {
  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 4;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.03 * col;
  }
  Q.insert(0, 4) = 0.18;
  Q.insert(4, 0) = 0.18;
  Q.insert(2, 10) = -0.11;
  Q.insert(10, 2) = -0.11;
  Q.insert(6, 14) = 0.09;
  Q.insert(14, 6) = 0.09;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.7;
  G.insert(1, 5) = 0.35;
  G.insert(2, 9) = -0.25;
  problem.setG(G);

  Matrix Y = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    for (unsigned col = 0; col < d; ++col) {
      Y(col, pose * (d + 1) + col) = 1.0;
    }
    Y(3, pose * (d + 1) + d) = 0.04 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  Y = manifold.project(Y);

  auto run = [&](bool useGradientBoundary) {
    ReducedRotationQuadraticOptimizer optimizer(&problem);
    optimizer.setProfileRuntime(true);
    optimizer.setUseGradientBoundaryCandidate(useGradientBoundary);
    optimizer.setTrustRegionIterations(2);
    optimizer.setTrustRegionAcceptedIterations(1);
    optimizer.setTrustRegionMaxInnerIterations(4);
    const Matrix Xopt = optimizer.optimize(Y);
    EXPECT_TRUE(Xopt.allFinite());
    EXPECT_TRUE(std::isfinite(problem.f(Xopt)));
    return optimizer.getProfileReducedObjectiveCount();
  };

  const std::size_t withBoundary = run(true);
  const std::size_t withoutBoundary = run(false);

  EXPECT_GT(withBoundary, 0u);
  EXPECT_GT(withBoundary, withoutBoundary);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerSurrogateTcgAcceptReducesMeritWork) {
  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 4;
  QuadraticProblem problem(n, d, r);
  const unsigned cols = n * (d + 1);

  SparseMatrix Q(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    Q.insert(col, col) = 1.0 + 0.025 * col;
  }
  Q.insert(0, 4) = 0.15;
  Q.insert(4, 0) = 0.15;
  Q.insert(2, 10) = -0.10;
  Q.insert(10, 2) = -0.10;
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.65;
  G.insert(1, 5) = 0.31;
  G.insert(2, 9) = -0.23;
  problem.setG(G);

  Matrix Y = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    for (unsigned col = 0; col < d; ++col) {
      Y(col, pose * (d + 1) + col) = 1.0;
    }
    Y(3, pose * (d + 1) + d) = 0.03 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  Y = manifold.project(Y);

  auto run = [&](bool surrogateAccept) {
    ReducedRotationQuadraticOptimizer optimizer(&problem);
    optimizer.setProfileRuntime(true);
    optimizer.setUseGradientBoundaryCandidate(false);
    optimizer.setUseSurrogateTcgAccept(surrogateAccept);
    optimizer.setTrustRegionIterations(2);
    optimizer.setTrustRegionAcceptedIterations(1);
    optimizer.setTrustRegionMaxInnerIterations(4);
    const Matrix Xopt = optimizer.optimize(Y);
    EXPECT_TRUE(Xopt.allFinite());
    EXPECT_TRUE(std::isfinite(problem.f(Xopt)));
    return optimizer.getProfileReducedObjectiveCount();
  };

  const std::size_t exactMerit = run(false);
  const std::size_t surrogateMerit = run(true);

  EXPECT_GT(exactMerit, 0u);
  EXPECT_GT(exactMerit, surrogateMerit);
}

TEST(testDPGO, ManualQuadraticOptimizerSurrogateBeatsRoptlibOnIllConditionedTranslationQuadratic) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 4;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  const double stiffness[4] = {0.05, 40.0, 0.08, 25.0};
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned tCol = pose * (d + 1) + d;
    Q.insert(tCol, tCol) = stiffness[pose] + 0.4;
  }
  for (unsigned pose = 0; pose + 1 < n; ++pose) {
    const unsigned c1 = pose * (d + 1) + d;
    const unsigned c2 = (pose + 1) * (d + 1) + d;
    Q.coeffRef(c1, c1) += 0.6;
    Q.coeffRef(c2, c2) += 0.6;
    Q.insert(c1, c2) = -0.6;
    Q.insert(c2, c1) = -0.6;
  }
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  const double g[3][4] = {{-3.5, 14.0, -2.4, 9.0},
                          {2.0, -8.0, 1.5, -7.5},
                          {-1.0, 4.0, -0.8, 3.0}};
  for (unsigned row = 0; row < r; ++row) {
    for (unsigned pose = 0; pose < n; ++pose) {
      G.insert(row, pose * (d + 1) + d) = g[row][pose];
    }
  }
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  QuadraticOptimizer ropt(&problem);
  ropt.setAlgorithm(ROPTALG::RTR);
  ropt.setTrustRegionIterations(1);
  ropt.setTrustRegionMaxInnerIterations(1);
  ropt.setTrustRegionTolerance(1e-12);
  ropt.setTrustRegionInitialRadius(10.0);
  const Matrix Xropt = ropt.optimize(X0);
  const double fRopt = problem.f(Xropt);

  ManualQuadraticOptimizer manual(&problem);
  manual.setMaxIterations(1);
  manual.setMaxCgIterations(1);
  manual.setGradientTolerance(1e-12);
  manual.setTrustRegionInitialRadius(10.0);
  const Matrix Xmanual = manual.optimize(X0);
  const double fManual = problem.f(Xmanual);

  EXPECT_LE(fManual, fRopt - 1e-3);
  EXPECT_LT(problem.RieGradNorm(Xmanual), problem.RieGradNorm(Xropt));
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerEliminatesTranslations) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 3;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned rCol = pose * (d + 1);
    const unsigned tCol = pose * (d + 1) + d;
    Q.insert(rCol, rCol) = 0.2;
    Q.insert(tCol, tCol) = 2.0 + pose;
    Q.insert(rCol, tCol) = 0.3 + 0.1 * pose;
    Q.insert(tCol, rCol) = 0.3 + 0.1 * pose;
  }
  Q.insert(d, 2 * (d + 1) + d) = -0.2;
  Q.insert(2 * (d + 1) + d, d) = -0.2;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned tCol = pose * (d + 1) + d;
    G.insert(0, tCol) = 1.5 - pose;
    G.insert(1, tCol) = -0.5 + 0.25 * pose;
    G.insert(2, tCol) = 0.75;
  }
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
    X0.col(pose * (d + 1) + d).setConstant(3.0 - pose);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setTrustRegionIterations(0);
  optimizer.setTrustRegionTolerance(1e-12);

  const Matrix Xopt = optimizer.optimize(X0);
  const Matrix egrad = Xopt * problem.getQ() + problem.getG();

  EXPECT_LT(problem.f(Xopt), problem.f(X0) - 1.0);
  for (unsigned pose = 0; pose < n; ++pose) {
    EXPECT_LE(egrad.col(pose * (d + 1) + d).norm(), 1e-8);
  }
  EXPECT_LE((Xopt.block(0, 0, d, d).transpose() *
                 Xopt.block(0, 0, d, d) -
             Matrix::Identity(d, d))
                .norm(),
            1e-9);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerExposesVariableProjectedSchurModel) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 1;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q(d + 1, d + 1);
  Q.insert(0, 0) = 4.0;
  Q.insert(1, 1) = 5.0;
  Q.insert(2, 2) = 6.0;
  Q.insert(3, 3) = 8.0;
  Q.insert(0, 3) = 1.2;
  Q.insert(3, 0) = 1.2;
  Q.insert(1, 3) = -0.8;
  Q.insert(3, 1) = -0.8;
  Q.insert(2, 3) = 0.4;
  Q.insert(3, 2) = 0.4;
  problem.setQ(Q);

  SparseMatrix G(r, d + 1);
  G.insert(0, 3) = -2.0;
  G.insert(1, 3) = 1.0;
  G.insert(2, 3) = -0.5;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, d + 1);
  X0.block(0, 0, d, d) = Matrix::Identity(d, d);
  X0(0, d) = 5.0;
  X0(1, d) = -4.0;
  X0(2, d) = 3.0;
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setValidateTranslationRecoveryCost(false);
  const Matrix recovered = optimizer.recoverTranslationsForRotations(X0);

  ASSERT_EQ(recovered.rows(), X0.rows());
  ASSERT_EQ(recovered.cols(), X0.cols());
  EXPECT_LT(problem.f(recovered), problem.f(X0) - 1.0);
  EXPECT_NEAR(optimizer.evaluateReducedObjective(X0), problem.f(recovered),
              1e-12);

  const Matrix euclideanGradient = recovered * problem.getQ() + problem.getG();
  EXPECT_LE(euclideanGradient.col(d).norm(), 1e-10);

  const Matrix reducedGradient = optimizer.buildReducedGradient(X0);
  EXPECT_LE(reducedGradient.col(d).norm(), 1e-14);
  const Matrix expectedProjectedGradient =
      euclideanGradient.leftCols(d) -
      X0.leftCols(d) *
          (0.5 * (X0.leftCols(d).transpose() *
                      euclideanGradient.leftCols(d) +
                  euclideanGradient.leftCols(d).transpose() *
                      X0.leftCols(d)));
  EXPECT_LE((reducedGradient.leftCols(d) - expectedProjectedGradient).norm(),
            1e-10);

  Matrix eta = Matrix::Zero(r, d + 1);
  eta(0, 1) = -0.3;
  eta(1, 0) = 0.3;
  eta(0, 2) = 0.2;
  eta(2, 0) = -0.2;
  eta(1, 2) = -0.1;
  eta(2, 1) = 0.1;

  const Matrix hv = optimizer.applyReducedSchurHv(X0, eta);
  EXPECT_LE(hv.col(d).norm(), 1e-14);

  Matrix qRR = Matrix::Zero(d, d);
  for (unsigned row = 0; row < d; ++row) {
    for (unsigned col = 0; col < d; ++col) {
      qRR(row, col) = Q.coeff(row, col);
    }
  }
  Matrix qRt = Matrix::Zero(d, 1);
  Matrix qtR = Matrix::Zero(1, d);
  for (unsigned col = 0; col < d; ++col) {
    qRt(col, 0) = Q.coeff(col, d);
    qtR(0, col) = Q.coeff(d, col);
  }
  const Matrix schur = qRR - (qRt * qtR) / Q.coeff(d, d);
  const Matrix ambientHv = eta.leftCols(d) * schur;
  const Matrix expectedProjectedHv =
      ambientHv -
      X0.leftCols(d) *
          (0.5 * (X0.leftCols(d).transpose() * ambientHv +
                  ambientHv.transpose() * X0.leftCols(d)));
  EXPECT_LE((hv.leftCols(d) - expectedProjectedHv).norm(), 1e-10);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerCompactHvMatchesExplicitMultiPoseSchur) {
  const char *previousCompactRaw =
      std::getenv("DRAN_REDUCED_ROTATION_COMPACT_HVP");
  const char *previousPrecomputedRaw =
      std::getenv("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP");
  const bool hadPreviousCompact = previousCompactRaw != nullptr;
  const bool hadPreviousPrecomputed = previousPrecomputedRaw != nullptr;
  const std::string previousCompact =
      hadPreviousCompact ? std::string(previousCompactRaw) : std::string();
  const std::string previousPrecomputed =
      hadPreviousPrecomputed ? std::string(previousPrecomputedRaw)
                             : std::string();
  setenv("DRAN_REDUCED_ROTATION_COMPACT_HVP", "true", 1);
  setenv("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP", "true", 1);

  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 2;
  const unsigned cols = (d + 1) * n;
  QuadraticProblem problem(n, d, r);

  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    dense(col, col) = 4.0 + 0.25 * col;
  }
  dense(0, d) = 0.7;
  dense(d, 0) = 0.7;
  dense(1, d) = -0.2;
  dense(d, 1) = -0.2;
  dense(4, 7) = -0.5;
  dense(7, 4) = -0.5;
  dense(5, 7) = 0.4;
  dense(7, 5) = 0.4;
  dense(d, 7) = -0.35;
  dense(7, d) = -0.35;
  dense(2, 4) = 0.15;
  dense(4, 2) = 0.15;
  SparseMatrix Q = dense.sparseView();
  problem.setQ(Q);

  SparseMatrix G(r, cols);
  G.insert(0, d) = -0.8;
  G.insert(1, 7) = 0.6;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  X0(0, d) = 0.25;
  X0(1, 7) = -0.4;
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  Matrix eta = Matrix::Zero(r, cols);
  eta(0, 1) = -0.11;
  eta(1, 0) = 0.11;
  eta(0, 2) = 0.07;
  eta(2, 0) = -0.07;
  eta(1, 2) = -0.05;
  eta(2, 1) = 0.05;
  eta(0, 5) = -0.13;
  eta(2, 4) = 0.13;
  eta(1, 6) = 0.09;
  eta(2, 5) = -0.09;

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setValidateTranslationRecoveryCost(false);
  optimizer.setProfileRuntime(true);
  const Matrix base = optimizer.recoverTranslationsForRotations(X0);
  const Matrix hv = optimizer.applyReducedSchurHv(X0, eta);

  std::vector<unsigned> rotationCols;
  std::vector<unsigned> translationCols;
  for (unsigned pose = 0; pose < n; ++pose) {
    for (unsigned localCol = 0; localCol < d; ++localCol) {
      rotationCols.push_back(pose * (d + 1) + localCol);
    }
    translationCols.push_back(pose * (d + 1) + d);
  }

  Matrix qTT = Matrix::Zero(n, n);
  Matrix etaRQt = Matrix::Zero(r, n);
  for (unsigned col = 0; col < n; ++col) {
    const unsigned tCol = translationCols[col];
    for (unsigned row = 0; row < n; ++row) {
      qTT(row, col) = Q.coeff(translationCols[row], tCol);
    }
    for (unsigned rotIdx = 0; rotIdx < rotationCols.size(); ++rotIdx) {
      etaRQt.col(col).noalias() +=
          eta.col(rotationCols[rotIdx]) *
          Q.coeff(rotationCols[rotIdx], tCol);
    }
  }
  const Matrix etaTranslation =
      -qTT.ldlt().solve(etaRQt.transpose()).transpose();
  Matrix fullEta = eta;
  for (unsigned pose = 0; pose < n; ++pose) {
    fullEta.col(translationCols[pose]) = etaTranslation.col(pose);
  }
  const Matrix ambientHv = fullEta * Q;

  Matrix expected = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    const Matrix R = base.block(0, colStart, r, d);
    const Matrix ZR = ambientHv.block(0, colStart, r, d);
    Matrix sym = R.transpose() * ZR;
    sym = (0.5 * (sym + sym.transpose())).eval();
    expected.block(0, colStart, r, d) = ZR - R * sym;
  }

  EXPECT_LE((hv - expected).norm(), 1e-10);

  optimizer.setTrustRegionIterations(1);
  optimizer.setTrustRegionMaxInnerIterations(2);
  optimizer.optimize(X0);
  EXPECT_GT(optimizer.getProfileReducedHessianProductCount(), 0u);

  if (hadPreviousCompact) {
    setenv("DRAN_REDUCED_ROTATION_COMPACT_HVP", previousCompact.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_COMPACT_HVP");
  }
  if (hadPreviousPrecomputed) {
    setenv("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP",
           previousPrecomputed.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP");
  }
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerDampsTranslationEliminationAroundInput) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 1;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  Q.insert(d, d) = 1.0;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = -10.0;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, d + 1);
  X0.block(0, 0, d, d) = Matrix::Identity(d, d);
  X0(0, d) = 2.0;
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer exact(&problem);
  exact.setTrustRegionIterations(0);
  exact.setValidateTranslationRecoveryCost(false);
  const Matrix Xexact = exact.optimize(X0);

  ReducedRotationQuadraticOptimizer damped(&problem);
  damped.setTrustRegionIterations(0);
  damped.setValidateTranslationRecoveryCost(false);
  damped.setTranslationEliminationProxWeight(3.0);
  const Matrix Xdamped = damped.optimize(X0);

  EXPECT_NEAR(Xexact(0, d), 10.0, 1e-10);
  EXPECT_NEAR(Xdamped(0, d), 4.0, 1e-10);
  EXPECT_GT(std::abs(Xdamped(0, d) - X0(0, d)),
            1e-10);
  EXPECT_LT(std::abs(Xdamped(0, d) - X0(0, d)),
            std::abs(Xexact(0, d) - X0(0, d)));
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerReusesTranslationFactorization) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 2;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned tCol = pose * (d + 1) + d;
    Q.insert(tCol, tCol) = 2.0 + pose;
  }
  Q.insert(d, d + 1 + d) = -0.25;
  Q.insert(d + 1 + d, d) = -0.25;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = 1.0;
  G.insert(1, d + 1 + d) = -2.0;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setTrustRegionIterations(0);
  optimizer.setTrustRegionTolerance(1e-12);

  const Matrix X1 = optimizer.optimize(X0);
  EXPECT_EQ(optimizer.getTranslationFactorizationCount(), 1u);

  SparseMatrix G2(r, (d + 1) * n);
  G2.insert(2, d) = -1.5;
  G2.insert(0, d + 1 + d) = 0.5;
  problem.setG(G2);
  const Matrix X2 = optimizer.optimize(X1);

  EXPECT_EQ(optimizer.getTranslationFactorizationCount(), 1u);
  EXPECT_TRUE(std::isfinite(problem.f(X2)));
}

TEST(testDPGO, QuadraticProblemCanUpdateQWithoutRebuildingPreconditioner) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 1;
  QuadraticProblem reference(n, d, r);
  QuadraticProblem fast(n, d, r);

  SparseMatrix Q1(d + 1, d + 1);
  SparseMatrix Q2(d + 1, d + 1);
  for (unsigned col = 0; col < d + 1; ++col) {
    Q1.insert(col, col) = 1.0 + col;
    Q2.insert(col, col) = 2.0 + 0.5 * col;
  }
  Q2.insert(0, d) = 0.25;
  Q2.insert(d, 0) = 0.25;

  SparseMatrix G(r, d + 1);
  G.insert(0, d) = -1.0;
  G.insert(1, 1) = 0.5;
  reference.setG(G);
  fast.setG(G);

  reference.setQ(Q2);
  fast.setQ(Q1);
  fast.setQWithoutPreconditioner(Q2);

  Matrix X = Matrix::Zero(r, d + 1);
  X.block(0, 0, d, d) = Matrix::Identity(d, d);
  X(0, d) = 1.25;
  LiftedSEManifold manifold(r, d, n);
  X = manifold.project(X);

  EXPECT_NEAR(fast.f(X), reference.f(X), 1e-12);
  EXPECT_LE((fast.RieGrad(X) - reference.RieGrad(X)).norm(), 1e-12);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerReusesMultipleTranslationFactorizations) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 2;
  QuadraticProblem problem(n, d, r);

  auto makeQ = [&](double firstTranslationDiagonal) {
    SparseMatrix Q((d + 1) * n, (d + 1) * n);
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned base = pose * (d + 1);
      for (unsigned col = 0; col < d; ++col) {
        Q.insert(base + col, base + col) = 1.0 + 0.1 * pose + 0.01 * col;
      }
      Q.insert(base + d, base + d) =
          pose == 0 ? firstTranslationDiagonal : 3.0;
    }
    Q.insert(d, d + 1 + d) = -0.25;
    Q.insert(d + 1 + d, d) = -0.25;
    return Q;
  };

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = 1.0;
  G.insert(1, d + 1 + d) = -2.0;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setTrustRegionIterations(0);
  optimizer.setTrustRegionTolerance(1e-12);
  optimizer.setValidateTranslationRecoveryCost(false);

  const SparseMatrix Q1 = makeQ(2.0);
  const SparseMatrix Q2 = makeQ(4.0);

  problem.setQ(Q1);
  const Matrix X1 = optimizer.optimize(X0);
  EXPECT_EQ(optimizer.getTranslationFactorizationCount(), 1u);

  problem.setQ(Q2);
  const Matrix X2 = optimizer.optimize(X1);
  EXPECT_EQ(optimizer.getTranslationFactorizationCount(), 2u);

  problem.setQ(Q1);
  const Matrix X3 = optimizer.optimize(X2);
  EXPECT_EQ(optimizer.getTranslationFactorizationCount(), 2u);
  EXPECT_TRUE(std::isfinite(problem.f(X3)));
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerJacobiPreconditionerHelpsIllConditionedRotation) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 2;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  const double stiffness[] = {0.05, 80.0, 0.08, 0.04, 120.0, 0.06};
  for (unsigned pose = 0; pose < n; ++pose) {
    for (unsigned col = 0; col < d; ++col) {
      const unsigned idx = pose * (d + 1) + col;
      Q.insert(idx, idx) = stiffness[pose * d + col];
    }
    Q.insert(pose * (d + 1) + d, pose * (d + 1) + d) = 1.0;
  }
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, 1) = -12.0;
  G.insert(1, 1) = 8.0;
  G.insert(0, 4) = 15.0;
  G.insert(2, 4) = -9.0;
  G.insert(1, 2) = -1.5;
  G.insert(2, 6) = 2.0;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer plain(&problem);
  plain.setTrustRegionIterations(1);
  plain.setTrustRegionMaxInnerIterations(1);
  plain.setTrustRegionTolerance(1e-12);
  plain.setTrustRegionInitialRadius(0.5);
  const Matrix Xplain = plain.optimize(X0);

  ReducedRotationQuadraticOptimizer jacobi(&problem);
  jacobi.setTrustRegionIterations(1);
  jacobi.setTrustRegionMaxInnerIterations(1);
  jacobi.setTrustRegionTolerance(1e-12);
  jacobi.setTrustRegionInitialRadius(0.5);
  jacobi.setUseJacobiPreconditioner(true);
  const Matrix Xjacobi = jacobi.optimize(X0);

  EXPECT_LE(problem.f(Xjacobi), problem.f(Xplain) + 1e-10);
  EXPECT_LE(problem.RieGradNorm(Xjacobi), problem.RieGradNorm(Xplain) + 1e-8);
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerPackedCholeskyProjectionMatchesDefault) {
  const char *previousRaw =
      std::getenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT");
  const bool hadPrevious = previousRaw != nullptr;
  const std::string previous = hadPrevious ? std::string(previousRaw)
                                           : std::string();

  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 3;
  const unsigned cols = n * (d + 1);
  QuadraticProblem problem(n, d, r);

  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    dense(col, col) = 3.0 + 0.2 * static_cast<double>(col);
  }
  dense(0, 3) = 0.4;
  dense(3, 0) = 0.4;
  dense(1, 7) = -0.25;
  dense(7, 1) = -0.25;
  dense(5, 11) = 0.3;
  dense(11, 5) = 0.3;
  dense(3, 7) = -0.18;
  dense(7, 3) = -0.18;
  dense(2, 4) = 0.12;
  dense(4, 2) = 0.12;
  dense += 1.0 * Matrix::Identity(cols, cols);
  problem.setQ(dense.sparseView());

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.7;
  G.insert(1, 4) = 0.5;
  G.insert(2, 7) = -0.4;
  G.insert(3, 11) = 0.3;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
    X0(3, pose * (d + 1) + d) = 0.05 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  auto run = [&](bool packed) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT",
           packed ? "true" : "false", 1);
    ReducedRotationQuadraticOptimizer optimizer(&problem);
    optimizer.setUseCholeskyPreconditioner(true);
    optimizer.setTrustRegionIterations(2);
    optimizer.setTrustRegionAcceptedIterations(2);
    optimizer.setTrustRegionMaxInnerIterations(4);
    optimizer.setTrustRegionTolerance(1e-12);
    optimizer.setTrustRegionInitialRadius(0.5);
    return optimizer.optimize(X0);
  };

  const Matrix defaultResult = run(false);
  const Matrix packedResult = run(true);

  EXPECT_LE((defaultResult - packedResult).norm(), 1e-12);
  EXPECT_NEAR(problem.f(defaultResult), problem.f(packedResult), 1e-12);
  EXPECT_NEAR(problem.RieGradNorm(defaultResult),
              problem.RieGradNorm(packedResult), 1e-12);

  if (hadPrevious) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT",
           previous.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT");
  }
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerPackedTcgMatchesDefault) {
  const char *previousRaw = std::getenv("DRAN_REDUCED_ROTATION_PACKED_TCG");
  const bool hadPrevious = previousRaw != nullptr;
  const std::string previous = hadPrevious ? std::string(previousRaw)
                                           : std::string();

  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 3;
  const unsigned cols = n * (d + 1);
  QuadraticProblem problem(n, d, r);

  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    dense(col, col) = 4.0 + 0.15 * static_cast<double>(col);
  }
  dense(0, 3) = 0.35;
  dense(3, 0) = 0.35;
  dense(1, 7) = -0.22;
  dense(7, 1) = -0.22;
  dense(5, 11) = 0.27;
  dense(11, 5) = 0.27;
  dense(2, 4) = 0.11;
  dense(4, 2) = 0.11;
  dense(6, 10) = -0.09;
  dense(10, 6) = -0.09;
  dense += 1.0 * Matrix::Identity(cols, cols);
  problem.setQ(dense.sparseView());

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.6;
  G.insert(1, 4) = 0.45;
  G.insert(2, 7) = -0.35;
  G.insert(3, 10) = 0.25;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
    X0(3, pose * (d + 1) + d) = 0.03 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  auto run = [&](bool packed) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_TCG",
           packed ? "true" : "false", 1);
    ReducedRotationQuadraticOptimizer optimizer(&problem);
    optimizer.setUseCholeskyPreconditioner(true);
    optimizer.setTrustRegionIterations(2);
    optimizer.setTrustRegionAcceptedIterations(2);
    optimizer.setTrustRegionMaxInnerIterations(4);
    optimizer.setTrustRegionTolerance(1e-12);
    optimizer.setTrustRegionInitialRadius(0.5);
    return optimizer.optimize(X0);
  };

  const Matrix defaultResult = run(false);
  const Matrix packedResult = run(true);

  EXPECT_LE((defaultResult - packedResult).norm(), 1e-10);
  EXPECT_NEAR(problem.f(defaultResult), problem.f(packedResult), 1e-10);
  EXPECT_NEAR(problem.RieGradNorm(defaultResult),
              problem.RieGradNorm(packedResult), 1e-10);

  if (hadPrevious) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_TCG", previous.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PACKED_TCG");
  }
}

TEST(testDPGO,
     ReducedRotationQuadraticOptimizerPackedTcgCholeskyProjectMatchesDefault) {
  const char *previousPackedRaw =
      std::getenv("DRAN_REDUCED_ROTATION_PACKED_TCG");
  const char *previousProjectRaw =
      std::getenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT");
  const bool hadPreviousPacked = previousPackedRaw != nullptr;
  const bool hadPreviousProject = previousProjectRaw != nullptr;
  const std::string previousPacked =
      hadPreviousPacked ? std::string(previousPackedRaw) : std::string();
  const std::string previousProject =
      hadPreviousProject ? std::string(previousProjectRaw) : std::string();

  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 3;
  const unsigned cols = n * (d + 1);
  QuadraticProblem problem(n, d, r);

  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    dense(col, col) = 4.0 + 0.17 * static_cast<double>(col);
  }
  dense(0, 3) = 0.31;
  dense(3, 0) = 0.31;
  dense(1, 7) = -0.24;
  dense(7, 1) = -0.24;
  dense(2, 8) = 0.16;
  dense(8, 2) = 0.16;
  dense(5, 11) = 0.23;
  dense(11, 5) = 0.23;
  dense(6, 10) = -0.13;
  dense(10, 6) = -0.13;
  dense += Matrix::Identity(cols, cols);
  problem.setQ(dense.sparseView());

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.55;
  G.insert(1, 4) = 0.40;
  G.insert(2, 7) = -0.32;
  G.insert(3, 10) = 0.22;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
    X0(3, pose * (d + 1) + d) = 0.025 * static_cast<double>(pose + 1);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  auto run = [&](bool packedProjection) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_TCG", "true", 1);
    setenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT",
           packedProjection ? "true" : "false", 1);
    ReducedRotationQuadraticOptimizer optimizer(&problem);
    optimizer.setUseCholeskyPreconditioner(true);
    optimizer.setTrustRegionIterations(2);
    optimizer.setTrustRegionAcceptedIterations(2);
    optimizer.setTrustRegionMaxInnerIterations(4);
    optimizer.setTrustRegionTolerance(1e-12);
    optimizer.setTrustRegionInitialRadius(0.5);
    return optimizer.optimize(X0);
  };

  const Matrix defaultResult = run(false);
  const Matrix projectedResult = run(true);

  EXPECT_LE((defaultResult - projectedResult).norm(), 1e-12);
  EXPECT_NEAR(problem.f(defaultResult), problem.f(projectedResult), 1e-12);
  EXPECT_NEAR(problem.RieGradNorm(defaultResult),
              problem.RieGradNorm(projectedResult), 1e-12);

  if (hadPreviousPacked) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_TCG", previousPacked.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PACKED_TCG");
  }
  if (hadPreviousProject) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT",
           previousProject.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT");
  }
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerPackedRiemannianHvRuns) {
  const char *previousPackedRaw =
      std::getenv("DRAN_REDUCED_ROTATION_PACKED_TCG");
  const char *previousRiemannianRaw =
      std::getenv("DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP");
  const bool hadPreviousPacked = previousPackedRaw != nullptr;
  const bool hadPreviousRiemannian = previousRiemannianRaw != nullptr;
  const std::string previousPacked =
      hadPreviousPacked ? std::string(previousPackedRaw) : std::string();
  const std::string previousRiemannian =
      hadPreviousRiemannian ? std::string(previousRiemannianRaw)
                            : std::string();

  const unsigned d = 3;
  const unsigned r = 4;
  const unsigned n = 3;
  const unsigned cols = n * (d + 1);
  QuadraticProblem problem(n, d, r);

  Matrix dense = Matrix::Zero(cols, cols);
  for (unsigned col = 0; col < cols; ++col) {
    dense(col, col) = 4.0 + 0.12 * static_cast<double>(col);
  }
  dense(0, 4) = 0.18;
  dense(4, 0) = 0.18;
  dense(1, 7) = -0.21;
  dense(7, 1) = -0.21;
  dense(5, 10) = 0.19;
  dense(10, 5) = 0.19;
  dense += Matrix::Identity(cols, cols);
  problem.setQ(dense.sparseView());

  SparseMatrix G(r, cols);
  G.insert(0, 1) = -0.5;
  G.insert(1, 4) = 0.35;
  G.insert(2, 8) = -0.25;
  G.insert(3, 10) = 0.2;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, cols);
  for (unsigned pose = 0; pose < n; ++pose) {
    X0.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  setenv("DRAN_REDUCED_ROTATION_PACKED_TCG", "true", 1);
  setenv("DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP", "true", 1);
  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setUseCholeskyPreconditioner(true);
  optimizer.setTrustRegionIterations(2);
  optimizer.setTrustRegionAcceptedIterations(2);
  optimizer.setTrustRegionMaxInnerIterations(4);
  optimizer.setTrustRegionTolerance(1e-12);
  optimizer.setTrustRegionInitialRadius(0.5);
  const Matrix result = optimizer.optimize(X0);

  EXPECT_TRUE(result.allFinite());
  EXPECT_TRUE(std::isfinite(problem.f(result)));
  EXPECT_TRUE(std::isfinite(problem.RieGradNorm(result)));

  if (hadPreviousPacked) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_TCG", previousPacked.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PACKED_TCG");
  }
  if (hadPreviousRiemannian) {
    setenv("DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP",
           previousRiemannian.c_str(), 1);
  } else {
    unsetenv("DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP");
  }
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerCanSkipResultStats) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 1;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  Q.insert(d, d) = 2.0;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = 1.0;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, d + 1);
  X0.block(0, 0, d, d) = Matrix::Identity(d, d);
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer optimizer(&problem);
  optimizer.setTrustRegionIterations(0);
  optimizer.setRecordResultStats(false);

  const Matrix Xopt = optimizer.optimize(X0);
  const ROPTResult result = optimizer.getOptResult();

  EXPECT_TRUE(std::isfinite(problem.f(Xopt)));
  EXPECT_TRUE(result.success);
  EXPECT_TRUE(std::isnan(result.fInit));
  EXPECT_TRUE(std::isnan(result.gradNormInit));
  EXPECT_TRUE(std::isnan(result.fOpt));
  EXPECT_TRUE(std::isnan(result.gradNormOpt));
  EXPECT_GE(result.elapsedMs, 0.0);
}

TEST(testDPGO, ReducedRotationQuadraticOptimizerCanSkipRecoveryCostValidation) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 1;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  Q.insert(d, d) = 2.0;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = 1.0;
  problem.setG(G);

  Matrix X0 = Matrix::Zero(r, d + 1);
  X0.block(0, 0, d, d) = Matrix::Identity(d, d);
  LiftedSEManifold manifold(r, d, n);
  X0 = manifold.project(X0);

  ReducedRotationQuadraticOptimizer checked(&problem);
  checked.setTrustRegionIterations(0);
  checked.optimize(X0);
  EXPECT_GT(checked.getTranslationRecoveryCostValidationCount(), 0u);

  ReducedRotationQuadraticOptimizer unchecked(&problem);
  unchecked.setTrustRegionIterations(0);
  unchecked.setValidateTranslationRecoveryCost(false);
  const Matrix Xunchecked = unchecked.optimize(X0);

  EXPECT_EQ(unchecked.getTranslationRecoveryCostValidationCount(), 0u);
  EXPECT_TRUE(std::isfinite(problem.f(Xunchecked)));
  EXPECT_LE(problem.f(Xunchecked), problem.f(X0));
}
