#include <DPGO/TEDCCI.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

Matrix deterministicMatrix(int rows, int cols, double phase) {
  Matrix A(rows, cols);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      A(r, c) = std::sin(phase + 0.13 * static_cast<double>((r + 1) * (c + 2))) +
                0.2 * std::cos(0.29 * static_cast<double>(r + c + 1));
    }
  }
  return A;
}

Vector deterministicVector(int rows, double phase) {
  Vector b(rows);
  for (int r = 0; r < rows; ++r) {
    b(r) = std::cos(phase + 0.19 * static_cast<double>(r + 1)) +
           0.15 * std::sin(0.31 * static_cast<double>(r + 2));
  }
  return b;
}

LinearFactorBlock makeFactor(const std::vector<PoseKey> &keys, int rows,
                             int blockDim, double phase) {
  LinearFactorBlock factor;
  factor.keys = keys;
  factor.A = deterministicMatrix(rows,
                                 static_cast<int>(keys.size()) * blockDim,
                                 phase);
  factor.b = deterministicVector(rows, phase);
  return factor;
}

size_t expectedCondensedBytes(const CondensedFactor &factor) {
  size_t matrixNonzeros = 0;
  for (int c = 0; c < factor.Abar.cols(); ++c) {
    for (int r = 0; r < factor.Abar.rows(); ++r) {
      if (factor.Abar(r, c) != 0.0) {
        ++matrixNonzeros;
      }
    }
  }
  size_t vectorNonzeros = 0;
  for (int r = 0; r < factor.bbar.rows(); ++r) {
    if (factor.bbar(r) != 0.0) {
      ++vectorNonzeros;
    }
  }
  const size_t header = sizeof(int) + sizeof(std::uint64_t) + sizeof(int) +
                        factor.boundary_keys.size() * 2u * sizeof(int) +
                        2u * sizeof(std::uint8_t);
  const size_t dense =
      header + 2u * sizeof(int) + sizeof(int) +
      static_cast<size_t>(factor.Abar.rows()) *
                   static_cast<size_t>(factor.Abar.cols()) * sizeof(double) +
      static_cast<size_t>(factor.bbar.rows()) * sizeof(double);
  const size_t sparse =
      header + 2u * sizeof(int) + sizeof(std::uint64_t) + sizeof(int) +
      sizeof(std::uint64_t) +
      matrixNonzeros * (2u * sizeof(int) + sizeof(double)) +
      vectorNonzeros * (sizeof(int) + sizeof(double));
  return std::min(dense, sparse);
}

size_t expectedSolutionBytes(const CondensedFactor &factor, int blockDim) {
  return sizeof(int) + sizeof(uint64_t) + sizeof(int) +
         factor.boundary_keys.size() * 2u * sizeof(int) +
         factor.boundary_keys.size() * static_cast<size_t>(blockDim) *
             sizeof(double);
}

Vector boundarySolutionFromFullProblem(
    const std::vector<LinearFactorBlock> &localFactors0,
    const std::vector<LinearFactorBlock> &localFactors1,
    const LinearFactorBlock &crossFactor, int blockDim) {
  const int xI0Dim = blockDim;
  const int xI1Dim = blockDim;
  const int xBDim = 2 * blockDim;
  const int totalCols = xI0Dim + xI1Dim + xBDim;
  int totalRows = crossFactor.A.rows();
  for (const auto &factor : localFactors0) {
    totalRows += factor.A.rows();
  }
  for (const auto &factor : localFactors1) {
    totalRows += factor.A.rows();
  }

  Matrix A = Matrix::Zero(totalRows, totalCols);
  Vector b(totalRows);
  int row = 0;
  for (const auto &factor : localFactors0) {
    A.block(row, 0, factor.A.rows(), blockDim) =
        factor.A.block(0, 0, factor.A.rows(), blockDim);
    A.block(row, xI0Dim + xI1Dim, factor.A.rows(), blockDim) =
        factor.A.block(0, blockDim, factor.A.rows(), blockDim);
    b.segment(row, factor.b.rows()) = factor.b;
    row += factor.A.rows();
  }
  for (const auto &factor : localFactors1) {
    A.block(row, xI0Dim, factor.A.rows(), blockDim) =
        factor.A.block(0, 0, factor.A.rows(), blockDim);
    A.block(row, xI0Dim + xI1Dim + blockDim, factor.A.rows(), blockDim) =
        factor.A.block(0, blockDim, factor.A.rows(), blockDim);
    b.segment(row, factor.b.rows()) = factor.b;
    row += factor.A.rows();
  }
  A.block(row, xI0Dim + xI1Dim, crossFactor.A.rows(), xBDim) =
      crossFactor.A;
  b.segment(row, crossFactor.b.rows()) = crossFactor.b;

  const Vector fullSolution = A.colPivHouseholderQr().solve(b);
  return fullSolution.segment(xI0Dim + xI1Dim, xBDim);
}

}  // namespace

TEST(testDPGO, TEDCCIInterfaceDirectSolverMatchesFullUncondensedSolve) {
  const int blockDim = 2;
  const PoseKey i0{0, 0};
  const PoseKey b0{0, 1};
  const PoseKey i1{1, 2};
  const PoseKey b1{1, 3};

  const std::vector<LinearFactorBlock> local0 = {
      makeFactor({i0, b0}, 5, blockDim, 0.1)};
  const std::vector<LinearFactorBlock> local1 = {
      makeFactor({i1, b1}, 6, blockDim, 0.7)};

  CondensedFactor condensed0;
  BackSubstitutionCache cache0;
  LocalQRCondensation::Condense(local0, {i0}, {b0}, blockDim,
                                &condensed0, &cache0);
  condensed0.owner_robot_id = 0;
  condensed0.factor_type = FactorType::TRANSLATION;

  CondensedFactor condensed1;
  BackSubstitutionCache cache1;
  LocalQRCondensation::Condense(local1, {i1}, {b1}, blockDim,
                                &condensed1, &cache1);
  condensed1.owner_robot_id = 1;
  condensed1.factor_type = FactorType::TRANSLATION;

  const LinearFactorBlock cross =
      makeFactor({b0, b1}, 4, blockDim, 1.3);

  TEDCCIParams params;
  TEDCCIStats stats;
  const Vector reduced = InterfaceDirectSolver::Solve(
      {condensed0, condensed1}, {cross}, blockDim, params, &stats);
  const Vector full = boundarySolutionFromFullProblem(local0, local1, cross,
                                                      blockDim);

  EXPECT_LT((reduced - full).norm(), 1e-8);
  EXPECT_EQ(stats.num_interface_vars, 2);
  EXPECT_EQ(stats.num_condensed_rows,
            condensed0.Abar.rows() + condensed1.Abar.rows());
  EXPECT_EQ(stats.num_factor_messages, 2);
  EXPECT_EQ(stats.num_solution_messages, 2);
  EXPECT_EQ(stats.bytes_sent_upward,
            expectedCondensedBytes(condensed0) +
                expectedCondensedBytes(condensed1));
  EXPECT_EQ(stats.bytes_sent_downward,
            expectedSolutionBytes(condensed0, blockDim) +
                expectedSolutionBytes(condensed1, blockDim));
}

TEST(testDPGO, TEDCCIInterfaceAsyncDDConvergesToDirectOnSmallSystem) {
  const int blockDim = 1;
  const PoseKey b0{0, 1};
  const PoseKey b1{1, 2};

  CondensedFactor condensed0;
  condensed0.owner_robot_id = 0;
  condensed0.factor_type = FactorType::TRANSLATION;
  condensed0.boundary_keys = {b0};
  condensed0.Abar.resize(1, 1);
  condensed0.Abar << 2.0;
  condensed0.bbar.resize(1);
  condensed0.bbar << 2.0;

  CondensedFactor condensed1;
  condensed1.owner_robot_id = 1;
  condensed1.factor_type = FactorType::TRANSLATION;
  condensed1.boundary_keys = {b1};
  condensed1.Abar.resize(1, 1);
  condensed1.Abar << 2.0;
  condensed1.bbar.resize(1);
  condensed1.bbar << 4.0;

  LinearFactorBlock cross;
  cross.keys = {b0, b1};
  cross.A.resize(1, 2);
  cross.A << 0.5, -0.5;
  cross.b.resize(1);
  cross.b << -0.5;

  TEDCCIParams params;
  params.async_dd_max_iters = 100;
  params.async_dd_rel_tol = 1e-12;

  const Vector direct = InterfaceDirectSolver::Solve(
      {condensed0, condensed1}, {cross}, blockDim, params);
  TEDCCIStats stats;
  const Vector async = InterfaceAsyncDDSolver::Solve(
      {condensed0, condensed1}, {cross}, blockDim, params, &stats);

  EXPECT_LT((async - direct).norm(), 1e-6);
  EXPECT_TRUE(stats.async_dd_converged);
  EXPECT_GT(stats.async_dd_iterations, 0);
  EXPECT_LT(stats.async_dd_final_residual,
            stats.async_dd_initial_residual);
  EXPECT_GT(stats.num_solution_messages, 0);
}

TEST(testDPGO, TEDCCIInterfaceAsyncDDSimulatedDelayAndReorderingConverges) {
  const int blockDim = 1;
  const PoseKey b0{0, 1};
  const PoseKey b1{1, 2};

  CondensedFactor condensed0;
  condensed0.owner_robot_id = 0;
  condensed0.factor_type = FactorType::TRANSLATION;
  condensed0.boundary_keys = {b0};
  condensed0.Abar.resize(1, 1);
  condensed0.Abar << 2.0;
  condensed0.bbar.resize(1);
  condensed0.bbar << 2.0;

  CondensedFactor condensed1;
  condensed1.owner_robot_id = 1;
  condensed1.factor_type = FactorType::TRANSLATION;
  condensed1.boundary_keys = {b1};
  condensed1.Abar.resize(1, 1);
  condensed1.Abar << 2.0;
  condensed1.bbar.resize(1);
  condensed1.bbar << 4.0;

  LinearFactorBlock cross;
  cross.keys = {b0, b1};
  cross.A.resize(1, 2);
  cross.A << 0.5, -0.5;
  cross.b.resize(1);
  cross.b << -0.5;

  TEDCCIParams params;
  params.async_dd_max_iters = 200;
  params.async_dd_rel_tol = 1e-12;

  const Vector direct = InterfaceDirectSolver::Solve(
      {condensed0, condensed1}, {cross}, blockDim, params);
  TEDCCIStats stats;
  const Vector async = InterfaceAsyncDDSolver::SolveWithSimulatedNetwork(
      {condensed0, condensed1}, {cross}, blockDim, params,
      /*max_delay_rounds=*/2, /*reverse_delivery_order=*/true, &stats);

  EXPECT_LT((async - direct).norm(), 1e-6);
  EXPECT_TRUE(stats.async_dd_converged);
  EXPECT_GT(stats.async_dd_iterations, 0);
  EXPECT_GT(stats.num_solution_messages, stats.async_dd_iterations);
  EXPECT_LT(stats.async_dd_final_residual,
            stats.async_dd_initial_residual);
}

TEST(testDPGO, TEDCCIInterfaceAsyncDDCoarseCorrectionHookIsNoOp) {
  const int blockDim = 1;
  const PoseKey b0{0, 1};
  const PoseKey b1{1, 2};

  CondensedFactor condensed0;
  condensed0.owner_robot_id = 0;
  condensed0.factor_type = FactorType::TRANSLATION;
  condensed0.boundary_keys = {b0};
  condensed0.Abar.resize(1, 1);
  condensed0.Abar << 2.0;
  condensed0.bbar.resize(1);
  condensed0.bbar << 2.0;

  CondensedFactor condensed1;
  condensed1.owner_robot_id = 1;
  condensed1.factor_type = FactorType::TRANSLATION;
  condensed1.boundary_keys = {b1};
  condensed1.Abar.resize(1, 1);
  condensed1.Abar << 2.0;
  condensed1.bbar.resize(1);
  condensed1.bbar << 4.0;

  LinearFactorBlock cross;
  cross.keys = {b0, b1};
  cross.A.resize(1, 2);
  cross.A << 0.5, -0.5;
  cross.b.resize(1);
  cross.b << -0.5;

  TEDCCIParams baselineParams;
  baselineParams.async_dd_max_iters = 100;
  baselineParams.async_dd_rel_tol = 1e-12;
  const Vector baseline = InterfaceAsyncDDSolver::Solve(
      {condensed0, condensed1}, {cross}, blockDim, baselineParams);

  TEDCCIParams hookParams = baselineParams;
  hookParams.async_dd_enable_coarse_correction = true;
  TEDCCIStats hookStats;
  const Vector hook = InterfaceAsyncDDSolver::Solve(
      {condensed0, condensed1}, {cross}, blockDim, hookParams, &hookStats);

  EXPECT_LT((hook - baseline).norm(), 1e-12);
  EXPECT_TRUE(hookStats.async_dd_converged);
  EXPECT_GT(hookStats.async_dd_coarse_correction_calls, 0);
}

TEST(testDPGO, TEDCCIInterfaceAsyncDDReportsNonConvergence) {
  const int blockDim = 1;
  const PoseKey b0{0, 1};

  CondensedFactor condensed;
  condensed.owner_robot_id = 0;
  condensed.factor_type = FactorType::TRANSLATION;
  condensed.boundary_keys = {b0};
  condensed.Abar.resize(1, 1);
  condensed.Abar << 2.0;
  condensed.bbar.resize(1);
  condensed.bbar << 2.0;

  TEDCCIParams params;
  params.async_dd_max_iters = 0;
  params.async_dd_rel_tol = 1e-12;

  TEDCCIStats stats;
  const Vector async = InterfaceAsyncDDSolver::Solve(
      {condensed}, {}, blockDim, params, &stats);

  EXPECT_EQ(async.rows(), 1);
  EXPECT_FALSE(stats.async_dd_converged);
  EXPECT_EQ(stats.async_dd_iterations, 0);
  EXPECT_GT(stats.async_dd_initial_residual, 0.0);
  EXPECT_EQ(stats.async_dd_final_residual,
            stats.async_dd_initial_residual);
}
