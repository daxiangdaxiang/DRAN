#include <DPGO/TEDCCI.h>

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

Matrix deterministicMatrix(int rows, int cols) {
  Matrix A(rows, cols);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      A(r, c) = std::sin(0.17 * static_cast<double>((r + 1) * (c + 2))) +
                0.25 * std::cos(0.31 * static_cast<double>(r + c + 1));
    }
  }
  return A;
}

Vector deterministicVector(int rows) {
  Vector b(rows);
  for (int r = 0; r < rows; ++r) {
    b(r) = std::cos(0.23 * static_cast<double>(r + 1)) +
           0.1 * std::sin(0.41 * static_cast<double>(r + 2));
  }
  return b;
}

LinearFactorBlock makeDenseFactor(const std::vector<PoseKey> &keys, int rows,
                                  int blockDim) {
  LinearFactorBlock factor;
  factor.keys = keys;
  factor.A = deterministicMatrix(rows, static_cast<int>(keys.size()) * blockDim);
  factor.b = deterministicVector(rows);
  return factor;
}

Matrix stackA(const std::vector<LinearFactorBlock> &factors) {
  int rows = 0;
  int cols = factors.empty() ? 0 : factors.front().A.cols();
  for (const auto &factor : factors) {
    rows += factor.A.rows();
  }
  Matrix A(rows, cols);
  int offset = 0;
  for (const auto &factor : factors) {
    A.middleRows(offset, factor.A.rows()) = factor.A;
    offset += factor.A.rows();
  }
  return A;
}

Vector stackB(const std::vector<LinearFactorBlock> &factors) {
  int rows = 0;
  for (const auto &factor : factors) {
    rows += factor.b.rows();
  }
  Vector b(rows);
  int offset = 0;
  for (const auto &factor : factors) {
    b.segment(offset, factor.b.rows()) = factor.b;
    offset += factor.b.rows();
  }
  return b;
}

void expectCondensationMatchesDirect(int blockDim, int interiorKeys,
                                     int boundaryKeys, int rows) {
  std::vector<PoseKey> interior;
  std::vector<PoseKey> boundary;
  std::vector<PoseKey> keys;
  for (int i = 0; i < interiorKeys; ++i) {
    interior.push_back(PoseKey{0, i});
    keys.push_back(PoseKey{0, i});
  }
  for (int i = 0; i < boundaryKeys; ++i) {
    boundary.push_back(PoseKey{1, i});
    keys.push_back(PoseKey{1, i});
  }

  const std::vector<LinearFactorBlock> factors = {
      makeDenseFactor(keys, rows, blockDim)};
  CondensedFactor condensed;
  BackSubstitutionCache cache;
  LocalQRCondensation::Condense(factors, interior, boundary, blockDim,
                                &condensed, &cache);

  const Matrix A = stackA(factors);
  const Vector b = stackB(factors);
  const int interiorDim = interiorKeys * blockDim;
  const int boundaryDim = boundaryKeys * blockDim;
  const Matrix AI = A.leftCols(interiorDim);
  const Matrix AB = A.rightCols(boundaryDim);
  const Vector xB = deterministicVector(boundaryDim);
  const Vector rhs = b - AB * xB;
  const Vector xIDirect = AI.colPivHouseholderQr().solve(rhs);
  const double directCost = (AI * xIDirect + AB * xB - b).squaredNorm();
  const double condensedCost = (condensed.Abar * xB - condensed.bbar).squaredNorm();
  const Vector xIBack = LocalQRCondensation::BackSubstitute(cache, xB);

  EXPECT_LT(std::abs(directCost - condensedCost), 1e-9);
  EXPECT_LT((xIBack - xIDirect).norm(), 1e-9);
  EXPECT_EQ(condensed.boundary_keys, boundary);
}

}  // namespace

TEST(testDPGO, TEDCCILocalQRCondensationMatchesOverdeterminedDirectSolve) {
  expectCondensationMatchesDirect(/*blockDim=*/2, /*interiorKeys=*/2,
                                  /*boundaryKeys=*/2, /*rows=*/12);
}

TEST(testDPGO, TEDCCILocalQRCondensationMatchesExactlyDeterminedDirectSolve) {
  expectCondensationMatchesDirect(/*blockDim=*/3, /*interiorKeys=*/2,
                                  /*boundaryKeys=*/1, /*rows=*/6);
}

TEST(testDPGO, TEDCCILocalQRCondensationUsesSparseBackendWhenRequested) {
  TEDCCIParams params;
  params.condensation_backend = TEDCCIBackend::SPARSE_SPQR;

  std::vector<PoseKey> interior = {PoseKey{0, 0}, PoseKey{0, 1},
                                   PoseKey{0, 2}};
  std::vector<PoseKey> boundary = {PoseKey{1, 3}, PoseKey{1, 4}};
  std::vector<PoseKey> keys = interior;
  keys.insert(keys.end(), boundary.begin(), boundary.end());

  const int blockDim = 2;
  const std::vector<LinearFactorBlock> factors = {
      makeDenseFactor(keys, /*rows=*/14, blockDim)};

  CondensedFactor sparseCondensed;
  BackSubstitutionCache sparseCache;
  LocalQRCondensation::Condense(factors, interior, boundary, blockDim,
                                &sparseCondensed, &sparseCache, params);

  TEDCCIParams denseParams;
  denseParams.condensation_backend = TEDCCIBackend::DENSE_HOUSEHOLDER;
  CondensedFactor denseCondensed;
  BackSubstitutionCache denseCache;
  LocalQRCondensation::Condense(factors, interior, boundary, blockDim,
                                &denseCondensed, &denseCache, denseParams);

  const Vector xB = deterministicVector(
      static_cast<int>(boundary.size()) * blockDim);
  const Vector sparseBack =
      LocalQRCondensation::BackSubstitute(sparseCache, xB);
  const Vector denseBack =
      LocalQRCondensation::BackSubstitute(denseCache, xB);

  EXPECT_EQ(sparseCondensed.boundary_keys, boundary);
  EXPECT_EQ(sparseCache.condensation_backend, TEDCCIBackend::SPARSE_SPQR);
  EXPECT_LT(std::abs((sparseCondensed.Abar * xB - sparseCondensed.bbar)
                         .squaredNorm() -
                     (denseCondensed.Abar * xB - denseCondensed.bbar)
                         .squaredNorm()),
            1e-9);
  EXPECT_LT((sparseBack - denseBack).norm(), 1e-9);
}
