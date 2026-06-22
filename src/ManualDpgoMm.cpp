#include <DPGO/ManualDpgoMm.h>

#include <DPGO/DPGO_utils.h>
#include <DPGO/FullEquivHybridOptimizer.h>
#include <DPGO/ManualQuadraticOptimizer.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/ReducedRotationQuadraticOptimizer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>

namespace DPGO {
namespace {

using ColMajorSparseMatrix = Eigen::SparseMatrix<double>;
using ProfileClock = std::chrono::steady_clock;

double denseSpectralNorm(const Matrix &matrix) {
  if (matrix.size() == 0 || !matrix.allFinite()) {
    return 0.0;
  }
  Eigen::JacobiSVD<Matrix> svd(matrix);
  if (svd.singularValues().size() == 0) {
    return 0.0;
  }
  const double norm = svd.singularValues()(0);
  return std::isfinite(norm) ? std::max(0.0, norm) : 0.0;
}

bool sparseMatrixAllFinite(const ColMajorSparseMatrix &matrix) {
  for (int outer = 0; outer < matrix.outerSize(); ++outer) {
    for (ColMajorSparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
      if (!std::isfinite(it.value())) {
        return false;
      }
    }
  }
  return true;
}

struct ManualDpgoMmAmmTrace {
  bool valid{false};
  unsigned localIter{0};
  double fobj{0.0};
  double trueFobj{0.0};
  double baselineSurrogateFobj{0.0};
  double baselineSurrogateOffset{0.0};
  double baselineSurrogateGradNorm{0.0};
  double recursiveSimpleF{0.0};
  double recursiveSimpleFobj{0.0};
  double recursiveSimpleDeltaQDelta{0.0};
  double recursiveSimplePTerm{0.0};
  double recursiveSimplePreviousGk{0.0};
  double refinedRatio{0.0};
  unsigned localAcceptedIterations{0};
  unsigned localAcceptedIterationBudget{0};
  int numOscillations{0};
  double F0{0.0};
  double F1{0.0};
  double gamma{0.0};
  double GkhInitial{0.0};
  double surrogateGkhInitial{0.0};
  double minG{0.0};
  double GkAfterAccelerated{0.0};
  double surrogateGkAfterAccelerated{0.0};
  double GkhAfterRestartCheck{0.0};
  double surrogateGkhAfterRestartCheck{0.0};
  double finalGk{0.0};
  double surrogateFinalGk{0.0};
  double phiLhs{0.0};
  double phiRhs{0.0};
  double restartCertificateLhs{0.0};
  double restartCertificateRhs{0.0};
  double restartCertificateMargin{0.0};
  bool restartCertificateValid{false};
  bool restartCertificatePassed{false};
  bool baselineSurrogateState{false};
  bool recursiveSimpleState{false};
  bool recursiveSimpleValid{false};
  bool refined{false};
  bool proxReset{false};
  bool hardRestart{false};
  bool softRestart{false};
  bool restartUsedXakh{false};
  bool phiFallback{false};
  std::size_t translationRecoveryAttemptCount{0};
  std::size_t translationRecoveryAcceptedCount{0};
  std::size_t translationRecoveryRejectedCount{0};
  int softRestartHits0{0};
  int softRestartHits1{0};
};

struct ManualDpgoMmAmmTraceRow {
  unsigned iter{0};
  unsigned robot{0};
  ManualDpgoMmAmmTrace trace;
};

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

double bytesToMegabytes(std::size_t byteCount) {
  return static_cast<double>(byteCount) / (1024.0 * 1024.0);
}

class ScopedSecondsAccumulator {
 public:
  explicit ScopedSecondsAccumulator(double &targetSeconds)
      : target(&targetSeconds),
        start(ProfileClock::now()) {}

  ScopedSecondsAccumulator(const ScopedSecondsAccumulator &) = delete;
  ScopedSecondsAccumulator &
  operator=(const ScopedSecondsAccumulator &) = delete;

  ~ScopedSecondsAccumulator() {
    if (target != nullptr) {
      *target += std::chrono::duration<double>(
                     ProfileClock::now() - start)
                     .count();
    }
  }

 private:
  double *target;
  ProfileClock::time_point start;
};

class ScopedOptionalSecondsAccumulator {
 public:
  explicit ScopedOptionalSecondsAccumulator(double *targetSeconds)
      : target(targetSeconds),
        start(targetSeconds != nullptr ? ProfileClock::now()
                                       : ProfileClock::time_point()) {}

  ScopedOptionalSecondsAccumulator(const ScopedOptionalSecondsAccumulator &) =
      delete;
  ScopedOptionalSecondsAccumulator &
  operator=(const ScopedOptionalSecondsAccumulator &) = delete;

  ~ScopedOptionalSecondsAccumulator() {
    if (target != nullptr) {
      *target += std::chrono::duration<double>(
                     ProfileClock::now() - start)
                     .count();
    }
  }

 private:
  double *target;
  ProfileClock::time_point start;
};

double secondsSince(const ProfileClock::time_point &start) {
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

struct ManualDpgoMmOptimizerProfile {
  double totalWallSec{0.0};
  double prepareProblemSec{0.0};
  double initializationExchangeSec{0.0};
  double distributedInitializationLocalModelSec{0.0};
  double distributedInitializationLocalSolveSec{0.0};
  double distributedInitializationExchangeSec{0.0};
  double initialLocalModelSec{0.0};
  double initialGlobalEvaluationSec{0.0};
  double iterationLocalPhaseWallSec{0.0};
  double localModelUpdateWallSec{0.0};
  double localOptimizationWallSec{0.0};
  double localGradientFreshExchangeSec{0.0};
  double localGradientCorrectionWallSec{0.0};
  double globalEvaluationWallSec{0.0};
  double postExchangeWallSec{0.0};
  double outputWriteSec{0.0};
  double agentOptimizeLocalModelSec{0.0};
  double reducedSolverSec{0.0};
  double reducedCandidateEvaluationSec{0.0};
  double lazyCandidateGradientEvaluationSec{0.0};
  double solverStartGradientEvaluationSec{0.0};
  double candidateConstructionSec{0.0};
  double localModelSnapshotSec{0.0};
  double localModelSwitchSec{0.0};
  double localSurrogatePreparationSec{0.0};
  double localCandidateSelectionSec{0.0};
  double localAmmOuterSec{0.0};
  double localFinalizationSec{0.0};
  double dpgoSimpleLinearRowsSec{0.0};
  double dpgoSimpleLinearMultiplySec{0.0};
  double dpgoSimpleLinearAssemblySec{0.0};
  double dpgoSimpleCacheBuildSec{0.0};
  double dpgoSimpleProximalStartSec{0.0};
  double dpgoSimpleProximalRowsSec{0.0};
  double dpgoSimpleProximalRotationSignalSec{0.0};
  double dpgoSimpleProximalGradientSec{0.0};
  double dpgoSimpleProximalProjectionSec{0.0};
  double dpgoSimpleProximalTranslationSec{0.0};
  double dpgoSimpleProximalAssemblySec{0.0};
  double reducedTranslationRecoverySec{0.0};
  double reducedTranslationFactorizationSec{0.0};
  double reducedTranslationCacheLookupSec{0.0};
  double reducedHessianProductSec{0.0};
  double reducedObjectiveSec{0.0};
  double reducedGradientBuildSec{0.0};
  double reducedPreconditionerSec{0.0};
  double reducedProjectionSec{0.0};
  double reducedRetractionSec{0.0};
  double reducedStepNormSec{0.0};
  double reducedTrustRegionScalingSec{0.0};
  double reducedTranslationResponseSec{0.0};
  double reducedRotationProductSec{0.0};
  double explicitTranslationRecoverySec{0.0};
  std::size_t iterationCount{0};
  std::size_t localModelUpdateCount{0};
  std::size_t localOptimizationRoundCount{0};
  std::size_t localGradientFreshExchangeCount{0};
  std::size_t localGradientCorrectionRoundCount{0};
  std::size_t globalEvaluationCount{0};
  std::size_t postExchangeCount{0};
  std::size_t agentOptimizeLocalModelCount{0};
  std::size_t reducedSolveCount{0};
  std::size_t reducedCandidateEvaluationCount{0};
  std::size_t lazyCandidateGradientEvaluationCount{0};
  std::size_t solverStartGradientEvaluationCount{0};
  std::size_t candidateObjectConstructionCount{0};
  std::size_t candidateObjectReuseCount{0};
  std::size_t localModelSnapshotCount{0};
  std::size_t localModelSwitchCount{0};
  std::size_t localSurrogatePreparationCount{0};
  std::size_t localCandidateSelectionCount{0};
  std::size_t localAmmOuterCount{0};
  std::size_t localFinalizationCount{0};
  std::size_t dpgoSimpleLinearCount{0};
  std::size_t dpgoSimpleCacheBuildCount{0};
  std::size_t dpgoSimpleProximalStartCount{0};
  std::size_t reducedTranslationRecoveryCount{0};
  std::size_t reducedTranslationFactorizationCount{0};
  std::size_t reducedTranslationCacheLookupCount{0};
  std::size_t reducedHessianProductCount{0};
  std::size_t reducedObjectiveCount{0};
  std::size_t reducedGradientBuildCount{0};
  std::size_t reducedPreconditionerCount{0};
  std::size_t reducedProjectionCount{0};
  std::size_t reducedRetractionCount{0};
  std::size_t reducedStepNormCount{0};
  std::size_t reducedTrustRegionScalingCount{0};
  std::size_t reducedTranslationResponseCount{0};
  std::size_t reducedRotationProductCount{0};
  std::size_t reducedCurvatureCauchyCandidateCount{0};
  std::size_t reducedCurvatureCauchyAcceptedCount{0};
  std::size_t reducedCurvatureCauchyFallbackCandidateCount{0};
  std::size_t reducedCurvatureCauchyFallbackAcceptedCount{0};
  std::size_t reducedCandidateProjectionSkipCount{0};
  std::size_t explicitTranslationRecoveryCount{0};
  std::size_t candidateEvaluationCacheRequestCount{0};
  std::size_t candidateEvaluationCacheHitCount{0};
  std::size_t candidateEvaluationCacheMissCount{0};
  std::size_t reducedPortfolioSelectionCount{0};
  std::array<std::size_t, 4> reducedPortfolioCandidateCounts{};
  std::array<std::size_t, 4> reducedPortfolioWinnerCounts{};
  std::size_t reducedPortfolioWinnerCostMarginCount{0};
  double reducedPortfolioWinnerCostMarginSum{0.0};
  double reducedPortfolioWinnerCostMarginMin{
      std::numeric_limits<double>::infinity()};
  std::size_t reducedAdaptivePortfolioFullSelectionCount{0};
  std::size_t reducedAdaptivePortfolioFastPathCount{0};
  std::size_t reducedAdaptivePortfolioCertifiedFastPathCount{0};
  std::size_t reducedAdaptivePortfolioFallbackCount{0};

  void addAgentProfile(const ManualDpgoMmOptimizerProfile &profile) {
    agentOptimizeLocalModelSec += profile.agentOptimizeLocalModelSec;
    reducedSolverSec += profile.reducedSolverSec;
    reducedCandidateEvaluationSec += profile.reducedCandidateEvaluationSec;
    lazyCandidateGradientEvaluationSec +=
        profile.lazyCandidateGradientEvaluationSec;
    solverStartGradientEvaluationSec +=
        profile.solverStartGradientEvaluationSec;
    candidateConstructionSec += profile.candidateConstructionSec;
    localModelSnapshotSec += profile.localModelSnapshotSec;
    localModelSwitchSec += profile.localModelSwitchSec;
    localSurrogatePreparationSec += profile.localSurrogatePreparationSec;
    localCandidateSelectionSec += profile.localCandidateSelectionSec;
    localAmmOuterSec += profile.localAmmOuterSec;
    localFinalizationSec += profile.localFinalizationSec;
    dpgoSimpleLinearRowsSec += profile.dpgoSimpleLinearRowsSec;
    dpgoSimpleLinearMultiplySec += profile.dpgoSimpleLinearMultiplySec;
    dpgoSimpleLinearAssemblySec += profile.dpgoSimpleLinearAssemblySec;
    dpgoSimpleCacheBuildSec += profile.dpgoSimpleCacheBuildSec;
    dpgoSimpleProximalStartSec += profile.dpgoSimpleProximalStartSec;
    dpgoSimpleProximalRowsSec += profile.dpgoSimpleProximalRowsSec;
    dpgoSimpleProximalRotationSignalSec +=
        profile.dpgoSimpleProximalRotationSignalSec;
    dpgoSimpleProximalGradientSec += profile.dpgoSimpleProximalGradientSec;
    dpgoSimpleProximalProjectionSec +=
        profile.dpgoSimpleProximalProjectionSec;
    dpgoSimpleProximalTranslationSec +=
        profile.dpgoSimpleProximalTranslationSec;
    dpgoSimpleProximalAssemblySec += profile.dpgoSimpleProximalAssemblySec;
    reducedTranslationRecoverySec += profile.reducedTranslationRecoverySec;
    reducedTranslationFactorizationSec +=
        profile.reducedTranslationFactorizationSec;
    reducedTranslationCacheLookupSec +=
        profile.reducedTranslationCacheLookupSec;
    reducedHessianProductSec += profile.reducedHessianProductSec;
    reducedObjectiveSec += profile.reducedObjectiveSec;
    reducedGradientBuildSec += profile.reducedGradientBuildSec;
    reducedPreconditionerSec += profile.reducedPreconditionerSec;
    reducedProjectionSec += profile.reducedProjectionSec;
    reducedRetractionSec += profile.reducedRetractionSec;
    reducedStepNormSec += profile.reducedStepNormSec;
    reducedTrustRegionScalingSec += profile.reducedTrustRegionScalingSec;
    reducedTranslationResponseSec += profile.reducedTranslationResponseSec;
    reducedRotationProductSec += profile.reducedRotationProductSec;
    explicitTranslationRecoverySec += profile.explicitTranslationRecoverySec;
    agentOptimizeLocalModelCount += profile.agentOptimizeLocalModelCount;
    reducedSolveCount += profile.reducedSolveCount;
    reducedCandidateEvaluationCount +=
        profile.reducedCandidateEvaluationCount;
    lazyCandidateGradientEvaluationCount +=
        profile.lazyCandidateGradientEvaluationCount;
    solverStartGradientEvaluationCount +=
        profile.solverStartGradientEvaluationCount;
    candidateObjectConstructionCount += profile.candidateObjectConstructionCount;
    candidateObjectReuseCount += profile.candidateObjectReuseCount;
    localModelSnapshotCount += profile.localModelSnapshotCount;
    localModelSwitchCount += profile.localModelSwitchCount;
    localSurrogatePreparationCount += profile.localSurrogatePreparationCount;
    localCandidateSelectionCount += profile.localCandidateSelectionCount;
    localAmmOuterCount += profile.localAmmOuterCount;
    localFinalizationCount += profile.localFinalizationCount;
    dpgoSimpleLinearCount += profile.dpgoSimpleLinearCount;
    dpgoSimpleCacheBuildCount += profile.dpgoSimpleCacheBuildCount;
    dpgoSimpleProximalStartCount += profile.dpgoSimpleProximalStartCount;
    reducedTranslationRecoveryCount +=
        profile.reducedTranslationRecoveryCount;
    reducedTranslationFactorizationCount +=
        profile.reducedTranslationFactorizationCount;
    reducedTranslationCacheLookupCount +=
        profile.reducedTranslationCacheLookupCount;
    reducedHessianProductCount += profile.reducedHessianProductCount;
    reducedObjectiveCount += profile.reducedObjectiveCount;
    reducedGradientBuildCount += profile.reducedGradientBuildCount;
    reducedPreconditionerCount += profile.reducedPreconditionerCount;
    reducedProjectionCount += profile.reducedProjectionCount;
    reducedRetractionCount += profile.reducedRetractionCount;
    reducedStepNormCount += profile.reducedStepNormCount;
    reducedTrustRegionScalingCount += profile.reducedTrustRegionScalingCount;
    reducedTranslationResponseCount +=
        profile.reducedTranslationResponseCount;
    reducedRotationProductCount += profile.reducedRotationProductCount;
    reducedCurvatureCauchyCandidateCount +=
        profile.reducedCurvatureCauchyCandidateCount;
    reducedCurvatureCauchyAcceptedCount +=
        profile.reducedCurvatureCauchyAcceptedCount;
    reducedCurvatureCauchyFallbackCandidateCount +=
        profile.reducedCurvatureCauchyFallbackCandidateCount;
    reducedCurvatureCauchyFallbackAcceptedCount +=
        profile.reducedCurvatureCauchyFallbackAcceptedCount;
    reducedCandidateProjectionSkipCount +=
        profile.reducedCandidateProjectionSkipCount;
    explicitTranslationRecoveryCount += profile.explicitTranslationRecoveryCount;
    candidateEvaluationCacheRequestCount +=
        profile.candidateEvaluationCacheRequestCount;
    candidateEvaluationCacheHitCount += profile.candidateEvaluationCacheHitCount;
    candidateEvaluationCacheMissCount +=
        profile.candidateEvaluationCacheMissCount;
    reducedPortfolioSelectionCount += profile.reducedPortfolioSelectionCount;
    for (std::size_t idx = 0; idx < reducedPortfolioCandidateCounts.size();
         ++idx) {
      reducedPortfolioCandidateCounts[idx] +=
          profile.reducedPortfolioCandidateCounts[idx];
      reducedPortfolioWinnerCounts[idx] +=
          profile.reducedPortfolioWinnerCounts[idx];
    }
    reducedPortfolioWinnerCostMarginCount +=
        profile.reducedPortfolioWinnerCostMarginCount;
    reducedPortfolioWinnerCostMarginSum +=
        profile.reducedPortfolioWinnerCostMarginSum;
    reducedPortfolioWinnerCostMarginMin =
        std::min(reducedPortfolioWinnerCostMarginMin,
                 profile.reducedPortfolioWinnerCostMarginMin);
    reducedAdaptivePortfolioFullSelectionCount +=
        profile.reducedAdaptivePortfolioFullSelectionCount;
    reducedAdaptivePortfolioFastPathCount +=
        profile.reducedAdaptivePortfolioFastPathCount;
    reducedAdaptivePortfolioCertifiedFastPathCount +=
        profile.reducedAdaptivePortfolioCertifiedFastPathCount;
    reducedAdaptivePortfolioFallbackCount +=
        profile.reducedAdaptivePortfolioFallbackCount;
  }
};

std::size_t hashCombine(std::size_t seed, std::size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

std::size_t matrixExactFingerprint(const Matrix &value) {
  std::size_t seed = 1469598103934665603ull;
  seed = hashCombine(seed, std::hash<int>{}(value.rows()));
  seed = hashCombine(seed, std::hash<int>{}(value.cols()));
  for (int idx = 0; idx < value.size(); ++idx) {
    seed = hashCombine(seed, std::hash<double>{}(value.data()[idx]));
  }
  return seed;
}

bool matricesExactlyEqual(const Matrix &a, const Matrix &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.array() == b.array()).all();
}

double scalarMegabytes(std::size_t scalarCount) {
  return bytesToMegabytes(scalarCount * sizeof(double));
}

std::size_t posePayloadBytesForBlock(const Matrix &pose) {
  if (pose.cols() <= 1) {
    return static_cast<std::size_t>(pose.size()) * sizeof(double);
  }
  const std::size_t d = static_cast<std::size_t>(pose.cols() - 1);
  return d * static_cast<std::size_t>(pose.cols()) * sizeof(double);
}

std::size_t sparseDeltaPayloadBytes(std::size_t entryCount) {
  constexpr std::size_t headerBytes = 2 * sizeof(std::uint32_t);
  constexpr std::size_t entryBytes = sizeof(std::uint32_t) + sizeof(double);
  return headerBytes + entryCount * entryBytes;
}

std::size_t tangentDeltaPayloadBytes(unsigned d) {
  const std::size_t rotationDof = d == 2 ? 1 : (d == 3 ? 3 : 0);
  if (rotationDof == 0) {
    return std::numeric_limits<std::size_t>::max();
  }
  constexpr std::size_t headerBytes = sizeof(std::uint32_t);
  return headerBytes + (rotationDof + d) * sizeof(double);
}

std::size_t liftedDeltaPayloadBytes(std::size_t rowCount,
                                    std::size_t columnCount,
                                    std::size_t rank) {
  if (rowCount == 0 || columnCount == 0 || rank == 0) {
    return std::numeric_limits<std::size_t>::max();
  }
  const std::size_t selectedColumns = std::min(rank, columnCount);
  constexpr std::size_t headerBytes = sizeof(std::uint32_t);
  const std::size_t columnBytes =
      selectedColumns * (sizeof(std::uint32_t) + rowCount * sizeof(double));
  const std::size_t translationBytes = rowCount * sizeof(double);
  return headerBytes + columnBytes + translationBytes;
}

double valueSchedulerScore(double poseDeltaNorm, double staleness,
                           double boundarySensitivity,
                           std::size_t payloadBytes,
                           unsigned hopCount) {
  if (!std::isfinite(poseDeltaNorm) || poseDeltaNorm < 0.0 ||
      !std::isfinite(staleness) || staleness < 0.0 ||
      !std::isfinite(boundarySensitivity) || boundarySensitivity < 0.0 ||
      payloadBytes == 0) {
    return 0.0;
  }
  const double denominator =
      static_cast<double>(payloadBytes) *
      static_cast<double>(std::max(1u, hopCount));
  if (!std::isfinite(denominator) || denominator <= 0.0) {
    return 0.0;
  }
  return poseDeltaNorm * (1.0 + staleness) *
         (1.0 + boundarySensitivity) / denominator;
}

bool interfaceModelPayloadHasDirection(const std::string &payload) {
  return payload == "direction_block" ||
         payload == "direction_block_diag_stiffness";
}

bool interfaceModelPayloadHasDiagStiffness(const std::string &payload) {
  return payload == "diag_stiffness" ||
         payload == "direction_block_diag_stiffness";
}

bool validInterfaceModelPayload(const std::string &payload) {
  return payload == "direction_block" ||
         payload == "diag_stiffness" ||
         payload == "direction_block_diag_stiffness";
}

Matrix projectPoseBlock(Matrix pose) {
  if (pose.cols() > 1 && pose.rows() >= pose.cols() - 1) {
    const int d = pose.cols() - 1;
    pose.block(0, 0, pose.rows(), d) =
        projectToStiefelManifold(pose.block(0, 0, pose.rows(), d));
  }
  return pose;
}

bool isPhysicalRotationBlock(const Matrix &rotation, double tolerance) {
  if (rotation.rows() != rotation.cols() || rotation.rows() < 2 ||
      !rotation.allFinite() || !std::isfinite(tolerance) ||
      tolerance < 0.0) {
    return false;
  }
  const Matrix orthogonality =
      rotation.transpose() * rotation -
      Matrix::Identity(rotation.cols(), rotation.cols());
  const double orthogonalityError = orthogonality.norm();
  const double determinant = rotation.determinant();
  return std::isfinite(orthogonalityError) &&
         std::isfinite(determinant) &&
         orthogonalityError <= tolerance &&
         std::abs(determinant - 1.0) <= tolerance;
}

bool hasCanonicalHomogeneousTail(const Matrix &pose, unsigned d,
                                 double tolerance) {
  if (pose.rows() == static_cast<int>(d)) {
    return true;
  }
  if (pose.rows() != static_cast<int>(d + 1) ||
      pose.cols() != static_cast<int>(d + 1) ||
      !std::isfinite(tolerance) || tolerance < 0.0) {
    return false;
  }
  for (unsigned col = 0; col < d; ++col) {
    if (std::abs(pose(static_cast<int>(d), static_cast<int>(col))) >
        tolerance) {
      return false;
    }
  }
  return std::abs(pose(static_cast<int>(d), static_cast<int>(d)) - 1.0) <=
         tolerance;
}

bool reconstructSparseDeltaPose(const Matrix &previous, const Matrix &current,
                                unsigned topK, double maxError,
                                Matrix &reconstructed,
                                std::size_t &payloadBytes) {
  if (topK == 0 || !std::isfinite(maxError) || maxError < 0.0 ||
      previous.rows() != current.rows() ||
      previous.cols() != current.cols() ||
      previous.size() == 0 || !previous.allFinite() ||
      !current.allFinite()) {
    return false;
  }
  const std::size_t fullPayloadBytes = posePayloadBytesForBlock(current);
  const std::size_t maxEntries =
      std::min<std::size_t>(topK, static_cast<std::size_t>(current.size()));
  const std::size_t sparseBytes = sparseDeltaPayloadBytes(maxEntries);
  if (sparseBytes >= fullPayloadBytes) {
    return false;
  }

  std::vector<std::pair<double, int>> magnitudes;
  magnitudes.reserve(static_cast<std::size_t>(current.size()));
  for (int idx = 0; idx < current.size(); ++idx) {
    const double delta = current.data()[idx] - previous.data()[idx];
    if (!std::isfinite(delta)) {
      return false;
    }
    magnitudes.emplace_back(std::abs(delta), idx);
  }
  const auto compareMagnitude = [](const auto &lhs, const auto &rhs) {
    if (lhs.first == rhs.first) {
      return lhs.second < rhs.second;
    }
    return lhs.first > rhs.first;
  };
  if (maxEntries < magnitudes.size()) {
    std::nth_element(magnitudes.begin(),
                     magnitudes.begin() + static_cast<long>(maxEntries),
                     magnitudes.end(), compareMagnitude);
    magnitudes.resize(maxEntries);
  }
  std::sort(magnitudes.begin(), magnitudes.end(), compareMagnitude);

  reconstructed = previous;
  for (const auto &entry : magnitudes) {
    reconstructed.data()[entry.second] = current.data()[entry.second];
  }
  reconstructed = projectPoseBlock(std::move(reconstructed));
  const double error = (reconstructed - current).norm();
  if (!std::isfinite(error) || error > maxError) {
    return false;
  }
  payloadBytes = sparseBytes;
  return true;
}

bool reconstructTangentDeltaPose(const Matrix &previous, const Matrix &current,
                                 double maxError, Matrix &reconstructed,
                                 std::size_t &payloadBytes) {
  if (!std::isfinite(maxError) || maxError < 0.0 ||
      previous.rows() != current.rows() ||
      previous.cols() != current.cols() ||
      previous.cols() <= 1 || !previous.allFinite() ||
      !current.allFinite()) {
    return false;
  }
  const unsigned d = static_cast<unsigned>(current.cols() - 1);
  if ((d != 2 && d != 3) || current.rows() < static_cast<int>(d)) {
    return false;
  }
  const std::size_t tangentBytes = tangentDeltaPayloadBytes(d);
  if (tangentBytes >= posePayloadBytesForBlock(current)) {
    return false;
  }

  const Matrix previousRawR = previous.block(0, 0, d, d);
  const Matrix currentRawR = current.block(0, 0, d, d);
  constexpr double rotationTolerance = 1e-8;
  if (isPhysicalRotationBlock(previousRawR, rotationTolerance) &&
      isPhysicalRotationBlock(currentRawR, rotationTolerance) &&
      hasCanonicalHomogeneousTail(previous, d, rotationTolerance) &&
      hasCanonicalHomogeneousTail(current, d, rotationTolerance)) {
    reconstructed = current;
    payloadBytes = tangentBytes;
    return true;
  }

  const Matrix previousR =
      projectToRotationGroup(previous.block(0, 0, d, d));
  const Matrix currentR =
      projectToRotationGroup(current.block(0, 0, d, d));
  const Matrix deltaR =
      projectToRotationGroup(currentR * previousR.transpose());
  Matrix reconstructedDeltaR = Matrix::Identity(d, d);
  if (d == 2) {
    const double theta = std::atan2(deltaR(1, 0), deltaR(0, 0));
    if (!std::isfinite(theta)) {
      return false;
    }
    reconstructedDeltaR(0, 0) = std::cos(theta);
    reconstructedDeltaR(0, 1) = -std::sin(theta);
    reconstructedDeltaR(1, 0) = std::sin(theta);
    reconstructedDeltaR(1, 1) = std::cos(theta);
  } else {
    const Matrix tangent = logmap(deltaR);
    if (tangent.rows() != 3 || tangent.cols() != 3 ||
        !tangent.allFinite()) {
      return false;
    }
    reconstructedDeltaR = expmap(tangent);
  }

  reconstructed = previous;
  reconstructed.block(0, 0, d, d) =
      projectToRotationGroup(reconstructedDeltaR * previousR);
  reconstructed.block(0, d, d, 1) =
      previous.block(0, d, d, 1) +
      (current.block(0, d, d, 1) - previous.block(0, d, d, 1));
  const double error = (reconstructed - current).norm();
  if (!std::isfinite(error) || error > maxError) {
    return false;
  }
  payloadBytes = tangentBytes;
  return true;
}

bool reconstructLiftedDeltaPose(const Matrix &previous, const Matrix &current,
                                unsigned rank, double maxError,
                                Matrix &reconstructed,
                                std::size_t &payloadBytes) {
  if (rank == 0 || !std::isfinite(maxError) || maxError < 0.0 ||
      previous.rows() != current.rows() ||
      previous.cols() != current.cols() || previous.cols() <= 1 ||
      previous.size() == 0 || !previous.allFinite() ||
      !current.allFinite()) {
    return false;
  }
  const unsigned d = static_cast<unsigned>(current.cols() - 1);
  if (current.rows() < static_cast<int>(d)) {
    return false;
  }
  const std::size_t fullPayloadBytes = posePayloadBytesForBlock(current);
  const std::size_t selectedColumns =
      std::min<std::size_t>(rank, static_cast<std::size_t>(d));
  const std::size_t liftedBytes =
      liftedDeltaPayloadBytes(static_cast<std::size_t>(current.rows()), d,
                              selectedColumns);
  if (liftedBytes >= fullPayloadBytes) {
    return false;
  }

  const Matrix previousY = projectToStiefelManifold(
      previous.block(0, 0, previous.rows(), static_cast<int>(d)));
  const Matrix currentY = projectToStiefelManifold(
      current.block(0, 0, current.rows(), static_cast<int>(d)));
  const Matrix deltaY = currentY - previousY;
  const Matrix tangent =
      deltaY -
      previousY * (0.5 * (previousY.transpose() * deltaY +
                          deltaY.transpose() * previousY));
  if (!tangent.allFinite()) {
    return false;
  }

  std::vector<std::pair<double, unsigned>> columnNorms;
  columnNorms.reserve(d);
  for (unsigned col = 0; col < d; ++col) {
    columnNorms.emplace_back(
        tangent.col(static_cast<int>(col)).norm(), col);
  }
  const auto compareMagnitude = [](const auto &lhs, const auto &rhs) {
    if (lhs.first == rhs.first) {
      return lhs.second < rhs.second;
    }
    return lhs.first > rhs.first;
  };
  if (selectedColumns < columnNorms.size()) {
    std::nth_element(columnNorms.begin(),
                     columnNorms.begin() +
                         static_cast<long>(selectedColumns),
                     columnNorms.end(), compareMagnitude);
    columnNorms.resize(selectedColumns);
  }
  std::sort(columnNorms.begin(), columnNorms.end(), compareMagnitude);

  Matrix tangentApprox = Matrix::Zero(tangent.rows(), tangent.cols());
  for (const auto &entry : columnNorms) {
    tangentApprox.col(static_cast<int>(entry.second)) =
        tangent.col(static_cast<int>(entry.second));
  }

  reconstructed = previous;
  reconstructed.block(0, 0, reconstructed.rows(), static_cast<int>(d)) =
      projectToStiefelManifold(previousY + tangentApprox);
  reconstructed.col(static_cast<int>(d)) = current.col(static_cast<int>(d));
  const double error = (reconstructed - current).norm();
  if (!std::isfinite(error) || error > maxError) {
    return false;
  }
  payloadBytes = liftedBytes;
  return true;
}

std::string datasetStem(const std::string &path) {
  std::filesystem::path p(path);
  return p.stem().string();
}

void ensureOutputDirectory(const std::string &path) {
  if (path.empty()) {
    return;
  }
  std::filesystem::create_directories(path);
}

std::string jsonEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

bool envFlagEnabled(const char *name, bool defaultValue) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return defaultValue;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  if (value == "0" || value == "false" || value == "off" ||
      value == "no" || value == "disabled") {
    return false;
  }
  if (value == "1" || value == "true" || value == "on" ||
      value == "yes" || value == "enabled") {
    return true;
  }
  return defaultValue;
}

const char *profileSchemeName(ManualDpgoMmScheme scheme) {
  return scheme == ManualDpgoMmScheme::AMM ? "amm" : "mm";
}

const char *profileLocalSolverName(ManualDpgoMmLocalSolver solver) {
  switch (solver) {
    case ManualDpgoMmLocalSolver::ManualFull:
      return "manual_full";
    case ManualDpgoMmLocalSolver::FullEquivHybrid:
      return "full_equiv_hybrid";
    case ManualDpgoMmLocalSolver::ReducedRotation:
      return "reduced_rotation";
  }
  return "unknown";
}

const char *profileReducedRotationPreconditionerName(
    ManualDpgoMmReducedRotationPreconditioner mode) {
  switch (mode) {
    case ManualDpgoMmReducedRotationPreconditioner::None:
      return "none";
    case ManualDpgoMmReducedRotationPreconditioner::Jacobi:
      return "jacobi";
    case ManualDpgoMmReducedRotationPreconditioner::SchurJacobi:
      return "schur_jacobi";
    case ManualDpgoMmReducedRotationPreconditioner::Cholesky:
      return "cholesky";
    case ManualDpgoMmReducedRotationPreconditioner::Portfolio:
      return "portfolio";
    case ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio:
      return "adaptive_portfolio";
  }
  return "unknown";
}

int profileReducedRotationPreconditionerIndex(
    ManualDpgoMmReducedRotationPreconditioner mode) {
  switch (mode) {
    case ManualDpgoMmReducedRotationPreconditioner::None:
      return 0;
    case ManualDpgoMmReducedRotationPreconditioner::Jacobi:
      return 1;
    case ManualDpgoMmReducedRotationPreconditioner::SchurJacobi:
      return 2;
    case ManualDpgoMmReducedRotationPreconditioner::Cholesky:
      return 3;
    case ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio:
    case ManualDpgoMmReducedRotationPreconditioner::Portfolio:
      return -1;
  }
  return -1;
}

void writeReducedPortfolioCountObject(
    std::ostream &output,
    const std::array<std::size_t, 4> &counts,
    const std::string &indent) {
  output << indent << "{\n";
  output << indent << "  \"none\": " << counts[0] << ",\n";
  output << indent << "  \"jacobi\": " << counts[1] << ",\n";
  output << indent << "  \"schur_jacobi\": " << counts[2] << ",\n";
  output << indent << "  \"cholesky\": " << counts[3] << "\n";
  output << indent << "}";
}

struct ManualDpgoMmAgentProfileEntry {
  unsigned robotId{0};
  ManualDpgoMmOptimizerProfile profile;
};

void writeAgentProfileEntry(std::ostream &output,
                            const ManualDpgoMmAgentProfileEntry &entry,
                            const std::string &indent) {
  const ManualDpgoMmOptimizerProfile &profile = entry.profile;
  output << indent << "{\n";
  output << indent << "  \"robot_id\": " << entry.robotId << ",\n";
  output << indent << "  \"seconds\": {\n";
  output << indent << "    \"agent_optimize_local_model_sec\": "
         << profile.agentOptimizeLocalModelSec << ",\n";
  output << indent << "    \"reduced_solver_sec\": "
         << profile.reducedSolverSec << ",\n";
  output << indent << "    \"reduced_candidate_evaluation_sec\": "
         << profile.reducedCandidateEvaluationSec << ",\n";
  output << indent << "    \"lazy_candidate_gradient_evaluation_sec\": "
         << profile.lazyCandidateGradientEvaluationSec << ",\n";
  output << indent << "    \"solver_start_gradient_evaluation_sec\": "
         << profile.solverStartGradientEvaluationSec << ",\n";
  output << indent << "    \"candidate_construction_sec\": "
         << profile.candidateConstructionSec << ",\n";
  output << indent << "    \"local_model_snapshot_sec\": "
         << profile.localModelSnapshotSec << ",\n";
  output << indent << "    \"local_model_switch_sec\": "
         << profile.localModelSwitchSec << ",\n";
  output << indent << "    \"local_surrogate_preparation_sec\": "
         << profile.localSurrogatePreparationSec << ",\n";
  output << indent << "    \"local_candidate_selection_sec\": "
         << profile.localCandidateSelectionSec << ",\n";
  output << indent << "    \"local_amm_outer_sec\": "
         << profile.localAmmOuterSec << ",\n";
  output << indent << "    \"local_finalization_sec\": "
         << profile.localFinalizationSec << ",\n";
  output << indent << "    \"dpgo_simple_linear_rows_sec\": "
         << profile.dpgoSimpleLinearRowsSec << ",\n";
  output << indent << "    \"dpgo_simple_linear_multiply_sec\": "
         << profile.dpgoSimpleLinearMultiplySec << ",\n";
  output << indent << "    \"dpgo_simple_linear_assembly_sec\": "
         << profile.dpgoSimpleLinearAssemblySec << ",\n";
  output << indent << "    \"dpgo_simple_cache_build_sec\": "
         << profile.dpgoSimpleCacheBuildSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_start_sec\": "
         << profile.dpgoSimpleProximalStartSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_rows_sec\": "
         << profile.dpgoSimpleProximalRowsSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_rotation_signal_sec\": "
         << profile.dpgoSimpleProximalRotationSignalSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_gradient_sec\": "
         << profile.dpgoSimpleProximalGradientSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_projection_sec\": "
         << profile.dpgoSimpleProximalProjectionSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_translation_sec\": "
         << profile.dpgoSimpleProximalTranslationSec << ",\n";
  output << indent << "    \"dpgo_simple_proximal_assembly_sec\": "
         << profile.dpgoSimpleProximalAssemblySec << ",\n";
  output << indent << "    \"reduced_translation_recovery_sec\": "
         << profile.reducedTranslationRecoverySec << ",\n";
  output << indent << "    \"reduced_translation_factorization_sec\": "
         << profile.reducedTranslationFactorizationSec << ",\n";
  output << indent << "    \"reduced_translation_cache_lookup_sec\": "
         << profile.reducedTranslationCacheLookupSec << ",\n";
  output << indent << "    \"reduced_hessian_product_sec\": "
         << profile.reducedHessianProductSec << ",\n";
  output << indent << "    \"reduced_objective_sec\": "
         << profile.reducedObjectiveSec << ",\n";
  output << indent << "    \"reduced_gradient_build_sec\": "
         << profile.reducedGradientBuildSec << ",\n";
  output << indent << "    \"reduced_preconditioner_sec\": "
         << profile.reducedPreconditionerSec << ",\n";
  output << indent << "    \"reduced_projection_sec\": "
         << profile.reducedProjectionSec << ",\n";
  output << indent << "    \"reduced_retraction_sec\": "
         << profile.reducedRetractionSec << ",\n";
  output << indent << "    \"reduced_step_norm_sec\": "
         << profile.reducedStepNormSec << ",\n";
  output << indent << "    \"reduced_trust_region_scaling_sec\": "
         << profile.reducedTrustRegionScalingSec << ",\n";
  output << indent << "    \"reduced_translation_response_sec\": "
         << profile.reducedTranslationResponseSec << ",\n";
  output << indent << "    \"reduced_rotation_product_sec\": "
         << profile.reducedRotationProductSec << ",\n";
  output << indent << "    \"explicit_translation_recovery_sec\": "
         << profile.explicitTranslationRecoverySec << "\n";
  output << indent << "  },\n";
  output << indent << "  \"counts\": {\n";
  output << indent << "    \"agent_optimize_local_model_count\": "
         << profile.agentOptimizeLocalModelCount << ",\n";
  output << indent << "    \"reduced_solve_count\": "
         << profile.reducedSolveCount << ",\n";
  output << indent << "    \"reduced_candidate_evaluation_count\": "
         << profile.reducedCandidateEvaluationCount << ",\n";
  output << indent << "    \"lazy_candidate_gradient_evaluation_count\": "
         << profile.lazyCandidateGradientEvaluationCount << ",\n";
  output << indent << "    \"solver_start_gradient_evaluation_count\": "
         << profile.solverStartGradientEvaluationCount << ",\n";
  output << indent << "    \"candidate_object_construction_count\": "
         << profile.candidateObjectConstructionCount << ",\n";
  output << indent << "    \"candidate_object_reuse_count\": "
         << profile.candidateObjectReuseCount << ",\n";
  output << indent << "    \"local_model_snapshot_count\": "
         << profile.localModelSnapshotCount << ",\n";
  output << indent << "    \"local_model_switch_count\": "
         << profile.localModelSwitchCount << ",\n";
  output << indent << "    \"local_surrogate_preparation_count\": "
         << profile.localSurrogatePreparationCount << ",\n";
  output << indent << "    \"local_candidate_selection_count\": "
         << profile.localCandidateSelectionCount << ",\n";
  output << indent << "    \"local_amm_outer_count\": "
         << profile.localAmmOuterCount << ",\n";
  output << indent << "    \"local_finalization_count\": "
         << profile.localFinalizationCount << ",\n";
  output << indent << "    \"dpgo_simple_linear_count\": "
         << profile.dpgoSimpleLinearCount << ",\n";
  output << indent << "    \"dpgo_simple_cache_build_count\": "
         << profile.dpgoSimpleCacheBuildCount << ",\n";
  output << indent << "    \"dpgo_simple_proximal_start_count\": "
         << profile.dpgoSimpleProximalStartCount << ",\n";
  output << indent << "    \"reduced_translation_recovery_count\": "
         << profile.reducedTranslationRecoveryCount << ",\n";
  output << indent << "    \"reduced_translation_factorization_count\": "
         << profile.reducedTranslationFactorizationCount << ",\n";
  output << indent << "    \"reduced_translation_cache_lookup_count\": "
         << profile.reducedTranslationCacheLookupCount << ",\n";
  output << indent << "    \"reduced_hessian_product_count\": "
         << profile.reducedHessianProductCount << ",\n";
  output << indent << "    \"reduced_objective_count\": "
         << profile.reducedObjectiveCount << ",\n";
  output << indent << "    \"reduced_gradient_build_count\": "
         << profile.reducedGradientBuildCount << ",\n";
  output << indent << "    \"reduced_preconditioner_count\": "
         << profile.reducedPreconditionerCount << ",\n";
  output << indent << "    \"reduced_projection_count\": "
         << profile.reducedProjectionCount << ",\n";
  output << indent << "    \"reduced_retraction_count\": "
         << profile.reducedRetractionCount << ",\n";
  output << indent << "    \"reduced_step_norm_count\": "
         << profile.reducedStepNormCount << ",\n";
  output << indent << "    \"reduced_trust_region_scaling_count\": "
         << profile.reducedTrustRegionScalingCount << ",\n";
  output << indent << "    \"reduced_translation_response_count\": "
         << profile.reducedTranslationResponseCount << ",\n";
  output << indent << "    \"reduced_rotation_product_count\": "
         << profile.reducedRotationProductCount << ",\n";
  output << indent << "    \"reduced_curvature_cauchy_candidate_count\": "
         << profile.reducedCurvatureCauchyCandidateCount << ",\n";
  output << indent << "    \"reduced_curvature_cauchy_accepted_count\": "
         << profile.reducedCurvatureCauchyAcceptedCount << ",\n";
  output << indent
         << "    \"reduced_curvature_cauchy_fallback_candidate_count\": "
         << profile.reducedCurvatureCauchyFallbackCandidateCount << ",\n";
  output << indent
         << "    \"reduced_curvature_cauchy_fallback_accepted_count\": "
         << profile.reducedCurvatureCauchyFallbackAcceptedCount << ",\n";
  output << indent << "    \"reduced_candidate_projection_skip_count\": "
         << profile.reducedCandidateProjectionSkipCount << ",\n";
  output << indent << "    \"explicit_translation_recovery_count\": "
         << profile.explicitTranslationRecoveryCount << ",\n";
  output << indent << "    \"candidate_evaluation_cache_request_count\": "
         << profile.candidateEvaluationCacheRequestCount << ",\n";
  output << indent << "    \"candidate_evaluation_cache_hit_count\": "
         << profile.candidateEvaluationCacheHitCount << ",\n";
  output << indent << "    \"candidate_evaluation_cache_miss_count\": "
         << profile.candidateEvaluationCacheMissCount << "\n";
  output << indent << "  },\n";
  output << indent << "  \"reduced_portfolio\": {\n";
  output << indent << "    \"selection_count\": "
         << profile.reducedPortfolioSelectionCount << ",\n";
  output << indent << "    \"candidate_counts\": ";
  writeReducedPortfolioCountObject(
      output, profile.reducedPortfolioCandidateCounts, indent + "    ");
  output << ",\n";
  output << indent << "    \"winner_counts\": ";
  writeReducedPortfolioCountObject(output, profile.reducedPortfolioWinnerCounts,
                                   indent + "    ");
  output << ",\n";
  output << indent << "    \"winner_cost_margin_count\": "
         << profile.reducedPortfolioWinnerCostMarginCount << ",\n";
  output << indent << "    \"winner_cost_margin_sum\": "
         << profile.reducedPortfolioWinnerCostMarginSum << ",\n";
  output << indent << "    \"winner_cost_margin_min\": "
         << (std::isfinite(profile.reducedPortfolioWinnerCostMarginMin)
                 ? profile.reducedPortfolioWinnerCostMarginMin
                 : 0.0)
         << "\n";
  output << indent << "  },\n";
  output << indent << "  \"reduced_adaptive_portfolio\": {\n";
  output << indent << "    \"full_selection_count\": "
         << profile.reducedAdaptivePortfolioFullSelectionCount << ",\n";
  output << indent << "    \"fast_path_count\": "
         << profile.reducedAdaptivePortfolioFastPathCount << ",\n";
  output << indent << "    \"certified_fast_path_count\": "
         << profile.reducedAdaptivePortfolioCertifiedFastPathCount << ",\n";
  output << indent << "    \"fallback_count\": "
         << profile.reducedAdaptivePortfolioFallbackCount << "\n";
  output << indent << "  }\n";
  output << indent << "}";
}

void writeOptimizerProfile(
    const std::string &path,
    const ManualDpgoMmOptimizerProfile &profile,
    const std::vector<ManualDpgoMmAgentProfileEntry> &agentProfiles,
    const ManualDpgoMmOptions &options,
    const std::string &datasetPath) {
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write optimizer profile: " + path);
  }

  std::vector<std::pair<std::string, double>> hotspots = {
      {"local_optimization_wall_sec", profile.localOptimizationWallSec},
      {"reduced_solver_sec", profile.reducedSolverSec},
      {"reduced_translation_recovery_sec",
       profile.reducedTranslationRecoverySec},
      {"reduced_translation_factorization_sec",
       profile.reducedTranslationFactorizationSec},
      {"reduced_translation_cache_lookup_sec",
       profile.reducedTranslationCacheLookupSec},
      {"reduced_hessian_product_sec", profile.reducedHessianProductSec},
      {"reduced_objective_sec", profile.reducedObjectiveSec},
      {"reduced_gradient_build_sec", profile.reducedGradientBuildSec},
      {"reduced_preconditioner_sec", profile.reducedPreconditionerSec},
      {"reduced_projection_sec", profile.reducedProjectionSec},
      {"reduced_retraction_sec", profile.reducedRetractionSec},
      {"reduced_step_norm_sec", profile.reducedStepNormSec},
      {"reduced_trust_region_scaling_sec",
       profile.reducedTrustRegionScalingSec},
      {"reduced_translation_response_sec",
       profile.reducedTranslationResponseSec},
      {"reduced_rotation_product_sec", profile.reducedRotationProductSec},
      {"lazy_candidate_gradient_evaluation_sec",
       profile.lazyCandidateGradientEvaluationSec},
      {"solver_start_gradient_evaluation_sec",
       profile.solverStartGradientEvaluationSec},
      {"candidate_construction_sec", profile.candidateConstructionSec},
      {"local_model_snapshot_sec", profile.localModelSnapshotSec},
      {"local_model_switch_sec", profile.localModelSwitchSec},
      {"local_surrogate_preparation_sec",
       profile.localSurrogatePreparationSec},
      {"local_candidate_selection_sec", profile.localCandidateSelectionSec},
      {"local_amm_outer_sec", profile.localAmmOuterSec},
      {"local_finalization_sec", profile.localFinalizationSec},
      {"dpgo_simple_linear_rows_sec", profile.dpgoSimpleLinearRowsSec},
      {"dpgo_simple_linear_multiply_sec",
       profile.dpgoSimpleLinearMultiplySec},
      {"dpgo_simple_linear_assembly_sec",
       profile.dpgoSimpleLinearAssemblySec},
      {"dpgo_simple_cache_build_sec",
       profile.dpgoSimpleCacheBuildSec},
      {"dpgo_simple_proximal_start_sec",
       profile.dpgoSimpleProximalStartSec},
      {"dpgo_simple_proximal_rows_sec",
       profile.dpgoSimpleProximalRowsSec},
      {"dpgo_simple_proximal_rotation_signal_sec",
       profile.dpgoSimpleProximalRotationSignalSec},
      {"dpgo_simple_proximal_gradient_sec",
       profile.dpgoSimpleProximalGradientSec},
      {"dpgo_simple_proximal_projection_sec",
       profile.dpgoSimpleProximalProjectionSec},
      {"dpgo_simple_proximal_translation_sec",
       profile.dpgoSimpleProximalTranslationSec},
      {"dpgo_simple_proximal_assembly_sec",
       profile.dpgoSimpleProximalAssemblySec},
      {"reduced_candidate_evaluation_sec",
       profile.reducedCandidateEvaluationSec},
      {"global_evaluation_wall_sec", profile.globalEvaluationWallSec},
      {"post_exchange_wall_sec", profile.postExchangeWallSec},
      {"initialization_exchange_sec", profile.initializationExchangeSec},
      {"output_write_sec", profile.outputWriteSec},
  };
  std::stable_sort(hotspots.begin(), hotspots.end(),
                   [](const auto &lhs, const auto &rhs) {
                     return lhs.second > rhs.second;
                   });

  output << std::setprecision(20);
  output << "{\n";
  output << "  \"version\": 1,\n";
  output << "  \"dataset\": \"" << jsonEscape(datasetPath) << "\",\n";
  output << "  \"scheme\": \"" << profileSchemeName(options.scheme)
         << "\",\n";
  output << "  \"local_solver\": \""
         << profileLocalSolverName(options.localSolver) << "\",\n";
  output << "  \"reduced_rotation_preconditioner\": \""
         << profileReducedRotationPreconditionerName(
                options.reducedRotationPreconditioner)
         << "\",\n";
  output << "  \"max_iterations\": " << options.maxIterations << ",\n";
  output << "  \"parallel_local_solves\": "
         << (options.parallelLocalSolves ? "true" : "false") << ",\n";
  output << "  \"clock\": \"std::chrono::steady_clock\",\n";
  output << "  \"profile_optimizer\": "
         << (options.profileOptimizer ? "true" : "false") << ",\n";
  output << "  \"candidate_evaluation_cache\": "
         << (options.candidateEvaluationCache ? "true" : "false") << ",\n";
  output << "  \"candidate_object_reuse\": "
         << (options.candidateObjectReuse ? "true" : "false") << ",\n";
  output << "  \"reduced_adaptive_portfolio_certified_fast_path\": "
         << (options.reducedAdaptivePortfolioCertifiedFastPath ? "true"
                                                               : "false")
         << ",\n";
  output << "  \"reduced_rotation_direct_objective\": "
         << (options.reducedRotationDirectObjective ? "true" : "false")
         << ",\n";
  output << "  \"fused_candidate_evaluation\": "
         << (options.fusedCandidateEvaluation ? "true" : "false")
         << ",\n";
  output << "  \"lazy_candidate_gradient_evaluation\": "
         << (options.lazyCandidateGradientEvaluation ? "true" : "false")
         << ",\n";
  output << "  \"lazy_solver_start_gradient_evaluation\": "
         << (options.lazySolverStartGradientEvaluation ? "true" : "false")
         << ",\n";
  output << "  \"lazy_surrogate_candidate_gradient_evaluation\": "
         << (options.lazySurrogateCandidateGradientEvaluation ? "true"
                                                              : "false")
         << ",\n";
  output << "  \"reduced_rotation_curvature_cauchy_candidate\": "
         << (options.reducedRotationCurvatureCauchyCandidate ? "true"
                                                             : "false")
         << ",\n";
  output << "  \"reduced_rotation_curvature_fallback_candidate\": "
         << (options.reducedRotationCurvatureFallbackCandidate ? "true"
                                                               : "false")
         << ",\n";
  output << "  \"reduced_rotation_gradient_boundary_candidate\": "
         << (options.reducedRotationGradientBoundaryCandidate ? "true"
                                                              : "false")
         << ",\n";
  output << "  \"reduced_rotation_surrogate_tcg_accept\": "
         << (options.reducedRotationSurrogateTcgAccept ? "true" : "false")
         << ",\n";
  output << "  \"amm_prox_reset_skip_refined_solve\": "
         << (options.ammProxResetSkipRefinedSolve ? "true" : "false")
         << ",\n";
  output << "  \"amm_dpgo_proximal_fallback_only\": "
         << (options.ammDpgoProximalFallbackOnly ? "true" : "false")
         << ",\n";
  output << "  \"amm_lazy_plain_after_certificate\": "
         << (options.ammLazyPlainAfterCertificate ? "true" : "false")
         << ",\n";
  output << "  \"amm_surrogate_first_exact_evaluation\": "
         << (options.ammSurrogateFirstExactEvaluation ? "true" : "false")
         << ",\n";
  output << "  \"reduced_rotation_skip_redundant_candidate_projection\": "
         << (options.reducedRotationSkipRedundantCandidateProjection ? "true"
                                                                     : "false")
         << ",\n";
  output << "  \"reduced_rotation_packed_cholesky_project\": "
         << (envFlagEnabled("DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT",
                            false)
                 ? "true"
                 : "false")
         << ",\n";
  output << "  \"reduced_rotation_packed_tcg\": "
         << (envFlagEnabled("DRAN_REDUCED_ROTATION_PACKED_TCG", false)
                 ? "true"
                 : "false")
         << ",\n";
  output << "  \"reduced_rotation_packed_riemannian_hvp\": "
         << (envFlagEnabled("DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP",
                            false)
                 ? "true"
                 : "false")
         << ",\n";
  output << "  \"reduced_rotation_compact_hvp\": "
         << (envFlagEnabled("DRAN_REDUCED_ROTATION_COMPACT_HVP", false)
                 ? "true"
                 : "false")
         << ",\n";
  output << "  \"reduced_rotation_precomputed_response_hvp\": "
         << (envFlagEnabled("DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP",
                            false)
                 ? "true"
                 : "false")
         << ",\n";
  output << "  \"manual_full_portfolio\": "
         << (options.manualFullPortfolio ? "true" : "false") << ",\n";
  output << "  \"seconds\": {\n";
  output << "    \"total_wall_sec\": " << profile.totalWallSec << ",\n";
  output << "    \"prepare_problem_sec\": " << profile.prepareProblemSec
         << ",\n";
  output << "    \"initialization_exchange_sec\": "
         << profile.initializationExchangeSec << ",\n";
  output << "    \"distributed_initialization_local_model_sec\": "
         << profile.distributedInitializationLocalModelSec << ",\n";
  output << "    \"distributed_initialization_local_solve_sec\": "
         << profile.distributedInitializationLocalSolveSec << ",\n";
  output << "    \"distributed_initialization_exchange_sec\": "
         << profile.distributedInitializationExchangeSec << ",\n";
  output << "    \"initial_local_model_sec\": "
         << profile.initialLocalModelSec << ",\n";
  output << "    \"initial_global_evaluation_sec\": "
         << profile.initialGlobalEvaluationSec << ",\n";
  output << "    \"iteration_local_phase_wall_sec\": "
         << profile.iterationLocalPhaseWallSec << ",\n";
  output << "    \"local_model_update_wall_sec\": "
         << profile.localModelUpdateWallSec << ",\n";
  output << "    \"local_optimization_wall_sec\": "
         << profile.localOptimizationWallSec << ",\n";
  output << "    \"local_gradient_fresh_exchange_sec\": "
         << profile.localGradientFreshExchangeSec << ",\n";
  output << "    \"local_gradient_correction_wall_sec\": "
         << profile.localGradientCorrectionWallSec << ",\n";
  output << "    \"global_evaluation_wall_sec\": "
         << profile.globalEvaluationWallSec << ",\n";
  output << "    \"post_exchange_wall_sec\": "
         << profile.postExchangeWallSec << ",\n";
  output << "    \"output_write_sec\": " << profile.outputWriteSec << ",\n";
  output << "    \"agent_optimize_local_model_sec\": "
         << profile.agentOptimizeLocalModelSec << ",\n";
  output << "    \"reduced_solver_sec\": " << profile.reducedSolverSec
         << ",\n";
  output << "    \"reduced_candidate_evaluation_sec\": "
         << profile.reducedCandidateEvaluationSec << ",\n";
  output << "    \"lazy_candidate_gradient_evaluation_sec\": "
         << profile.lazyCandidateGradientEvaluationSec << ",\n";
  output << "    \"solver_start_gradient_evaluation_sec\": "
         << profile.solverStartGradientEvaluationSec << ",\n";
  output << "    \"candidate_construction_sec\": "
         << profile.candidateConstructionSec << ",\n";
  output << "    \"local_model_snapshot_sec\": "
         << profile.localModelSnapshotSec << ",\n";
  output << "    \"local_model_switch_sec\": " << profile.localModelSwitchSec
         << ",\n";
  output << "    \"local_surrogate_preparation_sec\": "
         << profile.localSurrogatePreparationSec << ",\n";
  output << "    \"local_candidate_selection_sec\": "
         << profile.localCandidateSelectionSec << ",\n";
  output << "    \"local_amm_outer_sec\": " << profile.localAmmOuterSec
         << ",\n";
  output << "    \"local_finalization_sec\": "
         << profile.localFinalizationSec << ",\n";
  output << "    \"dpgo_simple_linear_rows_sec\": "
         << profile.dpgoSimpleLinearRowsSec << ",\n";
  output << "    \"dpgo_simple_linear_multiply_sec\": "
         << profile.dpgoSimpleLinearMultiplySec << ",\n";
  output << "    \"dpgo_simple_linear_assembly_sec\": "
         << profile.dpgoSimpleLinearAssemblySec << ",\n";
  output << "    \"dpgo_simple_cache_build_sec\": "
         << profile.dpgoSimpleCacheBuildSec << ",\n";
  output << "    \"dpgo_simple_proximal_start_sec\": "
         << profile.dpgoSimpleProximalStartSec << ",\n";
  output << "    \"dpgo_simple_proximal_rows_sec\": "
         << profile.dpgoSimpleProximalRowsSec << ",\n";
  output << "    \"dpgo_simple_proximal_rotation_signal_sec\": "
         << profile.dpgoSimpleProximalRotationSignalSec << ",\n";
  output << "    \"dpgo_simple_proximal_gradient_sec\": "
         << profile.dpgoSimpleProximalGradientSec << ",\n";
  output << "    \"dpgo_simple_proximal_projection_sec\": "
         << profile.dpgoSimpleProximalProjectionSec << ",\n";
  output << "    \"dpgo_simple_proximal_translation_sec\": "
         << profile.dpgoSimpleProximalTranslationSec << ",\n";
  output << "    \"dpgo_simple_proximal_assembly_sec\": "
         << profile.dpgoSimpleProximalAssemblySec << ",\n";
  output << "    \"reduced_translation_recovery_sec\": "
         << profile.reducedTranslationRecoverySec << ",\n";
  output << "    \"reduced_translation_factorization_sec\": "
         << profile.reducedTranslationFactorizationSec << ",\n";
  output << "    \"reduced_translation_cache_lookup_sec\": "
         << profile.reducedTranslationCacheLookupSec << ",\n";
  output << "    \"reduced_hessian_product_sec\": "
         << profile.reducedHessianProductSec << ",\n";
  output << "    \"reduced_objective_sec\": "
         << profile.reducedObjectiveSec << ",\n";
  output << "    \"reduced_gradient_build_sec\": "
         << profile.reducedGradientBuildSec << ",\n";
  output << "    \"reduced_preconditioner_sec\": "
         << profile.reducedPreconditionerSec << ",\n";
  output << "    \"reduced_projection_sec\": "
         << profile.reducedProjectionSec << ",\n";
  output << "    \"reduced_retraction_sec\": "
         << profile.reducedRetractionSec << ",\n";
  output << "    \"reduced_step_norm_sec\": "
         << profile.reducedStepNormSec << ",\n";
  output << "    \"reduced_trust_region_scaling_sec\": "
         << profile.reducedTrustRegionScalingSec << ",\n";
  output << "    \"reduced_translation_response_sec\": "
         << profile.reducedTranslationResponseSec << ",\n";
  output << "    \"reduced_rotation_product_sec\": "
         << profile.reducedRotationProductSec << ",\n";
  output << "    \"explicit_translation_recovery_sec\": "
         << profile.explicitTranslationRecoverySec << "\n";
  output << "  },\n";
  output << "  \"counts\": {\n";
  output << "    \"iteration_count\": " << profile.iterationCount << ",\n";
  output << "    \"local_model_update_count\": "
         << profile.localModelUpdateCount << ",\n";
  output << "    \"local_optimization_round_count\": "
         << profile.localOptimizationRoundCount << ",\n";
  output << "    \"local_gradient_fresh_exchange_count\": "
         << profile.localGradientFreshExchangeCount << ",\n";
  output << "    \"local_gradient_correction_round_count\": "
         << profile.localGradientCorrectionRoundCount << ",\n";
  output << "    \"global_evaluation_count\": "
         << profile.globalEvaluationCount << ",\n";
  output << "    \"post_exchange_count\": " << profile.postExchangeCount
         << ",\n";
  output << "    \"agent_optimize_local_model_count\": "
         << profile.agentOptimizeLocalModelCount << ",\n";
  output << "    \"reduced_solve_count\": " << profile.reducedSolveCount
         << ",\n";
  output << "    \"reduced_candidate_evaluation_count\": "
         << profile.reducedCandidateEvaluationCount << ",\n";
  output << "    \"lazy_candidate_gradient_evaluation_count\": "
         << profile.lazyCandidateGradientEvaluationCount << ",\n";
  output << "    \"solver_start_gradient_evaluation_count\": "
         << profile.solverStartGradientEvaluationCount << ",\n";
  output << "    \"candidate_object_construction_count\": "
         << profile.candidateObjectConstructionCount << ",\n";
  output << "    \"candidate_object_reuse_count\": "
         << profile.candidateObjectReuseCount << ",\n";
  output << "    \"local_model_snapshot_count\": "
         << profile.localModelSnapshotCount << ",\n";
  output << "    \"local_model_switch_count\": "
         << profile.localModelSwitchCount << ",\n";
  output << "    \"local_surrogate_preparation_count\": "
         << profile.localSurrogatePreparationCount << ",\n";
  output << "    \"local_candidate_selection_count\": "
         << profile.localCandidateSelectionCount << ",\n";
  output << "    \"local_amm_outer_count\": " << profile.localAmmOuterCount
         << ",\n";
  output << "    \"local_finalization_count\": "
         << profile.localFinalizationCount << ",\n";
  output << "    \"dpgo_simple_linear_count\": "
         << profile.dpgoSimpleLinearCount << ",\n";
  output << "    \"dpgo_simple_cache_build_count\": "
         << profile.dpgoSimpleCacheBuildCount << ",\n";
  output << "    \"dpgo_simple_proximal_start_count\": "
         << profile.dpgoSimpleProximalStartCount << ",\n";
  output << "    \"reduced_translation_recovery_count\": "
         << profile.reducedTranslationRecoveryCount << ",\n";
  output << "    \"reduced_translation_factorization_count\": "
         << profile.reducedTranslationFactorizationCount << ",\n";
  output << "    \"reduced_translation_cache_lookup_count\": "
         << profile.reducedTranslationCacheLookupCount << ",\n";
  output << "    \"reduced_hessian_product_count\": "
         << profile.reducedHessianProductCount << ",\n";
  output << "    \"reduced_objective_count\": "
         << profile.reducedObjectiveCount << ",\n";
  output << "    \"reduced_gradient_build_count\": "
         << profile.reducedGradientBuildCount << ",\n";
  output << "    \"reduced_preconditioner_count\": "
         << profile.reducedPreconditionerCount << ",\n";
  output << "    \"reduced_projection_count\": "
         << profile.reducedProjectionCount << ",\n";
  output << "    \"reduced_retraction_count\": "
         << profile.reducedRetractionCount << ",\n";
  output << "    \"reduced_step_norm_count\": "
         << profile.reducedStepNormCount << ",\n";
  output << "    \"reduced_trust_region_scaling_count\": "
         << profile.reducedTrustRegionScalingCount << ",\n";
  output << "    \"reduced_translation_response_count\": "
         << profile.reducedTranslationResponseCount << ",\n";
  output << "    \"reduced_rotation_product_count\": "
         << profile.reducedRotationProductCount << ",\n";
  output << "    \"reduced_curvature_cauchy_candidate_count\": "
         << profile.reducedCurvatureCauchyCandidateCount << ",\n";
  output << "    \"reduced_curvature_cauchy_accepted_count\": "
         << profile.reducedCurvatureCauchyAcceptedCount << ",\n";
  output << "    \"reduced_curvature_cauchy_fallback_candidate_count\": "
         << profile.reducedCurvatureCauchyFallbackCandidateCount << ",\n";
  output << "    \"reduced_curvature_cauchy_fallback_accepted_count\": "
         << profile.reducedCurvatureCauchyFallbackAcceptedCount << ",\n";
  output << "    \"reduced_candidate_projection_skip_count\": "
         << profile.reducedCandidateProjectionSkipCount << ",\n";
  output << "    \"explicit_translation_recovery_count\": "
         << profile.explicitTranslationRecoveryCount << ",\n";
  output << "    \"candidate_evaluation_cache_request_count\": "
         << profile.candidateEvaluationCacheRequestCount << ",\n";
  output << "    \"candidate_evaluation_cache_hit_count\": "
         << profile.candidateEvaluationCacheHitCount << ",\n";
  output << "    \"candidate_evaluation_cache_miss_count\": "
         << profile.candidateEvaluationCacheMissCount << "\n";
  output << "  },\n";
  output << "  \"reduced_portfolio\": {\n";
  output << "    \"selection_count\": "
         << profile.reducedPortfolioSelectionCount << ",\n";
  output << "    \"candidate_counts\": ";
  writeReducedPortfolioCountObject(
      output, profile.reducedPortfolioCandidateCounts, "    ");
  output << ",\n";
  output << "    \"winner_counts\": ";
  writeReducedPortfolioCountObject(output, profile.reducedPortfolioWinnerCounts,
                                   "    ");
  output << ",\n";
  output << "    \"winner_cost_margin_count\": "
         << profile.reducedPortfolioWinnerCostMarginCount << ",\n";
  output << "    \"winner_cost_margin_sum\": "
         << profile.reducedPortfolioWinnerCostMarginSum << ",\n";
  output << "    \"winner_cost_margin_min\": "
         << (std::isfinite(profile.reducedPortfolioWinnerCostMarginMin)
                 ? profile.reducedPortfolioWinnerCostMarginMin
                 : 0.0)
         << "\n";
  output << "  },\n";
  output << "  \"reduced_adaptive_portfolio\": {\n";
  output << "    \"full_selection_count\": "
         << profile.reducedAdaptivePortfolioFullSelectionCount << ",\n";
  output << "    \"fast_path_count\": "
         << profile.reducedAdaptivePortfolioFastPathCount << ",\n";
  output << "    \"certified_fast_path_count\": "
         << profile.reducedAdaptivePortfolioCertifiedFastPathCount << ",\n";
  output << "    \"fallback_count\": "
         << profile.reducedAdaptivePortfolioFallbackCount << "\n";
  output << "  },\n";
  output << "  \"agent_profiles\": [\n";
  for (std::size_t idx = 0; idx < agentProfiles.size(); ++idx) {
    writeAgentProfileEntry(output, agentProfiles[idx], "    ");
    output << (idx + 1 == agentProfiles.size() ? "\n" : ",\n");
  }
  output << "  ],\n";
  output << "  \"top_hotspots\": [\n";
  const std::size_t hotspotCount = std::min<std::size_t>(3, hotspots.size());
  for (std::size_t idx = 0; idx < hotspotCount; ++idx) {
    output << "    {\"name\": \"" << hotspots[idx].first
           << "\", \"seconds\": " << hotspots[idx].second << "}";
    output << (idx + 1 == hotspotCount ? "\n" : ",\n");
  }
  output << "  ],\n";
  output << "  \"notes\": [\n";
  output << "    \"Profile timings are observational only and are not used in optimizer decisions.\",\n";
  output << "    \"agent_optimize_local_model_sec is summed over agents; local_optimization_wall_sec is wall time for the parallel local-solve round.\",\n";
  output << "    \"reduced_translation_recovery_sec, reduced_translation_factorization_sec, and reduced_hessian_product_sec are measured inside ReducedRotationQuadraticOptimizer and are subsets of reduced_solver_sec.\",\n";
  output << "    \"explicit_translation_recovery_sec covers ManualDpgoMm explicit recovery calls outside the reduced optimizer.\",\n";
  output << "    \"agent_profiles contains summed per-robot local optimizer timings; in parallel local solves the slowest active robot, not the sum, controls local_optimization_wall_sec.\"\n";
  output << "  ]\n";
  output << "}\n";
}

void writeIterationSummary(
    const std::string &path,
    const std::vector<ManualDpgoMmIterationSummary> &summaries) {
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write iteration summary: " + path);
  }
  output << "iter,time,global_cost,gradient,comm_pose_count,iter_comm_mb,"
         << "cumulative_comm_pose_count,cumulative_comm_mb,"
         << "initialization_comm_pose_count,initialization_comm_mb,"
         << "outer_comm_pose_count,outer_comm_mb,"
         << "pose_iter_comm_mb,pose_cumulative_comm_mb,"
         << "local_gradient_scalar_count,"
         << "local_gradient_iter_scalar_comm_mb,"
         << "local_gradient_cumulative_scalar_comm_mb,"
         << "local_gradient_selected_step,"
         << "local_gradient_fresh_pose_count,"
         << "local_gradient_fresh_pose_comm_mb,"
         << "local_gradient_fresh_skipped_pose_count,"
         << "local_gradient_inner_round_count,"
         << "post_exchange_pose_count,"
         << "post_exchange_pose_comm_mb,"
         << "post_exchange_skipped_pose_count,"
         << "local_model_failures,local_optimization_failures,"
         << "local_accepted_iteration_count,"
         << "full_equiv_hybrid_warm_start_candidate_count,"
         << "full_equiv_hybrid_warm_start_accepted_count,"
         << "full_equiv_hybrid_warm_start_guard_rejected_count,"
         << "full_equiv_hybrid_schur_step_candidate_count,"
         << "full_equiv_hybrid_schur_step_accepted_count,"
         << "full_equiv_hybrid_schur_step_guard_rejected_count,"
         << "full_equiv_hybrid_local_portfolio_candidate_count,"
         << "full_equiv_hybrid_local_portfolio_selected_unsmoothed_count,"
         << "full_equiv_hybrid_local_portfolio_selected_schwarz_only_count,"
         << "full_equiv_hybrid_local_portfolio_selected_schwarz_feh_count,"
         << "full_equiv_hybrid_schwarz_smoothing_sweep_count,"
         << "full_equiv_hybrid_schwarz_smoothing_candidate_count,"
         << "full_equiv_hybrid_schwarz_smoothing_accepted_count,"
         << "full_equiv_hybrid_schwarz_smoothing_rejected_count,"
         << "full_equiv_hybrid_schwarz_smoothing_cost_decrease,"
         << "full_equiv_hybrid_schwarz_smoothing_time_sec,"
         << "full_equiv_hybrid_linear_pcg_iteration_count,"
         << "full_equiv_hybrid_linear_initial_residual,"
         << "full_equiv_hybrid_linear_final_residual,"
         << "full_equiv_hybrid_linear_full_residual,"
         << "full_equiv_hybrid_linear_solve_time_sec,"
         << "full_equiv_hybrid_step_trial_count,"
         << "full_equiv_hybrid_step_trial_accepted_count,"
         << "full_equiv_hybrid_step_trial_rejected_count,"
         << "full_equiv_hybrid_step_predicted_decrease_sum,"
         << "full_equiv_hybrid_step_actual_decrease_sum,"
         << "full_equiv_hybrid_step_rho_sum,"
         << "full_equiv_hybrid_step_rho_count,"
         << "full_equiv_hybrid_step_accepted_scale_sum,"
         << "full_equiv_hybrid_step_accepted_predicted_decrease_sum,"
         << "full_equiv_hybrid_step_accepted_actual_decrease_sum,"
         << "full_equiv_hybrid_step_accepted_rho_sum,"
         << "full_equiv_hybrid_step_accepted_rho_count,"
         << "full_equiv_hybrid_step_pareto_candidate_count,"
         << "full_equiv_hybrid_step_pareto_selected_count,"
         << "full_equiv_hybrid_step_pareto_selected_scale_sum,"
         << "full_equiv_hybrid_step_pareto_best_decrease_sum,"
         << "full_equiv_hybrid_step_pareto_selected_decrease_sum,"
         << "full_equiv_hybrid_step_pareto_selected_gradient_sum,"
         << "full_equiv_hybrid_step_pareto_gradient_eval_count,"
         << "full_equiv_hybrid_step_pareto_gradient_eval_time_sec,"
         << "full_equiv_hybrid_translation_recovery_step_trial_candidate_count,"
         << "full_equiv_hybrid_translation_recovery_step_trial_hard_accepted_count,"
         << "full_equiv_hybrid_translation_recovery_step_trial_selected_count,"
         << "full_equiv_hybrid_active_separator_candidate_count,"
         << "full_equiv_hybrid_active_separator_accepted_count,"
         << "full_equiv_hybrid_active_separator_rejected_count,"
         << "full_equiv_hybrid_active_separator_step_sum,"
         << "full_equiv_hybrid_active_separator_cost_decrease_sum,"
         << "full_equiv_hybrid_active_separator_lm_schur_candidate_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_accepted_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_guard_rejected_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_gradient_guard_rejected_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_solve_failure_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_fallback_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_boundary_col_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_private_col_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_backtracking_trial_count,"
         << "full_equiv_hybrid_active_separator_lm_schur_alpha_sum,"
         << "full_equiv_hybrid_active_separator_lm_schur_cost_decrease_sum,"
         << "full_equiv_hybrid_active_separator_lm_schur_gradient_change_sum,"
         << "full_equiv_hybrid_translation_recovery_polish_attempt_count,"
         << "full_equiv_hybrid_translation_recovery_polish_accepted_count,"
         << "full_equiv_hybrid_translation_recovery_polish_rejected_count,"
         << "full_equiv_hybrid_translation_recovery_polish_gradient_guard_rejected_count,"
         << "full_equiv_hybrid_translation_recovery_polish_backtracking_trigger_count,"
         << "full_equiv_hybrid_translation_recovery_polish_backtracking_trial_count,"
         << "full_equiv_hybrid_translation_recovery_polish_backtracking_accepted_count,"
         << "full_equiv_hybrid_translation_recovery_polish_backtracking_alpha_sum,"
         << "full_equiv_hybrid_translation_recovery_polish_merit_candidate_count,"
         << "full_equiv_hybrid_translation_recovery_polish_merit_selected_full_count,"
         << "full_equiv_hybrid_translation_recovery_polish_merit_selected_partial_count,"
         << "full_equiv_hybrid_translation_recovery_polish_merit_selected_alpha_sum,"
         << "full_equiv_hybrid_translation_recovery_polish_merit_best_decrease_sum,"
         << "full_equiv_hybrid_translation_recovery_polish_merit_selected_decrease_sum,"
         << "full_equiv_hybrid_translation_recovery_polish_cost_decrease_sum,"
         << "full_equiv_hybrid_translation_recovery_polish_gradient_change_sum,"
         << "full_equiv_hybrid_sparse_matvec_count,"
         << "full_equiv_hybrid_reduced_rotation_initial_guess_candidate_count,"
         << "full_equiv_hybrid_reduced_rotation_initial_guess_used_count,"
         << "full_equiv_hybrid_reduced_rotation_initial_guess_rejected_count,"
         << "full_equiv_hybrid_translation_recovery_initial_guess_candidate_count,"
         << "full_equiv_hybrid_translation_recovery_initial_guess_used_count,"
         << "full_equiv_hybrid_translation_recovery_initial_guess_rejected_count,"
         << "full_equiv_hybrid_translation_schur_preconditioner_application_count,"
         << "full_equiv_hybrid_translation_schur_preconditioner_factorization_count,"
         << "full_equiv_hybrid_translation_schur_preconditioner_fallback_count,"
         << "full_equiv_hybrid_local_chain_preconditioner_application_count,"
         << "full_equiv_hybrid_local_chain_preconditioner_factorization_count,"
         << "full_equiv_hybrid_local_chain_preconditioner_fallback_count,"
         << "full_equiv_hybrid_translation_block_preconditioner_application_count,"
         << "full_equiv_hybrid_translation_block_preconditioner_factorization_count,"
         << "full_equiv_hybrid_translation_block_preconditioner_fallback_count,"
         << "full_equiv_hybrid_translation_sparse_schur_preconditioner_application_count,"
         << "full_equiv_hybrid_translation_sparse_schur_preconditioner_factorization_count,"
         << "full_equiv_hybrid_translation_sparse_schur_preconditioner_fallback_count,"
         << "full_equiv_hybrid_translation_local_schur_preconditioner_application_count,"
         << "full_equiv_hybrid_translation_local_schur_preconditioner_factorization_count,"
         << "full_equiv_hybrid_translation_local_schur_preconditioner_fallback_count,"
         << "full_equiv_hybrid_translation_local_schur_preconditioner_active_pose_count,"
         << "full_equiv_hybrid_translation_local_schur_preconditioner_active_column_count,"
         << "full_equiv_hybrid_laplacian_deflation_preconditioner_application_count,"
         << "full_equiv_hybrid_laplacian_deflation_preconditioner_factorization_count,"
         << "full_equiv_hybrid_laplacian_deflation_preconditioner_fallback_count,"
         << "full_equiv_hybrid_laplacian_deflation_preconditioner_basis_dimension,"
         << "full_equiv_hybrid_reduced_rotation_preconditioner_application_count,"
         << "full_equiv_hybrid_reduced_rotation_preconditioner_factorization_count,"
         << "full_equiv_hybrid_rqn_used_count,"
         << "full_equiv_hybrid_rqn_accepted_pair_count,"
         << "full_equiv_hybrid_rqn_rejected_pair_count,"
         << "full_equiv_hybrid_rqn_memory_size,"
         << "full_equiv_hybrid_rqn_preconditioner_application_count,"
         << "adaptive_refinement_count,extrapolation_accepted_count,"
         << "extrapolation_rejected_count,g_extrapolation_accepted_count,"
         << "g_extrapolation_rejected_count,"
         << "coupled_extrapolation_accepted_count,"
         << "coupled_extrapolation_rejected_count,"
         << "anderson_accepted_count,anderson_rejected_count,"
         << "squarem_accepted_count,squarem_rejected_count,"
         << "global_extrapolation_accepted_count,"
         << "global_extrapolation_rejected_count,"
         << "global_anderson_accepted_count,"
         << "global_anderson_rejected_count,"
         << "local_gradient_accepted_count,"
         << "local_gradient_rejected_count,"
         << "local_gradient_coupled_direction_candidate_count,"
         << "local_gradient_coupled_direction_accepted_count,"
         << "local_gradient_coupled_direction_rejected_count,"
         << "local_gradient_coupled_direction_packet_count,"
         << "local_gradient_coupled_direction_skipped_packet_count,"
         << "local_gradient_coupled_direction_score_skipped_packet_count,"
         << "local_gradient_coupled_direction_byte_budget_skipped_packet_count,"
         << "local_gradient_coupled_direction_scalar_count,"
         << "local_gradient_coupled_direction_scalar_comm_mb,"
         << "local_gradient_compact_schur_candidate_count,"
         << "local_gradient_compact_schur_accepted_count,"
         << "local_gradient_compact_schur_guard_rejected_count,"
         << "local_gradient_compact_schur_gradient_guard_rejected_count,"
         << "local_gradient_compact_schur_solve_failure_count,"
         << "local_gradient_compact_schur_boundary_col_count,"
         << "local_gradient_compact_schur_private_col_count,"
         << "local_gradient_compact_schur_gradient_change_sum,"
         << "global_gradient_accepted_count,"
         << "global_gradient_rejected_count,"
         << "local_history_sync_count,"
         << "local_model_cost_before,local_model_cost_after,"
         << "local_model_gradient_before,local_model_gradient_after,"
         << "boundary_edge_cost_before,boundary_edge_cost_after,"
         << "separator_delta_norm,"
         << "boundary_proximal_candidate_count,"
         << "boundary_proximal_accepted_count,"
         << "edge_tight_quadratic_eval_count,"
         << "edge_tight_quadratic_surrogate_cost_sum,"
         << "edge_tight_quadratic_true_cost_sum,"
         << "edge_tight_quadratic_majorization_gap_min,"
         << "amm_accelerated_accepted_count,amm_restart_count,"
         << "amm_hard_restart_count,amm_soft_restart_count,"
         << "amm_phi_fallback_count,amm_local_merit_rejected_count,"
         << "amm_proximal_start_count,amm_skipped_count,"
         << "amm_trace_count,amm_refined_count,amm_prox_reset_count,"
         << "amm_restart_used_xakh_count,"
         << "amm_restart_certificate_count,"
         << "amm_restart_certificate_passed_count,"
         << "amm_restart_certificate_failed_count,"
         << "amm_restart_certificate_min_margin,"
         << "amm_translation_recovery_attempt_count,"
         << "amm_translation_recovery_accepted_count,"
         << "amm_translation_recovery_rejected_count,"
         << "amm_mixed_surrogate_candidate_count,"
         << "amm_mixed_surrogate_true_local_accepted_count,"
         << "amm_mixed_surrogate_simple_selected_count,"
         << "amm_mixed_surrogate_true_local_selected_count,"
         << "amm_mixed_surrogate_extrapolated_selected_count,"
         << "amm_mixed_surrogate_other_selected_count,"
         << "amm_mixed_surrogate_simple_skipped_count,"
         << "amm_mixed_surrogate_simple_forced_refresh_count,"
         << "amm_mean_gamma,"
         << "amm_mean_Gkh_initial,amm_mean_minG,"
         << "amm_mean_Gk_after_accelerated,"
         << "amm_mean_Gkh_after_restart_check,amm_mean_final_Gk,"
         << "amm_mean_phi_lhs,amm_mean_phi_rhs\n";
  for (const auto &row : summaries) {
    output << row.iter << "," << std::setprecision(16) << row.time << ","
           << row.globalCost << "," << row.gradient << ","
           << row.commPoseCount << "," << row.iterCommMb << ","
           << row.cumulativeCommPoseCount << "," << row.cumulativeCommMb
           << "," << row.initializationCommPoseCount << ","
           << row.initializationCommMb << "," << row.outerCommPoseCount
           << "," << row.outerCommMb
           << "," << row.poseIterCommMb << ","
           << row.poseCumulativeCommMb << ","
           << row.localGradientScalarCount << ","
           << row.localGradientIterScalarCommMb << ","
           << row.localGradientCumulativeScalarCommMb << ","
           << row.localGradientSelectedStep << ","
           << row.localGradientFreshPoseCount << ","
           << row.localGradientFreshPoseCommMb << ","
           << row.localGradientFreshSkippedPoseCount << ","
           << row.localGradientInnerRoundCount
           << "," << row.postExchangePoseCount << ","
           << row.postExchangePoseCommMb << ","
           << row.postExchangeSkippedPoseCount
           << "," << row.localModelFailures << ","
           << row.localOptimizationFailures << ","
           << row.localAcceptedIterationCount << ","
           << row.fullEquivHybridWarmStartCandidateCount << ","
           << row.fullEquivHybridWarmStartAcceptedCount << ","
           << row.fullEquivHybridWarmStartGuardRejectedCount << ","
           << row.fullEquivHybridSchurStepCandidateCount << ","
           << row.fullEquivHybridSchurStepAcceptedCount << ","
           << row.fullEquivHybridSchurStepGuardRejectedCount << ","
           << row.fullEquivHybridLocalPortfolioCandidateCount << ","
           << row.fullEquivHybridLocalPortfolioSelectedUnsmoothedCount << ","
           << row.fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount << ","
           << row.fullEquivHybridLocalPortfolioSelectedSchwarzFehCount << ","
           << row.fullEquivHybridSchwarzSmoothingSweepCount << ","
           << row.fullEquivHybridSchwarzSmoothingCandidateCount << ","
           << row.fullEquivHybridSchwarzSmoothingAcceptedCount << ","
           << row.fullEquivHybridSchwarzSmoothingRejectedCount << ","
           << row.fullEquivHybridSchwarzSmoothingCostDecrease << ","
           << row.fullEquivHybridSchwarzSmoothingTimeSec << ","
           << row.fullEquivHybridLinearPcgIterationCount << ","
           << row.fullEquivHybridLinearInitialResidual << ","
           << row.fullEquivHybridLinearFinalResidual << ","
           << row.fullEquivHybridLinearFullResidual << ","
           << row.fullEquivHybridLinearSolveTimeSec << ","
           << row.fullEquivHybridStepTrialCount << ","
           << row.fullEquivHybridStepTrialAcceptedCount << ","
           << row.fullEquivHybridStepTrialRejectedCount << ","
           << row.fullEquivHybridStepPredictedDecreaseSum << ","
           << row.fullEquivHybridStepActualDecreaseSum << ","
           << row.fullEquivHybridStepRhoSum << ","
           << row.fullEquivHybridStepRhoCount << ","
           << row.fullEquivHybridStepAcceptedScaleSum << ","
           << row.fullEquivHybridStepAcceptedPredictedDecreaseSum << ","
           << row.fullEquivHybridStepAcceptedActualDecreaseSum << ","
           << row.fullEquivHybridStepAcceptedRhoSum << ","
           << row.fullEquivHybridStepAcceptedRhoCount << ","
           << row.fullEquivHybridStepParetoCandidateCount << ","
           << row.fullEquivHybridStepParetoSelectedCount << ","
           << row.fullEquivHybridStepParetoSelectedScaleSum << ","
           << row.fullEquivHybridStepParetoBestDecreaseSum << ","
           << row.fullEquivHybridStepParetoSelectedDecreaseSum << ","
           << row.fullEquivHybridStepParetoSelectedGradientSum << ","
           << row.fullEquivHybridStepParetoGradientEvalCount << ","
           << row.fullEquivHybridStepParetoGradientEvalTimeSec << ","
           << row.fullEquivHybridTranslationRecoveryStepTrialCandidateCount
           << ","
           << row.fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount
           << ","
           << row.fullEquivHybridTranslationRecoveryStepTrialSelectedCount
           << ","
           << row.fullEquivHybridActiveSeparatorCandidateCount << ","
           << row.fullEquivHybridActiveSeparatorAcceptedCount << ","
           << row.fullEquivHybridActiveSeparatorRejectedCount << ","
           << row.fullEquivHybridActiveSeparatorStepSum << ","
           << row.fullEquivHybridActiveSeparatorCostDecreaseSum << ","
           << row.fullEquivHybridActiveSeparatorLmSchurCandidateCount << ","
           << row.fullEquivHybridActiveSeparatorLmSchurAcceptedCount << ","
           << row.fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount
           << ","
           << row.fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount
           << ","
           << row.fullEquivHybridActiveSeparatorLmSchurSolveFailureCount
           << ","
           << row.fullEquivHybridActiveSeparatorLmSchurFallbackCount << ","
           << row.fullEquivHybridActiveSeparatorLmSchurBoundaryColCount
           << ","
           << row.fullEquivHybridActiveSeparatorLmSchurPrivateColCount
           << ","
           << row.fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount
           << ","
           << row.fullEquivHybridActiveSeparatorLmSchurAlphaSum << ","
           << row.fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum << ","
           << row.fullEquivHybridActiveSeparatorLmSchurGradientChangeSum << ","
           << row.fullEquivHybridTranslationRecoveryPolishAttemptCount
           << ","
           << row.fullEquivHybridTranslationRecoveryPolishAcceptedCount
           << ","
           << row.fullEquivHybridTranslationRecoveryPolishRejectedCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishMeritCandidateCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum
           << ","
           << row.fullEquivHybridTranslationRecoveryPolishCostDecreaseSum
           << ","
           << row.fullEquivHybridTranslationRecoveryPolishGradientChangeSum
           << ","
           << row.fullEquivHybridSparseMatrixVectorProductCount << ","
           << row.fullEquivHybridReducedRotationInitialGuessCandidateCount
           << ","
           << row.fullEquivHybridReducedRotationInitialGuessUsedCount
           << ","
           << row.fullEquivHybridReducedRotationInitialGuessRejectedCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryInitialGuessCandidateCount
           << ","
           << row.fullEquivHybridTranslationRecoveryInitialGuessUsedCount
           << ","
           << row
                  .fullEquivHybridTranslationRecoveryInitialGuessRejectedCount
           << ","
           << row
                  .fullEquivHybridTranslationSchurPreconditionerApplicationCount
           << ","
           << row
                  .fullEquivHybridTranslationSchurPreconditionerFactorizationCount
           << ","
           << row.fullEquivHybridTranslationSchurPreconditionerFallbackCount
           << ","
           << row.fullEquivHybridLocalChainPreconditionerApplicationCount
           << ","
           << row.fullEquivHybridLocalChainPreconditionerFactorizationCount
           << ","
           << row.fullEquivHybridLocalChainPreconditionerFallbackCount
           << ","
           << row
                  .fullEquivHybridTranslationBlockPreconditionerApplicationCount
           << ","
           << row
                  .fullEquivHybridTranslationBlockPreconditionerFactorizationCount
           << ","
           << row.fullEquivHybridTranslationBlockPreconditionerFallbackCount
           << ","
           << row
                  .fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount
           << ","
           << row
                  .fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount
           << ","
           << row
                  .fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount
           << ","
           << row
                  .fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount
           << ","
           << row
                  .fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount
           << ","
           << row
                  .fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount
           << ","
           << row
                  .fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount
           << ","
           << row
                  .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount
           << ","
           << row
                  .fullEquivHybridLaplacianDeflationPreconditionerApplicationCount
           << ","
           << row
                  .fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount
           << ","
           << row.fullEquivHybridLaplacianDeflationPreconditionerFallbackCount
           << ","
           << row.fullEquivHybridLaplacianDeflationPreconditionerBasisDimension
           << ","
           << row
                  .fullEquivHybridReducedRotationPreconditionerApplicationCount
           << ","
           << row
                  .fullEquivHybridReducedRotationPreconditionerFactorizationCount
           << ","
           << row.fullEquivHybridRqnUsedCount << ","
           << row.fullEquivHybridRqnAcceptedPairCount << ","
           << row.fullEquivHybridRqnRejectedPairCount << ","
           << row.fullEquivHybridRqnMemorySize << ","
           << row.fullEquivHybridRqnPreconditionerApplicationCount << ","
           << row.adaptiveRefinementCount << ","
           << row.extrapolationAcceptedCount << ","
           << row.extrapolationRejectedCount << ","
           << row.gExtrapolationAcceptedCount << ","
           << row.gExtrapolationRejectedCount << ","
           << row.coupledExtrapolationAcceptedCount << ","
           << row.coupledExtrapolationRejectedCount << ","
           << row.andersonAcceptedCount << ","
           << row.andersonRejectedCount << ","
           << row.squaremAcceptedCount << ","
           << row.squaremRejectedCount << ","
           << row.globalExtrapolationAcceptedCount << ","
           << row.globalExtrapolationRejectedCount << ","
           << row.globalAndersonAcceptedCount << ","
           << row.globalAndersonRejectedCount << ","
           << row.localGradientAcceptedCount << ","
           << row.localGradientRejectedCount << ","
           << row.localGradientCoupledDirectionCandidateCount << ","
           << row.localGradientCoupledDirectionAcceptedCount << ","
           << row.localGradientCoupledDirectionRejectedCount << ","
           << row.localGradientCoupledDirectionPacketCount << ","
           << row.localGradientCoupledDirectionSkippedPacketCount << ","
           << row.localGradientCoupledDirectionScoreSkippedPacketCount << ","
           << row.localGradientCoupledDirectionByteBudgetSkippedPacketCount
           << ","
           << row.localGradientCoupledDirectionScalarCount << ","
           << row.localGradientCoupledDirectionScalarCommMb << ","
           << row.localGradientCompactSchurCandidateCount << ","
           << row.localGradientCompactSchurAcceptedCount << ","
           << row.localGradientCompactSchurGuardRejectedCount << ","
           << row.localGradientCompactSchurGradientGuardRejectedCount << ","
           << row.localGradientCompactSchurSolveFailureCount << ","
           << row.localGradientCompactSchurBoundaryColCount << ","
           << row.localGradientCompactSchurPrivateColCount << ","
           << row.localGradientCompactSchurGradientChangeSum << ","
           << row.globalGradientAcceptedCount << ","
           << row.globalGradientRejectedCount << ","
           << row.localHistorySyncCount << ","
           << row.localModelCostBefore << "," << row.localModelCostAfter
           << "," << row.localModelGradientBefore << ","
           << row.localModelGradientAfter << ","
           << row.boundaryEdgeCostBefore << ","
           << row.boundaryEdgeCostAfter << ","
           << row.separatorDeltaNorm << ","
           << row.boundaryProximalCandidateCount << ","
           << row.boundaryProximalAcceptedCount << ","
           << row.edgeTightQuadraticEvalCount << ","
           << row.edgeTightQuadraticSurrogateCostSum << ","
           << row.edgeTightQuadraticTrueCostSum << ","
           << row.edgeTightQuadraticMajorizationGapMin << ","
           << row.ammAcceleratedAcceptedCount << ","
           << row.ammRestartCount << "," << row.ammHardRestartCount << ","
           << row.ammSoftRestartCount << "," << row.ammPhiFallbackCount
           << "," << row.ammLocalMeritRejectedCount << ","
           << row.ammProximalStartCount << "," << row.ammSkippedCount << ","
           << row.ammTraceCount << "," << row.ammRefinedCount << ","
           << row.ammProxResetCount << "," << row.ammRestartUsedXakhCount
           << "," << row.ammRestartCertificateCount << ","
           << row.ammRestartCertificatePassedCount << ","
           << row.ammRestartCertificateFailedCount << ","
           << row.ammRestartCertificateMinMargin
           << "," << row.ammTranslationRecoveryAttemptCount << ","
           << row.ammTranslationRecoveryAcceptedCount << ","
           << row.ammTranslationRecoveryRejectedCount << ","
           << row.ammMixedSurrogateCandidateCount << ","
           << row.ammMixedSurrogateTrueLocalAcceptedCount << ","
           << row.ammMixedSurrogateSimpleSelectedCount << ","
           << row.ammMixedSurrogateTrueLocalSelectedCount << ","
           << row.ammMixedSurrogateExtrapolatedSelectedCount << ","
           << row.ammMixedSurrogateOtherSelectedCount << ","
           << row.ammMixedSurrogateSimpleSkippedCount << ","
           << row.ammMixedSurrogateSimpleForcedRefreshCount << ","
           << row.ammMeanGamma << "," << row.ammMeanGkhInitial
           << "," << row.ammMeanMinG << "," << row.ammMeanGkAfterAccelerated
           << "," << row.ammMeanGkhAfterRestartCheck << ","
           << row.ammMeanFinalGk << "," << row.ammMeanPhiLhs << ","
           << row.ammMeanPhiRhs
           << "\n";
  }
}

void writeAmmTrace(const std::string &path,
                   const std::vector<ManualDpgoMmAmmTraceRow> &rows) {
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write AMM trace: " + path);
  }
  output << "iter,robot,local_iter,fobj,true_fobj,"
         << "baseline_surrogate_fobj,baseline_surrogate_offset,"
         << "baseline_surrogate_grad_norm,"
         << "recursive_simple_f,recursive_simple_fobj,"
         << "recursive_simple_delta_Q_delta,recursive_simple_p_term,"
         << "recursive_simple_previous_Gk,refined_ratio,"
         << "local_accepted_iterations,"
         << "local_accepted_iteration_budget,num_oscillations,F0,F1,gamma,"
         << "Gkh_initial,"
         << "surrogate_Gkh_initial,minG,Gk_after_accelerated,"
         << "surrogate_Gk_after_accelerated,Gkh_after_restart_check,"
         << "surrogate_Gkh_after_restart_check,final_Gk,"
         << "surrogate_final_Gk,phi_lhs,phi_rhs,"
         << "restart_certificate_lhs,restart_certificate_rhs,"
         << "restart_certificate_margin,restart_certificate_valid,"
         << "restart_certificate_passed,baseline_surrogate_state,"
         << "recursive_simple_state,recursive_simple_valid,"
         << "refined,prox_reset,hard_restart,"
         << "soft_restart,restart_used_xakh,phi_fallback,"
         << "translation_recovery_attempt_count,"
         << "translation_recovery_accepted_count,"
         << "translation_recovery_rejected_count,"
         << "soft_restart_hits0,soft_restart_hits1\n";
  for (const auto &row : rows) {
    const auto &trace = row.trace;
    output << row.iter << "," << row.robot << "," << trace.localIter << ","
           << std::setprecision(16) << trace.fobj << "," << trace.trueFobj
           << "," << trace.baselineSurrogateFobj << ","
           << trace.baselineSurrogateOffset << ","
           << trace.baselineSurrogateGradNorm << ","
           << trace.recursiveSimpleF << "," << trace.recursiveSimpleFobj
           << "," << trace.recursiveSimpleDeltaQDelta << ","
           << trace.recursiveSimplePTerm << ","
           << trace.recursiveSimplePreviousGk << "," << trace.refinedRatio
           << "," << trace.localAcceptedIterations << ","
           << trace.localAcceptedIterationBudget << ","
           << trace.numOscillations << "," << trace.F0 << "," << trace.F1
           << "," << trace.gamma
           << "," << trace.GkhInitial << "," << trace.surrogateGkhInitial
           << "," << trace.minG << "," << trace.GkAfterAccelerated << ","
           << trace.surrogateGkAfterAccelerated << ","
           << trace.GkhAfterRestartCheck << ","
           << trace.surrogateGkhAfterRestartCheck << "," << trace.finalGk
           << "," << trace.surrogateFinalGk << "," << trace.phiLhs << ","
           << trace.phiRhs << "," << trace.restartCertificateLhs << ","
           << trace.restartCertificateRhs << ","
           << trace.restartCertificateMargin << ","
           << (trace.restartCertificateValid ? 1 : 0) << ","
           << (trace.restartCertificatePassed ? 1 : 0) << ","
           << (trace.baselineSurrogateState ? 1 : 0) << ","
           << (trace.recursiveSimpleState ? 1 : 0) << ","
           << (trace.recursiveSimpleValid ? 1 : 0) << ","
           << (trace.refined ? 1 : 0) << ","
           << (trace.proxReset ? 1 : 0) << ","
           << (trace.hardRestart ? 1 : 0) << ","
           << (trace.softRestart ? 1 : 0) << ","
           << (trace.restartUsedXakh ? 1 : 0) << ","
           << (trace.phiFallback ? 1 : 0) << ","
           << trace.translationRecoveryAttemptCount << ","
           << trace.translationRecoveryAcceptedCount << ","
           << trace.translationRecoveryRejectedCount << ","
           << trace.softRestartHits0 << "," << trace.softRestartHits1
           << "\n";
  }
}

void writeEstimate(const std::string &path, const Matrix &X, unsigned d) {
  Matrix aligned = X;
  const unsigned numPoses = static_cast<unsigned>(X.cols() / (d + 1));
  if (numPoses > 0) {
    const Vector t0 = aligned.col(d);
    for (unsigned pose = 0; pose < numPoses; ++pose) {
      aligned.col(pose * (d + 1) + d) -= t0;
    }
    const Matrix R0 = aligned.block(0, 0, d, d).transpose();
    aligned = R0 * aligned;
  }

  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write estimate: " + path);
  }
  output << std::setprecision(16) << aligned << "\n";
}

Matrix readManualInterleavedEstimate(const std::string &path, int rows,
                                     int cols) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to read external initial estimate: " +
                             path);
  }

  std::vector<double> values;
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }
  if (!input.eof()) {
    throw std::invalid_argument(
        "external initial estimate contains non-numeric data");
  }

  const std::size_t expected =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (values.size() != expected) {
    std::ostringstream message;
    message << "external initial estimate has wrong shape: expected "
            << rows << "x" << cols << " (" << expected
            << " values) but read " << values.size() << " values";
    throw std::invalid_argument(message.str());
  }

  Matrix estimate(rows, cols);
  std::size_t idx = 0;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      estimate(row, col) = values[idx++];
    }
  }
  if (!estimate.allFinite()) {
    throw std::invalid_argument(
        "external initial estimate contains non-finite values");
  }
  return estimate;
}

void writeIterationEstimates(const std::string &directory,
                             const std::vector<Matrix> &estimates,
                             unsigned d,
                             const std::string &prefix = "iter") {
  ensureOutputDirectory(directory);
  const int width = 4;
  for (std::size_t idx = 0; idx < estimates.size(); ++idx) {
    std::ostringstream name;
    name << directory << "/" << prefix << "_" << std::setw(width)
         << std::setfill('0')
         << idx << ".txt";
    writeEstimate(name.str(), estimates[idx], d);
  }
}

SparseMatrix padSparseMatrix(const SparseMatrix &input, std::size_t rows,
                             std::size_t cols) {
  if (input.rows() == static_cast<int>(rows) &&
      input.cols() == static_cast<int>(cols)) {
    return input;
  }
  if (input.rows() > static_cast<int>(rows) ||
      input.cols() > static_cast<int>(cols)) {
    throw std::runtime_error(
        "manual_dpgo_mm local Q is larger than the local pose block");
  }

  SparseMatrix padded(rows, cols);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(input.nonZeros()));
  for (int k = 0; k < input.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(input, k); it; ++it) {
      if (it.row() < static_cast<int>(rows) &&
          it.col() < static_cast<int>(cols)) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
    }
  }
  padded.setFromTriplets(triplets.begin(), triplets.end());
  padded.makeCompressed();
  return padded;
}

struct LocalGradientCorrectionCurvatureProbe {
  bool ok{false};
  bool currentFinite{false};
  double currentCost{0.0};
  double descentNumerator{0.0};
  double curvature{0.0};
  double maxStep{0.0};
};

struct LocalGradientCorrectionDiagnostics {
  std::size_t coupledDirectionCandidateCount{0};
  std::size_t coupledDirectionAcceptedCount{0};
  std::size_t coupledDirectionRejectedCount{0};
  std::size_t coupledDirectionPacketCount{0};
  std::size_t coupledDirectionSkippedPacketCount{0};
  std::size_t coupledDirectionScoreSkippedPacketCount{0};
  std::size_t coupledDirectionByteBudgetSkippedPacketCount{0};
  std::size_t coupledDirectionScalarCount{0};
  std::size_t compactSchurCandidateCount{0};
  std::size_t compactSchurAcceptedCount{0};
  std::size_t compactSchurGuardRejectedCount{0};
  std::size_t compactSchurGradientGuardRejectedCount{0};
  std::size_t compactSchurSolveFailureCount{0};
  std::size_t compactSchurBoundaryColCount{0};
  std::size_t compactSchurPrivateColCount{0};
  double compactSchurGradientChangeSum{0.0};

  void add(const LocalGradientCorrectionDiagnostics &other) {
    coupledDirectionCandidateCount += other.coupledDirectionCandidateCount;
    coupledDirectionAcceptedCount += other.coupledDirectionAcceptedCount;
    coupledDirectionRejectedCount += other.coupledDirectionRejectedCount;
    coupledDirectionPacketCount += other.coupledDirectionPacketCount;
    coupledDirectionSkippedPacketCount +=
        other.coupledDirectionSkippedPacketCount;
    coupledDirectionScoreSkippedPacketCount +=
        other.coupledDirectionScoreSkippedPacketCount;
    coupledDirectionByteBudgetSkippedPacketCount +=
        other.coupledDirectionByteBudgetSkippedPacketCount;
    coupledDirectionScalarCount += other.coupledDirectionScalarCount;
    compactSchurCandidateCount += other.compactSchurCandidateCount;
    compactSchurAcceptedCount += other.compactSchurAcceptedCount;
    compactSchurGuardRejectedCount += other.compactSchurGuardRejectedCount;
    compactSchurGradientGuardRejectedCount +=
        other.compactSchurGradientGuardRejectedCount;
    compactSchurSolveFailureCount += other.compactSchurSolveFailureCount;
    compactSchurBoundaryColCount += other.compactSchurBoundaryColCount;
    compactSchurPrivateColCount += other.compactSchurPrivateColCount;
    compactSchurGradientChangeSum += other.compactSchurGradientChangeSum;
  }
};

FullEquivHybridRqnOptions
makeFullEquivHybridRqnOptions(const ManualDpgoMmOptions &options) {
  FullEquivHybridRqnOptions rqnOptions;
  rqnOptions.memorySize = options.fullEquivHybridRqnMemorySize;
  rqnOptions.minCurvatureRatio =
      options.fullEquivHybridRqnMinCurvatureRatio;
  return rqnOptions;
}

FullEquivHybridReducedRotationPreconditioner
makeFullEquivHybridReducedRotationPreconditioner(
    ManualDpgoMmReducedRotationPreconditioner mode) {
  switch (mode) {
    case ManualDpgoMmReducedRotationPreconditioner::Jacobi:
      return FullEquivHybridReducedRotationPreconditioner::Jacobi;
    case ManualDpgoMmReducedRotationPreconditioner::SchurJacobi:
      return FullEquivHybridReducedRotationPreconditioner::SchurJacobi;
    case ManualDpgoMmReducedRotationPreconditioner::Cholesky:
    case ManualDpgoMmReducedRotationPreconditioner::Portfolio:
    case ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio:
      return FullEquivHybridReducedRotationPreconditioner::Cholesky;
    case ManualDpgoMmReducedRotationPreconditioner::None:
    default:
      return FullEquivHybridReducedRotationPreconditioner::None;
  }
}

class ManualDpgoMmAgent {
 public:
  struct TranslationRecoveryResult {
    Matrix value;
    bool attempted{false};
    bool accepted{false};
  };

  ManualDpgoMmAgent(unsigned robotID, unsigned dimension, unsigned rank,
                    unsigned localPoseCount,
                    std::vector<RelativeSEMeasurement> odometryIn,
                    std::vector<RelativeSEMeasurement> privateLoopsIn,
                    std::vector<RelativeSEMeasurement> sharedLoopsIn,
                    std::vector<double> robotMeasurementDegreesIn,
                    const ManualDpgoMmOptions &optionsIn)
      : id(robotID),
        d(dimension),
        r(rank),
        n(localPoseCount),
        odometry(std::move(odometryIn)),
        privateLoops(std::move(privateLoopsIn)),
        sharedLoops(std::move(sharedLoopsIn)),
        robotMeasurementDegrees(std::move(robotMeasurementDegreesIn)),
        options(optionsIn),
        problem(localPoseCount, dimension, rank),
        manualOptimizer(&problem),
        reducedOptimizer(&problem),
        fullEquivHybridRqnMemory(makeFullEquivHybridRqnOptions(optionsIn)) {
    const std::size_t directionBlockEntries =
        static_cast<std::size_t>(r) * static_cast<std::size_t>(d + 1);
    if (options.localGradientCorrectionCoupledDirectionTopKEntries >
        directionBlockEntries) {
      throw std::invalid_argument(
          "localGradientCorrectionCoupledDirectionTopKEntries cannot exceed the direction block size");
    }
    buildNeighborPublicPoseMap();
    constructQMatrix();
  }

  unsigned getID() const { return id; }

  void setOptimizationRound(unsigned round) {
    currentOptimizationRound = round;
  }

  std::vector<unsigned> getNeighbors() const {
    std::vector<unsigned> neighbors;
    neighbors.reserve(neighborPublicPoses.size());
    for (const auto &entry : neighborPublicPoses) {
      neighbors.push_back(entry.first);
    }
    return neighbors;
  }

  const std::vector<unsigned> &getNeighborPublicPoses(
      unsigned neighborID) const {
    static const std::vector<unsigned> empty;
    const auto it = neighborPublicPoses.find(neighborID);
    return it == neighborPublicPoses.end() ? empty : it->second;
  }

  double getNeighborPoseSensitivity(unsigned neighborID,
                                    unsigned neighborPoseIndex) const {
    double squaredSensitivity = 0.0;
    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(row, row) = m.weight * m.kappa;
      }
      Omega(d, d) = m.weight * m.tau;

      if (m.r1 == id && m.r2 == neighborID &&
          m.p2 == neighborPoseIndex) {
        squaredSensitivity += (Omega * T.transpose()).squaredNorm();
      } else if (m.r2 == id && m.r1 == neighborID &&
                 m.p1 == neighborPoseIndex) {
        squaredSensitivity += (T * Omega).squaredNorm();
      }
    }
    const double sensitivity = std::sqrt(squaredSensitivity);
    return std::isfinite(sensitivity) && sensitivity > 0.0 ? sensitivity : 1.0;
  }

  double getNeighborPoseResidualProxy(unsigned neighborID,
                                      unsigned neighborPoseIndex,
                                      const Matrix &neighborPose) const {
    if (X.rows() != static_cast<int>(r) ||
        X.cols() != static_cast<int>(n * (d + 1)) ||
        neighborPose.rows() != static_cast<int>(r) ||
        neighborPose.cols() != static_cast<int>(d + 1) ||
        !X.allFinite() || !neighborPose.allFinite()) {
      return 0.0;
    }

    double cost = 0.0;
    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1.0;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(row, row) = m.weight * m.kappa;
      }
      Omega(d, d) = m.weight * m.tau;

      Matrix residual;
      if (m.r1 == id && m.r2 == neighborID &&
          m.p2 == neighborPoseIndex) {
        const Matrix local =
            X.block(0, m.p1 * (d + 1), r, d + 1);
        residual = local * T - neighborPose;
      } else if (m.r2 == id && m.r1 == neighborID &&
                 m.p1 == neighborPoseIndex) {
        const Matrix local =
            X.block(0, m.p2 * (d + 1), r, d + 1);
        residual = neighborPose * T - local;
      } else {
        continue;
      }
      cost += 0.5 * residual.cwiseProduct(residual * Omega).sum();
    }
    if (!std::isfinite(cost) || cost <= 0.0) {
      return 0.0;
    }
    return std::sqrt(2.0 * cost);
  }

  unsigned getNeighborPoseStaleness(unsigned neighborID,
                                    unsigned neighborPoseIndex) const {
    return neighborPoseStaleness(PoseID(neighborID, neighborPoseIndex));
  }

  double getCoupledDirectionBudgetFraction() const {
    return options.localGradientCorrectionCoupledDirectionBudgetFraction;
  }

  unsigned getCoupledDirectionMaxPacketsPerReceiver() const {
    return options.localGradientCorrectionCoupledDirectionMaxPacketsPerReceiver;
  }

  unsigned getCoupledDirectionTopKEntries() const {
    return options.localGradientCorrectionCoupledDirectionTopKEntries;
  }

  double getCoupledDirectionMinScore() const {
    return options.localGradientCorrectionCoupledDirectionMinScore;
  }

  double getCoupledDirectionMinScoreRatio() const {
    return options.localGradientCorrectionCoupledDirectionMinScoreRatio;
  }

  double getCoupledDirectionByteBudgetMb() const {
    return options.localGradientCorrectionCoupledDirectionByteBudgetMb;
  }

  void setX(const Matrix &value) {
    if (value.rows() != static_cast<int>(r) ||
        value.cols() != static_cast<int>(n * (d + 1))) {
      throw std::invalid_argument("manual_dpgo_mm local X has wrong shape");
    }
    X = value;
  }

  const Matrix &getX() const { return X; }

  bool getSharedPose(unsigned poseIndex, Matrix &pose) const {
    if (poseIndex >= n || X.rows() == 0) {
      return false;
    }
    pose = X.block(0, poseIndex * (d + 1), r, d + 1);
    return true;
  }

  void updateNeighborPoses(const PoseDict &poses) {
    for (const auto &entry : poses) {
      const auto previous = neighborPoseDict.find(entry.first);
      if (previous != neighborPoseDict.end()) {
        neighborPosePreviousDict[entry.first] = previous->second;
        const auto previousRound =
            neighborPoseLastUpdateRound.find(entry.first);
        if (previousRound != neighborPoseLastUpdateRound.end()) {
          neighborPosePreviousUpdateRound[entry.first] =
              previousRound->second;
        }
      }
      neighborPoseDict[entry.first] = entry.second;
      neighborPoseLastUpdateRound[entry.first] = currentOptimizationRound;
    }
  }

  bool updateLocalModel() {
    return constructGMatrix();
  }

  bool evaluateLocalModel(double &cost, double &gradNorm) {
    if (!updateLocalModel()) {
      cost = 0.0;
      gradNorm = 0.0;
      return false;
    }
    cost = problem.f(X) + modelConstant;
    gradNorm = problem.RieGradNorm(X);
    return std::isfinite(cost) && std::isfinite(gradNorm);
  }

  bool fixedNeighborChordalRefit() {
    if (!constructGMatrix()) {
      return false;
    }
    const unsigned blockDim = d + 1;
    const unsigned totalCols = n * blockDim;
    if (n == 0 || X.rows() != static_cast<int>(r) ||
        X.cols() != static_cast<int>(totalCols) || !X.allFinite() ||
        problem.getQRef().rows() != static_cast<int>(totalCols) ||
        problem.getQRef().cols() != static_cast<int>(totalCols)) {
      return true;
    }

    const double currentCost = problem.f(X) + modelConstant;
    if (!std::isfinite(currentCost)) {
      return true;
    }

    const SparseMatrix &Q = problem.getQRef();
    const SparseMatrix &G = problem.getGRef();
    Matrix denseG = Matrix::Zero(static_cast<int>(r),
                                 static_cast<int>(totalCols));
    for (int outer = 0; outer < G.outerSize(); ++outer) {
      for (SparseMatrix::InnerIterator it(G, outer); it; ++it) {
        denseG(static_cast<int>(it.row()), static_cast<int>(it.col())) +=
            it.value();
      }
    }

    double diagScale = 0.0;
    unsigned diagCount = 0;
    for (unsigned col = 0; col < totalCols; ++col) {
      const double diag = std::abs(Q.coeff(static_cast<int>(col),
                                           static_cast<int>(col)));
      if (std::isfinite(diag) && diag > 0.0) {
        diagScale += diag;
        ++diagCount;
      }
    }
    diagScale = diagCount > 0 ? diagScale / static_cast<double>(diagCount)
                              : 1.0;
    if (!std::isfinite(diagScale) || diagScale <= 0.0) {
      diagScale = 1.0;
    }

    Matrix bestX = X;
    double bestCost = currentCost;
    const Matrix rhs = -denseG.transpose();
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
      ColMajorSparseMatrix system = Q;
      const double ridge =
          (attempt == 0)
              ? 0.0
              : 1e-10 * diagScale *
                    std::pow(10.0, static_cast<double>(attempt - 1));
      if (ridge > 0.0) {
        for (unsigned col = 0; col < totalCols; ++col) {
          system.coeffRef(static_cast<int>(col), static_cast<int>(col)) +=
              ridge;
        }
      }
      system.makeCompressed();

      Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
      solver.compute(system);
      if (solver.info() != Eigen::Success) {
        continue;
      }
      const Matrix solved = solver.solve(rhs);
      if (solver.info() != Eigen::Success ||
          solved.rows() != static_cast<int>(totalCols) ||
          solved.cols() != static_cast<int>(r) || !solved.allFinite()) {
        continue;
      }
      Matrix candidate = solved.transpose();
      if (candidate.rows() != X.rows() || candidate.cols() != X.cols() ||
          !candidate.allFinite()) {
        continue;
      }
      const Matrix target = projectRotationBlocks(candidate);
      if (target.rows() != X.rows() || target.cols() != X.cols() ||
          !target.allFinite()) {
        continue;
      }
      double alpha = 0.25;
      for (unsigned backtrack = 0; backtrack < 4; ++backtrack) {
        Matrix trial = projectRotationBlocks(X + alpha * (target - X));
        if (trial.rows() != X.rows() || trial.cols() != X.cols() ||
            !trial.allFinite()) {
          alpha *= 0.5;
          continue;
        }
        const double trialCost = problem.f(trial) + modelConstant;
        if (std::isfinite(trialCost) && trialCost < bestCost - 1e-10) {
          bestX = trial;
          bestCost = trialCost;
          break;
        }
        alpha *= 0.5;
      }
    }

    if (bestCost < currentCost - 1e-10) {
      X = bestX;
      resetLocalHistory();
    }
    return true;
  }

  double evaluateBoundaryEdgeCost(const Matrix &value) const {
    return boundaryEdgeCost(value);
  }

  bool optimizeLocalModel() {
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer ? &optimizerProfile.agentOptimizeLocalModelSec
                                 : nullptr);
    if (options.profileOptimizer) {
      ++optimizerProfile.agentOptimizeLocalModelCount;
    }
    ROPTResult result;
    Matrix candidate;
    const Matrix originalX = X;
    lastAdaptiveRefinementCount = 0;
    lastExtrapolationAcceptedCount = 0;
    lastExtrapolationRejectedCount = 0;
    lastGExtrapolationAcceptedCount = 0;
    lastGExtrapolationRejectedCount = 0;
    lastCoupledExtrapolationAcceptedCount = 0;
    lastCoupledExtrapolationRejectedCount = 0;
    lastAndersonAcceptedCount = 0;
    lastAndersonRejectedCount = 0;
    lastSquaremAcceptedCount = 0;
    lastSquaremRejectedCount = 0;
    lastAmmAcceleratedAcceptedCount = 0;
    lastAmmRestartCount = 0;
    lastAmmHardRestartCount = 0;
    lastAmmSoftRestartCount = 0;
    lastAmmPhiFallbackCount = 0;
    lastAmmLocalMeritRejectedCount = 0;
    lastAmmProximalStartCount = 0;
    lastAmmSkippedCount = 0;
    lastAmmMixedSurrogateCandidateCount = 0;
    lastAmmMixedSurrogateTrueLocalAcceptedCount = 0;
    lastAmmMixedSurrogateSimpleSelectedCount = 0;
    lastAmmMixedSurrogateTrueLocalSelectedCount = 0;
    lastAmmMixedSurrogateExtrapolatedSelectedCount = 0;
    lastAmmMixedSurrogateOtherSelectedCount = 0;
    lastAmmMixedSurrogateSimpleSkippedCount = 0;
    lastAmmMixedSurrogateSimpleForcedRefreshCount = 0;
    lastEdgeTightQuadraticEvalCount = 0;
    lastEdgeTightQuadraticSurrogateCostSum = 0.0;
    lastEdgeTightQuadraticTrueCostSum = 0.0;
    lastEdgeTightQuadraticMajorizationGapMin =
        std::numeric_limits<double>::infinity();
    lastVariableProjectedSchurCandidateCount = 0;
    lastVariableProjectedSchurAcceptedCount = 0;
    lastBoundaryProximalCandidateCount = 0;
    lastBoundaryProximalAcceptedCount = 0;
    lastSurrogateBoundCheckCount = 0;
    lastSurrogateBoundViolationCount = 0;
    lastSurrogateBoundMinMargin = std::numeric_limits<double>::infinity();
    lastBoundaryEdgeCostBefore = 0.0;
    lastBoundaryEdgeCostAfter = 0.0;
    lastSeparatorDeltaNorm = 0.0;
    lastFullEquivHybridWarmStartCandidateCount = 0;
    lastFullEquivHybridWarmStartAcceptedCount = 0;
    lastFullEquivHybridWarmStartGuardRejectedCount = 0;
    lastFullEquivHybridSchurStepCandidateCount = 0;
    lastFullEquivHybridSchurStepAcceptedCount = 0;
    lastFullEquivHybridSchurStepGuardRejectedCount = 0;
    lastFullEquivHybridLocalPortfolioCandidateCount = 0;
    lastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount = 0;
    lastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount = 0;
    lastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount = 0;
    lastFullEquivHybridSchwarzSmoothingSweepCount = 0;
    lastFullEquivHybridSchwarzSmoothingCandidateCount = 0;
    lastFullEquivHybridSchwarzSmoothingAcceptedCount = 0;
    lastFullEquivHybridSchwarzSmoothingRejectedCount = 0;
    lastFullEquivHybridSchwarzSmoothingCostDecrease = 0.0;
    lastFullEquivHybridSchwarzSmoothingTimeSec = 0.0;
    lastFullEquivHybridLinearPcgIterationCount = 0;
    lastFullEquivHybridLinearInitialResidual = 0.0;
    lastFullEquivHybridLinearFinalResidual = 0.0;
    lastFullEquivHybridLinearFullResidual = 0.0;
    lastFullEquivHybridLinearSolveTimeSec = 0.0;
    lastFullEquivHybridStepTrialCount = 0;
    lastFullEquivHybridStepTrialAcceptedCount = 0;
    lastFullEquivHybridStepTrialRejectedCount = 0;
    lastFullEquivHybridStepPredictedDecreaseSum = 0.0;
    lastFullEquivHybridStepActualDecreaseSum = 0.0;
    lastFullEquivHybridStepRhoSum = 0.0;
    lastFullEquivHybridStepRhoCount = 0;
    lastFullEquivHybridStepAcceptedScaleSum = 0.0;
    lastFullEquivHybridStepAcceptedPredictedDecreaseSum = 0.0;
    lastFullEquivHybridStepAcceptedActualDecreaseSum = 0.0;
    lastFullEquivHybridStepAcceptedRhoSum = 0.0;
    lastFullEquivHybridStepAcceptedRhoCount = 0;
    lastFullEquivHybridStepParetoCandidateCount = 0;
    lastFullEquivHybridStepParetoSelectedCount = 0;
    lastFullEquivHybridStepParetoSelectedScaleSum = 0.0;
    lastFullEquivHybridStepParetoBestDecreaseSum = 0.0;
    lastFullEquivHybridStepParetoSelectedDecreaseSum = 0.0;
    lastFullEquivHybridStepParetoSelectedGradientSum = 0.0;
    lastFullEquivHybridStepParetoGradientEvalCount = 0;
    lastFullEquivHybridStepParetoGradientEvalTimeSec = 0.0;
    lastFullEquivHybridTranslationRecoveryStepTrialCandidateCount = 0;
    lastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount = 0;
    lastFullEquivHybridTranslationRecoveryStepTrialSelectedCount = 0;
    lastFullEquivHybridActiveSeparatorCandidateCount = 0;
    lastFullEquivHybridActiveSeparatorAcceptedCount = 0;
    lastFullEquivHybridActiveSeparatorRejectedCount = 0;
    lastFullEquivHybridActiveSeparatorStepSum = 0.0;
    lastFullEquivHybridActiveSeparatorCostDecreaseSum = 0.0;
    lastFullEquivHybridActiveSeparatorLmSchurCandidateCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurAcceptedCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurFallbackCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurPrivateColCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount = 0;
    lastFullEquivHybridActiveSeparatorLmSchurAlphaSum = 0.0;
    lastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum = 0.0;
    lastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum = 0.0;
    lastFullEquivHybridTranslationRecoveryPolishAttemptCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishAcceptedCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishRejectedCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount =
        0;
    lastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum = 0.0;
    lastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount = 0;
    lastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum = 0.0;
    lastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum = 0.0;
    lastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum =
        0.0;
    lastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum = 0.0;
    lastFullEquivHybridTranslationRecoveryPolishGradientChangeSum = 0.0;
    lastFullEquivHybridSparseMatrixVectorProductCount = 0;
    lastFullEquivHybridReducedRotationInitialGuessCandidateCount = 0;
    lastFullEquivHybridReducedRotationInitialGuessUsedCount = 0;
    lastFullEquivHybridReducedRotationInitialGuessRejectedCount = 0;
    lastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount = 0;
    lastFullEquivHybridTranslationRecoveryInitialGuessUsedCount = 0;
    lastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount = 0;
    lastFullEquivHybridTranslationSchurPreconditionerApplicationCount = 0;
    lastFullEquivHybridTranslationSchurPreconditionerFactorizationCount = 0;
    lastFullEquivHybridTranslationSchurPreconditionerFallbackCount = 0;
    lastFullEquivHybridLocalChainPreconditionerApplicationCount = 0;
    lastFullEquivHybridLocalChainPreconditionerFactorizationCount = 0;
    lastFullEquivHybridLocalChainPreconditionerFallbackCount = 0;
    lastFullEquivHybridTranslationBlockPreconditionerApplicationCount = 0;
    lastFullEquivHybridTranslationBlockPreconditionerFactorizationCount = 0;
    lastFullEquivHybridTranslationBlockPreconditionerFallbackCount = 0;
    lastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount = 0;
    lastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount = 0;
    lastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount = 0;
    lastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount = 0;
    lastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount = 0;
    lastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount = 0;
    lastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount = 0;
    lastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount = 0;
    lastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount = 0;
    lastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount = 0;
    lastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount = 0;
    lastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension = 0;
    lastFullEquivHybridReducedRotationPreconditionerApplicationCount = 0;
    lastFullEquivHybridReducedRotationPreconditionerFactorizationCount = 0;
    lastFullEquivHybridRqnUsedCount = 0;
    lastFullEquivHybridRqnAcceptedPairCount = 0;
    lastFullEquivHybridRqnRejectedPairCount = 0;
    lastFullEquivHybridRqnMemorySize = fullEquivHybridRqnMemory.size();
    lastFullEquivHybridRqnPreconditionerApplicationCount = 0;
    lastAmmTrace = ManualDpgoMmAmmTrace();
    SparseMatrix trueQ;
    SparseMatrix trueG;
    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer ? &optimizerProfile.localModelSnapshotSec
                                   : nullptr);
      trueQ = problem.getQ();
      trueG = problem.getG();
    }
    if (options.profileOptimizer) {
      ++optimizerProfile.localModelSnapshotCount;
    }
    bool problemQIsTrue = true;
    bool problemGIsTrue = true;
    std::size_t candidateEvaluationModelVersion = 0;
    auto setProblemQForLocalSolve = [&](const SparseMatrix &Q,
                                        bool isTrueModel) {
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer ? &optimizerProfile.localModelSwitchSec
                                     : nullptr);
        problem.setQWithoutPreconditioner(Q);
      }
      if (options.profileOptimizer) {
        ++optimizerProfile.localModelSwitchCount;
      }
      problemQIsTrue = isTrueModel;
      ++candidateEvaluationModelVersion;
    };
    auto setProblemGForLocalSolve = [&](const SparseMatrix &G,
                                        bool isTrueModel) {
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer ? &optimizerProfile.localModelSwitchSec
                                     : nullptr);
        problem.setG(G);
      }
      if (options.profileOptimizer) {
        ++optimizerProfile.localModelSwitchCount;
      }
      problemGIsTrue = isTrueModel;
      ++candidateEvaluationModelVersion;
    };
    auto restoreTrueLocalModel = [&]() {
      if (!problemQIsTrue) {
        problem.setQWithoutPreconditioner(trueQ);
        problemQIsTrue = true;
        ++candidateEvaluationModelVersion;
      }
      if (!problemGIsTrue) {
        problem.setG(trueG);
        problemGIsTrue = true;
        ++candidateEvaluationModelVersion;
      }
    };
    SparseMatrix currentDpgoSimpleG;
    bool currentDpgoSimpleValid = false;
    const bool weightedEdgeSplitRequested =
        options.surrogateMode == ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
    const bool adaptiveSpectralRequested =
        options.surrogateMode == ManualDpgoMmSurrogateMode::AdaptiveSpectral;
    const bool variableProjectedSchurRequested =
        options.surrogateMode ==
        ManualDpgoMmSurrogateMode::VariableProjectedSchur;
    const bool weightedEdgeSplitSymmetric =
        weightedEdgeSplitRequested && edgeSplitThetaIsSymmetric();
    const bool dpgoSimpleRequestedWithoutReducedInterface =
        options.reducedSurrogateMode ==
            ManualDpgoMmReducedSurrogateMode::DpgoSimple ||
        options.reducedSurrogateMode ==
            ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic ||
        weightedEdgeSplitSymmetric || adaptiveSpectralRequested ||
        options.communicationTopologyBoundarySurrogate ||
        options.communicationTopologyBoundaryCandidate;
    const bool reducedInterfaceScheduled =
        options.communicationTopologyReducedInterfaceModel &&
        (options.communicationTopologyReducedInterfaceMaxLocalIterations == 0 ||
         ammLocalIter <
             options.communicationTopologyReducedInterfaceMaxLocalIterations);
    const bool reducedSurrogateRequestsDpgoSimple =
        dpgoSimpleRequestedWithoutReducedInterface ||
        reducedInterfaceScheduled;
    WeightedEdgeSplitModel currentWeightedEdgeSplit;
    WeightedEdgeSplitModel currentAdaptiveSpectral;
    WeightedEdgeSplitModel currentReducedInterface;
    const bool needsSurrogatePreparation =
        options.ammDpgoSurrogateParity || reducedSurrogateRequestsDpgoSimple ||
        (weightedEdgeSplitRequested && !weightedEdgeSplitSymmetric) ||
        adaptiveSpectralRequested || reducedInterfaceScheduled ||
        options.communicationTopologyBoundarySurrogate;
    if (needsSurrogatePreparation) {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer
              ? &optimizerProfile.localSurrogatePreparationSec
              : nullptr);
      if (options.ammDpgoSurrogateParity ||
          reducedSurrogateRequestsDpgoSimple) {
        currentDpgoSimpleG = makeDpgoSimpleLinearTermManual(X);
        currentDpgoSimpleValid =
            currentDpgoSimpleG.rows() == static_cast<int>(r) &&
            currentDpgoSimpleG.cols() == static_cast<int>(n * (d + 1)) &&
            dpgoSimpleQManual.rows() == static_cast<int>(n * (d + 1)) &&
            dpgoSimpleQManual.cols() == static_cast<int>(n * (d + 1)) &&
            dpgoSimpleQFull.rows() ==
                static_cast<int>((d + 1) *
                                 (n + dpgoSimpleNeighborOrder.size())) &&
            dpgoSimpleP.rows() == dpgoSimpleQFull.rows() &&
            dpgoSimpleP0.rows() == dpgoSimpleQFull.rows();
      }
      if (weightedEdgeSplitRequested && !weightedEdgeSplitSymmetric) {
        currentWeightedEdgeSplit = makeWeightedEdgeSplitModel(X);
      }
      if (adaptiveSpectralRequested ||
          options.communicationTopologyBoundarySurrogate) {
        currentAdaptiveSpectral = makeAdaptiveSpectralModel(X);
      }
      if (reducedInterfaceScheduled &&
          currentDpgoSimpleValid &&
          X.rows() == static_cast<int>(r) &&
          X.cols() == static_cast<int>(n * (d + 1)) && X.allFinite() &&
          trueQ.rows() == static_cast<int>(n * (d + 1)) &&
          trueQ.cols() == static_cast<int>(n * (d + 1)) &&
          trueG.rows() == static_cast<int>(r) &&
          trueG.cols() == static_cast<int>(n * (d + 1))) {
        const unsigned blockDim = d + 1;
        const unsigned totalDim = n * blockDim;
        Matrix simpleGrad = X * dpgoSimpleQManual;
        simpleGrad += Matrix(currentDpgoSimpleG);
        Matrix trueGrad = X * trueQ;
        trueGrad += Matrix(trueG);
        if (simpleGrad.rows() == static_cast<int>(r) &&
            simpleGrad.cols() == static_cast<int>(totalDim) &&
            trueGrad.rows() == static_cast<int>(r) &&
            trueGrad.cols() == static_cast<int>(totalDim) &&
            simpleGrad.allFinite() && trueGrad.allFinite()) {
          std::map<unsigned, double> freshnessWeightedSums;
          std::map<unsigned, double> stiffnessSums;
          std::map<unsigned, double> stiffnessAverageSums;
          std::map<unsigned, std::size_t> stiffnessAverageCounts;
          for (const auto &m : sharedLoops) {
            PoseID neighborPose;
            unsigned localPose = 0;
            bool localIsSource = false;
            if (m.r1 == id) {
              neighborPose = PoseID(m.r2, m.p2);
              localPose = m.p1;
              localIsSource = true;
            } else if (m.r2 == id) {
              neighborPose = PoseID(m.r1, m.p1);
              localPose = m.p2;
            } else {
              continue;
            }
            if (localPose >= n) {
              continue;
            }
            const double stiffness =
                localMeasurementStiffness(m, localIsSource);
            if (!std::isfinite(stiffness) || stiffness <= 0.0) {
              continue;
            }
            const double freshness =
                1.0 / (1.0 + static_cast<double>(
                                  neighborPoseStaleness(neighborPose)));
            if (!std::isfinite(freshness) || freshness <= 0.0) {
              continue;
            }
            freshnessWeightedSums[localPose] += stiffness * freshness;
            stiffnessSums[localPose] += stiffness;
            stiffnessAverageSums[localPose] += stiffness;
            ++stiffnessAverageCounts[localPose];
          }

          SparseMatrix q = dpgoSimpleQManual;
          SparseMatrix g = currentDpgoSimpleG;
          bool addedInterfaceTerm = false;
          const double modelWeight =
              std::max(0.0,
                       options.communicationTopologyReducedInterfaceWeight);
          for (const auto &entry : stiffnessSums) {
            const unsigned pose = entry.first;
            if (pose >= n || entry.second <= 0.0) {
              continue;
            }
            const double freshnessConfidence =
                freshnessWeightedSums[pose] / entry.second;
            if (!std::isfinite(freshnessConfidence) ||
                freshnessConfidence <= 0.0) {
              continue;
            }
            double diagScale = 0.0;
            std::size_t diagCount = 0;
            const unsigned colStart = pose * blockDim;
            for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
              const int col = static_cast<int>(colStart + localCol);
              diagScale += std::abs(dpgoSimpleQManual.coeff(col, col));
              ++diagCount;
            }
            if (diagCount > 0) {
              diagScale /= static_cast<double>(diagCount);
            }
            if (!std::isfinite(diagScale) || diagScale <= 1e-14) {
              const auto countIt = stiffnessAverageCounts.find(pose);
              if (countIt != stiffnessAverageCounts.end() &&
                  countIt->second > 0) {
                diagScale = stiffnessAverageSums[pose] /
                            static_cast<double>(countIt->second);
              }
            }
            if (!std::isfinite(diagScale) || diagScale <= 1e-14) {
              diagScale = 1.0;
            }

            const double beta =
                std::min(1.0, modelWeight * freshnessConfidence);
            const double lambda = beta * diagScale;
            if (!std::isfinite(beta) || beta <= 0.0 ||
                !std::isfinite(lambda) || lambda < 0.0) {
              continue;
            }
            for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
              const int col = static_cast<int>(colStart + localCol);
              q.coeffRef(col, col) += lambda;
              for (unsigned row = 0; row < r; ++row) {
                const int rowIndex = static_cast<int>(row);
                g.coeffRef(rowIndex, col) +=
                    beta * (trueGrad(rowIndex, col) -
                            simpleGrad(rowIndex, col)) -
                    lambda * X(rowIndex, col);
              }
            }
            addedInterfaceTerm = true;
          }
          if (addedInterfaceTerm) {
            q.makeCompressed();
            g.makeCompressed();
            currentReducedInterface.q = q;
            currentReducedInterface.g = g;
            currentReducedInterface.valid =
                currentReducedInterface.q.rows() ==
                    static_cast<int>(totalDim) &&
                currentReducedInterface.q.cols() ==
                    static_cast<int>(totalDim) &&
                currentReducedInterface.g.rows() == static_cast<int>(r) &&
                currentReducedInterface.g.cols() ==
                    static_cast<int>(totalDim);
          }
        }
      }
      if (options.profileOptimizer) {
        ++optimizerProfile.localSurrogatePreparationCount;
      }
    }
    const SparseMatrix *historyGForThisStepPtr =
        (currentDpgoSimpleValid &&
         (options.ammDpgoSurrogateParity ||
          dpgoSimpleRequestedWithoutReducedInterface))
            ? &currentDpgoSimpleG
            : &trueG;
    const SparseMatrix &historyGForThisStep = *historyGForThisStepPtr;
    if (options.ammDpgoRecursiveSimpleState &&
        options.ammDpgoSurrogateParity &&
        options.ammRecursiveSimpleReanchorPeriod > 0 && ammLocalIter > 0 &&
        ammLocalIter % options.ammRecursiveSimpleReanchorPeriod == 0) {
      recursiveSimpleInitialized = false;
      recursiveSimplePreviousZRows.resize(0, 0);
      recursiveSimplePreviousGk = 0.0;
    }
    if (options.localSolver == ManualDpgoMmLocalSolver::ManualFull ||
        options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid) {
      if (options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid &&
          (options.fullEquivHybridBackend ==
               ManualDpgoMmFullEquivHybridBackend::SparseDirectSchur ||
           options.fullEquivHybridBackend ==
               ManualDpgoMmFullEquivHybridBackend::PcgSchur ||
           options.fullEquivHybridBackend ==
               ManualDpgoMmFullEquivHybridBackend::PcgFull)) {
        enum class FehLocalPortfolioSource {
          Unsmoothed,
          SchwarzOnly,
          SchwarzFeh,
          LocalStateExtrapolated,
          SimpleSurrogate,
        };
        struct FehLocalPortfolioCandidate {
          Matrix x;
          ROPTResult result;
          double cost{std::numeric_limits<double>::infinity()};
          double modelCost{std::numeric_limits<double>::infinity()};
          double gradient{std::numeric_limits<double>::infinity()};
          double modelGradient{std::numeric_limits<double>::infinity()};
          FehLocalPortfolioSource source{FehLocalPortfolioSource::Unsmoothed};
          bool ok{false};
        };
        auto fehCandidateIsBetter =
            [&](const FehLocalPortfolioCandidate &candidateResult,
                const FehLocalPortfolioCandidate &bestResult) {
              if (!candidateResult.ok) {
                return false;
              }
              if (!bestResult.ok) {
                return true;
              }
              const double tolerance =
                  std::max(0.0, options.localCandidateCostTieTolerance);
              if (candidateResult.cost < bestResult.cost - tolerance) {
                return true;
              }
              return tolerance > 0.0 &&
                     candidateResult.cost <= bestResult.cost + tolerance &&
                     candidateResult.gradient <
                         bestResult.gradient - 1e-12;
        };
        auto applyFehTranslationRecoveryPolish =
            [&](FehLocalPortfolioCandidate &evaluated) {
              if (!options.fullEquivHybridTranslationRecoveryPolish ||
                  !evaluated.ok || evaluated.x.rows() != X.rows() ||
                  evaluated.x.cols() != X.cols() ||
                  !evaluated.x.allFinite()) {
                return;
              }
              if (!problemQIsTrue || !problemGIsTrue) {
                return;
              }
              const double originalCost = evaluated.cost;
              const double originalGradient = evaluated.gradient;
              const double originalModelCost = evaluated.modelCost;
              const double originalModelGradient = evaluated.modelGradient;
              struct TranslationPolishCandidate {
                Matrix value;
                double cost{std::numeric_limits<double>::infinity()};
                double gradient{std::numeric_limits<double>::infinity()};
                double alpha{1.0};
                bool partial{false};
              };
              auto acceptPolishedCandidate =
                  [&](const Matrix &value, double recoveredCost,
                      double recoveredGradient, double alpha,
                      bool fromBacktracking, bool fromMeritSelector) {
                    evaluated.x = value;
                    evaluated.cost = recoveredCost;
                    evaluated.gradient = recoveredGradient;
                    if (evaluated.source ==
                        FehLocalPortfolioSource::SimpleSurrogate) {
                      evaluated.modelCost = originalModelCost;
                      evaluated.modelGradient = originalModelGradient;
                    } else {
                      evaluated.modelCost = recoveredCost;
                      evaluated.modelGradient = recoveredGradient;
                    }
                    evaluated.result.fOpt = recoveredCost;
                    evaluated.result.gradNormOpt = recoveredGradient;
                    lastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum +=
                        originalCost - recoveredCost;
                    lastFullEquivHybridTranslationRecoveryPolishGradientChangeSum +=
                        recoveredGradient - originalGradient;
                    if (fromBacktracking) {
                      ++lastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount;
                      lastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum +=
                          alpha;
                    }
                    if (fromMeritSelector) {
                      if (alpha >= 1.0 - 1e-12) {
                        ++lastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount;
                      } else {
                        ++lastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount;
                      }
                      lastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum +=
                          alpha;
                    }
                    ++lastFullEquivHybridTranslationRecoveryPolishAcceptedCount;
                  };
              const TranslationRecoveryResult recovery =
                  recoverMajorizedTranslationsWithStats(evaluated.x);
              if (!recovery.attempted) {
                return;
              }
              ++lastFullEquivHybridTranslationRecoveryPolishAttemptCount;
              if (!recovery.accepted || recovery.value.rows() != X.rows() ||
                  recovery.value.cols() != X.cols() ||
                  !recovery.value.allFinite()) {
                ++lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
                return;
              }
              const double recoveredCost =
                  problem.f(recovery.value) + modelConstant;
              const double recoveredGradient =
                  problem.RieGradNorm(recovery.value);
              if (!std::isfinite(recoveredCost) ||
                  !std::isfinite(recoveredGradient) ||
                  !std::isfinite(originalCost) ||
                  !std::isfinite(originalGradient) ||
                  recoveredCost >= originalCost - 1e-12) {
                ++lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
                return;
              }
              if (options
                      .fullEquivHybridTranslationRecoveryPolishMeritSelector) {
                std::vector<TranslationPolishCandidate> meritCandidates;
                auto addMeritCandidate =
                    [&](const Matrix &value, double cost, double gradient,
                        double alpha, bool partial) {
                      if (value.rows() != evaluated.x.rows() ||
                          value.cols() != evaluated.x.cols() ||
                          !value.allFinite() || !std::isfinite(cost) ||
                          !std::isfinite(gradient) ||
                          cost >= originalCost - 1e-12) {
                        return;
                      }
                      TranslationPolishCandidate candidate;
                      candidate.value = value;
                      candidate.cost = cost;
                      candidate.gradient = gradient;
                      candidate.alpha = alpha;
                      candidate.partial = partial;
                      meritCandidates.push_back(std::move(candidate));
                      ++lastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount;
                    };

                addMeritCandidate(recovery.value, recoveredCost,
                                  recoveredGradient, 1.0, false);
                const Matrix delta = recovery.value - evaluated.x;
                double alpha = 0.5;
                for (unsigned trial = 0;
                     trial <
                     options
                         .fullEquivHybridTranslationRecoveryPolishBacktrackingSteps;
                     ++trial, alpha *= 0.5) {
                  Matrix partial = evaluated.x;
                  for (unsigned pose = 0; pose < n; ++pose) {
                    const unsigned col = pose * (d + 1) + d;
                    partial.col(static_cast<int>(col)) =
                        evaluated.x.col(static_cast<int>(col)) +
                        alpha * delta.col(static_cast<int>(col));
                  }
                  const double partialCost =
                      partial.allFinite()
                          ? problem.f(partial) + modelConstant
                          : std::numeric_limits<double>::infinity();
                  const double partialGradient =
                      partial.allFinite()
                          ? problem.RieGradNorm(partial)
                          : std::numeric_limits<double>::infinity();
                  addMeritCandidate(partial, partialCost, partialGradient,
                                    alpha, true);
                }
                if (meritCandidates.empty()) {
                  ++lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
                  return;
                }

                double bestDecrease = 0.0;
                for (const auto &candidate : meritCandidates) {
                  bestDecrease =
                      std::max(bestDecrease, originalCost - candidate.cost);
                }
                if (!std::isfinite(bestDecrease) || bestDecrease <= 0.0) {
                  ++lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
                  return;
                }
                lastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum +=
                    bestDecrease;
                const double minRecoveryRatio =
                    std::min(1.0,
                             std::max(0.0, options
                                               .fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio));
                const double minRequiredDecrease =
                    minRecoveryRatio * bestDecrease;
                const TranslationPolishCandidate *selected = nullptr;
                for (const auto &candidate : meritCandidates) {
                  const double decrease = originalCost - candidate.cost;
                  if (decrease + 1e-12 < minRequiredDecrease) {
                    continue;
                  }
                  if (selected == nullptr ||
                      candidate.gradient < selected->gradient - 1e-12 ||
                      (candidate.gradient <= selected->gradient + 1e-12 &&
                       candidate.cost < selected->cost - 1e-12)) {
                    selected = &candidate;
                  }
                }
                if (selected == nullptr) {
                  ++lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
                  return;
                }
                lastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum +=
                    originalCost - selected->cost;
                acceptPolishedCandidate(selected->value, selected->cost,
                                        selected->gradient, selected->alpha,
                                        false, true);
                return;
              }
              if (options.fullEquivHybridTranslationRecoveryPolishGradientGuard) {
                const double maxIncreaseRatio = std::max(
                    0.0,
                    options
                        .fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio);
                const double gradientLimit =
                    originalGradient * (1.0 + maxIncreaseRatio) +
                    1e-12 * std::max(1.0, originalGradient);
                if (recoveredGradient > gradientLimit) {
                  if (options
                          .fullEquivHybridTranslationRecoveryPolishBacktracking) {
                    ++lastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount;
                    const Matrix delta = recovery.value - evaluated.x;
                    double alpha = 0.5;
                    for (unsigned trial = 0;
                         trial <
                         options
                             .fullEquivHybridTranslationRecoveryPolishBacktrackingSteps;
                         ++trial, alpha *= 0.5) {
                      Matrix partial = evaluated.x + alpha * delta;
                      ++lastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount;
                      if (partial.rows() != evaluated.x.rows() ||
                          partial.cols() != evaluated.x.cols() ||
                          !partial.allFinite()) {
                        continue;
                      }
                      const double partialCost =
                          problem.f(partial) + modelConstant;
                      const double partialGradient =
                          problem.RieGradNorm(partial);
                      if (std::isfinite(partialCost) &&
                          std::isfinite(partialGradient) &&
                          partialCost < originalCost - 1e-12 &&
                          partialGradient <= gradientLimit) {
                        acceptPolishedCandidate(partial, partialCost,
                                                partialGradient, alpha, true,
                                                false);
                        return;
                      }
                    }
                  }
                  ++lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
                  ++lastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount;
                  return;
                }
              }
              acceptPolishedCandidate(recovery.value, recoveredCost,
                                      recoveredGradient, 1.0, false, false);
            };
        auto evaluateFehLocalCandidate =
            [&](const Matrix &start, bool useSchwarz, bool useFehStep,
                FehLocalPortfolioSource source, bool countAsPortfolio,
                bool allowTranslationRecoveryPolish = true,
                bool allowTranslationRecoveryStepTrials = true) {
              if (countAsPortfolio) {
                ++lastFullEquivHybridLocalPortfolioCandidateCount;
              }
              FehLocalPortfolioCandidate evaluated;
              evaluated.source = source;
              if (start.rows() != X.rows() || start.cols() != X.cols() ||
                  !start.allFinite()) {
                return evaluated;
              }
              X = start;
              if (useSchwarz &&
                  !applyFullEquivHybridSchwarzPreSmoothing()) {
                return evaluated;
              }
              const Matrix solverStart = X;
              const double fInit = problem.f(solverStart) + modelConstant;
              const double gradInit = problem.RieGradNorm(solverStart);
              if (!std::isfinite(fInit) || !std::isfinite(gradInit)) {
                return evaluated;
              }
              const std::size_t schurAcceptedBefore =
                  lastFullEquivHybridSchurStepAcceptedCount;
              const bool allowStepTrialsForSource =
                  allowTranslationRecoveryStepTrials &&
                  (source == FehLocalPortfolioSource::Unsmoothed ||
                   source == FehLocalPortfolioSource::SchwarzFeh);
              const bool allowInitialGuessForSource =
                  problemQIsTrue && problemGIsTrue &&
                  (source == FehLocalPortfolioSource::Unsmoothed ||
                   source == FehLocalPortfolioSource::SchwarzFeh);
              evaluated.x =
                  useFehStep
                      ? makeFullEquivHybridSchurStep(
                            solverStart,
                            problemQIsTrue && problemGIsTrue &&
                                allowStepTrialsForSource,
                            allowInitialGuessForSource)
                      : solverStart;
              if (evaluated.x.rows() != solverStart.rows() ||
                  evaluated.x.cols() != solverStart.cols() ||
                  !evaluated.x.allFinite()) {
                return evaluated;
              }
              const double fOpt = problem.f(evaluated.x) + modelConstant;
              const double gradOpt = problem.RieGradNorm(evaluated.x);
              if (!std::isfinite(fOpt) || !std::isfinite(gradOpt)) {
                return evaluated;
              }
              const std::size_t schurAcceptedDelta =
                  lastFullEquivHybridSchurStepAcceptedCount -
                  schurAcceptedBefore;
              evaluated.result =
                  ROPTResult(true, fInit, gradInit, fOpt, gradOpt,
                             (evaluated.x - solverStart).norm() /
                                 (solverStart.norm() + 1e-12),
                             0.0);
              evaluated.result.rtrAcceptedIterations =
                  static_cast<unsigned>(schurAcceptedDelta);
              evaluated.cost = fOpt;
              evaluated.modelCost = fOpt;
              evaluated.gradient = gradOpt;
              evaluated.modelGradient = gradOpt;
              evaluated.ok = true;
              if (allowTranslationRecoveryPolish &&
                  !options
                       .fullEquivHybridTranslationRecoveryPolishSelectedOnly) {
                applyFehTranslationRecoveryPolish(evaluated);
              }
              return evaluated;
            };
        auto evaluateFehSimpleSurrogateCandidate =
            [&](const Matrix &start) {
              FehLocalPortfolioCandidate evaluated;
              evaluated.source = FehLocalPortfolioSource::SimpleSurrogate;
              if (!currentDpgoSimpleValid ||
                  dpgoSimpleQManual.rows() !=
                      static_cast<int>(n * (d + 1)) ||
                  dpgoSimpleQManual.cols() !=
                      static_cast<int>(n * (d + 1)) ||
                  currentDpgoSimpleG.rows() != static_cast<int>(r) ||
                  currentDpgoSimpleG.cols() !=
                      static_cast<int>(n * (d + 1))) {
                return evaluated;
              }
              try {
                setProblemQForLocalSolve(dpgoSimpleQManual, false);
                setProblemGForLocalSolve(currentDpgoSimpleG, false);
                Matrix solverStart = start;
                const Matrix proximalStart =
                    makeDpgoMajorizedProximalStartUnfiltered(start);
                if (proximalStart.rows() == start.rows() &&
                    proximalStart.cols() == start.cols() &&
                    proximalStart.allFinite()) {
                  solverStart = proximalStart;
                  ++lastAmmProximalStartCount;
                }
                const bool useConfiguredSchwarz =
                    options.fullEquivHybridSchwarzPreSmoothing &&
                    options.fullEquivHybridSchwarzSweeps > 0;
                evaluated = evaluateFehLocalCandidate(
                    solverStart, useConfiguredSchwarz, true,
                    FehLocalPortfolioSource::SimpleSurrogate, false, false);
                evaluated.modelCost = evaluated.cost;
                evaluated.modelGradient = evaluated.gradient;
              } catch (...) {
                restoreTrueLocalModel();
                throw;
              }
              restoreTrueLocalModel();
              if (!evaluated.ok) {
                return evaluated;
              }
              const double trueCost = problem.f(evaluated.x) + modelConstant;
              const double trueGradient = problem.RieGradNorm(evaluated.x);
              const double trueInitCost = problem.f(originalX) + modelConstant;
              const double trueInitGradient = problem.RieGradNorm(originalX);
              evaluated.ok = std::isfinite(trueCost) &&
                             std::isfinite(trueGradient) &&
                             std::isfinite(trueInitCost) &&
                             std::isfinite(trueInitGradient);
              if (!evaluated.ok) {
                return evaluated;
              }
              evaluated.cost = trueCost;
              evaluated.gradient = trueGradient;
              evaluated.result.fInit = trueInitCost;
              evaluated.result.gradNormInit = trueInitGradient;
              evaluated.result.fOpt = trueCost;
              evaluated.result.gradNormOpt = trueGradient;
              if (!options
                       .fullEquivHybridTranslationRecoveryPolishSelectedOnly) {
                applyFehTranslationRecoveryPolish(evaluated);
              }
              return evaluated;
            };
        auto fehSourceIsTrueLocal =
            [](FehLocalPortfolioSource source) {
              return source == FehLocalPortfolioSource::Unsmoothed ||
                     source == FehLocalPortfolioSource::SchwarzOnly ||
                     source == FehLocalPortfolioSource::SchwarzFeh;
            };
        auto recordFehMixedSurrogateSelectedSource =
            [&](FehLocalPortfolioSource source) {
              if (source == FehLocalPortfolioSource::SimpleSurrogate) {
                ++lastAmmMixedSurrogateSimpleSelectedCount;
                ammMixedSurrogateTrueLocalWinStreak = 0;
                ammMixedSurrogateConsecutiveSimpleSkips = 0;
              } else if (source ==
                         FehLocalPortfolioSource::LocalStateExtrapolated) {
                ++lastAmmMixedSurrogateExtrapolatedSelectedCount;
                ammMixedSurrogateTrueLocalWinStreak = 0;
                ammMixedSurrogateConsecutiveSimpleSkips = 0;
              } else if (fehSourceIsTrueLocal(source)) {
                ++lastAmmMixedSurrogateTrueLocalSelectedCount;
                ++ammMixedSurrogateTrueLocalWinStreak;
              } else {
                ++lastAmmMixedSurrogateOtherSelectedCount;
                ammMixedSurrogateTrueLocalWinStreak = 0;
                ammMixedSurrogateConsecutiveSimpleSkips = 0;
              }
            };

        FehLocalPortfolioSource candidateSource =
            FehLocalPortfolioSource::Unsmoothed;
        auto applySelectedOnlyFehTranslationRecoveryPolish = [&]() {
          if (!options.fullEquivHybridTranslationRecoveryPolish ||
              !options
                   .fullEquivHybridTranslationRecoveryPolishSelectedOnly) {
            return;
          }
          restoreTrueLocalModel();
          FehLocalPortfolioCandidate selected;
          selected.x = candidate;
          selected.result = result;
          selected.source = candidateSource;
          if (candidate.rows() != originalX.rows() ||
              candidate.cols() != originalX.cols() || !candidate.allFinite()) {
            return;
          }
          selected.cost = problem.f(candidate) + modelConstant;
          selected.gradient = problem.RieGradNorm(candidate);
          selected.modelCost = selected.cost;
          selected.modelGradient = selected.gradient;
          selected.ok =
              std::isfinite(selected.cost) && std::isfinite(selected.gradient);
          applyFehTranslationRecoveryPolish(selected);
          if (!selected.ok) {
            return;
          }
          candidate = selected.x;
          result = selected.result;
        };
        if (options.fullEquivHybridLocalPortfolio) {
          FehLocalPortfolioCandidate best =
              evaluateFehLocalCandidate(originalX, false, true,
                                        FehLocalPortfolioSource::Unsmoothed,
                                        true);
          X = originalX;
          if (options.fullEquivHybridSchwarzPreSmoothing &&
              options.fullEquivHybridSchwarzSweeps > 0) {
            FehLocalPortfolioCandidate schwarzOnly =
                evaluateFehLocalCandidate(
                    originalX, true, false,
                    FehLocalPortfolioSource::SchwarzOnly, true);
            if (fehCandidateIsBetter(schwarzOnly, best)) {
              best = schwarzOnly;
            }
            X = originalX;
            FehLocalPortfolioCandidate schwarzFeh =
                evaluateFehLocalCandidate(
                    originalX, true, true,
                    FehLocalPortfolioSource::SchwarzFeh, true);
            if (fehCandidateIsBetter(schwarzFeh, best)) {
              best = schwarzFeh;
            }
            X = originalX;
          }
          if (!best.ok) {
            restoreTrueLocalModel();
            return false;
          }
          if (best.source == FehLocalPortfolioSource::SchwarzOnly) {
            ++lastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount;
          } else if (best.source == FehLocalPortfolioSource::SchwarzFeh) {
            ++lastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount;
          } else {
            ++lastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount;
          }
          candidate = best.x;
          result = best.result;
          candidateSource = best.source;
        } else {
          if (!applyFullEquivHybridSchwarzPreSmoothing()) {
            restoreTrueLocalModel();
            return false;
          }
          const double fInit = problem.f(X) + modelConstant;
          const double gradInit = problem.RieGradNorm(X);
          const Matrix solverStart = X;
          const std::size_t schurAcceptedBefore =
              lastFullEquivHybridSchurStepAcceptedCount;
          candidate =
              makeFullEquivHybridSchurStep(X, problemQIsTrue && problemGIsTrue);
          const double fOpt = problem.f(candidate) + modelConstant;
          const double gradOpt = problem.RieGradNorm(candidate);
          if (!std::isfinite(fInit) || !std::isfinite(gradInit) ||
              !std::isfinite(fOpt) || !std::isfinite(gradOpt)) {
            restoreTrueLocalModel();
            return false;
          }
          result = ROPTResult(true, fInit, gradInit, fOpt, gradOpt,
                              (candidate - solverStart).norm() /
                                  (solverStart.norm() + 1e-12),
                              0.0);
          result.rtrAcceptedIterations =
              static_cast<unsigned>(
                  lastFullEquivHybridSchurStepAcceptedCount -
                  schurAcceptedBefore);
          FehLocalPortfolioCandidate evaluated;
          evaluated.x = candidate;
          evaluated.result = result;
          evaluated.source = FehLocalPortfolioSource::Unsmoothed;
          evaluated.cost = fOpt;
          evaluated.modelCost = fOpt;
          evaluated.gradient = gradOpt;
          evaluated.modelGradient = gradOpt;
          evaluated.ok = true;
          if (!options
                   .fullEquivHybridTranslationRecoveryPolishSelectedOnly) {
            applyFehTranslationRecoveryPolish(evaluated);
          }
          candidate = evaluated.x;
          result = evaluated.result;
          candidateSource = FehLocalPortfolioSource::Unsmoothed;
        }
        if (options.localStateExtrapolation && hasPreviousX &&
            previousX.rows() == originalX.rows() &&
            previousX.cols() == originalX.cols()) {
          FehLocalPortfolioCandidate incumbent;
          incumbent.x = candidate;
          incumbent.result = result;
          incumbent.source = candidateSource;
          incumbent.cost = problem.f(candidate) + modelConstant;
          incumbent.gradient = problem.RieGradNorm(candidate);
          incumbent.ok = candidate.rows() == originalX.rows() &&
                         candidate.cols() == originalX.cols() &&
                         candidate.allFinite() &&
                         std::isfinite(incumbent.cost) &&
                         std::isfinite(incumbent.gradient);

          const std::vector<double> gammas = extrapolationGammas();
          FehLocalPortfolioCandidate bestExtrapolated;
          std::size_t attemptedExtrapolations = 0;
          const bool useConfiguredSchwarz =
              options.fullEquivHybridSchwarzPreSmoothing &&
              options.fullEquivHybridSchwarzSweeps > 0;
          for (const double gamma : gammas) {
            const Matrix extrapolatedStart =
                makeExtrapolatedStart(originalX, previousX, gamma);
            FehLocalPortfolioCandidate extrapolated =
                evaluateFehLocalCandidate(
                    extrapolatedStart, useConfiguredSchwarz, true,
                    FehLocalPortfolioSource::LocalStateExtrapolated, false,
                    true, false);
            ++attemptedExtrapolations;
            if (fehCandidateIsBetter(extrapolated, bestExtrapolated)) {
              bestExtrapolated = extrapolated;
            }
            X = originalX;
          }
          if (fehCandidateIsBetter(bestExtrapolated, incumbent)) {
            candidate = bestExtrapolated.x;
            result = bestExtrapolated.result;
            candidateSource = bestExtrapolated.source;
            lastExtrapolationAcceptedCount = 1;
            lastExtrapolationRejectedCount =
                attemptedExtrapolations > 0 ? attemptedExtrapolations - 1 : 0;
          } else {
            lastExtrapolationRejectedCount = attemptedExtrapolations;
          }
        }
        if (options.scheme == ManualDpgoMmScheme::AMM &&
            options.ammDpgoSurrogateParity &&
            options.ammDpgoMixedSurrogatePortfolio) {
          const bool eligibleToSkipSimple =
              options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak > 0 &&
              ammMixedSurrogateTrueLocalWinStreak >=
                  static_cast<std::size_t>(
                      options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak);
          bool forceSimpleRefresh = false;
          bool skipSimpleSurrogate = eligibleToSkipSimple;
          if (eligibleToSkipSimple &&
              options.ammMixedSurrogateForceSimpleEverySkippedRounds > 0 &&
              ammMixedSurrogateConsecutiveSimpleSkips >=
                  static_cast<std::size_t>(
                      options
                          .ammMixedSurrogateForceSimpleEverySkippedRounds)) {
            forceSimpleRefresh = true;
            skipSimpleSurrogate = false;
            ++lastAmmMixedSurrogateSimpleForcedRefreshCount;
          }

          FehLocalPortfolioCandidate incumbent;
          incumbent.x = candidate;
          incumbent.result = result;
          incumbent.source = candidateSource;
          incumbent.cost = problem.f(candidate) + modelConstant;
          incumbent.gradient = problem.RieGradNorm(candidate);
          incumbent.modelCost = incumbent.cost;
          incumbent.modelGradient = incumbent.gradient;
          incumbent.ok = candidate.rows() == originalX.rows() &&
                         candidate.cols() == originalX.cols() &&
                         candidate.allFinite() &&
                         std::isfinite(incumbent.cost) &&
                         std::isfinite(incumbent.gradient);

          ++lastAmmMixedSurrogateCandidateCount;
          if (skipSimpleSurrogate) {
            ++lastAmmMixedSurrogateSimpleSkippedCount;
            recordFehMixedSurrogateSelectedSource(incumbent.source);
            ++ammMixedSurrogateConsecutiveSimpleSkips;
          } else {
            if (forceSimpleRefresh) {
              ammMixedSurrogateConsecutiveSimpleSkips = 0;
            }
            FehLocalPortfolioCandidate simpleCandidate =
                evaluateFehSimpleSurrogateCandidate(originalX);
            X = originalX;
            if (fehCandidateIsBetter(simpleCandidate, incumbent)) {
              candidate = simpleCandidate.x;
              result = simpleCandidate.result;
              candidateSource = simpleCandidate.source;
              recordFehMixedSurrogateSelectedSource(candidateSource);
            } else {
              if (fehSourceIsTrueLocal(incumbent.source)) {
                ++lastAmmMixedSurrogateTrueLocalAcceptedCount;
              }
              recordFehMixedSurrogateSelectedSource(incumbent.source);
            }
          }
        }
        applySelectedOnlyFehTranslationRecoveryPolish();
        if (options.fullEquivHybridFinalPolishRtr) {
          manualOptimizer.setVerbose(options.verbose);
          manualOptimizer.setTrustRegionTolerance(options.trustRegionTolerance);
          manualOptimizer.setTrustRegionIterations(options.trustRegionIterations);
          manualOptimizer.setTrustRegionAcceptedIterations(
              options.trustRegionAcceptedIterations);
          manualOptimizer.setTrustRegionMaxInnerIterations(
              options.trustRegionMaxInnerIterations);
          manualOptimizer.setTrustRegionInitialRadius(
              options.trustRegionInitialRadius);
          candidate = manualOptimizer.optimize(candidate);
          result = manualOptimizer.getOptResult();
        }
      } else {
        if (options.localSolver == ManualDpgoMmLocalSolver::ManualFull) {
          enum class ManualFullCandidateSource {
            TrueLocal,
            LocalStateExtrapolated,
            AmmPredicted,
          };

          struct ManualFullCandidate {
            Matrix x;
            ROPTResult result;
            double cost{std::numeric_limits<double>::infinity()};
            double modelCost{std::numeric_limits<double>::infinity()};
            double gradient{std::numeric_limits<double>::infinity()};
            double modelGradient{std::numeric_limits<double>::infinity()};
            ManualFullCandidateSource source{
                ManualFullCandidateSource::TrueLocal};
            bool ok{false};
          };

          auto configureManualFullOptimizer = [&]() {
            manualOptimizer.setVerbose(options.verbose);
            manualOptimizer.setTrustRegionTolerance(
                options.trustRegionTolerance);
            manualOptimizer.setTrustRegionIterations(
                options.trustRegionIterations);
            manualOptimizer.setTrustRegionAcceptedIterations(
                options.trustRegionAcceptedIterations);
            manualOptimizer.setTrustRegionMaxInnerIterations(
                options.trustRegionMaxInnerIterations);
            manualOptimizer.setTrustRegionInitialRadius(
                options.trustRegionInitialRadius);
          };

          auto manualFullCandidateIsBetter =
              [&](const ManualFullCandidate &candidateResult,
                  const ManualFullCandidate &bestResult) {
                if (!candidateResult.ok) {
                  return false;
                }
                if (!bestResult.ok) {
                  return true;
                }
                const double tolerance =
                    std::max(0.0, options.localCandidateCostTieTolerance);
                if (candidateResult.cost < bestResult.cost - tolerance) {
                  return true;
                }
                return tolerance > 0.0 &&
                       candidateResult.cost <= bestResult.cost + tolerance &&
                       candidateResult.gradient <
                           bestResult.gradient - 1e-12;
          };

          auto manualFullCandidatePassesLocalMerit =
              [&](const ManualFullCandidate &candidateResult,
                  const ManualFullCandidate &plainResult) {
                if (!candidateResult.ok || !plainResult.ok) {
                  return false;
                }
                const double tolerance =
                    std::max({0.0, options.localCandidateCostTieTolerance,
                              options.ammLocalMeritCostTieTolerance});
                if (candidateResult.cost < plainResult.cost - tolerance) {
                  return true;
                }
                return candidateResult.cost <= plainResult.cost + tolerance &&
                       candidateResult.gradient <
                           plainResult.gradient - 1e-12;
          };

          auto solveManualFullAt =
              [&](const Matrix &start, const SparseMatrix *solveG,
                  ManualFullCandidateSource source) {
                ManualFullCandidate candidateResult;
                candidateResult.source = source;
                if (start.rows() != X.rows() || start.cols() != X.cols() ||
                    !start.allFinite()) {
                  return candidateResult;
                }

                try {
                  if (solveG != nullptr) {
                    setProblemGForLocalSolve(*solveG, false);
                  }
                  configureManualFullOptimizer();
                  Matrix solverStart = start;
                  if (options.ammProximalStart && solveG != nullptr) {
                    const Matrix proximalStart =
                        makeMajorizedProximalStart(start);
                    if (proximalStart.rows() == start.rows() &&
                        proximalStart.cols() == start.cols() &&
                        proximalStart.allFinite()) {
                      solverStart = proximalStart;
                      ++lastAmmProximalStartCount;
                    }
                  }
                  candidateResult.x =
                      manualOptimizer.optimize(solverStart);
                  candidateResult.result = manualOptimizer.getOptResult();
                  if (candidateResult.x.rows() != start.rows() ||
                      candidateResult.x.cols() != start.cols() ||
                      !candidateResult.x.allFinite()) {
                    if (solveG != nullptr) {
                      restoreTrueLocalModel();
                    }
                    return candidateResult;
                  }
                  candidateResult.modelCost =
                      problem.f(candidateResult.x) + modelConstant;
                  candidateResult.modelGradient =
                      problem.RieGradNorm(candidateResult.x);
                  if (solveG != nullptr) {
                    restoreTrueLocalModel();
                    candidateResult.cost =
                        problem.f(candidateResult.x) + modelConstant;
                    candidateResult.gradient =
                        problem.RieGradNorm(candidateResult.x);
                  } else {
                    candidateResult.cost = candidateResult.modelCost;
                    candidateResult.gradient = candidateResult.modelGradient;
                  }
                  candidateResult.ok =
                      std::isfinite(candidateResult.cost) &&
                      std::isfinite(candidateResult.gradient) &&
                      std::isfinite(candidateResult.modelCost) &&
                      std::isfinite(candidateResult.modelGradient);
                  if (candidateResult.ok) {
                    candidateResult.result.fOpt = candidateResult.cost;
                    candidateResult.result.gradNormOpt =
                        candidateResult.gradient;
                  }
                } catch (...) {
                  if (solveG != nullptr) {
                    restoreTrueLocalModel();
                  }
                  throw;
                }
                return candidateResult;
              };

          ManualFullCandidate plain = solveManualFullAt(
              X, nullptr, ManualFullCandidateSource::TrueLocal);
          ManualFullCandidate best = plain;

          if (options.manualFullPortfolio &&
              options.localStateExtrapolation && hasPreviousX &&
              previousX.rows() == originalX.rows() &&
              previousX.cols() == originalX.cols()) {
            const std::vector<double> gammas = extrapolationGammas();
            ManualFullCandidate bestExtrapolated;
            std::size_t attemptedExtrapolations = 0;
            for (const double gamma : gammas) {
              const Matrix extrapolatedStart =
                  makeExtrapolatedStart(originalX, previousX, gamma);
              ManualFullCandidate extrapolated = solveManualFullAt(
                  extrapolatedStart, nullptr,
                  ManualFullCandidateSource::LocalStateExtrapolated);
              ++attemptedExtrapolations;
              if (manualFullCandidateIsBetter(extrapolated,
                                               bestExtrapolated)) {
                bestExtrapolated = extrapolated;
              }
            }
            if (manualFullCandidateIsBetter(bestExtrapolated, best)) {
              best = bestExtrapolated;
              lastExtrapolationAcceptedCount = 1;
              lastExtrapolationRejectedCount =
                  attemptedExtrapolations > 0 ? attemptedExtrapolations - 1 : 0;
            } else {
              lastExtrapolationRejectedCount = attemptedExtrapolations;
            }
          }

          if (options.scheme == ManualDpgoMmScheme::AMM) {
            double trueCurrentCost = std::numeric_limits<double>::infinity();
            double trueCurrentGradient =
                std::numeric_limits<double>::infinity();
            if (options.fusedCandidateEvaluation) {
              const auto fusedCurrent = problem.fAndRieGradNorm(originalX);
              trueCurrentCost = fusedCurrent.first + modelConstant;
              trueCurrentGradient = fusedCurrent.second;
            } else {
              trueCurrentCost = problem.f(originalX) + modelConstant;
              trueCurrentGradient = problem.RieGradNorm(originalX);
            }
            const AmmStepState ammStep = beginAmmStep(trueCurrentCost);
            ManualDpgoMmAmmTrace trace;
            trace.localIter = ammLocalIter;
            trace.fobj = trueCurrentCost;
            trace.trueFobj = trueCurrentCost;
            trace.baselineSurrogateFobj = trueCurrentCost;
            trace.baselineSurrogateGradNorm = trueCurrentGradient;
            trace.localAcceptedIterationBudget =
                options.trustRegionAcceptedIterations;
            trace.numOscillations = ammStep.numOscillations;
            trace.F0 = ammFk0;
            trace.F1 = ammFk1;
            trace.gamma = ammStep.gamma;
            const double refinedDenominator =
                std::max(1e-12, std::abs(trueCurrentCost));
            trace.refinedRatio =
                (trueCurrentGradient * trueCurrentGradient) /
                refinedDenominator;
            trace.refined = options.trustRegionIterations > 0 &&
                            options.trustRegionMaxInnerIterations > 0;
            trace.softRestartHits0 = ammSoftRestartHits0;
            trace.softRestartHits1 = ammSoftRestartHits1;

            bool hardRestart = false;
            bool softRestart = false;
            bool phiFallback = false;
            bool acceleratedAccepted = false;
            bool localMeritRejected = false;
            bool attemptedAcceleration = false;
            bool cooldownSkipped = false;
            bool canAttemptAcceleration = ammStep.canAccelerate;
            if (canAttemptAcceleration && ammCooldown > 0) {
              --ammCooldown;
              canAttemptAcceleration = false;
              cooldownSkipped = true;
            }

            if (canAttemptAcceleration && hasPreviousX && hasPreviousG &&
                previousX.rows() == originalX.rows() &&
                previousX.cols() == originalX.cols() &&
                previousG.rows() == historyGForThisStep.rows() &&
                previousG.cols() == historyGForThisStep.cols()) {
              ManualFullCandidate accelerated;
              ManualDpgoMmAmmTrace acceleratedTrace;
              std::size_t attemptedAmmGammas = 0;
              for (const double gammaScale : ammGammaScales()) {
                const double gamma = ammStep.gamma * gammaScale;
                if (!std::isfinite(gamma) || gamma <= 0.0) {
                  continue;
                }
                const Matrix acceleratedStart =
                    makeExtrapolatedStart(originalX, previousX, gamma);
                SparseMatrix predictedG =
                    historyGForThisStep +
                    gamma * (historyGForThisStep - previousG);
                predictedG.makeCompressed();
                ManualFullCandidate trial = solveManualFullAt(
                    acceleratedStart, &predictedG,
                    ManualFullCandidateSource::AmmPredicted);
                ++attemptedAmmGammas;
                ManualDpgoMmAmmTrace trialTrace = trace;
                trialTrace.valid = true;
                trialTrace.gamma = gamma;
                trialTrace.GkhInitial =
                    trial.ok ? trial.modelCost
                             : std::numeric_limits<double>::infinity();
                trialTrace.surrogateGkhInitial = trialTrace.GkhInitial;
                trialTrace.GkhAfterRestartCheck = trialTrace.GkhInitial;
                trialTrace.surrogateGkhAfterRestartCheck =
                    trialTrace.surrogateGkhInitial;
                trialTrace.GkAfterAccelerated =
                    trial.ok ? trial.cost
                             : std::numeric_limits<double>::infinity();
                trialTrace.surrogateGkAfterAccelerated =
                    trial.ok ? trial.modelCost
                             : trialTrace.GkAfterAccelerated;
                trialTrace.finalGk = trialTrace.GkAfterAccelerated;
                trialTrace.surrogateFinalGk =
                    trialTrace.surrogateGkAfterAccelerated;
                trialTrace.localAcceptedIterations =
                    trial.result.rtrAcceptedIterations;
                if (manualFullCandidateIsBetter(trial, accelerated)) {
                  accelerated = trial;
                  acceleratedTrace = trialTrace;
                }
              }
              attemptedAcceleration = attemptedAmmGammas > 0;

              if (accelerated.ok) {
                ManualFullCandidate selected = accelerated;
                trace = acceleratedTrace;
                acceleratedAccepted = true;
                const double acceleratedAmmCost = accelerated.cost;
                const double plainAmmCost = best.cost;
                hardRestart = acceleratedAmmCost > ammFk0;
                softRestart =
                    (acceleratedAmmCost > ammFk1 &&
                     ammSoftRestartHits0 >= options.ammMaxSoftRestartHits0) ||
                    (acceleratedAmmCost > trueCurrentCost &&
                     ammSoftRestartHits1 > options.ammMaxSoftRestartHits1);
                if ((hardRestart || softRestart) && best.ok) {
                  selected = best;
                  acceleratedAccepted = false;
                } else if (options.ammLocalMeritFilter && best.ok &&
                           !manualFullCandidatePassesLocalMerit(
                               accelerated, best)) {
                  selected = best;
                  acceleratedAccepted = false;
                  localMeritRejected = true;
                } else if (best.ok) {
                  const double lhs = ammFk0 - acceleratedAmmCost;
                  const double rhs = options.ammPhi * (ammFk0 - plainAmmCost);
                  if (std::isfinite(lhs) && std::isfinite(rhs) &&
                      lhs < rhs) {
                    selected = best;
                    acceleratedAccepted = false;
                    phiFallback = true;
                  }
                }
                trace.hardRestart = hardRestart;
                trace.softRestart = softRestart;
                trace.phiFallback = phiFallback;
                trace.phiLhs = ammFk0 - acceleratedAmmCost;
                trace.phiRhs = options.ammPhi * (ammFk0 - plainAmmCost);
                if (!acceleratedAccepted && selected.ok) {
                  trace.localAcceptedIterations =
                      selected.result.rtrAcceptedIterations;
                  trace.finalGk = selected.cost;
                  trace.surrogateFinalGk = selected.modelCost;
                }
                best = selected;
              }
            }

            const bool rejectedAcceleration =
                attemptedAcceleration && !acceleratedAccepted;
            if (acceleratedAccepted) {
              ammCooldown = 0;
            } else if (rejectedAcceleration &&
                       options.ammCooldownAfterRejected > 0) {
              ammCooldown = options.ammCooldownAfterRejected;
            }

            const double nextS = cooldownSkipped ? 1.0 : ammStep.nextS;
            finishAmmStep(nextS, hardRestart, hardRestart || softRestart);
            lastAmmAcceleratedAcceptedCount =
                acceleratedAccepted ? 1 : 0;
            lastAmmHardRestartCount = hardRestart ? 1 : 0;
            lastAmmSoftRestartCount = softRestart ? 1 : 0;
            lastAmmRestartCount = (hardRestart || softRestart) ? 1 : 0;
            lastAmmPhiFallbackCount = phiFallback ? 1 : 0;
            lastAmmLocalMeritRejectedCount =
                localMeritRejected ? 1 : 0;
            lastAmmSkippedCount = cooldownSkipped ? 1 : 0;
            if (trace.valid) {
              lastAmmTrace = trace;
            }
            ++ammLocalIter;
          }

          if (!best.ok) {
            restoreTrueLocalModel();
            return false;
          }
          candidate = best.x;
          result = best.result;
        } else {
          const Matrix fullSolverStart =
              options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid &&
                      options.fullEquivHybridSchurWarmStart
                  ? makeFullEquivHybridSchurWarmStart(X)
                  : X;
          manualOptimizer.setVerbose(options.verbose);
          manualOptimizer.setTrustRegionTolerance(options.trustRegionTolerance);
          manualOptimizer.setTrustRegionIterations(options.trustRegionIterations);
          manualOptimizer.setTrustRegionAcceptedIterations(
              options.trustRegionAcceptedIterations);
          manualOptimizer.setTrustRegionMaxInnerIterations(
              options.trustRegionMaxInnerIterations);
          manualOptimizer.setTrustRegionInitialRadius(
              options.trustRegionInitialRadius);
          candidate = manualOptimizer.optimize(fullSolverStart);
          result = manualOptimizer.getOptResult();
        }
      }
    } else {
      enum class CandidateSource {
        Unknown,
        SimpleSurrogate,
        EdgeTightQuadratic,
        TrueLocal,
        LocalStateExtrapolated,
        GExtrapolated,
        CoupledExtrapolated,
        Anderson,
        Squarem,
        VariableProjectedSchur,
        BoundaryProximal,
        ReducedInterface,
      };

      struct CandidateResult {
        Matrix x;
        ROPTResult result;
        double cost{std::numeric_limits<double>::infinity()};
        double modelCost{std::numeric_limits<double>::infinity()};
        double gradient{std::numeric_limits<double>::infinity()};
        double modelGradient{std::numeric_limits<double>::infinity()};
        double boundaryCost{std::numeric_limits<double>::quiet_NaN()};
        std::size_t adaptiveRefinements{0};
        CandidateSource source{CandidateSource::Unknown};
        bool gradientEvaluated{false};
        bool trueEvaluationDeferred{false};
        bool ok{false};
      };

      struct CandidateEvaluationCacheEntry {
        std::size_t modelVersion{0};
        std::size_t fingerprint{0};
        int rows{0};
        int cols{0};
        CandidateResult result;
      };

      std::vector<CandidateEvaluationCacheEntry> candidateEvaluationCache;

      auto recordMixedSurrogateSelectedSource =
          [&](CandidateSource source) {
            if (!options.ammDpgoSurrogateParity ||
                !options.ammDpgoMixedSurrogatePortfolio) {
              return;
            }
            switch (source) {
              case CandidateSource::SimpleSurrogate:
                ++lastAmmMixedSurrogateSimpleSelectedCount;
                break;
              case CandidateSource::TrueLocal:
                ++lastAmmMixedSurrogateTrueLocalSelectedCount;
                break;
              case CandidateSource::LocalStateExtrapolated:
                ++lastAmmMixedSurrogateExtrapolatedSelectedCount;
                break;
              case CandidateSource::Unknown:
              case CandidateSource::EdgeTightQuadratic:
              case CandidateSource::GExtrapolated:
              case CandidateSource::CoupledExtrapolated:
              case CandidateSource::Anderson:
              case CandidateSource::Squarem:
              case CandidateSource::VariableProjectedSchur:
              case CandidateSource::BoundaryProximal:
              case CandidateSource::ReducedInterface:
                ++lastAmmMixedSurrogateOtherSelectedCount;
                break;
            }
          };

      const auto isAcceleratedCandidateSource = [](CandidateSource source) {
        switch (source) {
          case CandidateSource::LocalStateExtrapolated:
          case CandidateSource::GExtrapolated:
          case CandidateSource::CoupledExtrapolated:
          case CandidateSource::Anderson:
          case CandidateSource::Squarem:
            return true;
          case CandidateSource::Unknown:
          case CandidateSource::SimpleSurrogate:
          case CandidateSource::EdgeTightQuadratic:
          case CandidateSource::TrueLocal:
          case CandidateSource::VariableProjectedSchur:
          case CandidateSource::BoundaryProximal:
          case CandidateSource::ReducedInterface:
            return false;
        }
        return false;
      };

      auto boundarySafeguardRejectsAcceleratedCandidate =
          [&](const CandidateResult &candidateResult,
              const CandidateResult &bestResult) {
            if (options.mmSafeguard !=
                    ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary ||
                !isAcceleratedCandidateSource(candidateResult.source)) {
              return false;
            }
            if (!std::isfinite(candidateResult.boundaryCost) ||
                !std::isfinite(bestResult.boundaryCost)) {
              return true;
            }
            const double tolerance =
                std::max({1e-12, options.localCandidateCostTieTolerance,
                          options.ammLocalMeritCostTieTolerance});
            return candidateResult.boundaryCost >
                   bestResult.boundaryCost + tolerance;
          };

      auto ensureCandidateGradient = [&](CandidateResult &candidateResult) {
        if (!candidateResult.ok || candidateResult.gradientEvaluated) {
          return candidateResult.ok;
        }
        if (candidateResult.x.rows() != X.rows() ||
            candidateResult.x.cols() != X.cols() ||
            !candidateResult.x.allFinite()) {
          candidateResult.ok = false;
          return false;
        }
        {
          ScopedOptionalSecondsAccumulator profileTimer(
              options.profileOptimizer
                  ? &optimizerProfile.lazyCandidateGradientEvaluationSec
                  : nullptr);
          candidateResult.gradient = problem.RieGradNorm(candidateResult.x);
        }
        if (options.profileOptimizer) {
          ++optimizerProfile.lazyCandidateGradientEvaluationCount;
        }
        candidateResult.modelGradient = candidateResult.gradient;
        candidateResult.gradientEvaluated =
            std::isfinite(candidateResult.gradient);
        candidateResult.ok =
            candidateResult.ok && candidateResult.gradientEvaluated;
        return candidateResult.ok;
      };

      auto candidateIsBetter = [&](CandidateResult &candidateResult,
                                   CandidateResult &bestResult) {
        if (!candidateResult.ok) {
          return false;
        }
        if (!bestResult.ok) {
          return true;
        }
        if (boundarySafeguardRejectsAcceleratedCandidate(candidateResult,
                                                        bestResult)) {
          return false;
        }
        const double tolerance =
            std::max(0.0, options.localCandidateCostTieTolerance);
        if (candidateResult.cost < bestResult.cost - tolerance) {
          return true;
        }
        if (tolerance <= 0.0 ||
            candidateResult.cost > bestResult.cost + tolerance) {
          return false;
        }
        if (!ensureCandidateGradient(candidateResult) ||
            !ensureCandidateGradient(bestResult)) {
          return false;
        }
        return candidateResult.gradient < bestResult.gradient - 1e-12;
      };

      auto ammCandidatePassesLocalMerit =
          [&](CandidateResult &candidateResult,
              CandidateResult &plainResult) {
            if (!candidateResult.ok || !plainResult.ok) {
              return false;
            }
            if (options.mmSafeguard ==
                ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary) {
              if (!std::isfinite(candidateResult.boundaryCost) ||
                  !std::isfinite(plainResult.boundaryCost)) {
                return false;
              }
              const double boundaryTolerance =
                  std::max({1e-12, options.localCandidateCostTieTolerance,
                            options.ammLocalMeritCostTieTolerance});
              if (candidateResult.boundaryCost >
                  plainResult.boundaryCost + boundaryTolerance) {
                return false;
              }
            }
            const double tolerance =
                std::max({0.0, options.localCandidateCostTieTolerance,
                          options.ammLocalMeritCostTieTolerance});
            if (candidateResult.cost < plainResult.cost - tolerance) {
              return true;
            }
            if (candidateResult.cost > plainResult.cost + tolerance) {
              return false;
            }
            if (!ensureCandidateGradient(candidateResult) ||
                !ensureCandidateGradient(plainResult)) {
              return false;
            }
            return candidateResult.gradient < plainResult.gradient - 1e-12;
          };

      auto evaluateConstructedCandidate = [&](CandidateResult evaluated) {
        if (evaluated.x.rows() != X.rows() ||
            evaluated.x.cols() != X.cols() || !evaluated.x.allFinite()) {
          return evaluated;
        }
        if (options.profileOptimizer) {
          ++optimizerProfile.reducedCandidateEvaluationCount;
        }
        const bool useEvaluationCache =
            options.candidateEvaluationCache &&
            !options.lazyCandidateGradientEvaluation;
        std::size_t fingerprint = 0;
        if (useEvaluationCache) {
          if (options.profileOptimizer) {
            ++optimizerProfile.candidateEvaluationCacheRequestCount;
          }
          fingerprint = matrixExactFingerprint(evaluated.x);
          for (const auto &entry : candidateEvaluationCache) {
            if (entry.modelVersion == candidateEvaluationModelVersion &&
                entry.rows == evaluated.x.rows() &&
                entry.cols == evaluated.x.cols() &&
                entry.fingerprint == fingerprint &&
                matricesExactlyEqual(entry.result.x, evaluated.x)) {
              if (options.profileOptimizer) {
                ++optimizerProfile.candidateEvaluationCacheHitCount;
              }
              return entry.result;
            }
          }
          if (options.profileOptimizer) {
            ++optimizerProfile.candidateEvaluationCacheMissCount;
          }
        }
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer
                ? &optimizerProfile.reducedCandidateEvaluationSec
                : nullptr);
        if (options.lazyCandidateGradientEvaluation) {
          evaluated.cost = problem.f(evaluated.x);
          evaluated.gradient = std::numeric_limits<double>::infinity();
          evaluated.modelGradient = evaluated.gradient;
          evaluated.gradientEvaluated = false;
        } else if (options.fusedCandidateEvaluation) {
          const auto fused = problem.fAndRieGradNorm(evaluated.x);
          evaluated.cost = fused.first;
          evaluated.gradient = fused.second;
          evaluated.modelGradient = evaluated.gradient;
          evaluated.gradientEvaluated = std::isfinite(evaluated.gradient);
        } else {
          evaluated.cost = problem.f(evaluated.x);
          evaluated.gradient = problem.RieGradNorm(evaluated.x);
          evaluated.modelGradient = evaluated.gradient;
          evaluated.gradientEvaluated = std::isfinite(evaluated.gradient);
        }
        evaluated.modelCost = evaluated.cost;
        if (options.mmSafeguard ==
            ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary) {
          evaluated.boundaryCost = boundaryEdgeCost(evaluated.x);
        }
        evaluated.ok = std::isfinite(evaluated.cost) &&
                       (options.lazyCandidateGradientEvaluation ||
                        std::isfinite(evaluated.gradient)) &&
                       (options.mmSafeguard !=
                            ManualDpgoMmMmSafeguard::
                                LocalSurrogatePlusBoundary ||
                        std::isfinite(evaluated.boundaryCost));
        if (useEvaluationCache && evaluated.ok) {
          CandidateEvaluationCacheEntry entry;
          entry.modelVersion = candidateEvaluationModelVersion;
          entry.fingerprint = fingerprint;
          entry.rows = evaluated.x.rows();
          entry.cols = evaluated.x.cols();
          entry.result = evaluated;
          candidateEvaluationCache.push_back(entry);
        }
        return evaluated;
      };

      auto evaluateCurrentModelCandidate = [&](const Matrix &value) {
        CandidateResult evaluated;
        {
          ScopedOptionalSecondsAccumulator constructionTimer(
              options.profileOptimizer
                  ? &optimizerProfile.candidateConstructionSec
                  : nullptr);
          evaluated.x = value;
        }
        if (options.profileOptimizer) {
          ++optimizerProfile.candidateObjectConstructionCount;
        }
        return evaluateConstructedCandidate(std::move(evaluated));
      };

      auto evaluateMovedCurrentModelCandidate = [&](Matrix &&value) {
        CandidateResult evaluated;
        {
          ScopedOptionalSecondsAccumulator constructionTimer(
              options.profileOptimizer
                  ? &optimizerProfile.candidateConstructionSec
                  : nullptr);
          evaluated.x = std::move(value);
        }
        if (options.profileOptimizer) {
          ++optimizerProfile.candidateObjectConstructionCount;
          ++optimizerProfile.candidateObjectReuseCount;
        }
        return evaluateConstructedCandidate(std::move(evaluated));
      };

      auto materializeDeferredTrueEvaluation =
          [&](CandidateResult &candidateResult) {
            if (!candidateResult.ok) {
              return false;
            }
            if (!candidateResult.trueEvaluationDeferred) {
              return candidateResult.ok;
            }
            if (candidateResult.x.rows() != X.rows() ||
                candidateResult.x.cols() != X.cols() ||
                !candidateResult.x.allFinite()) {
              candidateResult.ok = false;
              candidateResult.trueEvaluationDeferred = false;
              return false;
            }
            const double surrogateCost = candidateResult.modelCost;
            const double surrogateGradient = candidateResult.modelGradient;
            restoreTrueLocalModel();
            CandidateResult trueEvaluation =
                evaluateCurrentModelCandidate(candidateResult.x);
            candidateResult.cost = trueEvaluation.cost;
            candidateResult.gradient = trueEvaluation.gradient;
            candidateResult.gradientEvaluated =
                trueEvaluation.gradientEvaluated;
            candidateResult.boundaryCost = trueEvaluation.boundaryCost;
            candidateResult.modelCost = surrogateCost;
            candidateResult.modelGradient = surrogateGradient;
            candidateResult.trueEvaluationDeferred = false;
            candidateResult.ok =
                candidateResult.ok && trueEvaluation.ok &&
                std::isfinite(candidateResult.cost) &&
                (options.lazyCandidateGradientEvaluation ||
                 std::isfinite(candidateResult.gradient));
            return candidateResult.ok;
          };

      auto runReduced = [&](const Matrix &start, int cgIterations,
                            ManualDpgoMmReducedRotationPreconditioner
                                preconditioner) {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer ? &optimizerProfile.reducedSolverSec
                                     : nullptr);
        if (options.profileOptimizer) {
          ++optimizerProfile.reducedSolveCount;
        }
        reducedOptimizer.setVerbose(options.verbose);
        reducedOptimizer.setTrustRegionTolerance(options.trustRegionTolerance);
        reducedOptimizer.setTrustRegionIterations(options.trustRegionIterations);
        reducedOptimizer.setTrustRegionAcceptedIterations(
            options.trustRegionAcceptedIterations);
        reducedOptimizer.setTrustRegionMaxInnerIterations(cgIterations);
        reducedOptimizer.setTruncatedCgRelativeTolerance(
            options.reducedRotationTcgRelativeTolerance);
        reducedOptimizer.setTrustRegionInitialRadius(
            options.trustRegionInitialRadius);
        reducedOptimizer.setRecordResultStats(
            options.recordLocalModelDiagnostics);
        reducedOptimizer.setProfileRuntime(options.profileOptimizer);
        reducedOptimizer.setValidateTranslationRecoveryCost(
            options.recordLocalModelDiagnostics);
        reducedOptimizer.setUseDirectObjectiveEvaluation(
            options.reducedRotationDirectObjective);
        reducedOptimizer.setUseCurvatureCauchyCandidate(
            options.reducedRotationCurvatureCauchyCandidate);
        reducedOptimizer.setUseCurvatureCauchyFallbackCandidate(
            options.reducedRotationCurvatureFallbackCandidate);
        reducedOptimizer.setUseGradientBoundaryCandidate(
            options.reducedRotationGradientBoundaryCandidate);
        reducedOptimizer.setUseSurrogateTcgAccept(
            options.reducedRotationSurrogateTcgAccept);
        reducedOptimizer.setSkipRedundantCandidateProjection(
            options.reducedRotationSkipRedundantCandidateProjection);
        reducedOptimizer.setUseJacobiPreconditioner(
            preconditioner ==
            ManualDpgoMmReducedRotationPreconditioner::Jacobi);
        reducedOptimizer.setUseSchurJacobiPreconditioner(
            preconditioner ==
            ManualDpgoMmReducedRotationPreconditioner::SchurJacobi);
        reducedOptimizer.setUseCholeskyPreconditioner(
            preconditioner ==
            ManualDpgoMmReducedRotationPreconditioner::Cholesky);
        Matrix reducedCandidate = reducedOptimizer.optimize(start);
        return std::make_pair(reducedCandidate, reducedOptimizer.getOptResult());
      };

      auto solveReducedAt = [&](const Matrix &start,
                                const SparseMatrix *solveG = nullptr,
                                bool useProximalStart = false,
                                const SparseMatrix *solveQ = nullptr,
                                bool deferTrueEvaluation = false) {
        CandidateResult candidateResult;
        if (start.rows() != X.rows() || start.cols() != X.cols() ||
            !start.allFinite()) {
          return candidateResult;
        }

        if (solveQ != nullptr) {
          setProblemQForLocalSolve(*solveQ, false);
        }
        if (solveG != nullptr) {
          setProblemGForLocalSolve(*solveG, false);
        }
        Matrix solverStart = start;
        (void)useProximalStart;
        if ((options.ammDpgoSurrogateParity ||
             options.reducedSurrogateMode !=
                 ManualDpgoMmReducedSurrogateMode::TrueLocal ||
             options.surrogateMode ==
                 ManualDpgoMmSurrogateMode::WeightedEdgeSplit) &&
            useProximalStart) {
          const Matrix proximalStart =
              makeDpgoMajorizedProximalStartUnfiltered(start);
          if (proximalStart.rows() == start.rows() &&
              proximalStart.cols() == start.cols() &&
              proximalStart.allFinite()) {
            solverStart = proximalStart;
            ++lastAmmProximalStartCount;
          }
        } else if (options.ammProximalStart) {
          const Matrix proximalStart = makeMajorizedProximalStart(start);
          if (proximalStart.rows() == start.rows() &&
              proximalStart.cols() == start.cols() &&
              proximalStart.allFinite()) {
            solverStart = proximalStart;
            ++lastAmmProximalStartCount;
          }
        }
        const double beforeCost = problem.f(solverStart);
        double beforeGradient = std::numeric_limits<double>::quiet_NaN();
        bool beforeGradientEvaluated = false;
        auto ensureBeforeGradient = [&]() {
          if (beforeGradientEvaluated) {
            return std::isfinite(beforeGradient);
          }
          {
            ScopedOptionalSecondsAccumulator profileTimer(
                options.profileOptimizer
                    ? &optimizerProfile.solverStartGradientEvaluationSec
                    : nullptr);
            beforeGradient = problem.RieGradNorm(solverStart);
          }
          beforeGradientEvaluated = true;
          if (options.profileOptimizer) {
            ++optimizerProfile.solverStartGradientEvaluationCount;
          }
          return std::isfinite(beforeGradient);
        };
        if (!options.lazySolverStartGradientEvaluation) {
          ensureBeforeGradient();
        }
        auto evaluateReducedRun =
            [&](int cgIterations,
                ManualDpgoMmReducedRotationPreconditioner preconditioner) {
          auto reduced = runReduced(solverStart, cgIterations,
                                    preconditioner);
          CandidateResult evaluated =
              options.candidateObjectReuse
                  ? evaluateMovedCurrentModelCandidate(
                        std::move(reduced.first))
                  : evaluateCurrentModelCandidate(reduced.first);
          evaluated.result = reduced.second;
          return evaluated;
        };

        auto bestReducedRun = [&](int cgIterations) {
          const auto mode = options.reducedRotationPreconditioner;
          auto runFullPortfolio = [&](bool adaptiveMode) {
            CandidateResult noneCandidate =
                evaluateReducedRun(
                    cgIterations,
                    ManualDpgoMmReducedRotationPreconditioner::None);
            CandidateResult bestCandidate = noneCandidate;
            ManualDpgoMmReducedRotationPreconditioner winnerMode =
                ManualDpgoMmReducedRotationPreconditioner::None;
            CandidateResult jacobiCandidate =
                evaluateReducedRun(
                    cgIterations,
                    ManualDpgoMmReducedRotationPreconditioner::Jacobi);
            if (candidateIsBetter(jacobiCandidate, bestCandidate)) {
              bestCandidate = jacobiCandidate;
              winnerMode = ManualDpgoMmReducedRotationPreconditioner::Jacobi;
            }
            CandidateResult schurJacobiCandidate =
                evaluateReducedRun(
                    cgIterations,
                    ManualDpgoMmReducedRotationPreconditioner::SchurJacobi);
            if (candidateIsBetter(schurJacobiCandidate, bestCandidate)) {
              bestCandidate = schurJacobiCandidate;
              winnerMode =
                  ManualDpgoMmReducedRotationPreconditioner::SchurJacobi;
            }
            CandidateResult choleskyCandidate =
                evaluateReducedRun(
                    cgIterations,
                    ManualDpgoMmReducedRotationPreconditioner::Cholesky);
            if (candidateIsBetter(choleskyCandidate, bestCandidate)) {
              bestCandidate = choleskyCandidate;
              winnerMode = ManualDpgoMmReducedRotationPreconditioner::Cholesky;
            }
            if (adaptiveMode && choleskyCandidate.ok &&
                winnerMode !=
                    ManualDpgoMmReducedRotationPreconditioner::Cholesky) {
              const double choleskyLoss =
                  choleskyCandidate.cost - bestCandidate.cost;
              const double materialOverride =
                  std::max(1e-8, 1e-6 * std::max(
                                      1.0, std::abs(choleskyCandidate.cost)));
              if (!std::isfinite(choleskyLoss) ||
                  choleskyLoss <= materialOverride) {
                bestCandidate = choleskyCandidate;
                winnerMode =
                    ManualDpgoMmReducedRotationPreconditioner::Cholesky;
              }
            }
            double winnerMargin = std::numeric_limits<double>::quiet_NaN();
            if (options.profileOptimizer) {
              ++optimizerProfile.reducedPortfolioSelectionCount;
              if (adaptiveMode) {
                ++optimizerProfile
                      .reducedAdaptivePortfolioFullSelectionCount;
              }
              const std::array<ManualDpgoMmReducedRotationPreconditioner, 4>
                  portfolioModes = {
                      ManualDpgoMmReducedRotationPreconditioner::None,
                      ManualDpgoMmReducedRotationPreconditioner::Jacobi,
                      ManualDpgoMmReducedRotationPreconditioner::SchurJacobi,
                      ManualDpgoMmReducedRotationPreconditioner::Cholesky};
              const std::array<const CandidateResult *, 4>
                  portfolioCandidates = {&noneCandidate, &jacobiCandidate,
                                         &schurJacobiCandidate,
                                         &choleskyCandidate};
              for (const auto candidateMode : portfolioModes) {
                const int idx =
                    profileReducedRotationPreconditionerIndex(candidateMode);
                if (idx >= 0) {
                  ++optimizerProfile.reducedPortfolioCandidateCounts
                        [static_cast<std::size_t>(idx)];
                }
              }
              const int winnerIdx =
                  profileReducedRotationPreconditionerIndex(winnerMode);
              if (bestCandidate.ok && winnerIdx >= 0) {
                ++optimizerProfile.reducedPortfolioWinnerCounts
                      [static_cast<std::size_t>(winnerIdx)];
                double secondCost = std::numeric_limits<double>::infinity();
                for (std::size_t idx = 0; idx < portfolioCandidates.size();
                     ++idx) {
                  const CandidateResult &candidate = *portfolioCandidates[idx];
                  if (!candidate.ok || portfolioModes[idx] == winnerMode ||
                      !std::isfinite(candidate.cost)) {
                    continue;
                  }
                  secondCost = std::min(secondCost, candidate.cost);
                }
                if (std::isfinite(secondCost) &&
                    std::isfinite(bestCandidate.cost)) {
                  const double margin = secondCost - bestCandidate.cost;
                  winnerMargin = margin;
                  ++optimizerProfile.reducedPortfolioWinnerCostMarginCount;
                  optimizerProfile.reducedPortfolioWinnerCostMarginSum +=
                      margin;
                  optimizerProfile.reducedPortfolioWinnerCostMarginMin =
                      std::min(
                          optimizerProfile
                              .reducedPortfolioWinnerCostMarginMin,
                          margin);
                }
              }
            }
            if (adaptiveMode) {
              ++reducedAdaptivePortfolioCalibrationCount;
              reducedAdaptivePortfolioFastPathSinceCalibration = 0;
              if (bestCandidate.ok &&
                  winnerMode ==
                      ManualDpgoMmReducedRotationPreconditioner::Cholesky) {
                ++reducedAdaptivePortfolioCholeskyWinStreak;
              } else {
                reducedAdaptivePortfolioCholeskyWinStreak = 0;
              }
              (void)winnerMargin;
            }
            return bestCandidate;
          };

          if (mode == ManualDpgoMmReducedRotationPreconditioner::Portfolio) {
            return runFullPortfolio(false);
          }
          if (mode ==
              ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio) {
            constexpr std::size_t kWarmupFullSelections = 3;
            constexpr std::size_t kRequiredCholeskyWinStreak = 2;
            constexpr std::size_t kMaxFastPathBeforeRefresh = 8;
            const bool periodicRefreshDue =
                reducedAdaptivePortfolioFastPathSinceCalibration >=
                kMaxFastPathBeforeRefresh;
            const bool canUseFastPath =
                reducedAdaptivePortfolioCalibrationCount >=
                    kWarmupFullSelections &&
                reducedAdaptivePortfolioCholeskyWinStreak >=
                    kRequiredCholeskyWinStreak &&
                (!periodicRefreshDue ||
                 options.reducedAdaptivePortfolioCertifiedFastPath);
            if (!canUseFastPath) {
              return runFullPortfolio(true);
            }
            CandidateResult choleskyCandidate = evaluateReducedRun(
                cgIterations,
                ManualDpgoMmReducedRotationPreconditioner::Cholesky);
            const bool costDoesNotIncrease =
                !std::isfinite(beforeCost) ||
                (choleskyCandidate.ok &&
                 choleskyCandidate.cost <= beforeCost + 1e-10);
            bool gradientMakesProgress = false;
            if (!costDoesNotIncrease && ensureBeforeGradient() &&
                beforeGradient > 1e-14 && choleskyCandidate.ok &&
                ensureCandidateGradient(choleskyCandidate)) {
              gradientMakesProgress =
                  choleskyCandidate.gradient <= 0.995 * beforeGradient;
            }
            if (choleskyCandidate.ok &&
                (costDoesNotIncrease || gradientMakesProgress)) {
              ++reducedAdaptivePortfolioFastPathSinceCalibration;
              if (options.profileOptimizer) {
                ++optimizerProfile.reducedAdaptivePortfolioFastPathCount;
                if (periodicRefreshDue &&
                    options.reducedAdaptivePortfolioCertifiedFastPath) {
                  ++optimizerProfile
                        .reducedAdaptivePortfolioCertifiedFastPathCount;
                }
              }
              return choleskyCandidate;
            }
            reducedAdaptivePortfolioCholeskyWinStreak = 0;
            reducedAdaptivePortfolioFastPathSinceCalibration = 0;
            if (options.profileOptimizer) {
              ++optimizerProfile.reducedAdaptivePortfolioFallbackCount;
            }
            return runFullPortfolio(true);
          }
          return evaluateReducedRun(cgIterations, mode);
        };

        candidateResult =
            bestReducedRun(options.trustRegionMaxInnerIterations);
        if (options.reducedRotationSurrogateTcgAccept &&
            solveG == nullptr && solveQ == nullptr && candidateResult.ok &&
            std::isfinite(beforeCost) &&
            candidateResult.cost > beforeCost + 1e-10) {
          candidateResult.x = solverStart;
          candidateResult.cost = beforeCost;
          candidateResult.modelCost = beforeCost;
          candidateResult.gradient =
              std::numeric_limits<double>::infinity();
          candidateResult.modelGradient = candidateResult.gradient;
          candidateResult.gradientEvaluated = false;
          candidateResult.ok = true;
          candidateResult.result =
              ROPTResult(true, beforeCost,
                         std::numeric_limits<double>::quiet_NaN(),
                         beforeCost,
                         std::numeric_limits<double>::quiet_NaN(),
                         0.0, 0.0);
          candidateResult.result.rtrAcceptedIterations = 0;
        }

        if (candidateResult.ok && options.adaptiveReducedTcg &&
            options.adaptiveReducedTcgMaxIterations >
                options.trustRegionMaxInnerIterations &&
            ensureBeforeGradient() && beforeGradient > 1e-14) {
          const double triggerRatio =
              std::max(0.0, options.adaptiveReducedTcgGradientRatio);
          if (ensureCandidateGradient(candidateResult) &&
              candidateResult.gradient > triggerRatio * beforeGradient) {
            CandidateResult refined =
                bestReducedRun(options.adaptiveReducedTcgMaxIterations);
            if (refined.ok && refined.cost <= candidateResult.cost + 1e-10 &&
                (refined.cost < candidateResult.cost - 1e-10 ||
                 (ensureCandidateGradient(refined) &&
                  refined.gradient < candidateResult.gradient))) {
              refined.adaptiveRefinements = 1;
              candidateResult = refined;
            }
          }
        }
        if (solveG != nullptr || solveQ != nullptr) {
          const bool deferSurrogateGradients =
              options.lazySurrogateCandidateGradientEvaluation &&
              options.lazyCandidateGradientEvaluation;
          if (!deferSurrogateGradients) {
            ensureCandidateGradient(candidateResult);
          }
          candidateResult.modelCost = candidateResult.cost;
          candidateResult.modelGradient = candidateResult.gradient;
          restoreTrueLocalModel();
          const bool canDeferTrueEvaluation =
              deferTrueEvaluation && candidateResult.ok &&
              std::isfinite(candidateResult.modelCost) &&
              (deferSurrogateGradients ||
               std::isfinite(candidateResult.modelGradient));
          if (canDeferTrueEvaluation) {
            candidateResult.trueEvaluationDeferred = true;
            return candidateResult;
          }
          CandidateResult trueEvaluation =
              evaluateCurrentModelCandidate(candidateResult.x);
          if (!deferSurrogateGradients) {
            ensureCandidateGradient(trueEvaluation);
          }
          candidateResult.cost = trueEvaluation.cost;
          candidateResult.gradient = trueEvaluation.gradient;
          candidateResult.gradientEvaluated =
              trueEvaluation.gradientEvaluated;
          candidateResult.ok =
              candidateResult.ok && std::isfinite(candidateResult.modelCost) &&
              (deferSurrogateGradients ||
               std::isfinite(candidateResult.modelGradient)) &&
              std::isfinite(candidateResult.cost) &&
              (deferSurrogateGradients ||
               std::isfinite(candidateResult.gradient));
        }
        return candidateResult;
      };

      auto solveDpgoSimpleAt = [&](const Matrix &start,
                                   const SparseMatrix &linearTerm,
                                   bool useProximalStart) {
        if (currentDpgoSimpleValid &&
            linearTerm.rows() == static_cast<int>(r) &&
            linearTerm.cols() == static_cast<int>(n * (d + 1))) {
          return solveReducedAt(start, &linearTerm, useProximalStart,
                                &dpgoSimpleQManual);
        }
        return solveReducedAt(start, nullptr, useProximalStart, nullptr);
      };

      auto solveWeightedEdgeSplitAt = [&](const Matrix &start) {
        if (weightedEdgeSplitSymmetric && currentDpgoSimpleValid) {
          return solveDpgoSimpleAt(start, currentDpgoSimpleG, true);
        }
        if (currentWeightedEdgeSplit.valid) {
          return solveReducedAt(start, &currentWeightedEdgeSplit.g, false,
                                &currentWeightedEdgeSplit.q);
        }
        return solveReducedAt(start, nullptr, false, nullptr);
      };

      auto solveAdaptiveSpectralAt = [&](const Matrix &start) {
        if (currentAdaptiveSpectral.valid) {
          return solveReducedAt(start, &currentAdaptiveSpectral.g, false,
                                &currentAdaptiveSpectral.q);
        }
        if (currentDpgoSimpleValid) {
          return solveDpgoSimpleAt(start, currentDpgoSimpleG, true);
        }
        return solveReducedAt(start, nullptr, false, nullptr);
      };

      auto solveVariableProjectedSchurAt = [&](const Matrix &start) {
        runVariableProjectedSchurDiagnostic(start);
        CandidateResult bestCandidate = solveReducedAt(start, nullptr, false,
                                                       nullptr);
        Matrix majorized =
            makeVariableProjectedSchurMajorizedCandidate(start);
        if (majorized.rows() == start.rows() &&
            majorized.cols() == start.cols() && majorized.allFinite()) {
          CandidateResult evaluated =
              options.candidateObjectReuse
                  ? evaluateMovedCurrentModelCandidate(std::move(majorized))
                  : evaluateCurrentModelCandidate(majorized);
          evaluated.source = CandidateSource::VariableProjectedSchur;
          if (candidateIsBetter(evaluated, bestCandidate)) {
            bestCandidate = evaluated;
          }
        }
        return bestCandidate;
      };

      auto makeBoundaryProximalModel =
          [&](const Matrix &anchor, double weight, SparseMatrix &proxQ,
              SparseMatrix &proxG) {
            if (weight <= 0.0 || !std::isfinite(weight) ||
                anchor.rows() != static_cast<int>(r) ||
                anchor.cols() != static_cast<int>(n * (d + 1)) ||
                !anchor.allFinite()) {
              return false;
            }
            std::set<unsigned> separatorPoses;
            for (const auto &m : sharedLoops) {
              if (m.r1 == id) {
                separatorPoses.insert(m.p1);
              } else if (m.r2 == id) {
                separatorPoses.insert(m.p2);
              }
            }
            if (separatorPoses.empty()) {
              return false;
            }

            double diagScale = 0.0;
            std::size_t diagCount = 0;
            for (const unsigned pose : separatorPoses) {
              if (pose >= n) {
                return false;
              }
              const unsigned colStart = pose * (d + 1);
              for (unsigned localCol = 0; localCol < d + 1; ++localCol) {
                const int col = static_cast<int>(colStart + localCol);
                diagScale += std::abs(trueQ.coeff(col, col));
                ++diagCount;
              }
            }
            if (diagCount > 0) {
              diagScale /= static_cast<double>(diagCount);
            }
            if (!std::isfinite(diagScale) || diagScale <= 1e-14) {
              diagScale = 1.0;
            }
            const double mu = weight * diagScale;
            if (!std::isfinite(mu) || mu <= 0.0) {
              return false;
            }

            proxQ = trueQ;
            proxG = trueG;
            for (const unsigned pose : separatorPoses) {
              const unsigned colStart = pose * (d + 1);
              for (unsigned localCol = 0; localCol < d + 1; ++localCol) {
                const int col = static_cast<int>(colStart + localCol);
                proxQ.coeffRef(col, col) += mu;
                for (unsigned row = 0; row < r; ++row) {
                  proxG.coeffRef(static_cast<int>(row), col) +=
                      -mu * anchor(static_cast<int>(row), col);
                }
              }
            }
            proxQ.makeCompressed();
            proxG.makeCompressed();
            return true;
          };

      auto boundaryProximalWeights = [&]() {
        std::vector<double> weights = options.localBoundaryProximalWeights;
        if (weights.empty()) {
          weights.push_back(options.localBoundaryProximalWeight);
        }
        weights.erase(std::remove_if(weights.begin(), weights.end(),
                                     [](double weight) {
                                       return !std::isfinite(weight) ||
                                              weight <= 0.0;
                                     }),
                      weights.end());
        return weights;
      };

      auto solveBoundaryProximalPortfolioAt = [&](const Matrix &start) {
        CandidateResult bestBoundaryProximal;
        for (const double weight : boundaryProximalWeights()) {
          SparseMatrix proxQ;
          SparseMatrix proxG;
          if (!makeBoundaryProximalModel(start, weight, proxQ, proxG)) {
            continue;
          }
          ++lastBoundaryProximalCandidateCount;
          CandidateResult candidate =
              solveReducedAt(start, &proxG, false, &proxQ);
          candidate.source = CandidateSource::BoundaryProximal;
          if (candidateIsBetter(candidate, bestBoundaryProximal)) {
            bestBoundaryProximal = candidate;
          }
        }
        return bestBoundaryProximal;
      };

      auto makeTopologyBoundaryAnchor =
          [&](const Matrix &start, Matrix &anchor,
              std::map<unsigned, double> &poseFreshnessConfidence) {
            if (start.rows() != static_cast<int>(r) ||
                start.cols() != static_cast<int>(n * (d + 1)) ||
                !start.allFinite()) {
              return false;
            }
            const unsigned blockDim = d + 1;
            anchor = start;
            std::map<unsigned, Matrix> poseAnchorSums;
            std::map<unsigned, double> poseAnchorWeights;
            std::map<unsigned, double> freshnessSums;
            std::map<unsigned, std::size_t> observationCounts;

            for (const auto &m : sharedLoops) {
              PoseID neighborPose;
              unsigned localPose = 0;
              bool localIsSource = false;
              if (m.r1 == id) {
                neighborPose = PoseID(m.r2, m.p2);
                localPose = m.p1;
                localIsSource = true;
              } else if (m.r2 == id) {
                neighborPose = PoseID(m.r1, m.p1);
                localPose = m.p2;
              } else {
                continue;
              }
              if (localPose >= n) {
                continue;
              }
              Matrix neighborBlock;
              if (!neighborPoseForLocalModel(neighborPose, neighborBlock) ||
                  neighborBlock.rows() != static_cast<int>(r) ||
                  neighborBlock.cols() != static_cast<int>(blockDim) ||
                  !neighborBlock.allFinite()) {
                continue;
              }

              Matrix T = Matrix::Zero(blockDim, blockDim);
              T.block(0, 0, d, d) = m.R;
              T.block(0, d, d, 1) = m.t;
              T(d, d) = 1.0;
              Matrix predicted;
              if (localIsSource) {
                predicted = neighborBlock * T.inverse();
              } else {
                predicted = neighborBlock * T;
              }
              if (predicted.rows() != static_cast<int>(r) ||
                  predicted.cols() != static_cast<int>(blockDim) ||
                  !predicted.allFinite()) {
                continue;
              }
              predicted.block(0, 0, r, d) =
                  projectToStiefelManifold(predicted.block(0, 0, r, d));
              if (!predicted.allFinite()) {
                continue;
              }

              const unsigned age = neighborPoseStaleness(neighborPose);
              const double freshness = 1.0 / (1.0 + static_cast<double>(age));
              const double stiffness =
                  localMeasurementStiffness(m, localIsSource);
              const double observationWeight = stiffness * freshness;
              if (!std::isfinite(observationWeight) ||
                  observationWeight <= 0.0) {
                continue;
              }
              auto sumIt = poseAnchorSums.find(localPose);
              if (sumIt == poseAnchorSums.end()) {
                sumIt =
                    poseAnchorSums
                        .emplace(localPose,
                                 Matrix::Zero(static_cast<int>(r),
                                              static_cast<int>(blockDim)))
                        .first;
              }
              sumIt->second += observationWeight * predicted;
              poseAnchorWeights[localPose] += observationWeight;
              freshnessSums[localPose] += freshness;
              ++observationCounts[localPose];
            }

            for (const auto &entry : poseAnchorSums) {
              const unsigned pose = entry.first;
              const double weight = poseAnchorWeights[pose];
              if (!std::isfinite(weight) || weight <= 0.0) {
                continue;
              }
              Matrix block = entry.second / weight;
              block.block(0, 0, r, d) =
                  projectToStiefelManifold(block.block(0, 0, r, d));
              if (!block.allFinite()) {
                continue;
              }
              anchor.block(0, pose * blockDim, r, blockDim) = block;
              const std::size_t count = observationCounts[pose];
              if (count > 0) {
                poseFreshnessConfidence[pose] =
                    freshnessSums[pose] / static_cast<double>(count);
              }
            }
            return !poseFreshnessConfidence.empty() && anchor.allFinite();
          };

      auto makeTopologyBoundarySurrogateModel =
          [&](const Matrix &anchor,
              const std::map<unsigned, double> &poseFreshnessConfidence,
              double weight, SparseMatrix &proxQ, SparseMatrix &proxG) {
            if (!currentDpgoSimpleValid || weight <= 0.0 ||
                !std::isfinite(weight) ||
                anchor.rows() != static_cast<int>(r) ||
                anchor.cols() != static_cast<int>(n * (d + 1)) ||
                !anchor.allFinite() ||
                dpgoSimpleQManual.rows() != static_cast<int>(n * (d + 1)) ||
                dpgoSimpleQManual.cols() != static_cast<int>(n * (d + 1)) ||
                currentDpgoSimpleG.rows() != static_cast<int>(r) ||
                currentDpgoSimpleG.cols() !=
                    static_cast<int>(n * (d + 1))) {
              return false;
            }
            const unsigned blockDim = d + 1;
            proxQ = dpgoSimpleQManual;
            proxG = currentDpgoSimpleG;

            bool addedTerm = false;
            for (const auto &entry : poseFreshnessConfidence) {
              const unsigned pose = entry.first;
              if (pose >= n) {
                continue;
              }
              const double freshnessConfidence =
                  std::max(0.0, std::min(1.0, entry.second));
              if (!std::isfinite(freshnessConfidence) ||
                  freshnessConfidence <= 0.0) {
                continue;
              }
              double diagScale = 0.0;
              std::size_t diagCount = 0;
              const unsigned colStart = pose * blockDim;
              for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
                const int col = static_cast<int>(colStart + localCol);
                diagScale += std::abs(dpgoSimpleQManual.coeff(col, col));
                ++diagCount;
              }
              if (diagCount > 0) {
                diagScale /= static_cast<double>(diagCount);
              }
              if (!std::isfinite(diagScale) || diagScale <= 1e-14) {
                diagScale = 1.0;
              }
              const double mu = weight * diagScale * freshnessConfidence;
              if (!std::isfinite(mu) || mu <= 0.0) {
                continue;
              }
              for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
                const int col = static_cast<int>(colStart + localCol);
                proxQ.coeffRef(col, col) += mu;
                for (unsigned row = 0; row < r; ++row) {
                  proxG.coeffRef(static_cast<int>(row), col) +=
                      -mu * anchor(static_cast<int>(row), col);
                }
              }
              addedTerm = true;
            }
            if (!addedTerm) {
              return false;
            }
            proxQ.makeCompressed();
            proxG.makeCompressed();
            return true;
          };

      auto solveTopologyBoundaryPortfolioAt = [&](const Matrix &start) {
        CandidateResult bestTopologyBoundary;
        Matrix anchor;
        std::map<unsigned, double> poseFreshnessConfidence;
        if (!makeTopologyBoundaryAnchor(start, anchor,
                                        poseFreshnessConfidence)) {
          return bestTopologyBoundary;
        }
        for (const double weight : boundaryProximalWeights()) {
          SparseMatrix proxQ;
          SparseMatrix proxG;
          if (!makeTopologyBoundarySurrogateModel(
                  anchor, poseFreshnessConfidence, weight, proxQ, proxG)) {
            continue;
          }
          ++lastBoundaryProximalCandidateCount;
          CandidateResult candidate =
              solveReducedAt(start, &proxG, false, &proxQ);
          candidate.source = CandidateSource::BoundaryProximal;
          if (candidateIsBetter(candidate, bestTopologyBoundary)) {
            bestTopologyBoundary = candidate;
          }
        }
        return bestTopologyBoundary;
      };

      auto recordEdgeTightQuadraticDiagnostics =
          [&](const CandidateResult &candidateResult) {
        if (!candidateResult.ok ||
            !std::isfinite(candidateResult.modelCost) ||
            !std::isfinite(candidateResult.cost)) {
          return;
        }
        ++lastEdgeTightQuadraticEvalCount;
        lastEdgeTightQuadraticSurrogateCostSum +=
            candidateResult.modelCost;
        lastEdgeTightQuadraticTrueCostSum += candidateResult.cost;
        double gap = candidateResult.modelCost - candidateResult.cost;
        if (std::abs(gap) <= 1e-10) {
          gap = 0.0;
        }
        lastEdgeTightQuadraticMajorizationGapMin =
            std::min(lastEdgeTightQuadraticMajorizationGapMin, gap);
      };

      auto evaluateDpgoProximalFallbackCandidate =
          [&](const Matrix &start, const SparseMatrix &linearTerm) {
        CandidateResult evaluated;
        evaluated.source = CandidateSource::SimpleSurrogate;
        if (!currentDpgoSimpleValid || start.rows() != X.rows() ||
            start.cols() != X.cols() || !start.allFinite()) {
          return evaluated;
        }

        Matrix proximal = makeDpgoMajorizedProximalStartUnfiltered(start);
        if (proximal.rows() != start.rows() ||
            proximal.cols() != start.cols() || !proximal.allFinite()) {
          proximal = start;
        }
        evaluated.x = proximal;
        evaluated.result =
            ROPTResult(true, problem.f(start), 0.0, 0.0, 0.0, 0.0, 0.0);
        evaluated.result.rtrAcceptedIterations = 0;

        try {
          setProblemQForLocalSolve(dpgoSimpleQManual, false);
          setProblemGForLocalSolve(linearTerm, false);
          CandidateResult modelEvaluation =
              evaluateCurrentModelCandidate(evaluated.x);
          restoreTrueLocalModel();
          CandidateResult trueEvaluation =
              evaluateCurrentModelCandidate(evaluated.x);
          evaluated.cost = trueEvaluation.cost;
          evaluated.gradient = trueEvaluation.gradient;
          evaluated.gradientEvaluated = trueEvaluation.gradientEvaluated;
          evaluated.modelCost = modelEvaluation.cost;
          evaluated.modelGradient = modelEvaluation.gradient;
          evaluated.ok = modelEvaluation.ok && trueEvaluation.ok &&
                         std::isfinite(evaluated.cost) &&
                         std::isfinite(evaluated.modelCost);
          if (evaluated.ok) {
            evaluated.result.fOpt = evaluated.cost;
            evaluated.result.gradNormOpt =
                evaluated.gradientEvaluated
                    ? evaluated.gradient
                    : std::numeric_limits<double>::quiet_NaN();
          }
        } catch (...) {
          restoreTrueLocalModel();
          throw;
        }
        return evaluated;
      };

      auto evaluateCurrentStateCandidate = [&]() {
        CandidateResult evaluated = evaluateCurrentModelCandidate(X);
        evaluated.source = CandidateSource::TrueLocal;
        if (evaluated.ok) {
          const double gradient =
              evaluated.gradientEvaluated
                  ? evaluated.gradient
                  : std::numeric_limits<double>::quiet_NaN();
          evaluated.result =
              ROPTResult(true, evaluated.cost, gradient, evaluated.cost,
                         gradient, 0.0, 0.0);
          evaluated.result.rtrAcceptedIterations = 0;
        }
        return evaluated;
      };

      const bool useMixedSurrogate =
          options.ammDpgoSurrogateParity &&
          options.ammDpgoMixedSurrogatePortfolio;
      const bool eligibleToSkipSimple =
          useMixedSurrogate &&
          options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak > 0 &&
          ammMixedSurrogateTrueLocalWinStreak >=
              static_cast<std::size_t>(
                  options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak);
      bool forceSimpleRefresh = false;
      bool skipSimpleSurrogate = eligibleToSkipSimple;
      if (eligibleToSkipSimple &&
          options.ammMixedSurrogateForceSimpleEverySkippedRounds > 0 &&
          ammMixedSurrogateConsecutiveSimpleSkips >=
              static_cast<std::size_t>(
                  options.ammMixedSurrogateForceSimpleEverySkippedRounds)) {
        forceSimpleRefresh = true;
        skipSimpleSurrogate = false;
        ++lastAmmMixedSurrogateSimpleForcedRefreshCount;
      }

      const auto localCandidateSelectionStart =
          options.profileOptimizer ? ProfileClock::now()
                                   : ProfileClock::time_point();
      CandidateResult plain;
      const bool useSimpleSurrogate =
          !weightedEdgeSplitRequested &&
          (options.ammDpgoSurrogateParity ||
           options.reducedSurrogateMode ==
               ManualDpgoMmReducedSurrogateMode::DpgoSimple ||
           options.reducedSurrogateMode ==
               ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic) &&
          currentDpgoSimpleValid;
      const bool useWeightedEdgeSplit =
          weightedEdgeSplitRequested &&
          (weightedEdgeSplitSymmetric ? currentDpgoSimpleValid
                                      : currentWeightedEdgeSplit.valid);
      const bool useAdaptiveSpectral =
          (adaptiveSpectralRequested ||
           options.communicationTopologyBoundarySurrogate) &&
          (currentAdaptiveSpectral.valid || currentDpgoSimpleValid);
      const bool useReducedInterface =
          reducedInterfaceScheduled && currentReducedInterface.valid;
      const bool useReducedInterfaceCandidate =
          useReducedInterface &&
          options.communicationTopologyReducedInterfaceCandidate;
      const bool useReducedInterfaceAsActive =
          useReducedInterface && !useReducedInterfaceCandidate;
      const bool useAmmDpgoProximalFallbackOnly =
          options.ammDpgoProximalFallbackOnly &&
          options.scheme == ManualDpgoMmScheme::AMM &&
          options.ammDpgoSurrogateParity && useSimpleSurrogate &&
          !weightedEdgeSplitRequested && !adaptiveSpectralRequested &&
          !variableProjectedSchurRequested;
      const bool useAmmLazyPlainAfterCertificate =
          options.ammLazyPlainAfterCertificate &&
          options.scheme == ManualDpgoMmScheme::AMM &&
          options.localSolver == ManualDpgoMmLocalSolver::ReducedRotation &&
          options.ammDpgoSurrogateParity &&
          options.ammDpgoMixedSurrogatePortfolio && useSimpleSurrogate &&
          skipSimpleSurrogate &&
          !useAmmDpgoProximalFallbackOnly && !weightedEdgeSplitRequested &&
          !adaptiveSpectralRequested &&
          !useReducedInterfaceAsActive &&
          !options.communicationTopologyBoundarySurrogate &&
          !variableProjectedSchurRequested &&
          options.reducedSurrogateMode !=
              ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic &&
          !options.localBoundaryProximalCandidate &&
          !options.communicationTopologyBoundaryCandidate &&
          !options.localStateExtrapolation &&
          !options.localModelGExtrapolation &&
          !options.coupledStateGExtrapolation &&
          !options.localAndersonAcceleration &&
          options.mmAcceleratorMode !=
              ManualDpgoMmMmAcceleratorMode::Anderson &&
          options.mmAcceleratorMode != ManualDpgoMmMmAcceleratorMode::Squarem;
      const bool useAmmSurrogateFirstExactEvaluation =
          options.ammSurrogateFirstExactEvaluation &&
          useAmmLazyPlainAfterCertificate &&
          options.mmSafeguard !=
              ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary;
      auto solveActiveReducedMapAt = [&](const Matrix &start) {
        CandidateResult result =
            variableProjectedSchurRequested
                ? solveVariableProjectedSchurAt(start)
                : (useReducedInterfaceAsActive
                       ? solveReducedAt(start, &currentReducedInterface.g,
                                        false, &currentReducedInterface.q)
                       : (useAdaptiveSpectral
                              ? solveAdaptiveSpectralAt(start)
                              : (useWeightedEdgeSplit
                                     ? solveWeightedEdgeSplitAt(start)
                                     : (useSimpleSurrogate
                                            ? solveDpgoSimpleAt(
                                                  start, currentDpgoSimpleG,
                                                  true)
                                            : solveReducedAt(start, nullptr,
                                                             false)))));
        result.source =
            variableProjectedSchurRequested
                ? CandidateSource::VariableProjectedSchur
                : ((useSimpleSurrogate || useWeightedEdgeSplit ||
                    useAdaptiveSpectral || useReducedInterfaceAsActive)
                       ? CandidateSource::SimpleSurrogate
                       : CandidateSource::TrueLocal);
        return result;
      };

      auto solveBaselinePlainCandidate = [&]() {
        CandidateResult baseline;
        if (useAmmDpgoProximalFallbackOnly) {
          baseline =
              evaluateDpgoProximalFallbackCandidate(X, currentDpgoSimpleG);
          ++lastAmmMixedSurrogateCandidateCount;
        } else if (skipSimpleSurrogate) {
          baseline = solveReducedAt(X, nullptr, false);
          baseline.source = CandidateSource::TrueLocal;
          ++lastAmmMixedSurrogateCandidateCount;
          ++lastAmmMixedSurrogateSimpleSkippedCount;
          ++ammMixedSurrogateConsecutiveSimpleSkips;
        } else {
          if (forceSimpleRefresh) {
            ammMixedSurrogateConsecutiveSimpleSkips = 0;
          }
          baseline = solveActiveReducedMapAt(X);
        }
        return baseline;
      };

      bool lazyPlainMaterialized = !useAmmLazyPlainAfterCertificate;
      if (useAmmLazyPlainAfterCertificate) {
        plain = evaluateCurrentStateCandidate();
      } else {
        plain = solveBaselinePlainCandidate();
      }
      CandidateResult best = plain;
      auto materializeLazyPlainCandidate = [&]() -> CandidateResult & {
        if (useAmmLazyPlainAfterCertificate && !lazyPlainMaterialized) {
          plain = solveBaselinePlainCandidate();
          best = plain;
          lazyPlainMaterialized = true;
        }
        return best;
      };
      const bool lazyPlainRepresentsSkippedTrueLocal =
          useAmmLazyPlainAfterCertificate && skipSimpleSurrogate;
      if (options.reducedSurrogateMode ==
          ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic) {
        CandidateResult edgeTight = solveReducedAt(X, nullptr, false);
        edgeTight.source = CandidateSource::EdgeTightQuadratic;
        recordEdgeTightQuadraticDiagnostics(edgeTight);
        if (candidateIsBetter(edgeTight, best)) {
          best = edgeTight;
        }
      }
      if (useMixedSurrogate && !skipSimpleSurrogate &&
          !useAmmDpgoProximalFallbackOnly &&
          !useAmmLazyPlainAfterCertificate) {
        CandidateResult trueLocalPlain = solveReducedAt(X, nullptr, false);
        trueLocalPlain.source = CandidateSource::TrueLocal;
        ++lastAmmMixedSurrogateCandidateCount;
        if (candidateIsBetter(trueLocalPlain, best)) {
          ++lastAmmMixedSurrogateTrueLocalAcceptedCount;
          best = trueLocalPlain;
        }
      }
      if (useReducedInterfaceCandidate) {
        if (useAmmLazyPlainAfterCertificate) {
          materializeLazyPlainCandidate();
        }
        CandidateResult reducedInterface =
            solveReducedAt(X, &currentReducedInterface.g, false,
                           &currentReducedInterface.q);
        reducedInterface.source = CandidateSource::ReducedInterface;
        ++lastAmmMixedSurrogateCandidateCount;
        if (candidateIsBetter(reducedInterface, best)) {
          best = reducedInterface;
        }
      }
      if (options.localBoundaryProximalCandidate) {
        CandidateResult boundaryProximal =
            solveBoundaryProximalPortfolioAt(X);
        if (candidateIsBetter(boundaryProximal, best)) {
          best = boundaryProximal;
        }
      }
      if (options.communicationTopologyBoundaryCandidate) {
        CandidateResult topologyBoundary =
            solveTopologyBoundaryPortfolioAt(X);
        if (candidateIsBetter(topologyBoundary, best)) {
          best = topologyBoundary;
        }
      }

      if (options.localStateExtrapolation && hasPreviousX &&
          previousX.rows() == X.rows() && previousX.cols() == X.cols()) {
        const std::vector<double> gammas = extrapolationGammas();
        CandidateResult bestExtrapolated;
        std::size_t attemptedExtrapolations = 0;
        for (const double gamma : gammas) {
          const Matrix extrapolatedStart =
              makeExtrapolatedStart(X, previousX, gamma);
          CandidateResult extrapolated =
              options.localStateExtrapolationUseActiveSurrogate
                  ? solveActiveReducedMapAt(extrapolatedStart)
                  : solveReducedAt(extrapolatedStart);
          extrapolated.source = CandidateSource::LocalStateExtrapolated;
          ++attemptedExtrapolations;
          if (candidateIsBetter(extrapolated, bestExtrapolated)) {
            bestExtrapolated = extrapolated;
          }
        }
        if (candidateIsBetter(bestExtrapolated, best)) {
          best = bestExtrapolated;
          lastExtrapolationAcceptedCount = 1;
          lastExtrapolationRejectedCount = attemptedExtrapolations - 1;
        } else {
          lastExtrapolationRejectedCount = attemptedExtrapolations;
        }
      }

      if (options.localModelGExtrapolation && hasPreviousG &&
          previousG.rows() == trueG.rows() && previousG.cols() == trueG.cols() &&
          !options.ammDpgoSurrogateParity) {
        const std::vector<double> gammas = gExtrapolationGammas();
        CandidateResult bestPredictedG;
        std::size_t attemptedGExtrapolations = 0;
        for (const double gamma : gammas) {
          SparseMatrix predictedG = trueG + gamma * (trueG - previousG);
          predictedG.makeCompressed();
          CandidateResult predicted = solveReducedAt(X, &predictedG);
          predicted.source = CandidateSource::GExtrapolated;
          ++attemptedGExtrapolations;
          if (candidateIsBetter(predicted, bestPredictedG)) {
            bestPredictedG = predicted;
          }
        }
        if (candidateIsBetter(bestPredictedG, best)) {
          best = bestPredictedG;
          lastGExtrapolationAcceptedCount = 1;
          lastGExtrapolationRejectedCount = attemptedGExtrapolations - 1;
        } else {
          lastGExtrapolationRejectedCount = attemptedGExtrapolations;
        }
      }

      if (options.coupledStateGExtrapolation && hasPreviousX && hasPreviousG &&
          previousX.rows() == X.rows() && previousX.cols() == X.cols() &&
          previousG.rows() == trueG.rows() && previousG.cols() == trueG.cols() &&
          !options.ammDpgoSurrogateParity) {
        const std::vector<double> gammas = coupledExtrapolationGammas();
        CandidateResult bestCoupled;
        std::size_t attemptedCoupledExtrapolations = 0;
        for (const double gamma : gammas) {
          const Matrix extrapolatedStart =
              makeExtrapolatedStart(X, previousX, gamma);
          SparseMatrix predictedG = trueG + gamma * (trueG - previousG);
          predictedG.makeCompressed();
          CandidateResult coupled =
              solveReducedAt(extrapolatedStart, &predictedG);
          coupled.source = CandidateSource::CoupledExtrapolated;
          ++attemptedCoupledExtrapolations;
          if (candidateIsBetter(coupled, bestCoupled)) {
            bestCoupled = coupled;
          }
        }
        if (candidateIsBetter(bestCoupled, best)) {
          best = bestCoupled;
          lastCoupledExtrapolationAcceptedCount = 1;
          lastCoupledExtrapolationRejectedCount =
              attemptedCoupledExtrapolations - 1;
        } else {
          lastCoupledExtrapolationRejectedCount =
              attemptedCoupledExtrapolations;
        }
      }

      const bool useLocalAnderson =
          options.localAndersonAcceleration ||
          options.mmAcceleratorMode == ManualDpgoMmMmAcceleratorMode::Anderson;
      if (useLocalAnderson && plain.ok &&
          hasPreviousFixedPoint &&
          previousFixedPointInput.rows() == originalX.rows() &&
          previousFixedPointInput.cols() == originalX.cols() &&
          previousFixedPointOutput.rows() == plain.x.rows() &&
          previousFixedPointOutput.cols() == plain.x.cols()) {
        const Matrix currentResidual = plain.x - originalX;
        const Matrix previousResidual =
            previousFixedPointOutput - previousFixedPointInput;
        const Matrix residualDelta = currentResidual - previousResidual;
        const double denominator =
            residualDelta.cwiseProduct(residualDelta).sum();
        if (std::isfinite(denominator) && denominator > 1e-20) {
          double alpha =
              currentResidual.cwiseProduct(residualDelta).sum() / denominator;
          const double maxAlpha =
              std::max(0.0, options.localAndersonMaxAlpha);
          if (std::isfinite(alpha) && maxAlpha > 0.0) {
            alpha = std::max(-maxAlpha, std::min(maxAlpha, alpha));
            const Matrix andersonStart = projectRotationBlocks(
                plain.x - alpha * (plain.x - previousFixedPointOutput));
            CandidateResult anderson = solveReducedAt(andersonStart);
            anderson.source = CandidateSource::Anderson;
            if (candidateIsBetter(anderson, best)) {
              best = anderson;
              lastAndersonAcceptedCount = 1;
            } else {
              lastAndersonRejectedCount = 1;
            }
          } else {
            lastAndersonRejectedCount = 1;
          }
        } else {
          lastAndersonRejectedCount = 1;
        }
      }

      if (options.mmAcceleratorMode ==
              ManualDpgoMmMmAcceleratorMode::Squarem &&
          plain.ok && hasPreviousFixedPoint) {
        CandidateResult second = solveReducedAt(plain.x);
        if (second.ok &&
            second.x.rows() == originalX.rows() &&
            second.x.cols() == originalX.cols()) {
          const Matrix residual = plain.x - originalX;
          const Matrix curvature = second.x - plain.x - residual;
          const double residualNorm =
              std::sqrt(residual.cwiseProduct(residual).sum());
          const double curvatureNorm =
              std::sqrt(curvature.cwiseProduct(curvature).sum());
          const double maxAlpha =
              std::max(0.0, options.localSquaremMaxAlpha);
          if (std::isfinite(residualNorm) &&
              std::isfinite(curvatureNorm) && residualNorm > 1e-14 &&
              curvatureNorm > 1e-14 && maxAlpha > 0.0) {
            double alpha = -residualNorm / curvatureNorm;
            if (std::isfinite(alpha)) {
              alpha = std::max(-maxAlpha, std::min(maxAlpha, alpha));
              Matrix squaremState =
                  projectRotationBlocks(originalX - 2.0 * alpha * residual +
                                        alpha * alpha * curvature);
              squaremState = recoverMajorizedTranslations(squaremState);
              CandidateResult squarem = solveReducedAt(squaremState);
              squarem.source = CandidateSource::Squarem;
              if (candidateIsBetter(squarem, best)) {
                best = squarem;
                lastSquaremAcceptedCount = 1;
              } else {
                lastSquaremRejectedCount = 1;
              }
            } else {
              lastSquaremRejectedCount = 1;
            }
          } else {
            lastSquaremRejectedCount = 1;
          }
        } else {
          lastSquaremRejectedCount = 1;
        }
      }

      if (options.profileOptimizer) {
        optimizerProfile.localCandidateSelectionSec +=
            secondsSince(localCandidateSelectionStart);
        ++optimizerProfile.localCandidateSelectionCount;
      }

      auto updateMixedSurrogateStateForSelectedSource =
          [&](CandidateSource selectedSource) {
            recordMixedSurrogateSelectedSource(selectedSource);
            if (selectedSource == CandidateSource::BoundaryProximal) {
              ++lastBoundaryProximalAcceptedCount;
            }
            if (useMixedSurrogate) {
              if (selectedSource == CandidateSource::TrueLocal) {
                ++ammMixedSurrogateTrueLocalWinStreak;
              } else {
                ammMixedSurrogateTrueLocalWinStreak = 0;
                ammMixedSurrogateConsecutiveSimpleSkips = 0;
              }
            } else {
              ammMixedSurrogateTrueLocalWinStreak = 0;
              ammMixedSurrogateConsecutiveSimpleSkips = 0;
            }
          };
      if (!useAmmLazyPlainAfterCertificate) {
        updateMixedSurrogateStateForSelectedSource(best.source);
      }

      if (options.scheme == ManualDpgoMmScheme::AMM) {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer ? &optimizerProfile.localAmmOuterSec
                                     : nullptr);
        if (options.profileOptimizer) {
          ++optimizerProfile.localAmmOuterCount;
        }
        double trueCurrentCost = std::numeric_limits<double>::infinity();
        double baselineSurrogateGradNorm =
            std::numeric_limits<double>::infinity();
        if (options.fusedCandidateEvaluation) {
          const auto fusedCurrent = problem.fAndRieGradNorm(originalX);
          trueCurrentCost = fusedCurrent.first + modelConstant;
          baselineSurrogateGradNorm = fusedCurrent.second;
        } else {
          trueCurrentCost = problem.f(originalX) + modelConstant;
          baselineSurrogateGradNorm = problem.RieGradNorm(originalX);
        }
        double ammReferenceCost = trueCurrentCost;
        double baselineSurrogateFobj = trueCurrentCost;
        double baselineSurrogateOffset = 0.0;
        RecursiveSimpleEvaluation recursiveSimple;
        const bool recursiveSimpleRequested =
            options.ammDpgoRecursiveSimpleState &&
            options.ammDpgoSurrogateParity && currentDpgoSimpleValid;
        if (recursiveSimpleRequested) {
          recursiveSimple =
              evaluateRecursiveSimpleState(originalX, currentDpgoSimpleG);
        }
        const bool recursiveSimpleActive =
            recursiveSimpleRequested && recursiveSimple.valid;
        const bool baselineSurrogateActive =
            options.ammBaselineSurrogateState &&
            options.ammDpgoSurrogateParity && currentDpgoSimpleValid &&
            !recursiveSimpleActive;
        if (baselineSurrogateActive) {
          setProblemQForLocalSolve(dpgoSimpleQManual, false);
          setProblemGForLocalSolve(historyGForThisStep, false);
          const CandidateResult baselineModel =
              evaluateCurrentModelCandidate(originalX);
          restoreTrueLocalModel();
          if (baselineModel.ok) {
            baselineSurrogateOffset = trueCurrentCost - baselineModel.cost;
            baselineSurrogateFobj =
                baselineModel.cost + baselineSurrogateOffset;
            baselineSurrogateGradNorm = baselineModel.gradient;
            ammReferenceCost = baselineSurrogateFobj;
          }
        }
        if (recursiveSimpleActive) {
          setProblemQForLocalSolve(dpgoSimpleQManual, false);
          setProblemGForLocalSolve(historyGForThisStep, false);
          const CandidateResult baselineModel =
              evaluateCurrentModelCandidate(originalX);
          if (options.ammDpgoStrictRefinedGate) {
            const double strictGradNorm =
                dpgoSimpleFullGradientNorm(originalX, historyGForThisStep);
            if (std::isfinite(strictGradNorm)) {
              baselineSurrogateGradNorm = strictGradNorm;
            }
          }
          restoreTrueLocalModel();
          if (baselineModel.ok) {
            if (!options.ammDpgoStrictRefinedGate) {
              baselineSurrogateGradNorm = baselineModel.gradient;
            }
          }
          baselineSurrogateFobj = recursiveSimple.fobj;
          baselineSurrogateOffset = 0.0;
          ammReferenceCost = recursiveSimple.fobj;
        }
        const bool ammSurrogateStateActive =
            baselineSurrogateActive || recursiveSimpleActive;
        const AmmStepState ammStep = beginAmmStep(ammReferenceCost);
        ManualDpgoMmAmmTrace trace;
        trace.valid = recursiveSimpleActive;
        trace.localIter = ammLocalIter;
        trace.fobj = ammReferenceCost;
        trace.trueFobj = trueCurrentCost;
        trace.baselineSurrogateFobj = baselineSurrogateFobj;
        trace.baselineSurrogateOffset = baselineSurrogateOffset;
        trace.baselineSurrogateGradNorm = baselineSurrogateGradNorm;
        trace.recursiveSimpleState = recursiveSimpleRequested;
        trace.recursiveSimpleValid = recursiveSimpleActive;
        if (recursiveSimple.valid) {
          trace.recursiveSimpleF = recursiveSimple.f;
          trace.recursiveSimpleFobj = recursiveSimple.fobj;
          trace.recursiveSimpleDeltaQDelta = recursiveSimple.deltaQDelta;
          trace.recursiveSimplePTerm = recursiveSimple.pTerm;
          trace.recursiveSimplePreviousGk = recursiveSimple.previousGk;
        }
        trace.baselineSurrogateState = baselineSurrogateActive;
        trace.localAcceptedIterationBudget =
            options.trustRegionAcceptedIterations;
        trace.numOscillations = ammStep.numOscillations;
        trace.F0 = ammFk0;
        trace.F1 = ammFk1;
        trace.gamma = ammStep.gamma;
        const double refinedDenominator =
            std::max(1e-12, std::abs(ammReferenceCost));
        trace.refinedRatio =
            (baselineSurrogateGradNorm * baselineSurrogateGradNorm) /
            refinedDenominator;
        const bool hasLocalAcceptedBudget =
            options.trustRegionIterations > 0 &&
            options.trustRegionAcceptedIterations > 0;
        trace.refined =
            ammSurrogateStateActive
                ? (hasLocalAcceptedBudget &&
                   (trace.refinedRatio > options.ammAcceptedDelta ||
                    trace.numOscillations >= options.ammMaxOscillations))
                : (options.trustRegionIterations > 0 &&
                   options.trustRegionMaxInnerIterations > 0);
        trace.softRestartHits0 = ammSoftRestartHits0;
        trace.softRestartHits1 = ammSoftRestartHits1;
        bool hardRestart = false;
        bool softRestart = false;
        bool phiFallback = false;
        bool acceleratedAccepted = false;
        bool localMeritRejected = false;
        bool attemptedAcceleration = false;
        bool cooldownSkipped = false;
        const double surrogateTraceOffset =
            ammSurrogateStateActive ? baselineSurrogateOffset : modelConstant;
        auto recursiveSurrogateCost =
            [&](const Matrix &value, const SparseMatrix &linearTerm) {
              return recursiveSimpleActive
                         ? evaluateDpgoSimpleGValue(value, linearTerm,
                                                   recursiveSimple.f)
                         : std::numeric_limits<double>::quiet_NaN();
            };
        auto surrogateCandidateValue =
            [&](const CandidateResult &candidateResult) {
              if (!candidateResult.ok) {
                return std::numeric_limits<double>::infinity();
              }
              if (recursiveSimpleActive) {
                return recursiveSurrogateCost(candidateResult.x,
                                              historyGForThisStep);
              }
              if (std::isfinite(candidateResult.modelCost)) {
                return candidateResult.modelCost + surrogateTraceOffset;
              }
              if (!candidateResult.trueEvaluationDeferred &&
                  std::isfinite(candidateResult.cost)) {
                return candidateResult.cost + modelConstant;
              }
              return std::numeric_limits<double>::infinity();
            };
        auto surrogateCandidateIsBetter =
            [&](const CandidateResult &candidateResult,
                const CandidateResult &bestResult) {
              if (!candidateResult.ok) {
                return false;
              }
              if (!bestResult.ok) {
                return true;
              }
              return surrogateCandidateValue(candidateResult) <
                     surrogateCandidateValue(bestResult) - 1e-12;
            };
        bool canAttemptAcceleration = ammStep.canAccelerate;
        if (canAttemptAcceleration && ammCooldown > 0) {
          --ammCooldown;
          canAttemptAcceleration = false;
          cooldownSkipped = true;
        }

        if (canAttemptAcceleration && hasPreviousX && hasPreviousG &&
            previousX.rows() == X.rows() && previousX.cols() == X.cols() &&
            previousG.rows() == historyGForThisStep.rows() &&
            previousG.cols() == historyGForThisStep.cols()) {
          CandidateResult accelerated;
          CandidateResult acceleratedXakh;
          SparseMatrix acceleratedPredictedG;
          ManualDpgoMmAmmTrace acceleratedTrace;
          std::size_t attemptedAmmGammas = 0;
          const std::vector<double> gammaScales =
              options.ammDpgoSurrogateParity ? std::vector<double>{1.0}
                                             : ammGammaScales();
          for (const double gammaScale : gammaScales) {
            const double gamma = ammStep.gamma * gammaScale;
            if (!std::isfinite(gamma) || gamma <= 0.0) {
              continue;
            }
            const Matrix acceleratedStart =
                makeExtrapolatedStart(X, previousX, gamma);
            SparseMatrix predictedG =
                historyGForThisStep + gamma * (historyGForThisStep - previousG);
            predictedG.makeCompressed();
            if (options.ammDpgoSurrogateParity) {
              setProblemQForLocalSolve(dpgoSimpleQManual, false);
            }
            setProblemGForLocalSolve(predictedG, false);
            const Matrix xakh =
                options.ammDpgoSurrogateParity
                    ? makeDpgoMajorizedProximalStartUnfiltered(
                          acceleratedStart)
                    : makeMajorizedProximalStart(acceleratedStart);
            ManualDpgoMmAmmTrace trialTrace = trace;
            trialTrace.valid = true;
            trialTrace.gamma = gamma;
            Matrix xakFromXakhStorage;
            const Matrix *xakFromXakh = &xakFromXakhStorage;
            bool xakFromXakhMatchesGuard = false;
            if (options.ammDpgoSurrogateParity &&
                options.ammDpgoRecoverTranslationsAfterProximal) {
              const TranslationRecoveryResult recovery =
                  recoverMajorizedTranslationsWithStats(xakh);
              {
                ScopedOptionalSecondsAccumulator constructionTimer(
                    options.profileOptimizer
                        ? &optimizerProfile.candidateConstructionSec
                        : nullptr);
                xakFromXakhStorage = recovery.value;
              }
              if (options.profileOptimizer) {
                ++optimizerProfile.candidateObjectConstructionCount;
              }
              trialTrace.translationRecoveryAttemptCount +=
                  recovery.attempted ? 1 : 0;
              trialTrace.translationRecoveryAcceptedCount +=
                  recovery.accepted ? 1 : 0;
              trialTrace.translationRecoveryRejectedCount +=
                  (recovery.attempted && !recovery.accepted) ? 1 : 0;
            } else if (options.candidateObjectReuse) {
              xakFromXakh = &xakh;
              xakFromXakhMatchesGuard = true;
              if (options.profileOptimizer) {
                ++optimizerProfile.candidateObjectReuseCount;
              }
            } else {
              {
                ScopedOptionalSecondsAccumulator constructionTimer(
                    options.profileOptimizer
                        ? &optimizerProfile.candidateConstructionSec
                        : nullptr);
                xakFromXakhStorage = xakh;
              }
              if (options.profileOptimizer) {
                ++optimizerProfile.candidateObjectConstructionCount;
              }
              xakFromXakhMatchesGuard = true;
            }
            const CandidateResult xakhGuardModel =
                evaluateCurrentModelCandidate(xakh);
            CandidateResult xakhModel;
            if (ammSurrogateStateActive) {
              xakhModel = options.fusedCandidateEvaluation &&
                                  xakFromXakhMatchesGuard
                              ? xakhGuardModel
                              : evaluateCurrentModelCandidate(*xakFromXakh);
            } else if (!options.ammDpgoSurrogateParity) {
              xakhModel = options.fusedCandidateEvaluation &&
                                  xakFromXakhMatchesGuard
                              ? xakhGuardModel
                              : evaluateCurrentModelCandidate(*xakFromXakh);
            }
            trialTrace.minG =
                ammFk0 - options.ammPsi * (xakh - X).squaredNorm();
            if (recursiveSimpleActive) {
              trialTrace.surrogateGkhInitial =
                  recursiveSurrogateCost(xakh, historyGForThisStep);
            } else {
              trialTrace.surrogateGkhInitial =
                  xakhGuardModel.ok
                      ? xakhGuardModel.cost + surrogateTraceOffset
                                : std::numeric_limits<double>::infinity();
            }
            const bool deferXakhTrueEvaluation =
                useAmmSurrogateFirstExactEvaluation &&
                options.ammDpgoSurrogateParity &&
                !options.ammDpgoRecoverTranslationsAfterProximal;
            CandidateResult xakhGuardTrue;
            CandidateResult xakhTrue;
            restoreTrueLocalModel();
            if (deferXakhTrueEvaluation) {
              xakhGuardTrue = xakhGuardModel;
              xakhGuardTrue.trueEvaluationDeferred = true;
              xakhTrue = xakhGuardTrue;
            } else {
              xakhGuardTrue = evaluateCurrentModelCandidate(xakh);
              xakhTrue =
                  options.fusedCandidateEvaluation && xakFromXakhMatchesGuard
                      ? xakhGuardTrue
                      : evaluateCurrentModelCandidate(*xakFromXakh);
            }
            trialTrace.GkhInitial =
                (!deferXakhTrueEvaluation && xakhGuardTrue.ok)
                    ? xakhGuardTrue.cost + modelConstant
                    : std::numeric_limits<double>::quiet_NaN();
            trialTrace.GkhAfterRestartCheck = trialTrace.GkhInitial;
            trialTrace.surrogateGkhAfterRestartCheck =
                trialTrace.surrogateGkhInitial;
            trialTrace.proxReset =
                (ammSurrogateStateActive || deferXakhTrueEvaluation
                     ? trialTrace.surrogateGkhInitial
                     : trialTrace.GkhInitial) > trialTrace.minG;
            CandidateResult xakhCandidate =
                deferXakhTrueEvaluation
                    ? xakhGuardModel
                    : (options.ammDpgoSurrogateParity ? xakhTrue : xakhModel);
            if (deferXakhTrueEvaluation) {
              xakhCandidate.trueEvaluationDeferred = true;
              xakhCandidate.modelCost = xakhGuardModel.cost;
              xakhCandidate.modelGradient = xakhGuardModel.gradient;
            } else if (options.ammDpgoSurrogateParity) {
              xakhCandidate.modelCost =
                  xakhModel.ok ? xakhModel.cost : xakhTrue.cost;
              xakhCandidate.modelGradient =
                  xakhModel.ok ? xakhModel.gradient : xakhTrue.gradient;
            } else {
              xakhCandidate.modelCost = xakhModel.cost;
              xakhCandidate.modelGradient = xakhModel.gradient;
              xakhCandidate.cost = xakhTrue.cost;
              xakhCandidate.gradient = xakhTrue.gradient;
              xakhCandidate.ok = xakhModel.ok && xakhTrue.ok;
            }
            const bool useDpgoSurrogate =
                options.ammDpgoRestartFallback ||
                options.ammDpgoSurrogateParity;
            if (useDpgoSurrogate && trialTrace.proxReset) {
              if (options.ammDpgoSurrogateParity) {
                setProblemQForLocalSolve(dpgoSimpleQManual, false);
                setProblemGForLocalSolve(historyGForThisStep, false);
              } else {
                setProblemGForLocalSolve(trueG, true);
              }
              const Matrix resetXakh =
                  options.ammDpgoSurrogateParity
                      ? makeDpgoMajorizedProximalStartUnfiltered(X)
                      : makeMajorizedProximalStart(X);
              Matrix resetXakFromXakhStorage;
              const Matrix *resetXakFromXakh = &resetXakFromXakhStorage;
              bool resetXakFromXakhMatchesGuard = false;
              if (options.ammDpgoSurrogateParity &&
                  options.ammDpgoRecoverTranslationsAfterProximal) {
                const TranslationRecoveryResult recovery =
                    recoverMajorizedTranslationsWithStats(resetXakh);
                {
                  ScopedOptionalSecondsAccumulator constructionTimer(
                      options.profileOptimizer
                          ? &optimizerProfile.candidateConstructionSec
                          : nullptr);
                  resetXakFromXakhStorage = recovery.value;
                }
                if (options.profileOptimizer) {
                  ++optimizerProfile.candidateObjectConstructionCount;
                }
                trialTrace.translationRecoveryAttemptCount +=
                    recovery.attempted ? 1 : 0;
                trialTrace.translationRecoveryAcceptedCount +=
                    recovery.accepted ? 1 : 0;
                trialTrace.translationRecoveryRejectedCount +=
                    (recovery.attempted && !recovery.accepted) ? 1 : 0;
              } else if (options.candidateObjectReuse) {
                resetXakFromXakh = &resetXakh;
                resetXakFromXakhMatchesGuard = true;
                if (options.profileOptimizer) {
                  ++optimizerProfile.candidateObjectReuseCount;
                }
              } else {
                {
                  ScopedOptionalSecondsAccumulator constructionTimer(
                      options.profileOptimizer
                          ? &optimizerProfile.candidateConstructionSec
                          : nullptr);
                  resetXakFromXakhStorage = resetXakh;
                }
                if (options.profileOptimizer) {
                  ++optimizerProfile.candidateObjectConstructionCount;
                }
                resetXakFromXakhMatchesGuard = true;
              }
              const CandidateResult resetGuardModel =
                  evaluateCurrentModelCandidate(resetXakh);
              CandidateResult resetModel;
              if (ammSurrogateStateActive) {
                resetModel =
                    options.fusedCandidateEvaluation &&
                            resetXakFromXakhMatchesGuard
                        ? resetGuardModel
                        : evaluateCurrentModelCandidate(*resetXakFromXakh);
              } else if (!options.ammDpgoSurrogateParity) {
                setProblemGForLocalSolve(predictedG, false);
                resetModel = evaluateCurrentModelCandidate(*resetXakFromXakh);
              }
              const bool deferResetTrueEvaluation =
                  useAmmSurrogateFirstExactEvaluation &&
                  options.ammDpgoSurrogateParity &&
                  !options.ammDpgoRecoverTranslationsAfterProximal;
              CandidateResult resetGuardTrue;
              CandidateResult resetTrue;
              restoreTrueLocalModel();
              if (deferResetTrueEvaluation) {
                resetGuardTrue = resetGuardModel;
                resetGuardTrue.trueEvaluationDeferred = true;
                resetTrue = resetGuardTrue;
              } else {
                resetGuardTrue = evaluateCurrentModelCandidate(resetXakh);
                resetTrue =
                    options.fusedCandidateEvaluation &&
                            resetXakFromXakhMatchesGuard
                        ? resetGuardTrue
                        : evaluateCurrentModelCandidate(*resetXakFromXakh);
              }
              trialTrace.GkhAfterRestartCheck =
                  (!deferResetTrueEvaluation && resetGuardTrue.ok)
                      ? resetGuardTrue.cost + modelConstant
                      : trialTrace.GkhInitial;
              if (recursiveSimpleActive) {
                trialTrace.surrogateGkhAfterRestartCheck =
                    recursiveSurrogateCost(resetXakh, historyGForThisStep);
              } else {
                trialTrace.surrogateGkhAfterRestartCheck =
                    resetGuardModel.ok
                        ? resetGuardModel.cost + surrogateTraceOffset
                        : trialTrace.surrogateGkhInitial;
              }
              CandidateResult resetCandidate =
                  deferResetTrueEvaluation
                      ? resetGuardModel
                      : (options.ammDpgoSurrogateParity ? resetTrue
                                                        : resetModel);
              if (deferResetTrueEvaluation) {
                resetCandidate.trueEvaluationDeferred = true;
                resetCandidate.modelCost = resetGuardModel.cost;
                resetCandidate.modelGradient = resetGuardModel.gradient;
              } else if (options.ammDpgoSurrogateParity) {
                resetCandidate.modelCost =
                    resetModel.ok ? resetModel.cost : resetTrue.cost;
                resetCandidate.modelGradient =
                    resetModel.ok ? resetModel.gradient : resetTrue.gradient;
              } else {
                resetCandidate.modelCost = resetModel.cost;
                resetCandidate.modelGradient = resetModel.gradient;
                resetCandidate.cost = resetTrue.cost;
                resetCandidate.gradient = resetTrue.gradient;
                resetCandidate.ok = resetModel.ok && resetTrue.ok;
              }
              if (resetCandidate.ok) {
                xakhCandidate = resetCandidate;
              }
            }
            CandidateResult trial;
            const bool skipRefinedSolveAfterProxReset =
                options.ammProxResetSkipRefinedSolve && useDpgoSurrogate &&
                trialTrace.proxReset && xakhCandidate.ok;
            if (skipRefinedSolveAfterProxReset) {
              trial = xakhCandidate;
              trialTrace.refined = false;
              trialTrace.localAcceptedIterations = 0;
            } else if (ammSurrogateStateActive && !trialTrace.refined) {
              trial = xakhCandidate;
            } else {
              const Matrix &refinedStart =
                  options.ammDpgoSurrogateParity &&
                          options
                              .ammDpgoRefinedStartsAtRecoveredProximal
                      ? *xakFromXakh
                      : acceleratedStart;
              const bool refinedUseProximalStart =
                  !(options.ammDpgoSurrogateParity &&
                    options.ammDpgoRefinedStartsAtRecoveredProximal);
              const bool deferAcceleratedTrueEvaluation =
                  useAmmSurrogateFirstExactEvaluation &&
                  options.ammDpgoSurrogateParity;
              trial = options.ammDpgoSurrogateParity
                          ? solveReducedAt(refinedStart, &predictedG,
                                           refinedUseProximalStart,
                                           &dpgoSimpleQManual,
                                           deferAcceleratedTrueEvaluation)
                          : solveReducedAt(acceleratedStart, &predictedG,
                                           false);
            }
            ++attemptedAmmGammas;
            trialTrace.GkAfterAccelerated =
                (trial.ok && !trial.trueEvaluationDeferred)
                    ? trial.cost + modelConstant
                    : std::numeric_limits<double>::quiet_NaN();
            trialTrace.surrogateGkAfterAccelerated =
                recursiveSimpleActive && trial.ok
                    ? recursiveSurrogateCost(trial.x, historyGForThisStep)
                    : (trial.ok && std::isfinite(trial.modelCost)
                           ? trial.modelCost + surrogateTraceOffset
                           : trialTrace.GkAfterAccelerated);
            trialTrace.finalGk = trialTrace.GkAfterAccelerated;
            trialTrace.surrogateFinalGk =
                trialTrace.surrogateGkAfterAccelerated;
            if (trial.trueEvaluationDeferred
                    ? surrogateCandidateIsBetter(trial, accelerated)
                    : candidateIsBetter(trial, accelerated)) {
              accelerated = trial;
              acceleratedXakh = xakhCandidate;
              acceleratedPredictedG = predictedG;
              acceleratedTrace = trialTrace;
            }
          }
          attemptedAcceleration = attemptedAmmGammas > 0;

          if (accelerated.ok) {
            CandidateResult plainBest = best;
            CandidateResult selected = accelerated;
            trace = acceleratedTrace;
            acceleratedAccepted = true;

            const bool surrogateFirstDeferred =
                useAmmSurrogateFirstExactEvaluation &&
                selected.trueEvaluationDeferred;
            double acceleratedAmmCost =
                surrogateFirstDeferred ? trace.surrogateGkAfterAccelerated
                                       : accelerated.cost + modelConstant;
            if (ammSurrogateStateActive) {
              acceleratedAmmCost = trace.surrogateGkAfterAccelerated;
            } else if (options.ammDpgoRestartFallback ||
                       options.ammDpgoSurrogateParity) {
              acceleratedAmmCost = trace.GkAfterAccelerated;
            }
            if (surrogateFirstDeferred && !ammSurrogateStateActive &&
                options.ammDpgoSurrogateParity) {
              acceleratedAmmCost = trace.surrogateGkAfterAccelerated;
            }
            const double plainAmmCost = plainBest.cost + modelConstant;

            hardRestart = acceleratedAmmCost > ammFk0;
            if (options.ammProxResetSkipRefinedSolve && trace.proxReset) {
              hardRestart = true;
            }
            softRestart =
                (acceleratedAmmCost > ammFk1 &&
                 ammSoftRestartHits0 >= options.ammMaxSoftRestartHits0) ||
                (acceleratedAmmCost > ammReferenceCost &&
                 ammSoftRestartHits1 > options.ammMaxSoftRestartHits1);

            if (!hardRestart && !softRestart && !phiFallback &&
                surrogateFirstDeferred) {
              if (materializeDeferredTrueEvaluation(selected)) {
                accelerated = selected;
                if (acceleratedXakh.trueEvaluationDeferred &&
                    materializeDeferredTrueEvaluation(acceleratedXakh)) {
                  trace.GkhAfterRestartCheck =
                      acceleratedXakh.cost + modelConstant;
                }
                trace.GkAfterAccelerated = selected.cost + modelConstant;
                if (!std::isfinite(trace.surrogateGkAfterAccelerated)) {
                  trace.surrogateGkAfterAccelerated =
                      surrogateCandidateValue(selected);
                }
                trace.finalGk = trace.GkAfterAccelerated;
                trace.surrogateFinalGk =
                    trace.surrogateGkAfterAccelerated;
                acceleratedAmmCost = trace.GkAfterAccelerated;
                hardRestart = acceleratedAmmCost > ammFk0;
                softRestart =
                    (acceleratedAmmCost > ammFk1 &&
                     ammSoftRestartHits0 >= options.ammMaxSoftRestartHits0) ||
                    (acceleratedAmmCost > ammReferenceCost &&
                     ammSoftRestartHits1 > options.ammMaxSoftRestartHits1);
              } else {
                hardRestart = true;
              }
            }

            if ((hardRestart || softRestart) &&
                useAmmLazyPlainAfterCertificate) {
              plainBest = materializeLazyPlainCandidate();
            }
            if ((hardRestart || softRestart) && plainBest.ok) {
              selected = plainBest;
              acceleratedAccepted = false;
              if ((options.ammDpgoRestartFallback ||
                   options.ammDpgoSurrogateParity) &&
                  acceleratedXakh.ok &&
                  (ammSurrogateStateActive ||
                           useAmmSurrogateFirstExactEvaluation
                       ? trace.surrogateGkhAfterRestartCheck
                       : trace.GkhAfterRestartCheck) <= ammReferenceCost) {
                CandidateResult xakhRefined;
                if (ammSurrogateStateActive && trace.refined) {
                  xakhRefined =
                      solveReducedAt(acceleratedXakh.x, &historyGForThisStep,
                                     false, &dpgoSimpleQManual);
                } else if (!ammSurrogateStateActive) {
                  xakhRefined =
                      solveReducedAt(acceleratedXakh.x, nullptr, false);
                }
                if (xakhRefined.ok) {
                  selected = xakhRefined;
                  trace.restartUsedXakh = true;
                } else {
                  selected = acceleratedXakh;
                  trace.restartUsedXakh = true;
                }
              }
            } else if (!options.ammDpgoRestartFallback &&
                       !options.ammDpgoSurrogateParity &&
                       options.ammLocalMeritFilter && plainBest.ok &&
                       !ammCandidatePassesLocalMerit(accelerated, plainBest)) {
              selected = plainBest;
              acceleratedAccepted = false;
              localMeritRejected = true;
            } else if (phiFallback) {
              selected = (options.ammDpgoRestartFallback ||
                          options.ammDpgoSurrogateParity) &&
                                 acceleratedXakh.ok
                             ? acceleratedXakh
                             : plainBest;
              acceleratedAccepted = false;
            } else if (plainBest.ok) {
              double rhsReference = plainAmmCost;
              if (ammSurrogateStateActive) {
                rhsReference = trace.surrogateGkhAfterRestartCheck;
              } else if (options.ammDpgoRestartFallback ||
                         options.ammDpgoSurrogateParity) {
                rhsReference =
                    std::isfinite(trace.GkhAfterRestartCheck)
                        ? trace.GkhAfterRestartCheck
                        : trace.surrogateGkhAfterRestartCheck;
              }
              const double lhs = ammFk0 - acceleratedAmmCost;
              const double rhs = options.ammPhi * (ammFk0 - rhsReference);
              if (std::isfinite(lhs) && std::isfinite(rhs) && lhs < rhs) {
                selected = (options.ammDpgoRestartFallback ||
                            options.ammDpgoSurrogateParity) &&
                                   acceleratedXakh.ok
                               ? acceleratedXakh
                               : plainBest;
                acceleratedAccepted = false;
                phiFallback = true;
              }
            }

            if (selected.trueEvaluationDeferred) {
              if (!materializeDeferredTrueEvaluation(selected)) {
                plainBest = materializeLazyPlainCandidate();
                selected = plainBest;
                acceleratedAccepted = false;
                localMeritRejected = true;
              }
            }

            const bool traceUsesSurrogateCertificate =
                surrogateFirstDeferred &&
                !std::isfinite(trace.GkAfterAccelerated);
            const bool traceUsesSurrogateRestartReference =
                ammSurrogateStateActive ||
                (useAmmSurrogateFirstExactEvaluation &&
                 !std::isfinite(trace.GkhAfterRestartCheck));
            const double traceAcceleratedCertificateCost =
                traceUsesSurrogateCertificate
                    ? trace.surrogateGkAfterAccelerated
                    : (ammSurrogateStateActive
                           ? trace.surrogateGkAfterAccelerated
                           : trace.GkAfterAccelerated);
            const double traceRestartReferenceCost =
                traceUsesSurrogateCertificate ||
                        traceUsesSurrogateRestartReference
                    ? trace.surrogateGkhAfterRestartCheck
                    : trace.GkhAfterRestartCheck;
            trace.hardRestart = hardRestart;
            trace.softRestart = softRestart;
            trace.phiFallback = phiFallback;
            trace.phiLhs = ammFk0 - traceAcceleratedCertificateCost;
            trace.phiRhs =
                options.ammPhi * (ammFk0 - traceRestartReferenceCost);
            trace.restartCertificateLhs = trace.phiLhs;
            trace.restartCertificateRhs = trace.phiRhs;
            trace.restartCertificateMargin =
                trace.restartCertificateLhs - trace.restartCertificateRhs;
            trace.restartCertificateValid =
                std::isfinite(trace.restartCertificateLhs) &&
                std::isfinite(trace.restartCertificateRhs) &&
                std::isfinite(trace.restartCertificateMargin);
            trace.restartCertificatePassed =
                trace.restartCertificateValid &&
                trace.restartCertificateMargin >= -1e-12 &&
                !hardRestart && !softRestart;
            if (!acceleratedAccepted && selected.ok) {
              trace.localAcceptedIterations =
                  selected.result.rtrAcceptedIterations;
              trace.finalGk =
                  selected.cost + modelConstant;
              if (recursiveSimpleActive) {
                trace.surrogateFinalGk =
                    recursiveSurrogateCost(selected.x, historyGForThisStep);
              } else {
                trace.surrogateFinalGk =
                    std::isfinite(selected.modelCost)
                        ? selected.modelCost + surrogateTraceOffset
                        : trace.finalGk;
              }
            } else {
              trace.localAcceptedIterations =
                  accelerated.result.rtrAcceptedIterations;
              trace.finalGk = trace.GkAfterAccelerated;
              trace.surrogateFinalGk =
                  trace.surrogateGkAfterAccelerated;
            }
            best = selected;
          } else if (useAmmLazyPlainAfterCertificate) {
            materializeLazyPlainCandidate();
          }
        }

        if (recursiveSimpleActive && !attemptedAcceleration && best.ok) {
          setProblemQForLocalSolve(dpgoSimpleQManual, false);
          setProblemGForLocalSolve(historyGForThisStep, false);
          const Matrix xakh = makeDpgoMajorizedProximalStartUnfiltered(X);
          restoreTrueLocalModel();
          if (xakh.rows() == X.rows() && xakh.cols() == X.cols() &&
              xakh.allFinite()) {
            const CandidateResult xakhGuardTrue =
                evaluateCurrentModelCandidate(xakh);
            trace.GkhInitial =
                xakhGuardTrue.ok ? xakhGuardTrue.cost + modelConstant
                                  : trace.GkhInitial;
            trace.surrogateGkhInitial =
                recursiveSurrogateCost(xakh, historyGForThisStep);
            trace.minG =
                ammFk0 - options.ammPsi * (xakh - X).squaredNorm();
          }
          trace.localAcceptedIterations = best.result.rtrAcceptedIterations;
          trace.GkAfterAccelerated = best.cost + modelConstant;
          trace.surrogateGkAfterAccelerated =
              recursiveSurrogateCost(best.x, historyGForThisStep);
          trace.GkhAfterRestartCheck = trace.GkhInitial;
          trace.surrogateGkhAfterRestartCheck = trace.surrogateGkhInitial;
          trace.finalGk = trace.GkAfterAccelerated;
          trace.surrogateFinalGk = trace.surrogateGkAfterAccelerated;
          trace.phiLhs = ammFk0 - trace.surrogateFinalGk;
          trace.phiRhs = 0.0;
        }

        const bool rejectedAcceleration =
            attemptedAcceleration && !acceleratedAccepted;
        if (acceleratedAccepted) {
          ammCooldown = 0;
        } else if (rejectedAcceleration && options.ammCooldownAfterRejected > 0) {
          ammCooldown = options.ammCooldownAfterRejected;
        }

        const double nextS = cooldownSkipped ? 1.0 : ammStep.nextS;
        finishAmmStep(nextS, hardRestart, hardRestart || softRestart);
        lastAmmAcceleratedAcceptedCount = acceleratedAccepted ? 1 : 0;
        lastAmmHardRestartCount = hardRestart ? 1 : 0;
        lastAmmSoftRestartCount = softRestart ? 1 : 0;
        lastAmmRestartCount = (hardRestart || softRestart) ? 1 : 0;
        lastAmmPhiFallbackCount = phiFallback ? 1 : 0;
        lastAmmLocalMeritRejectedCount = localMeritRejected ? 1 : 0;
        lastAmmSkippedCount = cooldownSkipped ? 1 : 0;
        if (trace.valid) {
          lastAmmTrace = trace;
        }
        if (recursiveSimpleActive && recursiveSimple.valid) {
          double nextRecursiveGk = trace.surrogateFinalGk;
          if (!std::isfinite(nextRecursiveGk) && best.ok) {
            nextRecursiveGk =
                recursiveSurrogateCost(best.x, historyGForThisStep);
          }
          if (std::isfinite(nextRecursiveGk)) {
            recursiveSimplePreviousZRows = recursiveSimple.zRows;
            recursiveSimplePreviousGk = nextRecursiveGk;
            recursiveSimpleInitialized = true;
          }
        }
        ++ammLocalIter;
      }

      if (useAmmLazyPlainAfterCertificate) {
        if (!lazyPlainMaterialized && best.source == CandidateSource::TrueLocal) {
          materializeLazyPlainCandidate();
        }
        if (lazyPlainMaterialized) {
          updateMixedSurrogateStateForSelectedSource(best.source);
        } else if (lazyPlainRepresentsSkippedTrueLocal) {
          ++lastAmmMixedSurrogateTrueLocalSelectedCount;
          ++lastAmmMixedSurrogateSimpleSkippedCount;
          ++ammMixedSurrogateTrueLocalWinStreak;
          ++ammMixedSurrogateConsecutiveSimpleSkips;
        }
      }

      if (!best.ok) {
        restoreTrueLocalModel();
        return false;
      }
      candidate = best.x;
      result = best.result;
      lastAdaptiveRefinementCount = best.adaptiveRefinements;
      previousFixedPointInput = originalX;
      previousFixedPointOutput = plain.x;
      hasPreviousFixedPoint = plain.ok;
    }

    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer ? &optimizerProfile.localFinalizationSec
                                   : nullptr);
      if (options.profileOptimizer) {
        ++optimizerProfile.localFinalizationCount;
      }
      if (candidate.rows() != X.rows() || candidate.cols() != X.cols() ||
          !candidate.allFinite()) {
        return false;
      }
      lastBoundaryEdgeCostBefore = boundaryEdgeCost(originalX);
      lastBoundaryEdgeCostAfter = boundaryEdgeCost(candidate);
      lastSeparatorDeltaNorm = separatorDeltaNorm(originalX, candidate);
      previousX = originalX;
      hasPreviousX = true;
      previousG = historyGForThisStep;
      hasPreviousG = true;
      restoreTrueLocalModel();
      X = candidate;
      if (options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid &&
          !applyFullEquivHybridActiveSeparatorCorrection(result)) {
        return false;
      }
    }
    lastOptimizationResult = result;
    return true;
  }

  std::size_t getLastAdaptiveRefinementCount() const {
    return lastAdaptiveRefinementCount;
  }

  void resetLocalGradientCorrectionDiagnostics() {
    localGradientCorrectionDiagnostics = LocalGradientCorrectionDiagnostics();
  }

  LocalGradientCorrectionDiagnostics
  getLocalGradientCorrectionDiagnostics() const {
    return localGradientCorrectionDiagnostics;
  }

  std::size_t getLastExtrapolationAcceptedCount() const {
    return lastExtrapolationAcceptedCount;
  }

  std::size_t getLastExtrapolationRejectedCount() const {
    return lastExtrapolationRejectedCount;
  }

  std::size_t getLastGExtrapolationAcceptedCount() const {
    return lastGExtrapolationAcceptedCount;
  }

  std::size_t getLastGExtrapolationRejectedCount() const {
    return lastGExtrapolationRejectedCount;
  }

  std::size_t getLastCoupledExtrapolationAcceptedCount() const {
    return lastCoupledExtrapolationAcceptedCount;
  }

  std::size_t getLastCoupledExtrapolationRejectedCount() const {
    return lastCoupledExtrapolationRejectedCount;
  }

  std::size_t getLastAndersonAcceptedCount() const {
    return lastAndersonAcceptedCount;
  }

  std::size_t getLastAndersonRejectedCount() const {
    return lastAndersonRejectedCount;
  }

  std::size_t getLastSquaremAcceptedCount() const {
    return lastSquaremAcceptedCount;
  }

  std::size_t getLastSquaremRejectedCount() const {
    return lastSquaremRejectedCount;
  }

  std::size_t getLastAmmAcceleratedAcceptedCount() const {
    return lastAmmAcceleratedAcceptedCount;
  }

  std::size_t getLastAmmRestartCount() const { return lastAmmRestartCount; }

  std::size_t getLastAmmHardRestartCount() const {
    return lastAmmHardRestartCount;
  }

  std::size_t getLastAmmSoftRestartCount() const {
    return lastAmmSoftRestartCount;
  }

  std::size_t getLastAmmPhiFallbackCount() const {
    return lastAmmPhiFallbackCount;
  }

  std::size_t getLastAmmLocalMeritRejectedCount() const {
    return lastAmmLocalMeritRejectedCount;
  }

  std::size_t getLastAmmProximalStartCount() const {
    return lastAmmProximalStartCount;
  }

  std::size_t getLastAmmSkippedCount() const {
    return lastAmmSkippedCount;
  }

  std::size_t getLastAmmMixedSurrogateCandidateCount() const {
    return lastAmmMixedSurrogateCandidateCount;
  }

  std::size_t getLastAmmMixedSurrogateTrueLocalAcceptedCount() const {
    return lastAmmMixedSurrogateTrueLocalAcceptedCount;
  }

  std::size_t getLastAmmMixedSurrogateSimpleSelectedCount() const {
    return lastAmmMixedSurrogateSimpleSelectedCount;
  }

  std::size_t getLastAmmMixedSurrogateTrueLocalSelectedCount() const {
    return lastAmmMixedSurrogateTrueLocalSelectedCount;
  }

  std::size_t getLastAmmMixedSurrogateExtrapolatedSelectedCount() const {
    return lastAmmMixedSurrogateExtrapolatedSelectedCount;
  }

  std::size_t getLastAmmMixedSurrogateOtherSelectedCount() const {
    return lastAmmMixedSurrogateOtherSelectedCount;
  }

  std::size_t getLastAmmMixedSurrogateSimpleSkippedCount() const {
    return lastAmmMixedSurrogateSimpleSkippedCount;
  }

  std::size_t getLastAmmMixedSurrogateSimpleForcedRefreshCount() const {
    return lastAmmMixedSurrogateSimpleForcedRefreshCount;
  }

  std::size_t getLastEdgeTightQuadraticEvalCount() const {
    return lastEdgeTightQuadraticEvalCount;
  }

  double getLastEdgeTightQuadraticSurrogateCostSum() const {
    return lastEdgeTightQuadraticSurrogateCostSum;
  }

  double getLastEdgeTightQuadraticTrueCostSum() const {
    return lastEdgeTightQuadraticTrueCostSum;
  }

  double getLastEdgeTightQuadraticMajorizationGapMin() const {
    return lastEdgeTightQuadraticEvalCount == 0
               ? 0.0
               : lastEdgeTightQuadraticMajorizationGapMin;
  }

  std::size_t getLastVariableProjectedSchurCandidateCount() const {
    return lastVariableProjectedSchurCandidateCount;
  }

  std::size_t getLastVariableProjectedSchurAcceptedCount() const {
    return lastVariableProjectedSchurAcceptedCount;
  }

  std::size_t getLastBoundaryProximalCandidateCount() const {
    return lastBoundaryProximalCandidateCount;
  }

  std::size_t getLastBoundaryProximalAcceptedCount() const {
    return lastBoundaryProximalAcceptedCount;
  }

  std::size_t getLastSurrogateBoundCheckCount() const {
    return lastSurrogateBoundCheckCount;
  }

  std::size_t getLastSurrogateBoundViolationCount() const {
    return lastSurrogateBoundViolationCount;
  }

  double getLastSurrogateBoundMinMargin() const {
    return lastSurrogateBoundCheckCount == 0 ? 0.0
                                             : lastSurrogateBoundMinMargin;
  }

  double getLastBoundaryEdgeCostBefore() const {
    return lastBoundaryEdgeCostBefore;
  }

  double getLastBoundaryEdgeCostAfter() const {
    return lastBoundaryEdgeCostAfter;
  }

  double getLastSeparatorDeltaNorm() const {
    return lastSeparatorDeltaNorm;
  }

  std::size_t getLastAcceptedIterationCount() const {
    return lastOptimizationResult.rtrAcceptedIterations;
  }

  std::size_t getLastFullEquivHybridWarmStartCandidateCount() const {
    return lastFullEquivHybridWarmStartCandidateCount;
  }

  std::size_t getLastFullEquivHybridWarmStartAcceptedCount() const {
    return lastFullEquivHybridWarmStartAcceptedCount;
  }

  std::size_t getLastFullEquivHybridWarmStartGuardRejectedCount() const {
    return lastFullEquivHybridWarmStartGuardRejectedCount;
  }

  std::size_t getLastFullEquivHybridSchurStepCandidateCount() const {
    return lastFullEquivHybridSchurStepCandidateCount;
  }

  std::size_t getLastFullEquivHybridSchurStepAcceptedCount() const {
    return lastFullEquivHybridSchurStepAcceptedCount;
  }

  std::size_t getLastFullEquivHybridSchurStepGuardRejectedCount() const {
    return lastFullEquivHybridSchurStepGuardRejectedCount;
  }

  std::size_t getLastFullEquivHybridLocalPortfolioCandidateCount() const {
    return lastFullEquivHybridLocalPortfolioCandidateCount;
  }

  std::size_t
  getLastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount() const {
    return lastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount;
  }

  std::size_t
  getLastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount() const {
    return lastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount;
  }

  std::size_t
  getLastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount() const {
    return lastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount;
  }

  std::size_t getLastFullEquivHybridSchwarzSmoothingSweepCount() const {
    return lastFullEquivHybridSchwarzSmoothingSweepCount;
  }

  std::size_t getLastFullEquivHybridSchwarzSmoothingCandidateCount() const {
    return lastFullEquivHybridSchwarzSmoothingCandidateCount;
  }

  std::size_t getLastFullEquivHybridSchwarzSmoothingAcceptedCount() const {
    return lastFullEquivHybridSchwarzSmoothingAcceptedCount;
  }

  std::size_t getLastFullEquivHybridSchwarzSmoothingRejectedCount() const {
    return lastFullEquivHybridSchwarzSmoothingRejectedCount;
  }

  double getLastFullEquivHybridSchwarzSmoothingCostDecrease() const {
    return lastFullEquivHybridSchwarzSmoothingCostDecrease;
  }

  double getLastFullEquivHybridSchwarzSmoothingTimeSec() const {
    return lastFullEquivHybridSchwarzSmoothingTimeSec;
  }

  std::size_t getLastFullEquivHybridLinearPcgIterationCount() const {
    return lastFullEquivHybridLinearPcgIterationCount;
  }

  double getLastFullEquivHybridLinearInitialResidual() const {
    return lastFullEquivHybridLinearInitialResidual;
  }

  double getLastFullEquivHybridLinearFinalResidual() const {
    return lastFullEquivHybridLinearFinalResidual;
  }

  double getLastFullEquivHybridLinearFullResidual() const {
    return lastFullEquivHybridLinearFullResidual;
  }

  double getLastFullEquivHybridLinearSolveTimeSec() const {
    return lastFullEquivHybridLinearSolveTimeSec;
  }

  std::size_t getLastFullEquivHybridStepTrialCount() const {
    return lastFullEquivHybridStepTrialCount;
  }

  std::size_t getLastFullEquivHybridStepTrialAcceptedCount() const {
    return lastFullEquivHybridStepTrialAcceptedCount;
  }

  std::size_t getLastFullEquivHybridStepTrialRejectedCount() const {
    return lastFullEquivHybridStepTrialRejectedCount;
  }

  double getLastFullEquivHybridStepPredictedDecreaseSum() const {
    return lastFullEquivHybridStepPredictedDecreaseSum;
  }

  double getLastFullEquivHybridStepActualDecreaseSum() const {
    return lastFullEquivHybridStepActualDecreaseSum;
  }

  double getLastFullEquivHybridStepRhoSum() const {
    return lastFullEquivHybridStepRhoSum;
  }

  std::size_t getLastFullEquivHybridStepRhoCount() const {
    return lastFullEquivHybridStepRhoCount;
  }

  double getLastFullEquivHybridStepAcceptedScaleSum() const {
    return lastFullEquivHybridStepAcceptedScaleSum;
  }

  double getLastFullEquivHybridStepAcceptedPredictedDecreaseSum() const {
    return lastFullEquivHybridStepAcceptedPredictedDecreaseSum;
  }

  double getLastFullEquivHybridStepAcceptedActualDecreaseSum() const {
    return lastFullEquivHybridStepAcceptedActualDecreaseSum;
  }

  double getLastFullEquivHybridStepAcceptedRhoSum() const {
    return lastFullEquivHybridStepAcceptedRhoSum;
  }

  std::size_t getLastFullEquivHybridStepAcceptedRhoCount() const {
    return lastFullEquivHybridStepAcceptedRhoCount;
  }

  std::size_t getLastFullEquivHybridStepParetoCandidateCount() const {
    return lastFullEquivHybridStepParetoCandidateCount;
  }

  std::size_t getLastFullEquivHybridStepParetoSelectedCount() const {
    return lastFullEquivHybridStepParetoSelectedCount;
  }

  double getLastFullEquivHybridStepParetoSelectedScaleSum() const {
    return lastFullEquivHybridStepParetoSelectedScaleSum;
  }

  double getLastFullEquivHybridStepParetoBestDecreaseSum() const {
    return lastFullEquivHybridStepParetoBestDecreaseSum;
  }

  double getLastFullEquivHybridStepParetoSelectedDecreaseSum() const {
    return lastFullEquivHybridStepParetoSelectedDecreaseSum;
  }

  double getLastFullEquivHybridStepParetoSelectedGradientSum() const {
    return lastFullEquivHybridStepParetoSelectedGradientSum;
  }

  std::size_t getLastFullEquivHybridStepParetoGradientEvalCount() const {
    return lastFullEquivHybridStepParetoGradientEvalCount;
  }

  double getLastFullEquivHybridStepParetoGradientEvalTimeSec() const {
    return lastFullEquivHybridStepParetoGradientEvalTimeSec;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryStepTrialCandidateCount() const {
    return lastFullEquivHybridTranslationRecoveryStepTrialCandidateCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount() const {
    return lastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryStepTrialSelectedCount() const {
    return lastFullEquivHybridTranslationRecoveryStepTrialSelectedCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount()
      const {
    return lastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryInitialGuessUsedCount() const {
    return lastFullEquivHybridTranslationRecoveryInitialGuessUsedCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount() const {
    return lastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorCandidateCount() const {
    return lastFullEquivHybridActiveSeparatorCandidateCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorAcceptedCount() const {
    return lastFullEquivHybridActiveSeparatorAcceptedCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorRejectedCount() const {
    return lastFullEquivHybridActiveSeparatorRejectedCount;
  }

  double getLastFullEquivHybridActiveSeparatorStepSum() const {
    return lastFullEquivHybridActiveSeparatorStepSum;
  }

  double getLastFullEquivHybridActiveSeparatorCostDecreaseSum() const {
    return lastFullEquivHybridActiveSeparatorCostDecreaseSum;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurCandidateCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurCandidateCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurAcceptedCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurAcceptedCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount;
  }

  std::size_t
  getLastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount()
      const {
    return
        lastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurFallbackCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurFallbackCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount;
  }

  std::size_t getLastFullEquivHybridActiveSeparatorLmSchurPrivateColCount()
      const {
    return lastFullEquivHybridActiveSeparatorLmSchurPrivateColCount;
  }

  std::size_t
  getLastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount() const {
    return lastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount;
  }

  double getLastFullEquivHybridActiveSeparatorLmSchurAlphaSum() const {
    return lastFullEquivHybridActiveSeparatorLmSchurAlphaSum;
  }

  double getLastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum() const {
    return lastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum;
  }

  double getLastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum() const {
    return lastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishAttemptCount() const {
    return lastFullEquivHybridTranslationRecoveryPolishAttemptCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishAcceptedCount() const {
    return lastFullEquivHybridTranslationRecoveryPolishAcceptedCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishRejectedCount() const {
    return lastFullEquivHybridTranslationRecoveryPolishRejectedCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount;
  }

  double
  getLastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum() const {
    return lastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount() const {
    return lastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount;
  }

  double
  getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum;
  }

  double
  getLastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum() const {
    return lastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum;
  }

  double
  getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum()
      const {
    return lastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum;
  }

  double
  getLastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum() const {
    return lastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum;
  }

  double
  getLastFullEquivHybridTranslationRecoveryPolishGradientChangeSum() const {
    return lastFullEquivHybridTranslationRecoveryPolishGradientChangeSum;
  }

  std::size_t getLastFullEquivHybridSparseMatrixVectorProductCount() const {
    return lastFullEquivHybridSparseMatrixVectorProductCount;
  }

  std::size_t
  getLastFullEquivHybridReducedRotationInitialGuessCandidateCount() const {
    return lastFullEquivHybridReducedRotationInitialGuessCandidateCount;
  }

  std::size_t
  getLastFullEquivHybridReducedRotationInitialGuessUsedCount() const {
    return lastFullEquivHybridReducedRotationInitialGuessUsedCount;
  }

  std::size_t
  getLastFullEquivHybridReducedRotationInitialGuessRejectedCount() const {
    return lastFullEquivHybridReducedRotationInitialGuessRejectedCount;
  }

  std::size_t
  getLastFullEquivHybridReducedRotationPreconditionerApplicationCount() const {
    return lastFullEquivHybridReducedRotationPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationSchurPreconditionerApplicationCount() const {
    return
        lastFullEquivHybridTranslationSchurPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationSchurPreconditionerFactorizationCount() const {
    return
        lastFullEquivHybridTranslationSchurPreconditionerFactorizationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationSchurPreconditionerFallbackCount() const {
    return lastFullEquivHybridTranslationSchurPreconditionerFallbackCount;
  }

  std::size_t
  getLastFullEquivHybridLocalChainPreconditionerApplicationCount() const {
    return lastFullEquivHybridLocalChainPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridLocalChainPreconditionerFactorizationCount() const {
    return lastFullEquivHybridLocalChainPreconditionerFactorizationCount;
  }

  std::size_t
  getLastFullEquivHybridLocalChainPreconditionerFallbackCount() const {
    return lastFullEquivHybridLocalChainPreconditionerFallbackCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationBlockPreconditionerApplicationCount() const {
    return lastFullEquivHybridTranslationBlockPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationBlockPreconditionerFactorizationCount() const {
    return lastFullEquivHybridTranslationBlockPreconditionerFactorizationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationBlockPreconditionerFallbackCount() const {
    return lastFullEquivHybridTranslationBlockPreconditionerFallbackCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount()
      const {
    return
        lastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount()
      const {
    return
        lastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount()
      const {
    return lastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount()
      const {
    return
        lastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount()
      const {
    return
        lastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount()
      const {
    return lastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount()
      const {
    return lastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount;
  }

  std::size_t
  getLastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount()
      const {
    return
        lastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount;
  }

  std::size_t
  getLastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount()
      const {
    return
        lastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount;
  }

  std::size_t
  getLastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount()
      const {
    return
        lastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount;
  }

  std::size_t
  getLastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount() const {
    return lastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount;
  }

  std::size_t
  getLastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension() const {
    return lastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension;
  }

  std::size_t
  getLastFullEquivHybridReducedRotationPreconditionerFactorizationCount() const {
    return lastFullEquivHybridReducedRotationPreconditionerFactorizationCount;
  }

  std::size_t getLastFullEquivHybridRqnUsedCount() const {
    return lastFullEquivHybridRqnUsedCount;
  }

  std::size_t getLastFullEquivHybridRqnAcceptedPairCount() const {
    return lastFullEquivHybridRqnAcceptedPairCount;
  }

  std::size_t getLastFullEquivHybridRqnRejectedPairCount() const {
    return lastFullEquivHybridRqnRejectedPairCount;
  }

  std::size_t getLastFullEquivHybridRqnMemorySize() const {
    return lastFullEquivHybridRqnMemorySize;
  }

  std::size_t
  getLastFullEquivHybridRqnPreconditionerApplicationCount() const {
    return lastFullEquivHybridRqnPreconditionerApplicationCount;
  }

  ManualDpgoMmAmmTrace getLastAmmTrace() const { return lastAmmTrace; }

  ManualDpgoMmOptimizerProfile getOptimizerProfile() const {
    ManualDpgoMmOptimizerProfile profile = optimizerProfile;
    profile.reducedTranslationRecoverySec +=
        reducedOptimizer.getProfileTranslationRecoverySeconds();
    profile.reducedTranslationFactorizationSec +=
        reducedOptimizer.getProfileTranslationFactorizationSeconds();
    profile.reducedTranslationCacheLookupSec +=
        reducedOptimizer.getProfileTranslationCacheLookupSeconds();
    profile.reducedHessianProductSec +=
        reducedOptimizer.getProfileReducedHessianProductSeconds();
    profile.reducedObjectiveSec +=
        reducedOptimizer.getProfileReducedObjectiveSeconds();
    profile.reducedGradientBuildSec +=
        reducedOptimizer.getProfileReducedGradientBuildSeconds();
    profile.reducedPreconditionerSec +=
        reducedOptimizer.getProfileReducedPreconditionerSeconds();
    profile.reducedProjectionSec +=
        reducedOptimizer.getProfileReducedProjectionSeconds();
    profile.reducedRetractionSec +=
        reducedOptimizer.getProfileReducedRetractionSeconds();
    profile.reducedStepNormSec +=
        reducedOptimizer.getProfileReducedStepNormSeconds();
    profile.reducedTrustRegionScalingSec +=
        reducedOptimizer.getProfileReducedTrustRegionScalingSeconds();
    profile.reducedTranslationResponseSec +=
        reducedOptimizer.getProfileReducedTranslationResponseSeconds();
    profile.reducedRotationProductSec +=
        reducedOptimizer.getProfileReducedRotationProductSeconds();
    profile.reducedTranslationRecoveryCount +=
        reducedOptimizer.getProfileTranslationRecoveryCount();
    profile.reducedTranslationFactorizationCount +=
        reducedOptimizer.getProfileTranslationFactorizationCount();
    profile.reducedTranslationCacheLookupCount +=
        reducedOptimizer.getProfileTranslationCacheLookupCount();
    profile.reducedHessianProductCount +=
        reducedOptimizer.getProfileReducedHessianProductCount();
    profile.reducedObjectiveCount +=
        reducedOptimizer.getProfileReducedObjectiveCount();
    profile.reducedGradientBuildCount +=
        reducedOptimizer.getProfileReducedGradientBuildCount();
    profile.reducedPreconditionerCount +=
        reducedOptimizer.getProfileReducedPreconditionerCount();
    profile.reducedProjectionCount +=
        reducedOptimizer.getProfileReducedProjectionCount();
    profile.reducedRetractionCount +=
        reducedOptimizer.getProfileReducedRetractionCount();
    profile.reducedStepNormCount +=
        reducedOptimizer.getProfileReducedStepNormCount();
    profile.reducedTrustRegionScalingCount +=
        reducedOptimizer.getProfileReducedTrustRegionScalingCount();
    profile.reducedTranslationResponseCount +=
        reducedOptimizer.getProfileReducedTranslationResponseCount();
    profile.reducedRotationProductCount +=
        reducedOptimizer.getProfileReducedRotationProductCount();
    profile.reducedCurvatureCauchyCandidateCount +=
        reducedOptimizer.getProfileCurvatureCauchyCandidateCount();
    profile.reducedCurvatureCauchyAcceptedCount +=
        reducedOptimizer.getProfileCurvatureCauchyAcceptedCount();
    profile.reducedCurvatureCauchyFallbackCandidateCount +=
        reducedOptimizer.getProfileCurvatureCauchyFallbackCandidateCount();
    profile.reducedCurvatureCauchyFallbackAcceptedCount +=
        reducedOptimizer.getProfileCurvatureCauchyFallbackAcceptedCount();
    profile.reducedCandidateProjectionSkipCount +=
        reducedOptimizer.getProfileReducedCandidateProjectionSkipCount();
    return profile;
  }

  void resetLocalHistory() {
    previousX.resize(0, 0);
    hasPreviousX = false;
    previousG.resize(0, 0);
    hasPreviousG = false;
    previousFixedPointInput.resize(0, 0);
    previousFixedPointOutput.resize(0, 0);
    hasPreviousFixedPoint = false;
    ammInitialized = false;
    ammS = 1.0;
    ammFk0 = 0.0;
    ammFk1 = 0.0;
    ammPreviousCost = 0.0;
    ammSoftRestartHits0 = 0;
    ammSoftRestartHits1 = 0;
    ammCooldown = 0;
    ammNumOscillations = 0;
    ammOscillations.clear();
    recursiveSimpleInitialized = false;
    recursiveSimplePreviousZRows.resize(0, 0);
    recursiveSimplePreviousGk = 0.0;
    fullEquivHybridRqnMemory.clear();
    lastFullEquivHybridRqnMemorySize = 0;
  }

  bool syncAmmReferenceToCurrentState() {
    if (options.scheme != ManualDpgoMmScheme::AMM) {
      return false;
    }
    const double currentCost = problem.f(X) + modelConstant;
    if (!std::isfinite(currentCost)) {
      return false;
    }
    ammInitialized = true;
    ammFk0 = currentCost;
    ammFk1 = currentCost;
    ammPreviousCost = currentCost;
    ammSoftRestartHits0 = 0;
    ammSoftRestartHits1 = 0;
    ammCooldown = 0;
    ammNumOscillations = 0;
    ammOscillations.clear();
    ammOscillations.push_back(1);
    recursiveSimpleInitialized = false;
    recursiveSimplePreviousZRows.resize(0, 0);
    recursiveSimplePreviousGk = 0.0;
    return true;
  }

  bool applyLocalGradientCorrection(const std::vector<double> &steps,
                                    std::size_t &accepted,
                                    std::size_t &rejected) {
    accepted = 0;
    rejected = 0;
    if (steps.empty()) {
      return true;
    }
    if (!updateLocalModel()) {
      return false;
    }
    const double currentCost = problem.f(X) + modelConstant;
    if (!std::isfinite(currentCost)) {
      rejected = steps.size();
      return true;
    }
    const Matrix gradient = localGradientCorrectionDirection(
        problem.RieGrad(X));
    if (gradient.rows() != X.rows() || gradient.cols() != X.cols() ||
        !gradient.allFinite() || gradient.squaredNorm() <= 0.0) {
      rejected = steps.size();
      return true;
    }

    Matrix bestX = X;
    double bestCost = currentCost;
    bool improved = false;
    for (const double step : steps) {
      const Matrix trialX = projectRotationBlocks(X - step * gradient);
      if (trialX.rows() != X.rows() || trialX.cols() != X.cols() ||
          !trialX.allFinite()) {
        ++rejected;
        continue;
      }
      const double trialCost = problem.f(trialX) + modelConstant;
      if (std::isfinite(trialCost) && trialCost < bestCost - 1e-12) {
        bestX = trialX;
        bestCost = trialCost;
        improved = true;
      } else {
        ++rejected;
      }
    }

    if (!improved) {
      return true;
    }
    X = bestX;
    accepted = 1;
    if (hasPreviousFixedPoint &&
        previousFixedPointOutput.rows() == X.rows() &&
        previousFixedPointOutput.cols() == X.cols()) {
      previousFixedPointOutput = X;
    }
    if (options.scheme == ManualDpgoMmScheme::AMM &&
        options.syncAmmReferenceAfterLocalGradientCorrection) {
      syncAmmReferenceToCurrentState();
    }
    return true;
  }

  bool evaluateLocalGradientCorrectionCosts(
      const std::vector<double> &steps, double &currentCost,
      std::vector<double> &trialCosts) {
    trialCosts.assign(steps.size(),
                      std::numeric_limits<double>::infinity());
    if (steps.empty()) {
      currentCost = 0.0;
      return true;
    }
    if (!updateLocalModel()) {
      return false;
    }
    currentCost = problem.f(X) + modelConstant;
    if (!std::isfinite(currentCost)) {
      return true;
    }
    const Matrix rawGradient = problem.RieGrad(X);
    const Matrix direction = localGradientCorrectionDirection(rawGradient);
    if (direction.rows() != X.rows() || direction.cols() != X.cols() ||
        !direction.allFinite() || direction.squaredNorm() <= 0.0) {
      return true;
    }
    for (std::size_t idx = 0; idx < steps.size(); ++idx) {
      const Matrix trialX = projectRotationBlocks(X - steps[idx] * direction);
      if (trialX.rows() != X.rows() || trialX.cols() != X.cols() ||
          !trialX.allFinite()) {
        continue;
      }
      trialCosts[idx] = problem.f(trialX) + modelConstant;
    }
    return true;
  }

  bool evaluateLocalGradientCorrectionCurvature(
      LocalGradientCorrectionCurvatureProbe &probe) {
    probe = LocalGradientCorrectionCurvatureProbe();
    if (!updateLocalModel()) {
      return false;
    }
    probe.maxStep = localGradientCorrectionStepCap();
    probe.currentCost = problem.f(X) + modelConstant;
    probe.currentFinite = std::isfinite(probe.currentCost);
    if (!probe.currentFinite) {
      probe.ok = true;
      return true;
    }
    const Matrix rawGradient = problem.RieGrad(X);
    const Matrix direction = localGradientCorrectionDirection(rawGradient);
    if (rawGradient.rows() != X.rows() || rawGradient.cols() != X.cols() ||
        !rawGradient.allFinite() || direction.rows() != X.rows() ||
        direction.cols() != X.cols() || !direction.allFinite() ||
        direction.squaredNorm() <= 0.0) {
      probe.ok = true;
      return true;
    }
    const Matrix hessianDirection = direction * problem.getQRef();
    if (hessianDirection.rows() != X.rows() ||
        hessianDirection.cols() != X.cols() || !hessianDirection.allFinite()) {
      probe.ok = true;
      return true;
    }
    probe.descentNumerator = (rawGradient.cwiseProduct(direction)).sum();
    probe.curvature = (hessianDirection.cwiseProduct(direction)).sum();
    probe.ok = true;
    return true;
  }

  bool evaluateLocalGradientCorrectionBaseDirection(Matrix &direction) {
    direction = Matrix();
    if (!updateLocalModel()) {
      return false;
    }
    const Matrix rawGradient = problem.RieGrad(X);
    if (rawGradient.rows() != X.rows() || rawGradient.cols() != X.cols() ||
        !rawGradient.allFinite()) {
      return true;
    }
    direction = localGradientCorrectionDirection(rawGradient);
    if (direction.rows() != X.rows() || direction.cols() != X.cols() ||
        !direction.allFinite()) {
      direction = Matrix();
    }
    return true;
  }

  bool getSharedDirection(const Matrix &direction, unsigned poseIndex,
                          Matrix &block) const {
    if (poseIndex >= n || direction.rows() != static_cast<int>(r) ||
        direction.cols() != static_cast<int>(n * (d + 1))) {
      return false;
    }
    block = direction.block(0, poseIndex * (d + 1), r, d + 1);
    return block.allFinite();
  }

  bool evaluateLocalGradientCorrectionCoupledCurvature(
      const PoseDict &neighborDirections,
      LocalGradientCorrectionCurvatureProbe &probe, Matrix &direction) {
    probe = LocalGradientCorrectionCurvatureProbe();
    direction = Matrix();
    if (!updateLocalModel()) {
      return false;
    }
    probe.maxStep = localGradientCorrectionStepCap();
    probe.currentCost = problem.f(X) + modelConstant;
    probe.currentFinite = std::isfinite(probe.currentCost);
    if (!probe.currentFinite) {
      probe.ok = true;
      return true;
    }

    const Matrix rawGradient = problem.RieGrad(X);
    if (rawGradient.rows() != X.rows() || rawGradient.cols() != X.cols() ||
        !rawGradient.allFinite()) {
      probe.ok = true;
      return true;
    }

    Matrix coupledGradient = rawGradient;
    const unsigned blockDim = d + 1;
    for (const auto &m : sharedLoops) {
      PoseID neighborPose;
      unsigned localPose = 0;
      bool localIsSource = false;
      if (m.r1 == id) {
        neighborPose = std::make_pair(m.r2, m.p2);
        localPose = m.p1;
        localIsSource = true;
      } else if (m.r2 == id) {
        neighborPose = std::make_pair(m.r1, m.p1);
        localPose = m.p2;
      } else {
        continue;
      }
      if (localPose >= n) {
        continue;
      }
      const auto directionIt = neighborDirections.find(neighborPose);
      if (directionIt == neighborDirections.end() ||
          directionIt->second.rows() != static_cast<int>(r) ||
          directionIt->second.cols() != static_cast<int>(blockDim) ||
          !directionIt->second.allFinite()) {
        continue;
      }

      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(row, row) = m.weight * m.kappa;
      }
      Omega(d, d) = m.weight * m.tau;

      Matrix crossAction;
      if (localIsSource) {
        crossAction = -directionIt->second * Omega * T.transpose();
      } else {
        crossAction = -directionIt->second * T * Omega;
      }
      if (crossAction.rows() == static_cast<int>(r) &&
          crossAction.cols() == static_cast<int>(blockDim) &&
          crossAction.allFinite()) {
        coupledGradient.block(0, localPose * blockDim, r, blockDim) -=
            crossAction;
      }
    }

    direction = localGradientCorrectionDirection(coupledGradient);
    if (direction.rows() != X.rows() || direction.cols() != X.cols() ||
        !direction.allFinite() || direction.squaredNorm() <= 0.0) {
      probe.ok = true;
      return true;
    }
    const Matrix hessianDirection = direction * problem.getQRef();
    if (hessianDirection.rows() != X.rows() ||
        hessianDirection.cols() != X.cols() || !hessianDirection.allFinite()) {
      probe.ok = true;
      return true;
    }
    probe.descentNumerator = (rawGradient.cwiseProduct(direction)).sum();
    probe.curvature = (hessianDirection.cwiseProduct(direction)).sum();
    probe.ok = true;
    return true;
  }

  bool applyLocalGradientCorrectionStep(double step, std::size_t &accepted,
                                        std::size_t &rejected,
                                        bool requireDecrease = false) {
    accepted = 0;
    rejected = 0;
    if (!std::isfinite(step) || step < 0.0 || !updateLocalModel()) {
      rejected = 1;
      return std::isfinite(step) && step >= 0.0;
    }
    const double currentCost = problem.f(X) + modelConstant;
    const Matrix rawGradient = problem.RieGrad(X);
    const Matrix direction = localGradientCorrectionDirection(rawGradient);
    if (!std::isfinite(currentCost) || direction.rows() != X.rows() ||
        direction.cols() != X.cols() || !direction.allFinite() ||
        direction.squaredNorm() <= 0.0) {
      rejected = 1;
      return true;
    }
    const Matrix trialX = projectRotationBlocks(X - step * direction);
    if (trialX.rows() != X.rows() || trialX.cols() != X.cols() ||
        !trialX.allFinite()) {
      rejected = 1;
      return true;
    }
    const double trialCost = problem.f(trialX) + modelConstant;
    if (!std::isfinite(trialCost)) {
      rejected = 1;
      return true;
    }
    if (requireDecrease && trialCost >= currentCost - 1e-12) {
      rejected = 1;
      return true;
    }
    X = trialX;
    accepted = 1;
    if (hasPreviousFixedPoint &&
        previousFixedPointOutput.rows() == X.rows() &&
        previousFixedPointOutput.cols() == X.cols()) {
      previousFixedPointOutput = X;
    }
    if (options.scheme == ManualDpgoMmScheme::AMM &&
        options.syncAmmReferenceAfterLocalGradientCorrection) {
      syncAmmReferenceToCurrentState();
    }
    return true;
  }

  bool applyLocalGradientCorrectionDirectionStep(
      const Matrix &direction, double step, std::size_t &accepted,
      std::size_t &rejected, bool requireDecrease = false) {
    accepted = 0;
    rejected = 0;
    if (!std::isfinite(step) || step < 0.0 || !updateLocalModel()) {
      rejected = 1;
      return std::isfinite(step) && step >= 0.0;
    }
    const double currentCost = problem.f(X) + modelConstant;
    if (!std::isfinite(currentCost) || direction.rows() != X.rows() ||
        direction.cols() != X.cols() || !direction.allFinite() ||
        direction.squaredNorm() <= 0.0) {
      rejected = 1;
      return true;
    }
    const Matrix trialX = projectRotationBlocks(X - step * direction);
    if (trialX.rows() != X.rows() || trialX.cols() != X.cols() ||
        !trialX.allFinite()) {
      rejected = 1;
      return true;
    }
    const double trialCost = problem.f(trialX) + modelConstant;
    if (!std::isfinite(trialCost)) {
      rejected = 1;
      return true;
    }
    if (requireDecrease && trialCost >= currentCost - 1e-12) {
      rejected = 1;
      return true;
    }
    X = trialX;
    accepted = 1;
    if (hasPreviousFixedPoint &&
        previousFixedPointOutput.rows() == X.rows() &&
        previousFixedPointOutput.cols() == X.cols()) {
      previousFixedPointOutput = X;
    }
    if (options.scheme == ManualDpgoMmScheme::AMM &&
        options.syncAmmReferenceAfterLocalGradientCorrection) {
      syncAmmReferenceToCurrentState();
    }
    return true;
  }

  bool applyFullEquivHybridSchwarzPreSmoothing() {
    if (!options.fullEquivHybridSchwarzPreSmoothing ||
        options.fullEquivHybridSchwarzSweeps == 0) {
      return true;
    }
    if (X.rows() != static_cast<int>(r) ||
        X.cols() != static_cast<int>(n * (d + 1)) || !X.allFinite()) {
      return false;
    }
    ScopedSecondsAccumulator smoothingTimer(
        lastFullEquivHybridSchwarzSmoothingTimeSec);
    try {
      const SparseMatrix &Q = problem.getQRef();
      const Matrix denseG = Matrix(problem.getGRef());
      const int blockCols = static_cast<int>(d + 1);
      if (Q.rows() != static_cast<int>(n * (d + 1)) ||
          Q.cols() != static_cast<int>(n * (d + 1)) ||
          denseG.rows() != X.rows() || denseG.cols() != X.cols()) {
        return false;
      }

      double currentCost = problem.f(X);
      if (!std::isfinite(currentCost)) {
        return false;
      }

      const unsigned backtrackingSteps =
          std::max(1u, options.fullEquivHybridWarmStartBacktrackingSteps);
      auto applySchwarzBlock = [&](const std::vector<unsigned> &poses) {
        ++lastFullEquivHybridSchwarzSmoothingCandidateCount;
        if (poses.empty()) {
          ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
          return false;
        }

        const int blockDim =
            blockCols * static_cast<int>(poses.size());
        Matrix gradientBlock =
            Matrix::Zero(static_cast<int>(r), blockDim);
        Matrix hBlock = Matrix::Zero(blockDim, blockDim);
        Matrix currentBlock =
            Matrix::Zero(static_cast<int>(r), blockDim);

        for (std::size_t localI = 0; localI < poses.size(); ++localI) {
          const unsigned poseI = poses[localI];
          if (poseI >= n) {
            ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
            return false;
          }
          const int colI = static_cast<int>(poseI * (d + 1));
          const int localColI = static_cast<int>(localI) * blockCols;
          gradientBlock.block(0, localColI, static_cast<int>(r), blockCols) =
              X * Q.block(0, colI, Q.rows(), blockCols) +
              denseG.block(0, colI, static_cast<int>(r), blockCols);
          currentBlock.block(0, localColI, static_cast<int>(r), blockCols) =
              X.block(0, colI, static_cast<int>(r), blockCols);
          for (std::size_t localJ = 0; localJ < poses.size(); ++localJ) {
            const unsigned poseJ = poses[localJ];
            if (poseJ >= n) {
              ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
              return false;
            }
            const int colJ = static_cast<int>(poseJ * (d + 1));
            const int localColJ = static_cast<int>(localJ) * blockCols;
            hBlock.block(localColI, localColJ, blockCols, blockCols) =
                Matrix(Q.block(colI, colJ, blockCols, blockCols));
          }
        }

        hBlock = 0.5 * (hBlock + hBlock.transpose());
        if (!gradientBlock.allFinite() || !hBlock.allFinite() ||
            !currentBlock.allFinite()) {
          ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
          return false;
        }

        const double baseDamping =
            std::max(0.0, options.fullEquivHybridSchurDamping);
        bool solved = false;
        Matrix rawStepBlock = Matrix::Zero(static_cast<int>(r), blockDim);
        for (unsigned jitter = 0; jitter < 5 && !solved; ++jitter) {
          const double damping =
              baseDamping +
              (jitter == 0
                   ? 0.0
                   : std::pow(10.0, static_cast<int>(jitter) - 12));
          const Matrix regularizedBlock =
              hBlock + damping * Matrix::Identity(blockDim, blockDim);
          Eigen::LDLT<Matrix> ldlt(regularizedBlock);
          if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
            continue;
          }
          const Matrix solvedBlock = ldlt.solve(-gradientBlock.transpose());
          if (ldlt.info() != Eigen::Success ||
              solvedBlock.rows() != blockDim ||
              solvedBlock.cols() != static_cast<int>(r) ||
              !solvedBlock.allFinite()) {
            continue;
          }
          rawStepBlock = solvedBlock.transpose();
          solved = true;
        }
        if (!solved) {
          ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
          return false;
        }

        Matrix tangentBlock = Matrix::Zero(static_cast<int>(r), blockDim);
        for (std::size_t localI = 0; localI < poses.size(); ++localI) {
          const int localCol = static_cast<int>(localI) * blockCols;
          const Matrix projectedBlock = projectFullEquivHybridTangent(
              currentBlock.block(0, localCol, static_cast<int>(r), blockCols),
              rawStepBlock.block(0, localCol, static_cast<int>(r), blockCols),
              d);
          if (projectedBlock.rows() != static_cast<int>(r) ||
              projectedBlock.cols() != blockCols ||
              !projectedBlock.allFinite()) {
            ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
            return false;
          }
          tangentBlock.block(0, localCol, static_cast<int>(r), blockCols) =
              projectedBlock;
        }
        if (tangentBlock.squaredNorm() <= 0.0) {
          ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
          return false;
        }

        const double maxBlockNorm =
            options.fullEquivHybridSchwarzMaxBlockNorm;
        const double stepNorm = tangentBlock.norm();
        if (std::isfinite(maxBlockNorm) && maxBlockNorm > 0.0 &&
            stepNorm > maxBlockNorm) {
          tangentBlock *= maxBlockNorm / stepNorm;
        }

        bool accepted = false;
        double scale = 1.0;
        for (unsigned attempt = 0; attempt < backtrackingSteps; ++attempt) {
          Matrix candidateBlock = Matrix::Zero(static_cast<int>(r), blockDim);
          bool validCandidate = true;
          for (std::size_t localI = 0; localI < poses.size(); ++localI) {
            const int localCol = static_cast<int>(localI) * blockCols;
            const Matrix retractedBlock = retractFullEquivHybridByProjection(
                currentBlock.block(0, localCol, static_cast<int>(r),
                                   blockCols),
                scale * tangentBlock.block(0, localCol, static_cast<int>(r),
                                           blockCols),
                d);
            if (retractedBlock.rows() != static_cast<int>(r) ||
                retractedBlock.cols() != blockCols ||
                !retractedBlock.allFinite()) {
              validCandidate = false;
              break;
            }
            candidateBlock.block(0, localCol, static_cast<int>(r),
                                 blockCols) = retractedBlock;
          }
          if (validCandidate) {
            const Matrix deltaBlock = candidateBlock - currentBlock;
            const double linearDelta =
                (deltaBlock.cwiseProduct(gradientBlock)).sum();
            const double quadraticDelta =
                0.5 * ((deltaBlock * hBlock).cwiseProduct(deltaBlock)).sum();
            const double candidateCost =
                currentCost + linearDelta + quadraticDelta;
            if (std::isfinite(candidateCost) &&
                candidateCost < currentCost - 1e-12) {
              lastFullEquivHybridSchwarzSmoothingCostDecrease +=
                  currentCost - candidateCost;
              currentCost = candidateCost;
              for (std::size_t localI = 0; localI < poses.size(); ++localI) {
                const int modelCol =
                    static_cast<int>(poses[localI] * (d + 1));
                const int localCol = static_cast<int>(localI) * blockCols;
                X.block(0, modelCol, static_cast<int>(r), blockCols) =
                    candidateBlock.block(0, localCol, static_cast<int>(r),
                                         blockCols);
              }
              ++lastFullEquivHybridSchwarzSmoothingAcceptedCount;
              accepted = true;
              break;
            }
          }
          scale *= 0.5;
        }
        if (!accepted) {
          ++lastFullEquivHybridSchwarzSmoothingRejectedCount;
        }
        return accepted;
      };

      auto collectEdgePairBlocks = [&]() {
        std::set<std::pair<unsigned, unsigned>> uniquePairs;
        for (int outer = 0; outer < Q.outerSize(); ++outer) {
          for (SparseMatrix::InnerIterator it(Q, outer); it; ++it) {
            const unsigned poseI =
                static_cast<unsigned>(it.row() / blockCols);
            const unsigned poseJ =
                static_cast<unsigned>(it.col() / blockCols);
            if (poseI >= n || poseJ >= n || poseI == poseJ ||
                it.value() == 0.0) {
              continue;
            }
            const unsigned a = std::min(poseI, poseJ);
            const unsigned b = std::max(poseI, poseJ);
            uniquePairs.emplace(a, b);
          }
        }
        std::vector<std::vector<unsigned>> blocks;
        blocks.reserve(uniquePairs.size());
        for (const auto &pair : uniquePairs) {
          blocks.push_back({pair.first, pair.second});
        }
        return blocks;
      };

      for (unsigned sweep = 0; sweep < options.fullEquivHybridSchwarzSweeps;
           ++sweep) {
        ++lastFullEquivHybridSchwarzSmoothingSweepCount;
        std::size_t acceptedThisSweep = 0;
        if (options.fullEquivHybridSchwarzBlockMode ==
            ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair) {
          const std::vector<std::vector<unsigned>> edgeBlocks =
              collectEdgePairBlocks();
          for (const auto &block : edgeBlocks) {
            if (applySchwarzBlock(block)) {
              ++acceptedThisSweep;
            }
          }
          if (edgeBlocks.empty()) {
            for (unsigned pose = 0; pose < n; ++pose) {
              if (applySchwarzBlock({pose})) {
                ++acceptedThisSweep;
              }
            }
          }
        } else {
          for (unsigned pose = 0; pose < n; ++pose) {
            if (applySchwarzBlock({pose})) {
              ++acceptedThisSweep;
            }
          }
        }
        if (acceptedThisSweep == 0) {
          break;
        }
      }
    } catch (const std::exception &) {
      return false;
    }
    return true;
  }

  Matrix makeFullEquivHybridSchurWarmStart(const Matrix &start) {
    return makeGuardedFullEquivHybridSchurCandidate(
        start, lastFullEquivHybridWarmStartCandidateCount,
        lastFullEquivHybridWarmStartAcceptedCount,
        lastFullEquivHybridWarmStartGuardRejectedCount, false, false);
  }

  Matrix makeFullEquivHybridSchurStep(
      const Matrix &start, bool allowTranslationRecoveryStepTrials = true,
      bool allowTranslationRecoveryInitialGuess = true) {
    return makeGuardedFullEquivHybridSchurCandidate(
        start, lastFullEquivHybridSchurStepCandidateCount,
        lastFullEquivHybridSchurStepAcceptedCount,
        lastFullEquivHybridSchurStepGuardRejectedCount,
        allowTranslationRecoveryStepTrials,
        allowTranslationRecoveryInitialGuess);
  }

  Matrix makeReducedRotationInitialStepGuessForFullEquivHybrid(
      const Matrix &start, double baseCost, bool &valid) {
    valid = false;
    ++lastFullEquivHybridReducedRotationInitialGuessCandidateCount;
    auto reject = [&]() {
      ++lastFullEquivHybridReducedRotationInitialGuessRejectedCount;
      return Matrix();
    };
    struct ReducedInitialGuessCandidate {
      Matrix step;
      double cost{std::numeric_limits<double>::quiet_NaN()};
      bool ok{false};
    };
    auto candidateIsBetter =
        [&](const ReducedInitialGuessCandidate &candidate,
            const ReducedInitialGuessCandidate &best) {
          if (!candidate.ok) {
            return false;
          }
          if (!best.ok) {
            return true;
          }
          const double tolerance =
              std::max(0.0, options.localCandidateCostTieTolerance);
          if (candidate.cost < best.cost - tolerance) {
            return true;
          }
          return tolerance > 0.0 &&
                 candidate.cost <= best.cost + tolerance &&
                 candidate.step.squaredNorm() < best.step.squaredNorm();
        };
    auto evaluateReducedInitialGuess =
        [&](ManualDpgoMmReducedRotationPreconditioner preconditioner) {
          ReducedInitialGuessCandidate candidate;
          reducedOptimizer.setVerbose(false);
          reducedOptimizer.setTrustRegionTolerance(options.trustRegionTolerance);
          reducedOptimizer.setTrustRegionIterations(
              options.trustRegionIterations);
          reducedOptimizer.setTrustRegionAcceptedIterations(
              options.trustRegionAcceptedIterations);
          reducedOptimizer.setTrustRegionMaxInnerIterations(
              options.trustRegionMaxInnerIterations);
          reducedOptimizer.setTruncatedCgRelativeTolerance(
              options.reducedRotationTcgRelativeTolerance);
          reducedOptimizer.setTrustRegionInitialRadius(
              options.trustRegionInitialRadius);
          reducedOptimizer.setRecordResultStats(false);
          reducedOptimizer.setValidateTranslationRecoveryCost(false);
          reducedOptimizer.setUseJacobiPreconditioner(
              preconditioner ==
              ManualDpgoMmReducedRotationPreconditioner::Jacobi);
          reducedOptimizer.setUseSchurJacobiPreconditioner(
              preconditioner ==
              ManualDpgoMmReducedRotationPreconditioner::SchurJacobi);
          reducedOptimizer.setUseCholeskyPreconditioner(
              preconditioner ==
              ManualDpgoMmReducedRotationPreconditioner::Cholesky);
          reducedOptimizer.setSkipRedundantCandidateProjection(
              options.reducedRotationSkipRedundantCandidateProjection);

          const Matrix reducedCandidate = reducedOptimizer.optimize(start);
          if (reducedCandidate.rows() != start.rows() ||
              reducedCandidate.cols() != start.cols() ||
              !reducedCandidate.allFinite()) {
            return candidate;
          }
          candidate.cost = problem.f(reducedCandidate);
          if (!std::isfinite(candidate.cost) ||
              candidate.cost >= baseCost - 1e-12) {
            return candidate;
          }
          candidate.step = projectFullEquivHybridTangent(
              start, reducedCandidate - start, d);
          candidate.ok =
              candidate.step.rows() == start.rows() &&
              candidate.step.cols() == start.cols() &&
              candidate.step.allFinite() && candidate.step.squaredNorm() > 0.0;
          return candidate;
        };
    try {
      ReducedInitialGuessCandidate bestCandidate;
      if (options.reducedRotationPreconditioner ==
          ManualDpgoMmReducedRotationPreconditioner::Portfolio) {
        bestCandidate = evaluateReducedInitialGuess(
            ManualDpgoMmReducedRotationPreconditioner::None);
        const ReducedInitialGuessCandidate jacobiCandidate =
            evaluateReducedInitialGuess(
                ManualDpgoMmReducedRotationPreconditioner::Jacobi);
        if (candidateIsBetter(jacobiCandidate, bestCandidate)) {
          bestCandidate = jacobiCandidate;
        }
        const ReducedInitialGuessCandidate choleskyCandidate =
            evaluateReducedInitialGuess(
                ManualDpgoMmReducedRotationPreconditioner::Cholesky);
        if (candidateIsBetter(choleskyCandidate, bestCandidate)) {
          bestCandidate = choleskyCandidate;
        }
      } else {
        bestCandidate =
            evaluateReducedInitialGuess(options.reducedRotationPreconditioner);
      }
      if (!bestCandidate.ok) {
        return reject();
      }
      valid = true;
      ++lastFullEquivHybridReducedRotationInitialGuessUsedCount;
      return bestCandidate.step;
    } catch (const std::exception &) {
      return reject();
    }
  }

  Matrix makeTranslationRecoveryInitialStepGuessForFullEquivHybrid(
      const Matrix &start, const Matrix &euclideanGradient, double damping,
      bool &valid) {
    valid = false;
    ++lastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount;
    auto reject = [&]() {
      ++lastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount;
      return Matrix();
    };

    const unsigned blockDim = d + 1;
    if (start.rows() != static_cast<int>(r) ||
        start.cols() != static_cast<int>(n * blockDim) ||
        euclideanGradient.rows() != start.rows() ||
        euclideanGradient.cols() != start.cols() || !start.allFinite() ||
        !euclideanGradient.allFinite()) {
      return reject();
    }

    const SparseMatrix &Q = problem.getQRef();
    if (Q.rows() != static_cast<int>(n * blockDim) ||
        Q.cols() != static_cast<int>(n * blockDim)) {
      return reject();
    }

    Matrix hTT = Matrix::Zero(static_cast<int>(n), static_cast<int>(n));
    for (unsigned poseI = 0; poseI < n; ++poseI) {
      const int colI = static_cast<int>(poseI * blockDim + d);
      for (unsigned poseJ = 0; poseJ < n; ++poseJ) {
        const int colJ = static_cast<int>(poseJ * blockDim + d);
        hTT(static_cast<int>(poseI), static_cast<int>(poseJ)) =
            Q.coeff(colI, colJ);
      }
      hTT(static_cast<int>(poseI), static_cast<int>(poseI)) +=
          std::max(0.0, damping);
    }
    if (!hTT.allFinite()) {
      return reject();
    }

    Matrix rhs = Matrix::Zero(static_cast<int>(n), static_cast<int>(r));
    for (unsigned pose = 0; pose < n; ++pose) {
      rhs.row(static_cast<int>(pose)) =
          -euclideanGradient.col(static_cast<int>(pose * blockDim + d))
               .transpose();
    }
    Eigen::LDLT<Matrix> ldlt(hTT);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
      return reject();
    }
    const Matrix solvedTranslations = ldlt.solve(rhs);
    if (ldlt.info() != Eigen::Success || !solvedTranslations.allFinite()) {
      return reject();
    }

    Matrix step = Matrix::Zero(start.rows(), start.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      step.col(static_cast<int>(pose * blockDim + d)) =
          solvedTranslations.row(static_cast<int>(pose)).transpose();
    }
    step = projectFullEquivHybridTangent(start, step, d);
    if (step.rows() != start.rows() || step.cols() != start.cols() ||
        !step.allFinite() || step.squaredNorm() <= 0.0) {
      return reject();
    }

    const double maxNorm = options.fullEquivHybridWarmStartMaxNorm;
    const double stepNorm = step.norm();
    if (std::isfinite(maxNorm) && maxNorm > 0.0 && stepNorm > maxNorm) {
      step *= maxNorm / stepNorm;
    }
    const double predictedDecrease =
        fullEquivHybridProjectedModelPredictedDecrease(
            step, euclideanGradient, damping);
    if (!std::isfinite(predictedDecrease) || predictedDecrease <= 0.0) {
      return reject();
    }

    valid = true;
    ++lastFullEquivHybridTranslationRecoveryInitialGuessUsedCount;
    return step;
  }

  void recordFullEquivHybridStepTrial(double predictedDecrease,
                                      double actualDecrease, double scale,
                                      bool accepted) {
    if (!std::isfinite(predictedDecrease) ||
        !std::isfinite(actualDecrease)) {
      return;
    }
    ++lastFullEquivHybridStepTrialCount;
    if (accepted) {
      ++lastFullEquivHybridStepTrialAcceptedCount;
      if (std::isfinite(scale)) {
        lastFullEquivHybridStepAcceptedScaleSum += scale;
      }
      lastFullEquivHybridStepAcceptedPredictedDecreaseSum += predictedDecrease;
      lastFullEquivHybridStepAcceptedActualDecreaseSum += actualDecrease;
      if (predictedDecrease > 1e-18) {
        lastFullEquivHybridStepAcceptedRhoSum +=
            actualDecrease / predictedDecrease;
        ++lastFullEquivHybridStepAcceptedRhoCount;
      }
    } else {
      ++lastFullEquivHybridStepTrialRejectedCount;
    }
    lastFullEquivHybridStepPredictedDecreaseSum += predictedDecrease;
    lastFullEquivHybridStepActualDecreaseSum += actualDecrease;
    if (predictedDecrease > 1e-18) {
      lastFullEquivHybridStepRhoSum += actualDecrease / predictedDecrease;
      ++lastFullEquivHybridStepRhoCount;
    }
  }

  double fullEquivHybridSharedPoseStepSquaredNorm(const Matrix &step) const {
    if (step.rows() != static_cast<int>(r) ||
        step.cols() != static_cast<int>(n * (d + 1)) || !step.allFinite()) {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<char> isShared(n, 0);
    for (const auto &m : sharedLoops) {
      unsigned localPose = n;
      if (m.r1 == id) {
        localPose = m.p1;
      } else if (m.r2 == id) {
        localPose = m.p2;
      }
      if (localPose < n) {
        isShared[localPose] = 1;
      }
    }
    double squaredNorm = 0.0;
    for (unsigned pose = 0; pose < n; ++pose) {
      if (!isShared[pose]) {
        continue;
      }
      squaredNorm += step.block(0, pose * (d + 1), r, d + 1).squaredNorm();
    }
    return squaredNorm;
  }

  double fullEquivHybridProjectedModelPredictedDecrease(
      const Matrix &realizedStep, const Matrix &euclideanGradient,
      double damping) const {
    if (realizedStep.rows() != static_cast<int>(r) ||
        realizedStep.cols() != static_cast<int>(n * (d + 1)) ||
        euclideanGradient.rows() != realizedStep.rows() ||
        euclideanGradient.cols() != realizedStep.cols() ||
        !realizedStep.allFinite() || !euclideanGradient.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double linearDelta =
        (realizedStep.cwiseProduct(euclideanGradient)).sum();
    const double quadraticDelta =
        0.5 * ((realizedStep * problem.getQRef()).cwiseProduct(realizedStep))
                  .sum();
    const double dampingDelta =
        0.5 * std::max(0.0, damping) * realizedStep.squaredNorm();
    return -(linearDelta + quadraticDelta + dampingDelta);
  }

  Matrix makeGuardedFullEquivHybridSchurCandidate(
      const Matrix &start, std::size_t &candidateCount,
      std::size_t &acceptedCount, std::size_t &guardRejectedCount,
      bool allowTranslationRecoveryStepTrials = true,
      bool allowTranslationRecoveryInitialGuess = true) {
    ++candidateCount;
    auto reject = [&]() {
      ++guardRejectedCount;
      return start;
    };
    if (start.rows() != static_cast<int>(r) ||
        start.cols() != static_cast<int>(n * (d + 1)) ||
        !start.allFinite()) {
      return reject();
    }
    const double baseCost = problem.f(start);
    if (!std::isfinite(baseCost)) {
      return reject();
    }

    try {
      const Matrix denseG = Matrix(problem.getGRef());
      if (denseG.rows() != start.rows() || denseG.cols() != start.cols()) {
        return reject();
      }
      const Matrix euclideanGradient = start * problem.getQRef() + denseG;
      if (!euclideanGradient.allFinite() ||
          euclideanGradient.squaredNorm() <= 0.0) {
        return reject();
      }

      Matrix rawStep;
      double linearFullResidual = std::numeric_limits<double>::quiet_NaN();
      {
        ScopedSecondsAccumulator linearSolveTimer(
            lastFullEquivHybridLinearSolveTimeSec);
        if (options.fullEquivHybridBackend ==
            ManualDpgoMmFullEquivHybridBackend::PcgSchur) {
          const FullEquivHybridEliminationPlan plan =
              makeFullEquivHybridTranslationEliminationPlan(n, d);
          const FullEquivHybridSchurSystem schur =
              buildFullEquivHybridExactSchurSystem(
                  problem.getQRef(), euclideanGradient, plan,
                  options.fullEquivHybridSchurDamping);
          FullEquivHybridPcgOptions pcgOptions;
          pcgOptions.relativeTolerance =
              options.fullEquivHybridLinearRelativeTolerance;
          pcgOptions.absoluteTolerance =
              options.fullEquivHybridLinearAbsoluteTolerance;
          pcgOptions.maxIterations =
              options.fullEquivHybridLinearMaxIterations;
          pcgOptions.useBlockJacobiPreconditioner =
              options.fullEquivHybridLinearBlockJacobiPreconditioner;
          pcgOptions.useSparseMatrixVectorProduct =
              options.fullEquivHybridSparseMatrixVectorProduct;
          pcgOptions.useTranslationSchurPreconditioner =
              options.fullEquivHybridLinearTranslationSchurPreconditioner;
          const FullEquivHybridPcgResult pcg =
              solveFullEquivHybridSchurPcg(
                  schur, euclideanGradient, plan, pcgOptions);
          lastFullEquivHybridLinearPcgIterationCount += pcg.iterations;
          lastFullEquivHybridLinearInitialResidual =
              std::max(lastFullEquivHybridLinearInitialResidual,
                       pcg.initialResidual);
          lastFullEquivHybridLinearFinalResidual =
              std::max(lastFullEquivHybridLinearFinalResidual,
                       pcg.finalResidual);
          if (!pcg.converged || pcg.step.rows() != start.rows() ||
              pcg.step.cols() != start.cols() || !pcg.step.allFinite()) {
            return reject();
          }
          rawStep = pcg.step;
        } else if (options.fullEquivHybridBackend ==
                   ManualDpgoMmFullEquivHybridBackend::PcgFull) {
          FullEquivHybridPcgOptions pcgOptions;
          pcgOptions.relativeTolerance =
              options.fullEquivHybridLinearRelativeTolerance;
          pcgOptions.absoluteTolerance =
              options.fullEquivHybridLinearAbsoluteTolerance;
          pcgOptions.maxIterations =
              options.fullEquivHybridLinearMaxIterations;
          pcgOptions.useBlockJacobiPreconditioner =
              options.fullEquivHybridLinearBlockJacobiPreconditioner;
          pcgOptions.useSparseMatrixVectorProduct =
              options.fullEquivHybridSparseMatrixVectorProduct;
          pcgOptions.useTranslationSchurPreconditioner =
              options.fullEquivHybridLinearTranslationSchurPreconditioner;
          pcgOptions.useLocalChainPreconditioner =
              options.fullEquivHybridLocalChainPreconditioner;
          pcgOptions.useTranslationBlockPreconditioner =
              options.fullEquivHybridTranslationBlockPreconditioner;
          pcgOptions.useTranslationSparseSchurPreconditioner =
              options.fullEquivHybridTranslationSparseSchurPreconditioner;
          pcgOptions.useTranslationLocalSchurPreconditioner =
              options.fullEquivHybridTranslationLocalSchurPreconditioner;
          pcgOptions.translationLocalSchurMaxActivePoses =
              options.fullEquivHybridTranslationLocalSchurMaxActivePoses;
          pcgOptions.useLaplacianDeflationPreconditioner =
              options.fullEquivHybridLaplacianDeflationPreconditioner;
          pcgOptions.laplacianDeflationBasisSize =
              options.fullEquivHybridLaplacianDeflationBasisSize;
          pcgOptions.laplacianDeflationMaxEigenPoses =
              options.fullEquivHybridLaplacianDeflationMaxEigenPoses;
          pcgOptions.useReducedRotationPreconditioner =
              options.fullEquivHybridReducedRotationPreconditioner;
          pcgOptions.reducedRotationPreconditioner =
              makeFullEquivHybridReducedRotationPreconditioner(
                  options.reducedRotationPreconditioner);
          pcgOptions.useRqnMemoryPreconditioner =
              options.fullEquivHybridRqnMemoryPreconditioner;
          if (options.fullEquivHybridRqnMemoryPreconditioner &&
              fullEquivHybridRqnMemory.size() > 0) {
            pcgOptions.rqnMemoryPreconditioner =
                &fullEquivHybridRqnMemory;
          }
          bool hasInitialStepGuess = false;
          if (options.fullEquivHybridReducedRotationInitialGuess) {
            bool reducedInitialGuessValid = false;
            const Matrix reducedInitialGuess =
                makeReducedRotationInitialStepGuessForFullEquivHybrid(
                    start, baseCost, reducedInitialGuessValid);
            if (reducedInitialGuessValid) {
              pcgOptions.initialStepGuess = reducedInitialGuess;
              hasInitialStepGuess = true;
            }
          }
          if (!hasInitialStepGuess &&
              options.fullEquivHybridTranslationRecoveryInitialGuess &&
              allowTranslationRecoveryInitialGuess) {
            bool translationInitialGuessValid = false;
            const Matrix translationInitialGuess =
                makeTranslationRecoveryInitialStepGuessForFullEquivHybrid(
                    start, euclideanGradient,
                    options.fullEquivHybridSchurDamping,
                    translationInitialGuessValid);
            if (translationInitialGuessValid) {
              pcgOptions.initialStepGuess = translationInitialGuess;
              hasInitialStepGuess = true;
            }
          }
          if (!hasInitialStepGuess &&
              options.fullEquivHybridRqnWarmStart &&
              fullEquivHybridRqnMemory.size() > 0) {
            try {
              const Matrix projectedGradient =
                  projectFullEquivHybridTangent(
                      start, euclideanGradient, d);
              const Matrix initialStep =
                  -fullEquivHybridRqnMemory.applyInverseHessian(
                      projectedGradient);
              if (initialStep.rows() == start.rows() &&
                  initialStep.cols() == start.cols() &&
                  initialStep.allFinite()) {
                pcgOptions.initialStepGuess = initialStep;
                ++lastFullEquivHybridRqnUsedCount;
              }
            } catch (const std::exception &) {
            }
          }
          const FullEquivHybridPcgResult pcg =
              solveFullEquivHybridFullPcg(
                  problem.getQRef(), euclideanGradient,
                  options.fullEquivHybridSchurDamping, d, pcgOptions);
          lastFullEquivHybridLinearPcgIterationCount += pcg.iterations;
          lastFullEquivHybridLinearInitialResidual =
              std::max(lastFullEquivHybridLinearInitialResidual,
                       pcg.initialResidual);
          lastFullEquivHybridLinearFinalResidual =
              std::max(lastFullEquivHybridLinearFinalResidual,
                       pcg.finalResidual);
          lastFullEquivHybridSparseMatrixVectorProductCount +=
              pcg.sparseMatrixVectorProductCount;
          lastFullEquivHybridTranslationSchurPreconditionerApplicationCount +=
              pcg.translationSchurPreconditionerApplicationCount;
          lastFullEquivHybridTranslationSchurPreconditionerFactorizationCount +=
              pcg.translationSchurPreconditionerFactorizationCount;
          lastFullEquivHybridTranslationSchurPreconditionerFallbackCount +=
              pcg.translationSchurPreconditionerFallbackCount;
          lastFullEquivHybridLocalChainPreconditionerApplicationCount +=
              pcg.localChainPreconditionerApplicationCount;
          lastFullEquivHybridLocalChainPreconditionerFactorizationCount +=
              pcg.localChainPreconditionerFactorizationCount;
          lastFullEquivHybridLocalChainPreconditionerFallbackCount +=
              pcg.localChainPreconditionerFallbackCount;
          lastFullEquivHybridTranslationBlockPreconditionerApplicationCount +=
              pcg.translationBlockPreconditionerApplicationCount;
          lastFullEquivHybridTranslationBlockPreconditionerFactorizationCount +=
              pcg.translationBlockPreconditionerFactorizationCount;
          lastFullEquivHybridTranslationBlockPreconditionerFallbackCount +=
              pcg.translationBlockPreconditionerFallbackCount;
          lastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount +=
              pcg.translationSparseSchurPreconditionerApplicationCount;
          lastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount +=
              pcg.translationSparseSchurPreconditionerFactorizationCount;
          lastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount +=
              pcg.translationSparseSchurPreconditionerFallbackCount;
          lastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount +=
              pcg.translationLocalSchurPreconditionerApplicationCount;
          lastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount +=
              pcg.translationLocalSchurPreconditionerFactorizationCount;
          lastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount +=
              pcg.translationLocalSchurPreconditionerFallbackCount;
          lastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount =
              std::max(
                  lastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount,
                  pcg.translationLocalSchurPreconditionerActivePoseCount);
          lastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount =
              std::max(
                  lastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount,
                  pcg.translationLocalSchurPreconditionerActiveColumnCount);
          lastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount +=
              pcg.laplacianDeflationPreconditionerApplicationCount;
          lastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount +=
              pcg.laplacianDeflationPreconditionerFactorizationCount;
          lastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount +=
              pcg.laplacianDeflationPreconditionerFallbackCount;
          lastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension =
              std::max(
                  lastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension,
                  pcg.laplacianDeflationPreconditionerBasisDimension);
          lastFullEquivHybridReducedRotationPreconditionerApplicationCount +=
              pcg.reducedRotationPreconditionerApplicationCount;
          lastFullEquivHybridReducedRotationPreconditionerFactorizationCount +=
              pcg.reducedRotationPreconditionerFactorizationCount;
          lastFullEquivHybridRqnPreconditionerApplicationCount +=
              pcg.rqnMemoryPreconditionerApplicationCount;
          if (!pcg.converged || pcg.step.rows() != start.rows() ||
              pcg.step.cols() != start.cols() || !pcg.step.allFinite()) {
            return reject();
          }
          rawStep = pcg.step;
        } else {
          const FullEquivHybridEliminationPlan plan =
              makeFullEquivHybridTranslationEliminationPlan(n, d);
          const FullEquivHybridSchurSystem schur =
              buildFullEquivHybridExactSchurSystem(
                  problem.getQRef(), euclideanGradient, plan,
                  options.fullEquivHybridSchurDamping);
          rawStep =
              solveFullEquivHybridSchurDirect(schur, euclideanGradient, plan);
        }
      }
      if (rawStep.rows() == start.rows() && rawStep.cols() == start.cols() &&
          rawStep.allFinite()) {
        const double fullResidual = fullEquivHybridRelativeLinearResidual(
            problem.getQRef(), euclideanGradient, rawStep,
            options.fullEquivHybridSchurDamping);
        if (std::isfinite(fullResidual)) {
          linearFullResidual = fullResidual;
        }
      }
      Matrix tangentStep = projectFullEquivHybridTangent(start, rawStep, d);
      if (tangentStep.rows() != start.rows() ||
          tangentStep.cols() != start.cols() || !tangentStep.allFinite() ||
          tangentStep.squaredNorm() <= 0.0) {
        return reject();
      }

      const double maxNorm = options.fullEquivHybridWarmStartMaxNorm;
      const double stepNorm = tangentStep.norm();
      if (std::isfinite(maxNorm) && maxNorm > 0.0 && stepNorm > maxNorm) {
        tangentStep *= maxNorm / stepNorm;
      }

      const unsigned backtrackingSteps =
          std::max(1u, options.fullEquivHybridWarmStartBacktrackingSteps);
      const double sharedPoseProxWeight = std::max(
          0.0, options.fullEquivHybridBacktrackingSharedPoseProxWeight);
      const bool projectedModelRhoGuard =
          options.fullEquivHybridProjectedModelRhoGuard;
      const double projectedModelRhoEta =
          std::max(0.0, options.fullEquivHybridProjectedModelRhoEta);
      const bool enableTranslationRecoveryStepTrials =
          options.fullEquivHybridTranslationRecoveryStepTrials &&
          allowTranslationRecoveryStepTrials;
      const bool selectBestBacktrackingTrial =
          options.fullEquivHybridSelectBestBacktrackingTrial ||
          sharedPoseProxWeight > 0.0 ||
          options.fullEquivHybridStepParetoSelector ||
          enableTranslationRecoveryStepTrials;
      struct BacktrackingTrialRecord {
        double predictedDecrease{std::numeric_limits<double>::quiet_NaN()};
        double actualDecrease{std::numeric_limits<double>::quiet_NaN()};
        double scale{std::numeric_limits<double>::quiet_NaN()};
        double candidateCost{std::numeric_limits<double>::quiet_NaN()};
        double candidateMerit{std::numeric_limits<double>::quiet_NaN()};
        double candidateGradient{std::numeric_limits<double>::quiet_NaN()};
        bool hardAccepted{false};
        bool translationRecovered{false};
        Matrix step;
        Matrix candidate;
      };
      std::vector<BacktrackingTrialRecord> trialRecords;
      if (selectBestBacktrackingTrial) {
        trialRecords.reserve(backtrackingSteps);
      }
      bool bestTrialValid = false;
      std::size_t bestTrialIndex = 0;
      double bestTrialMerit = std::numeric_limits<double>::infinity();
      double bestTrialScale = 1.0;
      Matrix bestTrialStep;
      Matrix bestTrialCandidate;
      auto finishAcceptedCandidate = [&](const Matrix &acceptedStep) {
        ++acceptedCount;
        if (std::isfinite(linearFullResidual)) {
          lastFullEquivHybridLinearFullResidual =
              std::max(lastFullEquivHybridLinearFullResidual,
                       linearFullResidual);
        }
        if ((options.fullEquivHybridRqnWarmStart ||
             options.fullEquivHybridRqnMemoryPreconditioner) &&
            options.fullEquivHybridBackend ==
                ManualDpgoMmFullEquivHybridBackend::PcgFull) {
          Matrix curvature =
              acceptedStep * problem.getQRef() +
              options.fullEquivHybridSchurDamping * acceptedStep;
          curvature =
              projectFullEquivHybridTangent(start, curvature, d);
          if (curvature.rows() == acceptedStep.rows() &&
              curvature.cols() == acceptedStep.cols() &&
              curvature.allFinite() &&
              fullEquivHybridRqnMemory.addPair(acceptedStep, curvature)) {
            ++lastFullEquivHybridRqnAcceptedPairCount;
          } else {
            ++lastFullEquivHybridRqnRejectedPairCount;
          }
          lastFullEquivHybridRqnMemorySize =
              fullEquivHybridRqnMemory.size();
        }
      };
      auto considerBacktrackingTrial =
          [&](const Matrix &candidate, const Matrix &modelStep, double scale,
              bool translationRecovered) {
            if (candidate.rows() != start.rows() ||
                candidate.cols() != start.cols() || !candidate.allFinite() ||
                modelStep.rows() != start.rows() ||
                modelStep.cols() != start.cols() || !modelStep.allFinite()) {
              return false;
            }
            const Matrix projectedCandidateStep =
                projectFullEquivHybridTangent(start, candidate - start, d);
            if (projectedCandidateStep.rows() != start.rows() ||
                projectedCandidateStep.cols() != start.cols() ||
                !projectedCandidateStep.allFinite() ||
                projectedCandidateStep.squaredNorm() <= 0.0) {
              return false;
            }
            if (translationRecovered) {
              ++lastFullEquivHybridTranslationRecoveryStepTrialCandidateCount;
            }
          const double candidateCost = problem.f(candidate);
            const double predictedDecrease =
                fullEquivHybridProjectedModelPredictedDecrease(
                    projectedCandidateStep, euclideanGradient,
                    options.fullEquivHybridSchurDamping);
          const double actualDecrease = baseCost - candidateCost;
          if (selectBestBacktrackingTrial) {
            BacktrackingTrialRecord record;
            record.predictedDecrease = predictedDecrease;
            record.actualDecrease = actualDecrease;
            record.scale = scale;
            record.candidateCost = candidateCost;
              record.translationRecovered = translationRecovered;
            trialRecords.push_back(std::move(record));
          }
          double candidateMerit = candidateCost;
          if (std::isfinite(candidateCost) && sharedPoseProxWeight > 0.0) {
            const double sharedStepSquaredNorm =
                fullEquivHybridSharedPoseStepSquaredNorm(candidate - start);
            if (std::isfinite(sharedStepSquaredNorm)) {
              candidateMerit +=
                  sharedPoseProxWeight * sharedStepSquaredNorm;
            } else {
              candidateMerit = std::numeric_limits<double>::infinity();
            }
          }
          const bool candidateCostDecreases =
              std::isfinite(candidateCost) &&
              candidateCost < baseCost - 1e-12;
          const bool candidateMeritDecreases =
              sharedPoseProxWeight <= 0.0 ||
              (std::isfinite(candidateMerit) &&
               candidateMerit < baseCost - 1e-12);
          bool projectedModelAccepts = true;
          if (projectedModelRhoGuard && candidateCostDecreases) {
            projectedModelAccepts =
                std::isfinite(predictedDecrease) &&
                predictedDecrease > 1e-18 &&
                actualDecrease / predictedDecrease >= projectedModelRhoEta;
          }
          if (candidateCostDecreases && candidateMeritDecreases &&
              projectedModelAccepts) {
              if (translationRecovered) {
                ++lastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount;
              }
            double candidateGradient =
                std::numeric_limits<double>::quiet_NaN();
            if (options.fullEquivHybridStepParetoSelector) {
              ++lastFullEquivHybridStepParetoGradientEvalCount;
              try {
                ScopedSecondsAccumulator gradientTimer(
                    lastFullEquivHybridStepParetoGradientEvalTimeSec);
                candidateGradient = problem.RieGradNorm(candidate);
              } catch (const std::exception &) {
                candidateGradient = std::numeric_limits<double>::infinity();
              }
            }
            if (selectBestBacktrackingTrial) {
              if (!trialRecords.empty()) {
                BacktrackingTrialRecord &record = trialRecords.back();
                record.candidateMerit = candidateMerit;
                record.candidateGradient = candidateGradient;
                record.hardAccepted = true;
                  record.step = projectedCandidateStep;
                record.candidate = candidate;
              }
              const double tolerance =
                  std::max(0.0, options.localCandidateCostTieTolerance);
              if (!bestTrialValid ||
                  candidateMerit < bestTrialMerit - tolerance ||
                  (tolerance > 0.0 &&
                   candidateMerit <= bestTrialMerit + tolerance &&
                   scale < bestTrialScale)) {
                bestTrialValid = true;
                bestTrialIndex = trialRecords.size() - 1u;
                bestTrialMerit = candidateMerit;
                bestTrialScale = scale;
                  bestTrialStep = projectedCandidateStep;
                bestTrialCandidate = candidate;
              }
            } else {
              recordFullEquivHybridStepTrial(predictedDecrease,
                                             actualDecrease, scale, true);
                finishAcceptedCandidate(projectedCandidateStep);
                if (translationRecovered) {
                  ++lastFullEquivHybridTranslationRecoveryStepTrialSelectedCount;
                }
                return true;
            }
          } else if (std::isfinite(candidateCost)) {
            if (!selectBestBacktrackingTrial) {
              recordFullEquivHybridStepTrial(
                  predictedDecrease, actualDecrease, scale, false);
            }
          }
            return false;
          };

      auto addTranslationRecoveryStepTrials =
          [&](const Matrix &candidate, double scale,
              Matrix &immediateAcceptedCandidate) {
            if (!enableTranslationRecoveryStepTrials) {
              return false;
            }
            const TranslationRecoveryResult recovery =
                recoverMajorizedTranslationsWithStats(candidate);
            if (!recovery.accepted ||
                recovery.value.rows() != candidate.rows() ||
                recovery.value.cols() != candidate.cols() ||
                !recovery.value.allFinite()) {
              return false;
            }
            bool immediateAccepted = false;
            auto considerRecoveredValue = [&](const Matrix &value) {
              if (immediateAccepted || value.rows() != start.rows() ||
                  value.cols() != start.cols() || !value.allFinite()) {
                return;
              }
              const Matrix recoveredStep =
                  projectFullEquivHybridTangent(start, value - start, d);
              if (recoveredStep.rows() != start.rows() ||
                  recoveredStep.cols() != start.cols() ||
                  !recoveredStep.allFinite() ||
                  recoveredStep.squaredNorm() <= 0.0) {
                return;
              }
              immediateAccepted = considerBacktrackingTrial(
                  value, recoveredStep, scale, true);
              if (immediateAccepted) {
                immediateAcceptedCandidate = value;
              }
            };

            considerRecoveredValue(recovery.value);
            if (options.fullEquivHybridTranslationRecoveryPolishMeritSelector) {
              const Matrix delta = recovery.value - candidate;
              double alpha = 0.5;
              for (unsigned trial = 0;
                   trial <
                   options
                       .fullEquivHybridTranslationRecoveryPolishBacktrackingSteps;
                   ++trial, alpha *= 0.5) {
                Matrix partial = candidate;
                for (unsigned pose = 0; pose < n; ++pose) {
                  const unsigned col = pose * (d + 1) + d;
                  partial.col(static_cast<int>(col)) =
                      candidate.col(static_cast<int>(col)) +
                      alpha * delta.col(static_cast<int>(col));
                }
                considerRecoveredValue(partial);
              }
            }
            return immediateAccepted;
          };

      double scale = 1.0;
      for (unsigned attempt = 0; attempt < backtrackingSteps; ++attempt) {
        const Matrix trialStep = scale * tangentStep;
        const Matrix candidate =
            retractFullEquivHybridByProjection(start, trialStep, d);
        if (considerBacktrackingTrial(candidate, trialStep, scale, false)) {
          return candidate;
        }
        Matrix immediateTranslationRecoveryCandidate;
        if (addTranslationRecoveryStepTrials(
                candidate, scale, immediateTranslationRecoveryCandidate)) {
          return immediateTranslationRecoveryCandidate;
        }
        scale *= 0.5;
      }
      if (selectBestBacktrackingTrial) {
        bool selectedTrialValid = bestTrialValid;
        std::size_t selectedTrialIndex = bestTrialIndex;
        Matrix selectedTrialStep = bestTrialStep;
        Matrix selectedTrialCandidate = bestTrialCandidate;
        if (options.fullEquivHybridStepParetoSelector && bestTrialValid) {
          const double tolerance =
              std::max(0.0, options.localCandidateCostTieTolerance);
          double bestActualDecrease =
              -std::numeric_limits<double>::infinity();
          std::size_t paretoCandidateCount = 0;
          for (const auto &record : trialRecords) {
            if (record.hardAccepted) {
              ++paretoCandidateCount;
              if (std::isfinite(record.actualDecrease)) {
                bestActualDecrease =
                    std::max(bestActualDecrease, record.actualDecrease);
              }
            }
          }
          if (paretoCandidateCount > 0 &&
              std::isfinite(bestActualDecrease)) {
            lastFullEquivHybridStepParetoCandidateCount +=
                paretoCandidateCount;
            const double minRequiredDecrease =
                std::max(0.0,
                         options.fullEquivHybridStepParetoMinDecreaseRatio) *
                bestActualDecrease;
            const double proxMeritThreshold =
                sharedPoseProxWeight > 0.0 && std::isfinite(bestTrialMerit)
                    ? bestTrialMerit + std::max(1e-12, tolerance)
                    : std::numeric_limits<double>::infinity();
            bool paretoSelected = false;
            std::size_t paretoIndex = selectedTrialIndex;
            double selectedGradient = std::numeric_limits<double>::infinity();
            double selectedCost = std::numeric_limits<double>::infinity();
            double selectedDecrease =
                -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < trialRecords.size(); ++i) {
              const BacktrackingTrialRecord &record = trialRecords[i];
              if (!record.hardAccepted ||
                  !std::isfinite(record.actualDecrease) ||
                  record.actualDecrease + 1e-12 < minRequiredDecrease) {
                continue;
              }
              if (sharedPoseProxWeight > 0.0 &&
                  (!std::isfinite(record.candidateMerit) ||
                   record.candidateMerit > proxMeritThreshold)) {
                continue;
              }
              if (!std::isfinite(record.candidateGradient)) {
                continue;
              }
              const bool betterGradient =
                  !paretoSelected ||
                  record.candidateGradient < selectedGradient - 1e-12;
              const bool tiedGradient =
                  paretoSelected &&
                  std::abs(record.candidateGradient - selectedGradient) <=
                      1e-12;
              const bool betterTie =
                  tiedGradient &&
                  (record.candidateCost < selectedCost - tolerance ||
                   (record.candidateCost <= selectedCost + tolerance &&
                    record.actualDecrease > selectedDecrease + 1e-12));
              if (betterGradient || betterTie) {
                paretoSelected = true;
                paretoIndex = i;
                selectedGradient = record.candidateGradient;
                selectedCost = record.candidateCost;
                selectedDecrease = record.actualDecrease;
              }
            }
            if (paretoSelected) {
              const BacktrackingTrialRecord &selected =
                  trialRecords[paretoIndex];
              selectedTrialValid = true;
              selectedTrialIndex = paretoIndex;
              selectedTrialStep = selected.step;
              selectedTrialCandidate = selected.candidate;
              ++lastFullEquivHybridStepParetoSelectedCount;
              lastFullEquivHybridStepParetoSelectedScaleSum +=
                  selected.scale;
              lastFullEquivHybridStepParetoBestDecreaseSum +=
                  bestActualDecrease;
              lastFullEquivHybridStepParetoSelectedDecreaseSum +=
                  selected.actualDecrease;
              lastFullEquivHybridStepParetoSelectedGradientSum +=
                  selected.candidateGradient;
            }
          }
        }
        for (std::size_t i = 0; i < trialRecords.size(); ++i) {
          const BacktrackingTrialRecord &record = trialRecords[i];
          recordFullEquivHybridStepTrial(
              record.predictedDecrease, record.actualDecrease, record.scale,
              selectedTrialValid && i == selectedTrialIndex);
        }
        if (selectedTrialValid &&
            selectedTrialCandidate.rows() == start.rows() &&
            selectedTrialCandidate.cols() == start.cols() &&
            selectedTrialCandidate.allFinite() &&
            selectedTrialStep.rows() == start.rows() &&
            selectedTrialStep.cols() == start.cols() &&
            selectedTrialStep.allFinite()) {
          if (selectedTrialIndex < trialRecords.size() &&
              trialRecords[selectedTrialIndex].translationRecovered) {
            ++lastFullEquivHybridTranslationRecoveryStepTrialSelectedCount;
          }
          finishAcceptedCandidate(selectedTrialStep);
          return selectedTrialCandidate;
        }
      }
    } catch (const std::exception &) {
      return reject();
    }
    return reject();
  }

  SparseMatrix getQ() const { return problem.getQ(); }

  SparseMatrix getG() const { return problem.getG(); }

 private:
  double boundaryEdgeCost(const Matrix &value) const {
    if (value.rows() != static_cast<int>(r) ||
        value.cols() != static_cast<int>(n * (d + 1)) ||
        !value.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double cost = 0.0;
    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1.0;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(row, row) = m.weight * m.kappa;
      }
      Omega(d, d) = m.weight * m.tau;

      Matrix residual;
      if (m.r1 == id) {
        const auto neighborIt = neighborPoseDict.find(PoseID(m.r2, m.p2));
        if (neighborIt == neighborPoseDict.end()) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        const Matrix local =
            value.block(0, m.p1 * (d + 1), r, d + 1);
        residual = local * T - neighborIt->second;
      } else if (m.r2 == id) {
        const auto neighborIt = neighborPoseDict.find(PoseID(m.r1, m.p1));
        if (neighborIt == neighborPoseDict.end()) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        const Matrix local =
            value.block(0, m.p2 * (d + 1), r, d + 1);
        residual = neighborIt->second * T - local;
      } else {
        continue;
      }
      cost += 0.5 * residual.cwiseProduct(residual * Omega).sum();
    }
    return std::isfinite(cost) ? std::max(0.0, cost)
                               : std::numeric_limits<double>::quiet_NaN();
  }

  double separatorDeltaNorm(const Matrix &before, const Matrix &after) const {
    if (before.rows() != after.rows() || before.cols() != after.cols() ||
        before.rows() != static_cast<int>(r) ||
        before.cols() != static_cast<int>(n * (d + 1)) ||
        !before.allFinite() || !after.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::set<unsigned> separatorPoses;
    for (const auto &m : sharedLoops) {
      if (m.r1 == id) {
        separatorPoses.insert(m.p1);
      } else if (m.r2 == id) {
        separatorPoses.insert(m.p2);
      }
    }
    double squaredNorm = 0.0;
    for (const unsigned pose : separatorPoses) {
      if (pose >= n) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      const Matrix delta =
          after.block(0, pose * (d + 1), r, d + 1) -
          before.block(0, pose * (d + 1), r, d + 1);
      squaredNorm += delta.squaredNorm();
    }
    const double norm = std::sqrt(squaredNorm);
    return std::isfinite(norm) ? norm
                               : std::numeric_limits<double>::quiet_NaN();
  }

  unsigned id;
  unsigned d;
  unsigned r;
  unsigned n;
  std::vector<RelativeSEMeasurement> odometry;
  std::vector<RelativeSEMeasurement> privateLoops;
  std::vector<RelativeSEMeasurement> sharedLoops;
  std::vector<double> robotMeasurementDegrees;
  ManualDpgoMmOptions options;
  QuadraticProblem problem;
  SparseMatrix baseQ;
  double modelConstant{0.0};
  unsigned currentOptimizationRound{0};
  ManualQuadraticOptimizer manualOptimizer;
  ReducedRotationQuadraticOptimizer reducedOptimizer;
  Matrix X;
  PoseDict neighborPoseDict;
  PoseDict neighborPosePreviousDict;
  std::map<PoseID, unsigned> neighborPoseLastUpdateRound;
  std::map<PoseID, unsigned> neighborPosePreviousUpdateRound;
  std::map<unsigned, std::vector<unsigned>> neighborPublicPoses;
  ROPTResult lastOptimizationResult;
  std::size_t lastAdaptiveRefinementCount{0};
  std::size_t lastExtrapolationAcceptedCount{0};
  std::size_t lastExtrapolationRejectedCount{0};
  std::size_t lastGExtrapolationAcceptedCount{0};
  std::size_t lastGExtrapolationRejectedCount{0};
  std::size_t lastCoupledExtrapolationAcceptedCount{0};
  std::size_t lastCoupledExtrapolationRejectedCount{0};
  std::size_t lastAndersonAcceptedCount{0};
  std::size_t lastAndersonRejectedCount{0};
  std::size_t lastSquaremAcceptedCount{0};
  std::size_t lastSquaremRejectedCount{0};
  std::size_t lastAmmAcceleratedAcceptedCount{0};
  std::size_t lastAmmRestartCount{0};
  std::size_t lastAmmHardRestartCount{0};
  std::size_t lastAmmSoftRestartCount{0};
  std::size_t lastAmmPhiFallbackCount{0};
  std::size_t lastAmmLocalMeritRejectedCount{0};
  std::size_t lastAmmProximalStartCount{0};
  std::size_t lastAmmSkippedCount{0};
  std::size_t lastAmmMixedSurrogateCandidateCount{0};
  std::size_t lastAmmMixedSurrogateTrueLocalAcceptedCount{0};
  std::size_t lastAmmMixedSurrogateSimpleSelectedCount{0};
  std::size_t lastAmmMixedSurrogateTrueLocalSelectedCount{0};
  std::size_t lastAmmMixedSurrogateExtrapolatedSelectedCount{0};
  std::size_t lastAmmMixedSurrogateOtherSelectedCount{0};
  std::size_t lastAmmMixedSurrogateSimpleSkippedCount{0};
  std::size_t lastAmmMixedSurrogateSimpleForcedRefreshCount{0};
  std::size_t lastEdgeTightQuadraticEvalCount{0};
  double lastEdgeTightQuadraticSurrogateCostSum{0.0};
  double lastEdgeTightQuadraticTrueCostSum{0.0};
  double lastEdgeTightQuadraticMajorizationGapMin{
      std::numeric_limits<double>::infinity()};
  std::size_t lastVariableProjectedSchurCandidateCount{0};
  std::size_t lastVariableProjectedSchurAcceptedCount{0};
  std::size_t lastBoundaryProximalCandidateCount{0};
  std::size_t lastBoundaryProximalAcceptedCount{0};
  std::size_t lastSurrogateBoundCheckCount{0};
  std::size_t lastSurrogateBoundViolationCount{0};
  double lastSurrogateBoundMinMargin{
      std::numeric_limits<double>::infinity()};
  double lastBoundaryEdgeCostBefore{0.0};
  double lastBoundaryEdgeCostAfter{0.0};
  double lastSeparatorDeltaNorm{0.0};
  std::size_t lastFullEquivHybridWarmStartCandidateCount{0};
  std::size_t lastFullEquivHybridWarmStartAcceptedCount{0};
  std::size_t lastFullEquivHybridWarmStartGuardRejectedCount{0};
  std::size_t lastFullEquivHybridSchurStepCandidateCount{0};
  std::size_t lastFullEquivHybridSchurStepAcceptedCount{0};
  std::size_t lastFullEquivHybridSchurStepGuardRejectedCount{0};
  std::size_t lastFullEquivHybridLocalPortfolioCandidateCount{0};
  std::size_t lastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount{0};
  std::size_t lastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount{0};
  std::size_t lastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount{0};
  std::size_t lastFullEquivHybridSchwarzSmoothingSweepCount{0};
  std::size_t lastFullEquivHybridSchwarzSmoothingCandidateCount{0};
  std::size_t lastFullEquivHybridSchwarzSmoothingAcceptedCount{0};
  std::size_t lastFullEquivHybridSchwarzSmoothingRejectedCount{0};
  double lastFullEquivHybridSchwarzSmoothingCostDecrease{0.0};
  double lastFullEquivHybridSchwarzSmoothingTimeSec{0.0};
  std::size_t lastFullEquivHybridLinearPcgIterationCount{0};
  double lastFullEquivHybridLinearInitialResidual{0.0};
  double lastFullEquivHybridLinearFinalResidual{0.0};
  double lastFullEquivHybridLinearFullResidual{0.0};
  double lastFullEquivHybridLinearSolveTimeSec{0.0};
  std::size_t lastFullEquivHybridStepTrialCount{0};
  std::size_t lastFullEquivHybridStepTrialAcceptedCount{0};
  std::size_t lastFullEquivHybridStepTrialRejectedCount{0};
  double lastFullEquivHybridStepPredictedDecreaseSum{0.0};
  double lastFullEquivHybridStepActualDecreaseSum{0.0};
  double lastFullEquivHybridStepRhoSum{0.0};
  std::size_t lastFullEquivHybridStepRhoCount{0};
  double lastFullEquivHybridStepAcceptedScaleSum{0.0};
  double lastFullEquivHybridStepAcceptedPredictedDecreaseSum{0.0};
  double lastFullEquivHybridStepAcceptedActualDecreaseSum{0.0};
  double lastFullEquivHybridStepAcceptedRhoSum{0.0};
  std::size_t lastFullEquivHybridStepAcceptedRhoCount{0};
  std::size_t lastFullEquivHybridStepParetoCandidateCount{0};
  std::size_t lastFullEquivHybridStepParetoSelectedCount{0};
  double lastFullEquivHybridStepParetoSelectedScaleSum{0.0};
  double lastFullEquivHybridStepParetoBestDecreaseSum{0.0};
  double lastFullEquivHybridStepParetoSelectedDecreaseSum{0.0};
  double lastFullEquivHybridStepParetoSelectedGradientSum{0.0};
  std::size_t lastFullEquivHybridStepParetoGradientEvalCount{0};
  double lastFullEquivHybridStepParetoGradientEvalTimeSec{0.0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryStepTrialCandidateCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryStepTrialSelectedCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryInitialGuessUsedCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorCandidateCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorAcceptedCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorRejectedCount{0};
  double lastFullEquivHybridActiveSeparatorStepSum{0.0};
  double lastFullEquivHybridActiveSeparatorCostDecreaseSum{0.0};
  std::size_t lastFullEquivHybridActiveSeparatorLmSchurCandidateCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorLmSchurAcceptedCount{0};
  std::size_t
      lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount{0};
  std::size_t
      lastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount{0};
  std::size_t
      lastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorLmSchurFallbackCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount{0};
  std::size_t lastFullEquivHybridActiveSeparatorLmSchurPrivateColCount{0};
  std::size_t
      lastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount{0};
  double lastFullEquivHybridActiveSeparatorLmSchurAlphaSum{0.0};
  double lastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum{0.0};
  double lastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum{0.0};
  std::size_t lastFullEquivHybridTranslationRecoveryPolishAttemptCount{0};
  std::size_t lastFullEquivHybridTranslationRecoveryPolishAcceptedCount{0};
  std::size_t lastFullEquivHybridTranslationRecoveryPolishRejectedCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount{
          0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount{0};
  double lastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum{0.0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount{0};
  std::size_t
      lastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount{0};
  double lastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum{
      0.0};
  double lastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum{0.0};
  double lastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum{
      0.0};
  double lastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum{0.0};
  double lastFullEquivHybridTranslationRecoveryPolishGradientChangeSum{0.0};
  std::size_t lastFullEquivHybridSparseMatrixVectorProductCount{0};
  std::size_t
      lastFullEquivHybridReducedRotationInitialGuessCandidateCount{0};
  std::size_t lastFullEquivHybridReducedRotationInitialGuessUsedCount{0};
  std::size_t
      lastFullEquivHybridReducedRotationInitialGuessRejectedCount{0};
  std::size_t
      lastFullEquivHybridTranslationSchurPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridTranslationSchurPreconditionerFactorizationCount{0};
  std::size_t
      lastFullEquivHybridTranslationSchurPreconditionerFallbackCount{0};
  std::size_t
      lastFullEquivHybridLocalChainPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridLocalChainPreconditionerFactorizationCount{0};
  std::size_t lastFullEquivHybridLocalChainPreconditionerFallbackCount{0};
  std::size_t
      lastFullEquivHybridTranslationBlockPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridTranslationBlockPreconditionerFactorizationCount{0};
  std::size_t
      lastFullEquivHybridTranslationBlockPreconditionerFallbackCount{0};
  std::size_t
      lastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount{
          0};
  std::size_t
      lastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount{0};
  std::size_t
      lastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount{
          0};
  std::size_t
      lastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount{0};
  std::size_t
      lastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount{0};
  std::size_t
      lastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount{
          0};
  std::size_t
      lastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount{0};
  std::size_t
      lastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount{0};
  std::size_t
      lastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension{0};
  std::size_t
      lastFullEquivHybridReducedRotationPreconditionerApplicationCount{0};
  std::size_t
      lastFullEquivHybridReducedRotationPreconditionerFactorizationCount{0};
  std::size_t lastFullEquivHybridRqnUsedCount{0};
  std::size_t lastFullEquivHybridRqnAcceptedPairCount{0};
  std::size_t lastFullEquivHybridRqnRejectedPairCount{0};
  std::size_t lastFullEquivHybridRqnMemorySize{0};
  std::size_t lastFullEquivHybridRqnPreconditionerApplicationCount{0};
  FullEquivHybridRqnMemory fullEquivHybridRqnMemory;
  std::size_t ammMixedSurrogateTrueLocalWinStreak{0};
  std::size_t ammMixedSurrogateConsecutiveSimpleSkips{0};
  std::size_t reducedAdaptivePortfolioCalibrationCount{0};
  std::size_t reducedAdaptivePortfolioCholeskyWinStreak{0};
  std::size_t reducedAdaptivePortfolioFastPathSinceCalibration{0};
  ManualDpgoMmAmmTrace lastAmmTrace;
  mutable ManualDpgoMmOptimizerProfile optimizerProfile;
  Matrix previousX;
  bool hasPreviousX{false};
  SparseMatrix previousG;
  bool hasPreviousG{false};
  Matrix previousFixedPointInput;
  Matrix previousFixedPointOutput;
  bool hasPreviousFixedPoint{false};
  bool ammInitialized{false};
  double ammS{1.0};
  double ammFk0{0.0};
  double ammFk1{0.0};
  double ammPreviousCost{0.0};
  int ammSoftRestartHits0{0};
  int ammSoftRestartHits1{0};
  int ammCooldown{0};
  int ammNumOscillations{0};
  std::vector<int> ammOscillations;
  unsigned ammLocalIter{0};
  bool recursiveSimpleInitialized{false};
  Matrix recursiveSimplePreviousZRows;
  double recursiveSimplePreviousGk{0.0};
  mutable bool dpgoSimpleProximalCacheReady{false};
  mutable bool dpgoSimpleProximalCacheValid{false};
  mutable Vector dpgoSimpleInvDiagT;
  mutable Matrix dpgoSimpleScaleN;
  mutable ColMajorSparseMatrix dpgoSimpleScaleNSparse;
  mutable Matrix dpgoSimpleU;
  mutable ColMajorSparseMatrix dpgoSimpleUSparse;
  mutable SparseMatrix dpgoSimpleQManual;
  mutable SparseMatrix dpgoSimpleS;
  mutable SparseMatrix dpgoSimpleQFull;
  mutable SparseMatrix dpgoSimpleP;
  mutable SparseMatrix dpgoSimpleP0;
  mutable std::vector<PoseID> dpgoSimpleNeighborOrder;
  mutable LocalGradientCorrectionDiagnostics
      localGradientCorrectionDiagnostics;

  struct WeightedEdgeSplitModel {
    SparseMatrix q;
    SparseMatrix g;
    double theta{0.5};
    bool valid{false};
  };

  struct AmmStepState {
    double gamma{0.0};
    double nextS{1.0};
    int numOscillations{0};
    bool canAccelerate{false};
  };

  AmmStepState beginAmmStep(double currentCost) {
    AmmStepState step;
    if (!std::isfinite(currentCost)) {
      return step;
    }

    if (!ammInitialized) {
      ammInitialized = true;
      ammS = 1.0;
      ammFk0 = currentCost;
      ammFk1 = currentCost;
      ammPreviousCost = currentCost;
      ammNumOscillations = 0;
      ammOscillations.clear();
      ammOscillations.push_back(1);
    } else {
      if (currentCost <= ammFk1) {
        ammSoftRestartHits0 =
            ammSoftRestartHits0 > 2 ? ammSoftRestartHits0 - 2 : 0;
      } else {
        ++ammSoftRestartHits0;
      }

      if (currentCost <= ammPreviousCost) {
        ammSoftRestartHits1 = 0;
      } else {
        ++ammSoftRestartHits1;
      }
      const int oscillationState = currentCost <= ammPreviousCost ? 1 : 0;
      if (!ammOscillations.empty() &&
          oscillationState != ammOscillations.back()) {
        ++ammNumOscillations;
      }
      ammOscillations.push_back(oscillationState);
      const int oscillationPeriod =
          std::max(0, options.ammOscillationCountPeriod);
      const int currentIndex =
          static_cast<int>(ammOscillations.size()) - 1;
      if (oscillationPeriod > 0 && currentIndex > oscillationPeriod) {
        const int newer = currentIndex - oscillationPeriod;
        const int older = newer - 1;
        if (older >= 0 && ammOscillations[static_cast<std::size_t>(newer)] !=
                              ammOscillations[static_cast<std::size_t>(older)]) {
          ammNumOscillations =
              std::max(0, ammNumOscillations - 1);
        }
      }
      ammPreviousCost = currentCost;

      ammFk0 = ammFk0 * (1.0 - options.ammEta0) +
               currentCost * options.ammEta0;
      ammFk1 = std::max(currentCost, ammFk1 * (1.0 - options.ammEta1) +
                                         currentCost * options.ammEta1);
    }

    step.nextS = 0.5 + 0.5 * std::sqrt(4.0 * ammS * ammS + 1.0);
    step.gamma = options.ammGammaScale * (ammS - 1.0) / step.nextS;
    step.numOscillations = ammNumOscillations;
    step.canAccelerate = std::isfinite(step.gamma) && step.gamma > 0.0;
    return step;
  }

  void finishAmmStep(double nextS, bool hardRestart, bool anyRestart) {
    if (hardRestart) {
      nextS = std::max(0.5 * nextS, 1.0);
    }
    if (anyRestart) {
      ammSoftRestartHits0 /= 3;
      ammSoftRestartHits1 = 0;
    }
    if (std::isfinite(nextS) && nextS >= 1.0) {
      ammS = nextS;
    }
  }

  TranslationRecoveryResult
  recoverMajorizedTranslationsWithStats(const Matrix &start) const {
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer
            ? &optimizerProfile.explicitTranslationRecoverySec
            : nullptr);
    if (options.profileOptimizer) {
      ++optimizerProfile.explicitTranslationRecoveryCount;
    }
    TranslationRecoveryResult result;
    result.value = start;
    if (n == 0 || start.cols() != static_cast<int>((d + 1) * n)) {
      return result;
    }
    result.attempted = true;

    const SparseMatrix &Q = problem.getQRef();
    const SparseMatrix &G = problem.getGRef();
    std::vector<int> translationCols;
    translationCols.reserve(n);
    std::vector<char> isTranslation(start.cols(), 0);
    for (unsigned pose = 0; pose < n; ++pose) {
      const int col = static_cast<int>(pose * (d + 1) + d);
      translationCols.push_back(col);
      isTranslation[col] = 1;
    }

    Matrix fixedStart = start;
    for (int col : translationCols) {
      fixedStart.col(col).setZero();
    }

    const Matrix fixedTimesQ = fixedStart * Q;
    Matrix rhs(start.rows(), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      const int col = translationCols[pose];
      rhs.col(static_cast<int>(pose)) = -fixedTimesQ.col(col);
      for (int row = 0; row < G.rows(); ++row) {
        rhs(row, static_cast<int>(pose)) -= G.coeff(row, col);
      }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(Q.nonZeros()) + n);
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

    const double startCost = problem.f(start);
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
      std::vector<Eigen::Triplet<double>> regularizedTriplets = triplets;
      if (attempt > 0) {
        const double ridge =
            1e-10 * std::max(1.0, diagScale) * std::pow(10.0, attempt - 1);
        for (unsigned idx = 0; idx < n; ++idx) {
          regularizedTriplets.emplace_back(static_cast<int>(idx),
                                           static_cast<int>(idx), ridge);
        }
      }

      ColMajorSparseMatrix translationQ(static_cast<int>(n),
                                        static_cast<int>(n));
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
          solved.cols() != start.rows() || !solved.allFinite()) {
        continue;
      }

      Matrix recovered = start;
      for (unsigned pose = 0; pose < n; ++pose) {
        recovered.col(pose * (d + 1) + d) =
            solved.row(static_cast<int>(pose)).transpose();
      }
      const double recoveredCost = problem.f(recovered);
      if (std::isfinite(recoveredCost) &&
          (!std::isfinite(startCost) || recoveredCost <= startCost + 1e-10)) {
        result.value = recovered;
        result.accepted = true;
        return result;
      }
    }

    return result;
  }

  Matrix recoverMajorizedTranslations(const Matrix &start) const {
    return recoverMajorizedTranslationsWithStats(start).value;
  }

  int dpgoLocalRowToManualCol(unsigned row) const {
    if (row < n) {
      return static_cast<int>(row * (d + 1) + d);
    }
    const unsigned rotationIndex = row - n;
    const unsigned pose = rotationIndex / d;
    const unsigned localCol = rotationIndex % d;
    return static_cast<int>(pose * (d + 1) + localCol);
  }

  Matrix makeDpgoOrderedStateRows(const Matrix &base) const {
    if (n == 0 || base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite() ||
        !ensureDpgoSimpleProximalCache()) {
      return Matrix();
    }

    const unsigned neighborCount =
        static_cast<unsigned>(dpgoSimpleNeighborOrder.size());
    const unsigned totalRows = (d + 1) * (n + neighborCount);
    Matrix z = Matrix::Zero(static_cast<int>(totalRows), static_cast<int>(r));

    auto localT = [&](unsigned pose) { return static_cast<int>(pose); };
    auto localR = [&](unsigned pose, unsigned col) {
      return static_cast<int>(n + pose * d + col);
    };
    auto neighborT = [&](unsigned neighborPose) {
      return static_cast<int>((d + 1) * n + neighborPose);
    };
    auto neighborR = [&](unsigned neighborPose, unsigned col) {
      return static_cast<int>((d + 1) * n + neighborCount +
                                  neighborPose * d + col);
    };

    for (unsigned pose = 0; pose < n; ++pose) {
      z.row(localT(pose)) =
          base.col(static_cast<int>(pose * (d + 1) + d)).transpose();
      for (unsigned k = 0; k < d; ++k) {
        z.row(localR(pose, k)) =
            base.col(static_cast<int>(pose * (d + 1) + k)).transpose();
      }
    }
    for (unsigned idx = 0; idx < neighborCount; ++idx) {
      const auto poseIt = neighborPoseDict.find(dpgoSimpleNeighborOrder[idx]);
      if (poseIt == neighborPoseDict.end() ||
          poseIt->second.rows() != static_cast<int>(r) ||
          poseIt->second.cols() != static_cast<int>(d + 1) ||
          !poseIt->second.allFinite()) {
        return Matrix();
      }
      z.row(neighborT(idx)) =
          poseIt->second.col(static_cast<int>(d)).transpose();
      for (unsigned k = 0; k < d; ++k) {
        z.row(neighborR(idx, k)) =
            poseIt->second.col(static_cast<int>(k)).transpose();
      }
    }

    return z.allFinite() ? z : Matrix();
  }

  SparseMatrix makeDpgoSimpleLinearTermManual(const Matrix &base) const {
    if (options.profileOptimizer) {
      ++optimizerProfile.dpgoSimpleLinearCount;
    }
    SparseMatrix empty(r, n * (d + 1));
    if (base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite() ||
        !ensureDpgoSimpleProximalCache() ||
        dpgoSimpleS.rows() != static_cast<int>((d + 1) * n)) {
      return empty;
    }
    Matrix z;
    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer ? &optimizerProfile.dpgoSimpleLinearRowsSec
                                   : nullptr);
      z = makeDpgoOrderedStateRows(base);
    }
    if (z.rows() != dpgoSimpleS.cols() || z.cols() != static_cast<int>(r) ||
        !z.allFinite()) {
      return empty;
    }

    Matrix manualDense;
    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleLinearMultiplySec
              : nullptr);
      const Matrix linearRows = dpgoSimpleS * z;
      if (linearRows.rows() != static_cast<int>((d + 1) * n) ||
          linearRows.cols() != static_cast<int>(r) || !linearRows.allFinite()) {
        return empty;
      }
      manualDense =
          Matrix::Zero(static_cast<int>(r), static_cast<int>(n * (d + 1)));
      for (unsigned localRow = 0; localRow < (d + 1) * n; ++localRow) {
        const int manualCol = dpgoLocalRowToManualCol(localRow);
        if (manualCol < 0 || manualCol >= manualDense.cols()) {
          return empty;
        }
        manualDense.col(manualCol) =
            linearRows.row(static_cast<int>(localRow)).transpose();
      }
    }

    SparseMatrix manualG(r, n * (d + 1));
    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleLinearAssemblySec
              : nullptr);
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(
          static_cast<std::size_t>(manualDense.rows() * manualDense.cols()));
      for (int col = 0; col < manualDense.cols(); ++col) {
        for (unsigned row = 0; row < r; ++row) {
          const double value = manualDense(static_cast<int>(row), col);
          if (value != 0.0) {
            triplets.emplace_back(static_cast<int>(row), col, value);
          }
        }
      }

      manualG.setFromTriplets(triplets.begin(), triplets.end());
      manualG.makeCompressed();
    }
    return manualG;
  }

  double traceRowsQuadratic(const Matrix &rows,
                            const SparseMatrix &quadratic) const {
    if (rows.rows() != quadratic.rows() || rows.rows() != quadratic.cols() ||
        rows.cols() != static_cast<int>(r) || !rows.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const Matrix product = quadratic * rows;
    if (!product.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return rows.cwiseProduct(product).sum();
  }

  double traceManualLinear(const Matrix &value,
                           const SparseMatrix &linearTerm) const {
    if (value.rows() != static_cast<int>(r) ||
        value.cols() != static_cast<int>(n * (d + 1)) ||
        linearTerm.rows() != value.rows() || linearTerm.cols() != value.cols() ||
        !value.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double linear = 0.0;
    for (int outer = 0; outer < linearTerm.outerSize(); ++outer) {
      for (SparseMatrix::InnerIterator it(linearTerm, outer); it; ++it) {
        linear += value(it.row(), it.col()) * it.value();
      }
    }
    return linear;
  }

  double evaluateDpgoSimpleGValue(const Matrix &value,
                                  const SparseMatrix &linearTerm,
                                  double constant) const {
    if (!std::isfinite(constant) ||
        dpgoSimpleQManual.rows() != static_cast<int>(n * (d + 1)) ||
        dpgoSimpleQManual.cols() != static_cast<int>(n * (d + 1))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double linear = traceManualLinear(value, linearTerm);
    if (!std::isfinite(linear)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const Matrix valueQ = value * dpgoSimpleQManual;
    if (!valueQ.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double quadratic = value.cwiseProduct(valueQ).sum();
    return linear + 0.5 * quadratic + constant;
  }

  double dpgoSimpleFullGradientNorm(const Matrix &value,
                                    const SparseMatrix &linearTerm) const {
    if (value.rows() != static_cast<int>(r) ||
        value.cols() != static_cast<int>(n * (d + 1)) ||
        linearTerm.rows() != value.rows() ||
        linearTerm.cols() != value.cols() ||
        dpgoSimpleQManual.rows() != value.cols() ||
        dpgoSimpleQManual.cols() != value.cols() || !value.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    Matrix gradient = value * dpgoSimpleQManual + linearTerm;
    if (!gradient.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      const Matrix rotation =
          value.block(0, static_cast<int>(colStart), static_cast<int>(r),
                      static_cast<int>(d));
      const Matrix rotationGradient =
          gradient.block(0, static_cast<int>(colStart), static_cast<int>(r),
                         static_cast<int>(d));
      const Matrix normal =
          0.5 * (rotation.transpose() * rotationGradient +
                 rotationGradient.transpose() * rotation);
      gradient.block(0, static_cast<int>(colStart), static_cast<int>(r),
                     static_cast<int>(d)) = rotationGradient - rotation * normal;
    }

    return gradient.norm();
  }

  std::pair<double, double> edgeSplitThetaRange() const {
    double thetaMin = options.edgeSplitThetaMin;
    double thetaMax = options.edgeSplitThetaMax;
    if (!std::isfinite(thetaMin) || !std::isfinite(thetaMax) ||
        thetaMin <= 0.0 || thetaMax >= 1.0 || thetaMin >= thetaMax) {
      thetaMin = 0.15;
      thetaMax = 0.85;
    }
    return std::make_pair(thetaMin, thetaMax);
  }

  double constantEdgeSplitTheta() const {
    double theta = options.edgeSplitThetaDefault;
    const auto range = edgeSplitThetaRange();
    if (!std::isfinite(theta)) {
      theta = 0.5;
    }
    return std::min(range.second, std::max(range.first, theta));
  }

  std::pair<double, double>
  edgeSplitCurvatureForMeasurement(const RelativeSEMeasurement &m) const {
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1.0;

    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      const double value = m.weight * m.kappa;
      if (!std::isfinite(value) || value < 0.0) {
        return std::make_pair(0.0, 0.0);
      }
      Omega(static_cast<int>(row), static_cast<int>(row)) = value;
    }
    const double translationWeight = m.weight * m.tau;
    if (!std::isfinite(translationWeight) || translationWeight < 0.0) {
      return std::make_pair(0.0, 0.0);
    }
    Omega(static_cast<int>(d), static_cast<int>(d)) = translationWeight;

    const Matrix sourceBlock = T * Omega * T.transpose();
    const double sourceCurvature = sourceBlock.trace();
    const double targetCurvature = Omega.trace();
    return std::make_pair(
        std::isfinite(sourceCurvature) ? std::max(0.0, sourceCurvature) : 0.0,
        std::isfinite(targetCurvature) ? std::max(0.0, targetCurvature) : 0.0);
  }

  double edgeSplitThetaForMeasurement(const RelativeSEMeasurement &m) const {
    if (options.edgeSplitThetaMode ==
        ManualDpgoMmEdgeSplitThetaMode::Degree) {
      const auto range = edgeSplitThetaRange();
      const double alpha =
          m.r1 < robotMeasurementDegrees.size()
              ? robotMeasurementDegrees[static_cast<std::size_t>(m.r1)]
              : 0.0;
      const double beta =
          m.r2 < robotMeasurementDegrees.size()
              ? robotMeasurementDegrees[static_cast<std::size_t>(m.r2)]
              : 0.0;
      return evaluateManualDpgoMmDegreeEdgeSplitTheta(alpha, beta, range.first,
                                                      range.second);
    }
    if (options.edgeSplitThetaMode ==
        ManualDpgoMmEdgeSplitThetaMode::Curvature) {
      const auto range = edgeSplitThetaRange();
      const auto curvatures = edgeSplitCurvatureForMeasurement(m);
      return evaluateManualDpgoMmCurvatureEdgeSplitTheta(
          curvatures.first, curvatures.second, range.first, range.second);
    }
    if (options.edgeSplitThetaMode ==
        ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned) {
      const auto range = edgeSplitThetaRange();
      const double degreeAlpha =
          m.r1 < robotMeasurementDegrees.size()
              ? robotMeasurementDegrees[static_cast<std::size_t>(m.r1)]
              : 0.0;
      const double degreeBeta =
          m.r2 < robotMeasurementDegrees.size()
              ? robotMeasurementDegrees[static_cast<std::size_t>(m.r2)]
              : 0.0;
      const auto curvatures = edgeSplitCurvatureForMeasurement(m);
      return evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
          degreeAlpha, degreeBeta, curvatures.first, curvatures.second,
          options.edgeSplitThetaCandidates, range.first, range.second);
    }
    return constantEdgeSplitTheta();
  }

  bool edgeSplitThetaIsSymmetric() const {
    if (options.edgeSplitThetaMode ==
            ManualDpgoMmEdgeSplitThetaMode::Degree ||
        options.edgeSplitThetaMode ==
            ManualDpgoMmEdgeSplitThetaMode::Curvature ||
        options.edgeSplitThetaMode ==
            ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned) {
      for (const auto &m : sharedLoops) {
        if (std::abs(edgeSplitThetaForMeasurement(m) - 0.5) > 1e-14) {
          return false;
        }
      }
      return true;
    }
    return std::abs(constantEdgeSplitTheta() - 0.5) <= 1e-14;
  }

  void recordSurrogateBoundCheckMargin(double margin, bool ok) {
    ++lastSurrogateBoundCheckCount;
    if (!ok) {
      ++lastSurrogateBoundViolationCount;
    }
    lastSurrogateBoundMinMargin =
        std::min(lastSurrogateBoundMinMargin, margin);
  }

  void recordWeightedEdgeSplitBoundCheck(const Matrix &a, const Matrix &b,
                                         const Matrix &ak, const Matrix &bk,
                                         double theta) {
    double margin = -std::numeric_limits<double>::infinity();
    bool ok = false;
    try {
      const ManualDpgoMmWeightedEdgeSplitStats stats =
          evaluateManualDpgoMmWeightedEdgeSplit(a, b, ak, bk, theta);
      margin = stats.gap;
      ok = std::isfinite(stats.gap) && stats.gap >= -1e-10 &&
           std::isfinite(stats.surrogate) && std::isfinite(stats.objective);
    } catch (const std::exception &) {
      ok = false;
    }
    recordSurrogateBoundCheckMargin(margin, ok);
  }

  void recordScalarYoungBoundCheck(const Matrix &deltaAlpha,
                                   const Matrix &deltaBeta,
                                   const Matrix &crossBlock, double eta) {
    double margin = -std::numeric_limits<double>::infinity();
    bool ok = false;
    try {
      margin = evaluateManualDpgoMmScalarYoungMajorizerGap(
          deltaAlpha, deltaBeta, crossBlock, eta);
      ok = std::isfinite(margin) && margin >= -1e-10;
    } catch (const std::exception &) {
      ok = false;
    }
    recordSurrogateBoundCheckMargin(margin, ok);
  }

  void runWeightedEdgeSplitBoundCheck(const Matrix &base) {
    if (!options.debugSurrogateBoundCheck ||
        options.debugSurrogateBoundSamples == 0 || n == 0 || d == 0 ||
        base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite()) {
      return;
    }

    const std::size_t blockDim = d + 1;
    const unsigned sampleCount = options.debugSurrogateBoundSamples;
    auto perturbation = [&](unsigned pose, unsigned sample) {
      Matrix delta = Matrix::Zero(static_cast<int>(r),
                                  static_cast<int>(blockDim));
      const double scale = 1e-3 * static_cast<double>(sample + 1);
      for (unsigned row = 0; row < r; ++row) {
        for (std::size_t col = 0; col < blockDim; ++col) {
          const double phase =
              0.37 * static_cast<double>(sample + 1) +
              0.19 * static_cast<double>(pose + 1) +
              0.11 * static_cast<double>(row + 1) +
              0.07 * static_cast<double>(col + 1);
          delta(static_cast<int>(row), static_cast<int>(col)) =
              scale * std::sin(phase);
        }
      }
      return delta;
    };

    for (const auto &m : sharedLoops) {
      const double theta = edgeSplitThetaForMeasurement(m);
      if (!std::isfinite(theta) || theta <= 0.0 || theta >= 1.0) {
        continue;
      }

      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1.0;

      Matrix sqrtOmega = Matrix::Zero(d + 1, d + 1);
      bool omegaOk = true;
      for (unsigned row = 0; row < d; ++row) {
        const double value = m.weight * m.kappa;
        if (!std::isfinite(value) || value < 0.0) {
          omegaOk = false;
          break;
        }
        sqrtOmega(static_cast<int>(row), static_cast<int>(row)) =
            std::sqrt(value);
      }
      const double translationWeight = m.weight * m.tau;
      if (!std::isfinite(translationWeight) || translationWeight < 0.0) {
        omegaOk = false;
      }
      sqrtOmega(static_cast<int>(d), static_cast<int>(d)) =
          omegaOk ? std::sqrt(translationWeight) : 0.0;
      if (!omegaOk) {
        continue;
      }

      if (m.r1 == id) {
        if (m.p1 >= n) {
          continue;
        }
        const PoseID neighborPose = std::make_pair(m.r2, m.p2);
        const auto neighborIt = neighborPoseDict.find(neighborPose);
        if (neighborIt == neighborPoseDict.end() ||
            neighborIt->second.rows() != static_cast<int>(r) ||
            neighborIt->second.cols() != static_cast<int>(blockDim) ||
            !neighborIt->second.allFinite()) {
          continue;
        }
        const Matrix localAnchor =
            base.block(0, static_cast<int>(m.p1 * blockDim),
                       static_cast<int>(r), static_cast<int>(blockDim));
        const Matrix ak = localAnchor * T;
        const Matrix bk = neighborIt->second;
        for (unsigned sample = 0; sample < sampleCount; ++sample) {
          const Matrix localSample =
              localAnchor + perturbation(m.p1, sample);
          recordWeightedEdgeSplitBoundCheck(
              (localSample * T) * sqrtOmega, bk * sqrtOmega,
              ak * sqrtOmega, bk * sqrtOmega, theta);
        }
      } else if (m.r2 == id) {
        if (m.p2 >= n) {
          continue;
        }
        const PoseID neighborPose = std::make_pair(m.r1, m.p1);
        const auto neighborIt = neighborPoseDict.find(neighborPose);
        if (neighborIt == neighborPoseDict.end() ||
            neighborIt->second.rows() != static_cast<int>(r) ||
            neighborIt->second.cols() != static_cast<int>(blockDim) ||
            !neighborIt->second.allFinite()) {
          continue;
        }
        const Matrix ak = neighborIt->second * T;
        const Matrix bk =
            base.block(0, static_cast<int>(m.p2 * blockDim),
                       static_cast<int>(r), static_cast<int>(blockDim));
        for (unsigned sample = 0; sample < sampleCount; ++sample) {
          const Matrix localSample = bk + perturbation(m.p2, sample);
          recordWeightedEdgeSplitBoundCheck(
              ak * sqrtOmega, localSample * sqrtOmega, ak * sqrtOmega,
              bk * sqrtOmega, theta);
        }
      }
    }
  }

  double scalarYoungEtaForMeasurement(const RelativeSEMeasurement &m,
                                      double sigma) const {
    if (!std::isfinite(sigma) || sigma <= 0.0) {
      return 0.0;
    }
    const double theta = edgeSplitThetaForMeasurement(m);
    if (!std::isfinite(theta) || theta <= 0.0 || theta >= 1.0) {
      return sigma;
    }
    const double eta = sigma * (1.0 - theta) / theta;
    return std::isfinite(eta) ? std::max(1e-12, eta) : sigma;
  }

  void runAdaptiveSpectralBoundCheck(const Matrix &base) {
    if (!options.debugSurrogateBoundCheck ||
        options.debugSurrogateBoundSamples == 0 || n == 0 || d == 0 ||
        base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite()) {
      return;
    }

    const std::size_t blockDim = d + 1;
    const unsigned sampleCount = options.debugSurrogateBoundSamples;
    auto perturbation = [&](unsigned pose, unsigned sample) {
      Matrix delta = Matrix::Zero(static_cast<int>(r),
                                  static_cast<int>(blockDim));
      const double scale = 1e-3 * static_cast<double>(sample + 1);
      for (unsigned row = 0; row < r; ++row) {
        for (std::size_t col = 0; col < blockDim; ++col) {
          const double phase =
              0.29 * static_cast<double>(sample + 1) +
              0.23 * static_cast<double>(pose + 1) +
              0.17 * static_cast<double>(row + 1) +
              0.05 * static_cast<double>(col + 1);
          delta(static_cast<int>(row), static_cast<int>(col)) =
              scale * std::cos(phase);
        }
      }
      return delta;
    };

    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1.0;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      bool omegaOk = true;
      for (unsigned row = 0; row < d; ++row) {
        const double value = m.weight * m.kappa;
        if (!std::isfinite(value) || value < 0.0) {
          omegaOk = false;
          break;
        }
        Omega(static_cast<int>(row), static_cast<int>(row)) = value;
      }
      const double translationWeight = m.weight * m.tau;
      if (!std::isfinite(translationWeight) || translationWeight < 0.0) {
        omegaOk = false;
      }
      Omega(static_cast<int>(d), static_cast<int>(d)) =
          omegaOk ? translationWeight : 0.0;
      if (!omegaOk) {
        continue;
      }

      const Matrix crossBlock = T * Omega;
      const double sigma = denseSpectralNorm(crossBlock);
      const double eta = scalarYoungEtaForMeasurement(m, sigma);
      if (!std::isfinite(eta) || eta <= 0.0) {
        continue;
      }

      const unsigned alphaPose = m.r1 == id ? m.p1 : m.p1 + 17u;
      const unsigned betaPose = m.r2 == id ? m.p2 : m.p2 + 31u;
      for (unsigned sample = 0; sample < sampleCount; ++sample) {
        const Matrix deltaAlpha = perturbation(alphaPose, sample);
        const Matrix deltaBeta = perturbation(betaPose, sample);
        recordScalarYoungBoundCheck(deltaAlpha, deltaBeta, crossBlock, eta);
      }
    }
  }

  Matrix projectLocalRotationTangent(const Matrix &base,
                                     const Matrix &value) const {
    if (base.rows() != value.rows() || base.cols() != value.cols()) {
      return Matrix::Zero(value.rows(), value.cols());
    }
    Matrix projected = Matrix::Zero(value.rows(), value.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      const Matrix R = base.block(0, colStart, r, d);
      const Matrix ZR = value.block(0, colStart, r, d);
      Matrix sym = R.transpose() * ZR;
      sym = (0.5 * (sym + sym.transpose())).eval();
      projected.block(0, colStart, r, d) = ZR - R * sym;
    }
    return projected;
  }

  void runVariableProjectedSchurDiagnostic(const Matrix &base) {
    if (!options.debugSurrogateBoundCheck ||
        options.debugSurrogateBoundSamples == 0 || n == 0 || d == 0 ||
        base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite()) {
      return;
    }

    ReducedRotationQuadraticOptimizer inspector(&problem);
    inspector.setValidateTranslationRecoveryCost(false);
    const Matrix recovered = inspector.recoverTranslationsForRotations(base);
    if (recovered.rows() != base.rows() || recovered.cols() != base.cols() ||
        !recovered.allFinite()) {
      recordSurrogateBoundCheckMargin(-1.0, false);
      return;
    }

    const Matrix euclideanGradient =
        recovered * problem.getQRef() + problem.getGRef();
    const double translationGradientTolerance =
        1e-8 * std::max(1.0, euclideanGradient.norm());
    double translationGradientNorm = 0.0;
    for (unsigned pose = 0; pose < n; ++pose) {
      translationGradientNorm = std::max(
          translationGradientNorm,
          euclideanGradient.col(static_cast<int>(pose * (d + 1) + d)).norm());
    }
    recordSurrogateBoundCheckMargin(
        translationGradientTolerance - translationGradientNorm,
        translationGradientNorm <= translationGradientTolerance);

    const unsigned sampleCount = options.debugSurrogateBoundSamples;
    for (unsigned sample = 0; sample < sampleCount; ++sample) {
      Matrix raw = Matrix::Zero(base.rows(), base.cols());
      for (unsigned pose = 0; pose < n; ++pose) {
        const unsigned colStart = pose * (d + 1);
        for (unsigned localCol = 0; localCol < d; ++localCol) {
          const unsigned row = (pose + localCol + sample) % r;
          raw(static_cast<int>(row),
              static_cast<int>(colStart + localCol)) =
              1e-4 *
              (1.0 + static_cast<double>((sample + 1) * (localCol + 1)));
        }
      }
      Matrix eta = projectLocalRotationTangent(recovered, raw);
      const double etaNorm = eta.norm();
      if (!std::isfinite(etaNorm) || etaNorm <= 1e-14) {
        continue;
      }
      eta *= 1e-4 / etaNorm;
      const Matrix hv = inspector.applyReducedSchurHv(recovered, eta);
      double hvTranslationNorm = 0.0;
      for (unsigned pose = 0; pose < n; ++pose) {
        hvTranslationNorm = std::max(
            hvTranslationNorm,
            hv.col(static_cast<int>(pose * (d + 1) + d)).norm());
      }
      const bool ok = hv.rows() == eta.rows() && hv.cols() == eta.cols() &&
                      hv.allFinite() && hvTranslationNorm <= 1e-12;
      recordSurrogateBoundCheckMargin(1e-12 - hvTranslationNorm, ok);
    }
  }

  std::vector<Matrix> variableProjectedSchurBlockMajorizers() const {
    const SparseMatrix &Q = problem.getQRef();
    const int totalCols = static_cast<int>(n * (d + 1));
    const int rotationColsCount = static_cast<int>(n * d);
    if (Q.rows() != totalCols || Q.cols() != totalCols ||
        rotationColsCount <= 0 || n == 0) {
      return {};
    }

    std::vector<int> rotationIndexByCol(static_cast<std::size_t>(totalCols),
                                        -1);
    std::vector<int> translationIndexByCol(
        static_cast<std::size_t>(totalCols), -1);
    std::vector<int> rotationCols;
    std::vector<int> translationCols;
    rotationCols.reserve(static_cast<std::size_t>(rotationColsCount));
    translationCols.reserve(n);
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int col = static_cast<int>(colStart + localCol);
        rotationIndexByCol[static_cast<std::size_t>(col)] =
            static_cast<int>(rotationCols.size());
        rotationCols.push_back(col);
      }
      const int translationCol = static_cast<int>(colStart + d);
      translationIndexByCol[static_cast<std::size_t>(translationCol)] =
          static_cast<int>(translationCols.size());
      translationCols.push_back(translationCol);
    }

    Matrix qRR = Matrix::Zero(rotationColsCount, rotationColsCount);
    Matrix qRT = Matrix::Zero(rotationColsCount, static_cast<int>(n));
    Matrix qTR = Matrix::Zero(static_cast<int>(n), rotationColsCount);
    std::vector<Eigen::Triplet<double>> qTTTriplets;
    qTTTriplets.reserve(Q.nonZeros() + n);
    double translationDiagScale = 0.0;
    unsigned translationDiagCount = 0;

    for (int outer = 0; outer < Q.outerSize(); ++outer) {
      for (SparseMatrix::InnerIterator it(Q, outer); it; ++it) {
        const int row = static_cast<int>(it.row());
        const int col = static_cast<int>(it.col());
        const int rRow = rotationIndexByCol[static_cast<std::size_t>(row)];
        const int rCol = rotationIndexByCol[static_cast<std::size_t>(col)];
        const int tRow = translationIndexByCol[static_cast<std::size_t>(row)];
        const int tCol = translationIndexByCol[static_cast<std::size_t>(col)];
        if (rRow >= 0 && rCol >= 0) {
          qRR(rRow, rCol) += it.value();
        } else if (rRow >= 0 && tCol >= 0) {
          qRT(rRow, tCol) += it.value();
        } else if (tRow >= 0 && rCol >= 0) {
          qTR(tRow, rCol) += it.value();
        } else if (tRow >= 0 && tCol >= 0) {
          qTTTriplets.emplace_back(tRow, tCol, it.value());
          if (tRow == tCol) {
            translationDiagScale += std::abs(it.value());
            ++translationDiagCount;
          }
        }
      }
    }

    if (translationDiagCount > 0) {
      translationDiagScale /= static_cast<double>(translationDiagCount);
    }
    if (!std::isfinite(translationDiagScale) || translationDiagScale <= 0.0) {
      translationDiagScale = 1.0;
    }

    Matrix solvedTR;
    bool solved = false;
    for (unsigned attempt = 0; attempt < 6 && !solved; ++attempt) {
      std::vector<Eigen::Triplet<double>> regularized = qTTTriplets;
      const double ridge =
          (attempt == 0 ? 0.0
                        : std::pow(10.0, static_cast<int>(attempt) - 12) *
                              std::max(1.0, translationDiagScale));
      if (ridge > 0.0) {
        for (unsigned pose = 0; pose < n; ++pose) {
          regularized.emplace_back(static_cast<int>(pose),
                                   static_cast<int>(pose), ridge);
        }
      }
      ColMajorSparseMatrix qTT(static_cast<int>(n), static_cast<int>(n));
      qTT.setFromTriplets(regularized.begin(), regularized.end());
      qTT.makeCompressed();
      Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
      solver.compute(qTT);
      if (solver.info() != Eigen::Success) {
        continue;
      }
      solvedTR = solver.solve(qTR);
      solved = solver.info() == Eigen::Success &&
               solvedTR.rows() == static_cast<int>(n) &&
               solvedTR.cols() == rotationColsCount && solvedTR.allFinite();
    }

    if (!solved) {
      return {};
    }

    Matrix schur = qRR - qRT * solvedTR;
    schur = (0.5 * (schur + schur.transpose())).eval();
    if (!schur.allFinite()) {
      return {};
    }

    std::vector<Matrix> blockMajorizers(n);
    for (unsigned pose = 0; pose < n; ++pose) {
      const int blockStart = static_cast<int>(pose * d);
      Matrix block = schur.block(blockStart, blockStart, static_cast<int>(d),
                                 static_cast<int>(d));
      block = (0.5 * (block + block.transpose())).eval();
      double offBlockNormSum = 0.0;
      for (unsigned other = 0; other < n; ++other) {
        if (other == pose) {
          continue;
        }
        const int otherStart = static_cast<int>(other * d);
        offBlockNormSum += denseSpectralNorm(
            schur.block(blockStart, otherStart, static_cast<int>(d),
                        static_cast<int>(d)));
      }
      if (!std::isfinite(offBlockNormSum) || offBlockNormSum < 0.0) {
        return {};
      }
      block.diagonal().array() += offBlockNormSum;

      double diagScale = block.diagonal().cwiseAbs().mean();
      if (!std::isfinite(diagScale) || diagScale <= 0.0) {
        diagScale = 1.0;
      }
      bool madeSpd = false;
      for (unsigned attempt = 0; attempt < 6; ++attempt) {
        Matrix regularized = block;
        const double ridge =
            (1e-10 * std::max(1.0, diagScale)) *
            std::pow(10.0, static_cast<int>(attempt));
        regularized.diagonal().array() += ridge;
        Eigen::LDLT<Matrix> solver(regularized);
        if (solver.info() == Eigen::Success && solver.isPositive() &&
            regularized.allFinite()) {
          blockMajorizers[pose] =
              (0.5 * (regularized + regularized.transpose())).eval();
          madeSpd = true;
          break;
        }
      }
      if (!madeSpd) {
        double fallbackBound = 1e-8;
        for (unsigned localCol = 0; localCol < d; ++localCol) {
          const int row = static_cast<int>(pose * d + localCol);
          double rowAbsSum = 0.0;
          for (int col = 0; col < schur.cols(); ++col) {
            rowAbsSum += std::abs(schur(row, col));
          }
          fallbackBound = std::max(fallbackBound, rowAbsSum);
        }
        blockMajorizers[pose] =
            std::max(1e-8, 1.05 * fallbackBound) *
            Matrix::Identity(static_cast<int>(d), static_cast<int>(d));
      }
    }
    return blockMajorizers;
  }

  Matrix makeVariableProjectedSchurMajorizedCandidate(const Matrix &base) {
    ++lastVariableProjectedSchurCandidateCount;
    if (base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite()) {
      return Matrix();
    }

    const std::vector<Matrix> blockMajorizers =
        variableProjectedSchurBlockMajorizers();
    if (blockMajorizers.size() != n) {
      return Matrix();
    }

    ReducedRotationQuadraticOptimizer inspector(&problem);
    inspector.setValidateTranslationRecoveryCost(false);
    const Matrix anchor = inspector.recoverTranslationsForRotations(base);
    if (anchor.rows() != base.rows() || anchor.cols() != base.cols() ||
        !anchor.allFinite()) {
      return Matrix();
    }
    const Matrix reducedGradient = inspector.buildReducedGradient(anchor);
    if (reducedGradient.rows() != anchor.rows() ||
        reducedGradient.cols() != anchor.cols() ||
        !reducedGradient.allFinite()) {
      return Matrix();
    }

    Matrix step = Matrix::Zero(anchor.rows(), anchor.cols());
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      const Matrix &majorizer = blockMajorizers[pose];
      if (majorizer.rows() != static_cast<int>(d) ||
          majorizer.cols() != static_cast<int>(d) ||
          !majorizer.allFinite()) {
        return Matrix();
      }
      Eigen::LDLT<Matrix> solver(majorizer);
      if (solver.info() != Eigen::Success || !solver.isPositive()) {
        return Matrix();
      }
      const Matrix gradientBlock =
          reducedGradient.block(0, colStart, r, d);
      const Matrix solved =
          solver.solve(gradientBlock.transpose()).transpose();
      if (solved.rows() != gradientBlock.rows() ||
          solved.cols() != gradientBlock.cols() || !solved.allFinite()) {
        return Matrix();
      }
      step.block(0, colStart, r, d) =
          -solved;
    }
    step = projectLocalRotationTangent(anchor, step);
    const double stepNorm = step.norm();
    if (!std::isfinite(stepNorm) || stepNorm <= 1e-14) {
      return Matrix();
    }
    const double maxNorm = std::max(1e-6, options.trustRegionInitialRadius);
    if (stepNorm > maxNorm) {
      step *= maxNorm / stepNorm;
    }

    const double baseCost = inspector.evaluateReducedObjective(anchor);
    if (!std::isfinite(baseCost)) {
      return Matrix();
    }

    for (unsigned attempt = 0; attempt < 8; ++attempt) {
      const double alpha = std::pow(0.5, static_cast<int>(attempt));
      Matrix trial = anchor;
      for (unsigned pose = 0; pose < n; ++pose) {
        const unsigned colStart = pose * (d + 1);
        trial.block(0, colStart, r, d) =
            projectToStiefelManifold(
                anchor.block(0, colStart, r, d) +
                alpha * step.block(0, colStart, r, d));
      }
      trial = inspector.recoverTranslationsForRotations(trial);
      if (trial.rows() != anchor.rows() || trial.cols() != anchor.cols() ||
          !trial.allFinite()) {
        continue;
      }
      const double trialCost = problem.f(trial);
      if (std::isfinite(trialCost) && trialCost < baseCost - 1e-12) {
        ++lastVariableProjectedSchurAcceptedCount;
        return trial;
      }
    }
    return Matrix();
  }

  WeightedEdgeSplitModel makeWeightedEdgeSplitModel(const Matrix &base) {
    WeightedEdgeSplitModel model;
    model.theta = constantEdgeSplitTheta();
    if (n == 0 || d == 0 || base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite() ||
        model.theta <= 0.0 || model.theta >= 1.0) {
      return model;
    }

    std::vector<RelativeSEMeasurement> privateMeasurements = odometry;
    privateMeasurements.insert(privateMeasurements.end(), privateLoops.begin(),
                               privateLoops.end());

    const std::size_t blockDim = d + 1;
    const std::size_t totalDim = n * blockDim;
    SparseMatrix q = padSparseMatrix(
        constructConnectionLaplacianSE(privateMeasurements), totalDim,
        totalDim);
    SparseMatrix g(static_cast<int>(r), static_cast<int>(totalDim));

    auto addDenseBlockToQ = [&](unsigned pose, const Matrix &block,
                                double scale) {
      if (pose >= n || block.rows() != static_cast<int>(blockDim) ||
          block.cols() != static_cast<int>(blockDim) ||
          !std::isfinite(scale)) {
        return false;
      }
      for (std::size_t col = 0; col < blockDim; ++col) {
        for (std::size_t row = 0; row < blockDim; ++row) {
          const double value = scale * block(static_cast<int>(row),
                                             static_cast<int>(col));
          if (value != 0.0) {
            q.coeffRef(pose * blockDim + row, pose * blockDim + col) +=
                value;
          }
        }
      }
      return true;
    };

    auto addDenseBlockToG = [&](unsigned pose, const Matrix &block) {
      if (pose >= n || block.rows() != static_cast<int>(r) ||
          block.cols() != static_cast<int>(blockDim)) {
        return false;
      }
      for (std::size_t col = 0; col < blockDim; ++col) {
        for (unsigned row = 0; row < r; ++row) {
          const double value = block(static_cast<int>(row),
                                     static_cast<int>(col));
          if (value != 0.0) {
            g.coeffRef(static_cast<int>(row),
                       static_cast<int>(pose * blockDim + col)) += value;
          }
        }
      }
      return true;
    };

    for (const auto &m : sharedLoops) {
      const double theta = edgeSplitThetaForMeasurement(m);
      if (!std::isfinite(theta) || theta <= 0.0 || theta >= 1.0) {
        return model;
      }

      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1.0;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(static_cast<int>(row), static_cast<int>(row)) =
            m.weight * m.kappa;
      }
      Omega(static_cast<int>(d), static_cast<int>(d)) = m.weight * m.tau;

      if (m.r1 == id) {
        if (m.p1 >= n) {
          return model;
        }
        const PoseID neighborPose = std::make_pair(m.r2, m.p2);
        const auto neighborIt = neighborPoseDict.find(neighborPose);
        if (neighborIt == neighborPoseDict.end() ||
            neighborIt->second.rows() != static_cast<int>(r) ||
            neighborIt->second.cols() != static_cast<int>(blockDim) ||
            !neighborIt->second.allFinite()) {
          return model;
        }
        const Matrix localAnchor =
            base.block(0, static_cast<int>(m.p1 * blockDim),
                       static_cast<int>(r), static_cast<int>(blockDim)) *
            T;
        const Matrix splitAnchor =
            (1.0 - theta) * localAnchor + theta * neighborIt->second;
        const Matrix qBlock = T * Omega * T.transpose();
        const Matrix gBlock =
            -(1.0 / theta) * splitAnchor * Omega * T.transpose();
        if (!addDenseBlockToQ(m.p1, qBlock, 1.0 / theta) ||
            !addDenseBlockToG(m.p1, gBlock)) {
          return model;
        }
      } else if (m.r2 == id) {
        if (m.p2 >= n) {
          return model;
        }
        const PoseID neighborPose = std::make_pair(m.r1, m.p1);
        const auto neighborIt = neighborPoseDict.find(neighborPose);
        if (neighborIt == neighborPoseDict.end() ||
            neighborIt->second.rows() != static_cast<int>(r) ||
            neighborIt->second.cols() != static_cast<int>(blockDim) ||
            !neighborIt->second.allFinite()) {
          return model;
        }
        const Matrix sourceAnchor = neighborIt->second * T;
        const Matrix localAnchor =
            base.block(0, static_cast<int>(m.p2 * blockDim),
                       static_cast<int>(r), static_cast<int>(blockDim));
        const Matrix splitAnchor =
            (1.0 - theta) * sourceAnchor + theta * localAnchor;
        const Matrix gBlock =
            -(1.0 / (1.0 - theta)) * splitAnchor * Omega;
        if (!addDenseBlockToQ(m.p2, Omega, 1.0 / (1.0 - theta)) ||
            !addDenseBlockToG(m.p2, gBlock)) {
          return model;
        }
      }
    }

    q.makeCompressed();
    g.makeCompressed();
    model.q = q;
    model.g = g;
    model.valid =
        model.q.rows() == static_cast<int>(totalDim) &&
        model.q.cols() == static_cast<int>(totalDim) &&
        model.g.rows() == static_cast<int>(r) &&
        model.g.cols() == static_cast<int>(totalDim);
    if (model.valid) {
      runWeightedEdgeSplitBoundCheck(base);
    }
    return model;
  }

  WeightedEdgeSplitModel makeAdaptiveSpectralModel(const Matrix &base) {
    WeightedEdgeSplitModel model;
    model.theta = constantEdgeSplitTheta();
    if (n == 0 || d == 0 || base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite()) {
      return model;
    }

    std::vector<RelativeSEMeasurement> privateMeasurements = odometry;
    privateMeasurements.insert(privateMeasurements.end(), privateLoops.begin(),
                               privateLoops.end());

    const std::size_t blockDim = d + 1;
    const std::size_t totalDim = n * blockDim;
    SparseMatrix q = padSparseMatrix(
        constructConnectionLaplacianSE(privateMeasurements), totalDim,
        totalDim);
    SparseMatrix g(static_cast<int>(r), static_cast<int>(totalDim));

    auto addDenseBlockToQ = [&](unsigned pose, const Matrix &block) {
      if (pose >= n || block.rows() != static_cast<int>(blockDim) ||
          block.cols() != static_cast<int>(blockDim) || !block.allFinite()) {
        return false;
      }
      for (std::size_t col = 0; col < blockDim; ++col) {
        for (std::size_t row = 0; row < blockDim; ++row) {
          const double value =
              block(static_cast<int>(row), static_cast<int>(col));
          if (value != 0.0) {
            q.coeffRef(pose * blockDim + row, pose * blockDim + col) +=
                value;
          }
        }
      }
      return true;
    };

    auto addDenseBlockToG = [&](unsigned pose, const Matrix &block) {
      if (pose >= n || block.rows() != static_cast<int>(r) ||
          block.cols() != static_cast<int>(blockDim) || !block.allFinite()) {
        return false;
      }
      for (std::size_t col = 0; col < blockDim; ++col) {
        for (unsigned row = 0; row < r; ++row) {
          const double value =
              block(static_cast<int>(row), static_cast<int>(col));
          if (value != 0.0) {
            g.coeffRef(static_cast<int>(row),
                       static_cast<int>(pose * blockDim + col)) += value;
          }
        }
      }
      return true;
    };

    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1.0;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      bool omegaOk = true;
      for (unsigned row = 0; row < d; ++row) {
        const double value = m.weight * m.kappa;
        if (!std::isfinite(value) || value < 0.0) {
          omegaOk = false;
          break;
        }
        Omega(static_cast<int>(row), static_cast<int>(row)) = value;
      }
      const double translationWeight = m.weight * m.tau;
      if (!std::isfinite(translationWeight) || translationWeight < 0.0) {
        omegaOk = false;
      }
      Omega(static_cast<int>(d), static_cast<int>(d)) =
          omegaOk ? translationWeight : 0.0;
      if (!omegaOk) {
        return model;
      }

      const Matrix crossBlock = T * Omega;
      const double sigma = denseSpectralNorm(crossBlock);
      const double eta = scalarYoungEtaForMeasurement(m, sigma);
      const double alphaDamping =
          sigma > 0.0 ? std::max(0.0, eta) : 0.0;
      const double betaDamping =
          sigma > 0.0 && eta > 0.0
              ? (sigma * sigma) / eta
              : 0.0;
      if (!std::isfinite(alphaDamping) || !std::isfinite(betaDamping)) {
        return model;
      }

      if (m.r1 == id) {
        if (m.p1 >= n) {
          return model;
        }
        const PoseID neighborPose = std::make_pair(m.r2, m.p2);
        const auto neighborIt = neighborPoseDict.find(neighborPose);
        if (neighborIt == neighborPoseDict.end() ||
            neighborIt->second.rows() != static_cast<int>(r) ||
            neighborIt->second.cols() != static_cast<int>(blockDim) ||
            !neighborIt->second.allFinite()) {
          return model;
        }
        const Matrix localAnchor =
            base.block(0, static_cast<int>(m.p1 * blockDim),
                       static_cast<int>(r), static_cast<int>(blockDim));
        const Matrix residualAnchor = localAnchor * T - neighborIt->second;
        const Matrix exactHessian = T * Omega * T.transpose();
        double topologyDampingScale = 1.0;
        if (options.communicationTopologyBoundarySurrogate) {
          topologyDampingScale +=
              options.communicationTopologyBoundarySurrogateStaleGain *
              static_cast<double>(neighborPoseStaleness(neighborPose));
        }
        if (!std::isfinite(topologyDampingScale) ||
            topologyDampingScale < 1.0) {
          topologyDampingScale = 1.0;
        }
        const Matrix qBlock =
            exactHessian +
            topologyDampingScale * alphaDamping *
                Matrix::Identity(static_cast<int>(blockDim),
                                 static_cast<int>(blockDim));
        const Matrix gradBlock = residualAnchor * Omega * T.transpose();
        const Matrix gBlock = gradBlock - localAnchor * qBlock;
        if (!addDenseBlockToQ(m.p1, qBlock) ||
            !addDenseBlockToG(m.p1, gBlock)) {
          return model;
        }
      } else if (m.r2 == id) {
        if (m.p2 >= n) {
          return model;
        }
        const PoseID neighborPose = std::make_pair(m.r1, m.p1);
        const auto neighborIt = neighborPoseDict.find(neighborPose);
        if (neighborIt == neighborPoseDict.end() ||
            neighborIt->second.rows() != static_cast<int>(r) ||
            neighborIt->second.cols() != static_cast<int>(blockDim) ||
            !neighborIt->second.allFinite()) {
          return model;
        }
        const Matrix sourceAnchor = neighborIt->second * T;
        const Matrix localAnchor =
            base.block(0, static_cast<int>(m.p2 * blockDim),
                       static_cast<int>(r), static_cast<int>(blockDim));
        double topologyDampingScale = 1.0;
        if (options.communicationTopologyBoundarySurrogate) {
          topologyDampingScale +=
              options.communicationTopologyBoundarySurrogateStaleGain *
              static_cast<double>(neighborPoseStaleness(neighborPose));
        }
        if (!std::isfinite(topologyDampingScale) ||
            topologyDampingScale < 1.0) {
          topologyDampingScale = 1.0;
        }
        const Matrix qBlock =
            Omega +
            topologyDampingScale * betaDamping *
                Matrix::Identity(static_cast<int>(blockDim),
                                 static_cast<int>(blockDim));
        const Matrix gradBlock = (localAnchor - sourceAnchor) * Omega;
        const Matrix gBlock = gradBlock - localAnchor * qBlock;
        if (!addDenseBlockToQ(m.p2, qBlock) ||
            !addDenseBlockToG(m.p2, gBlock)) {
          return model;
        }
      }
    }

    q.makeCompressed();
    g.makeCompressed();
    model.q = q;
    model.g = g;
    model.valid =
        model.q.rows() == static_cast<int>(totalDim) &&
        model.q.cols() == static_cast<int>(totalDim) &&
        model.g.rows() == static_cast<int>(r) &&
        model.g.cols() == static_cast<int>(totalDim);
    if (model.valid) {
      runAdaptiveSpectralBoundCheck(base);
    }
    return model;
  }

  struct RecursiveSimpleEvaluation {
    bool valid{false};
    Matrix zRows;
    double f{0.0};
    double fobj{0.0};
    double deltaQDelta{0.0};
    double pTerm{0.0};
    double previousGk{0.0};
  };

  RecursiveSimpleEvaluation
  evaluateRecursiveSimpleState(const Matrix &base,
                               const SparseMatrix &linearTerm) const {
    RecursiveSimpleEvaluation evaluation;
    if (linearTerm.rows() != static_cast<int>(r) ||
        linearTerm.cols() != static_cast<int>(n * (d + 1)) ||
        !ensureDpgoSimpleProximalCache()) {
      return evaluation;
    }
    const Matrix z = makeDpgoOrderedStateRows(base);
    const unsigned totalRows =
        static_cast<unsigned>((d + 1) *
                              (n + dpgoSimpleNeighborOrder.size()));
    if (z.rows() != static_cast<int>(totalRows) ||
        z.cols() != static_cast<int>(r) || !z.allFinite() ||
        dpgoSimpleQFull.rows() != z.rows() || dpgoSimpleP.rows() != z.rows() ||
        dpgoSimpleP0.rows() != z.rows()) {
      return evaluation;
    }

    if (!recursiveSimpleInitialized) {
      const double f0 = 0.5 * traceRowsQuadratic(z, dpgoSimpleP0);
      const double fobj = evaluateDpgoSimpleGValue(base, linearTerm, f0);
      if (!std::isfinite(f0) || !std::isfinite(fobj)) {
        return evaluation;
      }
      evaluation.f = f0;
      evaluation.fobj = fobj;
      evaluation.pTerm = traceRowsQuadratic(z, dpgoSimpleP0);
      evaluation.previousGk = fobj;
    } else {
      if (recursiveSimplePreviousZRows.rows() != z.rows() ||
          recursiveSimplePreviousZRows.cols() != z.cols() ||
          !recursiveSimplePreviousZRows.allFinite() ||
          !std::isfinite(recursiveSimplePreviousGk)) {
        return evaluation;
      }
      const Matrix delta = z - recursiveSimplePreviousZRows;
      const double deltaQDelta = traceRowsQuadratic(delta, dpgoSimpleQFull);
      const double pTerm = traceRowsQuadratic(z, dpgoSimpleP);
      if (!std::isfinite(deltaQDelta) || !std::isfinite(pTerm)) {
        return evaluation;
      }
      evaluation.deltaQDelta = deltaQDelta;
      evaluation.pTerm = pTerm;
      evaluation.previousGk = recursiveSimplePreviousGk;
      evaluation.fobj = recursiveSimplePreviousGk + 0.5 * deltaQDelta;
      evaluation.f = evaluation.fobj + 0.5 * pTerm;
    }

    if (!std::isfinite(evaluation.f) || !std::isfinite(evaluation.fobj)) {
      return evaluation;
    }
    evaluation.zRows = z;
    evaluation.valid = true;
    return evaluation;
  }

  bool ensureDpgoSimpleProximalCache() const {
    if (dpgoSimpleProximalCacheReady) {
      return dpgoSimpleProximalCacheValid;
    }
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer ? &optimizerProfile.dpgoSimpleCacheBuildSec
                                 : nullptr);
    if (options.profileOptimizer) {
      ++optimizerProfile.dpgoSimpleCacheBuildCount;
    }
    dpgoSimpleProximalCacheReady = true;
    dpgoSimpleProximalCacheValid = false;
    dpgoSimpleInvDiagT.resize(0);
    dpgoSimpleScaleN.resize(0, 0);
    dpgoSimpleScaleNSparse.resize(0, 0);
    dpgoSimpleU.resize(0, 0);
    dpgoSimpleUSparse.resize(0, 0);
    dpgoSimpleQManual.resize(0, 0);
    dpgoSimpleS.resize(0, 0);
    dpgoSimpleQFull.resize(0, 0);
    dpgoSimpleP.resize(0, 0);
    dpgoSimpleP0.resize(0, 0);
    dpgoSimpleNeighborOrder.clear();

    if (n == 0 || d == 0) {
      return false;
    }

    for (const auto &entry : neighborPublicPoses) {
      for (const unsigned pose : entry.second) {
        dpgoSimpleNeighborOrder.emplace_back(entry.first, pose);
      }
    }
    std::map<PoseID, unsigned> neighborIndex;
    for (unsigned idx = 0;
         idx < static_cast<unsigned>(dpgoSimpleNeighborOrder.size()); ++idx) {
      neighborIndex[dpgoSimpleNeighborOrder[idx]] = idx;
    }

    const unsigned neighborCount =
        static_cast<unsigned>(dpgoSimpleNeighborOrder.size());
    const unsigned rotationDim = n * d;
    const unsigned totalCols = (d + 1) * (n + neighborCount);
    Vector diagT = Vector::Zero(static_cast<int>(n));
    Matrix nmat =
        Matrix::Zero(static_cast<int>(n), static_cast<int>(rotationDim));
    std::vector<Eigen::Triplet<double>> tripletsG;
    std::vector<Eigen::Triplet<double>> tripletsS;
    std::vector<Eigen::Triplet<double>> tripletsQ;
    std::vector<Eigen::Triplet<double>> tripletsP;
    std::vector<Eigen::Triplet<double>> tripletsP0;
    std::vector<Eigen::Triplet<double>> tripletsVTop;
    std::vector<Eigen::Triplet<double>> tripletsVBottom;

    auto localT = [&](unsigned pose) { return static_cast<int>(pose); };
    auto localR = [&](unsigned pose, unsigned col) {
      return static_cast<int>(n + pose * d + col);
    };
    auto neighborT = [&](unsigned neighborPose) {
      return static_cast<int>((d + 1) * n + neighborPose);
    };
    auto neighborR = [&](unsigned neighborPose, unsigned col) {
      return static_cast<int>((d + 1) * n + neighborCount +
                              neighborPose * d + col);
    };
    auto addV = [&](int row, int col, double value) {
      if (row >= 0 && row < static_cast<int>(n + rotationDim) && col >= 0 &&
          col < static_cast<int>(totalCols) && value != 0.0) {
        if (row < static_cast<int>(n)) {
          tripletsVTop.emplace_back(row, col, value);
        } else {
          tripletsVBottom.emplace_back(row - static_cast<int>(n), col, value);
        }
      }
    };
    auto addG = [&](int row, int col, double value) {
      if (row >= 0 && row < static_cast<int>((d + 1) * n) && col >= 0 &&
          col < static_cast<int>((d + 1) * n) && value != 0.0) {
        tripletsG.emplace_back(row, col, value);
      }
    };
    auto addS = [&](int row, int col, double value) {
      if (row >= 0 && row < static_cast<int>((d + 1) * n) && col >= 0 &&
          col < static_cast<int>(totalCols) && value != 0.0) {
        tripletsS.emplace_back(row, col, value);
      }
    };
    auto addFull = [&](std::vector<Eigen::Triplet<double>> &triplets, int row,
                       int col, double value) {
      if (row >= 0 && row < static_cast<int>(totalCols) && col >= 0 &&
          col < static_cast<int>(totalCols) && value != 0.0) {
        triplets.emplace_back(row, col, value);
      }
    };
    auto addQ = [&](int row, int col, double value) {
      addFull(tripletsQ, row, col, value);
    };
    auto addP = [&](int row, int col, double value) {
      addFull(tripletsP, row, col, value);
    };
    auto addP0 = [&](int row, int col, double value) {
      addFull(tripletsP0, row, col, value);
    };

    auto addIntraMeasurement = [&](const RelativeSEMeasurement &m) {
      if (m.p1 >= n || m.p2 >= n) {
        return;
      }
      const double tau = m.weight * m.tau;
      const double kappa = m.weight * m.kappa;
      const unsigned i = static_cast<unsigned>(m.p1);
      const unsigned j = static_cast<unsigned>(m.p2);
      const int ti = localT(i);
      const int tj = localT(j);
      addG(ti, ti, tau);
      addG(tj, tj, tau);
      addG(ti, tj, -tau);
      addG(tj, ti, -tau);
      addP(ti, ti, -tau);
      addP(tj, tj, -tau);
      addP(ti, tj, tau);
      addP(tj, ti, tau);

      diagT(ti) += 2.0 * tau;
      diagT(tj) += 2.0 * tau;

      addV(ti, ti, -tau);
      addV(tj, tj, -tau);
      addV(ti, tj, -tau);
      addV(tj, ti, -tau);

      for (unsigned k = 0; k < d; ++k) {
        const double t = m.t(static_cast<int>(k));
        addG(ti, localR(i, k), tau * t);
        addG(tj, localR(i, k), -tau * t);
        addG(localR(i, k), ti, tau * t);
        addG(localR(i, k), tj, -tau * t);
        addG(localR(i, k), localR(i, k), kappa);
        addG(localR(j, k), localR(j, k), kappa);
        addP(ti, localR(i, k), -tau * t);
        addP(tj, localR(i, k), tau * t);
        addP(localR(i, k), ti, -tau * t);
        addP(localR(i, k), tj, tau * t);
        addP(localR(i, k), localR(i, k), -kappa);
        addP(localR(j, k), localR(j, k), -kappa);

        nmat(ti, static_cast<int>(i * d + k)) += 2.0 * tau * t;
        addV(ti, localR(i, k), -tau * t);
        addV(tj, localR(i, k), -tau * t);
        addV(localR(i, k), ti, -tau * t);
        addV(localR(i, k), tj, -tau * t);
        addV(localR(i, k), localR(i, k), -kappa);
        addV(localR(j, k), localR(j, k), -kappa);
      }

      for (unsigned row = 0; row < d; ++row) {
        for (unsigned col = 0; col < d; ++col) {
          addG(localR(i, row), localR(j, col),
               -kappa * m.R(static_cast<int>(row),
                            static_cast<int>(col)));
          addG(localR(j, row), localR(i, col),
               -kappa * m.R(static_cast<int>(col),
                            static_cast<int>(row)));
          addG(localR(i, row), localR(i, col),
               tau * m.t(static_cast<int>(row)) *
                   m.t(static_cast<int>(col)));
          addP(localR(i, row), localR(j, col),
               kappa * m.R(static_cast<int>(row),
                           static_cast<int>(col)));
          addP(localR(j, row), localR(i, col),
               kappa * m.R(static_cast<int>(col),
                           static_cast<int>(row)));
          addP(localR(i, row), localR(i, col),
               -tau * m.t(static_cast<int>(row)) *
                   m.t(static_cast<int>(col)));

          addV(localR(i, row), localR(j, col),
               -kappa * m.R(static_cast<int>(row),
                            static_cast<int>(col)));
          addV(localR(j, row), localR(i, col),
               -kappa * m.R(static_cast<int>(col),
                            static_cast<int>(row)));
          addV(localR(i, row), localR(i, col),
               -tau * m.t(static_cast<int>(row)) *
                   m.t(static_cast<int>(col)));
        }
      }
    };

    auto addInterMeasurement = [&](const RelativeSEMeasurement &m) {
      const double tau = m.weight * m.tau;
      const double kappa = m.weight * m.kappa;
      auto addInterRecursiveTerms = [&](int sourceT, int targetT,
                                        const auto &sourceR,
                                        const auto &targetR) {
        addQ(sourceT, sourceT, -0.5 * tau);
        addQ(targetT, targetT, -0.5 * tau);
        addQ(sourceT, targetT, -0.5 * tau);
        addQ(targetT, sourceT, -0.5 * tau);
        addP(sourceT, targetT, tau);
        addP(targetT, sourceT, tau);
        addP0(sourceT, sourceT, 0.5 * tau);
        addP0(targetT, targetT, 0.5 * tau);
        addP0(sourceT, targetT, 0.5 * tau);
        addP0(targetT, sourceT, 0.5 * tau);

        for (unsigned k = 0; k < d; ++k) {
          const double t = m.t(static_cast<int>(k));
          const int sourceRk = sourceR(k);
          const int targetRk = targetR(k);
          addQ(sourceT, sourceRk, -0.5 * tau * t);
          addQ(targetT, sourceRk, -0.5 * tau * t);
          addP(targetT, sourceRk, tau * t);
          addP0(sourceT, sourceRk, 0.5 * tau * t);
          addP0(targetT, sourceRk, 0.5 * tau * t);

          addQ(sourceRk, sourceT, -0.5 * tau * t);
          addQ(sourceRk, targetT, -0.5 * tau * t);
          addP(sourceRk, targetT, tau * t);
          addP0(sourceRk, sourceT, 0.5 * tau * t);
          addP0(sourceRk, targetT, 0.5 * tau * t);

          addQ(sourceRk, sourceRk, -0.5 * kappa);
          addQ(targetRk, targetRk, -0.5 * kappa);
          addP0(sourceRk, sourceRk, 0.5 * kappa);
          addP0(targetRk, targetRk, 0.5 * kappa);
        }

        for (unsigned row = 0; row < d; ++row) {
          for (unsigned col = 0; col < d; ++col) {
            const int sourceRRow = sourceR(row);
            const int sourceRCol = sourceR(col);
            const int targetRRow = targetR(row);
            const int targetRCol = targetR(col);
            const double rotationForward =
                kappa * m.R(static_cast<int>(row), static_cast<int>(col));
            const double rotationBackward =
                kappa * m.R(static_cast<int>(col), static_cast<int>(row));
            addQ(sourceRRow, targetRCol, -0.5 * rotationForward);
            addP(sourceRRow, targetRCol, rotationForward);
            addP0(sourceRRow, targetRCol, 0.5 * rotationForward);
            addQ(targetRRow, sourceRCol, -0.5 * rotationBackward);
            addP(targetRRow, sourceRCol, rotationBackward);
            addP0(targetRRow, sourceRCol, 0.5 * rotationBackward);

            const double translationCurvature =
                tau * m.t(static_cast<int>(row)) *
                m.t(static_cast<int>(col));
            addQ(sourceRRow, sourceRCol, -0.5 * translationCurvature);
            addP0(sourceRRow, sourceRCol, 0.5 * translationCurvature);
          }
        }
      };
      if (m.r1 == id) {
        if (m.p1 >= n) {
          return;
        }
        const auto neighborIt =
            neighborIndex.find(std::make_pair(m.r2, m.p2));
        if (neighborIt == neighborIndex.end()) {
          return;
        }
        const unsigned i = static_cast<unsigned>(m.p1);
        const unsigned q = neighborIt->second;
        const int ti = localT(i);
        const int tq = neighborT(q);
        addInterRecursiveTerms(
            ti, tq, [&](unsigned k) { return localR(i, k); },
            [&](unsigned k) { return neighborR(q, k); });
        addG(ti, ti, 2.0 * tau);
        addS(ti, ti, -tau);
        addS(ti, tq, -tau);

        diagT(ti) += 2.0 * tau;
        addV(ti, ti, -tau);
        addV(ti, tq, -tau);
        for (unsigned k = 0; k < d; ++k) {
          const double t = m.t(static_cast<int>(k));
          addG(ti, localR(i, k), 2.0 * tau * t);
          addG(localR(i, k), ti, 2.0 * tau * t);
          addG(localR(i, k), localR(i, k), 2.0 * kappa);
          addS(ti, localR(i, k), -tau * t);
          addS(localR(i, k), ti, -tau * t);
          addS(localR(i, k), tq, -tau * t);
          addS(localR(i, k), localR(i, k), -kappa);

          nmat(ti, static_cast<int>(i * d + k)) += 2.0 * tau * t;
          addV(ti, localR(i, k), -tau * t);
          addV(localR(i, k), ti, -tau * t);
          addV(localR(i, k), tq, -tau * t);
          addV(localR(i, k), localR(i, k), -kappa);
        }
        for (unsigned row = 0; row < d; ++row) {
          for (unsigned col = 0; col < d; ++col) {
            addG(localR(i, row), localR(i, col),
                 2.0 * tau * m.t(static_cast<int>(row)) *
                     m.t(static_cast<int>(col)));
            addS(localR(i, row), localR(i, col),
                 -tau * m.t(static_cast<int>(row)) *
                     m.t(static_cast<int>(col)));
            addS(localR(i, row), neighborR(q, col),
                 -kappa * m.R(static_cast<int>(row),
                              static_cast<int>(col)));

            addV(localR(i, row), localR(i, col),
                 -tau * m.t(static_cast<int>(row)) *
                     m.t(static_cast<int>(col)));
            addV(localR(i, row), neighborR(q, col),
                 -kappa * m.R(static_cast<int>(row),
                              static_cast<int>(col)));
          }
        }
      } else if (m.r2 == id) {
        if (m.p2 >= n) {
          return;
        }
        const auto neighborIt =
            neighborIndex.find(std::make_pair(m.r1, m.p1));
        if (neighborIt == neighborIndex.end()) {
          return;
        }
        const unsigned j = static_cast<unsigned>(m.p2);
        const unsigned q = neighborIt->second;
        const int tj = localT(j);
        const int tq = neighborT(q);
        addInterRecursiveTerms(
            tq, tj, [&](unsigned k) { return neighborR(q, k); },
            [&](unsigned k) { return localR(j, k); });
        addG(tj, tj, 2.0 * tau);
        addS(tj, tj, -tau);
        addS(tj, tq, -tau);

        diagT(tj) += 2.0 * tau;
        addV(tj, tj, -tau);
        addV(tj, tq, -tau);
        for (unsigned k = 0; k < d; ++k) {
          addG(localR(j, k), localR(j, k), 2.0 * kappa);
          addS(tj, neighborR(q, k), -tau * m.t(static_cast<int>(k)));
          addS(localR(j, k), localR(j, k), -kappa);

          addV(tj, neighborR(q, k), -tau * m.t(static_cast<int>(k)));
          addV(localR(j, k), localR(j, k), -kappa);
        }
        for (unsigned row = 0; row < d; ++row) {
          for (unsigned col = 0; col < d; ++col) {
            addS(localR(j, row), neighborR(q, col),
                 -kappa * m.R(static_cast<int>(col),
                              static_cast<int>(row)));

            addV(localR(j, row), neighborR(q, col),
                 -kappa * m.R(static_cast<int>(col),
                              static_cast<int>(row)));
          }
        }
      }
    };

    for (const auto &m : odometry) {
      addIntraMeasurement(m);
    }
    for (const auto &m : privateLoops) {
      addIntraMeasurement(m);
    }
    for (const auto &m : sharedLoops) {
      addInterMeasurement(m);
    }

    const double regularizer = 1e-10;
    for (unsigned pose = 0; pose < n; ++pose) {
      addG(localT(pose), localT(pose), regularizer);
      addS(localT(pose), localT(pose), -regularizer);
      addQ(localT(pose), localT(pose), -regularizer);
      addP(localT(pose), localT(pose), regularizer);
      addP0(localT(pose), localT(pose), regularizer);
      diagT(static_cast<int>(pose)) += 1.5 * regularizer;
      addV(localT(pose), localT(pose), -1.5 * regularizer);
      for (unsigned k = 0; k < d; ++k) {
        addG(localR(pose, k), localR(pose, k), regularizer);
        addS(localR(pose, k), localR(pose, k), -regularizer);
        addQ(localR(pose, k), localR(pose, k), -regularizer);
        addP(localR(pose, k), localR(pose, k), regularizer);
        addP0(localR(pose, k), localR(pose, k), regularizer);
        addV(localR(pose, k), localR(pose, k), -1.5 * regularizer);
      }
    }

    dpgoSimpleInvDiagT = Vector::Zero(static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      const double value = diagT(static_cast<int>(pose));
      if (!std::isfinite(value) || value <= 1e-14) {
        return false;
      }
      dpgoSimpleInvDiagT(static_cast<int>(pose)) = 1.0 / value;
    }

    dpgoSimpleScaleN = nmat;
    for (unsigned pose = 0; pose < n; ++pose) {
      dpgoSimpleScaleN.row(static_cast<int>(pose)) *=
          dpgoSimpleInvDiagT(static_cast<int>(pose));
    }
    ColMajorSparseMatrix vTop(static_cast<int>(n),
                              static_cast<int>(totalCols));
    vTop.setFromTriplets(tripletsVTop.begin(), tripletsVTop.end());
    vTop.makeCompressed();
    ColMajorSparseMatrix vBottom(static_cast<int>(rotationDim),
                                 static_cast<int>(totalCols));
    vBottom.setFromTriplets(tripletsVBottom.begin(), tripletsVBottom.end());
    vBottom.makeCompressed();

    std::vector<Eigen::Triplet<double>> scaleNTriplets;
    for (int row = 0; row < dpgoSimpleScaleN.rows(); ++row) {
      for (int col = 0; col < dpgoSimpleScaleN.cols(); ++col) {
        const double value = dpgoSimpleScaleN(row, col);
        if (value != 0.0) {
          scaleNTriplets.emplace_back(row, col, value);
        }
      }
    }
    ColMajorSparseMatrix scaleNSparse(static_cast<int>(n),
                                      static_cast<int>(rotationDim));
    scaleNSparse.setFromTriplets(scaleNTriplets.begin(),
                                 scaleNTriplets.end());
    scaleNSparse.makeCompressed();
    dpgoSimpleScaleNSparse = scaleNSparse;
    dpgoSimpleUSparse = scaleNSparse.transpose() * vTop;
    dpgoSimpleUSparse = dpgoSimpleUSparse - vBottom;
    dpgoSimpleUSparse.makeCompressed();

    SparseMatrix gDpgo(static_cast<int>((d + 1) * n),
                       static_cast<int>((d + 1) * n));
    gDpgo.setFromTriplets(tripletsG.begin(), tripletsG.end());
    gDpgo.makeCompressed();

    std::vector<Eigen::Triplet<double>> manualQTriplets;
    manualQTriplets.reserve(static_cast<std::size_t>(gDpgo.nonZeros()));
    for (int outer = 0; outer < gDpgo.outerSize(); ++outer) {
      for (SparseMatrix::InnerIterator it(gDpgo, outer); it; ++it) {
        const int manualRow =
            dpgoLocalRowToManualCol(static_cast<unsigned>(it.row()));
        const int manualCol =
            dpgoLocalRowToManualCol(static_cast<unsigned>(it.col()));
        manualQTriplets.emplace_back(manualRow, manualCol, it.value());
      }
    }
    dpgoSimpleQManual.resize(static_cast<int>(n * (d + 1)),
                             static_cast<int>(n * (d + 1)));
    dpgoSimpleQManual.setFromTriplets(manualQTriplets.begin(),
                                      manualQTriplets.end());
    dpgoSimpleQManual.makeCompressed();

    dpgoSimpleS.resize(static_cast<int>((d + 1) * n),
                       static_cast<int>(totalCols));
    dpgoSimpleS.setFromTriplets(tripletsS.begin(), tripletsS.end());
    dpgoSimpleS.makeCompressed();

    dpgoSimpleQFull.resize(static_cast<int>(totalCols),
                           static_cast<int>(totalCols));
    dpgoSimpleQFull.setFromTriplets(tripletsQ.begin(), tripletsQ.end());
    dpgoSimpleQFull.makeCompressed();
    dpgoSimpleP.resize(static_cast<int>(totalCols),
                       static_cast<int>(totalCols));
    dpgoSimpleP.setFromTriplets(tripletsP.begin(), tripletsP.end());
    dpgoSimpleP.makeCompressed();
    dpgoSimpleP0.resize(static_cast<int>(totalCols),
                        static_cast<int>(totalCols));
    dpgoSimpleP0.setFromTriplets(tripletsP0.begin(), tripletsP0.end());
    dpgoSimpleP0.makeCompressed();

    dpgoSimpleProximalCacheValid =
        dpgoSimpleInvDiagT.allFinite() && dpgoSimpleScaleN.allFinite() &&
        dpgoSimpleScaleNSparse.rows() == static_cast<int>(n) &&
        dpgoSimpleScaleNSparse.cols() == static_cast<int>(rotationDim) &&
        sparseMatrixAllFinite(dpgoSimpleScaleNSparse) &&
        dpgoSimpleUSparse.rows() == static_cast<int>(rotationDim) &&
        dpgoSimpleUSparse.cols() == static_cast<int>(totalCols) &&
        sparseMatrixAllFinite(dpgoSimpleUSparse) &&
        dpgoSimpleQManual.rows() == static_cast<int>(n * (d + 1)) &&
        dpgoSimpleS.rows() == static_cast<int>((d + 1) * n) &&
        dpgoSimpleQFull.rows() == static_cast<int>(totalCols) &&
        dpgoSimpleP.rows() == static_cast<int>(totalCols) &&
        dpgoSimpleP0.rows() == static_cast<int>(totalCols);
    return dpgoSimpleProximalCacheValid;
  }

  Matrix makeDpgoSimpleProximalStart(const Matrix &base) const {
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer ? &optimizerProfile.dpgoSimpleProximalStartSec
                                 : nullptr);
    if (options.profileOptimizer) {
      ++optimizerProfile.dpgoSimpleProximalStartCount;
    }
    if (n == 0 || base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite() ||
        !ensureDpgoSimpleProximalCache()) {
      return Matrix();
    }

    const unsigned neighborCount =
        static_cast<unsigned>(dpgoSimpleNeighborOrder.size());
    const unsigned totalCols = (d + 1) * (n + neighborCount);
    if (dpgoSimpleScaleNSparse.rows() != static_cast<int>(n) ||
        dpgoSimpleScaleNSparse.cols() != static_cast<int>(n * d) ||
        dpgoSimpleUSparse.rows() != static_cast<int>(n * d) ||
        dpgoSimpleUSparse.cols() != static_cast<int>(totalCols)) {
      return Matrix();
    }

    auto localT = [&](unsigned pose) { return static_cast<int>(pose); };
    auto localR = [&](unsigned pose, unsigned col) {
      return static_cast<int>(n + pose * d + col);
    };
    auto neighborT = [&](unsigned neighborPose) {
      return static_cast<int>((d + 1) * n + neighborPose);
    };
    auto neighborR = [&](unsigned neighborPose, unsigned col) {
      return static_cast<int>((d + 1) * n + neighborCount +
                              neighborPose * d + col);
    };

    Matrix z;
    {
      ScopedOptionalSecondsAccumulator profileRowsTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleProximalRowsSec
              : nullptr);
      z = Matrix::Zero(static_cast<int>(r), static_cast<int>(totalCols));
      for (unsigned pose = 0; pose < n; ++pose) {
        z.col(localT(pose)) =
            base.col(static_cast<int>(pose * (d + 1) + d));
        for (unsigned k = 0; k < d; ++k) {
          z.col(localR(pose, k)) =
              base.col(static_cast<int>(pose * (d + 1) + k));
        }
      }
      for (unsigned idx = 0; idx < neighborCount; ++idx) {
        const auto poseIt = neighborPoseDict.find(dpgoSimpleNeighborOrder[idx]);
        if (poseIt == neighborPoseDict.end() ||
            poseIt->second.rows() != static_cast<int>(r) ||
            poseIt->second.cols() != static_cast<int>(d + 1) ||
            !poseIt->second.allFinite()) {
          return Matrix();
        }
        z.col(neighborT(idx)) = poseIt->second.col(static_cast<int>(d));
        for (unsigned k = 0; k < d; ++k) {
          z.col(neighborR(idx, k)) =
              poseIt->second.col(static_cast<int>(k));
        }
      }
    }

    Matrix rotationSignal;
    {
      ScopedOptionalSecondsAccumulator profileRotationTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleProximalRotationSignalSec
              : nullptr);
      rotationSignal = z * dpgoSimpleUSparse.transpose();
    }
    if (!rotationSignal.allFinite()) {
      return Matrix();
    }

    Matrix rotationBase =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n * d));
    Matrix rotationCandidate =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n * d));
    Matrix translationBase =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
    Matrix gradient;
    {
      ScopedOptionalSecondsAccumulator profileGradientTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleProximalGradientSec
              : nullptr);
      gradient = base * problem.getQRef() + problem.getGRef();
    }
    Matrix gradientTranslation =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      translationBase.col(static_cast<int>(pose)) =
          base.col(static_cast<int>(pose * (d + 1) + d));
      gradientTranslation.col(static_cast<int>(pose)) =
          gradient.col(static_cast<int>(pose * (d + 1) + d));
      for (unsigned k = 0; k < d; ++k) {
        rotationBase.col(static_cast<int>(pose * d + k)) =
            base.col(static_cast<int>(pose * (d + 1) + k));
      }
    }

    {
      ScopedOptionalSecondsAccumulator profileProjectionTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleProximalProjectionSec
              : nullptr);
      for (unsigned pose = 0; pose < n; ++pose) {
        const int colStart = static_cast<int>(pose * d);
        rotationCandidate.block(0, colStart, static_cast<int>(r),
                                static_cast<int>(d)) =
            projectToStiefelManifold(rotationSignal.block(
                0, colStart, static_cast<int>(r), static_cast<int>(d)));
      }
    }

    Matrix translationCandidate;
    {
      ScopedOptionalSecondsAccumulator profileTranslationTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleProximalTranslationSec
              : nullptr);
      translationCandidate =
          translationBase - (rotationCandidate - rotationBase) *
                                dpgoSimpleScaleNSparse.transpose();
      for (unsigned pose = 0; pose < n; ++pose) {
        translationCandidate.col(static_cast<int>(pose)).noalias() -=
            dpgoSimpleInvDiagT(static_cast<int>(pose)) *
            gradientTranslation.col(static_cast<int>(pose));
      }
    }

    Matrix candidate;
    {
      ScopedOptionalSecondsAccumulator profileAssemblyTimer(
          options.profileOptimizer
              ? &optimizerProfile.dpgoSimpleProximalAssemblySec
              : nullptr);
      candidate = base;
      for (unsigned pose = 0; pose < n; ++pose) {
        candidate.col(static_cast<int>(pose * (d + 1) + d)) =
            translationCandidate.col(static_cast<int>(pose));
        for (unsigned k = 0; k < d; ++k) {
          candidate.col(static_cast<int>(pose * (d + 1) + k)) =
              rotationCandidate.col(static_cast<int>(pose * d + k));
        }
      }
    }

    return candidate.allFinite() ? candidate : Matrix();
  }

  Matrix makeDpgoMajorizedProximalStart(const Matrix &base,
                                        double baseCost) const {
    if (n == 0 || base.rows() != static_cast<int>(r) ||
        base.cols() != static_cast<int>(n * (d + 1)) || !base.allFinite() ||
        !std::isfinite(baseCost)) {
      return Matrix();
    }

    const std::size_t rotationDim = n * d;
    const double regularizer = 1e-10;
    Vector diagT = Vector::Constant(static_cast<int>(n), 0.5 * regularizer);
    Matrix Nmat =
        Matrix::Zero(static_cast<int>(n), static_cast<int>(rotationDim));
    Matrix Vmat = 0.5 * regularizer *
                  Matrix::Identity(static_cast<int>(rotationDim),
                                   static_cast<int>(rotationDim));

    auto addTailMajorizer = [&](unsigned pose,
                                const RelativeSEMeasurement &m) {
      if (pose >= n) {
        return;
      }
      const double tau = m.weight * m.tau;
      const double kappa = m.weight * m.kappa;
      const int row = static_cast<int>(pose);
      const int colStart = static_cast<int>(pose * d);
      diagT(row) += 2.0 * tau;
      for (unsigned k = 0; k < d; ++k) {
        Nmat(row, colStart + static_cast<int>(k)) += 2.0 * tau * m.t(k);
      }
      Vmat.block(colStart, colStart, static_cast<int>(d),
                 static_cast<int>(d))
          .noalias() +=
          2.0 * kappa * Matrix::Identity(static_cast<int>(d),
                                         static_cast<int>(d)) +
          2.0 * tau * (m.t * m.t.transpose());
    };

    auto addHeadMajorizer = [&](unsigned pose,
                                const RelativeSEMeasurement &m) {
      if (pose >= n) {
        return;
      }
      const double tau = m.weight * m.tau;
      const double kappa = m.weight * m.kappa;
      const int row = static_cast<int>(pose);
      const int colStart = static_cast<int>(pose * d);
      diagT(row) += 2.0 * tau;
      Vmat.block(colStart, colStart, static_cast<int>(d),
                 static_cast<int>(d))
          .noalias() +=
          2.0 * kappa * Matrix::Identity(static_cast<int>(d),
                                         static_cast<int>(d));
    };

    for (const auto &m : odometry) {
      addTailMajorizer(m.p1, m);
      addHeadMajorizer(m.p2, m);
    }
    for (const auto &m : privateLoops) {
      addTailMajorizer(m.p1, m);
      addHeadMajorizer(m.p2, m);
    }
    for (const auto &m : sharedLoops) {
      if (m.r1 == id) {
        addTailMajorizer(m.p1, m);
      } else if (m.r2 == id) {
        addHeadMajorizer(m.p2, m);
      }
    }

    Vector invDiagT = Vector::Zero(static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      const double value = diagT(static_cast<int>(pose));
      if (!std::isfinite(value) || value <= 1e-14) {
        return Matrix();
      }
      invDiagT(static_cast<int>(pose)) = 1.0 / value;
    }
    Matrix scaleN = Nmat;
    for (unsigned pose = 0; pose < n; ++pose) {
      scaleN.row(static_cast<int>(pose)) *=
          invDiagT(static_cast<int>(pose));
    }
    Vmat.noalias() -= Nmat.transpose() * scaleN;
    Vmat = 0.5 * (Vmat + Vmat.transpose());

    const SparseMatrix &Q = problem.getQRef();
    const Matrix gradient = base * Q + problem.getGRef();
    Matrix rotationBase =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(rotationDim));
    Matrix translationBase =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
    Matrix gradientRotation =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(rotationDim));
    Matrix gradientTranslation =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      const int translationCol = static_cast<int>(pose * (d + 1) + d);
      translationBase.col(static_cast<int>(pose)) =
          base.col(translationCol);
      gradientTranslation.col(static_cast<int>(pose)) =
          gradient.col(translationCol);
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int packedCol = static_cast<int>(pose * d + localCol);
        const int originalCol =
            static_cast<int>(pose * (d + 1) + localCol);
        rotationBase.col(packedCol) = base.col(originalCol);
        gradientRotation.col(packedCol) = gradient.col(originalCol);
      }
    }

    const Matrix rotationSignal =
        rotationBase * Vmat - gradientRotation + gradientTranslation * scaleN;
    if (!rotationSignal.allFinite()) {
      return Matrix();
    }

    Matrix rotationCandidate = rotationBase;
    for (unsigned pose = 0; pose < n; ++pose) {
      const int colStart = static_cast<int>(pose * d);
      rotationCandidate.block(0, colStart, static_cast<int>(r),
                              static_cast<int>(d)) =
          projectToStiefelManifold(rotationSignal.block(
              0, colStart, static_cast<int>(r), static_cast<int>(d)));
    }

    Matrix translationStep =
        Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
    for (unsigned pose = 0; pose < n; ++pose) {
      translationStep.col(static_cast<int>(pose)) =
          invDiagT(static_cast<int>(pose)) *
          gradientTranslation.col(static_cast<int>(pose));
    }
    const Matrix translationCandidate =
        translationBase - (rotationCandidate - rotationBase) *
                              scaleN.transpose() -
        translationStep;

    Matrix candidate = base;
    for (unsigned pose = 0; pose < n; ++pose) {
      const int translationCol = static_cast<int>(pose * (d + 1) + d);
      candidate.col(translationCol) =
          translationCandidate.col(static_cast<int>(pose));
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int packedCol = static_cast<int>(pose * d + localCol);
        const int originalCol =
            static_cast<int>(pose * (d + 1) + localCol);
        candidate.col(originalCol) = rotationCandidate.col(packedCol);
      }
    }

    if (!candidate.allFinite()) {
      return Matrix();
    }
    const double candidateCost = problem.f(candidate);
    if (std::isfinite(candidateCost) && candidateCost <= baseCost + 1e-10) {
      return candidate;
    }
    return Matrix();
  }

  Matrix makeDpgoMajorizedProximalStartUnfiltered(const Matrix &base) const {
    const Matrix simpleStart = makeDpgoSimpleProximalStart(base);
    if (simpleStart.rows() == base.rows() && simpleStart.cols() == base.cols() &&
        simpleStart.allFinite()) {
      return simpleStart;
    }
    const double permissiveCost =
        0.25 * std::numeric_limits<double>::max();
    return makeDpgoMajorizedProximalStart(base, permissiveCost);
  }

  Matrix makeMajorizedProximalStart(const Matrix &start) const {
    if (n == 0 || start.rows() != static_cast<int>(r) ||
        start.cols() != static_cast<int>(n * (d + 1)) ||
        !start.allFinite()) {
      return start;
    }

    const SparseMatrix &Q = problem.getQRef();
    Matrix base = recoverMajorizedTranslations(start);
    const double baseCost = problem.f(base);
    if (!std::isfinite(baseCost)) {
      return start;
    }

    Matrix bestStart = base;
    double bestCost = baseCost;
    auto considerCandidate = [&](const Matrix &candidate) {
      if (candidate.rows() != base.rows() || candidate.cols() != base.cols() ||
          !candidate.allFinite()) {
        return false;
      }
      const double candidateCost = problem.f(candidate);
      if (!std::isfinite(candidateCost) || candidateCost > bestCost + 1e-10) {
        return false;
      }
      bestStart = candidate;
      bestCost = candidateCost;
      return true;
    };

    const Matrix dpgoProximalStart =
        makeDpgoMajorizedProximalStart(base, baseCost);
    considerCandidate(dpgoProximalStart);

    const std::size_t blockDim = d + 1;
    const std::size_t rotationDim = n * d;
    std::vector<int> translationCols;
    std::vector<int> rotationCols;
    translationCols.reserve(n);
    rotationCols.reserve(rotationDim);
    std::vector<char> isTranslation(Q.cols(), 0);
    std::vector<int> translationIndex(Q.cols(), -1);
    std::vector<int> rotationIndex(Q.cols(), -1);
    for (unsigned pose = 0; pose < n; ++pose) {
      const int translationCol = static_cast<int>(pose * blockDim + d);
      translationIndex[translationCol] = static_cast<int>(pose);
      isTranslation[translationCol] = 1;
      translationCols.push_back(translationCol);
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        const int rotationCol =
            static_cast<int>(pose * blockDim + localCol);
        rotationIndex[rotationCol] =
            static_cast<int>(rotationCols.size());
        rotationCols.push_back(rotationCol);
      }
    }

    std::vector<Eigen::Triplet<double>> translationTriplets;
    translationTriplets.reserve(Q.nonZeros() + n);
    Matrix translationRotation =
        Matrix::Zero(static_cast<int>(n), static_cast<int>(rotationDim));
    double diagScale = 0.0;
    unsigned diagCount = 0;
    for (int outer = 0; outer < Q.outerSize(); ++outer) {
      for (SparseMatrix::InnerIterator it(Q, outer); it; ++it) {
        const int row = static_cast<int>(it.row());
        const int col = static_cast<int>(it.col());
        if (isTranslation[row] && isTranslation[col]) {
          const int translationRow = translationIndex[row];
          const int translationCol = translationIndex[col];
          translationTriplets.emplace_back(translationRow, translationCol,
                                           it.value());
          if (translationRow == translationCol) {
            diagScale += std::abs(it.value());
            ++diagCount;
          }
        } else if (isTranslation[row] && rotationIndex[col] >= 0) {
          translationRotation(translationIndex[row], rotationIndex[col]) +=
              it.value();
        }
      }
    }
    if (diagCount > 0) {
      diagScale /= static_cast<double>(diagCount);
    }
    if (!std::isfinite(diagScale) || diagScale <= 0.0) {
      diagScale = 1.0;
    }

    Matrix schurCandidate;
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
      std::vector<Eigen::Triplet<double>> regularizedTriplets =
          translationTriplets;
      const double ridge =
          (attempt == 0)
              ? 0.0
              : 1e-10 * std::max(1.0, diagScale) *
                    std::pow(10.0, static_cast<double>(attempt - 1));
      if (ridge > 0.0) {
        for (unsigned pose = 0; pose < n; ++pose) {
          regularizedTriplets.emplace_back(static_cast<int>(pose),
                                           static_cast<int>(pose), ridge);
        }
      }

      ColMajorSparseMatrix translationSystem(static_cast<int>(n),
                                             static_cast<int>(n));
      translationSystem.setFromTriplets(regularizedTriplets.begin(),
                                        regularizedTriplets.end());
      translationSystem.makeCompressed();

      Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
      solver.compute(translationSystem);
      if (solver.info() != Eigen::Success) {
        continue;
      }

      const Matrix scaleN = solver.solve(translationRotation);
      if (solver.info() != Eigen::Success ||
          scaleN.rows() != static_cast<int>(n) ||
          scaleN.cols() != static_cast<int>(rotationDim) ||
          !scaleN.allFinite()) {
        continue;
      }

      const Matrix gradient = base * Q + problem.getGRef();
      Matrix rotationBase =
          Matrix::Zero(static_cast<int>(r), static_cast<int>(rotationDim));
      Matrix translationBase =
          Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
      Matrix gradientRotation =
          Matrix::Zero(static_cast<int>(r), static_cast<int>(rotationDim));
      Matrix gradientTranslation =
          Matrix::Zero(static_cast<int>(r), static_cast<int>(n));
      for (unsigned pose = 0; pose < n; ++pose) {
        translationBase.col(static_cast<int>(pose)) =
            base.col(translationCols[pose]);
        gradientTranslation.col(static_cast<int>(pose)) =
            gradient.col(translationCols[pose]);
        for (unsigned localCol = 0; localCol < d; ++localCol) {
          const int packedCol = static_cast<int>(pose * d + localCol);
          const int originalCol =
              static_cast<int>(pose * blockDim + localCol);
          rotationBase.col(packedCol) = base.col(originalCol);
          gradientRotation.col(packedCol) = gradient.col(originalCol);
        }
      }

      const Matrix reducedGradient =
          gradientRotation - gradientTranslation * scaleN;
      Matrix rotationCandidate = rotationBase;
      bool validRotationCandidate = true;
      for (unsigned pose = 0; pose < n; ++pose) {
        Matrix hessianBlock = Matrix::Zero(d, d);
        for (unsigned row = 0; row < d; ++row) {
          const int originalRow =
              static_cast<int>(pose * blockDim + row);
          for (unsigned col = 0; col < d; ++col) {
            const int originalCol =
                static_cast<int>(pose * blockDim + col);
            const int packedCol = static_cast<int>(pose * d + col);
            double value = Q.coeff(originalRow, originalCol);
            for (unsigned translation = 0; translation < n; ++translation) {
              value -= Q.coeff(originalRow, translationCols[translation]) *
                       scaleN(static_cast<int>(translation), packedCol);
            }
            hessianBlock(static_cast<int>(row), static_cast<int>(col)) =
                value;
          }
        }
        hessianBlock = 0.5 * (hessianBlock + hessianBlock.transpose());
        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(hessianBlock);
        if (eigensolver.info() != Eigen::Success ||
            !eigensolver.eigenvalues().allFinite()) {
          double fallbackScale = 0.0;
          for (unsigned row = 0; row < d; ++row) {
            for (unsigned col = 0; col < d; ++col) {
              fallbackScale += std::abs(
                  hessianBlock(static_cast<int>(row), static_cast<int>(col)));
            }
          }
          fallbackScale = std::max(1.0, fallbackScale);
          hessianBlock = fallbackScale * Matrix::Identity(d, d);
        } else {
          const double minEigenvalue = eigensolver.eigenvalues().minCoeff();
          if (minEigenvalue < 1e-9) {
            hessianBlock.diagonal().array() += 1e-9 - minEigenvalue;
          }
        }

        const int colStart = static_cast<int>(pose * d);
        const Matrix blockBase = rotationBase.block(
            0, colStart, static_cast<int>(r), static_cast<int>(d));
        const Matrix blockGradient = reducedGradient.block(
            0, colStart, static_cast<int>(r), static_cast<int>(d));
        const Matrix proximalBlock = blockBase * hessianBlock - blockGradient;
        if (!proximalBlock.allFinite()) {
          validRotationCandidate = false;
          break;
        }
        rotationCandidate.block(
            0, colStart, static_cast<int>(r), static_cast<int>(d)) =
            projectToStiefelManifold(proximalBlock);
      }
      if (!validRotationCandidate) {
        continue;
      }

      const Matrix translationStep =
          solver.solve(gradientTranslation.transpose()).transpose();
      if (solver.info() != Eigen::Success ||
          translationStep.rows() != static_cast<int>(r) ||
          translationStep.cols() != static_cast<int>(n) ||
          !translationStep.allFinite()) {
        continue;
      }
      const Matrix translationCandidate =
          translationBase - (rotationCandidate - rotationBase) *
                                scaleN.transpose() -
          translationStep;

      schurCandidate = base;
      for (unsigned pose = 0; pose < n; ++pose) {
        schurCandidate.col(translationCols[pose]) =
            translationCandidate.col(static_cast<int>(pose));
        for (unsigned localCol = 0; localCol < d; ++localCol) {
          const int packedCol = static_cast<int>(pose * d + localCol);
          const int originalCol =
              static_cast<int>(pose * blockDim + localCol);
          schurCandidate.col(originalCol) =
              rotationCandidate.col(packedCol);
        }
      }

      if (!schurCandidate.allFinite()) {
        schurCandidate.resize(0, 0);
        continue;
      }
      const double schurCost = problem.f(schurCandidate);
      if (std::isfinite(schurCost)) {
        if (schurCost <= bestCost + 1e-10) {
          bestStart = schurCandidate;
          bestCost = schurCost;
        }
      }
      break;
    }

    std::vector<double> rowAbsSums(static_cast<std::size_t>(Q.rows()), 0.0);
    for (int k = 0; k < Q.outerSize(); ++k) {
      for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
        rowAbsSums[static_cast<std::size_t>(it.row())] +=
            std::abs(it.value());
      }
    }

    Matrix candidate = base;
    const Matrix grad = base * Q + problem.getGRef();
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      double lipschitz = 1e-8;
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        lipschitz = std::max(
            lipschitz,
            rowAbsSums[static_cast<std::size_t>(colStart + localCol)]);
      }
      const Matrix rotationBlock =
          base.block(0, colStart, r, d) -
          grad.block(0, colStart, r, d) / lipschitz;
      candidate.block(0, colStart, r, d) =
          projectToStiefelManifold(rotationBlock);
    }
    candidate = recoverMajorizedTranslations(candidate);

    const double candidateCost = problem.f(candidate);
    if (std::isfinite(candidateCost) && candidateCost <= bestCost + 1e-10) {
      bestStart = candidate;
      bestCost = candidateCost;
    }
    return bestStart;
  }

  Matrix projectRotationBlocks(Matrix value) const {
    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      value.block(0, colStart, r, d) =
          projectToStiefelManifold(value.block(0, colStart, r, d));
    }
    return value;
  }

  Matrix localGradientCorrectionDirection(const Matrix &gradient) const {
    Matrix direction = gradient;
    if (options.localGradientCorrectionBoundaryOnly) {
      direction.setZero();
      std::set<unsigned> separatorPoses;
      for (const auto &m : sharedLoops) {
        if (m.r1 == id && m.p1 < n) {
          separatorPoses.insert(m.p1);
        } else if (m.r2 == id && m.p2 < n) {
          separatorPoses.insert(m.p2);
        }
      }
      for (const unsigned pose : separatorPoses) {
        const unsigned colStart = pose * (d + 1);
        direction.block(0, colStart, r, d + 1) =
            gradient.block(0, colStart, r, d + 1);
      }
    }
    if (options.localGradientCorrectionBlockJacobi) {
      direction = localGradientCorrectionBlockJacobiDirection(direction);
    }
    if (options.localGradientCorrectionCompactSchur) {
      direction =
          localGradientCorrectionCompactSchurDirection(
              direction, gradient, localGradientCorrectionStepCap());
    }
    return direction;
  }

  Matrix localGradientCorrectionBlockJacobiDirection(
      const Matrix &direction) const {
    const SparseMatrix &Q = problem.getQRef();
    const unsigned blockDim = d + 1;
    if (Q.rows() != static_cast<int>(n * blockDim) ||
        Q.cols() != static_cast<int>(n * blockDim) || n == 0) {
      return direction;
    }

    std::vector<double> rowAbsSums(static_cast<std::size_t>(Q.rows()), 0.0);
    for (int outer = 0; outer < Q.outerSize(); ++outer) {
      for (SparseMatrix::InnerIterator it(Q, outer); it; ++it) {
        rowAbsSums[static_cast<std::size_t>(it.row())] +=
            std::abs(it.value());
      }
    }

    std::vector<double> blockStiffness(n, 0.0);
    double stiffnessSum = 0.0;
    unsigned finiteBlocks = 0;
    for (unsigned pose = 0; pose < n; ++pose) {
      double stiffness = 0.0;
      const unsigned colStart = pose * blockDim;
      for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
        stiffness = std::max(
            stiffness,
            rowAbsSums[static_cast<std::size_t>(colStart + localCol)]);
      }
      if (std::isfinite(stiffness) && stiffness > 1e-12) {
        blockStiffness[pose] = stiffness;
        stiffnessSum += stiffness;
        ++finiteBlocks;
      }
    }
    if (finiteBlocks == 0 || !std::isfinite(stiffnessSum) ||
        stiffnessSum <= 0.0) {
      return direction;
    }

    const double meanStiffness =
        stiffnessSum / static_cast<double>(finiteBlocks);
    Matrix preconditioned = direction;
    for (unsigned pose = 0; pose < n; ++pose) {
      if (blockStiffness[pose] <= 0.0) {
        continue;
      }
      const double ratio = meanStiffness / blockStiffness[pose];
      if (!std::isfinite(ratio) || ratio <= 0.0) {
        continue;
      }
      const unsigned colStart = pose * blockDim;
      preconditioned.block(0, colStart, r, blockDim) *= ratio;
    }
    return preconditioned;
  }

  struct ActiveSeparatorCandidate {
    bool valid{false};
    Matrix value;
    double cost{std::numeric_limits<double>::infinity()};
    double gradient{std::numeric_limits<double>::infinity()};
    double step{0.0};
    double alpha{0.0};
  };

  ActiveSeparatorCandidate makeFullEquivHybridActiveSeparatorCauchyCandidate(
      double currentCost, const Matrix &gradient, const Matrix &direction,
      double stepCap) const {
    ActiveSeparatorCandidate candidate;
    const Matrix hessianDirection = direction * problem.getQRef();
    if (hessianDirection.rows() != X.rows() ||
        hessianDirection.cols() != X.cols() ||
        !hessianDirection.allFinite()) {
      return candidate;
    }
    const double descentNumerator =
        (gradient.cwiseProduct(direction)).sum();
    const double curvature =
        (hessianDirection.cwiseProduct(direction)).sum();
    if (!std::isfinite(descentNumerator) || !std::isfinite(curvature) ||
        descentNumerator <= 0.0 || curvature <= 1e-18) {
      return candidate;
    }
    const double rawStep = descentNumerator / curvature;
    const double step =
        std::isfinite(rawStep) && rawStep > 0.0
            ? std::min(rawStep, stepCap)
            : 0.0;
    if (step <= 0.0) {
      return candidate;
    }

    const Matrix trialX = projectRotationBlocks(X - step * direction);
    if (trialX.rows() != X.rows() || trialX.cols() != X.cols() ||
        !trialX.allFinite()) {
      return candidate;
    }
    const double trialCost = problem.f(trialX) + modelConstant;
    if (!std::isfinite(trialCost) ||
        trialCost >= currentCost - 1e-12) {
      return candidate;
    }
    candidate.valid = true;
    candidate.value = trialX;
    candidate.cost = trialCost;
    if (options.fullEquivHybridActiveSeparatorLmSchurGradientGuard) {
      candidate.gradient = problem.RieGradNorm(trialX);
    }
    candidate.step = step;
    candidate.alpha = step;
    return candidate;
  }

  ActiveSeparatorCandidate makeFullEquivHybridActiveSeparatorLmSchurCandidate(
      double currentCost, const Matrix &gradient) {
    ActiveSeparatorCandidate best;
    ++lastFullEquivHybridActiveSeparatorLmSchurCandidateCount;
    const SparseMatrix &Q = problem.getQRef();
    const unsigned blockDim = d + 1;
    const unsigned totalCols = n * blockDim;
    if (Q.rows() != static_cast<int>(totalCols) ||
        Q.cols() != static_cast<int>(totalCols) ||
        gradient.rows() != static_cast<int>(r) ||
        gradient.cols() != static_cast<int>(totalCols) || n == 0) {
      ++lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount;
      return best;
    }

    std::set<unsigned> separatorPoseSet;
    for (const auto &m : sharedLoops) {
      if (m.r1 == id && m.p1 < n) {
        separatorPoseSet.insert(m.p1);
      } else if (m.r2 == id && m.p2 < n) {
        separatorPoseSet.insert(m.p2);
      }
    }
    if (separatorPoseSet.empty()) {
      ++lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount;
      return best;
    }

    std::vector<std::pair<unsigned, double>> scoredBoundaryPoses;
    scoredBoundaryPoses.reserve(separatorPoseSet.size());
    for (const unsigned pose : separatorPoseSet) {
      const unsigned colStart = pose * blockDim;
      const Matrix gradientBlock =
          gradient.block(0, colStart, r, blockDim);
      double score = gradientBlock.squaredNorm();
      if (options.fullEquivHybridActiveSeparatorLmSchurScoreMode ==
          ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode::
              ModelDecrease) {
        std::vector<unsigned> poseCols;
        poseCols.reserve(blockDim);
        for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
          poseCols.push_back(colStart + localCol);
        }
        const Matrix Qii = denseQBlock(poseCols, poseCols);
        const Matrix hessianGradientBlock = gradientBlock * Qii;
        const double curvature =
            (hessianGradientBlock.cwiseProduct(gradientBlock)).sum();
        if (std::isfinite(curvature) && curvature > 1e-18 &&
            std::isfinite(score) && score > 0.0) {
          score = (score * score) / curvature;
        } else {
          score = -std::numeric_limits<double>::infinity();
        }
      }
      if (std::isfinite(score) && score > 0.0) {
        scoredBoundaryPoses.emplace_back(pose, score);
      }
    }
    if (scoredBoundaryPoses.empty()) {
      ++lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount;
      return best;
    }
    std::sort(scoredBoundaryPoses.begin(), scoredBoundaryPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second == rhs.second) {
                  return lhs.first < rhs.first;
                }
                return lhs.second > rhs.second;
              });
    if (options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses > 0 &&
        scoredBoundaryPoses.size() >
            options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses) {
      scoredBoundaryPoses.resize(
          options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses);
    }
    std::sort(scoredBoundaryPoses.begin(), scoredBoundaryPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    std::vector<unsigned> boundaryCols;
    std::vector<char> isBoundaryCol(totalCols, 0);
    boundaryCols.reserve(scoredBoundaryPoses.size() * blockDim);
    for (const auto &entry : scoredBoundaryPoses) {
      const unsigned colStart = entry.first * blockDim;
      for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
        const unsigned col = colStart + localCol;
        boundaryCols.push_back(col);
        isBoundaryCol[col] = 1;
      }
    }

    std::map<unsigned, double> couplingByPose;
    for (const unsigned row : boundaryCols) {
      for (SparseMatrix::InnerIterator it(Q, static_cast<int>(row)); it;
           ++it) {
        const unsigned col = static_cast<unsigned>(it.col());
        if (col >= totalCols || isBoundaryCol[col]) {
          continue;
        }
        couplingByPose[col / blockDim] += std::abs(it.value());
      }
    }

    std::vector<std::pair<unsigned, double>> coupledPoses(
        couplingByPose.begin(), couplingByPose.end());
    std::sort(coupledPoses.begin(), coupledPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second == rhs.second) {
                  return lhs.first < rhs.first;
                }
                return lhs.second > rhs.second;
              });
    const unsigned maxPrivateCols =
        options.fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols;
    if (maxPrivateCols > 0) {
      const std::size_t maxPrivatePoses =
          std::max<std::size_t>(1, maxPrivateCols / blockDim);
      if (coupledPoses.size() > maxPrivatePoses) {
        coupledPoses.resize(maxPrivatePoses);
      }
    }
    std::sort(coupledPoses.begin(), coupledPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    std::vector<unsigned> privateCols;
    privateCols.reserve(coupledPoses.size() * blockDim);
    for (const auto &entry : coupledPoses) {
      const unsigned colStart = entry.first * blockDim;
      for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
        const unsigned col = colStart + localCol;
        if (col < totalCols && !isBoundaryCol[col]) {
          privateCols.push_back(col);
        }
      }
    }

    lastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount +=
        boundaryCols.size();
    lastFullEquivHybridActiveSeparatorLmSchurPrivateColCount +=
        privateCols.size();

    Matrix boundaryGradient(gradient.rows(), boundaryCols.size());
    for (unsigned col = 0; col < boundaryCols.size(); ++col) {
      boundaryGradient.col(static_cast<int>(col)) =
          gradient.col(static_cast<int>(boundaryCols[col]));
    }
    Matrix privateGradient(gradient.rows(), privateCols.size());
    for (unsigned col = 0; col < privateCols.size(); ++col) {
      privateGradient.col(static_cast<int>(col)) =
          gradient.col(static_cast<int>(privateCols[col]));
    }
    const Matrix Qbb = denseQBlock(boundaryCols, boundaryCols);
    const Matrix Qbp = denseQBlock(boundaryCols, privateCols);
    const Matrix Qpb = Qbp.transpose();
    const Matrix QppBase = denseQBlock(privateCols, privateCols);
    const double baseDamping =
        std::max(0.0, options.fullEquivHybridActiveSeparatorLmSchurDamping);
    const unsigned backtrackingSteps = std::max(
        1u, options.fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps);
    bool solvedAny = false;

    for (unsigned dampingRetry = 0; dampingRetry < 4; ++dampingRetry) {
      const double damping =
          baseDamping +
          (dampingRetry == 0
               ? 0.0
               : std::pow(10.0, static_cast<int>(dampingRetry) - 10));
      Matrix schur = Qbb;
      Matrix rhs = boundaryGradient.transpose();
      Matrix qppInvQpb;
      Matrix qppInvGp;
      if (!privateCols.empty()) {
        Matrix Qpp = QppBase;
        for (int col = 0; col < Qpp.cols(); ++col) {
          Qpp(col, col) += damping;
        }
        if (!solveSpdSystem(Qpp, Qpb, qppInvQpb) ||
            !solveSpdSystem(Qpp, privateGradient.transpose(), qppInvGp)) {
          continue;
        }
        schur -= Qbp * qppInvQpb;
        rhs -= Qbp * qppInvGp;
      }
      for (int col = 0; col < schur.cols(); ++col) {
        schur(col, col) += damping;
      }
      Matrix boundaryStepTranspose;
      if (!solveSpdSystem(schur, rhs, boundaryStepTranspose)) {
        continue;
      }
      Matrix privateStepTranspose;
      if (!privateCols.empty()) {
        privateStepTranspose = qppInvGp - qppInvQpb * boundaryStepTranspose;
      }
      solvedAny = true;

      Matrix rawDirection = Matrix::Zero(X.rows(), X.cols());
      const Matrix boundaryStep = boundaryStepTranspose.transpose();
      for (unsigned col = 0; col < boundaryCols.size(); ++col) {
        rawDirection.col(static_cast<int>(boundaryCols[col])) =
            boundaryStep.col(static_cast<int>(col));
      }
      if (!privateCols.empty()) {
        const Matrix privateStep = privateStepTranspose.transpose();
        for (unsigned col = 0; col < privateCols.size(); ++col) {
          rawDirection.col(static_cast<int>(privateCols[col])) =
              privateStep.col(static_cast<int>(col));
        }
      }
      Matrix tangentDirection =
          projectFullEquivHybridTangent(X, rawDirection, d);
      if (tangentDirection.rows() != X.rows() ||
          tangentDirection.cols() != X.cols() ||
          !tangentDirection.allFinite() ||
          tangentDirection.squaredNorm() <= 0.0) {
        continue;
      }
      const double maxNorm = options.fullEquivHybridActiveSeparatorStepCap;
      const double directionNorm = tangentDirection.norm();
      if (std::isfinite(maxNorm) && maxNorm > 0.0 &&
          directionNorm > maxNorm) {
        tangentDirection *= maxNorm / directionNorm;
      }

      double alpha = 1.0;
      for (unsigned attempt = 0; attempt < backtrackingSteps; ++attempt) {
        ++lastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount;
        const Matrix trialStep = -alpha * tangentDirection;
        const Matrix trialX =
            retractFullEquivHybridByProjection(X, trialStep, d);
        if (trialX.rows() == X.rows() && trialX.cols() == X.cols() &&
            trialX.allFinite()) {
          const double trialCost = problem.f(trialX) + modelConstant;
          if (std::isfinite(trialCost) &&
              trialCost < best.cost - 1e-12 &&
              trialCost < currentCost - 1e-12) {
            best.valid = true;
            best.value = trialX;
            best.cost = trialCost;
            if (options.fullEquivHybridActiveSeparatorLmSchurGradientGuard) {
              best.gradient = problem.RieGradNorm(trialX);
            }
            best.step = alpha * tangentDirection.norm();
            best.alpha = alpha;
          }
        }
        alpha *= 0.5;
      }
      if (best.valid) {
        break;
      }
    }

    if (!solvedAny) {
      ++lastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount;
    } else if (!best.valid) {
      ++lastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount;
    }
    return best;
  }

  Matrix fullEquivHybridActiveSeparatorDirection(
      const Matrix &gradient) const {
    if (gradient.rows() != static_cast<int>(r) ||
        gradient.cols() != static_cast<int>(n * (d + 1)) ||
        !gradient.allFinite()) {
      return Matrix();
    }
    Matrix direction = Matrix::Zero(gradient.rows(), gradient.cols());
    std::set<unsigned> separatorPoses;
    for (const auto &m : sharedLoops) {
      if (m.r1 == id && m.p1 < n) {
        separatorPoses.insert(m.p1);
      } else if (m.r2 == id && m.p2 < n) {
        separatorPoses.insert(m.p2);
      }
    }
    for (const unsigned pose : separatorPoses) {
      const unsigned colStart = pose * (d + 1);
      direction.block(0, colStart, r, d + 1) =
          gradient.block(0, colStart, r, d + 1);
    }
    if (options.fullEquivHybridActiveSeparatorBlockJacobi) {
      direction = localGradientCorrectionBlockJacobiDirection(direction);
    }
    if (options.fullEquivHybridActiveSeparatorCompactSchur) {
      direction =
          localGradientCorrectionCompactSchurDirection(
              direction, gradient,
              options.fullEquivHybridActiveSeparatorStepCap, true);
    }
    return direction;
  }

  bool applyFullEquivHybridActiveSeparatorCorrection(
      ROPTResult &result) {
    if (!options.fullEquivHybridActiveSeparatorCorrection) {
      return true;
    }
    ++lastFullEquivHybridActiveSeparatorCandidateCount;
    const double stepCap =
        options.fullEquivHybridActiveSeparatorStepCap;
    if (!std::isfinite(stepCap) || stepCap <= 0.0 ||
        X.rows() != static_cast<int>(r) ||
        X.cols() != static_cast<int>(n * (d + 1)) ||
        !X.allFinite()) {
      ++lastFullEquivHybridActiveSeparatorRejectedCount;
      return true;
    }

    const double currentCost = problem.f(X) + modelConstant;
    const Matrix rawGradient = problem.RieGrad(X);
    const double currentGradientNorm = rawGradient.norm();
    const Matrix direction =
        fullEquivHybridActiveSeparatorDirection(rawGradient);
    if (!std::isfinite(currentCost) ||
        !std::isfinite(currentGradientNorm) ||
        rawGradient.rows() != X.rows() ||
        rawGradient.cols() != X.cols() ||
        !rawGradient.allFinite() || direction.rows() != X.rows() ||
        direction.cols() != X.cols() || !direction.allFinite() ||
        direction.squaredNorm() <= 0.0) {
      ++lastFullEquivHybridActiveSeparatorRejectedCount;
      return true;
    }

    ActiveSeparatorCandidate best =
        makeFullEquivHybridActiveSeparatorCauchyCandidate(
            currentCost, rawGradient, direction, stepCap);
    bool selectedLmSchur = false;
    const bool lmSchurRoundAllowed =
        options.fullEquivHybridActiveSeparatorLmSchurMaxRounds == 0 ||
        currentOptimizationRound == 0 ||
        currentOptimizationRound <=
            options.fullEquivHybridActiveSeparatorLmSchurMaxRounds;
    if (options.fullEquivHybridActiveSeparatorLmSchur &&
        lmSchurRoundAllowed) {
      const ActiveSeparatorCandidate lmSchur =
          makeFullEquivHybridActiveSeparatorLmSchurCandidate(
              currentCost, rawGradient);
      const double lmSchurTieTolerance =
          std::max(
              1e-12,
              options.fullEquivHybridActiveSeparatorLmSchurMinCostImprovement);
      bool gradientGuardRejected = false;
      if (lmSchur.valid &&
          options.fullEquivHybridActiveSeparatorLmSchurGradientGuard) {
        const double referenceGradient =
            best.valid && std::isfinite(best.gradient)
                ? best.gradient
                : currentGradientNorm;
        if (std::isfinite(lmSchur.gradient) &&
            std::isfinite(referenceGradient)) {
          lastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum +=
              lmSchur.gradient - referenceGradient;
        }
        const double maxGradient =
            referenceGradient *
                (1.0 +
                 options
                     .fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio) +
            1e-12;
        if (!std::isfinite(lmSchur.gradient) ||
            !std::isfinite(maxGradient) ||
            lmSchur.gradient > maxGradient) {
          gradientGuardRejected = true;
          ++lastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount;
        }
      }
      if (lmSchur.valid && !gradientGuardRejected &&
          (!best.valid || lmSchur.cost < best.cost - lmSchurTieTolerance)) {
        best = lmSchur;
        selectedLmSchur = true;
      } else if (lmSchur.valid && !gradientGuardRejected) {
        ++lastFullEquivHybridActiveSeparatorLmSchurFallbackCount;
      }
    }
    if (!best.valid) {
      ++lastFullEquivHybridActiveSeparatorRejectedCount;
      return true;
    }

    X = best.value;
    ++lastFullEquivHybridActiveSeparatorAcceptedCount;
    lastFullEquivHybridActiveSeparatorStepSum += best.step;
    lastFullEquivHybridActiveSeparatorCostDecreaseSum +=
        currentCost - best.cost;
    if (selectedLmSchur) {
      ++lastFullEquivHybridActiveSeparatorLmSchurAcceptedCount;
      lastFullEquivHybridActiveSeparatorLmSchurAlphaSum += best.alpha;
      lastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum +=
          currentCost - best.cost;
    }
    result.fOpt = best.cost;
    const double trialGradient = problem.RieGradNorm(X);
    if (std::isfinite(trialGradient)) {
      result.gradNormOpt = trialGradient;
    }
    if (hasPreviousFixedPoint &&
        previousFixedPointOutput.rows() == X.rows() &&
        previousFixedPointOutput.cols() == X.cols()) {
      previousFixedPointOutput = X;
    }
    if (options.scheme == ManualDpgoMmScheme::AMM) {
      syncAmmReferenceToCurrentState();
    }
    return true;
  }

  Matrix denseQBlock(const std::vector<unsigned> &rows,
                     const std::vector<unsigned> &cols) const {
    const SparseMatrix &Q = problem.getQRef();
    Matrix block = Matrix::Zero(rows.size(), cols.size());
    for (unsigned row = 0; row < rows.size(); ++row) {
      for (unsigned col = 0; col < cols.size(); ++col) {
        block(static_cast<int>(row), static_cast<int>(col)) =
            Q.coeff(static_cast<int>(rows[row]),
                    static_cast<int>(cols[col]));
      }
    }
    return block;
  }

  bool solveSpdSystem(const Matrix &AIn, const Matrix &rhs,
                      Matrix &solution) const {
    if (AIn.rows() != AIn.cols() || AIn.rows() != rhs.rows()) {
      return false;
    }
    Matrix A = 0.5 * (AIn + AIn.transpose());
    Eigen::LDLT<Matrix> ldlt(A);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
      return false;
    }
    solution = ldlt.solve(rhs);
    return solution.rows() == rhs.rows() &&
           solution.cols() == rhs.cols() && solution.allFinite();
  }

  double localGradientCorrectionDirectionMerit(
      const Matrix &gradient, const Matrix &direction) const {
    if (gradient.rows() != direction.rows() ||
        gradient.cols() != direction.cols() || !gradient.allFinite() ||
        !direction.allFinite() || direction.squaredNorm() <= 0.0) {
      return -std::numeric_limits<double>::infinity();
    }
    const Matrix hessianDirection = direction * problem.getQRef();
    if (hessianDirection.rows() != direction.rows() ||
        hessianDirection.cols() != direction.cols() ||
        !hessianDirection.allFinite()) {
      return -std::numeric_limits<double>::infinity();
    }
    const double numerator = (gradient.cwiseProduct(direction)).sum();
    const double curvature = (hessianDirection.cwiseProduct(direction)).sum();
    if (!std::isfinite(numerator) || !std::isfinite(curvature) ||
        numerator <= 0.0 || curvature <= 0.0) {
      return -std::numeric_limits<double>::infinity();
    }
    return numerator * numerator / curvature;
  }

  double localGradientCorrectionDirectionTrialCost(
      const Matrix &gradient, const Matrix &direction,
      double stepCap) const {
    const double merit =
        localGradientCorrectionDirectionMerit(gradient, direction);
    if (!std::isfinite(merit)) {
      return std::numeric_limits<double>::infinity();
    }
    const Matrix hessianDirection = direction * problem.getQRef();
    const double numerator = (gradient.cwiseProduct(direction)).sum();
    const double curvature = (hessianDirection.cwiseProduct(direction)).sum();
    const double effectiveStepCap =
        std::isfinite(stepCap) ? std::max(0.0, stepCap)
                               : localGradientCorrectionStepCap();
    const double step = std::min(
        effectiveStepCap,
        (curvature > 0.0 ? numerator / curvature : 0.0));
    if (!std::isfinite(step) || step <= 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    const Matrix trial = projectRotationBlocks(X - step * direction);
    if (trial.rows() != X.rows() || trial.cols() != X.cols() ||
        !trial.allFinite()) {
      return std::numeric_limits<double>::infinity();
    }
    const double cost = problem.f(trial) + modelConstant;
    return std::isfinite(cost) ? cost : std::numeric_limits<double>::infinity();
  }

  double localGradientCorrectionDirectionTrialCost(
      const Matrix &gradient, const Matrix &direction) const {
    return localGradientCorrectionDirectionTrialCost(
        gradient, direction, localGradientCorrectionStepCap());
  }

  double localGradientCorrectionDirectionTrialGradient(
      const Matrix &gradient, const Matrix &direction,
      double stepCap) const {
    const double merit =
        localGradientCorrectionDirectionMerit(gradient, direction);
    if (!std::isfinite(merit)) {
      return std::numeric_limits<double>::infinity();
    }
    const Matrix hessianDirection = direction * problem.getQRef();
    const double numerator = (gradient.cwiseProduct(direction)).sum();
    const double curvature = (hessianDirection.cwiseProduct(direction)).sum();
    const double effectiveStepCap =
        std::isfinite(stepCap) ? std::max(0.0, stepCap)
                               : localGradientCorrectionStepCap();
    const double step = std::min(
        effectiveStepCap,
        (curvature > 0.0 ? numerator / curvature : 0.0));
    if (!std::isfinite(step) || step <= 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    const Matrix trial = projectRotationBlocks(X - step * direction);
    if (trial.rows() != X.rows() || trial.cols() != X.cols() ||
        !trial.allFinite()) {
      return std::numeric_limits<double>::infinity();
    }
    const double gradientNorm = problem.RieGradNorm(trial);
    return std::isfinite(gradientNorm) ? gradientNorm
                                       : std::numeric_limits<double>::infinity();
  }

  bool localGradientCorrectionCompactSchurGradientGuardAccepts(
      const Matrix &gradient, const Matrix &baseDirection,
      const Matrix &candidateDirection, double trialStepCap) const {
    if (!options.localGradientCorrectionCompactSchurGradientGuard) {
      return true;
    }
    const double baseGradient =
        localGradientCorrectionDirectionTrialGradient(
            gradient, baseDirection, trialStepCap);
    const double candidateGradient =
        localGradientCorrectionDirectionTrialGradient(
            gradient, candidateDirection, trialStepCap);
    if (std::isfinite(baseGradient) && std::isfinite(candidateGradient)) {
      localGradientCorrectionDiagnostics.compactSchurGradientChangeSum +=
          candidateGradient - baseGradient;
    }
    const double maxGradient =
        baseGradient *
            (1.0 +
             options
                 .localGradientCorrectionCompactSchurMaxGradientIncreaseRatio) +
        1e-12 * std::max(1.0, std::abs(baseGradient));
    if (!std::isfinite(baseGradient) ||
        !std::isfinite(candidateGradient) ||
        !std::isfinite(maxGradient) ||
        candidateGradient > maxGradient) {
      ++localGradientCorrectionDiagnostics
            .compactSchurGradientGuardRejectedCount;
      return false;
    }
    return true;
  }

  Matrix localGradientCorrectionCompactSchurDirection(
      const Matrix &baseDirection, const Matrix &gradient,
      double trialStepCap, bool keepBoundaryDirection = false) const {
    const SparseMatrix &Q = problem.getQRef();
    const unsigned blockDim = d + 1;
    const unsigned totalCols = n * blockDim;
    if (Q.rows() != static_cast<int>(totalCols) ||
        Q.cols() != static_cast<int>(totalCols) ||
        gradient.rows() != static_cast<int>(r) ||
        gradient.cols() != static_cast<int>(totalCols) ||
        baseDirection.rows() != gradient.rows() ||
        baseDirection.cols() != gradient.cols() || n == 0) {
      return baseDirection;
    }

    std::set<unsigned> separatorPoseSet;
    for (const auto &m : sharedLoops) {
      if (m.r1 == id && m.p1 < n) {
        separatorPoseSet.insert(m.p1);
      } else if (m.r2 == id && m.p2 < n) {
        separatorPoseSet.insert(m.p2);
      }
    }
    if (separatorPoseSet.empty()) {
      return baseDirection;
    }

    std::vector<std::pair<unsigned, double>> scoredBoundaryPoses;
    scoredBoundaryPoses.reserve(separatorPoseSet.size());
    for (const unsigned pose : separatorPoseSet) {
      const unsigned colStart = pose * blockDim;
      const double score =
          gradient.block(0, colStart, r, blockDim).squaredNorm();
      if (std::isfinite(score) && score > 0.0) {
        scoredBoundaryPoses.emplace_back(pose, score);
      }
    }
    if (scoredBoundaryPoses.empty()) {
      return baseDirection;
    }
    std::sort(scoredBoundaryPoses.begin(), scoredBoundaryPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second == rhs.second) {
                  return lhs.first < rhs.first;
                }
                return lhs.second > rhs.second;
              });
    if (options.localGradientCorrectionCompactSchurMaxBoundaryPoses > 0 &&
        scoredBoundaryPoses.size() >
            options.localGradientCorrectionCompactSchurMaxBoundaryPoses) {
      scoredBoundaryPoses.resize(
          options.localGradientCorrectionCompactSchurMaxBoundaryPoses);
    }
    std::sort(scoredBoundaryPoses.begin(), scoredBoundaryPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    std::vector<unsigned> boundaryCols;
    std::vector<char> isBoundaryCol(totalCols, 0);
    boundaryCols.reserve(scoredBoundaryPoses.size() * blockDim);
    for (const auto &entry : scoredBoundaryPoses) {
      const unsigned colStart = entry.first * blockDim;
      for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
        const unsigned col = colStart + localCol;
        boundaryCols.push_back(col);
        isBoundaryCol[col] = 1;
      }
    }

    std::map<unsigned, double> couplingByPose;
    for (const unsigned row : boundaryCols) {
      for (SparseMatrix::InnerIterator it(Q, static_cast<int>(row)); it;
           ++it) {
        const unsigned col = static_cast<unsigned>(it.col());
        if (col >= totalCols || isBoundaryCol[col]) {
          continue;
        }
        const unsigned pose = col / blockDim;
        couplingByPose[pose] += std::abs(it.value());
      }
    }

    std::vector<std::pair<unsigned, double>> coupledPoses(
        couplingByPose.begin(), couplingByPose.end());
    std::sort(coupledPoses.begin(), coupledPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second == rhs.second) {
                  return lhs.first < rhs.first;
                }
                return lhs.second > rhs.second;
              });
    const unsigned maxPrivateCols =
        options.localGradientCorrectionCompactSchurMaxPrivateCols;
    if (maxPrivateCols > 0) {
      const std::size_t maxPrivatePoses =
          std::max<std::size_t>(1, maxPrivateCols / blockDim);
      if (coupledPoses.size() > maxPrivatePoses) {
        coupledPoses.resize(maxPrivatePoses);
      }
    }
    std::sort(coupledPoses.begin(), coupledPoses.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });

    std::vector<unsigned> privateCols;
    privateCols.reserve(coupledPoses.size() * blockDim);
    for (const auto &entry : coupledPoses) {
      const unsigned colStart = entry.first * blockDim;
      for (unsigned localCol = 0; localCol < blockDim; ++localCol) {
        const unsigned col = colStart + localCol;
        if (col < totalCols && !isBoundaryCol[col]) {
          privateCols.push_back(col);
        }
      }
    }

    ++localGradientCorrectionDiagnostics.compactSchurCandidateCount;
    localGradientCorrectionDiagnostics.compactSchurBoundaryColCount +=
        boundaryCols.size();
    localGradientCorrectionDiagnostics.compactSchurPrivateColCount +=
        privateCols.size();

    Matrix Qbb = denseQBlock(boundaryCols, boundaryCols);
    Matrix boundaryGradient(gradient.rows(), boundaryCols.size());
    for (unsigned col = 0; col < boundaryCols.size(); ++col) {
      boundaryGradient.col(static_cast<int>(col)) =
          gradient.col(static_cast<int>(boundaryCols[col]));
    }

    const double damping =
        std::max(0.0, options.localGradientCorrectionCompactSchurDamping);
    const double costImprovementMargin =
        std::max(
            1e-12,
            options.localGradientCorrectionCompactSchurMinCostDecrease);
    if (keepBoundaryDirection) {
      if (privateCols.empty()) {
        ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
        return baseDirection;
      }

      Matrix Qbp = denseQBlock(boundaryCols, privateCols);
      Matrix Qpp = denseQBlock(privateCols, privateCols);
      for (int col = 0; col < Qpp.cols(); ++col) {
        Qpp(col, col) += damping;
      }

      Matrix privateGradient(gradient.rows(), privateCols.size());
      for (unsigned col = 0; col < privateCols.size(); ++col) {
        privateGradient.col(static_cast<int>(col)) =
            gradient.col(static_cast<int>(privateCols[col]));
      }
      Matrix boundaryDirection(baseDirection.rows(), boundaryCols.size());
      for (unsigned col = 0; col < boundaryCols.size(); ++col) {
        boundaryDirection.col(static_cast<int>(col)) =
            baseDirection.col(static_cast<int>(boundaryCols[col]));
      }

      Matrix privateStepTranspose;
      const Matrix responseRhs =
          privateGradient.transpose() -
          Qbp.transpose() * boundaryDirection.transpose();
      if (!solveSpdSystem(Qpp, responseRhs, privateStepTranspose)) {
        ++localGradientCorrectionDiagnostics.compactSchurSolveFailureCount;
        return baseDirection;
      }

      Matrix candidate = baseDirection;
      const Matrix privateStep = privateStepTranspose.transpose();
      for (unsigned col = 0; col < privateCols.size(); ++col) {
        candidate.col(static_cast<int>(privateCols[col])) =
            privateStep.col(static_cast<int>(col));
      }
      const double candidateMerit =
          localGradientCorrectionDirectionMerit(gradient, candidate);
      const double baseMerit =
          localGradientCorrectionDirectionMerit(gradient, baseDirection);
      const double candidateCost =
          localGradientCorrectionDirectionTrialCost(
              gradient, candidate, trialStepCap);
      const double baseCost =
          localGradientCorrectionDirectionTrialCost(
              gradient, baseDirection, trialStepCap);
      if (std::isfinite(baseCost) &&
          (!std::isfinite(candidateCost) ||
           candidateCost >= baseCost - costImprovementMargin)) {
        ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
        return baseDirection;
      }
      if (!std::isfinite(candidateMerit) ||
          candidateMerit <= baseMerit + 1e-12) {
        ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
        return baseDirection;
      }
      if (!localGradientCorrectionCompactSchurGradientGuardAccepts(
              gradient, baseDirection, candidate, trialStepCap)) {
        return baseDirection;
      }
      ++localGradientCorrectionDiagnostics.compactSchurAcceptedCount;
      return candidate;
    }

    Matrix schur = Qbb;
    Matrix effectiveBoundaryGradient = boundaryGradient;
    Matrix privateStepTranspose;
    if (!privateCols.empty()) {
      Matrix Qbp = denseQBlock(boundaryCols, privateCols);
      Matrix Qpp = denseQBlock(privateCols, privateCols);
      for (int col = 0; col < Qpp.cols(); ++col) {
        Qpp(col, col) += damping;
      }

      Matrix privateGradient(gradient.rows(), privateCols.size());
      for (unsigned col = 0; col < privateCols.size(); ++col) {
        privateGradient.col(static_cast<int>(col)) =
            gradient.col(static_cast<int>(privateCols[col]));
      }

      Matrix qppInvQpb;
      Matrix qppInvGp;
      if (!solveSpdSystem(Qpp, Qbp.transpose(), qppInvQpb) ||
          !solveSpdSystem(Qpp, privateGradient.transpose(), qppInvGp)) {
        ++localGradientCorrectionDiagnostics.compactSchurSolveFailureCount;
        return baseDirection;
      }
      schur = Qbb - Qbp * qppInvQpb;
      effectiveBoundaryGradient =
          boundaryGradient - (Qbp * qppInvGp).transpose();

      Matrix boundaryStepTranspose;
      for (int col = 0; col < schur.cols(); ++col) {
        schur(col, col) += damping;
      }
      if (!solveSpdSystem(schur, effectiveBoundaryGradient.transpose(),
                          boundaryStepTranspose)) {
        ++localGradientCorrectionDiagnostics.compactSchurSolveFailureCount;
        return baseDirection;
      }
      privateStepTranspose = qppInvGp - qppInvQpb * boundaryStepTranspose;

      Matrix candidate = baseDirection;
      const Matrix boundaryStep = boundaryStepTranspose.transpose();
      const Matrix privateStep = privateStepTranspose.transpose();
      for (unsigned col = 0; col < boundaryCols.size(); ++col) {
        candidate.col(static_cast<int>(boundaryCols[col])) =
            boundaryStep.col(static_cast<int>(col));
      }
      for (unsigned col = 0; col < privateCols.size(); ++col) {
        candidate.col(static_cast<int>(privateCols[col])) =
            privateStep.col(static_cast<int>(col));
      }
      const double candidateMerit =
          localGradientCorrectionDirectionMerit(gradient, candidate);
      const double baseMerit =
          localGradientCorrectionDirectionMerit(gradient, baseDirection);
      const double candidateCost =
          localGradientCorrectionDirectionTrialCost(
              gradient, candidate, trialStepCap);
      const double baseCost =
          localGradientCorrectionDirectionTrialCost(
              gradient, baseDirection, trialStepCap);
      if (std::isfinite(baseCost) &&
          (!std::isfinite(candidateCost) ||
           candidateCost >= baseCost - costImprovementMargin)) {
        ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
        return baseDirection;
      }
      if (!std::isfinite(candidateMerit) ||
          candidateMerit <= baseMerit + 1e-12) {
        ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
        return baseDirection;
      }
      if (!localGradientCorrectionCompactSchurGradientGuardAccepts(
              gradient, baseDirection, candidate, trialStepCap)) {
        return baseDirection;
      }
      ++localGradientCorrectionDiagnostics.compactSchurAcceptedCount;
      return candidate;
    }

    for (int col = 0; col < schur.cols(); ++col) {
      schur(col, col) += damping;
    }
    Matrix boundaryStepTranspose;
    if (!solveSpdSystem(schur, effectiveBoundaryGradient.transpose(),
                        boundaryStepTranspose)) {
      ++localGradientCorrectionDiagnostics.compactSchurSolveFailureCount;
      return baseDirection;
    }
    Matrix candidate = baseDirection;
    const Matrix boundaryStep = boundaryStepTranspose.transpose();
    for (unsigned col = 0; col < boundaryCols.size(); ++col) {
      candidate.col(static_cast<int>(boundaryCols[col])) =
          boundaryStep.col(static_cast<int>(col));
    }
    const double candidateMerit =
        localGradientCorrectionDirectionMerit(gradient, candidate);
    const double baseMerit =
        localGradientCorrectionDirectionMerit(gradient, baseDirection);
    const double candidateCost =
        localGradientCorrectionDirectionTrialCost(
            gradient, candidate, trialStepCap);
    const double baseCost =
        localGradientCorrectionDirectionTrialCost(
            gradient, baseDirection, trialStepCap);
    if (std::isfinite(baseCost) &&
        (!std::isfinite(candidateCost) ||
         candidateCost >= baseCost - costImprovementMargin)) {
      ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
      return baseDirection;
    }
    if (!std::isfinite(candidateMerit) ||
        candidateMerit <= baseMerit + 1e-12) {
      ++localGradientCorrectionDiagnostics.compactSchurGuardRejectedCount;
      return baseDirection;
    }
    if (!localGradientCorrectionCompactSchurGradientGuardAccepts(
            gradient, baseDirection, candidate, trialStepCap)) {
      return baseDirection;
    }
    ++localGradientCorrectionDiagnostics.compactSchurAcceptedCount;
    return candidate;
  }

  double localGradientCorrectionStepCap() const {
    double cap = options.localGradientCorrectionStep;
    for (const double step : options.localGradientCorrectionSteps) {
      if (std::isfinite(step) && step > cap) {
        cap = step;
      }
    }
    return std::max(0.0, cap);
  }

  Matrix makeExtrapolatedStart(const Matrix &current, const Matrix &previous,
                               double gamma) const {
    return projectRotationBlocks(current + gamma * (current - previous));
  }

  std::vector<double> extrapolationGammas() const {
    if (!options.localStateExtrapolationGammas.empty()) {
      return options.localStateExtrapolationGammas;
    }
    return {options.localStateExtrapolationGamma};
  }

  std::vector<double> gExtrapolationGammas() const {
    if (!options.localModelGExtrapolationGammas.empty()) {
      return options.localModelGExtrapolationGammas;
    }
    return {options.localModelGExtrapolationGamma};
  }

  std::vector<double> coupledExtrapolationGammas() const {
    if (!options.coupledStateGExtrapolationGammas.empty()) {
      return options.coupledStateGExtrapolationGammas;
    }
    return {options.coupledStateGExtrapolationGamma};
  }

  std::vector<double> ammGammaScales() const {
    if (!options.ammGammaScales.empty()) {
      return options.ammGammaScales;
    }
    return {1.0};
  }

  void buildNeighborPublicPoseMap() {
    std::map<unsigned, std::set<unsigned>> uniquePoses;
    for (const auto &m : sharedLoops) {
      if (m.r1 == id) {
        uniquePoses[m.r2].insert(m.p2);
      } else if (m.r2 == id) {
        uniquePoses[m.r1].insert(m.p1);
      }
    }
    for (const auto &entry : uniquePoses) {
      neighborPublicPoses[entry.first] =
          std::vector<unsigned>(entry.second.begin(), entry.second.end());
    }
  }

  unsigned neighborPoseStaleness(const PoseID &pose) const {
    const auto it = neighborPoseLastUpdateRound.find(pose);
    if (it == neighborPoseLastUpdateRound.end()) {
      return currentOptimizationRound + 1;
    }
    if (currentOptimizationRound <= it->second + 1) {
      return 0;
    }
    return currentOptimizationRound - it->second - 1;
  }

  bool neighborPoseForLocalModel(const PoseID &pose, Matrix &value) const {
    const auto currentIt = neighborPoseDict.find(pose);
    if (currentIt == neighborPoseDict.end()) {
      return false;
    }
    value = currentIt->second;
    if (!options.staleBoundaryPrediction ||
        !std::isfinite(options.staleBoundaryPredictionGain) ||
        options.staleBoundaryPredictionGain == 0.0 ||
        neighborPoseStaleness(pose) == 0) {
      return value.allFinite();
    }
    const auto previousIt = neighborPosePreviousDict.find(pose);
    const auto currentRoundIt = neighborPoseLastUpdateRound.find(pose);
    const auto previousRoundIt =
        neighborPosePreviousUpdateRound.find(pose);
    if (previousIt == neighborPosePreviousDict.end() ||
        currentRoundIt == neighborPoseLastUpdateRound.end() ||
        previousRoundIt == neighborPosePreviousUpdateRound.end() ||
        currentRoundIt->second <= previousRoundIt->second ||
        previousIt->second.rows() != value.rows() ||
        previousIt->second.cols() != value.cols() ||
        !previousIt->second.allFinite()) {
      return value.allFinite();
    }
    const unsigned age = neighborPoseStaleness(pose);
    const unsigned observedDt =
        currentRoundIt->second - previousRoundIt->second;
    const double scale =
        options.staleBoundaryPredictionGain *
        static_cast<double>(age) / static_cast<double>(observedDt);
    if (!std::isfinite(scale)) {
      return value.allFinite();
    }
    Matrix predicted = value + scale * (value - previousIt->second);
    if (predicted.rows() == static_cast<int>(r) &&
        predicted.cols() == static_cast<int>(d + 1) &&
        predicted.allFinite()) {
      predicted.block(0, 0, r, d) =
          projectToStiefelManifold(predicted.block(0, 0, r, d));
      if (predicted.allFinite()) {
        value = predicted;
      }
    }
    return value.allFinite();
  }

  double localMeasurementStiffness(const RelativeSEMeasurement &m,
                                   bool localIsSource) const {
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1.0;

    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;

    const Matrix localWeight =
        localIsSource ? T * Omega * T.transpose() : Omega;
    const double stiffness =
        localWeight.trace() / static_cast<double>(d + 1);
    return std::isfinite(stiffness) && stiffness > 0.0 ? stiffness : 1.0;
  }

  void addStaleBoundaryProximal(SparseMatrix &Q, SparseMatrix &G,
                                double &constant) const {
    if (!options.staleBoundaryProximal ||
        options.staleBoundaryProximalWeight <= 0.0 || X.rows() == 0) {
      return;
    }
    const unsigned blockDim = d + 1;
    for (const auto &m : sharedLoops) {
      PoseID neighborPose;
      unsigned localPose = 0;
      bool localIsSource = false;
      if (m.r1 == id) {
        neighborPose = PoseID(m.r2, m.p2);
        localPose = m.p1;
        localIsSource = true;
      } else if (m.r2 == id) {
        neighborPose = PoseID(m.r1, m.p1);
        localPose = m.p2;
      } else {
        continue;
      }
      if (localPose >= n) {
        continue;
      }
      const unsigned age = neighborPoseStaleness(neighborPose);
      if (age == 0) {
        continue;
      }
      const double lambda =
          options.staleBoundaryProximalWeight *
          static_cast<double>(age) *
          localMeasurementStiffness(m, localIsSource);
      if (!std::isfinite(lambda) || lambda <= 0.0) {
        continue;
      }
      const Matrix anchor =
          X.block(0, localPose * blockDim, r, blockDim);
      if (!anchor.allFinite()) {
        continue;
      }
      for (unsigned col = 0; col < blockDim; ++col) {
        Q.coeffRef(localPose * blockDim + col,
                   localPose * blockDim + col) += lambda;
        for (unsigned row = 0; row < r; ++row) {
          G.coeffRef(row, localPose * blockDim + col) -=
              lambda * anchor(row, col);
        }
      }
      constant += 0.5 * lambda * anchor.squaredNorm();
    }
  }

  void constructQMatrix() {
    std::vector<RelativeSEMeasurement> privateMeasurements = odometry;
    privateMeasurements.insert(privateMeasurements.end(), privateLoops.begin(),
                               privateLoops.end());

    const std::size_t blockDim = d + 1;
    const std::size_t totalDim = n * blockDim;
    SparseMatrix Q = padSparseMatrix(
        constructConnectionLaplacianSE(privateMeasurements), totalDim,
        totalDim);

    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(row, row) = m.weight * m.kappa;
      }
      Omega(d, d) = m.weight * m.tau;

      Matrix W;
      std::size_t idx = 0;
      if (m.r1 == id) {
        idx = m.p1;
        W = T * Omega * T.transpose();
      } else {
        idx = m.p2;
        W = Omega;
      }

      for (std::size_t col = 0; col < blockDim; ++col) {
        for (std::size_t row = 0; row < blockDim; ++row) {
          Q.coeffRef(idx * blockDim + row, idx * blockDim + col) +=
              W(row, col);
        }
      }
    }
    Q.makeCompressed();
    baseQ = Q;
    problem.setQ(Q);
  }

  bool constructGMatrix() {
    SparseMatrix Q = baseQ;
    SparseMatrix G(r, (d + 1) * n);
    double constant = 0.0;

    for (const auto &m : sharedLoops) {
      Matrix T = Matrix::Zero(d + 1, d + 1);
      T.block(0, 0, d, d) = m.R;
      T.block(0, d, d, 1) = m.t;
      T(d, d) = 1;

      Matrix Omega = Matrix::Zero(d + 1, d + 1);
      for (unsigned row = 0; row < d; ++row) {
        Omega(row, row) = m.weight * m.kappa;
      }
      Omega(d, d) = m.weight * m.tau;

      Matrix L;
      Matrix neighborPose;
      std::size_t idx = 0;
      if (m.r1 == id) {
        const PoseID neighborID = std::make_pair(m.r2, m.p2);
        if (!neighborPoseForLocalModel(neighborID, neighborPose)) {
          return false;
        }
        idx = m.p1;
        L = -neighborPose * Omega * T.transpose();
        constant +=
            0.5 * ((neighborPose * Omega).cwiseProduct(neighborPose)).sum();
      } else {
        const PoseID neighborID = std::make_pair(m.r1, m.p1);
        if (!neighborPoseForLocalModel(neighborID, neighborPose)) {
          return false;
        }
        idx = m.p2;
        L = -neighborPose * T * Omega;
        const Matrix neighborWeight = T * Omega * T.transpose();
        constant +=
            0.5 * ((neighborPose * neighborWeight).cwiseProduct(neighborPose))
                      .sum();
      }

      for (std::size_t col = 0; col < d + 1; ++col) {
        for (std::size_t row = 0; row < r; ++row) {
          G.coeffRef(row, idx * (d + 1) + col) += L(row, col);
        }
      }
    }
    addStaleBoundaryProximal(Q, G, constant);
    Q.makeCompressed();
    G.makeCompressed();
    if (options.staleBoundaryProximal) {
      problem.setQWithoutPreconditioner(Q);
    } else if (problem.getQRef().rows() != baseQ.rows() ||
               problem.getQRef().cols() != baseQ.cols() ||
               problem.getQRef().nonZeros() != baseQ.nonZeros()) {
      problem.setQWithoutPreconditioner(baseQ);
    }
    problem.setG(G);
    modelConstant = constant;
    return true;
  }
};

struct ManualDpgoMmPreparedProblem {
  std::vector<RelativeSEMeasurement> dataset;
  unsigned numPoses{0};
  unsigned d{0};
  unsigned r{0};
  unsigned posesPerRobot{0};
  std::unique_ptr<QuadraticProblem> centralProblem;
  std::vector<std::unique_ptr<ManualDpgoMmAgent>> agents;
};

Matrix identityPoseInitialization(unsigned d, unsigned numPoses) {
  Matrix T = Matrix::Zero(d, numPoses * (d + 1));
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    T.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  return T;
}

Matrix localPoseChordalInitialization(
    unsigned d, unsigned numPoses,
    const std::vector<RelativeSEMeasurement> &odometry,
    const std::vector<RelativeSEMeasurement> &privateLoops) {
  Matrix TLocal;
  if (numPoses <= 1) {
    TLocal = identityPoseInitialization(d, numPoses);
  } else {
    std::vector<RelativeSEMeasurement> localMeasurements = odometry;
    localMeasurements.insert(localMeasurements.end(), privateLoops.begin(),
                             privateLoops.end());
    if (localMeasurements.empty()) {
      TLocal = identityPoseInitialization(d, numPoses);
    } else {
      TLocal = chordalInitialization(d, numPoses, localMeasurements);
      if (!TLocal.allFinite() && odometry.size() + 1 == numPoses) {
        TLocal = odometryInitialization(d, numPoses, odometry);
      }
      if (!TLocal.allFinite()) {
        TLocal = identityPoseInitialization(d, numPoses);
      }
    }
  }
  return TLocal;
}

struct GaugeTransform {
  Matrix R;
  Matrix t;
  bool known{false};
};

struct GaugeCandidate {
  Matrix R;
  Matrix t;
  double weight{1.0};
};

GaugeTransform identityGauge(unsigned d) {
  GaugeTransform gauge;
  gauge.R = Matrix::Identity(d, d);
  gauge.t = Matrix::Zero(d, 1);
  gauge.known = true;
  return gauge;
}

bool estimateNeighborGaugeFromMeasurement(
    const GaugeTransform &knownGauge, const Matrix &knownLocalPose,
    const Matrix &unknownLocalPose, const RelativeSEMeasurement &m,
    bool knownIsFirstRobot, unsigned d, GaugeCandidate &candidate) {
  if (!knownGauge.known || knownLocalPose.rows() != static_cast<int>(d) ||
      unknownLocalPose.rows() != static_cast<int>(d) ||
      knownLocalPose.cols() != static_cast<int>(d + 1) ||
      unknownLocalPose.cols() != static_cast<int>(d + 1)) {
    return false;
  }
  const Matrix RKnownLocal = knownLocalPose.block(0, 0, d, d);
  const Matrix tKnownLocal = knownLocalPose.block(0, d, d, 1);
  const Matrix RUnknownLocal = unknownLocalPose.block(0, 0, d, d);
  const Matrix tUnknownLocal = unknownLocalPose.block(0, d, d, 1);
  const Matrix RKnown = knownGauge.R * RKnownLocal;
  const Matrix tKnown = knownGauge.t + knownGauge.R * tKnownLocal;

  Matrix RPredictedUnknown;
  Matrix tPredictedUnknown;
  if (knownIsFirstRobot) {
    RPredictedUnknown = RKnown * m.R;
    tPredictedUnknown = tKnown + RKnown * m.t;
  } else {
    RPredictedUnknown = RKnown * m.R.transpose();
    tPredictedUnknown = tKnown - RPredictedUnknown * m.t;
  }

  candidate.R =
      projectToRotationGroup(RPredictedUnknown * RUnknownLocal.transpose());
  candidate.t = tPredictedUnknown - candidate.R * tUnknownLocal;
  candidate.weight =
      std::max(1e-12, m.weight * (std::max(0.0, m.kappa) +
                                  std::max(0.0, m.tau)));
  return candidate.R.allFinite() && candidate.t.allFinite() &&
         std::isfinite(candidate.weight);
}

unsigned localPoseCount(const Matrix &localT, unsigned d) {
  const unsigned blockDim = d + 1;
  if (blockDim == 0 || localT.cols() % blockDim != 0) {
    return 0;
  }
  return static_cast<unsigned>(localT.cols() / blockDim);
}

bool getLocalPoseBlock(const std::vector<Matrix> &localTs, unsigned robot,
                       unsigned poseIndex, unsigned d, Matrix &pose) {
  if (robot >= localTs.size()) {
    return false;
  }
  const Matrix &localT = localTs[robot];
  const unsigned blockDim = d + 1;
  const unsigned count = localPoseCount(localT, d);
  if (localT.rows() != static_cast<int>(d) || poseIndex >= count) {
    return false;
  }
  pose = localT.block(0, poseIndex * blockDim, d, blockDim);
  return pose.allFinite();
}

bool isContiguousBoundaryOdometryMeasurement(const std::vector<Matrix> &localTs,
                                             const RelativeSEMeasurement &m,
                                             unsigned d) {
  if (m.r1 >= localTs.size() || m.r2 >= localTs.size() || m.r1 == m.r2) {
    return false;
  }
  const unsigned n1 = localPoseCount(localTs[m.r1], d);
  const unsigned n2 = localPoseCount(localTs[m.r2], d);
  if (n1 == 0 || n2 == 0) {
    return false;
  }
  return (m.r1 + 1 == m.r2 && m.p1 + 1 == n1 && m.p2 == 0) ||
         (m.r2 + 1 == m.r1 && m.p2 + 1 == n2 && m.p1 == 0);
}

bool robotGaugeGraphConnected(
    unsigned numRobots,
    const std::vector<RelativeSEMeasurement> &robotMeasurements) {
  if (numRobots == 0) {
    return false;
  }
  std::vector<std::vector<unsigned>> adjacency(numRobots);
  for (const auto &m : robotMeasurements) {
    if (m.r1 >= numRobots || m.r2 >= numRobots || m.r1 == m.r2) {
      continue;
    }
    adjacency[m.r1].push_back(m.r2);
    adjacency[m.r2].push_back(m.r1);
  }

  std::vector<char> visited(numRobots, 0);
  std::queue<unsigned> frontier;
  visited[0] = 1;
  frontier.push(0);
  unsigned visitedCount = 0;
  while (!frontier.empty()) {
    const unsigned robot = frontier.front();
    frontier.pop();
    ++visitedCount;
    for (const unsigned neighbor : adjacency[robot]) {
      if (neighbor < numRobots && !visited[neighbor]) {
        visited[neighbor] = 1;
        frontier.push(neighbor);
      }
    }
  }
  return visitedCount == numRobots;
}

void applyGaugeToLocalPoseMatrix(Matrix &localT, const GaugeTransform &gauge,
                                 unsigned d) {
  if (!gauge.known || localT.rows() != static_cast<int>(d)) {
    return;
  }
  const unsigned blockDim = d + 1;
  const unsigned count = localPoseCount(localT, d);
  for (unsigned pose = 0; pose < count; ++pose) {
    const unsigned col = pose * blockDim;
    const Matrix RLocal = localT.block(0, col, d, d);
    const Matrix tLocal = localT.block(0, col + d, d, 1);
    localT.block(0, col, d, d) = projectToRotationGroup(gauge.R * RLocal);
    localT.block(0, col + d, d, 1) = gauge.t + gauge.R * tLocal;
  }
}

bool computeRobotGaugeChordal(
    const std::vector<Matrix> &localTs,
    const std::vector<std::vector<RelativeSEMeasurement>> &sharedLoops,
    unsigned d, bool boundaryOdometryOnly,
    std::vector<GaugeTransform> &gauges) {
  std::vector<RelativeSEMeasurement> robotMeasurements;
  for (unsigned robot = 0; robot < sharedLoops.size(); ++robot) {
    for (const auto &m : sharedLoops[robot]) {
      if (m.r1 != robot || m.r1 == m.r2 || m.r1 >= localTs.size() ||
          m.r2 >= localTs.size()) {
        continue;
      }
      if (boundaryOdometryOnly &&
          !isContiguousBoundaryOdometryMeasurement(localTs, m, d)) {
        continue;
      }
      Matrix firstLocalPose;
      Matrix secondLocalPose;
      if (!getLocalPoseBlock(localTs, m.r1, m.p1, d, firstLocalPose) ||
          !getLocalPoseBlock(localTs, m.r2, m.p2, d, secondLocalPose)) {
        continue;
      }
      const Matrix RFirst = firstLocalPose.block(0, 0, d, d);
      const Matrix tFirst = firstLocalPose.block(0, d, d, 1);
      const Matrix RSecond = secondLocalPose.block(0, 0, d, d);
      const Matrix tSecond = secondLocalPose.block(0, d, d, 1);
      const Matrix RRelative =
          projectToRotationGroup(RFirst * m.R * RSecond.transpose());
      const Matrix tRelative =
          tFirst + RFirst * m.t - RRelative * tSecond;
      if (!RRelative.allFinite() || !tRelative.allFinite()) {
        continue;
      }
      RelativeSEMeasurement robotMeasurement(
          m.r1, m.r2, m.r1, m.r2, RRelative, tRelative, m.kappa, m.tau);
      robotMeasurement.weight = m.weight;
      robotMeasurements.push_back(robotMeasurement);
    }
  }

  if (robotMeasurements.empty() ||
      !robotGaugeGraphConnected(static_cast<unsigned>(localTs.size()),
                                robotMeasurements)) {
    return false;
  }

  const Matrix TGauge = chordalInitialization(d, localTs.size(),
                                              robotMeasurements);
  if (!TGauge.allFinite()) {
    return false;
  }
  gauges.assign(localTs.size(), GaugeTransform());
  for (unsigned robot = 0; robot < localTs.size(); ++robot) {
    gauges[robot].R = TGauge.block(0, robot * (d + 1), d, d);
    gauges[robot].t = TGauge.block(0, robot * (d + 1) + d, d, 1);
    gauges[robot].known =
        gauges[robot].R.allFinite() && gauges[robot].t.allFinite();
  }
  return std::all_of(gauges.begin(), gauges.end(),
                     [](const GaugeTransform &gauge) {
                       return gauge.known;
                     });
}

double evaluateRobotGaugeSharedResidual(
    const std::vector<Matrix> &localTs,
    const std::vector<std::vector<RelativeSEMeasurement>> &sharedLoops,
    const std::vector<GaugeTransform> &gauges, unsigned d,
    bool boundaryOdometryOnly) {
  if (gauges.size() != localTs.size()) {
    return std::numeric_limits<double>::infinity();
  }

  double cost = 0.0;
  unsigned evaluated = 0;
  for (unsigned robot = 0; robot < sharedLoops.size(); ++robot) {
    for (const auto &m : sharedLoops[robot]) {
      if (m.r1 != robot || m.r1 == m.r2 || m.r1 >= localTs.size() ||
          m.r2 >= localTs.size()) {
        continue;
      }
      if (boundaryOdometryOnly &&
          !isContiguousBoundaryOdometryMeasurement(localTs, m, d)) {
        continue;
      }
      if (!gauges[m.r1].known || !gauges[m.r2].known) {
        return std::numeric_limits<double>::infinity();
      }

      Matrix firstLocalPose;
      Matrix secondLocalPose;
      if (!getLocalPoseBlock(localTs, m.r1, m.p1, d, firstLocalPose) ||
          !getLocalPoseBlock(localTs, m.r2, m.p2, d, secondLocalPose)) {
        return std::numeric_limits<double>::infinity();
      }
      const Matrix R1Local = firstLocalPose.block(0, 0, d, d);
      const Matrix t1Local = firstLocalPose.block(0, d, d, 1);
      const Matrix R2Local = secondLocalPose.block(0, 0, d, d);
      const Matrix t2Local = secondLocalPose.block(0, d, d, 1);
      const Matrix R1 = gauges[m.r1].R * R1Local;
      const Matrix t1 = gauges[m.r1].t + gauges[m.r1].R * t1Local;
      const Matrix R2 = gauges[m.r2].R * R2Local;
      const Matrix t2 = gauges[m.r2].t + gauges[m.r2].R * t2Local;
      const double weight = std::max(0.0, m.weight);
      const double edgeCost = weight * computeMeasurementError(m, R1, t1,
                                                               R2, t2);
      if (!std::isfinite(edgeCost)) {
        return std::numeric_limits<double>::infinity();
      }
      cost += edgeCost;
      ++evaluated;
    }
  }
  if (evaluated == 0 || !std::isfinite(cost)) {
    return std::numeric_limits<double>::infinity();
  }
  return cost;
}

bool computeBfsRobotGauges(
    const std::vector<Matrix> &localTs,
    const std::vector<std::vector<RelativeSEMeasurement>> &sharedLoops,
    unsigned d, std::vector<GaugeTransform> &gauges) {
  if (localTs.empty()) {
    return false;
  }

  gauges.assign(localTs.size(), GaugeTransform());
  gauges[0] = identityGauge(d);
  std::queue<unsigned> frontier;
  frontier.push(0);

  while (!frontier.empty()) {
    const unsigned knownRobot = frontier.front();
    frontier.pop();
    std::map<unsigned, std::vector<GaugeCandidate>> candidatesByNeighbor;
    for (const auto &m : sharedLoops[knownRobot]) {
      bool knownIsFirstRobot = false;
      unsigned neighbor = 0;
      unsigned knownPoseIndex = 0;
      unsigned neighborPoseIndex = 0;
      if (m.r1 == knownRobot) {
        knownIsFirstRobot = true;
        neighbor = m.r2;
        knownPoseIndex = m.p1;
        neighborPoseIndex = m.p2;
      } else if (m.r2 == knownRobot) {
        knownIsFirstRobot = false;
        neighbor = m.r1;
        knownPoseIndex = m.p2;
        neighborPoseIndex = m.p1;
      } else {
        continue;
      }
      if (neighbor >= localTs.size() || gauges[neighbor].known) {
        continue;
      }

      Matrix knownLocalPose;
      Matrix neighborLocalPose;
      if (!getLocalPoseBlock(localTs, knownRobot, knownPoseIndex, d,
                             knownLocalPose) ||
          !getLocalPoseBlock(localTs, neighbor, neighborPoseIndex, d,
                             neighborLocalPose)) {
        continue;
      }
      GaugeCandidate candidate;
      if (estimateNeighborGaugeFromMeasurement(
              gauges[knownRobot], knownLocalPose, neighborLocalPose, m,
              knownIsFirstRobot, d, candidate)) {
        candidatesByNeighbor[neighbor].push_back(candidate);
      }
    }

    for (const auto &entry : candidatesByNeighbor) {
      const unsigned neighbor = entry.first;
      if (gauges[neighbor].known || entry.second.empty()) {
        continue;
      }
      Matrix RSum = Matrix::Zero(d, d);
      Matrix tSum = Matrix::Zero(d, 1);
      double weightSum = 0.0;
      for (const auto &candidate : entry.second) {
        RSum += candidate.weight * candidate.R;
        tSum += candidate.weight * candidate.t;
        weightSum += candidate.weight;
      }
      if (weightSum <= 0.0 || !std::isfinite(weightSum)) {
        continue;
      }
      gauges[neighbor].R = projectToRotationGroup(RSum / weightSum);
      gauges[neighbor].t = tSum / weightSum;
      gauges[neighbor].known = gauges[neighbor].R.allFinite() &&
                               gauges[neighbor].t.allFinite();
      if (gauges[neighbor].known) {
        frontier.push(neighbor);
      }
    }
  }

  return std::all_of(gauges.begin(), gauges.end(),
                     [](const GaugeTransform &gauge) {
                       return gauge.known;
                     });
}

void alignDistributedLocalChordalGauges(
    std::vector<Matrix> &localTs,
    const std::vector<std::vector<RelativeSEMeasurement>> &sharedLoops,
    unsigned d) {
  if (localTs.empty()) {
    return;
  }

  std::vector<GaugeTransform> allSharedGauges;
  std::vector<GaugeTransform> boundaryGauges;
  std::vector<GaugeTransform> bfsGauges;
  const bool hasAllSharedGauge =
      computeRobotGaugeChordal(localTs, sharedLoops, d, false,
                               allSharedGauges);
  const bool hasBoundaryGauge =
      computeRobotGaugeChordal(localTs, sharedLoops, d, true, boundaryGauges);
  const bool hasBfsGauge =
      computeBfsRobotGauges(localTs, sharedLoops, d, bfsGauges);

  std::vector<GaugeTransform> gauges;
  double bestCost = std::numeric_limits<double>::infinity();
  auto considerGauge = [&](bool valid,
                           const std::vector<GaugeTransform> &candidate) {
    if (!valid) {
      return;
    }
    const double cost =
        evaluateRobotGaugeSharedResidual(localTs, sharedLoops, candidate, d,
                                         false);
    if (std::isfinite(cost) && cost < bestCost) {
      bestCost = cost;
      gauges = candidate;
    }
  };
  considerGauge(hasAllSharedGauge, allSharedGauges);
  considerGauge(hasBoundaryGauge, boundaryGauges);
  considerGauge(hasBfsGauge, bfsGauges);
  if (gauges.empty()) {
    return;
  }

  for (unsigned robot = 0; robot < localTs.size(); ++robot) {
    applyGaugeToLocalPoseMatrix(localTs[robot], gauges[robot], d);
  }
}

void validateManualDpgoMmOptions(const ManualDpgoMmOptions &options) {
  if (options.numRobots == 0) {
    throw std::invalid_argument("numRobots must be positive");
  }
  if (options.localStateExtrapolation &&
      !std::isfinite(options.localStateExtrapolationGamma)) {
    throw std::invalid_argument("localStateExtrapolationGamma must be finite");
  }
  for (const double gamma : options.localStateExtrapolationGammas) {
    if (!std::isfinite(gamma)) {
      throw std::invalid_argument(
          "localStateExtrapolationGammas must contain finite values");
    }
  }
  if ((options.localBoundaryProximalCandidate ||
       options.communicationTopologyBoundaryCandidate) &&
      (!std::isfinite(options.localBoundaryProximalWeight) ||
       options.localBoundaryProximalWeight <= 0.0)) {
    throw std::invalid_argument(
        "localBoundaryProximalWeight must be finite and positive");
  }
  if (!std::isfinite(options.staleBoundaryProximalWeight) ||
      options.staleBoundaryProximalWeight < 0.0) {
    throw std::invalid_argument(
        "staleBoundaryProximalWeight must be finite and nonnegative");
  }
  if (!std::isfinite(options.staleBoundaryPredictionGain) ||
      options.staleBoundaryPredictionGain < 0.0) {
    throw std::invalid_argument(
        "staleBoundaryPredictionGain must be finite and nonnegative");
  }
  if (!std::isfinite(options.communicationTopologyHopBudgetFraction) ||
      options.communicationTopologyHopBudgetFraction < 0.0 ||
      options.communicationTopologyHopBudgetFraction > 1.0) {
    throw std::invalid_argument(
        "communicationTopologyHopBudgetFraction must be finite in [0, 1]");
  }
  if (!std::isfinite(options.communicationTopologyStaleAwareRelayAgeGain) ||
      options.communicationTopologyStaleAwareRelayAgeGain < 0.0) {
    throw std::invalid_argument(
        "communicationTopologyStaleAwareRelayAgeGain must be finite and nonnegative");
  }
  if (!std::isfinite(
          options.communicationTopologyDeltaMaxReconstructionError) ||
      options.communicationTopologyDeltaMaxReconstructionError < 0.0) {
    throw std::invalid_argument(
        "communicationTopologyDeltaMaxReconstructionError must be finite and nonnegative");
  }
  if (!std::isfinite(
          options.communicationTopologyLiftedDeltaMaxReconstructionError) ||
      options.communicationTopologyLiftedDeltaMaxReconstructionError < 0.0) {
    throw std::invalid_argument(
        "communicationTopologyLiftedDeltaMaxReconstructionError must be finite and nonnegative");
  }
  if (options.communicationTopologyLiftedDeltaRank == 0) {
    throw std::invalid_argument(
        "communicationTopologyLiftedDeltaRank must be positive");
  }
  if (options.communicationTopologyValueSchedulerMode !=
          "static_sensitivity_staleness" &&
      options.communicationTopologyValueSchedulerMode !=
          "boundary_residual_staleness") {
    throw std::invalid_argument(
        "communicationTopologyValueSchedulerMode must be static_sensitivity_staleness or boundary_residual_staleness");
  }
  if (!std::isfinite(options.communicationTopologyValueByteBudgetMb) ||
      options.communicationTopologyValueByteBudgetMb < 0.0) {
    throw std::invalid_argument(
        "communicationTopologyValueByteBudgetMb must be finite and nonnegative");
  }
  if (!std::isfinite(options.communicationTopologyValueMinScoreRatio) ||
      options.communicationTopologyValueMinScoreRatio < 0.0 ||
      options.communicationTopologyValueMinScoreRatio > 1.0) {
    throw std::invalid_argument(
        "communicationTopologyValueMinScoreRatio must be finite in [0, 1]");
  }
  if (!validInterfaceModelPayload(
          options.communicationTopologyInterfaceModelPayload)) {
    throw std::invalid_argument(
        "communicationTopologyInterfaceModelPayload must be direction_block, diag_stiffness, or direction_block_diag_stiffness");
  }
  if (!std::isfinite(options.communicationTopologyBoundarySurrogateStaleGain) ||
      options.communicationTopologyBoundarySurrogateStaleGain < 0.0) {
    throw std::invalid_argument(
        "communicationTopologyBoundarySurrogateStaleGain must be finite and nonnegative");
  }
  if (!std::isfinite(options.communicationTopologyReducedInterfaceWeight) ||
      options.communicationTopologyReducedInterfaceWeight < 0.0) {
    throw std::invalid_argument(
        "communicationTopologyReducedInterfaceWeight must be finite and nonnegative");
  }
  if (options.communicationTopologyReducedInterfaceModel &&
      options.ammDpgoRecursiveSimpleState) {
    throw std::invalid_argument(
        "communicationTopologyReducedInterfaceModel is not compatible with ammDpgoRecursiveSimpleState");
  }
  for (const double weight : options.localBoundaryProximalWeights) {
    if (!std::isfinite(weight) || weight <= 0.0) {
      throw std::invalid_argument(
          "localBoundaryProximalWeights must contain positive finite values");
    }
  }
  if (options.localModelGExtrapolation &&
      !std::isfinite(options.localModelGExtrapolationGamma)) {
    throw std::invalid_argument("localModelGExtrapolationGamma must be finite");
  }
  for (const double gamma : options.localModelGExtrapolationGammas) {
    if (!std::isfinite(gamma)) {
      throw std::invalid_argument(
          "localModelGExtrapolationGammas must contain finite values");
    }
  }
  if (options.coupledStateGExtrapolation &&
      !std::isfinite(options.coupledStateGExtrapolationGamma)) {
    throw std::invalid_argument(
        "coupledStateGExtrapolationGamma must be finite");
  }
  for (const double gamma : options.coupledStateGExtrapolationGammas) {
    if (!std::isfinite(gamma)) {
      throw std::invalid_argument(
          "coupledStateGExtrapolationGammas must contain finite values");
    }
  }
  if (!std::isfinite(options.localAndersonMaxAlpha) ||
      options.localAndersonMaxAlpha < 0.0) {
    throw std::invalid_argument(
        "localAndersonMaxAlpha must be finite and nonnegative");
  }
  if (!std::isfinite(options.localSquaremMaxAlpha) ||
      options.localSquaremMaxAlpha < 0.0) {
    throw std::invalid_argument(
        "localSquaremMaxAlpha must be finite and nonnegative");
  }
  if (!std::isfinite(options.reducedRotationTcgRelativeTolerance) ||
      options.reducedRotationTcgRelativeTolerance < 0.0) {
    throw std::invalid_argument(
        "reducedRotationTcgRelativeTolerance must be finite and nonnegative");
  }
  if (!std::isfinite(options.fullEquivHybridSchurDamping) ||
      options.fullEquivHybridSchurDamping < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridSchurDamping must be finite and nonnegative");
  }
  if (!std::isfinite(options.fullEquivHybridLinearRelativeTolerance) ||
      options.fullEquivHybridLinearRelativeTolerance < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridLinearRelativeTolerance must be finite and nonnegative");
  }
  if (!std::isfinite(options.fullEquivHybridLinearAbsoluteTolerance) ||
      options.fullEquivHybridLinearAbsoluteTolerance < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridLinearAbsoluteTolerance must be finite and nonnegative");
  }
  if (options.fullEquivHybridLinearMaxIterations == 0) {
    throw std::invalid_argument(
        "fullEquivHybridLinearMaxIterations must be positive");
  }
  if (options.fullEquivHybridRqnMemorySize == 0) {
    throw std::invalid_argument(
        "fullEquivHybridRqnMemorySize must be positive");
  }
  if (options.fullEquivHybridRqnMemoryPreconditioner) {
    if (options.fullEquivHybridBackend !=
        ManualDpgoMmFullEquivHybridBackend::PcgFull) {
      throw std::invalid_argument(
          "fullEquivHybridRqnMemoryPreconditioner requires PCG_FULL");
    }
    if (options.fullEquivHybridLinearTranslationSchurPreconditioner ||
        options.fullEquivHybridLocalChainPreconditioner ||
        options.fullEquivHybridTranslationBlockPreconditioner ||
        options.fullEquivHybridTranslationSparseSchurPreconditioner ||
        options.fullEquivHybridTranslationLocalSchurPreconditioner ||
        options.fullEquivHybridLaplacianDeflationPreconditioner ||
        options.fullEquivHybridReducedRotationPreconditioner) {
      throw std::invalid_argument(
          "fullEquivHybridRqnMemoryPreconditioner cannot be combined with other explicit FE-Hybrid preconditioners");
    }
  }
  if (options.fullEquivHybridActiveSeparatorCorrection) {
    if (options.localSolver != ManualDpgoMmLocalSolver::FullEquivHybrid) {
      throw std::invalid_argument(
          "fullEquivHybridActiveSeparatorCorrection requires the full_equiv_hybrid local solver");
    }
    if (options.fullEquivHybridRqnWarmStart ||
        options.fullEquivHybridRqnMemoryPreconditioner) {
      throw std::invalid_argument(
          "fullEquivHybridActiveSeparatorCorrection cannot be combined with FE-Hybrid RQN memory options");
    }
  }
  if (options.fullEquivHybridActiveSeparatorCompactSchur &&
      !options.fullEquivHybridActiveSeparatorCorrection) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorCompactSchur requires fullEquivHybridActiveSeparatorCorrection");
  }
  if (options.fullEquivHybridActiveSeparatorLmSchur &&
      !options.fullEquivHybridActiveSeparatorCorrection) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchur requires fullEquivHybridActiveSeparatorCorrection");
  }
  if (options.fullEquivHybridActiveSeparatorLmSchur &&
      options.fullEquivHybridActiveSeparatorCompactSchur) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchur cannot be combined with fullEquivHybridActiveSeparatorCompactSchur");
  }
  if (options.fullEquivHybridActiveSeparatorLmSchurGradientGuard &&
      !options.fullEquivHybridActiveSeparatorLmSchur) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchurGradientGuard requires fullEquivHybridActiveSeparatorLmSchur");
  }
  if (!std::isfinite(
          options
              .fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio) ||
      options.fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio <
          0.0) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio must be finite and nonnegative");
  }
  if (!std::isfinite(
          options
              .fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio) ||
      options
              .fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio <
          0.0) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio must be finite and nonnegative");
  }
  if (options.fullEquivHybridTranslationRecoveryPolishGradientGuard &&
      !options.fullEquivHybridTranslationRecoveryPolish) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishGradientGuard requires fullEquivHybridTranslationRecoveryPolish");
  }
  if (options.fullEquivHybridTranslationRecoveryPolishBacktracking &&
      !options.fullEquivHybridTranslationRecoveryPolishGradientGuard) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishBacktracking requires fullEquivHybridTranslationRecoveryPolishGradientGuard");
  }
  if (options.fullEquivHybridTranslationRecoveryPolishBacktracking &&
      options.fullEquivHybridTranslationRecoveryPolishBacktrackingSteps == 0) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishBacktrackingSteps must be positive when backtracking is enabled");
  }
  if (options.fullEquivHybridTranslationRecoveryPolishMeritSelector &&
      !options.fullEquivHybridTranslationRecoveryPolish) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishMeritSelector requires fullEquivHybridTranslationRecoveryPolish");
  }
  if (options.fullEquivHybridTranslationRecoveryPolishSelectedOnly &&
      !options.fullEquivHybridTranslationRecoveryPolish) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishSelectedOnly requires fullEquivHybridTranslationRecoveryPolish");
  }
  if (options.fullEquivHybridTranslationRecoveryStepTrials) {
    if (options.localSolver != ManualDpgoMmLocalSolver::FullEquivHybrid) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryStepTrials requires the full_equiv_hybrid local solver");
    }
    if (options.fullEquivHybridBackend ==
        ManualDpgoMmFullEquivHybridBackend::ManualFullRtr) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryStepTrials requires a FE-Hybrid Schur/PCG backend");
    }
  }
  if (options.fullEquivHybridTranslationRecoveryInitialGuess) {
    if (options.localSolver != ManualDpgoMmLocalSolver::FullEquivHybrid) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryInitialGuess requires the full_equiv_hybrid local solver");
    }
    if (options.fullEquivHybridBackend !=
        ManualDpgoMmFullEquivHybridBackend::PcgFull) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryInitialGuess requires PCG_FULL");
    }
    if (options.fullEquivHybridReducedRotationInitialGuess ||
        options.fullEquivHybridRqnWarmStart ||
        options.fullEquivHybridRqnMemoryPreconditioner ||
        options.fullEquivHybridTranslationRecoveryPolish ||
        options.fullEquivHybridTranslationRecoveryStepTrials) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryInitialGuess cannot be combined with other FE-Hybrid initial-guess or translation-recovery options");
    }
  }
  if (!std::isfinite(
          options
              .fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio) ||
      options
              .fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio <
          0.0 ||
      options
              .fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio >
          1.0) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio must be finite and in [0, 1]");
  }
  if (options.fullEquivHybridTranslationRecoveryPolishMeritSelector &&
      options.fullEquivHybridTranslationRecoveryPolishBacktrackingSteps == 0) {
    throw std::invalid_argument(
        "fullEquivHybridTranslationRecoveryPolishBacktrackingSteps must be positive when merit selector is enabled");
  }
  if (!std::isfinite(options.fullEquivHybridStepParetoMinDecreaseRatio) ||
      options.fullEquivHybridStepParetoMinDecreaseRatio < 0.0 ||
      options.fullEquivHybridStepParetoMinDecreaseRatio > 1.0) {
    throw std::invalid_argument(
        "fullEquivHybridStepParetoMinDecreaseRatio must be finite and in [0, 1]");
  }
  if (options.fullEquivHybridStepParetoSelector &&
      options.localSolver != ManualDpgoMmLocalSolver::FullEquivHybrid) {
    throw std::invalid_argument(
        "fullEquivHybridStepParetoSelector requires the full_equiv_hybrid local solver");
  }
  if (options.fullEquivHybridStepParetoSelector &&
      options.fullEquivHybridBackend ==
          ManualDpgoMmFullEquivHybridBackend::ManualFullRtr) {
    throw std::invalid_argument(
        "fullEquivHybridStepParetoSelector requires a FE-Hybrid Schur/PCG backend");
  }
  if (options.fullEquivHybridTranslationRecoveryPolish) {
    if (options.localSolver != ManualDpgoMmLocalSolver::FullEquivHybrid) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryPolish requires the full_equiv_hybrid local solver");
    }
    if (options.fullEquivHybridBackend ==
        ManualDpgoMmFullEquivHybridBackend::ManualFullRtr) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryPolish requires a FE-Hybrid Schur/PCG backend");
    }
    if (options.fullEquivHybridRqnWarmStart ||
        options.fullEquivHybridRqnMemoryPreconditioner) {
      throw std::invalid_argument(
          "fullEquivHybridTranslationRecoveryPolish cannot be combined with FE-Hybrid RQN memory options");
    }
  }
  if (!std::isfinite(options.fullEquivHybridActiveSeparatorStepCap) ||
      options.fullEquivHybridActiveSeparatorStepCap < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorStepCap must be finite and nonnegative");
  }
  if (options.fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps == 0) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps must be positive");
  }
  if (!std::isfinite(
          options.fullEquivHybridActiveSeparatorLmSchurDamping) ||
      options.fullEquivHybridActiveSeparatorLmSchurDamping < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchurDamping must be finite and nonnegative");
  }
  if (!std::isfinite(
          options.fullEquivHybridActiveSeparatorLmSchurMinCostImprovement) ||
      options.fullEquivHybridActiveSeparatorLmSchurMinCostImprovement < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridActiveSeparatorLmSchurMinCostImprovement must be finite and nonnegative");
  }
  if (options.fullEquivHybridLocalPortfolio &&
      (options.fullEquivHybridRqnWarmStart ||
       options.fullEquivHybridRqnMemoryPreconditioner)) {
    throw std::invalid_argument(
        "fullEquivHybridLocalPortfolio cannot be combined with FE-Hybrid RQN memory options");
  }
  if (options.manualFullPortfolio &&
      options.localSolver != ManualDpgoMmLocalSolver::ManualFull) {
    throw std::invalid_argument(
        "manualFullPortfolio requires the manual_full local solver");
  }
  if (options.localSolver == ManualDpgoMmLocalSolver::ManualFull &&
      options.ammDpgoSurrogateParity) {
    throw std::invalid_argument(
        "manual_full AMM does not support DPGO simple-surrogate parity");
  }
  if (options.localSolver == ManualDpgoMmLocalSolver::ManualFull &&
      options.ammDpgoMixedSurrogatePortfolio) {
    throw std::invalid_argument(
        "manual_full AMM does not support the reduced mixed-surrogate portfolio");
  }
  if (options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid &&
      options.localStateExtrapolation &&
      (options.fullEquivHybridRqnWarmStart ||
       options.fullEquivHybridRqnMemoryPreconditioner)) {
    throw std::invalid_argument(
        "FE-Hybrid localStateExtrapolation cannot be combined with FE-Hybrid RQN memory options");
  }
  if (options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid &&
      options.ammDpgoMixedSurrogatePortfolio &&
      (options.fullEquivHybridRqnWarmStart ||
       options.fullEquivHybridRqnMemoryPreconditioner)) {
    throw std::invalid_argument(
        "FE-Hybrid AMM mixed surrogate portfolio cannot be combined with FE-Hybrid RQN memory options");
  }
  if (options.fullEquivHybridLocalChainPreconditioner) {
    if (options.fullEquivHybridBackend !=
        ManualDpgoMmFullEquivHybridBackend::PcgFull) {
      throw std::invalid_argument(
          "fullEquivHybridLocalChainPreconditioner requires PCG_FULL");
    }
    if (options.fullEquivHybridLinearTranslationSchurPreconditioner ||
        options.fullEquivHybridReducedRotationPreconditioner) {
      throw std::invalid_argument(
          "fullEquivHybridLocalChainPreconditioner requires the sparse PCG_FULL path and cannot be combined with dense-only FE-Hybrid preconditioners");
    }
  }
  if (!std::isfinite(options.fullEquivHybridRqnMinCurvatureRatio) ||
      options.fullEquivHybridRqnMinCurvatureRatio < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridRqnMinCurvatureRatio must be finite and nonnegative");
  }
  if (!std::isfinite(options.fullEquivHybridWarmStartMaxNorm) ||
      options.fullEquivHybridWarmStartMaxNorm < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridWarmStartMaxNorm must be finite and nonnegative");
  }
  if (!std::isfinite(options.fullEquivHybridSchwarzMaxBlockNorm) ||
      options.fullEquivHybridSchwarzMaxBlockNorm < 0.0) {
    throw std::invalid_argument(
        "fullEquivHybridSchwarzMaxBlockNorm must be finite and nonnegative");
  }
  if (!std::isfinite(options.localCandidateCostTieTolerance) ||
      options.localCandidateCostTieTolerance < 0.0) {
    throw std::invalid_argument(
        "localCandidateCostTieTolerance must be finite and nonnegative");
  }
  if (options.globalStateExtrapolation &&
      !std::isfinite(options.globalStateExtrapolationGamma)) {
    throw std::invalid_argument(
        "globalStateExtrapolationGamma must be finite");
  }
  for (const double gamma : options.globalStateExtrapolationGammas) {
    if (!std::isfinite(gamma)) {
      throw std::invalid_argument(
          "globalStateExtrapolationGammas must contain finite values");
    }
  }
  if (options.globalAndersonAcceleration &&
      (!std::isfinite(options.globalAndersonMaxAlpha) ||
       options.globalAndersonMaxAlpha < 0.0)) {
    throw std::invalid_argument(
        "globalAndersonMaxAlpha must be finite and nonnegative");
  }
  if (options.localGradientCorrection &&
      (!std::isfinite(options.localGradientCorrectionStep) ||
       options.localGradientCorrectionStep < 0.0)) {
    throw std::invalid_argument(
        "localGradientCorrectionStep must be finite and nonnegative");
  }
  if (options.localGradientCorrectionInnerRounds == 0) {
    throw std::invalid_argument(
        "localGradientCorrectionInnerRounds must be positive");
  }
  if (!std::isfinite(
          options.localGradientCorrectionCoupledDirectionBudgetFraction) ||
      options.localGradientCorrectionCoupledDirectionBudgetFraction < 0.0 ||
      options.localGradientCorrectionCoupledDirectionBudgetFraction > 1.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCoupledDirectionBudgetFraction must be finite in [0, 1]");
  }
  if (!std::isfinite(
          options.localGradientCorrectionCoupledDirectionMinScore) ||
      options.localGradientCorrectionCoupledDirectionMinScore < 0.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCoupledDirectionMinScore must be finite and nonnegative");
  }
  if (!std::isfinite(
          options.localGradientCorrectionCoupledDirectionMinScoreRatio) ||
      options.localGradientCorrectionCoupledDirectionMinScoreRatio < 0.0 ||
      options.localGradientCorrectionCoupledDirectionMinScoreRatio > 1.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCoupledDirectionMinScoreRatio must be finite in [0, 1]");
  }
  if (!std::isfinite(
          options.localGradientCorrectionCoupledDirectionByteBudgetMb) ||
      options.localGradientCorrectionCoupledDirectionByteBudgetMb < 0.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCoupledDirectionByteBudgetMb must be finite and nonnegative");
  }
  if (options.postExchangePeriod == 0 &&
      !options.localGradientCorrectionFreshNeighborExchange) {
    throw std::invalid_argument(
        "postExchangePeriod=0 requires a fresh-neighbor exchange path");
  }
  if (!std::isfinite(options.localGradientCorrectionFreshMinPoseDelta) ||
      options.localGradientCorrectionFreshMinPoseDelta < 0.0) {
    throw std::invalid_argument(
        "localGradientCorrectionFreshMinPoseDelta must be finite and nonnegative");
  }
  if (!std::isfinite(options.localGradientCorrectionFreshBudgetFraction) ||
      options.localGradientCorrectionFreshBudgetFraction < 0.0 ||
      options.localGradientCorrectionFreshBudgetFraction > 1.0) {
    throw std::invalid_argument(
        "localGradientCorrectionFreshBudgetFraction must be finite in [0, 1]");
  }
  if (!std::isfinite(options.postExchangeMinPoseDelta) ||
      options.postExchangeMinPoseDelta < 0.0) {
    throw std::invalid_argument(
        "postExchangeMinPoseDelta must be finite and nonnegative");
  }
  if (!std::isfinite(options.postExchangeBudgetFraction) ||
      options.postExchangeBudgetFraction < 0.0 ||
      options.postExchangeBudgetFraction > 1.0) {
    throw std::invalid_argument(
        "postExchangeBudgetFraction must be finite in [0, 1]");
  }
  for (const double step : options.localGradientCorrectionSteps) {
    if (!std::isfinite(step) || step < 0.0) {
      throw std::invalid_argument(
          "localGradientCorrectionSteps must contain finite nonnegative values");
    }
  }
  if (!std::isfinite(options.localGradientCorrectionCompactSchurDamping) ||
      options.localGradientCorrectionCompactSchurDamping < 0.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCompactSchurDamping must be finite and nonnegative");
  }
  if (!std::isfinite(
          options.localGradientCorrectionCompactSchurMinCostDecrease) ||
      options.localGradientCorrectionCompactSchurMinCostDecrease < 0.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCompactSchurMinCostDecrease must be finite and nonnegative");
  }
  if (options.localGradientCorrectionCompactSchurGradientGuard &&
      !options.localGradientCorrectionCompactSchur &&
      !options.fullEquivHybridActiveSeparatorCompactSchur) {
    throw std::invalid_argument(
        "localGradientCorrectionCompactSchurGradientGuard requires a compact-Schur correction path");
  }
  if (!std::isfinite(
          options.localGradientCorrectionCompactSchurMaxGradientIncreaseRatio) ||
      options.localGradientCorrectionCompactSchurMaxGradientIncreaseRatio <
          0.0) {
    throw std::invalid_argument(
        "localGradientCorrectionCompactSchurMaxGradientIncreaseRatio must be finite and nonnegative");
  }
  if (options.globalGradientCorrection &&
      (!std::isfinite(options.globalGradientCorrectionStep) ||
       options.globalGradientCorrectionStep < 0.0)) {
    throw std::invalid_argument(
        "globalGradientCorrectionStep must be finite and nonnegative");
  }
  for (const double step : options.globalGradientCorrectionSteps) {
    if (!std::isfinite(step) || step < 0.0) {
      throw std::invalid_argument(
          "globalGradientCorrectionSteps must contain finite nonnegative values");
    }
  }
  if (!std::isfinite(options.ammEta0) || options.ammEta0 < 0.0 ||
      options.ammEta0 > 1.0 || !std::isfinite(options.ammEta1) ||
      options.ammEta1 < 0.0 || options.ammEta1 > 1.0) {
    throw std::invalid_argument("AMM eta values must be finite in [0, 1]");
  }
  if (!std::isfinite(options.ammPsi) || options.ammPsi < 0.0 ||
      !std::isfinite(options.ammPhi) || options.ammPhi < 0.0) {
    throw std::invalid_argument("AMM psi/phi values must be finite and nonnegative");
  }
  if (!std::isfinite(options.ammGammaScale) ||
      options.ammGammaScale < 0.0) {
    throw std::invalid_argument(
        "AMM gamma scale must be finite and nonnegative");
  }
  if (options.ammDpgoMixedSurrogatePortfolio &&
      !options.ammDpgoSurrogateParity) {
    throw std::invalid_argument(
        "AMM mixed surrogate portfolio requires DPGO surrogate parity");
  }
  if (options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak < 0) {
    throw std::invalid_argument(
        "AMM mixed surrogate skip streak must be nonnegative");
  }
  if (options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak > 0 &&
      !options.ammDpgoMixedSurrogatePortfolio) {
    throw std::invalid_argument(
        "AMM mixed surrogate skip streak requires mixed surrogate portfolio");
  }
  if (options.ammMixedSurrogateForceSimpleEverySkippedRounds < 0) {
    throw std::invalid_argument(
        "AMM mixed surrogate forced refresh period must be nonnegative");
  }
  if (options.ammMixedSurrogateForceSimpleEverySkippedRounds > 0 &&
      options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak <= 0) {
    throw std::invalid_argument(
        "AMM mixed surrogate forced refresh requires simple-skip streak");
  }
  for (const double gammaScale : options.ammGammaScales) {
    if (!std::isfinite(gammaScale) || gammaScale < 0.0) {
      throw std::invalid_argument(
          "AMM gamma scales must contain finite nonnegative values");
    }
  }
  if (options.ammCooldownAfterRejected < 0) {
    throw std::invalid_argument("AMM cooldown must be nonnegative");
  }
  if (options.ammMaxSoftRestartHits0 < 0 ||
      options.ammMaxSoftRestartHits1 < 0) {
    throw std::invalid_argument("AMM soft restart hit limits must be nonnegative");
  }
  if (!std::isfinite(options.ammLocalMeritCostTieTolerance) ||
      options.ammLocalMeritCostTieTolerance < 0.0) {
    throw std::invalid_argument(
        "AMM local merit cost tie tolerance must be finite and nonnegative");
  }
}

ManualDpgoMmPreparedProblem prepareManualDpgoMmProblem(
    const std::string &datasetPath, const ManualDpgoMmOptions &options,
    bool buildCentralProblem) {
  validateManualDpgoMmOptions(options);

  ManualDpgoMmPreparedProblem prepared;
  size_t numPosesSize = 0;
  prepared.dataset = read_g2o_file(datasetPath, numPosesSize);
  if (prepared.dataset.empty()) {
    throw std::runtime_error("Dataset has no supported measurements: " +
                             datasetPath);
  }
  prepared.numPoses = static_cast<unsigned>(numPosesSize);
  prepared.d = static_cast<unsigned>(prepared.dataset.front().t.rows());
  prepared.r =
      options.relaxationRank > 0 ? options.relaxationRank : prepared.d;
  if (prepared.r != prepared.d) {
    throw std::invalid_argument(
        "manual_dpgo_mm reduced-rotation comparison currently requires r=d");
  }

  prepared.posesPerRobot = prepared.numPoses / options.numRobots;
  if (prepared.posesPerRobot == 0) {
    throw std::invalid_argument("More robots than poses");
  }

  if (buildCentralProblem) {
    SparseMatrix QCentral = constructConnectionLaplacianSE(prepared.dataset);
    prepared.centralProblem = std::make_unique<QuadraticProblem>(
        prepared.numPoses, prepared.d, prepared.r);
    prepared.centralProblem->setQ(QCentral);
  }

  std::map<unsigned, PoseID> poseMap;
  for (unsigned robot = 0; robot < options.numRobots; ++robot) {
    const unsigned startIdx = robot * prepared.posesPerRobot;
    unsigned endIdx = (robot + 1) * prepared.posesPerRobot;
    if (robot + 1 == options.numRobots) {
      endIdx = prepared.numPoses;
    }
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      poseMap[idx] = std::make_pair(robot, idx - startIdx);
    }
  }

  std::vector<std::vector<RelativeSEMeasurement>> odometry(options.numRobots);
  std::vector<std::vector<RelativeSEMeasurement>> privateLoops(
      options.numRobots);
  std::vector<std::vector<RelativeSEMeasurement>> sharedLoops(
      options.numRobots);
  std::vector<double> robotMeasurementDegrees(options.numRobots, 0.0);
  for (const auto &mIn : prepared.dataset) {
    const PoseID src = poseMap.at(mIn.p1);
    const PoseID dst = poseMap.at(mIn.p2);
    robotMeasurementDegrees[src.first] += 1.0;
    robotMeasurementDegrees[dst.first] += 1.0;
    RelativeSEMeasurement m(src.first, dst.first, src.second, dst.second,
                            mIn.R, mIn.t, mIn.kappa, mIn.tau);
    if (src.first == dst.first) {
      if (src.second + 1 == dst.second) {
        odometry[src.first].push_back(m);
      } else {
        privateLoops[src.first].push_back(m);
      }
    } else {
      sharedLoops[src.first].push_back(m);
      sharedLoops[dst.first].push_back(m);
    }
  }

  prepared.agents.reserve(options.numRobots);
  for (unsigned robot = 0; robot < options.numRobots; ++robot) {
    const unsigned startIdx = robot * prepared.posesPerRobot;
    unsigned endIdx = (robot + 1) * prepared.posesPerRobot;
    if (robot + 1 == options.numRobots) {
      endIdx = prepared.numPoses;
    }
    auto agent = std::make_unique<ManualDpgoMmAgent>(
        robot, prepared.d, prepared.r, endIdx - startIdx, odometry[robot],
        privateLoops[robot], sharedLoops[robot], robotMeasurementDegrees,
        options);
    prepared.agents.push_back(std::move(agent));
  }

  if (options.centralizedChordalInit) {
    const Matrix TChordal =
        chordalInitialization(prepared.d, prepared.numPoses, prepared.dataset);
    const Matrix XChordal = fixedStiefelVariable(prepared.d, prepared.r) *
                            TChordal;
    for (unsigned robot = 0; robot < options.numRobots; ++robot) {
      const unsigned startIdx = robot * prepared.posesPerRobot;
      unsigned endIdx = (robot + 1) * prepared.posesPerRobot;
      if (robot + 1 == options.numRobots) {
        endIdx = prepared.numPoses;
      }
      prepared.agents[robot]->setX(XChordal.block(
          0, startIdx * (prepared.d + 1), prepared.r,
          (endIdx - startIdx) * (prepared.d + 1)));
    }
  } else {
    std::vector<Matrix> localPoseInitializations(options.numRobots);
    for (unsigned robot = 0; robot < options.numRobots; ++robot) {
      const unsigned startIdx = robot * prepared.posesPerRobot;
      unsigned endIdx = (robot + 1) * prepared.posesPerRobot;
      if (robot + 1 == options.numRobots) {
        endIdx = prepared.numPoses;
      }
      localPoseInitializations[robot] = localPoseChordalInitialization(
          prepared.d, endIdx - startIdx, odometry[robot],
          privateLoops[robot]);
    }
    alignDistributedLocalChordalGauges(localPoseInitializations, sharedLoops,
                                       prepared.d);
    const Matrix lift = fixedStiefelVariable(prepared.d, prepared.r);
    for (unsigned robot = 0; robot < options.numRobots; ++robot) {
      prepared.agents[robot]->setX(lift * localPoseInitializations[robot]);
    }
  }

  return prepared;
}

Matrix assembleGlobalEstimate(
    const std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    unsigned numPoses, unsigned numRobots, unsigned posesPerRobot,
    unsigned d, unsigned r) {
  Matrix X = Matrix::Zero(r, numPoses * (d + 1));
  for (unsigned robot = 0; robot < numRobots; ++robot) {
    const Matrix &localX = agents[robot]->getX();
    const unsigned startIdx = robot * posesPerRobot;
    unsigned endIdx = (robot + 1) * posesPerRobot;
    if (robot + 1 == numRobots) {
      endIdx = numPoses;
    }
    X.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)) =
        localX.block(0, 0, r, (endIdx - startIdx) * (d + 1));
  }
  return X;
}

void scatterGlobalEstimateToAgents(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents, const Matrix &X,
    unsigned numPoses, unsigned numRobots, unsigned posesPerRobot, unsigned d,
    unsigned r) {
  if (X.rows() != static_cast<int>(r) ||
      X.cols() != static_cast<int>(numPoses * (d + 1))) {
    throw std::invalid_argument("global estimate has wrong shape");
  }
  for (unsigned robot = 0; robot < numRobots; ++robot) {
    const unsigned startIdx = robot * posesPerRobot;
    unsigned endIdx = (robot + 1) * posesPerRobot;
    if (robot + 1 == numRobots) {
      endIdx = numPoses;
    }
    agents[robot]->setX(X.block(0, startIdx * (d + 1), r,
                                (endIdx - startIdx) * (d + 1)));
  }
}

void resetLocalHistories(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  for (auto &agent : agents) {
    agent->resetLocalHistory();
  }
}

std::size_t syncAmmReferences(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  std::size_t synced = 0;
  for (auto &agent : agents) {
    if (agent->syncAmmReferenceToCurrentState()) {
      ++synced;
    }
  }
  return synced;
}

Matrix projectGlobalRotationBlocks(Matrix value, unsigned d) {
  const unsigned blockDim = d + 1;
  if (blockDim == 0 || value.cols() % blockDim != 0) {
    return value;
  }
  const unsigned numPoses = static_cast<unsigned>(value.cols() / blockDim);
  for (unsigned pose = 0; pose < numPoses; ++pose) {
    const unsigned colStart = pose * blockDim;
    value.block(0, colStart, value.rows(), d) =
        projectToStiefelManifold(value.block(0, colStart, value.rows(), d));
  }
  return value;
}

using DirectedPoseKey = std::tuple<unsigned, unsigned, unsigned>;
using DirectedPoseCache = std::map<DirectedPoseKey, Matrix>;
using CommunicationEdge = std::pair<unsigned, unsigned>;
using CommunicationRound = std::set<CommunicationEdge>;
using CommunicationSchedule = std::vector<CommunicationRound>;

struct PoseExchangeAggregate {
  std::size_t sent{0};
  std::size_t skipped{0};
  double commMb{0.0};
};

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

CommunicationEdge normalizedCommunicationEdge(unsigned robotA,
                                               unsigned robotB) {
  if (robotA > robotB) {
    std::swap(robotA, robotB);
  }
  return CommunicationEdge(robotA, robotB);
}

bool communicationRoundAllows(const CommunicationRound *round,
                              unsigned robotA, unsigned robotB) {
  if (round == nullptr) {
    return true;
  }
  return round->count(normalizedCommunicationEdge(robotA, robotB)) > 0;
}

unsigned communicationRoundHopCount(const CommunicationRound *round,
                                    unsigned robotA, unsigned robotB,
                                    unsigned maxRelayHops,
                                    unsigned numRobots) {
  if (robotA == robotB) {
    return 0;
  }
  if (round == nullptr) {
    return 1;
  }
  if (communicationRoundAllows(round, robotA, robotB)) {
    return 1;
  }
  if (maxRelayHops == 0 || numRobots == 0 ||
      robotA >= numRobots || robotB >= numRobots) {
    return 0;
  }

  std::vector<std::vector<unsigned>> adjacency(numRobots);
  for (const auto &edge : *round) {
    if (edge.first >= numRobots || edge.second >= numRobots) {
      continue;
    }
    adjacency[edge.first].push_back(edge.second);
    adjacency[edge.second].push_back(edge.first);
  }

  std::vector<unsigned> distance(
      numRobots, std::numeric_limits<unsigned>::max());
  std::queue<unsigned> frontier;
  distance[robotA] = 0;
  frontier.push(robotA);
  while (!frontier.empty()) {
    const unsigned node = frontier.front();
    frontier.pop();
    if (distance[node] >= maxRelayHops) {
      continue;
    }
    for (const unsigned next : adjacency[node]) {
      if (distance[next] != std::numeric_limits<unsigned>::max()) {
        continue;
      }
      distance[next] = distance[node] + 1;
      if (next == robotB) {
        return distance[next] <= maxRelayHops ? distance[next] : 0;
      }
      frontier.push(next);
    }
  }
  return 0;
}

CommunicationSchedule readCommunicationSchedule(const std::string &path,
                                                unsigned numRobots) {
  CommunicationSchedule schedule;
  if (path.empty()) {
    return schedule;
  }

  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open communication topology file: " +
                             path);
  }

  std::string line;
  unsigned lineNumber = 0;
  bool sawData = false;
  while (std::getline(input, line)) {
    ++lineNumber;
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = splitCsvLine(line);
    if (fields.empty()) {
      continue;
    }
    if (lineNumber == 1 && fields[0] == "round") {
      continue;
    }
    if (fields.size() < 3) {
      throw std::runtime_error(
          "Communication topology row needs round,src,dst fields at line " +
          std::to_string(lineNumber));
    }

    const unsigned round =
        static_cast<unsigned>(std::stoul(fields[0]));
    const unsigned src = static_cast<unsigned>(std::stoul(fields[1]));
    const unsigned dst = static_cast<unsigned>(std::stoul(fields[2]));
    if (src >= numRobots || dst >= numRobots || src == dst) {
      throw std::runtime_error(
          "Invalid communication topology edge at line " +
          std::to_string(lineNumber));
    }
    if (fields.size() >= 4) {
      const double weight = std::stod(fields[3]);
      if (!std::isfinite(weight) || weight < 0.0) {
        throw std::runtime_error(
            "Invalid communication topology weight at line " +
            std::to_string(lineNumber));
      }
    }
    if (schedule.size() <= round) {
      schedule.resize(round + 1);
    }
    schedule[round].insert(normalizedCommunicationEdge(src, dst));
    sawData = true;
  }

  if (!sawData) {
    throw std::runtime_error("Communication topology file has no edges: " +
                             path);
  }
  for (unsigned round = 0; round < schedule.size(); ++round) {
    if (schedule[round].empty()) {
      throw std::runtime_error(
          "Communication topology has an empty/non-contiguous round: " +
          std::to_string(round));
    }
  }
  return schedule;
}

const CommunicationRound *communicationRoundFor(
    const CommunicationSchedule &schedule, unsigned round) {
  if (schedule.empty()) {
    return nullptr;
  }
  return &schedule[round % schedule.size()];
}

PoseExchangeAggregate exchangeOwnedSeparatorPosesInternal(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    DirectedPoseCache *deliveredPoses, bool thresholdPostExchange,
    double minPoseDelta, ManualDpgoMmPostExchangeDeltaMode deltaMode,
    double budgetFraction = 1.0,
    unsigned maxPosesPerReceiver = 0,
    const CommunicationRound *allowedEdges = nullptr,
    unsigned maxRelayHops = 0,
    double hopBudgetFraction = 1.0,
    bool staleAwareRelayScore = false,
    double staleAwareRelayAgeGain = 1.0,
    bool deltaCompression = false,
    ManualDpgoMmCommunicationDeltaMode deltaCompressionMode =
        ManualDpgoMmCommunicationDeltaMode::Sparse,
    unsigned deltaTopK = 4,
    double deltaMaxReconstructionError = 5e-4,
    bool liftedDeltaCompression = false,
    unsigned liftedDeltaRank = 2,
    double liftedDeltaMaxReconstructionError = 5e-4,
    bool valueScheduler = false,
    double valueByteBudgetMb = 0.0,
    double valueMinScoreRatio = 0.0,
    bool valueSchedulerUseResidualProxy = false) {
  struct Candidate {
    unsigned senderID{0};
    unsigned poseIndex{0};
    Matrix pose;
    double score{0.0};
    double valueScore{0.0};
    std::size_t payloadBytes{0};
    std::size_t order{0};
    unsigned hopCount{1};
  };

  PoseExchangeAggregate aggregate;
  const unsigned numRobots = static_cast<unsigned>(agents.size());
  const bool useCountBudget =
      thresholdPostExchange &&
      (budgetFraction < 1.0 || maxPosesPerReceiver > 0);
  const bool useHopBudget =
      thresholdPostExchange && hopBudgetFraction < 1.0;
  const bool useValueByteBudget =
      thresholdPostExchange && valueScheduler && valueByteBudgetMb > 0.0;
  const bool useValueRatio =
      thresholdPostExchange && valueScheduler && valueMinScoreRatio > 0.0;
  const bool useBudget =
      useCountBudget || useHopBudget || useValueByteBudget || useValueRatio;
  const std::size_t valueByteBudgetLimit =
      useValueByteBudget
          ? static_cast<std::size_t>(
                std::floor(valueByteBudgetMb * 1024.0 * 1024.0))
          : 0;
  std::size_t valueByteBudgetUsed = 0;
  for (auto &receiver : agents) {
    const unsigned receiverID = receiver->getID();
    std::vector<Candidate> candidates;
    std::size_t order = 0;
    for (unsigned senderID : receiver->getNeighbors()) {
      const unsigned hopCount = communicationRoundHopCount(
          allowedEdges, receiverID, senderID, maxRelayHops, numRobots);
      if (hopCount == 0) {
        aggregate.skipped +=
            receiver->getNeighborPublicPoses(senderID).size();
        continue;
      }
      for (unsigned poseIndex :
           receiver->getNeighborPublicPoses(senderID)) {
        Matrix pose;
        if (agents[senderID]->getSharedPose(poseIndex, pose)) {
          const DirectedPoseKey key(receiverID, senderID, poseIndex);
          bool sendPose = true;
          double deltaScore = std::numeric_limits<double>::infinity();
          if (thresholdPostExchange && deliveredPoses != nullptr &&
              (minPoseDelta > 0.0 || useBudget)) {
            const auto previous = deliveredPoses->find(key);
            if (previous != deliveredPoses->end()) {
              deltaScore = (pose - previous->second).norm();
              if (deltaMode ==
                  ManualDpgoMmPostExchangeDeltaMode::Weighted) {
                deltaScore *= receiver->getNeighborPoseSensitivity(
                    senderID, poseIndex);
              }
              if (deltaScore <= minPoseDelta) {
                sendPose = false;
              }
              if (sendPose && staleAwareRelayScore &&
                  std::isfinite(deltaScore)) {
                const double staleFactor =
                    1.0 + staleAwareRelayAgeGain *
                              static_cast<double>(
                                  receiver->getNeighborPoseStaleness(
                                      senderID, poseIndex));
                if (std::isfinite(staleFactor) && staleFactor > 0.0) {
                  deltaScore *= staleFactor;
                }
              }
            }
          }
          if (!sendPose) {
            ++aggregate.skipped;
            continue;
          }
          const double poseDeltaNorm =
              std::isfinite(deltaScore) ? deltaScore : 0.0;
          const double staleness = static_cast<double>(
              receiver->getNeighborPoseStaleness(senderID, poseIndex));
          double boundarySignal =
              receiver->getNeighborPoseSensitivity(senderID, poseIndex);
          if (valueSchedulerUseResidualProxy) {
            const double residualProxy =
                receiver->getNeighborPoseResidualProxy(senderID, poseIndex,
                                                       pose);
            if (std::isfinite(residualProxy) && residualProxy > 0.0) {
              boundarySignal = residualProxy;
            }
          }
          const std::size_t payloadBytes = posePayloadBytesForBlock(pose);
          const double valueScore =
              valueScheduler
                  ? valueSchedulerScore(poseDeltaNorm, staleness,
                                        boundarySignal, payloadBytes,
                                        hopCount)
                  : 0.0;
          candidates.push_back(Candidate{senderID, poseIndex, pose,
                                         deltaScore, valueScore,
                                         payloadBytes, order++, hopCount});
        }
      }
    }

    double maxValueScore = 0.0;
    if (useValueRatio) {
      for (const Candidate &candidate : candidates) {
        if (std::isfinite(candidate.valueScore)) {
          maxValueScore = std::max(maxValueScore, candidate.valueScore);
        }
      }
      if (maxValueScore > 0.0) {
        const double minAllowedScore =
            valueMinScoreRatio * maxValueScore;
        auto keepEnd = std::remove_if(
            candidates.begin(), candidates.end(),
            [minAllowedScore](const Candidate &candidate) {
              return !std::isfinite(candidate.valueScore) ||
                     candidate.valueScore < minAllowedScore;
            });
        aggregate.skipped +=
            static_cast<std::size_t>(candidates.end() - keepEnd);
        candidates.erase(keepEnd, candidates.end());
      }
    }

    std::size_t sendLimit = candidates.size();
    if (useCountBudget) {
      if (budgetFraction < 1.0) {
        sendLimit =
            std::min(sendLimit, static_cast<std::size_t>(
                                    std::ceil(budgetFraction *
                                              static_cast<double>(
                                                  candidates.size()))));
      }
      if (maxPosesPerReceiver > 0) {
        sendLimit =
            std::min(sendLimit, static_cast<std::size_t>(maxPosesPerReceiver));
      }
    }

    if (useBudget) {
      std::stable_sort(candidates.begin(), candidates.end(),
                       [valueScheduler, useHopBudget](
                           const Candidate &lhs, const Candidate &rhs) {
                         const double lhsScore =
                             valueScheduler
                                 ? lhs.valueScore
                                 : (useHopBudget
                                        ? lhs.score /
                                              static_cast<double>(
                                                  std::max(1u, lhs.hopCount))
                                        : lhs.score);
                         const double rhsScore =
                             valueScheduler
                                 ? rhs.valueScore
                                 : (useHopBudget
                                        ? rhs.score /
                                              static_cast<double>(
                                                  std::max(1u, rhs.hopCount))
                                        : rhs.score);
                         if (lhsScore == rhsScore) {
                           return lhs.order < rhs.order;
                         }
                         return lhsScore > rhsScore;
                       });
    }

    std::vector<std::size_t> selectedIndices;
    selectedIndices.reserve(sendLimit);
    if (useValueByteBudget) {
      for (std::size_t idx = 0;
           idx < candidates.size() && selectedIndices.size() < sendLimit;
           ++idx) {
        const std::size_t candidateBytes =
            candidates[idx].payloadBytes *
            static_cast<std::size_t>(std::max(1u, candidates[idx].hopCount));
        if (candidateBytes == 0 ||
            valueByteBudgetUsed + candidateBytes > valueByteBudgetLimit) {
          continue;
        }
        selectedIndices.push_back(idx);
        valueByteBudgetUsed += candidateBytes;
      }
    } else if (useHopBudget) {
      std::size_t totalHopCost = 0;
      for (const Candidate &candidate : candidates) {
        totalHopCost += std::max(1u, candidate.hopCount);
      }
      std::size_t hopLimit = static_cast<std::size_t>(
          std::ceil(hopBudgetFraction * static_cast<double>(totalHopCost)));
      if (hopBudgetFraction <= 0.0) {
        hopLimit = 0;
      } else if (hopLimit == 0 && !candidates.empty()) {
        hopLimit = 1;
      }
      std::size_t usedHopCost = 0;
      for (std::size_t idx = 0;
           idx < candidates.size() && selectedIndices.size() < sendLimit;
           ++idx) {
        const std::size_t hopCost =
            static_cast<std::size_t>(std::max(1u, candidates[idx].hopCount));
        if (usedHopCost + hopCost > hopLimit) {
          continue;
        }
        selectedIndices.push_back(idx);
        usedHopCost += hopCost;
      }
    } else {
      for (std::size_t idx = 0; idx < sendLimit; ++idx) {
        selectedIndices.push_back(idx);
      }
    }
    if (selectedIndices.size() < candidates.size()) {
      aggregate.skipped += candidates.size() - selectedIndices.size();
    }

    PoseDict poses;
    std::size_t selectedHopCost = 0;
    for (const std::size_t idx : selectedIndices) {
      const Candidate &candidate = candidates[idx];
      const DirectedPoseKey key(receiverID, candidate.senderID,
                                candidate.poseIndex);
      Matrix deliveredPose = candidate.pose;
      std::size_t payloadBytes = candidate.payloadBytes;
      if (deltaCompression && deliveredPoses != nullptr) {
        const auto previous = deliveredPoses->find(key);
        if (previous != deliveredPoses->end()) {
          Matrix reconstructed;
          std::size_t deltaPayloadBytes = 0;
          bool compressed = false;
          if (deltaCompression &&
              (deltaCompressionMode ==
                   ManualDpgoMmCommunicationDeltaMode::Tangent ||
               deltaCompressionMode ==
                   ManualDpgoMmCommunicationDeltaMode::Hybrid)) {
            compressed = reconstructTangentDeltaPose(
                previous->second, candidate.pose,
                deltaMaxReconstructionError, reconstructed,
                deltaPayloadBytes);
          }
          if (!compressed && deltaCompression &&
              (deltaCompressionMode ==
                   ManualDpgoMmCommunicationDeltaMode::Sparse ||
               deltaCompressionMode ==
                   ManualDpgoMmCommunicationDeltaMode::Hybrid)) {
            compressed = reconstructSparseDeltaPose(
                previous->second, candidate.pose, deltaTopK,
                deltaMaxReconstructionError, reconstructed,
                deltaPayloadBytes);
          }
          if (!compressed && liftedDeltaCompression) {
            compressed = reconstructLiftedDeltaPose(
                previous->second, candidate.pose, liftedDeltaRank,
                liftedDeltaMaxReconstructionError, reconstructed,
                deltaPayloadBytes);
          }
          if (compressed && deltaPayloadBytes < payloadBytes) {
            deliveredPose = std::move(reconstructed);
            payloadBytes = deltaPayloadBytes;
          }
        }
      }
      poses[std::make_pair(candidate.senderID, candidate.poseIndex)] =
          deliveredPose;
      if (deliveredPoses != nullptr) {
        (*deliveredPoses)[key] = deliveredPose;
      }
      selectedHopCost += candidate.hopCount;
      aggregate.commMb += bytesToMegabytes(
          payloadBytes * static_cast<std::size_t>(candidate.hopCount));
    }
    if (!poses.empty()) {
      receiver->updateNeighborPoses(poses);
      aggregate.sent += selectedHopCost;
    }
  }
  return aggregate;
}

PoseExchangeAggregate exchangeOwnedSeparatorPoses(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    DirectedPoseCache &deliveredPoses,
    const CommunicationRound *allowedEdges = nullptr,
    unsigned maxRelayHops = 0,
    double hopBudgetFraction = 1.0,
    bool staleAwareRelayScore = false,
    double staleAwareRelayAgeGain = 1.0,
    bool deltaCompression = false,
    ManualDpgoMmCommunicationDeltaMode deltaCompressionMode =
        ManualDpgoMmCommunicationDeltaMode::Sparse,
    unsigned deltaTopK = 4,
    double deltaMaxReconstructionError = 5e-4,
    bool liftedDeltaCompression = false,
    unsigned liftedDeltaRank = 2,
    double liftedDeltaMaxReconstructionError = 5e-4,
    bool valueScheduler = false,
    double valueByteBudgetMb = 0.0,
    double valueMinScoreRatio = 0.0,
    bool valueSchedulerUseResidualProxy = false) {
  return exchangeOwnedSeparatorPosesInternal(agents, &deliveredPoses, false,
                                             0.0,
                                             ManualDpgoMmPostExchangeDeltaMode::
                                                 Absolute,
                                             1.0, 0, allowedEdges,
                                             maxRelayHops,
                                             hopBudgetFraction,
                                             staleAwareRelayScore,
                                             staleAwareRelayAgeGain,
                                             deltaCompression,
                                             deltaCompressionMode, deltaTopK,
                                             deltaMaxReconstructionError,
                                             liftedDeltaCompression,
                                             liftedDeltaRank,
                                             liftedDeltaMaxReconstructionError,
                                             valueScheduler,
                                             valueByteBudgetMb,
                                             valueMinScoreRatio,
                                             valueSchedulerUseResidualProxy);
}

PoseExchangeAggregate exchangeChangedOwnedSeparatorPoses(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    DirectedPoseCache &deliveredPoses, double minPoseDelta,
    ManualDpgoMmPostExchangeDeltaMode deltaMode, double budgetFraction,
    unsigned maxPosesPerReceiver,
    const CommunicationRound *allowedEdges = nullptr,
    unsigned maxRelayHops = 0,
    double hopBudgetFraction = 1.0,
    bool staleAwareRelayScore = false,
    double staleAwareRelayAgeGain = 1.0,
    bool deltaCompression = false,
    ManualDpgoMmCommunicationDeltaMode deltaCompressionMode =
        ManualDpgoMmCommunicationDeltaMode::Sparse,
    unsigned deltaTopK = 4,
    double deltaMaxReconstructionError = 5e-4,
    bool liftedDeltaCompression = false,
    unsigned liftedDeltaRank = 2,
    double liftedDeltaMaxReconstructionError = 5e-4,
    bool valueScheduler = false,
    double valueByteBudgetMb = 0.0,
    double valueMinScoreRatio = 0.0,
    bool valueSchedulerUseResidualProxy = false) {
  return exchangeOwnedSeparatorPosesInternal(agents, &deliveredPoses, true,
                                             minPoseDelta, deltaMode,
                                             budgetFraction,
                                             maxPosesPerReceiver,
                                             allowedEdges, maxRelayHops,
                                             hopBudgetFraction,
                                             staleAwareRelayScore,
                                             staleAwareRelayAgeGain,
                                             deltaCompression,
                                             deltaCompressionMode, deltaTopK,
                                             deltaMaxReconstructionError,
                                             liftedDeltaCompression,
                                             liftedDeltaRank,
                                             liftedDeltaMaxReconstructionError,
                                             valueScheduler,
                                             valueByteBudgetMb,
                                             valueMinScoreRatio,
                                             valueSchedulerUseResidualProxy);
}

struct LocalModelAggregate {
  std::size_t failures{0};
  double cost{std::numeric_limits<double>::quiet_NaN()};
  double gradient{std::numeric_limits<double>::quiet_NaN()};
};

struct LocalOptimizationAggregate {
  std::size_t failures{0};
  std::size_t acceptedIterations{0};
  std::size_t adaptiveRefinements{0};
  std::size_t extrapolationAccepted{0};
  std::size_t extrapolationRejected{0};
  std::size_t gExtrapolationAccepted{0};
  std::size_t gExtrapolationRejected{0};
  std::size_t coupledExtrapolationAccepted{0};
  std::size_t coupledExtrapolationRejected{0};
  std::size_t andersonAccepted{0};
  std::size_t andersonRejected{0};
  std::size_t squaremAccepted{0};
  std::size_t squaremRejected{0};
  std::size_t fullEquivHybridWarmStartCandidateCount{0};
  std::size_t fullEquivHybridWarmStartAcceptedCount{0};
  std::size_t fullEquivHybridWarmStartGuardRejectedCount{0};
  std::size_t fullEquivHybridSchurStepCandidateCount{0};
  std::size_t fullEquivHybridSchurStepAcceptedCount{0};
  std::size_t fullEquivHybridSchurStepGuardRejectedCount{0};
  std::size_t fullEquivHybridLocalPortfolioCandidateCount{0};
  std::size_t fullEquivHybridLocalPortfolioSelectedUnsmoothedCount{0};
  std::size_t fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount{0};
  std::size_t fullEquivHybridLocalPortfolioSelectedSchwarzFehCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingSweepCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingCandidateCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingAcceptedCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingRejectedCount{0};
  double fullEquivHybridSchwarzSmoothingCostDecrease{0.0};
  double fullEquivHybridSchwarzSmoothingTimeSec{0.0};
  std::size_t fullEquivHybridLinearPcgIterationCount{0};
  double fullEquivHybridLinearInitialResidual{0.0};
  double fullEquivHybridLinearFinalResidual{0.0};
  double fullEquivHybridLinearFullResidual{0.0};
  double fullEquivHybridLinearSolveTimeSec{0.0};
  std::size_t fullEquivHybridStepTrialCount{0};
  std::size_t fullEquivHybridStepTrialAcceptedCount{0};
  std::size_t fullEquivHybridStepTrialRejectedCount{0};
  double fullEquivHybridStepPredictedDecreaseSum{0.0};
  double fullEquivHybridStepActualDecreaseSum{0.0};
  double fullEquivHybridStepRhoSum{0.0};
  std::size_t fullEquivHybridStepRhoCount{0};
  double fullEquivHybridStepAcceptedScaleSum{0.0};
  double fullEquivHybridStepAcceptedPredictedDecreaseSum{0.0};
  double fullEquivHybridStepAcceptedActualDecreaseSum{0.0};
  double fullEquivHybridStepAcceptedRhoSum{0.0};
  std::size_t fullEquivHybridStepAcceptedRhoCount{0};
  std::size_t fullEquivHybridStepParetoCandidateCount{0};
  std::size_t fullEquivHybridStepParetoSelectedCount{0};
  double fullEquivHybridStepParetoSelectedScaleSum{0.0};
  double fullEquivHybridStepParetoBestDecreaseSum{0.0};
  double fullEquivHybridStepParetoSelectedDecreaseSum{0.0};
  double fullEquivHybridStepParetoSelectedGradientSum{0.0};
  std::size_t fullEquivHybridStepParetoGradientEvalCount{0};
  double fullEquivHybridStepParetoGradientEvalTimeSec{0.0};
  std::size_t fullEquivHybridTranslationRecoveryStepTrialCandidateCount{0};
  std::size_t fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount{0};
  std::size_t fullEquivHybridTranslationRecoveryStepTrialSelectedCount{0};
  std::size_t fullEquivHybridActiveSeparatorCandidateCount{0};
  std::size_t fullEquivHybridActiveSeparatorAcceptedCount{0};
  std::size_t fullEquivHybridActiveSeparatorRejectedCount{0};
  double fullEquivHybridActiveSeparatorStepSum{0.0};
  double fullEquivHybridActiveSeparatorCostDecreaseSum{0.0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurCandidateCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurAcceptedCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount{0};
  std::size_t
      fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurSolveFailureCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurFallbackCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurBoundaryColCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurPrivateColCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount{0};
  double fullEquivHybridActiveSeparatorLmSchurAlphaSum{0.0};
  double fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum{0.0};
  double fullEquivHybridActiveSeparatorLmSchurGradientChangeSum{0.0};
  std::size_t fullEquivHybridTranslationRecoveryPolishAttemptCount{0};
  std::size_t fullEquivHybridTranslationRecoveryPolishAcceptedCount{0};
  std::size_t fullEquivHybridTranslationRecoveryPolishRejectedCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount{0};
  std::size_t fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount{
      0};
  std::size_t fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount{
      0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount{0};
  double fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum{0.0};
  std::size_t fullEquivHybridTranslationRecoveryPolishMeritCandidateCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount{0};
  double fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum{0.0};
  double fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum{0.0};
  double fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum{
      0.0};
  double fullEquivHybridTranslationRecoveryPolishCostDecreaseSum{0.0};
  double fullEquivHybridTranslationRecoveryPolishGradientChangeSum{0.0};
  LocalGradientCorrectionDiagnostics localGradientDiagnostics;
  std::size_t fullEquivHybridSparseMatrixVectorProductCount{0};
  std::size_t
      fullEquivHybridReducedRotationInitialGuessCandidateCount{0};
  std::size_t fullEquivHybridReducedRotationInitialGuessUsedCount{0};
  std::size_t
      fullEquivHybridReducedRotationInitialGuessRejectedCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryInitialGuessCandidateCount{0};
  std::size_t fullEquivHybridTranslationRecoveryInitialGuessUsedCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryInitialGuessRejectedCount{0};
  std::size_t
      fullEquivHybridTranslationSchurPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridTranslationSchurPreconditionerFactorizationCount{0};
  std::size_t
      fullEquivHybridTranslationSchurPreconditionerFallbackCount{0};
  std::size_t fullEquivHybridLocalChainPreconditionerApplicationCount{0};
  std::size_t fullEquivHybridLocalChainPreconditionerFactorizationCount{0};
  std::size_t fullEquivHybridLocalChainPreconditionerFallbackCount{0};
  std::size_t
      fullEquivHybridTranslationBlockPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridTranslationBlockPreconditionerFactorizationCount{0};
  std::size_t fullEquivHybridTranslationBlockPreconditionerFallbackCount{0};
  std::size_t
      fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount{0};
  std::size_t
      fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount{0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount{0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount{0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount{0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount{0};
  std::size_t
      fullEquivHybridLaplacianDeflationPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount{0};
  std::size_t
      fullEquivHybridLaplacianDeflationPreconditionerFallbackCount{0};
  std::size_t
      fullEquivHybridLaplacianDeflationPreconditionerBasisDimension{0};
  std::size_t
      fullEquivHybridReducedRotationPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridReducedRotationPreconditionerFactorizationCount{0};
  std::size_t fullEquivHybridRqnUsedCount{0};
  std::size_t fullEquivHybridRqnAcceptedPairCount{0};
  std::size_t fullEquivHybridRqnRejectedPairCount{0};
  std::size_t fullEquivHybridRqnMemorySize{0};
  std::size_t fullEquivHybridRqnPreconditionerApplicationCount{0};
  std::size_t ammAcceleratedAccepted{0};
  std::size_t ammRestart{0};
  std::size_t ammHardRestart{0};
  std::size_t ammSoftRestart{0};
  std::size_t ammPhiFallback{0};
  std::size_t ammLocalMeritRejected{0};
  std::size_t ammProximalStart{0};
  std::size_t ammSkipped{0};
  std::size_t ammMixedSurrogateCandidateCount{0};
  std::size_t ammMixedSurrogateTrueLocalAcceptedCount{0};
  std::size_t ammMixedSurrogateSimpleSelectedCount{0};
  std::size_t ammMixedSurrogateTrueLocalSelectedCount{0};
  std::size_t ammMixedSurrogateExtrapolatedSelectedCount{0};
  std::size_t ammMixedSurrogateOtherSelectedCount{0};
  std::size_t ammMixedSurrogateSimpleSkippedCount{0};
  std::size_t ammMixedSurrogateSimpleForcedRefreshCount{0};
  std::size_t edgeTightQuadraticEvalCount{0};
  double edgeTightQuadraticSurrogateCostSum{0.0};
  double edgeTightQuadraticTrueCostSum{0.0};
  double edgeTightQuadraticMajorizationGapMin{0.0};
  double boundaryEdgeCostBefore{0.0};
  double boundaryEdgeCostAfter{0.0};
  double separatorDeltaNormSquared{0.0};
  std::size_t variableProjectedSchurCandidateCount{0};
  std::size_t variableProjectedSchurAcceptedCount{0};
  std::size_t boundaryProximalCandidateCount{0};
  std::size_t boundaryProximalAcceptedCount{0};
  std::size_t surrogateBoundCheckCount{0};
  std::size_t surrogateBoundViolationCount{0};
  double surrogateBoundMinMargin{0.0};
  std::vector<ManualDpgoMmAmmTraceRow> ammTraceRows;
  std::size_t ammTraceCount{0};
  std::size_t ammRefined{0};
  std::size_t ammProxReset{0};
  std::size_t ammRestartUsedXakh{0};
  std::size_t ammRestartCertificateCount{0};
  std::size_t ammRestartCertificatePassedCount{0};
  std::size_t ammRestartCertificateFailedCount{0};
  double ammRestartCertificateMinMargin{0.0};
  std::size_t ammTranslationRecoveryAttemptCount{0};
  std::size_t ammTranslationRecoveryAcceptedCount{0};
  std::size_t ammTranslationRecoveryRejectedCount{0};
  double ammGammaSum{0.0};
  double ammGkhInitialSum{0.0};
  double ammMinGSum{0.0};
  double ammGkAfterAcceleratedSum{0.0};
  double ammGkhAfterRestartCheckSum{0.0};
  double ammFinalGkSum{0.0};
  double ammPhiLhsSum{0.0};
  double ammPhiRhsSum{0.0};
};

struct LocalGradientCorrectionAggregate {
  std::size_t failures{0};
  std::size_t accepted{0};
  std::size_t rejected{0};
  std::size_t scalarCount{0};
  double scalarCommMb{0.0};
  double selectedStep{0.0};
  LocalGradientCorrectionDiagnostics diagnostics;
};

void resetLocalGradientCorrectionDiagnostics(
    const std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  for (const auto &agent : agents) {
    agent->resetLocalGradientCorrectionDiagnostics();
  }
}

LocalGradientCorrectionDiagnostics collectLocalGradientCorrectionDiagnostics(
    const std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  LocalGradientCorrectionDiagnostics diagnostics;
  for (const auto &agent : agents) {
    diagnostics.add(agent->getLocalGradientCorrectionDiagnostics());
  }
  return diagnostics;
}

double curvatureCauchyStep(double descentNumerator, double curvature,
                           double maxStep) {
  if (!std::isfinite(descentNumerator) || !std::isfinite(curvature) ||
      !std::isfinite(maxStep) || descentNumerator <= 0.0 || maxStep <= 0.0 ||
      curvature <= 1e-18) {
    return 0.0;
  }
  const double rawStep = descentNumerator / curvature;
  if (!std::isfinite(rawStep) || rawStep <= 0.0) {
    return 0.0;
  }
  return std::min(rawStep, maxStep);
}

void addAmmTraceToAggregate(LocalOptimizationAggregate &aggregate,
                            unsigned robot,
                            const ManualDpgoMmAmmTrace &trace) {
  if (!trace.valid) {
    return;
  }
  ManualDpgoMmAmmTraceRow row;
  row.robot = robot;
  row.trace = trace;
  aggregate.ammTraceRows.push_back(row);
  ++aggregate.ammTraceCount;
  aggregate.ammRefined += trace.refined ? 1 : 0;
  aggregate.ammProxReset += trace.proxReset ? 1 : 0;
  aggregate.ammRestartUsedXakh += trace.restartUsedXakh ? 1 : 0;
  if (trace.restartCertificateValid) {
    const bool hadPrevious = aggregate.ammRestartCertificateCount > 0;
    ++aggregate.ammRestartCertificateCount;
    aggregate.ammRestartCertificatePassedCount +=
        trace.restartCertificatePassed ? 1 : 0;
    aggregate.ammRestartCertificateFailedCount +=
        trace.restartCertificatePassed ? 0 : 1;
    aggregate.ammRestartCertificateMinMargin =
        hadPrevious
            ? std::min(aggregate.ammRestartCertificateMinMargin,
                       trace.restartCertificateMargin)
            : trace.restartCertificateMargin;
  }
  aggregate.ammTranslationRecoveryAttemptCount +=
      trace.translationRecoveryAttemptCount;
  aggregate.ammTranslationRecoveryAcceptedCount +=
      trace.translationRecoveryAcceptedCount;
  aggregate.ammTranslationRecoveryRejectedCount +=
      trace.translationRecoveryRejectedCount;
  aggregate.ammGammaSum += trace.gamma;
  aggregate.ammGkhInitialSum += trace.GkhInitial;
  aggregate.ammMinGSum += trace.minG;
  aggregate.ammGkAfterAcceleratedSum += trace.GkAfterAccelerated;
  aggregate.ammGkhAfterRestartCheckSum += trace.GkhAfterRestartCheck;
  aggregate.ammFinalGkSum += trace.finalGk;
  aggregate.ammPhiLhsSum += trace.phiLhs;
  aggregate.ammPhiRhsSum += trace.phiRhs;
}

void addEdgeTightDiagnosticsToAggregate(
    LocalOptimizationAggregate &aggregate, const ManualDpgoMmAgent &agent) {
  const std::size_t count = agent.getLastEdgeTightQuadraticEvalCount();
  if (count == 0) {
    return;
  }
  const bool hadPrevious = aggregate.edgeTightQuadraticEvalCount > 0;
  aggregate.edgeTightQuadraticEvalCount += count;
  aggregate.edgeTightQuadraticSurrogateCostSum +=
      agent.getLastEdgeTightQuadraticSurrogateCostSum();
  aggregate.edgeTightQuadraticTrueCostSum +=
      agent.getLastEdgeTightQuadraticTrueCostSum();
  const double agentMin =
      agent.getLastEdgeTightQuadraticMajorizationGapMin();
  aggregate.edgeTightQuadraticMajorizationGapMin =
      hadPrevious
          ? std::min(aggregate.edgeTightQuadraticMajorizationGapMin,
                     agentMin)
          : agentMin;
}

void addSurrogateBoundDiagnosticsToAggregate(
    LocalOptimizationAggregate &aggregate, const ManualDpgoMmAgent &agent) {
  const std::size_t count = agent.getLastSurrogateBoundCheckCount();
  if (count == 0) {
    return;
  }
  const bool hadPrevious = aggregate.surrogateBoundCheckCount > 0;
  aggregate.surrogateBoundCheckCount += count;
  aggregate.surrogateBoundViolationCount +=
      agent.getLastSurrogateBoundViolationCount();
  const double agentMin = agent.getLastSurrogateBoundMinMargin();
  aggregate.surrogateBoundMinMargin =
      hadPrevious ? std::min(aggregate.surrogateBoundMinMargin, agentMin)
                  : agentMin;
}

void addBoundarySafeguardDiagnosticsToAggregate(
    LocalOptimizationAggregate &aggregate, const ManualDpgoMmAgent &agent) {
  const double before = agent.getLastBoundaryEdgeCostBefore();
  const double after = agent.getLastBoundaryEdgeCostAfter();
  const double deltaNorm = agent.getLastSeparatorDeltaNorm();
  if (std::isfinite(before)) {
    aggregate.boundaryEdgeCostBefore += before;
  }
  if (std::isfinite(after)) {
    aggregate.boundaryEdgeCostAfter += after;
  }
  if (std::isfinite(deltaNorm)) {
    aggregate.separatorDeltaNormSquared += deltaNorm * deltaNorm;
  }
}

void addVariableProjectedSchurDiagnosticsToAggregate(
    LocalOptimizationAggregate &aggregate, const ManualDpgoMmAgent &agent) {
  aggregate.variableProjectedSchurCandidateCount +=
      agent.getLastVariableProjectedSchurCandidateCount();
  aggregate.variableProjectedSchurAcceptedCount +=
      agent.getLastVariableProjectedSchurAcceptedCount();
}

void addBoundaryProximalDiagnosticsToAggregate(
    LocalOptimizationAggregate &aggregate, const ManualDpgoMmAgent &agent) {
  aggregate.boundaryProximalCandidateCount +=
      agent.getLastBoundaryProximalCandidateCount();
  aggregate.boundaryProximalAcceptedCount +=
      agent.getLastBoundaryProximalAcceptedCount();
}

LocalModelAggregate updateLocalModels(
    const std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  LocalModelAggregate aggregate;
  for (const auto &agent : agents) {
    if (!agent->updateLocalModel()) {
      ++aggregate.failures;
    }
  }
  return aggregate;
}

LocalOptimizationAggregate fixedNeighborChordalRefitLocalModels(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    bool parallelLocalSolves) {
  LocalOptimizationAggregate aggregate;
  if (parallelLocalSolves && agents.size() > 1) {
    std::vector<std::future<bool>> futures;
    futures.reserve(agents.size());
    for (auto &agent : agents) {
      ManualDpgoMmAgent *agentPtr = agent.get();
      futures.push_back(std::async(std::launch::async, [agentPtr]() {
        return agentPtr->fixedNeighborChordalRefit();
      }));
    }
    for (auto &future : futures) {
      if (!future.get()) {
        ++aggregate.failures;
      }
    }
    return aggregate;
  }

  for (auto &agent : agents) {
    if (!agent->fixedNeighborChordalRefit()) {
      ++aggregate.failures;
    }
  }
  return aggregate;
}

LocalModelAggregate evaluateLocalModels(
    const std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  LocalModelAggregate aggregate;
  aggregate.cost = 0.0;
  aggregate.gradient = 0.0;
  double gradSquared = 0.0;
  for (const auto &agent : agents) {
    double localCost = 0.0;
    double localGrad = 0.0;
    if (!agent->evaluateLocalModel(localCost, localGrad)) {
      ++aggregate.failures;
      continue;
    }
    aggregate.cost += localCost;
    gradSquared += localGrad * localGrad;
  }
  aggregate.gradient = std::sqrt(gradSquared);
  return aggregate;
}

LocalOptimizationAggregate refineLocalModels(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    bool parallelLocalSolves, unsigned optimizationRound = 0) {
  LocalOptimizationAggregate aggregate;
  if (parallelLocalSolves && agents.size() > 1) {
    std::vector<std::future<bool>> futures;
    futures.reserve(agents.size());
    for (auto &agent : agents) {
      ManualDpgoMmAgent *agentPtr = agent.get();
      agentPtr->resetLocalGradientCorrectionDiagnostics();
      agentPtr->setOptimizationRound(optimizationRound);
      futures.push_back(std::async(std::launch::async, [agentPtr]() {
        return agentPtr->optimizeLocalModel();
      }));
    }
    for (auto &future : futures) {
      if (!future.get()) {
        ++aggregate.failures;
      }
    }
    for (const auto &agent : agents) {
      aggregate.acceptedIterations += agent->getLastAcceptedIterationCount();
      aggregate.adaptiveRefinements += agent->getLastAdaptiveRefinementCount();
      aggregate.extrapolationAccepted +=
          agent->getLastExtrapolationAcceptedCount();
      aggregate.extrapolationRejected +=
          agent->getLastExtrapolationRejectedCount();
      aggregate.gExtrapolationAccepted +=
          agent->getLastGExtrapolationAcceptedCount();
      aggregate.gExtrapolationRejected +=
          agent->getLastGExtrapolationRejectedCount();
      aggregate.coupledExtrapolationAccepted +=
          agent->getLastCoupledExtrapolationAcceptedCount();
      aggregate.coupledExtrapolationRejected +=
          agent->getLastCoupledExtrapolationRejectedCount();
      aggregate.andersonAccepted += agent->getLastAndersonAcceptedCount();
      aggregate.andersonRejected += agent->getLastAndersonRejectedCount();
      aggregate.squaremAccepted += agent->getLastSquaremAcceptedCount();
      aggregate.squaremRejected += agent->getLastSquaremRejectedCount();
      aggregate.fullEquivHybridWarmStartCandidateCount +=
          agent->getLastFullEquivHybridWarmStartCandidateCount();
      aggregate.fullEquivHybridWarmStartAcceptedCount +=
          agent->getLastFullEquivHybridWarmStartAcceptedCount();
      aggregate.fullEquivHybridWarmStartGuardRejectedCount +=
          agent->getLastFullEquivHybridWarmStartGuardRejectedCount();
      aggregate.fullEquivHybridSchurStepCandidateCount +=
          agent->getLastFullEquivHybridSchurStepCandidateCount();
      aggregate.fullEquivHybridSchurStepAcceptedCount +=
          agent->getLastFullEquivHybridSchurStepAcceptedCount();
      aggregate.fullEquivHybridSchurStepGuardRejectedCount +=
          agent->getLastFullEquivHybridSchurStepGuardRejectedCount();
      aggregate.fullEquivHybridLocalPortfolioCandidateCount +=
          agent->getLastFullEquivHybridLocalPortfolioCandidateCount();
      aggregate.fullEquivHybridLocalPortfolioSelectedUnsmoothedCount +=
          agent
              ->getLastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount();
      aggregate.fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount +=
          agent
              ->getLastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount();
      aggregate.fullEquivHybridLocalPortfolioSelectedSchwarzFehCount +=
          agent->getLastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount();
      aggregate.fullEquivHybridSchwarzSmoothingSweepCount +=
          agent->getLastFullEquivHybridSchwarzSmoothingSweepCount();
      aggregate.fullEquivHybridSchwarzSmoothingCandidateCount +=
          agent->getLastFullEquivHybridSchwarzSmoothingCandidateCount();
      aggregate.fullEquivHybridSchwarzSmoothingAcceptedCount +=
          agent->getLastFullEquivHybridSchwarzSmoothingAcceptedCount();
      aggregate.fullEquivHybridSchwarzSmoothingRejectedCount +=
          agent->getLastFullEquivHybridSchwarzSmoothingRejectedCount();
      aggregate.fullEquivHybridSchwarzSmoothingCostDecrease +=
          agent->getLastFullEquivHybridSchwarzSmoothingCostDecrease();
      aggregate.fullEquivHybridSchwarzSmoothingTimeSec +=
          agent->getLastFullEquivHybridSchwarzSmoothingTimeSec();
      aggregate.fullEquivHybridLinearPcgIterationCount +=
          agent->getLastFullEquivHybridLinearPcgIterationCount();
      aggregate.fullEquivHybridLinearInitialResidual =
          std::max(aggregate.fullEquivHybridLinearInitialResidual,
                   agent->getLastFullEquivHybridLinearInitialResidual());
      aggregate.fullEquivHybridLinearFinalResidual =
          std::max(aggregate.fullEquivHybridLinearFinalResidual,
                   agent->getLastFullEquivHybridLinearFinalResidual());
      aggregate.fullEquivHybridLinearFullResidual =
          std::max(aggregate.fullEquivHybridLinearFullResidual,
                   agent->getLastFullEquivHybridLinearFullResidual());
      aggregate.fullEquivHybridLinearSolveTimeSec +=
          agent->getLastFullEquivHybridLinearSolveTimeSec();
      aggregate.fullEquivHybridStepTrialCount +=
          agent->getLastFullEquivHybridStepTrialCount();
      aggregate.fullEquivHybridStepTrialAcceptedCount +=
          agent->getLastFullEquivHybridStepTrialAcceptedCount();
      aggregate.fullEquivHybridStepTrialRejectedCount +=
          agent->getLastFullEquivHybridStepTrialRejectedCount();
      aggregate.fullEquivHybridStepPredictedDecreaseSum +=
          agent->getLastFullEquivHybridStepPredictedDecreaseSum();
      aggregate.fullEquivHybridStepActualDecreaseSum +=
          agent->getLastFullEquivHybridStepActualDecreaseSum();
      aggregate.fullEquivHybridStepRhoSum +=
          agent->getLastFullEquivHybridStepRhoSum();
      aggregate.fullEquivHybridStepRhoCount +=
          agent->getLastFullEquivHybridStepRhoCount();
      aggregate.fullEquivHybridStepAcceptedScaleSum +=
          agent->getLastFullEquivHybridStepAcceptedScaleSum();
      aggregate.fullEquivHybridStepAcceptedPredictedDecreaseSum +=
          agent->getLastFullEquivHybridStepAcceptedPredictedDecreaseSum();
      aggregate.fullEquivHybridStepAcceptedActualDecreaseSum +=
          agent->getLastFullEquivHybridStepAcceptedActualDecreaseSum();
      aggregate.fullEquivHybridStepAcceptedRhoSum +=
          agent->getLastFullEquivHybridStepAcceptedRhoSum();
      aggregate.fullEquivHybridStepAcceptedRhoCount +=
          agent->getLastFullEquivHybridStepAcceptedRhoCount();
      aggregate.fullEquivHybridStepParetoCandidateCount +=
          agent->getLastFullEquivHybridStepParetoCandidateCount();
      aggregate.fullEquivHybridStepParetoSelectedCount +=
          agent->getLastFullEquivHybridStepParetoSelectedCount();
      aggregate.fullEquivHybridStepParetoSelectedScaleSum +=
          agent->getLastFullEquivHybridStepParetoSelectedScaleSum();
      aggregate.fullEquivHybridStepParetoBestDecreaseSum +=
          agent->getLastFullEquivHybridStepParetoBestDecreaseSum();
      aggregate.fullEquivHybridStepParetoSelectedDecreaseSum +=
          agent->getLastFullEquivHybridStepParetoSelectedDecreaseSum();
      aggregate.fullEquivHybridStepParetoSelectedGradientSum +=
          agent->getLastFullEquivHybridStepParetoSelectedGradientSum();
      aggregate.fullEquivHybridStepParetoGradientEvalCount +=
          agent->getLastFullEquivHybridStepParetoGradientEvalCount();
      aggregate.fullEquivHybridStepParetoGradientEvalTimeSec +=
          agent->getLastFullEquivHybridStepParetoGradientEvalTimeSec();
      aggregate.fullEquivHybridTranslationRecoveryStepTrialCandidateCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryStepTrialCandidateCount();
      aggregate.fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount();
      aggregate.fullEquivHybridTranslationRecoveryStepTrialSelectedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryStepTrialSelectedCount();
      aggregate.fullEquivHybridActiveSeparatorCandidateCount +=
          agent->getLastFullEquivHybridActiveSeparatorCandidateCount();
      aggregate.fullEquivHybridActiveSeparatorAcceptedCount +=
          agent->getLastFullEquivHybridActiveSeparatorAcceptedCount();
      aggregate.fullEquivHybridActiveSeparatorRejectedCount +=
          agent->getLastFullEquivHybridActiveSeparatorRejectedCount();
      aggregate.fullEquivHybridActiveSeparatorStepSum +=
          agent->getLastFullEquivHybridActiveSeparatorStepSum();
      aggregate.fullEquivHybridActiveSeparatorCostDecreaseSum +=
          agent->getLastFullEquivHybridActiveSeparatorCostDecreaseSum();
      aggregate.fullEquivHybridActiveSeparatorLmSchurCandidateCount +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurCandidateCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurAcceptedCount +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurAcceptedCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount +=
          agent
              ->getLastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount();
      aggregate
          .fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount +=
          agent
              ->getLastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurSolveFailureCount +=
          agent
              ->getLastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurFallbackCount +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurFallbackCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurBoundaryColCount +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurPrivateColCount +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurPrivateColCount();
      aggregate
          .fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount +=
          agent
              ->getLastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount();
      aggregate.fullEquivHybridActiveSeparatorLmSchurAlphaSum +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurAlphaSum();
      aggregate.fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum +=
          agent->getLastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum();
      aggregate.fullEquivHybridActiveSeparatorLmSchurGradientChangeSum +=
          agent
              ->getLastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum();
      aggregate.fullEquivHybridTranslationRecoveryPolishAttemptCount +=
          agent->getLastFullEquivHybridTranslationRecoveryPolishAttemptCount();
      aggregate.fullEquivHybridTranslationRecoveryPolishAcceptedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishAcceptedCount();
      aggregate.fullEquivHybridTranslationRecoveryPolishRejectedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishRejectedCount();
      aggregate
          .fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount();
      aggregate
          .fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount();
      aggregate.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount();
      aggregate
          .fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount();
      aggregate.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum();
      aggregate.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount();
      aggregate
          .fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount();
      aggregate
          .fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount();
      aggregate.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum();
      aggregate.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum();
      aggregate
          .fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum();
      aggregate.fullEquivHybridTranslationRecoveryPolishCostDecreaseSum +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum();
      aggregate.fullEquivHybridTranslationRecoveryPolishGradientChangeSum +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryPolishGradientChangeSum();
      aggregate.localGradientDiagnostics.add(
          agent->getLocalGradientCorrectionDiagnostics());
      aggregate.fullEquivHybridSparseMatrixVectorProductCount +=
          agent->getLastFullEquivHybridSparseMatrixVectorProductCount();
      aggregate.fullEquivHybridReducedRotationInitialGuessCandidateCount +=
          agent
              ->getLastFullEquivHybridReducedRotationInitialGuessCandidateCount();
      aggregate.fullEquivHybridReducedRotationInitialGuessUsedCount +=
          agent->getLastFullEquivHybridReducedRotationInitialGuessUsedCount();
      aggregate.fullEquivHybridReducedRotationInitialGuessRejectedCount +=
          agent
              ->getLastFullEquivHybridReducedRotationInitialGuessRejectedCount();
      aggregate
          .fullEquivHybridTranslationRecoveryInitialGuessCandidateCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount();
      aggregate.fullEquivHybridTranslationRecoveryInitialGuessUsedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryInitialGuessUsedCount();
      aggregate
          .fullEquivHybridTranslationRecoveryInitialGuessRejectedCount +=
          agent
              ->getLastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount();
      aggregate
          .fullEquivHybridTranslationSchurPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridTranslationSchurPreconditionerApplicationCount();
      aggregate
          .fullEquivHybridTranslationSchurPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridTranslationSchurPreconditionerFactorizationCount();
      aggregate.fullEquivHybridTranslationSchurPreconditionerFallbackCount +=
          agent
              ->getLastFullEquivHybridTranslationSchurPreconditionerFallbackCount();
      aggregate.fullEquivHybridLocalChainPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridLocalChainPreconditionerApplicationCount();
      aggregate.fullEquivHybridLocalChainPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridLocalChainPreconditionerFactorizationCount();
      aggregate.fullEquivHybridLocalChainPreconditionerFallbackCount +=
          agent
              ->getLastFullEquivHybridLocalChainPreconditionerFallbackCount();
      aggregate
          .fullEquivHybridTranslationBlockPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridTranslationBlockPreconditionerApplicationCount();
      aggregate
          .fullEquivHybridTranslationBlockPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridTranslationBlockPreconditionerFactorizationCount();
      aggregate.fullEquivHybridTranslationBlockPreconditionerFallbackCount +=
          agent
              ->getLastFullEquivHybridTranslationBlockPreconditionerFallbackCount();
      aggregate
          .fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount();
      aggregate
          .fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount();
      aggregate.fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount +=
          agent
              ->getLastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount();
      aggregate
          .fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount();
      aggregate
          .fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount();
      aggregate.fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount +=
          agent
              ->getLastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount();
      aggregate.fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount =
          std::max(
              aggregate
                  .fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount,
              agent
                  ->getLastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount());
      aggregate
          .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount =
          std::max(
              aggregate
                  .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount,
              agent
                  ->getLastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount());
      aggregate
          .fullEquivHybridLaplacianDeflationPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount();
      aggregate
          .fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount();
      aggregate
          .fullEquivHybridLaplacianDeflationPreconditionerFallbackCount +=
          agent
              ->getLastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount();
      aggregate
          .fullEquivHybridLaplacianDeflationPreconditionerBasisDimension =
          std::max(
              aggregate
                  .fullEquivHybridLaplacianDeflationPreconditionerBasisDimension,
              agent
                  ->getLastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension());
      aggregate
          .fullEquivHybridReducedRotationPreconditionerApplicationCount +=
          agent
              ->getLastFullEquivHybridReducedRotationPreconditionerApplicationCount();
      aggregate
          .fullEquivHybridReducedRotationPreconditionerFactorizationCount +=
          agent
              ->getLastFullEquivHybridReducedRotationPreconditionerFactorizationCount();
      aggregate.fullEquivHybridRqnUsedCount +=
          agent->getLastFullEquivHybridRqnUsedCount();
      aggregate.fullEquivHybridRqnAcceptedPairCount +=
          agent->getLastFullEquivHybridRqnAcceptedPairCount();
      aggregate.fullEquivHybridRqnRejectedPairCount +=
          agent->getLastFullEquivHybridRqnRejectedPairCount();
      aggregate.fullEquivHybridRqnMemorySize +=
          agent->getLastFullEquivHybridRqnMemorySize();
      aggregate.fullEquivHybridRqnPreconditionerApplicationCount +=
          agent->getLastFullEquivHybridRqnPreconditionerApplicationCount();
      aggregate.ammAcceleratedAccepted +=
          agent->getLastAmmAcceleratedAcceptedCount();
      aggregate.ammRestart += agent->getLastAmmRestartCount();
      aggregate.ammHardRestart += agent->getLastAmmHardRestartCount();
      aggregate.ammSoftRestart += agent->getLastAmmSoftRestartCount();
      aggregate.ammPhiFallback += agent->getLastAmmPhiFallbackCount();
      aggregate.ammLocalMeritRejected +=
          agent->getLastAmmLocalMeritRejectedCount();
      aggregate.ammProximalStart += agent->getLastAmmProximalStartCount();
      aggregate.ammSkipped += agent->getLastAmmSkippedCount();
      aggregate.ammMixedSurrogateCandidateCount +=
          agent->getLastAmmMixedSurrogateCandidateCount();
      aggregate.ammMixedSurrogateTrueLocalAcceptedCount +=
          agent->getLastAmmMixedSurrogateTrueLocalAcceptedCount();
      aggregate.ammMixedSurrogateSimpleSelectedCount +=
          agent->getLastAmmMixedSurrogateSimpleSelectedCount();
      aggregate.ammMixedSurrogateTrueLocalSelectedCount +=
          agent->getLastAmmMixedSurrogateTrueLocalSelectedCount();
      aggregate.ammMixedSurrogateExtrapolatedSelectedCount +=
          agent->getLastAmmMixedSurrogateExtrapolatedSelectedCount();
      aggregate.ammMixedSurrogateOtherSelectedCount +=
          agent->getLastAmmMixedSurrogateOtherSelectedCount();
      aggregate.ammMixedSurrogateSimpleSkippedCount +=
          agent->getLastAmmMixedSurrogateSimpleSkippedCount();
      aggregate.ammMixedSurrogateSimpleForcedRefreshCount +=
          agent->getLastAmmMixedSurrogateSimpleForcedRefreshCount();
      addEdgeTightDiagnosticsToAggregate(aggregate, *agent);
      addBoundarySafeguardDiagnosticsToAggregate(aggregate, *agent);
      addVariableProjectedSchurDiagnosticsToAggregate(aggregate, *agent);
      addBoundaryProximalDiagnosticsToAggregate(aggregate, *agent);
      addSurrogateBoundDiagnosticsToAggregate(aggregate, *agent);
      addAmmTraceToAggregate(aggregate, agent->getID(),
                             agent->getLastAmmTrace());
    }
    return aggregate;
  }

  for (auto &agent : agents) {
    agent->resetLocalGradientCorrectionDiagnostics();
    agent->setOptimizationRound(optimizationRound);
    if (!agent->optimizeLocalModel()) {
      ++aggregate.failures;
    }
    aggregate.acceptedIterations += agent->getLastAcceptedIterationCount();
    aggregate.adaptiveRefinements += agent->getLastAdaptiveRefinementCount();
    aggregate.extrapolationAccepted +=
        agent->getLastExtrapolationAcceptedCount();
    aggregate.extrapolationRejected +=
        agent->getLastExtrapolationRejectedCount();
    aggregate.gExtrapolationAccepted +=
        agent->getLastGExtrapolationAcceptedCount();
    aggregate.gExtrapolationRejected +=
        agent->getLastGExtrapolationRejectedCount();
    aggregate.coupledExtrapolationAccepted +=
        agent->getLastCoupledExtrapolationAcceptedCount();
    aggregate.coupledExtrapolationRejected +=
        agent->getLastCoupledExtrapolationRejectedCount();
    aggregate.andersonAccepted += agent->getLastAndersonAcceptedCount();
    aggregate.andersonRejected += agent->getLastAndersonRejectedCount();
    aggregate.squaremAccepted += agent->getLastSquaremAcceptedCount();
    aggregate.squaremRejected += agent->getLastSquaremRejectedCount();
    aggregate.fullEquivHybridWarmStartCandidateCount +=
        agent->getLastFullEquivHybridWarmStartCandidateCount();
    aggregate.fullEquivHybridWarmStartAcceptedCount +=
        agent->getLastFullEquivHybridWarmStartAcceptedCount();
    aggregate.fullEquivHybridWarmStartGuardRejectedCount +=
        agent->getLastFullEquivHybridWarmStartGuardRejectedCount();
    aggregate.fullEquivHybridSchurStepCandidateCount +=
        agent->getLastFullEquivHybridSchurStepCandidateCount();
    aggregate.fullEquivHybridSchurStepAcceptedCount +=
        agent->getLastFullEquivHybridSchurStepAcceptedCount();
    aggregate.fullEquivHybridSchurStepGuardRejectedCount +=
        agent->getLastFullEquivHybridSchurStepGuardRejectedCount();
    aggregate.fullEquivHybridLocalPortfolioCandidateCount +=
        agent->getLastFullEquivHybridLocalPortfolioCandidateCount();
    aggregate.fullEquivHybridLocalPortfolioSelectedUnsmoothedCount +=
        agent->getLastFullEquivHybridLocalPortfolioSelectedUnsmoothedCount();
    aggregate.fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount +=
        agent->getLastFullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount();
    aggregate.fullEquivHybridLocalPortfolioSelectedSchwarzFehCount +=
        agent->getLastFullEquivHybridLocalPortfolioSelectedSchwarzFehCount();
    aggregate.fullEquivHybridSchwarzSmoothingSweepCount +=
        agent->getLastFullEquivHybridSchwarzSmoothingSweepCount();
    aggregate.fullEquivHybridSchwarzSmoothingCandidateCount +=
        agent->getLastFullEquivHybridSchwarzSmoothingCandidateCount();
    aggregate.fullEquivHybridSchwarzSmoothingAcceptedCount +=
        agent->getLastFullEquivHybridSchwarzSmoothingAcceptedCount();
    aggregate.fullEquivHybridSchwarzSmoothingRejectedCount +=
        agent->getLastFullEquivHybridSchwarzSmoothingRejectedCount();
    aggregate.fullEquivHybridSchwarzSmoothingCostDecrease +=
        agent->getLastFullEquivHybridSchwarzSmoothingCostDecrease();
    aggregate.fullEquivHybridSchwarzSmoothingTimeSec +=
        agent->getLastFullEquivHybridSchwarzSmoothingTimeSec();
    aggregate.fullEquivHybridLinearPcgIterationCount +=
        agent->getLastFullEquivHybridLinearPcgIterationCount();
    aggregate.fullEquivHybridLinearInitialResidual =
        std::max(aggregate.fullEquivHybridLinearInitialResidual,
                 agent->getLastFullEquivHybridLinearInitialResidual());
    aggregate.fullEquivHybridLinearFinalResidual =
        std::max(aggregate.fullEquivHybridLinearFinalResidual,
                 agent->getLastFullEquivHybridLinearFinalResidual());
    aggregate.fullEquivHybridLinearFullResidual =
        std::max(aggregate.fullEquivHybridLinearFullResidual,
                 agent->getLastFullEquivHybridLinearFullResidual());
    aggregate.fullEquivHybridLinearSolveTimeSec +=
        agent->getLastFullEquivHybridLinearSolveTimeSec();
    aggregate.fullEquivHybridStepTrialCount +=
        agent->getLastFullEquivHybridStepTrialCount();
    aggregate.fullEquivHybridStepTrialAcceptedCount +=
        agent->getLastFullEquivHybridStepTrialAcceptedCount();
    aggregate.fullEquivHybridStepTrialRejectedCount +=
        agent->getLastFullEquivHybridStepTrialRejectedCount();
    aggregate.fullEquivHybridStepPredictedDecreaseSum +=
        agent->getLastFullEquivHybridStepPredictedDecreaseSum();
    aggregate.fullEquivHybridStepActualDecreaseSum +=
        agent->getLastFullEquivHybridStepActualDecreaseSum();
    aggregate.fullEquivHybridStepRhoSum +=
        agent->getLastFullEquivHybridStepRhoSum();
    aggregate.fullEquivHybridStepRhoCount +=
        agent->getLastFullEquivHybridStepRhoCount();
    aggregate.fullEquivHybridStepAcceptedScaleSum +=
        agent->getLastFullEquivHybridStepAcceptedScaleSum();
    aggregate.fullEquivHybridStepAcceptedPredictedDecreaseSum +=
        agent->getLastFullEquivHybridStepAcceptedPredictedDecreaseSum();
    aggregate.fullEquivHybridStepAcceptedActualDecreaseSum +=
        agent->getLastFullEquivHybridStepAcceptedActualDecreaseSum();
    aggregate.fullEquivHybridStepAcceptedRhoSum +=
        agent->getLastFullEquivHybridStepAcceptedRhoSum();
    aggregate.fullEquivHybridStepAcceptedRhoCount +=
        agent->getLastFullEquivHybridStepAcceptedRhoCount();
    aggregate.fullEquivHybridStepParetoCandidateCount +=
        agent->getLastFullEquivHybridStepParetoCandidateCount();
    aggregate.fullEquivHybridStepParetoSelectedCount +=
        agent->getLastFullEquivHybridStepParetoSelectedCount();
    aggregate.fullEquivHybridStepParetoSelectedScaleSum +=
        agent->getLastFullEquivHybridStepParetoSelectedScaleSum();
    aggregate.fullEquivHybridStepParetoBestDecreaseSum +=
        agent->getLastFullEquivHybridStepParetoBestDecreaseSum();
    aggregate.fullEquivHybridStepParetoSelectedDecreaseSum +=
        agent->getLastFullEquivHybridStepParetoSelectedDecreaseSum();
    aggregate.fullEquivHybridStepParetoSelectedGradientSum +=
        agent->getLastFullEquivHybridStepParetoSelectedGradientSum();
    aggregate.fullEquivHybridStepParetoGradientEvalCount +=
        agent->getLastFullEquivHybridStepParetoGradientEvalCount();
    aggregate.fullEquivHybridStepParetoGradientEvalTimeSec +=
        agent->getLastFullEquivHybridStepParetoGradientEvalTimeSec();
    aggregate.fullEquivHybridTranslationRecoveryStepTrialCandidateCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryStepTrialCandidateCount();
    aggregate.fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount();
    aggregate.fullEquivHybridTranslationRecoveryStepTrialSelectedCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryStepTrialSelectedCount();
    aggregate.fullEquivHybridActiveSeparatorCandidateCount +=
        agent->getLastFullEquivHybridActiveSeparatorCandidateCount();
    aggregate.fullEquivHybridActiveSeparatorAcceptedCount +=
        agent->getLastFullEquivHybridActiveSeparatorAcceptedCount();
    aggregate.fullEquivHybridActiveSeparatorRejectedCount +=
        agent->getLastFullEquivHybridActiveSeparatorRejectedCount();
    aggregate.fullEquivHybridActiveSeparatorStepSum +=
        agent->getLastFullEquivHybridActiveSeparatorStepSum();
    aggregate.fullEquivHybridActiveSeparatorCostDecreaseSum +=
        agent->getLastFullEquivHybridActiveSeparatorCostDecreaseSum();
    aggregate.fullEquivHybridActiveSeparatorLmSchurCandidateCount +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurCandidateCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurAcceptedCount +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurAcceptedCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount +=
        agent
            ->getLastFullEquivHybridActiveSeparatorLmSchurGuardRejectedCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount +=
        agent
            ->getLastFullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurSolveFailureCount +=
        agent
            ->getLastFullEquivHybridActiveSeparatorLmSchurSolveFailureCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurFallbackCount +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurFallbackCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurBoundaryColCount +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurBoundaryColCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurPrivateColCount +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurPrivateColCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount +=
        agent
            ->getLastFullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount();
    aggregate.fullEquivHybridActiveSeparatorLmSchurAlphaSum +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurAlphaSum();
    aggregate.fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum +=
        agent->getLastFullEquivHybridActiveSeparatorLmSchurCostDecreaseSum();
    aggregate.fullEquivHybridActiveSeparatorLmSchurGradientChangeSum +=
        agent
            ->getLastFullEquivHybridActiveSeparatorLmSchurGradientChangeSum();
    aggregate.fullEquivHybridTranslationRecoveryPolishAttemptCount +=
        agent->getLastFullEquivHybridTranslationRecoveryPolishAttemptCount();
    aggregate.fullEquivHybridTranslationRecoveryPolishAcceptedCount +=
        agent->getLastFullEquivHybridTranslationRecoveryPolishAcceptedCount();
    aggregate.fullEquivHybridTranslationRecoveryPolishRejectedCount +=
        agent->getLastFullEquivHybridTranslationRecoveryPolishRejectedCount();
    aggregate
        .fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount();
    aggregate
        .fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount();
    aggregate.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount();
    aggregate
        .fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount();
    aggregate.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum();
    aggregate.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishMeritCandidateCount();
    aggregate
        .fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount();
    aggregate
        .fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount();
    aggregate.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum();
    aggregate.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum();
    aggregate.fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum();
    aggregate.fullEquivHybridTranslationRecoveryPolishCostDecreaseSum +=
        agent->getLastFullEquivHybridTranslationRecoveryPolishCostDecreaseSum();
    aggregate.fullEquivHybridTranslationRecoveryPolishGradientChangeSum +=
        agent->getLastFullEquivHybridTranslationRecoveryPolishGradientChangeSum();
    aggregate.localGradientDiagnostics.add(
        agent->getLocalGradientCorrectionDiagnostics());
    aggregate.fullEquivHybridSparseMatrixVectorProductCount +=
        agent->getLastFullEquivHybridSparseMatrixVectorProductCount();
    aggregate.fullEquivHybridReducedRotationInitialGuessCandidateCount +=
        agent
            ->getLastFullEquivHybridReducedRotationInitialGuessCandidateCount();
    aggregate.fullEquivHybridReducedRotationInitialGuessUsedCount +=
        agent->getLastFullEquivHybridReducedRotationInitialGuessUsedCount();
    aggregate.fullEquivHybridReducedRotationInitialGuessRejectedCount +=
        agent
            ->getLastFullEquivHybridReducedRotationInitialGuessRejectedCount();
    aggregate.fullEquivHybridTranslationRecoveryInitialGuessCandidateCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryInitialGuessCandidateCount();
    aggregate.fullEquivHybridTranslationRecoveryInitialGuessUsedCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryInitialGuessUsedCount();
    aggregate.fullEquivHybridTranslationRecoveryInitialGuessRejectedCount +=
        agent
            ->getLastFullEquivHybridTranslationRecoveryInitialGuessRejectedCount();
    aggregate.fullEquivHybridTranslationSchurPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridTranslationSchurPreconditionerApplicationCount();
    aggregate.fullEquivHybridTranslationSchurPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridTranslationSchurPreconditionerFactorizationCount();
    aggregate.fullEquivHybridTranslationSchurPreconditionerFallbackCount +=
        agent
            ->getLastFullEquivHybridTranslationSchurPreconditionerFallbackCount();
    aggregate.fullEquivHybridLocalChainPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridLocalChainPreconditionerApplicationCount();
    aggregate.fullEquivHybridLocalChainPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridLocalChainPreconditionerFactorizationCount();
    aggregate.fullEquivHybridLocalChainPreconditionerFallbackCount +=
        agent->getLastFullEquivHybridLocalChainPreconditionerFallbackCount();
    aggregate.fullEquivHybridTranslationBlockPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridTranslationBlockPreconditionerApplicationCount();
    aggregate.fullEquivHybridTranslationBlockPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridTranslationBlockPreconditionerFactorizationCount();
    aggregate.fullEquivHybridTranslationBlockPreconditionerFallbackCount +=
        agent
            ->getLastFullEquivHybridTranslationBlockPreconditionerFallbackCount();
    aggregate.fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridTranslationSparseSchurPreconditionerApplicationCount();
    aggregate.fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount();
    aggregate.fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount +=
        agent
            ->getLastFullEquivHybridTranslationSparseSchurPreconditionerFallbackCount();
    aggregate.fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridTranslationLocalSchurPreconditionerApplicationCount();
    aggregate.fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount();
    aggregate.fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount +=
        agent
            ->getLastFullEquivHybridTranslationLocalSchurPreconditionerFallbackCount();
    aggregate.fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount =
        std::max(
            aggregate
                .fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount,
            agent
                ->getLastFullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount());
    aggregate
        .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount =
        std::max(
            aggregate
                .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount,
            agent
                ->getLastFullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount());
    aggregate.fullEquivHybridLaplacianDeflationPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridLaplacianDeflationPreconditionerApplicationCount();
    aggregate
        .fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridLaplacianDeflationPreconditionerFactorizationCount();
    aggregate.fullEquivHybridLaplacianDeflationPreconditionerFallbackCount +=
        agent
            ->getLastFullEquivHybridLaplacianDeflationPreconditionerFallbackCount();
    aggregate.fullEquivHybridLaplacianDeflationPreconditionerBasisDimension =
        std::max(
            aggregate
                .fullEquivHybridLaplacianDeflationPreconditionerBasisDimension,
            agent
                ->getLastFullEquivHybridLaplacianDeflationPreconditionerBasisDimension());
    aggregate.fullEquivHybridReducedRotationPreconditionerApplicationCount +=
        agent
            ->getLastFullEquivHybridReducedRotationPreconditionerApplicationCount();
    aggregate.fullEquivHybridReducedRotationPreconditionerFactorizationCount +=
        agent
            ->getLastFullEquivHybridReducedRotationPreconditionerFactorizationCount();
    aggregate.fullEquivHybridRqnUsedCount +=
        agent->getLastFullEquivHybridRqnUsedCount();
    aggregate.fullEquivHybridRqnAcceptedPairCount +=
        agent->getLastFullEquivHybridRqnAcceptedPairCount();
    aggregate.fullEquivHybridRqnRejectedPairCount +=
        agent->getLastFullEquivHybridRqnRejectedPairCount();
    aggregate.fullEquivHybridRqnMemorySize +=
        agent->getLastFullEquivHybridRqnMemorySize();
    aggregate.fullEquivHybridRqnPreconditionerApplicationCount +=
        agent->getLastFullEquivHybridRqnPreconditionerApplicationCount();
    aggregate.ammAcceleratedAccepted +=
        agent->getLastAmmAcceleratedAcceptedCount();
    aggregate.ammRestart += agent->getLastAmmRestartCount();
    aggregate.ammHardRestart += agent->getLastAmmHardRestartCount();
    aggregate.ammSoftRestart += agent->getLastAmmSoftRestartCount();
    aggregate.ammPhiFallback += agent->getLastAmmPhiFallbackCount();
    aggregate.ammLocalMeritRejected +=
        agent->getLastAmmLocalMeritRejectedCount();
    aggregate.ammProximalStart += agent->getLastAmmProximalStartCount();
    aggregate.ammSkipped += agent->getLastAmmSkippedCount();
    aggregate.ammMixedSurrogateCandidateCount +=
        agent->getLastAmmMixedSurrogateCandidateCount();
    aggregate.ammMixedSurrogateTrueLocalAcceptedCount +=
        agent->getLastAmmMixedSurrogateTrueLocalAcceptedCount();
    aggregate.ammMixedSurrogateSimpleSelectedCount +=
        agent->getLastAmmMixedSurrogateSimpleSelectedCount();
    aggregate.ammMixedSurrogateTrueLocalSelectedCount +=
        agent->getLastAmmMixedSurrogateTrueLocalSelectedCount();
    aggregate.ammMixedSurrogateExtrapolatedSelectedCount +=
        agent->getLastAmmMixedSurrogateExtrapolatedSelectedCount();
    aggregate.ammMixedSurrogateOtherSelectedCount +=
        agent->getLastAmmMixedSurrogateOtherSelectedCount();
    aggregate.ammMixedSurrogateSimpleSkippedCount +=
        agent->getLastAmmMixedSurrogateSimpleSkippedCount();
    aggregate.ammMixedSurrogateSimpleForcedRefreshCount +=
        agent->getLastAmmMixedSurrogateSimpleForcedRefreshCount();
    addEdgeTightDiagnosticsToAggregate(aggregate, *agent);
    addBoundarySafeguardDiagnosticsToAggregate(aggregate, *agent);
    addVariableProjectedSchurDiagnosticsToAggregate(aggregate, *agent);
    addBoundaryProximalDiagnosticsToAggregate(aggregate, *agent);
    addSurrogateBoundDiagnosticsToAggregate(aggregate, *agent);
    addAmmTraceToAggregate(aggregate, agent->getID(),
                           agent->getLastAmmTrace());
  }
  return aggregate;
}

LocalGradientCorrectionAggregate applyLocalGradientCorrections(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    const std::vector<double> &steps, bool parallelLocalSolves) {
  LocalGradientCorrectionAggregate aggregate;
  if (steps.empty()) {
    return aggregate;
  }
  if (parallelLocalSolves && agents.size() > 1) {
    struct RobotCorrection {
      bool ok{false};
      std::size_t accepted{0};
      std::size_t rejected{0};
    };
    std::vector<std::future<RobotCorrection>> futures;
    futures.reserve(agents.size());
    for (auto &agent : agents) {
      ManualDpgoMmAgent *agentPtr = agent.get();
      futures.push_back(std::async(std::launch::async, [agentPtr, &steps]() {
        RobotCorrection correction;
        correction.ok = agentPtr->applyLocalGradientCorrection(
            steps, correction.accepted, correction.rejected);
        return correction;
      }));
    }
    for (auto &future : futures) {
      const RobotCorrection correction = future.get();
      if (!correction.ok) {
        ++aggregate.failures;
      }
      aggregate.accepted += correction.accepted;
      aggregate.rejected += correction.rejected;
    }
    return aggregate;
  }

  for (auto &agent : agents) {
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agent->applyLocalGradientCorrection(steps, accepted, rejected)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
  }
  return aggregate;
}

LocalGradientCorrectionAggregate applyCurvatureLocalGradientCorrections(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  LocalGradientCorrectionAggregate aggregate;
  double selectedStepSum = 0.0;
  std::size_t selectedRobotCount = 0;
  for (auto &agent : agents) {
    LocalGradientCorrectionCurvatureProbe probe;
    if (!agent->evaluateLocalGradientCorrectionCurvature(probe)) {
      ++aggregate.failures;
      continue;
    }
    const double step = curvatureCauchyStep(
        probe.descentNumerator, probe.curvature, probe.maxStep);
    if (!probe.ok || !probe.currentFinite || step <= 0.0) {
      ++aggregate.rejected;
      continue;
    }
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agent->applyLocalGradientCorrectionStep(step, accepted, rejected,
                                                 true)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
    selectedStepSum += step;
    ++selectedRobotCount;
  }
  aggregate.selectedStep =
      selectedRobotCount == 0
          ? 0.0
          : selectedStepSum / static_cast<double>(selectedRobotCount);
  return aggregate;
}

LocalGradientCorrectionAggregate applySharedCurvatureLocalGradientCorrection(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  LocalGradientCorrectionAggregate aggregate;
  if (agents.empty()) {
    return aggregate;
  }
  aggregate.scalarCount = agents.size() * static_cast<std::size_t>(2);
  aggregate.scalarCommMb = scalarMegabytes(aggregate.scalarCount);

  double descentNumeratorSum = 0.0;
  double curvatureSum = 0.0;
  double maxStep = std::numeric_limits<double>::infinity();
  for (auto &agent : agents) {
    LocalGradientCorrectionCurvatureProbe probe;
    if (!agent->evaluateLocalGradientCorrectionCurvature(probe)) {
      ++aggregate.failures;
      continue;
    }
    if (!probe.ok || !probe.currentFinite || probe.descentNumerator <= 0.0 ||
        probe.curvature <= 0.0 || probe.maxStep <= 0.0) {
      ++aggregate.rejected;
      continue;
    }
    descentNumeratorSum += probe.descentNumerator;
    curvatureSum += probe.curvature;
    maxStep = std::min(maxStep, probe.maxStep);
  }
  if (aggregate.failures > 0) {
    aggregate.rejected += agents.size();
    return aggregate;
  }

  const double step =
      curvatureCauchyStep(descentNumeratorSum, curvatureSum, maxStep);
  aggregate.selectedStep = step;
  if (step <= 0.0) {
    aggregate.rejected += agents.size();
    return aggregate;
  }

  for (auto &agent : agents) {
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agent->applyLocalGradientCorrectionStep(step, accepted, rejected,
                                                 true)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
  }
  return aggregate;
}

LocalGradientCorrectionAggregate applySharedStepLocalGradientCorrection(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    const std::vector<double> &steps) {
  LocalGradientCorrectionAggregate aggregate;
  if (steps.empty() || agents.empty()) {
    return aggregate;
  }
  aggregate.scalarCount =
      agents.size() * (steps.size() + static_cast<std::size_t>(2));
  aggregate.scalarCommMb = scalarMegabytes(aggregate.scalarCount);

  double currentCostSum = 0.0;
  std::vector<double> trialCostSums(
      steps.size(), std::numeric_limits<double>::infinity());
  bool haveFiniteCurrent = true;
  std::vector<std::vector<double>> robotTrialCosts;
  robotTrialCosts.reserve(agents.size());
  for (const auto &agent : agents) {
    double currentCost = 0.0;
    std::vector<double> trialCosts;
    if (!agent->evaluateLocalGradientCorrectionCosts(steps, currentCost,
                                                     trialCosts)) {
      ++aggregate.failures;
      continue;
    }
    if (!std::isfinite(currentCost)) {
      haveFiniteCurrent = false;
    } else {
      currentCostSum += currentCost;
    }
    robotTrialCosts.push_back(std::move(trialCosts));
  }
  if (aggregate.failures > 0 || !haveFiniteCurrent ||
      robotTrialCosts.size() != agents.size()) {
    aggregate.rejected = agents.size() * steps.size();
    return aggregate;
  }

  for (std::size_t stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
    double sum = 0.0;
    bool finite = true;
    for (const auto &costs : robotTrialCosts) {
      if (stepIdx >= costs.size() || !std::isfinite(costs[stepIdx])) {
        finite = false;
        break;
      }
      sum += costs[stepIdx];
    }
    if (finite) {
      trialCostSums[stepIdx] = sum;
    }
  }

  std::size_t bestStepIdx = steps.size();
  double bestCost = currentCostSum;
  for (std::size_t stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
    if (std::isfinite(trialCostSums[stepIdx]) &&
        trialCostSums[stepIdx] < bestCost - 1e-12) {
      bestStepIdx = stepIdx;
      bestCost = trialCostSums[stepIdx];
    }
  }
  if (bestStepIdx == steps.size()) {
    aggregate.rejected = agents.size() * steps.size();
    return aggregate;
  }
  aggregate.selectedStep = steps[bestStepIdx];

  for (auto &agent : agents) {
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agent->applyLocalGradientCorrectionStep(steps[bestStepIdx], accepted,
                                                 rejected)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
  }
  aggregate.rejected += agents.size() * (steps.size() - 1);
  return aggregate;
}

LocalGradientCorrectionAggregate
applyNeighborhoodCurvatureLocalGradientCorrection(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents) {
  LocalGradientCorrectionAggregate aggregate;
  if (agents.empty()) {
    return aggregate;
  }

  std::vector<LocalGradientCorrectionCurvatureProbe> probes(agents.size());
  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    if (!agents[robot]->evaluateLocalGradientCorrectionCurvature(
            probes[robot])) {
      ++aggregate.failures;
    }
  }
  if (aggregate.failures > 0) {
    aggregate.rejected = agents.size();
    return aggregate;
  }

  std::size_t directedNeighborMessages = 0;
  for (const auto &agent : agents) {
    directedNeighborMessages += agent->getNeighbors().size();
  }
  aggregate.scalarCount = directedNeighborMessages * static_cast<std::size_t>(2);
  aggregate.scalarCommMb = scalarMegabytes(aggregate.scalarCount);

  std::vector<double> selectedSteps(agents.size(), 0.0);
  double selectedStepSum = 0.0;
  std::size_t selectedRobotCount = 0;
  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    const LocalGradientCorrectionCurvatureProbe &selfProbe = probes[robot];
    if (!selfProbe.ok || !selfProbe.currentFinite) {
      continue;
    }
    double descentNumeratorSum = selfProbe.descentNumerator;
    double curvatureSum = selfProbe.curvature;
    double maxStep = selfProbe.maxStep;
    bool neighborhoodFinite = selfProbe.descentNumerator > 0.0 &&
                              selfProbe.curvature > 0.0 &&
                              selfProbe.maxStep > 0.0;
    for (const unsigned neighborID : agents[robot]->getNeighbors()) {
      if (neighborID >= probes.size()) {
        neighborhoodFinite = false;
        break;
      }
      const LocalGradientCorrectionCurvatureProbe &neighborProbe =
          probes[neighborID];
      if (!neighborProbe.ok || !neighborProbe.currentFinite ||
          neighborProbe.descentNumerator <= 0.0 ||
          neighborProbe.curvature <= 0.0 || neighborProbe.maxStep <= 0.0) {
        neighborhoodFinite = false;
        break;
      }
      descentNumeratorSum += neighborProbe.descentNumerator;
      curvatureSum += neighborProbe.curvature;
      maxStep = std::min(maxStep, neighborProbe.maxStep);
    }
    if (!neighborhoodFinite) {
      continue;
    }
    const double step =
        curvatureCauchyStep(descentNumeratorSum, curvatureSum, maxStep);
    if (step > 0.0) {
      selectedSteps[robot] = step;
      selectedStepSum += step;
      ++selectedRobotCount;
    }
  }

  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    if (selectedSteps[robot] <= 0.0) {
      ++aggregate.rejected;
      continue;
    }
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agents[robot]->applyLocalGradientCorrectionStep(
            selectedSteps[robot], accepted, rejected, true)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
  }
  aggregate.selectedStep =
      selectedRobotCount == 0
          ? 0.0
          : selectedStepSum / static_cast<double>(selectedRobotCount);
  return aggregate;
}

std::size_t maybeSparsifyDirectionPacket(Matrix &directionBlock,
                                         unsigned topKEntries) {
  const std::size_t denseScalarCount =
      static_cast<std::size_t>(directionBlock.size());
  if (topKEntries == 0 ||
      static_cast<std::size_t>(topKEntries) >= denseScalarCount ||
      static_cast<std::size_t>(topKEntries) * 2u >= denseScalarCount) {
    return denseScalarCount;
  }

  struct Entry {
    int row{0};
    int col{0};
    double magnitude{0.0};
    std::size_t order{0};
  };

  std::vector<Entry> entries;
  entries.reserve(denseScalarCount);
  std::size_t order = 0;
  for (int col = 0; col < directionBlock.cols(); ++col) {
    for (int row = 0; row < directionBlock.rows(); ++row) {
      entries.push_back(Entry{row, col, std::abs(directionBlock(row, col)),
                              order++});
    }
  }
  std::stable_sort(entries.begin(), entries.end(),
                   [](const Entry &lhs, const Entry &rhs) {
                     if (lhs.magnitude == rhs.magnitude) {
                       return lhs.order < rhs.order;
                     }
                     return lhs.magnitude > rhs.magnitude;
                   });

  Matrix sparse = Matrix::Zero(directionBlock.rows(), directionBlock.cols());
  for (unsigned idx = 0; idx < topKEntries; ++idx) {
    sparse(entries[idx].row, entries[idx].col) =
        directionBlock(entries[idx].row, entries[idx].col);
  }
  directionBlock = sparse;
  return static_cast<std::size_t>(topKEntries) * 2u;
}

LocalGradientCorrectionAggregate
applyCoupledDirectionCurvatureLocalGradientCorrection(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    const CommunicationRound *allowedEdges = nullptr,
    unsigned maxRelayHops = 0,
    const std::string &interfacePayload = "direction_block",
    bool requireDecrease = true) {
  LocalGradientCorrectionAggregate aggregate;
  if (agents.empty()) {
    return aggregate;
  }
  const bool sendsDirection =
      interfaceModelPayloadHasDirection(interfacePayload);
  const bool sendsDiagStiffness =
      interfaceModelPayloadHasDiagStiffness(interfacePayload);

  std::vector<Matrix> baseDirections(agents.size());
  if (sendsDirection) {
    for (std::size_t robot = 0; robot < agents.size(); ++robot) {
      if (!agents[robot]->evaluateLocalGradientCorrectionBaseDirection(
              baseDirections[robot])) {
        ++aggregate.failures;
      }
    }
    if (aggregate.failures > 0) {
      aggregate.rejected = agents.size();
      return aggregate;
    }
  }

  std::vector<PoseDict> receiverDirectionPackets(agents.size());
  std::size_t packetCount = 0;
  std::size_t skippedPacketCount = 0;
  std::size_t scoreSkippedPacketCount = 0;
  std::size_t byteBudgetSkippedPacketCount = 0;
  std::size_t scalarCount = 0;
  for (std::size_t receiver = 0; receiver < agents.size(); ++receiver) {
    struct Candidate {
      unsigned senderID{0};
      unsigned poseIndex{0};
      Matrix direction;
      double score{0.0};
      std::size_t payloadScalarCount{0};
      std::size_t order{0};
      unsigned hopCount{1};
    };

    std::vector<Candidate> candidates;
    std::size_t order = 0;
    for (const unsigned senderID : agents[receiver]->getNeighbors()) {
      if (senderID >= agents.size()) {
        continue;
      }
      const unsigned hopCount = communicationRoundHopCount(
          allowedEdges, static_cast<unsigned>(receiver), senderID,
          maxRelayHops, static_cast<unsigned>(agents.size()));
      if (hopCount == 0) {
        skippedPacketCount +=
            agents[receiver]->getNeighborPublicPoses(senderID).size();
        continue;
      }
      for (const unsigned poseIndex :
           agents[receiver]->getNeighborPublicPoses(senderID)) {
        Matrix directionBlock;
        bool haveDirection = true;
        if (sendsDirection) {
          haveDirection = agents[senderID]->getSharedDirection(
              baseDirections[senderID], poseIndex, directionBlock);
        }
        if (haveDirection) {
          double score =
              agents[receiver]->getNeighborPoseSensitivity(senderID,
                                                           poseIndex);
          std::size_t payloadScalarCount = sendsDiagStiffness ? 1u : 0u;
          if (sendsDirection) {
            score *= directionBlock.norm();
            payloadScalarCount += maybeSparsifyDirectionPacket(
                directionBlock,
                agents[receiver]->getCoupledDirectionTopKEntries());
          }
          candidates.push_back(
              Candidate{senderID, poseIndex, directionBlock,
                        std::isfinite(score) ? score : 0.0,
                        payloadScalarCount, order++, hopCount});
        }
      }
    }
    const double minScore = agents[receiver]->getCoupledDirectionMinScore();
    const double minScoreRatio =
        agents[receiver]->getCoupledDirectionMinScoreRatio();
    if (!candidates.empty() && (minScore > 0.0 || minScoreRatio > 0.0)) {
      double maxScore = 0.0;
      for (const Candidate &candidate : candidates) {
        maxScore = std::max(maxScore, candidate.score);
      }
      const double ratioThreshold =
          maxScore > 0.0 ? maxScore * minScoreRatio : 0.0;
      const double threshold = std::max(minScore, ratioThreshold);
      std::vector<Candidate> filtered;
      filtered.reserve(candidates.size());
      for (const Candidate &candidate : candidates) {
        if (candidate.score >= threshold) {
          filtered.push_back(candidate);
        }
      }
      scoreSkippedPacketCount += candidates.size() - filtered.size();
      candidates = std::move(filtered);
    }
    std::size_t sendLimit = candidates.size();
    const double budgetFraction =
        agents[receiver]->getCoupledDirectionBudgetFraction();
    if (budgetFraction < 1.0) {
      sendLimit = std::min(
          sendLimit,
          static_cast<std::size_t>(
              std::ceil(budgetFraction *
                        static_cast<double>(candidates.size()))));
    }
    const unsigned maxPackets =
        agents[receiver]->getCoupledDirectionMaxPacketsPerReceiver();
    if (maxPackets > 0) {
      sendLimit = std::min(
          sendLimit, static_cast<std::size_t>(maxPackets));
    }
    const double byteBudgetMb =
        agents[receiver]->getCoupledDirectionByteBudgetMb();
    const bool useByteBudget = byteBudgetMb > 0.0;
    if (sendLimit < candidates.size() || useByteBudget) {
      std::stable_sort(candidates.begin(), candidates.end(),
                       [](const Candidate &lhs, const Candidate &rhs) {
                         if (lhs.score == rhs.score) {
                           return lhs.order < rhs.order;
                         }
                         return lhs.score > rhs.score;
                       });
    }
    skippedPacketCount += candidates.size() - sendLimit;
    std::size_t scalarBudget = std::numeric_limits<std::size_t>::max();
    if (useByteBudget) {
      const double rawScalarBudget =
          byteBudgetMb * 1024.0 * 1024.0 / static_cast<double>(sizeof(double));
      scalarBudget =
          rawScalarBudget <= 0.0
              ? 0u
              : static_cast<std::size_t>(std::floor(rawScalarBudget + 1e-12));
    }
    std::size_t receiverScalarCount = 0;
    for (std::size_t idx = 0; idx < sendLimit; ++idx) {
      const Candidate &candidate = candidates[idx];
      const std::size_t hopWeightedScalarCount =
          candidate.payloadScalarCount *
          static_cast<std::size_t>(std::max(1u, candidate.hopCount));
      if (useByteBudget &&
          (hopWeightedScalarCount > scalarBudget ||
           receiverScalarCount >
               scalarBudget - hopWeightedScalarCount)) {
        ++byteBudgetSkippedPacketCount;
        continue;
      }
      if (sendsDirection) {
        receiverDirectionPackets[receiver][std::make_pair(
            candidate.senderID, candidate.poseIndex)] = candidate.direction;
      }
      ++packetCount;
      scalarCount += hopWeightedScalarCount;
      receiverScalarCount += hopWeightedScalarCount;
    }
  }
  skippedPacketCount += scoreSkippedPacketCount + byteBudgetSkippedPacketCount;
  aggregate.scalarCount = scalarCount;
  aggregate.scalarCommMb = scalarMegabytes(scalarCount);
  aggregate.diagnostics.coupledDirectionPacketCount = packetCount;
  aggregate.diagnostics.coupledDirectionSkippedPacketCount =
      skippedPacketCount;
  aggregate.diagnostics.coupledDirectionScoreSkippedPacketCount =
      scoreSkippedPacketCount;
  aggregate.diagnostics.coupledDirectionByteBudgetSkippedPacketCount =
      byteBudgetSkippedPacketCount;
  aggregate.diagnostics.coupledDirectionScalarCount = scalarCount;

  if (!sendsDirection) {
    return aggregate;
  }

  double selectedStepSum = 0.0;
  std::size_t selectedRobotCount = 0;
  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    LocalGradientCorrectionCurvatureProbe probe;
    Matrix direction;
    if (!agents[robot]->evaluateLocalGradientCorrectionCoupledCurvature(
            receiverDirectionPackets[robot], probe, direction)) {
      ++aggregate.failures;
      continue;
    }
    ++aggregate.diagnostics.coupledDirectionCandidateCount;
    const double step = curvatureCauchyStep(
        probe.descentNumerator, probe.curvature, probe.maxStep);
    if (!probe.ok || !probe.currentFinite || step <= 0.0 ||
        direction.rows() == 0) {
      ++aggregate.rejected;
      ++aggregate.diagnostics.coupledDirectionRejectedCount;
      continue;
    }
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agents[robot]->applyLocalGradientCorrectionDirectionStep(
            direction, step, accepted, rejected, requireDecrease)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
    aggregate.diagnostics.coupledDirectionAcceptedCount += accepted;
    aggregate.diagnostics.coupledDirectionRejectedCount += rejected;
    selectedStepSum += step;
    ++selectedRobotCount;
  }
  aggregate.selectedStep =
      selectedRobotCount == 0
          ? 0.0
          : selectedStepSum / static_cast<double>(selectedRobotCount);
  return aggregate;
}

LocalGradientCorrectionAggregate applyNeighborhoodStepLocalGradientCorrection(
    std::vector<std::unique_ptr<ManualDpgoMmAgent>> &agents,
    const std::vector<double> &steps) {
  LocalGradientCorrectionAggregate aggregate;
  if (steps.empty() || agents.empty()) {
    return aggregate;
  }

  struct RobotMerit {
    bool ok{false};
    bool currentFinite{false};
    double currentCost{0.0};
    std::vector<double> trialCosts;
  };

  std::vector<RobotMerit> robotMerits(agents.size());
  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    double currentCost = 0.0;
    std::vector<double> trialCosts;
    if (!agents[robot]->evaluateLocalGradientCorrectionCosts(
            steps, currentCost, trialCosts)) {
      ++aggregate.failures;
      continue;
    }
    robotMerits[robot].ok = true;
    robotMerits[robot].currentFinite = std::isfinite(currentCost);
    robotMerits[robot].currentCost = currentCost;
    robotMerits[robot].trialCosts = std::move(trialCosts);
  }
  if (aggregate.failures > 0) {
    aggregate.rejected = agents.size() * steps.size();
    return aggregate;
  }

  std::size_t directedNeighborMessages = 0;
  for (const auto &agent : agents) {
    directedNeighborMessages += agent->getNeighbors().size();
  }
  aggregate.scalarCount =
      directedNeighborMessages * (steps.size() + static_cast<std::size_t>(2));
  aggregate.scalarCommMb = scalarMegabytes(aggregate.scalarCount);

  std::vector<std::size_t> selectedStepIndex(agents.size(), steps.size());
  double selectedStepSum = 0.0;
  std::size_t selectedRobotCount = 0;
  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    const RobotMerit &selfMerit = robotMerits[robot];
    if (!selfMerit.ok || !selfMerit.currentFinite) {
      continue;
    }
    double neighborhoodCurrentCost = selfMerit.currentCost;
    std::vector<double> neighborhoodTrialCosts(
        steps.size(), std::numeric_limits<double>::infinity());
    for (std::size_t stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
      if (stepIdx < selfMerit.trialCosts.size() &&
          std::isfinite(selfMerit.trialCosts[stepIdx])) {
        neighborhoodTrialCosts[stepIdx] = selfMerit.trialCosts[stepIdx];
      }
    }

    bool neighborhoodFinite = true;
    for (const unsigned neighborID : agents[robot]->getNeighbors()) {
      if (neighborID >= robotMerits.size()) {
        neighborhoodFinite = false;
        break;
      }
      const RobotMerit &neighborMerit = robotMerits[neighborID];
      if (!neighborMerit.ok || !neighborMerit.currentFinite) {
        neighborhoodFinite = false;
        break;
      }
      neighborhoodCurrentCost += neighborMerit.currentCost;
      for (std::size_t stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
        if (!std::isfinite(neighborhoodTrialCosts[stepIdx]) ||
            stepIdx >= neighborMerit.trialCosts.size() ||
            !std::isfinite(neighborMerit.trialCosts[stepIdx])) {
          neighborhoodTrialCosts[stepIdx] =
              std::numeric_limits<double>::infinity();
        } else {
          neighborhoodTrialCosts[stepIdx] +=
              neighborMerit.trialCosts[stepIdx];
        }
      }
    }
    if (!neighborhoodFinite || !std::isfinite(neighborhoodCurrentCost)) {
      continue;
    }

    std::size_t bestStepIdx = steps.size();
    double bestCost = neighborhoodCurrentCost;
    for (std::size_t stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
      if (std::isfinite(neighborhoodTrialCosts[stepIdx]) &&
          neighborhoodTrialCosts[stepIdx] < bestCost - 1e-12) {
        bestStepIdx = stepIdx;
        bestCost = neighborhoodTrialCosts[stepIdx];
      }
    }
    if (bestStepIdx != steps.size()) {
      selectedStepIndex[robot] = bestStepIdx;
      selectedStepSum += steps[bestStepIdx];
      ++selectedRobotCount;
    }
  }

  for (std::size_t robot = 0; robot < agents.size(); ++robot) {
    if (selectedStepIndex[robot] == steps.size()) {
      aggregate.rejected += steps.size();
      continue;
    }
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    if (!agents[robot]->applyLocalGradientCorrectionStep(
            steps[selectedStepIndex[robot]], accepted, rejected)) {
      ++aggregate.failures;
    }
    aggregate.accepted += accepted;
    aggregate.rejected += rejected;
    aggregate.rejected += steps.size() - 1;
  }
  aggregate.selectedStep =
      selectedRobotCount == 0
          ? 0.0
          : selectedStepSum / static_cast<double>(selectedRobotCount);
  return aggregate;
}

void throwOnLocalFailures(const std::string &stage, unsigned iter,
                          std::size_t failures) {
  if (failures == 0) {
    return;
  }
  throw std::runtime_error("manual_dpgo_mm " + stage + " failed at iter " +
                           std::to_string(iter) + " for " +
                           std::to_string(failures) + " robot(s)");
}

}  // namespace

ManualDpgoMmScheme parseManualDpgoMmScheme(const std::string &value,
                                           bool accelerated) {
  const std::string lower = lowercase(value);
  if (lower == "auto") {
    return accelerated ? ManualDpgoMmScheme::AMM : ManualDpgoMmScheme::MM;
  }
  if (lower == "amm" || lower == "accelerated") {
    return ManualDpgoMmScheme::AMM;
  }
  if (lower == "mm" || lower == "plain") {
    return ManualDpgoMmScheme::MM;
  }
  throw std::invalid_argument("Unknown manual DPGO-MM scheme: " + value);
}

ManualDpgoMmLocalSolver
parseManualDpgoMmLocalSolver(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "manual_full" || lower == "full" ||
      lower == "full_pose" || lower == "manual_tr" ||
      lower == "manual_newton") {
    return ManualDpgoMmLocalSolver::ManualFull;
  }
  if (lower == "full_equiv_hybrid" || lower == "full_equivalent_hybrid" ||
      lower == "fe_hybrid" || lower == "fe-hybrid") {
    return ManualDpgoMmLocalSolver::FullEquivHybrid;
  }
  if (lower == "reduced_rotation" || lower == "translation_eliminated" ||
      lower == "reduced" || lower == "reduced_rotation_newton") {
    return ManualDpgoMmLocalSolver::ReducedRotation;
  }
  throw std::invalid_argument("Unknown manual DPGO-MM local solver: " + value);
}

ManualDpgoMmReducedRotationPreconditioner
parseManualDpgoMmReducedRotationPreconditioner(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "none" || lower == "off" || lower == "false") {
    return ManualDpgoMmReducedRotationPreconditioner::None;
  }
  if (lower == "jacobi" || lower == "diagonal") {
    return ManualDpgoMmReducedRotationPreconditioner::Jacobi;
  }
  if (lower == "schur_jacobi" || lower == "reduced_schur_jacobi" ||
      lower == "translation_schur_jacobi" ||
      lower == "schur-jacobi" || lower == "reduced-schur-jacobi") {
    return ManualDpgoMmReducedRotationPreconditioner::SchurJacobi;
  }
  if (lower == "cholesky" || lower == "regularized_cholesky" ||
      lower == "regularized-cholesky") {
    return ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  }
  if (lower == "portfolio" || lower == "best_of_none_jacobi" ||
      lower == "best-of-none-jacobi" ||
      lower == "best_of_none_jacobi_cholesky" ||
      lower == "best-of-none-jacobi-cholesky") {
    return ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  }
  if (lower == "adaptive_portfolio" || lower == "adaptive-portfolio" ||
      lower == "gated_portfolio" || lower == "gated-portfolio" ||
      lower == "selective_portfolio" || lower == "selective-portfolio") {
    return ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  }
  throw std::invalid_argument(
      "Unknown manual DPGO-MM reduced preconditioner: " + value);
}

ManualDpgoMmReducedSurrogateMode
parseManualDpgoMmReducedSurrogateMode(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "true_local" || lower == "true-local" ||
      lower == "local" || lower == "none" || lower == "off" ||
      lower == "false") {
    return ManualDpgoMmReducedSurrogateMode::TrueLocal;
  }
  if (lower == "dpgo_simple" || lower == "dpgo-simple" ||
      lower == "simple") {
    return ManualDpgoMmReducedSurrogateMode::DpgoSimple;
  }
  if (lower == "edge_tight_quadratic" ||
      lower == "edge-tight-quadratic" || lower == "edge_tight" ||
      lower == "edge-tight") {
    return ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic;
  }
  throw std::invalid_argument(
      "Unknown manual DPGO-MM reduced surrogate mode: " + value);
}

ManualDpgoMmSurrogateMode
parseManualDpgoMmSurrogateMode(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "legacy" || lower == "none" || lower == "off" ||
      lower == "false") {
    return ManualDpgoMmSurrogateMode::Legacy;
  }
  if (lower == "weighted_edge_split" ||
      lower == "weighted-edge-split" || lower == "edge_split" ||
      lower == "edge-split") {
    return ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  }
  if (lower == "adaptive_spectral" || lower == "adaptive-spectral" ||
      lower == "spectral") {
    return ManualDpgoMmSurrogateMode::AdaptiveSpectral;
  }
  if (lower == "variable_projected_schur" ||
      lower == "variable-projected-schur" ||
      lower == "projected_schur" || lower == "projected-schur") {
    return ManualDpgoMmSurrogateMode::VariableProjectedSchur;
  }
  throw std::invalid_argument("Unknown manual DPGO-MM surrogate mode: " +
                              value);
}

ManualDpgoMmEdgeSplitThetaMode
parseManualDpgoMmEdgeSplitThetaMode(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "constant" || lower == "fixed") {
    return ManualDpgoMmEdgeSplitThetaMode::Constant;
  }
  if (lower == "degree" || lower == "degree_balanced" ||
      lower == "degree-balanced") {
    return ManualDpgoMmEdgeSplitThetaMode::Degree;
  }
  if (lower == "curvature" || lower == "curvature_balanced" ||
      lower == "curvature-balanced") {
    return ManualDpgoMmEdgeSplitThetaMode::Curvature;
  }
  if (lower == "adaptive_conditioned" ||
      lower == "adaptive-conditioned" || lower == "conditioned") {
    return ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned;
  }
  throw std::invalid_argument(
      "Unknown manual DPGO-MM edge-split theta mode: " + value);
}

ManualDpgoMmMmAcceleratorMode
parseManualDpgoMmMmAcceleratorMode(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "none" || lower == "off" || lower == "false") {
    return ManualDpgoMmMmAcceleratorMode::None;
  }
  if (lower == "nesterov_legacy" || lower == "nesterov-legacy" ||
      lower == "legacy" || lower == "nesterov") {
    return ManualDpgoMmMmAcceleratorMode::NesterovLegacy;
  }
  if (lower == "anderson" || lower == "aa") {
    return ManualDpgoMmMmAcceleratorMode::Anderson;
  }
  if (lower == "squarem" || lower == "squared_extrapolation" ||
      lower == "squared-extrapolation") {
    return ManualDpgoMmMmAcceleratorMode::Squarem;
  }
  throw std::invalid_argument(
      "Unknown manual DPGO-MM MM accelerator mode: " + value);
}

ManualDpgoMmMmSafeguard
parseManualDpgoMmMmSafeguard(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "local_surrogate" || lower == "local-surrogate" ||
      lower == "local") {
    return ManualDpgoMmMmSafeguard::LocalSurrogate;
  }
  if (lower == "local_surrogate_plus_boundary" ||
      lower == "local-surrogate-plus-boundary" ||
      lower == "plus_boundary" || lower == "plus-boundary") {
    return ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary;
  }
  if (lower == "debug_global" || lower == "debug-global" ||
      lower == "global") {
    return ManualDpgoMmMmSafeguard::DebugGlobal;
  }
  throw std::invalid_argument("Unknown manual DPGO-MM MM safeguard: " +
                              value);
}

ManualDpgoMmFullEquivHybridBackend
parseManualDpgoMmFullEquivHybridBackend(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "manual_full_rtr" || lower == "manual_full" ||
      lower == "manual" || lower == "rtr" || lower == "full_rtr") {
    return ManualDpgoMmFullEquivHybridBackend::ManualFullRtr;
  }
  if (lower == "sparse_direct_schur" || lower == "direct_schur" ||
      lower == "schur" || lower == "sparse-direct-schur" ||
      lower == "direct-schur") {
    return ManualDpgoMmFullEquivHybridBackend::SparseDirectSchur;
  }
  if (lower == "pcg_schur" || lower == "pcg-schur" ||
      lower == "schur_pcg" || lower == "schur-pcg") {
    return ManualDpgoMmFullEquivHybridBackend::PcgSchur;
  }
  if (lower == "pcg_full" || lower == "pcg-full" ||
      lower == "full_pcg" || lower == "full-pcg") {
    return ManualDpgoMmFullEquivHybridBackend::PcgFull;
  }
  throw std::invalid_argument(
      "Unknown FE-Hybrid linear backend: " + value);
}

ManualDpgoMmFullEquivHybridSchwarzBlockMode
parseManualDpgoMmFullEquivHybridSchwarzBlockMode(const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "per_pose" || lower == "per-pose" || lower == "pose" ||
      lower == "single_pose" || lower == "single-pose") {
    return ManualDpgoMmFullEquivHybridSchwarzBlockMode::PerPose;
  }
  if (lower == "edge_pair" || lower == "edge-pair" || lower == "pair" ||
      lower == "pose_pair" || lower == "pose-pair") {
    return ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair;
  }
  throw std::invalid_argument("Unknown FE-Hybrid Schwarz block mode: " +
                              value);
}

ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode
parseManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode(
    const std::string &value) {
  const std::string lower = lowercase(value);
  if (lower == "gradient" || lower == "grad" || lower == "norm") {
    return ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode::
        Gradient;
  }
  if (lower == "model_decrease" || lower == "model-decrease" ||
      lower == "cauchy" || lower == "predicted_decrease" ||
      lower == "predicted-decrease") {
    return ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode::
        ModelDecrease;
  }
  throw std::invalid_argument(
      "Unknown FE-Hybrid active-separator LM-Schur score mode: " + value);
}

std::string manualDpgoMmSchemeName(ManualDpgoMmScheme scheme) {
  return scheme == ManualDpgoMmScheme::AMM ? "amm" : "mm";
}

std::string manualDpgoMmLocalSolverName(ManualDpgoMmLocalSolver solver) {
  if (solver == ManualDpgoMmLocalSolver::ReducedRotation) {
    return "reduced_rotation";
  }
  if (solver == ManualDpgoMmLocalSolver::FullEquivHybrid) {
    return "full_equiv_hybrid";
  }
  return "manual_full";
}

std::string manualDpgoMmFullEquivHybridBackendName(
    ManualDpgoMmFullEquivHybridBackend backend) {
  if (backend == ManualDpgoMmFullEquivHybridBackend::SparseDirectSchur) {
    return "sparse_direct_schur";
  }
  if (backend == ManualDpgoMmFullEquivHybridBackend::PcgSchur) {
    return "pcg_schur";
  }
  if (backend == ManualDpgoMmFullEquivHybridBackend::PcgFull) {
    return "pcg_full";
  }
  return "manual_full_rtr";
}

std::string manualDpgoMmReducedRotationPreconditionerName(
    ManualDpgoMmReducedRotationPreconditioner preconditioner) {
  if (preconditioner == ManualDpgoMmReducedRotationPreconditioner::Jacobi) {
    return "jacobi";
  }
  if (preconditioner ==
      ManualDpgoMmReducedRotationPreconditioner::SchurJacobi) {
    return "schur_jacobi";
  }
  if (preconditioner == ManualDpgoMmReducedRotationPreconditioner::Cholesky) {
    return "cholesky";
  }
  if (preconditioner ==
      ManualDpgoMmReducedRotationPreconditioner::Portfolio) {
    return "portfolio";
  }
  if (preconditioner ==
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio) {
    return "adaptive_portfolio";
  }
  return "none";
}

std::string manualDpgoMmReducedSurrogateModeName(
    ManualDpgoMmReducedSurrogateMode mode) {
  if (mode == ManualDpgoMmReducedSurrogateMode::DpgoSimple) {
    return "dpgo_simple";
  }
  if (mode == ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic) {
    return "edge_tight_quadratic";
  }
  return "true_local";
}

std::string manualDpgoMmSurrogateModeName(ManualDpgoMmSurrogateMode mode) {
  if (mode == ManualDpgoMmSurrogateMode::WeightedEdgeSplit) {
    return "weighted_edge_split";
  }
  if (mode == ManualDpgoMmSurrogateMode::AdaptiveSpectral) {
    return "adaptive_spectral";
  }
  if (mode == ManualDpgoMmSurrogateMode::VariableProjectedSchur) {
    return "variable_projected_schur";
  }
  return "legacy";
}

std::string manualDpgoMmEdgeSplitThetaModeName(
    ManualDpgoMmEdgeSplitThetaMode mode) {
  if (mode == ManualDpgoMmEdgeSplitThetaMode::Degree) {
    return "degree";
  }
  if (mode == ManualDpgoMmEdgeSplitThetaMode::Curvature) {
    return "curvature";
  }
  if (mode == ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned) {
    return "adaptive_conditioned";
  }
  return "constant";
}

std::string manualDpgoMmMmAcceleratorModeName(
    ManualDpgoMmMmAcceleratorMode mode) {
  if (mode == ManualDpgoMmMmAcceleratorMode::None) {
    return "none";
  }
  if (mode == ManualDpgoMmMmAcceleratorMode::Anderson) {
    return "anderson";
  }
  if (mode == ManualDpgoMmMmAcceleratorMode::Squarem) {
    return "squarem";
  }
  return "nesterov_legacy";
}

std::string manualDpgoMmMmSafeguardName(ManualDpgoMmMmSafeguard mode) {
  if (mode == ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary) {
    return "local_surrogate_plus_boundary";
  }
  if (mode == ManualDpgoMmMmSafeguard::DebugGlobal) {
    return "debug_global";
  }
  return "local_surrogate";
}

std::string manualDpgoMmFullEquivHybridSchwarzBlockModeName(
    ManualDpgoMmFullEquivHybridSchwarzBlockMode mode) {
  if (mode == ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair) {
    return "edge_pair";
  }
  return "per_pose";
}

std::string manualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreModeName(
    ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode mode) {
  if (mode ==
      ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode::
          ModelDecrease) {
    return "model_decrease";
  }
  return "gradient";
}

std::size_t manualDpgoMmPosePayloadBytes(unsigned dimension) {
  return static_cast<std::size_t>(dimension + 1) *
         static_cast<std::size_t>(dimension) * sizeof(double);
}

ManualDpgoMmQuadraticStats
evaluateManualDpgoMmQuadraticStats(const QuadraticProblem &problem,
                                   const Matrix &X) {
  const auto fused = problem.fAndRieGradNorm(X);
  ManualDpgoMmQuadraticStats stats;
  stats.globalCost = 2.0 * fused.first;
  stats.gradient = 2.0 * fused.second;
  return stats;
}

ManualDpgoMmWeightedEdgeSplitStats evaluateManualDpgoMmWeightedEdgeSplit(
    const Matrix &a, const Matrix &b, const Matrix &ak, const Matrix &bk,
    double theta) {
  if (a.rows() != b.rows() || a.cols() != b.cols() ||
      a.rows() != ak.rows() || a.cols() != ak.cols() ||
      a.rows() != bk.rows() || a.cols() != bk.cols()) {
    throw std::invalid_argument(
        "Weighted edge-split inputs must have matching dimensions");
  }
  if (!std::isfinite(theta) || theta <= 0.0 || theta >= 1.0) {
    throw std::invalid_argument(
        "Weighted edge-split theta must be finite and in (0, 1)");
  }

  const Matrix anchor =
      (1.0 - theta) * ak + theta * bk;
  const Matrix currentWeightedMean =
      (1.0 - theta) * a + theta * b;

  ManualDpgoMmWeightedEdgeSplitStats stats;
  stats.objective = 0.5 * (a - b).squaredNorm();
  stats.surrogate =
      0.5 / theta * (a - anchor).squaredNorm() +
      0.5 / (1.0 - theta) * (b - anchor).squaredNorm();
  stats.gap = stats.surrogate - stats.objective;
  stats.identityGap =
      0.5 / (theta * (1.0 - theta)) *
      (anchor - currentWeightedMean).squaredNorm();
  return stats;
}

double evaluateManualDpgoMmDegreeEdgeSplitTheta(double degreeAlpha,
                                                double degreeBeta,
                                                double thetaMin,
                                                double thetaMax) {
  if (!std::isfinite(thetaMin) || !std::isfinite(thetaMax) ||
      thetaMin <= 0.0 || thetaMax >= 1.0 || thetaMin >= thetaMax ||
      !std::isfinite(degreeAlpha) || !std::isfinite(degreeBeta) ||
      degreeAlpha <= 0.0 || degreeBeta <= 0.0) {
    return 0.5;
  }
  const double alpha = std::sqrt(degreeAlpha);
  const double beta = std::sqrt(degreeBeta);
  const double denom = alpha + beta;
  if (!std::isfinite(denom) || denom <= 0.0) {
    return 0.5;
  }
  const double theta = alpha / denom;
  if (!std::isfinite(theta)) {
    return 0.5;
  }
  return std::min(thetaMax, std::max(thetaMin, theta));
}

double evaluateManualDpgoMmCurvatureEdgeSplitTheta(double curvatureAlpha,
                                                   double curvatureBeta,
                                                   double thetaMin,
                                                   double thetaMax) {
  return evaluateManualDpgoMmDegreeEdgeSplitTheta(
      curvatureAlpha, curvatureBeta, thetaMin, thetaMax);
}

double evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
    double degreeAlpha, double degreeBeta, double curvatureAlpha,
    double curvatureBeta, const std::vector<double> &thetaCandidates,
    double thetaMin, double thetaMax) {
  if (!std::isfinite(thetaMin) || !std::isfinite(thetaMax) ||
      thetaMin <= 0.0 || thetaMax >= 1.0 || thetaMin >= thetaMax) {
    thetaMin = 0.15;
    thetaMax = 0.85;
  }

  const bool degreeValid = std::isfinite(degreeAlpha) &&
                           std::isfinite(degreeBeta) && degreeAlpha > 0.0 &&
                           degreeBeta > 0.0;
  const bool curvatureValid =
      std::isfinite(curvatureAlpha) && std::isfinite(curvatureBeta) &&
      curvatureAlpha > 0.0 && curvatureBeta > 0.0;
  if (!degreeValid && !curvatureValid) {
    return 0.5;
  }
  if (!degreeValid) {
    return evaluateManualDpgoMmCurvatureEdgeSplitTheta(
        curvatureAlpha, curvatureBeta, thetaMin, thetaMax);
  }
  if (!curvatureValid) {
    return evaluateManualDpgoMmDegreeEdgeSplitTheta(degreeAlpha, degreeBeta,
                                                   thetaMin, thetaMax);
  }

  const double alphaLoad = std::sqrt(degreeAlpha * curvatureAlpha);
  const double betaLoad = std::sqrt(degreeBeta * curvatureBeta);
  if (!std::isfinite(alphaLoad) || !std::isfinite(betaLoad) ||
      alphaLoad <= 0.0 || betaLoad <= 0.0) {
    return evaluateManualDpgoMmDegreeEdgeSplitTheta(degreeAlpha, degreeBeta,
                                                   thetaMin, thetaMax);
  }

  double bestTheta = 0.5;
  double bestScore = std::numeric_limits<double>::infinity();
  bool found = false;
  for (const double rawTheta : thetaCandidates) {
    if (!std::isfinite(rawTheta)) {
      continue;
    }
    const double theta = std::min(thetaMax, std::max(thetaMin, rawTheta));
    if (theta <= 0.0 || theta >= 1.0) {
      continue;
    }
    const double alphaScore = alphaLoad / theta;
    const double betaScore = betaLoad / (1.0 - theta);
    const double score = std::max(alphaScore, betaScore);
    if (!std::isfinite(score)) {
      continue;
    }
    const double tieBreak = std::abs(theta - 0.5);
    const double bestTieBreak = std::abs(bestTheta - 0.5);
    if (!found || score < bestScore - 1e-12 ||
        (std::abs(score - bestScore) <= 1e-12 &&
         tieBreak < bestTieBreak)) {
      found = true;
      bestTheta = theta;
      bestScore = score;
    }
  }

  if (!found) {
    return evaluateManualDpgoMmDegreeEdgeSplitTheta(degreeAlpha, degreeBeta,
                                                   thetaMin, thetaMax);
  }
  return bestTheta;
}

double evaluateManualDpgoMmScalarYoungMajorizerGap(
    const Matrix &deltaAlpha, const Matrix &deltaBeta,
    const Matrix &crossBlock, double eta) {
  if (deltaAlpha.rows() != deltaBeta.rows() ||
      deltaAlpha.cols() != crossBlock.rows() ||
      deltaBeta.cols() != crossBlock.cols()) {
    throw std::invalid_argument(
        "Scalar Young majorizer inputs have incompatible dimensions");
  }
  if (!deltaAlpha.allFinite() || !deltaBeta.allFinite() ||
      !crossBlock.allFinite() || !std::isfinite(eta) || eta <= 0.0) {
    throw std::invalid_argument(
        "Scalar Young majorizer inputs must be finite and eta must be positive");
  }

  const double sigma = denseSpectralNorm(crossBlock);
  const double alphaTerm = eta * deltaAlpha.squaredNorm();
  const double betaTerm =
      sigma > 0.0 ? (sigma * sigma / eta) * deltaBeta.squaredNorm() : 0.0;
  const double crossTerm =
      2.0 * (deltaAlpha * crossBlock).cwiseProduct(deltaBeta).sum();
  const double gap = 0.5 * (alphaTerm + betaTerm + crossTerm);
  return std::isfinite(gap) ? gap : -std::numeric_limits<double>::infinity();
}

double evaluateManualDpgoMmBlockGershgorinMajorizerGap(
    const Matrix &delta, const Matrix &schur, unsigned blockDim) {
  if (blockDim == 0 || schur.rows() != schur.cols() ||
      schur.rows() == 0 || schur.rows() % static_cast<int>(blockDim) != 0 ||
      delta.cols() != schur.cols()) {
    throw std::invalid_argument(
        "Block Gershgorin majorizer inputs have incompatible dimensions");
  }
  if (!delta.allFinite() || !schur.allFinite()) {
    throw std::invalid_argument(
        "Block Gershgorin majorizer inputs must be finite");
  }

  const int numBlocks = schur.rows() / static_cast<int>(blockDim);
  Matrix majorized = Matrix::Zero(schur.rows(), schur.cols());
  for (int block = 0; block < numBlocks; ++block) {
    const int colStart = block * static_cast<int>(blockDim);
    Matrix diagonal = schur.block(colStart, colStart,
                                  static_cast<int>(blockDim),
                                  static_cast<int>(blockDim));
    diagonal = (0.5 * (diagonal + diagonal.transpose())).eval();
    double offBlockNormSum = 0.0;
    for (int other = 0; other < numBlocks; ++other) {
      if (other == block) {
        continue;
      }
      const int otherStart = other * static_cast<int>(blockDim);
      offBlockNormSum += denseSpectralNorm(
          schur.block(colStart, otherStart, static_cast<int>(blockDim),
                      static_cast<int>(blockDim)));
    }
    diagonal.diagonal().array() += offBlockNormSum;
    majorized.block(colStart, colStart, static_cast<int>(blockDim),
                    static_cast<int>(blockDim)) = diagonal;
  }

  const Matrix residual = majorized - schur;
  const double gap =
      0.5 * (delta * residual).cwiseProduct(delta).sum();
  return std::isfinite(gap) ? gap : -std::numeric_limits<double>::infinity();
}

ManualDpgoMmRunResult runManualDpgoMm(const std::string &datasetPath,
                                      const ManualDpgoMmOptions &options) {
  ManualDpgoMmOptimizerProfile optimizerProfile;
  const auto profileTotalStart =
      options.profileOptimizer ? ProfileClock::now()
                               : ProfileClock::time_point();
  const auto prepareProfileStart =
      options.profileOptimizer ? ProfileClock::now()
                               : ProfileClock::time_point();
  ManualDpgoMmPreparedProblem prepared =
      prepareManualDpgoMmProblem(datasetPath, options, true);
  if (options.profileOptimizer) {
    optimizerProfile.prepareProblemSec += secondsSince(prepareProfileStart);
  }
  auto &agents = prepared.agents;
  const unsigned numPoses = prepared.numPoses;
  const unsigned d = prepared.d;
  const unsigned r = prepared.r;
  const unsigned posesPerRobot = prepared.posesPerRobot;
  QuadraticProblem &centralProblem = *prepared.centralProblem;
  const CommunicationSchedule communicationSchedule =
      readCommunicationSchedule(options.communicationTopologyFile,
                                options.numRobots);
  auto topologyRound = [&](unsigned round) -> const CommunicationRound * {
    return communicationRoundFor(communicationSchedule, round);
  };
  const bool valueSchedulerUseResidualProxy =
      options.communicationTopologyValueSchedulerMode ==
      "boundary_residual_staleness";

  ManualDpgoMmRunResult result;
  std::vector<ManualDpgoMmAmmTraceRow> ammTraceRows;
  DirectedPoseCache deliveredPoseCache;
  std::size_t cumulativeCommPoses = 0;
  double cumulativePosePayloadMb = 0.0;
  double cumulativeScalarCommMb = 0.0;
  double cumulativeValueSchedulerCommMb = 0.0;
  auto remainingValueSchedulerBudgetMb = [&]() {
    if (!options.communicationTopologyValueScheduler ||
        options.communicationTopologyValueByteBudgetMb <= 0.0) {
      return options.communicationTopologyValueByteBudgetMb;
    }
    return std::max(0.0,
                    options.communicationTopologyValueByteBudgetMb -
                        cumulativeValueSchedulerCommMb);
  };
  auto pacedValueSchedulerBudgetMb = [&](unsigned iter) {
    const double remaining = remainingValueSchedulerBudgetMb();
    if (!options.communicationTopologyValueScheduler ||
        options.communicationTopologyValueByteBudgetMb <= 0.0 ||
        remaining <= 0.0 ||
        !options.communicationTopologyValueBudgetPacing) {
      return remaining;
    }
    unsigned remainingPostExchanges = 0;
    for (unsigned futureIter = iter; futureIter < options.maxIterations;
         ++futureIter) {
      if (options.postExchangePeriod > 0 &&
          ((futureIter + 1) % options.postExchangePeriod == 0)) {
        ++remainingPostExchanges;
      }
    }
    if (remainingPostExchanges == 0) {
      return remaining;
    }
    return remaining / static_cast<double>(remainingPostExchanges);
  };
  auto recordInitializationEstimate = [&]() {
    if (!options.saveIterationEstimates) {
      return;
    }
    result.initializationEstimates.push_back(assembleGlobalEstimate(
        agents, numPoses, options.numRobots, posesPerRobot, d, r));
  };
  const bool hasExternalInitialEstimate =
      !options.externalInitialEstimatePath.empty();
  if (hasExternalInitialEstimate) {
    const Matrix externalInitialEstimate = readManualInterleavedEstimate(
        options.externalInitialEstimatePath, static_cast<int>(r),
        static_cast<int>(numPoses * (d + 1)));
    scatterGlobalEstimateToAgents(agents, externalInitialEstimate, numPoses,
                                  options.numRobots, posesPerRobot, d, r);
    resetLocalHistories(agents);
  }
  recordInitializationEstimate();
  double elapsed = 0.0;
  PoseExchangeAggregate initExchange;
  {
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer ? &optimizerProfile.initializationExchangeSec
                                 : nullptr);
    const CommunicationRound *initialTopology =
        options.communicationTopologyInitialFull ? nullptr : topologyRound(0);
    initExchange = exchangeOwnedSeparatorPoses(
        agents, deliveredPoseCache, initialTopology,
        options.communicationTopologyMaxRelayHops,
        options.communicationTopologyHopBudgetFraction,
        options.communicationTopologyStaleAwareRelayScore,
        options.communicationTopologyStaleAwareRelayAgeGain,
        options.communicationTopologyDeltaCompression,
        options.communicationTopologyDeltaMode,
        options.communicationTopologyDeltaTopK,
        options.communicationTopologyDeltaMaxReconstructionError,
        options.communicationTopologyLiftedDeltaCompression,
        options.communicationTopologyLiftedDeltaRank,
        options.communicationTopologyLiftedDeltaMaxReconstructionError,
        options.communicationTopologyValueScheduler,
        options.communicationTopologyValueByteBudgetMb,
        options.communicationTopologyValueMinScoreRatio,
        valueSchedulerUseResidualProxy);
  }
  std::size_t initCommPoses = initExchange.sent;
  cumulativeCommPoses += initExchange.sent;
  cumulativePosePayloadMb += initExchange.commMb;
  if (!options.centralizedChordalInit && !hasExternalInitialEstimate) {
    if (options.distributedInitializationFixedNeighborChordalRefit) {
      LocalOptimizationAggregate initRefit;
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer
                ? &optimizerProfile.distributedInitializationLocalSolveSec
                : nullptr);
        initRefit = fixedNeighborChordalRefitLocalModels(
            agents, options.parallelLocalSolves);
      }
      throwOnLocalFailures(
          "distributed initialization fixed-neighbor chordal refit", 0,
          initRefit.failures);
      recordInitializationEstimate();
    }
    for (unsigned initRound = 0;
         initRound < options.distributedInitializationRefinementRounds;
         ++initRound) {
      LocalModelAggregate initRoundModel;
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer
                ? &optimizerProfile.distributedInitializationLocalModelSec
                : nullptr);
        initRoundModel = updateLocalModels(agents);
      }
      throwOnLocalFailures("distributed initialization local model",
                           initRound + 1, initRoundModel.failures);
      LocalOptimizationAggregate initRoundOptimization;
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer
                ? &optimizerProfile.distributedInitializationLocalSolveSec
                : nullptr);
        initRoundOptimization =
            refineLocalModels(agents, options.parallelLocalSolves);
      }
      throwOnLocalFailures("distributed initialization local optimization",
                           initRound + 1, initRoundOptimization.failures);
      recordInitializationEstimate();
      PoseExchangeAggregate initRoundExchange;
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer
                ? &optimizerProfile.distributedInitializationExchangeSec
                : nullptr);
        const CommunicationRound *initRoundTopology =
            options.communicationTopologyInitialFull
                ? nullptr
                : topologyRound(initRound + 1);
        initRoundExchange = exchangeOwnedSeparatorPoses(
            agents, deliveredPoseCache, initRoundTopology,
            options.communicationTopologyMaxRelayHops,
            options.communicationTopologyHopBudgetFraction,
            options.communicationTopologyStaleAwareRelayScore,
            options.communicationTopologyStaleAwareRelayAgeGain,
            options.communicationTopologyDeltaCompression,
            options.communicationTopologyDeltaMode,
            options.communicationTopologyDeltaTopK,
            options.communicationTopologyDeltaMaxReconstructionError,
            options.communicationTopologyLiftedDeltaCompression,
            options.communicationTopologyLiftedDeltaRank,
            options.communicationTopologyLiftedDeltaMaxReconstructionError,
            options.communicationTopologyValueScheduler,
            remainingValueSchedulerBudgetMb(),
            options.communicationTopologyValueMinScoreRatio,
            valueSchedulerUseResidualProxy);
      }
      initCommPoses += initRoundExchange.sent;
      cumulativeCommPoses += initRoundExchange.sent;
      cumulativePosePayloadMb += initRoundExchange.commMb;
    }
    resetLocalHistories(agents);
  }
  const double initCommMb = cumulativePosePayloadMb;
  LocalModelAggregate initLocalModel;
  {
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer ? &optimizerProfile.initialLocalModelSec
                                 : nullptr);
    initLocalModel = options.recordLocalModelDiagnostics
                         ? evaluateLocalModels(agents)
                         : updateLocalModels(agents);
  }
  throwOnLocalFailures("local model construction", 0,
                       initLocalModel.failures);

  Matrix X;
  ManualDpgoMmQuadraticStats globalStats;
  {
    ScopedOptionalSecondsAccumulator profileTimer(
        options.profileOptimizer
            ? &optimizerProfile.initialGlobalEvaluationSec
            : nullptr);
    X = assembleGlobalEstimate(agents, numPoses, options.numRobots,
                               posesPerRobot, d, r);
    globalStats = evaluateManualDpgoMmQuadraticStats(centralProblem, X);
  }
  double globalCost = globalStats.globalCost;
  double globalGrad = globalStats.gradient;
  Matrix previousGlobalFixedPointInput;
  Matrix previousGlobalFixedPointOutput;
  bool hasPreviousGlobalFixedPoint = false;
  ManualDpgoMmIterationSummary initSummary;
  initSummary.iter = 0;
  initSummary.time = 0.0;
  initSummary.globalCost = globalCost;
  initSummary.gradient = globalGrad;
  initSummary.commPoseCount = initCommPoses;
  initSummary.iterCommMb = initCommMb;
  initSummary.cumulativeCommPoseCount = cumulativeCommPoses;
  initSummary.cumulativeCommMb = initCommMb;
  initSummary.initializationCommPoseCount = initCommPoses;
  initSummary.initializationCommMb = initCommMb;
  initSummary.outerCommPoseCount = 0;
  initSummary.outerCommMb = 0.0;
  initSummary.poseIterCommMb = initCommMb;
  initSummary.poseCumulativeCommMb = initCommMb;
  initSummary.localModelFailures = initLocalModel.failures;
  initSummary.localModelCostBefore = initLocalModel.cost;
  initSummary.localModelCostAfter = initLocalModel.cost;
  initSummary.localModelGradientBefore = initLocalModel.gradient;
  initSummary.localModelGradientAfter = initLocalModel.gradient;
  initSummary.postExchangePoseCount = 0;
  initSummary.postExchangePoseCommMb = 0.0;
  result.iterations.push_back(initSummary);
  if (options.saveIterationEstimates) {
    result.iterationEstimates.push_back(X);
  }
  if (options.printIterationSummary) {
    std::cout << "ITER_SUMMARY iter=0 global_cost=" << std::setprecision(20)
              << globalCost << " gradient=" << globalGrad
            << " comm_pose_count=" << initCommPoses
            << " iter_comm_mb=" << initCommMb
            << " cumulative_comm_pose_count=" << cumulativeCommPoses
            << " cumulative_comm_mb=" << initCommMb
            << " initialization_comm_pose_count=" << initCommPoses
            << " initialization_comm_mb=" << initCommMb
            << " outer_comm_pose_count=0"
            << " outer_comm_mb=0"
            << " pose_iter_comm_mb=" << initCommMb
            << " pose_cumulative_comm_mb=" << initCommMb
            << " local_gradient_scalar_count=0"
            << " local_gradient_iter_scalar_comm_mb=0"
            << " local_gradient_cumulative_scalar_comm_mb=0"
            << " local_gradient_selected_step=0"
            << " local_gradient_fresh_pose_count=0"
            << " local_gradient_fresh_pose_comm_mb=0"
            << " local_gradient_fresh_skipped_pose_count=0"
            << " local_gradient_inner_round_count=0"
            << " post_exchange_pose_count=0"
            << " post_exchange_pose_comm_mb=0"
            << " post_exchange_skipped_pose_count=0"
            << " local_model_failures=" << initLocalModel.failures
            << " local_optimization_failures=0"
            << " local_accepted_iteration_count=0"
            << " full_equiv_hybrid_warm_start_candidate_count=0"
            << " full_equiv_hybrid_warm_start_accepted_count=0"
            << " full_equiv_hybrid_warm_start_guard_rejected_count=0"
            << " full_equiv_hybrid_schur_step_candidate_count=0"
            << " full_equiv_hybrid_schur_step_accepted_count=0"
            << " full_equiv_hybrid_schur_step_guard_rejected_count=0"
            << " full_equiv_hybrid_local_portfolio_candidate_count=0"
            << " full_equiv_hybrid_local_portfolio_selected_unsmoothed_count=0"
            << " full_equiv_hybrid_local_portfolio_selected_schwarz_only_count=0"
            << " full_equiv_hybrid_local_portfolio_selected_schwarz_feh_count=0"
            << " full_equiv_hybrid_schwarz_smoothing_sweep_count=0"
            << " full_equiv_hybrid_schwarz_smoothing_candidate_count=0"
            << " full_equiv_hybrid_schwarz_smoothing_accepted_count=0"
            << " full_equiv_hybrid_schwarz_smoothing_rejected_count=0"
            << " full_equiv_hybrid_schwarz_smoothing_cost_decrease=0"
            << " full_equiv_hybrid_schwarz_smoothing_time_sec=0"
            << " full_equiv_hybrid_linear_pcg_iteration_count=0"
            << " full_equiv_hybrid_linear_initial_residual=0"
            << " full_equiv_hybrid_linear_final_residual=0"
            << " full_equiv_hybrid_linear_full_residual=0"
            << " full_equiv_hybrid_linear_solve_time_sec=0"
            << " full_equiv_hybrid_step_trial_count=0"
            << " full_equiv_hybrid_step_trial_accepted_count=0"
            << " full_equiv_hybrid_step_trial_rejected_count=0"
            << " full_equiv_hybrid_step_predicted_decrease_sum=0"
            << " full_equiv_hybrid_step_actual_decrease_sum=0"
            << " full_equiv_hybrid_step_rho_sum=0"
            << " full_equiv_hybrid_step_rho_count=0"
            << " full_equiv_hybrid_step_accepted_scale_sum=0"
            << " full_equiv_hybrid_step_accepted_predicted_decrease_sum=0"
            << " full_equiv_hybrid_step_accepted_actual_decrease_sum=0"
            << " full_equiv_hybrid_step_accepted_rho_sum=0"
            << " full_equiv_hybrid_step_accepted_rho_count=0"
            << " full_equiv_hybrid_step_pareto_candidate_count=0"
            << " full_equiv_hybrid_step_pareto_selected_count=0"
            << " full_equiv_hybrid_step_pareto_selected_scale_sum=0"
            << " full_equiv_hybrid_step_pareto_best_decrease_sum=0"
            << " full_equiv_hybrid_step_pareto_selected_decrease_sum=0"
            << " full_equiv_hybrid_step_pareto_selected_gradient_sum=0"
            << " full_equiv_hybrid_step_pareto_gradient_eval_count=0"
            << " full_equiv_hybrid_step_pareto_gradient_eval_time_sec=0"
            << " full_equiv_hybrid_translation_recovery_step_trial_candidate_count=0"
            << " full_equiv_hybrid_translation_recovery_step_trial_hard_accepted_count=0"
            << " full_equiv_hybrid_translation_recovery_step_trial_selected_count=0"
            << " full_equiv_hybrid_active_separator_candidate_count=0"
            << " full_equiv_hybrid_active_separator_accepted_count=0"
            << " full_equiv_hybrid_active_separator_rejected_count=0"
            << " full_equiv_hybrid_active_separator_step_sum=0"
            << " full_equiv_hybrid_active_separator_cost_decrease_sum=0"
            << " full_equiv_hybrid_active_separator_lm_schur_candidate_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_accepted_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_guard_rejected_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_gradient_guard_rejected_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_solve_failure_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_fallback_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_boundary_col_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_private_col_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_backtracking_trial_count=0"
            << " full_equiv_hybrid_active_separator_lm_schur_alpha_sum=0"
            << " full_equiv_hybrid_active_separator_lm_schur_cost_decrease_sum=0"
            << " full_equiv_hybrid_active_separator_lm_schur_gradient_change_sum=0"
            << " full_equiv_hybrid_translation_recovery_polish_attempt_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_accepted_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_rejected_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_gradient_guard_rejected_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_backtracking_trigger_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_backtracking_trial_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_backtracking_accepted_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_backtracking_alpha_sum=0"
            << " full_equiv_hybrid_translation_recovery_polish_merit_candidate_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_merit_selected_full_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_merit_selected_partial_count=0"
            << " full_equiv_hybrid_translation_recovery_polish_merit_selected_alpha_sum=0"
            << " full_equiv_hybrid_translation_recovery_polish_merit_best_decrease_sum=0"
            << " full_equiv_hybrid_translation_recovery_polish_merit_selected_decrease_sum=0"
            << " full_equiv_hybrid_translation_recovery_polish_cost_decrease_sum=0"
            << " full_equiv_hybrid_translation_recovery_polish_gradient_change_sum=0"
            << " full_equiv_hybrid_sparse_matvec_count=0"
            << " full_equiv_hybrid_reduced_rotation_initial_guess_candidate_count=0"
            << " full_equiv_hybrid_reduced_rotation_initial_guess_used_count=0"
            << " full_equiv_hybrid_reduced_rotation_initial_guess_rejected_count=0"
            << " full_equiv_hybrid_translation_recovery_initial_guess_candidate_count=0"
            << " full_equiv_hybrid_translation_recovery_initial_guess_used_count=0"
            << " full_equiv_hybrid_translation_recovery_initial_guess_rejected_count=0"
            << " full_equiv_hybrid_translation_schur_preconditioner_application_count=0"
            << " full_equiv_hybrid_translation_schur_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_translation_schur_preconditioner_fallback_count=0"
            << " full_equiv_hybrid_local_chain_preconditioner_application_count=0"
            << " full_equiv_hybrid_local_chain_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_local_chain_preconditioner_fallback_count=0"
            << " full_equiv_hybrid_translation_block_preconditioner_application_count=0"
            << " full_equiv_hybrid_translation_block_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_translation_block_preconditioner_fallback_count=0"
            << " full_equiv_hybrid_translation_sparse_schur_preconditioner_application_count=0"
            << " full_equiv_hybrid_translation_sparse_schur_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_translation_sparse_schur_preconditioner_fallback_count=0"
            << " full_equiv_hybrid_translation_local_schur_preconditioner_application_count=0"
            << " full_equiv_hybrid_translation_local_schur_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_translation_local_schur_preconditioner_fallback_count=0"
            << " full_equiv_hybrid_translation_local_schur_preconditioner_active_pose_count=0"
            << " full_equiv_hybrid_translation_local_schur_preconditioner_active_column_count=0"
            << " full_equiv_hybrid_laplacian_deflation_preconditioner_application_count=0"
            << " full_equiv_hybrid_laplacian_deflation_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_laplacian_deflation_preconditioner_fallback_count=0"
            << " full_equiv_hybrid_laplacian_deflation_preconditioner_basis_dimension=0"
            << " full_equiv_hybrid_reduced_rotation_preconditioner_application_count=0"
            << " full_equiv_hybrid_reduced_rotation_preconditioner_factorization_count=0"
            << " full_equiv_hybrid_rqn_used_count=0"
            << " full_equiv_hybrid_rqn_accepted_pair_count=0"
            << " full_equiv_hybrid_rqn_rejected_pair_count=0"
            << " full_equiv_hybrid_rqn_memory_size=0"
            << " full_equiv_hybrid_rqn_preconditioner_application_count=0"
            << " adaptive_refinement_count=0"
            << " extrapolation_accepted_count=0"
            << " extrapolation_rejected_count=0"
            << " g_extrapolation_accepted_count=0"
            << " g_extrapolation_rejected_count=0"
            << " coupled_extrapolation_accepted_count=0"
            << " coupled_extrapolation_rejected_count=0"
            << " anderson_accepted_count=0"
            << " anderson_rejected_count=0"
            << " global_extrapolation_accepted_count=0"
            << " global_extrapolation_rejected_count=0"
            << " global_anderson_accepted_count=0"
            << " global_anderson_rejected_count=0"
            << " local_gradient_accepted_count=0"
            << " local_gradient_rejected_count=0"
            << " local_gradient_coupled_direction_candidate_count=0"
            << " local_gradient_coupled_direction_accepted_count=0"
            << " local_gradient_coupled_direction_rejected_count=0"
            << " local_gradient_coupled_direction_packet_count=0"
            << " local_gradient_coupled_direction_skipped_packet_count=0"
            << " local_gradient_coupled_direction_score_skipped_packet_count=0"
            << " local_gradient_coupled_direction_byte_budget_skipped_packet_count=0"
            << " local_gradient_coupled_direction_scalar_count=0"
            << " local_gradient_coupled_direction_scalar_comm_mb=0"
            << " local_gradient_compact_schur_candidate_count=0"
            << " local_gradient_compact_schur_accepted_count=0"
            << " local_gradient_compact_schur_guard_rejected_count=0"
            << " local_gradient_compact_schur_gradient_guard_rejected_count=0"
            << " local_gradient_compact_schur_solve_failure_count=0"
            << " local_gradient_compact_schur_boundary_col_count=0"
            << " local_gradient_compact_schur_private_col_count=0"
            << " local_gradient_compact_schur_gradient_change_sum=0"
            << " global_gradient_accepted_count=0"
            << " global_gradient_rejected_count=0"
            << " local_history_sync_count=0"
            << " local_model_cost_before=" << initLocalModel.cost
            << " local_model_cost_after=" << initLocalModel.cost
            << " local_model_gradient_before=" << initLocalModel.gradient
            << " local_model_gradient_after=" << initLocalModel.gradient
            << " boundary_edge_cost_before=0"
            << " boundary_edge_cost_after=0"
            << " separator_delta_norm=0"
            << " boundary_proximal_candidate_count=0"
            << " boundary_proximal_accepted_count=0"
            << " amm_accelerated_accepted_count=0"
            << " amm_restart_count=0"
            << " amm_hard_restart_count=0"
            << " amm_soft_restart_count=0"
            << " amm_phi_fallback_count=0"
            << " amm_local_merit_rejected_count=0"
            << " amm_proximal_start_count=0"
            << " amm_skipped_count=0"
            << " amm_trace_count=0"
            << " amm_refined_count=0"
            << " amm_prox_reset_count=0"
            << " amm_restart_used_xakh_count=0"
            << " amm_restart_certificate_count=0"
            << " amm_restart_certificate_passed_count=0"
            << " amm_restart_certificate_failed_count=0"
            << " amm_restart_certificate_min_margin=0"
            << " amm_translation_recovery_attempt_count=0"
            << " amm_translation_recovery_accepted_count=0"
            << " amm_translation_recovery_rejected_count=0"
            << " amm_mean_gamma=0"
            << " amm_mean_Gkh_initial=0"
            << " amm_mean_minG=0"
            << " amm_mean_Gk_after_accelerated=0"
            << " amm_mean_Gkh_after_restart_check=0"
            << " amm_mean_final_Gk=0"
            << " amm_mean_phi_lhs=0"
              << " amm_mean_phi_rhs=0"
              << std::endl;
  }

  for (unsigned iter = 0; iter < options.maxIterations; ++iter) {
    if (options.profileOptimizer) {
      ++optimizerProfile.iterationCount;
    }
    for (auto &agent : agents) {
      agent->setOptimizationRound(iter + 1);
    }
    const Matrix iterationStartX = X;
    const auto start = ProfileClock::now();
    LocalModelAggregate localModelBefore;
    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer ? &optimizerProfile.localModelUpdateWallSec
                                   : nullptr);
      localModelBefore = options.recordLocalModelDiagnostics
                             ? evaluateLocalModels(agents)
                             : updateLocalModels(agents);
      if (options.profileOptimizer) {
        ++optimizerProfile.localModelUpdateCount;
      }
    }
    throwOnLocalFailures("local model construction", iter + 1,
                         localModelBefore.failures);
    LocalOptimizationAggregate localOptimization;
    {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer ? &optimizerProfile.localOptimizationWallSec
                                   : nullptr);
      localOptimization =
          refineLocalModels(agents, options.parallelLocalSolves, iter + 1);
      if (options.profileOptimizer) {
        ++optimizerProfile.localOptimizationRoundCount;
      }
    }
    throwOnLocalFailures("local optimization", iter + 1,
                         localOptimization.failures);
    if (localOptimization.surrogateBoundCheckCount > 0) {
      const bool hadPrevious = result.surrogateBoundCheckCount > 0;
      result.surrogateBoundCheckCount +=
          localOptimization.surrogateBoundCheckCount;
      result.surrogateBoundViolationCount +=
          localOptimization.surrogateBoundViolationCount;
      result.surrogateBoundMinMargin =
          hadPrevious
              ? std::min(result.surrogateBoundMinMargin,
                         localOptimization.surrogateBoundMinMargin)
              : localOptimization.surrogateBoundMinMargin;
    }
    result.variableProjectedSchurCandidateCount +=
        localOptimization.variableProjectedSchurCandidateCount;
    result.variableProjectedSchurAcceptedCount +=
        localOptimization.variableProjectedSchurAcceptedCount;
    result.boundaryProximalCandidateCount +=
        localOptimization.boundaryProximalCandidateCount;
    result.boundaryProximalAcceptedCount +=
        localOptimization.boundaryProximalAcceptedCount;
    LocalGradientCorrectionAggregate localGradientCorrection;
    const bool localGradientCorrectionActive =
        options.localGradientCorrection &&
        (options.localGradientCorrectionMaxRounds == 0 ||
         iter < options.localGradientCorrectionMaxRounds);
    std::size_t localGradientFreshCommPoses = 0;
    std::size_t localGradientFreshSkippedPoses = 0;
    double localGradientFreshCommMb = 0.0;
    if (localGradientCorrectionActive &&
        options.localGradientCorrectionFreshNeighborExchange) {
      PoseExchangeAggregate freshExchange;
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer
                ? &optimizerProfile.localGradientFreshExchangeSec
                : nullptr);
        freshExchange = exchangeChangedOwnedSeparatorPoses(
            agents, deliveredPoseCache,
            options.localGradientCorrectionFreshMinPoseDelta,
            options.localGradientCorrectionFreshDeltaMode,
            options.localGradientCorrectionFreshBudgetFraction,
            options.localGradientCorrectionFreshMaxPosesPerReceiver,
            topologyRound(iter), options.communicationTopologyMaxRelayHops,
            options.communicationTopologyHopBudgetFraction,
            options.communicationTopologyStaleAwareRelayScore,
            options.communicationTopologyStaleAwareRelayAgeGain,
            options.communicationTopologyDeltaCompression,
            options.communicationTopologyDeltaMode,
            options.communicationTopologyDeltaTopK,
            options.communicationTopologyDeltaMaxReconstructionError,
            options.communicationTopologyLiftedDeltaCompression,
            options.communicationTopologyLiftedDeltaRank,
            options.communicationTopologyLiftedDeltaMaxReconstructionError,
            options.communicationTopologyValueScheduler,
            remainingValueSchedulerBudgetMb(),
            options.communicationTopologyValueMinScoreRatio,
            valueSchedulerUseResidualProxy);
        if (options.profileOptimizer) {
          ++optimizerProfile.localGradientFreshExchangeCount;
        }
      }
      localGradientFreshCommPoses = freshExchange.sent;
      localGradientFreshSkippedPoses = freshExchange.skipped;
      localGradientFreshCommMb = freshExchange.commMb;
      if (options.communicationTopologyValueScheduler &&
          options.communicationTopologyValueByteBudgetMb > 0.0) {
        cumulativeValueSchedulerCommMb += freshExchange.commMb;
      }
      cumulativeCommPoses += localGradientFreshCommPoses;
      cumulativePosePayloadMb += localGradientFreshCommMb;
    }
    if (options.communicationTopologyInterfaceModel) {
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer
              ? &optimizerProfile.localGradientCorrectionWallSec
              : nullptr);
      if (options.profileOptimizer) {
        ++optimizerProfile.localGradientCorrectionRoundCount;
      }
      resetLocalGradientCorrectionDiagnostics(agents);
      LocalGradientCorrectionAggregate interfaceCorrection =
          applyCoupledDirectionCurvatureLocalGradientCorrection(
              agents, topologyRound(iter),
              options.communicationTopologyMaxRelayHops,
              options.communicationTopologyInterfaceModelPayload,
              options.communicationTopologyInterfaceModelLocalMerit);
      const LocalGradientCorrectionDiagnostics packetDiagnostics =
          interfaceCorrection.diagnostics;
      interfaceCorrection.diagnostics =
          collectLocalGradientCorrectionDiagnostics(agents);
      interfaceCorrection.diagnostics.add(packetDiagnostics);
      localGradientCorrection.failures += interfaceCorrection.failures;
      localGradientCorrection.accepted += interfaceCorrection.accepted;
      localGradientCorrection.rejected += interfaceCorrection.rejected;
      localGradientCorrection.scalarCount += interfaceCorrection.scalarCount;
      localGradientCorrection.scalarCommMb += interfaceCorrection.scalarCommMb;
      localGradientCorrection.selectedStep = interfaceCorrection.selectedStep;
      localGradientCorrection.diagnostics.add(
          interfaceCorrection.diagnostics);
      throwOnLocalFailures("interface model correction", iter + 1,
                           interfaceCorrection.failures);
    }
    std::size_t localGradientInnerRoundCount = 0;
    if (localGradientCorrectionActive) {
      std::vector<double> localGradientSteps =
          options.localGradientCorrectionSteps;
      if (localGradientSteps.empty()) {
        localGradientSteps.push_back(options.localGradientCorrectionStep);
      }
      ScopedOptionalSecondsAccumulator profileTimer(
          options.profileOptimizer
              ? &optimizerProfile.localGradientCorrectionWallSec
              : nullptr);
      for (unsigned innerRound = 0;
           innerRound < options.localGradientCorrectionInnerRounds;
           ++innerRound) {
        if (options.profileOptimizer) {
          ++optimizerProfile.localGradientCorrectionRoundCount;
        }
        resetLocalGradientCorrectionDiagnostics(agents);
        LocalGradientCorrectionAggregate roundCorrection =
            options.localGradientCorrectionCurvatureStep
                ? (options.localGradientCorrectionCoupledDirection
                       ? applyCoupledDirectionCurvatureLocalGradientCorrection(
                             agents, topologyRound(iter),
                             options.communicationTopologyMaxRelayHops)
                       : options.localGradientCorrectionNeighborhoodStep
                       ? applyNeighborhoodCurvatureLocalGradientCorrection(
                             agents)
                       : options.localGradientCorrectionSharedStep
                             ? applySharedCurvatureLocalGradientCorrection(
                                   agents)
                             : applyCurvatureLocalGradientCorrections(agents))
                : options.localGradientCorrectionNeighborhoodStep
                      ? applyNeighborhoodStepLocalGradientCorrection(
                            agents, localGradientSteps)
                      : options.localGradientCorrectionSharedStep
                            ? applySharedStepLocalGradientCorrection(
                                  agents, localGradientSteps)
                            : applyLocalGradientCorrections(
                                  agents, localGradientSteps,
                                  options.parallelLocalSolves);
        const LocalGradientCorrectionDiagnostics aggregateDiagnostics =
            roundCorrection.diagnostics;
        roundCorrection.diagnostics =
            collectLocalGradientCorrectionDiagnostics(agents);
        roundCorrection.diagnostics.add(aggregateDiagnostics);
        ++localGradientInnerRoundCount;
        localGradientCorrection.failures += roundCorrection.failures;
        localGradientCorrection.accepted += roundCorrection.accepted;
        localGradientCorrection.rejected += roundCorrection.rejected;
        localGradientCorrection.scalarCount += roundCorrection.scalarCount;
        localGradientCorrection.scalarCommMb += roundCorrection.scalarCommMb;
        localGradientCorrection.selectedStep = roundCorrection.selectedStep;
        localGradientCorrection.diagnostics.add(roundCorrection.diagnostics);
        throwOnLocalFailures("local gradient correction", iter + 1,
                             roundCorrection.failures);
        if (roundCorrection.accepted == 0) {
          break;
        }
      }
    }
    cumulativeScalarCommMb += localGradientCorrection.scalarCommMb;
    const LocalModelAggregate localModelAfter =
        options.recordLocalModelDiagnostics ? evaluateLocalModels(agents)
                                            : LocalModelAggregate();
    throwOnLocalFailures("local model construction after optimization",
                         iter + 1, localModelAfter.failures);
    elapsed += std::chrono::duration<double>(
                   ProfileClock::now() - start)
                   .count();
    if (options.profileOptimizer) {
      optimizerProfile.iterationLocalPhaseWallSec += secondsSince(start);
    }

    const auto globalProfileStart =
        options.profileOptimizer ? ProfileClock::now()
                                 : ProfileClock::time_point();
    X = assembleGlobalEstimate(agents, numPoses, options.numRobots,
                               posesPerRobot, d, r);
    const Matrix fixedPointOutputX = X;
    globalStats = evaluateManualDpgoMmQuadraticStats(centralProblem, X);
    globalCost = globalStats.globalCost;
    globalGrad = globalStats.gradient;

    std::size_t globalExtrapolationAccepted = 0;
    std::size_t globalExtrapolationRejected = 0;
    std::size_t globalAndersonAccepted = 0;
    std::size_t globalAndersonRejected = 0;
    std::size_t globalGradientAccepted = 0;
    std::size_t globalGradientRejected = 0;
    std::size_t localHistorySyncCount = 0;
    bool globalCorrectionAccepted = false;

    enum class GlobalCorrectionKind { None, State, Anderson, Gradient };
    GlobalCorrectionKind bestGlobalCorrectionKind = GlobalCorrectionKind::None;
    Matrix bestGlobalCorrectionX = fixedPointOutputX;
    double bestGlobalCorrectionCost = globalCost;
    double bestGlobalCorrectionGradient = globalGrad;
    bool stateHadImprovingCandidate = false;
    bool andersonHadImprovingCandidate = false;
    bool gradientHadImprovingCandidate = false;

    const auto considerGlobalCorrection =
        [&](const Matrix &trialX,
            const ManualDpgoMmQuadraticStats &trialStats,
            GlobalCorrectionKind kind) {
          if (trialX.rows() != fixedPointOutputX.rows() ||
              trialX.cols() != fixedPointOutputX.cols() ||
              !trialX.allFinite() ||
              !std::isfinite(trialStats.globalCost)) {
            return false;
          }
          if (trialStats.globalCost < bestGlobalCorrectionCost - 1e-12) {
            bestGlobalCorrectionX = trialX;
            bestGlobalCorrectionCost = trialStats.globalCost;
            bestGlobalCorrectionGradient = trialStats.gradient;
            bestGlobalCorrectionKind = kind;
            return true;
          }
          return false;
        };

    if (options.globalStateExtrapolation &&
        iterationStartX.rows() == fixedPointOutputX.rows() &&
        iterationStartX.cols() == fixedPointOutputX.cols()) {
      std::vector<double> gammas = options.globalStateExtrapolationGammas;
      if (gammas.empty()) {
        gammas.push_back(options.globalStateExtrapolationGamma);
      }
      for (const double gamma : gammas) {
        const Matrix trialX = projectGlobalRotationBlocks(
            fixedPointOutputX + gamma * (fixedPointOutputX - iterationStartX),
            d);
        if (trialX.rows() != fixedPointOutputX.rows() ||
            trialX.cols() != fixedPointOutputX.cols() ||
            !trialX.allFinite()) {
          ++globalExtrapolationRejected;
          continue;
        }
        const ManualDpgoMmQuadraticStats trialStats =
            evaluateManualDpgoMmQuadraticStats(centralProblem, trialX);
        if (considerGlobalCorrection(trialX, trialStats,
                                     GlobalCorrectionKind::State)) {
          stateHadImprovingCandidate = true;
        } else {
          ++globalExtrapolationRejected;
        }
      }
    }

    if (options.globalAndersonAcceleration && hasPreviousGlobalFixedPoint &&
        previousGlobalFixedPointInput.rows() == iterationStartX.rows() &&
        previousGlobalFixedPointInput.cols() == iterationStartX.cols() &&
        previousGlobalFixedPointOutput.rows() == fixedPointOutputX.rows() &&
        previousGlobalFixedPointOutput.cols() == fixedPointOutputX.cols()) {
      const Matrix currentResidual = fixedPointOutputX - iterationStartX;
      const Matrix previousResidual =
          previousGlobalFixedPointOutput - previousGlobalFixedPointInput;
      const Matrix residualDelta = currentResidual - previousResidual;
      const double denominator =
          residualDelta.cwiseProduct(residualDelta).sum();
      if (std::isfinite(denominator) && denominator > 1e-20) {
        double alpha =
            currentResidual.cwiseProduct(residualDelta).sum() / denominator;
        const double maxAlpha = std::max(0.0, options.globalAndersonMaxAlpha);
        if (std::isfinite(alpha) && maxAlpha > 0.0) {
          alpha = std::max(-maxAlpha, std::min(maxAlpha, alpha));
          const Matrix trialX = projectGlobalRotationBlocks(
              fixedPointOutputX -
                  alpha * (fixedPointOutputX -
                           previousGlobalFixedPointOutput),
              d);
          if (trialX.rows() == fixedPointOutputX.rows() &&
              trialX.cols() == fixedPointOutputX.cols() &&
              trialX.allFinite()) {
            const ManualDpgoMmQuadraticStats trialStats =
                evaluateManualDpgoMmQuadraticStats(centralProblem, trialX);
            if (considerGlobalCorrection(trialX, trialStats,
                                         GlobalCorrectionKind::Anderson)) {
              andersonHadImprovingCandidate = true;
            } else {
              globalAndersonRejected = 1;
            }
          } else {
            globalAndersonRejected = 1;
          }
        } else {
          globalAndersonRejected = 1;
        }
      } else {
        globalAndersonRejected = 1;
      }
    }

    if (options.globalGradientCorrection) {
      std::vector<double> steps = options.globalGradientCorrectionSteps;
      if (steps.empty()) {
        steps.push_back(options.globalGradientCorrectionStep);
      }
      const Matrix gradient = centralProblem.RieGrad(fixedPointOutputX);
      if (gradient.rows() == fixedPointOutputX.rows() &&
          gradient.cols() == fixedPointOutputX.cols() && gradient.allFinite()) {
        for (const double step : steps) {
          const Matrix trialX = projectGlobalRotationBlocks(
              fixedPointOutputX - step * gradient, d);
          if (trialX.rows() != fixedPointOutputX.rows() ||
              trialX.cols() != fixedPointOutputX.cols() ||
              !trialX.allFinite()) {
            ++globalGradientRejected;
            continue;
          }
          const ManualDpgoMmQuadraticStats trialStats =
              evaluateManualDpgoMmQuadraticStats(centralProblem, trialX);
          if (considerGlobalCorrection(trialX, trialStats,
                                       GlobalCorrectionKind::Gradient)) {
            gradientHadImprovingCandidate = true;
          } else {
            ++globalGradientRejected;
          }
        }
      } else {
        globalGradientRejected = steps.size();
      }
    }

    if (bestGlobalCorrectionKind != GlobalCorrectionKind::None) {
      X = bestGlobalCorrectionX;
      globalCost = bestGlobalCorrectionCost;
      globalGrad = bestGlobalCorrectionGradient;
      scatterGlobalEstimateToAgents(agents, X, numPoses, options.numRobots,
                                    posesPerRobot, d, r);
      globalCorrectionAccepted = true;
      switch (bestGlobalCorrectionKind) {
      case GlobalCorrectionKind::State:
        globalExtrapolationAccepted = 1;
        if (andersonHadImprovingCandidate) {
          ++globalAndersonRejected;
        }
        if (gradientHadImprovingCandidate) {
          ++globalGradientRejected;
        }
        break;
      case GlobalCorrectionKind::Anderson:
        globalAndersonAccepted = 1;
        if (stateHadImprovingCandidate) {
          ++globalExtrapolationRejected;
        }
        if (gradientHadImprovingCandidate) {
          ++globalGradientRejected;
        }
        break;
      case GlobalCorrectionKind::Gradient:
        globalGradientAccepted = 1;
        if (stateHadImprovingCandidate) {
          ++globalExtrapolationRejected;
        }
        if (andersonHadImprovingCandidate) {
          ++globalAndersonRejected;
        }
        break;
      case GlobalCorrectionKind::None:
        break;
      }
    }

    if (options.resetLocalHistoryAfterGlobalCorrection &&
        globalCorrectionAccepted) {
      resetLocalHistories(agents);
      localHistorySyncCount = agents.size();
    } else if (options.syncAmmReferenceAfterGlobalCorrection &&
               globalCorrectionAccepted) {
      localHistorySyncCount = syncAmmReferences(agents);
    }

    previousGlobalFixedPointInput = iterationStartX;
    previousGlobalFixedPointOutput = fixedPointOutputX;
    hasPreviousGlobalFixedPoint = true;
    if (options.profileOptimizer) {
      optimizerProfile.globalEvaluationWallSec +=
          secondsSince(globalProfileStart);
      ++optimizerProfile.globalEvaluationCount;
    }

    std::size_t postCorrectionCommPoses = 0;
    std::size_t postCorrectionSkippedPoses = 0;
    double postCorrectionCommMb = 0.0;
    if (options.postExchangePeriod > 0 &&
        ((iter + 1) % options.postExchangePeriod == 0)) {
      PoseExchangeAggregate postExchange;
      {
        ScopedOptionalSecondsAccumulator profileTimer(
            options.profileOptimizer ? &optimizerProfile.postExchangeWallSec
                                     : nullptr);
        postExchange = exchangeChangedOwnedSeparatorPoses(
            agents, deliveredPoseCache, options.postExchangeMinPoseDelta,
            options.postExchangeDeltaMode, options.postExchangeBudgetFraction,
            options.postExchangeMaxPosesPerReceiver, topologyRound(iter),
            options.communicationTopologyMaxRelayHops,
            options.communicationTopologyHopBudgetFraction,
            options.communicationTopologyStaleAwareRelayScore,
            options.communicationTopologyStaleAwareRelayAgeGain,
            options.communicationTopologyDeltaCompression,
            options.communicationTopologyDeltaMode,
            options.communicationTopologyDeltaTopK,
            options.communicationTopologyDeltaMaxReconstructionError,
            options.communicationTopologyLiftedDeltaCompression,
            options.communicationTopologyLiftedDeltaRank,
            options.communicationTopologyLiftedDeltaMaxReconstructionError,
            options.communicationTopologyValueScheduler,
            pacedValueSchedulerBudgetMb(iter),
            options.communicationTopologyValueMinScoreRatio,
            valueSchedulerUseResidualProxy);
        if (options.profileOptimizer) {
          ++optimizerProfile.postExchangeCount;
        }
      }
      postCorrectionCommPoses = postExchange.sent;
      postCorrectionSkippedPoses = postExchange.skipped;
      postCorrectionCommMb = postExchange.commMb;
      if (options.communicationTopologyValueScheduler &&
          options.communicationTopologyValueByteBudgetMb > 0.0) {
        cumulativeValueSchedulerCommMb += postCorrectionCommMb;
      }
    }
    cumulativeCommPoses += postCorrectionCommPoses;
    cumulativePosePayloadMb += postCorrectionCommMb;
    const std::size_t iterCommPoses =
        localGradientFreshCommPoses + postCorrectionCommPoses;
    const double iterPoseCommMb =
        localGradientFreshCommMb + postCorrectionCommMb;
    const double iterTotalCommMb =
        iterPoseCommMb + localGradientCorrection.scalarCommMb;
    const double cumulativeTotalCommMb =
        cumulativePosePayloadMb + cumulativeScalarCommMb;
    const std::size_t outerCommPoses = cumulativeCommPoses - initCommPoses;
    const double outerCommMb = cumulativeTotalCommMb - initCommMb;

    if (options.traceAmm) {
      for (auto row : localOptimization.ammTraceRows) {
        row.iter = iter + 1;
        ammTraceRows.push_back(row);
      }
    }
    const auto ammMean = [&](double sum) {
      return localOptimization.ammTraceCount == 0
                 ? 0.0
                 : sum / static_cast<double>(localOptimization.ammTraceCount);
    };

    ManualDpgoMmIterationSummary iterSummary;
    iterSummary.iter = iter + 1;
    iterSummary.time = elapsed;
    iterSummary.globalCost = globalCost;
    iterSummary.gradient = globalGrad;
    iterSummary.commPoseCount = iterCommPoses;
    iterSummary.iterCommMb = iterTotalCommMb;
    iterSummary.cumulativeCommPoseCount = cumulativeCommPoses;
    iterSummary.cumulativeCommMb = cumulativeTotalCommMb;
    iterSummary.initializationCommPoseCount = initCommPoses;
    iterSummary.initializationCommMb = initCommMb;
    iterSummary.outerCommPoseCount = outerCommPoses;
    iterSummary.outerCommMb = outerCommMb;
    iterSummary.poseIterCommMb = iterPoseCommMb;
    iterSummary.poseCumulativeCommMb = cumulativePosePayloadMb;
    iterSummary.localGradientScalarCount =
        localGradientCorrection.scalarCount;
    iterSummary.localGradientIterScalarCommMb =
        localGradientCorrection.scalarCommMb;
    iterSummary.localGradientCumulativeScalarCommMb =
        cumulativeScalarCommMb;
    iterSummary.localGradientSelectedStep =
        localGradientCorrection.selectedStep;
    iterSummary.localGradientFreshPoseCount = localGradientFreshCommPoses;
    iterSummary.localGradientFreshPoseCommMb =
        localGradientFreshCommMb;
    iterSummary.localGradientFreshSkippedPoseCount =
        localGradientFreshSkippedPoses;
    iterSummary.localGradientInnerRoundCount =
        localGradientInnerRoundCount;
    iterSummary.postExchangePoseCount = postCorrectionCommPoses;
    iterSummary.postExchangePoseCommMb =
        postCorrectionCommMb;
    iterSummary.postExchangeSkippedPoseCount = postCorrectionSkippedPoses;
    iterSummary.localModelFailures = localModelBefore.failures;
    iterSummary.localOptimizationFailures = localOptimization.failures;
    iterSummary.localAcceptedIterationCount =
        localOptimization.acceptedIterations;
    iterSummary.fullEquivHybridWarmStartCandidateCount =
        localOptimization.fullEquivHybridWarmStartCandidateCount;
    iterSummary.fullEquivHybridWarmStartAcceptedCount =
        localOptimization.fullEquivHybridWarmStartAcceptedCount;
    iterSummary.fullEquivHybridWarmStartGuardRejectedCount =
        localOptimization.fullEquivHybridWarmStartGuardRejectedCount;
    iterSummary.fullEquivHybridSchurStepCandidateCount =
        localOptimization.fullEquivHybridSchurStepCandidateCount;
    iterSummary.fullEquivHybridSchurStepAcceptedCount =
        localOptimization.fullEquivHybridSchurStepAcceptedCount;
    iterSummary.fullEquivHybridSchurStepGuardRejectedCount =
        localOptimization.fullEquivHybridSchurStepGuardRejectedCount;
    iterSummary.fullEquivHybridLocalPortfolioCandidateCount =
        localOptimization.fullEquivHybridLocalPortfolioCandidateCount;
    iterSummary.fullEquivHybridLocalPortfolioSelectedUnsmoothedCount =
        localOptimization
            .fullEquivHybridLocalPortfolioSelectedUnsmoothedCount;
    iterSummary.fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount =
        localOptimization
            .fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount;
    iterSummary.fullEquivHybridLocalPortfolioSelectedSchwarzFehCount =
        localOptimization.fullEquivHybridLocalPortfolioSelectedSchwarzFehCount;
    iterSummary.fullEquivHybridSchwarzSmoothingSweepCount =
        localOptimization.fullEquivHybridSchwarzSmoothingSweepCount;
    iterSummary.fullEquivHybridSchwarzSmoothingCandidateCount =
        localOptimization.fullEquivHybridSchwarzSmoothingCandidateCount;
    iterSummary.fullEquivHybridSchwarzSmoothingAcceptedCount =
        localOptimization.fullEquivHybridSchwarzSmoothingAcceptedCount;
    iterSummary.fullEquivHybridSchwarzSmoothingRejectedCount =
        localOptimization.fullEquivHybridSchwarzSmoothingRejectedCount;
    iterSummary.fullEquivHybridSchwarzSmoothingCostDecrease =
        localOptimization.fullEquivHybridSchwarzSmoothingCostDecrease;
    iterSummary.fullEquivHybridSchwarzSmoothingTimeSec =
        localOptimization.fullEquivHybridSchwarzSmoothingTimeSec;
    iterSummary.fullEquivHybridLinearPcgIterationCount =
        localOptimization.fullEquivHybridLinearPcgIterationCount;
    iterSummary.fullEquivHybridLinearInitialResidual =
        localOptimization.fullEquivHybridLinearInitialResidual;
    iterSummary.fullEquivHybridLinearFinalResidual =
        localOptimization.fullEquivHybridLinearFinalResidual;
    iterSummary.fullEquivHybridLinearFullResidual =
        localOptimization.fullEquivHybridLinearFullResidual;
    iterSummary.fullEquivHybridLinearSolveTimeSec =
        localOptimization.fullEquivHybridLinearSolveTimeSec;
    iterSummary.fullEquivHybridStepTrialCount =
        localOptimization.fullEquivHybridStepTrialCount;
    iterSummary.fullEquivHybridStepTrialAcceptedCount =
        localOptimization.fullEquivHybridStepTrialAcceptedCount;
    iterSummary.fullEquivHybridStepTrialRejectedCount =
        localOptimization.fullEquivHybridStepTrialRejectedCount;
    iterSummary.fullEquivHybridStepPredictedDecreaseSum =
        localOptimization.fullEquivHybridStepPredictedDecreaseSum;
    iterSummary.fullEquivHybridStepActualDecreaseSum =
        localOptimization.fullEquivHybridStepActualDecreaseSum;
    iterSummary.fullEquivHybridStepRhoSum =
        localOptimization.fullEquivHybridStepRhoSum;
    iterSummary.fullEquivHybridStepRhoCount =
        localOptimization.fullEquivHybridStepRhoCount;
    iterSummary.fullEquivHybridStepAcceptedScaleSum =
        localOptimization.fullEquivHybridStepAcceptedScaleSum;
    iterSummary.fullEquivHybridStepAcceptedPredictedDecreaseSum =
        localOptimization.fullEquivHybridStepAcceptedPredictedDecreaseSum;
    iterSummary.fullEquivHybridStepAcceptedActualDecreaseSum =
        localOptimization.fullEquivHybridStepAcceptedActualDecreaseSum;
    iterSummary.fullEquivHybridStepAcceptedRhoSum =
        localOptimization.fullEquivHybridStepAcceptedRhoSum;
    iterSummary.fullEquivHybridStepAcceptedRhoCount =
        localOptimization.fullEquivHybridStepAcceptedRhoCount;
    iterSummary.fullEquivHybridStepParetoCandidateCount =
        localOptimization.fullEquivHybridStepParetoCandidateCount;
    iterSummary.fullEquivHybridStepParetoSelectedCount =
        localOptimization.fullEquivHybridStepParetoSelectedCount;
    iterSummary.fullEquivHybridStepParetoSelectedScaleSum =
        localOptimization.fullEquivHybridStepParetoSelectedScaleSum;
    iterSummary.fullEquivHybridStepParetoBestDecreaseSum =
        localOptimization.fullEquivHybridStepParetoBestDecreaseSum;
    iterSummary.fullEquivHybridStepParetoSelectedDecreaseSum =
        localOptimization.fullEquivHybridStepParetoSelectedDecreaseSum;
    iterSummary.fullEquivHybridStepParetoSelectedGradientSum =
        localOptimization.fullEquivHybridStepParetoSelectedGradientSum;
    iterSummary.fullEquivHybridStepParetoGradientEvalCount =
        localOptimization.fullEquivHybridStepParetoGradientEvalCount;
    iterSummary.fullEquivHybridStepParetoGradientEvalTimeSec =
        localOptimization.fullEquivHybridStepParetoGradientEvalTimeSec;
    iterSummary.fullEquivHybridTranslationRecoveryStepTrialCandidateCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryStepTrialCandidateCount;
    iterSummary.fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount;
    iterSummary.fullEquivHybridTranslationRecoveryStepTrialSelectedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryStepTrialSelectedCount;
    iterSummary.fullEquivHybridActiveSeparatorCandidateCount =
        localOptimization.fullEquivHybridActiveSeparatorCandidateCount;
    iterSummary.fullEquivHybridActiveSeparatorAcceptedCount =
        localOptimization.fullEquivHybridActiveSeparatorAcceptedCount;
    iterSummary.fullEquivHybridActiveSeparatorRejectedCount =
        localOptimization.fullEquivHybridActiveSeparatorRejectedCount;
    iterSummary.fullEquivHybridActiveSeparatorStepSum =
        localOptimization.fullEquivHybridActiveSeparatorStepSum;
    iterSummary.fullEquivHybridActiveSeparatorCostDecreaseSum =
        localOptimization.fullEquivHybridActiveSeparatorCostDecreaseSum;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurCandidateCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurCandidateCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurAcceptedCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurAcceptedCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount;
    iterSummary
        .fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurSolveFailureCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurSolveFailureCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurFallbackCount =
        localOptimization.fullEquivHybridActiveSeparatorLmSchurFallbackCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurBoundaryColCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurBoundaryColCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurPrivateColCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurPrivateColCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurAlphaSum =
        localOptimization.fullEquivHybridActiveSeparatorLmSchurAlphaSum;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum;
    iterSummary.fullEquivHybridActiveSeparatorLmSchurGradientChangeSum =
        localOptimization
            .fullEquivHybridActiveSeparatorLmSchurGradientChangeSum;
    iterSummary.fullEquivHybridTranslationRecoveryPolishAttemptCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishAttemptCount;
    iterSummary.fullEquivHybridTranslationRecoveryPolishAcceptedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishAcceptedCount;
    iterSummary.fullEquivHybridTranslationRecoveryPolishRejectedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishRejectedCount;
    iterSummary
        .fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount;
    iterSummary
        .fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount;
    iterSummary.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount;
    iterSummary
        .fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount;
    iterSummary.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum;
    iterSummary.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishMeritCandidateCount;
    iterSummary
        .fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount;
    iterSummary
        .fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount;
    iterSummary.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum;
    iterSummary.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum;
    iterSummary
        .fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum;
    iterSummary.fullEquivHybridTranslationRecoveryPolishCostDecreaseSum =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishCostDecreaseSum;
    iterSummary.fullEquivHybridTranslationRecoveryPolishGradientChangeSum =
        localOptimization
            .fullEquivHybridTranslationRecoveryPolishGradientChangeSum;
    iterSummary.fullEquivHybridSparseMatrixVectorProductCount =
        localOptimization.fullEquivHybridSparseMatrixVectorProductCount;
    iterSummary.fullEquivHybridReducedRotationInitialGuessCandidateCount =
        localOptimization
            .fullEquivHybridReducedRotationInitialGuessCandidateCount;
    iterSummary.fullEquivHybridReducedRotationInitialGuessUsedCount =
        localOptimization.fullEquivHybridReducedRotationInitialGuessUsedCount;
    iterSummary.fullEquivHybridReducedRotationInitialGuessRejectedCount =
        localOptimization
            .fullEquivHybridReducedRotationInitialGuessRejectedCount;
    iterSummary.fullEquivHybridTranslationRecoveryInitialGuessCandidateCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryInitialGuessCandidateCount;
    iterSummary.fullEquivHybridTranslationRecoveryInitialGuessUsedCount =
        localOptimization.fullEquivHybridTranslationRecoveryInitialGuessUsedCount;
    iterSummary.fullEquivHybridTranslationRecoveryInitialGuessRejectedCount =
        localOptimization
            .fullEquivHybridTranslationRecoveryInitialGuessRejectedCount;
    iterSummary
        .fullEquivHybridTranslationSchurPreconditionerApplicationCount =
        localOptimization
            .fullEquivHybridTranslationSchurPreconditionerApplicationCount;
    iterSummary
        .fullEquivHybridTranslationSchurPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridTranslationSchurPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridTranslationSchurPreconditionerFallbackCount =
        localOptimization
            .fullEquivHybridTranslationSchurPreconditionerFallbackCount;
    iterSummary.fullEquivHybridLocalChainPreconditionerApplicationCount =
        localOptimization.fullEquivHybridLocalChainPreconditionerApplicationCount;
    iterSummary.fullEquivHybridLocalChainPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridLocalChainPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridLocalChainPreconditionerFallbackCount =
        localOptimization.fullEquivHybridLocalChainPreconditionerFallbackCount;
    iterSummary
        .fullEquivHybridTranslationBlockPreconditionerApplicationCount =
        localOptimization
            .fullEquivHybridTranslationBlockPreconditionerApplicationCount;
    iterSummary
        .fullEquivHybridTranslationBlockPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridTranslationBlockPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridTranslationBlockPreconditionerFallbackCount =
        localOptimization
            .fullEquivHybridTranslationBlockPreconditionerFallbackCount;
    iterSummary
        .fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount =
        localOptimization
            .fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount;
    iterSummary
        .fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount =
        localOptimization
            .fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount;
    iterSummary
        .fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount =
        localOptimization
            .fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount;
    iterSummary
        .fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount =
        localOptimization
            .fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount;
    iterSummary
        .fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount =
        localOptimization
            .fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount;
    iterSummary
        .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount =
        localOptimization
            .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount;
    iterSummary
        .fullEquivHybridLaplacianDeflationPreconditionerApplicationCount =
        localOptimization
            .fullEquivHybridLaplacianDeflationPreconditionerApplicationCount;
    iterSummary
        .fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridLaplacianDeflationPreconditionerFallbackCount =
        localOptimization
            .fullEquivHybridLaplacianDeflationPreconditionerFallbackCount;
    iterSummary.fullEquivHybridLaplacianDeflationPreconditionerBasisDimension =
        localOptimization
            .fullEquivHybridLaplacianDeflationPreconditionerBasisDimension;
    iterSummary
        .fullEquivHybridReducedRotationPreconditionerApplicationCount =
        localOptimization
            .fullEquivHybridReducedRotationPreconditionerApplicationCount;
    iterSummary
        .fullEquivHybridReducedRotationPreconditionerFactorizationCount =
        localOptimization
            .fullEquivHybridReducedRotationPreconditionerFactorizationCount;
    iterSummary.fullEquivHybridRqnUsedCount =
        localOptimization.fullEquivHybridRqnUsedCount;
    iterSummary.fullEquivHybridRqnAcceptedPairCount =
        localOptimization.fullEquivHybridRqnAcceptedPairCount;
    iterSummary.fullEquivHybridRqnRejectedPairCount =
        localOptimization.fullEquivHybridRqnRejectedPairCount;
    iterSummary.fullEquivHybridRqnMemorySize =
        localOptimization.fullEquivHybridRqnMemorySize;
    iterSummary.fullEquivHybridRqnPreconditionerApplicationCount =
        localOptimization.fullEquivHybridRqnPreconditionerApplicationCount;
    iterSummary.adaptiveRefinementCount =
        localOptimization.adaptiveRefinements;
    iterSummary.extrapolationAcceptedCount =
        localOptimization.extrapolationAccepted;
    iterSummary.extrapolationRejectedCount =
        localOptimization.extrapolationRejected;
    iterSummary.gExtrapolationAcceptedCount =
        localOptimization.gExtrapolationAccepted;
    iterSummary.gExtrapolationRejectedCount =
        localOptimization.gExtrapolationRejected;
    iterSummary.coupledExtrapolationAcceptedCount =
        localOptimization.coupledExtrapolationAccepted;
    iterSummary.coupledExtrapolationRejectedCount =
        localOptimization.coupledExtrapolationRejected;
    iterSummary.andersonAcceptedCount = localOptimization.andersonAccepted;
    iterSummary.andersonRejectedCount = localOptimization.andersonRejected;
    iterSummary.squaremAcceptedCount = localOptimization.squaremAccepted;
    iterSummary.squaremRejectedCount = localOptimization.squaremRejected;
    iterSummary.globalExtrapolationAcceptedCount =
        globalExtrapolationAccepted;
    iterSummary.globalExtrapolationRejectedCount =
        globalExtrapolationRejected;
    iterSummary.globalAndersonAcceptedCount = globalAndersonAccepted;
    iterSummary.globalAndersonRejectedCount = globalAndersonRejected;
    iterSummary.localGradientAcceptedCount =
        localGradientCorrection.accepted;
    iterSummary.localGradientRejectedCount =
        localGradientCorrection.rejected;
    LocalGradientCorrectionDiagnostics mergedLocalGradientDiagnostics =
        localOptimization.localGradientDiagnostics;
    mergedLocalGradientDiagnostics.add(localGradientCorrection.diagnostics);
    iterSummary.localGradientCoupledDirectionCandidateCount =
        mergedLocalGradientDiagnostics.coupledDirectionCandidateCount;
    iterSummary.localGradientCoupledDirectionAcceptedCount =
        mergedLocalGradientDiagnostics.coupledDirectionAcceptedCount;
    iterSummary.localGradientCoupledDirectionRejectedCount =
        mergedLocalGradientDiagnostics.coupledDirectionRejectedCount;
    iterSummary.localGradientCoupledDirectionPacketCount =
        mergedLocalGradientDiagnostics.coupledDirectionPacketCount;
    iterSummary.localGradientCoupledDirectionSkippedPacketCount =
        mergedLocalGradientDiagnostics.coupledDirectionSkippedPacketCount;
    iterSummary.localGradientCoupledDirectionScoreSkippedPacketCount =
        mergedLocalGradientDiagnostics
            .coupledDirectionScoreSkippedPacketCount;
    iterSummary.localGradientCoupledDirectionByteBudgetSkippedPacketCount =
        mergedLocalGradientDiagnostics
            .coupledDirectionByteBudgetSkippedPacketCount;
    iterSummary.localGradientCoupledDirectionScalarCount =
        mergedLocalGradientDiagnostics.coupledDirectionScalarCount;
    iterSummary.localGradientCoupledDirectionScalarCommMb =
        scalarMegabytes(
            mergedLocalGradientDiagnostics.coupledDirectionScalarCount);
    iterSummary.localGradientCompactSchurCandidateCount =
        mergedLocalGradientDiagnostics.compactSchurCandidateCount;
    iterSummary.localGradientCompactSchurAcceptedCount =
        mergedLocalGradientDiagnostics.compactSchurAcceptedCount;
    iterSummary.localGradientCompactSchurGuardRejectedCount =
        mergedLocalGradientDiagnostics.compactSchurGuardRejectedCount;
    iterSummary.localGradientCompactSchurGradientGuardRejectedCount =
        mergedLocalGradientDiagnostics.compactSchurGradientGuardRejectedCount;
    iterSummary.localGradientCompactSchurSolveFailureCount =
        mergedLocalGradientDiagnostics.compactSchurSolveFailureCount;
    iterSummary.localGradientCompactSchurBoundaryColCount =
        mergedLocalGradientDiagnostics.compactSchurBoundaryColCount;
    iterSummary.localGradientCompactSchurPrivateColCount =
        mergedLocalGradientDiagnostics.compactSchurPrivateColCount;
    iterSummary.localGradientCompactSchurGradientChangeSum =
        mergedLocalGradientDiagnostics.compactSchurGradientChangeSum;
    iterSummary.globalGradientAcceptedCount = globalGradientAccepted;
    iterSummary.globalGradientRejectedCount = globalGradientRejected;
    iterSummary.localHistorySyncCount = localHistorySyncCount;
    iterSummary.localModelCostBefore = localModelBefore.cost;
    iterSummary.localModelCostAfter = localModelAfter.cost;
    iterSummary.localModelGradientBefore = localModelBefore.gradient;
    iterSummary.localModelGradientAfter = localModelAfter.gradient;
    iterSummary.boundaryEdgeCostBefore =
        localOptimization.boundaryEdgeCostBefore;
    iterSummary.boundaryEdgeCostAfter =
        localOptimization.boundaryEdgeCostAfter;
    iterSummary.separatorDeltaNorm =
        std::sqrt(localOptimization.separatorDeltaNormSquared);
    iterSummary.boundaryProximalCandidateCount =
        localOptimization.boundaryProximalCandidateCount;
    iterSummary.boundaryProximalAcceptedCount =
        localOptimization.boundaryProximalAcceptedCount;
    iterSummary.edgeTightQuadraticEvalCount =
        localOptimization.edgeTightQuadraticEvalCount;
    iterSummary.edgeTightQuadraticSurrogateCostSum =
        localOptimization.edgeTightQuadraticSurrogateCostSum;
    iterSummary.edgeTightQuadraticTrueCostSum =
        localOptimization.edgeTightQuadraticTrueCostSum;
    iterSummary.edgeTightQuadraticMajorizationGapMin =
        localOptimization.edgeTightQuadraticEvalCount == 0
            ? 0.0
            : localOptimization.edgeTightQuadraticMajorizationGapMin;
    iterSummary.ammAcceleratedAcceptedCount =
        localOptimization.ammAcceleratedAccepted;
    iterSummary.ammRestartCount = localOptimization.ammRestart;
    iterSummary.ammHardRestartCount = localOptimization.ammHardRestart;
    iterSummary.ammSoftRestartCount = localOptimization.ammSoftRestart;
    iterSummary.ammPhiFallbackCount = localOptimization.ammPhiFallback;
    iterSummary.ammLocalMeritRejectedCount =
        localOptimization.ammLocalMeritRejected;
    iterSummary.ammProximalStartCount =
        localOptimization.ammProximalStart;
    iterSummary.ammSkippedCount = localOptimization.ammSkipped;
    iterSummary.ammTraceCount = localOptimization.ammTraceCount;
    iterSummary.ammRefinedCount = localOptimization.ammRefined;
    iterSummary.ammProxResetCount = localOptimization.ammProxReset;
    iterSummary.ammRestartUsedXakhCount =
        localOptimization.ammRestartUsedXakh;
    iterSummary.ammRestartCertificateCount =
        localOptimization.ammRestartCertificateCount;
    iterSummary.ammRestartCertificatePassedCount =
        localOptimization.ammRestartCertificatePassedCount;
    iterSummary.ammRestartCertificateFailedCount =
        localOptimization.ammRestartCertificateFailedCount;
    iterSummary.ammRestartCertificateMinMargin =
        localOptimization.ammRestartCertificateCount == 0
            ? 0.0
            : localOptimization.ammRestartCertificateMinMargin;
    iterSummary.ammTranslationRecoveryAttemptCount =
        localOptimization.ammTranslationRecoveryAttemptCount;
    iterSummary.ammTranslationRecoveryAcceptedCount =
        localOptimization.ammTranslationRecoveryAcceptedCount;
    iterSummary.ammTranslationRecoveryRejectedCount =
        localOptimization.ammTranslationRecoveryRejectedCount;
    iterSummary.ammMixedSurrogateCandidateCount =
        localOptimization.ammMixedSurrogateCandidateCount;
    iterSummary.ammMixedSurrogateTrueLocalAcceptedCount =
        localOptimization.ammMixedSurrogateTrueLocalAcceptedCount;
    iterSummary.ammMixedSurrogateSimpleSelectedCount =
        localOptimization.ammMixedSurrogateSimpleSelectedCount;
    iterSummary.ammMixedSurrogateTrueLocalSelectedCount =
        localOptimization.ammMixedSurrogateTrueLocalSelectedCount;
    iterSummary.ammMixedSurrogateExtrapolatedSelectedCount =
        localOptimization.ammMixedSurrogateExtrapolatedSelectedCount;
    iterSummary.ammMixedSurrogateOtherSelectedCount =
        localOptimization.ammMixedSurrogateOtherSelectedCount;
    iterSummary.ammMixedSurrogateSimpleSkippedCount =
        localOptimization.ammMixedSurrogateSimpleSkippedCount;
    iterSummary.ammMixedSurrogateSimpleForcedRefreshCount =
        localOptimization.ammMixedSurrogateSimpleForcedRefreshCount;
    iterSummary.ammMeanGamma = ammMean(localOptimization.ammGammaSum);
    iterSummary.ammMeanGkhInitial =
        ammMean(localOptimization.ammGkhInitialSum);
    iterSummary.ammMeanMinG = ammMean(localOptimization.ammMinGSum);
    iterSummary.ammMeanGkAfterAccelerated =
        ammMean(localOptimization.ammGkAfterAcceleratedSum);
    iterSummary.ammMeanGkhAfterRestartCheck =
        ammMean(localOptimization.ammGkhAfterRestartCheckSum);
    iterSummary.ammMeanFinalGk =
        ammMean(localOptimization.ammFinalGkSum);
    iterSummary.ammMeanPhiLhs = ammMean(localOptimization.ammPhiLhsSum);
    iterSummary.ammMeanPhiRhs = ammMean(localOptimization.ammPhiRhsSum);
    result.iterations.push_back(iterSummary);
    if (options.saveIterationEstimates) {
      result.iterationEstimates.push_back(X);
    }
    if (options.printIterationSummary) {
      std::cout << "ITER_SUMMARY iter=" << iter + 1
                << " global_cost=" << std::setprecision(20) << globalCost
              << " gradient=" << globalGrad
              << " comm_pose_count=" << iterCommPoses
              << " iter_comm_mb=" << iterTotalCommMb
              << " cumulative_comm_pose_count=" << cumulativeCommPoses
              << " cumulative_comm_mb=" << cumulativeTotalCommMb
              << " initialization_comm_pose_count=" << initCommPoses
              << " initialization_comm_mb=" << initCommMb
              << " outer_comm_pose_count=" << outerCommPoses
              << " outer_comm_mb=" << outerCommMb
              << " pose_iter_comm_mb=" << iterPoseCommMb
              << " pose_cumulative_comm_mb=" << cumulativePosePayloadMb
              << " local_gradient_scalar_count="
              << localGradientCorrection.scalarCount
              << " local_gradient_iter_scalar_comm_mb="
              << localGradientCorrection.scalarCommMb
              << " local_gradient_cumulative_scalar_comm_mb="
              << cumulativeScalarCommMb
              << " local_gradient_selected_step="
              << localGradientCorrection.selectedStep
              << " local_gradient_fresh_pose_count="
              << localGradientFreshCommPoses
              << " local_gradient_fresh_pose_comm_mb="
              << localGradientFreshCommMb
              << " local_gradient_fresh_skipped_pose_count="
              << localGradientFreshSkippedPoses
              << " local_gradient_inner_round_count="
              << localGradientInnerRoundCount
              << " post_exchange_pose_count=" << postCorrectionCommPoses
              << " post_exchange_pose_comm_mb="
              << postCorrectionCommMb
              << " post_exchange_skipped_pose_count="
              << postCorrectionSkippedPoses
              << " local_model_failures=" << localModelBefore.failures
              << " local_optimization_failures="
              << localOptimization.failures
              << " local_accepted_iteration_count="
              << localOptimization.acceptedIterations
              << " full_equiv_hybrid_warm_start_candidate_count="
              << localOptimization.fullEquivHybridWarmStartCandidateCount
              << " full_equiv_hybrid_warm_start_accepted_count="
              << localOptimization.fullEquivHybridWarmStartAcceptedCount
              << " full_equiv_hybrid_warm_start_guard_rejected_count="
              << localOptimization
                     .fullEquivHybridWarmStartGuardRejectedCount
              << " full_equiv_hybrid_schur_step_candidate_count="
              << localOptimization.fullEquivHybridSchurStepCandidateCount
              << " full_equiv_hybrid_schur_step_accepted_count="
              << localOptimization.fullEquivHybridSchurStepAcceptedCount
              << " full_equiv_hybrid_schur_step_guard_rejected_count="
              << localOptimization
                     .fullEquivHybridSchurStepGuardRejectedCount
              << " full_equiv_hybrid_local_portfolio_candidate_count="
              << localOptimization.fullEquivHybridLocalPortfolioCandidateCount
              << " full_equiv_hybrid_local_portfolio_selected_unsmoothed_count="
              << localOptimization
                     .fullEquivHybridLocalPortfolioSelectedUnsmoothedCount
              << " full_equiv_hybrid_local_portfolio_selected_schwarz_only_count="
              << localOptimization
                     .fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount
              << " full_equiv_hybrid_local_portfolio_selected_schwarz_feh_count="
              << localOptimization
                     .fullEquivHybridLocalPortfolioSelectedSchwarzFehCount
              << " full_equiv_hybrid_schwarz_smoothing_sweep_count="
              << localOptimization.fullEquivHybridSchwarzSmoothingSweepCount
              << " full_equiv_hybrid_schwarz_smoothing_candidate_count="
              << localOptimization
                     .fullEquivHybridSchwarzSmoothingCandidateCount
              << " full_equiv_hybrid_schwarz_smoothing_accepted_count="
              << localOptimization
                     .fullEquivHybridSchwarzSmoothingAcceptedCount
              << " full_equiv_hybrid_schwarz_smoothing_rejected_count="
              << localOptimization
                     .fullEquivHybridSchwarzSmoothingRejectedCount
              << " full_equiv_hybrid_schwarz_smoothing_cost_decrease="
              << localOptimization
                     .fullEquivHybridSchwarzSmoothingCostDecrease
              << " full_equiv_hybrid_schwarz_smoothing_time_sec="
              << localOptimization.fullEquivHybridSchwarzSmoothingTimeSec
              << " full_equiv_hybrid_linear_pcg_iteration_count="
              << localOptimization.fullEquivHybridLinearPcgIterationCount
              << " full_equiv_hybrid_linear_initial_residual="
              << localOptimization.fullEquivHybridLinearInitialResidual
              << " full_equiv_hybrid_linear_final_residual="
              << localOptimization.fullEquivHybridLinearFinalResidual
              << " full_equiv_hybrid_linear_full_residual="
              << localOptimization.fullEquivHybridLinearFullResidual
              << " full_equiv_hybrid_linear_solve_time_sec="
              << localOptimization.fullEquivHybridLinearSolveTimeSec
              << " full_equiv_hybrid_step_trial_count="
              << localOptimization.fullEquivHybridStepTrialCount
              << " full_equiv_hybrid_step_trial_accepted_count="
              << localOptimization.fullEquivHybridStepTrialAcceptedCount
              << " full_equiv_hybrid_step_trial_rejected_count="
              << localOptimization.fullEquivHybridStepTrialRejectedCount
              << " full_equiv_hybrid_step_predicted_decrease_sum="
              << localOptimization.fullEquivHybridStepPredictedDecreaseSum
              << " full_equiv_hybrid_step_actual_decrease_sum="
              << localOptimization.fullEquivHybridStepActualDecreaseSum
              << " full_equiv_hybrid_step_rho_sum="
              << localOptimization.fullEquivHybridStepRhoSum
              << " full_equiv_hybrid_step_rho_count="
              << localOptimization.fullEquivHybridStepRhoCount
              << " full_equiv_hybrid_step_accepted_scale_sum="
              << localOptimization.fullEquivHybridStepAcceptedScaleSum
              << " full_equiv_hybrid_step_accepted_predicted_decrease_sum="
              << localOptimization
                     .fullEquivHybridStepAcceptedPredictedDecreaseSum
              << " full_equiv_hybrid_step_accepted_actual_decrease_sum="
              << localOptimization.fullEquivHybridStepAcceptedActualDecreaseSum
              << " full_equiv_hybrid_step_accepted_rho_sum="
              << localOptimization.fullEquivHybridStepAcceptedRhoSum
              << " full_equiv_hybrid_step_accepted_rho_count="
              << localOptimization.fullEquivHybridStepAcceptedRhoCount
              << " full_equiv_hybrid_step_pareto_candidate_count="
              << localOptimization.fullEquivHybridStepParetoCandidateCount
              << " full_equiv_hybrid_step_pareto_selected_count="
              << localOptimization.fullEquivHybridStepParetoSelectedCount
              << " full_equiv_hybrid_step_pareto_selected_scale_sum="
              << localOptimization.fullEquivHybridStepParetoSelectedScaleSum
              << " full_equiv_hybrid_step_pareto_best_decrease_sum="
              << localOptimization.fullEquivHybridStepParetoBestDecreaseSum
              << " full_equiv_hybrid_step_pareto_selected_decrease_sum="
              << localOptimization.fullEquivHybridStepParetoSelectedDecreaseSum
              << " full_equiv_hybrid_step_pareto_selected_gradient_sum="
              << localOptimization.fullEquivHybridStepParetoSelectedGradientSum
              << " full_equiv_hybrid_step_pareto_gradient_eval_count="
              << localOptimization.fullEquivHybridStepParetoGradientEvalCount
              << " full_equiv_hybrid_step_pareto_gradient_eval_time_sec="
              << localOptimization.fullEquivHybridStepParetoGradientEvalTimeSec
              << " full_equiv_hybrid_translation_recovery_step_trial_candidate_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryStepTrialCandidateCount
              << " full_equiv_hybrid_translation_recovery_step_trial_hard_accepted_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount
              << " full_equiv_hybrid_translation_recovery_step_trial_selected_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryStepTrialSelectedCount
              << " full_equiv_hybrid_active_separator_candidate_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorCandidateCount
              << " full_equiv_hybrid_active_separator_accepted_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorAcceptedCount
              << " full_equiv_hybrid_active_separator_rejected_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorRejectedCount
              << " full_equiv_hybrid_active_separator_step_sum="
              << localOptimization.fullEquivHybridActiveSeparatorStepSum
              << " full_equiv_hybrid_active_separator_cost_decrease_sum="
              << localOptimization
                     .fullEquivHybridActiveSeparatorCostDecreaseSum
              << " full_equiv_hybrid_active_separator_lm_schur_candidate_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurCandidateCount
              << " full_equiv_hybrid_active_separator_lm_schur_accepted_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurAcceptedCount
              << " full_equiv_hybrid_active_separator_lm_schur_guard_rejected_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount
              << " full_equiv_hybrid_active_separator_lm_schur_gradient_guard_rejected_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount
              << " full_equiv_hybrid_active_separator_lm_schur_solve_failure_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurSolveFailureCount
              << " full_equiv_hybrid_active_separator_lm_schur_fallback_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurFallbackCount
              << " full_equiv_hybrid_active_separator_lm_schur_boundary_col_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurBoundaryColCount
              << " full_equiv_hybrid_active_separator_lm_schur_private_col_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurPrivateColCount
              << " full_equiv_hybrid_active_separator_lm_schur_backtracking_trial_count="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount
              << " full_equiv_hybrid_active_separator_lm_schur_alpha_sum="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurAlphaSum
              << " full_equiv_hybrid_active_separator_lm_schur_cost_decrease_sum="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum
              << " full_equiv_hybrid_active_separator_lm_schur_gradient_change_sum="
              << localOptimization
                     .fullEquivHybridActiveSeparatorLmSchurGradientChangeSum
              << " full_equiv_hybrid_translation_recovery_polish_attempt_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishAttemptCount
              << " full_equiv_hybrid_translation_recovery_polish_accepted_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishAcceptedCount
              << " full_equiv_hybrid_translation_recovery_polish_rejected_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishRejectedCount
              << " full_equiv_hybrid_translation_recovery_polish_gradient_guard_rejected_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount
              << " full_equiv_hybrid_translation_recovery_polish_backtracking_trigger_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount
              << " full_equiv_hybrid_translation_recovery_polish_backtracking_trial_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount
              << " full_equiv_hybrid_translation_recovery_polish_backtracking_accepted_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount
              << " full_equiv_hybrid_translation_recovery_polish_backtracking_alpha_sum="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum
              << " full_equiv_hybrid_translation_recovery_polish_merit_candidate_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishMeritCandidateCount
              << " full_equiv_hybrid_translation_recovery_polish_merit_selected_full_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount
              << " full_equiv_hybrid_translation_recovery_polish_merit_selected_partial_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount
              << " full_equiv_hybrid_translation_recovery_polish_merit_selected_alpha_sum="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum
              << " full_equiv_hybrid_translation_recovery_polish_merit_best_decrease_sum="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum
              << " full_equiv_hybrid_translation_recovery_polish_merit_selected_decrease_sum="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum
              << " full_equiv_hybrid_translation_recovery_polish_cost_decrease_sum="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishCostDecreaseSum
              << " full_equiv_hybrid_translation_recovery_polish_gradient_change_sum="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryPolishGradientChangeSum
              << " full_equiv_hybrid_sparse_matvec_count="
              << localOptimization.fullEquivHybridSparseMatrixVectorProductCount
              << " full_equiv_hybrid_reduced_rotation_initial_guess_candidate_count="
              << localOptimization
                     .fullEquivHybridReducedRotationInitialGuessCandidateCount
              << " full_equiv_hybrid_reduced_rotation_initial_guess_used_count="
              << localOptimization
                     .fullEquivHybridReducedRotationInitialGuessUsedCount
              << " full_equiv_hybrid_reduced_rotation_initial_guess_rejected_count="
              << localOptimization
                     .fullEquivHybridReducedRotationInitialGuessRejectedCount
              << " full_equiv_hybrid_translation_recovery_initial_guess_candidate_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryInitialGuessCandidateCount
              << " full_equiv_hybrid_translation_recovery_initial_guess_used_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryInitialGuessUsedCount
              << " full_equiv_hybrid_translation_recovery_initial_guess_rejected_count="
              << localOptimization
                     .fullEquivHybridTranslationRecoveryInitialGuessRejectedCount
              << " full_equiv_hybrid_translation_schur_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridTranslationSchurPreconditionerApplicationCount
              << " full_equiv_hybrid_translation_schur_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridTranslationSchurPreconditionerFactorizationCount
              << " full_equiv_hybrid_translation_schur_preconditioner_fallback_count="
              << localOptimization
                     .fullEquivHybridTranslationSchurPreconditionerFallbackCount
              << " full_equiv_hybrid_local_chain_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridLocalChainPreconditionerApplicationCount
              << " full_equiv_hybrid_local_chain_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridLocalChainPreconditionerFactorizationCount
              << " full_equiv_hybrid_local_chain_preconditioner_fallback_count="
              << localOptimization
                     .fullEquivHybridLocalChainPreconditionerFallbackCount
              << " full_equiv_hybrid_translation_block_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridTranslationBlockPreconditionerApplicationCount
              << " full_equiv_hybrid_translation_block_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridTranslationBlockPreconditionerFactorizationCount
              << " full_equiv_hybrid_translation_block_preconditioner_fallback_count="
              << localOptimization
                     .fullEquivHybridTranslationBlockPreconditionerFallbackCount
              << " full_equiv_hybrid_translation_sparse_schur_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount
              << " full_equiv_hybrid_translation_sparse_schur_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount
              << " full_equiv_hybrid_translation_sparse_schur_preconditioner_fallback_count="
              << localOptimization
                     .fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount
              << " full_equiv_hybrid_translation_local_schur_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount
              << " full_equiv_hybrid_translation_local_schur_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount
              << " full_equiv_hybrid_translation_local_schur_preconditioner_fallback_count="
              << localOptimization
                     .fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount
              << " full_equiv_hybrid_translation_local_schur_preconditioner_active_pose_count="
              << localOptimization
                     .fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount
              << " full_equiv_hybrid_translation_local_schur_preconditioner_active_column_count="
              << localOptimization
                     .fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount
              << " full_equiv_hybrid_laplacian_deflation_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridLaplacianDeflationPreconditionerApplicationCount
              << " full_equiv_hybrid_laplacian_deflation_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount
              << " full_equiv_hybrid_laplacian_deflation_preconditioner_fallback_count="
              << localOptimization
                     .fullEquivHybridLaplacianDeflationPreconditionerFallbackCount
              << " full_equiv_hybrid_laplacian_deflation_preconditioner_basis_dimension="
              << localOptimization
                     .fullEquivHybridLaplacianDeflationPreconditionerBasisDimension
              << " full_equiv_hybrid_reduced_rotation_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridReducedRotationPreconditionerApplicationCount
              << " full_equiv_hybrid_reduced_rotation_preconditioner_factorization_count="
              << localOptimization
                     .fullEquivHybridReducedRotationPreconditionerFactorizationCount
              << " full_equiv_hybrid_rqn_used_count="
              << localOptimization.fullEquivHybridRqnUsedCount
              << " full_equiv_hybrid_rqn_accepted_pair_count="
              << localOptimization.fullEquivHybridRqnAcceptedPairCount
              << " full_equiv_hybrid_rqn_rejected_pair_count="
              << localOptimization.fullEquivHybridRqnRejectedPairCount
              << " full_equiv_hybrid_rqn_memory_size="
              << localOptimization.fullEquivHybridRqnMemorySize
              << " full_equiv_hybrid_rqn_preconditioner_application_count="
              << localOptimization
                     .fullEquivHybridRqnPreconditionerApplicationCount
              << " adaptive_refinement_count="
              << localOptimization.adaptiveRefinements
              << " extrapolation_accepted_count="
              << localOptimization.extrapolationAccepted
              << " extrapolation_rejected_count="
              << localOptimization.extrapolationRejected
              << " g_extrapolation_accepted_count="
              << localOptimization.gExtrapolationAccepted
              << " g_extrapolation_rejected_count="
              << localOptimization.gExtrapolationRejected
              << " coupled_extrapolation_accepted_count="
              << localOptimization.coupledExtrapolationAccepted
              << " coupled_extrapolation_rejected_count="
              << localOptimization.coupledExtrapolationRejected
              << " anderson_accepted_count="
              << localOptimization.andersonAccepted
              << " anderson_rejected_count="
              << localOptimization.andersonRejected
              << " squarem_accepted_count="
              << localOptimization.squaremAccepted
              << " squarem_rejected_count="
              << localOptimization.squaremRejected
              << " global_extrapolation_accepted_count="
              << globalExtrapolationAccepted
              << " global_extrapolation_rejected_count="
              << globalExtrapolationRejected
              << " global_anderson_accepted_count="
              << globalAndersonAccepted
              << " global_anderson_rejected_count="
              << globalAndersonRejected
              << " local_gradient_accepted_count="
              << localGradientCorrection.accepted
              << " local_gradient_rejected_count="
              << localGradientCorrection.rejected
              << " local_gradient_coupled_direction_candidate_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionCandidateCount
              << " local_gradient_coupled_direction_accepted_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionAcceptedCount
              << " local_gradient_coupled_direction_rejected_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionRejectedCount
              << " local_gradient_coupled_direction_packet_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionPacketCount
              << " local_gradient_coupled_direction_skipped_packet_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionSkippedPacketCount
              << " local_gradient_coupled_direction_score_skipped_packet_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionScoreSkippedPacketCount
              << " local_gradient_coupled_direction_byte_budget_skipped_packet_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionByteBudgetSkippedPacketCount
              << " local_gradient_coupled_direction_scalar_count="
              << mergedLocalGradientDiagnostics
                     .coupledDirectionScalarCount
              << " local_gradient_coupled_direction_scalar_comm_mb="
              << scalarMegabytes(
                     mergedLocalGradientDiagnostics
                         .coupledDirectionScalarCount)
              << " local_gradient_compact_schur_candidate_count="
              << mergedLocalGradientDiagnostics.compactSchurCandidateCount
              << " local_gradient_compact_schur_accepted_count="
              << mergedLocalGradientDiagnostics.compactSchurAcceptedCount
              << " local_gradient_compact_schur_guard_rejected_count="
              << mergedLocalGradientDiagnostics
                     .compactSchurGuardRejectedCount
              << " local_gradient_compact_schur_gradient_guard_rejected_count="
              << mergedLocalGradientDiagnostics
                     .compactSchurGradientGuardRejectedCount
              << " local_gradient_compact_schur_solve_failure_count="
              << mergedLocalGradientDiagnostics
                     .compactSchurSolveFailureCount
              << " local_gradient_compact_schur_boundary_col_count="
              << mergedLocalGradientDiagnostics.compactSchurBoundaryColCount
              << " local_gradient_compact_schur_private_col_count="
              << mergedLocalGradientDiagnostics.compactSchurPrivateColCount
              << " local_gradient_compact_schur_gradient_change_sum="
              << mergedLocalGradientDiagnostics.compactSchurGradientChangeSum
              << " global_gradient_accepted_count="
              << globalGradientAccepted
              << " global_gradient_rejected_count="
              << globalGradientRejected
              << " local_history_sync_count="
              << localHistorySyncCount
              << " local_model_cost_before=" << localModelBefore.cost
              << " local_model_cost_after=" << localModelAfter.cost
              << " local_model_gradient_before=" << localModelBefore.gradient
              << " local_model_gradient_after=" << localModelAfter.gradient
              << " boundary_edge_cost_before="
              << localOptimization.boundaryEdgeCostBefore
              << " boundary_edge_cost_after="
              << localOptimization.boundaryEdgeCostAfter
              << " separator_delta_norm="
              << std::sqrt(localOptimization.separatorDeltaNormSquared)
              << " boundary_proximal_candidate_count="
              << localOptimization.boundaryProximalCandidateCount
              << " boundary_proximal_accepted_count="
              << localOptimization.boundaryProximalAcceptedCount
              << " edge_tight_quadratic_eval_count="
              << localOptimization.edgeTightQuadraticEvalCount
              << " edge_tight_quadratic_surrogate_cost_sum="
              << localOptimization.edgeTightQuadraticSurrogateCostSum
              << " edge_tight_quadratic_true_cost_sum="
              << localOptimization.edgeTightQuadraticTrueCostSum
              << " edge_tight_quadratic_majorization_gap_min="
              << (localOptimization.edgeTightQuadraticEvalCount == 0
                      ? 0.0
                      : localOptimization
                            .edgeTightQuadraticMajorizationGapMin)
              << " amm_accelerated_accepted_count="
              << localOptimization.ammAcceleratedAccepted
              << " amm_restart_count=" << localOptimization.ammRestart
              << " amm_hard_restart_count="
              << localOptimization.ammHardRestart
              << " amm_soft_restart_count="
              << localOptimization.ammSoftRestart
              << " amm_phi_fallback_count="
              << localOptimization.ammPhiFallback
              << " amm_local_merit_rejected_count="
              << localOptimization.ammLocalMeritRejected
              << " amm_proximal_start_count="
              << localOptimization.ammProximalStart
              << " amm_skipped_count="
              << localOptimization.ammSkipped
              << " amm_trace_count=" << localOptimization.ammTraceCount
              << " amm_refined_count=" << localOptimization.ammRefined
              << " amm_prox_reset_count=" << localOptimization.ammProxReset
              << " amm_restart_used_xakh_count="
              << localOptimization.ammRestartUsedXakh
              << " amm_restart_certificate_count="
              << localOptimization.ammRestartCertificateCount
              << " amm_restart_certificate_passed_count="
              << localOptimization.ammRestartCertificatePassedCount
              << " amm_restart_certificate_failed_count="
              << localOptimization.ammRestartCertificateFailedCount
              << " amm_restart_certificate_min_margin="
              << (localOptimization.ammRestartCertificateCount == 0
                      ? 0.0
                      : localOptimization.ammRestartCertificateMinMargin)
              << " amm_translation_recovery_attempt_count="
              << localOptimization.ammTranslationRecoveryAttemptCount
              << " amm_translation_recovery_accepted_count="
              << localOptimization.ammTranslationRecoveryAcceptedCount
              << " amm_translation_recovery_rejected_count="
              << localOptimization.ammTranslationRecoveryRejectedCount
              << " amm_mixed_surrogate_candidate_count="
              << localOptimization.ammMixedSurrogateCandidateCount
              << " amm_mixed_surrogate_true_local_accepted_count="
              << localOptimization.ammMixedSurrogateTrueLocalAcceptedCount
              << " amm_mixed_surrogate_simple_selected_count="
              << localOptimization.ammMixedSurrogateSimpleSelectedCount
              << " amm_mixed_surrogate_true_local_selected_count="
              << localOptimization.ammMixedSurrogateTrueLocalSelectedCount
              << " amm_mixed_surrogate_extrapolated_selected_count="
              << localOptimization.ammMixedSurrogateExtrapolatedSelectedCount
              << " amm_mixed_surrogate_other_selected_count="
              << localOptimization.ammMixedSurrogateOtherSelectedCount
              << " amm_mixed_surrogate_simple_skipped_count="
              << localOptimization.ammMixedSurrogateSimpleSkippedCount
              << " amm_mixed_surrogate_simple_forced_refresh_count="
              << localOptimization.ammMixedSurrogateSimpleForcedRefreshCount
              << " amm_mean_gamma="
              << ammMean(localOptimization.ammGammaSum)
              << " amm_mean_Gkh_initial="
              << ammMean(localOptimization.ammGkhInitialSum)
              << " amm_mean_minG=" << ammMean(localOptimization.ammMinGSum)
              << " amm_mean_Gk_after_accelerated="
              << ammMean(localOptimization.ammGkAfterAcceleratedSum)
              << " amm_mean_Gkh_after_restart_check="
              << ammMean(localOptimization.ammGkhAfterRestartCheckSum)
              << " amm_mean_final_Gk="
              << ammMean(localOptimization.ammFinalGkSum)
              << " amm_mean_phi_lhs="
              << ammMean(localOptimization.ammPhiLhsSum)
                << " amm_mean_phi_rhs="
                << ammMean(localOptimization.ammPhiRhsSum)
                << std::endl;
    }
  }

  result.finalEstimate = X;
  result.finalObjective = globalCost;
  result.finalGradient = globalGrad;
  result.elapsedSeconds = elapsed / static_cast<double>(options.numRobots);
  std::vector<ManualDpgoMmAgentProfileEntry> agentProfiles;
  if (options.profileOptimizer) {
    agentProfiles.reserve(agents.size());
    for (const auto &agent : agents) {
      ManualDpgoMmAgentProfileEntry entry;
      entry.robotId = agent->getID();
      entry.profile = agent->getOptimizerProfile();
      agentProfiles.push_back(entry);
      optimizerProfile.addAgentProfile(entry.profile);
    }
    result.optimizerReducedSolveCount = optimizerProfile.reducedSolveCount;
  }

  if (options.save) {
    const auto outputProfileStart =
        options.profileOptimizer ? ProfileClock::now()
                                 : ProfileClock::time_point();
    const std::string outputDir =
        options.outputDirectory.empty() ? "." : options.outputDirectory;
    ensureOutputDirectory(outputDir);
    const std::string baseName = datasetStem(datasetPath);
    const std::string resultPath =
        outputDir + "/results_chordal_" + baseName + "_" +
        std::to_string(options.numRobots) + "_" +
        manualDpgoMmSchemeName(options.scheme) + "_" +
        manualDpgoMmLocalSolverName(options.localSolver) + ".txt";
    const std::string estimatePath = outputDir + "/estimates_trivial.txt";
    const std::string iterationSummaryPath =
        outputDir + "/iteration_summary.csv";

    writeIterationSummary(iterationSummaryPath, result.iterations);
    writeEstimate(estimatePath, result.finalEstimate, d);
    if (options.saveIterationEstimates) {
      writeIterationEstimates(outputDir + "/initialization_estimates",
                              result.initializationEstimates, d, "init");
      writeIterationEstimates(outputDir + "/iteration_estimates",
                              result.iterationEstimates, d);
    }
    if (options.traceAmm) {
      writeAmmTrace(outputDir + "/amm_trace.csv", ammTraceRows);
    }

    std::ofstream output(resultPath);
    if (!output.is_open()) {
      throw std::runtime_error("Unable to write result summary: " +
                               resultPath);
    }
    output << std::setprecision(20);
    output << "final objective: " << result.finalObjective << "\n";
    output << "final gradient: " << result.finalGradient << "\n";
    if (options.debugSurrogateBoundCheck) {
      output << "surrogate bound checks: "
             << result.surrogateBoundCheckCount
             << " violations: " << result.surrogateBoundViolationCount
             << " min margin: " << result.surrogateBoundMinMargin << "\n";
    }
    output << "time: " << result.elapsedSeconds << " s/node.\n";
    if (!result.iterations.empty()) {
      const ManualDpgoMmIterationSummary &last = result.iterations.back();
      output << "total comm poses: " << last.cumulativeCommPoseCount << "\n";
      output << "total comm MB: " << last.cumulativeCommMb << "\n";
    }

    if (options.profileOptimizer) {
      optimizerProfile.outputWriteSec += secondsSince(outputProfileStart);
      optimizerProfile.totalWallSec = secondsSince(profileTotalStart);
      writeOptimizerProfile(outputDir + "/optimizer_profile.json",
                            optimizerProfile, agentProfiles, options,
                            datasetPath);
    }
  }

  return result;
}

std::vector<ManualDpgoMmLocalModelSummary>
inspectManualDpgoMmInitialLocalModels(const std::string &datasetPath,
                                      const ManualDpgoMmOptions &options) {
  ManualDpgoMmPreparedProblem prepared =
      prepareManualDpgoMmProblem(datasetPath, options, false);
  DirectedPoseCache deliveredPoseCache;
  exchangeOwnedSeparatorPoses(prepared.agents, deliveredPoseCache);

  std::vector<ManualDpgoMmLocalModelSummary> summaries;
  summaries.reserve(prepared.agents.size());
  for (const auto &agent : prepared.agents) {
    ManualDpgoMmLocalModelSummary summary;
    summary.robot = agent->getID();
    summary.ok = agent->evaluateLocalModel(summary.cost, summary.gradient);
    summaries.push_back(summary);
  }
  return summaries;
}

std::vector<ManualDpgoMmLocalModelSnapshot>
inspectManualDpgoMmInitialLocalModelSnapshots(
    const std::string &datasetPath, const ManualDpgoMmOptions &options) {
  ManualDpgoMmPreparedProblem prepared =
      prepareManualDpgoMmProblem(datasetPath, options, false);
  DirectedPoseCache deliveredPoseCache;
  exchangeOwnedSeparatorPoses(prepared.agents, deliveredPoseCache);

  std::vector<ManualDpgoMmLocalModelSnapshot> snapshots;
  snapshots.reserve(prepared.agents.size());
  for (const auto &agent : prepared.agents) {
    ManualDpgoMmLocalModelSnapshot snapshot;
    snapshot.robot = agent->getID();
    snapshot.ok = agent->updateLocalModel();
    if (snapshot.ok) {
      snapshot.q = agent->getQ();
      snapshot.g = agent->getG();
    }
    snapshots.push_back(std::move(snapshot));
  }
  return snapshots;
}

}  // namespace DPGO
