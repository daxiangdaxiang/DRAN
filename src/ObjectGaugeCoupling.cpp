#include <DPGO/ObjectGaugeCoupling.h>

#include <DPGO/DPGO_utils.h>
#include <DPGO/GeodesicSE3.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace DPGO {

namespace {

double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

bool validPoseBlock(const Matrix &pose, int d) {
  return pose.rows() == d && pose.cols() == d + 1;
}

double correctionStepNorm(const Matrix &R, const Matrix &t, unsigned d) {
  if (d == 3) {
    const Eigen::Vector3d w = so3Log(R);
    return std::sqrt(w.squaredNorm() + t.squaredNorm());
  }
  return std::sqrt((R - Matrix::Identity(d, d)).squaredNorm() +
                   t.squaredNorm());
}

}  // namespace

double cacheFreshnessWeight(size_t currentIter, size_t lastUpdateIter,
                            size_t maxAge, double decayRate) {
  const size_t age =
      currentIter >= lastUpdateIter ? currentIter - lastUpdateIter : 0;
  if (maxAge > 0 && age > maxAge) {
    return 0.0;
  }
  if (decayRate <= 0.0) {
    return 1.0;
  }
  return std::exp(-decayRate * static_cast<double>(age));
}

double frameRegistrationPayloadMb(
    const std::vector<size_t> &commonObjectCounts, unsigned d) {
  if (d == 0) {
    return 0.0;
  }
  size_t poseBlocks = 0;
  for (size_t count : commonObjectCounts) {
    poseBlocks += count;
  }
  const double bytes =
      static_cast<double>(poseBlocks) * static_cast<double>(d) *
      static_cast<double>(d + 1) * sizeof(double);
  return bytes / (1024.0 * 1024.0);
}

bool estimateSE3GaugeCorrection(
    const Matrix &state,
    const std::vector<GaugeCouplingObservation> &observations,
    unsigned d, size_t minObjects, Matrix &R, Matrix &t,
    GaugeCouplingCorrectionStats *stats) {
  if (stats) {
    *stats = GaugeCouplingCorrectionStats();
  }
  if (d != 3 || state.rows() != static_cast<int>(d) ||
      state.cols() % static_cast<int>(d + 1) != 0) {
    return false;
  }

  const int dim = static_cast<int>(d);
  const int blockCols = static_cast<int>(d + 1);
  Matrix rotationMoment = Matrix::Zero(dim, dim);
  Matrix sourceTranslationSum = Matrix::Zero(dim, 1);
  Matrix targetTranslationSum = Matrix::Zero(dim, 1);
  double weightSum = 0.0;
  size_t usedObjects = 0;

  for (const GaugeCouplingObservation &obs : observations) {
    if (obs.weight <= 0.0 || !std::isfinite(obs.weight) ||
        !validPoseBlock(obs.targetPose, dim)) {
      continue;
    }
    const int startCol = static_cast<int>(obs.poseIndex) * blockCols;
    if (startCol < 0 || startCol + blockCols > state.cols()) {
      continue;
    }

    const Matrix sourceR = state.block(0, startCol, dim, dim);
    const Matrix sourceT = state.block(0, startCol + dim, dim, 1);
    const Matrix targetR = obs.targetPose.block(0, 0, dim, dim);
    const Matrix targetT = obs.targetPose.block(0, dim, dim, 1);
    rotationMoment += obs.weight * targetR * sourceR.transpose();
    sourceTranslationSum += obs.weight * sourceT;
    targetTranslationSum += obs.weight * targetT;
    weightSum += obs.weight;
    ++usedObjects;
  }

  if (usedObjects < minObjects ||
      weightSum <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  R = projectToRotationGroup(rotationMoment);
  t = targetTranslationSum / weightSum - R * (sourceTranslationSum / weightSum);

  double translationSquared = 0.0;
  double rotationSquared = 0.0;
  for (const GaugeCouplingObservation &obs : observations) {
    if (obs.weight <= 0.0 || !std::isfinite(obs.weight) ||
        !validPoseBlock(obs.targetPose, dim)) {
      continue;
    }
    const int startCol = static_cast<int>(obs.poseIndex) * blockCols;
    if (startCol < 0 || startCol + blockCols > state.cols()) {
      continue;
    }
    const Matrix sourceR = state.block(0, startCol, dim, dim);
    const Matrix sourceT = state.block(0, startCol + dim, dim, 1);
    const Matrix targetR = obs.targetPose.block(0, 0, dim, dim);
    const Matrix targetT = obs.targetPose.block(0, dim, dim, 1);
    const Matrix translationResidual = R * sourceT + t - targetT;
    const Eigen::Vector3d rotationResidual =
        so3Log(targetR.transpose() * R * sourceR);
    translationSquared += obs.weight * translationResidual.squaredNorm();
    rotationSquared += obs.weight * rotationResidual.squaredNorm();
  }

  if (stats) {
    stats->valid = true;
    stats->usedObjects = usedObjects;
    stats->translationRms = std::sqrt(translationSquared / weightSum);
    stats->rotationRms = std::sqrt(rotationSquared / weightSum);
    stats->score =
        std::sqrt((translationSquared + rotationSquared) / weightSum);
    stats->rawStepNorm = correctionStepNorm(R, t, d);
    stats->appliedScale = 0.0;
    stats->appliedStepNorm = 0.0;
  }
  return true;
}

Matrix applySE3GaugeCorrection(
    const Matrix &state, const Matrix &R, const Matrix &t, double alpha,
    double maxStep, unsigned d, GaugeCouplingCorrectionStats *stats) {
  if (d != 3 || state.rows() != static_cast<int>(d) ||
      state.cols() % static_cast<int>(d + 1) != 0) {
    return state;
  }

  const int dim = static_cast<int>(d);
  const int blockCols = static_cast<int>(d + 1);
  double scale = clamp(alpha, 0.0, 1.0);
  const double rawStepNorm = correctionStepNorm(R, t, d);
  if (maxStep > 0.0 && rawStepNorm * scale > maxStep) {
    scale *= maxStep / std::max(rawStepNorm * scale, 1e-12);
  }

  const Eigen::Vector3d w = so3Log(R);
  const Matrix stepR = so3Exp(scale * w);
  const Matrix stepT = scale * t;

  Matrix corrected = state;
  for (int startCol = 0; startCol < state.cols(); startCol += blockCols) {
    const Matrix sourceR = state.block(0, startCol, dim, dim);
    const Matrix sourceT = state.block(0, startCol + dim, dim, 1);
    corrected.block(0, startCol, dim, dim) = stepR * sourceR;
    corrected.block(0, startCol + dim, dim, 1) = stepR * sourceT + stepT;
  }

  if (stats) {
    stats->valid = true;
    stats->rawStepNorm = rawStepNorm;
    stats->appliedScale = scale;
    stats->appliedStepNorm = rawStepNorm * scale;
  }
  return corrected;
}

Matrix applySE3GaugeCorrectionWithMerit(
    const Matrix &state, const Matrix &R, const Matrix &t, double alpha,
    double maxStep, unsigned d, size_t maxBacktracking,
    double meritTolerance,
    const std::function<double(const Matrix &)> &meritFunction,
    GaugeCouplingCorrectionStats *stats) {
  if (!meritFunction) {
    return applySE3GaugeCorrection(state, R, t, alpha, maxStep, d, stats);
  }

  GaugeCouplingCorrectionStats localStats;
  const double initialMerit = meritFunction(state);
  double trialAlpha = alpha;
  for (size_t attempt = 0; attempt <= maxBacktracking; ++attempt) {
    GaugeCouplingCorrectionStats trialStats;
    Matrix candidate =
        applySE3GaugeCorrection(state, R, t, trialAlpha, maxStep, d,
                                &trialStats);
    const double candidateMerit = meritFunction(candidate);
    if (candidateMerit <= initialMerit + meritTolerance) {
      localStats = trialStats;
      localStats.meritAccepted = true;
      localStats.meritRejectedSteps = attempt;
      localStats.meritInitial = initialMerit;
      localStats.meritFinal = candidateMerit;
      if (stats) {
        *stats = localStats;
      }
      return candidate;
    }
    trialAlpha *= 0.5;
  }

  localStats.valid = false;
  localStats.meritAccepted = false;
  localStats.meritRejectedSteps = maxBacktracking + 1;
  localStats.meritInitial = initialMerit;
  localStats.meritFinal = initialMerit;
  if (stats) {
    *stats = localStats;
  }
  return state;
}

}  // namespace DPGO
