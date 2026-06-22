#include <DPGO/GeodesicSE3.h>
#include <DPGO/ObjectGaugeCoupling.h>

#include <gtest/gtest.h>

#include <cmath>

using DPGO::Matrix;

namespace {

Matrix makeState(const std::vector<Matrix> &poses) {
  Matrix state = Matrix::Zero(3, 4 * poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    state.block(0, 4 * i, 3, 4) = poses[i];
  }
  return state;
}

Matrix makePose(const Eigen::Vector3d &w, const Eigen::Vector3d &t) {
  Matrix pose = Matrix::Zero(3, 4);
  pose.leftCols(3) = DPGO::so3Exp(w);
  pose.rightCols(1) = t;
  return pose;
}

Matrix leftTransformState(const Matrix &state, const Matrix &R,
                          const Matrix &t) {
  Matrix transformed = state;
  for (int pose = 0; pose < state.cols() / 4; ++pose) {
    const Matrix Rpose = state.block(0, 4 * pose, 3, 3);
    const Matrix tpose = state.block(0, 4 * pose + 3, 3, 1);
    transformed.block(0, 4 * pose, 3, 3) = R * Rpose;
    transformed.block(0, 4 * pose + 3, 3, 1) = R * tpose + t;
  }
  return transformed;
}

double relativeResidualNorm(const Matrix &state, size_t i, size_t j,
                            const Matrix &Rij, const Matrix &tij) {
  const Matrix Ri = state.block(0, 4 * i, 3, 3);
  const Matrix ti = state.block(0, 4 * i + 3, 3, 1);
  const Matrix Rj = state.block(0, 4 * j, 3, 3);
  const Matrix tj = state.block(0, 4 * j + 3, 3, 1);
  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3LogResidual(Ri, ti, Rij, tij, Rj, tj);
  return std::sqrt(residual.rotation.squaredNorm() +
                   residual.translation.squaredNorm());
}

}  // namespace

TEST(ObjectGaugeCoupling, EstimatesAndAppliesLeftGaugeCorrection) {
  const Matrix pose0 =
      makePose(Eigen::Vector3d(0.1, -0.2, 0.05),
               Eigen::Vector3d(0.2, -0.1, 0.3));
  const Matrix pose1 =
      makePose(Eigen::Vector3d(-0.2, 0.15, 0.08),
               Eigen::Vector3d(0.9, -0.4, 0.6));
  const Matrix object0 =
      makePose(Eigen::Vector3d(0.25, 0.05, -0.1),
               Eigen::Vector3d(1.7, -0.2, 0.4));
  const Matrix object1 =
      makePose(Eigen::Vector3d(-0.1, 0.35, 0.2),
               Eigen::Vector3d(-0.4, 1.1, 0.8));
  const Matrix targetState = makeState({pose0, pose1, object0, object1});

  const Matrix gaugeR =
      DPGO::so3Exp(Eigen::Vector3d(0.2, -0.12, 0.18));
  const Matrix gaugeT =
      (Eigen::Vector3d() << 0.7, -0.3, 0.25).finished();
  const Matrix localState = leftTransformState(targetState, gaugeR, gaugeT);

  std::vector<DPGO::GaugeCouplingObservation> observations;
  observations.push_back(DPGO::GaugeCouplingObservation{2, object0, 1.0});
  observations.push_back(DPGO::GaugeCouplingObservation{3, object1, 1.0});

  DPGO::GaugeCouplingCorrectionStats stats;
  Matrix correctionR;
  Matrix correctionT;
  ASSERT_TRUE(DPGO::estimateSE3GaugeCorrection(
      localState, observations, 3, 2, correctionR, correctionT, &stats));
  ASSERT_TRUE(stats.valid);
  ASSERT_EQ(stats.usedObjects, 2u);

  const Matrix corrected = DPGO::applySE3GaugeCorrection(
      localState, correctionR, correctionT, 1.0, 0.0, 3, &stats);
  ASSERT_LE((corrected - targetState).norm(), 1e-8);

  const Matrix Rij = pose0.leftCols(3).transpose() * pose1.leftCols(3);
  const Matrix tij = pose0.leftCols(3).transpose() *
                     (pose1.rightCols(1) - pose0.rightCols(1));
  ASSERT_NEAR(relativeResidualNorm(localState, 0, 1, Rij, tij),
              relativeResidualNorm(corrected, 0, 1, Rij, tij), 1e-10);
}

TEST(ObjectGaugeCoupling, CapsGaugeCorrectionStep) {
  const Matrix object0 =
      makePose(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.0));
  const Matrix object1 =
      makePose(Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0));
  const Matrix targetState = makeState({object0, object1});
  const Matrix gaugeR = Matrix::Identity(3, 3);
  const Matrix gaugeT =
      (Eigen::Vector3d() << 1.0, 0.0, 0.0).finished();
  const Matrix localState = leftTransformState(targetState, gaugeR, gaugeT);

  std::vector<DPGO::GaugeCouplingObservation> observations;
  observations.push_back(DPGO::GaugeCouplingObservation{0, object0, 1.0});
  observations.push_back(DPGO::GaugeCouplingObservation{1, object1, 1.0});

  DPGO::GaugeCouplingCorrectionStats stats;
  Matrix correctionR;
  Matrix correctionT;
  ASSERT_TRUE(DPGO::estimateSE3GaugeCorrection(
      localState, observations, 3, 2, correctionR, correctionT, &stats));

  const Matrix corrected = DPGO::applySE3GaugeCorrection(
      localState, correctionR, correctionT, 1.0, 0.25, 3, &stats);
  ASSERT_LT(stats.appliedScale, 1.0);
  ASSERT_NEAR(stats.appliedStepNorm, 0.25, 1e-10);
  ASSERT_GT((corrected - targetState).norm(), 0.1);
}

TEST(ObjectGaugeCoupling, FreshnessWeightDecaysAndRejectsOldCache) {
  ASSERT_DOUBLE_EQ(DPGO::cacheFreshnessWeight(10, 10, 0, 0.0), 1.0);
  ASSERT_DOUBLE_EQ(DPGO::cacheFreshnessWeight(10, 4, 0, 0.0), 1.0);
  ASSERT_NEAR(DPGO::cacheFreshnessWeight(10, 8, 0, 0.5),
              std::exp(-1.0), 1e-12);
  ASSERT_DOUBLE_EQ(DPGO::cacheFreshnessWeight(10, 6, 3, 0.5), 0.0);
}

TEST(ObjectGaugeCoupling, CountsFrameRegistrationPayloadMb) {
  const double mb = DPGO::frameRegistrationPayloadMb({2, 3}, 3);
  const double expected =
      static_cast<double>((2 + 3) * 3 * 4 * sizeof(double)) /
      (1024.0 * 1024.0);
  ASSERT_NEAR(mb, expected, 1e-12);
  ASSERT_DOUBLE_EQ(DPGO::frameRegistrationPayloadMb({}, 3), 0.0);
}

TEST(ObjectGaugeCoupling, MeritFilterBacktracksAndRejectsBadSteps) {
  const Matrix object0 =
      makePose(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.0));
  const Matrix object1 =
      makePose(Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0));
  const Matrix state = makeState({object0, object1});
  const Matrix R = Matrix::Identity(3, 3);
  const Matrix t =
      (Eigen::Vector3d() << 1.0, 0.0, 0.0).finished();

  DPGO::GaugeCouplingCorrectionStats stats;
  auto boundedMerit = [](const Matrix &candidate) {
    const double displacement = std::abs(candidate(0, 3));
    return displacement <= 0.5 ? displacement : 10.0 + displacement;
  };
  const Matrix accepted = DPGO::applySE3GaugeCorrectionWithMerit(
      state, R, t, 1.0, 0.0, 3, 3, 0.5, boundedMerit, &stats);
  ASSERT_TRUE(stats.meritAccepted);
  ASSERT_EQ(stats.meritRejectedSteps, 1u);
  ASSERT_NEAR(stats.appliedScale, 0.5, 1e-12);
  ASSERT_NEAR(accepted(0, 3), 0.5, 1e-12);

  auto rejectingMerit = [](const Matrix &candidate) {
    return candidate(0, 3) == 0.0 ? 0.0 : 1.0;
  };
  const Matrix rejected = DPGO::applySE3GaugeCorrectionWithMerit(
      state, R, t, 1.0, 0.0, 3, 2, 0.0, rejectingMerit, &stats);
  ASSERT_FALSE(stats.meritAccepted);
  ASSERT_EQ(stats.meritRejectedSteps, 3u);
  ASSERT_LE((rejected - state).norm(), 1e-12);
}
