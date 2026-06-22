#include <DPGO/GeodesicSE3.h>

#include <gtest/gtest.h>

#include <cmath>

using DPGO::Matrix;

TEST(GeodesicSE3, ExpLogRoundTripSmallAngle) {
  Eigen::Vector3d w;
  w << 1e-4, -2e-4, 3e-4;
  Matrix R = DPGO::so3Exp(w);
  Eigen::Vector3d recovered = DPGO::so3Log(R);
  ASSERT_LE((recovered - w).norm(), 1e-10);
}

TEST(GeodesicSE3, ExpLogRoundTripModerateAngle) {
  Eigen::Vector3d w;
  w << 0.2, -0.3, 0.1;
  Matrix R = DPGO::so3Exp(w);
  Eigen::Vector3d recovered = DPGO::so3Log(R);
  ASSERT_LE((recovered - w).norm(), 1e-10);
}

TEST(GeodesicSE3, ExpLogRoundTripNearPi) {
  Eigen::Vector3d axis;
  axis << 1.0, -2.0, 0.5;
  axis.normalize();
  Eigen::Vector3d w = (M_PI - 1e-7) * axis;
  Matrix R = DPGO::so3Exp(w);
  Eigen::Vector3d recovered = DPGO::so3Log(R);
  ASSERT_LE((recovered - w).norm(), 1e-6);
}

TEST(GeodesicSE3, LogProjectsNoisyRotationMatrix) {
  Eigen::Vector3d w;
  w << 0.15, -0.25, 0.08;
  Matrix R = DPGO::so3Exp(w);
  Matrix noisy = R;
  noisy(0, 1) += 1e-4;
  noisy(2, 0) -= 2e-4;
  Eigen::Vector3d recovered = DPGO::so3Log(noisy);
  ASSERT_LE((recovered - w).norm(), 5e-4);
}

TEST(GeodesicSE3, RelativeResidualIsZeroAtMeasurement) {
  Matrix Ri = Matrix::Identity(3, 3);
  Matrix ti = Matrix::Zero(3, 1);
  Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << 0.1, 0.2, -0.1).finished());
  Matrix tij = (Eigen::Vector3d() << 1.0, 2.0, 3.0).finished();
  Matrix Rj = Ri * Rij;
  Matrix tj = ti + Ri * tij;

  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3Residual(Ri, ti, Rij, tij, Rj, tj);

  ASSERT_LE(residual.rotation.norm(), 1e-10);
  ASSERT_LE(residual.translation.norm(), 1e-10);
}

TEST(GeodesicSE3, RelativeSE3LogResidualIsZeroAtMeasurement) {
  Matrix Ri =
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, -0.1, 0.05).finished());
  Matrix ti = (Eigen::Vector3d() << 0.3, -0.2, 0.4).finished();
  Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << 0.1, 0.2, -0.1).finished());
  Matrix tij = (Eigen::Vector3d() << 1.0, 2.0, 3.0).finished();
  Matrix Rj = Ri * Rij;
  Matrix tj = ti + Ri * tij;

  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3LogResidual(Ri, ti, Rij, tij, Rj, tj);

  ASSERT_LE(residual.rotation.norm(), 1e-10);
  ASSERT_LE(residual.translation.norm(), 1e-10);
}

TEST(GeodesicSE3, SE3LogResidualDiffersFromSeparableTranslationResidual) {
  Matrix Ri =
      DPGO::so3Exp((Eigen::Vector3d() << 0.35, -0.2, 0.15).finished());
  Matrix ti = (Eigen::Vector3d() << 0.3, -0.2, 0.4).finished();
  Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, 0.1, -0.25).finished());
  Matrix tij = (Eigen::Vector3d() << 1.0, -0.3, 0.7).finished();
  Matrix Rj =
      DPGO::so3Exp((Eigen::Vector3d() << -0.4, 0.25, 0.3).finished());
  Matrix tj = (Eigen::Vector3d() << 2.1, -0.8, 1.6).finished();

  const DPGO::GeodesicSE3Residual separable =
      DPGO::evaluateRelativeSE3Residual(Ri, ti, Rij, tij, Rj, tj);
  const DPGO::GeodesicSE3Residual lieLog =
      DPGO::evaluateRelativeSE3LogResidual(Ri, ti, Rij, tij, Rj, tj);

  ASSERT_LE((separable.rotation - lieLog.rotation).norm(), 1e-12);
  ASSERT_GT((separable.translation - Ri * Rij * lieLog.translation).norm(),
            1e-2);
}

namespace {

Eigen::Matrix<double, 6, 1> residualVector(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj) {
  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3Residual(Ri, ti, Rij, tij, Rj, tj);
  Eigen::Matrix<double, 6, 1> out;
  out.segment<3>(0) = residual.rotation;
  out.segment<3>(3) = residual.translation;
  return out;
}

Eigen::Matrix<double, 6, 1> se3LogResidualVector(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj) {
  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3LogResidual(Ri, ti, Rij, tij, Rj, tj);
  Eigen::Matrix<double, 6, 1> out;
  out.segment<3>(0) = residual.rotation;
  out.segment<3>(3) = residual.translation;
  return out;
}

}  // namespace

TEST(GeodesicSE3, RelativeLinearizationMatchesFiniteDifference) {
  const double eps = 1e-7;
  Matrix Ri = DPGO::so3Exp((Eigen::Vector3d() << 0.1, -0.2, 0.05).finished());
  Matrix ti = (Eigen::Vector3d() << 0.2, -0.4, 0.1).finished();
  Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << -0.05, 0.1, 0.2).finished());
  Matrix tij = (Eigen::Vector3d() << 1.0, -0.3, 0.7).finished();
  Matrix Rj = DPGO::so3Exp((Eigen::Vector3d() << 0.2, 0.1, -0.3).finished());
  Matrix tj = (Eigen::Vector3d() << 1.5, -0.8, 0.9).finished();

  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3Residual(Ri, ti, Rij, tij, Rj, tj);
  const DPGO::GeodesicSE3Jacobians J =
      DPGO::linearizeRelativeSE3Residual(Ri, tij, Rj, residual.rotation);
  const Eigen::Matrix<double, 6, 1> base =
      residualVector(Ri, ti, Rij, tij, Rj, tj);

  Eigen::Matrix<double, 6, 3> Jr_i_numeric;
  Eigen::Matrix<double, 6, 3> Jt_i_numeric;
  Eigen::Matrix<double, 6, 3> Jr_j_numeric;
  Eigen::Matrix<double, 6, 3> Jt_j_numeric;

  for (int c = 0; c < 3; ++c) {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta(c) = eps;

    Jr_i_numeric.col(c) =
        (residualVector(Ri * DPGO::so3Exp(delta), ti, Rij, tij, Rj, tj) -
         base) /
        eps;
    Jt_i_numeric.col(c) =
        (residualVector(Ri, ti + delta, Rij, tij, Rj, tj) - base) / eps;
    Jr_j_numeric.col(c) =
        (residualVector(Ri, ti, Rij, tij, Rj * DPGO::so3Exp(delta), tj) -
         base) /
        eps;
    Jt_j_numeric.col(c) =
        (residualVector(Ri, ti, Rij, tij, Rj, tj + delta) - base) / eps;
  }

  ASSERT_LE((Jr_i_numeric - J.Jr_i).norm(), 1e-5);
  ASSERT_LE((Jt_i_numeric - J.Jt_i).norm(), 1e-5);
  ASSERT_LE((Jr_j_numeric - J.Jr_j).norm(), 1e-5);
  ASSERT_LE((Jt_j_numeric - J.Jt_j).norm(), 1e-5);
}

TEST(GeodesicSE3, RelativeSE3LogLinearizationMatchesFiniteDifference) {
  const double eps = 1e-7;
  Matrix Ri = DPGO::so3Exp((Eigen::Vector3d() << 0.1, -0.2, 0.05).finished());
  Matrix ti = (Eigen::Vector3d() << 0.2, -0.4, 0.1).finished();
  Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << -0.05, 0.1, 0.2).finished());
  Matrix tij = (Eigen::Vector3d() << 1.0, -0.3, 0.7).finished();
  Matrix Rj = DPGO::so3Exp((Eigen::Vector3d() << 0.2, 0.1, -0.3).finished());
  Matrix tj = (Eigen::Vector3d() << 1.5, -0.8, 0.9).finished();

  const DPGO::GeodesicSE3Jacobians J =
      DPGO::linearizeRelativeSE3LogResidual(Ri, ti, Rij, tij, Rj, tj);
  const Eigen::Matrix<double, 6, 1> base =
      se3LogResidualVector(Ri, ti, Rij, tij, Rj, tj);

  Eigen::Matrix<double, 6, 3> Jr_i_numeric;
  Eigen::Matrix<double, 6, 3> Jt_i_numeric;
  Eigen::Matrix<double, 6, 3> Jr_j_numeric;
  Eigen::Matrix<double, 6, 3> Jt_j_numeric;

  for (int c = 0; c < 3; ++c) {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta(c) = eps;

    Jr_i_numeric.col(c) =
        (se3LogResidualVector(Ri * DPGO::so3Exp(delta), ti, Rij, tij, Rj,
                              tj) -
         base) /
        eps;
    Jt_i_numeric.col(c) =
        (se3LogResidualVector(Ri, ti + delta, Rij, tij, Rj, tj) - base) /
        eps;
    Jr_j_numeric.col(c) =
        (se3LogResidualVector(Ri, ti, Rij, tij, Rj * DPGO::so3Exp(delta),
                              tj) -
         base) /
        eps;
    Jt_j_numeric.col(c) =
        (se3LogResidualVector(Ri, ti, Rij, tij, Rj, tj + delta) - base) /
        eps;
  }

  ASSERT_LE((Jr_i_numeric - J.Jr_i).norm(), 2e-5);
  ASSERT_LE((Jt_i_numeric - J.Jt_i).norm(), 2e-5);
  ASSERT_LE((Jr_j_numeric - J.Jr_j).norm(), 2e-5);
  ASSERT_LE((Jt_j_numeric - J.Jt_j).norm(), 2e-5);
}

TEST(GeodesicSE3, RightTangentStepMatchesRelativeSE3Log) {
  const Matrix R = DPGO::so3Exp(Eigen::Vector3d(0.2, -0.1, 0.05));
  const Matrix t = (Matrix(3, 1) << 0.4, -0.2, 0.7).finished();
  Matrix pose = Matrix::Zero(3, 4);
  pose.leftCols(3) = R;
  pose.rightCols(1) = t;

  Eigen::Matrix<double, 6, 1> tangent;
  tangent << 0.03, -0.02, 0.015, 0.08, -0.04, 0.025;
  const double scale = 0.7;

  const Matrix stepped = DPGO::applySE3RightTangentStep(pose, tangent, scale);
  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3LogResidual(
          pose.leftCols(3), pose.rightCols(1), Matrix::Identity(3, 3),
          Matrix::Zero(3, 1), stepped.leftCols(3), stepped.rightCols(1));

  EXPECT_NEAR((residual.rotation - scale * tangent.head<3>()).norm(), 0.0,
              1e-10);
  EXPECT_NEAR((residual.translation - scale * tangent.tail<3>()).norm(), 0.0,
              1e-10);
}
