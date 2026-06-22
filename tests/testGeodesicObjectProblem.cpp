#include <DPGO/GeodesicObjectProblem.h>
#include <DPGO/GeodesicSE3.h>
#include <DPGO/RoptOptimizer.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <gtest/gtest.h>

using DPGO::Matrix;

namespace {

Matrix makeState(const Matrix &R0, const Matrix &t0, const Matrix &R1,
                 const Matrix &t1) {
  Matrix Y = Matrix::Zero(3, 8);
  Y.block(0, 0, 3, 3) = R0;
  Y.block(0, 3, 3, 1) = t0;
  Y.block(0, 4, 3, 3) = R1;
  Y.block(0, 7, 3, 1) = t1;
  return Y;
}

Matrix tangentMatrix(const Matrix &Y, const Eigen::Vector3d &w0,
                     const Eigen::Vector3d &v0,
                     const Eigen::Vector3d &w1,
                     const Eigen::Vector3d &v1) {
  Matrix eta = Matrix::Zero(3, 8);
  eta.block(0, 0, 3, 3) = Y.block(0, 0, 3, 3) * DPGO::skew3(w0);
  eta.block(0, 3, 3, 1) = v0;
  eta.block(0, 4, 3, 3) = Y.block(0, 4, 3, 3) * DPGO::skew3(w1);
  eta.block(0, 7, 3, 1) = v1;
  return eta;
}

Matrix perturbState(const Matrix &Y, const Eigen::Vector3d &w0,
                    const Eigen::Vector3d &v0, const Eigen::Vector3d &w1,
                    const Eigen::Vector3d &v1, double eps) {
  Matrix perturbed = Y;
  perturbed.block(0, 0, 3, 3) =
      Y.block(0, 0, 3, 3) * DPGO::so3Exp(eps * w0);
  perturbed.block(0, 3, 3, 1) = Y.block(0, 3, 3, 1) + eps * v0;
  perturbed.block(0, 4, 3, 3) =
      Y.block(0, 4, 3, 3) * DPGO::so3Exp(eps * w1);
  perturbed.block(0, 7, 3, 1) = Y.block(0, 7, 3, 1) + eps * v1;
  return perturbed;
}

double ambientInner(const Matrix &A, const Matrix &B) {
  return (A.cwiseProduct(B)).sum();
}

Matrix projectAmbientToTangent(const Matrix &Y, const Matrix &V) {
  Matrix projected = V;
  for (int pose = 0; pose < Y.cols() / 4; ++pose) {
    const Matrix R = Y.block(0, pose * 4, 3, 3);
    const Matrix A = R.transpose() * V.block(0, pose * 4, 3, 3);
    projected.block(0, pose * 4, 3, 3) =
        V.block(0, pose * 4, 3, 3) - R * (0.5 * (A + A.transpose()));
  }
  return projected;
}

double stateDeltaNorm(const Matrix &A, const Matrix &B) {
  double sqNorm = 0.0;
  for (int pose = 0; pose < A.cols() / 4; ++pose) {
    const Matrix Ra = A.block(0, pose * 4, 3, 3);
    const Matrix Rb = B.block(0, pose * 4, 3, 3);
    const Matrix ta = A.block(0, pose * 4 + 3, 3, 1);
    const Matrix tb = B.block(0, pose * 4 + 3, 3, 1);
    sqNorm += DPGO::so3Log(Ra.transpose() * Rb).squaredNorm();
    sqNorm += (tb - ta).squaredNorm();
  }
  return std::sqrt(sqNorm);
}

double maxStateBlockDeltaNorm(const Matrix &A, const Matrix &B) {
  double maxNorm = 0.0;
  for (int pose = 0; pose < A.cols() / 4; ++pose) {
    const Matrix Ra = A.block(0, pose * 4, 3, 3);
    const Matrix Rb = B.block(0, pose * 4, 3, 3);
    const Matrix ta = A.block(0, pose * 4 + 3, 3, 1);
    const Matrix tb = B.block(0, pose * 4 + 3, 3, 1);
    const double blockNorm =
        std::sqrt(DPGO::so3Log(Ra.transpose() * Rb).squaredNorm() +
                  (tb - ta).squaredNorm());
    maxNorm = std::max(maxNorm, blockNorm);
  }
  return maxNorm;
}

DPGO::GeodesicMeasurementTerm makeMeasurement(const Matrix &Rij,
                                              const Matrix &tij) {
  DPGO::GeodesicMeasurementTerm term;
  term.firstPose = 0;
  term.secondPose = 1;
  term.relativeRotation = Rij;
  term.relativeTranslation = tij;
  term.rotationalPrecision = 1.7;
  term.translationalPrecision = 2.3;
  return term;
}

}  // namespace

TEST(GeodesicObjectProblem, MeasurementCostIsZeroAtConsistentState) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, -0.1, 0.05).finished());
  const Matrix t0 = (Eigen::Vector3d() << 0.3, -0.2, 0.4).finished();
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << -0.1, 0.15, 0.2).finished());
  const Matrix tij = (Eigen::Vector3d() << 1.0, -0.4, 0.7).finished();
  const Matrix R1 = R0 * Rij;
  const Matrix t1 = t0 + R0 * tij;
  const Matrix Y = makeState(R0, t0, R1, t1);

  DPGO::GeodesicObjectProblem problem(2);
  problem.setMeasurements({makeMeasurement(Rij, tij)});

  ASSERT_LE(problem.f(Y), 1e-18);
  ASSERT_LE(problem.RieGradNorm(Y), 1e-10);
}

TEST(GeodesicObjectProblem, RiemannianGradientMatchesDirectionalDerivative) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.1, -0.25, 0.05).finished());
  const Matrix t0 = (Eigen::Vector3d() << 0.2, -0.1, 0.5).finished();
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << 0.05, 0.1, -0.2).finished());
  const Matrix tij = (Eigen::Vector3d() << 0.8, -0.2, 0.6).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << -0.2, 0.15, 0.3).finished());
  const Matrix t1 = (Eigen::Vector3d() << 0.9, -0.4, 1.1).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);
  const Eigen::Vector3d w0(0.2, -0.1, 0.05);
  const Eigen::Vector3d v0(-0.3, 0.2, 0.1);
  const Eigen::Vector3d w1(-0.15, 0.07, 0.12);
  const Eigen::Vector3d v1(0.05, -0.08, 0.2);
  const Matrix eta = tangentMatrix(Y, w0, v0, w1, v1);

  DPGO::GeodesicObjectProblem problem(2);
  problem.setMeasurements({makeMeasurement(Rij, tij)});

  const double eps = 1e-7;
  const double numeric =
      (problem.f(perturbState(Y, w0, v0, w1, v1, eps)) -
       problem.f(perturbState(Y, w0, v0, w1, v1, -eps))) /
      (2.0 * eps);
  const double analytic = ambientInner(problem.RieGrad(Y), eta);
  ASSERT_LE(std::abs(numeric - analytic), 1e-5);
}

TEST(GeodesicObjectProblem, ConsensusGradientMatchesDirectionalDerivative) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.1, -0.25, 0.05).finished());
  const Matrix t0 = (Eigen::Vector3d() << 0.2, -0.1, 0.5).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << -0.2, 0.15, 0.3).finished());
  const Matrix t1 = (Eigen::Vector3d() << 0.9, -0.4, 1.1).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);

  Matrix target = Matrix::Zero(3, 4);
  target.leftCols(3) =
      DPGO::so3Exp((Eigen::Vector3d() << -0.1, 0.2, 0.05).finished());
  target.rightCols(1) = (Eigen::Vector3d() << 1.2, -0.2, 0.8).finished();

  DPGO::GeodesicConsensusTerm consensus;
  consensus.pose = 1;
  consensus.targetPose = target;
  consensus.beta = 0.7;
  consensus.weight = 1.3;

  const Eigen::Vector3d w0(0.0, 0.0, 0.0);
  const Eigen::Vector3d v0(0.0, 0.0, 0.0);
  const Eigen::Vector3d w1(-0.15, 0.07, 0.12);
  const Eigen::Vector3d v1(0.05, -0.08, 0.2);
  const Matrix eta = tangentMatrix(Y, w0, v0, w1, v1);

  DPGO::GeodesicObjectProblem problem(2);
  problem.setConsensusTerms({consensus});

  const double eps = 1e-7;
  const double numeric =
      (problem.f(perturbState(Y, w0, v0, w1, v1, eps)) -
       problem.f(perturbState(Y, w0, v0, w1, v1, -eps))) /
      (2.0 * eps);
  const double analytic = ambientInner(problem.RieGrad(Y), eta);
  ASSERT_LE(std::abs(numeric - analytic), 1e-5);
}

TEST(GeodesicObjectProblem, ConsensusCostUsesFullSE3LogResidual) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.4, -0.15, 0.25).finished());
  const Matrix t0 = (Eigen::Vector3d() << -0.3, 0.2, 0.7).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << -0.2, 0.35, 0.1).finished());
  const Matrix t1 = (Eigen::Vector3d() << 1.4, -0.5, 0.9).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);

  Matrix target = Matrix::Zero(3, 4);
  target.leftCols(3) =
      DPGO::so3Exp((Eigen::Vector3d() << 0.3, -0.4, 0.15).finished());
  target.rightCols(1) = (Eigen::Vector3d() << 0.8, 1.1, -0.6).finished();

  DPGO::GeodesicConsensusTerm consensus;
  consensus.pose = 1;
  consensus.targetPose = target;
  consensus.beta = 0.7;
  consensus.weight = 1.3;

  DPGO::GeodesicObjectProblem problem(2);
  problem.setConsensusTerms({consensus});

  const DPGO::GeodesicSE3Residual residual =
      DPGO::evaluateRelativeSE3LogResidual(
          R1, t1, Matrix::Identity(3, 3), Matrix::Zero(3, 1),
          target.leftCols(3), target.rightCols(1));
  const double expected =
      consensus.beta * consensus.weight * consensus.weight *
      (residual.rotation.squaredNorm() +
       residual.translation.squaredNorm());
  ASSERT_NEAR(problem.f(Y), expected, 1e-12);
}

TEST(GeodesicObjectProblem, GaussNewtonHessianVectorMatchesZeroResidualCurvature) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, -0.1, 0.05).finished());
  const Matrix t0 = (Eigen::Vector3d() << 0.3, -0.2, 0.4).finished();
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << -0.1, 0.15, 0.2).finished());
  const Matrix tij = (Eigen::Vector3d() << 1.0, -0.4, 0.7).finished();
  const Matrix R1 = R0 * Rij;
  const Matrix t1 = t0 + R0 * tij;
  const Matrix Y = makeState(R0, t0, R1, t1);
  const Eigen::Vector3d w0(0.1, -0.07, 0.03);
  const Eigen::Vector3d v0(0.02, -0.05, 0.08);
  const Eigen::Vector3d w1(-0.04, 0.09, -0.02);
  const Eigen::Vector3d v1(0.06, 0.01, -0.03);
  const Matrix eta = tangentMatrix(Y, w0, v0, w1, v1);

  DPGO::GeodesicObjectProblem problem(2);
  problem.setMeasurements({makeMeasurement(Rij, tij)});

  DPGO::LiftedSEVariable var(3, 3, 2);
  DPGO::LiftedSEVector etaVec(3, 3, 2);
  DPGO::LiftedSEVector hVec(3, 3, 2);
  var.setData(Y);
  etaVec.setData(eta);
  problem.EucHessianEta(var.var(), etaVec.vec(), hVec.vec());

  const double eps = 1e-5;
  const double numeric =
      (problem.f(perturbState(Y, w0, v0, w1, v1, eps)) +
       problem.f(perturbState(Y, w0, v0, w1, v1, -eps)) -
       2.0 * problem.f(Y)) /
      (eps * eps);
  const double analytic = ambientInner(hVec.getData(), eta);
  ASSERT_LE(std::abs(numeric - analytic), 1e-4);
}

TEST(GeodesicObjectProblem, RoptHessianVectorUsesProjectedGaussNewtonModel) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.15, -0.18, 0.08).finished());
  const Matrix t0 = (Eigen::Vector3d() << -0.2, 0.1, 0.3).finished();
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << -0.25, 0.1, 0.22).finished());
  const Matrix tij = (Eigen::Vector3d() << 1.2, -0.7, 0.4).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.4, -0.15, 0.35).finished());
  const Matrix t1 = (Eigen::Vector3d() << 1.4, -0.8, 1.1).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);
  const Matrix eta =
      tangentMatrix(Y, Eigen::Vector3d(0.12, -0.04, 0.09),
                    Eigen::Vector3d(-0.08, 0.05, 0.13),
                    Eigen::Vector3d(-0.06, 0.11, -0.03),
                    Eigen::Vector3d(0.10, -0.04, 0.02));

  DPGO::GeodesicObjectProblem problem(2);
  problem.setMeasurements({makeMeasurement(Rij, tij)});

  DPGO::LiftedSEVariable var(3, 3, 2);
  DPGO::LiftedSEVector etaVec(3, 3, 2);
  DPGO::LiftedSEVector gnVec(3, 3, 2);
  DPGO::LiftedSEVector roptHvVec(3, 3, 2);
  var.setData(Y);
  etaVec.setData(eta);
  problem.EucHessianEta(var.var(), etaVec.vec(), gnVec.vec());
  static_cast<ROPTLIB::Problem &>(problem).RieGrad(var.var(), roptHvVec.vec());
  problem.HessianEta(var.var(), etaVec.vec(), roptHvVec.vec());

  const Matrix expected = projectAmbientToTangent(Y, gnVec.getData());
  ASSERT_LE((roptHvVec.getData() - expected).norm(), 1e-10);
}

TEST(GeodesicObjectProblem, RoptHessianVectorIsApproximateAtNonzeroResidual) {
  const Matrix R0 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.15, -0.18, 0.08).finished());
  const Matrix t0 = (Eigen::Vector3d() << -0.2, 0.1, 0.3).finished();
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << -0.25, 0.1, 0.22).finished());
  const Matrix tij = (Eigen::Vector3d() << 1.2, -0.7, 0.4).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.4, -0.15, 0.35).finished());
  const Matrix t1 = (Eigen::Vector3d() << 1.4, -0.8, 1.1).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);
  const Eigen::Vector3d w0(0.12, -0.04, 0.09);
  const Eigen::Vector3d v0(-0.08, 0.05, 0.13);
  const Eigen::Vector3d w1(-0.06, 0.11, -0.03);
  const Eigen::Vector3d v1(0.10, -0.04, 0.02);
  const Matrix eta = tangentMatrix(Y, w0, v0, w1, v1);

  DPGO::GeodesicObjectProblem problem(2);
  problem.setMeasurements({makeMeasurement(Rij, tij)});

  DPGO::LiftedSEVariable var(3, 3, 2);
  DPGO::LiftedSEVector etaVec(3, 3, 2);
  DPGO::LiftedSEVector gradVec(3, 3, 2);
  DPGO::LiftedSEVector hVec(3, 3, 2);
  var.setData(Y);
  etaVec.setData(eta);
  static_cast<ROPTLIB::Problem &>(problem).RieGrad(var.var(), gradVec.vec());
  problem.HessianEta(var.var(), etaVec.vec(), hVec.vec());

  const double eps = 1e-5;
  const double numeric =
      (problem.f(perturbState(Y, w0, v0, w1, v1, eps)) +
       problem.f(perturbState(Y, w0, v0, w1, v1, -eps)) -
       2.0 * problem.f(Y)) /
      (eps * eps);
  const double roptHessian = ambientInner(hVec.getData(), eta);

  ASSERT_GT(std::abs(numeric - roptHessian), 1e-2);
}

TEST(GeodesicObjectProblem, RoptTrustRegionDecreasesNonzeroResidualCost) {
  const Matrix R0 = Matrix::Identity(3, 3);
  const Matrix t0 = Matrix::Zero(3, 1);
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << 0.05, -0.08, 0.03).finished());
  const Matrix tij = (Eigen::Vector3d() << 0.5, -0.2, 0.4).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << -0.2, 0.1, 0.15).finished());
  const Matrix t1 = (Eigen::Vector3d() << 1.2, -0.7, 0.9).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);

  DPGO::GeodesicObjectProblem problem(2);
  problem.setMeasurements({makeMeasurement(Rij, tij)});
  DPGO::RoptOptimizer optimizer(&problem, 2, 3, 3);
  optimizer.setAlgorithm(DPGO::ROPTALG::RTR);
  optimizer.setTrustRegionIterations(5);
  optimizer.setTrustRegionTolerance(1e-8);
  optimizer.setTrustRegionInitialRadius(1.0);
  optimizer.setTrustRegionMaxInnerIterations(30);

  const Matrix Yopt = optimizer.optimize(Y);
  const DPGO::ROPTResult result = optimizer.getOptResult();
  ASSERT_TRUE(result.success);
  ASSERT_LT(problem.f(Yopt), problem.f(Y));
  ASSERT_LT(result.gradNormOpt, result.gradNormInit);
}

TEST(GeodesicObjectProblem, ProximalTermsLimitRtrStepAwayFromReference) {
  const Matrix R0 = Matrix::Identity(3, 3);
  const Matrix t0 = Matrix::Zero(3, 1);
  const Matrix Rij =
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, -0.1, 0.15).finished());
  const Matrix tij = (Eigen::Vector3d() << 0.8, -0.4, 0.5).finished();
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << -0.3, 0.2, -0.1).finished());
  const Matrix t1 = (Eigen::Vector3d() << 1.4, -0.7, 0.8).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);

  DPGO::GeodesicObjectProblem freeProblem(2);
  freeProblem.setMeasurements({makeMeasurement(Rij, tij)});
  DPGO::RoptOptimizer freeOptimizer(&freeProblem, 2, 3, 3);
  freeOptimizer.setTrustRegionIterations(3);
  freeOptimizer.setTrustRegionTolerance(1e-8);
  freeOptimizer.setTrustRegionInitialRadius(1.0);
  freeOptimizer.setTrustRegionMaxInnerIterations(30);
  const Matrix freeY = freeOptimizer.optimize(Y);

  DPGO::GeodesicObjectProblem proxProblem(2);
  proxProblem.setMeasurements({makeMeasurement(Rij, tij)});
  proxProblem.setConsensusTerms(DPGO::makeGeodesicProximalTerms(Y, 1e3));
  DPGO::RoptOptimizer proxOptimizer(&proxProblem, 2, 3, 3);
  proxOptimizer.setTrustRegionIterations(3);
  proxOptimizer.setTrustRegionTolerance(1e-8);
  proxOptimizer.setTrustRegionInitialRadius(1.0);
  proxOptimizer.setTrustRegionMaxInnerIterations(30);
  const Matrix proxY = proxOptimizer.optimize(Y);

  ASSERT_LT(stateDeltaNorm(Y, proxY), 0.25 * stateDeltaNorm(Y, freeY));
  ASSERT_LT(proxProblem.f(proxY), proxProblem.f(Y));
}

TEST(GeodesicObjectProblem, LimitGeodesicStateStepCapsLargestBlockMotion) {
  const Matrix R0 = Matrix::Identity(3, 3);
  const Matrix t0 = Matrix::Zero(3, 1);
  const Matrix R1 =
      DPGO::so3Exp((Eigen::Vector3d() << 0.1, -0.2, 0.05).finished());
  const Matrix t1 = (Eigen::Vector3d() << 0.3, -0.1, 0.2).finished();
  const Matrix Y = makeState(R0, t0, R1, t1);
  const Matrix trial =
      perturbState(Y, Eigen::Vector3d(0.6, 0.0, 0.0),
                   Eigen::Vector3d(0.8, 0.0, 0.0),
                   Eigen::Vector3d(0.1, 0.0, 0.0),
                   Eigen::Vector3d(0.1, 0.0, 0.0), 1.0);

  double scale = 1.0;
  double maxBlockDelta = 0.0;
  const Matrix limited =
      DPGO::limitGeodesicStateStep(Y, trial, 0.25, &scale, &maxBlockDelta);

  ASSERT_NEAR(maxBlockDelta, 1.0, 1e-9);
  ASSERT_NEAR(scale, 0.25, 1e-9);
  ASSERT_LE(maxStateBlockDeltaNorm(Y, limited), 0.25 + 1e-9);
  ASSERT_LT(stateDeltaNorm(Y, limited), stateDeltaNorm(Y, trial));

  const Matrix unchanged =
      DPGO::limitGeodesicStateStep(Y, trial, 2.0, &scale, &maxBlockDelta);
  ASSERT_NEAR(scale, 1.0, 1e-12);
  ASSERT_NEAR((unchanged - trial).norm(), 0.0, 1e-12);
}
