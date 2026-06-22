#include <DPGO/DCCI_linear_operators.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/PCGSolver.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

Matrix rotation2(double theta) {
  Matrix R(2, 2);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  R << c, -s, s, c;
  return R;
}

Matrix rotation3(double roll, double pitch, double yaw) {
  const Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
  return (Rz * Ry * Rx).toRotationMatrix();
}

RelativeSEMeasurement makeMeasurement(size_t i, size_t j, const Matrix &Ri,
                                      const Vector &ti, const Matrix &Rj,
                                      const Vector &tj, double kappa,
                                      double tau) {
  return RelativeSEMeasurement(0, 0, i, j, Ri.transpose() * Rj,
                               Ri.transpose() * (tj - ti), kappa, tau);
}

std::vector<RelativeSEMeasurement> makeSynthetic2DGraph(size_t *numPoses) {
  *numPoses = 4;
  const std::vector<Matrix> rotations = {rotation2(0.0), rotation2(0.25),
                                         rotation2(0.55), rotation2(0.80)};
  std::vector<Vector> translations(*numPoses, Vector::Zero(2));
  translations[0] << 0.0, 0.0;
  translations[1] << 1.0, 0.2;
  translations[2] << 2.1, 0.7;
  translations[3] << 3.0, 1.1;

  std::vector<RelativeSEMeasurement> measurements;
  measurements.emplace_back(makeMeasurement(0, 1, rotations[0],
                                            translations[0], rotations[1],
                                            translations[1], 3.0, 2.0));
  measurements.emplace_back(makeMeasurement(1, 2, rotations[1],
                                            translations[1], rotations[2],
                                            translations[2], 2.0, 4.0));
  measurements.emplace_back(makeMeasurement(2, 3, rotations[2],
                                            translations[2], rotations[3],
                                            translations[3], 5.0, 3.0));
  measurements.emplace_back(makeMeasurement(0, 2, rotations[0],
                                            translations[0], rotations[2],
                                            translations[2], 4.0, 2.5));
  measurements.emplace_back(makeMeasurement(1, 3, rotations[1],
                                            translations[1], rotations[3],
                                            translations[3], 2.5, 3.5));
  measurements.emplace_back(makeMeasurement(3, 0, rotations[3],
                                            translations[3], rotations[0],
                                            translations[0], 1.7, 2.2));
  return measurements;
}

std::vector<RelativeSEMeasurement> makeSynthetic3DGraph(size_t *numPoses) {
  *numPoses = 4;
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
      rotation3(0.18, 0.08, 0.45), rotation3(0.22, -0.12, 0.70)};
  std::vector<Vector> translations(*numPoses, Vector::Zero(3));
  translations[0] << 0.0, 0.0, 0.0;
  translations[1] << 1.0, 0.2, 0.1;
  translations[2] << 1.8, 0.9, 0.4;
  translations[3] << 2.7, 1.4, 0.8;

  std::vector<RelativeSEMeasurement> measurements;
  measurements.emplace_back(makeMeasurement(0, 1, rotations[0],
                                            translations[0], rotations[1],
                                            translations[1], 3.0, 2.0));
  measurements.emplace_back(makeMeasurement(1, 2, rotations[1],
                                            translations[1], rotations[2],
                                            translations[2], 2.0, 4.0));
  measurements.emplace_back(makeMeasurement(2, 3, rotations[2],
                                            translations[2], rotations[3],
                                            translations[3], 5.0, 3.0));
  measurements.emplace_back(makeMeasurement(0, 2, rotations[0],
                                            translations[0], rotations[2],
                                            translations[2], 4.0, 2.5));
  measurements.emplace_back(makeMeasurement(1, 3, rotations[1],
                                            translations[1], rotations[3],
                                            translations[3], 2.5, 3.5));
  measurements.emplace_back(makeMeasurement(3, 0, rotations[3],
                                            translations[3], rotations[0],
                                            translations[0], 1.7, 2.2));
  return measurements;
}

Matrix extractRotations(const Matrix &T, size_t d, size_t numPoses) {
  Matrix rotations(d, d * numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    rotations.block(0, pose * d, d, d) =
        T.block(0, pose * (d + 1), d, d);
  }
  return rotations;
}

double relativeError(const Vector &actual, const Vector &expected) {
  return (actual - expected).norm() / std::max(1.0, expected.norm());
}

Vector applyJacobi(const Vector &diagonal, const Vector &r) {
  Vector z = r;
  for (int i = 0; i < z.size(); ++i) {
    z(i) = std::abs(diagonal(i)) > 1e-14 ? r(i) / diagonal(i) : r(i);
  }
  return z;
}

PCGParams tightPCGParams() {
  PCGParams params;
  params.max_iters = 500;
  params.rel_tol = 1e-14;
  params.abs_tol = 1e-14;
  return params;
}

Vector referenceRotationSolve(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements) {
  SparseMatrix B1, B2, B3;
  constructBMatrices(measurements, B1, B2, B3);
  const size_t d2 = d * d;
  SparseMatrix B3red = B3.rightCols((numPoses - 1) * d2);
  B3red.makeCompressed();
  Matrix I = Matrix::Identity(d, d);
  Eigen::Map<Vector> IdVec(I.data(), d2);
  const Vector cR = B3.leftCols(d2) * IdVec;
  Eigen::SPQR<SparseMatrix> qr(B3red);
  return -qr.solve(cR);
}

Vector referenceTranslationSolve(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const Matrix &projectedRotations) {
  SparseMatrix B1, B2, B3;
  constructBMatrices(measurements, B1, B2, B3);
  SparseMatrix B1red = B1.rightCols((numPoses - 1) * d);
  B1red.makeCompressed();
  Eigen::Map<const Vector> rvec(projectedRotations.data(), d * d * numPoses);
  const Vector rhs = B2 * rvec;
  Eigen::SPQR<SparseMatrix> qr(B1red);
  return -qr.solve(rhs);
}

void expectPCGSolvesRotationSystem(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements) {
  const ChordalRotationLinearOperator op(d, numPoses, measurements);
  const Vector diagonal = op.buildJacobiPreconditionerDiagonal();
  const PCGResult pcg = solvePCG(
      op.buildRHS(), [&](const Vector &x) { return op.applyH(x); },
      [&](const Vector &r) { return applyJacobi(diagonal, r); },
      [](const Vector &a, const Vector &b) { return a.dot(b); },
      tightPCGParams());
  const Vector reference = referenceRotationSolve(d, numPoses, measurements);
  const double err = relativeError(pcg.x, reference);
  std::cout << "DCCI PCG rotation d=" << d << " iters=" << pcg.iters
            << " residual=" << pcg.final_residual_norm
            << " rel_error=" << err << std::endl;
  EXPECT_TRUE(pcg.converged);
  EXPECT_LT(err, 1e-10);
}

void expectPCGSolvesTranslationSystem(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements) {
  const Matrix T = chordalInitialization(d, numPoses, measurements);
  const Matrix projectedRotations = extractRotations(T, d, numPoses);
  const ChordalTranslationLinearOperator op(d, numPoses, measurements,
                                            projectedRotations);
  const Vector diagonal = op.buildJacobiPreconditionerDiagonal();
  const PCGResult pcg = solvePCG(
      op.buildRHS(), [&](const Vector &x) { return op.applyH(x); },
      [&](const Vector &r) { return applyJacobi(diagonal, r); },
      [](const Vector &a, const Vector &b) { return a.dot(b); },
      tightPCGParams());
  const Vector reference =
      referenceTranslationSolve(d, numPoses, measurements, projectedRotations);
  const double err = relativeError(pcg.x, reference);
  std::cout << "DCCI PCG translation d=" << d << " iters=" << pcg.iters
            << " residual=" << pcg.final_residual_norm
            << " rel_error=" << err << std::endl;
  EXPECT_TRUE(pcg.converged);
  EXPECT_LT(err, 1e-10);
}

}  // namespace

TEST(testDPGO, DCCIPCGSolvesSyntheticSPDSystem) {
  Matrix A(4, 4);
  A << 4.0, 1.0, 0.5, 0.0, 1.0, 3.5, 0.2, 0.1, 0.5, 0.2, 2.7, 0.3, 0.0,
      0.1, 0.3, 2.2;
  const Vector b = (Vector(4) << 1.0, -2.0, 0.5, 3.0).finished();
  const Vector diagonal = A.diagonal();
  unsigned dotCalls = 0;
  PCGParams params = tightPCGParams();
  const PCGResult pcg = solvePCG(
      b, [&](const Vector &x) { return A * x; },
      [&](const Vector &r) { return applyJacobi(diagonal, r); },
      [&](const Vector &a, const Vector &c) {
        ++dotCalls;
        return a.dot(c);
      },
      params);
  const Vector reference = A.ldlt().solve(b);
  const double err = relativeError(pcg.x, reference);
  std::cout << "DCCI PCG synthetic iters=" << pcg.iters
            << " residual=" << pcg.final_residual_norm
            << " rel_error=" << err << " dot_calls=" << dotCalls
            << std::endl;
  EXPECT_TRUE(pcg.converged);
  EXPECT_GT(dotCalls, 0u);
  EXPECT_LT(err, 1e-12);
}

TEST(testDPGO, DCCIPCGMatchesSPQRRotationSolve2D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic2DGraph(&numPoses);
  expectPCGSolvesRotationSystem(2, numPoses, measurements);
}

TEST(testDPGO, DCCIPCGMatchesSPQRRotationSolve3D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  expectPCGSolvesRotationSystem(3, numPoses, measurements);
}

TEST(testDPGO, DCCIPCGMatchesSPQRTranslationSolve2D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic2DGraph(&numPoses);
  expectPCGSolvesTranslationSystem(2, numPoses, measurements);
}

TEST(testDPGO, DCCIPCGMatchesSPQRTranslationSolve3D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  expectPCGSolvesTranslationSystem(3, numPoses, measurements);
}
