/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/GeodesicObjectProblem.h>

#include <DPGO/GeodesicSE3.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace DPGO {

namespace {

Matrix poseRotation(const Matrix &Y, size_t pose) {
  return Y.block(0, pose * 4, 3, 3);
}

Matrix poseTranslation(const Matrix &Y, size_t pose) {
  return Y.block(0, pose * 4 + 3, 3, 1);
}

void addAmbientGradientBlock(const Matrix &Y, size_t pose,
                             const Eigen::Vector3d &rotationGradient,
                             const Eigen::Vector3d &translationGradient,
                             Matrix &ambientGradient) {
  const Matrix R = poseRotation(Y, pose);
  ambientGradient.block(0, pose * 4, 3, 3) +=
      0.5 * R * skew3(rotationGradient);
  ambientGradient.block(0, pose * 4 + 3, 3, 1) += translationGradient;
}

Eigen::Vector3d tangentRotationVector(const Matrix &Y, const Matrix &eta,
                                      size_t pose) {
  const Matrix R = poseRotation(Y, pose);
  const Matrix A = R.transpose() * eta.block(0, pose * 4, 3, 3);
  return vee3(0.5 * (A - A.transpose()));
}

Eigen::Vector3d tangentTranslationVector(const Matrix &eta, size_t pose) {
  return eta.block(0, pose * 4 + 3, 3, 1);
}

void addMeasurementCostAndGradient(const GeodesicMeasurementTerm &term,
                                   const Matrix &Y, Matrix *ambientGradient,
                                   double &cost) {
  const Matrix Ri = poseRotation(Y, term.firstPose);
  const Matrix ti = poseTranslation(Y, term.firstPose);
  const Matrix Rj = poseRotation(Y, term.secondPose);
  const Matrix tj = poseTranslation(Y, term.secondPose);
  const GeodesicSE3Residual residual = evaluateRelativeSE3LogResidual(
      Ri, ti, term.relativeRotation, term.relativeTranslation, Rj, tj);

  cost += term.rotationalPrecision * residual.rotation.squaredNorm() +
          term.translationalPrecision * residual.translation.squaredNorm();
  if (ambientGradient == nullptr) {
    return;
  }

  const GeodesicSE3Jacobians J = linearizeRelativeSE3LogResidual(
      Ri, ti, term.relativeRotation, term.relativeTranslation, Rj, tj);
  Eigen::Matrix<double, 6, 1> weightedResidual;
  weightedResidual.head<3>() =
      term.rotationalPrecision * residual.rotation;
  weightedResidual.tail<3>() =
      term.translationalPrecision * residual.translation;

  const Eigen::Vector3d giRot = 2.0 * J.Jr_i.transpose() * weightedResidual;
  const Eigen::Vector3d giTrans = 2.0 * J.Jt_i.transpose() * weightedResidual;
  const Eigen::Vector3d gjRot = 2.0 * J.Jr_j.transpose() * weightedResidual;
  const Eigen::Vector3d gjTrans = 2.0 * J.Jt_j.transpose() * weightedResidual;
  addAmbientGradientBlock(Y, term.firstPose, giRot, giTrans,
                          *ambientGradient);
  addAmbientGradientBlock(Y, term.secondPose, gjRot, gjTrans,
                          *ambientGradient);
}

void addMeasurementHessianVector(const GeodesicMeasurementTerm &term,
                                 const Matrix &Y, const Matrix &eta,
                                 Matrix &ambientHessian) {
  const Matrix Ri = poseRotation(Y, term.firstPose);
  const Matrix Rj = poseRotation(Y, term.secondPose);
  const Matrix ti = poseTranslation(Y, term.firstPose);
  const Matrix tj = poseTranslation(Y, term.secondPose);
  const GeodesicSE3Jacobians J = linearizeRelativeSE3LogResidual(
      Ri, ti, term.relativeRotation, term.relativeTranslation, Rj, tj);

  const Eigen::Vector3d viRot = tangentRotationVector(Y, eta, term.firstPose);
  const Eigen::Vector3d viTrans =
      tangentTranslationVector(eta, term.firstPose);
  const Eigen::Vector3d vjRot = tangentRotationVector(Y, eta, term.secondPose);
  const Eigen::Vector3d vjTrans =
      tangentTranslationVector(eta, term.secondPose);

  Eigen::Matrix<double, 6, 1> deltaResidual =
      J.Jr_i * viRot + J.Jt_i * viTrans + J.Jr_j * vjRot +
      J.Jt_j * vjTrans;
  Eigen::Matrix<double, 6, 1> weightedDelta;
  weightedDelta.head<3>() =
      term.rotationalPrecision * deltaResidual.head<3>();
  weightedDelta.tail<3>() =
      term.translationalPrecision * deltaResidual.tail<3>();

  const Eigen::Vector3d hiRot = 2.0 * J.Jr_i.transpose() * weightedDelta;
  const Eigen::Vector3d hiTrans = 2.0 * J.Jt_i.transpose() * weightedDelta;
  const Eigen::Vector3d hjRot = 2.0 * J.Jr_j.transpose() * weightedDelta;
  const Eigen::Vector3d hjTrans = 2.0 * J.Jt_j.transpose() * weightedDelta;
  addAmbientGradientBlock(Y, term.firstPose, hiRot, hiTrans,
                          ambientHessian);
  addAmbientGradientBlock(Y, term.secondPose, hjRot, hjTrans,
                          ambientHessian);
}

void addConsensusCostAndGradient(const GeodesicConsensusTerm &term,
                                 const Matrix &Y, Matrix *ambientGradient,
                                 double &cost) {
  const Matrix localPose = Y.block(0, term.pose * 4, 3, 4);
  const Matrix localR = localPose.leftCols(3);
  const Matrix localT = localPose.rightCols(1);
  const Matrix targetR = term.targetPose.leftCols(3);
  const Matrix targetT = term.targetPose.rightCols(1);
  const GeodesicSE3Residual residual = evaluateRelativeSE3LogResidual(
      localR, localT, Matrix::Identity(3, 3), Matrix::Zero(3, 1),
      targetR, targetT);
  const double weightedBeta = term.beta * term.weight * term.weight;

  cost += weightedBeta * (residual.rotation.squaredNorm() +
                          residual.translation.squaredNorm());
  if (ambientGradient == nullptr || weightedBeta <= 0.0) {
    return;
  }

  const GeodesicSE3Jacobians J = linearizeRelativeSE3LogResidual(
      localR, localT, Matrix::Identity(3, 3), Matrix::Zero(3, 1),
      targetR, targetT);
  Eigen::Matrix<double, 6, 1> weightedResidual;
  weightedResidual.head<3>() = weightedBeta * residual.rotation;
  weightedResidual.tail<3>() = weightedBeta * residual.translation;
  const Eigen::Vector3d gRot =
      2.0 * J.Jr_i.transpose() * weightedResidual;
  const Eigen::Vector3d gTrans =
      2.0 * J.Jt_i.transpose() * weightedResidual;
  addAmbientGradientBlock(Y, term.pose, gRot, gTrans, *ambientGradient);
}

void addConsensusHessianVector(const GeodesicConsensusTerm &term,
                               const Matrix &Y, const Matrix &eta,
                               Matrix &ambientHessian) {
  const Matrix localPose = Y.block(0, term.pose * 4, 3, 4);
  const Matrix localR = localPose.leftCols(3);
  const Matrix localT = localPose.rightCols(1);
  const Matrix targetR = term.targetPose.leftCols(3);
  const Matrix targetT = term.targetPose.rightCols(1);
  const double weightedBeta = term.beta * term.weight * term.weight;
  if (weightedBeta <= 0.0) {
    return;
  }

  const GeodesicSE3Jacobians J = linearizeRelativeSE3LogResidual(
      localR, localT, Matrix::Identity(3, 3), Matrix::Zero(3, 1),
      targetR, targetT);
  const Eigen::Vector3d vRot = tangentRotationVector(Y, eta, term.pose);
  const Eigen::Vector3d vTrans = tangentTranslationVector(eta, term.pose);
  Eigen::Matrix<double, 6, 1> deltaResidual =
      J.Jr_i * vRot + J.Jt_i * vTrans;
  Eigen::Matrix<double, 6, 1> weightedDelta;
  weightedDelta.head<3>() = weightedBeta * deltaResidual.head<3>();
  weightedDelta.tail<3>() = weightedBeta * deltaResidual.tail<3>();
  const Eigen::Vector3d hRot =
      2.0 * J.Jr_i.transpose() * weightedDelta;
  const Eigen::Vector3d hTrans =
      2.0 * J.Jt_i.transpose() * weightedDelta;
  addAmbientGradientBlock(Y, term.pose, hRot, hTrans, ambientHessian);
}

}  // namespace

GeodesicObjectProblem::GeodesicObjectProblem(size_t numPoses)
    : n(numPoses), M(new LiftedSEManifold(r, d, n)) {
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());
}

GeodesicObjectProblem::~GeodesicObjectProblem() { delete M; }

std::vector<GeodesicConsensusTerm> makeGeodesicProximalTerms(
    const Matrix &referenceState, double beta) {
  std::vector<GeodesicConsensusTerm> terms;
  if (beta <= 0.0 || referenceState.rows() != 3 ||
      referenceState.cols() % 4 != 0) {
    return terms;
  }

  const size_t numPoses = static_cast<size_t>(referenceState.cols()) / 4;
  terms.reserve(numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    GeodesicConsensusTerm term;
    term.pose = pose;
    term.targetPose = referenceState.block(0, pose * 4, 3, 4);
    term.beta = beta;
    term.weight = 1.0;
    terms.push_back(std::move(term));
  }
  return terms;
}

Matrix limitGeodesicStateStep(const Matrix &previousState,
                              const Matrix &trialState,
                              double maxBlockStep,
                              double *appliedScale,
                              double *maxBlockDelta) {
  if (appliedScale != nullptr) {
    *appliedScale = 1.0;
  }
  if (maxBlockDelta != nullptr) {
    *maxBlockDelta = 0.0;
  }
  if (maxBlockStep <= 0.0 || previousState.rows() != 3 ||
      trialState.rows() != previousState.rows() ||
      trialState.cols() != previousState.cols() ||
      previousState.cols() % 4 != 0) {
    return trialState;
  }

  const size_t numPoses = static_cast<size_t>(previousState.cols()) / 4;
  double maxDelta = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    const Matrix previousPose = previousState.block(0, pose * 4, 3, 4);
    const Matrix trialPose = trialState.block(0, pose * 4, 3, 4);
    const Eigen::Vector3d rotationDelta =
        so3Log(previousPose.leftCols(3).transpose() *
               trialPose.leftCols(3));
    const Eigen::Vector3d translationDelta =
        trialPose.rightCols(1) - previousPose.rightCols(1);
    maxDelta = std::max(
        maxDelta,
        std::sqrt(rotationDelta.squaredNorm() +
                  translationDelta.squaredNorm()));
  }
  if (maxBlockDelta != nullptr) {
    *maxBlockDelta = maxDelta;
  }
  if (maxDelta <= maxBlockStep || maxDelta <= 0.0) {
    return trialState;
  }

  const double scale = maxBlockStep / maxDelta;
  if (appliedScale != nullptr) {
    *appliedScale = scale;
  }
  Matrix limited = previousState;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    const size_t colOffset = pose * 4;
    const Matrix previousPose = previousState.block(0, colOffset, 3, 4);
    const Matrix trialPose = trialState.block(0, colOffset, 3, 4);
    const Eigen::Vector3d rotationDelta =
        so3Log(previousPose.leftCols(3).transpose() *
               trialPose.leftCols(3));
    limited.block(0, colOffset, 3, 3) =
        previousPose.leftCols(3) * so3Exp(scale * rotationDelta);
    limited.block(0, colOffset + 3, 3, 1) =
        (1.0 - scale) * previousPose.rightCols(1) +
        scale * trialPose.rightCols(1);
  }
  return limited;
}

void GeodesicObjectProblem::setMeasurements(
    const std::vector<GeodesicMeasurementTerm> &terms) {
  measurements = terms;
}

void GeodesicObjectProblem::setConsensusTerms(
    const std::vector<GeodesicConsensusTerm> &terms) {
  consensusTerms = terms;
}

void GeodesicObjectProblem::clearConsensusTerms() { consensusTerms.clear(); }

double GeodesicObjectProblem::f(const Matrix &Y) const {
  assert((size_t)Y.rows() == r);
  assert((size_t)Y.cols() == (d + 1) * n);
  double cost = 0.0;
  for (const GeodesicMeasurementTerm &term : measurements) {
    addMeasurementCostAndGradient(term, Y, nullptr, cost);
  }
  for (const GeodesicConsensusTerm &term : consensusTerms) {
    addConsensusCostAndGradient(term, Y, nullptr, cost);
  }
  return cost;
}

double GeodesicObjectProblem::f(ROPTLIB::Variable *x) const {
  return f(readElement(x));
}

void GeodesicObjectProblem::EucGrad(ROPTLIB::Variable *x,
                                    ROPTLIB::Vector *g) const {
  const Matrix Y = readElement(x);
  Matrix gradient = Matrix::Zero(r, (d + 1) * n);
  double unusedCost = 0.0;
  for (const GeodesicMeasurementTerm &term : measurements) {
    addMeasurementCostAndGradient(term, Y, &gradient, unusedCost);
  }
  for (const GeodesicConsensusTerm &term : consensusTerms) {
    addConsensusCostAndGradient(term, Y, &gradient, unusedCost);
  }
  Eigen::Map<Matrix>((double *)g->ObtainWriteEntireData(), r,
                     (d + 1) * n) = gradient;
}

void GeodesicObjectProblem::EucHessianEta(ROPTLIB::Variable *x,
                                          ROPTLIB::Vector *v,
                                          ROPTLIB::Vector *Hv) const {
  const Matrix Y = readElement(x);
  const Matrix eta = readElement(v);
  Matrix hessian = Matrix::Zero(r, (d + 1) * n);
  for (const GeodesicMeasurementTerm &term : measurements) {
    addMeasurementHessianVector(term, Y, eta, hessian);
  }
  for (const GeodesicConsensusTerm &term : consensusTerms) {
    addConsensusHessianVector(term, Y, eta, hessian);
  }
  Eigen::Map<Matrix>((double *)Hv->ObtainWriteEntireData(), r,
                     (d + 1) * n) = hessian;
}

Matrix GeodesicObjectProblem::RieGrad(const Matrix &Y) const {
  LiftedSEVariable var(r, d, n);
  LiftedSEVector eGrad(r, d, n);
  LiftedSEVector rGrad(r, d, n);
  var.setData(Y);
  EucGrad(var.var(), eGrad.vec());
  M->getManifold()->Projection(var.var(), eGrad.vec(), rGrad.vec());
  return rGrad.getData();
}

double GeodesicObjectProblem::RieGradNorm(const Matrix &Y) const {
  return RieGrad(Y).norm();
}

Matrix GeodesicObjectProblem::readElement(
    const ROPTLIB::Element *element) const {
  return Eigen::Map<Matrix>((double *)element->ObtainReadData(), r,
                            n * (d + 1));
}

}  // namespace DPGO
