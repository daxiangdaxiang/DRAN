/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/ObjectConsensus.h>

#include <cmath>

namespace DPGO {

ObjectConsensus::ObjectConsensus(unsigned d, unsigned r) : mD(d), mR(r) {}

void ObjectConsensus::clear() {
  mStates.clear();
  mLambda.clear();
}

bool ObjectConsensus::hasDual(unsigned neighborID, unsigned objectID) const {
  return findDual(neighborID, objectID) != nullptr;
}

Matrix ObjectConsensus::getDual(unsigned neighborID, unsigned objectID) const {
  const Matrix *lambda = findDual(neighborID, objectID);
  if (lambda == nullptr) {
    if (mR == 0 || mD == 0) {
      return Matrix();
    }
    return Matrix::Zero(mR, mD + 1);
  }
  return *lambda;
}

void ObjectConsensus::setDual(unsigned neighborID, unsigned objectID,
                              const Matrix &lambda) {
  if (mR == 0 && mD == 0) {
    mR = static_cast<unsigned>(lambda.rows());
    mD = static_cast<unsigned>(lambda.cols() - 1);
  }
  if (lambda.rows() != static_cast<int>(mR) ||
      lambda.cols() != static_cast<int>(mD + 1)) {
    throw std::invalid_argument(
        "ObjectConsensus::setDual received a lambda with incompatible size");
  }
  mLambda[neighborID][objectID] = lambda;
}

void ObjectConsensus::updateDual(unsigned neighborID, unsigned objectID,
                                 const Matrix &localPose,
                                 const Matrix &neighborPose, double beta,
                                 double eta, double weight) {
  if (beta <= 0.0) {
    throw std::invalid_argument(
        "ObjectConsensus::updateDual requires beta > 0");
  }
  if (eta < 0.0) {
    throw std::invalid_argument("ObjectConsensus::updateDual requires eta >= 0");
  }

  ensureDimensions(localPose, neighborPose);

  PairState &state = stateRef(neighborID, objectID);
  state.localPose = localPose;
  state.neighborPose = neighborPose;
  state.weight = weight;
  state.hasLocal = true;
  state.hasNeighbor = true;

  Matrix &lambda = dualRef(neighborID, objectID);
  const Matrix residual = localPose - neighborPose;
  lambda += eta * beta * weight * residual;
}

double ObjectConsensus::evaluateConsensusCost(double beta,
                                              double &primalResidualNorm) const {
  if (beta <= 0.0) {
    throw std::invalid_argument(
        "ObjectConsensus::evaluateConsensusCost requires beta > 0");
  }

  double cost = 0.0;
  double primalResidualSquaredNorm = 0.0;
  for (const auto &neighborEntry : mStates) {
    for (const auto &objectEntry : neighborEntry.second) {
      const PairState &state = objectEntry.second;
      if (!state.hasLocal || !state.hasNeighbor) {
        continue;
      }
      const Matrix residual = state.localPose - state.neighborPose;
      const Matrix weightedResidual = state.weight * residual;
      const Matrix lambda =
          getDual(neighborEntry.first, objectEntry.first);
      const Matrix shifted = weightedResidual + lambda / beta;
      cost += 0.5 * beta * shifted.squaredNorm();
      primalResidualSquaredNorm += weightedResidual.squaredNorm();
    }
  }

  primalResidualNorm = std::sqrt(primalResidualSquaredNorm);
  return cost;
}

double ObjectConsensus::evaluateConsensusCost(
    unsigned neighborID, unsigned objectID, const Matrix &localPose,
    const Matrix &neighborPose, double beta, double weight,
    double &primalResidualNorm) const {
  if (beta <= 0.0) {
    throw std::invalid_argument(
        "ObjectConsensus::evaluateConsensusCost requires beta > 0");
  }
  if (localPose.rows() != neighborPose.rows() ||
      localPose.cols() != neighborPose.cols()) {
    throw std::invalid_argument(
        "ObjectConsensus requires local and neighbor pose blocks to have the "
        "same size");
  }

  const Matrix residual = localPose - neighborPose;
  const Matrix weightedResidual = weight * residual;
  const Matrix *storedLambda = findDual(neighborID, objectID);
  const Matrix lambda = storedLambda == nullptr
                            ? Matrix::Zero(localPose.rows(), localPose.cols())
                            : *storedLambda;
  const Matrix shifted = weightedResidual + lambda / beta;

  primalResidualNorm = weightedResidual.norm();
  return 0.5 * beta * shifted.squaredNorm();
}

bool ObjectConsensus::getPairState(unsigned neighborID, unsigned objectID,
                                   PairState &state) const {
  const PairState *stored = findState(neighborID, objectID);
  if (stored == nullptr) {
    return false;
  }
  state = *stored;
  return true;
}

void ObjectConsensus::ensureDimensions(const Matrix &localPose,
                                       const Matrix &neighborPose) {
  if (localPose.rows() != neighborPose.rows() ||
      localPose.cols() != neighborPose.cols()) {
    throw std::invalid_argument(
        "ObjectConsensus requires local and neighbor pose blocks to have the "
        "same size");
  }
  if (localPose.cols() < 2) {
    throw std::invalid_argument(
        "ObjectConsensus requires lifted pose blocks with at least one state "
        "column and one translation column");
  }

  if (mR == 0 && mD == 0) {
    mR = static_cast<unsigned>(localPose.rows());
    mD = static_cast<unsigned>(localPose.cols() - 1);
    return;
  }

  if (localPose.rows() != static_cast<int>(mR) ||
      localPose.cols() != static_cast<int>(mD + 1)) {
    throw std::invalid_argument(
        "ObjectConsensus received a pose block with incompatible size");
  }
}

Matrix &ObjectConsensus::dualRef(unsigned neighborID, unsigned objectID) {
  Matrix &lambda = mLambda[neighborID][objectID];
  if (lambda.size() == 0) {
    if (mR == 0 || mD == 0) {
      throw std::runtime_error(
          "ObjectConsensus dual state is not initialized yet");
    }
    lambda = Matrix::Zero(mR, mD + 1);
  }
  return lambda;
}

const Matrix *ObjectConsensus::findDual(unsigned neighborID,
                                        unsigned objectID) const {
  const auto neighborIt = mLambda.find(neighborID);
  if (neighborIt == mLambda.end()) {
    return nullptr;
  }
  const auto objectIt = neighborIt->second.find(objectID);
  if (objectIt == neighborIt->second.end()) {
    return nullptr;
  }
  return &objectIt->second;
}

ObjectConsensus::PairState &ObjectConsensus::stateRef(unsigned neighborID,
                                                      unsigned objectID) {
  return mStates[neighborID][objectID];
}

const ObjectConsensus::PairState *ObjectConsensus::findState(
    unsigned neighborID, unsigned objectID) const {
  const auto neighborIt = mStates.find(neighborID);
  if (neighborIt == mStates.end()) {
    return nullptr;
  }
  const auto objectIt = neighborIt->second.find(objectID);
  if (objectIt == neighborIt->second.end()) {
    return nullptr;
  }
  return &objectIt->second;
}

}  // namespace DPGO
