/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef OBJECTCONSENSUS_H
#define OBJECTCONSENSUS_H

#include <DPGO/DPGO_types.h>

#include <map>
#include <stdexcept>

namespace DPGO {

/**
 * @brief Independent helper for object consensus dual state.
 *
 * The helper stores lambda[neighborID][objectID] and the latest local / neighbor
 * pose blocks for each pair. Pose blocks are expected in lifted SE(d) chordal
 * form with shape r x (d + 1), where the residual is localPose - neighborPose.
 */
class ObjectConsensus {
 public:
  struct PairState {
    Matrix localPose;
    Matrix neighborPose;
    double weight{1.0};
    bool hasLocal{false};
    bool hasNeighbor{false};
  };

  explicit ObjectConsensus(unsigned d = 0, unsigned r = 0);

  unsigned dimension() const { return mD; }
  unsigned relaxationRank() const { return mR; }

  void clear();

  bool hasDual(unsigned neighborID, unsigned objectID) const;
  Matrix getDual(unsigned neighborID, unsigned objectID) const;
  void setDual(unsigned neighborID, unsigned objectID, const Matrix &lambda);

  /**
   * @brief Update the dual variable for one consensus pair.
   *
   * Uses a simple dual ascent step:
   * lambda <- lambda + eta * beta * weight * (localPose - neighborPose)
   */
  void updateDual(unsigned neighborID, unsigned objectID,
                  const Matrix &localPose, const Matrix &neighborPose,
                  double beta, double eta, double weight);

  /**
   * @brief Evaluate the stored consensus cost over all known pairs.
   *
   * Returns sum beta / 2 * || w * (local - neighbor) + lambda / beta ||^2.
   * The primalResidualNorm output reports the aggregated weighted residual norm
   * sqrt(sum ||w * (local - neighbor)||^2).
   */
  double evaluateConsensusCost(double beta, double &primalResidualNorm) const;

  /**
   * @brief Evaluate consensus cost for a single pair without mutating state.
   */
  double evaluateConsensusCost(unsigned neighborID, unsigned objectID,
                               const Matrix &localPose,
                               const Matrix &neighborPose, double beta,
                               double weight,
                               double &primalResidualNorm) const;

  /**
   * @brief Access stored local / neighbor pose blocks for a pair.
   */
  bool getPairState(unsigned neighborID, unsigned objectID,
                    PairState &state) const;

 private:
  using ObjectMap = std::map<unsigned, PairState>;
  using DualMap = std::map<unsigned, Matrix>;

  void ensureDimensions(const Matrix &localPose, const Matrix &neighborPose);
  Matrix &dualRef(unsigned neighborID, unsigned objectID);
  const Matrix *findDual(unsigned neighborID, unsigned objectID) const;
  PairState &stateRef(unsigned neighborID, unsigned objectID);
  const PairState *findState(unsigned neighborID, unsigned objectID) const;

  unsigned mD;
  unsigned mR;
  std::map<unsigned, ObjectMap> mStates;
  std::map<unsigned, DualMap> mLambda;
};

}  // namespace DPGO

#endif
