/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef IN_PROCESS_DCCI_COMMUNICATOR_H
#define IN_PROCESS_DCCI_COMMUNICATOR_H

#include <DPGO/DCCI_communicator.h>

namespace DPGO {

class InProcessDCCICommunicator : public DPGOCommunicator {
 public:
  explicit InProcessDCCICommunicator(
      const std::vector<DistributedPartition> &partitions);

  int numRobots() const;
  const DistributedPartition &partition(int robot_id) const;

  void setRotationBlock(const PoseID &pose_id, const Matrix &block);
  void setTranslationBlock(const PoseID &pose_id, const Vector &block);
  Matrix rotationBlock(const PoseID &pose_id) const;
  Vector translationBlock(const PoseID &pose_id) const;

  double allReduceSum(double value) override;
  std::map<PoseID, Matrix> exchangeRotationBlocks(
      int requester_robot,
      const std::vector<PoseID> &requested_pose_ids) override;
  std::map<PoseID, Vector> exchangeTranslationBlocks(
      int requester_robot,
      const std::vector<PoseID> &requested_pose_ids) override;
  const DCCICommunicationStats &stats() const override { return stats_; }
  void resetStats() override { stats_ = DCCICommunicationStats{}; }

 private:
  int ownerOf(const PoseID &pose_id) const;
  void validateRobotId(int robot_id) const;
  void validatePoseKnown(const PoseID &pose_id) const;

  std::vector<DistributedPartition> partitions_;
  std::unordered_map<PoseID, Matrix, PoseIDHash> rotationBlocks_;
  std::unordered_map<PoseID, Vector, PoseIDHash> translationBlocks_;
  DCCICommunicationStats stats_;
};

}  // namespace DPGO

#endif
