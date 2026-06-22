/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef MPI_DCCI_COMMUNICATOR_H
#define MPI_DCCI_COMMUNICATOR_H

#include <DPGO/DCCI_communicator.h>

#include <mpi.h>

namespace DPGO {

class MPIDCCICommunicator : public DPGOCommunicator {
 public:
  explicit MPIDCCICommunicator(const DistributedPartition &partition,
                               MPI_Comm comm = MPI_COMM_WORLD);

  void setRotationBlock(const PoseID &pose_id, const Matrix &block);
  void setTranslationBlock(const PoseID &pose_id, const Vector &block);

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
  void validatePoseKnown(const PoseID &pose_id) const;
  void validateRequester(int requester_robot) const;

  template <typename BlockT>
  std::map<PoseID, BlockT> exchangeBlocks(
      int requester_robot, const std::vector<PoseID> &requested_pose_ids,
      const std::unordered_map<PoseID, BlockT, PoseIDHash> &local_blocks);

  MPI_Comm comm_;
  int rank_ = 0;
  int size_ = 1;
  DistributedPartition partition_;
  std::unordered_map<PoseID, Matrix, PoseIDHash> rotationBlocks_;
  std::unordered_map<PoseID, Vector, PoseIDHash> translationBlocks_;
  DCCICommunicationStats stats_;
};

}  // namespace DPGO

#endif
