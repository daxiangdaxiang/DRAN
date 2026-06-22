/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DCCI_COMMUNICATOR_H
#define DCCI_COMMUNICATOR_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

namespace DPGO {

struct PoseIDHash {
  size_t operator()(const PoseID &pose) const {
    const size_t h1 = std::hash<unsigned>{}(pose.first);
    const size_t h2 = std::hash<unsigned>{}(pose.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

struct DistributedPartition {
  std::vector<PoseID> global_pose_ids;
  std::unordered_map<PoseID, size_t, PoseIDHash> global_to_local;
  std::unordered_map<PoseID, int, PoseIDHash> owner_robot;
  int my_robot_id = 0;
  int num_robots = 1;
};

struct DCCICommunicationStats {
  size_t num_scalar_reductions = 0;
  size_t num_block_messages = 0;
  size_t bytes_sent = 0;
};

class DPGOCommunicator {
 public:
  virtual ~DPGOCommunicator() = default;

  virtual double allReduceSum(double value) = 0;
  virtual std::map<PoseID, Matrix> exchangeRotationBlocks(
      int requester_robot, const std::vector<PoseID> &requested_pose_ids) = 0;
  virtual std::map<PoseID, Vector> exchangeTranslationBlocks(
      int requester_robot, const std::vector<PoseID> &requested_pose_ids) = 0;
  virtual const DCCICommunicationStats &stats() const = 0;
  virtual void resetStats() = 0;
};

}  // namespace DPGO

#endif
