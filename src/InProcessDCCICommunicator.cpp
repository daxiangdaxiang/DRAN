/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/InProcessDCCICommunicator.h>

#include <set>
#include <stdexcept>
#include <string>

namespace DPGO {

InProcessDCCICommunicator::InProcessDCCICommunicator(
    const std::vector<DistributedPartition> &partitions)
    : partitions_(partitions) {
  if (partitions_.empty()) {
    throw std::invalid_argument(
        "InProcessDCCICommunicator requires at least one partition");
  }
  const int expectedNumRobots = static_cast<int>(partitions_.size());
  for (int robot = 0; robot < expectedNumRobots; ++robot) {
    const auto &partition = partitions_[robot];
    if (partition.my_robot_id != robot) {
      throw std::invalid_argument(
          "InProcessDCCICommunicator partitions must be ordered by robot id");
    }
    if (partition.num_robots != expectedNumRobots) {
      throw std::invalid_argument(
          "InProcessDCCICommunicator partition num_robots mismatch");
    }
    for (const auto &pose_id : partition.global_pose_ids) {
      validatePoseKnown(pose_id);
      const auto local = partition.global_to_local.find(pose_id);
      if (local == partition.global_to_local.end()) {
        throw std::invalid_argument(
            "InProcessDCCICommunicator partition missing local index");
      }
    }
  }
}

int InProcessDCCICommunicator::numRobots() const {
  return static_cast<int>(partitions_.size());
}

const DistributedPartition &InProcessDCCICommunicator::partition(
    int robot_id) const {
  validateRobotId(robot_id);
  return partitions_[robot_id];
}

void InProcessDCCICommunicator::setRotationBlock(const PoseID &pose_id,
                                                const Matrix &block) {
  validatePoseKnown(pose_id);
  rotationBlocks_[pose_id] = block;
}

void InProcessDCCICommunicator::setTranslationBlock(const PoseID &pose_id,
                                                   const Vector &block) {
  validatePoseKnown(pose_id);
  translationBlocks_[pose_id] = block;
}

Matrix InProcessDCCICommunicator::rotationBlock(const PoseID &pose_id) const {
  const auto it = rotationBlocks_.find(pose_id);
  if (it == rotationBlocks_.end()) {
    throw std::invalid_argument(
        "InProcessDCCICommunicator missing rotation block");
  }
  return it->second;
}

Vector InProcessDCCICommunicator::translationBlock(const PoseID &pose_id) const {
  const auto it = translationBlocks_.find(pose_id);
  if (it == translationBlocks_.end()) {
    throw std::invalid_argument(
        "InProcessDCCICommunicator missing translation block");
  }
  return it->second;
}

double InProcessDCCICommunicator::allReduceSum(double value) {
  ++stats_.num_scalar_reductions;
  return value;
}

std::map<PoseID, Matrix> InProcessDCCICommunicator::exchangeRotationBlocks(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids) {
  validateRobotId(requester_robot);
  std::map<PoseID, Matrix> result;
  std::set<PoseID> uniqueRequests(requested_pose_ids.begin(),
                                  requested_pose_ids.end());
  for (const auto &pose_id : uniqueRequests) {
    validatePoseKnown(pose_id);
    if (ownerOf(pose_id) == requester_robot) {
      continue;
    }
    const Matrix block = rotationBlock(pose_id);
    result.emplace(pose_id, block);
    ++stats_.num_block_messages;
    stats_.bytes_sent += static_cast<size_t>(block.size()) * sizeof(double);
  }
  return result;
}

std::map<PoseID, Vector> InProcessDCCICommunicator::exchangeTranslationBlocks(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids) {
  validateRobotId(requester_robot);
  std::map<PoseID, Vector> result;
  std::set<PoseID> uniqueRequests(requested_pose_ids.begin(),
                                  requested_pose_ids.end());
  for (const auto &pose_id : uniqueRequests) {
    validatePoseKnown(pose_id);
    if (ownerOf(pose_id) == requester_robot) {
      continue;
    }
    const Vector block = translationBlock(pose_id);
    result.emplace(pose_id, block);
    ++stats_.num_block_messages;
    stats_.bytes_sent += static_cast<size_t>(block.size()) * sizeof(double);
  }
  return result;
}

int InProcessDCCICommunicator::ownerOf(const PoseID &pose_id) const {
  const auto it = partitions_.front().owner_robot.find(pose_id);
  if (it == partitions_.front().owner_robot.end()) {
    throw std::invalid_argument(
        "InProcessDCCICommunicator pose owner is unknown");
  }
  validateRobotId(it->second);
  return it->second;
}

void InProcessDCCICommunicator::validateRobotId(int robot_id) const {
  if (robot_id < 0 || robot_id >= static_cast<int>(partitions_.size())) {
    throw std::invalid_argument("InProcessDCCICommunicator invalid robot id " +
                                std::to_string(robot_id));
  }
}

void InProcessDCCICommunicator::validatePoseKnown(const PoseID &pose_id) const {
  const auto it = partitions_.front().owner_robot.find(pose_id);
  if (it == partitions_.front().owner_robot.end()) {
    throw std::invalid_argument(
        "InProcessDCCICommunicator unknown pose id");
  }
}

}  // namespace DPGO
