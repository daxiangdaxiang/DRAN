/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/MPIDCCICommunicator.h>

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace DPGO {

namespace {

int checkedMPIIntSize(size_t value, const char *name) {
  if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string("MPI payload too large for ") + name);
  }
  return static_cast<int>(value);
}

template <typename T>
std::vector<T> allGatherVector(const std::vector<T> &local,
                               MPI_Datatype datatype, MPI_Comm comm,
                               std::vector<int> *counts_out = nullptr) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  const int localCount = checkedMPIIntSize(local.size(), "Allgatherv");
  std::vector<int> counts(size, 0);
  MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  std::vector<int> displacements(size, 0);
  int total = 0;
  for (int rank = 0; rank < size; ++rank) {
    displacements[rank] = total;
    total += counts[rank];
  }

  std::vector<T> gathered(static_cast<size_t>(total));
  MPI_Allgatherv(local.empty() ? nullptr : local.data(), localCount, datatype,
                 gathered.empty() ? nullptr : gathered.data(), counts.data(),
                 displacements.data(), datatype, comm);
  if (counts_out != nullptr) {
    *counts_out = counts;
  }
  return gathered;
}

template <typename BlockT>
void appendBlockPayload(const PoseID &pose_id, const BlockT &block,
                        std::vector<long long> *meta,
                        std::vector<double> *values) {
  meta->push_back(static_cast<long long>(pose_id.first));
  meta->push_back(static_cast<long long>(pose_id.second));
  meta->push_back(static_cast<long long>(block.rows()));
  meta->push_back(static_cast<long long>(block.cols()));
  for (int col = 0; col < block.cols(); ++col) {
    for (int row = 0; row < block.rows(); ++row) {
      values->push_back(block(row, col));
    }
  }
}

template <typename BlockT>
BlockT readBlockPayload(const std::vector<long long> &meta, size_t meta_offset,
                        const std::vector<double> &values,
                        size_t *value_offset) {
  const int rows = static_cast<int>(meta[meta_offset + 2]);
  const int cols = static_cast<int>(meta[meta_offset + 3]);
  BlockT block(rows, cols);
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      if (*value_offset >= values.size()) {
        throw std::runtime_error("MPI DCCI block payload is truncated");
      }
      block(row, col) = values[*value_offset];
      ++(*value_offset);
    }
  }
  return block;
}

}  // namespace

MPIDCCICommunicator::MPIDCCICommunicator(
    const DistributedPartition &partition, MPI_Comm comm)
    : comm_(comm), partition_(partition) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::invalid_argument(
        "MPIDCCICommunicator requires MPI_Init before construction");
  }
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  if (partition_.num_robots != size_) {
    throw std::invalid_argument(
        "MPIDCCICommunicator partition num_robots must match MPI size");
  }
  if (partition_.my_robot_id != rank_) {
    throw std::invalid_argument(
        "MPIDCCICommunicator partition my_robot_id must match MPI rank");
  }
  for (const auto &entry : partition_.owner_robot) {
    if (entry.second < 0 || entry.second >= size_) {
      throw std::invalid_argument(
          "MPIDCCICommunicator partition owner out of range");
    }
  }
}

void MPIDCCICommunicator::setRotationBlock(const PoseID &pose_id,
                                          const Matrix &block) {
  validatePoseKnown(pose_id);
  if (ownerOf(pose_id) != rank_) {
    throw std::invalid_argument(
        "MPIDCCICommunicator can only publish locally owned rotation blocks");
  }
  rotationBlocks_[pose_id] = block;
}

void MPIDCCICommunicator::setTranslationBlock(const PoseID &pose_id,
                                             const Vector &block) {
  validatePoseKnown(pose_id);
  if (ownerOf(pose_id) != rank_) {
    throw std::invalid_argument(
        "MPIDCCICommunicator can only publish locally owned translation blocks");
  }
  translationBlocks_[pose_id] = block;
}

double MPIDCCICommunicator::allReduceSum(double value) {
  double result = 0.0;
  MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, comm_);
  ++stats_.num_scalar_reductions;
  return result;
}

std::map<PoseID, Matrix> MPIDCCICommunicator::exchangeRotationBlocks(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids) {
  return exchangeBlocks(requester_robot, requested_pose_ids, rotationBlocks_);
}

std::map<PoseID, Vector> MPIDCCICommunicator::exchangeTranslationBlocks(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids) {
  return exchangeBlocks(requester_robot, requested_pose_ids,
                        translationBlocks_);
}

int MPIDCCICommunicator::ownerOf(const PoseID &pose_id) const {
  const auto owner = partition_.owner_robot.find(pose_id);
  if (owner == partition_.owner_robot.end()) {
    throw std::invalid_argument("MPIDCCICommunicator unknown pose owner");
  }
  return owner->second;
}

void MPIDCCICommunicator::validatePoseKnown(const PoseID &pose_id) const {
  if (partition_.owner_robot.find(pose_id) == partition_.owner_robot.end()) {
    throw std::invalid_argument("MPIDCCICommunicator unknown pose id");
  }
}

void MPIDCCICommunicator::validateRequester(int requester_robot) const {
  if (requester_robot < 0 || requester_robot >= size_) {
    throw std::invalid_argument(
        "MPIDCCICommunicator requester rank out of range");
  }
}

template <typename BlockT>
std::map<PoseID, BlockT> MPIDCCICommunicator::exchangeBlocks(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids,
    const std::unordered_map<PoseID, BlockT, PoseIDHash> &local_blocks) {
  validateRequester(requester_robot);
  std::set<PoseID> requested(requested_pose_ids.begin(),
                             requested_pose_ids.end());
  for (const PoseID &pose_id : requested) {
    validatePoseKnown(pose_id);
  }

  std::vector<long long> localMeta;
  std::vector<double> localValues;
  localMeta.reserve(local_blocks.size() * 4);
  for (const auto &entry : local_blocks) {
    appendBlockPayload(entry.first, entry.second, &localMeta, &localValues);
  }

  std::vector<int> metaCounts;
  const std::vector<long long> gatheredMeta =
      allGatherVector(localMeta, MPI_LONG_LONG, comm_, &metaCounts);
  std::vector<int> valueCounts;
  const std::vector<double> gatheredValues =
      allGatherVector(localValues, MPI_DOUBLE, comm_, &valueCounts);

  size_t totalMetaScalars = 0;
  size_t totalValueScalars = 0;
  for (int count : metaCounts) {
    totalMetaScalars += static_cast<size_t>(count);
  }
  for (int count : valueCounts) {
    totalValueScalars += static_cast<size_t>(count);
  }
  stats_.num_block_messages += (totalMetaScalars / 4) *
                               static_cast<size_t>(std::max(0, size_ - 1));
  stats_.bytes_sent +=
      (totalMetaScalars * sizeof(long long) +
       totalValueScalars * sizeof(double)) *
      static_cast<size_t>(std::max(0, size_ - 1));

  std::map<PoseID, BlockT> result;
  size_t metaOffset = 0;
  size_t valueOffset = 0;
  while (metaOffset < gatheredMeta.size()) {
    if (metaOffset + 4 > gatheredMeta.size()) {
      throw std::runtime_error("MPI DCCI metadata payload is truncated");
    }
    const PoseID pose_id{
        static_cast<unsigned>(gatheredMeta[metaOffset]),
        static_cast<unsigned>(gatheredMeta[metaOffset + 1])};
    const BlockT block =
        readBlockPayload<BlockT>(gatheredMeta, metaOffset, gatheredValues,
                                 &valueOffset);
    metaOffset += 4;

    if (requested.find(pose_id) == requested.end()) {
      continue;
    }
    if (ownerOf(pose_id) == requester_robot) {
      continue;
    }
    result[pose_id] = block;
  }

  for (const PoseID &pose_id : requested) {
    if (ownerOf(pose_id) == requester_robot) {
      continue;
    }
    if (result.find(pose_id) == result.end()) {
      throw std::runtime_error("MPI DCCI did not receive requested ghost block");
    }
  }
  return result;
}

template std::map<PoseID, Matrix> MPIDCCICommunicator::exchangeBlocks<Matrix>(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids,
    const std::unordered_map<PoseID, Matrix, PoseIDHash> &local_blocks);

template std::map<PoseID, Vector> MPIDCCICommunicator::exchangeBlocks<Vector>(
    int requester_robot, const std::vector<PoseID> &requested_pose_ids,
    const std::unordered_map<PoseID, Vector, PoseIDHash> &local_blocks);

}  // namespace DPGO
