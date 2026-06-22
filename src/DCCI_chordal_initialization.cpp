/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DCCI_chordal_initialization.h>
#include <DPGO/DCCI_linear_operators.h>
#include <DPGO/DPGO_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DPGO {

namespace {

PoseID poseID(size_t pose) {
  return PoseID{0, static_cast<unsigned>(pose)};
}

void validateInputs(size_t dimension, size_t num_poses,
                    const std::vector<RelativeSEMeasurement> &measurements,
                    const DistributedPartition &partition) {
  if (dimension == 0) {
    throw std::invalid_argument(
        "distributedChordalInitialization requires positive dimension");
  }
  if (num_poses == 0) {
    throw std::invalid_argument(
        "distributedChordalInitialization requires at least one pose");
  }
  if (partition.num_robots <= 0 || partition.my_robot_id < 0 ||
      partition.my_robot_id >= partition.num_robots) {
    throw std::invalid_argument(
        "distributedChordalInitialization received invalid partition");
  }
  for (size_t pose = 0; pose < num_poses; ++pose) {
    if (partition.owner_robot.find(poseID(pose)) ==
        partition.owner_robot.end()) {
      throw std::invalid_argument(
          "distributedChordalInitialization partition missing pose owner");
    }
  }
  for (const auto &measurement : measurements) {
    if (measurement.p1 >= num_poses || measurement.p2 >= num_poses) {
      throw std::invalid_argument(
          "distributedChordalInitialization measurement pose index out of range");
    }
    if (static_cast<size_t>(measurement.R.rows()) != dimension ||
        static_cast<size_t>(measurement.R.cols()) != dimension ||
        static_cast<size_t>(measurement.t.rows()) != dimension ||
        measurement.t.cols() != 1) {
      throw std::invalid_argument(
          "distributedChordalInitialization measurement dimension mismatch");
    }
  }
}

void validateAnchorConnectivity(
    size_t num_poses, const std::vector<RelativeSEMeasurement> &measurements) {
  std::vector<std::vector<size_t>> adjacency(num_poses);
  for (const auto &measurement : measurements) {
    adjacency[measurement.p1].push_back(measurement.p2);
    adjacency[measurement.p2].push_back(measurement.p1);
  }

  std::vector<bool> visited(num_poses, false);
  std::queue<size_t> queue;
  visited[0] = true;
  queue.push(0);
  while (!queue.empty()) {
    const size_t pose = queue.front();
    queue.pop();
    for (const size_t neighbor : adjacency[pose]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        queue.push(neighbor);
      }
    }
  }

  for (size_t pose = 0; pose < num_poses; ++pose) {
    if (!visited[pose]) {
      throw std::invalid_argument(
          "distributedChordalInitialization graph is not connected to anchor");
    }
  }
}

Vector applyJacobi(const Vector &diagonal, const Vector &r) {
  Vector z = r;
  for (int i = 0; i < z.size(); ++i) {
    z(i) = std::abs(diagonal(i)) > 1e-14 ? r(i) / diagonal(i) : r(i);
  }
  return z;
}

void requireConverged(const PCGResult &result, const char *system_name) {
  if (!result.converged) {
    throw std::runtime_error(std::string(
        "distributedChordalInitialization PCG did not converge for ") +
                             system_name);
  }
}

Matrix recoverProjectedRotations(size_t d, size_t num_poses,
                                 const Vector &reduced_rotation_vector) {
  Matrix rotations(d, d * num_poses);
  rotations.leftCols(d) = Matrix::Identity(d, d);
  if (num_poses > 1) {
    rotations.rightCols((num_poses - 1) * d) =
        Eigen::Map<const Matrix>(reduced_rotation_vector.data(), d,
                                 (num_poses - 1) * d);
  }
  for (size_t pose = 1; pose < num_poses; ++pose) {
    rotations.block(0, pose * d, d, d) =
        projectToRotationGroup(rotations.block(0, pose * d, d, d));
  }
  return rotations;
}

Matrix assemblePoses(size_t d, size_t num_poses, const Matrix &rotations,
                     const Vector &reduced_translation_vector) {
  Matrix translations = Matrix::Zero(d, num_poses);
  if (num_poses > 1) {
    translations.rightCols(num_poses - 1) =
        Eigen::Map<const Matrix>(reduced_translation_vector.data(), d,
                                 num_poses - 1);
  }

  Matrix poses(d, num_poses * (d + 1));
  for (size_t pose = 0; pose < num_poses; ++pose) {
    poses.block(0, pose * (d + 1), d, d) =
        rotations.block(0, pose * d, d, d);
    poses.block(0, pose * (d + 1) + d, d, 1) =
        translations.block(0, pose, d, 1);
  }
  return poses;
}

unsigned traceStride(const DCCITraceOptions *trace, const PCGParams &params) {
  if (trace == nullptr || trace->frames == nullptr) {
    return 0;
  }
  if (trace->frame_stride > 0) {
    return trace->frame_stride;
  }
  if (trace->max_frames_per_stage == 0) {
    return 1;
  }
  return std::max(1u, (params.max_iters + trace->max_frames_per_stage - 1) /
                           trace->max_frames_per_stage);
}

void appendTraceFrame(const DCCITraceOptions *trace, size_t d,
                      size_t num_poses, const Matrix &rotations,
                      const Vector &translation_vector, unsigned pcg_iteration,
                      double residual_norm) {
  if (trace == nullptr || trace->frames == nullptr) {
    return;
  }
  DCCITraceFrame frame;
  frame.stage = DCCITraceStage::TranslationPCG;
  frame.pcg_iteration = pcg_iteration;
  frame.residual_norm = residual_norm;
  frame.poses = assemblePoses(d, num_poses, rotations, translation_vector);
  trace->frames->push_back(std::move(frame));
}

}  // namespace

Matrix distributedChordalInitialization(
    size_t dimension, size_t num_poses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const DistributedPartition &partition, DPGOCommunicator &comm,
    const DCCIParams &params, DCCIStats *stats,
    const DCCITraceOptions *trace) {
  validateInputs(dimension, num_poses, measurements, partition);
  validateAnchorConnectivity(num_poses, measurements);

  const ChordalRotationLinearOperator rotationOp(
      dimension, num_poses, measurements, params.use_measurement_weight);
  const Vector rotationDiagonal =
      rotationOp.buildJacobiPreconditionerDiagonal();
  const PCGResult rotationResult = solvePCG(
      rotationOp.buildRHS(),
      [&](const Vector &x) { return rotationOp.applyH(x); },
      [&](const Vector &r) { return applyJacobi(rotationDiagonal, r); },
      [&](const Vector &a, const Vector &b) {
        return comm.allReduceSum(a.dot(b));
      },
      params.rotation_pcg);
  requireConverged(rotationResult, "rotation");

  const Matrix rotations =
      recoverProjectedRotations(dimension, num_poses, rotationResult.x);

  const ChordalTranslationLinearOperator translationOp(
      dimension, num_poses, measurements, rotations,
      params.use_measurement_weight);
  const Vector translationDiagonal =
      translationOp.buildJacobiPreconditionerDiagonal();
  const Vector translationRhs = translationOp.buildRHS();
  PCGParams translationParams = params.translation_pcg;
  const auto userTranslationCallback =
      translationParams.iteration_callback;
  const unsigned stride = traceStride(trace, params.translation_pcg);
  unsigned traceFramesForStage = 0;
  unsigned lastTraceIteration = std::numeric_limits<unsigned>::max();
  if (trace != nullptr && trace->frames != nullptr) {
    appendTraceFrame(trace, dimension, num_poses, rotations,
                     Vector::Zero(translationOp.reducedDimension()), 0,
                     translationRhs.norm());
    traceFramesForStage = 1;
    lastTraceIteration = 0;
  }
  if (stride > 0 || userTranslationCallback) {
    translationParams.iteration_callback =
        [&, userTranslationCallback](unsigned iter, const Vector &x,
                                     double residual, bool converged) {
          if (userTranslationCallback) {
            userTranslationCallback(iter, x, residual, converged);
          }
          if (stride == 0) {
            return;
          }
          const bool underFrameLimit =
              trace->max_frames_per_stage == 0 ||
              traceFramesForStage < trace->max_frames_per_stage;
          const bool shouldSample =
              converged || iter == 1 || iter % stride == 0;
          if (shouldSample &&
              (underFrameLimit || converged) &&
              lastTraceIteration != iter) {
            appendTraceFrame(trace, dimension, num_poses, rotations, x, iter,
                             residual);
            ++traceFramesForStage;
            lastTraceIteration = iter;
          }
        };
  }
  const PCGResult translationResult = solvePCG(
      translationRhs,
      [&](const Vector &x) { return translationOp.applyH(x); },
      [&](const Vector &r) { return applyJacobi(translationDiagonal, r); },
      [&](const Vector &a, const Vector &b) {
        return comm.allReduceSum(a.dot(b));
      },
      translationParams);
  requireConverged(translationResult, "translation");
  if (trace != nullptr && trace->frames != nullptr &&
      lastTraceIteration != translationResult.iters) {
    appendTraceFrame(trace, dimension, num_poses, rotations,
                     translationResult.x, translationResult.iters,
                     translationResult.final_residual_norm);
  }

  if (stats != nullptr) {
    stats->rotation_pcg = rotationResult;
    stats->translation_pcg = translationResult;
    stats->communication = comm.stats();
  }

  return assemblePoses(dimension, num_poses, rotations, translationResult.x);
}

}  // namespace DPGO
