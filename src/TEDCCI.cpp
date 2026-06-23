/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/TEDCCI.h>
#include <DPGO/RIFTIF.h>
#include <DPGO/DPGO_utils.h>

#include <Eigen/QR>
#include <Eigen/SPQRSupport>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace DPGO {

namespace {

using ColMajorSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

template <typename Fn>
double elapsedMilliseconds(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto finish = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

TEDCCIBackend resolveBackend(TEDCCIBackend requested) {
  if (requested == TEDCCIBackend::AUTO) {
    return TEDCCIBackend::SPARSE_SPQR;
  }
  return requested;
}

PoseKey keyForFirstEndpoint(const RelativeSEMeasurement &measurement) {
  return PoseKey{static_cast<int>(measurement.r1),
                 static_cast<int>(measurement.p1)};
}

PoseKey keyForSecondEndpoint(const RelativeSEMeasurement &measurement) {
  return PoseKey{static_cast<int>(measurement.r2),
                 static_cast<int>(measurement.p2)};
}

PoseKey anchorKey(const TEDCCIParams &params) {
  return PoseKey{params.anchor_robot_id, params.anchor_pose_id};
}

bool isAnchor(const PoseKey &key, const TEDCCIParams &params) {
  return key == anchorKey(params);
}

double effectiveScale(double precision, double weight, bool useWeight,
                      const char *name) {
  const double scaledPrecision = precision * (useWeight ? weight : 1.0);
  if (!std::isfinite(scaledPrecision) || scaledPrecision < 0.0) {
    throw std::invalid_argument(std::string("TED-CCI invalid ") + name);
  }
  return std::sqrt(scaledPrecision);
}

int rotationVectorIndex(int d, int row, int col) {
  return row + col * d;
}

void addRotationProductBlock(Matrix &A, int rowOffset, int colOffset,
                             const Matrix &R, double scale) {
  const int d = static_cast<int>(R.rows());
  for (int col = 0; col < d; ++col) {
    for (int k = 0; k < d; ++k) {
      for (int row = 0; row < d; ++row) {
        A(rowOffset + rotationVectorIndex(d, row, col),
          colOffset + rotationVectorIndex(d, row, k)) +=
            scale * R(k, col);
      }
    }
  }
}

void addRotationIdentityBlock(Matrix &A, int rowOffset, int colOffset, int d,
                              double scale) {
  for (int idx = 0; idx < d * d; ++idx) {
    A(rowOffset + idx, colOffset + idx) += scale;
  }
}

Vector vectorizeMatrix(const Matrix &M) {
  return Eigen::Map<const Vector>(M.data(), M.rows() * M.cols());
}

void validateDimension(int dimension, const RelativeSEMeasurement &measurement) {
  if (dimension <= 0) {
    throw std::invalid_argument("TED-CCI factor dimension must be positive");
  }
  if (measurement.R.rows() != dimension || measurement.R.cols() != dimension ||
      measurement.t.rows() != dimension || measurement.t.cols() != 1) {
    throw std::invalid_argument("TED-CCI measurement dimension mismatch");
  }
}

std::vector<PoseKey> nonAnchorKeys(const PoseKey &first, const PoseKey &second,
                                   const TEDCCIParams &params) {
  std::vector<PoseKey> keys;
  if (!isAnchor(first, params)) {
    keys.push_back(first);
  }
  if (!isAnchor(second, params) && !(second == first)) {
    keys.push_back(second);
  }
  return keys;
}

int keyOffset(const std::vector<PoseKey> &keys, const PoseKey &key,
              int blockDim) {
  for (std::size_t idx = 0; idx < keys.size(); ++idx) {
    if (keys[idx] == key) {
      return static_cast<int>(idx) * blockDim;
    }
  }
  throw std::invalid_argument("TED-CCI requested key not present in factor");
}

std::vector<PoseKey> sortedUnique(std::vector<PoseKey> keys) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

bool containsKey(const std::vector<PoseKey> &keys, const PoseKey &key) {
  return std::binary_search(keys.begin(), keys.end(), key);
}

void addBoundaryForSharedEndpoint(
    int localRobotId, const std::vector<PoseKey> &localPoses,
    const RelativeSEMeasurement &measurement, std::vector<PoseKey> *boundary,
    std::map<int, std::vector<PoseKey>> *neighborBoundary) {
  const PoseKey first = keyForFirstEndpoint(measurement);
  const PoseKey second = keyForSecondEndpoint(measurement);
  if (first.robot_id == second.robot_id) {
    return;
  }

  PoseKey local;
  int neighbor = -1;
  if (first.robot_id == localRobotId) {
    local = first;
    neighbor = second.robot_id;
  } else if (second.robot_id == localRobotId) {
    local = second;
    neighbor = first.robot_id;
  } else {
    return;
  }

  if (!containsKey(localPoses, local)) {
    throw std::invalid_argument(
        "TED-CCI shared measurement references a non-local pose");
  }
  boundary->push_back(local);
  (*neighborBoundary)[neighbor].push_back(local);
}

std::map<PoseKey, int> buildColumnOffsets(const std::vector<PoseKey> &keys,
                                          int blockDim) {
  std::map<PoseKey, int> offsets;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    offsets[keys[i]] = static_cast<int>(i) * blockDim;
  }
  return offsets;
}

void stackFactors(const std::vector<LinearFactorBlock> &factors,
                  const std::vector<PoseKey> &interiorKeys,
                  const std::vector<PoseKey> &boundaryKeys, int blockDim,
                  Matrix *A, Vector *b) {
  if (blockDim <= 0) {
    throw std::invalid_argument("TED-CCI block dimension must be positive");
  }
  std::vector<PoseKey> allKeys = interiorKeys;
  allKeys.insert(allKeys.end(), boundaryKeys.begin(), boundaryKeys.end());
  const auto offsets = buildColumnOffsets(allKeys, blockDim);
  int rows = 0;
  for (const auto &factor : factors) {
    if (factor.A.rows() != factor.b.rows()) {
      throw std::invalid_argument("TED-CCI factor A/b row mismatch");
    }
    if (factor.A.cols() !=
        static_cast<int>(factor.keys.size()) * blockDim) {
      throw std::invalid_argument("TED-CCI factor column count mismatch");
    }
    rows += factor.A.rows();
  }

  A->setZero(rows, static_cast<int>(allKeys.size()) * blockDim);
  b->resize(rows);
  int rowOffset = 0;
  for (const auto &factor : factors) {
    for (std::size_t keyIndex = 0; keyIndex < factor.keys.size(); ++keyIndex) {
      const auto offset = offsets.find(factor.keys[keyIndex]);
      if (offset == offsets.end()) {
        throw std::invalid_argument(
            "TED-CCI factor references a key outside condensation split");
      }
      A->block(rowOffset, offset->second, factor.A.rows(), blockDim) +=
          factor.A.block(0, static_cast<int>(keyIndex) * blockDim,
                         factor.A.rows(), blockDim);
    }
    b->segment(rowOffset, factor.b.rows()) = factor.b;
    rowOffset += factor.A.rows();
  }
}

void stackFactorsSparse(
    const std::vector<LinearFactorBlock> &factors,
    const std::vector<PoseKey> &keys, int blockDim,
    ColMajorSparseMatrix *A, Vector *b) {
  if (blockDim <= 0) {
    throw std::invalid_argument("TED-CCI block dimension must be positive");
  }
  const auto offsets = buildColumnOffsets(keys, blockDim);
  int rows = 0;
  std::size_t tripletReserve = 0;
  for (const auto &factor : factors) {
    if (factor.A.rows() != factor.b.rows()) {
      throw std::invalid_argument("TED-CCI factor A/b row mismatch");
    }
    if (factor.A.cols() !=
        static_cast<int>(factor.keys.size()) * blockDim) {
      throw std::invalid_argument("TED-CCI factor column count mismatch");
    }
    rows += factor.A.rows();
    tripletReserve += static_cast<std::size_t>(factor.A.rows()) *
                      static_cast<std::size_t>(factor.A.cols());
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(tripletReserve);
  b->resize(rows);
  int rowOffset = 0;
  for (const auto &factor : factors) {
    for (std::size_t keyIndex = 0; keyIndex < factor.keys.size();
         ++keyIndex) {
      const auto offset = offsets.find(factor.keys[keyIndex]);
      if (offset == offsets.end()) {
        throw std::invalid_argument(
            "TED-CCI factor references a key outside sparse stack");
      }
      for (int r = 0; r < factor.A.rows(); ++r) {
        for (int c = 0; c < blockDim; ++c) {
          const double value =
              factor.A(r, static_cast<int>(keyIndex) * blockDim + c);
          if (value != 0.0) {
            triplets.emplace_back(rowOffset + r, offset->second + c, value);
          }
        }
      }
    }
    b->segment(rowOffset, factor.b.rows()) = factor.b;
    rowOffset += factor.A.rows();
  }

  A->resize(rows, static_cast<int>(keys.size()) * blockDim);
  A->setFromTriplets(triplets.begin(), triplets.end());
  A->makeCompressed();
}

void stackFactorsSparseSplit(
    const std::vector<LinearFactorBlock> &factors,
    const std::vector<PoseKey> &interiorKeys,
    const std::vector<PoseKey> &boundaryKeys, int blockDim,
    ColMajorSparseMatrix *AI, Matrix *boundaryAndRhs) {
  if (blockDim <= 0) {
    throw std::invalid_argument("TED-CCI block dimension must be positive");
  }
  const auto interiorOffsets = buildColumnOffsets(interiorKeys, blockDim);
  const auto boundaryOffsets = buildColumnOffsets(boundaryKeys, blockDim);
  int rows = 0;
  std::size_t tripletReserve = 0;
  for (const auto &factor : factors) {
    if (factor.A.rows() != factor.b.rows()) {
      throw std::invalid_argument("TED-CCI factor A/b row mismatch");
    }
    if (factor.A.cols() !=
        static_cast<int>(factor.keys.size()) * blockDim) {
      throw std::invalid_argument("TED-CCI factor column count mismatch");
    }
    rows += factor.A.rows();
    tripletReserve += static_cast<std::size_t>(factor.A.rows()) *
                      static_cast<std::size_t>(factor.A.cols());
  }

  const int boundaryDim = static_cast<int>(boundaryKeys.size()) * blockDim;
  boundaryAndRhs->setZero(rows, boundaryDim + 1);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(tripletReserve);

  int rowOffset = 0;
  for (const auto &factor : factors) {
    for (std::size_t keyIndex = 0; keyIndex < factor.keys.size();
         ++keyIndex) {
      const PoseKey key = factor.keys[keyIndex];
      const auto interiorOffset = interiorOffsets.find(key);
      const auto boundaryOffset = boundaryOffsets.find(key);
      if (interiorOffset == interiorOffsets.end() &&
          boundaryOffset == boundaryOffsets.end()) {
        throw std::invalid_argument(
            "TED-CCI factor references a key outside condensation split");
      }
      for (int r = 0; r < factor.A.rows(); ++r) {
        for (int c = 0; c < blockDim; ++c) {
          const double value =
              factor.A(r, static_cast<int>(keyIndex) * blockDim + c);
          if (value == 0.0) {
            continue;
          }
          if (interiorOffset != interiorOffsets.end()) {
            triplets.emplace_back(rowOffset + r, interiorOffset->second + c,
                                  value);
          } else {
            (*boundaryAndRhs)(rowOffset + r, boundaryOffset->second + c) +=
                value;
          }
        }
      }
    }
    boundaryAndRhs->block(rowOffset, boundaryDim, factor.b.rows(), 1) =
        factor.b;
    rowOffset += factor.A.rows();
  }

  AI->resize(rows, static_cast<int>(interiorKeys.size()) * blockDim);
  AI->setFromTriplets(triplets.begin(), triplets.end());
  AI->makeCompressed();
}

std::map<int, PoseKey> poseKeyByPoseId(
    int numPoses, const std::vector<TEDCCIPartition> &partitions) {
  std::map<int, PoseKey> result;
  for (const auto &partition : partitions) {
    for (const PoseKey &key : partition.local_poses) {
      if (key.pose_id < 0 || key.pose_id >= numPoses) {
        throw std::invalid_argument("TED-CCI partition pose id out of range");
      }
      if (result.count(key.pose_id) > 0) {
        throw std::invalid_argument("TED-CCI duplicate pose id in partitions");
      }
      result[key.pose_id] = key;
    }
  }
  for (int pose = 0; pose < numPoses; ++pose) {
    if (result.count(pose) == 0) {
      throw std::invalid_argument("TED-CCI partition missing pose id");
    }
  }
  return result;
}

std::vector<RelativeSEMeasurement> normalizeMeasurementOwners(
    int numPoses, const std::vector<RelativeSEMeasurement> &measurements,
    const std::vector<TEDCCIPartition> &partitions) {
  const auto poseKeys = poseKeyByPoseId(numPoses, partitions);
  std::vector<RelativeSEMeasurement> normalized = measurements;
  for (auto &measurement : normalized) {
    if (static_cast<int>(measurement.p1) >= numPoses ||
        static_cast<int>(measurement.p2) >= numPoses) {
      throw std::invalid_argument("TED-CCI measurement pose id out of range");
    }
    measurement.r1 =
        static_cast<unsigned>(poseKeys.at(static_cast<int>(measurement.p1))
                                  .robot_id);
    measurement.r2 =
        static_cast<unsigned>(poseKeys.at(static_cast<int>(measurement.p2))
                                  .robot_id);
  }
  return normalized;
}

std::vector<RelativeSEMeasurement> robotLocalMeasurements(
    int robot, const std::vector<RelativeSEMeasurement> &measurements) {
  std::vector<RelativeSEMeasurement> local;
  for (const auto &measurement : measurements) {
    if (measurement.r1 == measurement.r2 &&
        static_cast<int>(measurement.r1) == robot) {
      local.push_back(measurement);
    }
  }
  return local;
}

std::vector<RelativeSEMeasurement> sharedMeasurements(
    const std::vector<RelativeSEMeasurement> &measurements) {
  std::vector<RelativeSEMeasurement> shared;
  for (const auto &measurement : measurements) {
    if (measurement.r1 != measurement.r2) {
      shared.push_back(measurement);
    }
  }
  return shared;
}

LinearFactorBlock condensedAsFactor(const CondensedFactor &condensed) {
  LinearFactorBlock factor;
  factor.keys = condensed.boundary_keys;
  factor.A = condensed.Abar;
  factor.b = condensed.bbar;
  return factor;
}

std::size_t keyBytes(const std::vector<PoseKey> &keys) {
  return keys.size() * 2u * sizeof(int);
}

std::size_t denseBytes(const Matrix &A) {
  return static_cast<std::size_t>(A.rows()) *
         static_cast<std::size_t>(A.cols()) * sizeof(double);
}

std::size_t vectorBytes(const Vector &x) {
  return static_cast<std::size_t>(x.rows()) * sizeof(double);
}

std::size_t autoEncodingTagsBytes() { return 2u * sizeof(std::uint8_t); }

std::size_t condensedFactorDenseBytes(const CondensedFactor &factor) {
  const std::size_t matrixHeader = 2u * sizeof(int);
  const std::size_t vectorHeader = sizeof(int);
  return sizeof(int) + sizeof(std::uint64_t) + sizeof(int) +
         keyBytes(factor.boundary_keys) + autoEncodingTagsBytes() +
         matrixHeader + vectorHeader + denseBytes(factor.Abar) +
         vectorBytes(factor.bbar);
}

std::size_t matrixNonzeros(const Matrix &A) {
  std::size_t nonzeros = 0;
  for (int c = 0; c < A.cols(); ++c) {
    for (int r = 0; r < A.rows(); ++r) {
      if (A(r, c) != 0.0) {
        ++nonzeros;
      }
    }
  }
  return nonzeros;
}

std::size_t vectorNonzeros(const Vector &x) {
  std::size_t nonzeros = 0;
  for (int r = 0; r < x.rows(); ++r) {
    if (x(r) != 0.0) {
      ++nonzeros;
    }
  }
  return nonzeros;
}

std::size_t condensedFactorSparseTripletBytes(const CondensedFactor &factor) {
  const std::size_t matrixNnz = matrixNonzeros(factor.Abar);
  const std::size_t vectorNnz = vectorNonzeros(factor.bbar);
  const std::size_t matrixHeader =
      2u * sizeof(int) + sizeof(std::uint64_t);
  const std::size_t vectorHeader = sizeof(int) + sizeof(std::uint64_t);
  const std::size_t tripletBytes =
      matrixNnz * (2u * sizeof(int) + sizeof(double)) +
      vectorNnz * (sizeof(int) + sizeof(double));
  return sizeof(int) + sizeof(std::uint64_t) + sizeof(int) +
         keyBytes(factor.boundary_keys) + autoEncodingTagsBytes() +
         matrixHeader + vectorHeader + tripletBytes;
}

std::size_t condensedFactorSparseTripletNonzeros(
    const CondensedFactor &factor) {
  return matrixNonzeros(factor.Abar) + vectorNonzeros(factor.bbar);
}

std::size_t condensedFactorBytes(const CondensedFactor &factor) {
  return std::min(condensedFactorDenseBytes(factor),
                  condensedFactorSparseTripletBytes(factor));
}

std::size_t solutionMessageBytes(const CondensedFactor &factor, int blockDim) {
  return sizeof(int) + sizeof(std::uint64_t) + sizeof(int) +
         keyBytes(factor.boundary_keys) +
         factor.boundary_keys.size() * static_cast<std::size_t>(blockDim) *
             sizeof(double);
}

std::vector<PoseKey> collectInterfaceKeys(
    const std::vector<CondensedFactor> &condensedFactors,
    const std::vector<LinearFactorBlock> &crossFactors) {
  std::vector<PoseKey> keys;
  for (const auto &factor : condensedFactors) {
    keys.insert(keys.end(), factor.boundary_keys.begin(),
                factor.boundary_keys.end());
  }
  for (const auto &factor : crossFactors) {
    keys.insert(keys.end(), factor.keys.begin(), factor.keys.end());
  }
  return sortedUnique(std::move(keys));
}

void stackInterfaceSystem(
    const std::vector<CondensedFactor> &condensedFactors,
    const std::vector<LinearFactorBlock> &crossFactors, int blockDim,
    std::vector<PoseKey> *interfaceKeys, int *condensedRows, Matrix *A,
    Vector *b) {
  std::vector<LinearFactorBlock> factors;
  factors.reserve(condensedFactors.size() + crossFactors.size());
  if (condensedRows != nullptr) {
    *condensedRows = 0;
  }
  for (const auto &condensed : condensedFactors) {
    factors.push_back(condensedAsFactor(condensed));
    if (condensedRows != nullptr) {
      *condensedRows += condensed.Abar.rows();
    }
  }
  factors.insert(factors.end(), crossFactors.begin(), crossFactors.end());
  *interfaceKeys = collectInterfaceKeys(condensedFactors, crossFactors);
  stackFactors(factors, {}, *interfaceKeys, blockDim, A, b);
}

void stackInterfaceSystemSparse(
    const std::vector<CondensedFactor> &condensedFactors,
    const std::vector<LinearFactorBlock> &crossFactors, int blockDim,
    std::vector<PoseKey> *interfaceKeys, int *condensedRows,
    ColMajorSparseMatrix *A, Vector *b) {
  std::vector<LinearFactorBlock> factors;
  factors.reserve(condensedFactors.size() + crossFactors.size());
  if (condensedRows != nullptr) {
    *condensedRows = 0;
  }
  for (const auto &condensed : condensedFactors) {
    factors.push_back(condensedAsFactor(condensed));
    if (condensedRows != nullptr) {
      *condensedRows += condensed.Abar.rows();
    }
  }
  factors.insert(factors.end(), crossFactors.begin(), crossFactors.end());
  *interfaceKeys = collectInterfaceKeys(condensedFactors, crossFactors);
  stackFactorsSparse(factors, *interfaceKeys, blockDim, A, b);
}

Vector solveSparseLeastSquares(const ColMajorSparseMatrix &A, const Vector &b,
                               bool *usedFallback = nullptr) {
  if (usedFallback != nullptr) {
    *usedFallback = false;
  }
  if (A.cols() == 0) {
    return Vector::Zero(0);
  }
  Eigen::SPQR<ColMajorSparseMatrix> qr(A);
  if (qr.info() != Eigen::Success ||
      static_cast<int>(qr.rank()) < A.cols()) {
    if (usedFallback != nullptr) {
      *usedFallback = true;
    }
    const Matrix denseA = Matrix(A);
    return denseA.colPivHouseholderQr().solve(b);
  }
  return qr.solve(b);
}

std::map<PoseKey, Vector> splitSolutionByKey(
    const std::vector<PoseKey> &keys, const Vector &solution, int blockDim) {
  std::map<PoseKey, Vector> blocks;
  if (solution.rows() != static_cast<int>(keys.size()) * blockDim) {
    throw std::invalid_argument("TED-CCI solution vector has wrong size");
  }
  for (std::size_t i = 0; i < keys.size(); ++i) {
    blocks[keys[i]] = solution.segment(static_cast<int>(i) * blockDim,
                                       blockDim);
  }
  return blocks;
}

void insertBackSubstitutedBlocks(
    const BackSubstitutionCache &cache, const Vector &boundarySolution,
    int blockDim, std::map<PoseKey, Vector> *blocks) {
  const Vector interior = LocalQRCondensation::BackSubstitute(
      cache, boundarySolution);
  if (interior.rows() !=
      static_cast<int>(cache.interior_keys.size()) * blockDim) {
    throw std::invalid_argument("TED-CCI back-substitution returned bad size");
  }
  for (std::size_t i = 0; i < cache.interior_keys.size(); ++i) {
    (*blocks)[cache.interior_keys[i]] =
        interior.segment(static_cast<int>(i) * blockDim, blockDim);
  }
}

Matrix matrixFromVector(const Vector &x, int d) {
  return Eigen::Map<const Matrix>(x.data(), d, d);
}

Vector vectorizeRotationMultiRhsSolution(const std::vector<PoseKey> &keys,
                                         const Matrix &multiRhsSolution,
                                         int d) {
  if (multiRhsSolution.rows() != static_cast<int>(keys.size()) * d ||
      multiRhsSolution.cols() != d) {
    throw std::invalid_argument(
        "TED-CCI RIFT multi-RHS rotation solution has wrong size");
  }
  Vector vectorized(static_cast<int>(keys.size()) * d * d);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const Matrix rowWiseBlock =
        multiRhsSolution.block(static_cast<int>(i) * d, 0, d, d);
    const Matrix rotationBlock = rowWiseBlock.transpose();
    vectorized.segment(static_cast<int>(i) * d * d, d * d) =
        vectorizeMatrix(rotationBlock);
  }
  return vectorized;
}

std::map<PoseKey, Matrix> projectedRotationBlocks(
    int d, const std::map<PoseKey, Vector> &relaxedBlocks,
    const TEDCCIParams &params) {
  std::map<PoseKey, Matrix> rotations;
  const PoseKey anchor = anchorKey(params);
  for (const auto &entry : relaxedBlocks) {
    rotations[entry.first] = entry.first == anchor
                                 ? Matrix::Identity(d, d)
                                 : projectToRotationGroup(
                                       matrixFromVector(entry.second, d));
  }
  rotations[anchor] = Matrix::Identity(d, d);
  return rotations;
}

Vector boundarySolutionForCache(const BackSubstitutionCache &cache,
                                const std::map<PoseKey, Vector> &blocks,
                                int blockDim) {
  Vector x(static_cast<int>(cache.boundary_keys.size()) * blockDim);
  for (std::size_t i = 0; i < cache.boundary_keys.size(); ++i) {
    const auto block = blocks.find(cache.boundary_keys[i]);
    if (block == blocks.end()) {
      throw std::invalid_argument(
          "TED-CCI missing boundary block for back-substitution");
    }
    x.segment(static_cast<int>(i) * blockDim, blockDim) = block->second;
  }
  return x;
}

std::vector<PoseKey> removeAnchorKey(std::vector<PoseKey> keys,
                                     const TEDCCIParams &params) {
  const PoseKey anchor = anchorKey(params);
  keys.erase(std::remove(keys.begin(), keys.end(), anchor), keys.end());
  return keys;
}

std::vector<PoseKey> collectLinearFactorKeys(
    const std::vector<LinearFactorBlock> &factors) {
  std::vector<PoseKey> keys;
  for (const auto &factor : factors) {
    keys.insert(keys.end(), factor.keys.begin(), factor.keys.end());
  }
  return sortedUnique(std::move(keys));
}

bool robotInNode(const SeparatorNode &node, int robot) {
  return std::binary_search(node.robot_ids.begin(), node.robot_ids.end(),
                            robot);
}

bool factorInsideNode(const LinearFactorBlock &factor,
                      const SeparatorNode &node) {
  for (const PoseKey &key : factor.keys) {
    if (!robotInNode(node, key.robot_id)) {
      return false;
    }
  }
  return true;
}

bool factorTouchesOutsideNode(const LinearFactorBlock &factor,
                              const SeparatorNode &node) {
  bool hasInside = false;
  bool hasOutside = false;
  for (const PoseKey &key : factor.keys) {
    if (robotInNode(node, key.robot_id)) {
      hasInside = true;
    } else {
      hasOutside = true;
    }
  }
  return hasInside && hasOutside;
}

std::vector<PoseKey> separatorKeysForNode(
    const SeparatorNode &node, const std::vector<PoseKey> &nodeKeys,
    const std::vector<LinearFactorBlock> &crossFactors) {
  std::vector<PoseKey> separator;
  for (const PoseKey &key : nodeKeys) {
    for (const auto &factor : crossFactors) {
      if (std::find(factor.keys.begin(), factor.keys.end(), key) ==
          factor.keys.end()) {
        continue;
      }
      if (factorTouchesOutsideNode(factor, node)) {
        separator.push_back(key);
        break;
      }
    }
  }
  return sortedUnique(std::move(separator));
}

std::vector<PoseKey> subtractKeys(const std::vector<PoseKey> &keys,
                                  const std::vector<PoseKey> &remove) {
  std::vector<PoseKey> result;
  for (const PoseKey &key : keys) {
    if (!containsKey(remove, key)) {
      result.push_back(key);
    }
  }
  return result;
}

struct HierarchyCache {
  int node_id = -1;
  BackSubstitutionCache cache;
  std::vector<HierarchyCache> children;
};

struct HierarchyBuildResult {
  CondensedFactor factor;
  HierarchyCache cache;
  std::vector<LinearFactorBlock> root_factors;
};

void recoverHierarchy(const HierarchyCache &cache, int blockDim,
                      std::map<PoseKey, Vector> *blocks) {
  if (!cache.cache.interior_keys.empty()) {
    const Vector boundary =
        boundarySolutionForCache(cache.cache, *blocks, blockDim);
    insertBackSubstitutedBlocks(cache.cache, boundary, blockDim, blocks);
  }
  for (const auto &child : cache.children) {
    recoverHierarchy(child, blockDim, blocks);
  }
}

void hashCombine(std::uint64_t *seed, std::uint64_t value) {
  *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

void hashInt(std::uint64_t *seed, int value) {
  hashCombine(seed, static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(value)));
}

void hashDouble(std::uint64_t *seed, double value) {
  hashCombine(seed, static_cast<std::uint64_t>(std::hash<double>{}(value)));
}

void hashKeys(std::uint64_t *seed, const std::vector<PoseKey> &keys) {
  hashCombine(seed, static_cast<std::uint64_t>(keys.size()));
  for (const PoseKey &key : keys) {
    hashInt(seed, key.robot_id);
    hashInt(seed, key.pose_id);
  }
}

void hashMatrix(std::uint64_t *seed, const Matrix &A) {
  hashInt(seed, static_cast<int>(A.rows()));
  hashInt(seed, static_cast<int>(A.cols()));
  for (int col = 0; col < A.cols(); ++col) {
    for (int row = 0; row < A.rows(); ++row) {
      hashDouble(seed, A(row, col));
    }
  }
}

void hashVector(std::uint64_t *seed, const Vector &x) {
  hashInt(seed, static_cast<int>(x.rows()));
  for (int row = 0; row < x.rows(); ++row) {
    hashDouble(seed, x(row));
  }
}

std::pair<int, int> canonicalLink(int a, int b) {
  return std::minmax(a, b);
}

}  // namespace

std::uint64_t ComputeFactorMessageChecksum(const FactorMessage &msg) {
  std::uint64_t seed = 1469598103934665603ULL;
  hashInt(&seed, msg.sender_robot_id);
  hashInt(&seed, msg.receiver_robot_id);
  hashCombine(&seed, msg.epoch);
  hashCombine(&seed, msg.topology_epoch);
  hashInt(&seed, static_cast<int>(msg.factor_type));
  hashInt(&seed, msg.is_cross_factor ? 1 : 0);
  hashInt(&seed, msg.is_component_factor ? 1 : 0);
  hashInt(&seed, static_cast<int>(msg.component_robot_ids.size()));
  for (const int robot : msg.component_robot_ids) {
    hashInt(&seed, robot);
  }
  hashKeys(&seed, msg.keys);
  hashMatrix(&seed, msg.A);
  hashVector(&seed, msg.b);
  return seed;
}

std::uint64_t ComputeSolutionMessageChecksum(const SolutionMessage &msg) {
  std::uint64_t seed = 1099511628211ULL;
  hashInt(&seed, msg.sender_robot_id);
  hashInt(&seed, msg.receiver_robot_id);
  hashCombine(&seed, msg.epoch);
  hashCombine(&seed, msg.topology_epoch);
  hashInt(&seed, static_cast<int>(msg.factor_type));
  hashInt(&seed, msg.is_component_solution ? 1 : 0);
  hashInt(&seed, static_cast<int>(msg.component_robot_ids.size()));
  for (const int robot : msg.component_robot_ids) {
    hashInt(&seed, robot);
  }
  hashKeys(&seed, msg.keys);
  hashVector(&seed, msg.x);
  return seed;
}

std::vector<LinearFactorBlock> CCIFactorBuilder::BuildRotationFactors(
    int dimension, const std::vector<RelativeSEMeasurement> &measurements,
    const TEDCCIParams &params) {
  const int d = dimension;
  const int d2 = d * d;
  std::vector<LinearFactorBlock> factors;
  factors.reserve(measurements.size());

  for (const auto &measurement : measurements) {
    validateDimension(dimension, measurement);
    const PoseKey first = keyForFirstEndpoint(measurement);
    const PoseKey second = keyForSecondEndpoint(measurement);
    LinearFactorBlock factor;
    factor.keys = nonAnchorKeys(first, second, params);
    factor.A = Matrix::Zero(d2, static_cast<int>(factor.keys.size()) * d2);
    factor.b = Vector::Zero(d2);

    const double scale = effectiveScale(measurement.kappa, measurement.weight,
                                        params.use_measurement_weight,
                                        "rotation precision");
    if (isAnchor(first, params)) {
      factor.b -= vectorizeMatrix(scale * Matrix::Identity(d, d) *
                                  measurement.R);
    } else {
      addRotationProductBlock(
          factor.A, 0, keyOffset(factor.keys, first, d2), measurement.R,
          scale);
    }

    if (isAnchor(second, params)) {
      factor.b += vectorizeMatrix(scale * Matrix::Identity(d, d));
    } else {
      addRotationIdentityBlock(
          factor.A, 0, keyOffset(factor.keys, second, d2), d, -scale);
    }

    factors.push_back(std::move(factor));
  }

  return factors;
}

std::vector<LinearFactorBlock> CCIFactorBuilder::BuildTranslationFactors(
    int dimension, const std::vector<RelativeSEMeasurement> &measurements,
    const std::map<PoseKey, Matrix> &projected_rotations,
    const TEDCCIParams &params) {
  const int d = dimension;
  std::vector<LinearFactorBlock> factors;
  factors.reserve(measurements.size());

  for (const auto &measurement : measurements) {
    validateDimension(dimension, measurement);
    const PoseKey first = keyForFirstEndpoint(measurement);
    const PoseKey second = keyForSecondEndpoint(measurement);
    const auto rotation = projected_rotations.find(first);
    if (rotation == projected_rotations.end()) {
      throw std::invalid_argument(
          "TED-CCI translation factor missing first-endpoint rotation");
    }
    if (rotation->second.rows() != d || rotation->second.cols() != d) {
      throw std::invalid_argument(
          "TED-CCI translation factor rotation has wrong dimension");
    }

    LinearFactorBlock factor;
    factor.keys = nonAnchorKeys(first, second, params);
    factor.A = Matrix::Zero(d, static_cast<int>(factor.keys.size()) * d);
    const double scale = effectiveScale(measurement.tau, measurement.weight,
                                        params.use_measurement_weight,
                                        "translation precision");
    factor.b = scale * rotation->second * measurement.t;

    if (!isAnchor(first, params)) {
      const int offset = keyOffset(factor.keys, first, d);
      factor.A.block(0, offset, d, d) -= scale * Matrix::Identity(d, d);
    }
    if (!isAnchor(second, params)) {
      const int offset = keyOffset(factor.keys, second, d);
      factor.A.block(0, offset, d, d) += scale * Matrix::Identity(d, d);
    }

    factors.push_back(std::move(factor));
  }

  return factors;
}

TEDCCIPartition BoundarySelector::SelectBoundaryVariables(
    int local_robot_id, const std::vector<PoseKey> &local_poses,
    const std::vector<RelativeSEMeasurement> &local_measurements,
    const std::vector<RelativeSEMeasurement> &shared_measurements,
    const TEDCCIParams &params) {
  (void)local_measurements;

  TEDCCIPartition partition;
  partition.local_robot_id = local_robot_id;
  partition.local_poses = sortedUnique(local_poses);
  std::vector<PoseKey> boundary;

  const PoseKey anchor = anchorKey(params);
  if (anchor.robot_id == local_robot_id &&
      containsKey(partition.local_poses, anchor)) {
    boundary.push_back(anchor);
  }

  for (const auto &measurement : shared_measurements) {
    addBoundaryForSharedEndpoint(local_robot_id, partition.local_poses,
                                 measurement, &boundary,
                                 &partition.neighbor_boundary_poses);
  }

  boundary = sortedUnique(std::move(boundary));
  if (boundary.empty() && params.enable_component_fallback &&
      !partition.local_poses.empty()) {
    boundary.push_back(partition.local_poses.front());
  }
  partition.boundary_poses = boundary;

  for (auto &entry : partition.neighbor_boundary_poses) {
    entry.second = sortedUnique(std::move(entry.second));
  }

  for (const PoseKey &pose : partition.local_poses) {
    if (!containsKey(partition.boundary_poses, pose)) {
      partition.interior_poses.push_back(pose);
    }
  }

  return partition;
}

void LocalQRCondensation::Condense(
    const std::vector<LinearFactorBlock> &factors,
    const std::vector<PoseKey> &interior_keys,
    const std::vector<PoseKey> &boundary_keys, int block_dim,
    CondensedFactor *condensed_factor, BackSubstitutionCache *cache) {
  TEDCCIParams params;
  params.condensation_backend = TEDCCIBackend::DENSE_HOUSEHOLDER;
  Condense(factors, interior_keys, boundary_keys, block_dim, condensed_factor,
           cache, params);
}

void LocalQRCondensation::Condense(
    const std::vector<LinearFactorBlock> &factors,
    const std::vector<PoseKey> &interior_keys,
    const std::vector<PoseKey> &boundary_keys, int block_dim,
    CondensedFactor *condensed_factor, BackSubstitutionCache *cache,
    const TEDCCIParams &params) {
  if (condensed_factor == nullptr || cache == nullptr) {
    throw std::invalid_argument("TED-CCI condensation outputs must be non-null");
  }

  const std::vector<PoseKey> interior = sortedUnique(interior_keys);
  const std::vector<PoseKey> boundary = sortedUnique(boundary_keys);
  const int interiorDim = static_cast<int>(interior.size()) * block_dim;
  const int boundaryDim = static_cast<int>(boundary.size()) * block_dim;

  condensed_factor->boundary_keys = boundary;
  cache->interior_keys = interior;
  cache->boundary_keys = boundary;
  cache->interior_column_permutation.clear();
  cache->condensation_backend = resolveBackend(params.condensation_backend);

  if (cache->condensation_backend == TEDCCIBackend::SPARSE_SPQR) {
    ColMajorSparseMatrix AI;
    Matrix boundaryAndRhs;
    stackFactorsSparseSplit(factors, interior, boundary, block_dim, &AI,
                            &boundaryAndRhs);
    if (interiorDim > AI.rows()) {
      throw std::invalid_argument(
          "TED-CCI local QR condensation has more interior columns than rows");
    }
    if (interiorDim == 0) {
      condensed_factor->Abar = boundaryAndRhs.leftCols(boundaryDim);
      condensed_factor->bbar = boundaryAndRhs.col(boundaryDim);
      cache->R_II.resize(0, 0);
      cache->R_IB.resize(0, boundaryDim);
      cache->d_I.resize(0);
      return;
    }

    Eigen::SPQR<ColMajorSparseMatrix> qr(AI);
    if (qr.info() != Eigen::Success ||
        static_cast<int>(qr.rank()) < interiorDim) {
      throw std::invalid_argument(
          "TED-CCI local SPQR condensation requires full-rank interior block");
    }

    const Matrix transformed = qr.matrixQ().transpose() * boundaryAndRhs;
    const ColMajorSparseMatrix sparseR = qr.matrixR();
    const Matrix denseR = Matrix(sparseR);
    if (denseR.rows() < interiorDim || denseR.cols() < interiorDim ||
        transformed.rows() < AI.rows()) {
      throw std::invalid_argument(
          "TED-CCI local SPQR condensation returned inconsistent factors");
    }

    cache->R_II = denseR.topLeftCorner(interiorDim, interiorDim);
    cache->R_IB = transformed.block(0, 0, interiorDim, boundaryDim);
    cache->d_I = transformed.block(0, boundaryDim, interiorDim, 1);

    const auto permutation = qr.colsPermutation();
    cache->interior_column_permutation.resize(interiorDim);
    for (int i = 0; i < interiorDim; ++i) {
      cache->interior_column_permutation[i] =
          static_cast<int>(permutation.indices()[i]);
    }

    const int condensedRows = AI.rows() - interiorDim;
    condensed_factor->Abar =
        transformed.block(interiorDim, 0, condensedRows, boundaryDim);
    condensed_factor->bbar =
        transformed.block(interiorDim, boundaryDim, condensedRows, 1);
    return;
  }

  Matrix A;
  Vector b;
  stackFactors(factors, interior, boundary, block_dim, &A, &b);
  if (interiorDim > A.rows()) {
    throw std::invalid_argument(
        "TED-CCI local QR condensation has more interior columns than rows");
  }

  if (interiorDim == 0) {
    condensed_factor->Abar = A.leftCols(boundaryDim);
    condensed_factor->bbar = b;
    cache->R_II.resize(0, 0);
    cache->R_IB.resize(0, boundaryDim);
    cache->d_I.resize(0);
    return;
  }

  const Matrix AI = A.leftCols(interiorDim);
  const int augmentedCols = interiorDim + boundaryDim + 1;
  Matrix augmented(A.rows(), augmentedCols);
  augmented.leftCols(interiorDim + boundaryDim) = A;
  augmented.col(augmentedCols - 1) = b;

  Eigen::HouseholderQR<Matrix> qr(AI);
  const Matrix Q =
      qr.householderQ() * Matrix::Identity(A.rows(), A.rows());
  const Matrix transformed = Q.transpose() * augmented;

  cache->R_II = transformed.topLeftCorner(interiorDim, interiorDim);
  for (int i = 0; i < interiorDim; ++i) {
    if (std::abs(cache->R_II(i, i)) < 1e-12) {
      throw std::invalid_argument(
          "TED-CCI local QR condensation requires full-rank interior block");
    }
  }
  cache->R_IB =
      transformed.block(0, interiorDim, interiorDim, boundaryDim);
  cache->d_I = transformed.block(0, interiorDim + boundaryDim, interiorDim, 1);

  const int condensedRows = A.rows() - interiorDim;
  condensed_factor->Abar =
      transformed.block(interiorDim, interiorDim, condensedRows, boundaryDim);
  condensed_factor->bbar =
      transformed.block(interiorDim, interiorDim + boundaryDim, condensedRows,
                        1);
  cache->condensation_backend = TEDCCIBackend::DENSE_HOUSEHOLDER;
}

Vector LocalQRCondensation::BackSubstitute(
    const BackSubstitutionCache &cache, const Vector &boundary_solution) {
  if (cache.R_II.rows() == 0) {
    return Vector::Zero(0);
  }
  if (cache.R_IB.cols() != boundary_solution.rows()) {
    throw std::invalid_argument(
        "TED-CCI back-substitution boundary solution has wrong size");
  }
  const Vector rhs = cache.d_I - cache.R_IB * boundary_solution;
  const Vector pivoted =
      cache.R_II.triangularView<Eigen::Upper>().solve(rhs);
  if (cache.interior_column_permutation.empty()) {
    return pivoted;
  }
  if (static_cast<int>(cache.interior_column_permutation.size()) !=
      pivoted.rows()) {
    throw std::invalid_argument(
        "TED-CCI back-substitution has invalid SPQR column permutation");
  }
  Vector original = Vector::Zero(pivoted.rows());
  for (int i = 0; i < pivoted.rows(); ++i) {
    const int originalIndex = cache.interior_column_permutation[i];
    if (originalIndex < 0 || originalIndex >= pivoted.rows()) {
      throw std::invalid_argument(
          "TED-CCI back-substitution SPQR permutation index out of range");
    }
    original(originalIndex) = pivoted(i);
  }
  return original;
}

Vector InterfaceDirectSolver::Solve(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
  const TEDCCIParams &params, TEDCCIStats *stats) {
  if (block_dim <= 0) {
    throw std::invalid_argument(
        "TED-CCI interface block dimension must be positive");
  }
  if (!params.use_square_root_qr && !params.allow_normal_equation_debug_path) {
    throw std::invalid_argument(
        "TED-CCI interface direct solver requires square-root QR mode");
  }

  std::vector<PoseKey> interfaceKeys;
  int condensedRows = 0;
  Vector b;
  Vector solution;
  const TEDCCIBackend backend = resolveBackend(params.interface_backend);
  bool usedSparseFallback = false;
  if (backend == TEDCCIBackend::SPARSE_SPQR) {
    ColMajorSparseMatrix A;
    stackInterfaceSystemSparse(condensed_factors, cross_factors, block_dim,
                               &interfaceKeys, &condensedRows, &A, &b);
    solution = solveSparseLeastSquares(A, b, &usedSparseFallback);
  } else {
    Matrix A;
    stackInterfaceSystem(condensed_factors, cross_factors, block_dim,
                         &interfaceKeys, &condensedRows, &A, &b);
    solution = Vector::Zero(A.cols());
    if (A.cols() > 0) {
      solution = A.colPivHouseholderQr().solve(b);
    }
  }

  if (stats != nullptr) {
    stats->effective_mode = CCIInitMode::TED_CCI_SR_DIRECT;
    stats->effective_interface_backend = backend;
    stats->spqr_rank_deficient_fallbacks = usedSparseFallback ? 1 : 0;
    stats->peak_interface_cols =
        std::max(stats->peak_interface_cols,
                 static_cast<int>(solution.rows()));
    stats->num_interface_vars = static_cast<int>(interfaceKeys.size());
    stats->num_condensed_rows = condensedRows;
    stats->max_separator_size = static_cast<int>(interfaceKeys.size());
    stats->num_factor_messages =
        static_cast<int>(condensed_factors.size());
    stats->num_solution_messages =
        static_cast<int>(condensed_factors.size());
    stats->bytes_sent_upward = 0;
    stats->bytes_sent_downward = 0;
    stats->factor_dense_bytes_upward = 0;
    stats->factor_sparse_triplet_bytes_upward = 0;
    stats->factor_sparse_triplet_nonzeros = 0;
    for (const auto &factor : condensed_factors) {
      stats->bytes_sent_upward += condensedFactorBytes(factor);
      stats->factor_dense_bytes_upward += condensedFactorDenseBytes(factor);
      stats->factor_sparse_triplet_bytes_upward +=
          condensedFactorSparseTripletBytes(factor);
      stats->factor_sparse_triplet_nonzeros +=
          condensedFactorSparseTripletNonzeros(factor);
      stats->bytes_sent_downward += solutionMessageBytes(factor, block_dim);
    }
  }

  return solution;
}

Vector InterfaceAsyncDDSolver::Solve(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const TEDCCIParams &params, TEDCCIStats *stats) {
  if (block_dim <= 0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD block dimension must be positive");
  }
  if (params.async_dd_max_iters < 0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD max iterations must be nonnegative");
  }
  if (!std::isfinite(params.async_dd_rel_tol) ||
      params.async_dd_rel_tol < 0.0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD relative tolerance must be finite and nonnegative");
  }

  std::vector<PoseKey> interfaceKeys;
  int condensedRows = 0;
  Matrix A;
  Vector b;
  stackInterfaceSystem(condensed_factors, cross_factors, block_dim,
                       &interfaceKeys, &condensedRows, &A, &b);

  Vector x = Vector::Zero(A.cols());
  const auto residualNorm = [&]() {
    return A.rows() == 0 ? 0.0 : (A * x - b).norm();
  };
  const double initialResidual = residualNorm();
  double finalResidual = initialResidual;
  bool converged = initialResidual == 0.0;
  int iterations = 0;
  int coarseCorrectionCalls = 0;
  const double residualGate =
      params.async_dd_rel_tol * std::max(1.0, initialResidual);

  for (int iter = 0; iter < params.async_dd_max_iters && !converged; ++iter) {
    for (std::size_t keyIndex = 0; keyIndex < interfaceKeys.size();
         ++keyIndex) {
      const int col = static_cast<int>(keyIndex) * block_dim;
      const Matrix Ablock = A.block(0, col, A.rows(), block_dim);
      if (Ablock.cols() == 0) {
        continue;
      }
      const Vector rhs = b - A * x + Ablock * x.segment(col, block_dim);
      x.segment(col, block_dim) = Ablock.colPivHouseholderQr().solve(rhs);
    }
    if (params.async_dd_enable_coarse_correction) {
      ++coarseCorrectionCalls;
    }
    ++iterations;
    finalResidual = residualNorm();
    converged = finalResidual <= residualGate;
  }

  if (stats != nullptr) {
    stats->num_interface_vars = static_cast<int>(interfaceKeys.size());
    stats->num_condensed_rows = condensedRows;
    stats->max_separator_size = static_cast<int>(interfaceKeys.size());
    stats->num_factor_messages =
        static_cast<int>(condensed_factors.size());
    stats->num_solution_messages =
        iterations * static_cast<int>(interfaceKeys.size());
    stats->bytes_sent_upward = 0;
    stats->factor_dense_bytes_upward = 0;
    stats->factor_sparse_triplet_bytes_upward = 0;
    stats->factor_sparse_triplet_nonzeros = 0;
    for (const auto &factor : condensed_factors) {
      stats->bytes_sent_upward += condensedFactorBytes(factor);
      stats->factor_dense_bytes_upward += condensedFactorDenseBytes(factor);
      stats->factor_sparse_triplet_bytes_upward +=
          condensedFactorSparseTripletBytes(factor);
      stats->factor_sparse_triplet_nonzeros +=
          condensedFactorSparseTripletNonzeros(factor);
    }
    stats->bytes_sent_downward =
        static_cast<std::size_t>(stats->num_solution_messages) *
        (2u * sizeof(int) +
         static_cast<std::size_t>(block_dim) * sizeof(double));
    stats->async_dd_iterations = iterations;
    stats->async_dd_converged = converged;
    stats->async_dd_initial_residual = initialResidual;
    stats->async_dd_final_residual = finalResidual;
    stats->async_dd_coarse_correction_calls = coarseCorrectionCalls;
  }

  return x;
}

Vector InterfaceAsyncDDSolver::SolveWithSimulatedNetwork(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const TEDCCIParams &params, int max_delay_rounds,
    bool reverse_delivery_order, TEDCCIStats *stats) {
  if (block_dim <= 0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD block dimension must be positive");
  }
  if (params.async_dd_max_iters < 0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD max iterations must be nonnegative");
  }
  if (!std::isfinite(params.async_dd_rel_tol) ||
      params.async_dd_rel_tol < 0.0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD relative tolerance must be finite and nonnegative");
  }
  if (max_delay_rounds < 0) {
    throw std::invalid_argument(
        "TED-CCI Async-DD simulated delay must be nonnegative");
  }

  std::vector<PoseKey> interfaceKeys;
  int condensedRows = 0;
  Matrix A;
  Vector b;
  stackInterfaceSystem(condensed_factors, cross_factors, block_dim,
                       &interfaceKeys, &condensedRows, &A, &b);

  const int numBlocks = static_cast<int>(interfaceKeys.size());
  Vector ownerValues = Vector::Zero(A.cols());
  std::vector<std::vector<Vector>> receiverCaches(
      numBlocks, std::vector<Vector>(numBlocks, Vector::Zero(block_dim)));
  std::vector<std::vector<int>> receivedIteration(
      numBlocks, std::vector<int>(numBlocks, -1));

  struct BlockUpdateMessage {
    int sender = -1;
    int receiver = -1;
    int delivery_round = 0;
    int iteration = 0;
    Vector value;
  };
  std::vector<BlockUpdateMessage> pendingMessages;
  std::size_t sentMessages = 0;

  const auto residualNorm = [&]() {
    return A.rows() == 0 ? 0.0 : (A * ownerValues - b).norm();
  };
  const auto receiverView = [&](int receiver) {
    Vector view = Vector::Zero(A.cols());
    for (int block = 0; block < numBlocks; ++block) {
      const int col = block * block_dim;
      if (block == receiver) {
        view.segment(col, block_dim) =
            ownerValues.segment(col, block_dim);
      } else {
        view.segment(col, block_dim) = receiverCaches[receiver][block];
      }
    }
    return view;
  };
  const auto deliverMessages = [&](int round) {
    std::vector<BlockUpdateMessage> due;
    std::vector<BlockUpdateMessage> remaining;
    for (const auto &message : pendingMessages) {
      if (message.delivery_round <= round) {
        due.push_back(message);
      } else {
        remaining.push_back(message);
      }
    }
    if (reverse_delivery_order) {
      std::reverse(due.begin(), due.end());
    }
    for (const auto &message : due) {
      if (message.receiver < 0 || message.receiver >= numBlocks ||
          message.sender < 0 || message.sender >= numBlocks) {
        continue;
      }
      if (message.iteration >=
          receivedIteration[message.receiver][message.sender]) {
        receiverCaches[message.receiver][message.sender] = message.value;
        receivedIteration[message.receiver][message.sender] =
            message.iteration;
      }
    }
    pendingMessages = std::move(remaining);
  };

  const double initialResidual = residualNorm();
  double finalResidual = initialResidual;
  bool converged = initialResidual == 0.0;
  int iterations = 0;
  int coarseCorrectionCalls = 0;
  const double residualGate =
      params.async_dd_rel_tol * std::max(1.0, initialResidual);

  for (int round = 0; round < params.async_dd_max_iters && !converged;
       ++round) {
    deliverMessages(round);
    for (int receiver = 0; receiver < numBlocks; ++receiver) {
      const int col = receiver * block_dim;
      const Matrix Ablock = A.block(0, col, A.rows(), block_dim);
      Vector view = receiverView(receiver);
      const Vector rhs =
          b - A * view + Ablock * view.segment(col, block_dim);
      const Vector update = Ablock.colPivHouseholderQr().solve(rhs);
      ownerValues.segment(col, block_dim) = update;

      for (int other = 0; other < numBlocks; ++other) {
        if (other == receiver) {
          continue;
        }
        BlockUpdateMessage message;
        message.sender = receiver;
        message.receiver = other;
        const int delayCycle = max_delay_rounds + 1;
        const int delay = delayCycle == 0
                              ? 0
                              : (round + receiver + other) % delayCycle;
        message.delivery_round = round + delay;
        message.iteration = round * numBlocks + receiver;
        message.value = update;
        pendingMessages.push_back(std::move(message));
        ++sentMessages;
      }
    }
    if (params.async_dd_enable_coarse_correction) {
      ++coarseCorrectionCalls;
    }
    ++iterations;
    finalResidual = residualNorm();
    converged = finalResidual <= residualGate;
  }

  if (stats != nullptr) {
    stats->num_interface_vars = numBlocks;
    stats->num_condensed_rows = condensedRows;
    stats->max_separator_size = numBlocks;
    stats->num_factor_messages =
        static_cast<int>(condensed_factors.size());
    stats->num_solution_messages = static_cast<int>(sentMessages);
    stats->bytes_sent_upward = 0;
    for (const auto &factor : condensed_factors) {
      stats->bytes_sent_upward += condensedFactorBytes(factor);
    }
    stats->bytes_sent_downward =
        sentMessages *
        (2u * sizeof(int) +
         static_cast<std::size_t>(block_dim) * sizeof(double));
    stats->async_dd_iterations = iterations;
    stats->async_dd_converged = converged;
    stats->async_dd_initial_residual = initialResidual;
    stats->async_dd_final_residual = finalResidual;
    stats->async_dd_coarse_correction_calls = coarseCorrectionCalls;
  }

  return ownerValues;
}

static Vector solveInterfaceForMode(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params, TEDCCIStats *stats) {
  CCIInitMode selectedMode = params.mode;
  if (selectedMode == CCIInitMode::TED_CCI_SR_AUTO) {
    const int interfaceVars = static_cast<int>(
        collectInterfaceKeys(condensed_factors, cross_factors).size());
    selectedMode = interfaceVars <= params.direct_interface_threshold
                       ? CCIInitMode::TED_CCI_SR_DIRECT
                       : CCIInitMode::TED_CCI_SR_HIERARCHICAL;
  }

  if (selectedMode == CCIInitMode::TED_CCI_ASYNC_DD) {
    if (stats != nullptr) {
      stats->effective_mode = selectedMode;
      stats->effective_interface_backend =
          resolveBackend(params.interface_backend);
    }
    return InterfaceAsyncDDSolver::Solve(condensed_factors, cross_factors,
                                         block_dim, params, stats);
  }
  if (selectedMode == CCIInitMode::TED_CCI_RIFT_IF) {
    if (stats != nullptr) {
      stats->effective_mode = selectedMode;
    }
    return SolveTEDInterfaceWithRIFTExact(condensed_factors, cross_factors,
                                          block_dim, partitions, params,
                                          stats);
  }
  if (selectedMode == CCIInitMode::TED_CCI_SR_HIERARCHICAL) {
    const SeparatorTree tree =
        SeparatorTree::BuildFromRobotGraph(partitions, params);
    return tree.SolveHierarchical(condensed_factors, cross_factors, block_dim,
                                  params, stats);
  }
  return InterfaceDirectSolver::Solve(condensed_factors, cross_factors,
                                      block_dim, params, stats);
}

SeparatorTree SeparatorTree::BuildFromRobotGraph(
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params) {
  (void)params;

  std::vector<int> robotIds;
  robotIds.reserve(partitions.size());
  for (const auto &partition : partitions) {
    robotIds.push_back(partition.local_robot_id);
  }
  std::sort(robotIds.begin(), robotIds.end());
  robotIds.erase(std::unique(robotIds.begin(), robotIds.end()),
                 robotIds.end());
  if (robotIds.empty()) {
    throw std::invalid_argument("TED-CCI separator tree requires robots");
  }

  SeparatorTree tree;
  std::function<int(std::size_t, std::size_t, int)> build =
      [&](std::size_t begin, std::size_t end, int parent) -> int {
    const int nodeId = static_cast<int>(tree.nodes_.size());
    tree.nodes_.push_back(SeparatorNode{});
    SeparatorNode &node = tree.nodes_.back();
    node.node_id = nodeId;
    node.parent_id = parent;
    node.robot_ids.assign(robotIds.begin() + static_cast<std::ptrdiff_t>(begin),
                          robotIds.begin() + static_cast<std::ptrdiff_t>(end));
    node.leader_robot_id =
        node.robot_ids.empty() ? -1 : node.robot_ids.front();
    if (end - begin > 1) {
      const std::size_t mid = begin + (end - begin) / 2;
      const int left = build(begin, mid, nodeId);
      const int right = build(mid, end, nodeId);
      tree.nodes_[nodeId].children = {left, right};
    }
    return nodeId;
  };
  tree.root_node_id_ = build(0, robotIds.size(), -1);
  return tree;
}

SeparatorFailureRecoveryPlan SeparatorTree::PlanParentNodeFailureRecovery(
    int failed_node_id) const {
  if (failed_node_id < 0 ||
      failed_node_id >= static_cast<int>(nodes_.size())) {
    throw std::invalid_argument(
        "TED-CCI separator recovery failed node id is invalid");
  }
  const SeparatorNode &failed =
      nodes_.at(static_cast<std::size_t>(failed_node_id));
  if (failed.children.empty()) {
    throw std::invalid_argument(
        "TED-CCI separator recovery requires a parent separator node");
  }
  if (failed.leader_robot_id < 0) {
    throw std::invalid_argument(
        "TED-CCI separator recovery requires an elected leader");
  }

  int replacement = -1;
  for (const int robot : failed.robot_ids) {
    if (robot != failed.leader_robot_id) {
      replacement = robot;
      break;
    }
  }
  if (replacement < 0) {
    throw std::invalid_argument(
        "TED-CCI separator recovery has no alternate leader robot");
  }

  SeparatorFailureRecoveryPlan plan;
  plan.failed_node_id = failed_node_id;
  plan.replacement_robot_id = replacement;
  for (const int childId : failed.children) {
    const SeparatorNode &child =
        nodes_.at(static_cast<std::size_t>(childId));
    if (child.leader_robot_id < 0 ||
        child.leader_robot_id == replacement) {
      continue;
    }
    SeparatorResendRequest request;
    request.sender_node_id = childId;
    request.sender_robot_id = child.leader_robot_id;
    request.receiver_node_id = failed_node_id;
    request.receiver_robot_id = replacement;
    plan.resend_requests.push_back(request);
  }
  return plan;
}

std::vector<FactorMessage> SeparatorTree::MakeRecoveryFactorMessages(
    const SeparatorFailureRecoveryPlan &plan,
    const std::map<int, CondensedFactor> &latest_node_factors,
    std::uint64_t epoch, std::uint64_t topology_epoch) const {
  std::vector<FactorMessage> messages;
  messages.reserve(plan.resend_requests.size());
  for (const auto &request : plan.resend_requests) {
    const auto factorIt = latest_node_factors.find(request.sender_node_id);
    if (factorIt == latest_node_factors.end()) {
      continue;
    }
    const CondensedFactor &factor = factorIt->second;
    FactorMessage msg;
    msg.sender_robot_id = request.sender_robot_id;
    msg.receiver_robot_id = request.receiver_robot_id;
    msg.epoch = epoch;
    msg.topology_epoch = topology_epoch;
    msg.factor_type = factor.factor_type;
    msg.is_cross_factor = false;
    msg.is_component_factor = false;
    msg.keys = factor.boundary_keys;
    msg.A = factor.Abar;
    msg.b = factor.bbar;
    msg.checksum = ComputeFactorMessageChecksum(msg);
    messages.push_back(std::move(msg));
  }
  return messages;
}

SeparatorTreeRuntime::SeparatorTreeRuntime(const SeparatorTree &tree)
    : nodes_(tree.nodes()) {}

std::vector<SeparatorNodeOwnership> SeparatorTreeRuntime::ElectOwners(
    const std::set<int> &live_robot_ids,
    std::uint64_t topology_epoch) const {
  std::vector<SeparatorNodeOwnership> ownership;
  ownership.reserve(nodes_.size());
  for (const SeparatorNode &node : nodes_) {
    SeparatorNodeOwnership entry;
    entry.node_id = node.node_id;
    entry.topology_epoch = topology_epoch;
    for (const int robot : node.robot_ids) {
      if (live_robot_ids.count(robot) > 0) {
        entry.owner_robot_id = robot;
        break;
      }
    }
    ownership.push_back(entry);
  }
  return ownership;
}

int SeparatorTreeRuntime::ParentOwnerForChildNode(
    int child_node_id,
    const std::vector<SeparatorNodeOwnership> &ownership) const {
  if (child_node_id < 0 ||
      child_node_id >= static_cast<int>(nodes_.size())) {
    throw std::invalid_argument(
        "TED-CCI separator runtime child node id is invalid");
  }
  const int parentId =
      nodes_.at(static_cast<std::size_t>(child_node_id)).parent_id;
  if (parentId < 0) {
    return -1;
  }
  for (const SeparatorNodeOwnership &entry : ownership) {
    if (entry.node_id == parentId) {
      return entry.owner_robot_id;
    }
  }
  throw std::invalid_argument(
      "TED-CCI separator runtime ownership is missing parent node");
}

SeparatorFailureRecoveryPlan
SeparatorTreeRuntime::PlanParentOwnerChangeResend(
    const std::vector<SeparatorNodeOwnership> &previous_ownership,
    const std::vector<SeparatorNodeOwnership> &current_ownership,
    int parent_node_id) const {
  if (parent_node_id < 0 ||
      parent_node_id >= static_cast<int>(nodes_.size())) {
    throw std::invalid_argument(
        "TED-CCI separator runtime parent node id is invalid");
  }
  const SeparatorNode &parent =
      nodes_.at(static_cast<std::size_t>(parent_node_id));
  if (parent.children.empty()) {
    throw std::invalid_argument(
        "TED-CCI separator runtime resend requires a parent node");
  }

  auto findOwnership =
      [](const std::vector<SeparatorNodeOwnership> &ownership,
         int nodeId) -> const SeparatorNodeOwnership * {
    for (const SeparatorNodeOwnership &entry : ownership) {
      if (entry.node_id == nodeId) {
        return &entry;
      }
    }
    return nullptr;
  };

  const SeparatorNodeOwnership *previous =
      findOwnership(previous_ownership, parent_node_id);
  const SeparatorNodeOwnership *current =
      findOwnership(current_ownership, parent_node_id);
  if (previous == nullptr || current == nullptr) {
    throw std::invalid_argument(
        "TED-CCI separator runtime ownership is missing parent node");
  }

  SeparatorFailureRecoveryPlan plan;
  plan.failed_node_id = parent_node_id;
  plan.replacement_robot_id = current->owner_robot_id;
  plan.topology_epoch = current->topology_epoch;
  if (current->owner_robot_id < 0 ||
      current->owner_robot_id == previous->owner_robot_id) {
    return plan;
  }

  for (const int childId : parent.children) {
    const SeparatorNodeOwnership *child =
        findOwnership(current_ownership, childId);
    if (child == nullptr) {
      throw std::invalid_argument(
          "TED-CCI separator runtime ownership is missing child node");
    }
    if (child->owner_robot_id < 0 ||
        child->owner_robot_id == current->owner_robot_id) {
      continue;
    }
    SeparatorResendRequest request;
    request.sender_node_id = childId;
    request.sender_robot_id = child->owner_robot_id;
    request.receiver_node_id = parent_node_id;
    request.receiver_robot_id = current->owner_robot_id;
    plan.resend_requests.push_back(request);
  }
  return plan;
}

SeparatorTreeMessageController::SeparatorTreeMessageController(
    const SeparatorTree &tree)
    : tree_(tree), runtime_(tree_) {}

std::vector<SeparatorNodeOwnership>
SeparatorTreeMessageController::ElectOwners(
    const std::set<int> &live_robot_ids,
    std::uint64_t topology_epoch) const {
  return runtime_.ElectOwners(live_robot_ids, topology_epoch);
}

void SeparatorTreeMessageController::UpdateLatestNodeFactor(
    int node_id, const CondensedFactor &factor) {
  if (node_id < 0) {
    throw std::invalid_argument(
        "TED-CCI separator message controller node id is invalid");
  }
  const auto existing = latest_node_factors_.find(node_id);
  if (existing == latest_node_factors_.end() ||
      factor.epoch >= existing->second.epoch) {
    latest_node_factors_[node_id] = factor;
  }
}

std::vector<FactorMessage>
SeparatorTreeMessageController::MakeOwnerChangeRecoveryMessages(
    const std::vector<SeparatorNodeOwnership> &previous_ownership,
    const std::vector<SeparatorNodeOwnership> &current_ownership,
    int parent_node_id, std::uint64_t epoch) const {
  const SeparatorFailureRecoveryPlan plan =
      runtime_.PlanParentOwnerChangeResend(previous_ownership,
                                           current_ownership,
                                           parent_node_id);
  return tree_.MakeRecoveryFactorMessages(plan, latest_node_factors_, epoch,
                                          plan.topology_epoch);
}

std::size_t SeparatorTreeMessageController::SendOwnerChangeRecoveryMessages(
    const std::vector<SeparatorNodeOwnership> &previous_ownership,
    const std::vector<SeparatorNodeOwnership> &current_ownership,
    int parent_node_id, std::uint64_t epoch,
    AsyncFactorExchange *exchange) const {
  if (exchange == nullptr) {
    throw std::invalid_argument(
        "TED-CCI separator message controller requires an exchange");
  }
  const std::vector<FactorMessage> messages =
      MakeOwnerChangeRecoveryMessages(previous_ownership, current_ownership,
                                      parent_node_id, epoch);
  for (const FactorMessage &message : messages) {
    exchange->SendFactor(message);
  }
  return messages.size();
}

Vector SeparatorTree::SolveHierarchical(
    const std::vector<CondensedFactor> &leaf_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    TEDCCIStats *stats) const {
  TEDCCIParams params;
  params.condensation_backend = TEDCCIBackend::DENSE_HOUSEHOLDER;
  params.interface_backend = TEDCCIBackend::DENSE_HOUSEHOLDER;
  return SolveHierarchical(leaf_factors, cross_factors, block_dim, params,
                           stats);
}

Vector SeparatorTree::SolveHierarchical(
    const std::vector<CondensedFactor> &leaf_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const TEDCCIParams &params, TEDCCIStats *stats) const {
  if (root_node_id_ < 0 || root_node_id_ >= static_cast<int>(nodes_.size())) {
    throw std::invalid_argument("TED-CCI separator tree is empty");
  }
  if (block_dim <= 0) {
    throw std::invalid_argument(
        "TED-CCI separator tree block dimension must be positive");
  }

  std::map<int, std::vector<CondensedFactor>> leafFactorsByRobot;
  for (const auto &factor : leaf_factors) {
    leafFactorsByRobot[factor.owner_robot_id].push_back(factor);
  }

  std::map<int, std::vector<LinearFactorBlock>> crossByNode;
  std::function<int(int, const LinearFactorBlock &)> ownerNode =
      [&](int nodeId, const LinearFactorBlock &factor) -> int {
    const SeparatorNode &node = nodes_.at(static_cast<std::size_t>(nodeId));
    for (const int childId : node.children) {
      if (factorInsideNode(factor, nodes_.at(static_cast<std::size_t>(childId)))) {
        return ownerNode(childId, factor);
      }
    }
    return nodeId;
  };
  for (const auto &factor : cross_factors) {
    if (!factorInsideNode(factor, nodes_.at(static_cast<std::size_t>(
                                   root_node_id_)))) {
      throw std::invalid_argument(
          "TED-CCI cross factor references robot outside separator tree");
    }
    crossByNode[ownerNode(root_node_id_, factor)].push_back(factor);
  }

  int maxSeparatorSize = 0;
  std::size_t upwardBytes = 0;
  std::size_t downwardBytes = 0;
  std::size_t denseFactorBytes = 0;
  std::size_t sparseTripletFactorBytes = 0;
  std::size_t sparseTripletFactorNonzeros = 0;
  int factorMessages = 0;

  std::function<HierarchyBuildResult(int)> buildUpward =
      [&](int nodeId) -> HierarchyBuildResult {
    const SeparatorNode &node = nodes_.at(static_cast<std::size_t>(nodeId));
    HierarchyBuildResult result;
    result.cache.node_id = nodeId;

    std::vector<LinearFactorBlock> factors;
    if (node.children.empty()) {
      const int robot = node.robot_ids.empty() ? -1 : node.robot_ids.front();
      for (const auto &factor : leafFactorsByRobot[robot]) {
        factors.push_back(condensedAsFactor(factor));
      }
    } else {
      for (const int childId : node.children) {
        HierarchyBuildResult child = buildUpward(childId);
        result.cache.children.push_back(std::move(child.cache));
        factors.push_back(condensedAsFactor(child.factor));
      }
    }
    const auto crossIt = crossByNode.find(nodeId);
    if (crossIt != crossByNode.end()) {
      factors.insert(factors.end(), crossIt->second.begin(),
                     crossIt->second.end());
    }

    if (nodeId == root_node_id_) {
      result.root_factors = std::move(factors);
      return result;
    }

    std::vector<PoseKey> nodeKeys = collectLinearFactorKeys(factors);
    std::vector<PoseKey> separator =
        separatorKeysForNode(node, nodeKeys, cross_factors);
    std::vector<PoseKey> interior = subtractKeys(nodeKeys, separator);
    maxSeparatorSize = std::max(maxSeparatorSize,
                                static_cast<int>(separator.size()));

    result.factor.owner_robot_id = nodeId;
    result.factor.factor_type = leaf_factors.empty()
                                    ? FactorType::ROTATION
                                    : leaf_factors.front().factor_type;
    LocalQRCondensation::Condense(factors, interior, separator, block_dim,
                                  &result.factor, &result.cache.cache,
                                  params);
    upwardBytes += condensedFactorBytes(result.factor);
    denseFactorBytes += condensedFactorDenseBytes(result.factor);
    sparseTripletFactorBytes +=
        condensedFactorSparseTripletBytes(result.factor);
    sparseTripletFactorNonzeros +=
        condensedFactorSparseTripletNonzeros(result.factor);
    downwardBytes += solutionMessageBytes(result.factor, block_dim);
    ++factorMessages;
    return result;
  };

  HierarchyBuildResult root = buildUpward(root_node_id_);
  std::vector<PoseKey> rootKeys = collectLinearFactorKeys(root.root_factors);
  maxSeparatorSize =
      std::max(maxSeparatorSize, static_cast<int>(rootKeys.size()));
  Vector rootB;
  Vector rootSolution;
  const TEDCCIBackend interfaceBackend =
      resolveBackend(params.interface_backend);
  bool usedSparseFallback = false;
  if (interfaceBackend == TEDCCIBackend::SPARSE_SPQR) {
    ColMajorSparseMatrix rootA;
    stackFactorsSparse(root.root_factors, rootKeys, block_dim, &rootA,
                       &rootB);
    rootSolution = solveSparseLeastSquares(rootA, rootB, &usedSparseFallback);
  } else {
    Matrix rootA;
    stackFactors(root.root_factors, {}, rootKeys, block_dim, &rootA, &rootB);
    rootSolution = Vector::Zero(rootA.cols());
    if (rootA.cols() > 0) {
      rootSolution = rootA.colPivHouseholderQr().solve(rootB);
    }
  }

  std::map<PoseKey, Vector> blocks =
      splitSolutionByKey(rootKeys, rootSolution, block_dim);
  recoverHierarchy(root.cache, block_dim, &blocks);

  const std::vector<PoseKey> outputKeys =
      collectInterfaceKeys(leaf_factors, cross_factors);
  Vector output(static_cast<int>(outputKeys.size()) * block_dim);
  for (std::size_t i = 0; i < outputKeys.size(); ++i) {
    const auto block = blocks.find(outputKeys[i]);
    if (block == blocks.end()) {
      throw std::runtime_error(
          "TED-CCI separator tree failed to recover interface key");
    }
    output.segment(static_cast<int>(i) * block_dim, block_dim) =
        block->second;
  }

  if (stats != nullptr) {
    stats->effective_mode = CCIInitMode::TED_CCI_SR_HIERARCHICAL;
    stats->effective_interface_backend = interfaceBackend;
    stats->spqr_rank_deficient_fallbacks = usedSparseFallback ? 1 : 0;
    stats->peak_interface_cols =
        std::max(stats->peak_interface_cols,
                 static_cast<int>(rootSolution.rows()));
    stats->num_interface_vars = static_cast<int>(outputKeys.size());
    stats->num_condensed_rows = 0;
    for (const auto &factor : leaf_factors) {
      stats->num_condensed_rows += factor.Abar.rows();
    }
    stats->max_separator_size = maxSeparatorSize;
    stats->num_factor_messages = factorMessages;
    stats->num_solution_messages = factorMessages;
    stats->bytes_sent_upward = upwardBytes;
    stats->bytes_sent_downward = downwardBytes;
    stats->factor_dense_bytes_upward = denseFactorBytes;
    stats->factor_sparse_triplet_bytes_upward = sparseTripletFactorBytes;
    stats->factor_sparse_triplet_nonzeros = sparseTripletFactorNonzeros;
  }
  return output;
}

bool InProcessAsyncFactorExchange::MessageKey::operator<(
    const MessageKey &other) const {
  if (sender_robot_id != other.sender_robot_id) {
    return sender_robot_id < other.sender_robot_id;
  }
  if (factor_type != other.factor_type) {
    return static_cast<int>(factor_type) < static_cast<int>(other.factor_type);
  }
  if (topology_epoch != other.topology_epoch) {
    return topology_epoch < other.topology_epoch;
  }
  if (is_cross_factor != other.is_cross_factor) {
    return is_cross_factor < other.is_cross_factor;
  }
  if (is_component_message != other.is_component_message) {
    return is_component_message < other.is_component_message;
  }
  if (component_robot_ids != other.component_robot_ids) {
    return component_robot_ids < other.component_robot_ids;
  }
  return keys < other.keys;
}

InProcessAsyncFactorExchange::InProcessAsyncFactorExchange(
    bool queue_when_link_down)
    : queue_when_link_down_(queue_when_link_down) {}

bool InProcessAsyncFactorExchange::IsLinkDown(int sender, int receiver) const {
  return down_links_.count(canonicalLink(sender, receiver)) > 0;
}

void InProcessAsyncFactorExchange::SendFactor(const FactorMessage &msg) {
  if (msg.checksum != ComputeFactorMessageChecksum(msg)) {
    return;
  }
  if (IsLinkDown(msg.sender_robot_id, msg.receiver_robot_id)) {
    if (queue_when_link_down_) {
      queued_factors_.push_back(msg);
    }
    return;
  }
  const MessageKey key{msg.sender_robot_id, msg.factor_type,
                       msg.topology_epoch,
                       msg.is_cross_factor, msg.is_component_factor,
                       msg.component_robot_ids, msg.keys};
  auto &inbox = factor_inbox_[msg.receiver_robot_id];
  const auto existing = inbox.find(key);
  if (existing == inbox.end() || msg.epoch > existing->second.epoch) {
    inbox[key] = msg;
  }
}

void InProcessAsyncFactorExchange::SendSolution(const SolutionMessage &msg) {
  if (msg.checksum != ComputeSolutionMessageChecksum(msg)) {
    return;
  }
  if (IsLinkDown(msg.sender_robot_id, msg.receiver_robot_id)) {
    if (queue_when_link_down_) {
      queued_solutions_.push_back(msg);
    }
    return;
  }
  const MessageKey key{msg.sender_robot_id, msg.factor_type,
                       msg.topology_epoch, false,
                       msg.is_component_solution, msg.component_robot_ids,
                       msg.keys};
  auto &inbox = solution_inbox_[msg.receiver_robot_id];
  const auto existing = inbox.find(key);
  if (existing == inbox.end() || msg.epoch > existing->second.epoch) {
    inbox[key] = msg;
  }
}

std::vector<FactorMessage> InProcessAsyncFactorExchange::PollFactorMessages(
    int receiver_robot_id) {
  std::vector<FactorMessage> messages;
  auto it = factor_inbox_.find(receiver_robot_id);
  if (it == factor_inbox_.end()) {
    return messages;
  }
  for (const auto &entry : it->second) {
    messages.push_back(entry.second);
  }
  factor_inbox_.erase(it);
  return messages;
}

std::vector<SolutionMessage>
InProcessAsyncFactorExchange::PollSolutionMessages(int receiver_robot_id) {
  std::vector<SolutionMessage> messages;
  auto it = solution_inbox_.find(receiver_robot_id);
  if (it == solution_inbox_.end()) {
    return messages;
  }
  for (const auto &entry : it->second) {
    messages.push_back(entry.second);
  }
  solution_inbox_.erase(it);
  return messages;
}

void InProcessAsyncFactorExchange::MarkLinkDown(int robot_a, int robot_b) {
  down_links_.insert(canonicalLink(robot_a, robot_b));
}

void InProcessAsyncFactorExchange::MarkLinkUp(int robot_a, int robot_b) {
  const auto link = canonicalLink(robot_a, robot_b);
  down_links_.erase(link);

  std::vector<FactorMessage> remainingFactors;
  for (const auto &msg : queued_factors_) {
    if (canonicalLink(msg.sender_robot_id, msg.receiver_robot_id) == link) {
      SendFactor(msg);
    } else {
      remainingFactors.push_back(msg);
    }
  }
  queued_factors_ = std::move(remainingFactors);

  std::vector<SolutionMessage> remainingSolutions;
  for (const auto &msg : queued_solutions_) {
    if (canonicalLink(msg.sender_robot_id, msg.receiver_robot_id) == link) {
      SendSolution(msg);
    } else {
      remainingSolutions.push_back(msg);
    }
  }
  queued_solutions_ = std::move(remainingSolutions);
}

TEDCCIOrchestrator::TEDCCIOrchestrator(int local_robot_id,
                                       const TEDCCIParams &params,
                                       AsyncFactorExchange *exchange)
    : local_robot_id_(local_robot_id), params_(params), exchange_(exchange) {
  if (exchange_ == nullptr) {
    throw std::invalid_argument("TED-CCI orchestrator requires an exchange");
  }
}

void TEDCCIOrchestrator::SetProblemMetadata(
    int dimension, int num_poses, const TEDCCIPartition &local_partition,
    const std::vector<int> &expected_robot_ids) {
  if (dimension != 2 && dimension != 3) {
    throw std::invalid_argument(
        "TED-CCI orchestrator supports only SE(2) or SE(3)");
  }
  if (num_poses <= 0) {
    throw std::invalid_argument(
        "TED-CCI orchestrator requires at least one pose");
  }
  if (local_partition.local_robot_id != local_robot_id_) {
    throw std::invalid_argument(
        "TED-CCI orchestrator partition robot id mismatch");
  }

  dimension_ = dimension;
  num_poses_ = num_poses;
  local_partition_ = local_partition;
  local_partition_.local_poses = sortedUnique(local_partition_.local_poses);
  local_partition_.boundary_poses =
      sortedUnique(local_partition_.boundary_poses);
  local_partition_.interior_poses =
      sortedUnique(local_partition_.interior_poses);
  expected_robot_ids_ = expected_robot_ids;
  expected_robot_ids_.push_back(local_robot_id_);
  std::sort(expected_robot_ids_.begin(), expected_robot_ids_.end());
  expected_robot_ids_.erase(
      std::unique(expected_robot_ids_.begin(), expected_robot_ids_.end()),
      expected_robot_ids_.end());
  metadata_set_ = true;
}

void TEDCCIOrchestrator::SetLocalMeasurements(
    const std::vector<RelativeSEMeasurement> &local_measurements,
    const std::vector<RelativeSEMeasurement> &shared_measurements) {
  local_measurements_ = local_measurements;
  shared_measurements_ = shared_measurements;
  measurements_set_ = true;
}

void TEDCCIOrchestrator::ValidateReady() const {
  if (!metadata_set_) {
    throw std::runtime_error("TED-CCI orchestrator metadata not set");
  }
  if (!measurements_set_) {
    throw std::runtime_error("TED-CCI orchestrator measurements not set");
  }
}

Matrix TEDCCIOrchestrator::ComputeLocalOnlyInitialization() const {
  const int d = dimension_;
  const int localCount = static_cast<int>(local_partition_.local_poses.size());
  Matrix fallback(d, localCount * (d + 1));
  if (localCount == 0) {
    fallback.resize(d, 0);
    return fallback;
  }

  std::map<int, unsigned> localIndexByPose;
  for (std::size_t idx = 0; idx < local_partition_.local_poses.size(); ++idx) {
    localIndexByPose[local_partition_.local_poses[idx].pose_id] =
        static_cast<unsigned>(idx);
  }

  std::vector<RelativeSEMeasurement> remapped;
  for (const auto &measurement : local_measurements_) {
    const auto first = localIndexByPose.find(static_cast<int>(measurement.p1));
    const auto second = localIndexByPose.find(static_cast<int>(measurement.p2));
    if (first == localIndexByPose.end() ||
        second == localIndexByPose.end()) {
      continue;
    }
    RelativeSEMeasurement copy = measurement;
    copy.r1 = 0;
    copy.r2 = 0;
    copy.p1 = first->second;
    copy.p2 = second->second;
    remapped.push_back(copy);
  }

  if (!remapped.empty()) {
    return chordalInitialization(d, static_cast<size_t>(localCount), remapped,
                                 params_.use_measurement_weight);
  }

  fallback.setZero();
  for (int pose = 0; pose < localCount; ++pose) {
    fallback.block(0, pose * (d + 1), d, d) = Matrix::Identity(d, d);
  }
  return fallback;
}

void TEDCCIOrchestrator::StartEpoch(std::uint64_t epoch) {
  ValidateReady();
  epoch_ = epoch;

  rotation_factor_sent_ = false;
  rotation_cross_factor_sent_ = false;
  rotation_solved_ = false;
  translation_factor_sent_ = false;
  translation_cross_factor_sent_ = false;
  translation_solved_ = false;
  local_only_available_ = false;
  component_available_ = false;
  global_available_ = false;
  component_rotation_solved_ = false;
  component_translation_factor_sent_ = false;
  component_translation_cross_factor_sent_ = false;
  component_translation_solved_ = false;

  local_only_solution_.resize(0, 0);
  component_solution_.resize(0, 0);
  global_solution_.resize(0, 0);
  rotation_factors_.clear();
  translation_factors_.clear();
  rotation_cross_factors_.clear();
  translation_cross_factors_.clear();
  relaxed_rotation_blocks_.clear();
  projected_rotations_.clear();
  translation_blocks_.clear();
  received_rotation_solution_blocks_.clear();
  received_translation_solution_blocks_.clear();
  rotation_solution_senders_.clear();
  translation_solution_senders_.clear();
  active_component_robot_ids_.clear();
  component_relaxed_rotation_blocks_.clear();
  component_projected_rotations_.clear();
  component_translation_blocks_.clear();
  component_translation_factors_.clear();
  component_translation_cross_factors_.clear();
  component_received_rotation_solution_blocks_.clear();
  component_received_translation_solution_blocks_.clear();
  component_rotation_solution_senders_.clear();
  component_translation_solution_senders_.clear();

  local_only_solution_ = ComputeLocalOnlyInitialization();
  component_solution_ = local_only_solution_;
  local_only_available_ = true;
  component_available_ = true;

  BuildAndBroadcastRotationFactor();
  BuildAndBroadcastRotationCrossFactor();
}

void TEDCCIOrchestrator::BuildAndBroadcastRotationFactor() {
  if (rotation_factor_sent_) {
    return;
  }
  const int blockDim = dimension_ * dimension_;
  const auto factors = CCIFactorBuilder::BuildRotationFactors(
      dimension_, local_measurements_, params_);
  local_rotation_factor_ = CondensedFactor{};
  local_rotation_factor_.owner_robot_id = local_robot_id_;
  local_rotation_factor_.epoch = epoch_;
  local_rotation_factor_.factor_type = FactorType::ROTATION;
  rotation_cache_ = BackSubstitutionCache{};
  rotation_cache_.factor_type = FactorType::ROTATION;
  LocalQRCondensation::Condense(
      factors, local_partition_.interior_poses,
      removeAnchorKey(local_partition_.boundary_poses, params_), blockDim,
      &local_rotation_factor_, &rotation_cache_, params_);
  rotation_factors_[local_robot_id_] = local_rotation_factor_;
  BroadcastFactor(local_rotation_factor_);
  rotation_factor_sent_ = true;
}

LinearFactorBlock TEDCCIOrchestrator::BuildOwnedCrossFactor(
    FactorType factor_type) const {
  return BuildOwnedCrossFactorForRobots(
      factor_type, expected_robot_ids_, projected_rotations_);
}

LinearFactorBlock TEDCCIOrchestrator::BuildOwnedCrossFactorForRobots(
    FactorType factor_type, const std::vector<int> &component_robots,
    const std::map<PoseKey, Matrix> &rotations) const {
  std::set<int> componentSet(component_robots.begin(),
                             component_robots.end());
  std::vector<RelativeSEMeasurement> owned;
  for (const auto &measurement : shared_measurements_) {
    if (measurement.r1 == measurement.r2) {
      continue;
    }
    if (componentSet.count(static_cast<int>(measurement.r1)) == 0 ||
        componentSet.count(static_cast<int>(measurement.r2)) == 0) {
      continue;
    }
    const int owner = std::min(static_cast<int>(measurement.r1),
                               static_cast<int>(measurement.r2));
    if (owner == local_robot_id_) {
      owned.push_back(measurement);
    }
  }

  const int blockDim = factor_type == FactorType::ROTATION
                           ? dimension_ * dimension_
                           : dimension_;
  std::vector<LinearFactorBlock> factors;
  if (factor_type == FactorType::ROTATION) {
    factors = CCIFactorBuilder::BuildRotationFactors(dimension_, owned,
                                                     params_);
  } else {
    factors = CCIFactorBuilder::BuildTranslationFactors(
        dimension_, owned, rotations, params_);
  }

  LinearFactorBlock combined;
  combined.keys = collectLinearFactorKeys(factors);
  if (factors.empty()) {
    combined.A.resize(0, 0);
    combined.b.resize(0);
    return combined;
  }
  stackFactors(factors, {}, combined.keys, blockDim, &combined.A,
               &combined.b);
  return combined;
}

void TEDCCIOrchestrator::BuildAndBroadcastRotationCrossFactor() {
  if (rotation_cross_factor_sent_) {
    return;
  }
  local_rotation_cross_factor_ =
      BuildOwnedCrossFactor(FactorType::ROTATION);
  rotation_cross_factors_[local_robot_id_] = local_rotation_cross_factor_;
  BroadcastCrossFactor(FactorType::ROTATION, local_rotation_cross_factor_);
  rotation_cross_factor_sent_ = true;
}

void TEDCCIOrchestrator::BuildAndBroadcastTranslationCrossFactor() {
  if (translation_cross_factor_sent_) {
    return;
  }
  local_translation_cross_factor_ =
      BuildOwnedCrossFactor(FactorType::TRANSLATION);
  translation_cross_factors_[local_robot_id_] =
      local_translation_cross_factor_;
  BroadcastCrossFactor(FactorType::TRANSLATION,
                       local_translation_cross_factor_);
  translation_cross_factor_sent_ = true;
}

void TEDCCIOrchestrator::BroadcastFactor(const CondensedFactor &factor) {
  for (const int receiver : expected_robot_ids_) {
    if (receiver == local_robot_id_) {
      continue;
    }
    FactorMessage msg;
    msg.sender_robot_id = local_robot_id_;
    msg.receiver_robot_id = receiver;
    msg.epoch = factor.epoch;
    msg.factor_type = factor.factor_type;
    msg.is_cross_factor = false;
    msg.keys = factor.boundary_keys;
    msg.A = factor.Abar;
    msg.b = factor.bbar;
    msg.checksum = ComputeFactorMessageChecksum(msg);
    exchange_->SendFactor(msg);
  }
}

void TEDCCIOrchestrator::BroadcastCrossFactor(
    FactorType factor_type, const LinearFactorBlock &factor) {
  for (const int receiver : expected_robot_ids_) {
    if (receiver == local_robot_id_) {
      continue;
    }
    FactorMessage msg;
    msg.sender_robot_id = local_robot_id_;
    msg.receiver_robot_id = receiver;
    msg.epoch = epoch_;
    msg.factor_type = factor_type;
    msg.is_cross_factor = true;
    msg.keys = factor.keys;
    msg.A = factor.A;
    msg.b = factor.b;
    msg.checksum = ComputeFactorMessageChecksum(msg);
    exchange_->SendFactor(msg);
  }
}

void TEDCCIOrchestrator::BroadcastComponentFactor(
    const CondensedFactor &factor, const std::vector<int> &component_robots) {
  for (const int receiver : component_robots) {
    if (receiver == local_robot_id_) {
      continue;
    }
    FactorMessage msg;
    msg.sender_robot_id = local_robot_id_;
    msg.receiver_robot_id = receiver;
    msg.epoch = epoch_;
    msg.factor_type = factor.factor_type;
    msg.is_cross_factor = false;
    msg.is_component_factor = true;
    msg.component_robot_ids = component_robots;
    msg.keys = factor.boundary_keys;
    msg.A = factor.Abar;
    msg.b = factor.bbar;
    msg.checksum = ComputeFactorMessageChecksum(msg);
    exchange_->SendFactor(msg);
  }
}

void TEDCCIOrchestrator::BroadcastComponentCrossFactor(
    FactorType factor_type, const LinearFactorBlock &factor,
    const std::vector<int> &component_robots) {
  for (const int receiver : component_robots) {
    if (receiver == local_robot_id_) {
      continue;
    }
    FactorMessage msg;
    msg.sender_robot_id = local_robot_id_;
    msg.receiver_robot_id = receiver;
    msg.epoch = epoch_;
    msg.factor_type = factor_type;
    msg.is_cross_factor = true;
    msg.is_component_factor = true;
    msg.component_robot_ids = component_robots;
    msg.keys = factor.keys;
    msg.A = factor.A;
    msg.b = factor.b;
    msg.checksum = ComputeFactorMessageChecksum(msg);
    exchange_->SendFactor(msg);
  }
}

bool TEDCCIOrchestrator::HasAllFactors(FactorType factor_type) const {
  const auto &factors = factor_type == FactorType::ROTATION
                            ? rotation_factors_
                            : translation_factors_;
  for (const int robot : expected_robot_ids_) {
    if (factors.count(robot) == 0) {
      return false;
    }
  }
  return true;
}

bool TEDCCIOrchestrator::HasAllCrossFactors(FactorType factor_type) const {
  const auto &factors = factor_type == FactorType::ROTATION
                            ? rotation_cross_factors_
                            : translation_cross_factors_;
  for (const int robot : expected_robot_ids_) {
    if (factors.count(robot) == 0) {
      return false;
    }
  }
  return true;
}

std::vector<CondensedFactor> TEDCCIOrchestrator::FactorsForType(
    FactorType factor_type) const {
  const auto &factorMap = factor_type == FactorType::ROTATION
                              ? rotation_factors_
                              : translation_factors_;
  std::vector<CondensedFactor> factors;
  factors.reserve(expected_robot_ids_.size());
  for (const int robot : expected_robot_ids_) {
    const auto entry = factorMap.find(robot);
    if (entry == factorMap.end()) {
      throw std::runtime_error("TED-CCI missing expected factor");
    }
    factors.push_back(entry->second);
  }
  return factors;
}

std::vector<LinearFactorBlock> TEDCCIOrchestrator::CrossFactorsForType(
    FactorType factor_type) const {
  const auto &factorMap = factor_type == FactorType::ROTATION
                              ? rotation_cross_factors_
                              : translation_cross_factors_;
  std::vector<LinearFactorBlock> factors;
  factors.reserve(expected_robot_ids_.size());
  for (const int robot : expected_robot_ids_) {
    const auto entry = factorMap.find(robot);
    if (entry == factorMap.end()) {
      throw std::runtime_error("TED-CCI missing expected cross factor");
    }
    factors.push_back(entry->second);
  }
  return factors;
}

std::vector<int> TEDCCIOrchestrator::ReachableComponentRobotIds() const {
  std::vector<int> robots;
  for (const int robot : expected_robot_ids_) {
    if (rotation_factors_.count(robot) > 0 &&
        rotation_cross_factors_.count(robot) > 0) {
      robots.push_back(robot);
    }
  }
  if (std::find(robots.begin(), robots.end(), local_robot_id_) ==
      robots.end()) {
    robots.push_back(local_robot_id_);
  }
  std::sort(robots.begin(), robots.end());
  robots.erase(std::unique(robots.begin(), robots.end()), robots.end());
  return robots;
}

bool TEDCCIOrchestrator::FactorKeysInsideRobots(
    const LinearFactorBlock &factor,
    const std::vector<int> &component_robots) const {
  std::set<int> componentSet(component_robots.begin(),
                             component_robots.end());
  for (const PoseKey &key : factor.keys) {
    if (componentSet.count(key.robot_id) == 0) {
      return false;
    }
  }
  return true;
}

std::vector<CondensedFactor> TEDCCIOrchestrator::ComponentRotationFactors()
    const {
  std::vector<CondensedFactor> factors;
  factors.reserve(active_component_robot_ids_.size());
  for (const int robot : active_component_robot_ids_) {
    const auto entry = rotation_factors_.find(robot);
    if (entry == rotation_factors_.end()) {
      throw std::runtime_error("TED-CCI missing component rotation factor");
    }
    factors.push_back(entry->second);
  }
  return factors;
}

std::vector<LinearFactorBlock>
TEDCCIOrchestrator::ComponentRotationCrossFactors() const {
  std::vector<LinearFactorBlock> factors;
  factors.reserve(active_component_robot_ids_.size());
  for (const int robot : active_component_robot_ids_) {
    const auto entry = rotation_cross_factors_.find(robot);
    if (entry == rotation_cross_factors_.end()) {
      throw std::runtime_error(
          "TED-CCI missing component rotation cross factor");
    }
    if (FactorKeysInsideRobots(entry->second, active_component_robot_ids_)) {
      factors.push_back(entry->second);
    }
  }
  return factors;
}

bool TEDCCIOrchestrator::HasComponentTranslationFactors() const {
  for (const int robot : active_component_robot_ids_) {
    const auto entry = component_translation_factors_.find(robot);
    if (entry == component_translation_factors_.end() ||
        entry->second.first != active_component_robot_ids_) {
      return false;
    }
  }
  return true;
}

bool TEDCCIOrchestrator::HasComponentTranslationCrossFactors() const {
  for (const int robot : active_component_robot_ids_) {
    const auto entry = component_translation_cross_factors_.find(robot);
    if (entry == component_translation_cross_factors_.end() ||
        entry->second.first != active_component_robot_ids_) {
      return false;
    }
  }
  return true;
}

bool TEDCCIOrchestrator::HasComponentSolutionSenders(
    FactorType factor_type) const {
  const auto &senders = factor_type == FactorType::ROTATION
                            ? component_rotation_solution_senders_
                            : component_translation_solution_senders_;
  for (const int robot : active_component_robot_ids_) {
    const auto entry = senders.find(robot);
    if (entry == senders.end() ||
        entry->second != active_component_robot_ids_) {
      return false;
    }
  }
  return true;
}

std::vector<CondensedFactor>
TEDCCIOrchestrator::ComponentTranslationFactors() const {
  std::vector<CondensedFactor> factors;
  factors.reserve(active_component_robot_ids_.size());
  for (const int robot : active_component_robot_ids_) {
    const auto entry = component_translation_factors_.find(robot);
    if (entry == component_translation_factors_.end() ||
        entry->second.first != active_component_robot_ids_) {
      throw std::runtime_error("TED-CCI missing component translation factor");
    }
    factors.push_back(entry->second.second);
  }
  return factors;
}

std::vector<LinearFactorBlock>
TEDCCIOrchestrator::ComponentTranslationCrossFactors() const {
  std::vector<LinearFactorBlock> factors;
  factors.reserve(active_component_robot_ids_.size());
  for (const int robot : active_component_robot_ids_) {
    const auto entry = component_translation_cross_factors_.find(robot);
    if (entry == component_translation_cross_factors_.end() ||
        entry->second.first != active_component_robot_ids_) {
      throw std::runtime_error(
          "TED-CCI missing component translation cross factor");
    }
    if (FactorKeysInsideRobots(entry->second.second,
                               active_component_robot_ids_)) {
      factors.push_back(entry->second.second);
    }
  }
  return factors;
}

void TEDCCIOrchestrator::TrySolveRotationAndBroadcastTranslation() {
  if (rotation_solved_ || !HasAllFactors(FactorType::ROTATION) ||
      !HasAllCrossFactors(FactorType::ROTATION)) {
    return;
  }

  const int blockDim = dimension_ * dimension_;
  const std::vector<CondensedFactor> condensed =
      FactorsForType(FactorType::ROTATION);
  const auto crossFactors = CrossFactorsForType(FactorType::ROTATION);
  const std::vector<PoseKey> interfaceKeys =
      collectInterfaceKeys(condensed, crossFactors);
  const Vector interfaceSolution = InterfaceDirectSolver::Solve(
      condensed, crossFactors, blockDim, params_);
  relaxed_rotation_blocks_ =
      splitSolutionByKey(interfaceKeys, interfaceSolution, blockDim);
  relaxed_rotation_blocks_[anchorKey(params_)] =
      vectorizeMatrix(Matrix::Identity(dimension_, dimension_));

  const Vector boundary = boundarySolutionForCache(
      rotation_cache_, relaxed_rotation_blocks_, blockDim);
  insertBackSubstitutedBlocks(rotation_cache_, boundary, blockDim,
                              &relaxed_rotation_blocks_);
  projected_rotations_ =
      projectedRotationBlocks(dimension_, relaxed_rotation_blocks_, params_);
  rotation_solved_ = true;

  Vector localRotations(static_cast<int>(local_partition_.local_poses.size()) *
                        blockDim);
  for (std::size_t idx = 0; idx < local_partition_.local_poses.size(); ++idx) {
    const PoseKey key = local_partition_.local_poses[idx];
    localRotations.segment(static_cast<int>(idx) * blockDim, blockDim) =
        vectorizeMatrix(projected_rotations_.at(key));
    received_rotation_solution_blocks_[key] =
        localRotations.segment(static_cast<int>(idx) * blockDim, blockDim);
  }
  rotation_solution_senders_.insert(local_robot_id_);
  BroadcastLocalSolution(FactorType::ROTATION, local_partition_.local_poses,
                         localRotations);

  const auto translationFactors = CCIFactorBuilder::BuildTranslationFactors(
      dimension_, local_measurements_, projected_rotations_, params_);
  local_translation_factor_ = CondensedFactor{};
  local_translation_factor_.owner_robot_id = local_robot_id_;
  local_translation_factor_.epoch = epoch_;
  local_translation_factor_.factor_type = FactorType::TRANSLATION;
  translation_cache_ = BackSubstitutionCache{};
  translation_cache_.factor_type = FactorType::TRANSLATION;
  LocalQRCondensation::Condense(
      translationFactors, local_partition_.interior_poses,
      removeAnchorKey(local_partition_.boundary_poses, params_), dimension_,
      &local_translation_factor_, &translation_cache_, params_);
  translation_factors_[local_robot_id_] = local_translation_factor_;
  BroadcastFactor(local_translation_factor_);
  translation_factor_sent_ = true;
  BuildAndBroadcastTranslationCrossFactor();
}

void TEDCCIOrchestrator::BroadcastLocalSolution(
    FactorType factor_type, const std::vector<PoseKey> &keys,
    const Vector &x) {
  for (const int receiver : expected_robot_ids_) {
    if (receiver == local_robot_id_) {
      continue;
    }
    SolutionMessage msg;
    msg.sender_robot_id = local_robot_id_;
    msg.receiver_robot_id = receiver;
    msg.epoch = epoch_;
    msg.factor_type = factor_type;
    msg.keys = keys;
    msg.x = x;
    msg.checksum = ComputeSolutionMessageChecksum(msg);
    exchange_->SendSolution(msg);
  }
}

void TEDCCIOrchestrator::BroadcastComponentLocalSolution(
    FactorType factor_type, const std::vector<PoseKey> &keys, const Vector &x,
    const std::vector<int> &component_robots) {
  for (const int receiver : component_robots) {
    if (receiver == local_robot_id_) {
      continue;
    }
    SolutionMessage msg;
    msg.sender_robot_id = local_robot_id_;
    msg.receiver_robot_id = receiver;
    msg.epoch = epoch_;
    msg.factor_type = factor_type;
    msg.is_component_solution = true;
    msg.component_robot_ids = component_robots;
    msg.keys = keys;
    msg.x = x;
    msg.checksum = ComputeSolutionMessageChecksum(msg);
    exchange_->SendSolution(msg);
  }
}

void TEDCCIOrchestrator::TrySolveTranslationAndBroadcastSolution() {
  if (translation_solved_ || !rotation_solved_ ||
      !HasAllFactors(FactorType::TRANSLATION) ||
      !HasAllCrossFactors(FactorType::TRANSLATION)) {
    return;
  }

  const std::vector<CondensedFactor> condensed =
      FactorsForType(FactorType::TRANSLATION);
  const auto crossFactors = CrossFactorsForType(FactorType::TRANSLATION);
  const std::vector<PoseKey> interfaceKeys =
      collectInterfaceKeys(condensed, crossFactors);
  const Vector interfaceSolution = InterfaceDirectSolver::Solve(
      condensed, crossFactors, dimension_, params_);
  translation_blocks_ =
      splitSolutionByKey(interfaceKeys, interfaceSolution, dimension_);
  translation_blocks_[anchorKey(params_)] = Vector::Zero(dimension_);

  const Vector boundary =
      boundarySolutionForCache(translation_cache_, translation_blocks_,
                               dimension_);
  insertBackSubstitutedBlocks(translation_cache_, boundary, dimension_,
                              &translation_blocks_);
  translation_solved_ = true;

  Vector localTranslations(
      static_cast<int>(local_partition_.local_poses.size()) * dimension_);
  for (std::size_t idx = 0; idx < local_partition_.local_poses.size(); ++idx) {
    const PoseKey key = local_partition_.local_poses[idx];
    localTranslations.segment(static_cast<int>(idx) * dimension_,
                              dimension_) = translation_blocks_.at(key);
    received_translation_solution_blocks_[key] =
        localTranslations.segment(static_cast<int>(idx) * dimension_,
                                  dimension_);
  }
  translation_solution_senders_.insert(local_robot_id_);
  BroadcastLocalSolution(FactorType::TRANSLATION, local_partition_.local_poses,
                         localTranslations);
  TryAssembleGlobalSolution();
}

void TEDCCIOrchestrator::TrySolveComponentAndBroadcast() {
  if (global_available_) {
    return;
  }
  const std::vector<int> reachable = ReachableComponentRobotIds();
  if (reachable.size() <= 1 ||
      reachable.size() == expected_robot_ids_.size()) {
    return;
  }

  if (active_component_robot_ids_ != reachable) {
    active_component_robot_ids_ = reachable;
    component_rotation_solved_ = false;
    component_translation_factor_sent_ = false;
    component_translation_cross_factor_sent_ = false;
    component_translation_solved_ = false;
    component_relaxed_rotation_blocks_.clear();
    component_projected_rotations_.clear();
    component_translation_blocks_.clear();

    for (auto it = component_translation_factors_.begin();
         it != component_translation_factors_.end();) {
      if (it->second.first == active_component_robot_ids_) {
        ++it;
      } else {
        it = component_translation_factors_.erase(it);
      }
    }
    for (auto it = component_translation_cross_factors_.begin();
         it != component_translation_cross_factors_.end();) {
      if (it->second.first == active_component_robot_ids_) {
        ++it;
      } else {
        it = component_translation_cross_factors_.erase(it);
      }
    }
    for (auto it = component_rotation_solution_senders_.begin();
         it != component_rotation_solution_senders_.end();) {
      if (it->second == active_component_robot_ids_) {
        ++it;
      } else {
        it = component_rotation_solution_senders_.erase(it);
      }
    }
    for (auto it = component_translation_solution_senders_.begin();
         it != component_translation_solution_senders_.end();) {
      if (it->second == active_component_robot_ids_) {
        ++it;
      } else {
        it = component_translation_solution_senders_.erase(it);
      }
    }
  }

  if (!component_rotation_solved_) {
    const int blockDim = dimension_ * dimension_;
    const std::vector<CondensedFactor> condensed =
        ComponentRotationFactors();
    const auto crossFactors = ComponentRotationCrossFactors();
    const std::vector<PoseKey> interfaceKeys =
        collectInterfaceKeys(condensed, crossFactors);
    const Vector interfaceSolution = InterfaceDirectSolver::Solve(
        condensed, crossFactors, blockDim, params_);
    component_relaxed_rotation_blocks_ =
        splitSolutionByKey(interfaceKeys, interfaceSolution, blockDim);
    component_relaxed_rotation_blocks_[anchorKey(params_)] =
        vectorizeMatrix(Matrix::Identity(dimension_, dimension_));

    const Vector boundary = boundarySolutionForCache(
        rotation_cache_, component_relaxed_rotation_blocks_, blockDim);
    insertBackSubstitutedBlocks(rotation_cache_, boundary, blockDim,
                                &component_relaxed_rotation_blocks_);
    component_projected_rotations_ = projectedRotationBlocks(
        dimension_, component_relaxed_rotation_blocks_, params_);
    component_rotation_solved_ = true;

    Vector localRotations(
        static_cast<int>(local_partition_.local_poses.size()) * blockDim);
    for (std::size_t idx = 0; idx < local_partition_.local_poses.size();
         ++idx) {
      const PoseKey key = local_partition_.local_poses[idx];
      localRotations.segment(static_cast<int>(idx) * blockDim, blockDim) =
          vectorizeMatrix(component_projected_rotations_.at(key));
      component_received_rotation_solution_blocks_[key] =
          localRotations.segment(static_cast<int>(idx) * blockDim, blockDim);
    }
    component_rotation_solution_senders_[local_robot_id_] =
        active_component_robot_ids_;
    BroadcastComponentLocalSolution(FactorType::ROTATION,
                                    local_partition_.local_poses,
                                    localRotations,
                                    active_component_robot_ids_);

    const auto translationFactors = CCIFactorBuilder::BuildTranslationFactors(
        dimension_, local_measurements_, component_projected_rotations_,
        params_);
    local_component_translation_factor_ = CondensedFactor{};
    local_component_translation_factor_.owner_robot_id = local_robot_id_;
    local_component_translation_factor_.epoch = epoch_;
    local_component_translation_factor_.factor_type =
        FactorType::TRANSLATION;
    component_translation_cache_ = BackSubstitutionCache{};
    component_translation_cache_.factor_type = FactorType::TRANSLATION;
    LocalQRCondensation::Condense(
        translationFactors, local_partition_.interior_poses,
        removeAnchorKey(local_partition_.boundary_poses, params_), dimension_,
        &local_component_translation_factor_, &component_translation_cache_,
        params_);
    component_translation_factors_[local_robot_id_] =
        std::make_pair(active_component_robot_ids_,
                       local_component_translation_factor_);
    BroadcastComponentFactor(local_component_translation_factor_,
                             active_component_robot_ids_);
    component_translation_factor_sent_ = true;

    local_component_translation_cross_factor_ =
        BuildOwnedCrossFactorForRobots(FactorType::TRANSLATION,
                                       active_component_robot_ids_,
                                       component_projected_rotations_);
    component_translation_cross_factors_[local_robot_id_] =
        std::make_pair(active_component_robot_ids_,
                       local_component_translation_cross_factor_);
    BroadcastComponentCrossFactor(FactorType::TRANSLATION,
                                  local_component_translation_cross_factor_,
                                  active_component_robot_ids_);
    component_translation_cross_factor_sent_ = true;
  }

  if (!component_translation_solved_ && component_rotation_solved_ &&
      HasComponentTranslationFactors() &&
      HasComponentTranslationCrossFactors()) {
    const std::vector<CondensedFactor> condensed =
        ComponentTranslationFactors();
    const auto crossFactors = ComponentTranslationCrossFactors();
    const std::vector<PoseKey> interfaceKeys =
        collectInterfaceKeys(condensed, crossFactors);
    const Vector interfaceSolution = InterfaceDirectSolver::Solve(
        condensed, crossFactors, dimension_, params_);
    component_translation_blocks_ =
        splitSolutionByKey(interfaceKeys, interfaceSolution, dimension_);
    component_translation_blocks_[anchorKey(params_)] =
        Vector::Zero(dimension_);

    const Vector boundary = boundarySolutionForCache(
        component_translation_cache_, component_translation_blocks_,
        dimension_);
    insertBackSubstitutedBlocks(component_translation_cache_, boundary,
                                dimension_, &component_translation_blocks_);
    component_translation_solved_ = true;

    Vector localTranslations(
        static_cast<int>(local_partition_.local_poses.size()) * dimension_);
    for (std::size_t idx = 0; idx < local_partition_.local_poses.size();
         ++idx) {
      const PoseKey key = local_partition_.local_poses[idx];
      localTranslations.segment(static_cast<int>(idx) * dimension_,
                                dimension_) =
          component_translation_blocks_.at(key);
      component_received_translation_solution_blocks_[key] =
          localTranslations.segment(static_cast<int>(idx) * dimension_,
                                    dimension_);
    }
    component_translation_solution_senders_[local_robot_id_] =
        active_component_robot_ids_;
    BroadcastComponentLocalSolution(FactorType::TRANSLATION,
                                    local_partition_.local_poses,
                                    localTranslations,
                                    active_component_robot_ids_);
  }

  TryAssembleComponentSolution();
}

void TEDCCIOrchestrator::StoreSolutionBlocks(const SolutionMessage &msg) {
  if (msg.epoch != epoch_) {
    return;
  }
  const int blockDim =
      msg.factor_type == FactorType::ROTATION ? dimension_ * dimension_
                                              : dimension_;
  if (msg.x.rows() != static_cast<int>(msg.keys.size()) * blockDim) {
    return;
  }
  if (msg.is_component_solution) {
    std::map<PoseKey, Vector> &blocks =
        msg.factor_type == FactorType::ROTATION
            ? component_received_rotation_solution_blocks_
            : component_received_translation_solution_blocks_;
    for (std::size_t idx = 0; idx < msg.keys.size(); ++idx) {
      blocks[msg.keys[idx]] =
          msg.x.segment(static_cast<int>(idx) * blockDim, blockDim);
    }
    if (msg.factor_type == FactorType::ROTATION) {
      component_rotation_solution_senders_[msg.sender_robot_id] =
          msg.component_robot_ids;
    } else {
      component_translation_solution_senders_[msg.sender_robot_id] =
          msg.component_robot_ids;
    }
    return;
  }
  std::map<PoseKey, Vector> &blocks =
      msg.factor_type == FactorType::ROTATION
          ? received_rotation_solution_blocks_
          : received_translation_solution_blocks_;
  for (std::size_t idx = 0; idx < msg.keys.size(); ++idx) {
    blocks[msg.keys[idx]] =
        msg.x.segment(static_cast<int>(idx) * blockDim, blockDim);
  }
  if (msg.factor_type == FactorType::ROTATION) {
    rotation_solution_senders_.insert(msg.sender_robot_id);
  } else {
    translation_solution_senders_.insert(msg.sender_robot_id);
  }
}

void TEDCCIOrchestrator::StoreFactorMessage(const FactorMessage &msg) {
  if (msg.epoch != epoch_) {
    return;
  }
  if (msg.is_component_factor) {
    if (msg.factor_type != FactorType::TRANSLATION) {
      return;
    }
    if (msg.is_cross_factor) {
      LinearFactorBlock factor;
      factor.keys = msg.keys;
      factor.A = msg.A;
      factor.b = msg.b;
      if (component_translation_cross_factors_.count(msg.sender_robot_id) ==
          0) {
        component_translation_cross_factors_[msg.sender_robot_id] =
            std::make_pair(msg.component_robot_ids, std::move(factor));
      }
      return;
    }
    CondensedFactor factor;
    factor.owner_robot_id = msg.sender_robot_id;
    factor.epoch = msg.epoch;
    factor.factor_type = msg.factor_type;
    factor.boundary_keys = msg.keys;
    factor.Abar = msg.A;
    factor.bbar = msg.b;
    if (component_translation_factors_.count(msg.sender_robot_id) == 0) {
      component_translation_factors_[msg.sender_robot_id] =
          std::make_pair(msg.component_robot_ids, std::move(factor));
    }
    return;
  }
  if (msg.is_cross_factor) {
    LinearFactorBlock factor;
    factor.keys = msg.keys;
    factor.A = msg.A;
    factor.b = msg.b;
    auto &factorMap = msg.factor_type == FactorType::ROTATION
                          ? rotation_cross_factors_
                          : translation_cross_factors_;
    if (factorMap.count(msg.sender_robot_id) == 0) {
      factorMap[msg.sender_robot_id] = std::move(factor);
    }
    return;
  }

  auto &factorMap = msg.factor_type == FactorType::ROTATION
                        ? rotation_factors_
                        : translation_factors_;
  if (factorMap.count(msg.sender_robot_id) > 0) {
    return;
  }
  CondensedFactor factor;
  factor.owner_robot_id = msg.sender_robot_id;
  factor.epoch = msg.epoch;
  factor.factor_type = msg.factor_type;
  factor.boundary_keys = msg.keys;
  factor.Abar = msg.A;
  factor.bbar = msg.b;
  factorMap[msg.sender_robot_id] = std::move(factor);
}

void TEDCCIOrchestrator::ProcessIncomingMessages() {
  ValidateReady();
  for (const FactorMessage &msg :
       exchange_->PollFactorMessages(local_robot_id_)) {
    StoreFactorMessage(msg);
  }

  for (const SolutionMessage &msg :
       exchange_->PollSolutionMessages(local_robot_id_)) {
    StoreSolutionBlocks(msg);
  }

  TrySolveComponentAndBroadcast();
  TrySolveRotationAndBroadcastTranslation();
  TrySolveTranslationAndBroadcastSolution();
  TrySolveComponentAndBroadcast();
  TryAssembleGlobalSolution();
}

void TEDCCIOrchestrator::TryAssembleGlobalSolution() {
  if (global_available_) {
    return;
  }
  for (const int robot : expected_robot_ids_) {
    if (rotation_solution_senders_.count(robot) == 0 ||
        translation_solution_senders_.count(robot) == 0) {
      return;
    }
  }

  Matrix T(dimension_, num_poses_ * (dimension_ + 1));
  for (int pose = 0; pose < num_poses_; ++pose) {
    bool found = false;
    PoseKey key;
    for (const auto &entry : received_rotation_solution_blocks_) {
      if (entry.first.pose_id == pose) {
        key = entry.first;
        found = true;
        break;
      }
    }
    if (!found ||
        received_translation_solution_blocks_.count(key) == 0) {
      return;
    }
    T.block(0, pose * (dimension_ + 1), dimension_, dimension_) =
        matrixFromVector(received_rotation_solution_blocks_.at(key),
                         dimension_);
    T.block(0, pose * (dimension_ + 1) + dimension_, dimension_, 1) =
        received_translation_solution_blocks_.at(key);
  }
  global_solution_ = T;
  global_available_ = true;
}

void TEDCCIOrchestrator::TryAssembleComponentSolution() {
  if (active_component_robot_ids_.empty() || !component_translation_solved_ ||
      !HasComponentSolutionSenders(FactorType::ROTATION) ||
      !HasComponentSolutionSenders(FactorType::TRANSLATION)) {
    return;
  }

  std::set<int> componentSet(active_component_robot_ids_.begin(),
                             active_component_robot_ids_.end());
  std::vector<PoseKey> componentKeys;
  for (const auto &entry : component_received_rotation_solution_blocks_) {
    if (componentSet.count(entry.first.robot_id) > 0 &&
        component_received_translation_solution_blocks_.count(entry.first) >
            0) {
      componentKeys.push_back(entry.first);
    }
  }
  std::sort(componentKeys.begin(), componentKeys.end(),
            [](const PoseKey &lhs, const PoseKey &rhs) {
              if (lhs.pose_id != rhs.pose_id) {
                return lhs.pose_id < rhs.pose_id;
              }
              return lhs.robot_id < rhs.robot_id;
            });

  int expectedPoses = 0;
  for (const int robot : active_component_robot_ids_) {
    expectedPoses += static_cast<int>(
        component_rotation_solution_senders_.count(robot) > 0
            ? std::count_if(
                  component_received_rotation_solution_blocks_.begin(),
                  component_received_rotation_solution_blocks_.end(),
                  [robot](const auto &entry) {
                    return entry.first.robot_id == robot;
                  })
            : 0);
  }
  if (static_cast<int>(componentKeys.size()) != expectedPoses) {
    return;
  }

  Matrix T(dimension_,
           static_cast<int>(componentKeys.size()) * (dimension_ + 1));
  for (std::size_t idx = 0; idx < componentKeys.size(); ++idx) {
    const PoseKey key = componentKeys[idx];
    T.block(0, static_cast<int>(idx) * (dimension_ + 1), dimension_,
            dimension_) =
        matrixFromVector(component_received_rotation_solution_blocks_.at(key),
                         dimension_);
    T.block(0, static_cast<int>(idx) * (dimension_ + 1) + dimension_,
            dimension_, 1) =
        component_received_translation_solution_blocks_.at(key);
  }
  component_solution_ = T;
  component_available_ = true;
}

bool TEDCCIOrchestrator::HasComponentConsistentSolution() const {
  return component_available_;
}

bool TEDCCIOrchestrator::HasGlobalConsistentSolution() const {
  return global_available_;
}

Matrix TEDCCIOrchestrator::GetLocalOnlyInitialization() const {
  return local_only_solution_;
}

Matrix TEDCCIOrchestrator::GetComponentConsistentInitialization() const {
  return component_solution_;
}

Matrix TEDCCIOrchestrator::GetGlobalConsistentInitialization() const {
  return global_solution_;
}

Matrix TEDCCISolver::Initialize(
    int dimension, int num_poses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params, TEDCCIStats *stats) {
  switch (params.mode) {
    case CCIInitMode::CENTRALIZED_CCI:
      return chordalInitialization(static_cast<size_t>(dimension),
                                   static_cast<size_t>(num_poses),
                                   measurements,
                                   params.use_measurement_weight);
    case CCIInitMode::TED_CCI_SR_AUTO:
    case CCIInitMode::TED_CCI_SR_DIRECT:
    case CCIInitMode::TED_CCI_SR_HIERARCHICAL:
    case CCIInitMode::TED_CCI_RIFT_IF:
    case CCIInitMode::TED_CCI_ASYNC_DD:
      return InitializeSingleProcessDirect(dimension, num_poses, measurements,
                                           partitions, params, stats);
    case CCIInitMode::DPCG_CCI:
      throw std::invalid_argument(
          "TEDCCISolver::Initialize DPCG_CCI requires a DPGOCommunicator; "
          "use distributedChordalInitialization for the legacy DPCG path");
    case CCIInitMode::LOCAL_ONLY_CCI:
      throw std::invalid_argument(
          "TEDCCISolver::Initialize LOCAL_ONLY_CCI requires per-robot "
          "orchestrator/component context");
  }
  throw std::invalid_argument("TED-CCI unknown initialization mode");
}

Matrix TEDCCISolver::InitializeSingleProcessDirect(
    int dimension, int num_poses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params, TEDCCIStats *stats) {
  if (dimension != 2 && dimension != 3) {
    throw std::invalid_argument("TED-CCI direct supports only SE(2) or SE(3)");
  }
  if (num_poses <= 0) {
    throw std::invalid_argument("TED-CCI direct requires at least one pose");
  }
  const auto poseKeys = poseKeyByPoseId(num_poses, partitions);
  const std::vector<RelativeSEMeasurement> normalized =
      normalizeMeasurementOwners(num_poses, measurements, partitions);
  const std::vector<RelativeSEMeasurement> shared =
      sharedMeasurements(normalized);

  const int d = dimension;
  const int rotationBlockDim = d * d;
  const bool useRiftRotationMultiRhs =
      params.mode == CCIInitMode::TED_CCI_RIFT_IF &&
      params.rift_use_rotation_multi_rhs;
  std::vector<CondensedFactor> rotationCondensed;
  std::vector<BackSubstitutionCache> rotationCaches;
  std::vector<InterfaceFactor> rotationMultiRhsInterfaceFactors;
  RIFTFactorId rotationMultiRhsNextId = 0;
  rotationCondensed.reserve(partitions.size());
  rotationCaches.reserve(partitions.size());
  rotationMultiRhsInterfaceFactors.reserve(partitions.size() +
                                           shared.size());
  double localQrMs = 0.0;
  double interfaceSolveMs = 0.0;

  for (const auto &partition : partitions) {
    const auto localMeasurements =
        robotLocalMeasurements(partition.local_robot_id, normalized);
    const auto factors = CCIFactorBuilder::BuildRotationFactors(
        d, localMeasurements, params);
    if (useRiftRotationMultiRhs) {
      localQrMs += elapsedMilliseconds([&]() {
        std::vector<InterfaceFactor> multiRhsFactors;
        multiRhsFactors.reserve(factors.size());
        RIFTFactorId localFactorId = 0;
        for (const auto &factor : factors) {
          multiRhsFactors.push_back(BuildRotationMultiRHSFactorFromVectorized(
              localFactorId++, factor.keys, factor.A, factor.b, d,
              partition.local_robot_id));
        }
        rotationMultiRhsInterfaceFactors.push_back(CondenseMultiRHSFactors(
            rotationMultiRhsNextId++, FactorType::ROTATION, d, multiRhsFactors,
            partition.interior_poses,
            removeAnchorKey(partition.boundary_poses, params),
            partition.local_robot_id));
      });
    }
    CondensedFactor condensed;
    condensed.owner_robot_id = partition.local_robot_id;
    condensed.factor_type = FactorType::ROTATION;
    BackSubstitutionCache cache;
    cache.factor_type = FactorType::ROTATION;
    localQrMs += elapsedMilliseconds([&]() {
      LocalQRCondensation::Condense(
          factors, partition.interior_poses,
          removeAnchorKey(partition.boundary_poses, params), rotationBlockDim,
          &condensed, &cache, params);
    });
    rotationCondensed.push_back(std::move(condensed));
    rotationCaches.push_back(std::move(cache));
  }

  const auto crossRotationFactors =
      CCIFactorBuilder::BuildRotationFactors(d, shared, params);
  if (useRiftRotationMultiRhs) {
    for (const auto &factor : crossRotationFactors) {
      const int owner = factor.keys.empty() ? -1 : factor.keys.front().robot_id;
      rotationMultiRhsInterfaceFactors.push_back(
          BuildRotationMultiRHSFactorFromVectorized(
              rotationMultiRhsNextId++, factor.keys, factor.A, factor.b, d,
              owner));
    }
  }
  std::vector<PoseKey> rotationInterfaceKeys;
  TEDCCIStats rotationInterfaceStats;
  Vector rotationInterfaceSolution;
  interfaceSolveMs += elapsedMilliseconds([&]() {
    if (useRiftRotationMultiRhs) {
      const InterfaceProblem rotationProblem =
          InterfaceProblemBuilder::BuildFromInterfaceFactors(
              FactorType::ROTATION, d, d, d,
              rotationMultiRhsInterfaceFactors);
      const Matrix multiRhsSolution =
          SolveInterfaceProblemWithRIFTExact(rotationProblem, params,
                                             &rotationInterfaceStats);
      rotationInterfaceKeys = rotationProblem.variables;
      rotationInterfaceSolution = vectorizeRotationMultiRhsSolution(
          rotationInterfaceKeys, multiRhsSolution, d);
    } else {
      rotationInterfaceKeys =
          collectInterfaceKeys(rotationCondensed, crossRotationFactors);
      rotationInterfaceSolution = solveInterfaceForMode(
          rotationCondensed, crossRotationFactors, rotationBlockDim, partitions,
          params, &rotationInterfaceStats);
    }
  });
  std::map<PoseKey, Vector> relaxedRotationBlocks =
      splitSolutionByKey(rotationInterfaceKeys, rotationInterfaceSolution,
                         rotationBlockDim);
  relaxedRotationBlocks[anchorKey(params)] =
      vectorizeMatrix(Matrix::Identity(d, d));
  for (const auto &cache : rotationCaches) {
    const Vector boundary = boundarySolutionForCache(
        cache, relaxedRotationBlocks, rotationBlockDim);
    insertBackSubstitutedBlocks(cache, boundary, rotationBlockDim,
                                &relaxedRotationBlocks);
  }

  const std::map<PoseKey, Matrix> projectedRotations =
      projectedRotationBlocks(d, relaxedRotationBlocks, params);

  std::vector<CondensedFactor> translationCondensed;
  std::vector<BackSubstitutionCache> translationCaches;
  translationCondensed.reserve(partitions.size());
  translationCaches.reserve(partitions.size());

  for (const auto &partition : partitions) {
    const auto localMeasurements =
        robotLocalMeasurements(partition.local_robot_id, normalized);
    const auto factors = CCIFactorBuilder::BuildTranslationFactors(
        d, localMeasurements, projectedRotations, params);
    CondensedFactor condensed;
    condensed.owner_robot_id = partition.local_robot_id;
    condensed.factor_type = FactorType::TRANSLATION;
    BackSubstitutionCache cache;
    cache.factor_type = FactorType::TRANSLATION;
    localQrMs += elapsedMilliseconds([&]() {
      LocalQRCondensation::Condense(
          factors, partition.interior_poses,
          removeAnchorKey(partition.boundary_poses, params), d, &condensed,
          &cache, params);
    });
    translationCondensed.push_back(std::move(condensed));
    translationCaches.push_back(std::move(cache));
  }

  const auto crossTranslationFactors =
      CCIFactorBuilder::BuildTranslationFactors(d, shared, projectedRotations,
                                                params);
  const std::vector<PoseKey> translationInterfaceKeys =
      collectInterfaceKeys(translationCondensed, crossTranslationFactors);
  TEDCCIStats translationInterfaceStats;
  Vector translationInterfaceSolution;
  interfaceSolveMs += elapsedMilliseconds([&]() {
    translationInterfaceSolution = solveInterfaceForMode(
        translationCondensed, crossTranslationFactors, d, partitions, params,
        &translationInterfaceStats);
  });
  std::map<PoseKey, Vector> translationBlocks =
      splitSolutionByKey(translationInterfaceKeys, translationInterfaceSolution,
                         d);
  translationBlocks[anchorKey(params)] = Vector::Zero(d);
  for (const auto &cache : translationCaches) {
    const Vector boundary = boundarySolutionForCache(cache, translationBlocks,
                                                     d);
    insertBackSubstitutedBlocks(cache, boundary, d, &translationBlocks);
  }

  Matrix T(d, static_cast<int>(num_poses) * (d + 1));
  for (int pose = 0; pose < num_poses; ++pose) {
    const PoseKey key = poseKeys.at(pose);
    const auto rotation = projectedRotations.find(key);
    const auto translation = translationBlocks.find(key);
    if (rotation == projectedRotations.end() ||
        translation == translationBlocks.end()) {
      throw std::runtime_error("TED-CCI failed to recover all pose blocks");
    }
    T.block(0, pose * (d + 1), d, d) = rotation->second;
    T.block(0, pose * (d + 1) + d, d, 1) = translation->second;
  }

  if (stats != nullptr) {
    stats->num_interface_vars = static_cast<int>(
        rotationInterfaceKeys.size() + translationInterfaceKeys.size());
    stats->num_condensed_rows =
        rotationInterfaceStats.num_condensed_rows +
        translationInterfaceStats.num_condensed_rows;
    stats->max_separator_size =
        std::max(rotationInterfaceStats.max_separator_size,
                 translationInterfaceStats.max_separator_size);
    stats->num_factor_messages =
        rotationInterfaceStats.num_factor_messages +
        translationInterfaceStats.num_factor_messages;
    stats->num_solution_messages =
        rotationInterfaceStats.num_solution_messages +
        translationInterfaceStats.num_solution_messages;
    stats->bytes_sent_upward =
        rotationInterfaceStats.bytes_sent_upward +
        translationInterfaceStats.bytes_sent_upward;
    stats->bytes_sent_downward =
        rotationInterfaceStats.bytes_sent_downward +
        translationInterfaceStats.bytes_sent_downward;
    stats->async_dd_iterations =
        rotationInterfaceStats.async_dd_iterations +
        translationInterfaceStats.async_dd_iterations;
    stats->async_dd_converged =
        params.mode == CCIInitMode::TED_CCI_ASYNC_DD &&
        rotationInterfaceStats.async_dd_converged &&
        translationInterfaceStats.async_dd_converged;
    stats->async_dd_initial_residual = std::hypot(
        std::max(0.0, rotationInterfaceStats.async_dd_initial_residual),
        std::max(0.0, translationInterfaceStats.async_dd_initial_residual));
    stats->async_dd_final_residual = std::hypot(
        std::max(0.0, rotationInterfaceStats.async_dd_final_residual),
        std::max(0.0, translationInterfaceStats.async_dd_final_residual));
    stats->local_qr_ms = localQrMs;
    stats->interface_solve_ms = interfaceSolveMs;
    stats->effective_rotation_mode = rotationInterfaceStats.effective_mode;
    stats->effective_translation_mode =
        translationInterfaceStats.effective_mode;
    stats->effective_condensation_backend =
        resolveBackend(params.condensation_backend);
    stats->effective_interface_backend =
        resolveBackend(params.interface_backend);
    stats->spqr_rank_deficient_fallbacks =
        rotationInterfaceStats.spqr_rank_deficient_fallbacks +
        translationInterfaceStats.spqr_rank_deficient_fallbacks;
    stats->peak_interface_cols =
        std::max(rotationInterfaceStats.peak_interface_cols,
                 translationInterfaceStats.peak_interface_cols);
    stats->factor_dense_bytes_upward =
        rotationInterfaceStats.factor_dense_bytes_upward +
        translationInterfaceStats.factor_dense_bytes_upward;
    stats->factor_sparse_triplet_bytes_upward =
        rotationInterfaceStats.factor_sparse_triplet_bytes_upward +
        translationInterfaceStats.factor_sparse_triplet_bytes_upward;
    stats->factor_sparse_triplet_nonzeros =
        rotationInterfaceStats.factor_sparse_triplet_nonzeros +
        translationInterfaceStats.factor_sparse_triplet_nonzeros;
    stats->rift_selected_backend =
        rotationInterfaceStats.rift_selected_backend ==
                RIFTInterfaceBackend::RIFT_EXACT ||
            translationInterfaceStats.rift_selected_backend ==
                RIFTInterfaceBackend::RIFT_EXACT
            ? RIFTInterfaceBackend::RIFT_EXACT
            : rotationInterfaceStats.rift_selected_backend;
    stats->rift_num_cliques = rotationInterfaceStats.rift_num_cliques +
                              translationInterfaceStats.rift_num_cliques;
    stats->rift_num_tree_edges =
        rotationInterfaceStats.rift_num_tree_edges +
        translationInterfaceStats.rift_num_tree_edges;
    stats->rift_max_clique_blocks =
        std::max(rotationInterfaceStats.rift_max_clique_blocks,
                 translationInterfaceStats.rift_max_clique_blocks);
    stats->rift_max_separator_blocks =
        std::max(rotationInterfaceStats.rift_max_separator_blocks,
                 translationInterfaceStats.rift_max_separator_blocks);
    stats->rift_estimated_message_bytes =
        rotationInterfaceStats.rift_estimated_message_bytes +
        translationInterfaceStats.rift_estimated_message_bytes;
    stats->rift_actual_message_bytes =
        rotationInterfaceStats.rift_actual_message_bytes +
        translationInterfaceStats.rift_actual_message_bytes;
    stats->rift_directed_messages_sent =
        rotationInterfaceStats.rift_directed_messages_sent +
        translationInterfaceStats.rift_directed_messages_sent;
    stats->rift_symbolic_ms = rotationInterfaceStats.rift_symbolic_ms +
                              translationInterfaceStats.rift_symbolic_ms;
    stats->rift_message_qr_ms =
        rotationInterfaceStats.rift_message_qr_ms +
        translationInterfaceStats.rift_message_qr_ms;
    stats->rift_belief_solve_ms =
        rotationInterfaceStats.rift_belief_solve_ms +
        translationInterfaceStats.rift_belief_solve_ms;
    stats->rift_final_interface_residual = std::hypot(
        std::max(0.0, rotationInterfaceStats.rift_final_interface_residual),
        std::max(0.0, translationInterfaceStats.rift_final_interface_residual));
    stats->rift_used_global_matrix =
        rotationInterfaceStats.rift_used_global_matrix ||
        translationInterfaceStats.rift_used_global_matrix;
    stats->rift_used_direct_solver =
        rotationInterfaceStats.rift_used_direct_solver ||
        translationInterfaceStats.rift_used_direct_solver;
    stats->rift_used_collective =
        rotationInterfaceStats.rift_used_collective ||
        translationInterfaceStats.rift_used_collective;
  }

  return T;
}

}  // namespace DPGO
