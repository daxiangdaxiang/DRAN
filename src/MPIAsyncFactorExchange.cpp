/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/MPIAsyncFactorExchange.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace DPGO {

namespace {

constexpr int kFactorMessageTag = 42011;
constexpr int kSolutionMessageTag = 42012;
constexpr std::uint32_t kPayloadVersion = 2;
constexpr std::uint32_t kFactorPayloadMagic = 0x54434641u;    // TCFA
constexpr std::uint32_t kSolutionPayloadMagic = 0x5443534fu;  // TCSO

struct MessageKey {
  int sender_robot_id = -1;
  FactorType factor_type = FactorType::ROTATION;
  std::uint64_t topology_epoch = 0;
  bool is_cross_factor = false;
  bool is_component_message = false;
  std::vector<int> component_robot_ids;
  std::vector<PoseKey> keys;

  bool operator<(const MessageKey &other) const {
    if (sender_robot_id != other.sender_robot_id) {
      return sender_robot_id < other.sender_robot_id;
    }
    if (factor_type != other.factor_type) {
      return static_cast<int>(factor_type) <
             static_cast<int>(other.factor_type);
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
};

std::pair<int, int> canonicalLink(int a, int b) {
  return std::minmax(a, b);
}

int checkedMPIIntSize(std::size_t value, const char *name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string("MPI TED-CCI payload too large for ") +
                             name);
  }
  return static_cast<int>(value);
}

template <typename T>
void appendPod(std::vector<unsigned char> *payload, const T &value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "MPI TED-CCI payload requires trivially copyable fields");
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  payload->insert(payload->end(), bytes, bytes + sizeof(T));
}

template <typename T>
T readPod(const std::vector<unsigned char> &payload, std::size_t *offset) {
  static_assert(std::is_trivially_copyable<T>::value,
                "MPI TED-CCI payload requires trivially copyable fields");
  if (*offset + sizeof(T) > payload.size()) {
    throw std::runtime_error("MPI TED-CCI payload is truncated");
  }
  T value;
  std::memcpy(&value, payload.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return value;
}

void appendBool(std::vector<unsigned char> *payload, bool value) {
  appendPod<std::uint8_t>(payload, value ? 1u : 0u);
}

bool readBool(const std::vector<unsigned char> &payload,
              std::size_t *offset) {
  return readPod<std::uint8_t>(payload, offset) != 0;
}

void appendInts(std::vector<unsigned char> *payload,
                const std::vector<int> &values) {
  appendPod<std::uint64_t>(payload, values.size());
  for (const int value : values) {
    appendPod<int>(payload, value);
  }
}

std::vector<int> readInts(const std::vector<unsigned char> &payload,
                          std::size_t *offset) {
  const auto size = readPod<std::uint64_t>(payload, offset);
  if (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("MPI TED-CCI int vector is too large");
  }
  std::vector<int> values;
  values.reserve(static_cast<std::size_t>(size));
  for (std::uint64_t i = 0; i < size; ++i) {
    values.push_back(readPod<int>(payload, offset));
  }
  return values;
}

void appendKeys(std::vector<unsigned char> *payload,
                const std::vector<PoseKey> &keys) {
  appendPod<std::uint64_t>(payload, keys.size());
  for (const PoseKey &key : keys) {
    appendPod<int>(payload, key.robot_id);
    appendPod<int>(payload, key.pose_id);
  }
}

std::vector<PoseKey> readKeys(const std::vector<unsigned char> &payload,
                              std::size_t *offset) {
  const auto size = readPod<std::uint64_t>(payload, offset);
  if (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("MPI TED-CCI key vector is too large");
  }
  std::vector<PoseKey> keys;
  keys.reserve(static_cast<std::size_t>(size));
  for (std::uint64_t i = 0; i < size; ++i) {
    PoseKey key;
    key.robot_id = readPod<int>(payload, offset);
    key.pose_id = readPod<int>(payload, offset);
    keys.push_back(key);
  }
  return keys;
}

std::size_t denseMatrixPayloadBytes(const Matrix &matrix) {
  return 2u * sizeof(int) +
         static_cast<std::size_t>(matrix.rows()) *
             static_cast<std::size_t>(matrix.cols()) * sizeof(double);
}

std::size_t sparseMatrixPayloadBytes(const Matrix &matrix) {
  std::size_t nnz = 0;
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      if (matrix(row, col) != 0.0) {
        ++nnz;
      }
    }
  }
  return 2u * sizeof(int) + sizeof(std::uint64_t) +
         nnz * (2u * sizeof(int) + sizeof(double));
}

void appendMatrixAuto(std::vector<unsigned char> *payload,
                      const Matrix &matrix) {
  const bool useSparse =
      sparseMatrixPayloadBytes(matrix) < denseMatrixPayloadBytes(matrix);
  appendBool(payload, useSparse);
  appendPod<int>(payload, matrix.rows());
  appendPod<int>(payload, matrix.cols());
  if (!useSparse) {
    for (int col = 0; col < matrix.cols(); ++col) {
      for (int row = 0; row < matrix.rows(); ++row) {
        appendPod<double>(payload, matrix(row, col));
      }
    }
    return;
  }
  std::uint64_t nnz = 0;
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      if (matrix(row, col) != 0.0) {
        ++nnz;
      }
    }
  }
  appendPod<std::uint64_t>(payload, nnz);
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      const double value = matrix(row, col);
      if (value != 0.0) {
        appendPod<int>(payload, row);
        appendPod<int>(payload, col);
        appendPod<double>(payload, value);
      }
    }
  }
}

Matrix readMatrixAuto(const std::vector<unsigned char> &payload,
                      std::size_t *offset) {
  const bool sparse = readBool(payload, offset);
  const int rows = readPod<int>(payload, offset);
  const int cols = readPod<int>(payload, offset);
  if (rows < 0 || cols < 0) {
    throw std::runtime_error("MPI TED-CCI matrix has negative shape");
  }
  Matrix matrix = Matrix::Zero(rows, cols);
  if (!sparse) {
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        matrix(row, col) = readPod<double>(payload, offset);
      }
    }
    return matrix;
  }
  const auto nnz = readPod<std::uint64_t>(payload, offset);
  for (std::uint64_t k = 0; k < nnz; ++k) {
    const int row = readPod<int>(payload, offset);
    const int col = readPod<int>(payload, offset);
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      throw std::runtime_error("MPI TED-CCI sparse matrix index out of range");
    }
    matrix(row, col) = readPod<double>(payload, offset);
  }
  return matrix;
}

void appendVector(std::vector<unsigned char> *payload, const Vector &vector) {
  appendPod<int>(payload, vector.rows());
  for (int row = 0; row < vector.rows(); ++row) {
    appendPod<double>(payload, vector(row));
  }
}

std::size_t denseVectorPayloadBytes(const Vector &vector) {
  return sizeof(int) + static_cast<std::size_t>(vector.rows()) *
                           sizeof(double);
}

std::size_t sparseVectorPayloadBytes(const Vector &vector) {
  std::size_t nnz = 0;
  for (int row = 0; row < vector.rows(); ++row) {
    if (vector(row) != 0.0) {
      ++nnz;
    }
  }
  return sizeof(int) + sizeof(std::uint64_t) +
         nnz * (sizeof(int) + sizeof(double));
}

void appendVectorAuto(std::vector<unsigned char> *payload,
                      const Vector &vector) {
  const bool useSparse =
      sparseVectorPayloadBytes(vector) < denseVectorPayloadBytes(vector);
  appendBool(payload, useSparse);
  appendPod<int>(payload, vector.rows());
  if (!useSparse) {
    for (int row = 0; row < vector.rows(); ++row) {
      appendPod<double>(payload, vector(row));
    }
    return;
  }
  std::uint64_t nnz = 0;
  for (int row = 0; row < vector.rows(); ++row) {
    if (vector(row) != 0.0) {
      ++nnz;
    }
  }
  appendPod<std::uint64_t>(payload, nnz);
  for (int row = 0; row < vector.rows(); ++row) {
    const double value = vector(row);
    if (value != 0.0) {
      appendPod<int>(payload, row);
      appendPod<double>(payload, value);
    }
  }
}

Vector readVector(const std::vector<unsigned char> &payload,
                  std::size_t *offset) {
  const int rows = readPod<int>(payload, offset);
  if (rows < 0) {
    throw std::runtime_error("MPI TED-CCI vector has negative size");
  }
  Vector vector(rows);
  for (int row = 0; row < rows; ++row) {
    vector(row) = readPod<double>(payload, offset);
  }
  return vector;
}

Vector readVectorAuto(const std::vector<unsigned char> &payload,
                      std::size_t *offset) {
  const bool sparse = readBool(payload, offset);
  const int rows = readPod<int>(payload, offset);
  if (rows < 0) {
    throw std::runtime_error("MPI TED-CCI vector has negative size");
  }
  Vector vector = Vector::Zero(rows);
  if (!sparse) {
    for (int row = 0; row < rows; ++row) {
      vector(row) = readPod<double>(payload, offset);
    }
    return vector;
  }
  const auto nnz = readPod<std::uint64_t>(payload, offset);
  for (std::uint64_t k = 0; k < nnz; ++k) {
    const int row = readPod<int>(payload, offset);
    if (row < 0 || row >= rows) {
      throw std::runtime_error("MPI TED-CCI sparse vector index out of range");
    }
    vector(row) = readPod<double>(payload, offset);
  }
  return vector;
}

std::vector<unsigned char> serializeFactor(const FactorMessage &msg) {
  std::vector<unsigned char> payload;
  payload.reserve(128 + static_cast<std::size_t>(msg.A.size() + msg.b.size()) *
                            sizeof(double));
  appendPod<std::uint32_t>(&payload, kFactorPayloadMagic);
  appendPod<std::uint32_t>(&payload, kPayloadVersion);
  appendPod<int>(&payload, msg.sender_robot_id);
  appendPod<int>(&payload, msg.receiver_robot_id);
  appendPod<std::uint64_t>(&payload, msg.epoch);
  appendPod<std::uint64_t>(&payload, msg.topology_epoch);
  appendPod<int>(&payload, static_cast<int>(msg.factor_type));
  appendBool(&payload, msg.is_cross_factor);
  appendBool(&payload, msg.is_component_factor);
  appendInts(&payload, msg.component_robot_ids);
  appendKeys(&payload, msg.keys);
  appendMatrixAuto(&payload, msg.A);
  appendVectorAuto(&payload, msg.b);
  appendPod<std::uint64_t>(&payload, msg.checksum);
  return payload;
}

FactorMessage deserializeFactor(const std::vector<unsigned char> &payload) {
  std::size_t offset = 0;
  if (readPod<std::uint32_t>(payload, &offset) != kFactorPayloadMagic) {
    throw std::runtime_error("MPI TED-CCI factor payload has bad magic");
  }
  if (readPod<std::uint32_t>(payload, &offset) != kPayloadVersion) {
    throw std::runtime_error("MPI TED-CCI factor payload has bad version");
  }
  FactorMessage msg;
  msg.sender_robot_id = readPod<int>(payload, &offset);
  msg.receiver_robot_id = readPod<int>(payload, &offset);
  msg.epoch = readPod<std::uint64_t>(payload, &offset);
  msg.topology_epoch = readPod<std::uint64_t>(payload, &offset);
  msg.factor_type =
      static_cast<FactorType>(readPod<int>(payload, &offset));
  msg.is_cross_factor = readBool(payload, &offset);
  msg.is_component_factor = readBool(payload, &offset);
  msg.component_robot_ids = readInts(payload, &offset);
  msg.keys = readKeys(payload, &offset);
  msg.A = readMatrixAuto(payload, &offset);
  msg.b = readVectorAuto(payload, &offset);
  msg.checksum = readPod<std::uint64_t>(payload, &offset);
  if (offset != payload.size()) {
    throw std::runtime_error("MPI TED-CCI factor payload has trailing bytes");
  }
  return msg;
}

std::vector<unsigned char> serializeSolution(const SolutionMessage &msg) {
  std::vector<unsigned char> payload;
  payload.reserve(128 + static_cast<std::size_t>(msg.x.size()) *
                            sizeof(double));
  appendPod<std::uint32_t>(&payload, kSolutionPayloadMagic);
  appendPod<std::uint32_t>(&payload, kPayloadVersion);
  appendPod<int>(&payload, msg.sender_robot_id);
  appendPod<int>(&payload, msg.receiver_robot_id);
  appendPod<std::uint64_t>(&payload, msg.epoch);
  appendPod<std::uint64_t>(&payload, msg.topology_epoch);
  appendPod<int>(&payload, static_cast<int>(msg.factor_type));
  appendBool(&payload, msg.is_component_solution);
  appendInts(&payload, msg.component_robot_ids);
  appendKeys(&payload, msg.keys);
  appendVector(&payload, msg.x);
  appendPod<std::uint64_t>(&payload, msg.checksum);
  return payload;
}

SolutionMessage deserializeSolution(
    const std::vector<unsigned char> &payload) {
  std::size_t offset = 0;
  if (readPod<std::uint32_t>(payload, &offset) != kSolutionPayloadMagic) {
    throw std::runtime_error("MPI TED-CCI solution payload has bad magic");
  }
  if (readPod<std::uint32_t>(payload, &offset) != kPayloadVersion) {
    throw std::runtime_error("MPI TED-CCI solution payload has bad version");
  }
  SolutionMessage msg;
  msg.sender_robot_id = readPod<int>(payload, &offset);
  msg.receiver_robot_id = readPod<int>(payload, &offset);
  msg.epoch = readPod<std::uint64_t>(payload, &offset);
  msg.topology_epoch = readPod<std::uint64_t>(payload, &offset);
  msg.factor_type =
      static_cast<FactorType>(readPod<int>(payload, &offset));
  msg.is_component_solution = readBool(payload, &offset);
  msg.component_robot_ids = readInts(payload, &offset);
  msg.keys = readKeys(payload, &offset);
  msg.x = readVector(payload, &offset);
  msg.checksum = readPod<std::uint64_t>(payload, &offset);
  if (offset != payload.size()) {
    throw std::runtime_error("MPI TED-CCI solution payload has trailing bytes");
  }
  return msg;
}

MessageKey factorKey(const FactorMessage &msg) {
  return MessageKey{msg.sender_robot_id,
                    msg.factor_type,
                    msg.topology_epoch,
                    msg.is_cross_factor,
                    msg.is_component_factor,
                    msg.component_robot_ids,
                    msg.keys};
}

MessageKey solutionKey(const SolutionMessage &msg) {
  return MessageKey{msg.sender_robot_id,
                    msg.factor_type,
                    msg.topology_epoch,
                    false,
                    msg.is_component_solution,
                    msg.component_robot_ids,
                    msg.keys};
}

}  // namespace

MPIAsyncFactorExchange::MPIAsyncFactorExchange(MPI_Comm comm) : comm_(comm) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::invalid_argument(
        "MPIAsyncFactorExchange requires MPI_Init before construction");
  }
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
}

MPIAsyncFactorExchange::~MPIAsyncFactorExchange() {
  int initialized = 0;
  int finalized = 1;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (!initialized || finalized) {
    return;
  }
  for (PendingSend &send : pending_sends_) {
    if (send.request != MPI_REQUEST_NULL) {
      MPI_Wait(&send.request, MPI_STATUS_IGNORE);
    }
  }
}

bool MPIAsyncFactorExchange::IsLinkDown(int sender, int receiver) const {
  return down_links_.count(canonicalLink(sender, receiver)) > 0;
}

void MPIAsyncFactorExchange::SendPayload(std::vector<unsigned char> payload,
                                         int receiver, int tag) {
  if (receiver < 0 || receiver >= size_) {
    throw std::invalid_argument("MPI TED-CCI receiver rank out of range");
  }
  PendingSend pending;
  pending.payload = std::move(payload);
  const int count = checkedMPIIntSize(pending.payload.size(), "Isend");
  MPI_Isend(pending.payload.empty() ? nullptr : pending.payload.data(), count,
            MPI_UNSIGNED_CHAR, receiver, tag, comm_, &pending.request);
  pending_sends_.push_back(std::move(pending));
  DrainCompletedSends();
}

void MPIAsyncFactorExchange::DrainCompletedSends() {
  std::vector<PendingSend> active;
  active.reserve(pending_sends_.size());
  for (PendingSend &send : pending_sends_) {
    int complete = 0;
    MPI_Test(&send.request, &complete, MPI_STATUS_IGNORE);
    if (!complete) {
      active.push_back(std::move(send));
    }
  }
  pending_sends_ = std::move(active);
}

void MPIAsyncFactorExchange::SendFactor(const FactorMessage &msg) {
  if (msg.checksum != ComputeFactorMessageChecksum(msg)) {
    return;
  }
  if (IsLinkDown(msg.sender_robot_id, msg.receiver_robot_id)) {
    return;
  }
  SendPayload(serializeFactor(msg), msg.receiver_robot_id, kFactorMessageTag);
}

void MPIAsyncFactorExchange::SendSolution(const SolutionMessage &msg) {
  if (msg.checksum != ComputeSolutionMessageChecksum(msg)) {
    return;
  }
  if (IsLinkDown(msg.sender_robot_id, msg.receiver_robot_id)) {
    return;
  }
  SendPayload(serializeSolution(msg), msg.receiver_robot_id,
              kSolutionMessageTag);
}

std::vector<FactorMessage> MPIAsyncFactorExchange::PollFactorMessages(
    int receiver_robot_id) {
  DrainCompletedSends();
  if (receiver_robot_id != rank_) {
    return {};
  }

  std::map<MessageKey, FactorMessage> inbox;
  while (true) {
    int hasMessage = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, kFactorMessageTag, comm_, &hasMessage,
               &status);
    if (!hasMessage) {
      break;
    }
    int count = 0;
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
    std::vector<unsigned char> payload(static_cast<std::size_t>(count));
    MPI_Recv(payload.empty() ? nullptr : payload.data(), count,
             MPI_UNSIGNED_CHAR, status.MPI_SOURCE, kFactorMessageTag, comm_,
             MPI_STATUS_IGNORE);
    try {
      const FactorMessage msg = deserializeFactor(payload);
      if (msg.receiver_robot_id != receiver_robot_id ||
          msg.checksum != ComputeFactorMessageChecksum(msg)) {
        continue;
      }
      const MessageKey key = factorKey(msg);
      const auto existing = inbox.find(key);
      if (existing == inbox.end() || msg.epoch > existing->second.epoch) {
        inbox[key] = msg;
      }
    } catch (const std::exception &) {
      continue;
    }
  }

  std::vector<FactorMessage> messages;
  messages.reserve(inbox.size());
  for (const auto &entry : inbox) {
    messages.push_back(entry.second);
  }
  return messages;
}

std::vector<SolutionMessage> MPIAsyncFactorExchange::PollSolutionMessages(
    int receiver_robot_id) {
  DrainCompletedSends();
  if (receiver_robot_id != rank_) {
    return {};
  }

  std::map<MessageKey, SolutionMessage> inbox;
  while (true) {
    int hasMessage = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, kSolutionMessageTag, comm_, &hasMessage,
               &status);
    if (!hasMessage) {
      break;
    }
    int count = 0;
    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
    std::vector<unsigned char> payload(static_cast<std::size_t>(count));
    MPI_Recv(payload.empty() ? nullptr : payload.data(), count,
             MPI_UNSIGNED_CHAR, status.MPI_SOURCE, kSolutionMessageTag, comm_,
             MPI_STATUS_IGNORE);
    try {
      const SolutionMessage msg = deserializeSolution(payload);
      if (msg.receiver_robot_id != receiver_robot_id ||
          msg.checksum != ComputeSolutionMessageChecksum(msg)) {
        continue;
      }
      const MessageKey key = solutionKey(msg);
      const auto existing = inbox.find(key);
      if (existing == inbox.end() || msg.epoch > existing->second.epoch) {
        inbox[key] = msg;
      }
    } catch (const std::exception &) {
      continue;
    }
  }

  std::vector<SolutionMessage> messages;
  messages.reserve(inbox.size());
  for (const auto &entry : inbox) {
    messages.push_back(entry.second);
  }
  return messages;
}

void MPIAsyncFactorExchange::MarkLinkDown(int robot_a, int robot_b) {
  down_links_.insert(canonicalLink(robot_a, robot_b));
}

void MPIAsyncFactorExchange::MarkLinkUp(int robot_a, int robot_b) {
  down_links_.erase(canonicalLink(robot_a, robot_b));
}

}  // namespace DPGO
