#ifndef DPGO_RIFT_IF_H
#define DPGO_RIFT_IF_H

#include <DPGO/DPGO_types.h>
#include <DPGO/TEDCCI.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace DPGO {

using InterfaceKey = PoseKey;
using RIFTFactorId = int;
using RIFTCliqueId = int;

struct RIFTParams {
  RIFTInterfaceBackend backend = RIFTInterfaceBackend::RIFT_AUTO;
  int exact_max_separator_blocks_2d = 256;
  int exact_max_separator_blocks_3d = 128;
  std::size_t exact_max_message_bytes = 32ull * 1024ull * 1024ull;
  bool use_rotation_multi_rhs = true;
  bool forbid_global_interface_matrix = true;
  bool forbid_direct_solver_in_deployment = true;
  bool forbid_collectives = true;
};

struct RIFTStats {
  RIFTInterfaceBackend selected_backend = RIFTInterfaceBackend::DIRECT_ORACLE;
  int num_cliques = 0;
  int num_tree_edges = 0;
  int max_clique_blocks = 0;
  int max_separator_blocks = 0;
  std::size_t estimated_message_bytes = 0;
  std::size_t actual_message_bytes = 0;
  int directed_messages_sent = 0;
  double symbolic_ms = 0.0;
  double message_qr_ms = 0.0;
  double belief_solve_ms = 0.0;
  double final_interface_residual = -1.0;
  bool used_global_matrix = false;
  bool used_direct_solver = false;
  bool used_collective = false;
};

struct InterfaceFactor {
  RIFTFactorId id = -1;
  FactorType stage = FactorType::ROTATION;
  std::vector<InterfaceKey> scope;
  Matrix A;
  Matrix B;
  int owner_robot = -1;
  std::uint64_t graph_epoch = 0;
};

struct InterfaceProblem {
  FactorType stage = FactorType::ROTATION;
  int dimension = 0;
  int block_dim = 0;
  int rhs_dim = 1;
  std::vector<InterfaceFactor> factors;
  std::vector<InterfaceKey> variables;
};

struct InterfaceClique {
  RIFTCliqueId id = -1;
  int host_robot = -1;
  std::vector<InterfaceKey> variables;
  std::vector<RIFTFactorId> assigned_factors;
  std::vector<RIFTCliqueId> neighbors;
};

struct InterfaceCliqueTreeEdge {
  RIFTCliqueId a = -1;
  RIFTCliqueId b = -1;
  std::vector<InterfaceKey> separator;
  std::size_t estimated_message_bytes = 0;
};

struct InterfaceCliqueTree {
  std::vector<InterfaceClique> cliques;
  std::vector<InterfaceCliqueTreeEdge> edges;
};

struct DirectedCliqueEdge {
  RIFTCliqueId src = -1;
  RIFTCliqueId dst = -1;

  bool operator<(const DirectedCliqueEdge &other) const {
    if (src != other.src) {
      return src < other.src;
    }
    return dst < other.dst;
  }
};

struct RIFTMessageHeader {
  FactorType stage = FactorType::ROTATION;
  std::uint64_t graph_epoch = 0;
  RIFTCliqueId src = -1;
  RIFTCliqueId dst = -1;
  std::uint64_t seq = 0;
};

struct RIFTMessage {
  RIFTMessageHeader header;
  std::vector<InterfaceKey> separator_keys;
  Matrix R;
  Matrix D;
};

struct CliqueSolution {
  RIFTCliqueId clique_id = -1;
  std::vector<InterfaceKey> keys;
  Matrix X;
};

class DecentralizationGuard {
 public:
  void RecordGlobalInterfaceMatrixConstruction();
  void RecordDirectInterfaceSolverCall();
  void RecordCollectiveCall(const std::string &name);
  void RecordMessage(int src, int dst, std::size_t bytes);
  void RecordFactorTransfer(int src, int dst, RIFTFactorId id);

  void AssertNoGlobalInterfaceMatrixConstructed() const;
  void AssertNoDirectSolverCalled() const;
  void AssertNoCollectiveCommunication() const;
  void AssertNoNodeReceivedAllFactors(int num_total_factors) const;

  bool used_global_matrix() const { return used_global_matrix_; }
  bool used_direct_solver() const { return used_direct_solver_; }
  bool used_collective() const { return used_collective_; }
  std::size_t message_bytes() const { return message_bytes_; }
  int message_count() const { return message_count_; }

 private:
  bool used_global_matrix_ = false;
  bool used_direct_solver_ = false;
  bool used_collective_ = false;
  std::size_t message_bytes_ = 0;
  int message_count_ = 0;
  std::map<int, std::set<RIFTFactorId>> factors_by_receiver_;
};

class InterfaceProblemBuilder {
 public:
  static InterfaceProblem BuildFromInterfaceFactors(
      FactorType stage, int dimension, int block_dim, int rhs_dim,
      const std::vector<InterfaceFactor> &factors);

  static InterfaceProblem BuildFromTEDFactors(
      FactorType stage, int dimension, int block_dim,
      const std::vector<CondensedFactor> &condensed_factors,
      const std::vector<LinearFactorBlock> &cross_factors,
      bool use_rotation_multi_rhs);
};

InterfaceFactor BuildRotationMultiRHSFactorFromVectorized(
    RIFTFactorId id, const std::vector<PoseKey> &scope,
    const Matrix &vectorized_A, const Vector &vectorized_b, int dimension,
    int owner_robot, std::uint64_t graph_epoch = 0);

InterfaceFactor CondenseMultiRHSFactors(
    RIFTFactorId id, FactorType stage, int dimension,
    const std::vector<InterfaceFactor> &factors,
    const std::vector<PoseKey> &interior_keys,
    const std::vector<PoseKey> &boundary_keys, int owner_robot,
    std::uint64_t graph_epoch = 0);

class InterfaceCliqueTreeBuilder {
 public:
  static InterfaceCliqueTree Build(const InterfaceProblem &problem,
                                   const RIFTParams &params);

  static bool VerifyRunningIntersection(const InterfaceCliqueTree &tree);
};

class RIFTRootlessScheduler {
 public:
  void Initialize(const InterfaceCliqueTree &tree);
  void OnMessageReceived(const RIFTMessage &msg);
  std::vector<DirectedCliqueEdge> ReadyOutgoingMessages() const;
  void MarkMessageSent(RIFTCliqueId src, RIFTCliqueId dst);
  bool AllDirectedMessagesComplete() const;
  bool CliqueBeliefReady(RIFTCliqueId alpha) const;

 private:
  std::map<RIFTCliqueId, std::set<RIFTCliqueId>> neighbors_;
  std::set<DirectedCliqueEdge> sent_;
  std::set<DirectedCliqueEdge> received_;
};

class RIFTExactSolver {
 public:
  static Matrix SolveMatrix(const InterfaceProblem &problem,
                            const InterfaceCliqueTree &tree,
                            const RIFTParams &params,
                            RIFTStats *stats = nullptr,
                            DecentralizationGuard *guard = nullptr);

  static Vector Solve(const InterfaceProblem &problem,
                      const InterfaceCliqueTree &tree,
                      const RIFTParams &params, RIFTStats *stats = nullptr,
                      DecentralizationGuard *guard = nullptr);
};

Matrix SolveInterfaceProblemWithRIFTExact(
    const InterfaceProblem &problem, const TEDCCIParams &params,
    TEDCCIStats *stats = nullptr);

Vector SolveTEDInterfaceWithRIFTExact(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params, TEDCCIStats *stats = nullptr);

std::string RIFTInterfaceBackendName(RIFTInterfaceBackend backend);

}  // namespace DPGO

#endif
