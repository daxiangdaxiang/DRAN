#ifndef DPGO_RIFT_IF_H
#define DPGO_RIFT_IF_H

#include <DPGO/DPGO_types.h>
#include <DPGO/TEDCCI.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
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
  int async_schur_max_iters = 5000;
  double async_schur_rel_tol = 1e-8;
  double async_schur_relaxation = 0.25;
  double async_schur_damping = 1e-6;
};

struct RIFTStats {
  RIFTInterfaceBackend selected_backend = RIFTInterfaceBackend::DIRECT_ORACLE;
  int num_cliques = 0;
  int num_tree_edges = 0;
  int num_host_robots = 0;
  int max_host_clique_load = 0;
  int cross_host_tree_edges = 0;
  int estimated_route_hops = 0;
  int max_clique_blocks = 0;
  int max_separator_blocks = 0;
  std::size_t estimated_message_bytes = 0;
  std::size_t estimated_routed_message_bytes = 0;
  std::size_t actual_message_bytes = 0;
  int directed_messages_sent = 0;
  int directed_messages_reused = 0;
  int directed_messages_invalidated = 0;
  int cak_iterations = 0;
  int cak_scalar_reductions = 0;
  std::size_t cak_scalar_reduction_bytes = 0;
  double cak_final_residual = -1.0;
  int async_schur_iterations = 0;
  double async_schur_initial_residual = -1.0;
  double async_schur_final_residual = -1.0;
  bool async_schur_converged = false;
  bool async_schur_global_consistent = true;
  bool async_schur_component_consistent = false;
  int async_schur_components = 1;
  int async_schur_stale_messages_rejected = 0;
  int async_schur_reconnect_merges = 0;
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

  bool operator==(const DirectedCliqueEdge &other) const {
    return src == other.src && dst == other.dst;
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

enum class RIFTDisconnectPolicy {
  HOLD_UNTIL_RECONNECTED,
  DROP_WHILE_DISCONNECTED,
};

struct RIFTLinkModel {
  double latency_mean_ms = 0.0;
  double latency_jitter_ms = 0.0;
  double drop_prob = 0.0;
  double reorder_prob = 0.0;
  double bandwidth_bytes_per_sec = 0.0;
  bool bidirectional = true;
  RIFTDisconnectPolicy disconnect_policy =
      RIFTDisconnectPolicy::HOLD_UNTIL_RECONNECTED;
};

struct RIFTNetworkMessage {
  int src_robot = -1;
  int dst_robot = -1;
  std::uint64_t seq = 0;
  std::size_t payload_bytes = 0;
  double send_time_ms = 0.0;
  RIFTMessage rift_message;
};

struct RIFTDeliveredNetworkMessage {
  RIFTNetworkMessage message;
  double delivery_time_ms = 0.0;
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

class RIFTDirtyMessageTracker {
 public:
  explicit RIFTDirtyMessageTracker(const InterfaceCliqueTree &tree);

  void CacheMessage(const DirectedCliqueEdge &edge);
  void MarkFactorUpdated(RIFTFactorId factor_id);
  void MarkFactorUpdated(RIFTFactorId factor_id,
                         RIFTCliqueId assigned_clique);
  void MarkCliqueDirty(RIFTCliqueId clique_id);

  bool IsMessageCached(const DirectedCliqueEdge &edge) const;
  bool IsMessageInvalidated(const DirectedCliqueEdge &edge) const;
  std::vector<DirectedCliqueEdge> InvalidatedMessages() const;
  std::vector<RIFTCliqueId> DirtyCliques() const;
  int CachedMessageCount() const;
  int ReusableCachedMessageCount() const;
  int FullDirectedMessageCount() const;

 private:
  void ValidateClique(RIFTCliqueId clique_id) const;
  void ValidateDirectedEdge(const DirectedCliqueEdge &edge) const;
  void InvalidateDirectedMessage(const DirectedCliqueEdge &edge,
                                 std::queue<DirectedCliqueEdge> *queue);

  std::map<RIFTCliqueId, std::set<RIFTCliqueId>> neighbors_;
  std::map<RIFTFactorId, RIFTCliqueId> factor_owner_clique_;
  std::set<RIFTCliqueId> dirty_cliques_;
  std::set<DirectedCliqueEdge> cached_messages_;
  std::set<DirectedCliqueEdge> invalidated_messages_;
};

class RIFTP2PNetworkSimulator {
 public:
  explicit RIFTP2PNetworkSimulator(std::uint64_t seed = 1);

  void SetRandomSeed(std::uint64_t seed);
  void AddRobot(int robot_id);
  void AddLink(int a, int b, const RIFTLinkModel &model);
  void DropLink(int a, int b);
  void RestoreLink(int a, int b);
  bool HasLink(int a, int b) const;
  bool LinkUp(int a, int b) const;

  void Send(RIFTNetworkMessage message);
  std::vector<RIFTDeliveredNetworkMessage> DeliverReady(double now_ms);

  std::size_t PendingCount() const;
  std::size_t DroppedCount() const;

 private:
  struct LinkState {
    RIFTLinkModel model;
    bool up = true;
    double next_available_ms = 0.0;
  };

  struct PendingEvent {
    RIFTDeliveredNetworkMessage delivered;
    std::uint64_t insertion_order = 0;
  };

  std::map<std::pair<int, int>, LinkState> links_;
  std::set<int> robots_;
  std::vector<PendingEvent> pending_;
  std::uint64_t next_insertion_order_ = 0;
  std::size_t dropped_count_ = 0;
  std::mt19937_64 rng_;
};

class RIFTExactSolver {
 public:
  static Matrix SolveMatrix(const InterfaceProblem &problem,
                            const InterfaceCliqueTree &tree,
                            const RIFTParams &params,
                            RIFTStats *stats = nullptr,
                            DecentralizationGuard *guard = nullptr,
                            const std::map<DirectedCliqueEdge, RIFTMessage>
                                *reusable_messages = nullptr,
                            const RIFTDirtyMessageTracker *dirty_tracker =
                                nullptr,
                            std::map<DirectedCliqueEdge, RIFTMessage>
                                *updated_messages = nullptr);

  static Vector Solve(const InterfaceProblem &problem,
                      const InterfaceCliqueTree &tree,
                      const RIFTParams &params, RIFTStats *stats = nullptr,
                      DecentralizationGuard *guard = nullptr);
};

class RIFTCAKSolver {
 public:
  static Matrix ApplyNormalOperator(const InterfaceProblem &problem,
                                    const Matrix &X);

  static Matrix SolveMatrix(const InterfaceProblem &problem,
                            const RIFTParams &params,
                            RIFTStats *stats = nullptr,
                            DecentralizationGuard *guard = nullptr);

  static Vector Solve(const InterfaceProblem &problem,
                      const RIFTParams &params, RIFTStats *stats = nullptr,
                      DecentralizationGuard *guard = nullptr);
};

class RIFTAsyncSchurSolver {
 public:
  RIFTAsyncSchurSolver(const InterfaceProblem &problem,
                       const RIFTParams &params,
                       std::uint64_t seed = 1);

  void SetLinkModel(int a, int b, const RIFTLinkModel &model);
  void DropLink(int a, int b);
  void RestoreLink(int a, int b);
  void StepRobot(int robot_id);
  void Run(int max_iterations);

  bool ComponentConverged(const std::vector<int> &component) const;
  bool GlobalConverged() const;
  bool ComponentConsistent() const;
  Matrix Solution() const;
  double GlobalResidualNorm() const;
  double ComponentResidualNorm(const std::vector<int> &component) const;
  std::vector<std::vector<int>> ConnectedComponents() const;

  int iterations() const { return iterations_; }
  int stale_messages_rejected() const { return stale_messages_rejected_; }
  int reconnect_merges() const { return reconnect_merges_; }

  static Matrix SolveMatrix(const InterfaceProblem &problem,
                            const RIFTParams &params,
                            RIFTStats *stats = nullptr,
                            DecentralizationGuard *guard = nullptr);
  static Vector Solve(const InterfaceProblem &problem,
                      const RIFTParams &params, RIFTStats *stats = nullptr,
                      DecentralizationGuard *guard = nullptr);

 private:
  void DeliverReadyMessages();
  void BroadcastState(int robot_id);
  void BroadcastFactorGradients(int robot_id,
                                const std::vector<int> &component);
  std::vector<int> ReachableComponent(int robot_id) const;

  InterfaceProblem problem_;
  RIFTParams params_;
  std::vector<int> robots_;
  std::map<int, std::vector<int>> owned_key_indices_;
  Matrix values_;
  std::map<int, std::vector<Matrix>> value_cache_by_robot_;
  std::map<std::pair<int, int>, Matrix> gradient_cache_;
  std::map<std::pair<int, int>, Matrix> diagonal_cache_;
  std::map<std::tuple<int, int, int, int>, std::uint64_t> last_message_seq_;
  RIFTP2PNetworkSimulator network_;
  DecentralizationGuard *guard_ = nullptr;
  std::uint64_t next_seq_ = 1;
  double current_time_ms_ = 0.0;
  double initial_residual_ = 0.0;
  int iterations_ = 0;
  int stale_messages_rejected_ = 0;
  int reconnect_merges_ = 0;
  std::size_t message_bytes_ = 0;
  int message_count_ = 0;
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
