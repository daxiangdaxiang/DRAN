/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef TED_CCI_H
#define TED_CCI_H

#include <DPGO/DPGO_types.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace DPGO {

struct PoseKey {
  int robot_id = -1;
  int pose_id = -1;

  bool operator==(const PoseKey &other) const {
    return robot_id == other.robot_id && pose_id == other.pose_id;
  }

  bool operator<(const PoseKey &other) const {
    if (robot_id != other.robot_id) {
      return robot_id < other.robot_id;
    }
    return pose_id < other.pose_id;
  }
};

enum class CCIInitMode {
  CENTRALIZED_CCI,
  DPCG_CCI,
  TED_CCI_SR_AUTO,
  TED_CCI_SR_DIRECT,
  TED_CCI_SR_HIERARCHICAL,
  TED_CCI_RIFT_IF,
  TED_CCI_ASYNC_DD,
  LOCAL_ONLY_CCI,
};

enum class TEDCCIBackend {
  AUTO,
  DENSE_HOUSEHOLDER,
  SPARSE_SPQR,
};

enum class RIFTInterfaceBackend {
  DIRECT_ORACLE,
  RIFT_EXACT,
  RIFT_AUTO,
  RIFT_CAK,
  RIFT_ASYNC_SCHUR,
};

enum class FactorType {
  ROTATION,
  TRANSLATION,
};

struct TEDCCIParams {
  CCIInitMode mode = CCIInitMode::TED_CCI_SR_AUTO;
  bool use_measurement_weight = true;
  bool use_square_root_qr = true;
  bool allow_normal_equation_debug_path = false;
  TEDCCIBackend condensation_backend = TEDCCIBackend::AUTO;
  TEDCCIBackend interface_backend = TEDCCIBackend::AUTO;
  bool enable_component_fallback = true;
  bool enable_message_versioning = true;
  int anchor_robot_id = 0;
  int anchor_pose_id = 0;
  int direct_interface_threshold = 2000;
  int max_separator_size = 500;
  int async_dd_max_iters = 200;
  double async_dd_rel_tol = 1e-8;
  bool async_dd_enable_coarse_correction = false;
  RIFTInterfaceBackend rift_interface_backend =
      RIFTInterfaceBackend::DIRECT_ORACLE;
  bool rift_use_rotation_multi_rhs = false;
  int rift_exact_max_separator_blocks_2d = 256;
  int rift_exact_max_separator_blocks_3d = 128;
  std::size_t rift_exact_max_message_bytes = 32ull * 1024ull * 1024ull;
  bool rift_forbid_direct_interface_solver = false;
  bool rift_forbid_global_interface_matrix = false;
  bool rift_forbid_collectives = false;
  bool verbose = false;
};

struct TEDCCIStats {
  int num_local_interior_vars = 0;
  int num_local_boundary_vars = 0;
  int num_interface_vars = 0;
  int num_condensed_rows = 0;
  int max_separator_size = 0;
  int num_factor_messages = 0;
  int num_solution_messages = 0;
  std::size_t bytes_sent_upward = 0;
  std::size_t bytes_sent_downward = 0;
  double rotation_equivalence_error = -1.0;
  double translation_equivalence_error = -1.0;
  int async_dd_iterations = 0;
  bool async_dd_converged = false;
  double async_dd_initial_residual = -1.0;
  double async_dd_final_residual = -1.0;
  int async_dd_coarse_correction_calls = 0;
  double local_qr_ms = 0.0;
  double interface_solve_ms = 0.0;
  CCIInitMode effective_mode = CCIInitMode::CENTRALIZED_CCI;
  CCIInitMode effective_rotation_mode = CCIInitMode::CENTRALIZED_CCI;
  CCIInitMode effective_translation_mode = CCIInitMode::CENTRALIZED_CCI;
  TEDCCIBackend effective_condensation_backend = TEDCCIBackend::AUTO;
  TEDCCIBackend effective_interface_backend = TEDCCIBackend::AUTO;
  int spqr_rank_deficient_fallbacks = 0;
  int peak_interface_cols = 0;
  std::size_t factor_dense_bytes_upward = 0;
  std::size_t factor_sparse_triplet_bytes_upward = 0;
  std::size_t factor_sparse_triplet_nonzeros = 0;
  RIFTInterfaceBackend rift_selected_backend =
      RIFTInterfaceBackend::DIRECT_ORACLE;
  int rift_num_cliques = 0;
  int rift_num_tree_edges = 0;
  int rift_max_clique_blocks = 0;
  int rift_max_separator_blocks = 0;
  std::size_t rift_estimated_message_bytes = 0;
  std::size_t rift_actual_message_bytes = 0;
  int rift_directed_messages_sent = 0;
  double rift_symbolic_ms = 0.0;
  double rift_message_qr_ms = 0.0;
  double rift_belief_solve_ms = 0.0;
  double rift_final_interface_residual = -1.0;
  bool rift_used_global_matrix = false;
  bool rift_used_direct_solver = false;
  bool rift_used_collective = false;
};

struct LinearFactorBlock {
  std::vector<PoseKey> keys;
  Matrix A;
  Vector b;
};

struct TEDCCIPartition {
  int local_robot_id = -1;
  std::vector<PoseKey> local_poses;
  std::vector<PoseKey> boundary_poses;
  std::vector<PoseKey> interior_poses;
  std::map<int, std::vector<PoseKey>> neighbor_boundary_poses;
};

struct CondensedFactor {
  int owner_robot_id = -1;
  std::uint64_t epoch = 0;
  FactorType factor_type = FactorType::ROTATION;
  std::vector<PoseKey> boundary_keys;
  Matrix Abar;
  Vector bbar;
};

struct BackSubstitutionCache {
  FactorType factor_type = FactorType::ROTATION;
  std::vector<PoseKey> interior_keys;
  std::vector<PoseKey> boundary_keys;
  Matrix R_II;
  Matrix R_IB;
  Vector d_I;
  TEDCCIBackend condensation_backend = TEDCCIBackend::DENSE_HOUSEHOLDER;
  std::vector<int> interior_column_permutation;
};

struct FactorMessage {
  int sender_robot_id = -1;
  int receiver_robot_id = -1;
  std::uint64_t epoch = 0;
  std::uint64_t topology_epoch = 0;
  FactorType factor_type = FactorType::ROTATION;
  bool is_cross_factor = false;
  bool is_component_factor = false;
  std::vector<int> component_robot_ids;
  std::vector<PoseKey> keys;
  Matrix A;
  Vector b;
  std::uint64_t checksum = 0;
};

struct SolutionMessage {
  int sender_robot_id = -1;
  int receiver_robot_id = -1;
  std::uint64_t epoch = 0;
  std::uint64_t topology_epoch = 0;
  FactorType factor_type = FactorType::ROTATION;
  bool is_component_solution = false;
  std::vector<int> component_robot_ids;
  std::vector<PoseKey> keys;
  Vector x;
  std::uint64_t checksum = 0;
};

std::uint64_t ComputeFactorMessageChecksum(const FactorMessage &msg);
std::uint64_t ComputeSolutionMessageChecksum(const SolutionMessage &msg);

class CCIFactorBuilder {
 public:
  static std::vector<LinearFactorBlock> BuildRotationFactors(
      int dimension, const std::vector<RelativeSEMeasurement> &measurements,
      const TEDCCIParams &params);

  static std::vector<LinearFactorBlock> BuildTranslationFactors(
      int dimension, const std::vector<RelativeSEMeasurement> &measurements,
      const std::map<PoseKey, Matrix> &projected_rotations,
      const TEDCCIParams &params);
};

class LocalQRCondensation {
 public:
  static void Condense(const std::vector<LinearFactorBlock> &factors,
                       const std::vector<PoseKey> &interior_keys,
                       const std::vector<PoseKey> &boundary_keys,
                       int block_dim, CondensedFactor *condensed_factor,
                       BackSubstitutionCache *cache);

  static void Condense(const std::vector<LinearFactorBlock> &factors,
                       const std::vector<PoseKey> &interior_keys,
                       const std::vector<PoseKey> &boundary_keys,
                       int block_dim, CondensedFactor *condensed_factor,
                       BackSubstitutionCache *cache,
                       const TEDCCIParams &params);

  static Vector BackSubstitute(const BackSubstitutionCache &cache,
                               const Vector &boundary_solution);
};

class BoundarySelector {
 public:
  static TEDCCIPartition SelectBoundaryVariables(
      int local_robot_id, const std::vector<PoseKey> &local_poses,
      const std::vector<RelativeSEMeasurement> &local_measurements,
      const std::vector<RelativeSEMeasurement> &shared_measurements,
      const TEDCCIParams &params);
};

class InterfaceDirectSolver {
 public:
  static Vector Solve(const std::vector<CondensedFactor> &condensed_factors,
                      const std::vector<LinearFactorBlock> &cross_factors,
                      int block_dim, const TEDCCIParams &params,
                      TEDCCIStats *stats = nullptr);
};

class InterfaceAsyncDDSolver {
 public:
  static Vector Solve(const std::vector<CondensedFactor> &condensed_factors,
                      const std::vector<LinearFactorBlock> &cross_factors,
                      int block_dim, const TEDCCIParams &params,
                      TEDCCIStats *stats = nullptr);

  static Vector SolveWithSimulatedNetwork(
      const std::vector<CondensedFactor> &condensed_factors,
      const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
      const TEDCCIParams &params, int max_delay_rounds,
      bool reverse_delivery_order, TEDCCIStats *stats = nullptr);
};

struct SeparatorNode {
  int node_id = -1;
  int parent_id = -1;
  int leader_robot_id = -1;
  std::vector<int> children;
  std::vector<int> robot_ids;
  std::vector<PoseKey> separator_keys;
};

struct SeparatorResendRequest {
  int sender_node_id = -1;
  int sender_robot_id = -1;
  int receiver_node_id = -1;
  int receiver_robot_id = -1;
};

struct SeparatorNodeOwnership {
  int node_id = -1;
  int owner_robot_id = -1;
  std::uint64_t topology_epoch = 0;
};

struct SeparatorFailureRecoveryPlan {
  int failed_node_id = -1;
  int replacement_robot_id = -1;
  std::uint64_t topology_epoch = 0;
  std::vector<SeparatorResendRequest> resend_requests;
};

class AsyncFactorExchange;

class SeparatorTree {
 public:
  static SeparatorTree BuildFromRobotGraph(
      const std::vector<TEDCCIPartition> &partitions,
      const TEDCCIParams &params);

  Vector SolveHierarchical(
      const std::vector<CondensedFactor> &leaf_factors,
      const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
      TEDCCIStats *stats = nullptr) const;

  Vector SolveHierarchical(
      const std::vector<CondensedFactor> &leaf_factors,
      const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
      const TEDCCIParams &params, TEDCCIStats *stats = nullptr) const;

  SeparatorFailureRecoveryPlan PlanParentNodeFailureRecovery(
      int failed_node_id) const;
  std::vector<FactorMessage> MakeRecoveryFactorMessages(
      const SeparatorFailureRecoveryPlan &plan,
      const std::map<int, CondensedFactor> &latest_node_factors,
      std::uint64_t epoch, std::uint64_t topology_epoch = 0) const;

  const std::vector<SeparatorNode> &nodes() const { return nodes_; }

 private:
  std::vector<SeparatorNode> nodes_;
  int root_node_id_ = -1;
};

class SeparatorTreeRuntime {
 public:
  explicit SeparatorTreeRuntime(const SeparatorTree &tree);

  std::vector<SeparatorNodeOwnership> ElectOwners(
      const std::set<int> &live_robot_ids,
      std::uint64_t topology_epoch) const;

  int ParentOwnerForChildNode(
      int child_node_id,
      const std::vector<SeparatorNodeOwnership> &ownership) const;

  SeparatorFailureRecoveryPlan PlanParentOwnerChangeResend(
      const std::vector<SeparatorNodeOwnership> &previous_ownership,
      const std::vector<SeparatorNodeOwnership> &current_ownership,
      int parent_node_id) const;

 private:
  std::vector<SeparatorNode> nodes_;
};

class SeparatorTreeMessageController {
 public:
  explicit SeparatorTreeMessageController(const SeparatorTree &tree);

  std::vector<SeparatorNodeOwnership> ElectOwners(
      const std::set<int> &live_robot_ids,
      std::uint64_t topology_epoch) const;

  void UpdateLatestNodeFactor(int node_id, const CondensedFactor &factor);

  std::vector<FactorMessage> MakeOwnerChangeRecoveryMessages(
      const std::vector<SeparatorNodeOwnership> &previous_ownership,
      const std::vector<SeparatorNodeOwnership> &current_ownership,
      int parent_node_id, std::uint64_t epoch) const;

  std::size_t SendOwnerChangeRecoveryMessages(
      const std::vector<SeparatorNodeOwnership> &previous_ownership,
      const std::vector<SeparatorNodeOwnership> &current_ownership,
      int parent_node_id, std::uint64_t epoch,
      AsyncFactorExchange *exchange) const;

 private:
  SeparatorTree tree_;
  SeparatorTreeRuntime runtime_;
  std::map<int, CondensedFactor> latest_node_factors_;
};

class AsyncFactorExchange {
 public:
  virtual ~AsyncFactorExchange() = default;

  virtual void SendFactor(const FactorMessage &msg) = 0;
  virtual void SendSolution(const SolutionMessage &msg) = 0;

  virtual std::vector<FactorMessage> PollFactorMessages(
      int receiver_robot_id) = 0;
  virtual std::vector<SolutionMessage> PollSolutionMessages(
      int receiver_robot_id) = 0;

  virtual void MarkLinkDown(int robot_a, int robot_b) = 0;
  virtual void MarkLinkUp(int robot_a, int robot_b) = 0;
};

class InProcessAsyncFactorExchange : public AsyncFactorExchange {
 public:
  explicit InProcessAsyncFactorExchange(bool queue_when_link_down = true);

  void SendFactor(const FactorMessage &msg) override;
  void SendSolution(const SolutionMessage &msg) override;

  std::vector<FactorMessage> PollFactorMessages(
      int receiver_robot_id) override;
  std::vector<SolutionMessage> PollSolutionMessages(
      int receiver_robot_id) override;

  void MarkLinkDown(int robot_a, int robot_b) override;
  void MarkLinkUp(int robot_a, int robot_b) override;

 private:
  struct MessageKey {
    int sender_robot_id = -1;
    FactorType factor_type = FactorType::ROTATION;
    std::uint64_t topology_epoch = 0;
    bool is_cross_factor = false;
    bool is_component_message = false;
    std::vector<int> component_robot_ids;
    std::vector<PoseKey> keys;

    bool operator<(const MessageKey &other) const;
  };

  bool queue_when_link_down_ = true;
  std::set<std::pair<int, int>> down_links_;
  std::map<int, std::map<MessageKey, FactorMessage>> factor_inbox_;
  std::map<int, std::map<MessageKey, SolutionMessage>> solution_inbox_;
  std::vector<FactorMessage> queued_factors_;
  std::vector<SolutionMessage> queued_solutions_;

  bool IsLinkDown(int sender, int receiver) const;
};

class TEDCCIOrchestrator {
 public:
  TEDCCIOrchestrator(int local_robot_id, const TEDCCIParams &params,
                     AsyncFactorExchange *exchange);

  void SetProblemMetadata(int dimension, int num_poses,
                          const TEDCCIPartition &local_partition,
                          const std::vector<int> &expected_robot_ids);

  void SetLocalMeasurements(
      const std::vector<RelativeSEMeasurement> &local_measurements,
      const std::vector<RelativeSEMeasurement> &shared_measurements);

  void StartEpoch(std::uint64_t epoch);
  void ProcessIncomingMessages();

  bool HasComponentConsistentSolution() const;
  bool HasGlobalConsistentSolution() const;

  Matrix GetLocalOnlyInitialization() const;
  Matrix GetComponentConsistentInitialization() const;
  Matrix GetGlobalConsistentInitialization() const;

 private:
  void ValidateReady() const;
  Matrix ComputeLocalOnlyInitialization() const;
  void BuildAndBroadcastRotationFactor();
  void TrySolveRotationAndBroadcastTranslation();
  void TrySolveTranslationAndBroadcastSolution();
  void TryAssembleGlobalSolution();
  void TrySolveComponentAndBroadcast();
  void TryAssembleComponentSolution();
  void BroadcastFactor(const CondensedFactor &factor);
  void BuildAndBroadcastRotationCrossFactor();
  void BuildAndBroadcastTranslationCrossFactor();
  void BroadcastCrossFactor(FactorType factor_type,
                            const LinearFactorBlock &factor);
  void BroadcastComponentFactor(const CondensedFactor &factor,
                                const std::vector<int> &component_robots);
  void BroadcastComponentCrossFactor(
      FactorType factor_type, const LinearFactorBlock &factor,
      const std::vector<int> &component_robots);
  void BroadcastLocalSolution(FactorType factor_type,
                              const std::vector<PoseKey> &keys,
                              const Vector &x);
  void BroadcastComponentLocalSolution(
      FactorType factor_type, const std::vector<PoseKey> &keys,
      const Vector &x, const std::vector<int> &component_robots);
  void StoreFactorMessage(const FactorMessage &msg);
  void StoreSolutionBlocks(const SolutionMessage &msg);
  bool HasAllFactors(FactorType factor_type) const;
  bool HasAllCrossFactors(FactorType factor_type) const;
  std::vector<int> ReachableComponentRobotIds() const;
  bool FactorKeysInsideRobots(
      const LinearFactorBlock &factor,
      const std::vector<int> &component_robots) const;
  bool HasComponentTranslationFactors() const;
  bool HasComponentTranslationCrossFactors() const;
  bool HasComponentSolutionSenders(FactorType factor_type) const;
  std::vector<CondensedFactor> FactorsForType(FactorType factor_type) const;
  std::vector<LinearFactorBlock> CrossFactorsForType(
      FactorType factor_type) const;
  std::vector<CondensedFactor> ComponentRotationFactors() const;
  std::vector<LinearFactorBlock> ComponentRotationCrossFactors() const;
  std::vector<CondensedFactor> ComponentTranslationFactors() const;
  std::vector<LinearFactorBlock> ComponentTranslationCrossFactors() const;
  LinearFactorBlock BuildOwnedCrossFactor(FactorType factor_type) const;
  LinearFactorBlock BuildOwnedCrossFactorForRobots(
      FactorType factor_type, const std::vector<int> &component_robots,
      const std::map<PoseKey, Matrix> &rotations) const;

  int local_robot_id_ = -1;
  TEDCCIParams params_;
  AsyncFactorExchange *exchange_ = nullptr;

  bool metadata_set_ = false;
  bool measurements_set_ = false;
  int dimension_ = 0;
  int num_poses_ = 0;
  TEDCCIPartition local_partition_;
  std::vector<int> expected_robot_ids_;

  std::vector<RelativeSEMeasurement> local_measurements_;
  std::vector<RelativeSEMeasurement> shared_measurements_;
  std::uint64_t epoch_ = 0;

  bool rotation_factor_sent_ = false;
  bool rotation_cross_factor_sent_ = false;
  bool rotation_solved_ = false;
  bool translation_factor_sent_ = false;
  bool translation_cross_factor_sent_ = false;
  bool translation_solved_ = false;
  bool local_only_available_ = false;
  bool component_available_ = false;
  bool global_available_ = false;
  bool component_rotation_solved_ = false;
  bool component_translation_factor_sent_ = false;
  bool component_translation_cross_factor_sent_ = false;
  bool component_translation_solved_ = false;

  Matrix local_only_solution_;
  Matrix component_solution_;
  Matrix global_solution_;

  CondensedFactor local_rotation_factor_;
  CondensedFactor local_translation_factor_;
  LinearFactorBlock local_rotation_cross_factor_;
  LinearFactorBlock local_translation_cross_factor_;
  BackSubstitutionCache rotation_cache_;
  BackSubstitutionCache translation_cache_;

  std::map<int, CondensedFactor> rotation_factors_;
  std::map<int, CondensedFactor> translation_factors_;
  std::map<int, LinearFactorBlock> rotation_cross_factors_;
  std::map<int, LinearFactorBlock> translation_cross_factors_;
  std::map<PoseKey, Vector> relaxed_rotation_blocks_;
  std::map<PoseKey, Matrix> projected_rotations_;
  std::map<PoseKey, Vector> translation_blocks_;
  std::map<PoseKey, Vector> received_rotation_solution_blocks_;
  std::map<PoseKey, Vector> received_translation_solution_blocks_;
  std::set<int> rotation_solution_senders_;
  std::set<int> translation_solution_senders_;

  std::vector<int> active_component_robot_ids_;
  CondensedFactor local_component_translation_factor_;
  LinearFactorBlock local_component_translation_cross_factor_;
  BackSubstitutionCache component_translation_cache_;
  std::map<PoseKey, Vector> component_relaxed_rotation_blocks_;
  std::map<PoseKey, Matrix> component_projected_rotations_;
  std::map<PoseKey, Vector> component_translation_blocks_;
  std::map<int, std::pair<std::vector<int>, CondensedFactor>>
      component_translation_factors_;
  std::map<int, std::pair<std::vector<int>, LinearFactorBlock>>
      component_translation_cross_factors_;
  std::map<PoseKey, Vector> component_received_rotation_solution_blocks_;
  std::map<PoseKey, Vector> component_received_translation_solution_blocks_;
  std::map<int, std::vector<int>> component_rotation_solution_senders_;
  std::map<int, std::vector<int>> component_translation_solution_senders_;
};

class TEDCCISolver {
 public:
  static Matrix Initialize(
      int dimension, int num_poses,
      const std::vector<RelativeSEMeasurement> &measurements,
      const std::vector<TEDCCIPartition> &partitions,
      const TEDCCIParams &params, TEDCCIStats *stats = nullptr);

  static Matrix InitializeSingleProcessDirect(
      int dimension, int num_poses,
      const std::vector<RelativeSEMeasurement> &measurements,
      const std::vector<TEDCCIPartition> &partitions,
      const TEDCCIParams &params, TEDCCIStats *stats = nullptr);
};

}  // namespace DPGO

#endif
