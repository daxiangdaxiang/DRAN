#include <DPGO/RIFTIF.h>

#include <Eigen/QR>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace DPGO {
namespace {

template <typename Fn>
double elapsedMilliseconds(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto finish = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::vector<InterfaceKey> sortedUnique(std::vector<InterfaceKey> keys) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

bool containsKey(const std::vector<InterfaceKey> &keys,
                 const InterfaceKey &key) {
  return std::binary_search(keys.begin(), keys.end(), key);
}

bool containsScope(const std::vector<InterfaceKey> &clique,
                   const std::vector<InterfaceKey> &scope) {
  for (const auto &key : scope) {
    if (!containsKey(clique, key)) {
      return false;
    }
  }
  return true;
}

std::vector<InterfaceKey> intersectKeys(const std::vector<InterfaceKey> &a,
                                        const std::vector<InterfaceKey> &b) {
  std::vector<InterfaceKey> result;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
  return result;
}

std::vector<InterfaceKey> subtractKeys(const std::vector<InterfaceKey> &keys,
                                       const std::vector<InterfaceKey> &drop) {
  std::vector<InterfaceKey> result;
  std::set_difference(keys.begin(), keys.end(), drop.begin(), drop.end(),
                      std::back_inserter(result));
  return result;
}

int keyOffset(const std::vector<InterfaceKey> &keys, const InterfaceKey &key,
              int blockDim) {
  const auto it = std::find(keys.begin(), keys.end(), key);
  if (it == keys.end() || !(*it == key)) {
    throw std::invalid_argument("RIFT-IF key not present in variable list");
  }
  return static_cast<int>(std::distance(keys.begin(), it)) * blockDim;
}

Matrix vectorAsRhsMatrix(const Vector &b) {
  Matrix B(b.rows(), 1);
  B.col(0) = b;
  return B;
}

void appendExpandedRows(const std::vector<InterfaceKey> &targetKeys,
                        int blockDim, int rhsDim,
                        const std::vector<InterfaceKey> &factorKeys,
                        const Matrix &factorA, const Matrix &factorB,
                        Matrix *A, Matrix *B, int *rowOffset) {
  if (factorB.cols() != rhsDim) {
    throw std::invalid_argument("RIFT-IF factor RHS dimension mismatch");
  }
  if (factorA.rows() != factorB.rows()) {
    throw std::invalid_argument("RIFT-IF factor A/B row mismatch");
  }
  if (factorA.cols() != static_cast<int>(factorKeys.size()) * blockDim) {
    throw std::invalid_argument("RIFT-IF factor A column mismatch");
  }
  const int rows = factorA.rows();
  for (std::size_t k = 0; k < factorKeys.size(); ++k) {
    const int dst = keyOffset(targetKeys, factorKeys[k], blockDim);
    const int src = static_cast<int>(k) * blockDim;
    A->block(*rowOffset, dst, rows, blockDim) =
        factorA.block(0, src, rows, blockDim);
  }
  B->block(*rowOffset, 0, rows, rhsDim) = factorB;
  *rowOffset += rows;
}

std::pair<Matrix, Matrix> stackFactorsForClique(
    const std::vector<InterfaceKey> &cliqueKeys, int blockDim, int rhsDim,
    const std::vector<const InterfaceFactor *> &factors,
    const std::vector<const RIFTMessage *> &messages) {
  int rows = 0;
  for (const auto *factor : factors) {
    rows += factor->A.rows();
  }
  for (const auto *message : messages) {
    rows += message->R.rows();
  }

  Matrix A = Matrix::Zero(rows,
                          static_cast<int>(cliqueKeys.size()) * blockDim);
  Matrix B = Matrix::Zero(rows, rhsDim);
  int rowOffset = 0;
  for (const auto *factor : factors) {
    appendExpandedRows(cliqueKeys, blockDim, rhsDim, factor->scope, factor->A,
                       factor->B, &A, &B, &rowOffset);
  }
  for (const auto *message : messages) {
    appendExpandedRows(cliqueKeys, blockDim, rhsDim, message->separator_keys,
                       message->R, message->D, &A, &B, &rowOffset);
  }
  return {A, B};
}

std::pair<Matrix, Matrix> compressRowsByQR(const Matrix &A,
                                           const Matrix &B) {
  if (A.cols() == 0) {
    return {Matrix::Zero(0, 0), Matrix::Zero(0, B.cols())};
  }
  if (A.rows() == 0) {
    return {Matrix::Zero(0, A.cols()), Matrix::Zero(0, B.cols())};
  }
  Eigen::HouseholderQR<Matrix> qr(A);
  const Matrix transformedB = qr.householderQ().transpose() * B;
  const Matrix Rfull = qr.matrixQR()
                           .topLeftCorner(A.rows(), A.cols())
                           .template triangularView<Eigen::Upper>();
  const int rows = std::min(A.rows(), A.cols());
  return {Rfull.topRows(rows), transformedB.topRows(rows)};
}

RIFTMessage computeDirectedMessage(
    const InterfaceProblem &problem, const InterfaceCliqueTree &tree,
    const std::map<RIFTFactorId, InterfaceFactor> &factorById,
    RIFTCliqueId src, RIFTCliqueId dst,
    const std::map<DirectedCliqueEdge, RIFTMessage> &messageByEdge) {
  const InterfaceClique &srcClique = tree.cliques.at(src);
  const InterfaceClique &dstClique = tree.cliques.at(dst);
  const std::vector<InterfaceKey> separator =
      intersectKeys(srcClique.variables, dstClique.variables);
  const std::vector<InterfaceKey> eliminated =
      subtractKeys(srcClique.variables, separator);
  std::vector<InterfaceKey> orderedKeys = eliminated;
  orderedKeys.insert(orderedKeys.end(), separator.begin(), separator.end());

  std::vector<const InterfaceFactor *> factors;
  factors.reserve(srcClique.assigned_factors.size());
  for (const RIFTFactorId id : srcClique.assigned_factors) {
    factors.push_back(&factorById.at(id));
  }

  std::vector<const RIFTMessage *> incoming;
  for (const RIFTCliqueId neighbor : srcClique.neighbors) {
    if (neighbor == dst) {
      continue;
    }
    const auto it = messageByEdge.find(DirectedCliqueEdge{neighbor, src});
    if (it == messageByEdge.end()) {
      throw std::runtime_error("RIFT-IF directed message missing dependency");
    }
    incoming.push_back(&it->second);
  }

  const auto stacked = stackFactorsForClique(orderedKeys, problem.block_dim,
                                             problem.rhs_dim, factors,
                                             incoming);
  const Matrix &A = stacked.first;
  const Matrix &B = stacked.second;
  const int elimDim =
      static_cast<int>(eliminated.size()) * problem.block_dim;
  const int sepDim = static_cast<int>(separator.size()) * problem.block_dim;

  Matrix residualA;
  Matrix residualB;
  if (elimDim > 0 && A.rows() > 0) {
    const Matrix AE = A.leftCols(elimDim);
    Matrix rest(A.rows(), sepDim + problem.rhs_dim);
    if (sepDim > 0) {
      rest.leftCols(sepDim) = A.middleCols(elimDim, sepDim);
    }
    rest.rightCols(problem.rhs_dim) = B;
    Eigen::ColPivHouseholderQR<Matrix> qr(AE);
    const Matrix transformed = qr.householderQ().transpose() * rest;
    const int totalRows = static_cast<int>(A.rows());
    const int start = std::min(static_cast<int>(qr.rank()), totalRows);
    if (sepDim == 0) {
      residualA = Matrix::Zero(totalRows - start, 0);
    } else {
      residualA = transformed.block(start, 0, totalRows - start, sepDim);
    }
    residualB = transformed.block(start, sepDim, totalRows - start,
                                  problem.rhs_dim);
  } else {
    if (sepDim == 0) {
      residualA = Matrix::Zero(A.rows(), 0);
    } else {
      residualA = A.rightCols(sepDim);
    }
    residualB = B;
  }

  const auto compressed = compressRowsByQR(residualA, residualB);

  RIFTMessage msg;
  msg.header.stage = problem.stage;
  msg.header.src = src;
  msg.header.dst = dst;
  msg.separator_keys = separator;
  msg.R = compressed.first;
  msg.D = compressed.second;
  return msg;
}

std::size_t messageBytes(const RIFTMessage &message) {
  return sizeof(int) * 4u +
         message.separator_keys.size() * 2u * sizeof(int) +
         static_cast<std::size_t>(message.R.rows()) *
             static_cast<std::size_t>(message.R.cols()) * sizeof(double) +
         static_cast<std::size_t>(message.D.rows()) *
             static_cast<std::size_t>(message.D.cols()) * sizeof(double);
}

double interfaceResidualNorm(const InterfaceProblem &problem,
                             const Matrix &solution) {
  if (solution.rows() !=
          static_cast<int>(problem.variables.size()) * problem.block_dim ||
      solution.cols() != problem.rhs_dim) {
    throw std::invalid_argument("RIFT-IF solution matrix dimension mismatch");
  }
  double squared = 0.0;
  for (const auto &factor : problem.factors) {
    Matrix X(factor.A.cols(), problem.rhs_dim);
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const int globalOffset =
          keyOffset(problem.variables, factor.scope[k], problem.block_dim);
      const int localOffset = static_cast<int>(k) * problem.block_dim;
      X.block(localOffset, 0, problem.block_dim, problem.rhs_dim) =
          solution.block(globalOffset, 0, problem.block_dim, problem.rhs_dim);
    }
    const Matrix residual = factor.A * X - factor.B;
    squared += residual.squaredNorm();
  }
  return std::sqrt(squared);
}

std::size_t estimatedMessageBytesForSeparator(std::size_t separatorBlocks,
                                              int blockDim, int rhsDim) {
  const std::size_t sepDim = separatorBlocks * static_cast<std::size_t>(blockDim);
  return sizeof(int) * 4u + separatorBlocks * 2u * sizeof(int) +
         sepDim * sepDim * sizeof(double) +
         sepDim * static_cast<std::size_t>(rhsDim) * sizeof(double);
}

}  // namespace

void DecentralizationGuard::RecordGlobalInterfaceMatrixConstruction() {
  used_global_matrix_ = true;
}

void DecentralizationGuard::RecordDirectInterfaceSolverCall() {
  used_direct_solver_ = true;
}

void DecentralizationGuard::RecordCollectiveCall(const std::string &) {
  used_collective_ = true;
}

void DecentralizationGuard::RecordMessage(int, int, std::size_t bytes) {
  ++message_count_;
  message_bytes_ += bytes;
}

void DecentralizationGuard::RecordFactorTransfer(int, int dst,
                                                 RIFTFactorId id) {
  factors_by_receiver_[dst].insert(id);
}

void DecentralizationGuard::AssertNoGlobalInterfaceMatrixConstructed() const {
  if (used_global_matrix_) {
    throw std::runtime_error("RIFT-IF guard: global interface matrix used");
  }
}

void DecentralizationGuard::AssertNoDirectSolverCalled() const {
  if (used_direct_solver_) {
    throw std::runtime_error("RIFT-IF guard: direct interface solver used");
  }
}

void DecentralizationGuard::AssertNoCollectiveCommunication() const {
  if (used_collective_) {
    throw std::runtime_error("RIFT-IF guard: collective communication used");
  }
}

void DecentralizationGuard::AssertNoNodeReceivedAllFactors(
    int num_total_factors) const {
  if (num_total_factors <= 0) {
    return;
  }
  for (const auto &entry : factors_by_receiver_) {
    if (static_cast<int>(entry.second.size()) >= num_total_factors &&
        num_total_factors > 1) {
      throw std::runtime_error(
          "RIFT-IF guard: one node received all interface factors");
    }
  }
}

InterfaceProblem InterfaceProblemBuilder::BuildFromInterfaceFactors(
    FactorType stage, int dimension, int block_dim, int rhs_dim,
    const std::vector<InterfaceFactor> &factors) {
  if (dimension <= 0 || block_dim <= 0 || rhs_dim <= 0) {
    throw std::invalid_argument("RIFT-IF problem dimensions must be positive");
  }
  InterfaceProblem problem;
  problem.stage = stage;
  problem.dimension = dimension;
  problem.block_dim = block_dim;
  problem.rhs_dim = rhs_dim;
  std::vector<InterfaceKey> variables;
  for (const auto &factor : factors) {
    if (factor.stage != stage) {
      throw std::invalid_argument("RIFT-IF factor stage mismatch");
    }
    if (factor.A.rows() != factor.B.rows()) {
      throw std::invalid_argument("RIFT-IF factor A/B row mismatch");
    }
    if (factor.B.cols() != rhs_dim) {
      throw std::invalid_argument("RIFT-IF factor RHS dimension mismatch");
    }
    if (factor.A.cols() !=
        static_cast<int>(factor.scope.size()) * block_dim) {
      throw std::invalid_argument("RIFT-IF factor column mismatch");
    }
    variables.insert(variables.end(), factor.scope.begin(),
                     factor.scope.end());
    problem.factors.push_back(factor);
  }
  problem.variables = sortedUnique(std::move(variables));
  return problem;
}

InterfaceProblem InterfaceProblemBuilder::BuildFromTEDFactors(
    FactorType stage, int dimension, int block_dim,
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors,
    bool use_rotation_multi_rhs) {
  if (block_dim <= 0) {
    throw std::invalid_argument("RIFT-IF block dimension must be positive");
  }

  InterfaceProblem problem;
  problem.stage = stage;
  problem.dimension = dimension;
  problem.block_dim = block_dim;
  problem.rhs_dim = 1;
  if (stage == FactorType::ROTATION && use_rotation_multi_rhs) {
    throw std::invalid_argument(
        "RIFT-IF rotation multi-RHS must be built from row-wise factors");
  }

  RIFTFactorId nextId = 0;
  std::vector<InterfaceKey> variables;
  for (const auto &condensed : condensed_factors) {
    InterfaceFactor factor;
    factor.id = nextId++;
    factor.stage = stage;
    factor.scope = condensed.boundary_keys;
    factor.A = condensed.Abar;
    factor.B = vectorAsRhsMatrix(condensed.bbar);
    factor.owner_robot = condensed.owner_robot_id;
    factor.graph_epoch = condensed.epoch;
    if (factor.A.cols() != static_cast<int>(factor.scope.size()) * block_dim) {
      throw std::invalid_argument("RIFT-IF condensed factor column mismatch");
    }
    variables.insert(variables.end(), factor.scope.begin(), factor.scope.end());
    problem.factors.push_back(std::move(factor));
  }
  for (const auto &cross : cross_factors) {
    InterfaceFactor factor;
    factor.id = nextId++;
    factor.stage = stage;
    factor.scope = cross.keys;
    factor.A = cross.A;
    factor.B = vectorAsRhsMatrix(cross.b);
    factor.owner_robot =
        factor.scope.empty() ? -1 : factor.scope.front().robot_id;
    if (factor.A.cols() != static_cast<int>(factor.scope.size()) * block_dim) {
      throw std::invalid_argument("RIFT-IF cross factor column mismatch");
    }
    variables.insert(variables.end(), factor.scope.begin(), factor.scope.end());
    problem.factors.push_back(std::move(factor));
  }
  problem.variables = sortedUnique(std::move(variables));
  return problem;
}

InterfaceFactor BuildRotationMultiRHSFactorFromVectorized(
    RIFTFactorId id, const std::vector<PoseKey> &scope,
    const Matrix &vectorized_A, const Vector &vectorized_b, int dimension,
    int owner_robot, std::uint64_t graph_epoch) {
  if (dimension <= 0) {
    throw std::invalid_argument("RIFT-IF multi-RHS dimension must be positive");
  }
  if (vectorized_A.rows() != vectorized_b.rows()) {
    throw std::invalid_argument("RIFT-IF vectorized factor A/b row mismatch");
  }
  if (vectorized_A.rows() % dimension != 0) {
    throw std::invalid_argument(
        "RIFT-IF vectorized rotation rows are not divisible by dimension");
  }
  if (vectorized_A.cols() !=
      static_cast<int>(scope.size()) * dimension * dimension) {
    throw std::invalid_argument(
        "RIFT-IF vectorized rotation factor column mismatch");
  }

  const int rows = vectorized_A.rows() / dimension;
  InterfaceFactor factor;
  factor.id = id;
  factor.stage = FactorType::ROTATION;
  factor.scope = scope;
  factor.A =
      Matrix::Zero(rows, static_cast<int>(scope.size()) * dimension);
  factor.B = Matrix::Zero(rows, dimension);
  factor.owner_robot = owner_robot;
  factor.graph_epoch = graph_epoch;

  constexpr double kCoeffTol = 1e-8;
  for (int rhs = 0; rhs < dimension; ++rhs) {
    for (int row = 0; row < rows; ++row) {
      const int vecRow = rhs + row * dimension;
      factor.B(row, rhs) = vectorized_b(vecRow);
      for (std::size_t keyIndex = 0; keyIndex < scope.size(); ++keyIndex) {
        for (int col = 0; col < dimension; ++col) {
          const int multiCol =
              static_cast<int>(keyIndex) * dimension + col;
          const int vecCol = static_cast<int>(keyIndex) * dimension *
                                 dimension +
                             rhs + col * dimension;
          const double coeff = vectorized_A(vecRow, vecCol);
          for (int otherRhs = 0; otherRhs < dimension; ++otherRhs) {
            if (otherRhs == rhs) {
              continue;
            }
            const int offRhsVecCol = static_cast<int>(keyIndex) * dimension *
                                         dimension +
                                     otherRhs + col * dimension;
            const double offCoeff = vectorized_A(vecRow, offRhsVecCol);
            if (std::abs(offCoeff) > kCoeffTol) {
              throw std::invalid_argument(
                  "RIFT-IF vectorized rotation factor has off-row coupling");
            }
          }
          if (rhs == 0) {
            factor.A(row, multiCol) = coeff;
          } else if (std::abs(factor.A(row, multiCol) - coeff) >
                     kCoeffTol *
                         std::max(1.0, std::abs(factor.A(row, multiCol)))) {
            throw std::invalid_argument(
                "RIFT-IF vectorized rotation factor is not row-separable");
          }
        }
      }
    }
  }
  return factor;
}

InterfaceFactor CondenseMultiRHSFactors(
    RIFTFactorId id, FactorType stage, int dimension,
    const std::vector<InterfaceFactor> &factors,
    const std::vector<PoseKey> &interior_keys,
    const std::vector<PoseKey> &boundary_keys, int owner_robot,
    std::uint64_t graph_epoch) {
  if (dimension <= 0) {
    throw std::invalid_argument("RIFT-IF multi-RHS dimension must be positive");
  }
  const std::vector<InterfaceKey> interior = sortedUnique(interior_keys);
  const std::vector<InterfaceKey> boundary = sortedUnique(boundary_keys);
  std::vector<InterfaceKey> orderedKeys = interior;
  orderedKeys.insert(orderedKeys.end(), boundary.begin(), boundary.end());

  std::vector<const InterfaceFactor *> factorPtrs;
  factorPtrs.reserve(factors.size());
  for (const auto &factor : factors) {
    if (factor.stage != stage) {
      throw std::invalid_argument("RIFT-IF multi-RHS factor stage mismatch");
    }
    if (factor.B.cols() != dimension) {
      throw std::invalid_argument("RIFT-IF multi-RHS RHS dimension mismatch");
    }
    factorPtrs.push_back(&factor);
  }
  const auto stacked =
      stackFactorsForClique(orderedKeys, dimension, dimension, factorPtrs, {});
  const Matrix &A = stacked.first;
  const Matrix &B = stacked.second;
  const int interiorDim = static_cast<int>(interior.size()) * dimension;
  const int boundaryDim = static_cast<int>(boundary.size()) * dimension;

  InterfaceFactor condensed;
  condensed.id = id;
  condensed.stage = stage;
  condensed.scope = boundary;
  condensed.owner_robot = owner_robot;
  condensed.graph_epoch = graph_epoch;

  if (interiorDim == 0) {
    if (boundaryDim == 0) {
      condensed.A = Matrix::Zero(A.rows(), 0);
    } else {
      condensed.A = A.rightCols(boundaryDim);
    }
    condensed.B = B;
    return condensed;
  }
  if (A.rows() < interiorDim) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS condensation has underdetermined interior block");
  }

  const Matrix AI = A.leftCols(interiorDim);
  Matrix rest(A.rows(), boundaryDim + dimension);
  if (boundaryDim > 0) {
    rest.leftCols(boundaryDim) = A.middleCols(interiorDim, boundaryDim);
  }
  rest.rightCols(dimension) = B;

  Eigen::ColPivHouseholderQR<Matrix> qr(AI);
  if (static_cast<int>(qr.rank()) < interiorDim) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS condensation requires full-rank interior block");
  }
  const Matrix transformed = qr.householderQ().transpose() * rest;
  const int residualRows = A.rows() - interiorDim;
  if (boundaryDim == 0) {
    condensed.A = Matrix::Zero(residualRows, 0);
  } else {
    condensed.A =
        transformed.block(interiorDim, 0, residualRows, boundaryDim);
  }
  condensed.B =
      transformed.block(interiorDim, boundaryDim, residualRows, dimension);
  return condensed;
}

InterfaceCliqueTree InterfaceCliqueTreeBuilder::Build(
    const InterfaceProblem &problem, const RIFTParams &params) {
  (void)params;
  InterfaceCliqueTree tree;
  const int n = static_cast<int>(problem.variables.size());
  if (n == 0) {
    return tree;
  }

  std::map<InterfaceKey, int> indexOf;
  for (int i = 0; i < n; ++i) {
    indexOf[problem.variables[i]] = i;
  }

  std::vector<std::set<int>> adjacency(n);
  for (const auto &factor : problem.factors) {
    std::vector<int> scope;
    for (const auto &key : factor.scope) {
      scope.push_back(indexOf.at(key));
    }
    for (std::size_t i = 0; i < scope.size(); ++i) {
      for (std::size_t j = i + 1; j < scope.size(); ++j) {
        adjacency[scope[i]].insert(scope[j]);
        adjacency[scope[j]].insert(scope[i]);
      }
    }
  }

  std::vector<bool> active(n, true);
  std::vector<std::vector<int>> rawCliques;
  for (int step = 0; step < n; ++step) {
    int best = -1;
    int bestDegree = std::numeric_limits<int>::max();
    for (int v = 0; v < n; ++v) {
      if (!active[v]) {
        continue;
      }
      int degree = 0;
      for (const int nb : adjacency[v]) {
        if (active[nb]) {
          ++degree;
        }
      }
      if (degree < bestDegree || (degree == bestDegree && v < best)) {
        best = v;
        bestDegree = degree;
      }
    }
    if (best < 0) {
      break;
    }

    std::vector<int> clique = {best};
    for (const int nb : adjacency[best]) {
      if (active[nb]) {
        clique.push_back(nb);
      }
    }
    std::sort(clique.begin(), clique.end());
    rawCliques.push_back(clique);

    for (std::size_t i = 1; i < clique.size(); ++i) {
      for (std::size_t j = i + 1; j < clique.size(); ++j) {
        adjacency[clique[i]].insert(clique[j]);
        adjacency[clique[j]].insert(clique[i]);
      }
    }
    active[best] = false;
  }

  std::vector<std::vector<int>> maximal;
  for (const auto &candidate : rawCliques) {
    bool subset = false;
    for (const auto &other : rawCliques) {
      if (&candidate == &other || candidate.size() > other.size()) {
        continue;
      }
      if (std::includes(other.begin(), other.end(), candidate.begin(),
                        candidate.end()) &&
          candidate != other) {
        subset = true;
        break;
      }
    }
    if (!subset &&
        std::find(maximal.begin(), maximal.end(), candidate) == maximal.end()) {
      maximal.push_back(candidate);
    }
  }
  if (maximal.empty()) {
    maximal.push_back({0});
  }

  for (std::size_t c = 0; c < maximal.size(); ++c) {
    InterfaceClique clique;
    clique.id = static_cast<RIFTCliqueId>(c);
    for (const int variableIndex : maximal[c]) {
      clique.variables.push_back(problem.variables[variableIndex]);
    }
    std::map<int, int> robotCounts;
    for (const auto &key : clique.variables) {
      ++robotCounts[key.robot_id];
    }
    int bestRobot = clique.variables.empty() ? -1 : clique.variables.front().robot_id;
    int bestCount = -1;
    for (const auto &entry : robotCounts) {
      if (entry.second > bestCount ||
          (entry.second == bestCount && entry.first < bestRobot)) {
        bestRobot = entry.first;
        bestCount = entry.second;
      }
    }
    clique.host_robot = bestRobot;
    tree.cliques.push_back(std::move(clique));
  }

  struct WeightedEdge {
    int weight = 0;
    int a = -1;
    int b = -1;
    std::vector<InterfaceKey> separator;
  };
  std::vector<WeightedEdge> weightedEdges;
  for (std::size_t a = 0; a < tree.cliques.size(); ++a) {
    for (std::size_t b = a + 1; b < tree.cliques.size(); ++b) {
      WeightedEdge edge;
      edge.a = static_cast<int>(a);
      edge.b = static_cast<int>(b);
      edge.separator =
          intersectKeys(tree.cliques[a].variables, tree.cliques[b].variables);
      edge.weight = static_cast<int>(edge.separator.size());
      weightedEdges.push_back(std::move(edge));
    }
  }
  std::sort(weightedEdges.begin(), weightedEdges.end(),
            [](const WeightedEdge &lhs, const WeightedEdge &rhs) {
              if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
              }
              return std::tie(lhs.a, lhs.b) < std::tie(rhs.a, rhs.b);
            });

  std::vector<int> parent(tree.cliques.size());
  std::iota(parent.begin(), parent.end(), 0);
  const auto findRoot = [&](int x) {
    int root = x;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[x] != x) {
      const int next = parent[x];
      parent[x] = root;
      x = next;
    }
    return root;
  };
  auto unite = [&](int a, int b) {
    const int ra = findRoot(a);
    const int rb = findRoot(b);
    if (ra == rb) {
      return false;
    }
    parent[rb] = ra;
    return true;
  };

  for (const auto &weighted : weightedEdges) {
    if (!unite(weighted.a, weighted.b)) {
      continue;
    }
    InterfaceCliqueTreeEdge edge;
    edge.a = weighted.a;
    edge.b = weighted.b;
    edge.separator = weighted.separator;
    edge.estimated_message_bytes = estimatedMessageBytesForSeparator(
        edge.separator.size(), problem.block_dim, problem.rhs_dim);
    tree.edges.push_back(edge);
    tree.cliques[weighted.a].neighbors.push_back(weighted.b);
    tree.cliques[weighted.b].neighbors.push_back(weighted.a);
  }

  for (auto &clique : tree.cliques) {
    std::sort(clique.neighbors.begin(), clique.neighbors.end());
  }

  for (const auto &factor : problem.factors) {
    if (factor.scope.empty()) {
      continue;
    }
    bool assigned = false;
    for (auto &clique : tree.cliques) {
      if (containsScope(clique.variables, factor.scope)) {
        clique.assigned_factors.push_back(factor.id);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      throw std::runtime_error("RIFT-IF clique tree does not cover factor");
    }
  }

  return tree;
}

bool InterfaceCliqueTreeBuilder::VerifyRunningIntersection(
    const InterfaceCliqueTree &tree) {
  std::map<InterfaceKey, std::vector<RIFTCliqueId>> containing;
  for (const auto &clique : tree.cliques) {
    for (const auto &key : clique.variables) {
      containing[key].push_back(clique.id);
    }
  }

  for (const auto &entry : containing) {
    const std::set<RIFTCliqueId> wanted(entry.second.begin(),
                                        entry.second.end());
    std::queue<RIFTCliqueId> queue;
    std::set<RIFTCliqueId> visited;
    queue.push(*wanted.begin());
    visited.insert(*wanted.begin());
    while (!queue.empty()) {
      const RIFTCliqueId current = queue.front();
      queue.pop();
      for (const RIFTCliqueId neighbor : tree.cliques.at(current).neighbors) {
        if (wanted.count(neighbor) == 0 || visited.count(neighbor) != 0) {
          continue;
        }
        visited.insert(neighbor);
        queue.push(neighbor);
      }
    }
    if (visited != wanted) {
      return false;
    }
  }
  return true;
}

void RIFTRootlessScheduler::Initialize(const InterfaceCliqueTree &tree) {
  neighbors_.clear();
  sent_.clear();
  received_.clear();
  for (const auto &clique : tree.cliques) {
    neighbors_[clique.id] =
        std::set<RIFTCliqueId>(clique.neighbors.begin(),
                               clique.neighbors.end());
  }
}

void RIFTRootlessScheduler::OnMessageReceived(const RIFTMessage &msg) {
  received_.insert(DirectedCliqueEdge{msg.header.src, msg.header.dst});
}

std::vector<DirectedCliqueEdge> RIFTRootlessScheduler::ReadyOutgoingMessages()
    const {
  std::vector<DirectedCliqueEdge> ready;
  for (const auto &entry : neighbors_) {
    const RIFTCliqueId src = entry.first;
    for (const RIFTCliqueId dst : entry.second) {
      const DirectedCliqueEdge edge{src, dst};
      if (sent_.count(edge) != 0) {
        continue;
      }
      bool canSend = true;
      for (const RIFTCliqueId other : entry.second) {
        if (other == dst) {
          continue;
        }
        if (received_.count(DirectedCliqueEdge{other, src}) == 0) {
          canSend = false;
          break;
        }
      }
      if (canSend) {
        ready.push_back(edge);
      }
    }
  }
  std::sort(ready.begin(), ready.end());
  return ready;
}

void RIFTRootlessScheduler::MarkMessageSent(RIFTCliqueId src,
                                            RIFTCliqueId dst) {
  sent_.insert(DirectedCliqueEdge{src, dst});
}

bool RIFTRootlessScheduler::AllDirectedMessagesComplete() const {
  std::size_t expected = 0;
  for (const auto &entry : neighbors_) {
    expected += entry.second.size();
  }
  return sent_.size() == expected && received_.size() == expected;
}

bool RIFTRootlessScheduler::CliqueBeliefReady(RIFTCliqueId alpha) const {
  const auto it = neighbors_.find(alpha);
  if (it == neighbors_.end()) {
    return false;
  }
  for (const RIFTCliqueId neighbor : it->second) {
    if (received_.count(DirectedCliqueEdge{neighbor, alpha}) == 0) {
      return false;
    }
  }
  return true;
}

Matrix RIFTExactSolver::SolveMatrix(const InterfaceProblem &problem,
                                    const InterfaceCliqueTree &tree,
                                    const RIFTParams &params,
                                    RIFTStats *stats,
                                    DecentralizationGuard *guard) {
  if (problem.block_dim <= 0 || problem.rhs_dim <= 0) {
    throw std::invalid_argument("RIFT-IF problem dimensions must be positive");
  }
  if (!InterfaceCliqueTreeBuilder::VerifyRunningIntersection(tree)) {
    throw std::runtime_error("RIFT-IF clique tree violates running intersection");
  }

  RIFTStats localStats;
  localStats.selected_backend = RIFTInterfaceBackend::RIFT_EXACT;
  localStats.num_cliques = static_cast<int>(tree.cliques.size());
  localStats.num_tree_edges = static_cast<int>(tree.edges.size());
  for (const auto &clique : tree.cliques) {
    localStats.max_clique_blocks =
        std::max(localStats.max_clique_blocks,
                 static_cast<int>(clique.variables.size()));
  }
  for (const auto &edge : tree.edges) {
    localStats.max_separator_blocks =
        std::max(localStats.max_separator_blocks,
                 static_cast<int>(edge.separator.size()));
    localStats.estimated_message_bytes += 2u * edge.estimated_message_bytes;
  }

  const int maxSeparatorAllowed =
      problem.dimension == 2 ? params.exact_max_separator_blocks_2d
                             : params.exact_max_separator_blocks_3d;
  if (localStats.max_separator_blocks > maxSeparatorAllowed ||
      localStats.estimated_message_bytes > params.exact_max_message_bytes) {
    throw std::runtime_error("RIFT-IF exact symbolic gate rejected interface");
  }

  std::map<RIFTFactorId, InterfaceFactor> factorById;
  for (const auto &factor : problem.factors) {
    factorById[factor.id] = factor;
  }

  if (guard != nullptr) {
    for (const auto &clique : tree.cliques) {
      for (const RIFTFactorId factorId : clique.assigned_factors) {
        const auto &factor = factorById.at(factorId);
        guard->RecordFactorTransfer(factor.owner_robot, clique.host_robot,
                                    factorId);
      }
    }
  }

  std::map<DirectedCliqueEdge, RIFTMessage> messages;
  RIFTRootlessScheduler scheduler;
  scheduler.Initialize(tree);
  localStats.message_qr_ms += elapsedMilliseconds([&]() {
    while (!scheduler.AllDirectedMessagesComplete()) {
      const auto ready = scheduler.ReadyOutgoingMessages();
      if (ready.empty()) {
        throw std::runtime_error("RIFT-IF scheduler deadlock");
      }
      for (const auto &edge : ready) {
        RIFTMessage msg = computeDirectedMessage(problem, tree, factorById,
                                                 edge.src, edge.dst, messages);
        const std::size_t bytes = messageBytes(msg);
        localStats.actual_message_bytes += bytes;
        ++localStats.directed_messages_sent;
        if (guard != nullptr) {
          guard->RecordMessage(tree.cliques.at(edge.src).host_robot,
                               tree.cliques.at(edge.dst).host_robot, bytes);
        }
        scheduler.MarkMessageSent(edge.src, edge.dst);
        scheduler.OnMessageReceived(msg);
        messages[edge] = std::move(msg);
      }
    }
  });

  std::map<InterfaceKey, Matrix> blockByKey;
  localStats.belief_solve_ms += elapsedMilliseconds([&]() {
    for (const auto &clique : tree.cliques) {
      std::vector<const InterfaceFactor *> factors;
      for (const RIFTFactorId id : clique.assigned_factors) {
        factors.push_back(&factorById.at(id));
      }
      std::vector<const RIFTMessage *> incoming;
      for (const RIFTCliqueId neighbor : clique.neighbors) {
        incoming.push_back(
            &messages.at(DirectedCliqueEdge{neighbor, clique.id}));
      }
      const auto stacked = stackFactorsForClique(
          clique.variables, problem.block_dim, problem.rhs_dim, factors,
          incoming);
      Matrix X = Matrix::Zero(
          static_cast<int>(clique.variables.size()) * problem.block_dim,
          problem.rhs_dim);
      if (stacked.first.cols() > 0 && stacked.first.rows() > 0) {
        Eigen::ColPivHouseholderQR<Matrix> qr(stacked.first);
        for (int rhs = 0; rhs < problem.rhs_dim; ++rhs) {
          X.col(rhs) = qr.solve(stacked.second.col(rhs));
        }
      }
      for (std::size_t i = 0; i < clique.variables.size(); ++i) {
        const InterfaceKey &key = clique.variables[i];
        if (blockByKey.count(key) != 0) {
          continue;
        }
        blockByKey[key] = X.block(static_cast<int>(i) * problem.block_dim, 0,
                                  problem.block_dim, problem.rhs_dim);
      }
    }
  });

  Matrix solution = Matrix::Zero(
      static_cast<int>(problem.variables.size()) * problem.block_dim,
      problem.rhs_dim);
  for (std::size_t i = 0; i < problem.variables.size(); ++i) {
    const auto block = blockByKey.find(problem.variables[i]);
    if (block == blockByKey.end()) {
      throw std::runtime_error("RIFT-IF failed to extract interface variable");
    }
    solution.block(static_cast<int>(i) * problem.block_dim, 0,
                   problem.block_dim, problem.rhs_dim) = block->second;
  }

  localStats.final_interface_residual =
      interfaceResidualNorm(problem, solution);
  if (guard != nullptr) {
    if (params.forbid_global_interface_matrix) {
      guard->AssertNoGlobalInterfaceMatrixConstructed();
    }
    if (params.forbid_direct_solver_in_deployment) {
      guard->AssertNoDirectSolverCalled();
    }
    if (params.forbid_collectives) {
      guard->AssertNoCollectiveCommunication();
    }
    if (params.forbid_global_interface_matrix ||
        params.forbid_direct_solver_in_deployment ||
        params.forbid_collectives) {
      guard->AssertNoNodeReceivedAllFactors(
          static_cast<int>(problem.factors.size()));
    }
    localStats.used_global_matrix = guard->used_global_matrix();
    localStats.used_direct_solver = guard->used_direct_solver();
    localStats.used_collective = guard->used_collective();
  }
  if (stats != nullptr) {
    *stats = localStats;
  }
  return solution;
}

Vector RIFTExactSolver::Solve(const InterfaceProblem &problem,
                              const InterfaceCliqueTree &tree,
                              const RIFTParams &params, RIFTStats *stats,
                              DecentralizationGuard *guard) {
  if (problem.rhs_dim != 1) {
    throw std::invalid_argument(
        "RIFT-IF vector Solve requires a single RHS; use SolveMatrix");
  }
  return SolveMatrix(problem, tree, params, stats, guard).col(0);
}

Matrix SolveInterfaceProblemWithRIFTExact(
    const InterfaceProblem &problem, const TEDCCIParams &params,
    TEDCCIStats *stats) {
  RIFTParams riftParams;
  riftParams.backend = params.rift_interface_backend;
  if (riftParams.backend == RIFTInterfaceBackend::DIRECT_ORACLE) {
    throw std::invalid_argument(
        "RIFT-IF deployment mode requires rift_exact or rift_auto backend");
  }
  if (riftParams.backend == RIFTInterfaceBackend::RIFT_CAK ||
      riftParams.backend == RIFTInterfaceBackend::RIFT_ASYNC_SCHUR) {
    throw std::invalid_argument(
        "RIFT-IF requested backend is not implemented in this MVP");
  }
  riftParams.exact_max_separator_blocks_2d =
      params.rift_exact_max_separator_blocks_2d;
  riftParams.exact_max_separator_blocks_3d =
      params.rift_exact_max_separator_blocks_3d;
  riftParams.exact_max_message_bytes = params.rift_exact_max_message_bytes;
  riftParams.use_rotation_multi_rhs = params.rift_use_rotation_multi_rhs;
  riftParams.forbid_direct_solver_in_deployment =
      params.rift_forbid_direct_interface_solver;
  riftParams.forbid_global_interface_matrix =
      params.rift_forbid_global_interface_matrix;
  riftParams.forbid_collectives = params.rift_forbid_collectives;

  RIFTStats riftStats;
  InterfaceCliqueTree tree;
  double symbolicMs = elapsedMilliseconds([&]() {
    tree = InterfaceCliqueTreeBuilder::Build(problem, riftParams);
  });
  DecentralizationGuard guard;
  Matrix solution =
      RIFTExactSolver::SolveMatrix(problem, tree, riftParams, &riftStats,
                                   &guard);
  riftStats.symbolic_ms += symbolicMs;

  if (stats != nullptr) {
    stats->effective_mode = CCIInitMode::TED_CCI_RIFT_IF;
    stats->effective_interface_backend = params.interface_backend;
    stats->num_interface_vars = static_cast<int>(problem.variables.size());
    stats->num_condensed_rows = 0;
    for (const auto &factor : problem.factors) {
      stats->num_condensed_rows += factor.A.rows();
    }
    stats->max_separator_size = riftStats.max_separator_blocks;
    stats->num_factor_messages = static_cast<int>(problem.factors.size());
    stats->num_solution_messages = riftStats.directed_messages_sent;
    stats->bytes_sent_upward = riftStats.actual_message_bytes;
    stats->bytes_sent_downward = 0;
    stats->rift_selected_backend = riftStats.selected_backend;
    stats->rift_num_cliques = riftStats.num_cliques;
    stats->rift_num_tree_edges = riftStats.num_tree_edges;
    stats->rift_max_clique_blocks = riftStats.max_clique_blocks;
    stats->rift_max_separator_blocks = riftStats.max_separator_blocks;
    stats->rift_estimated_message_bytes = riftStats.estimated_message_bytes;
    stats->rift_actual_message_bytes = riftStats.actual_message_bytes;
    stats->rift_directed_messages_sent = riftStats.directed_messages_sent;
    stats->rift_symbolic_ms = riftStats.symbolic_ms;
    stats->rift_message_qr_ms = riftStats.message_qr_ms;
    stats->rift_belief_solve_ms = riftStats.belief_solve_ms;
    stats->rift_final_interface_residual =
        riftStats.final_interface_residual;
    stats->rift_used_global_matrix = riftStats.used_global_matrix;
    stats->rift_used_direct_solver = riftStats.used_direct_solver;
    stats->rift_used_collective = riftStats.used_collective;
  }
  return solution;
}

Vector SolveTEDInterfaceWithRIFTExact(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params, TEDCCIStats *stats) {
  if (block_dim <= 0) {
    throw std::invalid_argument("RIFT-IF TED interface block dimension invalid");
  }
  (void)partitions;
  const FactorType stage =
      !condensed_factors.empty()
          ? condensed_factors.front().factor_type
          : (!cross_factors.empty() ? FactorType::ROTATION
                                    : FactorType::TRANSLATION);
  const int dimension =
      stage == FactorType::ROTATION
          ? static_cast<int>(std::round(std::sqrt(static_cast<double>(block_dim))))
          : block_dim;
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          stage, dimension, block_dim, condensed_factors, cross_factors,
          /*use_rotation_multi_rhs=*/false);
  const Matrix solution =
      SolveInterfaceProblemWithRIFTExact(problem, params, stats);
  if (solution.cols() != 1) {
    throw std::invalid_argument(
        "RIFT-IF TED vector interface solve produced multiple RHS columns");
  }
  return solution.col(0);
}

std::string RIFTInterfaceBackendName(RIFTInterfaceBackend backend) {
  switch (backend) {
    case RIFTInterfaceBackend::DIRECT_ORACLE:
      return "direct_oracle";
    case RIFTInterfaceBackend::RIFT_EXACT:
      return "rift_exact";
    case RIFTInterfaceBackend::RIFT_AUTO:
      return "rift_auto";
    case RIFTInterfaceBackend::RIFT_CAK:
      return "rift_cak";
    case RIFTInterfaceBackend::RIFT_ASYNC_SCHUR:
      return "rift_async_schur";
  }
  return "unknown";
}

}  // namespace DPGO
