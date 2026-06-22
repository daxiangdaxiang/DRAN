/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/ManualQuadraticOptimizer.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/QR>

namespace DPGO {

using ColMajorSparseMatrix = Eigen::SparseMatrix<double>;

ManualQuadraticOptimizer::ManualQuadraticOptimizer(QuadraticProblem *p)
    : problem(p),
      verbose(false),
      maxIterations(1),
      maxAcceptedIterations(1),
      maxCgIterations(50),
      gradientTolerance(1e-2),
      initialRadius(10.0),
      initialDamping(1e-6) {}

double ManualQuadraticOptimizer::innerProduct(const Matrix &A,
                                              const Matrix &B) {
  return (A.cwiseProduct(B)).sum();
}

double ManualQuadraticOptimizer::boundaryTau(const Matrix &eta,
                                             const Matrix &p,
                                             double radius) {
  const double a = innerProduct(p, p);
  const double b = 2.0 * innerProduct(eta, p);
  const double c = innerProduct(eta, eta) - radius * radius;
  const double disc = std::max(0.0, b * b - 4.0 * a * c);
  if (a <= 0.0) {
    return 0.0;
  }
  return std::max(0.0, (-b + std::sqrt(disc)) / (2.0 * a));
}

Matrix ManualQuadraticOptimizer::projectTangent(const Matrix &Y,
                                                const Matrix &Z) const {
  const unsigned r = problem->relaxation_rank();
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  Matrix projected = Matrix::Zero(Z.rows(), Z.cols());

  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    const Matrix R = Y.block(0, colStart, r, d);
    const Matrix ZR = Z.block(0, colStart, r, d);
    Matrix sym = R.transpose() * ZR;
    sym = 0.5 * (sym + sym.transpose());
    projected.block(0, colStart, r, d) = ZR - R * sym;
    projected.col(colStart + d) = Z.col(colStart + d);
  }
  return projected;
}

Matrix ManualQuadraticOptimizer::retract(const Matrix &Y,
                                         const Matrix &eta) const {
  const unsigned r = problem->relaxation_rank();
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  Matrix candidate = Y + eta;

  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned colStart = pose * (d + 1);
    const Matrix block = candidate.block(0, colStart, r, d);
    Eigen::HouseholderQR<Matrix> qr(block);
    Matrix qFull = qr.householderQ() * Matrix::Identity(r, d);
    Matrix rTri = qr.matrixQR().topLeftCorner(d, d)
                      .template triangularView<Eigen::Upper>();
    for (unsigned col = 0; col < d; ++col) {
      if (rTri(col, col) < 0.0) {
        qFull.col(col) *= -1.0;
      }
    }
    candidate.block(0, colStart, r, d) = qFull;
  }
  return candidate;
}

Matrix ManualQuadraticOptimizer::hessianEta(const Matrix &Y, const Matrix &eta,
                                            const SparseMatrix &Q) const {
  LiftedSEVariable point(problem->relaxation_rank(), problem->dimension(),
                         problem->num_poses());
  LiftedSEVector etaVec(problem->relaxation_rank(), problem->dimension(),
                        problem->num_poses());
  LiftedSEVector gradVec(problem->relaxation_rank(), problem->dimension(),
                         problem->num_poses());
  LiftedSEVector hVec(problem->relaxation_rank(), problem->dimension(),
                      problem->num_poses());
  point.setData(Y);
  etaVec.setData(eta);

  const ROPTLIB::Problem *baseProblem = problem;
  baseProblem->RieGrad(point.var(), gradVec.vec());
  baseProblem->HessianEta(point.var(), etaVec.vec(), hVec.vec());

  Matrix hEta = hVec.getData();
  if (hEta.rows() != eta.rows() || hEta.cols() != eta.cols() ||
      !hEta.allFinite()) {
    hEta = projectTangent(Y, eta * Q);
  }
  if (initialDamping > 0.0) {
    hEta += initialDamping * eta;
  }
  return hEta;
}

Matrix ManualQuadraticOptimizer::polishTranslations(
    const Matrix &Y, const SparseMatrix &Q) const {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  if (n == 0 || Y.cols() != static_cast<int>((d + 1) * n)) {
    return Y;
  }

  std::vector<int> translationCols;
  translationCols.reserve(n);
  std::vector<char> isTranslation(Y.cols(), 0);
  for (unsigned pose = 0; pose < n; ++pose) {
    const int col = static_cast<int>(pose * (d + 1) + d);
    translationCols.push_back(col);
    isTranslation[col] = 1;
  }

  Matrix fixedY = Y;
  for (int col : translationCols) {
    fixedY.col(col).setZero();
  }

  const SparseMatrix G = problem->getG();
  const Matrix fixedTimesQ = fixedY * Q;
  Matrix rhs(Y.rows(), static_cast<int>(n));
  for (unsigned pose = 0; pose < n; ++pose) {
    const int col = translationCols[pose];
    rhs.col(static_cast<int>(pose)) = -fixedTimesQ.col(col);
    for (int row = 0; row < G.rows(); ++row) {
      rhs(row, static_cast<int>(pose)) -= G.coeff(row, col);
    }
  }

  ColMajorSparseMatrix translationQ(n, n);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(Q.nonZeros() + n);
  double diagScale = 0.0;
  unsigned diagCount = 0;
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
      if (!isTranslation[it.row()] || !isTranslation[it.col()]) {
        continue;
      }
      const int row = static_cast<int>(it.row() / (d + 1));
      const int col = static_cast<int>(it.col() / (d + 1));
      triplets.emplace_back(row, col, it.value());
      if (row == col) {
        diagScale += std::abs(it.value());
        ++diagCount;
      }
    }
  }
  if (diagCount > 0) {
    diagScale /= static_cast<double>(diagCount);
  }
  if (!std::isfinite(diagScale) || diagScale <= 0.0) {
    diagScale = 1.0;
  }

  Matrix polished = Y;
  const double baseRidge =
      std::max(initialDamping, 1e-10 * std::max(1.0, diagScale));
  for (unsigned attempt = 0; attempt < 6; ++attempt) {
    ColMajorSparseMatrix regularized(n, n);
    std::vector<Eigen::Triplet<double>> regularizedTriplets = triplets;
    const double ridge = baseRidge * std::pow(10.0, attempt);
    for (unsigned i = 0; i < n; ++i) {
      regularizedTriplets.emplace_back(static_cast<int>(i),
                                       static_cast<int>(i), ridge);
    }
    regularized.setFromTriplets(regularizedTriplets.begin(),
                                regularizedTriplets.end());
    regularized.makeCompressed();

    Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
    solver.compute(regularized);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    Matrix solved = solver.solve(rhs.transpose());
    if (solver.info() != Eigen::Success ||
        solved.rows() != static_cast<int>(n) || solved.cols() != Y.rows() ||
        !solved.allFinite()) {
      continue;
    }
    for (unsigned pose = 0; pose < n; ++pose) {
      polished.col(pose * (d + 1) + d) =
          solved.row(static_cast<int>(pose)).transpose();
    }
    if (problem->f(polished) <= problem->f(Y) + 1e-10) {
      return polished;
    }
  }

  return Y;
}

Matrix ManualQuadraticOptimizer::polishMajorizedRotationBlocks(
    const Matrix &Y, const SparseMatrix &Q) const {
  const unsigned d = problem->dimension();
  const unsigned n = problem->num_poses();
  if (n == 0 || Y.cols() != static_cast<int>((d + 1) * n)) {
    return Y;
  }

  std::vector<double> rowAbsSums(Q.rows(), 0.0);
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
      rowAbsSums[it.row()] += std::abs(it.value());
    }
  }

  Matrix current = Y;
  double currentCost = problem->f(current);
  for (unsigned sweep = 0; sweep < 2; ++sweep) {
    Matrix grad = problem->RieGrad(current);
    Matrix direction = Matrix::Zero(Y.rows(), Y.cols());

    for (unsigned pose = 0; pose < n; ++pose) {
      const unsigned colStart = pose * (d + 1);
      double lipschitz = initialDamping;
      for (unsigned localCol = 0; localCol < d; ++localCol) {
        lipschitz =
            std::max(lipschitz, rowAbsSums[colStart + localCol]);
      }
      lipschitz = std::max(lipschitz, 1e-6);
      Matrix block =
          -grad.block(0, colStart, Y.rows(), d) / lipschitz;
      const double blockNorm = block.norm();
      constexpr double maxBlockNorm = 0.5;
      if (blockNorm > maxBlockNorm) {
        block *= maxBlockNorm / blockNorm;
      }
      direction.block(0, colStart, Y.rows(), d) = block;
    }

    if (direction.norm() <= 1e-14) {
      break;
    }

    bool accepted = false;
    for (double scale : {1.0, 0.5, 0.25, 0.125}) {
      Matrix candidate =
          polishTranslations(retract(current, scale * direction), Q);
      const double candidateCost = problem->f(candidate);
      if (std::isfinite(candidateCost) &&
          candidateCost < currentCost - 1e-12) {
        current = candidate;
        currentCost = candidateCost;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      break;
    }
  }

  return current;
}

Matrix ManualQuadraticOptimizer::solveSurrogateDirection(
    const Matrix &Y, const Matrix &grad, double radius,
    const SparseMatrix &Q) const {
  if (radius <= 0.0 || grad.norm() <= 1e-14) {
    return Matrix::Zero(grad.rows(), grad.cols());
  }

  double diagScale = 0.0;
  unsigned diagCount = 0;
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(Q, k); it; ++it) {
      if (it.row() == it.col()) {
        diagScale += std::abs(it.value());
        ++diagCount;
      }
    }
  }
  if (diagCount > 0) {
    diagScale /= static_cast<double>(diagCount);
  }
  if (!std::isfinite(diagScale) || diagScale <= 0.0) {
    diagScale = 1.0;
  }

  Matrix direction = Matrix::Zero(grad.rows(), grad.cols());
  const double baseRidge =
      std::max(initialDamping, 1e-5 * std::max(1.0, diagScale));
  for (unsigned attempt = 0; attempt < 6; ++attempt) {
    ColMajorSparseMatrix surrogate = Q;
    const double ridge = baseRidge * std::pow(10.0, attempt);
    for (int i = 0; i < surrogate.rows(); ++i) {
      surrogate.coeffRef(i, i) += ridge;
    }
    surrogate.makeCompressed();

    Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
    solver.compute(surrogate);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    Matrix solved = solver.solve(grad.transpose());
    if (solver.info() != Eigen::Success || solved.rows() != grad.cols() ||
        solved.cols() != grad.rows() || !solved.allFinite()) {
      continue;
    }
    direction = -solved.transpose();
    direction = projectTangent(Y, direction);
    const double norm = direction.norm();
    if (norm > radius) {
      direction *= radius / norm;
    }
    return direction;
  }

  return direction;
}

Matrix ManualQuadraticOptimizer::solveTruncatedCg(const Matrix &Y,
                                                  const Matrix &grad,
                                                  double radius,
                                                  double damping,
                                                  const SparseMatrix &Q) const {
  Matrix eta = Matrix::Zero(grad.rows(), grad.cols());
  Matrix residual = grad;
  Matrix direction = -residual;
  double residualNormSq = innerProduct(residual, residual);
  const double residualNorm0 = std::sqrt(std::max(0.0, residualNormSq));
  const double cgTol = std::max(1e-12, 0.1 * residualNorm0);

  if (residualNorm0 <= cgTol || radius <= 0.0) {
    return eta;
  }

  const int iters = std::max(1, maxCgIterations);
  LiftedSEVariable precondPoint(problem->relaxation_rank(),
                                problem->dimension(),
                                problem->num_poses());
  precondPoint.setData(Y);
  auto applyPreconditioner = [&](const Matrix &input) {
    LiftedSEVector inVec(problem->relaxation_rank(), problem->dimension(),
                         problem->num_poses());
    LiftedSEVector outVec(problem->relaxation_rank(), problem->dimension(),
                          problem->num_poses());
    inVec.setData(input);
    problem->PreConditioner(precondPoint.var(), inVec.vec(), outVec.vec());
    Matrix output = outVec.getData();
    if (output.rows() != input.rows() || output.cols() != input.cols() ||
        !output.allFinite()) {
      return input;
    }
    return projectTangent(Y, output);
  };

  Matrix z = applyPreconditioner(residual);
  double zResidual = innerProduct(z, residual);
  if (!std::isfinite(zResidual) || zResidual <= 1e-14) {
    z = residual;
    zResidual = residualNormSq;
  }
  direction = -z;

  for (int iter = 0; iter < iters; ++iter) {
    Matrix hDir = hessianEta(Y, direction, Q);
    if (damping != initialDamping) {
      hDir += (damping - initialDamping) * direction;
    }
    const double curvature = innerProduct(direction, hDir);
    if (curvature <= 1e-14) {
      const double tau = boundaryTau(eta, direction, radius);
      return eta + tau * direction;
    }
    const double alpha = zResidual / curvature;
    Matrix nextEta = eta + alpha * direction;
    if (nextEta.norm() >= radius) {
      const double tau = boundaryTau(eta, direction, radius);
      return eta + tau * direction;
    }

    eta = nextEta;
    Matrix nextResidual = residual + alpha * hDir;
    const double nextResidualNormSq =
        innerProduct(nextResidual, nextResidual);
    if (std::sqrt(std::max(0.0, nextResidualNormSq)) <= cgTol) {
      return eta;
    }
    Matrix nextZ = applyPreconditioner(nextResidual);
    double nextZResidual = innerProduct(nextZ, nextResidual);
    if (!std::isfinite(nextZResidual) || nextZResidual <= 1e-14) {
      nextZ = nextResidual;
      nextZResidual = nextResidualNormSq;
    }
    const double beta = nextZResidual / zResidual;
    direction = -nextZ + beta * direction;
    residual = nextResidual;
    residualNormSq = nextResidualNormSq;
    z = nextZ;
    zResidual = nextZResidual;
  }
  return eta;
}

Matrix ManualQuadraticOptimizer::optimize(const Matrix &Y) {
  result = ROPTResult();
  result.fInit = problem->f(Y);
  result.gradNormInit = problem->RieGradNorm(Y);
  auto startTime = std::chrono::high_resolution_clock::now();

  const Matrix XStart = Y;
  double radius = std::max(1e-12, initialRadius);
  const double maxRadius = std::max(radius, 5.0 * radius);
  const SparseMatrix Q = problem->getQ();
  Matrix X = polishTranslations(Y, Q);

  unsigned acceptedIterations = 0;
  for (unsigned iter = 0;
       iter < maxIterations && acceptedIterations < maxAcceptedIterations;
       ++iter) {
    const double f = problem->f(X);
    Matrix grad = problem->RieGrad(X);
    const double gradNorm = grad.norm();
    if (gradNorm < gradientTolerance) {
      break;
    }

    bool accepted = false;
    for (unsigned attempt = 0; attempt < 12 && radius >= 1e-12; ++attempt) {
      std::vector<Matrix> candidates;
      candidates.push_back(
          solveTruncatedCg(X, grad, radius, initialDamping, Q));
      candidates.push_back(solveSurrogateDirection(X, grad, radius, Q));
      candidates.push_back(
          -(radius / std::max(gradNorm, 1e-14)) * grad);

      double bestCost = std::numeric_limits<double>::infinity();
      double bestRho = -std::numeric_limits<double>::infinity();
      Matrix bestCandidate = X;
      Matrix bestEta = Matrix::Zero(X.rows(), X.cols());
      bool hasCandidate = false;

      for (Matrix eta : candidates) {
        if (eta.norm() <= 1e-14) {
          continue;
        }
        eta = projectTangent(X, eta);
        const double etaNorm = eta.norm();
        if (etaNorm > radius) {
          eta *= radius / etaNorm;
        }

        Matrix hEta = hessianEta(X, eta, Q);
        double modelDecrease =
            -(innerProduct(grad, eta) + 0.5 * innerProduct(eta, hEta));
        if (modelDecrease <= 1e-14) {
          modelDecrease = -innerProduct(grad, eta);
        }
        if (modelDecrease <= 1e-14) {
          continue;
        }

        Matrix candidate = polishTranslations(retract(X, eta), Q);
        const double candidateCost = problem->f(candidate);
        const double actualDecrease = f - candidateCost;
        const double rho = actualDecrease / modelDecrease;
        if (std::isfinite(candidateCost) && actualDecrease > 0.0 &&
            rho > 0.1 && candidateCost < bestCost) {
          bestCost = candidateCost;
          bestRho = rho;
          bestCandidate = candidate;
          bestEta = eta;
          hasCandidate = true;
        }
      }

      if (!hasCandidate) {
        result.rtrRejectedSteps++;
        radius *= 0.25;
        continue;
      }

      Matrix polishedCandidate =
          polishMajorizedRotationBlocks(bestCandidate, Q);
      const double polishedCost = problem->f(polishedCandidate);
      if (std::isfinite(polishedCost) && polishedCost <= bestCost + 1e-12) {
        bestCandidate = polishedCandidate;
        bestCost = polishedCost;
      }

      X = bestCandidate;
      ++acceptedIterations;
      if (bestRho > 0.75 && bestEta.norm() > 0.8 * radius) {
        radius = std::min(maxRadius, 2.0 * radius);
      }
      accepted = true;
      break;
    }

    if (!accepted) {
      double step = std::min(radius, 1.0) / std::max(gradNorm, 1e-14);
      for (unsigned attempt = 0; attempt < 12; ++attempt) {
        Matrix candidate = polishTranslations(retract(X, -step * grad), Q);
        const double candidateCost = problem->f(candidate);
        if (std::isfinite(candidateCost) && candidateCost < f) {
          X = candidate;
          ++acceptedIterations;
          accepted = true;
          radius = std::max(radius, step * gradNorm);
          break;
        }
        step *= 0.5;
      }
    }

    if (radius < 1e-12) {
      break;
    }
  }

  auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
  result.elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  result.fOpt = problem->f(X);
  result.gradNormOpt = problem->RieGradNorm(X);
  result.relativeChange =
      std::sqrt((X - XStart).squaredNorm() / problem->num_poses());
  result.rtrAcceptedIterations = acceptedIterations;
  result.rtrAcceptedRadius = radius;
  result.success = true;
  return X;
}

}  // namespace DPGO
