/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/QuadraticProblem.h>
#include <iostream>
#include "DPGO/DPGO_types.h"
#include "DPGO/manifold/LiftedSEVariable.h"

using namespace std;

/*Define the namespace*/
namespace DPGO {

QuadraticProblem::QuadraticProblem(size_t nIn, size_t dIn, size_t rIn) :
    n(nIn), d(dIn), r(rIn),
    M(new LiftedSEManifold(r, d, n)) {
  assert(r >= d);
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());
  setQ(SparseMatrix((d + 1) * n, (d + 1) * n));
  setG(SparseMatrix(r, (d + 1) * n));
}

QuadraticProblem::~QuadraticProblem() {
  delete M;
}

void QuadraticProblem::setQ(const SparseMatrix &QIn) {
  assert((unsigned) QIn.rows() == (d + 1) * n);
  assert((unsigned) QIn.cols() == (d + 1) * n);
  mQ = QIn;

  // Update preconditioner
  SparseMatrix P = mQ;
  for (int i = 0; i < P.rows(); ++i) {
    P.coeffRef(i, i) += 1e-1;
  }
  solver.compute(P);
}

void QuadraticProblem::setQWithoutPreconditioner(const SparseMatrix &QIn) {
  assert((unsigned) QIn.rows() == (d + 1) * n);
  assert((unsigned) QIn.cols() == (d + 1) * n);
  mQ = QIn;
}

void QuadraticProblem::setG(const SparseMatrix &GIn) {
  assert((unsigned) GIn.rows() == r);
  assert((unsigned) GIn.cols() == (d + 1) * n);
  mG = GIn;
}

double QuadraticProblem::f(const Matrix &Y) const {
  assert((unsigned) Y.rows() == r);
  assert((unsigned) Y.cols() == (d + 1) * n);
  // returns 0.5 * (Y * Q * Y.transpose()).trace() + (Y * G.transpose()).trace()
//  std::cout<<"first term: "<<0.5 * ((Y * mQ).cwiseProduct(Y)).sum()<<" second term: "<<(Y.cwiseProduct(mG)).sum()<<std::endl;
  return 0.5 * ((Y * mQ).cwiseProduct(Y)).sum() + (Y.cwiseProduct(mG)).sum();
}
double QuadraticProblem::get_scondf(const Matrix &Y)const{
  return (Y.cwiseProduct(mG)).sum();
}

std::pair<double, double> QuadraticProblem::fAndRieGradNorm(
    const Matrix &Y) const {
  assert((unsigned) Y.rows() == r);
  assert((unsigned) Y.cols() == (d + 1) * n);
  const Matrix YQ = Y * mQ;
  const double objective =
      0.5 * (YQ.cwiseProduct(Y)).sum() + (Y.cwiseProduct(mG)).sum();
  return {objective, projectLiftedSETangent(Y, YQ + mG).norm()};
}

double QuadraticProblem::f(ROPTLIB::Variable *x) const {
  Eigen::Map<const Matrix> X((double *) x->ObtainReadData(), r, (d + 1) * n);
  return 0.5 * ((X * mQ).cwiseProduct(X)).sum() + (X.cwiseProduct(mG)).sum();
}

void QuadraticProblem::EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const {
  Eigen::Map<const Matrix> X((double *) x->ObtainReadData(), r, (d + 1) * n);
  Eigen::Map<Matrix> EG((double *) g->ObtainWriteEntireData(), r, (d + 1) * n);
  EG = X * mQ + mG;
}

void QuadraticProblem::EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                                     ROPTLIB::Vector *Hv) const {
  Eigen::Map<const Matrix> V((double *) v->ObtainReadData(), r, (d + 1) * n);
  Eigen::Map<Matrix> HV((double *) Hv->ObtainWriteEntireData(), r, (d + 1) * n);
  HV = V * mQ;
}

void QuadraticProblem::PreConditioner(ROPTLIB::Variable *x,
                                      ROPTLIB::Vector *inVec,
                                      ROPTLIB::Vector *outVec) const {
  Eigen::Map<const Matrix> INVEC((double *) inVec->ObtainReadData(), r, (d + 1) * n);
  Eigen::Map<Matrix> OUTVEC((double *) outVec->ObtainWriteEntireData(), r, (d + 1) * n);
  OUTVEC = solver.solve(INVEC.transpose()).transpose();
  if (solver.info() == Eigen::Success) {
    M->getManifold()->Projection(x, outVec, outVec);  // Project output to the tangent space at x
  } else {
    printf("Preconditioner failed.\n");
    OUTVEC = INVEC;
  }
}

Matrix QuadraticProblem::RieGrad(const Matrix &Y) const {
  assert((unsigned) Y.rows() == r);
  assert((unsigned) Y.cols() == (d + 1) * n);
  return projectLiftedSETangent(Y, Y * mQ + mG);
}

// Matrix QuadraticProblem::vectransport(const Matrix &X,const Matrix &Y,const Matrix &xix){
//   LiftedSEVector xGrad(r, d, 1);
//   xGrad.setData(xix);
//   LiftedSEVariable XVar(r, d, 1);
//   XVar.setData(X);
//   LiftedSEVariable YVar(r, d, 1);
//   YVar.setData(Y);
//   LiftedSEVector YGrad(r, d, 1);


// }
double QuadraticProblem::RieGradNorm(const Matrix &Y) const {
  return RieGrad(Y).norm();
}

Matrix QuadraticProblem::projectLiftedSETangent(const Matrix &Y,
                                                const Matrix &Z) const {
  assert((unsigned) Y.rows() == r);
  assert((unsigned) Y.cols() == (d + 1) * n);
  assert(Z.rows() == Y.rows());
  assert(Z.cols() == Y.cols());
  Matrix projected = Z;
  for (size_t pose = 0; pose < n; ++pose) {
    const size_t colStart = pose * (d + 1);
    const auto R =
        Y.block(0, static_cast<int>(colStart), static_cast<int>(r),
                static_cast<int>(d));
    const auto ZR =
        Z.block(0, static_cast<int>(colStart), static_cast<int>(r),
                static_cast<int>(d));
    Matrix sym = R.transpose() * ZR;
    sym = (0.5 * (sym + sym.transpose())).eval();
    projected.block(0, static_cast<int>(colStart), static_cast<int>(r),
                    static_cast<int>(d)) = ZR - R * sym;
  }
  return projected;
}

Matrix QuadraticProblem::readElement(const ROPTLIB::Element *element) const {
  return Eigen::Map<Matrix>((double *) element->ObtainReadData(), r, n * (d + 1));
}

void QuadraticProblem::setElement(ROPTLIB::Element *element, const Matrix *matrix) const {
  memcpy(element->ObtainWriteEntireData(), matrix->data(), sizeof(double) * r * (d + 1) * n);
}

}  // namespace DPGO
