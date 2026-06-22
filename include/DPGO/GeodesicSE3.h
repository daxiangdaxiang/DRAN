#ifndef DPGO_GEODESIC_SE3_H
#define DPGO_GEODESIC_SE3_H

#include <DPGO/DPGO_types.h>

#include <Eigen/Dense>

namespace DPGO {

Matrix skew3(const Eigen::Vector3d &v);
Eigen::Vector3d vee3(const Matrix &S);
Matrix so3Exp(const Eigen::Vector3d &omega);
Eigen::Vector3d so3Log(const Matrix &R);
Matrix so3LeftJacobian(const Eigen::Vector3d &omega);
Matrix so3RightJacobianInverse(const Eigen::Vector3d &omega);
Matrix so3LeftJacobianInverse(const Eigen::Vector3d &omega);
Matrix projectToSO3(const Matrix &R);
Matrix applySE3RightTangentStep(
    const Matrix &pose, const Eigen::Matrix<double, 6, 1> &tangent,
    double scale);

struct GeodesicSE3Residual {
  Eigen::Vector3d rotation;
  Eigen::Vector3d translation;
};

struct GeodesicSE3Jacobians {
  Matrix Jr_i;
  Matrix Jt_i;
  Matrix Jr_j;
  Matrix Jt_j;
};

GeodesicSE3Residual evaluateRelativeSE3Residual(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj);

GeodesicSE3Residual evaluateRelativeSE3LogResidual(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj);

GeodesicSE3Jacobians linearizeRelativeSE3Residual(
    const Matrix &Ri, const Matrix &tij, const Matrix &Rj,
    const Eigen::Vector3d &rotationResidual);

GeodesicSE3Jacobians linearizeRelativeSE3LogResidual(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj);

}  // namespace DPGO

#endif
