#include <DPGO/GeodesicSE3.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace DPGO {

namespace {

double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

bool isNearlySO3(const Matrix &R) {
  if (R.rows() != 3 || R.cols() != 3) {
    return false;
  }
  const Matrix orthogonality =
      R.transpose() * R - Matrix::Identity(3, 3);
  return orthogonality.norm() < 1e-8 && R.determinant() > 0.0;
}

Eigen::Vector3d leftJacobianInverseDirectionalProduct(
    const Eigen::Vector3d &phi, const Eigen::Vector3d &direction,
    const Eigen::Vector3d &p) {
  if (direction.squaredNorm() < 1e-24) {
    return Eigen::Vector3d::Zero();
  }
  const double theta = phi.norm();
  const Matrix W = skew3(phi);
  const Matrix U = skew3(direction);
  if (theta < 1e-8) {
    return (-0.5 * U + (1.0 / 12.0) * (W * U + U * W)) * p;
  }

  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);
  const double theta2 = theta * theta;
  const double coeff =
      1.0 / theta2 -
      (1.0 + cosTheta) / (2.0 * theta * sinTheta);
  const double numerator = 1.0 + cosTheta;
  const double denominator = 2.0 * theta * sinTheta;
  const double numeratorPrime = -sinTheta;
  const double denominatorPrime = 2.0 * sinTheta + 2.0 * theta * cosTheta;
  const double quotientPrime =
      (numeratorPrime * denominator -
       numerator * denominatorPrime) /
      (denominator * denominator);
  const double coeffPrime = -2.0 / (theta2 * theta) - quotientPrime;
  const double directionalCoeff =
      coeffPrime * phi.dot(direction) / theta;
  const Matrix dJ =
      -0.5 * U + directionalCoeff * W * W +
      coeff * (W * U + U * W);
  return dJ * p;
}

}  // namespace

Matrix skew3(const Eigen::Vector3d &v) {
  Matrix S = Matrix::Zero(3, 3);
  S(0, 1) = -v.z();
  S(0, 2) = v.y();
  S(1, 0) = v.z();
  S(1, 2) = -v.x();
  S(2, 0) = -v.y();
  S(2, 1) = v.x();
  return S;
}

Eigen::Vector3d vee3(const Matrix &S) {
  return Eigen::Vector3d(S(2, 1), S(0, 2), S(1, 0));
}

Matrix so3Exp(const Eigen::Vector3d &omega) {
  const double theta = omega.norm();
  const Matrix W = skew3(omega);
  const Matrix I = Matrix::Identity(3, 3);
  if (theta < 1e-12) {
    return I + W + 0.5 * W * W;
  }
  const double theta2 = theta * theta;
  return I + (std::sin(theta) / theta) * W +
         ((1.0 - std::cos(theta)) / theta2) * W * W;
}

Eigen::Vector3d so3Log(const Matrix &Rin) {
  const Matrix R = isNearlySO3(Rin) ? Rin : projectToSO3(Rin);
  const double cosTheta = clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
  const double theta = std::acos(cosTheta);
  const Eigen::Vector3d skewPart = vee3(0.5 * (R - R.transpose()));

  if (theta < 1e-10) {
    return skewPart;
  }

  if (M_PI - theta < 1e-6) {
    Eigen::Vector3d axis;
    int idx = 0;
    R.diagonal().maxCoeff(&idx);
    const double mainComponent =
        std::sqrt(std::max(0.0, 0.5 * (1.0 + R(idx, idx))));
    if (mainComponent < 1e-12) {
      axis = Eigen::Vector3d::UnitX();
    } else {
      axis(idx) = mainComponent;
      const int j = (idx + 1) % 3;
      const int k = (idx + 2) % 3;
      axis(j) = (R(j, idx) + R(idx, j)) / (4.0 * mainComponent);
      axis(k) = (R(k, idx) + R(idx, k)) / (4.0 * mainComponent);
      axis.normalize();
    }
    if (axis.dot(skewPart) < 0.0) {
      axis = -axis;
    }
    return theta * axis;
  }

  const double sinTheta = std::sin(theta);
  if (std::abs(sinTheta) < 1e-12) {
    return theta * skewPart / std::max(1e-12, skewPart.norm());
  }
  return (theta / sinTheta) * skewPart;
}

Matrix so3LeftJacobian(const Eigen::Vector3d &omega) {
  const double theta = omega.norm();
  const Matrix W = skew3(omega);
  const Matrix I = Matrix::Identity(3, 3);
  if (theta < 1e-8) {
    return I + 0.5 * W + (1.0 / 6.0) * W * W;
  }
  const double theta2 = theta * theta;
  const double theta3 = theta2 * theta;
  return I + ((1.0 - std::cos(theta)) / theta2) * W +
         ((theta - std::sin(theta)) / theta3) * W * W;
}

Matrix so3RightJacobianInverse(const Eigen::Vector3d &omega) {
  const double theta = omega.norm();
  const Matrix W = skew3(omega);
  const Matrix I = Matrix::Identity(3, 3);
  if (theta < 1e-8) {
    return I + 0.5 * W + (1.0 / 12.0) * W * W;
  }
  const double theta2 = theta * theta;
  const double coeff =
      1.0 / theta2 -
      (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
  return I + 0.5 * W + coeff * W * W;
}

Matrix so3LeftJacobianInverse(const Eigen::Vector3d &omega) {
  return so3RightJacobianInverse(-omega);
}

Matrix projectToSO3(const Matrix &R) {
  Eigen::JacobiSVD<Matrix> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix U = svd.matrixU();
  Matrix V = svd.matrixV();
  Matrix S = Matrix::Identity(3, 3);
  if ((U * V.transpose()).determinant() < 0.0) {
    S(2, 2) = -1.0;
  }
  return U * S * V.transpose();
}

Matrix applySE3RightTangentStep(
    const Matrix &pose, const Eigen::Matrix<double, 6, 1> &tangent,
    double scale) {
  if (pose.rows() != 3 || pose.cols() != 4) {
    throw std::runtime_error("applySE3RightTangentStep requires a 3x4 pose");
  }
  const Eigen::Matrix<double, 6, 1> scaled = scale * tangent;
  const Eigen::Vector3d phi = scaled.head<3>();
  const Eigen::Vector3d rho = scaled.tail<3>();
  Matrix stepped = Matrix::Zero(3, 4);
  stepped.leftCols(3) =
      projectToSO3(pose.leftCols(3) * so3Exp(phi));
  stepped.rightCols(1) =
      pose.rightCols(1) + pose.leftCols(3) * so3LeftJacobian(phi) * rho;
  return stepped;
}

GeodesicSE3Residual evaluateRelativeSE3Residual(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj) {
  GeodesicSE3Residual residual;
  residual.rotation = so3Log((Ri * Rij).transpose() * Rj);
  residual.translation = tj - ti - Ri * tij;
  return residual;
}

GeodesicSE3Residual evaluateRelativeSE3LogResidual(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj) {
  GeodesicSE3Residual residual;
  const Matrix Rerr = Rij.transpose() * Ri.transpose() * Rj;
  const Eigen::Vector3d phi = so3Log(Rerr);
  const Eigen::Vector3d p =
      Rij.transpose() * (Ri.transpose() * (tj - ti) - tij);
  residual.rotation = phi;
  residual.translation = so3LeftJacobianInverse(phi) * p;
  return residual;
}

GeodesicSE3Jacobians linearizeRelativeSE3Residual(
    const Matrix &Ri, const Matrix &tij, const Matrix &Rj,
    const Eigen::Vector3d &rotationResidual) {
  GeodesicSE3Jacobians J;
  J.Jr_i = Matrix::Zero(6, 3);
  J.Jt_i = Matrix::Zero(6, 3);
  J.Jr_j = Matrix::Zero(6, 3);
  J.Jt_j = Matrix::Zero(6, 3);

  const Matrix JrInv = so3RightJacobianInverse(rotationResidual);
  J.Jr_i.block(0, 0, 3, 3) = -JrInv * Rj.transpose() * Ri;
  J.Jr_j.block(0, 0, 3, 3) = JrInv;
  J.Jr_i.block(3, 0, 3, 3) = Ri * skew3(tij);
  J.Jt_i.block(3, 0, 3, 3) = -Matrix::Identity(3, 3);
  J.Jt_j.block(3, 0, 3, 3) = Matrix::Identity(3, 3);
  return J;
}

GeodesicSE3Jacobians linearizeRelativeSE3LogResidual(
    const Matrix &Ri, const Matrix &ti, const Matrix &Rij,
    const Matrix &tij, const Matrix &Rj, const Matrix &tj) {
  GeodesicSE3Jacobians J;
  J.Jr_i = Matrix::Zero(6, 3);
  J.Jt_i = Matrix::Zero(6, 3);
  J.Jr_j = Matrix::Zero(6, 3);
  J.Jt_j = Matrix::Zero(6, 3);

  const GeodesicSE3Residual residual =
      evaluateRelativeSE3LogResidual(Ri, ti, Rij, tij, Rj, tj);
  const Eigen::Vector3d phi = residual.rotation;
  const Matrix JrInv = so3RightJacobianInverse(phi);
  J.Jr_i.block(0, 0, 3, 3) = -JrInv * Rj.transpose() * Ri;
  J.Jr_j.block(0, 0, 3, 3) = JrInv;

  const Matrix RiT = Ri.transpose();
  const Matrix ZijT = Rij.transpose();
  const Eigen::Vector3d relativeWorld = tj - ti;
  const Eigen::Vector3d a = RiT * relativeWorld;
  const Eigen::Vector3d p = ZijT * (a - tij);
  const Matrix Vinv = so3LeftJacobianInverse(phi);

  const Matrix dp_dri = ZijT * skew3(a);
  const Matrix dp_dti = -ZijT * RiT;
  const Matrix dp_dtj = ZijT * RiT;
  J.Jt_i.block(3, 0, 3, 3) = Vinv * dp_dti;
  J.Jt_j.block(3, 0, 3, 3) = Vinv * dp_dtj;

  for (int col = 0; col < 3; ++col) {
    const Eigen::Vector3d dphi_i = J.Jr_i.block(0, col, 3, 1);
    const Eigen::Vector3d dphi_j = J.Jr_j.block(0, col, 3, 1);
    J.Jr_i.block(3, col, 3, 1) =
        Vinv * dp_dri.col(col) +
        leftJacobianInverseDirectionalProduct(phi, dphi_i, p);
    J.Jr_j.block(3, col, 3, 1) =
        leftJacobianInverseDirectionalProduct(phi, dphi_j, p);
  }
  return J;
}

}  // namespace DPGO
