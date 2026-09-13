// MIT License; see factor.hpp for copyright and license text.
#include "form/feature/summary.hpp"
#include "form/feature/factor.hpp"

#include <Eigen/QR>
#include <algorithm>

namespace form {

namespace {
Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d result;
  result << 0., -v.z(), v.y(), v.z(), 0., -v.x(), -v.y(), v.x(), 0.;
  return result;
}

// If M=a*b^T, this is a.cross(b). The operation remains linear for QR rows.
Eigen::Vector3d crossParts(const Eigen::Matrix3d &m) {
  return {m(1, 2) - m(2, 1), m(2, 0) - m(0, 2), m(0, 1) - m(1, 0)};
}

template <int Width>
Eigen::Matrix<double, Width, Width> upperFactor(Eigen::MatrixXd &features) {
  Eigen::Matrix<double, Width, Width> root =
      Eigen::Matrix<double, Width, Width>::Zero();
  if (features.rows() == 0) return root;
  // In-place QR avoids retaining a second correspondence-sized feature matrix.
  Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(features);
  const Eigen::Index rows = std::min<Eigen::Index>(features.rows(), Width);
  root.topRows(rows) =
      qr.matrixQR().topRows(rows).template triangularView<Eigen::Upper>();
  return root;
}

Eigen::Matrix3d planeTensor(const Eigen::Matrix<double, 13, 13> &root, int row) {
  Eigen::Matrix3d tensor;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) tensor(a, b) = root(row, 3 * a + b);
  return tensor;
}
} // namespace

FeatureSummary::FeatureSummary(const PlanePoint &planes, const PointPoint &points)
    : point_rows_(static_cast<int>(std::min<size_t>(7, points.num_constraints()))),
      plane_rows_(static_cast<int>(std::min<size_t>(13, planes.num_constraints()))) {
  Eigen::MatrixXd point_features = pointFeatures(points);
  point_root_ = upperFactor<7>(point_features);
  Eigen::MatrixXd plane_features = planeFeatures(planes);
  plane_root_ = upperFactor<13>(plane_features);
}

FeatureSummary::FeatureSummary(const Eigen::Matrix<double, 13, 13> &planeRoot,
                               const Eigen::Matrix<double, 7, 7> &pointRoot)
    : point_root_(pointRoot), plane_root_(planeRoot),
      point_rows_(pointRoot.isZero(0.) ? 0 : 7),
      plane_rows_(planeRoot.isZero(0.) ? 0 : 13) {}

Eigen::MatrixXd FeatureSummary::pointFeatures(const PointPoint &points) {
  points.ensureRaw();
  Eigen::MatrixXd features(points.num_constraints(), 7);
  for (Eigen::Index k = 0; k < features.rows(); ++k) {
    const Eigen::Map<const Eigen::Vector3d> pi(points.p_i.data() + 3 * k);
    const Eigen::Map<const Eigen::Vector3d> pj(points.p_j.data() + 3 * k);
    features(k, 0) = 1.;
    features.block<1, 3>(k, 1) = pi.transpose();
    features.block<1, 3>(k, 4) = (pj - pi).transpose();
  }
  return features;
}

Eigen::MatrixXd FeatureSummary::planeFeatures(const PlanePoint &planes) {
  planes.ensureRaw();
  Eigen::MatrixXd features(planes.num_constraints(), 13);
  for (Eigen::Index k = 0; k < features.rows(); ++k) {
    const Eigen::Map<const Eigen::Vector3d> pi(planes.p_i.data() + 3 * k);
    const Eigen::Map<const Eigen::Vector3d> pj(planes.p_j.data() + 3 * k);
    const Eigen::Map<const Eigen::Vector3d> n(planes.n_i.data() + 3 * k);
    for (int a = 0; a < 3; ++a)
      features.block<1, 3>(k, 3 * a) = n[a] * pj.transpose();
    features.block<1, 3>(k, 9) = n.transpose();
    features(k, 12) = n.dot(pj - pi);
  }
  return features;
}

double FeatureSummary::squaredError(const gtsam::Pose3 &Ti,
                                    const gtsam::Pose3 &Tj) const {
  const Eigen::Matrix3d Ri = Ti.rotation().matrix(), Rj = Tj.rotation().matrix();
  const Eigen::Vector3d dt = Tj.translation() - Ti.translation();
  double squared_error = 0.;
  const Eigen::Matrix3d rotation_difference = Rj - Ri;
  for (int k = 0; k < point_rows_; ++k) {
    const Eigen::Vector3d p = point_root_.block<1, 3>(k, 1).transpose();
    const Eigen::Vector3d d = point_root_.block<1, 3>(k, 4).transpose();
    const Eigen::Vector3d residual =
        rotation_difference * p + Rj * d + point_root_(k, 0) * dt;
    squared_error += residual.squaredNorm();
  }
  if (plane_rows_) {
    const Eigen::Matrix3d R_delta = Ri.transpose() * (Rj - Ri);
    const Eigen::Vector3d t = Ri.transpose() * dt;
    for (int k = 0; k < plane_rows_; ++k) {
      const Eigen::Matrix3d tensor = planeTensor(plane_root_, k);
      const double residual =
          tensor.cwiseProduct(R_delta).sum() +
          plane_root_.block<1, 3>(k, 9).dot(t) + plane_root_(k, 12);
      squared_error += residual * residual;
    }
  }
  return squared_error;
}

FeatureSummary::AugmentedHessian
FeatureSummary::augmentedHessian(const gtsam::Pose3 &Ti,
                                 const gtsam::Pose3 &Tj) const {
  const Eigen::Matrix3d Ri = Ti.rotation().matrix(), Rj = Tj.rotation().matrix();
  const Eigen::Vector3d dt = Tj.translation() - Ti.translation();
  AugmentedHessian result = AugmentedHessian::Zero();
  if (point_rows_) {
    Eigen::Matrix<double, 21, 13> augmented = Eigen::Matrix<double, 21, 13>::Zero();
    const Eigen::Matrix3d rotation_difference = Rj - Ri;
    for (int k = 0; k < point_rows_; ++k) {
      const double weight = point_root_(k, 0);
      const Eigen::Vector3d p = point_root_.block<1, 3>(k, 1).transpose();
      const Eigen::Vector3d d = point_root_.block<1, 3>(k, 4).transpose();
      augmented.block<3, 3>(3 * k, 0).noalias() = Ri * skew(p);
      augmented.block<3, 3>(3 * k, 3) = -weight * Ri;
      augmented.block<3, 3>(3 * k, 6).noalias() = -Rj * skew(p + d);
      augmented.block<3, 3>(3 * k, 9) = weight * Rj;
      augmented.block<3, 1>(3 * k, 12) =
          -(rotation_difference * p + Rj * d + weight * dt);
    }
    result.selfadjointView<Eigen::Lower>().rankUpdate(augmented.transpose());
  }
  if (plane_rows_) {
    const Eigen::Matrix3d R = Ri.transpose() * Rj;
    const Eigen::Matrix3d R_delta = Ri.transpose() * (Rj - Ri);
    const Eigen::Vector3d t = Ri.transpose() * dt;
    AugmentedHessian augmented = AugmentedHessian::Zero();
    for (int k = 0; k < plane_rows_; ++k) {
      const Eigen::Matrix3d tensor = planeTensor(plane_root_, k);
      const Eigen::Vector3d normal = plane_root_.block<1, 3>(k, 9).transpose();
      augmented.block<1, 3>(k, 0) =
          (crossParts(tensor * R.transpose()) + normal.cross(t)).transpose();
      augmented.block<1, 3>(k, 3) = -normal.transpose();
      augmented.block<1, 3>(k, 6) = crossParts(tensor.transpose() * R).transpose();
      augmented.block<1, 3>(k, 9) = normal.transpose() * R;
      augmented(k, 12) = -(tensor.cwiseProduct(R_delta).sum() + normal.dot(t) +
                           plane_root_(k, 12));
    }
    result.selfadjointView<Eigen::Lower>().rankUpdate(augmented.transpose());
  }
  result.triangularView<Eigen::StrictlyUpper>() = result.transpose();
  return result;
}
} // namespace form
