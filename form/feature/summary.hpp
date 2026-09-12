// MIT License; see factor.hpp for copyright and license text.
#pragma once

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

namespace form {
struct PlanePoint;
struct PointPoint;

/// Fixed-size sufficient statistics for isotropic feature least squares.
///
/// A point correspondence has the fixed feature row q=[1, pi, pj-pi]. Every
/// component of its WORLD-frame residual and its two right-local Pose3
/// Jacobians is linear in q. A plane correspondence has the feature row
/// z=[vec_row(n*pj^T), n, n.dot(pj-pi)]. Writing R=Ri^T*Rj and
/// t=Ri^T*(tj-ti), its residual is z.dot([vec_row(R-I), t, 1]); its two
/// Jacobians are also linear in z.
///
/// Householder QR of these feature matrices retains only their upper factors.
/// Multiplication by the discarded, pose-independent orthonormal matrices
/// preserves r^T*r, J^T*r and J^T*J for every pose. There are at most seven
/// synthetic point correspondences and thirteen synthetic plane residuals.
/// In particular the point Jacobians remain WORLD-frame Jacobians: rotating
/// their residual into pose i would preserve cost but change Gauss-Newton.
///
/// Difference features and QR avoid subtracting large Gram-matrix quadratic
/// terms when the residual is small. No eigenvalue cutoff or regularization
/// is applied, including for rank-deficient inputs. Construction snapshots
/// the correspondence arrays; later mutations require a new summary.
struct FeatureSummary {
  using AugmentedHessian = Eigen::Matrix<double, 13, 13>;

  FeatureSummary(const PlanePoint &planes, const PointPoint &points);
  /// Construct from externally computed QR upper factors, with zero padding
  /// for fewer than 13 plane or 7 point feature rows. Do not pass Gram matrices.
  FeatureSummary(const Eigen::Matrix<double, 13, 13> &planeRoot,
                 const Eigen::Matrix<double, 7, 7> &pointRoot);
  /// Column-major fixed feature matrices shared by CPU and batched QR paths.
  static Eigen::MatrixXd planeFeatures(const PlanePoint &planes);
  static Eigen::MatrixXd pointFeatures(const PointPoint &points);

  /// Unwhitened sum of squared residuals (without the usual factor of 1/2).
  double squaredError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj) const;

  /// Unwhitened [Ji, Jj, -r]^T [Ji, Jj, -r], with rotation then translation
  /// for each pose. Its last column is the Gaussian right-hand side -J^T*r.
  AugmentedHessian augmentedHessian(const gtsam::Pose3 &Ti,
                                    const gtsam::Pose3 &Tj) const;

  const auto& pointRoot() const noexcept { return point_root_; }
  const auto& planeRoot() const noexcept { return plane_root_; }

private:
  Eigen::Matrix<double, 7, 7> point_root_ = Eigen::Matrix<double, 7, 7>::Zero();
  Eigen::Matrix<double, 13, 13> plane_root_ = Eigen::Matrix<double, 13, 13>::Zero();
  int point_rows_ = 0;
  int plane_rows_ = 0;
};
} // namespace form
