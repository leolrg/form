// MIT License

// Copyright (c) 2025 Easton Potokar, Taylor Pool, and Michael Kaess

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <vector>
#include <functional>
#include <optional>
#include <mutex>

#include "form/feature/features.hpp"
#include "form/optimization/gtsam.hpp"

namespace form {
struct FeatureSummary;
struct PointPoint;

static void check(const gtsam::SharedNoiseModel &noiseModel, size_t m) {
  if (noiseModel && m != noiseModel->dim())
    throw std::invalid_argument("bad noise model number");
}

using OptionalJacobian = gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

/// @brief Structure that holds plane-point correspondences and computes their error
///
/// Points and normals are stored in a std::vector that is then mapped to Eigen
/// matrices for efficient computation.
struct PlanePoint {
  /// @brief Shared pointer type
  typedef std::shared_ptr<PlanePoint> Ptr;

  /// @brief Vectors to hold point and normal data.
  /// Call ensureRaw() before direct access, then invalidateSummary() after edits.
  mutable std::vector<double> p_i;
  mutable std::vector<double> n_i;
  mutable std::vector<double> p_j;
  // Reused by all graph instances until either correspondence set changes.
  std::shared_ptr<const FeatureSummary> summary_cache;
  size_t summary_point_revision = 0;
  std::weak_ptr<PointPoint> summary_point_owner;

  PlanePoint() = default;

  // Summary backends can retain rows outside host memory. Counts are available
  // immediately; raw evaluation and explicit array access call ensureRaw().
  void deferRaw(size_t count, std::function<void(const PlanePoint&)> loader) {
    clear();
    if (!loader) throw std::invalid_argument("Deferred rows require a loader");
    raw_mutex_ = std::make_shared<std::mutex>();
    raw_loader_ = std::move(loader);
    deferred_count_ = count;
  }
  void ensureRaw() const {
    if (!deferred_count_) return;
    std::lock_guard<std::mutex> lock(*raw_mutex_);
    if (raw_loader_) { raw_loader_(*this); raw_loader_ = {}; }
  }
private:
  std::optional<size_t> deferred_count_;
  std::shared_ptr<std::mutex> raw_mutex_;
  mutable std::function<void(const PlanePoint&)> raw_loader_;
public:


  /// Invalidate cached summaries after direct mutation of correspondence arrays.
  /// Factors already constructed retain their immutable summary snapshots.
  void invalidateSummary() {
    ensureRaw();
    deferred_count_.reset();
    summary_cache.reset();
    summary_point_owner.reset();
  }

  bool summaryValidFor(const std::shared_ptr<PointPoint> &points) const noexcept;

  /// @brief Add a new plane-point correspondence
  void push_back(const PlanarFeat &p_i_, const PlanarFeat &p_j_) {
    invalidateSummary();
    p_i.insert(p_i.end(), {p_i_.x, p_i_.y, p_i_.z});
    n_i.insert(n_i.end(), {p_i_.nx, p_i_.ny, p_i_.nz});
    p_j.insert(p_j.end(), {p_j_.x, p_j_.y, p_j_.z});
  }

  /// @brief Clear all stored correspondences
  void clear() noexcept {
    raw_loader_ = {};
    deferred_count_.reset();
    invalidateSummary();
    p_i.clear();
    n_i.clear();
    p_j.clear();
  }

  /// @brief Get the number of residuals (one per correspondence)
  size_t num_residuals() const noexcept { return num_constraints(); }

  /// @brief Get the number of constraints / correspondences
  size_t num_constraints() const noexcept { return deferred_count_ ? *deferred_count_ : p_i.size() / 3; }

  /// @brief Evaluate the residual given two poses
  ///
  /// @param Ti The pose of the first scan
  /// @param Tj The pose of the second scan
  /// @param residual_D_Ti Optional Jacobian of the residual w.r.t. Ti
  /// @param residual_D_Tj Optional Jacobian of the residual w.r.t. Tj
  /// @return The residual vector
  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const;
};

/// @brief Structure that holds point-point correspondences and computes their error
///
/// Points are stored in a std::vector that is then mapped to Eigen matrices for
/// efficient computation.
struct PointPoint {
  /// @brief Shared pointer type
  typedef std::shared_ptr<PointPoint> Ptr;

  /// @brief Vectors to hold point data.
  /// Call ensureRaw() before direct access, then invalidateSummary() after edits.
  mutable std::vector<double> p_i;
  mutable std::vector<double> p_j;
  size_t revision = 0;

  PointPoint() = default;

  // Summary backends can retain rows outside host memory. Counts are available
  // immediately; raw evaluation and explicit array access call ensureRaw().
  void deferRaw(size_t count, std::function<void(const PointPoint&)> loader) {
    clear();
    if (!loader) throw std::invalid_argument("Deferred rows require a loader");
    raw_mutex_ = std::make_shared<std::mutex>();
    raw_loader_ = std::move(loader);
    deferred_count_ = count;
  }
  void ensureRaw() const {
    if (!deferred_count_) return;
    std::lock_guard<std::mutex> lock(*raw_mutex_);
    if (raw_loader_) { raw_loader_(*this); raw_loader_ = {}; }
  }
private:
  std::optional<size_t> deferred_count_;
  std::shared_ptr<std::mutex> raw_mutex_;
  mutable std::function<void(const PointPoint&)> raw_loader_;
public:


  /// Invalidate dependent summaries after direct mutation of correspondence arrays.
  void invalidateSummary() { ensureRaw(); deferred_count_.reset(); ++revision; }

  /// @brief Add a new point-point correspondence
  void push_back(const PointFeat &p_i_, const PointFeat &p_j_) {
    invalidateSummary();
    p_i.insert(p_i.end(), {p_i_.x, p_i_.y, p_i_.z});
    p_j.insert(p_j.end(), {p_j_.x, p_j_.y, p_j_.z});
  }

  /// @brief Clear all stored correspondences
  void clear() noexcept {
    raw_loader_ = {};
    deferred_count_.reset();
    invalidateSummary();
    p_i.clear();
    p_j.clear();
  }

  /// @brief Get the number of residuals (three per correspondence)
  size_t num_residuals() const noexcept { return 3 * num_constraints(); }

  /// @brief Get the number of constraints / correspondences
  size_t num_constraints() const noexcept { return deferred_count_ ? *deferred_count_ : p_i.size() / 3; }

  /// @brief Evaluate the residual given two poses
  ///
  /// @param Ti The pose of the first scan
  /// @param Tj The pose of the second scan
  /// @param residual_D_Ti Optional Jacobian of the residual w.r.t. Ti
  /// @param residual_D_Tj Optional Jacobian of the residual w.r.t. Tj
  /// @return The residual vector
  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                OptionalJacobian residual_D_Ti = boost::none,
                OptionalJacobian residual_D_Tj = boost::none) const;
};

/// @brief A wrapper factor that holds both plane-point and point-point constraints
class FeatureFactor : public DenseFactor {
public:
  /// @brief Plane-point constraints
  PlanePoint::Ptr plane_point;

  /// @brief Point-point constraints
  PointPoint::Ptr point_point;

public:
  /// @brief Constructor
  ///
  /// @param i Key for the first pose
  /// @param j Key for the second pose
  /// @param constraint A tuple containing plane-point and point-point constraints
  /// @param sigma The isotropic noise standard deviation
  FeatureFactor(const gtsam::Key i, const gtsam::Key j,
                const std::tuple<PlanePoint::Ptr, PointPoint::Ptr> &constraint,
                double sigma, bool use_summary = false);

  boost::shared_ptr<gtsam::GaussianFactor>
  linearize(const gtsam::Values &values) const override;
  double error(const gtsam::Values &values) const override;
  const auto& summary() const noexcept { return summary_; }
  double inverseVariance() const noexcept { return inverse_variance_; }

private:
  std::shared_ptr<const FeatureSummary> summary_;
  double inverse_variance_;

public:

  /// @brief Evaluate the residual given two poses
  ///
  /// @param Ti The pose of the first scan
  /// @param Tj The pose of the second scan
  /// @param residual_D_Ti Optional Jacobian of the residual w.r.t. Ti
  /// @param residual_D_Tj Optional Jacobian of the residual w.r.t. Tj
  /// @return The residual vector
  [[nodiscard]] gtsam::Vector
  evaluateError(const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj,
                boost::optional<gtsam::Matrix &> residual_D_Ti = boost::none,
                boost::optional<gtsam::Matrix &> residual_D_Tj =
                    boost::none) const override;
};

} // namespace form
