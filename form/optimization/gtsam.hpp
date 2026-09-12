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

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/linear/Scatter.h>
#include "form/optimization/profile.hpp"
#ifdef FORM_ENABLE_CUDA
#include "form/optimization/cuda_solver.hpp"
#endif

/*
This file contains a handful of helper to help gtsam run dense optimizations.

Most of gtsam is optimized for sparse problems, so we shortcut a handful of places to
make it faster when sparsity isn't a concern.
*/
namespace form {
class CudaDenseSolver;

/// @brief Extension of LM that uses dense optimization instead of sparse bayes tree
/// solver.
class DenseLMOptimizer : public gtsam::LevenbergMarquardtOptimizer {
  std::shared_ptr<CudaDenseSolver> cuda_solver_;
  int cuda_min_dimension_;
public:
  /// @brief Constructor
  DenseLMOptimizer(const gtsam::NonlinearFactorGraph &graph,
                   const gtsam::Values &initialValues,
                   const gtsam::LevenbergMarquardtParams &params,
                   std::shared_ptr<CudaDenseSolver> cuda_solver = {},
                   int cuda_min_dimension = 240)
      : LevenbergMarquardtOptimizer(graph, initialValues, params),
        cuda_solver_(std::move(cuda_solver)), cuda_min_dimension_(cuda_min_dimension) {}

  /// @brief Solve densely instead of using the sparse solver
  gtsam::VectorValues
  solve(const gtsam::GaussianFactorGraph &gfg,
        const gtsam::NonlinearOptimizerParams &params) const override {
    if (!profile::enabled && !cuda_solver_) return gfg.optimizeDensely();
    // Same operations as GTSAM 4.2 optimizeDensely(), with the assembly and
    // numerical solve measured separately.
    auto start = profile::Clock::now();
    gtsam::Scatter scatter(gfg);
    gtsam::HessianFactor combined(gfg, scatter);
    gtsam::Matrix augmented = combined.info().selfadjointView();
    if (profile::enabled)
      profile::assemble += static_cast<uint64_t>(profile::milliseconds(start) * 1e6);
    if (profile::enabled && profile::solve_observer) profile::solve_observer(augmented);
    profile::Scope timer(profile::solve);
    const auto n = augmented.rows() - 1;
#ifdef FORM_ENABLE_CUDA
    if (cuda_solver_ && n >= cuda_min_dimension_) {
      profile::cuda_solve_calls.fetch_add(1, std::memory_order_relaxed);
      gtsam::Vector solution;
      if (cuda_solver_->solve(augmented, solution))
        return gtsam::VectorValues(solution, scatter);
      profile::cuda_solve_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
#endif
    Eigen::LLT<gtsam::Matrix, Eigen::Upper> llt(augmented.topLeftCorner(n, n));
    gtsam::Vector solution = llt.solve(augmented.topRightCorner(n, 1));
    return gtsam::VectorValues(solution, scatter);
  }

  gtsam::GaussianFactorGraph::shared_ptr linearize() const override {
    profile::Scope timer(profile::linearize_wall);
    profile::iterations.fetch_add(1, std::memory_order_relaxed);
    return LevenbergMarquardtOptimizer::linearize();
  }
};

/// @brief Factor that forces gtsam factors to use Hessian linearization instead of
/// Jacobian linearization. This is more efficient when residual sizes are large and
/// the problem is dense.
class DenseFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
public:
  DenseFactor(const gtsam::noiseModel::Base::shared_ptr &noiseModel,
              const gtsam::Key i, const gtsam::Key j)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(noiseModel, i, j) {}

  // Blatantly stolen from here, just with last line changed:
  // https://github.com/borglab/gtsam/blob/develop/gtsam/nonlinear/NonlinearFactor.cpp#L152
  boost::shared_ptr<gtsam::GaussianFactor>
  linearize(const gtsam::Values &x) const override {
    profile::Scope timer(profile::factor_linearize);
    // Call evaluate error to get Jacobians and RHS vector b
    std::vector<gtsam::Matrix> A(size());
    gtsam::Vector b = -unwhitenedError(x, A);

    // Whiten the corresponding system now
    if (noiseModel_)
      noiseModel_->WhitenSystem(A, b);

    // Fill in terms, needed to create gtsam::JacobianFactor below
    std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms(size());
    for (size_t j = 0; j < size(); ++j) {
      terms[j].first = keys()[j];
      terms[j].second.swap(A[j]);
    }

    return gtsam::GaussianFactor::shared_ptr(
        new gtsam::HessianFactor(gtsam::JacobianFactor(terms, b)));
  }
};

/// @brief A fast isotropic noise model that uses a single sigma value for all
/// dimensions. The gtsam version allocates a vector of sigmas, which is unnecessary
class FastIsotropic : public gtsam::noiseModel::Gaussian {
  typedef boost::shared_ptr<FastIsotropic> shared_ptr;

  double sigma_, invsigma_;

  FastIsotropic(double sigma, size_t dim)
      : sigma_(sigma), invsigma_(1.0 / sigma), gtsam::noiseModel::Gaussian(dim) {}

public:
  // For creation
  static shared_ptr Sigma(double sigma, size_t dim) {
    return shared_ptr(new FastIsotropic(sigma, dim));
  }

  // Override virtuals
  void print(const std::string &s = "") const override {
    std::cout << s << "FastIsotropic(sigma=" << sigma_ << ", dim=" << dim() << ")"
              << std::endl;
  }

  bool equals(const gtsam::noiseModel::Base &expected,
              double tol = 1e-9) const override {
    const FastIsotropic *p = dynamic_cast<const FastIsotropic *>(&expected);
    if (p == nullptr)
      return false;
    if (typeid(*this) != typeid(*p))
      return false;
    return gtsam::fpEqual(sigma_, p->sigma_, tol) && dim() == p->dim();
  }

  gtsam::Vector whiten(const gtsam::Vector &v) const override {
    return v * invsigma_;
  }

  gtsam::Vector unwhiten(const gtsam::Vector &v) const override {
    return v * sigma_;
  }

  gtsam::Matrix Whiten(const gtsam::Matrix &H) const override {
    return invsigma_ * H;
  }

  void WhitenInPlace(gtsam::Matrix &H) const override { H *= invsigma_; }

  void whitenInPlace(gtsam::Vector &v) const override { v *= invsigma_; }

  void WhitenInPlace(Eigen::Block<gtsam::Matrix> H) const override {
    H *= invsigma_;
  }
};

/// @brief Wrapper around a gtsam::NoiseModelFactor2 to allow for a fixed FIRST
/// variable
class BinaryFactorWrapper : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;
  using Wrapped = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

private:
  boost::shared_ptr<Wrapped> factor_;
  gtsam::Pose3 pose_i_;
  bool use_dense_;

public:
  BinaryFactorWrapper(const gtsam::Pose3 &pose_i, const gtsam::Key &key,
                      const boost::shared_ptr<Wrapped> &factor, bool use_dense = false)
      : Base(factor->noiseModel(), key), factor_(factor), pose_i_(pose_i),
        use_dense_(use_dense) {}

  template <typename T>
  static BinaryFactorWrapper Create(const gtsam::Pose3 &pose, const gtsam::Key &key,
                                    const T &factor, bool use_dense = false) {
    return BinaryFactorWrapper(pose, key, boost::make_shared<T>(factor), use_dense);
  }

  /// Preserve the wrapped factor's dense/summarized implementation when requested.
  boost::shared_ptr<gtsam::GaussianFactor>
  linearize(const gtsam::Values &values) const override {
    if (!use_dense_) return Base::linearize(values);
    if (!active(values)) return {};
    gtsam::Values binary_values;
    binary_values.insert(factor_->key1(), pose_i_);
    binary_values.insert(factor_->key2(), values.at<gtsam::Pose3>(key()));
    const auto linear = factor_->linearize(binary_values);
    if (!linear) return {};
    const auto h = linear->augmentedInformation();
    return boost::make_shared<gtsam::HessianFactor>(key(), h.block<6, 6>(6, 6),
                                                  h.block<6, 1>(6, 12), h(12, 12));
  }

  double error(const gtsam::Values &values) const override {
    if (!use_dense_) return Base::error(values);
    if (!active(values)) return 0.0;
    gtsam::Values binary_values;
    binary_values.insert(factor_->key1(), pose_i_);
    binary_values.insert(factor_->key2(), values.at<gtsam::Pose3>(key()));
    return factor_->error(binary_values);
  }

  // Evaluate the factor
  gtsam::Vector
  evaluateError(const gtsam::Pose3 &pose_j,
                boost::optional<gtsam::Matrix &> H = boost::none) const override {
    return factor_->evaluateError(pose_i_, pose_j, boost::none, H);
  }
};

} // namespace form
