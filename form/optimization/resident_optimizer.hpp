// MIT License; see form/feature/factor.hpp for copyright and license text.
#pragma once

#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <memory>

namespace form {
struct ResidentResult {
  gtsam::Values values;
  double initial_error = 0, final_error = 0, lambda = 0;
  size_t iterations = 0, inner_iterations = 0;
};

/// Reusable, serialized dense Pose3 graph workspace and GTSAM 4.2 LM controller.
/// reset() snapshots summarized/frozen factors; other factors retain virtual dispatch.
class ResidentOptimizer {
 public:
  explicit ResidentOptimizer(bool gpu = false);
  ~ResidentOptimizer();
  void reset(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values);
  const gtsam::KeyVector& keys() const;
  void linearize(const gtsam::Values& values);
  Eigen::MatrixXd model();  // Diagnostic: transfers the model on CUDA.
  double error(const gtsam::Values& values);
  bool solve(double lambda, bool diagonal, double min_diagonal, double max_diagonal,
             Eigen::VectorXd& delta, double& old_linear_error, double& new_linear_error);
  ResidentResult optimize(const gtsam::Values& initial,
                          const gtsam::LevenbergMarquardtParams& params);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace form
