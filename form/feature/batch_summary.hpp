// MIT License; see factor.hpp for copyright and license text.
#pragma once
#include "form/feature/summary.hpp"
#include "form/optimization/resident_system.hpp"
#include <memory>
#include <vector>
namespace form {
struct SummaryEdge {
  int i, j;
  std::shared_ptr<const FeatureSummary> summary;
  double weight; // inverse variance, applied to cost and normal equations
};
/// Snapshot of a pose graph's feature summaries. Calls on one instance must be serialized.
class BatchSummary {
 public:
  BatchSummary(int pose_count, std::vector<SummaryEdge> edges, bool cuda);
  ~BatchSummary();
  void reset(int pose_count, std::vector<SummaryEdge> edges);
  Eigen::MatrixXd linearize(const std::vector<gtsam::Pose3>& poses);
  double error(const std::vector<gtsam::Pose3>& poses);
  // CUDA-only resident pipeline. Frozen deltas concatenate 6 values per local
  // pose in each FrozenSystem; aux values concatenate full column-major blocks.
  void configureResident(const std::vector<FrozenSystem>& frozen,
                         const std::vector<std::vector<int>>& auxiliary_poses);
  void residentLinearize(const std::vector<gtsam::Pose3>& poses,
                         const std::vector<double>& frozen_deltas,
                         const std::vector<double>& auxiliary_values);
  double residentError(const std::vector<gtsam::Pose3>& poses,
                       const std::vector<double>& frozen_deltas);
  bool residentSolve(double lambda, bool diagonal, double min_diagonal,
                     double max_diagonal, Eigen::VectorXd& delta,
                     double& old_linear_error, double& new_linear_error);
  Eigen::MatrixXd residentHessian(); // diagnostic transfer, not used by optimizer

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
