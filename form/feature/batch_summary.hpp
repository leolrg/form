// MIT License; see factor.hpp for copyright and license text.
#pragma once
#include "form/feature/summary.hpp"
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
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
