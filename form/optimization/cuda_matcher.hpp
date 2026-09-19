#pragma once
#include <Eigen/Core>
#include <array>
#include <memory>
#include <vector>

namespace form {
// Scan-lifetime snapshot. No GTSAM or CUDA types cross this interface.
class CudaMatcher {
public:
  enum class ReuseMode { Disabled, Audit, Certified };
  enum class SearchKernel { Fused, Split };
  struct ReuseStats {
    unsigned long long total=0, certified=0, searched=0, cell_fallback=0, gap_fallback=0;
    unsigned long long unchanged=0, mismatches=0, oracle_searched=0, split_queries=0;
  };
  void setReuseMode(ReuseMode mode);
  // Research ablations; both setters invalidate existing certificates.
  void setSearchKernel(SearchKernel kernel);
  void setOccurrenceBound(bool enabled);
  // Statistics for the latest search; downloads only when requested.
  ReuseStats reuseStats();
  struct SummaryStats {
    size_t active_leaves=0, dirty_leaves=0, full_rebuild_checks=0;
    double relative_gram_error=0.;
    bool tree=false;
  };
  void setIncrementalSummaries(bool enabled);
  // Select cached ancestor maintenance; enabling this also enables summaries.
  void setSummaryTree(bool enabled);
  SummaryStats summaryStats();
  struct Voxel { int coords[3]; int begin, count; };
  struct MapPoint { double world[4], local[3], normal[3]; };
  struct Query { double point[4]; };
  struct Result { double distance; int index; };
  struct GroupedSummary {
    std::vector<Eigen::MatrixXd> roots;
    std::vector<size_t> counts;
  };
  CudaMatcher();
  ~CudaMatcher();
  CudaMatcher(const CudaMatcher&) = delete;
  CudaMatcher& operator=(const CudaMatcher&) = delete;
  // Optional inverse poses transform world coordinates and world normals once
  // on device into local/normal fields; world (including padding) stays intact.
  void reset(const std::vector<Voxel>& voxels, const std::vector<MapPoint>& points,
             const std::vector<Query>& queries, double voxel_width,
             const std::vector<std::array<double,12>>& inverse_poses = {},
             const std::vector<int>& point_pose_indices = {});
  const std::vector<Result>& search(const std::array<double,12>& world_T_query);
  // One entry per map point; -1 excludes a point, other IDs index group_count.
  // Configuration lasts until reset or the next setGroups call.
  void setGroups(const std::vector<int>& target_groups, size_t group_count);
  // Strict distance < threshold_squared; threshold must be finite and positive.
  // Stable query order within each group. Downloads only counts and roots.
  GroupedSummary searchGrouped(const std::array<double,12>& world_T_query,
                               double threshold_squared, bool plane);
  // Lazily downloads and caches all results from the latest completed search.
  const std::vector<Result>& downloadResults();
  // Query indices in each group define the row order. Uses the latest search;
  // only accepted matches may be supplied. Coordinates stay on device through QR.
  std::vector<Eigen::MatrixXd> summarize(const std::vector<std::vector<int>>& groups,
                                       bool plane);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
