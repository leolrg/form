#pragma once
#include <Eigen/Core>
#include <array>
#include <memory>
#include <vector>

namespace form {
// Scan-lifetime snapshot. No GTSAM or CUDA types cross this interface.
class CudaMatcher {
public:
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
