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
  CudaMatcher();
  ~CudaMatcher();
  CudaMatcher(const CudaMatcher&) = delete;
  CudaMatcher& operator=(const CudaMatcher&) = delete;
  void reset(const std::vector<Voxel>& voxels, const std::vector<MapPoint>& points,
             const std::vector<Query>& queries, double voxel_width);
  const std::vector<Result>& search(const std::array<double,12>& world_T_query);
  // Query indices in each group define the row order. Uses the latest search;
  // only accepted matches may be supplied. Coordinates stay on device through QR.
  std::vector<Eigen::MatrixXd> summarize(const std::vector<std::vector<int>>& groups,
                                       bool plane);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
