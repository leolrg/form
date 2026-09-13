#pragma once
#include "form/feature/factor.hpp"
#include "form/mapping/map.hpp"
#include <functional>
#include <memory>

namespace form {
// Adapter retaining raw FORM matches while producing exact summaries on device.
// One caller, one incoming scan at a time; reset after the world map changes.
class CudaMatching {
public:
  using ConstraintMap=tsl::robin_map<size_t,std::tuple<PlanePoint::Ptr,PointPoint::Ptr>>;
  CudaMatching();
  ~CudaMatching();
  void reset(const VoxelMap<PlanarFeat>& planes,const VoxelMap<PointFeat>& points,
             const std::vector<PlanarFeat>& plane_queries,const std::vector<PointFeat>& point_queries,
             const std::function<gtsam::Pose3(size_t)>& estimates,double voxel_width);
  void match(const gtsam::Pose3& pose,double max_distance,ConstraintMap& constraints,
             tbb::concurrent_vector<Match<PlanarFeat>>& planes,
             tbb::concurrent_vector<Match<PointFeat>>& points);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
