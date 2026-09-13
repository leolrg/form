#include "form/optimization/cuda_matching.hpp"
#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/summary.hpp"
#include <unordered_map>
#include <type_traits>

namespace form {
namespace {
template<class Point> struct Snapshot {
  CudaMatcher device;
  std::vector<Point> local_points,queries;
  void reset(const VoxelMap<Point>& map,const std::vector<Point>& query,
             const std::function<gtsam::Pose3(size_t)>& estimates,double width) {
    queries=query; local_points.clear();
    std::vector<CudaMatcher::Voxel> voxels;
    std::vector<CudaMatcher::MapPoint> points;
    std::unordered_map<size_t,gtsam::Pose3> inverse_poses;
    voxels.reserve(map.size());
    for(const auto& [coords,voxel]:map) {
      voxels.push_back({{coords.x(),coords.y(),coords.z()},int(points.size()),int(voxel.size())});
      for(const auto& world:voxel) {
        auto it=inverse_poses.find(world.scan);
        if(it==inverse_poses.end()) it=inverse_poses.emplace(world.scan,estimates(world.scan).inverse()).first;
        auto local=world.transform(it->second);
        local_points.push_back(local);
        CudaMatcher::MapPoint p{{world.x,world.y,world.z,world._},{local.x,local.y,local.z},{0,0,0}};
        if constexpr(std::is_same_v<Point,PlanarFeat>) {
          p.normal[0]=local.nx; p.normal[1]=local.ny; p.normal[2]=local.nz;
        }
        points.push_back(p);
      }
    }
    std::vector<CudaMatcher::Query> packed;
    packed.reserve(query.size());
    for(const auto& q:query) packed.push_back({{q.x,q.y,q.z,q._}});
    device.reset(voxels,points,packed,width);
  }
  template<int I> std::vector<Eigen::MatrixXd> match(const std::array<double,12>& pose,double threshold,
      CudaMatching::ConstraintMap& constraints,const std::unordered_map<size_t,size_t>& slots,
      tbb::concurrent_vector<Match<Point>>& matches) {
    const auto& results=device.search(pose);
    // Match the CPU adapter's empty-feature behavior, including retaining its
    // previous raw matches for map insertion. Do not silently change FORM here.
    if(queries.empty()) return device.summarize(std::vector<std::vector<int>>(slots.size()),I==0);
    matches.clear(); matches.reserve(queries.size());
    std::vector<std::vector<int>> groups(slots.size());
    for(size_t i=0;i<queries.size();++i) {
      Match<Point> match{};
      match.query=queries[i]; match.dist_sqrd=results[i].distance;
      if(results[i].index>=0) {
        match.point=local_points.at(results[i].index);
        if(match.dist_sqrd<threshold) {
          const auto scan=match.point.scan;
          groups.at(slots.at(scan)).push_back(int(i));
          std::get<I>(constraints.at(scan))->push_back(match.point,match.query);
        }
      }
      matches.push_back(match);
    }
    return device.summarize(groups,I==0);
  }
};
}
struct CudaMatching::Impl {
  Snapshot<PlanarFeat> planes;
  Snapshot<PointFeat> points;
};
CudaMatching::CudaMatching():impl_(std::make_unique<Impl>()) {}
CudaMatching::~CudaMatching()=default;
void CudaMatching::reset(const VoxelMap<PlanarFeat>& planes,const VoxelMap<PointFeat>& points,
    const std::vector<PlanarFeat>& pq,const std::vector<PointFeat>& qq,
    const std::function<gtsam::Pose3(size_t)>& estimates,double width) {
  impl_->planes.reset(planes,pq,estimates,width);
  impl_->points.reset(points,qq,estimates,width);
}
void CudaMatching::match(const gtsam::Pose3& pose,double max_distance,ConstraintMap& constraints,
    tbb::concurrent_vector<Match<PlanarFeat>>& planes,tbb::concurrent_vector<Match<PointFeat>>& points) {
  if(!std::isfinite(max_distance) || max_distance<=0) throw std::invalid_argument("Invalid CUDA matching distance");
  std::unordered_map<size_t,size_t> slots;
  std::vector<std::tuple<PlanePoint::Ptr,PointPoint::Ptr>> pairs;
  for(auto& [scan,pair]:constraints) {
    slots.emplace(scan,pairs.size()); pairs.push_back(pair);
    std::get<0>(pair)->clear(); std::get<1>(pair)->clear();
  }
  std::array<double,12> packed;
  const auto matrix=pose.matrix();
  for(int i=0;i<3;++i) for(int j=0;j<4;++j) packed[4*i+j]=matrix(i,j);
  auto pr=impl_->planes.match<0>(packed,max_distance*max_distance,constraints,slots,planes);
  auto qr=impl_->points.match<1>(packed,max_distance*max_distance,constraints,slots,points);
  for(size_t i=0;i<pairs.size();++i) {
    auto& p=std::get<0>(pairs[i]); auto& q=std::get<1>(pairs[i]);
    p->summary_cache=std::make_shared<FeatureSummary>(pr[i],qr[i]);
    p->summary_point_owner=q; p->summary_point_revision=q->revision;
  }
}
}
