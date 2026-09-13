#include "form/optimization/cuda_matching.hpp"
#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/summary.hpp"
#include <unordered_map>
#include <type_traits>
#include <mutex>
#include <tbb/parallel_for.h>

namespace form {
namespace {
std::array<double,12> packPose(const gtsam::Pose3& pose) {
  std::array<double,12> packed;
  const auto matrix=pose.matrix();
  for(int i=0;i<3;++i) for(int j=0;j<4;++j) packed[4*i+j]=matrix(i,j);
  return packed;
}

template<class Point> struct HostMap {
  std::vector<Point> world_points,queries;
  std::vector<int> pose_indices;
  std::vector<gtsam::Pose3> inverse_poses;
  Point local(size_t index) const {
    return world_points.at(index).transform(inverse_poses.at(pose_indices.at(index)));
  }
};

// A rematch owns immutable host metadata and leases the current device results.
// Before the device is reused, surviving leases are frozen to host memory.
// No correspondence owner is retained here: its loader owns this batch.
template<class Point> struct MatchBatch {
  std::shared_ptr<const HostMap<Point>> map;
  std::shared_ptr<CudaMatcher> device;
  std::shared_ptr<const std::vector<int>> target_groups;
  size_t group_count;
  double threshold;
  std::mutex mutex;
  bool loaded=false;
  std::vector<Match<Point>> matches;
  std::vector<std::vector<int>> groups;

  void ensure() {
    std::lock_guard<std::mutex> lock(mutex);
    if(loaded) return;
    const auto& results=device->downloadResults();
    matches.resize(map->queries.size());
    tbb::parallel_for(size_t(0),matches.size(),[&](size_t i) {
      Match<Point> match{};
      match.query=map->queries[i]; match.dist_sqrd=results[i].distance;
      if(results[i].index>=0) match.point=map->local(results[i].index);
      matches[i]=match;
    });
    groups.assign(group_count,{});
    for(size_t i=0;i<matches.size();++i) {
      const auto& result=results[i];
      if(result.index>=0 && result.distance<threshold) {
        const int group=target_groups->at(result.index);
        if(group>=0) groups.at(group).push_back(int(i));
      }
    }
    loaded=true;
    device.reset();
  }
  template<class Rows> void load(const Rows& rows,size_t group) {
    ensure();
    const auto& indices=groups.at(group);
    rows.p_i.resize(3*indices.size()); rows.p_j.resize(3*indices.size());
    if constexpr(std::is_same_v<Point,PlanarFeat>) rows.n_i.resize(3*indices.size());
    for(size_t k=0;k<indices.size();++k) {
      const auto& match=matches[indices[k]];
      for(int axis=0;axis<3;++axis) {
        rows.p_i[3*k+axis]=match.point.vec3()[axis];
        rows.p_j[3*k+axis]=match.query.vec3()[axis];
        if constexpr(std::is_same_v<Point,PlanarFeat>) rows.n_i[3*k+axis]=match.point.n_vec3()[axis];
      }
    }
  }
  void materialize(tbb::concurrent_vector<Match<Point>>& output) {
    // Preserve FORM's existing stale raw-match behavior for empty feature input.
    if(map->queries.empty()) return;
    ensure();
    output.clear();
    std::copy(matches.begin(),matches.end(),output.grow_by(matches.size()));
  }
};

template<class Point> struct Snapshot {
  std::shared_ptr<CudaMatcher> device=std::make_shared<CudaMatcher>();
  std::shared_ptr<HostMap<Point>> map;
  std::shared_ptr<MatchBatch<Point>> current,idle_batch;
  std::shared_ptr<std::vector<int>> target_groups;
  std::vector<size_t> scan_order;
  // Upload staging is never captured by deferred callbacks. reset() completes
  // device reads before returning, so these capacities can span snapshots.
  std::vector<CudaMatcher::Voxel> voxels;
  std::vector<CudaMatcher::MapPoint> points;
  std::vector<CudaMatcher::Query> packed;
  std::vector<std::array<double,12>> inverse_matrices;
  std::unordered_map<size_t,int> pose_slots;
  void freezeSurvivors() {
    if(!current) return;
    if(current.use_count()>1) {
      current->ensure();
      current.reset();
      return;
    }
    // Only an unleased batch can be recycled. Drop its snapshot references
    // before checking map uniqueness, but retain its large match allocation.
    idle_batch=std::move(current);
    idle_batch->map.reset(); idle_batch->device.reset(); idle_batch->target_groups.reset();
    idle_batch->group_count=0; idle_batch->threshold=0.; idle_batch->loaded=false;
    idle_batch->matches.clear(); idle_batch->groups.clear();
  }
  void reset(const VoxelMap<Point>& world_map,const std::vector<Point>& query,
             const std::function<gtsam::Pose3(size_t)>& estimates,double width) {
    freezeSurvivors();
    // Leased maps remain immutable, including for callbacks copied from older
    // factors. The common unleased path reuses point and query allocations.
    if(!map || map.use_count()!=1) map=std::make_shared<HostMap<Point>>();
    map->world_points.clear(); map->pose_indices.clear(); map->inverse_poses.clear();
    map->queries=query;
    voxels.clear(); points.clear(); packed.clear(); inverse_matrices.clear(); pose_slots.clear();
    voxels.reserve(world_map.size());
    size_t total=0;
    for(const auto& entry:world_map) total+=entry.second.size();
    if(total>size_t(std::numeric_limits<int>::max())) throw std::invalid_argument("CUDA matcher snapshot too large");
    points.reserve(total); map->world_points.reserve(total); map->pose_indices.reserve(total);
    for(const auto& [coords,voxel]:world_map) {
      voxels.push_back({{coords.x(),coords.y(),coords.z()},int(points.size()),int(voxel.size())});
      for(const auto& world:voxel) {
        auto it=pose_slots.find(world.scan);
        if(it==pose_slots.end()) {
          const auto inverse=estimates(world.scan).inverse();
          it=pose_slots.emplace(world.scan,int(map->inverse_poses.size())).first;
          map->inverse_poses.push_back(inverse);
          inverse_matrices.push_back(packPose(inverse));
        }
        map->world_points.push_back(world); map->pose_indices.push_back(it->second);
        CudaMatcher::MapPoint p{{world.x,world.y,world.z,world._},{world.x,world.y,world.z},{0,0,0}};
        if constexpr(std::is_same_v<Point,PlanarFeat>) {
          p.normal[0]=world.nx; p.normal[1]=world.ny; p.normal[2]=world.nz;
        }
        points.push_back(p);
      }
    }
    packed.reserve(query.size());
    for(const auto& q:query) packed.push_back({{q.x,q.y,q.z,q._}});
    device->reset(voxels,points,packed,width,inverse_matrices,map->pose_indices);
    scan_order.clear(); target_groups.reset();
  }
  template<int I> CudaMatcher::GroupedSummary match(const std::array<double,12>& pose,double threshold,
      const std::vector<size_t>& scans,const std::vector<std::tuple<PlanePoint::Ptr,PointPoint::Ptr>>& pairs) {
    if(!map) throw std::logic_error("CUDA matching must be reset before match");
    freezeSurvivors();
    if(!target_groups || scan_order!=scans) {
      scan_order=scans;
      std::unordered_map<size_t,int> slots;
      for(size_t i=0;i<scans.size();++i) slots.emplace(scans[i],int(i));
      target_groups=std::make_shared<std::vector<int>>();
      target_groups->reserve(map->world_points.size());
      for(const auto& p:map->world_points) target_groups->push_back(slots.at(p.scan));
      device->setGroups(*target_groups,scans.size());
    }
    auto summaries=device->searchGrouped(pose,threshold,I==0);
    if(idle_batch) current=std::move(idle_batch);
    else current=std::make_shared<MatchBatch<Point>>();
    current->map=map; current->device=device; current->target_groups=target_groups;
    current->group_count=scans.size(); current->threshold=threshold;
    for(size_t i=0;i<pairs.size();++i) {
      if(!summaries.counts[i]) continue;
      std::get<I>(pairs[i])->deferRaw(summaries.counts[i],[batch=current,i](const auto& rows) {batch->load(rows,i);});
    }
    return summaries;
  }
};
}
struct CudaMatching::Impl {
  bool ready=false;
  Snapshot<PlanarFeat> planes;
  Snapshot<PointFeat> points;
  std::vector<std::pair<std::weak_ptr<PlanePoint>,std::weak_ptr<PointPoint>>> pending;
};
CudaMatching::CudaMatching():impl_(std::make_unique<Impl>()) {}
CudaMatching::~CudaMatching()=default;
void CudaMatching::reset(const VoxelMap<PlanarFeat>& planes,const VoxelMap<PointFeat>& points,
    const std::vector<PlanarFeat>& pq,const std::vector<PointFeat>& qq,
    const std::function<gtsam::Pose3(size_t)>& estimates,double width) {
  impl_->ready=false;
  impl_->planes.reset(planes,pq,estimates,width);
  impl_->points.reset(points,qq,estimates,width);
  impl_->ready=true;
}
void CudaMatching::match(const gtsam::Pose3& pose,double max_distance,ConstraintMap& constraints,
    tbb::concurrent_vector<Match<PlanarFeat>>& planes,tbb::concurrent_vector<Match<PointFeat>>& points,
    bool defer_raw) {
  if(!impl_->ready) throw std::logic_error("CUDA matching requires a successful reset");
  if(!std::isfinite(max_distance) || max_distance<=0) throw std::invalid_argument("Invalid CUDA matching distance");
  std::vector<size_t> scans;
  std::vector<std::tuple<PlanePoint::Ptr,PointPoint::Ptr>> pairs;
  impl_->pending.clear();
  for(auto& [scan,pair]:constraints) {
    scans.push_back(scan); pairs.push_back(pair);
    std::get<0>(pair)->clear(); std::get<1>(pair)->clear();
    impl_->pending.emplace_back(std::get<0>(pair),std::get<1>(pair));
  }
  const auto packed=packPose(pose);
  auto pr=impl_->planes.match<0>(packed,max_distance*max_distance,scans,pairs);
  auto qr=impl_->points.match<1>(packed,max_distance*max_distance,scans,pairs);
  for(size_t i=0;i<pairs.size();++i) {
    auto& p=std::get<0>(pairs[i]); auto& q=std::get<1>(pairs[i]);
    p->summary_cache=std::make_shared<FeatureSummary>(pr.roots[i],qr.roots[i]);
    p->summary_point_owner=q; p->summary_point_revision=q->revision;
  }
  if(!defer_raw) materialize(planes,points);
}
void CudaMatching::materialize(tbb::concurrent_vector<Match<PlanarFeat>>& planes,
    tbb::concurrent_vector<Match<PointFeat>>& points) {
  for(const auto& pair:impl_->pending) {
    if(auto p=pair.first.lock()) p->ensureRaw();
    if(auto q=pair.second.lock()) q->ensureRaw();
  }
  if(impl_->planes.current) impl_->planes.current->materialize(planes);
  if(impl_->points.current) impl_->points.current->materialize(points);
}
}
