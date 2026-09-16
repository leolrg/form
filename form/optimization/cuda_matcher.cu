#include <optional>
#include "form/optimization/diagnostics.hpp"
#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/cuda_qr.hpp"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace form {
namespace {
void check(cudaError_t code) {
  if(code!=cudaSuccess) throw std::runtime_error(std::string("CUDA matcher: ")+cudaGetErrorString(code));
}
template<class T> struct Buffer {
  T* data=nullptr; size_t capacity=0;
  ~Buffer() { if(data) cudaFree(data); }
  void reserve(size_t n) {
    if(n<=capacity) return;
    T* next=nullptr; const size_t size=std::max(n,capacity+capacity/2);
    check(cudaMalloc(reinterpret_cast<void**>(&next),size*sizeof(T)));
    if(data) check(cudaFree(data));
    data=next; capacity=size;
  }
  void upload(const std::vector<T>& values,cudaStream_t stream) {
    reserve(values.size());
    if(!values.empty()) check(cudaMemcpyAsync(data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
  }
};
struct Bucket { int x,y,z,begin,count; };
struct Pose { double matrix[12]; };
struct Group { int first,rows; size_t offset; };
__host__ __device__ uint32_t hash(int x,int y,int z) {
  return uint32_t(x)*73856093u ^ uint32_t(y)*19349669u ^ uint32_t(z)*83492791u;
}
// Each coordinate uses two bits per neighbor (shift + 1), in FORM's exact
// visit order. Immediate masks avoid serialized divergent constant-memory reads.
__device__ int neighborShift(unsigned long long coordinates,int neighbor) {
  return int((coordinates>>(2*neighbor))&3)-1;
}

__global__ void extractWorld(CudaMatcher::MapPoint* points,int count,
                             size_t stride,double* world,const Pose* inverse_poses,
                             const int* pose_indices,int* invalid) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  for(int axis=0;axis<4;++axis) world[size_t(axis)*stride+i]=points[i].world[axis];
  if(inverse_poses) {
    const Pose pose=inverse_poses[pose_indices[i]];
    const double x=points[i].world[0],y=points[i].world[1],z=points[i].world[2];
    const double nx=points[i].normal[0],ny=points[i].normal[1],nz=points[i].normal[2];
    for(int axis=0;axis<3;++axis) {
      const double* r=pose.matrix+4*axis;
      const double local=((r[0]*x+r[1]*y)+r[2]*z)+r[3];
      const double normal=(r[0]*nx+r[1]*ny)+r[2]*nz;
      points[i].local[axis]=local; points[i].normal[axis]=normal;
      if(!isfinite(local) || !isfinite(normal)) atomicExch(invalid,1);
    }
  }
}

// One warp owns one query. Dense voxel rows are scanned cooperatively; the
// lexicographic (distance, neighbor order, point order) reduction retains FORM's
// first-hit tie rule even when different lanes discover equal-distance points.
__global__ void nearest(const Bucket* buckets,int mask,const double* __restrict__ points,size_t point_stride,
                        const CudaMatcher::Query* queries,int count,double width,Pose pose,
                        CudaMatcher::Result* results) {
  const int lane=threadIdx.x%32;
  const int i=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(i>=count) return; // Whole warp exits, including the final partial block.
  const auto q=queries[i];
  double world[3],coords[3];
  for(int axis=0;axis<3;++axis) {
    const double* r=pose.matrix+4*axis;
    world[axis]=((r[0]*q.point[0]+r[1]*q.point[1])+r[2]*q.point[2])+r[3];
    coords[axis]=floor(world[axis]/width);
    if(!isfinite(coords[axis]) || coords[axis]<-2147483647. || coords[axis]>2147483646.) {
      if(lane==0) results[i]={0.,-2};
      return;
    }
  }
  const int x=int(coords[0]), y=int(coords[1]), z=int(coords[2]);
  double best=__longlong_as_double(0x7fefffffffffffffLL);
  unsigned long long rank=~0ULL;
  // Independent hash probes occupy 27 lanes instead of serializing every
  // lookup through lane zero. Divergent collision chains reconverge at ballot.
  int neighbor_begin=-1,neighbor_length=0;
  if(lane<27) {
    const int a=x+neighborShift(0x2a9542829549ULL,lane);
    const int b=y+neighborShift(0x2828295489495ULL,lane);
    const int c=z+neighborShift(0x8888888954955ULL,lane);
    int slot=hash(a,b,c)&mask;
    Bucket voxel=buckets[slot];
    while(voxel.begin!=-1 && (voxel.x!=a || voxel.y!=b || voxel.z!=c)) {
      slot=(slot+1)&mask; voxel=buckets[slot];
    }
    neighbor_begin=voxel.begin; neighbor_length=voxel.count;
  }
  unsigned occupied=__ballot_sync(0xffffffff,neighbor_length>0);
  while(occupied) {
    const int n=__ffs(occupied)-1;
    occupied&=occupied-1;
    const int begin=__shfl_sync(0xffffffff,neighbor_begin,n);
    const int length=__shfl_sync(0xffffffff,neighbor_length,n);
    for(size_t offset=lane;offset<size_t(length);offset+=32) {
      const int j=begin+int(offset);
      const double dx=points[j]-world[0],dy=points[point_stride+j]-world[1];
      const double dz=points[2*point_stride+j]-world[2],dw=points[3*point_stride+j]-q.point[3];
      const double distance=(dx*dx+dz*dz)+(dy*dy+dw*dw);
      if(distance<best) { best=distance; rank=(static_cast<unsigned long long>(n)<<32)|unsigned(j); }
    }
  }
  for(int shift=16;shift;shift/=2) {
    const double other=__shfl_down_sync(0xffffffff,best,shift);
    const auto other_rank=__shfl_down_sync(0xffffffff,rank,shift);
    if(other<best || (other==best && other_rank<rank)) { best=other; rank=other_rank; }
  }
  if(lane==0) results[i]={best,rank==~0ULL?-1:int(unsigned(rank))};
}

__global__ void classify(const CudaMatcher::Result* results,int count,
                         const int* target_groups,int group_count,double threshold,
                         int* keys,int* sequence,int* counts) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  const auto result=results[i];
  int group=group_count; // Rejected queries sort after every real group.
  if(result.index==-2) atomicExch(counts+group_count,1);
  if(result.index>=0 && result.distance<threshold) {
    const int target=target_groups[result.index];
    if(target>=0) { group=target; atomicAdd(counts+group,1); }
  }
  keys[i]=group;
  sequence[i]=int(i);
}

__global__ void pack(const CudaMatcher::MapPoint* points,const CudaMatcher::Query* queries,
                     const CudaMatcher::Result* results,const int* indices,const Group* groups,
                     bool plane,double* output) {
  const Group group=groups[blockIdx.y];
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=group.rows) return;
  const int q=indices[group.first+row];
  const auto p=points[results[q].index];
  const auto query=queries[q];
  double v[7];
  if(plane) {
    for(int k=0;k<3;++k) { v[k]=query.point[k]; v[3+k]=p.normal[k]; }
    // Eigen Vector3 dot: x + (y + z).
    v[6]=p.normal[0]*(query.point[0]-p.local[0])+
      (p.normal[1]*(query.point[1]-p.local[1])+p.normal[2]*(query.point[2]-p.local[2]));
  } else {
    v[0]=1.;
    for(int k=0;k<3;++k) { v[1+k]=p.local[k]; v[4+k]=query.point[k]-p.local[k]; }
  }
  for(int k=0;k<7;++k) output[group.offset+row+size_t(k)*group.rows]=v[k];
}
}
struct CudaMatcher::Impl {
  cudaStream_t stream=nullptr;
  Buffer<Bucket> buckets;
  Buffer<MapPoint> points;
  Buffer<Pose> inverse_poses;
  Buffer<int> pose_indices,snapshot_error;
  Buffer<double> search_positions;
  size_t point_stride=0;
  Buffer<Query> queries;
  Buffer<Result> results;
  Buffer<int> indices;
  Buffer<Group> groups;
  Buffer<double> packed;
  Buffer<int> target_groups,group_counts,sort_keys,sorted_keys,sequence;
  Buffer<unsigned char> sort_storage;
  std::vector<int> host_counts;
  int group_count=0,point_count=0,sort_bits=1;
  size_t sort_bytes=0;
  bool groups_ready=false,host_results_ready=false;
  std::vector<Result> host_results;
  int mask=0,count=0;
  double width=1.;
  bool ready=false,searched=false;
  BatchedCudaQr qr;
  void launchSearch(const std::array<double,12>& matrix);
  std::vector<Eigen::MatrixXd> packSummaries(const std::vector<Group>& descriptors,
      const std::vector<BatchedCudaQr::DeviceInput>& shapes,size_t total,int max_rows,bool plane);
  Impl() { check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
  ~Impl() { if(stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
};
CudaMatcher::CudaMatcher():impl_(std::make_unique<Impl>()) {}
CudaMatcher::~CudaMatcher()=default;
void CudaMatcher::reset(const std::vector<Voxel>& voxels,const std::vector<MapPoint>& points,
                        const std::vector<Query>& queries,double width,
                        const std::vector<std::array<double,12>>& inverse_poses,
                        const std::vector<int>& point_pose_indices) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::matcher_reset);
  auto& s=*impl_;
  s.ready=false; s.searched=false; s.groups_ready=false; s.host_results_ready=false;
  if(!std::isfinite(width) || width<=0) throw std::invalid_argument("Invalid CUDA voxel width");
  constexpr size_t limit=std::numeric_limits<int>::max();
  if(points.size()>limit || queries.size()>limit || voxels.size()>limit/4)
    throw std::invalid_argument("CUDA matcher snapshot too large");
  if(inverse_poses.size()>limit ||
     (inverse_poses.empty()?!point_pose_indices.empty():point_pose_indices.size()!=points.size()))
    throw std::invalid_argument("Invalid CUDA inverse pose shape");
  std::vector<Pose> poses(inverse_poses.size());
  for(size_t p=0;p<inverse_poses.size();++p) {
    for(double v:inverse_poses[p]) if(!std::isfinite(v))
      throw std::invalid_argument("Nonfinite CUDA inverse pose");
    std::copy(inverse_poses[p].begin(),inverse_poses[p].end(),poses[p].matrix);
  }
  for(int index:point_pose_indices) if(index<0 || size_t(index)>=inverse_poses.size())
    throw std::invalid_argument("Invalid CUDA inverse pose index");
  for(const auto& p:points) {
    for(double v:p.world) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA map point");
    for(double v:p.local) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA local point");
    for(double v:p.normal) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA map normal");
  }
  for(const auto& q:queries) for(double v:q.point)
    if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA query");
  size_t size=1; while(size<std::max<size_t>(1,voxels.size()*2)) size*=2;
  std::vector<Bucket> buckets(size,Bucket{0,0,0,-1,0});
  for(const auto& v:voxels) {
    if(v.begin<0 || v.count<0 || size_t(v.begin)+v.count>points.size())
      throw std::invalid_argument("Invalid CUDA voxel point range");
    size_t index=hash(v.coords[0],v.coords[1],v.coords[2])&(size-1);
    while(buckets[index].begin!=-1) {
      const auto& b=buckets[index];
      if(b.x==v.coords[0] && b.y==v.coords[1] && b.z==v.coords[2])
        throw std::invalid_argument("Duplicate CUDA voxel");
      index=(index+1)&(size-1);
    }
    buckets[index]={v.coords[0],v.coords[1],v.coords[2],v.begin,v.count};
  }
  check(cudaStreamSynchronize(s.stream));
  s.buckets.upload(buckets,s.stream); s.points.upload(points,s.stream); s.queries.upload(queries,s.stream);
  int invalid_snapshot=0;
  if(!poses.empty()) {
    s.inverse_poses.upload(poses,s.stream); s.pose_indices.upload(point_pose_indices,s.stream);
    s.snapshot_error.reserve(1); check(cudaMemsetAsync(s.snapshot_error.data,0,sizeof(int),s.stream));
  }
  // Keep the public map records for summary packing. Nearest only reads four
  // world coordinates; plane starts are aligned for coalesced warp loads.
  s.point_stride=(points.size()+31)/32*32;
  s.search_positions.reserve(4*s.point_stride);
  if(!points.empty()) {
    extractWorld<<<1+(points.size()-1)/256,256,0,s.stream>>>(s.points.data,int(points.size()),s.point_stride,s.search_positions.data,
      poses.empty()?nullptr:s.inverse_poses.data,s.pose_indices.data,s.snapshot_error.data);
    check(cudaGetLastError());
  }
  s.results.reserve(queries.size()); s.host_results.resize(queries.size());
  s.mask=int(size-1); s.count=int(queries.size()); s.point_count=int(points.size()); s.width=width;
  if(!poses.empty()) check(cudaMemcpyAsync(&invalid_snapshot,s.snapshot_error.data,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  if(invalid_snapshot) throw std::invalid_argument("Nonfinite CUDA transformed map feature");
  s.ready=true;
}
void CudaMatcher::Impl::launchSearch(const std::array<double,12>& matrix) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_search_launch);
  auto& s=*this;
  s.searched=false; s.host_results_ready=false;
  if(!s.ready) throw std::logic_error("CUDA matcher must be reset before search");
  for(auto v:matrix) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA query pose");
  if(s.count) {
    Pose pose; std::copy(matrix.begin(),matrix.end(),pose.matrix);
    // Two queries per block: A100 sweeps over 64/128/256/512 favor 64 for
    // sparse and dense maps, with 62.5% occupancy and less block tail work.
    constexpr int search_threads=64;
    nearest<<<1+(s.count-1)/(search_threads/32),search_threads,0,s.stream>>>(s.buckets.data,s.mask,s.search_positions.data,s.point_stride,s.queries.data,s.count,s.width,pose,s.results.data);
    check(cudaGetLastError());
  }
}
const std::vector<CudaMatcher::Result>& CudaMatcher::search(const std::array<double,12>& matrix) {
  impl_->launchSearch(matrix);
  impl_->searched=true;
  return downloadResults();
}
const std::vector<CudaMatcher::Result>& CudaMatcher::downloadResults() {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_download);
  auto& s=*impl_;
  if(!s.searched) throw std::logic_error("CUDA results require a completed search");
  if(!s.host_results_ready) {
    if(s.count) check(cudaMemcpyAsync(s.host_results.data(),s.results.data,size_t(s.count)*sizeof(Result),cudaMemcpyDeviceToHost,s.stream));
    check(cudaStreamSynchronize(s.stream));
    for(const auto& r:s.host_results) if(r.index==-2) {
      s.searched=false;
      throw std::invalid_argument("CUDA transformed query exceeds voxel coordinate range");
    }
    s.host_results_ready=true;
  }
  return s.host_results;
}
void CudaMatcher::setGroups(const std::vector<int>& target_groups,size_t group_count) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_group_setup);
  auto& s=*impl_;
  s.groups_ready=false;
  if(!s.ready) throw std::logic_error("CUDA matcher must be reset before grouping");
  if(target_groups.size()!=size_t(s.point_count) || group_count>65535)
    throw std::invalid_argument("Invalid CUDA target group shape");
  for(int group:target_groups) if(group < -1 || (group>=0 && size_t(group)>=group_count))
    throw std::invalid_argument("Invalid CUDA target group index");
  s.target_groups.upload(target_groups,s.stream);
  s.group_count=int(group_count);
  s.group_counts.reserve(group_count+1); s.host_counts.resize(group_count+1);
  s.sort_keys.reserve(s.count); s.sorted_keys.reserve(s.count);
  s.sequence.reserve(s.count); s.indices.reserve(s.count);
  s.sort_bits=1;
  while((1u<<s.sort_bits)<=group_count) ++s.sort_bits;
  if(s.count) {
    check(cub::DeviceRadixSort::SortPairs(nullptr,s.sort_bytes,s.sort_keys.data,s.sorted_keys.data,
          s.sequence.data,s.indices.data,s.count,0,s.sort_bits,s.stream));
    s.sort_storage.reserve(s.sort_bytes);
  }
  check(cudaStreamSynchronize(s.stream));
  s.groups_ready=true;
}
CudaMatcher::GroupedSummary CudaMatcher::searchGrouped(const std::array<double,12>& matrix,
                                                       double threshold_squared,bool plane) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_search_grouped);
  auto& s=*impl_;
  s.searched=false; s.host_results_ready=false;
  if(!s.groups_ready) throw std::logic_error("CUDA target groups must be configured before grouped search");
  if(!std::isfinite(threshold_squared) || threshold_squared<=0)
    throw std::invalid_argument("Invalid CUDA squared match threshold");
  std::optional<diagnostics::Scope> phase; phase.emplace(diagnostics::Stage::match_classify_wait);
  s.launchSearch(matrix);
  check(cudaMemsetAsync(s.group_counts.data,0,s.host_counts.size()*sizeof(int),s.stream));
  if(s.count) {
    classify<<<1+(s.count-1)/256,256,0,s.stream>>>(s.results.data,s.count,s.target_groups.data,s.group_count,
      threshold_squared,s.sort_keys.data,s.sequence.data,s.group_counts.data);
    check(cudaGetLastError());
  }
  check(cudaMemcpyAsync(s.host_counts.data(),s.group_counts.data,s.host_counts.size()*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  phase.emplace(diagnostics::Stage::match_group_sort);
  if(s.host_counts.back()) throw std::invalid_argument("CUDA transformed query exceeds voxel coordinate range");
  s.searched=true;
  GroupedSummary result;
  std::vector<Group> descriptors;
  std::vector<BatchedCudaQr::DeviceInput> shapes;
  int first=0,max_rows=0;
  size_t total=0;
  for(int group=0;group<s.group_count;++group) {
    const int rows=s.host_counts[group];
    descriptors.push_back({first,rows,total}); shapes.push_back({size_t(rows),plane});
    result.counts.push_back(size_t(rows));
    first+=rows; total+=size_t(rows)*7; max_rows=std::max(max_rows,rows);
  }
  if(first) {
    // CUB's stable radix sort retains ascending input query order within each
    // group, independent of scheduling of the histogram's integer atomics.
    check(cub::DeviceRadixSort::SortPairs(s.sort_storage.data,s.sort_bytes,s.sort_keys.data,s.sorted_keys.data,
          s.sequence.data,s.indices.data,s.count,0,s.sort_bits,s.stream));
  }
  phase.reset();
  result.roots=s.packSummaries(descriptors,shapes,total,max_rows,plane);
  return result;
}
std::vector<Eigen::MatrixXd> CudaMatcher::Impl::packSummaries(const std::vector<Group>& descriptors,
    const std::vector<BatchedCudaQr::DeviceInput>& shapes,size_t total,int max_rows,bool plane) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_pack);
  auto& s=*this;
  s.groups.upload(descriptors,s.stream); s.packed.reserve(total);
  if(max_rows) {
    pack<<<dim3(1+(max_rows-1)/128,descriptors.size()),128,0,s.stream>>>(s.points.data,s.queries.data,s.results.data,s.indices.data,s.groups.data,plane,s.packed.data);
    check(cudaGetLastError());
  }
  return s.qr.computeDevicePacked(s.packed.data,shapes,s.stream);
}
std::vector<Eigen::MatrixXd> CudaMatcher::summarize(const std::vector<std::vector<int>>& groups,bool plane) {
  auto& s=*impl_;
  if(!s.searched) throw std::logic_error("CUDA summary requires a completed search");
  downloadResults();
  if(groups.size()>65535) throw std::invalid_argument("Too many CUDA match groups");
  std::vector<int> indices;
  std::vector<Group> descriptors;
  std::vector<BatchedCudaQr::DeviceInput> shapes;
  size_t total=0; int max_rows=0;
  for(const auto& g:groups) {
    if(g.size()+indices.size()>size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Too many CUDA summary rows");
    for(int q:g) if(q<0 || q>=s.count || s.host_results[q].index<0)
      throw std::invalid_argument("Invalid CUDA summary query index");
    descriptors.push_back({int(indices.size()),int(g.size()),total});
    shapes.push_back({g.size(),plane});
    indices.insert(indices.end(),g.begin(),g.end()); total+=g.size()*7;
    max_rows=std::max(max_rows,int(g.size()));
  }
  s.indices.upload(indices,s.stream);
  return s.packSummaries(descriptors,shapes,total,max_rows,plane);
}
}
