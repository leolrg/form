#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/cuda_qr.hpp"
#include <cuda_runtime.h>
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

__global__ void extractWorld(const CudaMatcher::MapPoint* points,int count,
                             size_t stride,double* world) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  for(int axis=0;axis<4;++axis) world[size_t(axis)*stride+i]=points[i].world[axis];
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
  Buffer<double> search_positions;
  size_t point_stride=0;
  Buffer<Query> queries;
  Buffer<Result> results;
  Buffer<int> indices;
  Buffer<Group> groups;
  Buffer<double> packed;
  std::vector<Result> host_results;
  int mask=0,count=0;
  double width=1.;
  bool ready=false,searched=false;
  BatchedCudaQr qr;
  Impl() { check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
  ~Impl() { if(stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
};
CudaMatcher::CudaMatcher():impl_(std::make_unique<Impl>()) {}
CudaMatcher::~CudaMatcher()=default;
void CudaMatcher::reset(const std::vector<Voxel>& voxels,const std::vector<MapPoint>& points,
                        const std::vector<Query>& queries,double width) {
  auto& s=*impl_;
  s.ready=false; s.searched=false;
  if(!std::isfinite(width) || width<=0) throw std::invalid_argument("Invalid CUDA voxel width");
  constexpr size_t limit=std::numeric_limits<int>::max();
  if(points.size()>limit || queries.size()>limit || voxels.size()>limit/4)
    throw std::invalid_argument("CUDA matcher snapshot too large");
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
  // Keep the public map records for summary packing. Nearest only reads four
  // world coordinates; plane starts are aligned for coalesced warp loads.
  s.point_stride=(points.size()+31)/32*32;
  s.search_positions.reserve(4*s.point_stride);
  if(!points.empty()) {
    extractWorld<<<1+(points.size()-1)/256,256,0,s.stream>>>(s.points.data,int(points.size()),s.point_stride,s.search_positions.data);
    check(cudaGetLastError());
  }
  s.results.reserve(queries.size()); s.host_results.resize(queries.size());
  s.mask=int(size-1); s.count=int(queries.size()); s.width=width;
  check(cudaStreamSynchronize(s.stream)); s.ready=true;
}
const std::vector<CudaMatcher::Result>& CudaMatcher::search(const std::array<double,12>& matrix) {
  auto& s=*impl_;
  s.searched=false;
  if(!s.ready) throw std::logic_error("CUDA matcher must be reset before search");
  for(auto v:matrix) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA query pose");
  if(s.count) {
    Pose pose; std::copy(matrix.begin(),matrix.end(),pose.matrix);
    // Two queries per block: A100 sweeps over 64/128/256/512 favor 64 for
    // sparse and dense maps, with 62.5% occupancy and less block tail work.
    constexpr int search_threads=64;
    nearest<<<1+(s.count-1)/(search_threads/32),search_threads,0,s.stream>>>(s.buckets.data,s.mask,s.search_positions.data,s.point_stride,s.queries.data,s.count,s.width,pose,s.results.data);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(s.host_results.data(),s.results.data,s.count*sizeof(Result),cudaMemcpyDeviceToHost,s.stream));
    check(cudaStreamSynchronize(s.stream));
  }
  for(const auto& r:s.host_results) if(r.index==-2)
    throw std::invalid_argument("CUDA transformed query exceeds voxel coordinate range");
  s.searched=true;
  return s.host_results;
}
std::vector<Eigen::MatrixXd> CudaMatcher::summarize(const std::vector<std::vector<int>>& groups,bool plane) {
  auto& s=*impl_;
  if(!s.searched) throw std::logic_error("CUDA summary requires a completed search");
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
  s.indices.upload(indices,s.stream); s.groups.upload(descriptors,s.stream); s.packed.reserve(total);
  if(max_rows) {
    pack<<<dim3(1+(max_rows-1)/128,groups.size()),128,0,s.stream>>>(s.points.data,s.queries.data,s.results.data,s.indices.data,s.groups.data,plane,s.packed.data);
    check(cudaGetLastError());
  }
  return s.qr.computeDevicePacked(s.packed.data,shapes,s.stream);
}
}
