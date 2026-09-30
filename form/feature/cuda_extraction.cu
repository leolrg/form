#include "form/optimization/diagnostics.hpp"
#include "form/feature/cuda_extraction.hpp"
#include <cuda_runtime.h>
#include "form/feature/cuda_normal_eigen.cuh"
#include "form/feature/cuda_legacy_sort.cuh"
#include "form/optimization/profile.hpp"
#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace form {
namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string("CUDA extraction: ") + cudaGetErrorString(code));
}
template<class T> struct Buffer {
  T* data = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (data) cudaFree(data); }
  void reserve(size_t n) {
    if (n <= capacity) return;
    T* next = nullptr;
    const auto size = std::max(n, capacity + capacity/2);
    check(cudaMalloc(reinterpret_cast<void**>(&next), size*sizeof(T)));
    if (data) check(cudaFree(data));
    data = next; capacity = size;
  }
};
template<class T> struct Point { T x,y,z,w; };

template<class T>
__device__ T distanceSquared(const Point<T>& p, const Point<T>& q,
                            CudaExtraction::Reduction reduction) {
  const T x=p.x-q.x, y=p.y-q.y, z=p.z-q.z, w=p.w-q.w;
  if (reduction == CudaExtraction::Reduction::Cross) return (x*x+z*z)+(y*y+w*w);
  if (reduction == CudaExtraction::Reduction::Adjacent) return (x*x+y*y)+(z*z+w*w);
  return ((x*x+y*y)+z*z)+w*w;
}

template<class T>
__global__ void curvature(const Point<T>* scan, const unsigned char* valid,
                         int count, int columns, int neighbors, double* out) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (!valid[i] || i%columns < neighbors || i%columns >= columns-neighbors) {
    out[i] = (sizeof(T)==sizeof(float) ? double(FLT_MAX) : DBL_MAX); return;
  }
  double dx = -(2.0*neighbors)*scan[i].x;
  double dy = -(2.0*neighbors)*scan[i].y;
  double dz = -(2.0*neighbors)*scan[i].z;
  for (int n=1; n<=neighbors; ++n) {
    dx = dx + scan[i-n].x + scan[i+n].x;
    dy = dy + scan[i-n].y + scan[i+n].y;
    dz = dz + scan[i-n].z + scan[i+n].z;
  }
  out[i] = double(T(dx*dx + dy*dy + dz*dz));
}

template<class T>
__global__ void nearestRowsKernel(const Point<T>* scan, const unsigned char* valid,
    const int* queries, int count, int columns, int rows, CudaExtraction::Reduction reduction, int* out) {
  const int lane = threadIdx.x%32;
  const int job = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
  if (job >= count*2) return;
  const int query = queries[job/2];
  const int row = query/columns + (job%2 == 0 ? -1 : 1);
  if (row < 0 || row >= rows) { if (lane == 0) out[job] = -1; return; }
  const auto q = scan[query];
  double best = DBL_MAX;
  int winner = INT_MAX;
  for (int c=lane; c<columns; c+=32) {
    const int i = row*columns+c;
    if (!valid[i]) continue;
    const auto p = scan[i];
    // Match the calling CPU's Eigen reduction, including scalar builds.
    const T d=distanceSquared(p,q,reduction);
    if (double(d) < best) { best=double(d); winner=i; }
  }
  for (int offset=16; offset; offset/=2) {
    const double other = __shfl_down_sync(0xffffffff, best, offset);
    const int index = __shfl_down_sync(0xffffffff, winner, offset);
    if (other < best || (other == best && index < winner)) {
      best=other; winner=index;
    }
  }
  if (lane == 0) out[job] = winner == INT_MAX ? -1 : winner;
}

// Two short traversals avoid a dynamically allocated per-candidate neighborhood.
// Divide each displacement before its outer product, as the CPU does with A/m.
template<class T, bool Accumulate> struct Neighborhood {
  Point<T> query;
  T divisor;
  int count=0;
  Eigen::Matrix<T,3,3> covariance=Eigen::Matrix<T,3,3>::Zero();
  __device__ void add(const Point<T>& p) {
    ++count;
    if constexpr (Accumulate) {
      const T d[3]={(p.x-query.x)/divisor,(p.y-query.y)/divisor,(p.z-query.z)/divisor};
      for(int col=0;col<3;++col) for(int row=col;row<3;++row)
        covariance(row,col)+=d[row]*d[col];
    }
  }
};

template<class T, class Output>
__device__ void gatherNeighbors(const Point<T>* scan, int center, int columns,
    int neighbors, double radius2, CudaExtraction::Reduction reduction,
    Output& output) {
  const int column=center%columns;
  for(int sign=1;sign>=-1;sign-=2) for(int n=1;n<=neighbors;++n) {
    const int c=column+sign*n;
    if(c<0 || c>=columns) break;
    const auto p=scan[center+sign*n];
    if(!(double(distanceSquared(p,scan[center],reduction))<radius2)) break;
    output.add(p);
  }
}

template<class T, class Output>
__device__ void gatherNormal(const Point<T>* scan, int query, const int* nearest,
    int columns, int neighbors, double radius2, CudaExtraction::Reduction reduction,
    Output& output) {
  gatherNeighbors(scan,query,columns,neighbors,radius2,reduction,output);
  for(int side=0;side<2;++side) if(nearest[side]>=0) {
    output.add(scan[nearest[side]]);
    gatherNeighbors(scan,nearest[side],columns,neighbors,radius2,reduction,output);
  }
}

// Eigen 3.4 uses coefficient dot products for A.transpose()*A when
// A.rows()+3+3 is below this threshold. Their SIMD reduction differs from GEMM.
constexpr int coefficient_threshold=EIGEN_GEMM_TO_COEFFBASED_THRESHOLD-6;
template<class T> struct SmallNeighborhood {
  Point<T> query;
  T divisor;
  int count=0;
  T displacement[coefficient_threshold-1][3];
  __device__ void add(const Point<T>& p) {
    displacement[count][0]=(p.x-query.x)/divisor;
    displacement[count][1]=(p.y-query.y)/divisor;
    displacement[count][2]=(p.z-query.z)/divisor;
    ++count;
  }
};

// Keep the rare short-neighborhood workspace off the common GEMM-order path.
template<class T>
__device__ __noinline__ Eigen::Matrix<T,3,3> smallCovariance(
    const Point<T>* scan, int query, const int* adjacent, int columns,
    int neighbors, double radius2, CudaExtraction::Reduction reduction,
    int count, int packet_size) {
  SmallNeighborhood<T> gathered{scan[query],T(count)};
  gatherNormal(scan,query,adjacent,columns,neighbors,radius2,reduction,gathered);
  Eigen::Matrix<T,3,3> covariance=Eigen::Matrix<T,3,3>::Zero();
  for(int col=0;col<3;++col) for(int row=col;row<3;++row) {
    T products[coefficient_threshold-1];
    for(int i=0;i<count;++i)
      products[i]=gathered.displacement[i][row]*gathered.displacement[i][col];
    T sum;
    const int end=count/packet_size*packet_size;
    if(packet_size>1 && end) {
      T lanes[16];
      for(int lane=0;lane<packet_size;++lane) {
        T first=products[lane];
        if(end>packet_size) {
          T second=products[packet_size+lane];
          const int paired_end=count/(2*packet_size)*(2*packet_size);
          for(int i=2*packet_size;i<paired_end;i+=2*packet_size) {
            first+=products[i+lane]; second+=products[i+packet_size+lane];
          }
          first+=second;
          if(end>paired_end) first+=products[paired_end+lane];
        }
        lanes[lane]=first;
      }
      // Eigen's x86 predux normally folds halves recursively. AVX512
      // double8 instead folds to four lanes, then adds adjacent pairs.
      if(sizeof(T)==sizeof(double) && packet_size==8) {
        for(int lane=0;lane<4;++lane) lanes[lane]+=lanes[lane+4];
        sum=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
      } else {
        for(int half=packet_size/2;half;half/=2)
          for(int lane=0;lane<half;++lane) lanes[lane]+=lanes[lane+half];
        sum=lanes[0];
      }
      for(int i=end;i<count;++i) sum+=products[i];
    } else {
      sum=products[0];
      for(int i=1;i<count;++i) sum+=products[i];
    }
    covariance(row,col)=sum;
  }
  return covariance;
}

template<class T>
__global__ void normalsKernel(const Point<T>* scan, const int* queries,
    const int* nearest, int count, int columns, int neighbors, double radius2,
    size_t min_points, CudaExtraction::Reduction reduction, int covariance_packet_size, Point<double>* out) {
  const int job=blockIdx.x*blockDim.x+threadIdx.x;
  if(job>=count) return;
  out[job]={0,0,0,0};
  const int* adjacent=nearest+2*job;
  if(adjacent[0]<0 && adjacent[1]<0) return;
  const int query=queries[job];
  Neighborhood<T,false> counter{scan[query],T(1)};
  gatherNormal(scan,query,adjacent,columns,neighbors,radius2,reduction,counter);
  if(size_t(counter.count)<min_points) return;
  Eigen::Matrix<T,3,3> covariance;
  if(counter.count>0 && counter.count<coefficient_threshold) {
    covariance=smallCovariance(scan,query,adjacent,columns,neighbors,radius2,
                               reduction,counter.count,covariance_packet_size);
  } else {
    Neighborhood<T,true> gathered{scan[query],T(counter.count)};
    gatherNormal(scan,query,adjacent,columns,neighbors,radius2,reduction,gathered);
    covariance=gathered.covariance;
  }
  Eigen::Matrix<T,3,1> normal;
  if(!cuda_detail::normalEigenvector(covariance,normal,
        sizeof(T)==sizeof(double) && covariance_packet_size>1)) {
    out[job].w=-1; return;
  }
  out[job]={double(normal.x()),double(normal.y()),double(normal.z()),1};
}
#include "form/feature/cuda_selection.cuh"
}

struct CudaExtraction::Impl {
  cudaStream_t stream = nullptr;
  Buffer<unsigned char> scan, valid;
  Buffer<double> curvatures;
  Buffer<int> queries, nearest;
  std::vector<int> host_queries;
  Buffer<Point<double>> normals;
  Buffer<unsigned char> point_mask,used;
  Buffer<int> sorted,scratch,selected_planes,selected_points,row_counts,totals,point_queries;
  int columns=0, rows=0, count=0, neighbor_points=0;
  bool single=false, ready=false;
  int covariance_packet_size=1;
  void setCovariancePacketSize(int size) {
    if(size<1 || size>16 || (size & (size-1)))
      throw std::invalid_argument("Invalid CUDA covariance packet size");
    covariance_packet_size=size;
  }
  CudaExtraction::Reduction reduction=CudaExtraction::Reduction::Cross;
  Impl() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Impl() { if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
  template<class T> std::vector<double> prepare(
      const std::vector<std::array<T,4>>& input,
      const std::vector<unsigned char>& mask, int cols, int neighbors, CudaExtraction::Reduction policy, int packet_size) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::extraction_prepare);
    ready=false;
    setCovariancePacketSize(packet_size);
    if (cols <= 0 || neighbors < 0 || neighbors > cols/2 ||
        input.empty() || input.size() > size_t(INT_MAX)/2 ||
        input.size()%cols || mask.size()!=input.size())
      throw std::invalid_argument("Invalid CUDA extraction shape");
    columns=cols; rows=int(input.size()/cols); count=int(input.size()); neighbor_points=neighbors;
    single=sizeof(T)==sizeof(float); reduction=policy;
    scan.reserve(input.size()*sizeof(Point<T>)); valid.reserve(mask.size()); curvatures.reserve(input.size());
    check(cudaMemcpyAsync(scan.data,input.data(),input.size()*sizeof(Point<T>),cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(valid.data,mask.data(),mask.size(),cudaMemcpyHostToDevice,stream));
    curvature<<<(count+255)/256,256,0,stream>>>(reinterpret_cast<const Point<T>*>(scan.data),valid.data,count,columns,neighbors,curvatures.data);
    check(cudaGetLastError());
    std::vector<double> output(input.size());
    check(cudaMemcpyAsync(output.data(),curvatures.data,output.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    ready=true;
    return output;
  }
  void search(const std::vector<size_t>& indices) {
    if (!ready) throw std::logic_error("Prepare CUDA extraction before normal search");
    if (indices.size()>size_t(INT_MAX)/2) throw std::invalid_argument("Too many CUDA normal queries");
    host_queries.clear(); host_queries.reserve(indices.size());
    for (auto i:indices) {
      if (i>=size_t(count)) throw std::out_of_range("CUDA normal query outside scan");
      host_queries.push_back(int(i));
    }
    if(indices.empty()) return;
    queries.reserve(indices.size()); nearest.reserve(2*indices.size());
    // Retain the upload source until the caller synchronizes this stream.
    check(cudaMemcpyAsync(queries.data,host_queries.data(),host_queries.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
    searchDevice(int(indices.size()));
  }
  void searchDevice(int size) {
    if(!size)return;
    nearest.reserve(size_t(size)*2);
    const int blocks=int((size_t(size)*2+3)/4);
    if(single)
      nearestRowsKernel<<<blocks,128,0,stream>>>(reinterpret_cast<const Point<float>*>(scan.data),valid.data,queries.data,size,columns,rows,reduction,nearest.data);
    else
      nearestRowsKernel<<<blocks,128,0,stream>>>(reinterpret_cast<const Point<double>*>(scan.data),valid.data,queries.data,size,columns,rows,reduction,nearest.data);
    check(cudaGetLastError());
  }
  template<class T> Features extract(const std::vector<std::array<T,4>>& input,
      const Selection& p,Reduction policy, int packet_size) {
    ready=false;
    setCovariancePacketSize(packet_size);
    if(p.columns<=0 || p.neighbors<0 || p.neighbors>p.columns/2 ||
       p.sectors<=0 || p.sectors>p.columns || p.spacing<0 || p.spacing>p.neighbors ||
       input.empty() || input.size()>size_t(INT_MAX)/2 || input.size()%p.columns ||
       !std::isfinite(p.min_squared) || !std::isfinite(p.max_squared) || p.min_squared<0 || p.min_squared>p.max_squared ||
       !std::isfinite(p.threshold) || !std::isfinite(p.radius) || p.radius<0 || !std::isfinite(p.radius*p.radius))
      throw std::invalid_argument("Invalid resident CUDA extraction parameters");
    int sort_threads=1;
    while(sort_threads<p.columns/p.sectors+p.columns%p.sectors)sort_threads*=2;
    if(sort_threads>1024)throw std::invalid_argument("Resident CUDA selection supports sectors of at most 1024 points");
    columns=p.columns;rows=int(input.size()/columns);count=int(input.size());neighbor_points=p.neighbors;
    single=sizeof(T)==sizeof(float);reduction=policy;
    auto timer=profile::enabled?profile::Clock::now():profile::Clock::time_point{};
    scan.reserve(input.size()*sizeof(Point<T>));valid.reserve(count);point_mask.reserve(count);used.reserve(count);curvatures.reserve(count);
    sorted.reserve(count);scratch.reserve(count);selected_planes.reserve(count);selected_points.reserve(count);
    queries.reserve(count);point_queries.reserve(count);row_counts.reserve(2*rows);totals.reserve(3);
    check(cudaMemcpyAsync(scan.data,input.data(),input.size()*sizeof(Point<T>),cudaMemcpyHostToDevice,stream));
    check(cudaMemsetAsync(totals.data,0,3*sizeof(int),stream));
    const auto* points=reinterpret_cast<const Point<T>*>(scan.data);
    selectionRawMask<<<(count+255)/256,256,0,stream>>>(points,point_mask.data,count,p,reduction);
    selectionPlanarMask<<<(count+255)/256,256,0,stream>>>(point_mask.data,valid.data,count,columns,neighbor_points);
    curvature<<<(count+255)/256,256,0,stream>>>(points,valid.data,count,columns,neighbor_points,curvatures.data);
    selectionSort<<<rows*p.sectors,sort_threads,0,stream>>>(curvatures.data,sorted.data,totals.data+2,columns,p.sectors,p.stable_order);
    selectionRows<<<rows,32,0,stream>>>(curvatures.data,sorted.data,valid.data,point_mask.data,used.data,scratch.data,selected_planes.data,selected_points.data,row_counts.data,totals.data+2,p);
    selectionCompact<<<rows,128,0,stream>>>(selected_planes.data,selected_points.data,row_counts.data,queries.data,point_queries.data,totals.data,rows,columns);
    check(cudaGetLastError());
    std::array<int,3> sizes;
    check(cudaMemcpyAsync(sizes.data(),totals.data,3*sizeof(int),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    if(sizes[2])throw std::invalid_argument("Nonfinite resident CUDA curvature");
    profile::checkpoint(profile::extract_planar_select,timer); // Includes masks/curvature/sorting/compaction.
    Features result;result.planes.resize(sizes[0]);result.points.resize(sizes[1]);result.normals.resize(sizes[0]);
    searchDevice(sizes[0]);
    if(sizes[0]) {
      normals.reserve(sizes[0]);
      normalsKernel<<<(sizes[0]+127)/128,128,0,stream>>>(points,queries.data,nearest.data,sizes[0],columns,neighbor_points,p.radius*p.radius,p.min_points,reduction,covariance_packet_size,normals.data);
      check(cudaGetLastError());
      check(cudaMemcpyAsync(result.normals.data(),normals.data,result.normals.size()*sizeof(Point<double>),cudaMemcpyDeviceToHost,stream));
      check(cudaMemcpyAsync(result.planes.data(),queries.data,result.planes.size()*sizeof(int),cudaMemcpyDeviceToHost,stream));
    }
    if(sizes[1])check(cudaMemcpyAsync(result.points.data(),point_queries.data,result.points.size()*sizeof(int),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    for(const auto& n:result.normals)if(n[3]<0)throw std::runtime_error("CUDA normal eigensolver failed");
    profile::checkpoint(profile::extract_normals,timer);
    ready=true;return result;
  }
};
CudaExtraction::CudaExtraction(): impl_(std::make_unique<Impl>()) {}
CudaExtraction::~CudaExtraction() = default;
CudaExtraction::Features CudaExtraction::extract(const std::vector<std::array<float,4>>& scan,const Selection& p,Reduction r,int packet_size){return impl_->extract(scan,p,r,packet_size);}
CudaExtraction::Features CudaExtraction::extract(const std::vector<std::array<double,4>>& scan,const Selection& p,Reduction r,int packet_size){return impl_->extract(scan,p,r,packet_size);}
std::vector<std::array<double,4>> CudaExtraction::normals(
    const std::vector<size_t>& indices, double radius, size_t min_points) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::extraction_search);
  if(!std::isfinite(radius) || radius<0 || !std::isfinite(radius*radius))
    throw std::invalid_argument("Invalid CUDA normal radius");
  auto& s=*impl_;
  s.search(indices);
  std::vector<std::array<double,4>> output(indices.size());
  if(indices.empty()) return output;
  s.normals.reserve(indices.size());
  const int blocks=int((indices.size()+127)/128);
  if(s.single)
    normalsKernel<<<blocks,128,0,s.stream>>>(reinterpret_cast<const Point<float>*>(s.scan.data),s.queries.data,s.nearest.data,int(indices.size()),s.columns,s.neighbor_points,radius*radius,min_points,s.reduction,s.covariance_packet_size,s.normals.data);
  else
    normalsKernel<<<blocks,128,0,s.stream>>>(reinterpret_cast<const Point<double>*>(s.scan.data),s.queries.data,s.nearest.data,int(indices.size()),s.columns,s.neighbor_points,radius*radius,min_points,s.reduction,s.covariance_packet_size,s.normals.data);
  check(cudaGetLastError());
  static_assert(sizeof(Point<double>)==sizeof(std::array<double,4>));
  check(cudaMemcpyAsync(output.data(),s.normals.data,output.size()*sizeof(Point<double>),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  for(const auto& n:output) if(n[3]<0) throw std::runtime_error("CUDA normal eigensolver failed");
  return output;
}
std::vector<double> CudaExtraction::prepare(const std::vector<std::array<float,4>>& scan,
    const std::vector<unsigned char>& valid,int columns,int neighbors, Reduction reduction, int packet_size) {
  return impl_->prepare(scan,valid,columns,neighbors,reduction,packet_size);
}
std::vector<double> CudaExtraction::prepare(const std::vector<std::array<double,4>>& scan,
    const std::vector<unsigned char>& valid,int columns,int neighbors, Reduction reduction, int packet_size) {
  return impl_->prepare(scan,valid,columns,neighbors,reduction,packet_size);
}
std::vector<std::array<int,2>> CudaExtraction::nearestRows(const std::vector<size_t>& indices) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::extraction_search);
  auto& s=*impl_;
  s.search(indices);
  std::vector<std::array<int,2>> output(indices.size());
  if (indices.empty()) return output;
  check(cudaMemcpyAsync(output.data(),s.nearest.data,output.size()*2*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  return output;
}
}
