#include "form/feature/cuda_extraction.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cfloat>
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
    const T x=p.x-q.x, y=p.y-q.y, z=p.z-q.z, w=p.w-q.w;
    // Match the calling CPU's Eigen reduction, including scalar builds.
    T d;
    if (reduction == CudaExtraction::Reduction::Cross) d=(x*x+z*z)+(y*y+w*w);
    else if (reduction == CudaExtraction::Reduction::Adjacent) d=(x*x+y*y)+(z*z+w*w);
    else d=((x*x+y*y)+z*z)+w*w;
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
}

struct CudaExtraction::Impl {
  cudaStream_t stream = nullptr;
  Buffer<unsigned char> scan, valid;
  Buffer<double> curvatures;
  Buffer<int> queries, nearest;
  int columns=0, rows=0, count=0;
  bool single=false, ready=false;
  CudaExtraction::Reduction reduction=CudaExtraction::Reduction::Cross;
  Impl() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Impl() { if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
  template<class T> std::vector<double> prepare(
      const std::vector<std::array<T,4>>& input,
      const std::vector<unsigned char>& mask, int cols, int neighbors, CudaExtraction::Reduction policy) {
    ready=false;
    if (cols <= 0 || neighbors < 0 || neighbors > cols/2 ||
        input.empty() || input.size() > size_t(INT_MAX)/2 ||
        input.size()%cols || mask.size()!=input.size())
      throw std::invalid_argument("Invalid CUDA extraction shape");
    columns=cols; rows=int(input.size()/cols); count=int(input.size());
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
};
CudaExtraction::CudaExtraction(): impl_(std::make_unique<Impl>()) {}
CudaExtraction::~CudaExtraction() = default;
std::vector<double> CudaExtraction::prepare(const std::vector<std::array<float,4>>& scan,
    const std::vector<unsigned char>& valid,int columns,int neighbors, Reduction reduction) {
  return impl_->prepare(scan,valid,columns,neighbors,reduction);
}
std::vector<double> CudaExtraction::prepare(const std::vector<std::array<double,4>>& scan,
    const std::vector<unsigned char>& valid,int columns,int neighbors, Reduction reduction) {
  return impl_->prepare(scan,valid,columns,neighbors,reduction);
}
std::vector<std::array<int,2>> CudaExtraction::nearestRows(const std::vector<size_t>& indices) {
  auto& s=*impl_;
  if (!s.ready) throw std::logic_error("Prepare CUDA extraction before normal search");
  if (indices.size()>size_t(INT_MAX)/2) throw std::invalid_argument("Too many CUDA normal queries");
  std::vector<int> queries; queries.reserve(indices.size());
  for (auto i:indices) {
    if (i>=size_t(s.count)) throw std::out_of_range("CUDA normal query outside scan");
    queries.push_back(int(i));
  }
  std::vector<std::array<int,2>> output(indices.size());
  if (indices.empty()) return output;
  s.queries.reserve(queries.size()); s.nearest.reserve(2*indices.size());
  check(cudaMemcpyAsync(s.queries.data,queries.data(),queries.size()*sizeof(int),cudaMemcpyHostToDevice,s.stream));
  const int blocks=int((indices.size()*2+3)/4);
  if (s.single)
    nearestRowsKernel<<<blocks,128,0,s.stream>>>(reinterpret_cast<const Point<float>*>(s.scan.data),s.valid.data,s.queries.data,int(indices.size()),s.columns,s.rows,s.reduction,s.nearest.data);
  else
    nearestRowsKernel<<<blocks,128,0,s.stream>>>(reinterpret_cast<const Point<double>*>(s.scan.data),s.valid.data,s.queries.data,int(indices.size()),s.columns,s.rows,s.reduction,s.nearest.data);
  check(cudaGetLastError());
  check(cudaMemcpyAsync(output.data(),s.nearest.data,output.size()*2*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  return output;
}
}
