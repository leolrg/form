#include "form/feature/cuda_qr.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>

namespace form {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA QR: ") + cudaGetErrorString(status));
}
BatchedCudaQr::InputObserver input_observer = nullptr;
template<class T> struct Buffer {
  T* data = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (data) cudaFree(data); }
  void reserve(size_t count) {
    if (count <= capacity) return;
    const size_t next_capacity = std::max(count, capacity + capacity / 2);
    T* next = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&next), next_capacity * sizeof(T)));
    if (data) cudaFree(data);
    data = next; capacity = next_capacity;
  }
};
template<class T> struct HostBuffer {
  T* data = nullptr;
  size_t capacity = 0;
  bool pinned;
  explicit HostBuffer(bool use_pinned) : pinned(use_pinned) {}
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  ~HostBuffer() { release(); }
  void release() {
    if (!data) return;
    if (pinned) cudaFreeHost(data);
    else delete[] data;
    data = nullptr;
  }
  void reserve(size_t count) {
    if (count <= capacity) return;
    const size_t next_capacity = std::max(count, capacity + capacity / 2);
    T* next = nullptr;
    if (pinned) check(cudaHostAlloc(reinterpret_cast<void**>(&next), next_capacity*sizeof(T), cudaHostAllocDefault));
    else next = new T[next_capacity];
    release();
    data = next; capacity = next_capacity;
  }
};
struct Task {
  uint64_t source, destination;
  int rows, width, stride, stacked;
};
struct Layout { size_t offset; int roots, width; };
struct InputLayout { int rows, width, stored_width; };

__device__ double warpSum(double value) {
  for (int shift=16; shift; shift/=2) value += __shfl_down_sync(0xffffffff, value, shift);
  return __shfl_sync(0xffffffff, value, 0);
}
// One full warp factors 32 or 64 rows. Every lane participates in every
// shuffle, including padded rows. All arithmetic and reductions are FP64.
// Fixed reduction order and stable Householder signs avoid atomics and avoid
// forming A^T A, whose condition number would square before compression.
template<int Width, int Rows>
__device__ void qrWarp(const double* input, double* output, const Task& task) {
  const int lane = threadIdx.x % 32;
  double a[Rows/32][Width];
#pragma unroll
  for(int slot=0;slot<Rows/32;++slot) {
  const int row=lane+32*slot;
#pragma unroll
  for (int col=0; col<Width; ++col) {
    const size_t index = task.stacked == 1
      ? task.source + (row/Width)*Width*Width + row%Width + col*Width
      : task.source + row + col*static_cast<size_t>(task.stride);
    if(task.stacked == 2) {
      // Compact plane row is [pj, n, n.dot(pj-pi)]. Expand the tensor
      // while loading this tile, without writing a full feature matrix.
      const size_t base = task.source + row;
      a[slot][col] = row >= task.rows ? 0. : col < 9
        ? input[base+(3+col/3)*static_cast<size_t>(task.stride)] *
          input[base+(col%3)*static_cast<size_t>(task.stride)]
        : input[base+(col<12?col-6:6)*static_cast<size_t>(task.stride)];
    } else a[slot][col] = row < task.rows ? input[index] : 0.;
  }
  }
#pragma unroll
  for (int col=0; col<Width; ++col) {
    double local_norm2=0.;
#pragma unroll
    for(int slot=0;slot<Rows/32;++slot) {
      const double x=lane+32*slot>=col?a[slot][col]:0.;
      local_norm2+=x*x;
    }
    const double norm2 = warpSum(local_norm2);
    if (norm2 == 0.) continue;
    const double diagonal = __shfl_sync(0xffffffff, a[0][col], col);
    const double alpha = -copysign(sqrt(norm2), diagonal);
    double v[Rows/32],local_vv=0.;
#pragma unroll
    for(int slot=0;slot<Rows/32;++slot) {
      const int row=lane+32*slot;
      v[slot]=row==col?diagonal-alpha:(row>col?a[slot][col]:0.);
      local_vv+=v[slot]*v[slot];
    }
    const double vv = warpSum(local_vv);
#pragma unroll
    for (int next=0; next<Width; ++next) {
      if(next>col) {
        double local_dot=0.;
#pragma unroll
        for(int slot=0;slot<Rows/32;++slot) local_dot+=v[slot]*a[slot][next];
        const double dot = warpSum(local_dot);
        const double scale=2.*dot/vv;
#pragma unroll
        for(int slot=0;slot<Rows/32;++slot) a[slot][next] -= scale*v[slot];
      }
    }
#pragma unroll
    for(int slot=0;slot<Rows/32;++slot) {
      const int row=lane+32*slot;
      if (row >= col) a[slot][col] = row == col ? alpha : 0.;
    }
  }
  if (lane < Width) {
#pragma unroll
    for(int col=0; col<Width; ++col)
      output[task.destination + lane + col*Width] = a[0][col];
  }
}
template<int Rows>
__global__ void qrTiles(const double* input, double* output,
                        const Task* tasks, int count) {
  const int tile = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (tile >= count) return;
  const Task task = tasks[tile];
  if(task.width==7) qrWarp<7,Rows>(input,output,task);
  else qrWarp<13,Rows>(input,output,task);
}
} // namespace

struct BatchedCudaQr::Impl {
  cudaStream_t stream = nullptr;
  Buffer<double> a, b;
  Buffer<Task> descriptors;
  HostBuffer<double> host_input, host_roots;
  HostBuffer<Task> host_tasks;
  int first_rows,reduction_rows,block_threads;
  std::vector<Eigen::MatrixXd> run(const std::vector<InputLayout>& input, size_t total);
  explicit Impl(bool pinned,int first,int reduction,int threads)
      : host_input(pinned), host_roots(pinned), host_tasks(pinned),
        first_rows(first),reduction_rows(reduction),block_threads(threads) {
    if((first!=32 && first!=64) || (reduction!=32 && reduction!=64) ||
       (threads!=32 && threads!=64 && threads!=128 && threads!=256))
      throw std::invalid_argument("Invalid CUDA QR tile or block size");
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  }
  ~Impl() {
    if(stream) cudaStreamSynchronize(stream);
    if(stream) cudaStreamDestroy(stream);
  }
};
BatchedCudaQr::BatchedCudaQr(bool pinned_host_buffers,int first_rows,int reduction_rows,int block_threads)
    : impl_(std::make_unique<Impl>(pinned_host_buffers,first_rows,reduction_rows,block_threads)) {}
void BatchedCudaQr::setInputObserver(InputObserver observer) { input_observer = observer; }
BatchedCudaQr::~BatchedCudaQr() = default;

std::vector<Eigen::MatrixXd>
BatchedCudaQr::computeCorrespondences(const std::vector<Correspondences>& input) {
  if(input.empty()) return {};
  size_t total=0;
  std::vector<InputLayout> shapes;
  for(const auto& item:input) {
    if(item.rows>static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("CUDA QR input has too many rows");
    if(item.rows && (!item.pi || !item.pj || (item.plane && !item.normal)))
      throw std::invalid_argument("CUDA QR correspondence pointer is null");
    shapes.push_back({static_cast<int>(item.rows),item.plane?13:7,7});
    total+=item.rows*7;
  }
  auto& state=*impl_;
  state.host_input.reserve(std::max<size_t>(1,total));
  size_t offset=0;
  std::vector<Eigen::MatrixXd> observed;
  for(const auto& item:input) {
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,7>> packed(state.host_input.data+offset,item.rows,7);
    for(size_t k=0;k<item.rows;++k) {
      const Eigen::Map<const Eigen::Vector3d> pi(item.pi+3*k),pj(item.pj+3*k);
      if(item.plane) {
        const Eigen::Map<const Eigen::Vector3d> n(item.normal+3*k);
        packed.template block<1,3>(k,0)=pj.transpose();
        packed.template block<1,3>(k,3)=n.transpose();
        packed(k,6)=n.dot(pj-pi);
      } else {
        packed(k,0)=1.;
        packed.template block<1,3>(k,1)=pi.transpose();
        packed.template block<1,3>(k,4)=(pj-pi).transpose();
      }
    }
    if(input_observer) {
      if(!item.plane) observed.emplace_back(packed);
      else {
        Eigen::MatrixXd expanded(item.rows,13);
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
          expanded.col(3*a+b)=packed.col(3+a).cwiseProduct(packed.col(b));
        expanded.middleCols(9,3)=packed.middleCols(3,3);
        expanded.col(12)=packed.col(6);
        observed.emplace_back(std::move(expanded));
      }
    }
    offset+=item.rows*7;
  }
  if(input_observer) input_observer(observed);
  return state.run(shapes,total);
}

std::vector<Eigen::MatrixXd>
BatchedCudaQr::compute(const std::vector<Eigen::MatrixXd>& input) {
  if (input.empty()) return {};
  size_t total = 0;
  for (const auto& matrix : input) {
    if (matrix.cols()!=7 && matrix.cols()!=13)
      throw std::invalid_argument("CUDA QR supports 7 or 13 columns");
    if (matrix.rows() > std::numeric_limits<int>::max())
      throw std::invalid_argument("CUDA QR input has too many rows");
    total += matrix.size();
  }
  if (input_observer) input_observer(input);
  auto& state = *impl_;
  state.host_input.reserve(std::max<size_t>(1,total));
  std::vector<InputLayout> shapes;
  size_t offset=0;
  for(const auto& matrix:input) {
    shapes.push_back({static_cast<int>(matrix.rows()),static_cast<int>(matrix.cols()),static_cast<int>(matrix.cols())});
    if(matrix.size()) std::memcpy(state.host_input.data+offset,matrix.data(),matrix.size()*sizeof(double));
    offset+=matrix.size();
  }
  return state.run(shapes,total);
}

std::vector<Eigen::MatrixXd>
BatchedCudaQr::Impl::run(const std::vector<InputLayout>& input,size_t total) {
  auto& state=*this;
  std::vector<Task> first;
  std::vector<Layout> layout;
  size_t input_offset=0, output_offset=0;
  for(const auto& matrix:input) {
    const int width=matrix.width, rows=matrix.rows;
    const int roots=rows?1+(rows-1)/first_rows:1;
    layout.push_back({output_offset,roots,width});
    for(int tile=0;tile<roots;++tile)
      first.push_back({input_offset+tile*first_rows,output_offset+tile*width*width,
                       std::max(0,std::min(first_rows,rows-tile*first_rows)),width,rows,width!=matrix.stored_width?2:0});
    input_offset += static_cast<size_t>(rows)*matrix.stored_width; output_offset += roots*width*width;
  }
  // Build every reduction level before upload: descriptor storage remains
  // immutable until the stream has finished all kernels.
  std::vector<Task> all=first;
  std::vector<std::pair<size_t,size_t>> levels{{0,first.size()}};
  size_t max_output=output_offset;
  while (std::any_of(layout.begin(),layout.end(),[](auto l){return l.roots>1;})) {
    std::vector<Layout> next;
    const size_t begin=all.size();
    size_t destination=0;
    for(auto l:layout) {
      const int group=reduction_rows/l.width;
      const int roots=(l.roots+group-1)/group;
      next.push_back({destination,roots,l.width});
      for(int r=0;r<roots;++r)
        all.push_back({l.offset+static_cast<size_t>(r*group*l.width*l.width),
                       destination+static_cast<size_t>(r*l.width*l.width),
                       std::min(group,l.roots-r*group)*l.width,l.width,l.width,1});
      destination += roots*l.width*l.width;
    }
    levels.push_back({begin,all.size()-begin});
    layout=std::move(next);
  }
  const size_t final_size=layout.back().offset+layout.back().width*layout.back().width;
  state.host_roots.reserve(final_size);
  state.a.reserve(std::max<size_t>(1,std::max(total,max_output)));
  state.b.reserve(std::max<size_t>(1,max_output));
  state.descriptors.reserve(all.size());
  state.host_tasks.reserve(all.size());
  std::memcpy(state.host_tasks.data, all.data(), all.size()*sizeof(Task));
  if(total) check(cudaMemcpyAsync(state.a.data,state.host_input.data,total*sizeof(double),cudaMemcpyHostToDevice,state.stream));
  check(cudaMemcpyAsync(state.descriptors.data,state.host_tasks.data,all.size()*sizeof(Task),cudaMemcpyHostToDevice,state.stream));
  auto launch=[&] {
    double* src=state.a.data; double* dst=state.b.data;
    bool first_level=true;
    for(auto level:levels) {
      const int warps=block_threads/32;
      const int blocks=(level.second+warps-1)/warps;
      if((first_level?first_rows:reduction_rows)==32)
        qrTiles<32><<<blocks,block_threads,0,state.stream>>>(src,dst,state.descriptors.data+level.first,level.second);
      else
        qrTiles<64><<<blocks,block_threads,0,state.stream>>>(src,dst,state.descriptors.data+level.first,level.second);
      check(cudaGetLastError());
      std::swap(src,dst);
      first_level=false;
    }
  };
  launch();
  double* src=levels.size()%2?state.b.data:state.a.data;
  check(cudaMemcpyAsync(state.host_roots.data,src,final_size*sizeof(double),cudaMemcpyDeviceToHost,state.stream));
  check(cudaStreamSynchronize(state.stream));
  std::vector<Eigen::MatrixXd> result;
  result.reserve(input.size());
  for(auto l:layout)
    result.emplace_back(Eigen::Map<const Eigen::MatrixXd>(state.host_roots.data+l.offset,l.width,l.width));
  return result;
}
} // namespace form
