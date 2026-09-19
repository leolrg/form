#include <optional>
#include "form/optimization/diagnostics.hpp"
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
size_t addElements(size_t total, size_t count) {
  if(total > std::numeric_limits<size_t>::max()/sizeof(double) ||
     count > std::numeric_limits<size_t>::max()/sizeof(double) - total)
    throw std::invalid_argument("CUDA QR input or scratch size overflows");
  return total+count;
}
struct Event {
  cudaEvent_t value = nullptr;
  Event() { check(cudaEventCreateWithFlags(&value,cudaEventDisableTiming)); }
  ~Event() { if(value) cudaEventDestroy(value); }
};
template<class T> size_t grownCapacity(size_t count,size_t capacity) {
  const size_t limit=std::numeric_limits<size_t>::max()/sizeof(T);
  if(count>limit) throw std::invalid_argument("CUDA QR allocation size overflows");
  return std::max(count,capacity+std::min(capacity/2,limit-capacity));
}
template<class T> struct Buffer {
  T* data = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (data) cudaFree(data); }
  void reserve(size_t count) {
    if (count <= capacity) return;
    const size_t next_capacity = grownCapacity<T>(count,capacity);
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
    const size_t next_capacity = grownCapacity<T>(count,capacity);
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
  const size_t tile = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (tile >= count) return;
  const Task task = tasks[tile];
  if(task.width==7) qrWarp<7,Rows>(input,output,task);
  else qrWarp<13,Rows>(input,output,task);
}
// Each dirty leaf uses exactly the same Householder arithmetic as full QR.
__global__ void qrDirtyLeaves(const double* input,double* roots,const unsigned char* dirty,
                              const Task* tasks,const int* task_groups,int count,
                              unsigned int* changed,unsigned long long* work) {
  const size_t tile=(static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  if(tile>=size_t(count)) return;
  const Task task=tasks[tile];
  if(!dirty[task.source/(64*7)]) return; // Warp-uniform, before any input read.
  if(threadIdx.x%32==0) { atomicExch(changed+task_groups[tile],1u); atomicAdd(work,1ULL); }
  if(task.width==7) qrWarp<7,64>(input,roots,task);
  else qrWarp<13,64>(input,roots,task);
}
__global__ void qrChangedGroups(const double* input,double* output,const Task* tasks,
                                const int* task_groups,int count,const unsigned int* changed,
                                unsigned long long* work) {
  const size_t tile=(static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  if(tile>=size_t(count) || !changed[task_groups[tile]]) return;
  const Task task=tasks[tile];
  if(threadIdx.x%32==0) atomicAdd(work+1,1ULL);
  if(task.width==7) qrWarp<7,64>(input,output,task);
  else qrWarp<13,64>(input,output,task);
}
__global__ void retainGroupRoots(const double* input,double* roots,const size_t* offsets,
                                 const size_t* active,const unsigned int* changed,
                                 int width,unsigned long long* work) {
  const size_t group=blockIdx.x;
  if(!changed[group]) return;
  if(threadIdx.x==0) atomicAdd(work+2,1ULL);
  for(int element=threadIdx.x;element<width*width;element+=blockDim.x)
    roots[group*width*width+element]=active[group]?input[offsets[group]+element]:0.;
}
} // namespace

struct BatchedCudaQr::Impl {
  cudaStream_t stream = nullptr;
  Event producer_ready;
  Buffer<double> a, b;
  Buffer<Task> descriptors;
  HostBuffer<double> host_input, host_roots;
  HostBuffer<Task> host_tasks;
  int first_rows,reduction_rows,block_threads;
  Buffer<double> incremental_leaves,incremental_roots;
  Buffer<int> incremental_task_groups;
  Buffer<size_t> incremental_offsets,incremental_active;
  Buffer<unsigned int> incremental_changed;
  Buffer<unsigned long long> incremental_work;
  size_t incremental_capacity=0,incremental_groups=0;
  bool incremental_plane=false,incremental_ready=false,incremental_stats_ready=false;
  std::vector<size_t> incremental_previous_active;
  size_t incremental_total_active=0;
  std::vector<Eigen::MatrixXd> run(const std::vector<InputLayout>& input, size_t total,
                                   const double* device_input = nullptr, bool device = false);
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
bool BatchedCudaQr::hasInputObserver() { return input_observer!=nullptr; }
BatchedCudaQr::~BatchedCudaQr() = default;
void BatchedCudaQr::resetIncremental() {
  impl_->incremental_ready=false; impl_->incremental_stats_ready=false;
}
BatchedCudaQr::IncrementalStats BatchedCudaQr::incrementalStats() {
  auto& state=*impl_;
  IncrementalStats result;
  if(!state.incremental_stats_ready) return result;
  result.active_leaves=state.incremental_total_active;
  unsigned long long work[3]={};
  check(cudaMemcpyAsync(work,state.incremental_work.data,sizeof(work),cudaMemcpyDeviceToHost,state.stream));
  check(cudaStreamSynchronize(state.stream));
  result.dirty_leaves=work[0]; result.merge_tiles=work[1]; result.changed_groups=work[2];
  return result;
}
std::vector<Eigen::MatrixXd> BatchedCudaQr::computeDevicePackedIncremental(
    const double* packed,const unsigned char* dirty,size_t capacity,
    const std::vector<size_t>& active,bool plane,void* producer_stream) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::qr_total);
  auto& state=*impl_;
  const bool retained=state.incremental_ready && state.incremental_capacity==capacity &&
    state.incremental_groups==active.size() && state.incremental_plane==plane;
  // Any failure leaves the next call conservatively zero-initialized.
  state.incremental_ready=false; state.incremental_stats_ready=false;
  const size_t groups=active.size();
  const int width=plane?13:7;
  constexpr size_t leaf_elements=64*7;
  const size_t square=size_t(width)*width;
  if(groups>size_t(std::numeric_limits<int>::max()) ||
     (groups && capacity>std::numeric_limits<size_t>::max()/groups))
    throw std::invalid_argument("CUDA incremental QR layout overflows");
  const size_t allocated=groups*capacity;
  if(allocated>std::numeric_limits<size_t>::max()/leaf_elements)
    throw std::invalid_argument("CUDA incremental QR leaf layout overflows");
  addElements(0,allocated*leaf_elements);
  const size_t leaf_root_elements=addElements(0,allocated*square);
  const size_t group_root_elements=addElements(0,groups*square);
  size_t total_active=0;
  for(size_t count:active) {
    if(count>capacity || count>size_t(std::numeric_limits<int>::max()) ||
       total_active>size_t(std::numeric_limits<int>::max())-count)
      throw std::invalid_argument("CUDA incremental QR active extent is invalid");
    total_active+=count;
  }
  if(total_active && (!packed || !dirty))
    throw std::invalid_argument("CUDA incremental QR device input is null");
  check(cudaEventRecord(state.producer_ready.value,static_cast<cudaStream_t>(producer_stream)));
  check(cudaStreamWaitEvent(state.stream,state.producer_ready.value,0));
  state.incremental_work.reserve(3);
  check(cudaMemsetAsync(state.incremental_work.data,0,3*sizeof(unsigned long long),state.stream));
  state.incremental_leaves.reserve(std::max<size_t>(1,leaf_root_elements));
  state.incremental_roots.reserve(std::max<size_t>(1,group_root_elements));
  if(!retained) {
    if(leaf_root_elements) check(cudaMemsetAsync(state.incremental_leaves.data,0,leaf_root_elements*sizeof(double),state.stream));
    if(group_root_elements) check(cudaMemsetAsync(state.incremental_roots.data,0,group_root_elements*sizeof(double),state.stream));
  }
  std::vector<unsigned int> changed(groups);
  std::vector<Task> tasks;
  std::vector<int> task_groups;
  std::vector<Layout> layout;
  for(size_t group=0;group<groups;++group) {
    changed[group]=!retained || active[group]!=state.incremental_previous_active[group];
    layout.push_back({group*capacity*square,int(active[group]),width});
    for(size_t leaf=0;leaf<active[group];++leaf) {
      const size_t flat=group*capacity+leaf;
      tasks.push_back({flat*leaf_elements,flat*square,64,width,64,plane?2:0});
      task_groups.push_back(int(group));
    }
  }
  const size_t first_count=tasks.size();
  std::vector<std::pair<size_t,size_t>> levels;
  while(std::any_of(layout.begin(),layout.end(),[](auto item){return item.roots>1;})) {
    const size_t begin=tasks.size();
    std::vector<Layout> next;
    size_t destination=0;
    for(size_t group=0;group<groups;++group) {
      const auto item=layout[group];
      const int fanin=64/width;
      const int roots=item.roots/fanin+(item.roots%fanin!=0);
      next.push_back({destination,roots,width});
      for(int root=0;root<roots;++root) {
        tasks.push_back({item.offset+size_t(root)*fanin*square,destination+size_t(root)*square,
                         std::min(fanin,item.roots-root*fanin)*width,width,width,1});
        task_groups.push_back(int(group));
      }
      destination=addElements(destination,size_t(roots)*square);
    }
    levels.push_back({begin,tasks.size()-begin}); layout=std::move(next);
  }
  std::vector<size_t> offsets;
  for(const auto& item:layout) offsets.push_back(item.offset);
  state.incremental_changed.reserve(groups); state.incremental_active.reserve(groups);
  state.incremental_offsets.reserve(groups); state.incremental_task_groups.reserve(tasks.size());
  state.descriptors.reserve(tasks.size()); state.host_tasks.reserve(tasks.size());
  state.a.reserve(std::max<size_t>(1,total_active*square));
  state.b.reserve(std::max<size_t>(1,total_active*square));
  state.host_roots.reserve(std::max<size_t>(1,group_root_elements));
  if(groups) {
    check(cudaMemcpyAsync(state.incremental_changed.data,changed.data(),groups*sizeof(unsigned int),cudaMemcpyHostToDevice,state.stream));
    check(cudaMemcpyAsync(state.incremental_active.data,active.data(),groups*sizeof(size_t),cudaMemcpyHostToDevice,state.stream));
    check(cudaMemcpyAsync(state.incremental_offsets.data,offsets.data(),groups*sizeof(size_t),cudaMemcpyHostToDevice,state.stream));
  }
  if(!tasks.empty()) {
    std::memcpy(state.host_tasks.data,tasks.data(),tasks.size()*sizeof(Task));
    check(cudaMemcpyAsync(state.descriptors.data,state.host_tasks.data,tasks.size()*sizeof(Task),cudaMemcpyHostToDevice,state.stream));
    check(cudaMemcpyAsync(state.incremental_task_groups.data,task_groups.data(),tasks.size()*sizeof(int),cudaMemcpyHostToDevice,state.stream));
  }
  const int warps=state.block_threads/32;
  if(first_count) {
    qrDirtyLeaves<<<(first_count+warps-1)/warps,state.block_threads,0,state.stream>>>(
      packed,state.incremental_leaves.data,dirty,state.descriptors.data,state.incremental_task_groups.data,
      int(first_count),state.incremental_changed.data,state.incremental_work.data);
    check(cudaGetLastError());
  }
  const double* source=state.incremental_leaves.data;
  double* destination=state.a.data;
  for(auto level:levels) {
    qrChangedGroups<<<(level.second+warps-1)/warps,state.block_threads,0,state.stream>>>(
      source,destination,state.descriptors.data+level.first,state.incremental_task_groups.data+level.first,
      int(level.second),state.incremental_changed.data,state.incremental_work.data);
    check(cudaGetLastError());
    source=destination; destination=destination==state.a.data?state.b.data:state.a.data;
  }
  if(groups) {
    retainGroupRoots<<<groups,32,0,state.stream>>>(source,state.incremental_roots.data,
      state.incremental_offsets.data,state.incremental_active.data,state.incremental_changed.data,width,state.incremental_work.data);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(state.host_roots.data,state.incremental_roots.data,group_root_elements*sizeof(double),cudaMemcpyDeviceToHost,state.stream));
  }
  check(cudaStreamSynchronize(state.stream));
  if(input_observer) {
    // Diagnostic-only download preserves group rows, including all-zero holes.
    state.host_input.reserve(std::max<size_t>(1,total_active*leaf_elements));
    size_t cursor=0;
    for(size_t group=0;group<groups;++group) {
      const size_t count=active[group]*leaf_elements;
      if(count) check(cudaMemcpyAsync(state.host_input.data+cursor,packed+group*capacity*leaf_elements,count*sizeof(double),cudaMemcpyDeviceToHost,state.stream));
      cursor+=count;
    }
    check(cudaStreamSynchronize(state.stream));
    std::vector<Eigen::MatrixXd> observed;
    cursor=0;
    for(size_t group=0;group<groups;++group) {
      Eigen::MatrixXd expanded(active[group]*64,width);
      for(size_t leaf=0;leaf<active[group];++leaf) {
        const Eigen::Map<const Eigen::Matrix<double,64,7>> compact(state.host_input.data+cursor);
        if(!plane) expanded.middleRows(leaf*64,64)=compact;
        else {
          for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            expanded.block(leaf*64,3*a+b,64,1)=compact.col(3+a).cwiseProduct(compact.col(b));
          expanded.block(leaf*64,9,64,3)=compact.middleCols(3,3);
          expanded.block(leaf*64,12,64,1)=compact.col(6);
        }
        cursor+=leaf_elements;
      }
      observed.emplace_back(std::move(expanded));
    }
    input_observer(observed);
  }
  std::vector<Eigen::MatrixXd> result;
  for(size_t group=0;group<groups;++group)
    result.emplace_back(Eigen::Map<const Eigen::MatrixXd>(state.host_roots.data+group*square,width,width));
  state.incremental_capacity=capacity; state.incremental_groups=groups; state.incremental_plane=plane;
  state.incremental_previous_active=active; state.incremental_total_active=total_active;
  state.incremental_ready=true; state.incremental_stats_ready=true;
  return result;
}

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
BatchedCudaQr::computeDevicePacked(const double* packed,
                                  const std::vector<DeviceInput>& input,
                                  void* producer_stream) {
  if(input.empty()) return {};
  size_t total=0;
  std::vector<InputLayout> shapes;
  shapes.reserve(input.size());
  for(const auto& item:input) {
    if(item.rows>static_cast<size_t>(std::numeric_limits<int>::max()) ||
       item.rows>std::numeric_limits<size_t>::max()/7)
      throw std::invalid_argument("CUDA QR input has too many rows");
    total=addElements(total,item.rows*7);
    shapes.push_back({static_cast<int>(item.rows),item.plane?13:7,7});
  }
  if(total && !packed) throw std::invalid_argument("CUDA QR device input pointer is null");
  auto& state=*impl_;
  check(cudaEventRecord(state.producer_ready.value,static_cast<cudaStream_t>(producer_stream)));
  check(cudaStreamWaitEvent(state.stream,state.producer_ready.value,0));
  if(input_observer) {
    state.host_input.reserve(std::max<size_t>(1,total));
    if(total) check(cudaMemcpyAsync(state.host_input.data,packed,total*sizeof(double),cudaMemcpyDeviceToHost,state.stream));
    check(cudaStreamSynchronize(state.stream));
    std::vector<Eigen::MatrixXd> observed;
    size_t offset=0;
    for(const auto& item:input) {
      const Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,7>> compact(state.host_input.data+offset,item.rows,7);
      if(!item.plane) observed.emplace_back(compact);
      else {
        Eigen::MatrixXd expanded(item.rows,13);
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
          expanded.col(3*a+b)=compact.col(3+a).cwiseProduct(compact.col(b));
        expanded.middleCols(9,3)=compact.middleCols(3,3);
        expanded.col(12)=compact.col(6);
        observed.emplace_back(std::move(expanded));
      }
      offset+=item.rows*7;
    }
    input_observer(observed);
  }
  return state.run(shapes,total,packed,true);
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
BatchedCudaQr::Impl::run(const std::vector<InputLayout>& input,size_t total,
                         const double* device_input,bool device) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::qr_total);
  std::optional<diagnostics::Scope> phase; phase.emplace(diagnostics::Stage::qr_plan);
  auto& state=*this;
  std::vector<Task> first;
  std::vector<Layout> layout;
  size_t input_offset=0, output_offset=0;
  for(const auto& matrix:input) {
    const int width=matrix.width, rows=matrix.rows;
    const int roots=rows?1+(rows-1)/first_rows:1;
    layout.push_back({output_offset,roots,width});
    for(int tile=0;tile<roots;++tile)
      first.push_back({input_offset+tile*first_rows,output_offset+static_cast<size_t>(tile)*width*width,
                       std::max(0,std::min(first_rows,rows-tile*first_rows)),width,rows,width!=matrix.stored_width?2:0});
    input_offset = addElements(input_offset,static_cast<size_t>(rows)*matrix.stored_width);
    output_offset = addElements(output_offset,static_cast<size_t>(roots)*width*width);
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
        all.push_back({l.offset+static_cast<size_t>(r)*group*l.width*l.width,
                       destination+static_cast<size_t>(r)*l.width*l.width,
                       std::min(group,l.roots-r*group)*l.width,l.width,l.width,1});
      destination = addElements(destination,static_cast<size_t>(roots)*l.width*l.width);
    }
    levels.push_back({begin,all.size()-begin});
    layout=std::move(next);
  }
  const size_t final_size=layout.back().offset+layout.back().width*layout.back().width;
  state.host_roots.reserve(final_size);
  state.a.reserve(std::max<size_t>(1,std::max(device?size_t(0):total,max_output)));
  state.b.reserve(std::max<size_t>(1,max_output));
  for(auto level:levels)
    if(level.second>static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("CUDA QR has too many tiles");
  state.descriptors.reserve(all.size());
  state.host_tasks.reserve(all.size());
  std::memcpy(state.host_tasks.data, all.data(), all.size()*sizeof(Task));
  phase.emplace(diagnostics::Stage::qr_execute);
  if(!device && total) check(cudaMemcpyAsync(state.a.data,state.host_input.data,total*sizeof(double),cudaMemcpyHostToDevice,state.stream));
  check(cudaMemcpyAsync(state.descriptors.data,state.host_tasks.data,all.size()*sizeof(Task),cudaMemcpyHostToDevice,state.stream));
  auto launch=[&] {
    const double* src=device?device_input:state.a.data;
    double* dst=state.b.data;
    bool first_level=true;
    for(auto level:levels) {
      const int warps=block_threads/32;
      const int blocks=(level.second+warps-1)/warps;
      if((first_level?first_rows:reduction_rows)==32)
        qrTiles<32><<<blocks,block_threads,0,state.stream>>>(src,dst,state.descriptors.data+level.first,level.second);
      else
        qrTiles<64><<<blocks,block_threads,0,state.stream>>>(src,dst,state.descriptors.data+level.first,level.second);
      check(cudaGetLastError());
      src=dst;
      // Never use the caller's device pointer as a reduction destination.
      dst=dst==state.b.data?state.a.data:state.b.data;
      first_level=false;
    }
  };
  launch();
  double* src=levels.size()%2?state.b.data:state.a.data;
  check(cudaMemcpyAsync(state.host_roots.data,src,final_size*sizeof(double),cudaMemcpyDeviceToHost,state.stream));
  check(cudaStreamSynchronize(state.stream));
  phase.emplace(diagnostics::Stage::qr_output);
  std::vector<Eigen::MatrixXd> result;
  result.reserve(input.size());
  for(auto l:layout)
    result.emplace_back(Eigen::Map<const Eigen::MatrixXd>(state.host_roots.data+l.offset,l.width,l.width));
  return result;
}
} // namespace form
