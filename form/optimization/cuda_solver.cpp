#include "form/optimization/cuda_solver.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef EIGEN_GPUCC
#error Compile the host-only CUDA dense solver with ordinary C++, not NVCC
#endif

namespace form {
namespace {

void checkCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA dense solver: ") + operation +
                             ": " + cudaGetErrorString(status));
}

void checkSolver(cusolverStatus_t status, const char* operation) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string("CUDA dense solver: ") + operation +
                             " failed with cuSolver status " + std::to_string(status));
}

// Owning buffers also clean up allocations made before initialization fails.
template <typename T, bool Pinned = false>
struct Buffer {
  T* data = nullptr;
  size_t capacity = 0;

  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { release(); }

  void release() noexcept {
    if (data) {
      if (Pinned) cudaFreeHost(data);
      else cudaFree(data);
    }
    data = nullptr;
    capacity = 0;
  }

  void reserve(size_t count) {
    if (count <= capacity) return;
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::length_error("CUDA dense solver allocation is too large");
    T* next = nullptr;
    if (Pinned)
      checkCuda(cudaMallocHost(reinterpret_cast<void**>(&next), count * sizeof(T)),
                "allocate pinned buffer");
    else
      checkCuda(cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(T)),
                "allocate device buffer");
    release();
    data = next;
    capacity = count;
  }
};

}  // namespace

struct CudaDenseSolver::Impl {
  cudaStream_t stream = nullptr;
  cusolverDnHandle_t handle = nullptr;
  Buffer<double> device, workspace;
  Buffer<double, true> host;
  Buffer<int> info;
  Buffer<int, true> hostInfo;
  size_t capacity = 0;
  int workspaceN = 0;
  int workspaceSize = 0;

  void initialize() {
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    checkSolver(cusolverDnCreate(&handle), "create handle");
    checkSolver(cusolverDnSetStream(handle, stream), "set stream");
    info.reserve(2);
    hostInfo.reserve(2);
  }

  ~Impl() noexcept {
    if (stream) cudaStreamSynchronize(stream);
    if (handle) cusolverDnDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  }

  void reserve(int n) {
    if (static_cast<size_t>(n) > capacity) {
      const size_t next = std::max(static_cast<size_t>(n), capacity + (capacity + 1) / 2);
      // A single allocation stores the matrix and RHS; only the largest
      // capacity survives. Actual matrices are packed with leading dimension N.
      if (next > std::numeric_limits<size_t>::max() / (next + 1) / sizeof(double))
        throw std::length_error("CUDA dense solver matrix is too large");
      const size_t count = next * (next + 1);
      device.reserve(count);
      host.reserve(count);
      capacity = next;
    }
    if (workspaceN != n) {
      int required = 0;
      checkSolver(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
                                             n, device.data, n, &required),
                  "query Cholesky workspace");
      if (required < 0)
        throw std::runtime_error("CUDA dense solver received an invalid workspace size");
      const size_t count = std::max(1, required);
      if (count > workspace.capacity)
        workspace.reserve(std::max(count, workspace.capacity + (workspace.capacity + 1) / 2));
      workspaceSize = required;
      workspaceN = n;
    }
  }

  bool solve(const Eigen::MatrixXd& augmented, Eigen::VectorXd& solution) {
    if (augmented.rows() != augmented.cols() || augmented.rows() < 2 ||
        augmented.rows() - 1 > std::numeric_limits<int>::max())
      throw std::invalid_argument("CUDA dense solver requires a square augmented matrix with N >= 1");
    const int n = static_cast<int>(augmented.rows() - 1);
    if (!augmented.topLeftCorner(n, n).allFinite() ||
        !augmented.topRightCorner(n, 1).allFinite()) return false;
    reserve(n);
    double* hostRhs = host.data + capacity * capacity;
    double* deviceRhs = device.data + capacity * capacity;
    for (int j = 0; j < n; ++j)
      std::memcpy(host.data + size_t(j) * n,
                  augmented.data() + size_t(j) * augmented.outerStride(),
                  size_t(n) * sizeof(double));
    std::memcpy(hostRhs, augmented.col(n).data(), size_t(n) * sizeof(double));

    try {
      checkCuda(cudaMemcpyAsync(device.data, host.data, size_t(n) * n * sizeof(double),
                                cudaMemcpyHostToDevice, stream), "upload matrix");
      checkCuda(cudaMemcpyAsync(deviceRhs, hostRhs, size_t(n) * sizeof(double),
                                cudaMemcpyHostToDevice, stream), "upload RHS");
      checkSolver(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, device.data, n,
                                   workspace.data, workspaceSize, info.data), "factor matrix");
      // Both operations share the stream. Numerical factor failure is checked
      // with the solve status after one synchronization, before accepting output.
      checkSolver(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1, device.data,
                                   n, deviceRhs, n, info.data + 1), "solve system");
      checkCuda(cudaMemcpyAsync(hostInfo.data, info.data, 2 * sizeof(int),
                                cudaMemcpyDeviceToHost, stream), "download solver status");
      checkCuda(cudaMemcpyAsync(hostRhs, deviceRhs, size_t(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, stream), "download solution");
      checkCuda(cudaStreamSynchronize(stream), "synchronize solve");
    } catch (...) {
      // Do not leave pending transfers accessing staging buffers if the caller
      // catches a runtime failure and later reuses this instance.
      cudaStreamSynchronize(stream);
      throw;
    }
    if (hostInfo.data[0] != 0 || hostInfo.data[1] != 0) return false;
    const Eigen::Map<const Eigen::VectorXd> result(hostRhs, n);
    if (!result.allFinite()) return false;
    solution = result;
    return true;
  }
};

CudaDenseSolver::CudaDenseSolver() : impl_(std::make_unique<Impl>()) {
  impl_->initialize();
}
CudaDenseSolver::~CudaDenseSolver() = default;
bool CudaDenseSolver::solve(const Eigen::MatrixXd& augmented, Eigen::VectorXd& solution) {
  return impl_->solve(augmented, solution);
}
}  // namespace form
