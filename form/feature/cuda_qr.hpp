#pragma once
#include <Eigen/Core>
#include <memory>
#include <vector>
namespace form {
/// Batched double-precision tall-skinny QR for 7/13-column feature matrices.
/// Returns zero-padded upper roots R with R^T R = A^T A. Owns reusable CUDA
/// buffers and a stream; one caller at a time. compute includes all transfers.
class BatchedCudaQr {
public:
  explicit BatchedCudaQr(bool pinned_host_buffers = true, int first_rows = 64,
                        int reduction_rows = 64, int block_threads = 32);
  using InputObserver = void (*)(const std::vector<Eigen::MatrixXd>&);
  /// Diagnostic hook; set before processing, one caller at a time.
  static void setInputObserver(InputObserver observer);
  ~BatchedCudaQr();
  BatchedCudaQr(const BatchedCudaQr&) = delete;
  BatchedCudaQr& operator=(const BatchedCudaQr&) = delete;
  std::vector<Eigen::MatrixXd> compute(const std::vector<Eigen::MatrixXd>& input);
  struct Correspondences {
    const double *pi, *pj, *normal;
    size_t rows;
    bool plane;
  };
  /// Coordinates/normals are interleaved xyz. Packs seven doubles per row
  /// directly into staging memory; expands plane tensors in the first kernel.
  std::vector<Eigen::MatrixXd> computeCorrespondences(const std::vector<Correspondences>& input);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
