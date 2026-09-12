#pragma once

#include <Eigen/Core>
#include <memory>

namespace form {

// FP64 Cholesky solve of the top-left N x N block of a full symmetric
// (N + 1) x (N + 1) augmented matrix, using its last column as the RHS.
// Instances retain bounded growable CUDA storage and are not thread-safe.
class CudaDenseSolver {
public:
  CudaDenseSolver();
  ~CudaDenseSolver();
  CudaDenseSolver(const CudaDenseSolver&) = delete;
  CudaDenseSolver& operator=(const CudaDenseSolver&) = delete;

  // Returns false on numerical failure, leaving solution unchanged. Invalid
  // dimensions throw invalid_argument; CUDA runtime/setup failures throw.
  bool solve(const Eigen::MatrixXd& augmented, Eigen::VectorXd& solution);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace form
