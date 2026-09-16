#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace form {
/// Reusable, serialized by FeatureExtractor, CUDA scanline workspace.
/// The CPU retains ordered selection and Eigen covariance/eigenvector arithmetic.
class CudaExtraction {
public:
  CudaExtraction();
  ~CudaExtraction();
  CudaExtraction(const CudaExtraction&) = delete;
  CudaExtraction& operator=(const CudaExtraction&) = delete;
  std::vector<double> prepare(const std::vector<std::array<float,4>>& scan,
      const std::vector<unsigned char>& valid, int columns, int neighbors);
  std::vector<double> prepare(const std::vector<std::array<double,4>>& scan,
      const std::vector<unsigned char>& valid, int columns, int neighbors);
  /// For each selected index, exact nearest valid index on preceding/following
  /// rows, or -1. Ties use the first scanline index, as in the CPU search.
  std::vector<std::array<int,2>> nearestRows(const std::vector<size_t>& indices);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
