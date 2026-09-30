#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace form {
/// Reusable, serialized by FeatureExtractor, CUDA scanline workspace.
/// Masks, ordered selection and normal construction can remain resident.
class CudaExtraction {
public:
  enum class Reduction { Cross, Adjacent, Sequential };
  struct Selection {
    int columns,neighbors,sectors,spacing;
    size_t planes,points,min_points;
    double min_squared,max_squared,threshold,radius;
    bool stable_order = false;
  };
  struct Features {
    std::vector<int> planes,points;
    std::vector<std::array<double,4>> normals;
  };
  CudaExtraction();
  ~CudaExtraction();
  CudaExtraction(const CudaExtraction&) = delete;
  CudaExtraction& operator=(const CudaExtraction&) = delete;
  std::vector<double> prepare(const std::vector<std::array<float,4>>& scan,
      const std::vector<unsigned char>& valid, int columns, int neighbors,
      Reduction reduction = Reduction::Cross);
  std::vector<double> prepare(const std::vector<std::array<double,4>>& scan,
      const std::vector<unsigned char>& valid, int columns, int neighbors,
      Reduction reduction = Reduction::Cross);
  /// For each selected index, exact nearest valid index on preceding/following
  /// rows, or -1. Ties use the first scanline index, as in the CPU search.
  std::vector<std::array<int,2>> nearestRows(const std::vector<size_t>& indices);
  /// GPU neighborhood/covariance/eigensolver. xyz is the unit normal; w is 1
  /// when the original neighborhood requirements are met, otherwise 0.
  std::vector<std::array<double,4>> normals(const std::vector<size_t>& indices,
      double radius, size_t min_points);
  /// Resident masks, ordered selection and normals. Only completed indices
  /// and normals return; intermediate masks/curvatures/queries stay on device.
  Features extract(const std::vector<std::array<float,4>>& scan,
                   const Selection& params,Reduction reduction);
  Features extract(const std::vector<std::array<double,4>>& scan,
                   const Selection& params,Reduction reduction);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
