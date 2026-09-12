// MIT License; see factor.hpp for copyright and license text.
#pragma once
#include <memory>
#include <vector>
namespace form {
struct BatchPose { double rotation[9], translation[3]; }; // row major
struct BatchRoot { double plane[169], point[49], weight; int i,j; }; // column major
class CudaSummaryBatch {
 public:
  CudaSummaryBatch(int poses,const std::vector<BatchRoot>& roots,
                   const std::vector<int>& offsets,const std::vector<int>& indices);
  ~CudaSummaryBatch();
  // Output is column-major augmented system, or one squared error (without 1/2).
  void evaluate(const std::vector<BatchPose>& poses, double* output, bool cost);
 private:
  struct Impl; std::unique_ptr<Impl> impl_;
};
}
