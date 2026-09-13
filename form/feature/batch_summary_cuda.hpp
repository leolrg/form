// MIT License; see factor.hpp for copyright and license text.
#pragma once
#include <memory>
#include "form/optimization/resident_system.hpp"
#include <vector>
namespace form {
struct BatchPose { double rotation[9], translation[3]; }; // row major
struct BatchRoot { double plane[169], point[49], weight; int i,j; }; // column major
class CudaSummaryBatch {
 public:
  CudaSummaryBatch(int poses,const std::vector<BatchRoot>& roots,
                   const std::vector<int>& offsets,const std::vector<int>& indices);
  ~CudaSummaryBatch();
  void reset(int poses,const std::vector<BatchRoot>& roots,
             const std::vector<int>& offsets,const std::vector<int>& indices);
  // Output is column-major augmented system, or one squared error (without 1/2).
  void evaluate(const std::vector<BatchPose>& poses, double* output, bool cost);
  void configureResident(const std::vector<FrozenSystem>& frozen,const std::vector<std::vector<int>>& auxiliary_poses);
  void residentLinearize(const std::vector<BatchPose>& poses,const std::vector<double>& deltas,const std::vector<double>& auxiliary);
  double residentError(const std::vector<BatchPose>& poses,const std::vector<double>& deltas);
  bool residentSolve(double lambda,bool diagonal,double minimum,double maximum,std::vector<double>& delta,double& old_error,double& new_error);
  void residentHessian(double* output);
 private:
  struct Impl; std::unique_ptr<Impl> impl_;
};
}
