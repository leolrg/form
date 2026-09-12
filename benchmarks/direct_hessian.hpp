#pragma once
#include <array>
#include <memory>
#include <vector>
namespace direct_experiment {
struct MatrixInput { const double* values; int rows,cols; };
struct PoseData { double Ri[9],Rj[9],R[9],D[9],dt[3],t[3]; };
// Row-major 13x13 augmented Hessian per matrix, or one squared cost per matrix.
class Batch {
public:
  Batch(); ~Batch();
  void upload(const std::vector<MatrixInput>& matrices);
  std::vector<double> evaluate(const std::vector<PoseData>& poses,bool cost_only=false);
private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
