// MIT License; see form/feature/factor.hpp for copyright and license text.
#include "form/optimization/resident_optimizer.hpp"

#include "form/feature/batch_summary.hpp"
#include "form/feature/factor.hpp"
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/internal/LevenbergMarquardtState.h>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <stdexcept>
#include <typeinfo>

namespace form {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Scatter a complete local augmented block, retaining its constant term.
void addBlock(Matrix& out, const Matrix& block, const std::vector<int>& poses) {
  const int n = out.rows()-1, m = 6*poses.size();
  for (int j = 0; j <= m; ++j) {
    const int c = j == m ? n : 6*poses[j/6]+j%6;
    for (int i = 0; i <= m; ++i) {
      const int r = i == m ? n : 6*poses[i/6]+i%6;
      out(r,c) += block(i,j);
    }
  }
}
}  // namespace

struct ResidentOptimizer::Impl {
  explicit Impl(bool cuda) : gpu(cuda) {}
  bool gpu, ready = false, linearized = false;
  gtsam::KeyVector keys;
  std::map<gtsam::Key, int> index;
  std::unique_ptr<BatchSummary> batch;
  std::vector<FrozenSystem> frozen;
  std::vector<std::vector<gtsam::Pose3>> anchors;
  struct Auxiliary {
    gtsam::NonlinearFactor::shared_ptr factor;
    std::vector<int> poses;
  };
  std::vector<Auxiliary> auxiliary;
  std::vector<gtsam::Pose3> poses;
  std::vector<double> displacements, auxiliary_values;
  Matrix hessian, damped, frozen_information;
  Vector candidate;
  Eigen::LLT<Matrix, Eigen::Upper> llt;

  void pack(const gtsam::Values& values) {
    diagnostics::Scope detail(diagnostics::Stage::pose_pack);
    if (!ready) throw std::logic_error("ResidentOptimizer requires reset");
    if (values.size() != keys.size()) throw std::invalid_argument("ResidentOptimizer value keys changed; reset required");
    poses.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) poses[i] = values.at<gtsam::Pose3>(keys[i]);
    displacements.clear();
    for (size_t f = 0; f < frozen.size(); ++f) {
      const auto& system = frozen[f];
      for (size_t j = 0; j < system.pose_indices.size(); ++j) {
        const gtsam::Vector6 d = system.has_anchor ? anchors[f][j].localCoordinates(poses[system.pose_indices[j]]) : gtsam::Vector6::Zero().eval();
        displacements.insert(displacements.end(), d.data(), d.data()+6);
      }
    }
  }
};

ResidentOptimizer::ResidentOptimizer(bool gpu) : impl_(std::make_unique<Impl>(gpu)) {
#ifndef FORM_ENABLE_CUDA
  if (gpu) throw std::runtime_error("CUDA resident optimizer requested in CPU-only build");
#endif
}
ResidentOptimizer::~ResidentOptimizer() = default;
const gtsam::KeyVector& ResidentOptimizer::keys() const { return impl_->keys; }

void ResidentOptimizer::reset(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values) {
  diagnostics::Scope detail(impl_->gpu ? diagnostics::Stage::gpu_reset : diagnostics::Stage::cpu_reset);
  profile::Scope timer(profile::resident_reset_wall);
  auto& s = *impl_;
  s.ready = s.linearized = false;
  s.keys = values.keys();
  if (s.keys.size() > 1000) throw std::invalid_argument("ResidentOptimizer supports at most 1000 poses");
  std::sort(s.keys.begin(), s.keys.end());
  s.index.clear(); s.frozen.clear(); s.anchors.clear(); s.auxiliary.clear();
  for (size_t i = 0; i < s.keys.size(); ++i) {
    if (!dynamic_cast<const gtsam::GenericValue<gtsam::Pose3>*>(&values.at(s.keys[i])))
      throw std::invalid_argument("ResidentOptimizer only supports Pose3 values");
    s.index.emplace(s.keys[i], i);
  }
  std::vector<SummaryEdge> edges;
  std::vector<std::vector<int>> auxiliary_poses;
  { diagnostics::Scope detail(diagnostics::Stage::reset_classify);
  for (const auto& factor : graph) {
    if (!factor) continue;
    if (typeid(*factor) == typeid(FeatureFactor)) {
      const auto& feature = static_cast<const FeatureFactor&>(*factor);
      if (feature.summary()) {
        edges.push_back({s.index.at(feature.key1()), s.index.at(feature.key2()), feature.summary(), feature.inverseVariance()});
        continue;
      }
    }
    if (typeid(*factor) == typeid(gtsam::LinearContainerFactor)) {
      const auto& container = static_cast<const gtsam::LinearContainerFactor&>(*factor);
      if (container.factor() && typeid(*container.factor()) == typeid(gtsam::HessianFactor)) {
        FrozenSystem system;
        system.has_anchor = container.hasLinearizationPoint();
        std::vector<gtsam::Pose3> anchors;
        for (auto key = container.factor()->begin(); key != container.factor()->end(); ++key) {
          if (container.factor()->getDim(key) != 6) throw std::invalid_argument("Resident frozen factor requires six-dimensional pose blocks");
          system.pose_indices.push_back(s.index.at(*key));
          if (system.has_anchor) anchors.push_back(container.linearizationPoint()->at<gtsam::Pose3>(*key));
        }
        const Matrix augmented = container.factor()->augmentedInformation();
        system.augmented.assign(augmented.data(), augmented.data()+augmented.size());
        s.frozen.push_back(std::move(system)); s.anchors.push_back(std::move(anchors));
        continue;
      }
    }
    // Inactive derived factors can mention absent keys. Their virtual linearize
    // may return null; if they activate later, the missing key fails explicitly.
    std::vector<int> indices;
    for (auto key : factor->keys()) {
      auto found = s.index.find(key);
      if (found != s.index.end()) indices.push_back(found->second);
    }
    s.auxiliary.push_back({factor, indices});
    auxiliary_poses.push_back(std::move(indices));
  }
  }
  if (s.keys.empty()) {
    s.batch.reset();
  } else {
    if (s.batch) s.batch->reset(s.keys.size(), std::move(edges));
    else s.batch = std::make_unique<BatchSummary>(s.keys.size(), std::move(edges), s.gpu);
#ifdef FORM_ENABLE_CUDA
    if (s.gpu) s.batch->configureResident(s.frozen, auxiliary_poses);
#endif
  }
  if (!s.gpu) {
    diagnostics::Scope detail(diagnostics::Stage::cpu_frozen_setup);
    s.frozen_information = Matrix::Zero(6*s.keys.size()+1, 6*s.keys.size()+1);
    for (const auto& frozen : s.frozen) {
      const int n = 6*frozen.pose_indices.size();
      Matrix information = Eigen::Map<const Matrix>(frozen.augmented.data(), n+1, n+1);
      information.row(n).setZero(); information.col(n).setZero();
      addBlock(s.frozen_information, information, frozen.pose_indices);
    }
  }
  s.ready = true;
}

void ResidentOptimizer::linearize(const gtsam::Values& values) {
  diagnostics::Scope detail(impl_->gpu ? diagnostics::Stage::gpu_linearize : diagnostics::Stage::cpu_linearize);
  profile::Scope timer(profile::linearize_wall);
  profile::iterations.fetch_add(1, std::memory_order_relaxed);
  auto& s = *impl_;
  s.linearized = false;
  s.pack(values);
  s.auxiliary_values.clear();
  if (!s.gpu || !s.batch) s.hessian = s.batch ? s.batch->linearize(s.poses) : Matrix::Zero(1,1).eval();
  { diagnostics::Scope detail(diagnostics::Stage::auxiliary_linearize);
  for (const auto& auxiliary : s.auxiliary) {
    const int dim = 6*auxiliary.poses.size()+1;
    Matrix augmented = Matrix::Zero(dim, dim);
    const auto linear = auxiliary.factor->linearize(values);
    if (linear) {
      const auto jacobian = boost::dynamic_pointer_cast<const gtsam::JacobianFactor>(linear);
      if (jacobian && jacobian->get_model() && jacobian->get_model()->isConstrained())
        throw std::invalid_argument("ResidentOptimizer does not support constrained Jacobian factors");
      std::vector<int> local;
      for (auto key = linear->begin(); key != linear->end(); ++key) {
        if (linear->getDim(key) != 6) throw std::invalid_argument("Resident auxiliary factor requires six-dimensional pose blocks");
        const int global = s.index.at(*key);
        auto where = std::find(auxiliary.poses.begin(), auxiliary.poses.end(), global);
        if (where == auxiliary.poses.end()) throw std::invalid_argument("Resident linear factor keys differ from nonlinear factor");
        local.push_back(where-auxiliary.poses.begin());
      }
      addBlock(augmented, linear->augmentedInformation(), local);
    }
    if (s.gpu && s.batch) s.auxiliary_values.insert(s.auxiliary_values.end(), augmented.data(), augmented.data()+augmented.size());
    else addBlock(s.hessian, augmented, auxiliary.poses);
  }
  }
  if (s.gpu && s.batch) {
#ifdef FORM_ENABLE_CUDA
    s.batch->residentLinearize(s.poses, s.displacements, s.auxiliary_values);
#endif
  } else {
    diagnostics::Scope detail(diagnostics::Stage::cpu_frozen_shift);
    if (!s.gpu) s.hessian += s.frozen_information;
    const int n = 6*s.keys.size();
    size_t offset = 0;
    for (const auto& frozen : s.frozen) {
      const int m = 6*frozen.pose_indices.size();
      const Eigen::Map<const Matrix> h(frozen.augmented.data(), m+1, m+1);
      const Eigen::Map<const Vector> d(s.displacements.data()+offset, m);
      const Vector gd = h.topLeftCorner(m,m)*d;
      s.hessian(n,n) += h(m,m)+d.dot(gd)-2*d.dot(h.col(m).head(m));
      for (size_t j = 0; j < frozen.pose_indices.size(); ++j)
        s.hessian.col(n).segment<6>(6*frozen.pose_indices[j]) += h.col(m).segment<6>(6*j)-gd.segment<6>(6*j);
      offset += m;
    }
    s.hessian.row(n).head(n) = s.hessian.col(n).head(n).transpose();
  }
  s.linearized = true;
}

Matrix ResidentOptimizer::model() {
  auto& s = *impl_;
  if (!s.linearized) throw std::logic_error("ResidentOptimizer requires linearize before model");
#ifdef FORM_ENABLE_CUDA
  if (s.gpu && s.batch) return s.batch->residentHessian();
#endif
  return s.hessian;
}

double ResidentOptimizer::error(const gtsam::Values& values) {
  diagnostics::Scope detail(impl_->gpu ? diagnostics::Stage::gpu_error : diagnostics::Stage::cpu_error);
  profile::Scope timer(profile::resident_error_wall);
  auto& s = *impl_;
  s.pack(values);
  double cost = 0;
#ifdef FORM_ENABLE_CUDA
  if (s.gpu && s.batch) cost = s.batch->residentError(s.poses, s.displacements);
  else
#endif
  {
    if (s.batch) cost = s.batch->error(s.poses);
    size_t offset = 0;
    for (const auto& frozen : s.frozen) {
      const int m = 6*frozen.pose_indices.size();
      if (frozen.has_anchor) {
        const Eigen::Map<const Matrix> h(frozen.augmented.data(), m+1, m+1);
        const Eigen::Map<const Vector> d(s.displacements.data()+offset, m);
        cost += .5*(h(m,m)+d.dot(h.topLeftCorner(m,m)*d)-2*d.dot(h.col(m).head(m)));
      }
      offset += m;
    }
  }
  { diagnostics::Scope detail(diagnostics::Stage::auxiliary_error);
    for (const auto& auxiliary : s.auxiliary) cost += auxiliary.factor->error(values); }
  return cost;
}

bool ResidentOptimizer::solve(double lambda, bool diagonal, double min_diagonal, double max_diagonal,
                              Vector& delta, double& old_linear_error, double& new_linear_error) {
  diagnostics::Scope detail(impl_->gpu ? diagnostics::Stage::gpu_solve : diagnostics::Stage::cpu_solve);
  auto& s = *impl_;
  if (!s.linearized) throw std::logic_error("ResidentOptimizer requires linearize before solve");
  if (!std::isfinite(lambda) || lambda < 0 ||
      (diagonal && (!std::isfinite(min_diagonal) || !std::isfinite(max_diagonal) ||
                    min_diagonal <= 0 || max_diagonal < min_diagonal)))
    throw std::invalid_argument("Invalid resident damping parameters");
  profile::Scope timer(profile::solve);
#ifdef FORM_ENABLE_CUDA
  if (s.gpu && s.batch) {
    profile::cuda_solve_calls.fetch_add(1, std::memory_order_relaxed);
    return s.batch->residentSolve(lambda, diagonal, min_diagonal, max_diagonal,
                                 delta, old_linear_error, new_linear_error);
  }
#endif
  const int n = s.hessian.rows()-1;
  { diagnostics::Scope detail(diagnostics::Stage::cpu_damping);
  s.damped = s.hessian.topLeftCorner(n,n);
  const double inverse_sigma = 1.0/(1.0/std::sqrt(lambda));
  for (int k = 0; k < n; ++k) {
    const double a = diagonal ? inverse_sigma*std::sqrt(std::clamp(s.hessian(k,k), min_diagonal, max_diagonal)) : inverse_sigma;
    s.damped(k,k) += a*a;
  }
  }
  { diagnostics::Scope detail(diagnostics::Stage::cpu_cholesky); s.llt.compute(s.damped); }
  if (s.llt.info() != Eigen::Success) return false;
  { diagnostics::Scope detail(diagnostics::Stage::cpu_backsolve);
    s.candidate = s.llt.solve(s.hessian.col(n).head(n)); }
  if (!s.candidate.allFinite()) return false;
  diagnostics::Scope model_detail(diagnostics::Stage::cpu_model_error);
  const double old_error = .5*s.hessian(n,n);
  const double new_error = old_error+.5*(s.candidate.dot(s.hessian.topLeftCorner(n,n)*s.candidate)-2*s.candidate.dot(s.hessian.col(n).head(n)));
  if (!std::isfinite(old_error) || !std::isfinite(new_error)) return false;
  delta = s.candidate; old_linear_error = old_error; new_linear_error = new_error;
  return true;
}

ResidentResult ResidentOptimizer::optimize(const gtsam::Values& initial,
                                          const gtsam::LevenbergMarquardtParams& params) {
  // The supported numerical solve is dense Cholesky. Custom elimination orders
  // have no effect on this sorted dense system, as in DenseLMOptimizer::solve.
  if ((params.linearSolverType != gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY &&
       params.linearSolverType != gtsam::NonlinearOptimizerParams::SEQUENTIAL_CHOLESKY) ||
      params.iterativeParams || !params.logFile.empty() ||
      params.verbosityLM != gtsam::LevenbergMarquardtParams::SILENT)
    throw std::invalid_argument("ResidentOptimizer supports Cholesky with silent LM logging only");
  if (!std::isfinite(params.lambdaInitial) || params.lambdaInitial <= 0 ||
      !std::isfinite(params.lambdaFactor) || params.lambdaFactor <= 1 ||
      !std::isfinite(params.lambdaUpperBound) || params.lambdaUpperBound <= 0 ||
      !std::isfinite(params.lambdaLowerBound) || params.lambdaLowerBound < 0 ||
      params.lambdaLowerBound > params.lambdaUpperBound ||
      !std::isfinite(params.minDiagonal) || params.minDiagonal <= 0 ||
      !std::isfinite(params.maxDiagonal) || params.maxDiagonal < params.minDiagonal)
    throw std::invalid_argument("ResidentOptimizer requires finite positive damping parameters");
  using State = gtsam::internal::LevenbergMarquardtState;
  ResidentResult result;
  result.initial_error = error(initial);
  auto state = std::make_unique<State>(initial, result.initial_error, params.lambdaInitial, params.lambdaFactor);
  if (!(state->error <= params.errorTol) && state->iterations < params.maxIterations) {
    double previous_error;
    do {
      previous_error = state->error;
      linearize(state->values);
      bool stop;
      do {
        stop = false;
        bool accepted = false;
        double fidelity = 0, new_error = std::numeric_limits<double>::infinity();
        gtsam::Values trial;
        Vector step;
        double old_linear, new_linear;
        if (solve(state->lambda, params.diagonalDamping, params.minDiagonal, params.maxDiagonal,
                  step, old_linear, new_linear)) {
          const double linear_change = old_linear-new_linear;
          if (linear_change >= 0) {
            { diagnostics::Scope detail(diagnostics::Stage::retraction);
            gtsam::VectorValues delta;
            for (size_t k = 0; k < impl_->keys.size(); ++k) delta.insert(impl_->keys[k], step.segment<6>(6*k));
            trial = state->values.retract(delta); }
            new_error = error(trial);
            const double cost_change = state->error-new_error;
            if (linear_change > std::numeric_limits<double>::epsilon()*old_linear) {
              fidelity = cost_change/linear_change;
              accepted = fidelity > params.minModelFidelity;
            }
            stop = std::abs(cost_change) < params.relativeErrorTol*state->error;
          }
        }
        else diagnostics::tick(diagnostics::Stage::lm_solve_failed);
        diagnostics::tick(accepted ? diagnostics::Stage::lm_accepted : diagnostics::Stage::lm_rejected);
        if (accepted) {
          state = state->decreaseLambda(params, fidelity, std::move(trial), new_error);
          stop = true;
        } else if (!stop) {
          state->increaseLambda(params);
          stop = state->lambda >= params.lambdaUpperBound;
        }
      } while (!stop);
      if (params.iterationHook) params.iterationHook(state->iterations, previous_error, state->error);
    } while (state->iterations < params.maxIterations &&
             !gtsam::checkConvergence(params, previous_error, state->error) && std::isfinite(previous_error));
  }
  result.values = state->values;
  result.final_error = state->error;
  result.lambda = state->lambda;
  result.iterations = state->iterations;
  result.inner_iterations = state->totalNumberInnerIterations;
  return result;
}
}  // namespace form
