#include <gtest/gtest.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/base/numericalDerivative.h>
#include <tbb/global_control.h>
#include "form/feature/factor.hpp"
#include "form/optimization/resident_optimizer.hpp"

namespace {
gtsam::NonlinearFactorGraph fixture(gtsam::Values& values) {
  values.insert(7, gtsam::Pose3(gtsam::Rot3::RzRyRx(.1, -.2, .3), {1, 2, 3}));
  values.insert(42, gtsam::Pose3(gtsam::Rot3::RzRyRx(-.2, .1, -.1), {2, 1, 3}));
  auto points = std::make_shared<form::PointPoint>();
  auto planes = std::make_shared<form::PlanePoint>();
  for (int k = 0; k < 16; ++k) {
    Eigen::Vector3d p(.2*k, .1*k*k, (k%3)-1.), q = p + Eigen::Vector3d(.2, -.1, .05);
    points->p_i.insert(points->p_i.end(), p.data(), p.data()+3);
    points->p_j.insert(points->p_j.end(), q.data(), q.data()+3);
  }
  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<form::FeatureFactor>(42, 7, std::make_tuple(planes, points), .7, true);
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(7, gtsam::Pose3{}, gtsam::noiseModel::Isotropic::Sigma(6, .5));
  return graph;
}

void expectModel(form::ResidentOptimizer& workspace, const gtsam::NonlinearFactorGraph& graph,
                 const gtsam::Values& values) {
  workspace.linearize(values);
  auto expected = graph.linearize(values)->augmentedHessian();
  EXPECT_TRUE(workspace.model().isApprox(expected, 2e-11));
  EXPECT_NEAR(workspace.error(values), graph.error(values), 2e-11*std::max(1., graph.error(values)));
}
}  // namespace

TEST(ResidentOptimizer, MixedModelFrozenShiftsAndDampedSteps) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    gtsam::Values values;
    auto graph = fixture(values);
    const auto raw = boost::dynamic_pointer_cast<form::FeatureFactor>(graph[0]);
    graph.emplace_shared<form::FeatureFactor>(7, 42, std::make_tuple(raw->plane_point, raw->point_point), 1.1, false);
    auto frozen = graph[0]->linearize(values);
    graph.emplace_shared<gtsam::LinearContainerFactor>(frozen, values);
    graph.emplace_shared<gtsam::LinearContainerFactor>(frozen);
    form::ResidentOptimizer workspace(gpu);
    workspace.reset(graph, values);
    EXPECT_EQ(workspace.keys(), (gtsam::KeyVector{7, 42}));
    for (int k = 0; k < 3; ++k) {
      values.update(42, values.at<gtsam::Pose3>(42).retract((gtsam::Vector6() << .01, -.02, .03, .1, -.1, .2).finished()));
      expectModel(workspace, graph, values);
      const auto linear = graph.linearize(values);
      for (bool diagonal : {false, true}) for (double lambda : {1e-5, .1, 1e3}) {
        gtsam::LevenbergMarquardtParams p;
        p.lambdaInitial = lambda; p.diagonalDamping = diagonal;
        p.minDiagonal = .3; p.maxDiagonal = 10;
        form::DenseLMOptimizer reference(graph, values, p);
        auto sqrtDiagonal = linear->hessianDiagonal();
        for (auto& entry : sqrtDiagonal) entry.second = entry.second.cwiseMax(p.minDiagonal).cwiseMin(p.maxDiagonal).cwiseSqrt();
        auto expected = reference.solve(reference.buildDampedSystem(*linear, sqrtDiagonal), p);
        Eigen::VectorXd delta;
        double oldError, newError;
        const auto calls = form::profile::cuda_solve_calls.load();
        ASSERT_TRUE(workspace.solve(lambda, diagonal, p.minDiagonal, p.maxDiagonal, delta, oldError, newError));
        EXPECT_EQ(form::profile::cuda_solve_calls.load(), calls+(gpu ? 1 : 0));
        EXPECT_TRUE(delta.isApprox(expected.vector(workspace.keys()), 2e-9));
        EXPECT_NEAR(oldError, linear->error(gtsam::VectorValues::Zero(expected)), 1e-8);
        EXPECT_NEAR(newError, linear->error(expected), 1e-8);
      }
    }
  }
}

namespace {
class SquaredTranslation : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  explicit SquaredTranslation(gtsam::Key key)
      : NoiseModelFactor1(gtsam::noiseModel::Isotropic::Sigma(6,1), key) {}
  gtsam::Vector evaluateError(const gtsam::Pose3& pose, boost::optional<gtsam::Matrix&> H = boost::none) const override {
    const std::function<gtsam::Vector(const gtsam::Pose3&)> residual = [](const gtsam::Pose3& p) {
      gtsam::Vector r = gtsam::Pose3::Logmap(p);
      r(3) = p.x()*p.x()-1;
      return r;
    };
    if (H) *H = gtsam::numericalDerivative11<gtsam::Vector,gtsam::Pose3>(residual, pose);
    return residual(pose);
  }
};
}

TEST(ResidentOptimizer, LMMatchesDenseWithRejectedStepsAndBothPolicies) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    for (bool diagonal : {false, true}) for (bool fixed : {false, true}) for (bool rejection : {false, true}) for (size_t limit : {3, 100}) {
      SCOPED_TRACE(::testing::Message() << "gpu=" << gpu << " diagonal=" << diagonal << " fixed=" << fixed << " rejection=" << rejection << " limit=" << limit);
      gtsam::Values initial;
      gtsam::NonlinearFactorGraph graph;
      if (rejection) {
        initial.insert(91, gtsam::Pose3(gtsam::Rot3{}, {.1, 0, 0}));
        graph.emplace_shared<SquaredTranslation>(91);
      } else {
        graph = fixture(initial);
        graph.emplace_shared<gtsam::LinearContainerFactor>(graph[0]->linearize(initial), initial);
      }
      gtsam::LevenbergMarquardtParams params;
      params.diagonalDamping = diagonal; params.useFixedLambdaFactor = fixed;
      params.lambdaInitial = 1e-5; params.lambdaFactor = fixed ? 10 : 2;
      params.absoluteErrorTol = 1e-8; params.relativeErrorTol = 1e-8;
      params.maxIterations = limit;
      std::vector<double> expectedErrors, actualErrors;
      std::vector<size_t> expectedIterations, actualIterations;
      params.iterationHook = [&](size_t i, double, double after) { expectedIterations.push_back(i); expectedErrors.push_back(after); };
      form::DenseLMOptimizer reference(graph, initial, params);
      const auto expected = reference.optimize();
      params.iterationHook = [&](size_t i, double, double after) { actualIterations.push_back(i); actualErrors.push_back(after); };
      form::ResidentOptimizer optimizer(gpu);
      optimizer.reset(graph, initial);
      const auto before = form::profile::iterations.load();
      const auto actual = optimizer.optimize(initial, params);
      EXPECT_TRUE(actual.values.equals(expected, 2e-7));
      EXPECT_NEAR(actual.initial_error, graph.error(initial), 1e-8);
      EXPECT_NEAR(actual.final_error, reference.error(), 1e-8);
      EXPECT_EQ(actual.iterations, reference.iterations());
      EXPECT_EQ(actual.inner_iterations, reference.getInnerIterations());
      // Compare adaptive lambda before the frozen approximation stalls: at its
      // nonzero cost floor, fidelity divides two cancellation-sized reductions.
      // The full run still checks every accepted error, state, and retry count.
      if (fixed || rejection || limit == 3)
        EXPECT_NEAR(actual.lambda, reference.lambda(), 1e-12*std::max(1., reference.lambda()));
      EXPECT_EQ(actualIterations, expectedIterations);
      ASSERT_EQ(actualErrors.size(), expectedErrors.size());
      for (size_t k = 0; k < actualErrors.size(); ++k) EXPECT_NEAR(actualErrors[k], expectedErrors[k], 1e-8);
      EXPECT_EQ(form::profile::iterations.load()-before, actualErrors.size());
      if (rejection) EXPECT_GT(actual.inner_iterations, actual.iterations);
    }
  }
}

TEST(ResidentOptimizer, PreservesSubtypeVirtualBehaviorAndInactiveMissingKeys) {
  class Inactive final : public form::FeatureFactor {
   public:
    using FeatureFactor::FeatureFactor;
    bool active(const gtsam::Values&) const override { return false; }
  };
  class DerivedContainer final : public gtsam::LinearContainerFactor {
   public:
    using LinearContainerFactor::LinearContainerFactor;
    double error(const gtsam::Values&) const override { return 13.; }
    gtsam::GaussianFactor::shared_ptr linearize(const gtsam::Values&) const override { return {}; }
  };
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    gtsam::Values values;
    auto graph = fixture(values);
    const auto feature = boost::dynamic_pointer_cast<form::FeatureFactor>(graph[0]);
    graph.emplace_shared<Inactive>(101, 200, std::make_tuple(feature->plane_point, feature->point_point), 1., true);
    graph.emplace_shared<DerivedContainer>(graph[0]->linearize(values), values);
    form::ResidentOptimizer optimizer(gpu);
    optimizer.reset(graph, values);
    expectModel(optimizer, graph, values);
  }
}

TEST(ResidentOptimizer, ResetEmptyGraphAndNumericalFailure) {
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    gtsam::Values values;
    auto graph = fixture(values);
    form::ResidentOptimizer optimizer(gpu);
    optimizer.reset(graph, values);
    expectModel(optimizer, graph, values);
    gtsam::Values single;
    single.insert(91, gtsam::Pose3{});
    gtsam::NonlinearFactorGraph indefinite;
    indefinite.emplace_shared<gtsam::LinearContainerFactor>(gtsam::HessianFactor(91, -gtsam::Matrix6::Identity(), gtsam::Vector6::Ones(), 4.), single);
    optimizer.reset(indefinite, single);
    expectModel(optimizer, indefinite, single);
    Eigen::VectorXd delta;
    double oldError, newError;
    delta = Eigen::VectorXd::Constant(2, 17.);
    EXPECT_THROW(optimizer.solve(-1, false, 1e-6, 1e32, delta, oldError, newError), std::invalid_argument);
    EXPECT_THROW(optimizer.solve(1, true, 0, 1e32, delta, oldError, newError), std::invalid_argument);
    EXPECT_FALSE(optimizer.solve(.1, false, 1e-6, 1e32, delta, oldError, newError));
    EXPECT_EQ(delta.size(), 2);
    EXPECT_TRUE(delta.isConstant(17.));
    EXPECT_TRUE(optimizer.solve(2, false, 1e-6, 1e32, delta, oldError, newError));
    EXPECT_TRUE(delta.isApprox(gtsam::Vector6::Ones()));
    gtsam::LevenbergMarquardtParams retry;
    retry.lambdaInitial = .1; retry.lambdaFactor = 10; retry.maxIterations = 1;
    const auto recovered = optimizer.optimize(single, retry);
    EXPECT_EQ(recovered.iterations, 1);
    EXPECT_EQ(recovered.inner_iterations, 3);
    EXPECT_LT(recovered.final_error, recovered.initial_error);
    EXPECT_NEAR(recovered.values.at<gtsam::Pose3>(91).localCoordinates(single.at<gtsam::Pose3>(91)).norm(), std::sqrt(6.)/9., 1e-10);
    optimizer.reset({}, {});
    EXPECT_EQ(optimizer.error({}), 0);
    optimizer.linearize({});
    ASSERT_EQ(optimizer.model().rows(), 1);
    EXPECT_EQ(optimizer.model()(0,0), 0);
    EXPECT_TRUE(optimizer.solve(1, false, 1e-6, 1e32, delta, oldError, newError));
    EXPECT_EQ(delta.size(), 0);
    const auto result = optimizer.optimize({}, {});
    EXPECT_EQ(result.iterations, 0);
    optimizer.reset(graph, values);
    expectModel(optimizer, graph, values);
  }
}

TEST(ResidentOptimizer, RejectsUnsupportedValuesParametersAndStaleModels) {
#ifndef FORM_ENABLE_CUDA
  EXPECT_THROW(form::ResidentOptimizer(true), std::runtime_error);
#endif
  form::ResidentOptimizer optimizer;
  EXPECT_THROW(optimizer.model(), std::logic_error);
  gtsam::Values values;
  values.insert(7, 1.);
  EXPECT_THROW(optimizer.reset({}, values), std::invalid_argument);
  values.clear();
  auto graph = fixture(values);
  optimizer.reset(graph, values);
  gtsam::LevenbergMarquardtParams p;
  p.linearSolverType = gtsam::NonlinearOptimizerParams::Iterative;
  EXPECT_THROW(optimizer.optimize(values, p), std::invalid_argument);
  p = {}; p.logFile = "unsupported.csv";
  EXPECT_THROW(optimizer.optimize(values, p), std::invalid_argument);
  p = {}; p.lambdaFactor = 1;
  EXPECT_THROW(optimizer.optimize(values, p), std::invalid_argument);
  p = {}; p.maxIterations = 0;
  EXPECT_EQ(optimizer.optimize(values, p).iterations, 0);
  optimizer.linearize(values);
  optimizer.reset(graph, values);
  EXPECT_THROW(optimizer.model(), std::logic_error);
}

TEST(ResidentOptimizer, RejectionUpperBoundPreservesStateAndCounts) {
  for (bool fixed : {false, true}) {
    gtsam::Values initial;
    initial.insert(91, gtsam::Pose3(gtsam::Rot3{}, {.1, 0, 0}));
    gtsam::NonlinearFactorGraph graph;
    graph.emplace_shared<SquaredTranslation>(91);
    gtsam::LevenbergMarquardtParams params;
    params.lambdaUpperBound = .001; params.useFixedLambdaFactor = fixed;
    form::DenseLMOptimizer reference(graph, initial, params);
    reference.optimize();
    form::ResidentOptimizer optimizer;
    optimizer.reset(graph, initial);
    const auto actual = optimizer.optimize(initial, params);
    EXPECT_TRUE(actual.values.equals(initial));
    EXPECT_EQ(actual.iterations, 0);
    EXPECT_EQ(actual.inner_iterations, reference.getInnerIterations());
    EXPECT_EQ(actual.lambda, reference.lambda());
  }
}

TEST(ResidentOptimizer, RejectsConstrainedJacobianFactorsIncludingSubtypes) {
  class DerivedJacobian final : public gtsam::JacobianFactor {
   public:
    explicit DerivedJacobian(const gtsam::JacobianFactor& factor) : JacobianFactor(factor) {}
  };
  class DerivedPrior final : public gtsam::PriorFactor<gtsam::Pose3> {
   public:
    using PriorFactor::PriorFactor;
    gtsam::GaussianFactor::shared_ptr linearize(const gtsam::Values& values) const override {
      const auto linear = PriorFactor::linearize(values);
      return boost::make_shared<DerivedJacobian>(*boost::dynamic_pointer_cast<gtsam::JacobianFactor>(linear));
    }
  };
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    for (bool derived : {false, true}) {
      gtsam::Values values;
      values.insert(7, gtsam::Pose3(gtsam::Rot3{}, {1, 0, 0}));
      gtsam::NonlinearFactorGraph graph;
      const auto constrained = gtsam::noiseModel::Constrained::All(6);
      if (derived) graph.emplace_shared<DerivedPrior>(7, gtsam::Pose3{}, constrained);
      else graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(7, gtsam::Pose3{}, constrained);
      EXPECT_THROW(graph.linearize(values)->optimizeDensely(), std::invalid_argument);
      form::ResidentOptimizer optimizer(gpu);
      optimizer.reset(graph, values);
      EXPECT_THROW(optimizer.linearize(values), std::invalid_argument);
      EXPECT_THROW(optimizer.model(), std::logic_error);
      EXPECT_THROW(optimizer.optimize(values, {}), std::invalid_argument);
    }
  }
}

namespace {
struct ReuseDiagnostics {
  bool previous = form::diagnostics::enabled;
  ReuseDiagnostics() { form::diagnostics::enabled = true; }
  ~ReuseDiagnostics() { form::diagnostics::enabled = previous; }
  uint64_t rebuilds(bool gpu) const {
    using namespace form::diagnostics;
    return calls[static_cast<size_t>(gpu ? Stage::cuda_configure_host : Stage::cpu_frozen_setup)].load();
  }
};
}

TEST(ResidentOptimizer, ReusesConstantsAndRefreshesAnchorsRootsAndAuxiliaryFactors) {
  ReuseDiagnostics counters;
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    SCOPED_TRACE(gpu);
    gtsam::Values values;
    auto graph = fixture(values);
    auto frozen = graph[0]->linearize(values);
    graph.emplace_shared<gtsam::LinearContainerFactor>(frozen, values);
    form::ResidentOptimizer workspace(gpu);
    workspace.reset(graph, values);
    expectModel(workspace, graph, values);
    for (int trial = 0; trial < 4; ++trial) {
      // Same constant Hessian; fresh anchors and auxiliary objects must still apply.
      auto anchor = values;
      anchor.update(7, values.at<gtsam::Pose3>(7).retract(gtsam::Vector6::Constant(.01*trial)));
      graph[2] = boost::make_shared<gtsam::LinearContainerFactor>(frozen, anchor);
      graph[1] = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(7,
          gtsam::Pose3(gtsam::Rot3{}, {.1*trial, 0, 0}), gtsam::noiseModel::Isotropic::Sigma(6, .5));
      const auto original = boost::dynamic_pointer_cast<form::FeatureFactor>(graph[0]);
      // Change summary weights and endpoint order, retaining the frozen graph.
      graph[0] = boost::make_shared<form::FeatureFactor>(trial%2 ? 7 : 42, trial%2 ? 42 : 7,
          std::make_tuple(original->plane_point, original->point_point), .4+.1*trial, true);
      const auto before = counters.rebuilds(gpu);
      workspace.reset(graph, values);
      EXPECT_EQ(counters.rebuilds(gpu), before) << "unchanged constants rebuilt";
      EXPECT_THROW(workspace.model(), std::logic_error);
      expectModel(workspace, graph, values);
      form::ResidentOptimizer fresh(gpu);
      fresh.reset(graph, values);
      gtsam::LevenbergMarquardtParams params; params.maxIterations = 3;
      const auto expected = fresh.optimize(values, params);
      const auto actual = workspace.optimize(values, params);
      EXPECT_TRUE(actual.values.equals(expected.values, 1e-12));
      EXPECT_EQ(actual.iterations, expected.iterations);
      EXPECT_EQ(actual.inner_iterations, expected.inner_iterations);
      EXPECT_DOUBLE_EQ(actual.lambda, expected.lambda);
    }
  }
}

TEST(ResidentOptimizer, InvalidatesConstantsForSameSizeChangesAndRecoversAfterFailure) {
  ReuseDiagnostics counters;
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    form::ResidentOptimizer workspace(gpu);
    gtsam::Values values;
    values.insert(7, gtsam::Pose3{}); values.insert(42, gtsam::Pose3{});
    auto make = [](gtsam::Key key, double weight, bool anchored, const gtsam::Values& anchors) {
      gtsam::NonlinearFactorGraph graph;
      gtsam::HessianFactor h(key, weight*gtsam::Matrix6::Identity(), gtsam::Vector6::Ones(), 20.);
      if (anchored) graph.emplace_shared<gtsam::LinearContainerFactor>(h, anchors);
      else graph.emplace_shared<gtsam::LinearContainerFactor>(h);
      for (auto global_key : anchors.keys())
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(global_key, gtsam::Pose3{},
            gtsam::noiseModel::Isotropic::Sigma(6, 1.));
      return graph;
    };
    auto graph = make(7, 2., true, values);
    workspace.reset(graph, values);
    // Same pointer, changed matrix: pointer identity cannot authorize reuse.
    auto container = boost::dynamic_pointer_cast<gtsam::LinearContainerFactor>(graph[0]);
    auto matrix = boost::dynamic_pointer_cast<gtsam::HessianFactor>(container->factor());
    *matrix = gtsam::HessianFactor(7, 3.*gtsam::Matrix6::Identity(), gtsam::Vector6::Ones(), 20.);
    auto before = counters.rebuilds(gpu);
    workspace.reset(graph, values);
    EXPECT_EQ(counters.rebuilds(gpu), before+1);
    expectModel(workspace, graph, values);
    for (auto change : {make(7, 4., true, values), make(42, 4., true, values), make(42, 4., false, values)}) {
      before = counters.rebuilds(gpu);
      workspace.reset(change, values);
      EXPECT_EQ(counters.rebuilds(gpu), before+1);
      expectModel(workspace, change, values);
    }
    // Same number of global poses but different keys (and shifted global mapping).
    auto changed = values; changed.erase(7); changed.insert(99, gtsam::Pose3{});
    auto remapped = make(42, 4., false, changed);
    before = counters.rebuilds(gpu);
    workspace.reset(remapped, changed);
    // CUDA constants already compare local indices; here 42 moves from slot 1 to 0.
    EXPECT_EQ(counters.rebuilds(gpu), before+1);
    expectModel(workspace, remapped, changed);
    auto invalid = remapped;
    invalid.emplace_shared<form::FeatureFactor>(7, 99,
        std::make_tuple(std::make_shared<form::PlanePoint>(),std::make_shared<form::PointPoint>()), 1., true);
    EXPECT_THROW(workspace.reset(invalid, changed), std::out_of_range);
    EXPECT_THROW(workspace.model(), std::logic_error);
    workspace.reset(remapped, changed);
    expectModel(workspace, remapped, changed);
    workspace.reset({}, {});
    EXPECT_EQ(workspace.error({}), 0.);
    workspace.reset(graph, values);
    expectModel(workspace, graph, values);
  }
}
