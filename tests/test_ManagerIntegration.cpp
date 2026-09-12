#include "form/optimization/constraints.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(DenseSolveCapture, ObservesSystemWithoutChangingSolution) {
  static Eigen::MatrixXd observed;
  observed.resize(0,0);
  struct Restore {
    bool enabled=form::profile::enabled;
    form::profile::SolveObserver observer=form::profile::solve_observer;
    ~Restore() { form::profile::enabled=enabled;form::profile::solve_observer=observer; }
  } restore;
  form::profile::enabled=true;
  form::profile::solve_observer=[](const Eigen::MatrixXd& a){observed=a;};
  const auto key=gtsam::symbol_shorthand::X(0);
  gtsam::Matrix h=2*gtsam::Matrix::Identity(6,6);
  h(0,1)=h(1,0)=.3;
  const gtsam::Vector rhs=gtsam::Vector::LinSpaced(6,-.5,.7);
  gtsam::GaussianFactorGraph graph;
  graph.push_back(gtsam::GaussianFactor::shared_ptr(new gtsam::HessianFactor(key,h,rhs,0.)));
  form::DenseLMOptimizer optimizer(gtsam::NonlinearFactorGraph{},gtsam::Values{},
                                   gtsam::LevenbergMarquardtParams{});
  const auto solution=optimizer.solve(graph,gtsam::LevenbergMarquardtParams{});
  ASSERT_EQ(observed.rows(),7);
  EXPECT_TRUE(observed.topLeftCorner(6,6).isApprox(h,1e-15));
  EXPECT_TRUE(observed.topRightCorner(6,1).isApprox(rhs,1e-15));
  EXPECT_TRUE(solution.at(key).isApprox(h.ldlt().solve(rhs),1e-14));
}
#ifdef FORM_ENABLE_CUDA
#include <cuda_runtime.h>

TEST(DenseCudaDispatch, PreservesFullSystemAndHonorsThresholdWithoutProfiling) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  struct Restore {
    bool enabled = form::profile::enabled;
    ~Restore() { form::profile::enabled = enabled; }
  } restore;
  form::profile::enabled = false;
  auto solver = std::make_shared<form::CudaDenseSolver>();
  for (const int n : {12, 258}) {
    gtsam::Matrix h = 2 * gtsam::Matrix::Identity(n, n);
    for (int i = 1; i < n; ++i) h(i-1,i) = h(i,i-1) = .3;
    const gtsam::Vector rhs = gtsam::Vector::LinSpaced(n, -.5, .7);
    gtsam::KeyVector keys;
    std::vector<gtsam::Matrix> blocks;
    std::vector<gtsam::Vector> pieces;
    for (int i = 0; i < n / 6; ++i) {
      // Reverse, nonconsecutive pose keys exercise scatter/reconstruction.
      keys.push_back(gtsam::symbol_shorthand::X(100 + 7 * (n / 6 - 1 - i)));
      pieces.emplace_back(rhs.segment(6 * i, 6));
      for (int j = i; j < n / 6; ++j) blocks.emplace_back(h.block(6*i, 6*j, 6, 6));
    }
    gtsam::GaussianFactorGraph graph;
    graph.push_back(gtsam::GaussianFactor::shared_ptr(new gtsam::HessianFactor(keys,blocks,pieces,0.)));
    const auto reference = graph.optimizeDensely();
    form::DenseLMOptimizer optimizer(gtsam::NonlinearFactorGraph{}, gtsam::Values{},
                                    gtsam::LevenbergMarquardtParams{}, solver, 240);
    form::profile::reset();
    const auto result = optimizer.solve(graph, gtsam::LevenbergMarquardtParams{});
    ASSERT_EQ(result.size(), keys.size());
    for (auto key : keys) EXPECT_TRUE(result.at(key).isApprox(reference.at(key), 1e-12));
    EXPECT_EQ(form::profile::cuda_solve_calls.load(), n >= 240 ? 1u : 0u);
    EXPECT_EQ(form::profile::cuda_solve_fallbacks.load(), 0u);
  }
}
#endif

namespace {
using form::ConstraintManager;
using gtsam::symbol_shorthand::X;

gtsam::Pose3 truth(size_t scan) {
  const double s = static_cast<double>(scan);
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(.025*s, -.018*s, .04*s),
                      gtsam::Point3(.35*s, -.12*s, .08*s));
}

gtsam::Pose3 initial(size_t scan) {
  if (scan == 0) return truth(scan);
  gtsam::Vector6 perturbation;
  perturbation << .018, -.012, .014, .08, -.05, .06;
  return truth(scan).retract(perturbation);
}

void fillCurrent(ConstraintManager &manager, size_t scan, int rematch = 0) {
  for (const auto &[older, pair] : manager.get_constraints(scan)) {
    auto planes = std::get<0>(pair);
    auto points = std::get<1>(pair);
    planes->clear();
    points->clear();
    for (int n = 0; n < 24; ++n) {
      const double a = .31*n + .09*rematch;
      const gtsam::Point3 world(2*std::sin(a), 1.4*std::cos(1.3*a),
                                .17*(n%7) - .5);
      const auto pi = truth(older).transformTo(world);
      const auto pj = truth(scan).transformTo(world);
      points->push_back(form::PointFeat(pi.x(), pi.y(), pi.z(), older),
                        form::PointFeat(pj.x(), pj.y(), pj.z(), scan));
      const gtsam::Vector3 normal = gtsam::Vector3(std::cos(a), std::sin(a), .6).normalized();
      const auto ni = truth(older).rotation().unrotate(normal);
      const auto nj = truth(scan).rotation().unrotate(normal);
      planes->push_back(form::PlanarFeat(pi.x(), pi.y(), pi.z(), ni.x(), ni.y(), ni.z(), older),
                        form::PlanarFeat(pj.x(), pj.y(), pj.z(), nj.x(), nj.y(), nj.z(), scan));
    }
  }
}

void expectValuesNear(const gtsam::Values &reference, const gtsam::Values &actual) {
  ASSERT_EQ(reference.keys(), actual.keys());
  for (auto key : reference.keys()) {
    EXPECT_LT(reference.at<gtsam::Pose3>(key).localCoordinates(
        actual.at<gtsam::Pose3>(key)).norm(), 2e-7) << gtsam::Symbol(key);
  }
}

void expectGraphNear(ConstraintManager &reference, ConstraintManager &accelerated,
                     bool fast, bool single = false) {
  const auto rg = single ? reference.get_single_graph() : reference.get_graph(fast);
  const auto ag = single ? accelerated.get_single_graph() : accelerated.get_graph(fast);
  const auto &values = reference.get_values();
  EXPECT_NEAR(rg.error(values), ag.error(values), 2e-7);
  gtsam::HessianFactor rh(*rg.linearize(values));
  gtsam::HessianFactor ah(*ag.linearize(values));
  EXPECT_TRUE(rh.augmentedInformation().isApprox(ah.augmentedInformation(), 1e-9));
}

class ManagerIntegration : public testing::TestWithParam<int> {
protected:
  void checkBackend() {
    if (!(GetParam() & 1)) return;
#ifdef FORM_ENABLE_CUDA
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
      GTEST_SKIP() << "CUDA device unavailable";
#else
    GTEST_SKIP() << "CUDA backend disabled";
#endif
  }

  ConstraintManager::Params params(bool single = false) const {
    ConstraintManager::Params result;
    result.use_summary = !(GetParam() & 1);
    result.use_cuda_summaries = GetParam() & 1;
    result.use_batch_summaries = GetParam() >= 2;
    result.batch_min_edges = 0;
    result.disable_smoothing = single;
    return result;
  }
};

TEST_P(ManagerIntegration, FastFullRematchMarginalizationAndNextScan) {
  checkBackend();
  if (IsSkipped()) return;
  ASSERT_FALSE(form::profile::enabled);
  ConstraintManager reference;
  ConstraintManager accelerated(params());
  for (size_t scan = 0; scan < 4; ++scan) {
    reference.step(initial(scan));
    accelerated.step(initial(scan));
    if (scan == 0) continue;
    fillCurrent(reference, scan);
    fillCurrent(accelerated, scan);
    const auto rv = reference.optimize(false);
    const auto av = accelerated.optimize(false);
    expectValuesNear(rv, av);
    EXPECT_LT(truth(scan).localCoordinates(av.at<gtsam::Pose3>(X(scan))).norm(), 1e-6);
    expectGraphNear(reference, accelerated, false);
    reference.update_values(rv);
    accelerated.update_values(av);
  }
  const auto rf = reference.optimize(true);
  const auto af = accelerated.optimize(true);
  expectValuesNear(rf, af);
  expectGraphNear(reference, accelerated, true);
  fillCurrent(reference, 3, 1);
  fillCurrent(accelerated, 3, 1);
  expectValuesNear(reference.optimize(true), accelerated.optimize(true));
  expectGraphNear(reference, accelerated, true);
  reference.update_values(reference.optimize(false));
  accelerated.update_values(accelerated.optimize(false));
  reference.marginalize({0});
  accelerated.marginalize({0});
  EXPECT_FALSE(reference.get_values().exists(X(0)));
  EXPECT_FALSE(accelerated.get_values().exists(X(0)));
  expectGraphNear(reference, accelerated, false);
  reference.step(initial(4));
  accelerated.step(initial(4));
  fillCurrent(reference, 4);
  fillCurrent(accelerated, 4);
  expectValuesNear(reference.optimize(true), accelerated.optimize(true));
  expectGraphNear(reference, accelerated, true);
  expectValuesNear(reference.optimize(false), accelerated.optimize(false));
}

TEST_P(ManagerIntegration, DisableSmoothingUsesUnarySummariesAcrossRematches) {
  checkBackend();
  if (IsSkipped()) return;
  ASSERT_FALSE(form::profile::enabled);
  ConstraintManager::Params reference_params;
  reference_params.disable_smoothing = true;
  ConstraintManager reference(reference_params);
  ConstraintManager accelerated(params(true));
  for (size_t scan = 0; scan < 5; ++scan) {
    reference.step(initial(scan));
    accelerated.step(initial(scan));
    if (scan == 0) continue;
    for (int rematch = 0; rematch < 2; ++rematch) {
      fillCurrent(reference, scan, rematch);
      fillCurrent(accelerated, scan, rematch);
      const auto rv = reference.optimize(rematch != 0);
      const auto av = accelerated.optimize(rematch != 0);
      ASSERT_EQ(rv.size(), 1);
      expectValuesNear(rv, av);
      EXPECT_LT(truth(scan).localCoordinates(av.at<gtsam::Pose3>(X(scan))).norm(), 1e-6);
      expectGraphNear(reference, accelerated, false, true);
      reference.update_values(rv);
      accelerated.update_values(av);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CpuAndCuda, ManagerIntegration, testing::Values(0,1,2,3));
} // namespace
