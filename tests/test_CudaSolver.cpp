#include "form/optimization/cuda_solver.hpp"

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

Eigen::MatrixXd system(int n, double condition, int seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal;
  Eigen::MatrixXd random(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) random(i, j) = normal(rng);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(random);
  const Eigen::MatrixXd q = qr.householderQ();
  Eigen::VectorXd spectrum(n), truth(n);
  for (int i = 0; i < n; ++i) {
    spectrum[i] = std::pow(condition, -double(i) / std::max(1, n - 1));
    truth[i] = std::sin(i + 0.3);
  }
  Eigen::MatrixXd a = q * spectrum.asDiagonal() * q.transpose();
  a = (0.5 * (a + a.transpose())).eval();
  Eigen::MatrixXd augmented(n + 1, n + 1);
  augmented.topLeftCorner(n, n) = a;
  augmented.topRightCorner(n, 1) = a * truth;
  augmented.bottomLeftCorner(1, n) = augmented.topRightCorner(n, 1).transpose();
  augmented(n, n) = 1234567.89;
  return augmented;
}

void expectAccurate(form::CudaDenseSolver& solver, const Eigen::MatrixXd& augmented,
                    double relativeTolerance) {
  const int n = static_cast<int>(augmented.rows()) - 1;
  const Eigen::MatrixXd a = augmented.topLeftCorner(n, n);
  const Eigen::VectorXd b = augmented.topRightCorner(n, 1);
  Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> factor(a);
  ASSERT_EQ(factor.info(), Eigen::Success);
  const Eigen::VectorXd expected = factor.solve(b);
  Eigen::VectorXd actual;
  ASSERT_TRUE(solver.solve(augmented, actual));
  ASSERT_EQ(actual.size(), n);
  ASSERT_TRUE(actual.allFinite());
  EXPECT_LE((actual - expected).norm() / expected.norm(), relativeTolerance);
  const double denominator = a.cwiseAbs().rowwise().sum().maxCoeff() *
                                 actual.lpNorm<Eigen::Infinity>() +
                             b.lpNorm<Eigen::Infinity>();
  EXPECT_LE((a * actual - b).lpNorm<Eigen::Infinity>() / denominator, 1e-12);
}

}  // namespace

TEST(CudaSolver, SolvesWellConditionedSystemsAcrossProductionAndRaggedSizes) {
  form::CudaDenseSolver solver;
  for (int n : {1, 6, 33, 108, 174, 257, 258}) {
    SCOPED_TRACE(n);
    expectAccurate(solver, system(n, 100.), 1e-10);
  }
}

TEST(CudaSolver, SolvesIllConditionedPositiveDefiniteSystems) {
  form::CudaDenseSolver solver;
  for (int n : {6, 108, 174, 258}) {
    SCOPED_TRACE(n);
    expectAccurate(solver, system(n, 1e10), 1e-5);
  }
}

TEST(CudaSolver, ReusesStorageAcrossRepeatedGrowthAndShrinkage) {
  form::CudaDenseSolver solver;
  for (int repeat = 0; repeat < 3; ++repeat)
    for (int n : {6, 108, 33, 174, 258, 1, 129, 259, 6}) {
      SCOPED_TRACE(n);
      SCOPED_TRACE(repeat);
      expectAccurate(solver, system(n, 10., 50 + repeat), 1e-11);
    }
}

TEST(CudaSolver, PacksAugmentedColumnsWithTheirExtraRowAndIgnoresConstant) {
  form::CudaDenseSolver solver;
  Eigen::MatrixXd augmented(4, 4);
  augmented << 4., 1., 2., 8.,
               1., 5., 0., -9.,
               2., 0., 6., 20.,
               8., -9., 20., std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd actual;
  ASSERT_TRUE(solver.solve(augmented, actual));
  ASSERT_EQ(actual.size(), 3);
  Eigen::Vector3d expected(1., -2., 3.);
  EXPECT_LE((actual - expected).norm(), 1e-12);
}

TEST(CudaSolver, RejectsIndefiniteSystemsWithoutReplacingSolutionAndRecovers) {
  form::CudaDenseSolver solver;
  Eigen::MatrixXd augmented = Eigen::MatrixXd::Identity(7, 7);
  augmented(2, 2) = -1.;
  Eigen::VectorXd actual = Eigen::VectorXd::Constant(2, 17.);
  EXPECT_FALSE(solver.solve(augmented, actual));
  EXPECT_EQ(actual.size(), 2);
  EXPECT_TRUE(actual.isConstant(17.));
  expectAccurate(solver, system(6, 10.), 1e-11);
}

TEST(CudaSolver, RejectsNonfiniteInputsAndOutputWithoutReplacingSolution) {
  form::CudaDenseSolver solver;
  Eigen::VectorXd actual = Eigen::VectorXd::Constant(2, 17.);
  Eigen::MatrixXd augmented = system(6, 10.);
  augmented(2, 1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(solver.solve(augmented, actual));
  augmented = system(6, 10.);
  augmented(1, 6) = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(solver.solve(augmented, actual));
  augmented = Eigen::MatrixXd::Ones(2, 2);
  augmented(0, 0) = 1e-300;
  augmented(0, 1) = augmented(1, 0) = 1e300;
  EXPECT_FALSE(solver.solve(augmented, actual));
  EXPECT_EQ(actual.size(), 2);
  EXPECT_TRUE(actual.isConstant(17.));
}

TEST(CudaSolver, RejectsEmptyAndNonsquareAugmentedInputs) {
  form::CudaDenseSolver solver;
  Eigen::VectorXd actual;
  EXPECT_THROW(solver.solve(Eigen::MatrixXd(0, 0), actual), std::invalid_argument);
  EXPECT_THROW(solver.solve(Eigen::MatrixXd(1, 1), actual), std::invalid_argument);
  EXPECT_THROW(solver.solve(Eigen::MatrixXd(3, 4), actual), std::invalid_argument);
}
