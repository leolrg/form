#include "form/feature/summary.hpp"
#include "form/feature/factor.hpp"

#include <gtest/gtest.h>
#include <Eigen/QR>
#include <random>
#include <thread>
#include <atomic>

namespace {
using Matrix13 = Eigen::Matrix<double, 13, 13>;

void append(form::PointPoint &points, form::PlanePoint &planes,
            const Eigen::Vector3d &pi, const Eigen::Vector3d &pj,
            const Eigen::Vector3d &normal, bool point, bool plane) {
  if (point) {
    points.p_i.insert(points.p_i.end(), pi.data(), pi.data() + 3);
    points.p_j.insert(points.p_j.end(), pj.data(), pj.data() + 3);
  }
  if (plane) {
    planes.p_i.insert(planes.p_i.end(), pi.data(), pi.data() + 3);
    planes.p_j.insert(planes.p_j.end(), pj.data(), pj.data() + 3);
    planes.n_i.insert(planes.n_i.end(), normal.data(), normal.data() + 3);
  }
}

Matrix13 direct(const form::PointPoint &points, const form::PlanePoint &planes,
                const gtsam::Pose3 &Ti, const gtsam::Pose3 &Tj) {
  Matrix13 result = Matrix13::Zero();
  auto add = [&](const auto &constraints) {
    if (!constraints.num_residuals()) return;
    gtsam::Matrix Ji, Jj;
    const gtsam::Vector residual = constraints.evaluateError(Ti, Tj, Ji, Jj);
    gtsam::Matrix augmented(residual.size(), 13);
    augmented << Ji, Jj, -residual;
    result.noalias() += augmented.transpose() * augmented;
  };
  add(points);
  add(planes);
  return result;
}

TEST(FeatureSummary, PreservesWorldNormalEquationsForBothPoses) {
  std::mt19937 random(712319);
  std::normal_distribution<double> gaussian;
  auto vector = [&]() {
    return Eigen::Vector3d(gaussian(random), gaussian(random), gaussian(random));
  };
  for (int mode = 0; mode < 3; ++mode) {
    for (int count : {1, 2, 6, 7, 12, 13, 37, 513}) {
      form::PointPoint points;
      form::PlanePoint planes;
      for (int k = 0; k < count; ++k)
        append(points, planes, 10 * vector(), 10 * vector(), vector(), mode != 1,
               mode != 0);
      const form::FeatureSummary summary(planes, points);
      for (int trial = 0; trial < 8; ++trial) {
        const auto a = vector(), b = vector();
        const gtsam::Pose3 Ti(gtsam::Rot3::RzRyRx(a[0], a[1], a[2]), 4 * vector());
        const gtsam::Pose3 Tj(gtsam::Rot3::RzRyRx(b[0], b[1], b[2]), 4 * vector());
        const auto expected = direct(points, planes, Ti, Tj);
        const auto actual = summary.augmentedHessian(Ti, Tj);
        for (int row = 0; row < 13; ++row)
          for (int col = 0; col < 13; ++col)
            EXPECT_NEAR(actual(row, col), expected(row, col),
                        2e-10 * std::max(1., std::abs(expected(row, col))))
                << "mode=" << mode << " count=" << count << " row=" << row
                << " col=" << col;
        EXPECT_NEAR(summary.squaredError(Ti, Tj), expected(12, 12),
                    2e-12 * std::max(1., expected(12, 12)));
      }
    }
  }
}

TEST(FeatureSummary, EmptyCorrespondencesHaveZeroErrorAndHessian) {
  const form::FeatureSummary summary(form::PlanePoint{}, form::PointPoint{});
  EXPECT_DOUBLE_EQ(summary.squaredError(gtsam::Pose3{}, gtsam::Pose3{}), 0.);
  EXPECT_TRUE(summary.augmentedHessian(gtsam::Pose3{}, gtsam::Pose3{}).isZero());
}

TEST(FeatureSummary, KeepsRankDeficientCorrespondencesWithoutRegularization) {
  form::PointPoint points;
  form::PlanePoint planes;
  for (int k = 0; k < 80; ++k)
    append(points, planes, {1, 2, 3}, {4, 5, 6}, {0, 0, 1}, true, true);
  const gtsam::Pose3 Ti(gtsam::Rot3::RzRyRx(.1, -.2, .3), {4, -1, 2});
  const gtsam::Pose3 Tj(gtsam::Rot3::RzRyRx(-.4, .3, .2), {-1, 5, 2});
  const form::FeatureSummary summary(planes, points);
  EXPECT_TRUE(summary.augmentedHessian(Ti, Tj).isApprox(direct(points, planes, Ti, Tj), 2e-13));
}

TEST(FeatureSummary, RetainsTinyErrorAtLargeCoordinates) {
  std::mt19937 random(19);
  std::normal_distribution<double> gaussian;
  auto vector = [&]() {
    return Eigen::Vector3d(gaussian(random), gaussian(random), gaussian(random));
  };
  for (bool plane : {false, true}) {
    form::PointPoint points;
    form::PlanePoint planes;
    for (int k = 0; k < 1000; ++k) {
      const Eigen::Vector3d pi = 1e4 * vector();
      const Eigen::Vector3d pj = pi + 1e-6 * vector();
      append(points, planes, pi, pj, vector(), !plane, plane);
    }
    const form::FeatureSummary summary(planes, points);
    const gtsam::Pose3 identity;
    const double expected = direct(points, planes, identity, identity)(12, 12);
    ASSERT_GT(expected, 0.);
    EXPECT_NEAR(summary.squaredError(identity, identity), expected, expected * 2e-12);
    EXPECT_NEAR(summary.augmentedHessian(identity, identity)(12, 12), expected,
                expected * 2e-12);
  }
}

TEST(FeatureSummary, CoincidentNonidentityPosesKeepExactlyZeroResidual) {
  form::PointPoint points;
  form::PlanePoint planes;
  for (int k = 0; k < 17; ++k) {
    const Eigen::Vector3d p(1e4 * k, 1e3 * k * k, -2e3 * k);
    append(points, planes, p, p, {0.3, -0.4, 0.5}, true, true);
  }
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(.31, -.73, .27), {12, -13, 17});
  const form::FeatureSummary summary(planes, points);
  EXPECT_DOUBLE_EQ(direct(points, planes, pose, pose)(12, 12), 0.);
  EXPECT_DOUBLE_EQ(summary.squaredError(pose, pose), 0.);
  EXPECT_DOUBLE_EQ(summary.augmentedHessian(pose, pose)(12, 12), 0.);
}

TEST(FeatureSummary, ExternalQrRootsHaveTheSameNormalEquations) {
  form::PointPoint points;
  form::PlanePoint planes;
  for (int k = 0; k < 30; ++k)
    append(points, planes, {double(k), 2., -3.}, {4., double(k * k), 6.},
           {1., -2., .5}, true, true);
  const Eigen::MatrixXd point_features = form::FeatureSummary::pointFeatures(points);
  const Eigen::MatrixXd plane_features = form::FeatureSummary::planeFeatures(planes);
  ASSERT_EQ(point_features.rows(), 30);
  ASSERT_EQ(point_features.cols(), 7);
  ASSERT_EQ(plane_features.rows(), 30);
  ASSERT_EQ(plane_features.cols(), 13);
  Eigen::HouseholderQR<Eigen::MatrixXd> point_qr(point_features), plane_qr(plane_features);
  Eigen::Matrix<double, 7, 7> point_root =
      point_qr.matrixQR().topRows(7).triangularView<Eigen::Upper>();
  Eigen::Matrix<double, 13, 13> plane_root =
      plane_qr.matrixQR().topRows(13).triangularView<Eigen::Upper>();
  const form::FeatureSummary summary(plane_root, point_root);
  const gtsam::Pose3 Ti(gtsam::Rot3::RzRyRx(.1, -.2, .3), {4, -1, 2});
  const gtsam::Pose3 Tj(gtsam::Rot3::RzRyRx(-.4, .3, .2), {-1, 5, 2});
  const auto expected = direct(points, planes, Ti, Tj);
  EXPECT_TRUE(summary.augmentedHessian(Ti, Tj).isApprox(expected, 2e-13));
  EXPECT_NEAR(summary.squaredError(Ti, Tj), expected(12, 12), 2e-13 * expected(12, 12));
}
} // namespace
TEST(FeatureSummary, DeferredRowsPreserveCountsAndMaterializeOnce) {
  form::PlanePoint planes;
  form::PointPoint points;
  int plane_loads=0, point_loads=0;
  planes.deferRaw(1,[&](const form::PlanePoint& p) {
    ++plane_loads; p.p_i={1,2,3}; p.p_j={2,4,6}; p.n_i={0,0,1};
  });
  points.deferRaw(1,[&](const form::PointPoint& p) {
    ++point_loads; p.p_i={1,2,3}; p.p_j={2,4,6};
  });
  EXPECT_EQ(planes.num_constraints(),1); EXPECT_EQ(points.num_residuals(),3);
  EXPECT_TRUE(planes.p_i.empty()); EXPECT_TRUE(points.p_i.empty());
  EXPECT_DOUBLE_EQ(planes.evaluateError({},{}).squaredNorm(),9.);
  EXPECT_DOUBLE_EQ(points.evaluateError({},{}).squaredNorm(),14.);
  form::FeatureSummary summary(planes,points);
  EXPECT_NEAR(summary.squaredError({},{}),23.,1e-12);
  EXPECT_EQ(plane_loads,1); EXPECT_EQ(point_loads,1);
}

TEST(FeatureSummary, DeferredRowsClearCancelsAndAppendMaterializes) {
  form::PointPoint points;
  int loads=0;
  auto load=[&](const form::PointPoint& p) { ++loads; p.p_i={1,2,3}; p.p_j={2,4,6}; };
  points.deferRaw(1,load);
  points.clear();
  EXPECT_EQ(loads,0); EXPECT_EQ(points.num_constraints(),0);
  points.deferRaw(1,load);
  points.push_back(form::PointFeat(0,0,0,0),form::PointFeat(1,0,0,1));
  EXPECT_EQ(loads,1); EXPECT_EQ(points.num_constraints(),2);
  EXPECT_DOUBLE_EQ(points.evaluateError({},{}).squaredNorm(),15.);
}

TEST(FeatureSummary, DeferredRowsConcurrentReadsCopiesAndRetry) {
  form::PointPoint points;
  std::atomic<int> loads{0};
  points.deferRaw(1,[&](const form::PointPoint& p) {
    ++loads; p.p_i={1,2,3}; p.p_j={2,4,6};
  });
  auto copy=points;
  std::vector<std::thread> readers;
  for(int i=0;i<8;++i) readers.emplace_back([&] {
    for(int j=0;j<50;++j) EXPECT_DOUBLE_EQ(points.evaluateError({},{}).squaredNorm(),14.);
  });
  for(auto& reader:readers) reader.join();
  EXPECT_EQ(loads,1);
  EXPECT_DOUBLE_EQ(copy.evaluateError({},{}).squaredNorm(),14.);
  EXPECT_EQ(loads,2);
  points.deferRaw(1,[&](const form::PointPoint& p) {
    if(++loads==3) throw std::runtime_error("download failed");
    p.p_i={0,0,0}; p.p_j={1,0,0};
  });
  EXPECT_THROW(points.evaluateError({},{}),std::runtime_error);
  EXPECT_EQ(points.num_constraints(),1);
  EXPECT_DOUBLE_EQ(points.evaluateError({},{}).squaredNorm(),1.);
}
