#include "form/feature/factor.hpp"
#include <gtsam/base/numericalDerivative.h>
#include <gtest/gtest.h>
#include <random>

using namespace form;
namespace {
std::tuple<PlanePoint::Ptr, PointPoint::Ptr> makeConstraints(int planes, int points) {
  auto plane = std::make_shared<PlanePoint>();
  auto point = std::make_shared<PointPoint>();
  std::mt19937 random(12);
  std::normal_distribution<double> normal;
  auto vec = [&]() { return Eigen::Vector3d(normal(random), normal(random), normal(random)); };
  for (int i = 0; i < planes; ++i) {
    auto a = vec(), b = vec(), n = vec().normalized();
    plane->push_back(PlanarFeat(a.x(),a.y(),a.z(),n.x(),n.y(),n.z(),0),
                     PlanarFeat(b.x(),b.y(),b.z(),0,0,1,1));
  }
  for (int i = 0; i < points; ++i) {
    auto a = vec(), b = vec();
    point->push_back(PointFeat(a.x(),a.y(),a.z(),0), PointFeat(b.x(),b.y(),b.z(),1));
  }
  return {plane, point};
}
}
TEST(FeatureFactor, BothPoseJacobiansMatchNumericalDerivatives) {
  const gtsam::Pose3 a(gtsam::Rot3::RzRyRx(0.4,-0.3,0.2), {2,-1,4});
  const gtsam::Pose3 b(gtsam::Rot3::RzRyRx(-0.5,0.8,-0.1), {-1,3,2});
  for (auto counts : {std::pair<int,int>{25,0}, {0,27}, {13,17}}) {
    FeatureFactor factor(0,1,makeConstraints(counts.first,counts.second),0.3);
    gtsam::Matrix A, B;
    const auto residual = factor.evaluateError(a,b,A,B);
    ASSERT_GT(residual.size(),0);
    std::function<gtsam::Vector(const gtsam::Pose3&,const gtsam::Pose3&)> f =
        [&](const auto& x, const auto& y) {return factor.evaluateError(x,y);};
    auto numericA = gtsam::numericalDerivative21<gtsam::Vector,gtsam::Pose3,gtsam::Pose3>(f,a,b,1e-5);
    auto numericB = gtsam::numericalDerivative22<gtsam::Vector,gtsam::Pose3,gtsam::Pose3>(f,a,b,1e-5);
    EXPECT_LT((A-numericA).cwiseAbs().maxCoeff(), 1e-8);
    EXPECT_LT((B-numericB).cwiseAbs().maxCoeff(), 1e-8);
  }
}
TEST(FeatureFactor, DenseLinearizationPreservesNoiseAndRhsSign) {
  FeatureFactor factor(0,1,makeConstraints(23,19),0.3);
  gtsam::Values values;
  values.insert(0,gtsam::Pose3(gtsam::Rot3::RzRyRx(0.1,0.2,0.3),{1,2,3}));
  values.insert(1,gtsam::Pose3(gtsam::Rot3::RzRyRx(0.4,0.5,0.6),{2,1,4}));
  gtsam::Matrix A,B;
  const auto r=factor.evaluateError(values.at<gtsam::Pose3>(0),values.at<gtsam::Pose3>(1),A,B);
  gtsam::Matrix augmented(r.size(),13);
  augmented << A,B,-r;
  const gtsam::Matrix expected=augmented.transpose()*augmented/(0.3*0.3);
  const auto linear=factor.linearize(values);
  EXPECT_TRUE(linear->augmentedInformation().isApprox(expected,1e-12));
  EXPECT_NEAR(factor.error(values),r.squaredNorm()/(2*0.3*0.3),1e-9);
}
