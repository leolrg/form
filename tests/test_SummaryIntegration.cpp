#include "form/feature/factor.hpp"
#include <gtest/gtest.h>

TEST(SummaryIntegration, ReusesAndInvalidatesCorrespondenceSnapshot) {
  using namespace form;
  auto planes = std::make_shared<PlanePoint>();
  auto points = std::make_shared<PointPoint>();
  auto constraints = std::make_tuple(planes,points);
  gtsam::Values values;
  values.insert(0,gtsam::Pose3());
  values.insert(1,gtsam::Pose3(gtsam::Rot3::RzRyRx(.1,.2,-.3),{1,2,3}));
  auto verify = [&]() {
    FeatureFactor original(0,1,constraints,0.1);
    FeatureFactor summary(0,1,constraints,0.1,true);
    EXPECT_TRUE(original.linearize(values)->augmentedInformation().isApprox(
                summary.linearize(values)->augmentedInformation(),1e-10));
    EXPECT_NEAR(original.error(values),summary.error(values),1e-8);
  };
  planes->push_back(PlanarFeat(1,2,3,0,0,1,0),PlanarFeat(2,1,3,0,0,1,1));
  verify();
  auto snapshot = planes->summary_cache;
  verify();
  EXPECT_EQ(snapshot,planes->summary_cache);
  points->push_back(PointFeat(1,0,2,0),PointFeat(2,3,1,1));
  verify();
  EXPECT_NE(snapshot,planes->summary_cache);
  snapshot = planes->summary_cache;
  planes->clear();
  planes->push_back(PlanarFeat(-1,0,2,1,0,0,0),PlanarFeat(1,4,0,1,0,0,1));
  verify();
  EXPECT_NE(snapshot,planes->summary_cache);
  points->clear();
  verify();
}

namespace {
gtsam::Values testValues() {
  gtsam::Values values;
  values.insert(0, gtsam::Pose3());
  values.insert(1, gtsam::Pose3(gtsam::Rot3::RzRyRx(.1, .2, -.3), {1, 2, 3}));
  return values;
}

void expectEquivalent(const form::FeatureFactor &reference,
                      const form::FeatureFactor &summary,
                      const gtsam::Values &values) {
  EXPECT_TRUE(reference.linearize(values)->augmentedInformation().isApprox(
      summary.linearize(values)->augmentedInformation(), 1e-10));
  EXPECT_NEAR(reference.error(values), summary.error(values), 1e-8);
}

class InactiveFeatureFactor : public form::FeatureFactor {
public:
  using FeatureFactor::FeatureFactor;
  bool active(const gtsam::Values &) const override { return false; }
};
} // namespace

TEST(SummaryIntegration, ReplacingPointSetWithSameRevisionRebuildsSummary) {
  auto planes = std::make_shared<form::PlanePoint>();
  auto first = std::make_shared<form::PointPoint>();
  auto replacement = std::make_shared<form::PointPoint>();
  first->push_back(form::PointFeat(1, 0, 2, 0), form::PointFeat(2, 3, 1, 1));
  replacement->push_back(form::PointFeat(-4, 2, 1, 0), form::PointFeat(3, -2, 4, 1));
  ASSERT_EQ(first->revision, replacement->revision);
  form::FeatureFactor initial(0, 1, {planes, first}, .1, true);
  form::FeatureFactor reference(0, 1, {planes, replacement}, .1);
  form::FeatureFactor summary(0, 1, {planes, replacement}, .1, true);
  expectEquivalent(reference, summary, testValues());
}

TEST(SummaryIntegration, ExistingFactorRetainsImmutableSnapshotAfterRematch) {
  auto planes = std::make_shared<form::PlanePoint>();
  auto points = std::make_shared<form::PointPoint>();
  points->push_back(form::PointFeat(1, 0, 2, 0), form::PointFeat(2, 3, 1, 1));
  form::FeatureFactor existing(0, 1, {planes, points}, .1, true);
  const auto values = testValues();
  const auto hessian = existing.linearize(values)->augmentedInformation();
  const double error = existing.error(values);
  points->clear();
  points->push_back(form::PointFeat(-4, 2, 1, 0), form::PointFeat(3, -2, 4, 1));
  form::FeatureFactor updated(0, 1, {planes, points}, .1, true);
  form::FeatureFactor reference(0, 1, {planes, points}, .1);
  expectEquivalent(reference, updated, values);
  EXPECT_TRUE(hessian.isApprox(existing.linearize(values)->augmentedInformation(), 1e-14));
  EXPECT_DOUBLE_EQ(error, existing.error(values));
  EXPECT_GT(std::abs(updated.error(values) - error), 1.0);
}

TEST(SummaryIntegration, InactiveSummaryFactorSkipsEvaluation) {
  auto planes = std::make_shared<form::PlanePoint>();
  auto points = std::make_shared<form::PointPoint>();
  points->push_back(form::PointFeat(1, 0, 2, 0), form::PointFeat(2, 3, 1, 1));
  InactiveFeatureFactor factor(0, 1, {planes, points}, .1, true);
  const gtsam::Values empty;
  EXPECT_NO_THROW(EXPECT_DOUBLE_EQ(0.0, factor.error(empty)));
  EXPECT_NO_THROW(EXPECT_FALSE(factor.linearize(empty)));
}

TEST(SummaryIntegration, FixedPoseWrapperUsesSummarySnapshot) {
  auto planes = std::make_shared<form::PlanePoint>();
  auto points = std::make_shared<form::PointPoint>();
  points->push_back(form::PointFeat(1, 0, 2, 0), form::PointFeat(2, 3, 1, 1));
  form::FeatureFactor factor(0, 1, {planes, points}, .1, true);
  auto values = testValues();
  const auto fixed_pose = values.at<gtsam::Pose3>(0);
  auto wrapper = form::BinaryFactorWrapper::Create(fixed_pose, 1, factor, true);
  gtsam::Values unary;
  unary.insert(1, values.at<gtsam::Pose3>(1));
  form::FeatureFactor reference_factor(0, 1, {planes, points}, .1);
  auto reference_wrapper = form::BinaryFactorWrapper::Create(fixed_pose, 1, reference_factor);
  EXPECT_TRUE(reference_wrapper.linearize(unary)->augmentedInformation().isApprox(
      wrapper.linearize(unary)->augmentedInformation(), 1e-10));
  EXPECT_NEAR(reference_wrapper.error(unary), wrapper.error(unary), 1e-8);
  const gtsam::Matrix original = wrapper.linearize(unary)->augmentedInformation();
  const double error = wrapper.error(unary);
  points->clear();
  points->push_back(form::PointFeat(-4, 2, 1, 0), form::PointFeat(3, -2, 4, 1));
  EXPECT_TRUE(original.isApprox(wrapper.linearize(unary)->augmentedInformation(), 1e-14));
  EXPECT_DOUBLE_EQ(error, wrapper.error(unary));
}

TEST(SummaryIntegration, DirectArrayMutationWithExplicitInvalidation) {
  auto planes = std::make_shared<form::PlanePoint>();
  auto points = std::make_shared<form::PointPoint>();
  planes->push_back(form::PlanarFeat(1, 2, 3, 0, 0, 1, 0),
                    form::PlanarFeat(2, 1, 3, 0, 0, 1, 1));
  points->push_back(form::PointFeat(1, 0, 2, 0), form::PointFeat(2, 3, 1, 1));
  form::FeatureFactor initial(0, 1, {planes, points}, .1, true);
  const auto snapshot = planes->summary_cache;
  planes->p_j[2] += 4;
  planes->invalidateSummary();
  EXPECT_FALSE(planes->summaryValidFor(points));
  form::FeatureFactor plane_update(0, 1, {planes, points}, .1, true);
  form::FeatureFactor plane_reference(0, 1, {planes, points}, .1);
  expectEquivalent(plane_reference, plane_update, testValues());
  EXPECT_NE(snapshot, planes->summary_cache);
  points->p_i[0] -= 3;
  points->invalidateSummary();
  EXPECT_FALSE(planes->summaryValidFor(points));
  form::FeatureFactor point_update(0, 1, {planes, points}, .1, true);
  form::FeatureFactor point_reference(0, 1, {planes, points}, .1);
  expectEquivalent(point_reference, point_update, testValues());
  EXPECT_TRUE(planes->summaryValidFor(points));
}
