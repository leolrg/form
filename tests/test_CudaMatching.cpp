#include "form/optimization/cuda_matching.hpp"
#include "form/optimization/matcher.hpp"
#include "form/feature/summary.hpp"
#include <gtest/gtest.h>

TEST(CudaMatching, MatchesCpuConstraintsAndSummaryAcrossRematches) {
  form::VoxelMap<form::PlanarFeat> planes(.8);
  form::VoxelMap<form::PointFeat> points(.8);
  std::vector<form::PlanarFeat> pq;
  std::vector<form::PointFeat> qq;
  gtsam::Pose3 base(gtsam::Rot3::RzRyRx(.02,-.03,.01),gtsam::Point3(.1,-.2,.3));
  auto estimates=[&](size_t i) { return i==0?base:gtsam::Pose3(); };
  for(int i=0;i<301;++i) {
    const double x=(i%17)*.15,y=(i/17)*.12;
    form::PlanarFeat p(x,y,.1,.3,.5,.7,i%2);
    form::PointFeat q(x,y,.1,i%2);
    planes.push_back(p.transform(estimates(i%2)));
    points.push_back(q.transform(estimates(i%2)));
    p.scan=2; p.x+=.03; pq.push_back(p);
    q.scan=2; q.y+=.04; qq.push_back(q);
  }
  form::CudaMatching gpu;
  gpu.reset(planes,points,pq,qq,estimates,.8);
  form::Matcher<form::PlanarFeat> pcpu({},1);
  form::Matcher<form::PointFeat> qcpu({},1);
  form::CudaMatching::ConstraintMap reference,candidate;
  for(size_t i=0;i<2;++i) {
    reference[i]={std::make_shared<form::PlanePoint>(),std::make_shared<form::PointPoint>()};
    candidate[i]={std::make_shared<form::PlanePoint>(),std::make_shared<form::PointPoint>()};
  }
  tbb::concurrent_vector<form::Match<form::PlanarFeat>> pm;
  tbb::concurrent_vector<form::Match<form::PointFeat>> qm;
  for(double t:{0.,.05,-.1}) {
    gtsam::Pose3 current(gtsam::Rot3::RzRyRx(t/7,t/3,t/5),gtsam::Point3(t,0,0));
    auto poses=[&](size_t i) {return i==2?current:estimates(i);};
    pcpu.match<0>(planes,pq,poses,reference);
    qcpu.match<1>(points,qq,poses,reference);
    gpu.match(current,.8,candidate,pm,qm);
    ASSERT_EQ(pm.size(),pq.size()); ASSERT_EQ(qm.size(),qq.size());
    for(size_t i=0;i<2;++i) {
      auto [a,b]=candidate.at(i); auto [c,d]=reference.at(i);
      EXPECT_EQ(a->num_constraints(),c->num_constraints()); EXPECT_EQ(b->num_constraints(),d->num_constraints());
      ASSERT_TRUE(a->summaryValidFor(b));
      form::FeatureSummary expected(*c,*d),raw(*a,*b);
      for(double delta:{0.,.2}) {
        auto trial=current.retract(gtsam::Vector6::Constant(delta));
        auto h=expected.augmentedHessian(estimates(i),trial);
        EXPECT_LE((a->summary_cache->augmentedHessian(estimates(i),trial)-h).norm(),1e-10*(1+h.norm()));
        EXPECT_LE((a->summary_cache->augmentedHessian(estimates(i),trial)-raw.augmentedHessian(estimates(i),trial)).norm(),1e-10*(1+h.norm()));
        EXPECT_NEAR(a->summary_cache->squaredError(estimates(i),trial),expected.squaredError(estimates(i),trial),1e-9);
      }
    }
  }
}

TEST(CudaMatching, StrictThresholdAndCpuEmptyFeatureCompatibility) {
  form::CudaMatching gpu;
  form::VoxelMap<form::PlanarFeat> planes(1.);
  form::VoxelMap<form::PointFeat> points(1.);
  points.push_back(form::PointFeat(0,0,0,0));
  auto pose=[](size_t) { return gtsam::Pose3(); };
  form::CudaMatching::ConstraintMap constraints;
  constraints[0]={std::make_shared<form::PlanePoint>(),std::make_shared<form::PointPoint>()};
  tbb::concurrent_vector<form::Match<form::PlanarFeat>> pm;
  tbb::concurrent_vector<form::Match<form::PointFeat>> qm;
  gpu.reset(planes,points,{},{{1,0,0,1},{.5,0,0,1}},pose,1.);
  gpu.match({},1.,constraints,pm,qm);
  EXPECT_EQ(std::get<1>(constraints.at(0))->num_constraints(),1);
  EXPECT_DOUBLE_EQ(qm[0].dist_sqrd,1.);
  gpu.reset(planes,points,{},{},pose,1.);
  gpu.match({},1.,constraints,pm,qm);
  EXPECT_TRUE(pm.empty()); EXPECT_EQ(qm.size(),2); // CPU retains the previous matches on empty feature input.
  EXPECT_EQ(std::get<1>(constraints.at(0))->num_constraints(),0);
}
