#include <gtest/gtest.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <tbb/global_control.h>
#include "form/feature/batch_summary.hpp"
#include "form/feature/batch_factor.hpp"
#include "form/feature/factor.hpp"

static void append(form::PointPoint& points, form::PlanePoint& planes, const Eigen::Vector3d& p, const Eigen::Vector3d& q, const Eigen::Vector3d& normal, bool point=true, bool plane=true) {
  if(point) {points.p_i.insert(points.p_i.end(),p.data(),p.data()+3); points.p_j.insert(points.p_j.end(),q.data(),q.data()+3);}
  if(plane) {planes.p_i.insert(planes.p_i.end(),p.data(),p.data()+3); planes.p_j.insert(planes.p_j.end(),q.data(),q.data()+3); planes.n_i.insert(planes.n_i.end(),normal.data(),normal.data()+3);}
}

TEST(BatchSummary, MatchesIndependentFactorGraphAndUpdatedPoses) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  for (bool gpu : {false, true}) {
#ifndef FORM_ENABLE_CUDA
    if (gpu) continue;
#endif
    for (int n : {2, 10, 41}) {
      std::vector<form::SummaryEdge> edges;
      for (int i=0;i<n-1;++i) for(int j=i+1;j<n;++j) {
        if ((i+j)%3==0) continue;
        auto planes=std::make_shared<form::PlanePoint>();
        auto points=std::make_shared<form::PointPoint>();
        for(int k=0;k<19;++k) {
          Eigen::Vector3d p(.1*k, i-.3*k, .2*j+k), q=p+Eigen::Vector3d(.01*j,-.02*i,.03);
          append(*points,*planes,p,q,{.2,.3,.5},i%3!=0,i%3!=1);
        }
        edges.push_back({i,j,std::make_shared<form::FeatureSummary>(*planes,*points),.5+.1*i});
      }
      form::BatchSummary batch(n,edges,gpu);
      for(int trial=0;trial<3;++trial) {
        std::vector<gtsam::Pose3> poses;
        for(int i=0;i<n;++i) poses.emplace_back(gtsam::Rot3::RzRyRx(.01*i*trial,-.02*i,.03*trial),gtsam::Point3(.1*i,.2*trial,-.03*i));
        gtsam::GaussianFactorGraph reference;
        double error=0;
        for(const auto& e:edges) {
          auto h=(e.weight*e.summary->augmentedHessian(poses[e.i],poses[e.j])).eval();
          reference.emplace_shared<gtsam::HessianFactor>(e.i,e.j,h.block<6,6>(0,0),h.block<6,6>(0,6),h.block<6,1>(0,12),h.block<6,6>(6,6),h.block<6,1>(6,12),h(12,12));
          error+=.5*e.weight*e.summary->squaredError(poses[e.i],poses[e.j]);
        }
        gtsam::Matrix expected=gtsam::HessianFactor(reference).info().selfadjointView();
        auto actual=batch.linearize(poses);
        ASSERT_EQ(actual.rows(),expected.rows());
        EXPECT_LT((actual-expected).cwiseAbs().maxCoeff(),2e-10*std::max(1.,expected.cwiseAbs().maxCoeff())) << gpu << " " << n;
        EXPECT_NEAR(batch.error(poses),error,2e-12*std::max(1.,error));
        EXPECT_TRUE(actual.isApprox(actual.transpose(),1e-14));
        for(int c=0;c<actual.cols();++c) for(int r=0;r<actual.rows();++r)
          EXPECT_NEAR(actual(r,c),expected(r,c),2e-10*std::max(1.,std::abs(expected(r,c))));
      }
    }
  }
}

TEST(BatchSummary, EmptyZeroResidualAndInvalidInputs) {
  for(bool gpu:{false,true}) {
#ifndef FORM_ENABLE_CUDA
    if(gpu) continue;
#endif
    std::vector<gtsam::Pose3> poses(2,gtsam::Pose3(gtsam::Rot3::RzRyRx(.3,-.7,.2),{12,-13,17}));
    form::BatchSummary empty(2,{},gpu);
    EXPECT_EQ(empty.error(poses),0.);
    EXPECT_TRUE(empty.linearize(poses).isZero());
    form::PointPoint points; form::PlanePoint planes;
    for(int k=0;k<20;++k) {Eigen::Vector3d p(1e4*k,1e3*k*k,-2e3*k); append(points,planes,p,p,{.3,-.4,.5});}
    auto root=std::make_shared<form::FeatureSummary>(planes,points);
    form::BatchSummary batch(2,{{0,1,root,1.}},gpu);
    EXPECT_EQ(batch.error(poses),0.);
    EXPECT_EQ(batch.linearize(poses)(12,12),0.);
    EXPECT_THROW(batch.error({poses[0]}),std::invalid_argument);
    EXPECT_THROW((form::BatchSummary(2,{{0,2,root,1.}},gpu)),std::invalid_argument);
    EXPECT_THROW((form::BatchSummary(2,{{0,0,root,1.}},gpu)),std::invalid_argument);
    EXPECT_THROW((form::BatchSummary(2,{{0,1,root,-1.}},gpu)),std::invalid_argument);
    EXPECT_THROW((form::BatchSummary(2,{{0,1,{},1.}},gpu)),std::invalid_argument);
  }
}

TEST(BatchSummary, ReusesWorkspaceAcrossGraphAndRootChanges) {
  auto a=std::make_shared<form::FeatureSummary>(Eigen::Matrix<double,13,13>::Identity(),Eigen::Matrix<double,7,7>::Identity());
  auto b=std::make_shared<form::FeatureSummary>((2*Eigen::Matrix<double,13,13>::Identity()).eval(),Eigen::Matrix<double,7,7>::Zero());
  for(bool gpu:{false,true}) {
#ifndef FORM_ENABLE_CUDA
    if(gpu)continue;
#endif
    form::BatchSummary batch(2,{{0,1,a,1.}},gpu);
    for(int n:{3,40,2,10}) {
      std::vector<form::SummaryEdge> edges;
      for(int i=0;i<n-1;++i)edges.push_back({i,n-1,i%2?a:b,.7});
      std::vector<gtsam::Pose3> poses(n);
      batch.reset(n,edges);
      form::BatchSummary cpu(n,edges,false);
      EXPECT_TRUE(batch.linearize(poses).isApprox(cpu.linearize(poses),2e-12));
      EXPECT_NEAR(batch.error(poses),cpu.error(poses),2e-12);
      for(auto& edge:edges){edge.summary=b;edge.weight=.3;}
      batch.reset(n,edges);form::BatchSummary changed(n,edges,false);
      EXPECT_TRUE(batch.linearize(poses).isApprox(changed.linearize(poses),2e-12));
      batch.reset(n,{});EXPECT_TRUE(batch.linearize(poses).isZero());
    }
  }
}

TEST(BatchSummary, RetainedGraphsKeepTheirOwnSnapshot) {
  for(bool gpu:{false,true}) {
#ifndef FORM_ENABLE_CUDA
    if(gpu)continue;
#endif
    std::shared_ptr<form::BatchSummary> workspace;
    auto makeGraph=[](double offset,gtsam::Key i,gtsam::Key j) {
      auto points=std::make_shared<form::PointPoint>();auto planes=std::make_shared<form::PlanePoint>();
      append(*points,*planes,{1,2,3},{1+offset,2,3},{1,0,0});
      gtsam::NonlinearFactorGraph graph;
      graph.push_back(form::FeatureFactor(i,j,std::make_tuple(planes,points),.5,true));return graph;
    };
    gtsam::Values values;for(auto key:{7,10,42,99})values.insert(key,gtsam::Pose3{});
    auto original=makeGraph(.1,10,42);
    auto first=form::batchFeatureGraph(original,gpu,0,workspace);
    const auto firstH=first.linearize(values)->augmentedHessian();
    auto other=makeGraph(.5,7,99);
    auto second=form::batchFeatureGraph(other,gpu,0,workspace);
    EXPECT_NEAR(second.error(values),other.error(values),1e-12);
    EXPECT_NEAR(first.error(values),original.error(values),1e-12);
    EXPECT_TRUE(first.linearize(values)->augmentedHessian().isApprox(firstH,1e-12));
  }
}
