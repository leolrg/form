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

TEST(BatchSummary, PreservesDerivedFactorActivation) {
  class Inactive final : public form::FeatureFactor {
   public:
    using FeatureFactor::FeatureFactor;
    bool active(const gtsam::Values&) const override { return false; }
  };
  auto points=std::make_shared<form::PointPoint>();auto planes=std::make_shared<form::PlanePoint>();
  append(*points,*planes,{1,2,3},{2,2,3},{1,0,0});
  std::shared_ptr<form::BatchSummary> workspace;
  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<Inactive>(7,9,std::make_tuple(planes,points),1.,true);
  auto batch=form::batchFeatureGraph(graph,false,0,workspace);
  EXPECT_EQ(batch[0],graph[0]);
  EXPECT_NO_THROW(EXPECT_EQ(batch.error(gtsam::Values{}),0.));
  graph.push_back(form::FeatureFactor(1,2,std::make_tuple(planes,points),1.,true));
  gtsam::Values values;values.insert(1,gtsam::Pose3{});values.insert(2,gtsam::Pose3{});
  batch=form::batchFeatureGraph(graph,false,0,workspace);
  EXPECT_EQ(batch[0],graph[0]);
  EXPECT_NO_THROW(EXPECT_NEAR(batch.error(values),graph.error(values),1e-12));
}

TEST(BatchSummary, ResidentFrozenAuxiliaryDampingAndTrialCost) {
#ifndef FORM_ENABLE_CUDA
  GTEST_SKIP() << "CUDA disabled";
#else
  const int n=18;
  auto root=std::make_shared<form::FeatureSummary>(Eigen::Matrix<double,13,13>::Identity(),Eigen::Matrix<double,7,7>::Identity());
  form::BatchSummary batch(3,{{0,2,root,.7}},true);
  Eigen::MatrixXd f=Eigen::MatrixXd::Zero(13,13),aux=Eigen::MatrixXd::Zero(7,7);
  f.topLeftCorner(12,12)=2*Eigen::MatrixXd::Identity(12,12);f(0,8)=f(8,0)=.2;
  f.topRightCorner(12,1)=Eigen::VectorXd::LinSpaced(12,-.3,.4);f.bottomLeftCorner(1,12)=f.topRightCorner(12,1).transpose();f(12,12)=50;
  aux.topLeftCorner(6,6)=3*Eigen::MatrixXd::Identity(6,6);aux(6,6)=10.;aux(2,6)=aux(6,2)=.2;
  std::vector<double> fv(f.data(),f.data()+f.size()),av(aux.data(),aux.data()+aux.size());
  batch.configureResident({{{0,2},fv,true}},{{1}});
  std::vector<gtsam::Pose3> poses(3);
  std::vector<double> offsets(12);for(int k=0;k<12;++k)offsets[k]=.01*k;
  Eigen::Map<const Eigen::VectorXd> d(offsets.data(),12);
  Eigen::MatrixXd shifted=f;const auto product=(f.topLeftCorner(12,12)*d).eval();
  shifted.topRightCorner(12,1)-=product;shifted.bottomLeftCorner(1,12)=shifted.topRightCorner(12,1).transpose();
  shifted(12,12)+=d.dot(product)-2*d.dot(f.col(12).head(12));
  Eigen::MatrixXd expected=Eigen::MatrixXd::Zero(n+1,n+1);
  auto scatter=[&](const Eigen::MatrixXd& h,std::vector<int> indices) {for(int c=0;c<h.cols();++c)for(int r=0;r<h.rows();++r)expected(indices[r],indices[c])+=h(r,c);};
  std::vector<int> mapping={0,1,2,3,4,5,12,13,14,15,16,17,18};
  scatter(.7*root->augmentedHessian(poses[0],poses[2]),mapping);scatter(shifted,mapping);scatter(aux,{6,7,8,9,10,11,18});
  batch.residentLinearize(poses,offsets,av);
  EXPECT_TRUE(batch.residentHessian().isApprox(expected,2e-12));
  double error=.5*shifted(12,12)+.35*root->squaredError(poses[0],poses[2]);
  EXPECT_NEAR(batch.residentError(poses,offsets),error,2e-12);
  // Trial costs must not destroy the retained linear model for a lambda retry.
  poses[2]=gtsam::Pose3(gtsam::Rot3::RzRyRx(.2,-.1,.3),{1,2,3});batch.residentError(poses,offsets);
  for(bool diagonal:{false,true})for(double lambda:{1e-5,.1,100.}) {
    Eigen::MatrixXd h=expected.topLeftCorner(n,n);
    for(int i=0;i<n;++i){double a=1./(1./std::sqrt(lambda));if(diagonal)a*=std::sqrt(std::clamp(h(i,i),.01,10.));h(i,i)+=a*a;}
    Eigen::VectorXd actual;double old_error,new_error;
    ASSERT_TRUE(batch.residentSolve(lambda,diagonal,.01,10.,actual,old_error,new_error));
    Eigen::VectorXd want=h.llt().solve(expected.topRightCorner(n,1));
    EXPECT_TRUE(actual.isApprox(want,2e-12));
    EXPECT_NEAR(old_error,.5*expected(n,n),2e-12);
    EXPECT_NEAR(new_error,.5*(expected(n,n)-2*actual.dot(expected.col(n).head(n))+actual.dot(expected.topLeftCorner(n,n)*actual)),2e-11);
  }
  batch.configureResident({{{0,2},fv,false}},{{1}});
  poses.assign(3,gtsam::Pose3{});
  EXPECT_NEAR(batch.residentError(poses,offsets),.35*root->squaredError(poses[0],poses[2]),2e-12);
  EXPECT_THROW(batch.residentLinearize(poses,{},av),std::invalid_argument);
#endif
}
