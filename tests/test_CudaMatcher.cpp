#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/cuda_qr.hpp"
#include "form/feature/summary.hpp"
#include "form/feature/features.hpp"
#include "form/mapping/map.hpp"
#include <gtest/gtest.h>
#include <random>
#include <cstdlib>

using form::CudaMatcher;

TEST(CudaMatcher, IncrementalBlocksRetainUnchangedLeavesAndHandleMigrationAndRejection) {
  const std::array<double,12> identity={1,0,0,0,0,1,0,0,0,0,1,0};
  auto shifted=identity; shifted[3]=.2;
  for(bool plane:{false,true}) for(int extra_group:{0,1}) for(int backend:{0,1,2}) {
    const bool tree=backend!=0,bounded=backend==2;
    CudaMatcher full,partial;
    full.setIncrementalSummaries(false);
    full.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    partial.setReuseMode(CudaMatcher::ReuseMode::Certified);
    partial.setIncrementalSummaries(true);
    partial.setSummaryTree(tree);
    partial.setSummaryTreeBounds(bounded);
    std::vector<CudaMatcher::MapPoint> points;
    std::vector<CudaMatcher::Query> queries;
    for(int i=0;i<192;++i) {
      points.push_back({{i+.1,.1,.1,0},{i*.7,.2,.3},{.3,.4,.5}});
      queries.push_back({{i+.1,.1,.1,0}});
    }
    points.push_back({{.4,.1,.1,0},{.15,.25,.35},{.4,.5,.6}});
    std::vector<int> groups(points.size(),0); groups.back()=extra_group;
    for(auto* matcher:{&full,&partial}) {
      matcher->reset({{{0,0,0},0,int(points.size())}},points,queries,1000.);
      matcher->setGroups(groups,2);
    }
    auto compare=[&](const auto& pose,double gate) {
      const auto expected=full.searchGrouped(pose,gate,plane);
      const auto actual=partial.searchGrouped(pose,gate,plane);
      ASSERT_EQ(actual.counts,expected.counts);
      for(size_t i=0;i<actual.roots.size();++i) {
        const Eigen::MatrixXd a=actual.roots[i].transpose()*actual.roots[i];
        const Eigen::MatrixXd b=expected.roots[i].transpose()*expected.roots[i];
        EXPECT_LE((a-b).norm(),1e-11*(1+b.norm()));
        Eigen::Matrix<double,13,13> ap=Eigen::Matrix<double,13,13>::Zero(),bp=ap;
        Eigen::Matrix<double,7,7> aq=Eigen::Matrix<double,7,7>::Zero(),bq=aq;
        if(plane) { ap=actual.roots[i]; bp=expected.roots[i]; }
        else { aq=actual.roots[i]; bq=expected.roots[i]; }
        const form::FeatureSummary sa(ap,aq),sb(bp,bq);
        for(double angle:{-.7,0.,.3}) {
          const gtsam::Pose3 ti(gtsam::Rot3::RzRyRx(angle,.1,-.2),gtsam::Point3(.2,-.3,.4));
          const gtsam::Pose3 tj(gtsam::Rot3::RzRyRx(-.1,angle,.2),gtsam::Point3(-.4,.2,.3));
          const auto ha=sa.augmentedHessian(ti,tj),hb=sb.augmentedHessian(ti,tj);
          EXPECT_LE((ha-hb).norm(),1e-10*(1+hb.norm()));
          EXPECT_NEAR(sa.squaredError(ti,tj),sb.squaredError(ti,tj),1e-10*(1+sb.squaredError(ti,tj)));
        }
      }
    };
    compare(identity,1.);
    EXPECT_EQ(partial.summaryStats().active_leaves,3u);
    EXPECT_EQ(partial.summaryStats().tree,tree);
    EXPECT_EQ(partial.summaryStats().bounded,bounded);
    EXPECT_EQ(partial.summaryStats().dirty_leaves,3u);
    compare(shifted,1.);
    EXPECT_EQ(partial.summaryStats().dirty_leaves,size_t(extra_group?2:1));
    compare(shifted,1.);
    EXPECT_EQ(partial.summaryStats().dirty_leaves,0u);
    compare(shifted,.005); // Every query rejected; delete all former rows.
    compare(identity,1.); // Reinsert into retained empty slots.
    partial.setGroups(std::vector<int>(points.size(),-1),2);
    const auto empty=partial.searchGrouped(identity,1.,plane);
    EXPECT_EQ(empty.counts,(std::vector<size_t>{0,0}));
    for(const auto& root:empty.roots) EXPECT_EQ(root.norm(),0.);
  }
}

TEST(CudaMatcher, IncrementalSlotsCrossBlockScanChunksAndRefillScatteredHoles) {
  const std::array<double,12> identity={1,0,0,0,0,1,0,0,0,0,1,0};
  auto shifted=identity; shifted[3]=.2;
  std::vector<CudaMatcher::MapPoint> points;
  std::vector<CudaMatcher::Query> queries;
  std::vector<int> groups;
  for(int i=0;i<777;++i) {
    const double x=2*i+.1;
    queries.push_back({{x,.1,.1,0}});
    points.push_back({{x,.1,.1,0},{x-.01,.1,.1},{.3,.4,.5}}); groups.push_back(0);
    if(i%17==0) { points.push_back({{x+.3,.1,.1,0},{x+.29,.1,.1},{.5,.4,.3}}); groups.push_back(1); }
  }
  for(bool plane:{false,true}) for(int backend:{0,1,2}) {
    const bool tree=backend!=0,bounded=backend==2;
    CudaMatcher full,partial;
    full.setIncrementalSummaries(false);
    partial.setIncrementalSummaries(true);
    partial.setSummaryTree(tree);
    partial.setSummaryTreeBounds(bounded);
    for(auto* matcher:{&full,&partial}) {
      matcher->reset({{{0,0,0},0,int(points.size())}},points,queries,2000.);
      matcher->setGroups(groups,3);
    }
    for(int iteration=0;iteration<20;++iteration) {
      const auto pose=iteration%3?shifted:identity;
      const double gate=iteration%5==4?1e-4:1.;
      const auto a=partial.searchGrouped(pose,gate,plane),b=full.searchGrouped(pose,gate,plane);
      ASSERT_EQ(a.counts,b.counts);
      for(size_t g=0;g<a.roots.size();++g) {
        const Eigen::MatrixXd aa=a.roots[g].transpose()*a.roots[g],bb=b.roots[g].transpose()*b.roots[g];
        EXPECT_LE((aa-bb).norm(),1e-11*(1+bb.norm()));
      }
    }
  }
}

TEST(CudaMatcher, EmptyMapAndQueryTail) {
  CudaMatcher matcher;
  std::vector<CudaMatcher::Query> queries(257);
  matcher.reset({}, {}, queries, 1.);
  auto matches = matcher.search({1,0,0,0,0,1,0,0,0,0,1,0});
  ASSERT_EQ(matches.size(), queries.size());
  for(auto m : matches) { EXPECT_EQ(m.index,-1); EXPECT_EQ(m.distance,std::numeric_limits<double>::max()); }
  matcher.reset({}, {}, {}, 1.);
  EXPECT_TRUE(matcher.search({1,0,0,0,0,1,0,0,0,0,1,0}).empty());
}

TEST(CudaMatcher, MatchesCpuVoxelSearchAcrossPosesAndMapReplacement) {
  CudaMatcher matcher;
  std::mt19937 rng(318);
  std::uniform_real_distribution<double> uniform(-4.,4.);
  for(int n : {1,513,67}) {
    form::VoxelMap<form::PointFeat> map(.8);
    for(int i=0;i<n;++i) map.push_back(form::PointFeat(uniform(rng),uniform(rng),uniform(rng),i));
    std::vector<CudaMatcher::MapPoint> points;
    std::vector<CudaMatcher::Voxel> voxels;
    for(const auto& [key,voxel] : map) {
      voxels.push_back({{key.x(),key.y(),key.z()},int(points.size()),int(voxel.size())});
      for(const auto& p:voxel) points.push_back({{p.x,p.y,p.z,p._},{p.x,p.y,p.z},{0,0,0}});
    }
    std::vector<CudaMatcher::Query> queries;
    for(int i=0;i<259;++i) queries.push_back({{uniform(rng),uniform(rng),uniform(rng),0}});
    matcher.reset(voxels,points,queries,.8);
    for(double angle : {0.,.1,-.3}) {
      gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(angle,angle/2,angle/3),gtsam::Point3(.13,-.24,.35));
      std::array<double,12> packed;
      auto matrix=pose.matrix();
      for(int i=0;i<3;++i) for(int j=0;j<4;++j) packed[4*i+j]=matrix(i,j);
      auto actual=matcher.search(packed);
      for(size_t i=0;i<queries.size();++i) {
        auto p=queries[i].point;
        auto expected=map.find_closest(form::PointFeat(p[0],p[1],p[2],0).transform(pose));
        if(!expected.found()) { EXPECT_EQ(actual[i].index,-1); continue; }
        ASSERT_GE(actual[i].index,0);
        const auto& a=points[actual[i].index];
        EXPECT_DOUBLE_EQ(a.world[0],expected.point.x);
        EXPECT_DOUBLE_EQ(a.world[1],expected.point.y);
        EXPECT_DOUBLE_EQ(a.world[2],expected.point.z);
        EXPECT_NEAR(actual[i].distance,expected.dist_sqrd,2e-14);
      }
    }
  }
}

TEST(CudaMatcher, VoxelOrderTiesNegativeFloorAndPaddingDistance) {
  CudaMatcher matcher;
  // Origin wins a tie over the +x voxel; within a voxel, first wins.
  std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,2},{{1,0,0},2,1},{{-1,0,0},3,1}};
  std::vector<CudaMatcher::MapPoint> points={
    {{.75,0,0,0},{},{ }},{{.75,0,0,0},{},{}},
    {{1.25,0,0,0},{},{}},{{-.1,0,0,0},{},{}}};
  matcher.reset(voxels,points,{{{1.,0,0,0}},{{-.01,0,0,0}},{{0,0,0,1}}},1.);
  auto result=matcher.search({1,0,0,0,0,1,0,0,0,0,1,0});
  EXPECT_EQ(result[0].index,2); // Query lies in +x voxel: visit its own voxel first.
  EXPECT_EQ(result[1].index,3);
  EXPECT_NEAR(result[2].distance,1.01,1e-14);
  matcher.reset({{{0,0,0},0,2}}, {points[0],points[1]}, {{{.8,0,0,0}}},1.);
  EXPECT_EQ(matcher.search({1,0,0,0,0,1,0,0,0,0,1,0})[0].index,0);
}

TEST(CudaMatcher, DeviceSummariesPreserveRawFeaturesAcrossRaggedGroups) {
  CudaMatcher matcher;
  std::vector<CudaMatcher::MapPoint> points={{{.1,.2,.3,0},{.11,.21,.31},{.3,.4,.5}}};
  std::vector<CudaMatcher::Query> queries(259);
  for(size_t i=0;i<queries.size();++i) queries[i]={{.1+i*1e-5,.2,.3,0}};
  matcher.reset({{{0,0,0},0,1}},points,queries,1.);
  matcher.search({1,0,0,0,0,1,0,0,0,0,1,0});
  std::vector<std::vector<int>> groups(4);
  for(int i=0;i<259;++i) groups[i%3].push_back(i);
  for(bool plane:{false,true}) {
    auto roots=matcher.summarize(groups,plane);
    for(size_t k=0;k<groups.size();++k) {
      Eigen::MatrixXd f(groups[k].size(),plane?13:7);
      for(size_t row=0;row<groups[k].size();++row) {
        Eigen::Map<const Eigen::Vector3d> p(points[0].local),n(points[0].normal),q(queries[groups[k][row]].point);
        if(plane) {
          for(int a=0;a<3;++a) f.block<1,3>(row,3*a)=n[a]*q.transpose();
          f.block<1,3>(row,9)=n.transpose(); f(row,12)=n.dot(q-p);
        } else { f(row,0)=1; f.block<1,3>(row,1)=p.transpose(); f.block<1,3>(row,4)=(q-p).transpose(); }
      }
      Eigen::MatrixXd expected=f.transpose()*f;
      EXPECT_LE((roots[k].transpose()*roots[k]-expected).norm(),1e-11*(1+expected.norm()));
    }
  }
  EXPECT_THROW(matcher.summarize({{-1}},false),std::invalid_argument);
  EXPECT_THROW(matcher.summarize({{259}},false),std::invalid_argument);
}

TEST(CudaMatcher, RejectsNonfiniteSnapshotAndOutOfRangeTransformedQueries) {
  CudaMatcher matcher;
  EXPECT_THROW(matcher.reset({}, {}, {{{std::numeric_limits<double>::infinity(),0,0,0}}},1.),std::invalid_argument);
  CudaMatcher::MapPoint p{{0,0,0,0},{0,0,0},{0,0,std::numeric_limits<double>::quiet_NaN()}};
  EXPECT_THROW(matcher.reset({{{0,0,0},0,1}},{p},{{{0,0,0,0}}},1.),std::invalid_argument);
  matcher.reset({}, {}, {{{0,0,0,0}}},1.);
  EXPECT_THROW(matcher.search({1,0,0,1e100,0,1,0,0,0,0,1,0}),std::invalid_argument);
  EXPECT_THROW(matcher.summarize({},false),std::logic_error);
  EXPECT_NO_THROW(matcher.search({1,0,0,0,0,1,0,0,0,0,1,0}));
}

TEST(CudaMatcher, VisitsEveryNeighborOfQueryVoxel) {
  CudaMatcher matcher;
  for(const auto& shift:form::voxel_shifts) {
    CudaMatcher::MapPoint p{{shift.x()+.5,shift.y()+.5,shift.z()+.5,0},{},{}};
    matcher.reset({{{shift.x(),shift.y(),shift.z()},0,1}},{p},{{{.5,.5,.5,0}}},1.);
    EXPECT_EQ(matcher.search({1,0,0,0,0,1,0,0,0,0,1,0})[0].index,0);
  }
}

TEST(CudaMatcher, DenseVoxelTailWinnerAndFirstHitAcrossLanes) {
  CudaMatcher matcher;
  for(int count:{33,65,257,4097}) {
    std::vector<CudaMatcher::MapPoint> points(count,CudaMatcher::MapPoint{{.7,0,0,0},{},{}});
    points.back().world[0]=.125;
    std::vector<CudaMatcher::Query> queries(137,CudaMatcher::Query{{0,0,0,0}});
    matcher.reset({{{0,0,0},0,count}},points,queries,1.);
    for(auto r:matcher.search({1,0,0,0,0,1,0,0,0,0,1,0})) EXPECT_EQ(r.index,count-1);
    points[1].world[0]=.125;
    matcher.reset({{{0,0,0},0,count}},points,queries,1.);
    for(auto r:matcher.search({1,0,0,0,0,1,0,0,0,0,1,0})) EXPECT_EQ(r.index,1);
  }
}


TEST(CudaMatcher, EveryNeighborTieUsesVisitOrderBeforeStorageOrder) {
  CudaMatcher matcher;
  // Reverse snapshot order so an index-only reduction cannot satisfy first hit.
  for(int first=0;first<26;++first) {
    std::vector<CudaMatcher::MapPoint> points;
    std::vector<CudaMatcher::Voxel> voxels;
    for(int n=26;n>=first;--n) {
      const auto shift=form::voxel_shifts[n];
      voxels.push_back({{shift.x(),shift.y(),shift.z()},int(points.size()),65});
      for(int j=0;j<65;++j) points.push_back({{.5,.5,.5,j==33?0.:1.},{},{}});
    }
    std::vector<CudaMatcher::Query> queries(131,CudaMatcher::Query{{.5,.5,.5,0}});
    matcher.reset(voxels,points,queries,1.);
    for(auto r:matcher.search({1,0,0,0,0,1,0,0,0,0,1,0})) {
      EXPECT_EQ(r.index,(26-first)*65+33);
      EXPECT_DOUBLE_EQ(r.distance,0.);
    }
  }
}

TEST(CudaMatcher, FourDimensionalSearchStorageRefreshesAcrossRaggedResets) {
  CudaMatcher matcher;
  for(int count:{1,33,4097,7,129}) {
    std::vector<CudaMatcher::MapPoint> points(count);
    for(int i=0;i<count;++i) {
      points[i]={{.2,.3,.4,2.},{1000.+i,-2000.,3000.},{-4.,5.,-6.}};
    }
    points.back().world[3]=.25;
    matcher.reset({{{0,0,0},0,count}},points,{{{.2,.3,.4,.5}}},1.);
    const auto result=matcher.search({1,0,0,0,0,1,0,0,0,0,1,0})[0];
    EXPECT_EQ(result.index,count-1);
    EXPECT_DOUBLE_EQ(result.distance,.0625);
  }
}


namespace {
const std::array<double,12> identity_pose{1,0,0,0,0,1,0,0,0,0,1,0};
std::vector<Eigen::MatrixXd> grouped_inputs;
void captureGroupedInputs(const std::vector<Eigen::MatrixXd>& inputs) { grouped_inputs=inputs; }
struct GroupedInputCapture {
  GroupedInputCapture() { grouped_inputs.clear(); form::BatchedCudaQr::setInputObserver(captureGroupedInputs); }
  ~GroupedInputCapture() { form::BatchedCudaQr::setInputObserver(nullptr); }
};
}

TEST(CudaMatcher, DeviceGroupingPreservesAcceptedQueryOrderAndLazyResults) {
  CudaMatcher matcher;
  std::vector<CudaMatcher::MapPoint> points={
    {{.1,.2,.3,0},{10,20,30},{.2,.3,.4}},
    {{.4,.2,.3,0},{11,21,31},{.5,.6,.7}},
    {{.7,.2,.3,0},{12,22,32},{.8,.9,1.}}};
  for(int rows:{1,259,4097,7}) {
    std::vector<CudaMatcher::Query> queries(rows);
    for(int i=0;i<rows;++i) queries[i]={{points[i%3].world[0]+(i%201)*1e-5,.2,.3,0}};
    matcher.reset({{{0,0,0},0,3}},points,queries,1.);
    matcher.setGroups({2,-1,0},4);
    for(bool plane:{false,true}) {
      auto pose=identity_pose; pose[3]=plane?.0001:0.;
      const auto expected_results=matcher.search(pose);
      std::vector<std::vector<int>> expected_groups(4);
      for(int i=0;i<rows;++i) if(expected_results[i].distance<1e-6) {
        const int group=std::array<int,3>{2,-1,0}[expected_results[i].index];
        if(group>=0) expected_groups[group].push_back(i);
      }
      GroupedInputCapture capture;
      const auto result=matcher.searchGrouped(pose,1e-6,plane);
      ASSERT_EQ(result.roots.size(),4); ASSERT_EQ(result.counts.size(),4); ASSERT_EQ(grouped_inputs.size(),4);
      for(int group=0;group<4;++group) {
        EXPECT_EQ(result.counts[group],expected_groups[group].size());
        Eigen::MatrixXd expected(expected_groups[group].size(),plane?13:7);
        for(size_t row=0;row<expected_groups[group].size();++row) {
          const int q=expected_groups[group][row]; const auto& p=points[expected_results[q].index];
          Eigen::Map<const Eigen::Vector3d> pi(p.local),n(p.normal),pj(queries[q].point);
          if(plane) {
            for(int a=0;a<3;++a) expected.block<1,3>(row,3*a)=n[a]*pj.transpose();
            expected.block<1,3>(row,9)=n.transpose(); expected(row,12)=n.dot(pj-pi);
          } else { expected(row,0)=1.; expected.block<1,3>(row,1)=pi.transpose(); expected.block<1,3>(row,4)=(pj-pi).transpose(); }
        }
        EXPECT_TRUE(grouped_inputs[group].isApprox(expected,1e-15));
        const Eigen::MatrixXd gram=expected.transpose()*expected;
        EXPECT_LE((result.roots[group].transpose()*result.roots[group]-gram).norm(),1e-10*(1+gram.norm()));
      }
      const auto& downloaded=matcher.downloadResults(); ASSERT_EQ(downloaded.size(),expected_results.size());
      for(size_t i=0;i<downloaded.size();++i) {
        EXPECT_EQ(downloaded[i].index,expected_results[i].index);
        EXPECT_DOUBLE_EQ(downloaded[i].distance,expected_results[i].distance);
      }
      EXPECT_EQ(&matcher.downloadResults(),&downloaded);
      EXPECT_NO_THROW(matcher.summarize(expected_groups,plane));
    }
  }
}

TEST(CudaMatcher, DeviceGroupingStrictThresholdEmptyGroupsAndValidation) {
  CudaMatcher matcher;
  EXPECT_THROW(matcher.setGroups({},0),std::logic_error);
  EXPECT_THROW(matcher.downloadResults(),std::logic_error);
  matcher.reset({{{0,0,0},0,1}},{{{0,0,0,0},{},{}}},{{{1,0,0,0}},{{.5,0,0,0}},{{5,0,0,0}}},1.);
  EXPECT_THROW(matcher.searchGrouped(identity_pose,1.,false),std::logic_error);
  EXPECT_THROW(matcher.setGroups({},1),std::invalid_argument);
  EXPECT_THROW(matcher.setGroups({1},1),std::invalid_argument);
  EXPECT_THROW(matcher.setGroups({-2},1),std::invalid_argument);
  EXPECT_THROW(matcher.setGroups({0},65536),std::invalid_argument);
  matcher.setGroups({0},1);
  for(double threshold:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(matcher.searchGrouped(identity_pose,threshold,false),std::invalid_argument);
  const auto accepted=matcher.searchGrouped(identity_pose,1.,false);
  EXPECT_EQ(accepted.counts,std::vector<size_t>{1}); // Equal distance is rejected.
  matcher.setGroups({-1},0);
  const auto empty=matcher.searchGrouped(identity_pose,1.,true);
  EXPECT_TRUE(empty.roots.empty()); EXPECT_TRUE(empty.counts.empty());
  EXPECT_EQ(matcher.downloadResults().size(),3);
  auto invalid=identity_pose; invalid[3]=1e100;
  EXPECT_THROW(matcher.searchGrouped(invalid,1.,true),std::invalid_argument);
  EXPECT_THROW(matcher.downloadResults(),std::logic_error);
  matcher.reset({}, {}, {},1.);
  EXPECT_THROW(matcher.searchGrouped(identity_pose,1.,false),std::logic_error);
  matcher.setGroups({},2);
  const auto no_queries=matcher.searchGrouped(identity_pose,1.,false);
  EXPECT_EQ(no_queries.counts,(std::vector<size_t>{0,0}));
  ASSERT_EQ(no_queries.roots.size(),2); EXPECT_TRUE(no_queries.roots[0].isZero()); EXPECT_TRUE(no_queries.roots[1].isZero());
  matcher.reset({}, {}, {{{0,0,0,0}}},1.); matcher.setGroups({},1);
  EXPECT_EQ(matcher.searchGrouped(identity_pose,1.,false).counts,std::vector<size_t>{0});
}


TEST(CudaMatcher, DeviceInversePosesPreserveWorldSearchAndLocalFeatureRows) {
  CudaMatcher matcher;
  const std::vector<gtsam::Pose3> inverse={
    gtsam::Pose3(gtsam::Rot3::RzRyRx(.13,-.21,.31),gtsam::Point3(1e4,-2e4,3e4)),
    gtsam::Pose3(gtsam::Rot3::RzRyRx(-.42,.17,-.08),gtsam::Point3(-4e4,5e4,-6e4))};
  std::vector<std::array<double,12>> matrices(2);
  for(int p=0;p<2;++p) for(int r=0;r<3;++r) for(int c=0;c<4;++c) matrices[p][4*r+c]=inverse[p].matrix()(r,c);
  for(int count:{1,257,17}) {
    std::vector<CudaMatcher::MapPoint> points;
    std::vector<CudaMatcher::Query> queries;
    std::vector<int> pose_indices;
    for(int i=0;i<count;++i) {
      points.push_back({{.1+i*.0001,.2,.3,.25+i*.01},{99,98,97},{.2+i*.001,-.3,.4}});
      queries.push_back({{points.back().world[0],.2,.3,points.back().world[3]}});
      pose_indices.push_back(i%2);
    }
    matcher.reset({{{0,0,0},0,count}},points,queries,1.,matrices,pose_indices);
    matcher.setGroups(std::vector<int>(count,0),1);
    for(bool plane:{false,true}) {
      GroupedInputCapture capture;
      const auto summary=matcher.searchGrouped(identity_pose,1e-8,plane);
      ASSERT_EQ(summary.counts,std::vector<size_t>{size_t(count)});
      ASSERT_EQ(grouped_inputs.size(),1);
      for(int i=0;i<count;++i) {
        const auto& p=points[i];
        const auto local=inverse[i%2]*gtsam::Point3(p.world[0],p.world[1],p.world[2]);
        const auto normal=inverse[i%2].rotation()*gtsam::Point3(p.normal[0],p.normal[1],p.normal[2]);
        Eigen::Map<const Eigen::Vector3d> query(queries[i].point);
        if(plane) {
          for(int a=0;a<3;++a) EXPECT_DOUBLE_EQ(grouped_inputs[0](i,9+a),normal[a]);
          EXPECT_NEAR(grouped_inputs[0](i,12),normal.dot(query-local),1e-10);
        } else for(int a=0;a<3;++a) EXPECT_DOUBLE_EQ(grouped_inputs[0](i,1+a),local[a]);
      }
      const auto& results=matcher.downloadResults();
      for(int i=0;i<count;++i) { EXPECT_EQ(results[i].index,i); EXPECT_DOUBLE_EQ(results[i].distance,0.); }
    }
    // A subsequent ordinary reset must use the supplied local coordinates.
    matcher.reset({{{0,0,0},0,count}},points,queries,1.);
    matcher.search(identity_pose);
    GroupedInputCapture capture;
    matcher.summarize({{0}},false);
    EXPECT_DOUBLE_EQ(grouped_inputs[0](0,1),99.);
  }
}

TEST(CudaMatcher, DeviceInversePoseValidationAndOverflow) {
  CudaMatcher matcher;
  const std::vector<CudaMatcher::MapPoint> points={{{2,0,0,0},{2,0,0},{0,0,1}}};
  const std::vector<CudaMatcher::Voxel> voxels={{{2,0,0},0,1}};
  const std::vector<CudaMatcher::Query> queries={{{2,0,0,0}}};
  EXPECT_THROW(matcher.reset(voxels,points,queries,1.,{}, {0}),std::invalid_argument);
  EXPECT_THROW(matcher.reset(voxels,points,queries,1.,{identity_pose},{}),std::invalid_argument);
  EXPECT_THROW(matcher.reset(voxels,points,queries,1.,{identity_pose},{-1}),std::invalid_argument);
  EXPECT_THROW(matcher.reset(voxels,points,queries,1.,{identity_pose},{1}),std::invalid_argument);
  auto bad=identity_pose; bad[0]=std::numeric_limits<double>::infinity();
  EXPECT_THROW(matcher.reset(voxels,points,queries,1.,{bad},{0}),std::invalid_argument);
  bad[0]=std::numeric_limits<double>::max();
  EXPECT_THROW(matcher.reset(voxels,points,queries,1.,{bad},{0}),std::invalid_argument);
  EXPECT_THROW(matcher.search(identity_pose),std::logic_error);
  EXPECT_NO_THROW(matcher.reset(voxels,points,queries,1.,{identity_pose},{0}));
  EXPECT_EQ(matcher.search(identity_pose)[0].index,0);
}


TEST(CudaMatcher, ReuseCertifiesSmallMotionWithExactCurrentDistances) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified}) {
    CudaMatcher matcher,oracle;
    matcher.setReuseMode(mode); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    std::vector<CudaMatcher::MapPoint> points={{{.2,.3,.4,.5},{},{}},{{.8,.7,.6,.2},{},{}}};
    std::vector<CudaMatcher::Query> queries(67,CudaMatcher::Query{{.21,.3,.4,.5}});
    matcher.reset({{{0,0,0},0,2}},points,queries,1.);
    oracle.reset({{{0,0,0},0,2}},points,queries,1.);
    matcher.search(identity_pose);
    EXPECT_EQ(matcher.reuseStats().searched,queries.size());
    for(int step=1;step<=4;++step) {
      auto pose=identity_pose; pose[3]=step*.001;
      const auto expected=oracle.search(pose),actual=matcher.search(pose);
      for(size_t i=0;i<actual.size();++i) {
        EXPECT_EQ(actual[i].index,expected[i].index);
        EXPECT_EQ(actual[i].distance,expected[i].distance);
      }
      const auto stats=matcher.reuseStats();
      EXPECT_EQ(stats.total,queries.size()); EXPECT_EQ(stats.certified,queries.size());
      EXPECT_EQ(stats.searched,mode==CudaMatcher::ReuseMode::Audit?queries.size():0);
      EXPECT_EQ(stats.mismatches,0);
      EXPECT_EQ(stats.oracle_searched,mode==CudaMatcher::ReuseMode::Audit?queries.size():0);
    }
  }
}

TEST(CudaMatcher, ReuseCrossingsTiesPaddingAndImmutableAnchor) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified}) {
    CudaMatcher matcher,oracle; matcher.setReuseMode(mode); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    // Arbitrary world placement is legal. Crossing introduces point 2 and changes tie order.
    std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,2},{{2,0,0},2,1}};
    std::vector<CudaMatcher::MapPoint> points={{{.8,.2,.2,.5},{},{}},{{1.2,.2,.2,.5},{},{}},{{1.,.2,.2,.5},{},{}}};
    matcher.reset(voxels,points,{{{.9,.2,.2,.5}}},1.); oracle.reset(voxels,points,{{{.9,.2,.2,.5}}},1.);
    for(double dx:{0.,.02,.04,.08,.099,.1,.101,.15,-.05}) {
      auto pose=identity_pose; pose[3]=dx;
      const auto expected=oracle.search(pose)[0],actual=matcher.search(pose)[0];
      EXPECT_EQ(actual.index,expected.index); EXPECT_EQ(actual.distance,expected.distance);
      EXPECT_EQ(matcher.reuseStats().mismatches,0);
      if(dx==.1) EXPECT_EQ(matcher.reuseStats().cell_fallback,1);
    }
    points={{{.2,.3,.4,0.},{},{}},{{.2,.3,.4,0.},{},{}},{{.2,.3,.4,2.},{},{}}};
    matcher.reset({{{0,0,0},0,3}},points,{{{.2,.3,.4,0}}},1.);
    matcher.search(identity_pose); matcher.search(identity_pose);
    EXPECT_EQ(matcher.reuseStats().certified,1); // Identical position safely preserves ties.
    auto pose=identity_pose; pose[3]=.001;
    EXPECT_EQ(matcher.search(pose)[0].index,0);
    EXPECT_EQ(matcher.reuseStats().certified,0); EXPECT_EQ(matcher.reuseStats().gap_fallback,1);
  }
}

TEST(CudaMatcher, ReuseRecomputesGateAndSummaryAndInvalidatesAfterFailure) {
  CudaMatcher matcher,oracle; matcher.setReuseMode(CudaMatcher::ReuseMode::Certified);
  oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
  for(int rows:{1,67,7,7}) {
    std::vector<CudaMatcher::MapPoint> points={{{.2+rows*.001,.3,.4,.5},{.1,.2,.3},{.3,.4,.5}},{{.8,.7,.6,.2},{},{}}};
    std::vector<CudaMatcher::Query> queries(rows,CudaMatcher::Query{{.21,.3,.4,.5}});
    for(auto* m:{&matcher,&oracle}) { m->reset({{{0,0,0},0,2}},points,queries,1.); m->setGroups({0,1},2); }
    for(bool plane:{false,true}) for(double dx:{0.,.001,.002}) for(double gate:{1e-6,1.}) {
      auto pose=identity_pose; pose[3]=dx;
      const auto expected=oracle.searchGrouped(pose,gate,plane),actual=matcher.searchGrouped(pose,gate,plane);
      EXPECT_EQ(actual.counts,expected.counts);
      for(size_t i=0;i<actual.roots.size();++i) EXPECT_TRUE(actual.roots[i].isApprox(expected.roots[i],1e-15));
      const auto expected_results=oracle.downloadResults(),actual_results=matcher.downloadResults();
      for(int i=0;i<rows;++i) { EXPECT_EQ(actual_results[i].index,expected_results[i].index); EXPECT_EQ(actual_results[i].distance,expected_results[i].distance); }
    }
    auto invalid=identity_pose; invalid[3]=std::numeric_limits<double>::infinity();
    EXPECT_THROW(matcher.search(invalid),std::invalid_argument);
    matcher.search(identity_pose); EXPECT_EQ(matcher.reuseStats().certified,0);
    invalid[3]=1e100; EXPECT_THROW(matcher.search(invalid),std::invalid_argument);
    matcher.search(identity_pose); EXPECT_EQ(matcher.reuseStats().certified,0);
    EXPECT_THROW(matcher.reset({}, {}, queries,-1.),std::invalid_argument);
    EXPECT_THROW(matcher.search(identity_pose),std::logic_error);
  }
}

TEST(CudaMatcher, ReuseOracleParityAcrossRandomizedMotionAndAdversarialScales) {
  std::mt19937 rng(71903);
  std::uniform_real_distribution<double> uniform(.1,.9);
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified}) {
    CudaMatcher matcher,oracle; matcher.setReuseMode(mode); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    for(double scale:{1.,1e-160,1e-310,1e150,1e155}) {
      std::vector<CudaMatcher::MapPoint> points;
      std::vector<CudaMatcher::Query> queries;
      for(int i=0;i<131;++i) {
        points.push_back({{scale*uniform(rng),scale*uniform(rng),scale*uniform(rng),scale*uniform(rng)},{},{}});
        if(i<67) queries.push_back({{points.back().world[0],points.back().world[1],points.back().world[2],points.back().world[3]}});
      }
      points[65]=points[0]; // Distinct equal-distance competitor in another lane/tile.
      // Explicitly overlapping ranges are legal, and repeat identical point indices.
      const std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,131},{{1,0,0},32,99}};
      for(auto* m:{&matcher,&oracle}) m->reset(voxels,points,queries,scale);
      for(int step=0;step<40;++step) {
        const double angle=step*1e-4;
        auto pose=identity_pose;
        pose[0]=std::cos(angle); pose[1]=-std::sin(angle); pose[4]=std::sin(angle); pose[5]=std::cos(angle);
        pose[3]=scale*step*.0002; pose[7]=-scale*step*.0001;
        const auto expected=oracle.search(pose),actual=matcher.search(pose);
        for(size_t i=0;i<actual.size();++i) {
          ASSERT_EQ(actual[i].index,expected[i].index) << scale << ":" << step << ":" << i;
          ASSERT_EQ(actual[i].distance,expected[i].distance) << scale << ":" << step << ":" << i;
        }
        EXPECT_EQ(matcher.reuseStats().mismatches,0);
      }
    }
    const std::vector<CudaMatcher::MapPoint> points={{{.2,.2,.2,0},{},{}},{{.8,.2,.2,0},{},{}}};
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},points,{{{.25,.2,.2,0}}},1.);
    for(int step=0;step<30;++step) {
      auto pose=identity_pose; pose[3]=step*.015;
      const auto expected=oracle.search(pose)[0],actual=matcher.search(pose)[0];
      ASSERT_EQ(actual.index,expected.index); ASSERT_EQ(actual.distance,expected.distance);
      EXPECT_EQ(matcher.reuseStats().mismatches,0);
    }
  }
}

TEST(CudaMatcher, ReuseEmptyDomainsOverflowAndModeChanges) {
  CudaMatcher matcher; matcher.setReuseMode(CudaMatcher::ReuseMode::Certified);
  matcher.reset({}, {}, {{{.2,.3,.4,0}}},1.);
  matcher.search(identity_pose);
  auto pose=identity_pose; pose[3]=.001;
  EXPECT_EQ(matcher.search(pose)[0].index,-1); EXPECT_EQ(matcher.reuseStats().certified,1);
  matcher.setReuseMode(CudaMatcher::ReuseMode::Certified);
  matcher.search(pose); EXPECT_EQ(matcher.reuseStats().searched,1);
  matcher.reset({{{0,0,0},0,1}},{{{1e200,0,0,0},{},{}}},{{{.2,.3,.4,0}}},1.);
  matcher.search(identity_pose);
  EXPECT_EQ(matcher.search(pose)[0].index,-1); EXPECT_EQ(matcher.reuseStats().certified,0);
  EXPECT_EQ(matcher.search(pose)[0].index,-1); EXPECT_EQ(matcher.reuseStats().certified,1);
  matcher.setGroups({0},1);
  EXPECT_THROW(matcher.searchGrouped(pose,-1.,false),std::invalid_argument);
  matcher.search(pose); EXPECT_EQ(matcher.reuseStats().certified,0);
  pose[3]=1e100;
  EXPECT_THROW(matcher.searchGrouped(pose,1.,false),std::invalid_argument);
  matcher.search(identity_pose); EXPECT_EQ(matcher.reuseStats().certified,0);
  matcher.setReuseMode(CudaMatcher::ReuseMode::Disabled);
  matcher.search(identity_pose); EXPECT_EQ(matcher.reuseStats().certified,0);
  EXPECT_EQ(matcher.reuseStats().searched,1);
  EXPECT_THROW(matcher.setReuseMode(static_cast<CudaMatcher::ReuseMode>(123)),std::invalid_argument);
  matcher.reset({}, {}, {},1.); matcher.setReuseMode(CudaMatcher::ReuseMode::Certified);
  EXPECT_TRUE(matcher.search(identity_pose).empty()); EXPECT_EQ(matcher.reuseStats().total,0);
}

TEST(CudaMatcher, ReuseRoundedTieAndOverflowedCompetitorFallBack) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified}) {
    CudaMatcher matcher,oracle; matcher.setReuseMode(mode); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    const double below=std::nextafter(.5,0.);
    // The tiny y difference is lost when added to x squared: computed tie.
    std::vector<CudaMatcher::MapPoint> points={
      {{.25,.5,.5,0},{},{}},{{.75,std::nextafter(.5,1.),.5,0},{},{}}};
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},points,{{{below,.5,.5,0}}},1.);
    matcher.search(identity_pose);
    auto pose=identity_pose; pose[3]=.5-below;
    const auto expected=oracle.search(pose)[0],actual=matcher.search(pose)[0];
    EXPECT_EQ(expected.index,0); EXPECT_EQ(expected.distance,.0625);
    EXPECT_EQ(actual.index,expected.index); EXPECT_EQ(actual.distance,expected.distance);
    EXPECT_EQ(matcher.reuseStats().certified,0); EXPECT_EQ(matcher.reuseStats().gap_fallback,1);
    // The anchor runner-up overflows, then becomes finite and wins in the same voxel.
    points={{{1e154,0,0,0},{},{}},{{2.5e154,0,0,0},{},{}}};
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},points,{{{1e154,0,0,0}}},1e155);
    EXPECT_EQ(matcher.search(identity_pose)[0].index,0);
    pose=identity_pose; pose[3]=.9e154;
    const auto finite_expected=oracle.search(pose)[0],finite_actual=matcher.search(pose)[0];
    EXPECT_EQ(finite_expected.index,1); EXPECT_TRUE(std::isfinite(finite_expected.distance));
    EXPECT_EQ(finite_actual.index,finite_expected.index); EXPECT_EQ(finite_actual.distance,finite_expected.distance);
    EXPECT_EQ(matcher.reuseStats().certified,0); EXPECT_EQ(matcher.reuseStats().gap_fallback,1);
    EXPECT_EQ(matcher.reuseStats().mismatches,0);
  }
}

TEST(CudaMatcher, ReuseGroupReassignmentAndExclusionUseCurrentGroups) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified}) {
    CudaMatcher matcher,oracle; matcher.setReuseMode(mode); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    std::vector<CudaMatcher::MapPoint> points={
      {{.2,.3,.4,.5},{.1,.2,.3},{.3,.4,.5}},{{.8,.7,.6,.2},{.7,.6,.5},{.2,.3,.4}}};
    const std::vector<CudaMatcher::Query> queries={{{.21,.3,.4,.5}},{{.81,.7,.6,.2}}};
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},points,queries,1.);
    matcher.search(identity_pose);
    for(const std::vector<int>& groups:{std::vector<int>{0,1},std::vector<int>{1,0},std::vector<int>{-1,1},std::vector<int>{0,-1}}) {
      for(auto* m:{&matcher,&oracle}) m->setGroups(groups,2);
      for(bool plane:{false,true}) {
        auto pose=identity_pose; pose[3]=.001;
        const auto expected=oracle.searchGrouped(pose,1.,plane),actual=matcher.searchGrouped(pose,1.,plane);
        EXPECT_EQ(actual.counts,expected.counts);
        for(size_t i=0;i<actual.roots.size();++i) EXPECT_TRUE(actual.roots[i].isApprox(expected.roots[i],1e-15));
        EXPECT_EQ(matcher.reuseStats().certified,2); EXPECT_EQ(matcher.reuseStats().mismatches,0);
      }
    }
  }
}

TEST(CudaMatcher, SearchVariantsSelectSplitAndConservativeOccurrenceBounds) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified})
  for(auto kernel:{CudaMatcher::SearchKernel::Fused,CudaMatcher::SearchKernel::Split})
  for(bool occurrences:{false,true}) {
    CudaMatcher matcher,oracle;
    matcher.setReuseMode(mode); matcher.setSearchKernel(kernel); matcher.setOccurrenceBound(occurrences);
    oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    const std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,2},{{1,0,0},0,1}};
    const std::vector<CudaMatcher::MapPoint> points={{{.2,.3,.4,0},{},{}},{{.8,.3,.4,0},{},{}}};
    for(auto* m:{&matcher,&oracle}) m->reset(voxels,points,{{{.21,.3,.4,0}}},1.);
    matcher.search(identity_pose);
    EXPECT_EQ(matcher.reuseStats().split_queries,0); // First search skips the prepass.
    auto pose=identity_pose; pose[3]=.001;
    const auto expected=oracle.search(pose)[0],actual=matcher.search(pose)[0];
    EXPECT_EQ(actual.index,expected.index); EXPECT_EQ(actual.distance,expected.distance);
    const auto stats=matcher.reuseStats();
    EXPECT_EQ(stats.certified,occurrences?0:1); // Repeated winner weakens occurrence bound.
    EXPECT_EQ(stats.split_queries,kernel==CudaMatcher::SearchKernel::Split?1:0);
    EXPECT_EQ(stats.oracle_searched,mode==CudaMatcher::ReuseMode::Audit?1:0); EXPECT_EQ(stats.mismatches,0);
    matcher.search(pose); EXPECT_EQ(matcher.reuseStats().certified,1); // Identical coordinates preserve ties.
    matcher.setSearchKernel(kernel); matcher.search(pose); EXPECT_EQ(matcher.reuseStats().certified,0);
    matcher.search(pose); matcher.setOccurrenceBound(occurrences);
    matcher.search(pose); EXPECT_EQ(matcher.reuseStats().certified,0);
    EXPECT_THROW(matcher.setSearchKernel(static_cast<CudaMatcher::SearchKernel>(123)),std::invalid_argument);
  }
}

TEST(CudaMatcher, SearchVariantsPreserveOracleAcrossMotionTiesScalesAndResets) {
  std::mt19937 rng(372);
  std::uniform_real_distribution<double> random(.1,.9);
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified})
  for(auto kernel:{CudaMatcher::SearchKernel::Fused,CudaMatcher::SearchKernel::Split})
  for(bool occurrences:{false,true}) {
    CudaMatcher matcher,oracle;
    matcher.setReuseMode(mode); matcher.setSearchKernel(kernel); matcher.setOccurrenceBound(occurrences);
    oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    for(double scale:{1.,1e-160,1e155}) for(int rows:{67,7,7}) {
      std::vector<CudaMatcher::MapPoint> points;
      std::vector<CudaMatcher::Query> queries;
      for(int i=0;i<131;++i) points.push_back({{scale*random(rng),scale*random(rng),scale*random(rng),scale*random(rng)},{},{}});
      points[65]=points[0];
      for(int i=0;i<rows;++i) queries.push_back({{points[i].world[0],points[i].world[1],points[i].world[2],points[i].world[3]}});
      const std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,131},{{1,0,0},0,66},{{2,0,0},65,66}};
      for(auto* m:{&matcher,&oracle}) m->reset(voxels,points,queries,scale);
      for(int step=0;step<22;++step) {
        auto pose=identity_pose; const double angle=step*.0001;
        pose[0]=std::cos(angle); pose[1]=-std::sin(angle); pose[4]=std::sin(angle); pose[5]=std::cos(angle);
        pose[3]=scale*(step<15?step*.001:(step-14)*.15);
        const auto expected=oracle.search(pose),actual=matcher.search(pose);
        for(size_t i=0;i<actual.size();++i) {
          ASSERT_EQ(actual[i].index,expected[i].index) << step << ":" << scale << ":" << occurrences;
          ASSERT_EQ(actual[i].distance,expected[i].distance) << step << ":" << scale << ":" << occurrences;
        }
        EXPECT_EQ(matcher.reuseStats().mismatches,0);
      }
      auto invalid=identity_pose; invalid[3]=scale*1e12;
      EXPECT_THROW(matcher.search(invalid),std::invalid_argument);
      matcher.search(identity_pose); EXPECT_EQ(matcher.reuseStats().certified,0);
      invalid[0]=std::numeric_limits<double>::quiet_NaN();
      EXPECT_THROW(matcher.search(invalid),std::invalid_argument);
      matcher.search(identity_pose); EXPECT_EQ(matcher.reuseStats().certified,0);
    }
  }
}

TEST(CudaMatcher, SearchVariantsPreserveCurrentGatesGroupsAndUnchangedAuditCounts) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified})
  for(auto kernel:{CudaMatcher::SearchKernel::Fused,CudaMatcher::SearchKernel::Split})
  for(bool occurrences:{false,true}) {
    CudaMatcher matcher,oracle;
    matcher.setReuseMode(mode); matcher.setSearchKernel(kernel); matcher.setOccurrenceBound(occurrences);
    oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    const std::vector<CudaMatcher::MapPoint> points={
      {{.2,.3,.4,0},{.1,.2,.3},{.3,.4,.5}},{{.8,.3,.4,0},{.7,.6,.5},{.2,.3,.4}}};
    for(auto* m:{&matcher,&oracle}) {
      m->reset({{{0,0,0},0,2}},points,{{{.21,.3,.4,0}},{{.81,.3,.4,0}}},1.); m->setGroups({0,1},2);
    }
    matcher.search(identity_pose);
    for(bool plane:{false,true}) for(double displacement:{.001,.002,.02}) for(double gate:{.00015,1.}) {
      auto pose=identity_pose; pose[3]=displacement;
      const auto expected=oracle.searchGrouped(pose,gate,plane),actual=matcher.searchGrouped(pose,gate,plane);
      EXPECT_EQ(actual.counts,expected.counts);
      for(size_t g=0;g<2;++g) EXPECT_TRUE(actual.roots[g].isApprox(expected.roots[g],1e-15));
      const auto stats=matcher.reuseStats(); EXPECT_EQ(stats.mismatches,0);
      EXPECT_EQ(stats.unchanged,mode==CudaMatcher::ReuseMode::Audit?2:0);
      const auto expected_results=oracle.downloadResults(),actual_results=matcher.downloadResults();
      for(size_t i=0;i<2;++i) { EXPECT_EQ(actual_results[i].index,expected_results[i].index); EXPECT_EQ(actual_results[i].distance,expected_results[i].distance); }
    }
    for(auto* m:{&matcher,&oracle}) m->setGroups({-1,0},2);
    const auto expected=oracle.searchGrouped(identity_pose,1.,false),actual=matcher.searchGrouped(identity_pose,1.,false);
    EXPECT_EQ(actual.counts,expected.counts); EXPECT_EQ(matcher.reuseStats().mismatches,0);
  }
}

namespace {
struct ScopedMatcherEnvironment {
  const char* name;
  bool existed;
  std::string previous;
  ScopedMatcherEnvironment(const char* key,const char* value):name(key),existed(std::getenv(key)!=nullptr),previous(existed?std::getenv(key):"") {
    if(setenv(name,value,1)!=0) throw std::runtime_error("Cannot set test environment");
  }
  ~ScopedMatcherEnvironment() { if(existed) setenv(name,previous.c_str(),1); else unsetenv(name); }
};
}
TEST(CudaMatcher, SearchVariantEnvironmentSelectionAndValidation) {
  ScopedMatcherEnvironment reuse("FORM_CUDA_MATCH_REUSE","certified");
  for(const char* kernel:{"fused","split"}) for(const char* bound:{"distinct","occurrences"}) {
    ScopedMatcherEnvironment kernel_env("FORM_CUDA_MATCH_KERNEL",kernel),bound_env("FORM_CUDA_MATCH_TOP2",bound);
    CudaMatcher matcher;
    matcher.reset({{{0,0,0},0,2},{{1,0,0},0,1}},
      {{{.2,.3,.4,0},{},{}},{{.8,.3,.4,0},{},{}}},{{{.21,.3,.4,0}}},1.);
    matcher.search(identity_pose);
    auto pose=identity_pose; pose[3]=.001; matcher.search(pose);
    EXPECT_EQ(matcher.reuseStats().split_queries,std::string(kernel)=="split"?1:0);
    EXPECT_EQ(matcher.reuseStats().certified,std::string(bound)=="occurrences"?0:1);
  }
  { ScopedMatcherEnvironment invalid("FORM_CUDA_MATCH_KERNEL","invalid"); EXPECT_THROW(CudaMatcher{},std::invalid_argument); }
  { ScopedMatcherEnvironment invalid("FORM_CUDA_MATCH_TOP2","invalid"); EXPECT_THROW(CudaMatcher{},std::invalid_argument); }
}

TEST(CudaMatcher, SquaredCertificateConservativelyFallsBackWhenPolynomialUnderflows) {
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified})
  for(auto kernel:{CudaMatcher::SearchKernel::Fused,CudaMatcher::SearchKernel::Split})
  for(bool occurrences:{false,true}) for(bool squared:{false,true}) {
    CudaMatcher matcher,oracle;
    matcher.setReuseMode(mode); matcher.setSearchKernel(kernel); matcher.setOccurrenceBound(occurrences);
    matcher.setSquaredCertificate(squared); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    const std::vector<CudaMatcher::MapPoint> points={{{0,0,0,0},{},{}},{{1e-100,0,0,0},{},{}}};
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},points,{{{0,0,0,0}}},1.);
    matcher.search(identity_pose);
    auto pose=identity_pose; pose[3]=1e-102;
    const auto expected=oracle.search(pose)[0],actual=matcher.search(pose)[0];
    EXPECT_EQ(actual.index,expected.index); EXPECT_EQ(actual.distance,expected.distance);
    EXPECT_EQ(matcher.reuseStats().certified,squared?0:1);
    EXPECT_EQ(matcher.reuseStats().mismatches,0);
    matcher.search(pose); EXPECT_EQ(matcher.reuseStats().certified,1);
    matcher.setSquaredCertificate(squared); matcher.search(pose);
    EXPECT_EQ(matcher.reuseStats().certified,0);
  }
}

TEST(CudaMatcher, SquaredCertificateMatchesOriginalOverAdversarialScalesPaddingAndMotion) {
  std::mt19937 rng(42871);
  std::uniform_real_distribution<double> random(.1,.9);
  for(auto mode:{CudaMatcher::ReuseMode::Audit,CudaMatcher::ReuseMode::Certified})
  for(auto kernel:{CudaMatcher::SearchKernel::Fused,CudaMatcher::SearchKernel::Split})
  for(bool occurrences:{false,true}) {
    CudaMatcher matcher,oracle;
    matcher.setReuseMode(mode); matcher.setSearchKernel(kernel); matcher.setOccurrenceBound(occurrences);
    matcher.setSquaredCertificate(true); oracle.setReuseMode(CudaMatcher::ReuseMode::Disabled);
    for(double scale:{1.,1e-100,1e-160,1e-310,1e150,1e155}) {
      std::vector<CudaMatcher::MapPoint> points;
      std::vector<CudaMatcher::Query> queries;
      for(int i=0;i<67;++i) {
        points.push_back({{scale*random(rng),scale*random(rng),scale*random(rng),scale*random(rng)},{},{}});
        if(i<33) queries.push_back({{points.back().world[0],points.back().world[1],points.back().world[2],points.back().world[3]}});
      }
      points[66]=points[0];
      const std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,67},{{1,0,0},0,33}};
      for(auto* m:{&matcher,&oracle}) m->reset(voxels,points,queries,scale);
      for(int step=0;step<23;++step) {
        auto pose=identity_pose; const double angle=step*.0001;
        pose[0]=std::cos(angle); pose[1]=-std::sin(angle); pose[4]=std::sin(angle); pose[5]=std::cos(angle);
        pose[3]=scale*(step<15?step*.001:(step-14)*.13);
        const auto expected=oracle.search(pose),actual=matcher.search(pose);
        for(size_t i=0;i<actual.size();++i) {
          ASSERT_EQ(actual[i].index,expected[i].index) << scale << ":" << step;
          ASSERT_EQ(actual[i].distance,expected[i].distance) << scale << ":" << step;
        }
        EXPECT_EQ(matcher.reuseStats().mismatches,0);
      }
    }
    // Rounded Voronoi tie: the tiny y term is lost when added to x squared.
    const double below=std::nextafter(.5,0.);
    const std::vector<CudaMatcher::MapPoint> tied={{{.25,.5,.5,.25},{},{}},{{.75,std::nextafter(.5,1.),.5,.25},{},{}}};
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},tied,{{{below,.5,.5,.25}}},1.);
    matcher.search(identity_pose); auto pose=identity_pose; pose[3]=.5-below;
    const auto expected=oracle.search(pose)[0],actual=matcher.search(pose)[0];
    EXPECT_EQ(actual.index,expected.index); EXPECT_EQ(actual.distance,expected.distance);
    EXPECT_EQ(matcher.reuseStats().certified,0); EXPECT_EQ(matcher.reuseStats().mismatches,0);
    // Ordinary-scale strict gaps actually certify; this is not an always-fallback implementation.
    for(auto* m:{&matcher,&oracle}) m->reset({{{0,0,0},0,2}},
      {{{.2,.3,.4,.25},{},{}},{{.8,.3,.4,.5},{},{}}},{{{.21,.3,.4,.25}}},1.);
    matcher.search(identity_pose); pose=identity_pose; pose[3]=.001;
    const auto stable_expected=oracle.search(pose)[0],stable_actual=matcher.search(pose)[0];
    EXPECT_EQ(stable_actual.index,stable_expected.index); EXPECT_EQ(stable_actual.distance,stable_expected.distance);
    EXPECT_EQ(matcher.reuseStats().certified,1); EXPECT_EQ(matcher.reuseStats().mismatches,0);
  }
}

TEST(CudaMatcher, SquaredCertificateEnvironmentSelectionAndValidation) {
  ScopedMatcherEnvironment reuse("FORM_CUDA_MATCH_REUSE","certified");
  for(const char* variant:{"norm","squared"}) {
    ScopedMatcherEnvironment certificate("FORM_CUDA_MATCH_CERTIFICATE",variant);
    CudaMatcher matcher;
    matcher.reset({{{0,0,0},0,2}},{{{0,0,0,0},{},{}},{{1e-100,0,0,0},{},{}}},{{{0,0,0,0}}},1.);
    matcher.search(identity_pose); auto pose=identity_pose; pose[3]=1e-102; matcher.search(pose);
    EXPECT_EQ(matcher.reuseStats().certified,std::string(variant)=="squared"?0:1);
  }
  ScopedMatcherEnvironment invalid("FORM_CUDA_MATCH_CERTIFICATE","invalid");
  EXPECT_THROW(CudaMatcher{},std::invalid_argument);
}


TEST(CudaMatcher, SummaryRowQueueHandlesMixedPartialWarpsAndSlotLifetimes) {
  for(bool plane:{false,true}) for(int backend:{0,1,2}) {
    CudaMatcher full,queued;
    full.setIncrementalSummaries(false);
    queued.setIncrementalSummaries(true); queued.setSummaryTree(backend!=0);
    queued.setSummaryTreeBounds(backend==2); queued.setSummaryRowQueue(true);
    for(int count:{0,31,33,64,257,777,33}) {
      std::vector<CudaMatcher::MapPoint> points;
      std::vector<CudaMatcher::Query> queries;
      std::vector<int> groups;
      for(int i=0;i<count;++i) {
        const double x=2.*i+.1; const int group=count==64?0:i%3;
        queries.push_back({{x,.1,.1,0}});
        points.push_back({{x,.1,.1,0},{x-.02,.15,.2},{.3,.4,.5}}); groups.push_back(group);
        if(i%7==0 || i%11==0) {
          points.push_back({{x+.3,.1,.1,0},{x+.27,.2,.25},{.5,.3,.4}});
          groups.push_back(i%11==0?(group+1)%3:group);
        }
      }
      const std::vector<CudaMatcher::Voxel> voxels={{{0,0,0},0,int(points.size())}};
      auto reset=[&] {
        for(auto* m:{&full,&queued}) { m->reset(voxels,points,queries,2000.); m->setGroups(groups,4); }
      };
      reset();
      auto compare=[&](double shift,double gate,bool active) {
        auto pose=identity_pose; pose[3]=shift;
        const auto a=queued.searchGrouped(pose,gate,plane),b=full.searchGrouped(pose,gate,plane);
        ASSERT_EQ(a.counts,b.counts); EXPECT_EQ(queued.summaryStats().queued,active);
        for(size_t g=0;g<a.roots.size();++g) {
          const Eigen::MatrixXd aa=a.roots[g].transpose()*a.roots[g],bb=b.roots[g].transpose()*b.roots[g];
          EXPECT_LE((aa-bb).norm(),1e-11*(1+bb.norm()));
          Eigen::VectorXd probe=Eigen::VectorXd::LinSpaced(a.roots[g].cols(),-.4,.6);
          EXPECT_NEAR((a.roots[g]*probe).squaredNorm(),(b.roots[g]*probe).squaredNorm(),1e-10*(1+bb.norm()));
        }
        const auto ar=queued.downloadResults(),br=full.downloadResults();
        ASSERT_EQ(ar.size(),br.size());
        for(size_t i=0;i<ar.size();++i) { EXPECT_EQ(ar[i].index,br[i].index); EXPECT_EQ(ar[i].distance,br[i].distance); }
      };
      for(int iteration=0;iteration<9;++iteration) compare(iteration%3?.2:0.,iteration%5==4?1e-5:1.,true);
      compare(0.,1.,true); compare(0.,1.,true); EXPECT_EQ(queued.summaryStats().dirty_leaves,0u);
      queued.setSummaryRowQueue(false); compare(.2,1.,false);
      queued.setSummaryRowQueue(true); compare(.2,1.,true);
      EXPECT_THROW(queued.searchGrouped(identity_pose,0.,plane),std::invalid_argument);
      compare(0.,1.,true);
      for(auto& p:points) { p.local[0]+=.17; p.normal[1]*=.7; }
      reset(); compare(.2,1.,true);
      groups.assign(points.size(),-1);
      for(auto* m:{&full,&queued}) m->setGroups(groups,4);
      compare(0.,1.,true);
    }
    queued.reset({}, {},std::vector<CudaMatcher::Query>(33),1.);
    queued.setGroups({},0);
    const auto empty=queued.searchGrouped(identity_pose,1.,plane);
    EXPECT_TRUE(empty.counts.empty()); EXPECT_TRUE(empty.roots.empty());
    EXPECT_TRUE(queued.summaryStats().queued);
  }
}

TEST(CudaMatcher, SummaryRowQueueEnvironmentAuditAndObserverFallback) {
  ScopedMatcherEnvironment reuse("FORM_CUDA_SUMMARY_REUSE","audit-tree64");
  ScopedMatcherEnvironment rows("FORM_CUDA_SUMMARY_ROWS","queue");
  CudaMatcher matcher;
  matcher.reset({{{0,0,0},0,2}},{{{.1,.1,.1,0},{.1,.1,.1},{.3,.4,.5}},{{.8,.1,.1,0},{.8,.1,.1},{.5,.4,.3}}},
    {{{.1,.1,.1,0}},{{.8,.1,.1,0}},{{.12,.1,.1,0}}},1.);
  matcher.setGroups({0,1},2);
  for(bool plane:{false,true}) {
    matcher.searchGrouped(identity_pose,1.,plane);
    EXPECT_TRUE(matcher.summaryStats().queued); EXPECT_EQ(matcher.summaryStats().full_rebuild_checks,2u);
    {
      GroupedInputCapture capture;
      matcher.searchGrouped(identity_pose,1.,plane);
      EXPECT_FALSE(matcher.summaryStats().queued);
      ASSERT_EQ(grouped_inputs.size(),2u); EXPECT_EQ(grouped_inputs[0].rows(),2);
      EXPECT_EQ(grouped_inputs[0](0,plane?0:4),plane?.03:0.);
      EXPECT_NEAR(grouped_inputs[0](1,plane?0:4),plane?.036:.02,1e-15);
    }
    matcher.searchGrouped(identity_pose,1.,plane); EXPECT_TRUE(matcher.summaryStats().queued);
  }
  { ScopedMatcherEnvironment sorted("FORM_CUDA_SUMMARY_ROWS","sorted"); CudaMatcher valid; }
  { ScopedMatcherEnvironment invalid("FORM_CUDA_SUMMARY_ROWS","invalid"); EXPECT_THROW(CudaMatcher{},std::invalid_argument); }
  matcher.setIncrementalSummaries(false);
  matcher.searchGrouped(identity_pose,1.,false); EXPECT_FALSE(matcher.summaryStats().queued);
}
