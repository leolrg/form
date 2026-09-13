#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/cuda_qr.hpp"
#include "form/feature/features.hpp"
#include "form/mapping/map.hpp"
#include <gtest/gtest.h>
#include <random>

using form::CudaMatcher;

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
