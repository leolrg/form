#include "form/feature/cuda_extraction.hpp"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>

namespace {
template<class T> void checkScalar() {
  constexpr int cols=67, rows=4, neighbors=5;
  std::mt19937 engine(714);
  std::uniform_real_distribution<double> random(-5.,5.);
  std::vector<std::array<T,4>> points(cols*rows);
  std::vector<unsigned char> mask(points.size(),1);
  for (size_t i=0;i<points.size();++i) {
    for (auto& x:points[i]) x=T(random(engine));
    if (i%11==0 || i%cols<neighbors || i%cols>=cols-neighbors) mask[i]=0;
  }
  // Equal positions must choose the first valid index, including across lanes.
  points[cols+6]=points[cols+40]=points[20];
  mask[cols+6]=mask[cols+40]=1;
  form::CudaExtraction gpu;
  auto result=gpu.prepare(points,mask,cols,neighbors);
  ASSERT_EQ(result.size(),points.size());
  for (size_t i=0;i<points.size();++i) {
    double expected=std::numeric_limits<T>::max();
    if(mask[i]) {
      double dx=-2.*neighbors*points[i][0],dy=-2.*neighbors*points[i][1],dz=-2.*neighbors*points[i][2];
      for(int n=1;n<=neighbors;++n) {
        dx=dx+points[i-n][0]+points[i+n][0];
        dy=dy+points[i-n][1]+points[i+n][1];
        dz=dz+points[i-n][2]+points[i+n][2];
      }
      expected=T(dx*dx+dy*dy+dz*dz);
    }
    EXPECT_EQ(result[i],expected) << i;
  }
  const std::vector<size_t> queries{20,90,150,210,10,100,200};
  auto nearest=gpu.nearestRows(queries);
  for(size_t q=0;q<queries.size();++q) for(int direction=0;direction<2;++direction) {
    int row=int(queries[q]/cols)+(direction ? 1 : -1), expected=-1;
    double best=std::numeric_limits<double>::max();
    if(row>=0 && row<rows) for(int col=0;col<cols;++col) {
      const int i=row*cols+col;
      if(!mask[i]) continue;
      const Eigen::Map<const Eigen::Matrix<T,4,1>> p(points[i].data()), x(points[queries[q]].data());
      const double distance=(p-x).squaredNorm();
      if(distance<best) { best=distance; expected=i; }
    }
    EXPECT_EQ(nearest[q][direction],expected) << q << ',' << direction;
  }
  EXPECT_EQ(nearest[0][1],cols+6);
  EXPECT_TRUE(gpu.nearestRows({}).empty());
}
}
TEST(CudaExtraction, FloatMatchesEigenDistancesAndCurvature) { checkScalar<float>(); }
TEST(CudaExtraction, DoubleMatchesEigenDistancesAndCurvature) { checkScalar<double>(); }
TEST(CudaExtraction, InvalidPreparationBlocksSearchAndCanRecover) {
  form::CudaExtraction gpu;
  EXPECT_THROW(gpu.nearestRows({0}),std::logic_error);
  std::vector<std::array<float,4>> scan(32,{1,2,3,0});
  std::vector<unsigned char> valid(32,1);
  gpu.prepare(scan,valid,16,2);
  EXPECT_THROW(gpu.nearestRows({32}),std::out_of_range);
  EXPECT_THROW(gpu.prepare(scan,valid,17,2),std::invalid_argument);
  EXPECT_THROW(gpu.nearestRows({0}),std::logic_error);
  gpu.prepare(scan,valid,16,2);
  auto nearest=gpu.nearestRows({8,24});
  EXPECT_EQ(nearest[0][0],-1); EXPECT_EQ(nearest[0][1],16);
  EXPECT_EQ(nearest[1][0],0); EXPECT_EQ(nearest[1][1],-1);
  std::fill(valid.begin(),valid.end(),0);
  gpu.prepare(scan,valid,16,2);
  nearest=gpu.nearestRows({8,24});
  for(auto pair:nearest) EXPECT_EQ(pair,(std::array<int,2>{-1,-1}));
}

TEST(CudaExtraction, ReductionPolicyPreservesRoundingSensitiveTie) {
  form::CudaExtraction gpu;
  std::vector<std::array<float,4>> points(8,{0,0,0,0});
  std::vector<unsigned char> mask(8,0);
  points[4]={1,1,4096,0}; points[5]={0,0,4096,0};
  mask[4]=mask[5]=1;
  gpu.prepare(points,mask,4,0,form::CudaExtraction::Reduction::Cross);
  EXPECT_EQ(gpu.nearestRows({0})[0][1],4);
  gpu.prepare(points,mask,4,0,form::CudaExtraction::Reduction::Adjacent);
  EXPECT_EQ(gpu.nearestRows({0})[0][1],5);
  gpu.prepare(points,mask,4,0,form::CudaExtraction::Reduction::Sequential);
  EXPECT_EQ(gpu.nearestRows({0})[0][1],5);
}
