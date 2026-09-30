#include "form/feature/cuda_extraction.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
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

namespace {
template<class T> void checkNormals() {
  form::CudaExtraction gpu;
  constexpr int cols=67, neighbors=5;
  for (int rows : {1, 3, 7, 2}) {
    std::vector<std::array<T,4>> scan(rows*cols);
    std::vector<unsigned char> valid(scan.size(),1);
    for(int row=0;row<rows;++row) for(int col=0;col<cols;++col) {
      const T y=T(0.03*col), z=T(0.07*row);
      scan[row*cols+col]={T(5)+T(.2)*y+T(.3)*z+T(.002)*T(std::sin(col*.71+row)),y,z,T(0)};
      if(col<neighbors || col>=cols-neighbors || (rows==7 && row==3)) valid[row*cols+col]=0;
    }
    gpu.prepare(scan,valid,cols,neighbors);
    std::vector<size_t> queries;
    for(size_t i=0;i<scan.size();++i) if(valid[i]) queries.push_back(i);
    for(double radius : {0., .04, .5}) {
      const auto actual=gpu.normals(queries,radius,5);
      ASSERT_EQ(actual.size(),queries.size());
      for(size_t q=0;q<queries.size();++q) {
        const int idx=int(queries[q]);
        auto point=[&](int i) { return Eigen::Map<const Eigen::Matrix<T,4,1>>(scan[i].data()).eval(); };
        std::vector<Eigen::Matrix<T,3,1>> neighborhood;
        auto gather=[&](int center) {
          for(int sign : {1,-1}) for(int n=1;n<=neighbors;++n) {
            const int i=center+sign*n;
            if(double((point(i)-point(center)).squaredNorm())>=radius*radius) break;
            neighborhood.push_back(point(i).template head<3>());
          }
        };
        gather(idx);
        bool other=false;
        for(int row : {idx/cols-1,idx/cols+1}) {
          if(row<0 || row>=rows) continue;
          double best=std::numeric_limits<double>::max(); int winner=-1;
          for(int col=0;col<cols;++col) {
            const int i=row*cols+col;
            if(!valid[i]) continue;
            const double d=(point(i)-point(idx)).squaredNorm();
            if(d<best) { best=d; winner=i; }
          }
          if(winner>=0) { other=true; neighborhood.push_back(point(winner).template head<3>()); gather(winner); }
        }
        const bool accepted=other && neighborhood.size()>=5;
        ASSERT_EQ(actual[q][3],accepted ? 1. : 0.);
        if(!accepted) continue;
        Eigen::Matrix<T,Eigen::Dynamic,3> a(neighborhood.size(),3);
        for(size_t j=0;j<neighborhood.size();++j) a.row(j)=neighborhood[j]-point(idx).template head<3>();
        a/=T(neighborhood.size());
        const Eigen::Matrix<T,3,3> cov=a.transpose()*a;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,3,3>> eig(cov);
        Eigen::Vector3d got(actual[q][0],actual[q][1],actual[q][2]);
        const Eigen::Vector3d expected=eig.eigenvectors().col(0).template cast<double>().normalized();
        const double tolerance=sizeof(T)==4 ? 2e-4 : 1e-10;
        EXPECT_NEAR(got.norm(),1.,tolerance);
        EXPECT_LT(std::min((got-expected).norm(),(got+expected).norm()),tolerance);
        const Eigen::Matrix3d c=cov.template cast<double>();
        EXPECT_LE((c*got-got.dot(c*got)*got).norm(),tolerance*c.norm());
      }
    }
    EXPECT_TRUE(gpu.normals({},.5,5).empty());
    const auto rejected=gpu.normals(queries,.5,10000);
    for(const auto& n:rejected) EXPECT_EQ(n[3],0.);
  }
}
}
TEST(CudaExtraction, FloatNormalsMatchCpuNeighborhoodAndEigen) { checkNormals<float>(); }
TEST(CudaExtraction, DoubleNormalsMatchCpuNeighborhoodAndEigen) { checkNormals<double>(); }
TEST(CudaExtraction, NormalArgumentsAndPreparationAreValidated) {
  form::CudaExtraction gpu;
  EXPECT_THROW(gpu.normals({0},1.,5),std::logic_error);
  std::vector<std::array<float,4>> scan(32,{1,2,3,0});
  std::vector<unsigned char> valid(32,1);
  gpu.prepare(scan,valid,16,2);
  EXPECT_THROW(gpu.normals({32},1.,5),std::out_of_range);
  EXPECT_THROW(gpu.normals({8},-1.,5),std::invalid_argument);
  EXPECT_THROW(gpu.normals({8},std::numeric_limits<double>::infinity(),5),std::invalid_argument);
  EXPECT_THROW(gpu.prepare(scan,valid,17,2),std::invalid_argument);
  EXPECT_THROW(gpu.normals({8},1.,5),std::logic_error);
}

TEST(CudaExtraction, NormalRadiusUsesCallerReductionAndStrictComparison) {
  form::CudaExtraction gpu;
  std::vector<std::array<float,4>> scan(16,{1e6,1e6,1e6,0});
  std::vector<unsigned char> valid(16,0);
  scan[2]=scan[10]={0,0,0,0}; valid[2]=valid[10]=1;
  scan[3]={1,1,4096,0};
  for(auto reduction : {form::CudaExtraction::Reduction::Cross,
                        form::CudaExtraction::Reduction::Adjacent,
                        form::CudaExtraction::Reduction::Sequential}) {
    gpu.prepare(scan,valid,8,1,reduction);
    auto n=gpu.normals({2},std::sqrt(16777217.),2);
    ASSERT_EQ(n.size(),1);
    EXPECT_EQ(n[0][3],reduction==form::CudaExtraction::Reduction::Cross ? 1. : 0.);
  }
  scan[3]={0,0,0,1};
  gpu.prepare(scan,valid,8,1);
  EXPECT_EQ(gpu.normals({2},1.,2)[0][3],0.); // Equal radius excluded, including w.
  EXPECT_EQ(gpu.normals({2},1.01,2)[0][3],1.);
}

TEST(CudaExtraction, DegenerateScaledTranslatedAndChangingDtypeNormals) {
  form::CudaExtraction gpu;
  constexpr int cols=37, rows=3;
  std::vector<unsigned char> valid(cols*rows,1);
  for(int i=0;i<cols*rows;++i) if(i%cols<3 || i%cols>=cols-3) valid[i]=0;
  for(double scale : {1e-10,1.,1e10}) {
    std::vector<std::array<double,4>> scan(cols*rows);
    for(int i=0;i<cols*rows;++i) scan[i]={1e5*scale,(i%cols)*.03*scale,(i/cols)*.07*scale,0};
    gpu.prepare(scan,valid,cols,3);
    const auto n=gpu.normals({3,20,40,90},scale,5);
    for(const auto& p:n) {
      ASSERT_EQ(p[3],1.); EXPECT_NEAR(std::abs(p[0]),1.,1e-12);
      EXPECT_NEAR(p[1],0.,1e-12); EXPECT_NEAR(p[2],0.,1e-12);
    }
    std::vector<std::array<float,4>> flat(cols*rows,{3,3,3,0});
    gpu.prepare(flat,valid,cols,3);
    const auto degenerate=gpu.normals({3,20,40,90},1.,5);
    for(const auto& p:degenerate) {
      EXPECT_EQ(p[3],1.);
      EXPECT_NEAR(p[0]*p[0]+p[1]*p[1]+p[2]*p[2],1.,1e-6);
    }
  }
}

TEST(CudaExtraction, NormalQueriesHandleRowEdgesAndBlockTails) {
  form::CudaExtraction gpu;
  std::vector<std::array<float,4>> scan(74,{3,3,3,0});
  std::vector<unsigned char> valid(scan.size(),1);
  gpu.prepare(scan,valid,37,3);
  for(size_t count : {size_t(1),size_t(127),size_t(128),size_t(129),size_t(257)}) {
    std::vector<size_t> queries(count);
    for(size_t i=0;i<count;++i) queries[i]=i%2 ? 0 : 73;
    const auto output=gpu.normals(queries,1.,5);
    ASSERT_EQ(output.size(),count);
    for(const auto& n:output) {
      EXPECT_EQ(n[3],1.);
      EXPECT_NEAR(n[0]*n[0]+n[1]*n[1]+n[2]*n[2],1.,1e-6);
    }
  }
}
