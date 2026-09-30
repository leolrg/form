#include "form/feature/cuda_extraction.hpp"
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>
#include <random>

TEST(CudaExtraction, SmallIllConditionedNeighborhoodMatchesCpuNormal) {
  // Hilti basement_2 frame 500: twelve displacements from a nearly collinear
  // neighborhood. Covariance rounding here changed later LM/rematch decisions.
  const std::array<float,4> displacements[] = {
    {0.017928361892700195f,0.0046586990356445312f,0.095769941806793213f,0},
    {-2.7003903388977051f,3.2431588172912598f,-0.67504322528839111f,0},
    {-2.7116873264312744f,3.248171329498291f,-0.67668610811233521f,0},
    {-2.7107043266296387f,3.2431786060333252f,-0.67613846063613892f,0},
    {-2.7157914638519287f,3.2432956695556641f,-0.67668610811233521f,0},
    {-2.7179701328277588f,3.2407326698303223f,-0.67668610811233521f,0},
    {-2.7170352935791016f,3.2357304096221924f,-0.67613846063613892f,0},
    {-2.7044451236724854f,3.2506871223449707f,-0.67613846063613892f,0},
    {-2.7116031646728516f,3.2606959342956543f,-0.67778134346008301f,0},
    {-2.7064189910888672f,3.2607772350311279f,-0.67723369598388672f,0},
    {-2.7013380527496338f,3.2607483863830566f,-0.67668610811233521f,0},
    {-2.6993370056152344f,3.263228178024292f,-0.67668610811233521f,0}
  };
  constexpr int cols=13, center=6, query=cols+center;
  std::vector<std::array<float,4>> scan(2*cols,{1e6f,1e6f,1e6f,0});
  std::vector<unsigned char> valid(scan.size(),0);
  scan[query]={0,0,0,0}; scan[query+1]=displacements[0];
  scan[center]=displacements[1]; valid[center]=1;
  for(int n=1;n<=5;++n) {
    scan[center+n]=displacements[1+n];
    scan[center-n]=displacements[6+n];
  }
  Eigen::Matrix<float,Eigen::Dynamic,3> a(12,3);
  for(int i=0;i<12;++i) for(int j=0;j<3;++j) a(i,j)=displacements[i][j];
  a/=12.f;
  const Eigen::Matrix3f covariance=a.transpose()*a;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance);
  Eigen::Vector3f expected=eig.eigenvectors().col(0);
  expected.normalize(); // Production normalizes in the point scalar type.
  form::CudaExtraction gpu;
  gpu.prepare(scan,valid,cols,5);
  const auto actual=gpu.normals({query},.5,12);
  ASSERT_EQ(actual.size(),1);
  ASSERT_EQ(actual[0][3],1.);
  for(int i=0;i<3;++i) EXPECT_EQ(actual[0][i],double(expected[i]));
}

namespace {
template<class T> void checkNeighborhoodSizes() {
  std::mt19937 random(530);
  std::uniform_real_distribution<double> noise(-.02,.02);
  form::CudaExtraction gpu;
  constexpr int cols=35, center=17, query=cols+center, neighbors=16;
  // Cover partial/full SIMD packets, scalar tails, and the switch to GEMM.
  for(int count=1;count<=33;++count) for(int trial=0;trial<12;++trial) {
    SCOPED_TRACE(::testing::Message()<<"count="<<count<<" trial="<<trial);
    std::vector<std::array<T,4>> scan(2*cols,{T(1e6),T(1e6),T(1e6),T(0)});
    std::vector<unsigned char> valid(scan.size(),0);
    scan[query]={0,0,0,0}; valid[center]=1;
    Eigen::Matrix<T,Eigen::Dynamic,3> a(count,3);
    for(int i=0;i<count;++i) {
      const T x=T(.3+noise(random));
      const T y=T(.6)*x+T(noise(random));
      const T z=T(.2)*x+T(.3)*y+T(noise(random)*(trial%2 ? .001 : 1.));
      const int index=i==0 ? center : (i<=neighbors ? center+i : center-(i-neighbors));
      scan[index]={x,y,z,0};
      a.row(i)<<x,y,z;
    }
    a/=T(count);
    const Eigen::Matrix<T,3,3> covariance=a.transpose()*a;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,3,3>> eig(covariance);
    Eigen::Matrix<T,3,1> expected=eig.eigenvectors().col(0); expected.normalize();
    gpu.prepare(scan,valid,cols,neighbors);
    const auto actual=gpu.normals({query},1.,count);
    ASSERT_EQ(actual[0][3],1.);
    EXPECT_EQ(gpu.normals({query},1.,count+1)[0][3],0.);
    for(int j=0;j<3;++j) EXPECT_EQ(actual[0][j],double(expected[j]));
  }
}
}
TEST(CudaExtraction, FloatNormalsPreserveCpuReductionAcrossNeighborhoodSizes) { checkNeighborhoodSizes<float>(); }
TEST(CudaExtraction, DoubleNormalsPreserveCpuReductionAcrossNeighborhoodSizes) { checkNeighborhoodSizes<double>(); }
