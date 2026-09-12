#include "form/feature/cuda_qr.hpp"
#include <gtest/gtest.h>
#include <random>
#include <array>

TEST(CudaQr, RaggedBatchesPreserveQuadraticFormsAndBufferReuse) {
  std::mt19937 rng(42); std::normal_distribution<double> normal;
  for(const auto config : {std::array<int,3>{32,32,128}, {32,64,64}, {64,64,64}, {64,64,128}}) {
  form::BatchedCudaQr qr(true,config[0],config[1],config[2]);
  for (int repeat=0; repeat<3; ++repeat) {
    std::vector<Eigen::MatrixXd> inputs;
    for (int width : {7,13}) for (int rows : {0,1,6,7,12,13,31,32,33,64,513,4097}) {
      Eigen::MatrixXd a(rows,width);
      for (int j=0;j<width;++j) for(int i=0;i<rows;++i) a(i,j)=normal(rng);
      if (repeat==1 && rows) a.col(width-1)=a.col(0);
      inputs.push_back(a);
    }
    auto roots=qr.compute(inputs);
    ASSERT_EQ(roots.size(),inputs.size());
    for(size_t k=0;k<inputs.size();++k) {
      auto actual=(roots[k].transpose()*roots[k]).eval();
      auto expected=(inputs[k].transpose()*inputs[k]).eval();
      EXPECT_LE((actual-expected).norm(),1e-10*(1+expected.norm()));
      EXPECT_TRUE(roots[k].allFinite());
    }
  }
  }
}
TEST(CudaQr, PreservesSmallDifferenceFeatureResiduals) {
  form::BatchedCudaQr qr;
  Eigen::MatrixXd a(1234,7);
  for(int i=0;i<a.rows();++i) a.row(i)<<1.,1e4+i,2e4-i,3e4,1e-6,-2e-6,3e-6;
  Eigen::VectorXd c=Eigen::VectorXd::Zero(7); c.tail(3)<<1.,2.,3.;
  auto root=qr.compute({a})[0];
  EXPECT_NEAR((root*c).squaredNorm(),(a*c).squaredNorm(),1e-20);
}
TEST(CudaQr, EmptyBatchAndInvalidWidth) {
  form::BatchedCudaQr qr;
  EXPECT_TRUE(qr.compute({}).empty());
  EXPECT_THROW(qr.compute({Eigen::MatrixXd::Zero(5,9)}),std::invalid_argument);
}

TEST(CudaQr, SmallFiniteInputsRemainFinite) {
  std::mt19937 rng(94); std::normal_distribution<double> normal;
  for(int tile : {32,64}) {
  form::BatchedCudaQr qr(true,tile,tile,64);
  for(int width : {7,13}) {
    Eigen::MatrixXd a(64,width);
    for(int j=0;j<width;++j) for(int i=0;i<a.rows();++i) a(i,j)=normal(rng);
    const auto root=qr.compute({(a*1e-155).eval()})[0];
    ASSERT_TRUE(root.allFinite());
    const Eigen::MatrixXd rescaled=root/1e-155;
    const Eigen::MatrixXd expected=a.transpose()*a;
    EXPECT_LE((rescaled.transpose()*rescaled-expected).norm(),1e-10*expected.norm());
  }
  }
}

TEST(CudaQr, DirectCorrespondencesPreserveFeaturesAndSmallResiduals) {
  std::mt19937 rng(51); std::normal_distribution<double> normal;
  for(int tile : {32,64}) {
  form::BatchedCudaQr qr(true,tile,tile,64);
  for(int rows : {0,1,13,31,32,33,51,52,53,63,64,65,127,128,129,255,256,257,4097}) {
    std::vector<double> pi(3*rows),pj(3*rows),n(3*rows);
    Eigen::MatrixXd point(rows,7),plane(rows,13);
    for(int i=0;i<rows;++i) {
      for(int j=0;j<3;++j) {
        pi[3*i+j]=1e4+normal(rng);
        pj[3*i+j]=pi[3*i+j]+1e-6*normal(rng);
        n[3*i+j]=normal(rng);
      }
      const Eigen::Map<const Eigen::Vector3d> p(pi.data()+3*i),q(pj.data()+3*i),v(n.data()+3*i);
      point(i,0)=1.; point.block<1,3>(i,1)=p.transpose(); point.block<1,3>(i,4)=(q-p).transpose();
      for(int j=0;j<3;++j) plane.block<1,3>(i,3*j)=v[j]*q.transpose();
      plane.block<1,3>(i,9)=v.transpose(); plane(i,12)=v.dot(q-p);
    }
    const auto roots=qr.computeCorrespondences({{pi.data(),pj.data(),n.data(),size_t(rows),true},
                                              {pi.data(),pj.data(),nullptr,size_t(rows),false}});
    ASSERT_EQ(roots.size(),2);
    const std::vector<Eigen::MatrixXd> expanded{plane,point};
    for(size_t k=0;k<2;++k) {
      ASSERT_TRUE(roots[k].allFinite());
      const Eigen::MatrixXd gram=expanded[k].transpose()*expanded[k];
      EXPECT_LE((roots[k].transpose()*roots[k]-gram).norm(),1e-10*(1.+gram.norm()));
      Eigen::VectorXd c=Eigen::VectorXd::Zero(expanded[k].cols()); c.tail(k==0?1:3).setOnes();
      EXPECT_NEAR((roots[k]*c).squaredNorm(),(expanded[k]*c).squaredNorm(),1e-18);
    }
  }
  }
}
