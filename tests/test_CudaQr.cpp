#include "form/feature/cuda_qr.hpp"
#include <gtest/gtest.h>
#include <random>
#include <array>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

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


namespace {
struct DevicePackedFixture {
  double* device = nullptr;
  double* pinned = nullptr;
  cudaStream_t producer = nullptr;
  explicit DevicePackedFixture(size_t count) {
    if(cudaMalloc(reinterpret_cast<void**>(&device),count*sizeof(double))!=cudaSuccess ||
       cudaMallocHost(reinterpret_cast<void**>(&pinned),count*sizeof(double))!=cudaSuccess ||
       cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking)!=cudaSuccess)
      throw std::runtime_error("CUDA test allocation failed");
  }
  ~DevicePackedFixture() {
    if(producer) cudaStreamSynchronize(producer);
    cudaFree(device); cudaFreeHost(pinned); cudaStreamDestroy(producer);
  }
};
std::vector<Eigen::MatrixXd> device_observed;
void observeDeviceInput(const std::vector<Eigen::MatrixXd>& input) { device_observed=input; }
struct ObserveDeviceInput {
  ObserveDeviceInput() { form::BatchedCudaQr::setInputObserver(observeDeviceInput); }
  ~ObserveDeviceInput() { form::BatchedCudaQr::setInputObserver(nullptr); }
};
}

TEST(CudaQr, DevicePackedRaggedBatchesPreserveInputAndCostsAcrossGrowthAndShrink) {
  std::mt19937 rng(731); std::normal_distribution<double> normal;
  for(int tile : {32,64}) {
    form::BatchedCudaQr qr(true,tile,tile,64);
    for(int largest : {65,4097,1,513}) {
      std::vector<form::BatchedCudaQr::DeviceInput> shapes;
      std::vector<Eigen::MatrixXd> expected;
      std::vector<double> packed;
      for(bool plane : {false,true}) for(int rows : {0,1,7,13,31,32,33,63,64,65,largest}) {
        Eigen::MatrixXd compact(rows,7), expanded(rows,plane?13:7);
        for(int i=0;i<rows;++i) {
          Eigen::Vector3d p,n,delta;
          for(int j=0;j<3;++j) { p[j]=1e4+normal(rng); n[j]=normal(rng); delta[j]=1e-6*normal(rng); }
          if(plane) {
            compact.block<1,3>(i,0)=p.transpose(); compact.block<1,3>(i,3)=n.transpose();
            compact(i,6)=n.dot(delta);
            for(int j=0;j<3;++j) expanded.block<1,3>(i,3*j)=n[j]*p.transpose();
            expanded.block<1,3>(i,9)=n.transpose(); expanded(i,12)=n.dot(delta);
          } else {
            compact(i,0)=1.; compact.block<1,3>(i,1)=p.transpose(); compact.block<1,3>(i,4)=delta.transpose();
            expanded.row(i)=compact.row(i);
          }
        }
        shapes.push_back({size_t(rows),plane}); expected.push_back(expanded);
        if(compact.size()) packed.insert(packed.end(),compact.data(),compact.data()+compact.size());
      }
      DevicePackedFixture memory(packed.size());
      std::memcpy(memory.pinned,packed.data(),packed.size()*sizeof(double));
      ASSERT_EQ(cudaMemcpyAsync(memory.device,memory.pinned,packed.size()*sizeof(double),cudaMemcpyHostToDevice,memory.producer),cudaSuccess);
      auto roots=qr.computeDevicePacked(memory.device,shapes,memory.producer);
      ASSERT_EQ(roots.size(),expected.size());
      for(size_t i=0;i<roots.size();++i) {
        ASSERT_EQ(roots[i].rows(),expected[i].cols()); ASSERT_TRUE(roots[i].allFinite());
        const Eigen::MatrixXd gram=expected[i].transpose()*expected[i];
        EXPECT_LE((roots[i].transpose()*roots[i]-gram).norm(),1e-10*(1.+gram.norm()));
        Eigen::VectorXd c=Eigen::VectorXd::Zero(expected[i].cols()); c.tail(shapes[i].plane?1:3).setOnes();
        EXPECT_NEAR((roots[i]*c).squaredNorm(),(expected[i]*c).squaredNorm(),1e-18);
      }
      std::vector<double> unchanged(packed.size());
      ASSERT_EQ(cudaMemcpy(unchanged.data(),memory.device,packed.size()*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
      EXPECT_EQ(std::memcmp(unchanged.data(),packed.data(),packed.size()*sizeof(double)),0);
    }
  }
}

TEST(CudaQr, DevicePackedWaitsForProducerAndSupportsDefaultStream) {
  form::BatchedCudaQr qr;
  Eigen::MatrixXd input=Eigen::MatrixXd::Random(129,7);
  DevicePackedFixture memory(input.size());
  std::memcpy(memory.pinned,input.data(),input.size()*sizeof(double));
  ASSERT_EQ(cudaMemset(memory.device,0,input.size()*sizeof(double)),cudaSuccess);
  // Warm scratch allocations before the delayed producer, so allocation cannot
  // accidentally provide the ordering that this test requires from the API.
  qr.computeDevicePacked(memory.device,{{129,false}},nullptr);
  std::atomic<bool> produced{false};
  ASSERT_EQ(cudaLaunchHostFunc(memory.producer,[](void* state) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    static_cast<std::atomic<bool>*>(state)->store(true);
  },&produced),cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(memory.device,memory.pinned,input.size()*sizeof(double),cudaMemcpyHostToDevice,memory.producer),cudaSuccess);
  const auto root=qr.computeDevicePacked(memory.device,{{129,false}},memory.producer)[0];
  EXPECT_TRUE(produced.load());
  EXPECT_EQ(cudaStreamQuery(memory.producer),cudaSuccess);
  const Eigen::MatrixXd gram=input.transpose()*input;
  EXPECT_LE((root.transpose()*root-gram).norm(),1e-10*gram.norm());
  ASSERT_EQ(cudaMemcpyAsync(memory.device,memory.pinned,input.size()*sizeof(double),cudaMemcpyHostToDevice,nullptr),cudaSuccess);
  const auto default_root=qr.computeDevicePacked(memory.device,{{129,false}},nullptr)[0];
  EXPECT_LE((default_root.transpose()*default_root-gram).norm(),1e-10*gram.norm());
}

TEST(CudaQr, DevicePackedEmptyInputValidationAndDiagnosticCapture) {
  form::BatchedCudaQr qr;
  EXPECT_TRUE(qr.computeDevicePacked(nullptr,{},nullptr).empty());
  auto roots=qr.computeDevicePacked(nullptr,{{0,false},{0,true}},nullptr);
  ASSERT_EQ(roots.size(),2); EXPECT_TRUE(roots[0].isZero()); EXPECT_TRUE(roots[1].isZero());
  EXPECT_THROW(qr.computeDevicePacked(nullptr,{{1,false}},nullptr),std::invalid_argument);
  EXPECT_THROW(qr.computeDevicePacked(nullptr,{{std::numeric_limits<size_t>::max(),true}},nullptr),std::invalid_argument);
  DevicePackedFixture memory(14);
  Eigen::Matrix<double,2,7> compact;
  compact << 1.,2.,3.,4.,5.,6.,1e-6, 7.,8.,9.,10.,11.,12.,-2e-6;
  std::memcpy(memory.pinned,compact.data(),14*sizeof(double));
  ASSERT_EQ(cudaMemcpyAsync(memory.device,memory.pinned,14*sizeof(double),cudaMemcpyHostToDevice,memory.producer),cudaSuccess);
  ObserveDeviceInput observer;
  qr.computeDevicePacked(memory.device,{{2,true}},memory.producer);
  ASSERT_EQ(device_observed.size(),1);
  Eigen::MatrixXd expanded(2,13);
  for(int a=0;a<3;++a) for(int b=0;b<3;++b) expanded.col(3*a+b)=compact.col(3+a).cwiseProduct(compact.col(b));
  expanded.middleCols(9,3)=compact.middleCols(3,3); expanded.col(12)=compact.col(6);
  EXPECT_TRUE(device_observed[0].isApprox(expanded,0.));
  qr.computeDevicePacked(memory.device,{{2,false}},memory.producer);
  ASSERT_EQ(device_observed.size(),1); EXPECT_TRUE(device_observed[0].isApprox(compact,0.));
}

namespace {
struct IncrementalFixture : DevicePackedFixture {
  unsigned char* dirty=nullptr;
  std::vector<unsigned char> host_dirty;
  explicit IncrementalFixture(size_t leaves):DevicePackedFixture(std::max<size_t>(1,leaves*64*7)),host_dirty(leaves,1) {
    if(cudaMalloc(reinterpret_cast<void**>(&dirty),std::max<size_t>(1,leaves))!=cudaSuccess)
      throw std::runtime_error("CUDA dirty allocation failed");
    std::fill(pinned,pinned+leaves*64*7,0.);
  }
  ~IncrementalFixture() { cudaStreamSynchronize(producer); cudaFree(dirty); }
  void upload() {
    if(cudaMemcpyAsync(device,pinned,host_dirty.size()*64*7*sizeof(double),cudaMemcpyHostToDevice,producer)!=cudaSuccess ||
       cudaMemcpyAsync(dirty,host_dirty.data(),host_dirty.size(),cudaMemcpyHostToDevice,producer)!=cudaSuccess)
      throw std::runtime_error("CUDA incremental upload failed");
  }
  void leaf(size_t index,const Eigen::MatrixXd& matrix) { std::memcpy(pinned+index*64*7,matrix.data(),64*7*sizeof(double)); }
};
Eigen::MatrixXd expandLeaf(const Eigen::MatrixXd& compact,bool plane) {
  if(!plane) return compact;
  Eigen::MatrixXd expanded(compact.rows(),13);
  for(int a=0;a<3;++a) for(int b=0;b<3;++b)
    expanded.col(3*a+b)=compact.col(3+a).cwiseProduct(compact.col(b));
  expanded.middleCols(9,3)=compact.middleCols(3,3); expanded.col(12)=compact.col(6);
  return expanded;
}
void expectIncrementalGram(const Eigen::MatrixXd& root,const std::vector<Eigen::MatrixXd>& leaves,bool plane) {
  Eigen::MatrixXd expected=Eigen::MatrixXd::Zero(plane?13:7,plane?13:7);
  for(const auto& compact:leaves) { const auto expanded=expandLeaf(compact,plane); expected+=expanded.transpose()*expanded; }
  ASSERT_TRUE(root.allFinite());
  EXPECT_LE((root.transpose()*root-expected).norm(),2e-11*(1.+expected.norm()));
}
}

TEST(CudaQr, IncrementalDirtyLeafRetainsUntouchedRootsAndSkipsUnchangedGroups) {
  for(bool plane:{false,true}) {
    form::BatchedCudaQr qr;
    const size_t capacity=10;
    IncrementalFixture memory(2*capacity);
    std::vector<Eigen::MatrixXd> leaves;
    for(size_t i=0;i<2*capacity;++i) {
      leaves.push_back(Eigen::MatrixXd::Random(64,7)); memory.leaf(i,leaves.back());
    }
    memory.upload();
    auto roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,capacity,{10,2},plane,memory.producer);
    ASSERT_EQ(roots.size(),2);
    expectIncrementalGram(roots[0],{leaves.begin(),leaves.begin()+10},plane);
    expectIncrementalGram(roots[1],{leaves.begin()+10,leaves.begin()+12},plane);
    EXPECT_EQ(qr.incrementalStats().dirty_leaves,12); EXPECT_EQ(qr.incrementalStats().active_leaves,12);
    const Eigen::MatrixXd unchanged=roots[1];
    leaves[1]*=2.; memory.leaf(1,leaves[1]);
    // Clean leaves must not be read from the producer again, even in the same group.
    for(size_t i=0;i<2*capacity;++i) if(i!=1) memory.leaf(i,Eigen::MatrixXd::Constant(64,7,std::numeric_limits<double>::quiet_NaN()));
    std::fill(memory.host_dirty.begin(),memory.host_dirty.end(),0); memory.host_dirty[1]=1; memory.upload();
    roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,capacity,{10,2},plane,memory.producer);
    expectIncrementalGram(roots[0],{leaves.begin(),leaves.begin()+10},plane);
    EXPECT_TRUE(roots[1].isApprox(unchanged,0.));
    EXPECT_EQ(qr.incrementalStats().dirty_leaves,1); EXPECT_EQ(qr.incrementalStats().changed_groups,1);
    EXPECT_GT(qr.incrementalStats().merge_tiles,0);
    memory.host_dirty[1]=0; memory.upload();
    qr.computeDevicePackedIncremental(memory.device,memory.dirty,capacity,{10,2},plane,memory.producer);
    EXPECT_EQ(qr.incrementalStats().dirty_leaves,0); EXPECT_EQ(qr.incrementalStats().merge_tiles,0);
    EXPECT_EQ(qr.incrementalStats().changed_groups,0);
  }
}

TEST(CudaQr, IncrementalInsertionDeletionMigrationAndRankDeficiency) {
  for(bool plane:{false,true}) {
    form::BatchedCudaQr qr;
    IncrementalFixture memory(6);
    Eigen::MatrixXd a=Eigen::MatrixXd::Zero(64,7),b=a;
    a.row(2)<<1.,2.,3.,4.,.1,.2,.3; a.row(7)=a.row(2); b.row(40)=a.row(2)*2.;
    memory.leaf(0,a); memory.leaf(1,b); memory.upload();
    auto roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,3,{2,0},plane,memory.producer);
    expectIncrementalGram(roots[0],{a,b},plane); EXPECT_TRUE(roots[1].isZero());
    const Eigen::MatrixXd zero=Eigen::MatrixXd::Zero(64,7);
    memory.leaf(0,zero); memory.leaf(3,a);
    memory.host_dirty={1,0,0,1,0,0}; memory.upload();
    roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,3,{2,1},plane,memory.producer);
    expectIncrementalGram(roots[0],{b},plane); expectIncrementalGram(roots[1],{a},plane);
    EXPECT_EQ(qr.incrementalStats().dirty_leaves,2);
    memory.host_dirty.assign(6,0); memory.upload();
    roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,3,{0,0},plane,memory.producer);
    EXPECT_TRUE(roots[0].isZero()); EXPECT_TRUE(roots[1].isZero()); EXPECT_EQ(qr.incrementalStats().changed_groups,2);
    roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,3,{2,1},plane,memory.producer);
    expectIncrementalGram(roots[0],{b},plane); expectIncrementalGram(roots[1],{a},plane);
    EXPECT_EQ(qr.incrementalStats().dirty_leaves,0); EXPECT_EQ(qr.incrementalStats().changed_groups,2);
  }
}

TEST(CudaQr, IncrementalResetConfigurationValidationAndProducerOrdering) {
  form::BatchedCudaQr qr;
  IncrementalFixture memory(2);
  const Eigen::MatrixXd a=Eigen::MatrixXd::Random(64,7);
  memory.leaf(0,a); memory.leaf(1,a); memory.upload();
  qr.computeDevicePackedIncremental(memory.device,memory.dirty,2,{2},false,memory.producer);
  memory.host_dirty={1,0};
  ASSERT_EQ(cudaMemcpyAsync(memory.dirty,memory.host_dirty.data(),2,cudaMemcpyHostToDevice,memory.producer),cudaSuccess);
  std::atomic<bool> produced{false};
  ASSERT_EQ(cudaLaunchHostFunc(memory.producer,[](void* p) {
    std::this_thread::sleep_for(std::chrono::milliseconds(30)); static_cast<std::atomic<bool>*>(p)->store(true);
  },&produced),cudaSuccess);
  const Eigen::MatrixXd changed=(a*3.).eval(); memory.leaf(0,changed);
  // Only pinned asynchronous input follows the delay; no pageable upload or
  // fresh allocation may accidentally satisfy the required producer ordering.
  ASSERT_EQ(cudaMemcpyAsync(memory.device,memory.pinned,2*64*7*sizeof(double),cudaMemcpyHostToDevice,memory.producer),cudaSuccess);
  auto roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,2,{2},false,memory.producer);
  EXPECT_TRUE(produced.load()); EXPECT_EQ(cudaStreamQuery(memory.producer),cudaSuccess);
  expectIncrementalGram(roots[0],{changed,a},false);
  memory.host_dirty={0,0}; memory.upload(); qr.resetIncremental();
  roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,2,{2},false,memory.producer);
  EXPECT_TRUE(roots[0].isZero());
  memory.host_dirty={1,1}; memory.upload();
  roots=qr.computeDevicePackedIncremental(memory.device,memory.dirty,2,{2},true,memory.producer);
  expectIncrementalGram(roots[0],{changed,a},true);
  EXPECT_THROW(qr.computeDevicePackedIncremental(memory.device,memory.dirty,1,{2},true,memory.producer),std::invalid_argument);
  EXPECT_THROW(qr.computeDevicePackedIncremental(nullptr,memory.dirty,2,{1},false),std::invalid_argument);
  EXPECT_THROW(qr.computeDevicePackedIncremental(memory.device,nullptr,2,{1},false),std::invalid_argument);
  EXPECT_THROW(qr.computeDevicePackedIncremental(memory.device,memory.dirty,std::numeric_limits<size_t>::max(),{1,1},false),std::invalid_argument);
  EXPECT_TRUE(qr.computeDevicePackedIncremental(nullptr,nullptr,0,{},false).empty());
  roots=qr.computeDevicePackedIncremental(nullptr,nullptr,0,{0,0},false);
  ASSERT_EQ(roots.size(),2); EXPECT_TRUE(roots[0].isZero()); EXPECT_TRUE(roots[1].isZero());
}
