// MIT License; see factor.hpp for copyright and license text.
#include "form/feature/batch_summary.hpp"
#ifdef FORM_ENABLE_CUDA
#include "form/feature/batch_summary_cuda.hpp"
#endif
#include <tbb/parallel_for.h>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace form {
struct BatchSummary::Impl {
  int poses=0,n=0; bool gpu=false;
  std::vector<SummaryEdge> edges;
  std::vector<int> offsets,indices;
  std::vector<FeatureSummary::AugmentedHessian> partials;
  std::vector<double> costs;
#ifdef FORM_ENABLE_CUDA
  std::unique_ptr<CudaSummaryBatch> cuda;
  std::vector<BatchPose> packed;
#endif
  void check(const std::vector<gtsam::Pose3>& values) {
    if(values.size()!=static_cast<size_t>(poses)) throw std::invalid_argument("BatchSummary pose count mismatch");
#ifdef FORM_ENABLE_CUDA
    if(cuda) for(int i=0;i<poses;++i) {
      const auto r=values[i].rotation().matrix();
      for(int j=0;j<3;++j) {packed[i].translation[j]=values[i].translation()[j];
        for(int k=0;k<3;++k) packed[i].rotation[j*3+k]=r(j,k);}
    }
#endif
  }
};
BatchSummary::BatchSummary(int pose_count,std::vector<SummaryEdge> edges,bool gpu):impl_(std::make_unique<Impl>()) {
  impl_->gpu=gpu;reset(pose_count,std::move(edges));
}
void BatchSummary::reset(int pose_count,std::vector<SummaryEdge> edges) {
  if(pose_count<1 || pose_count>1000 || edges.size()>static_cast<size_t>(std::numeric_limits<int>::max()/169))
    throw std::invalid_argument("BatchSummary supports 1..1000 poses and int-indexed factors");
  auto& s=*impl_;
  for(const auto& edge:edges)
    if(edge.i<0 || edge.j<0 || edge.i>=pose_count || edge.j>=pose_count || edge.i==edge.j || !edge.summary || !std::isfinite(edge.weight) || edge.weight<=0)
      throw std::invalid_argument("Invalid BatchSummary edge");
  bool same=pose_count==s.poses && edges.size()==s.edges.size();
  if(same) for(size_t e=0;e<edges.size();++e) if(edges[e].i!=s.edges[e].i || edges[e].j!=s.edges[e].j) {same=false;break;}
  s.poses=pose_count;s.n=pose_count*6+1;s.edges=std::move(edges);
  if(!same) {
    s.offsets.assign(static_cast<size_t>(s.n)*s.n+1,0);
    auto entries=[&](auto consume) {
      for(size_t e=0;e<s.edges.size();++e) {
        const auto& edge=s.edges[e];int mapping[13];
        for(int k=0;k<6;++k) {mapping[k]=6*edge.i+k;mapping[6+k]=6*edge.j+k;}mapping[12]=s.n-1;
        for(int c=0;c<13;++c) for(int r=0;r<13;++r)
          consume(mapping[r]+s.n*mapping[c],static_cast<int>(169*e+r+13*c));
      }
    };
    entries([&](int cell,int){++s.offsets[cell+1];});
    for(size_t k=1;k<s.offsets.size();++k)s.offsets[k]+=s.offsets[k-1];
    s.indices.resize(s.offsets.back());auto cursor=s.offsets;
    entries([&](int cell,int index){s.indices[cursor[cell]++]=index;});
  }
  if(s.gpu) {
#ifdef FORM_ENABLE_CUDA
    std::vector<BatchRoot> roots(s.edges.size());
    for(size_t e=0;e<roots.size();++e) {
      const auto& edge=s.edges[e];auto& root=roots[e];root.i=edge.i;root.j=edge.j;root.weight=edge.weight;
      std::copy_n(edge.summary->planeRoot().data(),169,root.plane);
      std::copy_n(edge.summary->pointRoot().data(),49,root.point);
    }
    s.packed.resize(pose_count);
    if(s.cuda)s.cuda->reset(pose_count,roots,s.offsets,s.indices);
    else s.cuda=std::make_unique<CudaSummaryBatch>(pose_count,roots,s.offsets,s.indices);
#else
    throw std::runtime_error("CUDA batch requested in CPU-only build");
#endif
  } else {s.partials.resize(s.edges.size());s.costs.resize(s.edges.size());}
}
BatchSummary::~BatchSummary()=default;
Eigen::MatrixXd BatchSummary::linearize(const std::vector<gtsam::Pose3>& poses) {
  auto& s=*impl_;s.check(poses);Eigen::MatrixXd h(s.n,s.n);
#ifdef FORM_ENABLE_CUDA
  if(s.cuda) {s.cuda->evaluate(s.packed,h.data(),false);return h;}
#endif
  tbb::parallel_for(size_t(0),s.edges.size(),[&](size_t k) {
    const auto& e=s.edges[k];s.partials[k]=e.weight*e.summary->augmentedHessian(poses[e.i],poses[e.j]);
  });
  for(int k=0;k<s.n*s.n;++k) {
    double sum=0.;for(int p=s.offsets[k];p<s.offsets[k+1];++p) {int index=s.indices[p];sum+=s.partials[index/169].data()[index%169];}h.data()[k]=sum;
  }
  return h;
}
double BatchSummary::error(const std::vector<gtsam::Pose3>& poses) {
  auto& s=*impl_;s.check(poses);
#ifdef FORM_ENABLE_CUDA
  if(s.cuda) {double sum;s.cuda->evaluate(s.packed,&sum,true);return .5*sum;}
#endif
  tbb::parallel_for(size_t(0),s.edges.size(),[&](size_t k) {
    const auto& e=s.edges[k];s.costs[k]=e.weight*e.summary->squaredError(poses[e.i],poses[e.j]);
  });
  double sum=0;for(double cost:s.costs) sum+=cost;return .5*sum;
}
}
