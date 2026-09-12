// Compare complete CPU/GPU evaluation+assembly calls, including transfers and synchronization.
#include "form/feature/batch_summary.hpp"
#include "form/feature/factor.hpp"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <tbb/global_control.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
using Clock=std::chrono::steady_clock;
static volatile double checksum=0;
template<class Fn> double timeUs(Fn&& fn) {auto start=Clock::now();fn();return std::chrono::duration<double,std::micro>(Clock::now()-start).count();}
static double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}
int main() {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism,32);
  // Initialize the CUDA context before measuring per-graph setup.
  {form::BatchSummary warm(2,{},true);warm.error(std::vector<gtsam::Pose3>(2));}
  std::cout<<std::setprecision(9)<<"poses,edges,points_per_edge,backend,setup_us,reset_reuse_us,linearize_assemble_us,cost_us,modeled_reset_plus_5_iterations_us\n";
  for(int n:{10,30,40,80}) for(bool dense:{false,true}) for(int count:{32,128}) {
    std::vector<form::SummaryEdge> edges;std::vector<gtsam::Pose3> poses;gtsam::Values values;
    gtsam::NonlinearFactorGraph original,summary;
    for(int i=0;i<n;++i) {poses.emplace_back(gtsam::Rot3::RzRyRx(.003*i,-.002*i,.01),gtsam::Point3(.01*i,.02*i,-.03));values.insert(i,poses.back());}
    for(int i=0;i<n-1;++i) for(int j=i+1;j<n;++j) {
      if(!dense && j!=n-1)continue;
      auto planes=std::make_shared<form::PlanePoint>();auto points=std::make_shared<form::PointPoint>();
      for(int k=0;k<count;++k) {
        Eigen::Vector3d p(.1*k,.03*k*k,.1*j-.2*i),q=p+Eigen::Vector3d(.02*j,-.01*i,.03),normal(.2,.3,.5);
        points->p_i.insert(points->p_i.end(),p.data(),p.data()+3);points->p_j.insert(points->p_j.end(),q.data(),q.data()+3);
        planes->p_i.insert(planes->p_i.end(),p.data(),p.data()+3);planes->p_j.insert(planes->p_j.end(),q.data(),q.data()+3);planes->n_i.insert(planes->n_i.end(),normal.data(),normal.data()+3);
      }
      original.push_back(form::FeatureFactor(i,j,std::make_pair(planes,points),1.,false));
      summary.push_back(form::FeatureFactor(i,j,std::make_pair(planes,points),1.,true));
      edges.push_back({i,j,planes->summary_cache,1.});
    }
    std::unique_ptr<form::BatchSummary> cpu,gpu;std::vector<double> setups[2];
    for(int r=0;r<5;++r) for(int k=0;k<2;++k) {int b=(r+k)%2;std::unique_ptr<form::BatchSummary> temp;setups[b].push_back(timeUs([&]{temp=std::make_unique<form::BatchSummary>(n,edges,b==1);}));if(b)gpu=std::move(temp);else cpu=std::move(temp);}
    std::vector<double> resets[2];
    for(int r=0;r<12;++r) for(int k=0;k<2;++k) {int b=(r+k)%2;resets[b].push_back(timeUs([&]{(b?gpu:cpu)->reset(n,edges);}));}
    std::vector<double> linear[4],cost[4];
    for(int r=-2;r<12;++r) for(int k=0;k<4;++k) {
      int b=(r+2+k)%4;
      double l=timeUs([&] {if(b<2){auto graph=(b?summary:original).linearize(values);gtsam::HessianFactor h(*graph);checksum+=h.constantTerm();}else{auto h=(b==2?cpu:gpu)->linearize(poses);checksum+=h(h.rows()-1,h.cols()-1);}});
      double c=timeUs([&] {checksum+=(b<2?(b?summary:original).error(values):(b==2?cpu:gpu)->error(poses));});
      if(r>=0){linear[b].push_back(l);cost[b].push_back(c);}
    }
    const char* labels[]={"original_cpu","summary_cpu","batch_cpu","batch_cuda"};
    for(int b=0;b<4;++b) {double setup=b<2?0.:median(setups[b-2]),reset=b<2?0.:median(resets[b-2]),l=median(linear[b]),c=median(cost[b]);
      std::cout<<n<<','<<edges.size()<<','<<count<<','<<labels[b]<<','<<setup<<','<<reset<<','<<l<<','<<c<<','<<reset+5*(l+c)<<'\n';}
    std::cout.flush();
  }
  std::cerr<<"checksum="<<checksum<<'\n';
}
