#include "form/feature/extraction.hpp"
#include <gtest/gtest.h>
#include <tbb/task_scheduler_observer.h>
#include <atomic>
#include <future>
#include <algorithm>

namespace {
struct Observer : tbb::task_scheduler_observer {
  std::atomic<int> workers{0};
  Observer() { observe(true); }
  ~Observer() { observe(false); }
  void on_scheduler_entry(bool worker) override { if(worker) ++workers; }
};
auto scanFor(const form::FeatureExtractor::Params& params) {
  std::vector<form::PointXYZf> scan;
  for(int r=0;r<params.num_rows;++r) for(int c=0;c<params.num_columns;++c)
    scan.emplace_back(5.f,0.02f*c,0.05f*r);
  return scan;
}
void sort(std::vector<form::PlanarFeat>& points) {
  std::sort(points.begin(),points.end(),[](const auto& a,const auto& b) {
    return std::tie(a.x,a.y,a.z)<std::tie(b.x,b.y,b.z);
  });
}
}
TEST(ParallelExtraction, RowSelectionActuallyUsesWorkers) {
  // Separate executable: no preceding test can install a permanent 1-thread cap.
  form::FeatureExtractor::Params params;
  params.num_rows=128; params.num_columns=1023; params.planar_threshold=0.;
  const auto scan=scanFor(params);
  form::FeatureExtractor reference(params,4);
  auto expected=reference.extract(scan,0);
  params.parallel_selection=true;
  form::FeatureExtractor parallel(params,4);
  Observer observer;
  auto actual=parallel.extract(scan,0);
  EXPECT_GT(observer.workers.load(),0);
  EXPECT_EQ(actual,expected);
}
#ifdef FORM_ENABLE_CUDA
TEST(ParallelExtraction, SharedCudaWorkspaceSerializesConcurrentCalls) {
  form::FeatureExtractor::Params params;
  params.num_rows=8; params.num_columns=131;
  using Output=std::tuple<std::vector<form::PlanarFeat>,std::vector<form::PointFeat>>;
  std::vector<std::vector<form::PointXYZf>> scans;
  std::vector<Output> expected;
  for(int i=0;i<4;++i) {
    scans.push_back(scanFor(params));
    for(auto& point:scans.back()) point.x+=0.2f*i;
    if(i) scans.back()[35*i].vec3().setZero();
    expected.push_back(form::FeatureExtractor(params,4).extract(scans.back(),7+i));
    sort(std::get<0>(expected.back()));
  }
  params.parallel_selection=true; params.use_cuda=true;
  form::FeatureExtractor gpu(params,4);
  std::vector<std::future<Output>> calls;
  for(int i=0;i<4;++i)
    calls.push_back(std::async(std::launch::async,[&,i] { return gpu.extract(scans[i],7+i); }));
  for(size_t i=0;i<calls.size();++i) {
    auto [p,q]=calls[i].get(); sort(p);
    EXPECT_EQ(p,std::get<0>(expected[i])); EXPECT_EQ(q,std::get<1>(expected[i]));
  }
}
#endif
