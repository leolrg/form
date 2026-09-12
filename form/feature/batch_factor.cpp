// MIT License; see factor.hpp for copyright and license text.
#include "form/feature/batch_factor.hpp"
#include "form/feature/factor.hpp"
#include "form/optimization/profile.hpp"
#include <gtsam/linear/HessianFactor.h>
#include <map>
#include <typeinfo>
namespace form {
namespace {
const FeatureFactor* summarizedFeature(const gtsam::NonlinearFactor* factor) {
  // Subclasses may override activation, error, or linearization. Keep their
  // virtual behavior rather than replacing it with the base summary equations.
  if (!factor || typeid(*factor) != typeid(FeatureFactor)) return nullptr;
  const auto* feature = static_cast<const FeatureFactor*>(factor);
  return feature->summary() ? feature : nullptr;
}
class BatchFactor final:public gtsam::NonlinearFactor {
  std::shared_ptr<BatchSummary> batch_;
  size_t residuals_;
  std::vector<gtsam::Pose3> poses(const gtsam::Values& values) const {
    std::vector<gtsam::Pose3> result;result.reserve(keys().size());
    for(auto key:keys())result.push_back(values.at<gtsam::Pose3>(key));
    return result;
  }
 public:
  BatchFactor(const gtsam::KeyVector& keys,std::shared_ptr<BatchSummary> batch,size_t residuals)
      :NonlinearFactor(keys),batch_(std::move(batch)),residuals_(residuals) {}
  size_t dim() const override{return residuals_;}
  double error(const gtsam::Values& values) const override {
    profile::Scope timer(profile::factor_error);return batch_->error(poses(values));
  }
  boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override {
    profile::Scope timer(profile::factor_linearize);
    const auto h=batch_->linearize(poses(values));
    gtsam::SymmetricBlockMatrix blocks(std::vector<size_t>(keys().size(),6),h,true);
    return boost::make_shared<gtsam::HessianFactor>(keys(),blocks);
  }
};
}
gtsam::NonlinearFactorGraph batchFeatureGraph(const gtsam::NonlinearFactorGraph& graph,
    bool cuda,size_t min_edges,std::shared_ptr<BatchSummary>& workspace) {
  std::vector<const FeatureFactor*> factors;std::map<gtsam::Key,int> indices;size_t residuals=0;
  for(const auto& factor:graph) {
    const auto* f=summarizedFeature(factor.get());
    if(!f)continue;
    factors.push_back(f);indices.emplace(f->key1(),0);indices.emplace(f->key2(),0);residuals+=f->dim();
  }
  // The zero threshold is for correctness testing. Production screening keeps
  // sparse star-shaped graphs on their existing per-factor path.
  if(factors.empty() || (min_edges && factors.size()<std::max(min_edges,2*indices.size())))return graph;
  gtsam::KeyVector keys;for(auto& [key,index]:indices){index=keys.size();keys.push_back(key);}
  std::vector<SummaryEdge> edges;edges.reserve(factors.size());
  for(const auto* f:factors)edges.push_back({indices.at(f->key1()),indices.at(f->key2()),f->summary(),f->inverseVariance()});
  if(workspace && workspace.use_count()==1)workspace->reset(keys.size(),std::move(edges));
  else workspace=std::make_shared<BatchSummary>(keys.size(),std::move(edges),cuda);
  gtsam::NonlinearFactorGraph result;
  for(const auto& factor:graph)if(!summarizedFeature(factor.get()))result.push_back(factor);
  result.emplace_shared<BatchFactor>(keys,workspace,residuals);return result;
}
}
