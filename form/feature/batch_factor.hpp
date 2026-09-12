// MIT License; see factor.hpp for copyright and license text.
#pragma once
#include "form/feature/batch_summary.hpp"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
namespace form {
/// Replace summarized binary feature factors with one dense batch. Other factors
/// stay in the graph. A workspace may be reused only when no older graph owns it.
/// The CUDA selection must remain fixed for a given workspace.
gtsam::NonlinearFactorGraph batchFeatureGraph(
    const gtsam::NonlinearFactorGraph& graph, bool cuda, size_t min_edges,
    std::shared_ptr<BatchSummary>& workspace);
}
