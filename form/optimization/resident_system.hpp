// MIT License; see form/feature/factor.hpp for copyright and license text.
#pragma once
#include <vector>
namespace form {
// Pose indices are in the global sorted pose order. Augmented matrices are
// column-major, dimension (6*pose_indices.size()+1)^2, unshifted [G,b;b',f].
struct FrozenSystem {
  std::vector<int> pose_indices;
  std::vector<double> augmented;
  bool has_anchor = true;
};
}
