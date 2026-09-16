// MIT License

// Copyright (c) 2025 Easton Potokar, Taylor Pool, and Michael Kaess

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include "form/feature/extraction.hpp"
#include "form/optimization/profile.hpp"

namespace form {

// ######################### Main Method ######################### //
template <typename Point>
[[nodiscard]] std::tuple<std::vector<PlanarFeat>, std::vector<PointFeat>>
FeatureExtractor::extract(const std::vector<Point> &scan, size_t scan_idx) const {
  if (params.feature_spacing > params.neighbor_points) {
    throw std::invalid_argument("feature_spacing must not exceed neighbor_points");
  }
#ifndef FORM_ENABLE_CUDA
  if (params.use_cuda) throw std::runtime_error("CUDA extraction requested without FORM_ENABLE_CUDA");
#else
  std::unique_lock<std::mutex> cuda_lock(*cuda_mutex_, std::defer_lock);
  if (params.use_cuda) cuda_lock.lock();
#endif
  using T = typename Point::Scalar;
  const size_t points_per_sector = params.num_columns / params.num_sectors;

  auto profile_start = profile::enabled ? profile::Clock::now() : profile::Clock::time_point{};

  // First we validate that all the points are good
  auto valid_mask = compute_valid_points(scan);
  profile::checkpoint(profile::extract_validate, profile_start);

  // ------------------------- Planar Features ------------------------- //
  std::vector<Curvature<T>> curvature;
#ifdef FORM_ENABLE_CUDA
  if (params.use_cuda) {
    if (!cuda_) cuda_ = std::make_shared<CudaExtraction>();
    std::vector<std::array<T,4>> packed(scan.size());
    std::vector<unsigned char> mask(valid_mask.begin(), valid_mask.end());
    for (size_t i=0; i<scan.size(); ++i)
      for (int axis=0; axis<4; ++axis) packed[i][axis] = scan[i].vec4()[axis];
    // Ask Eigen which reduction the actual four-coordinate expression uses.
    using Squared = std::decay_t<decltype((scan.front().vec4()-scan.front().vec4()).cwiseAbs2())>;
    using Redux = Eigen::internal::redux_traits<Eigen::internal::scalar_sum_op<T>,
        Eigen::internal::redux_evaluator<Squared>>;
    constexpr auto reduction = int(Redux::Traversal) != int(Eigen::DefaultTraversal)
        ? CudaExtraction::Reduction::Cross
        : (int(Redux::Unrolling) == int(Eigen::CompleteUnrolling)
            ? CudaExtraction::Reduction::Adjacent : CudaExtraction::Reduction::Sequential);
    const auto values = cuda_->prepare(packed, mask, params.num_columns,
                                      params.neighbor_points, reduction);
    curvature.reserve(values.size());
    for (size_t i=0; i<values.size(); ++i) curvature.emplace_back(i, values[i]);
  } else
#endif
    curvature = compute_curvature(scan, valid_mask, scan_idx);
  profile::checkpoint(profile::extract_curvature, profile_start);

  // Next get the planar features
  std::vector<size_t> planar_indices;
  std::vector<bool> used_points = valid_mask;
  auto select_row = [&](size_t scan_line_idx, auto& mask, auto& selected) {
    // Sectors on the same row share suppression state and remain ordered.
    for (size_t sector_idx=0; sector_idx<params.num_sectors; ++sector_idx) {
      const size_t first=scan_line_idx*params.num_columns+sector_idx*points_per_sector;
      const size_t end=sector_idx+1==params.num_sectors
          ? (scan_line_idx+1)*params.num_columns : first+points_per_sector;
      std::sort(curvature.begin()+first,curvature.begin()+end);
      extract_planar(first,end,curvature,selected,mask);
    }
  };
  if (params.parallel_selection) {
    // Byte masks avoid races between neighboring rows sharing a vector<bool> word.
    std::vector<unsigned char> mask(valid_mask.begin(),valid_mask.end());
    std::vector<std::vector<size_t>> row_features(params.num_rows);
    tbb::parallel_for(size_t(0),size_t(params.num_rows),[&](size_t row) {
      select_row(row,mask,row_features[row]);
    });
    for (size_t i=0;i<mask.size();++i) used_points[i]=mask[i];
    for (const auto& row:row_features)
      planar_indices.insert(planar_indices.end(),row.begin(),row.end());
  } else {
    for (size_t row=0;row<size_t(params.num_rows);++row)
      select_row(row,used_points,planar_indices);
  }

  profile::checkpoint(profile::extract_planar_select, profile_start);

  // ------------------------- Point Features ------------------------- //
  // Get valid point features mask
  auto valid_mask_points = compute_point_valid_points(scan);

  std::vector<size_t> point_indices;
  // used_points stores the points and their neighbors that have already been used
  for (size_t idx = 0; idx < used_points.size(); idx++) {
    valid_mask_points[idx] =
        // good if (wasn't used as planar) AND (is valid point)
        (used_points[idx] == valid_mask[idx]) && valid_mask_points[idx];
  }

  profile::checkpoint(profile::extract_point_mask, profile_start);

  for (size_t scan_line_idx = 0; scan_line_idx < params.num_rows; scan_line_idx++) {
    // Independently detect features in each sector of this scan_line
    for (size_t sector_idx = 0; sector_idx < params.num_sectors; sector_idx++) {
      const size_t sector_start_pt =
          (scan_line_idx * params.num_columns) + (sector_idx * points_per_sector);
      // Special case for end point as we add any reminder points to the last
      // sector
      const size_t sector_end_pt = (sector_idx == params.num_sectors - 1)
                                       ? ((scan_line_idx + 1) * params.num_columns)
                                       : sector_start_pt + points_per_sector;

      extract_point(sector_start_pt, sector_end_pt, point_indices,
                    valid_mask_points);
    }
  }

  profile::checkpoint(profile::extract_point_select, profile_start);

  std::vector<std::array<int,2>> nearest_rows;
#ifdef FORM_ENABLE_CUDA
  if (params.use_cuda) nearest_rows = cuda_->nearestRows(planar_indices);
#endif

  // Finally extract all normals
  tbb::concurrent_vector<PlanarFeat> result_planar_tbb;
  result_planar_tbb.reserve(planar_indices.size());
  const auto range =
      tbb::blocked_range{planar_indices.cbegin(), planar_indices.cend()};
  tbb::parallel_for(range, [&](const auto &range) {
    for (auto it = range.begin(); it != range.end(); ++it) {
      const size_t idx = *it;
      const Point &point = scan[idx];
      std::optional<Eigen::Matrix<T, 3, 1>> normal =
          compute_normal(idx, scan, valid_mask, params.use_cuda ? &nearest_rows[it-planar_indices.cbegin()] : nullptr);
      if (normal.has_value()) {
        result_planar_tbb.emplace_back(
            static_cast<double>(point.x), static_cast<double>(point.y),
            static_cast<double>(point.z), static_cast<double>(normal.value().x()),
            static_cast<double>(normal.value().y()),
            static_cast<double>(normal.value().z()), static_cast<size_t>(scan_idx));
      }
    }
  });
  profile::checkpoint(profile::extract_normals, profile_start);
  std::vector<PlanarFeat> result_planar(result_planar_tbb.begin(),
                                        result_planar_tbb.end());

  // Add the point features
  std::vector<PointFeat> result_point;
  result_point.reserve(point_indices.size());
  for (const size_t &idx : point_indices) {
    const Point &point = scan[idx];
    result_point.emplace_back(
        static_cast<double>(point.x), static_cast<double>(point.y),
        static_cast<double>(point.z), static_cast<size_t>(scan_idx));
  }

  profile::checkpoint(profile::extract_pack, profile_start);
  return std::make_tuple(result_planar, result_point);
}

// ------------------------- Validators ------------------------- //
// A bunch of methods to validate points
template <typename Point>
std::vector<bool>
FeatureExtractor::compute_valid_points(const std::vector<Point> &scan) const {
  using T = typename Point::Scalar;

  if (scan.size() != params.num_columns * params.num_rows) {
    throw std::runtime_error("Provided scan does not match the expected size " +
                             std::to_string(params.num_columns * params.num_rows) +
                             " != " + std::to_string(scan.size()));
  }

  std::vector<bool> mask(scan.size(), true);
  size_t num_points = scan.size();

  // Compute the valid points based on the parameters
  // Structured search (search over each scan line individually over all points
  // [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < params.num_rows; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < params.num_columns; line_pt_idx++) {
      const size_t idx = (scan_line_idx * params.num_columns) + line_pt_idx;

      // CHECK 1: Due to edge effects, the first and last neighbor_points points
      // of each scan line are invalid
      if (line_pt_idx < params.neighbor_points ||
          line_pt_idx >= params.num_columns - params.neighbor_points) {
        mask[idx] = false;
        continue;
      }

      // CHECK 2: Is the point in the valid range of the LiDAR
      const Point &point = scan[idx];
      const double range2 = point.squaredNorm();
      if (range2 < params.min_norm_squared || range2 > params.max_norm_squared) {
        mask[idx] = false;
        for (size_t i = 1; i <= params.neighbor_points; i++) {
          mask[idx - i] = false;
          mask[idx + i] = false;
        }
        continue;
      }
    } // end line point search
  } // end scan line search

  return mask;
}

template <typename Point>
std::vector<bool>
FeatureExtractor::compute_point_valid_points(const std::vector<Point> &scan) const {
  using T = typename Point::Scalar;

  if (scan.size() != params.num_columns * params.num_rows) {
    throw std::runtime_error("Provided scan does not match the expected size " +
                             std::to_string(params.num_columns * params.num_rows) +
                             " != " + std::to_string(scan.size()));
  }

  std::vector<bool> mask(scan.size(), true);
  size_t num_points = scan.size();

  // Compute the valid points based on the parameters
  // Structured search (search over each scan line individually over all points
  // [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < params.num_rows; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < params.num_columns; line_pt_idx++) {
      const size_t idx = (scan_line_idx * params.num_columns) + line_pt_idx;

      // CHECK 1: Due to edge effects, the first and last neighbor_points points
      // of each scan line are invalid
      if (line_pt_idx < params.neighbor_points ||
          line_pt_idx >= params.num_columns - params.neighbor_points) {
        mask[idx] = false;
        continue;
      }

      // CHECK 2: Is the point in the valid range of the LiDAR
      const Point &point = scan[idx];
      const double range2 = point.squaredNorm();
      if (range2 < params.min_norm_squared || range2 > params.max_norm_squared) {
        mask[idx] = false;
        continue;
      }
    } // end line point search
  } // end scan line search

  return mask;
}

// ------------------------- Computers ------------------------- //
// Used to compute various things
template <typename Point>
std::vector<Curvature<typename Point::Scalar>>
FeatureExtractor::compute_curvature(const std::vector<Point> &scan,
                                    const std::vector<bool> &mask,
                                    size_t scan_idx) const noexcept {

  using T = typename Point::Scalar;
  std::vector<Curvature<T>> curvature;

  // Structured search (search over each scan line individually over all points
  // [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < params.num_rows; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < params.num_columns; line_pt_idx++) {
      const size_t idx = (scan_line_idx * params.num_columns) + line_pt_idx;
      // If not valid, input max curvature
      if (!mask[idx]) {
        curvature.emplace_back(idx, std::numeric_limits<T>::max());
      }
      // If valid compute the curvature
      else {
        // Initialize with the difference term
        double dx = -(2.0 * params.neighbor_points) * scan[idx].x;
        double dy = -(2.0 * params.neighbor_points) * scan[idx].y;
        double dz = -(2.0 * params.neighbor_points) * scan[idx].z;
        // Iterate over neighbors and accumulate
        for (size_t n = 1; n <= params.neighbor_points; n++) {
          dx = dx + scan[idx - n].x + scan[idx + n].x;
          dy = dy + scan[idx - n].y + scan[idx + n].y;
          dz = dz + scan[idx - n].z + scan[idx + n].z;
        }
        curvature.emplace_back(idx, dx * dx + dy * dy + dz * dz);
      }
    }
  }
  return curvature;
}

template <typename Point>
std::optional<Eigen::Matrix<typename Point::Scalar, 3, 1>>
FeatureExtractor::compute_normal(
    const size_t &idx, const std::vector<Point> &scan,
    const std::vector<bool> &valid_mask,
    const std::array<int,2>* nearest_rows) const noexcept {
  using T = typename Point::Scalar;
  const size_t scan_line_idx = idx / params.num_columns;
  const auto start = scan.cbegin();
  const auto end = scan.cend();
  const auto &point = scan[idx];

  const bool profile_sample = profile::enabled &&
      (((static_cast<uint64_t>(idx) * 11400714819323198485ull) >> 58) == 0);
  auto normal_start = profile_sample ? profile::Clock::now() : profile::Clock::time_point{};

  // First find neighbors on own scan line
  std::vector<Point> neighbors;
  find_neighbors(idx, scan, neighbors);

  bool found_other_scanline = false;

  // Get the neighbors of the point on the previous scan line
  if (scan_line_idx > 0) {
    const size_t prev_scan_line_idx = scan_line_idx - 1;
    const auto closest_idx = nearest_rows
        ? ((*nearest_rows)[0] < 0 ? std::optional<size_t>{} : std::optional<size_t>{size_t((*nearest_rows)[0])})
        : find_closest(point, params.num_columns * prev_scan_line_idx,
                       params.num_columns * (prev_scan_line_idx + 1), scan, valid_mask);
    if (closest_idx.has_value()) {
      found_other_scanline = true;
      neighbors.push_back(scan[*closest_idx]);
      find_neighbors(*closest_idx, scan, neighbors);
    }
  }

  // Get the neighbors of the point on the next scan line
  if (scan_line_idx < params.num_rows - 1) {
    // std::printf("---- Searching next scan line %zu\n", scan_line_idx + 1);
    const size_t next_scan_line_idx = scan_line_idx + 1;
    const auto closest_idx = nearest_rows
        ? ((*nearest_rows)[1] < 0 ? std::optional<size_t>{} : std::optional<size_t>{size_t((*nearest_rows)[1])})
        : find_closest(point, params.num_columns * next_scan_line_idx,
                       params.num_columns * (next_scan_line_idx + 1), scan, valid_mask);
    if (closest_idx.has_value()) {
      found_other_scanline = true;
      neighbors.push_back(scan[*closest_idx]);
      find_neighbors(closest_idx.value(), scan, neighbors);
    }
  }

  if (profile_sample) {
    profile::checkpoint(profile::normal_search_sample_cpu, normal_start);
    profile::normal_samples.fetch_add(1, std::memory_order_relaxed);
  }

  // If there's not enough neighbors, return failed
  if (!found_other_scanline || neighbors.size() < params.min_points) {
    return std::nullopt;
  }

  // Compute the covariance matrix
  // std::printf("---- Found %zu neighbors\n", neighbors.size());
  Eigen::Matrix<T, Eigen::Dynamic, 3> A(neighbors.size(), 3);
  for (size_t j = 0; j < neighbors.size(); ++j) {
    A.row(j) = neighbors[j].vec3() - point.vec3();
  }
  A /= neighbors.size();
  Eigen::Matrix<T, 3, 3> Cov = A.transpose() * A;

  // Eigenvalues + normals
  // std::printf("---- computing eigenvalues\n");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> b(
      Cov, Eigen::ComputeEigenvectors);
  Eigen::Matrix<T, 3, 1> normal = b.eigenvectors().col(0);
  normal.normalize();
  if (profile_sample) profile::checkpoint(profile::normal_eigen_sample_cpu, normal_start);

  return normal;
}

// ------------------------- Extractors ------------------------- //
template <typename T, typename Mask>
void FeatureExtractor::extract_planar(const size_t &sector_start_point,
                                      const size_t &sector_end_point,
                                      const std::vector<Curvature<T>> &curvature,
                                      std::vector<size_t> &out_features,
                                      Mask &valid_mask) const noexcept {

  size_t num_sector_planar_features = 0;
  const size_t feature_spacing = params.feature_spacing != 0
                                     ? params.feature_spacing
                                     : params.neighbor_points;
  // Iterate through all points in the sector
  for (size_t sorted_curv_idx = sector_start_point;
       sorted_curv_idx < sector_end_point; sorted_curv_idx++) {
    const Curvature curv = curvature[sorted_curv_idx];
    if (valid_mask[curv.index] && curv.curvature < params.planar_threshold) {
      out_features.push_back(curv.index);
      // mark the neighbors as used so they aren't also added in
      for (size_t n = 0; n < feature_spacing; n++) {
        valid_mask[curv.index + n] = false;
        valid_mask[curv.index - n] = false;
      }
      num_sector_planar_features++;
    }
    // Early exit if we have found enough features
    if (num_sector_planar_features > params.planar_feats_per_sector)
      break;

  } // end feature search in sector
}

void FeatureExtractor::extract_point(const size_t &sector_start_point,
                                     const size_t &sector_end_point,
                                     std::vector<size_t> &out_features,
                                     std::vector<bool> &valid_mask) const noexcept {
  size_t num_sector_point_features = 0;
  const size_t feature_spacing = params.feature_spacing != 0
                                     ? params.feature_spacing
                                     : params.neighbor_points;

  if (params.point_feats_per_sector == 0) {
    return; // No point features to extract
  }

  // Figure out how many we may have
  std::vector<size_t> unused_points;
  for (size_t idx = sector_start_point; idx < sector_end_point; idx++) {
    if (valid_mask[idx]) {
      unused_points.push_back(idx);
    }
  }

  // By what factor do we have too many?
  size_t factor = 1 + unused_points.size() / params.point_feats_per_sector;
  // Do "factor" number of passes over the points until we get enough
  // This should help spread them out evenly
  for (size_t offset = 0; offset < factor; offset++) {
    for (size_t unused_idx = offset; unused_idx < unused_points.size();
         unused_idx += factor) {
      const size_t idx = unused_points[unused_idx];
      if (valid_mask[idx]) {
        out_features.push_back(idx);                          // Add to points
        for (size_t n = 0; n < feature_spacing; n++) { // update mask
          valid_mask[idx + n] = false;
          valid_mask[idx - n] = false;
        }
        num_sector_point_features++;
      }
      // Early exit if we have found enough features
      if (num_sector_point_features > params.point_feats_per_sector)
        break;
    }
    // Early exit if we have found enough features
    if (num_sector_point_features > params.point_feats_per_sector)
      break;
  }
}

// ------------------------- Helpers ------------------------- //
template <typename Point>
std::optional<size_t>
FeatureExtractor::find_closest(const Point &point, const size_t &start,
                               const size_t &end, const std::vector<Point> &scan,
                               const std::vector<bool> &valid_mask) const noexcept {
  std::optional<size_t> closest_point = std::nullopt;
  double min_dist2 = std::numeric_limits<double>::max();
  for (size_t idx = start; idx < end; idx++) {
    if (!valid_mask[idx]) {
      continue;
    }
    const double dist2 = (scan[idx].vec4() - point.vec4()).squaredNorm();
    if (dist2 < min_dist2) {
      min_dist2 = dist2;
      closest_point = idx;
    }
  }
  return closest_point;
}

template <typename Point>
void FeatureExtractor::find_neighbors(const size_t &idx,
                                      const std::vector<Point> &scan,
                                      std::vector<Point> &out) const noexcept {
  // search in the positive direction
  const auto &point = scan[idx];
  for (size_t i = 1; i <= params.neighbor_points; i++) {
    const auto &neighbor = scan[idx + i];
    const double range2 = (neighbor.vec4() - point.vec4()).squaredNorm();
    if (range2 < params.radius * params.radius) {
      out.push_back(neighbor);
    } else {
      break;
    }
  }

  // search in the negative direction
  for (size_t i = 1; i <= params.neighbor_points; i++) {
    const auto &neighbor = scan[idx - i];
    const double range2 = (neighbor.vec4() - point.vec4()).squaredNorm();
    if (range2 < params.radius * params.radius) {
      out.push_back(neighbor);
    } else {
      break;
    }
  }
}

} // namespace form
