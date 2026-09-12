#include "form/feature/extraction.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace {
struct ScanPoint : form::PointFeat {
  using Scalar = double;
  using form::PointFeat::PointFeat;
};

form::FeatureExtractor::Params planarParams() {
  form::FeatureExtractor::Params params;
  params.num_rows = 2;
  params.num_columns = 120;
  params.num_sectors = 2;
  params.planar_feats_per_sector = 1000;
  params.point_feats_per_sector = 1000;
  return params;
}

std::vector<ScanPoint> planarScan(const form::FeatureExtractor::Params &params) {
  std::vector<ScanPoint> scan;
  for (int row = 0; row < params.num_rows; ++row) {
    for (int column = 0; column < params.num_columns; ++column) {
      scan.emplace_back(5.0, 0.05 * column, 0.05 * row, 0);
    }
  }
  return scan;
}

template <typename Feature> void sortFeatures(std::vector<Feature> &features) {
  std::sort(features.begin(), features.end(), [](const auto &a, const auto &b) {
    return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
  });
}
} // namespace

TEST(FeatureExtraction, DefaultSpacingMatchesExplicitNeighborhood) {
  for (size_t neighbors : {size_t{3}, size_t{5}}) {
    for (double threshold : {0.0, 1.0}) {
      auto params = planarParams();
      params.neighbor_points = neighbors;
      params.planar_threshold = threshold;
      params.planar_feats_per_sector = 3;
      params.point_feats_per_sector = 3;
      const auto scan = planarScan(params);
      auto [default_planes, default_points] =
          form::FeatureExtractor(params, 1).extract(scan, 17);
      params.feature_spacing = neighbors;
      auto [explicit_planes, explicit_points] =
          form::FeatureExtractor(params, 1).extract(scan, 17);
      sortFeatures(default_planes);
      sortFeatures(explicit_planes);
      EXPECT_EQ(default_planes, explicit_planes);
      EXPECT_EQ(default_points, explicit_points);
      ASSERT_FALSE(default_points.empty());
      if (threshold > 0.0) {
        ASSERT_FALSE(default_planes.empty());
      }
    }
  }
}

TEST(FeatureExtraction, DenserSpacingAddsPlanesWithOriginalNormalNeighborhood) {
  auto params = planarParams();
  // Five neighbors on each side across two rows provide 21 neighbors. Using
  // feature_spacing=2 for normal estimation would provide only nine and fail.
  params.min_points = 15;
  const auto scan = planarScan(params);
  const auto [baseline, baseline_points] =
      form::FeatureExtractor(params, 1).extract(scan, 17);
  params.feature_spacing = 2;
  const auto [dense, dense_points] =
      form::FeatureExtractor(params, 1).extract(scan, 17);
  ASSERT_FALSE(baseline.empty());
  EXPECT_GT(dense.size(), baseline.size());
  for (const auto &plane : dense) {
    EXPECT_NEAR(std::abs(plane.nx), 1.0, 1e-12);
    EXPECT_NEAR(plane.ny, 0.0, 1e-12);
    EXPECT_NEAR(plane.nz, 0.0, 1e-12);
    EXPECT_EQ(plane.scan, 17);
  }
}

TEST(FeatureExtraction, DenserSpacingPreservesCurvatureNeighborhood) {
  auto params = planarParams();
  params.planar_threshold = 0.001;
  auto scan = planarScan(params);
  // Quadratic sampling yields curvature 0.003025 with five neighbors, but
  // 0.000025 with two. Only the latter would pass the planar threshold.
  for (size_t idx = 0; idx < scan.size(); ++idx) {
    const auto column = idx % params.num_columns;
    scan[idx].y = 0.0005 * column * column;
  }
  const auto [baseline, baseline_points] =
      form::FeatureExtractor(params, 1).extract(scan, 0);
  params.feature_spacing = 2;
  const auto [dense, dense_points] =
      form::FeatureExtractor(params, 1).extract(scan, 0);
  EXPECT_TRUE(baseline.empty());
  EXPECT_TRUE(dense.empty());
}

TEST(FeatureExtraction, PointSpacingControlsSuppressionAndPreservesRowBoundaries) {
  auto params = planarParams();
  params.planar_threshold = 0.0;
  const auto scan = planarScan(params);
  const size_t valid_per_row = params.num_columns - 2 * params.neighbor_points;
  for (size_t spacing : {size_t{1}, size_t{2}, size_t{5}}) {
    params.feature_spacing = spacing;
    const auto [planes, points] =
        form::FeatureExtractor(params, 1).extract(scan, 0);
    EXPECT_TRUE(planes.empty());
    EXPECT_EQ(points.size(), params.num_rows *
                                 ((valid_per_row + spacing - 1) / spacing));
    for (const auto &point : points) {
      const auto column = static_cast<size_t>(std::lround(point.y / 0.05));
      EXPECT_GE(column, params.neighbor_points);
      EXPECT_LT(column, params.num_columns - params.neighbor_points);
    }
  }
}

TEST(FeatureExtraction, DenserPlanarSpacingPreservesInvalidRangeNeighborhood) {
  auto params = planarParams();
  params.feature_spacing = 1;
  auto scan = planarScan(params);
  const size_t invalid_column = 60;
  scan[invalid_column].vec3().setZero();
  const auto [planes, points] =
      form::FeatureExtractor(params, 1).extract(scan, 0);
  ASSERT_FALSE(planes.empty());
  for (const auto &plane : planes) {
    const auto column = static_cast<size_t>(std::lround(plane.y / 0.05));
    EXPECT_GE(column, params.neighbor_points);
    EXPECT_LT(column, params.num_columns - params.neighbor_points);
    if (plane.z == 0.0) {
      EXPECT_TRUE(column < invalid_column - params.neighbor_points ||
                  column > invalid_column + params.neighbor_points);
    }
  }
}

TEST(FeatureExtraction, RejectsSpacingBeyondNeighborhoodAtEveryExtraction) {
  auto params = planarParams();
  const auto scan = planarScan(params);
  form::FeatureExtractor extractor(params, 1);
  EXPECT_NO_THROW((void)extractor.extract(scan, 0));
  extractor.params.feature_spacing = extractor.params.neighbor_points + 1;
  EXPECT_THROW((void)extractor.extract(scan, 0), std::invalid_argument);
  extractor.params.feature_spacing = 2;
  EXPECT_NO_THROW((void)extractor.extract(scan, 0));
  extractor.params.neighbor_points = 1;
  EXPECT_THROW((void)extractor.extract(scan, 0), std::invalid_argument);
}
