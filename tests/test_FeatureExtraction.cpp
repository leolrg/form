#include "form/feature/extraction.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include "form/feature/cuda_legacy_sort.cuh"
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

TEST(FeatureExtraction, LegacySortMatchesLibstdcxxPermutation) {
#if !defined(__GLIBCXX__) || !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE != 13 || defined(_GLIBCXX_PARALLEL)
  GTEST_SKIP()<<"Legacy compatibility policy is verified against serial libstdc++ 13";
#endif
  std::mt19937 random(321);
  for(int n=1;n<=1024;++n)for(int pattern=0;pattern<6;++pattern) {
    std::vector<double> keys(n);
    for(int i=0;i<n;++i) {
      switch(pattern) {
        case 0:keys[i]=0.;break;
        case 1:keys[i]=i;break;
        case 2:keys[i]=-i;break;
        case 3:keys[i]=std::min(i,n-1-i);break;
        case 4:keys[i]=random()%11;break;
        default:keys[i]=random();
      }
    }
    std::vector<int> a(n);std::iota(a.begin(),a.end(),0);auto b=a;
    std::sort(a.begin(),a.end(),[&](int x,int y){return keys[x]<keys[y];});
    form::detail::legacySelectionSort(b.data(),keys.data(),n);
    ASSERT_EQ(a,b)<<"n="<<n<<" pattern="<<pattern;
    std::iota(a.begin(),a.end(),0);b=a;
    auto less=[&](int x,int y){return keys[x]<keys[y];};
    std::make_heap(a.begin(),a.end(),less);std::sort_heap(a.begin(),a.end(),less);
    form::detail::legacyHeap(b.data(),keys.data(),n);
    ASSERT_EQ(a,b)<<"heap n="<<n<<" pattern="<<pattern;
  }
}

#ifdef FORM_ENABLE_CUDA
TEST(FeatureExtraction, ResidentSelectionMatchesCpuSelection) {
  for(bool stable : {false,true}) for(int sectors : {1,6}) for(int columns : {41,257,1024}) for(size_t spacing : {size_t{2},size_t{5}}) {
    auto p=planarParams();p.num_rows=3;p.num_columns=columns;p.num_sectors=sectors;
    p.planar_feats_per_sector=50;p.point_feats_per_sector=3;p.feature_spacing=spacing;
    p.stable_selection=stable;p.parallel_selection=true;p.use_cuda=true;p.use_cuda_normals=true;
    auto scan=planarScan(p);
    scan[columns+16].x=0.;scan[columns+16].y=0.;scan[columns+16].z=0.;
    auto reference=form::FeatureExtractor(p,4).extract(scan,0);
    p.use_cuda_selection=true;
    auto candidate=form::FeatureExtractor(p,4).extract(scan,0);
    const auto& a=std::get<0>(reference);const auto& b=std::get<0>(candidate);
    ASSERT_EQ(a.size(),b.size());
    for(size_t i=0;i<a.size();++i){EXPECT_EQ(a[i],b[i]);}
    ASSERT_EQ(std::get<1>(reference).size(),std::get<1>(candidate).size());
    for(size_t i=0;i<std::get<1>(reference).size();++i)EXPECT_EQ(std::get<1>(reference)[i],std::get<1>(candidate)[i]);
  }
}
TEST(FeatureExtraction, ResidentSelectionRequiresExplicitSemantics) {
  auto p=planarParams();p.use_cuda_selection=true;
  EXPECT_THROW(form::FeatureExtractor(p).extract(planarScan(p),0),std::invalid_argument);
}
TEST(FeatureExtraction, ResidentSelectionFloatMasksZeroCapsAndRaggedSectors) {
  auto p=planarParams();p.num_rows=3;p.num_columns=137;p.num_sectors=6;
  p.min_norm_squared=.01;p.max_norm_squared=2500.;p.planar_feats_per_sector=0;p.point_feats_per_sector=0;
  p.stable_selection=true;p.parallel_selection=true;p.use_cuda=true;p.use_cuda_normals=true;
  std::vector<form::PointXYZf> scan;
  for(int row=0;row<p.num_rows;++row)for(int c=0;c<p.num_columns;++c)scan.emplace_back(5.f,.015f*c,.04f*row);
  scan[16]._=60.f;scan[0].x=100.f;scan[31]=form::PointXYZf(0,0,0);
  for(bool stable : {false,true}) for(size_t cap : {size_t{0},size_t{3},size_t{200}}){
    p.stable_selection=stable;
    p.planar_feats_per_sector=cap;p.point_feats_per_sector=cap;p.use_cuda_selection=false;
    auto a=form::FeatureExtractor(p,4).extract(scan,7);p.use_cuda_selection=true;
    auto b=form::FeatureExtractor(p,4).extract(scan,7);
    EXPECT_EQ(std::get<0>(a),std::get<0>(b));EXPECT_EQ(std::get<1>(a),std::get<1>(b));
  }
  p.num_sectors=0;
  EXPECT_THROW(form::FeatureExtractor(p).extract(scan,7),std::invalid_argument);
}
TEST(FeatureExtraction, ResidentSelectionRejectsNonfiniteCurvatureAndRecovers) {
 for(bool stable : {false,true}) {
  auto p=planarParams();p.use_cuda=true;p.use_cuda_normals=true;p.stable_selection=stable;p.use_cuda_selection=true;
  auto scan=planarScan(p);auto extractor=form::FeatureExtractor(p);
  scan[17].x=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(extractor.extract(scan,0),std::invalid_argument);
  p.use_cuda_selection=false;
  if(stable)EXPECT_THROW(form::FeatureExtractor(p).extract(scan,0),std::invalid_argument);
  scan=planarScan(p);
  EXPECT_NO_THROW(extractor.extract(scan,1));
 }
}
#endif

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

#ifdef FORM_ENABLE_CUDA
TEST(FeatureExtraction, CudaPreservesFeaturesAndNormalsAcrossRepeatedScans) {
  auto params = planarParams();
  params.num_rows = 5;
  params.num_columns = 131; // partial warp and uneven sectors
  params.num_sectors = 6;
  params.planar_feats_per_sector = 5;
  params.point_feats_per_sector = 3;
  for (size_t spacing : {size_t{0}, size_t{2}}) {
    params.feature_spacing = spacing;
    form::FeatureExtractor cpu(params, 1);
    params.use_cuda = true;
    params.parallel_selection = true;
    form::FeatureExtractor gpu(params, 1);
    params.use_cuda = false;
    params.parallel_selection = false;
    for (int repetition = 0; repetition < 3; ++repetition) {
      auto scan = planarScan(params);
      scan[60].vec3().setZero();
      scan[131 + 65].vec3().setZero();
      for (size_t i = 0; i < scan.size(); ++i) {
        if (scan[i].x != 0) scan[i].x += 0.001 * std::sin(i * 0.37 + repetition);
      }
      auto [cp, cq] = cpu.extract(scan, repetition);
      auto [gp, gq] = gpu.extract(scan, repetition);
      sortFeatures(cp); sortFeatures(gp);
      ASSERT_FALSE(cp.empty());
      EXPECT_EQ(cp, gp);
      EXPECT_EQ(cq, gq);
      std::vector<form::PointXYZf> floats;
      for (const auto& p : scan) floats.emplace_back(p.x, p.y, p.z);
      auto [cfp, cfq] = cpu.extract(floats, repetition);
      auto [gfp, gfq] = gpu.extract(floats, repetition);
      sortFeatures(cfp); sortFeatures(gfp);
      EXPECT_EQ(cfp, gfp);
      EXPECT_EQ(cfq, gfq);
    }
  }
}

TEST(FeatureExtraction, CudaHandlesNoPlanesMissingRowsAndChangingShape) {
  auto params = planarParams();
  params.use_cuda = true;
  form::FeatureExtractor gpu(params, 1);
  for (int rows : {1, 3, 2}) {
    gpu.params.num_rows = rows;
    for (double threshold : {0., 1.}) {
      gpu.params.planar_threshold = threshold;
      auto scan = planarScan(gpu.params);
      if (rows == 3)
        for (int c=0; c<gpu.params.num_columns; ++c) scan[gpu.params.num_columns+c].vec3().setZero();
      auto reference_params = gpu.params; reference_params.use_cuda = false;
      auto [cp,cq] = form::FeatureExtractor(reference_params,1).extract(scan,5);
      auto [gp,gq] = gpu.extract(scan,5);
      sortFeatures(cp); sortFeatures(gp);
      EXPECT_EQ(cp,gp); EXPECT_EQ(cq,gq);
    }
  }
  EXPECT_THROW(gpu.extract(std::vector<ScanPoint>{},0), std::runtime_error);
}
#else
TEST(FeatureExtraction, CudaRequestFailsExplicitlyInCpuBuild) {
  auto params = planarParams(); params.use_cuda = true;
  EXPECT_THROW(form::FeatureExtractor(params,1).extract(planarScan(params),0), std::runtime_error);
}
#endif

TEST(FeatureExtraction, ParallelSelectionPreservesSectorSuppression) {
  auto params=planarParams(); params.num_rows=7; params.num_columns=131;
  params.num_sectors=6;
  auto scan=planarScan(params);
  scan[131+20].vec3().setZero();
  for(size_t spacing: {size_t{1},size_t{2},size_t{5}}) {
    params.feature_spacing=spacing; params.parallel_selection=false;
    auto [cp,cq]=form::FeatureExtractor(params,1).extract(scan,7);
    params.parallel_selection=true;
    auto [gp,gq]=form::FeatureExtractor(params,1).extract(scan,7);
    sortFeatures(cp); sortFeatures(gp);
    EXPECT_EQ(cp,gp); EXPECT_EQ(cq,gq);
  }
}

TEST(FeatureExtraction, GpuNormalsRequireCudaExtraction) {
  auto params=planarParams(); params.use_cuda_normals=true;
  EXPECT_THROW(form::FeatureExtractor(params,1).extract(planarScan(params),0),std::invalid_argument);
}

#ifdef FORM_ENABLE_CUDA
TEST(FeatureExtraction, GpuNormalsPreserveSelectedFeaturesAndDirections) {
  auto params=planarParams(); params.num_rows=7; params.num_columns=131;
  params.num_sectors=6; params.parallel_selection=true; params.use_cuda=true;
  form::FeatureExtractor control(params,1);
  params.use_cuda_normals=true;
  form::FeatureExtractor candidate(params,1);
  for(int repetition=0;repetition<3;++repetition) {
    auto scan=planarScan(params);
    scan[60].vec3().setZero(); scan[131+65].vec3().setZero();
    for(size_t i=0;i<scan.size();++i)
      if(scan[i].x!=0) scan[i].x+=.001*std::sin(i*.37+repetition);
    auto compare=[&](const auto& input, double tolerance) {
      auto [cp,cq]=control.extract(input,repetition);
      auto [gp,gq]=candidate.extract(input,repetition);
      sortFeatures(cp); sortFeatures(gp);
      ASSERT_FALSE(cp.empty()); ASSERT_EQ(cp.size(),gp.size()); EXPECT_EQ(cq,gq);
      for(size_t i=0;i<cp.size();++i) {
        EXPECT_EQ(cp[i].vec3(),gp[i].vec3()); EXPECT_EQ(cp[i].scan,gp[i].scan);
        Eigen::Vector3d c(cp[i].nx,cp[i].ny,cp[i].nz),g(gp[i].nx,gp[i].ny,gp[i].nz);
        EXPECT_LT(std::min((c-g).norm(),(c+g).norm()),tolerance);
      }
    };
    compare(scan,1e-10);
    std::vector<form::PointXYZf> floats;
    for(const auto& p:scan) floats.emplace_back(p.x,p.y,p.z);
    compare(floats,2e-4);
  }
  candidate.params.planar_threshold=0.;
  EXPECT_TRUE(std::get<0>(candidate.extract(planarScan(params),0)).empty());
}
#endif
