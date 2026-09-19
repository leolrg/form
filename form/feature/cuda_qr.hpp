#pragma once
#include <Eigen/Core>
#include <limits>
#include <memory>
#include <vector>
namespace form {
/// Batched double-precision tall-skinny QR for 7/13-column feature matrices.
/// Returns zero-padded upper roots R with R^T R = A^T A. Owns reusable CUDA
/// buffers and a stream; one caller at a time. compute includes all transfers.
class BatchedCudaQr {
public:
  explicit BatchedCudaQr(bool pinned_host_buffers = true, int first_rows = 64,
                        int reduction_rows = 64, int block_threads = 32);
  using InputObserver = void (*)(const std::vector<Eigen::MatrixXd>&);
  /// Diagnostic hook; set before processing, one caller at a time.
  static void setInputObserver(InputObserver observer);
  static bool hasInputObserver();
  ~BatchedCudaQr();
  BatchedCudaQr(const BatchedCudaQr&) = delete;
  BatchedCudaQr& operator=(const BatchedCudaQr&) = delete;
  std::vector<Eigen::MatrixXd> compute(const std::vector<Eigen::MatrixXd>& input);
  struct Correspondences {
    const double *pi, *pj, *normal;
    size_t rows;
    bool plane;
  };
  /// Coordinates/normals are interleaved xyz. Packs seven doubles per row
  /// directly into staging memory; expands plane tensors in the first kernel.
  std::vector<Eigen::MatrixXd> computeCorrespondences(const std::vector<Correspondences>& input);
  struct DeviceInput {
    size_t rows;
    bool plane;
  };
  /// Caller-owned device matrices concatenated in column-major rows x 7 layout.
  /// Point rows are [1, pi, pj-pi]; plane rows are [pj, n, n.dot(pj-pi)].
  /// The producer stream (nullptr for the default stream) is waited on by event.
  /// Input is never modified; all reads finish before this synchronous return.
  /// Diagnostic observers receive expanded host matrices when enabled.
  std::vector<Eigen::MatrixXd> computeDevicePacked(
      const double* packed, const std::vector<DeviceInput>& input,
      void* producer_stream = nullptr);
  struct IncrementalStats {
    size_t active_leaves=0;
    unsigned long long dirty_leaves=0, merge_tiles=0, changed_groups=0;
    // Latest compact call only; upload_bytes counts task/bound metadata H2D.
    size_t plan_refreshes=0, upload_bytes=0;
  };
  /// Fixed 64x7 column-major leaves; flattened index is group*capacity+leaf.
  /// All-zero rows represent holes. Clean leaves retain their cached roots.
  /// reset/configuration changes initialize clean leaves to zero; callers must
  /// mark every populated leaf dirty on those calls. Active extents may shrink.
  /// Producer buffers remain caller-owned; return waits for every input read.
  std::vector<Eigen::MatrixXd> computeDevicePackedIncremental(
      const double* packed, const unsigned char* dirty, size_t leaf_capacity_per_group,
      const std::vector<size_t>& active_leaves, bool plane,
      void* producer_stream = nullptr);
  /// Cache both leaf roots and their ancestors; no per-call host extent/plan upload.
  /// Device highwater[group] is a slot extent in [0,capacity*64]; its ceiling in
  /// units of 64 selects active leaves. Holes within those leaves must be zero.
  /// Shrunk-away leaves retain their roots for later clean reactivation. Clean
  /// leaves never previously activated start at zero. Reset, configuration changes,
  /// and failed calls start a zero cache; mark every populated leaf dirty afterward.
  /// This cache is independent of ordinary QR and computeDevicePackedIncremental.
  /// Input remains caller-owned and every read completes before return.
  /// Optional max_active_leaves bounds every device extent; a false bound throws
  /// and invalidates the cache. The default sentinel uses the full capacity.
  std::vector<Eigen::MatrixXd> computeDevicePackedIncrementalTree(
      const double* packed, const unsigned char* dirty, size_t leaf_capacity_per_group,
      size_t group_count, const int* device_highwater, bool plane,
      void* producer_stream = nullptr,
      size_t max_active_leaves = std::numeric_limits<size_t>::max());
  /// Compact scheduling shares the tree roots with the capped API. Bounds are
  /// per-group leaf counts; current device extents are validated against them.
  /// Cached node tasks and bounds upload only when their respective values change.
  /// Shrinking bounds process the preceding extent too; false bounds invalidate
  /// the shared tree cache. Warp caps apply only to the capped API.
  std::vector<Eigen::MatrixXd> computeDevicePackedIncrementalTreeCompact(
      const double* packed, const unsigned char* dirty, size_t leaf_capacity_per_group,
      const std::vector<size_t>& per_group_bounds, const int* device_highwater,
      bool plane, void* producer_stream = nullptr);
  /// Accepts 32/128/256 warps per group; default 32. Valid changes preserve caches.
  void setIncrementalTreeWarpCap(size_t cap);
  void resetIncremental();
  /// Latest call's work; device counters are downloaded only on request.
  IncrementalStats incrementalStats();
  IncrementalStats incrementalTreeStats();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
