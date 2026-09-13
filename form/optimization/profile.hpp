#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <Eigen/Core>

namespace form::profile {
// Optional process-wide counters for one estimator benchmark at a time. Factor
// counters sum CPU durations across TBB workers; they are not wall-time stages.
inline bool enabled = false;
using SolveObserver = void (*)(const Eigen::MatrixXd& augmented);
// Diagnostic only; install before a profiled scan, with one optimizer caller.
inline SolveObserver solve_observer = nullptr;
inline std::atomic<double> last_initial_error{0}, last_final_error{0};
inline std::atomic<uint64_t> factor_linearize{0}, factor_eval{0}, factor_error{0};
inline std::atomic<uint64_t> summary_build_cpu{0}, summary_prepare_wall{0};
inline std::atomic<uint64_t> linearize_wall{0}, assemble{0}, solve{0}, iterations{0};
inline std::atomic<uint64_t> cuda_solve_calls{0}, cuda_solve_fallbacks{0};
// Optional detailed diagnostics; normal timings sum a deterministic 1/64 sample
// across workers and must not be added to wall-time stage counters.
inline std::atomic<uint64_t> extract_validate{0}, extract_curvature{0}, extract_planar_select{0};
inline std::atomic<uint64_t> extract_point_mask{0}, extract_point_select{0}, extract_normals{0}, extract_pack{0};
inline std::atomic<uint64_t> normal_search_sample_cpu{0}, normal_eigen_sample_cpu{0}, normal_samples{0};
inline std::atomic<uint64_t> resident_reset_wall{0}, resident_error_wall{0}, graph_build_wall{0};
inline std::atomic<uint64_t> map_world_wall{0}, map_snapshot_wall{0}, match_materialize_wall{0};
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Scope {
  std::atomic<uint64_t>* counter;
  Clock::time_point start;
  explicit Scope(std::atomic<uint64_t>& c) : counter(enabled ? &c : nullptr) {
    if (counter) start = Clock::now();
  }
  ~Scope() {
    if (counter) counter->fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count(),
        std::memory_order_relaxed);
  }
};
// Mark consecutive serial stages without querying the clock when disabled.
inline void checkpoint(std::atomic<uint64_t>& counter, Clock::time_point& start) {
  if (!enabled) return;
  const auto now = Clock::now();
  counter.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(now-start).count(),
                    std::memory_order_relaxed);
  start = now;
}
inline void reset() {
  extract_validate = extract_curvature = extract_planar_select = 0;
  extract_point_mask = extract_point_select = extract_normals = extract_pack = 0;
  normal_search_sample_cpu = normal_eigen_sample_cpu = normal_samples = 0;
  resident_reset_wall = resident_error_wall = graph_build_wall = 0;
  map_world_wall = map_snapshot_wall = match_materialize_wall = 0;
  last_initial_error = 0; last_final_error = 0;
  summary_build_cpu = 0; summary_prepare_wall = 0;
  factor_linearize = 0; factor_eval = 0; factor_error = 0;
  linearize_wall = 0; assemble = 0; solve = 0; iterations = 0;
  cuda_solve_calls = 0; cuda_solve_fallbacks = 0;
}
inline double ms(const std::atomic<uint64_t>& c) { return c.load() * 1e-6; }
} // namespace form::profile
