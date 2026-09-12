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
inline void reset() {
  last_initial_error = 0; last_final_error = 0;
  summary_build_cpu = 0; summary_prepare_wall = 0;
  factor_linearize = 0; factor_eval = 0; factor_error = 0;
  linearize_wall = 0; assemble = 0; solve = 0; iterations = 0;
  cuda_solve_calls = 0; cuda_solve_fallbacks = 0;
}
inline double ms(const std::atomic<uint64_t>& c) { return c.load() * 1e-6; }
} // namespace form::profile
